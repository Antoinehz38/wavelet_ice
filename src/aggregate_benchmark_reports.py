from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


REPORT_RE = re.compile(r"^(?P<wavelet>.+)_(?P<timestamp>\d{8}_\d{6})_rapport\.json$")


def cli_path(value: str) -> Path:
    return Path(value).expanduser()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Agrege les rapports *_rapport.json d'une wavelet sur tous les sous-dossiers ex* "
            "du benchmark et affiche les moyennes des champs numeriques."
        )
    )
    parser.add_argument(
        "--wavelet",
        required=True,
        help="Nom exact de la wavelet a agreger, par ex. cmor100.0-1.0.",
    )
    parser.add_argument(
        "--benchmark-root",
        type=cli_path,
        default=Path("./data/benchmark"),
        help="Racine du benchmark contenant les sous-dossiers ex*.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def select_report_files(benchmark_root: Path, wavelet: str) -> list[Path]:
    latest_by_experiment: dict[str, tuple[str, Path]] = {}

    for report_path in sorted(benchmark_root.glob("ex*/**/*_rapport.json")):
        match = REPORT_RE.match(report_path.name)
        if not match:
            continue
        if match.group("wavelet") != wavelet:
            continue

        experiment = report_path.parent.name
        timestamp = match.group("timestamp")
        previous = latest_by_experiment.get(experiment)
        if previous is None or timestamp > previous[0]:
            latest_by_experiment[experiment] = (timestamp, report_path)

    return sorted(path for _, path in latest_by_experiment.values())


def average_numeric_mapping(items: list[dict[str, Any]]) -> dict[str, float]:
    totals: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)

    for item in items:
        for key, value in item.items():
            if is_number(value):
                totals[key] += float(value)
                counts[key] += 1

    return {key: totals[key] / counts[key] for key in sorted(totals)}


def aggregate_reports(report_files: list[Path]) -> dict[str, Any]:
    report_payloads = [load_json(path) for path in report_files]

    summary_average = average_numeric_mapping(
        [payload.get("summary", {}) for payload in report_payloads]
    )

    iou_thresholds = sorted(
        {
            threshold
            for payload in report_payloads
            for threshold in payload.get("iou_sweep", {}).keys()
        },
        key=float,
    )

    iou_average: dict[str, dict[str, float]] = {}
    for threshold in iou_thresholds:
        threshold_entries: list[dict[str, Any]] = []
        for payload in report_payloads:
            threshold_payload = payload.get("iou_sweep", {}).get(threshold)
            if isinstance(threshold_payload, dict):
                threshold_entries.append(threshold_payload)
        iou_average[threshold] = average_numeric_mapping(threshold_entries)

    return {
        "wavelet": REPORT_RE.match(report_files[0].name).group("wavelet"),
        "benchmark_root": str(report_files[0].parents[1]),
        "num_reports": len(report_files),
        "report_files": [str(path) for path in report_files],
        "summary_average": summary_average,
        "iou_sweep_average": iou_average,
    }


def main() -> None:
    args = parse_args()
    report_files = select_report_files(args.benchmark_root, args.wavelet)

    if not report_files:
        raise SystemExit(
            f"Aucun fichier *_rapport.json trouve pour la wavelet '{args.wavelet}' "
            f"dans {args.benchmark_root}."
        )

    aggregated = aggregate_reports(report_files)
    print(json.dumps(aggregated, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
