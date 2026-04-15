from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment

# Utilisation dossier complet :
# python -m src.evaluate_sigmf_predictions \
#   --pred-root ./data/benchmark \
#   --meta-root ~/raid/spawc21_challenge_dataset/train \
#   --output-root ./data/evaluation_reports

# Utilisation fichier unique :
# python -m src.evaluate_sigmf_predictions \
#   --prediction-json ./data/benchmark/ex1/cmor100.0-1.0_20260414_195140.json \
#   --metadata-json ~/raid/spawc21_challenge_dataset/train/west-wideband-modrec-ex1-tmpl2-20.04.sigmf-meta \
#   --output-json ./data/evaluation_reports/ex1/report.json



IOU_THRESHOLDS = np.arange(0.50, 0.96, 0.05)
EXPERIMENT_RE = re.compile(r"ex(\d+)", re.IGNORECASE)


@dataclass(frozen=True)
class BoxRecord:
    index: int
    label: str
    sample_start: float
    sample_count: float
    freq_lower_edge: float
    freq_upper_edge: float
    source_path: str

    @property
    def t0(self) -> float:
        return float(self.sample_start)

    @property
    def t1(self) -> float:
        return float(self.sample_start + self.sample_count)

    @property
    def f0(self) -> float:
        return float(self.freq_lower_edge)

    @property
    def f1(self) -> float:
        return float(self.freq_upper_edge)

    @property
    def tc(self) -> float:
        return float((self.t0 + self.t1) / 2.0)

    @property
    def fc(self) -> float:
        return float((self.f0 + self.f1) / 2.0)

    @property
    def B(self) -> float:
        return float(self.f1 - self.f0)

    @property
    def D(self) -> float:
        return float(self.t1 - self.t0)

    def rect(self) -> tuple[float, float, float, float]:
        return (self.t0, self.f0, self.t1, self.f1)

    def base_params(self) -> dict[str, float]:
        return {
            "t0": self.t0,
            "t1": self.t1,
            "f0": self.f0,
            "f1": self.f1,
        }

    def physical_params(self) -> dict[str, float]:
        return {
            "t0": self.t0,
            "t1": self.t1,
            "f0": self.f0,
            "f1": self.f1,
            "tc": self.tc,
            "fc": self.fc,
            "B": self.B,
            "D": self.D,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare des JSON SigMF de predictions avec les metadonnees SigMF "
            "et genere un rapport d'evaluation detaille."
        )
    )
    parser.add_argument(
        "--pred-root",
        type=Path,
        default=Path("./data/benchmark"),
        help="Dossier racine contenant les predictions dans des sous-dossiers exN.",
    )
    parser.add_argument(
        "--meta-root",
        type=Path,
        default=Path("~/raid/spawc21_challenge_dataset/train").expanduser(),
        help="Dossier contenant les fichiers de metadonnees .sigmf-meta.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("./data/evaluation_reports"),
        help="Dossier de sortie des rapports JSON.",
    )
    parser.add_argument(
        "--prediction-json",
        type=Path,
        help="Chemin d'un seul fichier JSON de predictions a evaluer.",
    )
    parser.add_argument(
        "--metadata-json",
        type=Path,
        help="Chemin du fichier .sigmf-meta a comparer avec --prediction-json.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Chemin du rapport JSON en mode fichier unique.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def get_capture_sample_count(payload: dict[str, Any]) -> float:
    captures = payload.get("captures", [])
    if not captures:
        return 0.0
    return float(captures[0].get("core:sample_count", 0.0))


def get_frequency_span(payload: dict[str, Any]) -> float:
    annotations = payload.get("annotations", [])
    if not annotations:
        return 1.0

    min_f = min(float(item["core:freq_lower_edge"]) for item in annotations)
    max_f = max(float(item["core:freq_upper_edge"]) for item in annotations)
    span = max_f - min_f
    return float(span if span > 0 else 1.0)


def load_boxes(payload: dict[str, Any], source_path: Path) -> list[BoxRecord]:
    boxes: list[BoxRecord] = []
    for idx, annotation in enumerate(payload.get("annotations", [])):
        boxes.append(
            BoxRecord(
                index=idx,
                label=str(annotation.get("core:description", "")),
                sample_start=float(annotation["core:sample_start"]),
                sample_count=float(annotation["core:sample_count"]),
                freq_lower_edge=float(annotation["core:freq_lower_edge"]),
                freq_upper_edge=float(annotation["core:freq_upper_edge"]),
                source_path=str(source_path),
            )
        )
    return boxes


def intersection_area(rect_a: tuple[float, float, float, float], rect_b: tuple[float, float, float, float]) -> float:
    x1 = max(rect_a[0], rect_b[0])
    y1 = max(rect_a[1], rect_b[1])
    x2 = min(rect_a[2], rect_b[2])
    y2 = min(rect_a[3], rect_b[3])
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    return float(width * height)


def rect_area(rect: tuple[float, float, float, float]) -> float:
    return float(max(0.0, rect[2] - rect[0]) * max(0.0, rect[3] - rect[1]))


def compute_iou(rect_a: tuple[float, float, float, float], rect_b: tuple[float, float, float, float]) -> float:
    inter = intersection_area(rect_a, rect_b)
    union = rect_area(rect_a) + rect_area(rect_b) - inter
    if union <= 0.0:
        return 0.0
    return float(inter / union)


def union_area(rects: Iterable[tuple[float, float, float, float]]) -> float:
    rect_list = [rect for rect in rects if rect_area(rect) > 0.0]
    if not rect_list:
        return 0.0

    x_points = sorted({rect[0] for rect in rect_list} | {rect[2] for rect in rect_list})
    if len(x_points) < 2:
        return 0.0

    total_area = 0.0
    for x_left, x_right in zip(x_points, x_points[1:]):
        if x_right <= x_left:
            continue

        intervals: list[tuple[float, float]] = []
        for rect in rect_list:
            if rect[0] < x_right and rect[2] > x_left:
                intervals.append((rect[1], rect[3]))

        if not intervals:
            continue

        intervals.sort()
        covered = 0.0
        current_start, current_end = intervals[0]
        for start, end in intervals[1:]:
            if start <= current_end:
                current_end = max(current_end, end)
            else:
                covered += max(0.0, current_end - current_start)
                current_start, current_end = start, end
        covered += max(0.0, current_end - current_start)
        total_area += (x_right - x_left) * covered

    return float(total_area)


def intersection_with_union_area(
    rect: tuple[float, float, float, float],
    other_rects: Iterable[tuple[float, float, float, float]],
) -> float:
    clipped_rects = []
    for other in other_rects:
        x1 = max(rect[0], other[0])
        y1 = max(rect[1], other[1])
        x2 = min(rect[2], other[2])
        y2 = min(rect[3], other[3])
        if x2 > x1 and y2 > y1:
            clipped_rects.append((x1, y1, x2, y2))
    return union_area(clipped_rects)


def build_iou_matrix(pred_boxes: list[BoxRecord], gt_boxes: list[BoxRecord], iou_threshold: float) -> np.ndarray:
    matrix = np.zeros((len(pred_boxes), len(gt_boxes)), dtype=float)
    for pred_idx, pred_box in enumerate(pred_boxes):
        for gt_idx, gt_box in enumerate(gt_boxes):
            iou = compute_iou(pred_box.rect(), gt_box.rect())
            if iou >= iou_threshold:
                matrix[pred_idx, gt_idx] = iou
    return matrix


def match_boxes(pred_boxes: list[BoxRecord], gt_boxes: list[BoxRecord], iou_threshold: float) -> list[tuple[int, int, float]]:
    if not pred_boxes or not gt_boxes:
        return []

    iou_matrix = build_iou_matrix(pred_boxes, gt_boxes, iou_threshold)
    size = max(len(pred_boxes), len(gt_boxes))
    score_matrix = np.zeros((size, size), dtype=float)
    score_matrix[: len(pred_boxes), : len(gt_boxes)] = iou_matrix

    row_ind, col_ind = linear_sum_assignment(score_matrix, maximize=True)
    matches: list[tuple[int, int, float]] = []
    for pred_idx, gt_idx in zip(row_ind, col_ind):
        if pred_idx < len(pred_boxes) and gt_idx < len(gt_boxes):
            iou = float(iou_matrix[pred_idx, gt_idx])
            if iou > 0.0:
                matches.append((int(pred_idx), int(gt_idx), iou))
    matches.sort(key=lambda item: item[0])
    return matches


def safe_relative_percent(delta: float, reference: float) -> float | None:
    if math.isclose(reference, 0.0, abs_tol=1e-12):
        return None
    return float((delta / reference) * 100.0)


def compute_match_scores(
    pred_box: BoxRecord,
    gt_box: BoxRecord,
    capture_sample_count: float,
    freq_span: float,
) -> dict[str, Any]:
    pred_params = pred_box.physical_params()
    gt_params = gt_box.physical_params()
    base_keys = ("t0", "t1", "f0", "f1")

    deltas = {
        key: float(gt_params[key] - pred_params[key])
        for key in ("t0", "t1", "f0", "f1", "tc", "fc", "B", "D")
    }
    delta_percents = {
        key: safe_relative_percent(deltas[key], gt_params[key])
        for key in deltas
    }

    l1 = float(sum(abs(deltas[key]) for key in base_keys))
    l2 = float(math.sqrt(sum(deltas[key] ** 2 for key in base_keys)))

    time_scale = capture_sample_count if capture_sample_count > 0 else 1.0
    freq_scale = freq_span if freq_span > 0 else 1.0
    accuracy_components = []
    for key in base_keys:
        reference = abs(gt_params[key])
        if math.isclose(reference, 0.0, abs_tol=1e-12):
            reference = time_scale if key.startswith("t") else freq_scale
        score = max(0.0, 100.0 * (1.0 - (abs(deltas[key]) / reference)))
        accuracy_components.append(score)

    return {
        "score_L1": l1,
        "score_L2": l2,
        "accuracy_percent": float(sum(accuracy_components) / len(accuracy_components)),
        "prediction": pred_params,
        "metadata": gt_params,
        "delta_meta_minus_prediction": deltas,
        "delta_meta_minus_prediction_percent": delta_percents,
    }


def build_box_coverage_summary(pred_boxes: list[BoxRecord], gt_boxes: list[BoxRecord]) -> dict[str, Any]:
    gt_rects = [box.rect() for box in gt_boxes]
    ratios = []
    no_overlap = 0
    details: dict[str, Any] = {}

    for pred_box in pred_boxes:
        pred_rect = pred_box.rect()
        pred_area = rect_area(pred_rect)
        inter_area = intersection_with_union_area(pred_rect, gt_rects)
        ratio = float(inter_area / pred_area) if pred_area > 0.0 else 0.0
        if math.isclose(ratio, 0.0, abs_tol=1e-12):
            no_overlap += 1
        ratios.append(ratio)

        overlapping_labels = [
            gt_box.label
            for gt_box in gt_boxes
            if intersection_area(pred_rect, gt_box.rect()) > 0.0
        ]
        label_suffix = " + ".join(dict.fromkeys(overlapping_labels)) if overlapping_labels else "NO_LABEL_OVERLAP"
        details[
            f"BBox #{pred_box.index} {label_suffix} | intersection(BB_i, union(labels)) / aire(BB_i)"
        ] = ratio

    return {
        "average_BB_coverage_ratio": float(sum(ratios) / len(ratios)) if ratios else 0.0,
        "Nb_BB_sans_rencouvrement": int(no_overlap),
        **details,
    }


def build_label_coverage_summary(pred_boxes: list[BoxRecord], gt_boxes: list[BoxRecord]) -> dict[str, Any]:
    pred_rects = [box.rect() for box in pred_boxes]
    ratios = []
    no_overlap = 0
    details: dict[str, Any] = {}

    for gt_box in gt_boxes:
        gt_rect = gt_box.rect()
        gt_area = rect_area(gt_rect)
        inter_area = intersection_with_union_area(gt_rect, pred_rects)
        ratio = float(inter_area / gt_area) if gt_area > 0.0 else 0.0
        if math.isclose(ratio, 0.0, abs_tol=1e-12):
            no_overlap += 1
        ratios.append(ratio)
        details[
            f"Label #{gt_box.index} {gt_box.label} | intersection(label_i, union(BB)) / aire(label_i)"
        ] = ratio

    return {
        "average_prediction_coverage_ratio_on_labels": float(sum(ratios) / len(ratios)) if ratios else 0.0,
        "Nb_labels_sans_recouvrement": int(no_overlap),
        **details,
    }


def evaluate_prediction_file(prediction_path: Path, metadata_path: Path) -> dict[str, Any]:
    pred_payload = load_json(prediction_path)
    meta_payload = load_json(metadata_path)

    pred_boxes = load_boxes(pred_payload, prediction_path)
    gt_boxes = load_boxes(meta_payload, metadata_path)

    capture_sample_count = get_capture_sample_count(meta_payload)
    freq_span = get_frequency_span(meta_payload)

    sweep_results: dict[str, Any] = {}
    f1_scores = []
    for threshold in IOU_THRESHOLDS:
        matches = match_boxes(pred_boxes, gt_boxes, float(threshold))
        tp = len(matches)
        fp = len(pred_boxes) - tp
        fn = len(gt_boxes) - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        f1_scores.append(f1)

        detailed_matches = []
        l1_scores = []
        l2_scores = []
        accuracies = []
        for pred_idx, gt_idx, iou in matches:
            pred_box = pred_boxes[pred_idx]
            gt_box = gt_boxes[gt_idx]
            match_scores = compute_match_scores(pred_box, gt_box, capture_sample_count, freq_span)
            l1_scores.append(match_scores["score_L1"])
            l2_scores.append(match_scores["score_L2"])
            accuracies.append(match_scores["accuracy_percent"])
            detailed_matches.append(
                {
                    "pred_index": int(pred_idx),
                    "gt_label": gt_box.label,
                    "iou": float(iou),
                    **match_scores,
                }
            )

        threshold_key = f"{threshold:.2f}"
        sweep_results[threshold_key] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "avg_L1_score": float(sum(l1_scores) / len(l1_scores)) if l1_scores else None,
            "avg_L2_score": float(sum(l2_scores) / len(l2_scores)) if l2_scores else None,
            "avg_accuracy_percent": float(sum(accuracies) / len(accuracies)) if accuracies else None,
            "matches": detailed_matches,
        }

    report = {
        "prediction_file": str(prediction_path),
        "metadata_file": str(metadata_path),
        "summary": {
            "num_predictions": int(len(pred_boxes)),
            "num_metadata_boxes": int(len(gt_boxes)),
            "prediction_minus_metadata": int(len(pred_boxes) - len(gt_boxes)),
            "mf1_50_95": float(sum(f1_scores) / len(f1_scores)) if f1_scores else 0.0,
            **build_box_coverage_summary(pred_boxes, gt_boxes),
            **build_label_coverage_summary(pred_boxes, gt_boxes),
        },
        "iou_sweep": sweep_results,
    }
    return report


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def extract_experiment_id(path: Path) -> str | None:
    for candidate in [path.name, path.stem, path.parent.name]:
        match = EXPERIMENT_RE.search(candidate)
        if match:
            return match.group(1)
    return None


def find_metadata_path(meta_root: Path, experiment_id: str) -> Path:
    candidates = sorted(meta_root.glob(f"*ex{experiment_id}*.sigmf-meta"))
    if not candidates:
        raise FileNotFoundError(
            f"Aucune metadonnee trouvee pour ex{experiment_id} dans {meta_root}"
        )
    if len(candidates) > 1:
        exact_prefix = [path for path in candidates if f"-ex{experiment_id}-" in path.name]
        if len(exact_prefix) == 1:
            return exact_prefix[0]
        raise RuntimeError(
            f"Plusieurs metadonnees candidates pour ex{experiment_id}: "
            + ", ".join(path.name for path in candidates)
        )
    return candidates[0]


def iter_prediction_files(pred_root: Path) -> Iterable[Path]:
    for path in sorted(pred_root.rglob("*.json")):
        if EXPERIMENT_RE.search(path.parent.name) or EXPERIMENT_RE.search(path.name):
            yield path


def evaluate_single_file(args: argparse.Namespace) -> int:
    if args.prediction_json is None or args.metadata_json is None:
        raise ValueError("--prediction-json et --metadata-json sont requis en mode fichier unique.")

    output_json = args.output_json
    if output_json is None:
        output_json = args.output_root / f"{args.prediction_json.stem}_evaluation.json"

    report = evaluate_prediction_file(args.prediction_json, args.metadata_json)
    write_json(output_json, report)
    print(f"Rapport genere: {output_json}")
    return 0


def evaluate_directory(args: argparse.Namespace) -> int:
    pred_root = args.pred_root.expanduser()
    meta_root = args.meta_root.expanduser()
    output_root = args.output_root.expanduser()

    prediction_files = list(iter_prediction_files(pred_root))
    if not prediction_files:
        raise FileNotFoundError(f"Aucun JSON de prediction trouve dans {pred_root}")

    manifest: list[dict[str, str]] = []
    for prediction_path in prediction_files:
        experiment_id = extract_experiment_id(prediction_path)
        if experiment_id is None:
            raise RuntimeError(f"Impossible d'extraire le numero exN pour {prediction_path}")

        metadata_path = find_metadata_path(meta_root, experiment_id)
        report = evaluate_prediction_file(prediction_path, metadata_path)

        rel_parent = prediction_path.parent.relative_to(pred_root)
        output_json = output_root / rel_parent / f"{prediction_path.stem}_evaluation.json"
        write_json(output_json, report)
        manifest.append(
            {
                "prediction_file": str(prediction_path),
                "metadata_file": str(metadata_path),
                "report_file": str(output_json),
            }
        )
        print(f"[ex{experiment_id}] Rapport genere: {output_json}")

    write_json(output_root / "manifest.json", {"reports": manifest})
    print(f"Manifest genere: {output_root / 'manifest.json'}")
    return 0


def main() -> int:
    args = parse_args()
    if args.prediction_json or args.metadata_json:
        return evaluate_single_file(args)
    return evaluate_directory(args)


if __name__ == "__main__":
    raise SystemExit(main())
