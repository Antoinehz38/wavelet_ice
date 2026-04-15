# wavelet_ice


## GET STARTED 

We highly recommend do set-up a virtual env using 

```bash
python3.12 -m venv .venv
source .venv/bin/activate  
```

Then to install dependencies just run: 

```bash
pip install -e . 
```

You are good to go


## TO RUN 

You have to run the main file, you can use this command :

```commandline
python3 -m src.main --input /raid/spawc21_challenge_dataset/train/west-wideband-modrec-ex110-tmpl13-20.04.sigmf-data --output ~/guido_leteurtre_hanachowicz/wavelet_ice/data/test
```

```
python3 -m src.main --input /raid/spawc21_challenge_dataset/train/west-wideband-modrec-ex110-tmpl13-20.04.sigmf-data --output ~/guido_leteurtre_hanachowicz/hanacho/wavelet_ice/data/benchmark --duration 100_000_000 --addPrediction true --transfoType cwt_rc --pointsPerWindow 2_000_000

```

## EVALUATION

Single prediction file:

```bash
python3 -m src.evaluate_sigmf_predictions \
  --prediction-json ./data/benchmark/ex1/cmor100.0-1.0_20260414_195140.json \
  --metadata /raid/spawc21_challenge_dataset/train
```

The script automatically finds the matching `.sigmf-meta` file from the `exN` identifier and writes the report next to the prediction JSON with suffix `_rapport.json`.

Whole benchmark directory:

```bash
python3 -m src.evaluate_sigmf_predictions \
  --pred-root ./data/benchmark \
  --metaroot /raid/spawc21_challenge_dataset/train
```

Each report is written next to its prediction JSON.

