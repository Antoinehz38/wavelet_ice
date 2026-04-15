# wavelet_ice


## GET STARTED 

We highly recommend do set-up a virtual env using, on bash : 

```bash
python3.12 -m venv .venv
source .venv/bin/activate  
```

On windows :

```sur windows
python -m venv .venv
source .venv/Scripts/activate 
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


