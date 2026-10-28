# FS-Go: per-position xP models

Predicts a Fantasy Premier League player's expected points (`xP`) for the next gameweek, with a separate
model for each position (GK, DEF, MID, FWD). Nine base models and a few ensembles are compared on
time-based folds of the 2024-25 season.

## Layout

```
inputs/                     data
  rolled_player_data_24-25.csv   raw source (downloaded, not committed)
  fg_data_{pos}.csv              columns kept per position (notebooks/get data.ipynb)
  {pos}_train_folds.csv          same, plus the kfold column (src/create_folds.py)
models/                     everything the pipeline writes (see "Outputs")
notebooks/
  get data.ipynb            downloads the raw data and writes fg_data_{pos}.csv
  model_comparison.ipynb    charts and tests comparing the base models
src/
  config.py                 positions, fold numbers and paths
  model_dispatcher.py       features and the nine base models per position
  create_folds.py           adds the fold column
  train.py                  trains one base model on one fold
  folds.sh                  runs train.py for every model, fold and position
  ensemble.py               builds one ensemble on one fold
  ensemble.sh               runs ensemble.py for every position and fold
  compare.py                prints the comparison table per position
  rank_features.py          optional: ranks candidate features (writes features.py; not used by the models yet)
```

## Setup

Python 3.11. All scripts run from `src/` and use paths relative to it.

```
pip install -r requirements.txt
```

## Run

```
# 1. Data: run notebooks/get data.ipynb (downloads about 33 MB the first time)

# 2. Base models: create the folds and train every model on every fold and position (about 5 minutes)
cd src
bash folds.sh

# 3. Ensembles (about 2 minutes)
bash ensemble.sh

# 4. Comparison table per position
python compare.py
python compare.py --pos GK --folds     # one position, with the MAE of each fold
```

`folds.sh ridge svr` trains only the named models. `bash ensemble.sh mean stack top` runs more ensemble
methods, and `TOP_K=4 BASE_MODELS="ridge svr cat" bash ensemble.sh` changes the model sets.

## How models are evaluated

A fold is a gameweek. Each fold trains on every gameweek before it and validates on that gameweek.

* **Validation folds 21, 26, 31, 36** are used to compare models and choose ensembles.
* **Test folds 37, 38** are held out and only reported.

Both lists are in `src/config.py`. Scores are mean absolute error (MAE, lower is better) and R².

`ensemble.sh` builds two equal-weight mean ensembles per position:

* `ens_mean_base`: ridge, hgb, lgbm, xgb and svr, the same for every position.
* `ens_mean_top3`: the three base models with the lowest validation MAE for that position. Because the
  models are picked on the validation folds, its validation score is slightly optimistic. Its test score is not.

`ensemble.py` also supports `median`, `stack` (non-negative weights fitted on the three gameweeks before
the fold) and `top` (the best models re-picked inside each fold). In earlier runs, neither `stack` nor `top`
beat the plain mean.

## Results

Mean MAE from `python compare.py`, for the best options of each position:

| Position | Best single model | Its test MAE | Best ensemble | Its test MAE |
|---|---|---|---|---|
| GK | svr | 0.800 | ens_mean_top3 (svr, cat, xgb) | 0.768 |
| DEF | lgbm | 0.803 | ens_mean_base | 0.791 |
| MID | svr | 0.683 | ens_mean_base | 0.698 |
| FWD | svr | 0.783 | ens_mean_top3 (svr, cat, rf) | 0.853 |

* **FWD:** svr alone is clearly best on both validation and test.
* **MID:** ensembles win on validation, svr alone wins on test. The gap is small either way.
* **DEF:** the base ensemble is best on both.
* **GK:** the top-3 ensemble is best on both, but each GK gameweek has only about 20 rows, so this is the
  least certain result.

Each validation or test fold is a single gameweek, so differences below about 0.02 MAE are within noise.

## Outputs

All written to `models/`. Result files are appended to; `compare.py` keeps the latest run per model and fold.

| File | Written by | Contents |
|---|---|---|
| `{POS}_results.csv` | train.py | MAE and R² per model and fold |
| `ensemble_results.csv` | ensemble.py | MAE and R² per ensemble and fold, its weights, and each base model's MAE on the same fold |
| `oof_{model}_{POS}_{fold}.csv` | both | validation predictions (`row` is the row number in `{pos}_train_folds.csv`) |
| `{model}_{POS}_{fold}.pkl` | both | fitted model, loadable with `joblib.load` (run from `src/`, or add it to `sys.path`) |
| `feature_ranking.csv` | rank_features.py | feature scores per position |

The `.pkl` and `oof_*.csv` files are not committed. Rebuild them with steps 2 and 3.

## Known issues

* Python sometimes crashes with a segmentation fault (exit code 139) after training lightgbm or xgboost.
  The crash is random and the same run passes when repeated, so `folds.sh` and `ensemble.sh` retry it up to
  three times.
* In `model_dispatcher.py`, the MID and FWD feature lists contain `threat_rolling_3` twice, so that feature is
  fed to the models twice. It was left as is so that results stay comparable with earlier runs.
* `features.py` (from `rank_features.py`) holds a ranked feature list per position, but `model_dispatcher.py`
  does not use it yet.
