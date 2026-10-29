# FS-Go 2024-25: per-position xP models

Predicts a Fantasy Premier League player's expected points (`xP`) for the next gameweek, with a separate
model for each position (GK, DEF, MID, FWD). Nine base models and two ensembles compete at four
model selection points in the 2024-25 season, and each selected model is scored on the gameweeks it
would have been used for.

## Layout

```
inputs/                     data
  rolled_player_data_24-25.csv           raw source (downloaded by src/prepare_data.py, not committed)
  rolled_player_data_24-25_xp_fixed.csv  same, with the xP features rebuilt (src/fix_missing_xp.py)
  fg_data_{pos}.csv                      columns the models use, per position (src/prepare_data.py)
  {pos}_train_folds.csv                  same, plus the kfold column (src/create_folds.py)
models/                     everything the pipeline writes (see "Outputs")
notebooks/
  model_comparison.ipynb    charts and tests comparing the models, and MAE and R² per gameweek
src/
  config.py                 positions, selection points, gameweeks without a target, paths
  run.sh                    runs every step below in order
  prepare_data.py           downloads the raw data, fixes the xP features, writes fg_data_{pos}.csv
  fix_missing_xp.py         writes a copy of the raw data with the xP rolling features rebuilt
  model_dispatcher.py       features and the nine base models per position
  data.py                   loads a position's fold file, without the gameweeks that have no target
  create_folds.py           adds the fold column
  train.py                  trains one base model at one selection point
  folds.sh                  runs create_folds.py, then train.py for every model, selection point and position
  ensemble.py               builds one ensemble at one selection point
  ensemble.sh               runs ensemble.py for every position and selection point
  deploy.py                 scores every saved model on the gameweeks after its selection point
  compare.py                prints the comparison table per position
  predictor.py              predicts every player for gameweeks 21 to 38 with the selected models
  performance.py            scores those predictions per gameweek and position
```

## Setup

Python 3.11. All scripts run from `src/` and use paths relative to it.

```
pip install -r requirements.txt
```

## Run

Everything at once:

```
bash src/run.sh
```

Or step by step:

```
cd src

# 1. Data: download (about 33 MB, first time only), fix the xP features, write fg_data_{pos}.csv
python prepare_data.py
python prepare_data.py --fresh         # download and rebuild the raw files again

# 2. Base models: create the folds and train every model at every selection point and position
bash folds.sh

# 3. Ensembles
bash ensemble.sh

# 4. Score the saved models on the gameweeks after each selection point
python deploy.py

# 5. Comparison table per position
python compare.py
python compare.py --pos GK --folds     # one position, with each selection point and window

# 6. Predictions for gameweeks 21 to 38 with the selected model of each window
python predictor.py

# 7. Performance of those predictions per gameweek and position
python performance.py
python performance.py --pos GK
```

`folds.sh ridge svr` trains only the named models. `bash ensemble.sh mean stack top` runs more ensemble
methods, and `TOP_K=4 BASE_MODELS="ridge svr cat" bash ensemble.sh` changes the model sets. Rerun
`deploy.py` after either script.

To change the features, edit the lists in `model_dispatcher.py` and rerun from step 1: `prepare_data.py`
takes each position's columns from those lists.

## How models are evaluated

A fold is a gameweek. Models are chosen at four **selection points: gameweeks 21, 26, 31 and 36**.

1. At gameweek 21, every model trains on gameweeks 1 to 20 and is validated on gameweek 21.
2. The model with the lowest validation MAE is selected. The dummy baseline is never selected.
3. The selected model is used, unchanged, for the next 5 gameweeks: 22 to 26.
4. The same is repeated at gameweeks 26, 31 and 36. The last model is used for 37 and 38, the end of
   the season.

`predictor.py` also predicts gameweek 21 with the model selected at 21. Those predictions were used to
select that model, so they are marked `in_selection` and left out of the scores.

The selection points and the 5-gameweek horizon are set in `src/config.py`. Scores are mean absolute
error (MAE, lower is better) and R². The score that matters is the MAE on the gameweeks after each
selection point, 22 to 38, because those gameweeks played no part in choosing the model.

Two ensembles compete with the base models at each selection point:

* `ens_mean_base`: the mean of ridge, hgb, lgbm, xgb and svr, the same for every position.
* `ens_top3`: the mean of the three base models with the lowest MAE on the three gameweeks before the
  selection point. It only uses information available at that point.

Gameweeks 22, 32 and 34 have no target in the source data, so they are left out of training and of every
score (see "Known issues"). The three gameweeks before a selection point are counted without them, so
before gameweek 36 they are 31, 33 and 35.

`ensemble.py` also supports `median` and `stack` (non-negative weights fitted on the three gameweeks
before the selection point).

## Results

MAE on the gameweeks after each selection point, from `python compare.py`. Gameweeks 22, 32 and 34 are
left out because the source data has no xP for them (see "Known issues"), which leaves 14 gameweeks.

| Position | Selection procedure | Best fixed model | Its MAE | Picks at gameweeks 21, 26, 31, 36 |
|---|---|---|---|---|
| GK | 1.022 | ens_top3 | 0.974 | ens_top3, dt, lgbm, rf |
| DEF | 0.812 | ens_mean_base | 0.801 | cat, rf, ens_mean_base, lgbm |
| MID | 0.834 | ens_mean_base | 0.835 | cat, ens_mean_base, ens_mean_base, ens_mean_base |
| FWD | 1.001 | ens_top3 | 0.963 | svr, rf, svr, cat |

* **The procedure only matches the best fixed model for MID**, where it picks ens_mean_base three times.
  It is behind by 0.01 to 0.05 elsewhere. A single validation gameweek is too noisy to pick between close
  models, so the pick changes at almost every selection point, and it sometimes lands on a weak model such
  as dt.
* **ens_mean_base is the most reliable choice.** It is within about 0.01 of the best fixed model for every
  position, and it is the best for DEF (tied with lgbm and ens_top3) and MID.
* The best fixed model is only known in hindsight, so its MAE is a little optimistic.

## Outputs

All written to `models/`. The results files of steps 2 and 3 are appended to, and `compare.py` keeps the
latest run per model and fold. `deploy.py` and `predictor.py` rewrite their files.

| File | Written by | Contents |
|---|---|---|
| `{POS}_results.csv` | train.py | MAE and R² per model and fold |
| `ensemble_results.csv` | ensemble.py | MAE and R² per ensemble and fold, its weights, and each base model's MAE on the same fold |
| `deploy_results.csv` | deploy.py | MAE and R² of every model on every gameweek after its selection point |
| `predictions.csv` | predictor.py | predicted and actual xP per player and gameweek, with the model and selection point used |
| `performance.csv` | performance.py | MAE, RMSE, R², bias, rank correlation and dummy-baseline MAE per position and gameweek |
| `oof_{model}_{POS}_{fold}.csv` | both | validation predictions (`row` is the row number in `{pos}_train_folds.csv`) |
| `{model}_{POS}_{fold}.pkl` | both | fitted model, loadable with `joblib.load` (run from `src/`, or add it to `sys.path`) |

The `.pkl` and `oof_*.csv` files are not committed. Rebuild them with steps 2 to 4.

## Known issues

* Python sometimes crashes with a segmentation fault (exit code 139) after training lightgbm or xgboost.
  The crash is random and the same run passes when repeated, so `folds.sh` and `ensemble.sh` retry it up to
  three times.
* In `model_dispatcher.py`, the MID and FWD feature lists contain `threat_rolling_3` twice, so that feature is
  fed to the models twice. It was left as is so that results stay comparable with earlier runs.
* The source data has no target for gameweeks 22, 32 and 34: every player's xP is 0 there, while players
  still scored about 2.7 points on average. They are listed in `config.MISSING_TARGET_GWS`, and `data.load`
  leaves them out of training and of every score; `predictor.py` still predicts them. The source data's
  rolling xP features (`xP_rolling_3`) count those zeros, so `fix_missing_xp.py` writes
  `config.RAW_DATA_XP_FIXED`, a copy where xP and xP_25 are empty in those gameweeks and their rolling and
  ewm features skip them. It first checks that its rebuild reproduces the source features exactly.
  `prepare_data.py` runs it and reads that file, so the models use the rebuilt features. `data.load` warns
  if the gameweeks with no xP in the data differ from the config list.
* The raw data only has players who played, so `predictor.py` predicts (and `performance.py` scores) those
  players only.
