# FS-Go: per-position xP models

Predicts a Fantasy Premier League player's expected points (`xP`) for the next gameweek, with a separate
model for each position (GK, DEF, MID, FWD). Nine base models and two ensembles compete at seven
model selection points in one season (`config.SEASON`, now 2023-24), and each selected model is scored on the gameweeks it
would have been used for.

## Layout

```
inputs/                     data
  sources/                               downloads cached by src/build_rolled_data.py (not committed)
  rolled_player_data_23-24.csv           2023-24 data in the 2024-25 column format (src/build_rolled_data.py)
  rolled_player_data_23-24_xp_fixed.csv  same, with the xP features rebuilt (src/fix_missing_xp.py)
  fg_data_{pos}.csv              columns kept per position (notebooks/get data.ipynb)
  {pos}_train_folds.csv          same, plus the kfold column (src/create_folds.py)
models/                     everything the pipeline writes (see "Outputs")
notebooks/
  get data.ipynb            builds the raw data (first run) and writes fg_data_{pos}.csv
  model_comparison.ipynb    charts and tests comparing the models, and MAE and R² per gameweek
src/
  config.py                 season, positions, selection points, gameweeks without a target, paths
  build_rolled_data.py      builds the 2023-24 raw data from vaastav, football-data and Understat
  data.py                   loads a position's fold file, without the gameweeks that have no target
  fix_missing_xp.py         writes a copy of the raw data with the xP rolling features rebuilt without them
  model_dispatcher.py       features and the nine base models per position
  create_folds.py           adds the fold column
  train.py                  trains one base model at one selection point
  folds.sh                  runs train.py for every model, selection point and position
  ensemble.py               builds one ensemble at one selection point
  ensemble.sh               runs ensemble.py for every position and selection point
  deploy.py                 scores every saved model on the gameweeks after its selection point
  compare.py                prints the comparison table per position
  predictor.py              predicts every player for gameweeks 5 to 38 with the selected models
  performance.py            scores those predictions per gameweek and position
  rank_features.py          optional: ranks candidate features (writes features.py; not used by the models yet)
```

## Setup

Python 3.11. All scripts run from `src/` and use paths relative to it.

```
pip install -r requirements.txt
```

## Run

```
# 1. Data: run notebooks/get data.ipynb (builds the 2023-24 data the first time, about a minute)

# 2. Base models: create the folds and train every model at every selection point and position
cd src
bash folds.sh

# 3. Ensembles
bash ensemble.sh

# 4. Score the saved models on the gameweeks after each selection point
python deploy.py

# 5. Comparison table per position
python compare.py
python compare.py --pos GK --folds     # one position, with each selection point and window

# 6. Predictions for gameweeks 5 to 38 with the selected model of each window
python predictor.py

# 7. Performance of those predictions per gameweek and position
python performance.py
python performance.py --pos GK
```

`folds.sh ridge svr` trains only the named models. `bash ensemble.sh mean stack top` runs more ensemble
methods, and `TOP_K=4 BASE_MODELS="ridge svr cat" bash ensemble.sh` changes the model sets. Rerun
`deploy.py` after either script.

## How models are evaluated

A fold is a gameweek. Models are chosen at seven **selection points: gameweeks 5, 10, 15, 20, 25, 30 and
35** (2023-24; gameweek 26 has no target, so the later points sit one gameweek earlier than 2024-25's).

1. At gameweek 5, every model trains on gameweeks 1 to 4 and is validated on gameweek 5.
2. The model with the lowest validation MAE is selected. The dummy baseline is never selected.
3. The selected model is used, unchanged, up to and including the next selection point: 6 to 10.
4. The same is repeated at every later point. The last model (35) is used for 36 to 38, the end of
   the season.

`predictor.py` also predicts gameweek 5 with the model selected at 5. Those predictions were used to
select that model, so they are marked `in_selection` and left out of the scores.

The selection points and the 5-gameweek horizon are set in `src/config.py`. Scores are mean absolute
error (MAE, lower is better) and R². The score that matters is the MAE on the gameweeks after each
selection point, 22 to 38, because those gameweeks played no part in choosing the model.

Two ensembles compete with the base models at each selection point:

* `ens_mean_base`: the mean of ridge, hgb, lgbm, xgb and svr, the same for every position.
* `ens_top3`: the mean of the three base models with the lowest MAE on the three gameweeks before the
  selection point. It only uses information available at that point.

Gameweeks 22, 32 and 34 (2024-25) and 26 (2023-24) have no target in the source data, so they are left out
of training and of every score (see "Known issues"). The three gameweeks before a selection point are counted without them, so
before gameweek 36 they are 31, 33 and 35.

`ensemble.py` also supports `median` and `stack` (non-negative weights fitted on the three gameweeks
before the selection point).

## Results

MAE on the gameweeks after each selection point (6 to 38), from `python compare.py`. Gameweek 26 is left
out because the source data has no xP for it (see "Known issues"), which leaves 32 gameweeks.

| Position | Selection procedure | Best fixed model | Its MAE | Picks at gameweeks 5, 10, 15, 20, 25, 30, 35 |
|---|---|---|---|---|
| GK | 1.045 | cat | 1.055 | hgb, ens_top3, hgb, ens_top3, rf, rf, ridge |
| DEF | 0.912 | cat | 0.915 | cat, ens_mean_base, ens_top3, cat, lgbm, ens_top3, lgbm |
| MID | 0.874 | ens_mean_base / cat | 0.870 | ens_mean_base, cat, lgbm, cat, cat, ens_mean_base, ens_mean_base |
| FWD | 1.109 | ens_top3 | 1.081 | ens_mean_base, lgbm, ens_mean_base, cat, lgbm, cat, svr |

* **The procedure beats every fixed model for GK and DEF**, is 0.004 behind the best for MID and 0.03 behind
  for FWD. A single validation gameweek is too noisy to pick between close models, so the pick changes at
  almost every selection point.
* **cat and ens_mean_base are the most reliable choices.** One of them is within 0.03 of the best fixed model
  for every position.
* The best fixed model is only known in hindsight, so its MAE is a little optimistic.

## Outputs

All written to `models/`. The results files of steps 2 and 3 are appended to, and `compare.py` keeps the
latest run per model and fold. `deploy.py` and `predictor.py`
rewrite their files.

| File | Written by | Contents |
|---|---|---|
| `{POS}_results.csv` | train.py | MAE and R² per model and fold |
| `ensemble_results.csv` | ensemble.py | MAE and R² per ensemble and fold, its weights, and each base model's MAE on the same fold |
| `deploy_results.csv` | deploy.py | MAE and R² of every model on every gameweek after its selection point |
| `predictions.csv` | predictor.py | predicted and actual xP per player and gameweek, with the model and selection point used |
| `performance.csv` | performance.py | MAE, RMSE, R², bias, rank correlation and dummy-baseline MAE per position and gameweek |
| `oof_{model}_{POS}_{fold}.csv` | both | validation predictions (`row` is the row number in `{pos}_train_folds.csv`) |
| `{model}_{POS}_{fold}.pkl` | both | fitted model, loadable with `joblib.load` (run from `src/`, or add it to `sys.path`) |
| `feature_ranking.csv` | rank_features.py | feature scores per position |

The `.pkl` and `oof_*.csv` files are not committed. Rebuild them with steps 2 to 4.

## Known issues

* Python sometimes crashes with a segmentation fault (exit code 139) after training lightgbm or xgboost.
  The crash is random and the same run passes when repeated, so `folds.sh` and `ensemble.sh` retry it up to
  three times.
* In `model_dispatcher.py`, the MID and FWD feature lists contain `threat_rolling_3` twice, so that feature is
  fed to the models twice. It was left as is so that results stay comparable with earlier runs.
* The source data has no target for gameweeks 22, 32 and 34: every player's xP is 0 there, while players
  still scored about 2.7 points on average. They are listed in `config.MISSING_TARGET_GWS`, and `data.load`
  leaves them out of training and of every score; `predictor.py` still predicts them. The rolling xP features
  of the gameweeks after them (`xP_rolling_3`) are computed in the source data and still include those
  zeros. `data.load` warns if the gameweeks with all-zero xP in the data differ from the config list.
  `python fix_missing_xp.py` writes `config.RAW_DATA_XP_FIXED`, a copy where xP (and xP_25 in 2024-25) is
  empty in those gameweeks and their rolling and ewm features skip them. It first checks that its
  rebuild reproduces the source features exactly. `notebooks/get data.ipynb` runs it and reads
  that file, so the models use the rebuilt features. In 2023-24 the gameweek without xP is 26.
* FPL-Optimal-Transfer's `rolled_data_23_24.csv` is a copy of the 2024-25 data, so
  `build_rolled_data.py` rebuilds 2023-24 from public sources in the same format. Differences from 2024-25:
  * no tackles or other defensive actions (FPL started publishing them in 2025-26), so `tackles_rolling_3`
    and `cbitr_ewm_probability` are not in the feature lists;
  * Elo is computed from the Premier League results of the five seasons before (ClubElo's API was
    unavailable), so it is on a different scale from the 2024-25 Elo;
  * `selected_by_percent` is estimated from `selected` (each manager owns 15 players);
  * Understat players are matched to FPL players with FPL-Optimal-Transfer's id_dict, and the rest on
    their appearances (same match, minutes within 15), ties broken by name.
  The results above are from 2024-25, with tackles and cbitr.
* `features.py` (from `rank_features.py`) holds a ranked feature list per position, but `model_dispatcher.py`
  does not use it yet.
