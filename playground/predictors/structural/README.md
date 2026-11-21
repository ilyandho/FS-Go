# Structural xP model

Predicts each FPL player's points for every gameweek by predicting each scoring component (minutes, team goals
conceded, goals, assists, bonus, saves, cards) and combining them with the FPL scoring rules. Every model for
gameweek *gw* is fitted on gameweeks before *gw* only. Works for 2023-24 and 2024-25 (`config.SEASONS`); the
scoring rules are the same in both and reproduce `total_points` exactly for every position.

## Layout

```
src/
  config.py               seasons, positions, paths(season)
  build_data.py           every player with a fixture, per gameweek, with lagged features   (once per season)
  build_availability.py   pre-deadline FPL status, chance of playing, set-piece orders       (once per season)
  structural.py           the model: load, fit, walk-forward prediction, scoring; also a CLI
notebooks/
  structural.ipynb        runs the model for one season (SEASON in the first code cell) and evaluates it:
                          scores by position, per-gameweek RMSE, calibration, expected points per component,
                          a sample of players' predicted vs actual points, one-gameweek example, xP data check
data/
  {season}/players.csv          build_data.py
  {season}/availability.csv     build_availability.py
  sources/                      download caches: vaastav_{season}/ (merged_gw, fixtures, teams, players_raw,
                                understat/, id_dict), football-data/ (results and odds, shared by seasons),
                                players_raw_snapshots_{season}/ (players_raw.csv at each pre-deadline commit)
outputs/
  {season}/structural_predictions.csv                       structural.py and the notebook
  {season}/structural_predictions_history_{seasons}.csv     the same, trained with earlier seasons (--history)
```

The folder is self-contained: it reads nothing from `23-24/` or `24-25/`. `build_data.py` is adapted from
`23-24/src/build_rolled_data.py` and gives the same columns (checked: the rebuilt 2023-24 file matches that
pipeline's all-players file in every column the model uses, and the model's predictions are unchanged).

## Run

Python 3.11 with the packages in `23-24/requirements.txt`. Scripts run from `src/` and take `--season`
(default `23-24`).

```
cd src
python build_data.py --season 24-25           # downloads once (about a minute), then uses data/sources/
python build_availability.py --season 24-25   # needs build_data.py's fixtures.csv; downloads ~35 snapshots once
python structural.py --season 24-25           # all positions, GW 5-38 -> ../outputs/24-25/structural_predictions.csv
```

`structural.py` options: `--history 23-24` (add earlier seasons, see below), `--pos GK DEF`, `--gws 21` or `--gws 20-38`, `--targets realised` (train goals, assists
and goals conceded on realised counts instead of xG, xA and xGA), `--no-availability` (leave the FPL status out of
the minutes model), `--no-set-pieces` (leave the set-piece duties out of the goal and assist models), `--out PATH`.

Pre-deadline inputs from the `players_raw.csv` snapshots (`build_availability.py`): the FPL status and chance of
playing feed the minutes model; the penalty, corner and direct free-kick taker orders feed the goal model
(first-choice and backup penalty taker, first-choice free-kick taker) and the assist model (first-choice corner and
free-kick taker).

### Training on earlier seasons too

`--history 23-24` (`walk_forward(..., history=["23-24"])`, `HISTORY` in the notebook) adds every gameweek of the
listed seasons to the training data of each gameweek. Predictions then start at GW1 (there is history before it) and
are written to `structural_predictions_history_23-24.csv`, next to the single-season file. For 2024-25:

| 2024-25, GW5-38 | 24-25 only | with 23-24 |
|---|---|---|
| Every player: RMSE / R² / Spearman | 1.877 / 0.337 / 0.732 | 1.871 / 0.341 / 0.735 |
| Players who played: RMSE / R² / Spearman | 2.740 / 0.108 / 0.395 | 2.733 / 0.112 / 0.401 |
| GW1-4, every player: RMSE / R² / Spearman | not possible | 1.984 / 0.288 / 0.704 |

Most of the gain is early in the season (GW5-10 RMSE 1.896 to 1.876) and for midfielders and forwards; the last 10
gameweeks of 2023-24 give nearly all of it. 2023-24 has no earlier season here, so to give both seasons the same
treatment, add "22-23" to `config.SEASONS` and build its data.

From Python:

```python
import structural
data = structural.load("24-25")                            # load once
preds = structural.walk_forward(gws=range(5, 39), data=data)
preds_h = structural.walk_forward(gws=range(1, 39), data=data, history=["23-24"])  # with 2023-24 as well
structural.score(preds, ("pred",), played_only=False)      # RMSE, R2, Spearman, MAE by position
```

To add a season: add it to `config.SEASONS` (with any gameweeks that have no target), check that the scoring rules
in `structural.SCORING` still hold (`structural.points` must reproduce `total_points`), then run the three scripts.
2025-26 added defensive contribution points, so it needs a new component before it will reproduce.

## Output columns

- `pred`: expected points, including the chance of not playing. Pick teams on this.
- `pred_if_plays`: expected points if the player plays.
- `p_play`, `p60`: probability of playing at all, and of playing 60+ minutes.
- `lam_ga`, `p_cs`: expected goals conceded by the player's team, and the clean-sheet probability.
- `e_app`, `e_cs`, `e_gc`, `e_goals`, `e_assists`, `e_bonus`, `e_saves`, `e_other`: expected points per
  component; they sum to `pred`.
- `season`, `has_target` (False in a gameweek without a target), and `total_points`, `minutes`, `played`: what
  actually happened, for scoring.

## Results (gameweeks 5-38, walk-forward)

| | 2023-24 | 2024-25 |
|---|---|---|
| Every player with a fixture: RMSE / R² / Spearman | 1.854 / 0.341 / 0.721 | 1.877 / 0.337 / 0.732 |
| Players who played: RMSE / R² / Spearman | 2.850 / 0.109 / 0.404 | 2.740 / 0.108 / 0.395 |

The set-piece features improve RMSE in both seasons (every player: 1.857 to 1.854 in 2023-24, 1.881 to 1.877 in
2024-25), mostly for midfielders and forwards. First-choice penalty takers' goal points are much better predicted
(2024-25: 1.50 predicted against 1.50 actual per 60+ appearance, from 1.24), but players without penalty duty are now
slightly under-predicted (0.38 against 0.42).

## Metric

RMSE. Squared error is minimised by the mean, which is what the optimiser needs; MAE is minimised by the median
and favours predictions that ignore hauls. Spearman is also reported, since picking a team depends on ranking
players.

## Assumptions and known gaps

- The minutes state is independent of the match result, and a player on for 60+ minutes is treated as on for the
  whole match (team clean sheet = player clean sheet). Midfield and forward clean sheets are under-predicted as a
  result (about 4-6 points of probability): a player subbed off after 60 minutes before a goal still gets one.
- Availability snapshots are a median of 44 hours (2023-24) and 39 hours (2024-25) old at the deadline, and over a
  week for GW5, 21, 22, 27, 30 (2023-24) and GW4, 8, 21, 23, 30, 33 (2024-25), so late team news there is missed.
- 2024-25: 39 players (315 appearances) have no Understat match, so their Understat features (shots, key passes,
  xG, xA) are 0; FPL's own xG and xA features still cover them.
- Set-piece orders are FPL's listed takers at the snapshot, not who actually took the kick; about 18 first-choice
  penalty takers are listed per gameweek, so a few teams list none.
- FPL's same-gameweek `xP` in the source data was recorded after the match (see the notebook's data check), so it
  is not used. It is missing in GW26 (2023-24) and GW22, 32, 34 (2024-25), where it is left empty.
