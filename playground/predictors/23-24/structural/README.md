# Structural xP model

Predicts each FPL player's points for every gameweek by predicting each scoring component (minutes, team goals
conceded, goals, assists, bonus, saves, cards) and combining them with the FPL scoring rules. Every model for
gameweek *gw* is fitted on gameweeks before *gw* only.

## Layout

```
src/
  config.py               season, positions, paths
  build_availability.py   pre-deadline FPL status and chance of playing per player and gameweek (run once)
  structural.py           the model: load, fit, walk-forward prediction, scoring; also a CLI
notebooks/
  structural.ipynb        runs the model and evaluates it (scores by position, per-gameweek RMSE,
                          calibration, expected points per component, one-gameweek example, xP data check)
inputs/
  availability_{season}.csv               written by build_availability.py
  sources/players_raw_snapshots_{season}/ its download cache (players_raw.csv at each pre-deadline commit)
outputs/
  structural_predictions_{season}.csv     written by structural.py and the notebook
```

Shared inputs are read from `../inputs/` (that is, `23-24/inputs/`), not copied:
`rolled_player_data_{season}_all_players_xp_fixed.csv` (every player with a fixture) and
`sources/vaastav_{season}/merged_gw.csv` and `fixtures.csv`. They are built by `23-24/src/build_rolled_data.py`
and `fix_missing_xp.py`; see `23-24/README.md`.

## Run

Python 3.11 with the packages in `23-24/requirements.txt`. Scripts run from `src/`.

```
cd src
python build_availability.py      # once; downloads ~37 snapshots from GitHub the first time, then uses the cache
python structural.py              # all positions, GW 5-38 -> ../outputs/structural_predictions_23-24.csv
```

Options: `--pos GK DEF`, `--gws 21` or `--gws 20-38`, `--targets realised` (train goals, assists and goals
conceded on realised counts instead of xG, xA and xGA), `--no-availability` (leave the FPL status out of the
minutes model), `--out PATH`.

From Python:

```python
import structural
data = structural.load()                                   # load once
preds = structural.walk_forward(gws=range(5, 39), data=data)
structural.score(preds, ("pred",), played_only=False)      # RMSE, R2, Spearman, MAE by position
```

## Output columns

- `pred`: expected points, including the chance of not playing. Pick teams on this.
- `pred_if_plays`: expected points if the player plays.
- `p_play`, `p60`: probability of playing at all, and of playing 60+ minutes.
- `lam_ga`, `p_cs`: expected goals conceded by the player's team, and the clean-sheet probability.
- `e_app`, `e_cs`, `e_gc`, `e_goals`, `e_assists`, `e_bonus`, `e_saves`, `e_other`: expected points per
  component; they sum to `pred`.
- `total_points`, `minutes`, `played`: what actually happened, for scoring.

## Metric

RMSE. Squared error is minimised by the mean, which is what the optimiser needs; MAE is minimised by the median
and favours predictions that ignore hauls. Spearman is also reported, since picking a team depends on ranking
players.

## Assumptions and known gaps

- The minutes state is independent of the match result, and a player on for 60+ minutes is treated as on for the
  whole match (team clean sheet = player clean sheet). Midfield clean sheets are under-predicted as a result
  (22% against 27%): a midfielder subbed off after 60 minutes before a goal still gets one.
- Availability snapshots are a median of 44 hours old at the deadline, up to 10 days for GW5, 21, 22, 27 and 30,
  so late team news in those weeks is missed.
- FPL's same-gameweek `xP` in the source data was recorded after the match (see the notebook's data check), so it
  is not used.
