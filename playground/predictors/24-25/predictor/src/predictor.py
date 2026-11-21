"""Predict every player's xP for gameweeks 21 to 38 with the model selected for each gameweek.

At each selection point s (config.VALID_FOLDS), the model with the lowest validation MAE at s is
selected (compare.selections) and used up to and including the next selection point
(deploy.windows). The first selection point comes before the first window, so it is predicted by
its own selected model. That model was trained on the gameweeks before it, but its predictions there
were also used to select it, so those rows are flagged with in_selection = True.

Writes ../models/predictions.csv, one row per player and gameweek:
  pos, round, fpl_id, name, team, selected_at, model, pred, xP, in_selection
xP is the actual target, left empty for the gameweeks without one (config.MISSING_TARGET_GWS).
name and team come from the raw data file, when it is present.

Run folds.sh, ensemble.sh and deploy.py first.

Usage: python predictor.py            # all positions
       python predictor.py --pos GK
"""

import argparse
import os

import compare
import config
import data
import deploy
import ensemble  # noqa: F401  (EnsembleRegressor must be importable to load ensemble models)
import joblib
import pandas as pd
from sklearn import metrics

OUTPUT = f"{config.MODEL_OUTPUT}predictions.csv"
PLAYER_DATA = config.FULL_DATA


def schedule(last_gw):
    """Gameweek -> selection point whose selected model predicts it."""
    first = min(config.VALID_FOLDS)
    plan = {first: first}
    for s, gws in deploy.windows(last_gw).items():
        plan.update({gw: s for gw in gws})
    return plan


def run(pos, results):
    df = data.load(pos, drop_missing_target=False)  # predict every gameweek
    picks = compare.selections(results, pos)
    missing = set(config.MISSING_TARGET_GWS)
    out = []
    for gw, s in schedule(df.kfold.max()).items():
        model_name = picks[s]
        model = joblib.load(f"{config.MODEL_OUTPUT}{model_name}_{pos}_{s}.pkl")
        d = df[df.kfold == gw]
        out.append(
            pd.DataFrame(
                {
                    "pos": pos,
                    "round": gw,
                    "fpl_id": d.fpl_id.to_numpy(),
                    "selected_at": s,
                    "model": model_name,
                    "pred": model.predict(data.features(d)),
                    "xP": d.xP.to_numpy() if gw not in missing else float("nan"),
                    "in_selection": gw == s,
                }
            )
        )
    out = pd.concat(out, ignore_index=True)
    print(
        f"Pos: {pos}, picks: "
        + ", ".join(f"gw{s} {m}" for s, m in picks.items())
        + f" | {len(out)} predictions for gameweeks {out['round'].min()}-{out['round'].max()}"
    )
    return out


def add_names(pred):
    """Add player and team names from the player data, when it is present."""
    if not os.path.exists(PLAYER_DATA):
        print(f"No {PLAYER_DATA}: names left out")
        return pred
    raw = pd.read_csv(
        PLAYER_DATA,
        usecols=["element", "round", "name", "team", "value", "total_points"],
        low_memory=False,
    ).rename(columns={"element": "fpl_id"})
    raw = raw.drop_duplicates(["fpl_id", "round"])
    named = pred.merge(raw, on=["fpl_id", "round"], how="left", validate="many_to_one")
    cols = [
        "pos",
        "round",
        "fpl_id",
        "value",
        "name",
        "team",
        "total_points",
        "selected_at",
        "model",
        "pred",
    ]
    return named[cols + ["xP", "in_selection"]]


def report(pred):
    """MAE of the predictions on the gameweeks that have a target and were not used to select."""
    scored = pred[pred.xP.notna() & ~pred.in_selection]
    by_gw = scored.groupby(["pos", "round"]).apply(
        lambda g: metrics.mean_absolute_error(g.xP, g.pred)
    )
    summary = by_gw.groupby("pos").agg(["mean", "size"])
    summary.columns = ["mae (mean per gameweek)", "gameweeks"]
    print(summary.round(3).to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pos", type=str.upper, choices=config.POSITIONS, help="Only this position"
    )
    args = parser.parse_args()
    positions = [args.pos] if args.pos else config.POSITIONS

    results = compare.load_results()
    new = add_names(pd.concat([run(pos, results) for pos in positions]))

    # Replace the rows of the positions just predicted and keep the others
    if os.path.exists(OUTPUT):
        old = pd.read_csv(OUTPUT)
        new = pd.concat([old[~old.pos.isin(positions)], new], ignore_index=True)
    new.to_csv(OUTPUT, index=False)
    print(f"Wrote {len(new)} rows to {OUTPUT}")
    report(new[new.pos.isin(positions)])
