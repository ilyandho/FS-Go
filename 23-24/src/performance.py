"""Performance of the selected models per gameweek and position, from predictor.py's predictions.

Columns
  model, selected_at  the model that predicted the gameweek and the selection point it was chosen at
  n                   players predicted
  mae, rmse, r2       errors against the actual xP
  bias                mean of (prediction - actual): above 0 means over-prediction
  spearman            rank correlation of prediction and actual: how well players are ordered
  dummy_mae           MAE of the dummy baseline (mean of the training targets) on the same gameweek
  skill               1 - mae / dummy_mae: share of the baseline error removed

Only players who played are scored, as in training and deploy.py.
Gameweeks without a target (config.MISSING_TARGET_GWS) are listed with empty scores. Gameweek 21 is marked
in_selection: its predictions were used to choose the model, so it is left out of the totals.

Writes ../models/performance.csv.

Usage: python performance.py            # every position, then MAE per gameweek and position
       python performance.py --pos GK
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn import metrics

import compare
import config

PREDICTIONS = f"{config.MODEL_OUTPUT}predictions.csv"
DEPLOY = f"{config.MODEL_OUTPUT}deploy_results.csv"
OUTPUT = f"{config.MODEL_OUTPUT}performance.csv"


def scores(g):
    """Scores of one position and gameweek; empty when the gameweek has no target."""
    row = {
        "model": g.model.iloc[0],
        "selected_at": g.selected_at.iloc[0],
        "in_selection": bool(g.in_selection.iloc[0]),
        "n": len(g),
    }
    if g.xP.notna().all():
        err = g.pred - g.xP
        row |= {
            "mae": err.abs().mean(),
            "rmse": np.sqrt((err**2).mean()),
            "r2": metrics.r2_score(g.xP, g.pred),
            "bias": err.mean(),
            "spearman": g.pred.corr(g.xP, method="spearman"),
        }
    return row


def per_gameweek(pred, dep):
    rows = [
        {"pos": pos, "round": gw, **scores(g)}
        for pos in config.POSITIONS
        for gw, g in pred[pred.pos == pos].groupby("round")
    ]
    table = pd.DataFrame(rows)
    # Dummy baseline from deploy.py, at the same selection point as the selected model
    dummy = dep[dep.model == "dummy"].rename(
        columns={"gw": "round", "mae": "dummy_mae"}
    )
    table = table.merge(
        dummy[["pos", "round", "selected_at", "dummy_mae"]],
        on=["pos", "round", "selected_at"],
        how="left",
    )
    # Gameweek 21 is a selection point, so its dummy score comes from the validation results
    valid = compare.load_results()
    valid = valid[valid.label == "dummy"].set_index(["pos", "fold"]).mae
    sel = table.in_selection
    table.loc[sel, "dummy_mae"] = [
        valid.get((p, r)) for p, r in zip(table.pos[sel], table["round"][sel])
    ]
    table["skill"] = 1 - table["mae"] / table["dummy_mae"]
    return table


def totals(table):
    scored = table[table.mae.notna() & ~table.in_selection]
    cols = ["mae", "rmse", "r2", "bias", "spearman", "dummy_mae", "skill"]
    out = scored.groupby("pos", sort=False)[cols].mean()
    out.insert(0, "gameweeks", scored.groupby("pos", sort=False).size())
    return out


def fmt(x):
    return f"{x:.3f}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pos", type=str.upper, choices=config.POSITIONS, help="Only this position"
    )
    args = parser.parse_args()
    for path in [PREDICTIONS, DEPLOY]:
        if not os.path.exists(path):
            sys.exit(f"No {path}. Run deploy.py and predictor.py first.")

    pd.set_option("display.width", 250)
    pred = pd.read_csv(PREDICTIONS)
    pred = pred[pred["played"]]  # scored like training and deploy.py: players who played
    table = per_gameweek(pred, pd.read_csv(DEPLOY))
    table.to_csv(OUTPUT, index=False)

    shown = table[table.pos == args.pos] if args.pos else table
    for pos, g in shown.groupby("pos", sort=False):
        print(f"\n=== {pos} ===")
        g = g.drop(columns="pos").set_index("round")
        g["in_selection"] = g.in_selection.map({True: "yes", False: ""})
        print(g.to_string(float_format=fmt, na_rep="-"))

    print("\n=== Mean per gameweek (scored gameweeks only) ===")
    print(totals(shown).to_string(float_format=fmt))
    if not args.pos:
        print("\n=== MAE per gameweek and position ===")
        wide = table.pivot(index="round", columns="pos", values="mae")[config.POSITIONS]
        print(wide.to_string(float_format=fmt, na_rep="-"))
    print(
        "\n'-' = no xP target in the source data. in_selection = used to choose the model."
    )
    print(f"Wrote {OUTPUT}")
