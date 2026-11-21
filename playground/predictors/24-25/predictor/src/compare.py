"""Print a comparison table of every base model and ensemble, one table per position.

Follows the selection procedure in config.py: at each selection point (config.VALID_FOLDS) the model
with the lowest validation MAE is chosen, then used up to the next selection point (deploy.windows).

Reads ../models/{POS}_results.csv (train.py), ../models/ensemble_results.csv (ensemble.py) and
../models/deploy_results.csv (deploy.py). When a model was run more than once on a fold, only the
latest run counts.

Rows
  one per model, as if that model were always chosen
  "selected"  the procedure: the model with the lowest validation MAE at each selection point

Columns
  valid_mae, valid_r2    mean over the selection points (the scores the choice is made on)
  use_mae, use_r2        mean over the gameweeks after each selection point (the honest scores)
  mae_<fold>             validation MAE at each selection point, with --folds
  use_<first>-<last>     MAE on each window of gameweeks, with --folds

Usage
  python compare.py                 # all positions
  python compare.py --pos GK --folds
"""

import argparse
import os
import sys

import config
import pandas as pd

SELECTED = "selected"


def load_results():
    """Validation scores of every model at every selection point."""
    frames = []
    for pos in config.POSITIONS:
        path = f"{config.MODEL_OUTPUT}{pos}_results.csv"
        if os.path.exists(path):
            df = pd.read_csv(path)
            df["pos"] = pos  # older files have no pos column
            df["kind"] = "base"
            frames.append(df)
    path = f"{config.MODEL_OUTPUT}ensemble_results.csv"
    if os.path.exists(path):
        df = pd.read_csv(path)
        df["kind"] = "ensemble"
        frames.append(df)
    if not frames:
        sys.exit(
            f"No results in {config.MODEL_OUTPUT}. Run folds.sh and ensemble.sh first."
        )
    res = pd.concat(frames, ignore_index=True)
    if "base_models" not in res:
        res["base_models"] = None
    res = res.rename(columns={"model": "label"})
    res = res[["pos", "kind", "label", "fold", "mae", "r2", "base_models"]]
    res = res[res.fold.isin(config.VALID_FOLDS)]
    return res.drop_duplicates(["pos", "label", "fold"], keep="last")


def load_deploy():
    """Scores of every model on the gameweeks after each selection point (deploy.py)."""
    path = f"{config.MODEL_OUTPUT}deploy_results.csv"
    if not os.path.exists(path):
        sys.exit(f"No {path}. Run deploy.py first.")
    return pd.read_csv(path).rename(columns={"model": "label"})


def selections(res, pos):
    """Selection point -> the model with the lowest validation MAE there (dummy excluded)."""
    df = res[(res.pos == pos) & (res.label != "dummy")]
    return {s: g.loc[g.mae.idxmin(), "label"] for s, g in df.groupby("fold")}


def summarize(res, dep, pos, show_folds=False):
    v = res[res.pos == pos]
    d = dep[dep.pos == pos]
    picks = selections(res, pos)
    d_sel = pd.concat(
        [d[(d.selected_at == s) & (d.label == m)] for s, m in picks.items()]
    ).assign(label=SELECTED)
    d_all = pd.concat([d, d_sel])

    table = v.groupby("label").agg(
        kind=("kind", "first"), valid_mae=("mae", "mean"), valid_r2=("r2", "mean")
    )
    table.loc[SELECTED, "kind"] = "procedure"
    use = d_all.groupby("label").agg(use_mae=("mae", "mean"), use_r2=("r2", "mean"))
    table = table.join(use)

    if show_folds:
        per_fold = v.pivot_table(index="label", columns="fold", values="mae")
        per_fold.columns = [f"mae_{f}" for f in per_fold.columns]
        d_all = d_all.assign(
            window=d_all.groupby("selected_at").gw.transform(
                lambda g: f"use_{g.min()}-{g.max()}"
            )
        )
        per_window = d_all.pivot_table(index="label", columns="window", values="mae")
        table = table.join(per_fold).join(per_window[sorted(per_window.columns)])
    return table.sort_values(["use_mae", "valid_mae"]), picks


def print_table(res, dep, pos, show_folds=False):
    table, picks = summarize(res, dep, pos, show_folds)
    gws = sorted(dep[dep.pos == pos].gw.unique())
    print(
        f"\n=== {pos} === selection points {config.VALID_FOLDS}, "
        f"used for gameweeks {gws[0]}-{gws[-1]} ({len(gws)} gameweeks)"
    )
    print(table.to_string(float_format=lambda x: f"{x:.3f}", na_rep="-"))
    print("selected: " + ", ".join(f"gw{s} {m}" for s, m in picks.items()))
    ens = res[(res.pos == pos) & (res.kind == "ensemble")]
    for label, b in ens.groupby("label").base_models.first().items():
        print(f"{label}: {str(b).replace('+', ' ')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--pos", type=str.upper, choices=config.POSITIONS, help="Only this position"
    )
    parser.add_argument(
        "--folds",
        action="store_true",
        help="Also show each selection point and each window of gameweeks",
    )
    args = parser.parse_args()

    pd.set_option("display.width", 250)
    res, dep = load_results(), load_deploy()
    for pos in [args.pos] if args.pos else config.POSITIONS:
        print_table(res, dep, pos, args.folds)
