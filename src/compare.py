"""Print a comparison table of every base model and ensemble, one table per position.

Reads ../models/{POS}_results.csv (train.py) and ../models/ensemble_results.csv (ensemble.py).
When a model was run more than once on a fold, only the latest run counts.

Columns
  valid_mae, valid_r2  mean over config.VALID_FOLDS (use these to choose a model)
  test_mae, test_r2    mean over config.TEST_FOLDS (held out; report only)
  mae_<fold>           per-fold MAE, with --folds

Usage
  python compare.py                 # all positions
  python compare.py --pos GK --folds
  python compare.py --rank GK 3     # print the 3 best base models for GK (used by ensemble.sh)
"""

import argparse
import os
import sys

import pandas as pd

import config


def load_results():
    frames = []
    for pos in config.POSITIONS:
        path = f"{config.MODEL_OUTPUT}{pos}_results.csv"
        if os.path.exists(path):
            df = pd.read_csv(path)
            df["pos"] = pos  # older files have no pos column
            df["kind"] = "base"
            df["label"] = df["model"]
            frames.append(df)
    path = f"{config.MODEL_OUTPUT}ensemble_results.csv"
    if os.path.exists(path):
        df = pd.read_csv(path)
        df["kind"] = "ensemble"
        df["label"] = df["model"] + " [" + df["base_models"].str.replace("+", " ") + "]"
        frames.append(df)
    if not frames:
        sys.exit(f"No results in {config.MODEL_OUTPUT}. Run folds.sh and ensemble.sh first.")
    res = pd.concat(frames, ignore_index=True)[["pos", "kind", "label", "fold", "mae", "r2"]]
    return res.drop_duplicates(["pos", "label", "fold"], keep="last")


def summarize(res, pos, show_folds=False):
    df = res[res.pos == pos]
    out = []
    for split, folds in [("valid", config.VALID_FOLDS), ("test", config.TEST_FOLDS)]:
        sub = df[df.fold.isin(folds)]
        agg = sub.groupby("label").agg(mae=("mae", "mean"), r2=("r2", "mean"), n=("fold", "nunique"))
        agg.columns = [f"{split}_{c}" for c in agg.columns]
        out.append(agg)
    table = pd.concat(out, axis=1)
    table.insert(0, "kind", df.groupby("label").kind.first())
    if show_folds:
        per_fold = df.pivot_table(index="label", columns="fold", values="mae")
        per_fold.columns = [f"mae_{f}" for f in per_fold.columns]
        table = table.join(per_fold)
    return table.sort_values(["valid_mae", "test_mae"])


def rank_base_models(res, pos, k):
    table = summarize(res, pos)
    table = table[(table.kind == "base") & (table.index != "dummy")]
    complete = table[table.valid_n == len(config.VALID_FOLDS)]
    return complete.index[:k].tolist()


def print_table(table, pos):
    n_valid, n_test = len(config.VALID_FOLDS), len(config.TEST_FOLDS)
    print(f"\n=== {pos} === validation folds {config.VALID_FOLDS}, test folds {config.TEST_FOLDS}")
    shown = table.copy()
    incomplete = (shown.valid_n < n_valid) | (shown.test_n.fillna(0) < n_test)
    shown.index = [f"{lab} *" if bad else lab for lab, bad in zip(shown.index, incomplete)]
    shown = shown.drop(columns=["valid_n", "test_n"])
    print(shown.to_string(float_format=lambda v: f"{v:.3f}", na_rep="-"))
    best = table[table.valid_n == n_valid].index
    if len(best):
        print(f"best on validation: {best[0]}")
    if incomplete.any():
        print("* missing some folds; its means cover fewer gameweeks")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--pos", type=str.upper, choices=config.POSITIONS, help="Only this position")
    parser.add_argument("--folds", action="store_true", help="Also show the MAE of every fold")
    parser.add_argument(
        "--rank", nargs=2, metavar=("POS", "K"),
        help="Print the K best base models of POS by validation MAE, space separated",
    )
    args = parser.parse_args()

    pd.set_option("display.width", 250)
    res = load_results()
    if args.rank:
        print(" ".join(rank_base_models(res, args.rank[0].upper(), int(args.rank[1]))))
        sys.exit()
    for pos in [args.pos] if args.pos else config.POSITIONS:
        print_table(summarize(res, pos, args.folds), pos)
