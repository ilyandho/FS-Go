"""Write a copy of the raw data with the xP rolling features rebuilt without the missing gameweeks.

The source data has no xP for config.MISSING_TARGET_GWS (22, 32 and 34): every player has xP = 0
there. Its lagged xP features (xP_rolling_k, xP_k_ewm and the same for xP_25) count those zeros as
real scores, so they are too low for the gameweeks that follow. This script:

1. sets xP and xP_25 to empty in the missing gameweeks,
2. rebuilds the lagged features the way the source builds them, but with the missing gameweeks
   taken out of each player's timeline: the window before gameweek 23 is 19, 20 and 21.

How the source builds them (checked by --check, which reproduces the source values exactly):
  * one value per player and gameweek, 0 when the player has no row that gameweek
  * shifted by one gameweek, so a row only sees earlier gameweeks; the first gameweek is 0
  * xP_rolling_k: mean of the last k gameweeks, 0 until k earlier gameweeks exist
  * xP_k_ewm: exponentially weighted mean with span k (adjust=False)

Reads  config.RAW_DATA
Writes config.RAW_DATA_XP_FIXED (same rows and columns)

Usage: python fix_missing_xp.py           # write the fixed file
       python fix_missing_xp.py --check   # only check the rebuild against the source values
"""

import argparse

import numpy as np
import pandas as pd

import config

SOURCE = config.RAW_DATA
OUTPUT = config.RAW_DATA_XP_FIXED
STATS = [
    "xP",
    "xP_25",
]  # xP_25 is a near-copy of xP and is missing in the same gameweeks
WINDOWS = [1, 3, 5]


def lagged_features(df, stat, skip_gws=(), id_col="fpl_id"):
    """Rolling and ewm features of `stat` for every player and gameweek, ignoring `skip_gws`.

    Each gameweek sees the gameweeks before it, minus skip_gws. A row in a skipped gameweek sees the
    same history as the gameweek after it would.
    """
    all_rounds = sorted(df["round"].unique())
    kept = [r for r in all_rounds if r not in set(skip_gws)]
    grid = (
        df[df["round"].isin(kept)]
        .pivot_table(index="round", columns=id_col, values=stat, aggfunc="sum")
        .reindex(kept)
        .fillna(0.0)  # no row for the player that gameweek
    )
    out = {}
    for k in WINDOWS:
        current = {
            f"{stat}_rolling_{k}": grid.rolling(k, min_periods=k).mean(),
            f"{stat}_{k}_ewm": grid.ewm(span=k, adjust=False).mean(),
        }
        for name, f in current.items():
            # Value up to the last kept gameweek before each round; 0 for the first round
            out[name] = f.reindex(all_rounds).ffill().shift(1).fillna(0.0)
    return pd.concat({n: f.stack() for n, f in out.items()}, axis=1).reset_index()


def rebuild(df, skip_gws, id_col="fpl_id"):
    """df with the lagged xP features recomputed, ignoring skip_gws."""
    out = df.copy()
    for stat in stats(df):
        feats = lagged_features(df, stat, skip_gws, id_col)
        cols = [c for c in feats.columns if c not in ("round", id_col)]
        merged = out[[id_col, "round"]].merge(
            feats, on=[id_col, "round"], how="left", validate="many_to_one"
        )
        out[cols] = merged[cols].to_numpy()
    return out


def stats(df):
    """The STATS columns df has (some files have no xP_25)."""
    return [s for s in STATS if s in df.columns]


def check(df, id_col="fpl_id"):
    """Rebuild without skipping anything and compare with the source values."""
    rebuilt = rebuild(df, skip_gws=[], id_col=id_col)
    worst = 0.0
    for stat in stats(df):
        for k in WINDOWS:
            for col in [f"{stat}_rolling_{k}", f"{stat}_{k}_ewm"]:
                diff = np.nanmax(np.abs(rebuilt[col] - df[col]))
                worst = max(worst, diff)
                print(f"{col:<18} max abs difference {diff:.2e}")
    return worst


def fix(df, id_col="fpl_id", check_only=False):
    """df with xP (and xP_25) empty in config.MISSING_TARGET_GWS and their lagged features rebuilt.

    First checks that the rebuild reproduces df's own lagged features, and stops if it does not.
    Also used by ../../optimizer/prepare_data.py, whose data has every player and `element` ids.
    """
    worst = check(df, id_col)
    if worst > 1e-6:
        raise SystemExit(
            f"Rebuild does not match the source (max difference {worst:.3g})"
        )
    print("Rebuild matches the source data.")
    if check_only:
        return df

    missing = config.MISSING_TARGET_GWS
    fixed = df.copy()
    fixed.loc[fixed["round"].isin(missing), stats(df)] = np.nan
    fixed = rebuild(fixed, missing, id_col)

    changed = {
        c: int((~np.isclose(fixed[c], df[c], equal_nan=True)).sum())
        for c in fixed.columns
        if c.startswith(tuple(f"{s}_" for s in stats(df))) and c not in STATS
    }
    print(f"Set {', '.join(stats(df))} to empty in gameweeks {missing}")
    print("Rows changed per feature:", changed)
    return fixed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Only check the rebuild")
    args = parser.parse_args()

    df = pd.read_csv(SOURCE, low_memory=False)
    fixed = fix(df, check_only=args.check)
    if not args.check:
        fixed.to_csv(OUTPUT, index=False)
        print(f"Wrote {OUTPUT}")
