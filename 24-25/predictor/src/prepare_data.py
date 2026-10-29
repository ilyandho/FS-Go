"""Build one data file per position from the raw 2024-25 data.

1. Downloads config.DATA_URL to config.RAW_DATA, unless the file is already there.
2. Runs fix_missing_xp.py to write config.RAW_DATA_XP_FIXED, unless that file is already there.
3. Keeps each position's rows and the columns its models use (model_dispatcher), plus the
   identifiers and the target.

Writes ../inputs/fg_data_{pos}.csv (read by create_folds.py)

Usage: python prepare_data.py           # reuse the raw files when present
       python prepare_data.py --fresh   # download and rebuild them
"""

import argparse
import os
import subprocess
import sys

import pandas as pd

import config
import model_dispatcher

ID_COLS = ["fpl_id", "round", "position", "xP"]
NUM_COLS = {
    "GK": model_dispatcher.num_cols_gk,
    "DEF": model_dispatcher.num_cols_def,
    "MID": model_dispatcher.num_cols_mid,
    "FWD": model_dispatcher.num_cols_fwd,
}


def columns(pos):
    """Identifiers, target and model features of a position, without duplicates."""
    cols = ID_COLS + model_dispatcher.cat_cols + NUM_COLS[pos]
    return list(dict.fromkeys(cols))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fresh", action="store_true", help="Download and rebuild the raw files"
    )
    args = parser.parse_args()

    if args.fresh or not os.path.exists(config.RAW_DATA_XP_FIXED):
        if args.fresh or not os.path.exists(config.RAW_DATA):
            print(f"Downloading {config.DATA_URL}")
            pd.read_csv(config.DATA_URL, low_memory=False).to_csv(
                config.RAW_DATA, index=False
            )
        subprocess.run([sys.executable, "fix_missing_xp.py"], check=True)

    raw = pd.read_csv(config.RAW_DATA_XP_FIXED, low_memory=False)
    for pos in config.POSITIONS:
        df = raw.loc[raw["position"] == pos, columns(pos)]
        path = f"../inputs/fg_data_{pos.lower()}.csv"
        df.to_csv(path, index=False)
        print(f"{pos}: {len(df)} rows, {df.shape[1]} columns -> {path}")
