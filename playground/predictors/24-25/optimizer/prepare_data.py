"""Build the optimizer's player data: every player in the FPL pool each gameweek, with xP fixed.

The beta estimator notebooks need every player each gameweek, also those who did not play, in the
same order as the *_binary columns of inputs/league_selections_df.csv, with the league's selection
shares (league_select_probs). The predictor's data only has players who played, so it cannot be
used here. The source is FPL-Optimal-Transfer's "FPL Data/rolled_data_23_24.csv", which despite
its name holds the 2024-25 season.

Like the predictor's data, the source has xP = 0 in the gameweeks without a target (22, 32 and 34),
and its xP rolling and ewm features count those zeros. ../predictor/src/fix_missing_xp.py sets xP to
empty there and rebuilds those features without them, after checking that it reproduces the
source's own features exactly.

Reads  inputs/rolled_data_24-25.csv           (downloaded the first time, about 90 MB)
Writes inputs/rolled_data_24-25_xp_fixed.csv  (same rows and columns)

Usage (from optimizer/): python prepare_data.py
                         python prepare_data.py --fresh   # download again
"""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "predictor", "src"))
import fix_missing_xp  # noqa: E402

DATA_URL = (
    "https://raw.githubusercontent.com/ilyandho/FPL-Optimal-Transfer/FantasyGo/"
    "FPL%20Data/rolled_data_23_24.csv"
)
HERE = os.path.dirname(os.path.abspath(__file__))
RAW_DATA = os.path.join(HERE, "inputs", "rolled_data_24-25.csv")
RAW_DATA_XP_FIXED = os.path.join(HERE, "inputs", "rolled_data_24-25_xp_fixed.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh", action="store_true", help="Download the source again")
    args = parser.parse_args()

    if args.fresh or not os.path.exists(RAW_DATA):
        print(f"Downloading {DATA_URL}")
        source = pd.read_csv(DATA_URL, index_col=0, low_memory=False)
        source.to_csv(RAW_DATA, index=False)

    df = pd.read_csv(RAW_DATA, low_memory=False)
    fixed = fix_missing_xp.fix(df, id_col="element")
    fixed.to_csv(RAW_DATA_XP_FIXED, index=False)
    print(f"Wrote {RAW_DATA_XP_FIXED}: {len(fixed)} rows, {fixed.shape[1]} columns")
