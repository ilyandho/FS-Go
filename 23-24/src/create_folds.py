"""Add a kfold column (the gameweek) to a position's data file.

Reads  ../inputs/fg_data_{pos}.csv      (written by notebooks/get data.ipynb)
Writes ../inputs/{pos}_train_folds.csv  (read by train.py and ensemble.py)

Usage: python create_folds.py --pos GK
"""

import argparse

import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pos", type=str, required=True, help="Position: GK, DEF, MID or FWD"
    )
    args = parser.parse_args()
    pos = args.pos.lower()

    print(f"Creating folds for {pos} position...")
    df = pd.read_csv(f"../inputs/fg_data_{pos}.csv")
    df["kfold"] = df["round"]  # one fold per gameweek
    df.to_csv(f"../inputs/{pos}_train_folds.csv", index=False)
