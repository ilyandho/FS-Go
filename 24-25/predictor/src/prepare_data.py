"""Build one data file per position from the full 2024-25 player data.

Reads config.FULL_DATA: every player in the FPL pool each gameweek, also those who did not play,
with the xP features rebuilt without the missing gameweeks (written by
../../optimizer/prepare_data.py, which this runs first if the file is missing). Keeps each
position's rows and the columns its models use (model_dispatcher), plus the identifiers and the
target, with the identifiers renamed to the predictor's (element -> fpl_id, GK/DEF/MID/FWD).

Every player is kept, not only those who played: whether a player plays is only known after the
gameweek, so filtering on it would choose the rows to predict by the outcome.

Writes ../inputs/fg_data_{pos}.csv (read by create_folds.py)

Usage: python prepare_data.py
"""

import os
import subprocess
import sys

import config
import model_dispatcher
import pandas as pd

ID_COLS = ["fpl_id", "round", "position", "xP"]
# Only for choosing training rows (data.training_rows), never a model input
FILTER_COLS = ["minutes_rolling_1"]
NUM_COLS = {
    "GK": model_dispatcher.num_cols_gk,
    "DEF": model_dispatcher.num_cols_def,
    "MID": model_dispatcher.num_cols_mid,
    "FWD": model_dispatcher.num_cols_fwd,
}


def columns(pos):
    """Identifiers, target and model features of a position, without duplicates."""
    cols = ID_COLS + FILTER_COLS + model_dispatcher.cat_cols + NUM_COLS[pos]
    return list(dict.fromkeys(cols))


if __name__ == "__main__":
    if not os.path.exists(config.FULL_DATA):
        builder = os.path.join("..", "..", "optimizer", "prepare_data.py")
        subprocess.run([sys.executable, builder], check=True)

    raw = pd.read_csv(config.FULL_DATA, low_memory=False).rename(
        columns={"element": "fpl_id"}
    )
    raw["position"] = raw["position"].map(config.FULL_DATA_POSITIONS)
    for pos in config.POSITIONS:
        df = raw.loc[raw["position"] == pos, columns(pos)]
        path = f"../inputs/fg_data_{pos.lower()}.csv"
        df.to_csv(path, index=False)
        print(f"{pos}: {len(df)} rows, {df.shape[1]} columns -> {path}")
