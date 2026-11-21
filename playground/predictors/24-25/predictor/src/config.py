"""Paths and fold settings shared by every script, for the 2024-25 season. Paths are relative to src/."""

POSITIONS = ["GK", "DEF", "MID", "FWD"]

# Folds are gameweeks. Each fold trains on every gameweek before it and validates on that gameweek.
# VALID_FOLDS are the model selection points: at gameweek 21, train on gameweeks 1-20, validate on 21,
# and use the selected model up to and including the next selection point (22-26). The first point,
# 5, makes gameweeks 6-38 all predicted out of sample. After the last point the model is used for
# HORIZON gameweeks, cut short by the end of the season (37-38).
VALID_FOLDS = [5, 10, 15, 21, 26, 31, 36]
HORIZON = 5

# Gameweeks with no target: the source data has xP = 0 for every player in them. They are left out of
# training and of every score (data.load). predictor.py still predicts them.
MISSING_TARGET_GWS = [22, 32, 34]
assert not set(MISSING_TARGET_GWS) & set(VALID_FOLDS), "a selection point has no target"

# The models are built from FULL_DATA: every player in the FPL pool each gameweek, also those who did
# not play, with the xP features rebuilt without the missing gameweeks. It is written by
# ../../optimizer/prepare_data.py. Players who did not play must be in the data, or each gameweek's
# rows (and so its predictions) would be chosen by the outcome being predicted.
FULL_DATA = "../../optimizer/inputs/rolled_data_24-25_xp_fixed.csv"
FULL_DATA_POSITIONS = {
    "Goalkeeper": "GK",
    "Defender": "DEF",
    "Midfielder": "MID",
    "Forward": "FWD",
}

# Training uses only players with minutes in the previous 3 gameweeks (data.training_rows), so long
# absentees don't dilute the fit; validation and prediction rows keep every player.
ACTIVE_COL = "minutes_rolling_3"

GK_TRAINING_FILE = "../inputs/gk_train_folds.csv"
DEF_TRAINING_FILE = "../inputs/def_train_folds.csv"
MID_TRAINING_FILE = "../inputs/mid_train_folds.csv"
FWD_TRAINING_FILE = "../inputs/fwd_train_folds.csv"

MODEL_OUTPUT = "../models/"
