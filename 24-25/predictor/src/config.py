"""Paths and fold settings shared by every script, for the 2024-25 season. Paths are relative to src/."""

POSITIONS = ["GK", "DEF", "MID", "FWD"]

# Folds are gameweeks. Each fold trains on every gameweek before it and validates on that gameweek.
# VALID_FOLDS are the model selection points: at gameweek 21, train on gameweeks 1-20, validate on 21,
# and use the selected model for the next HORIZON gameweeks (22-26). Then repeat the selection at 26, 31
# and 36. The last window is cut short by the end of the season (37-38).
VALID_FOLDS = [21, 26, 31, 36]
HORIZON = 5

# Gameweeks with no target: the source data has xP = 0 for every player in them. They are left out of
# training and of every score (data.load). predictor.py still predicts them.
MISSING_TARGET_GWS = [22, 32, 34]
assert not set(MISSING_TARGET_GWS) & set(VALID_FOLDS), "a selection point has no target"

# Raw data (players who played, one row per player and gameweek), downloaded by prepare_data.py, and
# the copy fix_missing_xp.py writes from it with the xP features rebuilt. The models are built from
# RAW_DATA_XP_FIXED.
DATA_URL = (
    "https://raw.githubusercontent.com/ilyandho/FPL-Optimal-Transfer/refs/heads/"
    "FantasyGo/Prediction%20with%20Affine%20PNN/data/rolled_player_data_24-25.csv"
)
RAW_DATA = "../inputs/rolled_player_data_24-25.csv"
RAW_DATA_XP_FIXED = "../inputs/rolled_player_data_24-25_xp_fixed.csv"

GK_TRAINING_FILE = "../inputs/gk_train_folds.csv"
DEF_TRAINING_FILE = "../inputs/def_train_folds.csv"
MID_TRAINING_FILE = "../inputs/mid_train_folds.csv"
FWD_TRAINING_FILE = "../inputs/fwd_train_folds.csv"

MODEL_OUTPUT = "../models/"
