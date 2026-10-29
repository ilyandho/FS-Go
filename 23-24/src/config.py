"""Paths and fold settings shared by every script. Paths are relative to src/."""

POSITIONS = ["GK", "DEF", "MID", "FWD"]

# Season of the raw data: "23-24" (built by build_rolled_data.py) or "24-25" (downloaded).
SEASON = "23-24"

# Folds are gameweeks. Each fold trains on every gameweek before it and validates on that gameweek.
# VALID_FOLDS are the model selection points: at gameweek 21, train on gameweeks 1-20, validate on 21,
# and use the selected model for the next HORIZON gameweeks (22-26). Then repeat the selection at 26, 31
# and 36. The last window is cut short by the end of the season (37-38).
# 2023-24 has no target in gameweek 26, so its points are moved one gameweek earlier.
VALID_FOLDS = {"23-24": [20, 25, 30, 35], "24-25": [21, 26, 31, 36]}[SEASON]
HORIZON = 5

# Gameweeks with no target: the source data has xP = 0 for every player in them. They are left out of
# training and of every score (data.load). predictor.py still predicts them.
MISSING_TARGET_GWS = {"23-24": [26], "24-25": [22, 32, 34]}[SEASON]
assert not set(MISSING_TARGET_GWS) & set(VALID_FOLDS), "a selection point has no target"

GK_TRAINING_FILE = "../inputs/gk_train_folds.csv"
DEF_TRAINING_FILE = "../inputs/def_train_folds.csv"
MID_TRAINING_FILE = "../inputs/mid_train_folds.csv"
FWD_TRAINING_FILE = "../inputs/fwd_train_folds.csv"

MODEL_OUTPUT = "../models/"

# Raw data in the 2024-25 column format, and the copy fix_missing_xp.py writes from it with the xP
# features rebuilt. The models are built from RAW_DATA_XP_FIXED (via notebooks/get data.ipynb).
RAW_DATA = f"../inputs/rolled_player_data_{SEASON}.csv"
RAW_DATA_XP_FIXED = f"../inputs/rolled_player_data_{SEASON}_xp_fixed.csv"
# The same for every player with a fixture, also those who did not play (build_rolled_data.py). The
# models train on RAW_DATA (players who played); predictor.py predicts these rows.
ALL_PLAYERS_DATA = f"../inputs/rolled_player_data_{SEASON}_all_players.csv"
ALL_PLAYERS_DATA_XP_FIXED = f"../inputs/rolled_player_data_{SEASON}_all_players_xp_fixed.csv"
