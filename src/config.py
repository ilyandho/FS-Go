"""Paths and fold settings shared by every script. Paths are relative to src/."""

POSITIONS = ["GK", "DEF", "MID", "FWD"]

# Folds are gameweeks. Each fold trains on every gameweek before it and validates on that gameweek.
VALID_FOLDS = [21, 26, 31, 36]  # used to compare models and choose ensembles
TEST_FOLDS = [37, 38]  # held out: only used to report the final scores

GK_TRAINING_FILE = "../inputs/gk_train_folds.csv"
DEF_TRAINING_FILE = "../inputs/def_train_folds.csv"
MID_TRAINING_FILE = "../inputs/mid_train_folds.csv"
FWD_TRAINING_FILE = "../inputs/fwd_train_folds.csv"

MODEL_OUTPUT = "../models/"
