"""Load a position's fold file, with or without the gameweeks that have no target."""

import warnings

import config
import pandas as pd

DROP_COLS = ["kfold", "fpl_id", "round", "total_points"]  # never used as features


def missing_target_gws(df):
    """Gameweeks where every total_points is 0 or empty, i.e. the target is missing in the source data.
    """
    missing = df.groupby("kfold").total_points.apply(lambda y: (y.fillna(0) == 0).all())
    return set(missing[missing].index)


def load(pos, drop_missing_target=True):
    """Rows of {pos}_train_folds.csv; `row` is the row number in that file.

    By default the gameweeks in config.MISSING_TARGET_GWS are dropped, so they are never trained or
    scored on. Pass drop_missing_target=False to keep them, for example to predict them.
    """
    df = pd.read_csv(getattr(config, f"{pos}_TRAINING_FILE"))
    df["row"] = df.index
    found = missing_target_gws(df)
    if found != set(config.MISSING_TARGET_GWS):
        warnings.warn(
            f"{pos}: gameweeks with total_points = 0 for every player are {sorted(found)}, "
            f"but config.MISSING_TARGET_GWS is {config.MISSING_TARGET_GWS}",
            stacklevel=2,
        )
    if drop_missing_target:
        df = df[~df.kfold.isin(config.MISSING_TARGET_GWS)]
    return df


def features(df):
    """Model inputs: every column except the identifiers, the target and `row`."""
    return df.drop(columns=DROP_COLS + ["row"], errors="ignore")
