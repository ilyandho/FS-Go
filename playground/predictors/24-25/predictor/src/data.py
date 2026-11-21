"""Load a position's fold file, with or without the gameweeks that have no target."""

import warnings

import pandas as pd

import config

DROP_COLS = ["kfold", "fpl_id", "round", "xP", "minutes_rolling_1"]  # never used as features


def missing_target_gws(df):
    """Gameweeks where every xP is 0 or empty, i.e. the target is missing in the source data.

    The raw source has 0 there; rolled_player_data_24-25_xp_fixed.csv (fix_missing_xp.py) has empty.
    """
    missing = df.groupby("kfold").xP.apply(lambda y: (y.fillna(0) == 0).all())
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
            f"{pos}: gameweeks with xP = 0 for every player are {sorted(found)}, "
            f"but config.MISSING_TARGET_GWS is {config.MISSING_TARGET_GWS}",
            stacklevel=2,
        )
    if drop_missing_target:
        df = df[~df.kfold.isin(config.MISSING_TARGET_GWS)]
    return df


def training_rows(df):
    """The rows models are trained on: players with minutes in the previous 3 gameweeks
    (config.ACTIVE_COL). That window is only complete from gameweek 4, so gameweeks 2-3 use the
    previous gameweek's minutes and gameweek 1, with no history, keeps everyone. Validation and
    prediction rows are never filtered, so every player still gets a prediction."""
    active = df[config.ACTIVE_COL] > 0
    active |= (df.kfold.between(2, 3)) & (df["minutes_rolling_1"] > 0)
    active |= df.kfold == 1
    return df[active]


def features(df):
    """Model inputs: every column except the identifiers, the target and `row`."""
    return df.drop(columns=DROP_COLS + ["row"], errors="ignore")
