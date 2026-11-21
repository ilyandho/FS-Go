import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn import (
    compose,
    dummy,
    ensemble,
    impute,
    linear_model,
    preprocessing,
    svm,
    tree,
)
from sklearn.pipeline import make_pipeline


def linear_prep(num_cols, cat_cols):
    return compose.ColumnTransformer(
        [
            (
                "num",
                make_pipeline(
                    impute.SimpleImputer(strategy="median"),
                    preprocessing.StandardScaler(),
                ),
                num_cols,
            ),
            ("cat", preprocessing.OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )


def tree_prep(num_cols, cat_cols):
    return compose.ColumnTransformer(
        [
            ("num", "passthrough", num_cols),
            (
                "cat",
                preprocessing.OrdinalEncoder(
                    handle_unknown="use_encoded_value", unknown_value=-1
                ),
                cat_cols,
            ),
        ]
    )


## FEATS
# Tackles (and cbitr, built from them) are not published for 2023-24, so they are not used
cat_cols = [
    "team",
    "opponent_team",
    "was_home",
]

num_cols_gk = [
    "elo",
    "elo_h",
    "elo_a",
    "opp_elo",
    "opp_elo_h",
    "opp_elo_a",
    "team_h_difficulty",
    "team_a_difficulty",
    "win_prob",
    "value",
    "selected_by_percent",
    "transfers_in",
    "transfers_out",
    "xP_rolling_3",
    "xGChain_rolling_3",
    "xGBuildup_rolling_3",
    "minutes_rolling_3",
    "starts_rolling_3",
    "expected_goals_conceded_rolling_3",
    "clean_sheets_rolling_3",
    "saves_rolling_3",
    "ict_index_rolling_3",
    "influence_rolling_3",
    "creativity_rolling_3",
    "threat_rolling_3",
]  # numeric features

num_cols_def = [
    # "xP",
    # "elo",
    "elo_h",
    "elo_a",
    # "opp_elo",
    # "opp_elo_h",
    # "opp_elo_a",
    # "team_h_difficulty",
    # "team_a_difficulty",
    "win_prob",
    "value",
    "selected_by_percent",
    "transfers_in",
    "transfers_out",
    "xP_rolling_3",
    "xGChain_rolling_3",
    "xGBuildup_rolling_3",
    "expected_goals_rolling_3",
    "expected_assists_rolling_3",
    "expected_goal_involvements_rolling_3",
    "minutes_rolling_3",
    "starts_rolling_3",
    "expected_goals_conceded_rolling_3",
    "clean_sheets_rolling_3",
    # "saves_rolling_3",
    "ict_index_rolling_3",
    "influence_rolling_3",
    "creativity_rolling_3",
    "threat_rolling_3",
]

# num_cols_def = [
#     "elo",
#     "elo_h",
#     "elo_a",
#     "opp_elo",
#     "opp_elo_h",
#     "opp_elo_a",
#     "team_h_difficulty",
#     "team_a_difficulty",
#     "win_prob",
#     "value",
#     "selected_by_percent",
#     "transfers_in",
#     "transfers_out",
#     "xP_rolling_3",
#     "xGChain_rolling_3",
#     "xGBuildup_rolling_3",
#     "expected_goals_rolling_3",
#     "expected_assists_rolling_3",
#     "expected_goal_involvements_rolling_3",
#     "minutes_rolling_3",
#     "starts_rolling_3",
#     "expected_goals_conceded_rolling_3",
#     "clean_sheets_rolling_3",
#     "saves_rolling_3",
#     "ict_index_rolling_3",
#     "influence_rolling_3",
#     "creativity_rolling_3",
#     "threat_rolling_3",
# ]  # numeric features

num_cols_mid = [
    "elo",
    "elo_h",
    "elo_a",
    "opp_elo",
    "opp_elo_h",
    "opp_elo_a",
    "team_h_difficulty",
    "team_a_difficulty",
    "win_prob",
    "value",
    "selected_by_percent",
    "transfers_in",
    "transfers_out",
    "xP_rolling_3",
    "xGChain_rolling_3",
    "xGBuildup_rolling_3",
    "expected_goals_rolling_3",
    "expected_assists_rolling_3",
    "expected_goal_involvements_rolling_3",
    "minutes_rolling_3",
    "starts_rolling_3",
    "ict_index_rolling_3",
    "influence_rolling_3",
    "threat_rolling_3",
    "creativity_rolling_3",
    "threat_rolling_3",
]  # numeric features

num_cols_fwd = [
    "elo",
    "elo_h",
    "elo_a",
    "opp_elo",
    "opp_elo_h",
    "opp_elo_a",
    "team_h_difficulty",
    "team_a_difficulty",
    "win_prob",
    "value",
    "selected_by_percent",
    "transfers_in",
    "transfers_out",
    "xP_rolling_3",
    "xGChain_rolling_3",
    "xGBuildup_rolling_3",
    "expected_goals_rolling_3",
    "expected_assists_rolling_3",
    "expected_goal_involvements_rolling_3",
    "minutes_rolling_3",
    "starts_rolling_3",
    "ict_index_rolling_3",
    "influence_rolling_3",
    "threat_rolling_3",
    "creativity_rolling_3",
    "threat_rolling_3",
]  # numeric features


## Models
def make_models(num_cols):
    """Fresh, unfitted models for one position's numeric feature list."""
    return {
        "dummy": dummy.DummyRegressor(strategy="mean"),  # floor
        "ridge": make_pipeline(
            linear_prep(num_cols, cat_cols), linear_model.Ridge(alpha=1.0)
        ),
        "dt": make_pipeline(
            tree_prep(num_cols, cat_cols),
            tree.DecisionTreeRegressor(
                max_depth=6, min_samples_leaf=30, random_state=42
            ),
        ),
        "rf": make_pipeline(
            tree_prep(num_cols, cat_cols),
            ensemble.RandomForestRegressor(
                n_estimators=300, min_samples_leaf=10, n_jobs=-1, random_state=42
            ),
        ),
        "hgb": make_pipeline(
            tree_prep(num_cols, cat_cols),
            ensemble.HistGradientBoostingRegressor(random_state=42),
        ),
        "lgbm": make_pipeline(
            tree_prep(num_cols, cat_cols),
            lgb.LGBMRegressor(
                n_estimators=300,
                learning_rate=0.03,
                num_leaves=15,
                min_child_samples=30,
                subsample=0.8,
                subsample_freq=1,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1,
            ),
        ),
        "xgb": make_pipeline(
            tree_prep(num_cols, cat_cols),
            xgb.XGBRegressor(
                n_estimators=300,
                learning_rate=0.03,
                max_depth=3,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="rmse",
                random_state=42,
            ),
        ),
        "cat": make_pipeline(
            tree_prep(num_cols, cat_cols),
            CatBoostRegressor(
                iterations=500,
                learning_rate=0.03,
                depth=4,
                random_seed=42,
                verbose=0,
                allow_writing_files=False,  # no catboost_info/ folder
            ),
        ),
        "svr": make_pipeline(linear_prep(num_cols, cat_cols), svm.SVR(C=1.0)),
    }


models_gk = make_models(num_cols_gk)
models_def = make_models(num_cols_def)
models_mid = make_models(num_cols_mid)
models_fwd = make_models(num_cols_fwd)

MODELS_BY_POSITION = {
    "GK": models_gk,
    "DEF": models_def,
    "MID": models_mid,
    "FWD": models_fwd,
}
