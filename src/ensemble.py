"""Combine base models into an ensemble for one position and fold, and record its scores.

Methods
  mean    equal-weight average of the base models
  median  median of the base models
  stack   non-negative weights fitted on the last --inner gameweeks before the fold
  top     equal-weight average of the --top-k models that score best on the last --inner gameweeks

Writes to ../models/:
  ensemble_results.csv          one row per run, with every base model's MAE on the same fold
  oof_{name}_{POS}_{fold}.csv   ensemble and base model predictions on the fold
  {name}_{POS}_{fold}.pkl       fitted EnsembleRegressor

Usage: python ensemble.py --fold 21 --pos GK --method mean --models svr cat xgb --name ens_top3
"""

import argparse
import os

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn import metrics
from sklearn.base import clone

import config
import model_dispatcher

DROP_COLS = ["kfold", "fpl_id", "round", "xP"]


class EnsembleRegressor:
    """Weighted mean (or median) of fitted base models. Picklable with joblib."""

    def __init__(self, models, weights=None, method="mean"):
        self.models = models  # dict: name -> fitted estimator
        self.weights = weights  # dict: name -> weight, or None for equal weights
        self.method = method

    def base_predictions(self, x):
        return pd.DataFrame({name: m.predict(x) for name, m in self.models.items()})

    def predict(self, x):
        preds = self.base_predictions(x)
        if self.method == "median":
            return preds.median(axis=1).to_numpy()
        if self.weights is None:
            return preds.mean(axis=1).to_numpy()
        w = np.array([self.weights[name] for name in preds.columns])
        return preds.to_numpy() @ w


def split_xy(df):
    return df.drop(columns=DROP_COLS), df.xP


MODELS_BY_POSITION = model_dispatcher.MODELS_BY_POSITION


def fit_models(pos, names, x, y):
    # clone: model_dispatcher holds single shared instances
    available = MODELS_BY_POSITION[pos.upper()]
    return {name: clone(available[name]).fit(x, y) for name in names}


def inner_holdout_preds(pos, names, df_train, inner):
    """Fit on training data before the last `inner` gameweeks and predict those gameweeks."""
    cutoff = df_train.kfold.max() - inner + 1
    df_fit = df_train[df_train.kfold < cutoff]
    df_hold = df_train[df_train.kfold >= cutoff]
    if df_fit.empty or df_hold.empty:
        raise ValueError(
            f"Not enough gameweeks before this fold for --inner {inner}; "
            "use a smaller --inner or a later fold."
        )

    x_fit, y_fit = split_xy(df_fit)
    x_hold, y_hold = split_xy(df_hold)
    models = fit_models(pos, names, x_fit, y_fit)
    preds = np.column_stack([models[n].predict(x_hold) for n in names])
    return preds, y_hold


def stack_weights(pos, names, df_train, inner):
    """Fit NNLS weights on the last `inner` gameweeks of the training data."""
    preds, y_hold = inner_holdout_preds(pos, names, df_train, inner)
    w, _ = nnls(preds, y_hold.to_numpy())
    if w.sum() == 0:  # degenerate case: fall back to equal weights
        w = np.ones(len(names))
    w = w / w.sum()
    return dict(zip(names, w))


def top_k_weights(pos, names, df_train, inner, k):
    """Equal weights on the k models with the lowest MAE on the last `inner` gameweeks."""
    preds, y_hold = inner_holdout_preds(pos, names, df_train, inner)
    mae = {n: metrics.mean_absolute_error(y_hold, preds[:, i]) for i, n in enumerate(names)}
    top = sorted(names, key=mae.get)[:k]
    return {n: (1 / len(top) if n in top else 0.0) for n in names}


def run(fold, pos, names, method, inner, top_k=3, name=None):
    training_file = getattr(config, f"{pos.upper()}_TRAINING_FILE")
    df = pd.read_csv(training_file)
    df["row"] = df.index

    # Time-based split, identical to train.py
    df_train = df[df.kfold < fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    if method == "stack":
        weights = stack_weights(pos, names, df_train.drop(columns="row"), inner)
    elif method == "top":
        weights = top_k_weights(pos, names, df_train.drop(columns="row"), inner, top_k)
    else:
        weights = None

    x_train, y_train = split_xy(df_train.drop(columns="row"))
    x_valid, y_valid = split_xy(df_valid.drop(columns="row"))

    ens = EnsembleRegressor(fit_models(pos, names, x_train, y_train), weights, method)
    base_preds = ens.base_predictions(x_valid)
    preds = ens.predict(x_valid)

    mae = metrics.mean_absolute_error(y_valid, preds)
    r2 = metrics.r2_score(y_valid, preds)
    tag = name or (f"ens_top{top_k}" if method == "top" else f"ens_{method}")

    row = {
        "model": tag,
        "pos": pos.upper(),
        "fold": fold,
        "mae": mae,
        "r2": r2,
        "base_models": "+".join(names),
        "weights": ""
        if weights is None
        else ";".join(f"{k}={v:.3f}" for k, v in weights.items()),
    }
    # Per-base-model MAE on the same fold, for a direct comparison
    for name in names:
        row[f"mae_{name}"] = metrics.mean_absolute_error(y_valid, base_preds[name])

    path = f"{config.MODEL_OUTPUT}ensemble_results.csv"
    # Rewrite rather than append: runs with different base models add different mae_* columns
    results = pd.DataFrame([row])
    if os.path.exists(path):
        results = pd.concat([pd.read_csv(path), results], ignore_index=True)
    results.to_csv(path, index=False)

    oof = pd.DataFrame(
        {"row": df_valid["row"], "fold": fold, "y": y_valid, "pred": preds}
    )
    oof = pd.concat([oof, base_preds.add_prefix("pred_")], axis=1)
    oof.to_csv(f"{config.MODEL_OUTPUT}oof_{tag}_{pos.upper()}_{fold}.csv", index=False)

    joblib.dump(ens, f"{config.MODEL_OUTPUT}{tag}_{pos.upper()}_{fold}.pkl")

    best_base = min(names, key=lambda n: row[f"mae_{n}"])
    print(
        f"Pos: {pos.upper()}, Fold: {fold}, {tag} MAE: {mae:.4f}, R2: {r2:.4f} "
        f"| best base {best_base}: {row[f'mae_{best_base}']:.4f}"
        + (f" | weights {row['weights']}" if weights else "")
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fold", type=int, required=True, help="Fold (gameweek) to validate on"
    )
    parser.add_argument(
        "--pos",
        type=str.upper,
        required=True,
        choices=list(MODELS_BY_POSITION),
        help="Position: GK, DEF, MID or FWD",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Base models to combine (default: every model of the position except dummy)",
    )
    parser.add_argument("--method", choices=["mean", "median", "stack", "top"], default="mean")
    parser.add_argument(
        "--inner",
        type=int,
        default=3,
        help="stack/top: number of gameweeks before the fold used to fit weights or rank models",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="top only: number of best models (ranked on the --inner gameweeks) to average",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Label for the results row and output files (default: ens_<method>)",
    )
    args = parser.parse_args()

    available_models = MODELS_BY_POSITION[args.pos]
    models = (
        args.models
        if args.models is not None
        else [m for m in available_models if m != "dummy"]
    )
    invalid_models = set(models) - set(available_models)
    if invalid_models:
        parser.error(f"Invalid models for {args.pos}: {sorted(invalid_models)}")

    # Call run() through the imported module so saved models reference ensemble.EnsembleRegressor
    # rather than __main__.EnsembleRegressor, and can be loaded from other scripts and notebooks.
    import ensemble

    ensemble.run(args.fold, args.pos, models, args.method, args.inner, args.top_k, args.name)
