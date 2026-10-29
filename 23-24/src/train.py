"""Train one base model for one position at one selection point (fold) and record its validation scores.

Writes to ../models/:
  {POS}_results.csv         one row per run (appended)
  oof_{model}_{POS}_{fold}.csv  validation predictions
  {model}_{POS}_{fold}.pkl      fitted model

Usage: python train.py --fold 21 --model ridge --pos GK
"""

import argparse
import os

import joblib
import pandas as pd
from sklearn import metrics

import config
import data
import model_dispatcher


def run(fold, model, pos):
    df = data.load(pos)  # without the gameweeks that have no target

    # Time-based split: train on every gameweek before the fold, validate on the fold
    df_train = df[df.kfold < fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    x_train = data.features(df_train)
    y_train = df_train.xP
    x_valid = data.features(df_valid)
    y_valid = df_valid.xP

    reg = model_dispatcher.MODELS_BY_POSITION[pos][model]
    reg.fit(x_train, y_train)

    preds = reg.predict(x_valid)
    mae = metrics.mean_absolute_error(y_valid, preds)
    r2 = metrics.r2_score(y_valid, preds)

    row = pd.DataFrame(
        [{"pos": pos, "model": model, "fold": fold, "mae": mae, "r2": r2}]
    )
    path = f"{config.MODEL_OUTPUT}{pos}_results.csv"
    row.to_csv(path, mode="a", header=not os.path.exists(path), index=False)

    oof = pd.DataFrame(
        {"row": df_valid["row"], "fold": fold, "y": y_valid, "pred": preds}
    )
    oof.to_csv(f"{config.MODEL_OUTPUT}oof_{model}_{pos}_{fold}.csv", index=False)
    joblib.dump(reg, f"{config.MODEL_OUTPUT}{model}_{pos}_{fold}.pkl")

    print(f"Pos: {pos}, Model: {model}, Fold: {fold}, MAE: {mae:.4f}, R2: {r2:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fold", type=int, required=True, help="Fold (gameweek) to validate on"
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Model name from model_dispatcher"
    )
    parser.add_argument(
        "--pos",
        type=str.upper,
        required=True,
        choices=config.POSITIONS,
        help="Position",
    )
    args = parser.parse_args()
    if args.model not in model_dispatcher.MODELS_BY_POSITION[args.pos]:
        parser.error(
            f"Unknown model '{args.model}' for {args.pos}: "
            f"{list(model_dispatcher.MODELS_BY_POSITION[args.pos])}"
        )
    run(args.fold, args.model, args.pos)
