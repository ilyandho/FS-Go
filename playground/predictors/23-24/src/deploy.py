"""Score every model on the gameweeks it would be used for after each selection point.

At selection point s (config.VALID_FOLDS), each model was trained on the gameweeks before s and
validated on s by train.py or ensemble.py. The selected model is then used unchanged up to and
including the next selection point, or for config.HORIZON gameweeks after the last one (windows).
This script loads every model saved at s and predicts each of those gameweeks, so compare.py can
score both fixed models and the selection procedure.

The gameweeks in config.MISSING_TARGET_GWS have no target, so data.load leaves them out and they
are not scored.

Writes ../models/deploy_results.csv: pos, model, selected_at, gw, mae, r2, n

Usage: python deploy.py            # all positions
       python deploy.py --pos GK
"""

import argparse
import os

import compare
import config
import data
import ensemble  # noqa: F401  (EnsembleRegressor must be importable to load ensemble models)
import joblib
import pandas as pd
from sklearn import metrics

OUTPUT = f"{config.MODEL_OUTPUT}deploy_results.csv"


def windows(last_gw):
    """Selection point -> the gameweeks its selected model is used for: up to and including the
    next selection point (whose own model is selected on it), and HORIZON gameweeks after the last
    one. Every gameweek after the first selection point is covered, however far apart they are."""
    folds = sorted(config.VALID_FOLDS)
    ends = folds[1:] + [folds[-1] + config.HORIZON]
    return {s: list(range(s + 1, min(end, last_gw) + 1)) for s, end in zip(folds, ends)}


def run(pos, results):
    df = data.load(pos)  # without the gameweeks that have no target
    present = set(df.kfold)
    rows = []
    for s, gws in windows(df.kfold.max()).items():
        gws = [gw for gw in gws if gw in present]
        models = results[(results.pos == pos) & (results.fold == s)].label
        for name in models:
            path = f"{config.MODEL_OUTPUT}{name}_{pos}_{s}.pkl"
            if not os.path.exists(path):
                print(f"Missing {path}, skipped")
                continue
            model = joblib.load(path)
            for gw in gws:
                d = df[df.kfold == gw]
                preds = model.predict(data.features(d))
                rows.append(
                    {
                        "pos": pos,
                        "model": name,
                        "selected_at": s,
                        "gw": gw,
                        "mae": metrics.mean_absolute_error(d.total_points, preds),
                        "r2": metrics.r2_score(d.total_points, preds),
                        "n": len(d),
                    }
                )
        print(
            f"Pos: {pos}, selection point {s}: {len(models)} models on gameweeks {gws}"
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pos", type=str.upper, choices=config.POSITIONS, help="Only this position"
    )
    args = parser.parse_args()
    positions = [args.pos] if args.pos else config.POSITIONS

    results = compare.load_results()
    new = pd.concat([run(pos, results) for pos in positions], ignore_index=True)

    # Replace the rows of the positions just scored and keep the others
    if os.path.exists(OUTPUT):
        old = pd.read_csv(OUTPUT)
        new = pd.concat([old[~old.pos.isin(positions)], new], ignore_index=True)
    new.to_csv(OUTPUT, index=False)
    print(f"Wrote {len(new)} rows to {OUTPUT}")
