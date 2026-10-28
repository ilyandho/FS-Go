"""Classify the raw columns and rank the usable (pre-match) features per position.

Writes
  ../models/feature_ranking.csv : every candidate feature, per position, with scores
  features.py                   : ordered, de-duplicated feature lists per position

Usage
  python rank_features.py                  # rank on gameweeks before the first CV fold
  python rank_features.py --max-round 38   # rank on all gameweeks (leaks into CV)
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression

DATA_URL = (
    "https://raw.githubusercontent.com/ilyandho/FPL-Optimal-Transfer/refs/heads/"
    "FantasyGo/Prediction%20with%20Affine%20PNN/data/rolled_player_data_24-25.csv"
)
LOCAL_COPY = Path("../inputs/rolled_player_data_24-25.csv")
POSITIONS = ["GK", "DEF", "MID", "FWD"]
TARGET = "xP"

# ---------------------------------------------------------------------------
# Column classification. Checked against the data (see the notes on each group).
# ---------------------------------------------------------------------------
IDENTIFIERS = [
    "fpl_id", "round", "name", "fpl_name", "understat_id", "fixture",
    "kickoff_time", "date", "season", "modified", "opponent_team_name",
    "team_h", "team_a",  # home/away team ids: redundant with team, opponent_team, was_home
    "anytime_odds", "two_or_more", "hatrick",  # raw odds strings; *_prob and *_logit carry them
]

CATEGORICAL = ["team", "opponent_team", "was_home"]  # "position" is constant within a position

# Known only after the match: using them leaks the result into the prediction.
MATCH_OUTCOMES = [
    "assists", "bonus", "bps", "clean_sheets", "creativity", "expected_assists",
    "expected_goal_involvements", "expected_goals", "expected_goals_conceded",
    "goals_conceded", "goals_scored", "ict_index", "influence", "minutes", "own_goals",
    "penalties_missed", "penalties_saved", "red_cards", "saves", "starts",
    "team_a_score", "team_h_score", "threat", "total_points", "yellow_cards",
    "shots", "xG", "xA", "key_passes", "npg", "npxG", "xGChain", "xGBuildup",
    "interceptions", "recoveries", "blocks", "clearances", "tackles",
    "clearances_blocks_interceptions", "cbit", "cbitr",
    "pts_bonus",         # equals total_points in ~90% of rows
    "form",              # its change tracks THIS gameweek's points (r=0.61), not last week's
    "opp_clean_sheets",  # equals "own team failed to score in this match" in ~88% of rows
]

TARGET_COPIES = ["xP_25"]  # identical to xP in ~92% of rows

# Everything else is treated as known before kickoff: fixture context, odds, Elo,
# price/ownership/transfers at the deadline, season-to-date totals (verified to
# exclude the current match) and the *_rolling_k / *_k_ewm lags (verified shifted).


def load(path):
    if path:
        return pd.read_csv(path, low_memory=False)
    if not LOCAL_COPY.exists():
        print(f"Downloading source data to {LOCAL_COPY} ...")
        pd.read_csv(DATA_URL, low_memory=False).to_csv(LOCAL_COPY, index=False)
    return pd.read_csv(LOCAL_COPY, low_memory=False)


def family(col):
    """Base stat a feature is derived from, e.g. 'threat_rolling_3' -> 'threat'."""
    col = re.sub(r"_rolling_\d+$", "", col)
    col = re.sub(r"_\d+_ewm$", "", col)
    return {"xP_25": "xP"}.get(col, col)  # xP_25 is a near-copy of xP


def rank_position(df, numeric, per_family=1, min_abs_corr_dup=0.97, seed=42):
    x = df[numeric].astype(float)
    y = df[TARGET].astype(float)
    keep = x.columns[x.notna().mean() > 0.8]
    keep = [c for c in keep if x[c].nunique() > 1]
    x = x[keep]

    spearman = x.corrwith(y, method="spearman")
    mi = pd.Series(
        mutual_info_regression(x.fillna(x.median()), y, random_state=seed), index=keep
    )
    out = pd.DataFrame({
        "spearman": spearman,
        "abs_spearman": spearman.abs(),
        "mutual_info": mi,
    })
    # Combined score: mean of the two percentile ranks (robust to scale differences).
    out["score"] = (out["abs_spearman"].rank(pct=True) + out["mutual_info"].rank(pct=True)) / 2
    out = out.sort_values("score", ascending=False)

    # Walk down the ranking. A feature is redundant if its base stat already has
    # `per_family` variants kept (e.g. threat_rolling_3 after threat_5_ewm), or if it
    # is ~collinear with any kept feature.
    corr = x[out.index].corr(method="spearman").abs()
    kept, redundant_with = [], {}
    for c in out.index:
        fam = family(c)
        same_fam = [k for k in kept if family(k) == fam]
        dup = same_fam[0] if len(same_fam) >= per_family else None
        dup = dup or next((k for k in kept if corr.loc[c, k] >= min_abs_corr_dup), None)
        if dup is None:
            kept.append(c)
        else:
            redundant_with[c] = dup
    out["redundant_with"] = pd.Series(redundant_with)
    out["family"] = [family(c) for c in out.index]
    out["n_rows"] = len(df)
    return out


def write_features_py(lists, max_round, path="features.py"):
    lines = [
        '"""Pre-match features per position, most relevant first.',
        "",
        f"Generated by rank_features.py (ranked on gameweeks < {max_round}).",
        "Only the best window of each base stat is kept, and near-duplicates",
        "(|Spearman| >= 0.97 with a higher-ranked feature) are removed.",
        "Match outcomes, `form`, `opp_clean_sheets` and `xP_25` are excluded as leakage.",
        '"""',
        "",
        f"CATEGORICAL = {CATEGORICAL!r}",
        "",
        "NUMERIC = {",
    ]
    for pos, cols in lists.items():
        lines.append(f'    "{pos}": [')
        lines += [f'        "{c}",' for c in cols]
        lines.append("    ],")
    lines += [
        "}",
        "",
        "",
        "def top(pos, n=None):",
        '    """Numeric features for a position; the first n if given."""',
        "    cols = NUMERIC[pos.upper()]",
        "    return cols if n is None else cols[:n]",
        "",
    ]
    Path(path).write_text("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", help="Path to the raw rolled player CSV (default: download)")
    parser.add_argument(
        "--max-round", type=int, default=10,
        help="Rank on gameweeks < this value. Default 10 = before the first CV fold, so "
             "feature selection does not see any validation gameweek.",
    )
    parser.add_argument("--top", type=int, default=40, help="Features kept per position in features.py")
    parser.add_argument("--per-family", type=int, default=1,
                        help="Windows (rolling/ewm variants) kept per base stat")
    args = parser.parse_args()

    df = load(args.data)
    excluded = set(IDENTIFIERS + MATCH_OUTCOMES + TARGET_COPIES + CATEGORICAL + ["position", TARGET])
    numeric = [c for c in df.columns if c not in excluded and pd.api.types.is_numeric_dtype(df[c])]
    unclassified = [c for c in df.columns if c not in excluded and c not in numeric]
    if unclassified:
        print("Non-numeric columns not classified (ignored):", unclassified)

    hist = df[df["round"] < args.max_round]
    rankings, lists = [], {}
    for pos in POSITIONS:
        r = rank_position(hist[hist.position == pos], numeric, per_family=args.per_family)
        r.insert(0, "position", pos)
        rankings.append(r.rename_axis("feature").reset_index())
        lists[pos] = r[r.redundant_with.isna()].index[: args.top].tolist()

    ranking = pd.concat(rankings, ignore_index=True)
    ranking.to_csv("../models/feature_ranking.csv", index=False)
    write_features_py(lists, args.max_round)

    for pos in POSITIONS:
        print(f"\n{pos} (top 15 of {len(lists[pos])}):")
        sub = ranking[(ranking.position == pos) & ranking.redundant_with.isna()].head(15)
        print(sub[["feature", "spearman", "mutual_info", "score"]].to_string(index=False))
