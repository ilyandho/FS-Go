"""
Sigma_delta with correlations by position and relationship (Haugh & Singal, Appendix C).

Sigma = diag(sigma) rho diag(sigma). sigma is each player's prediction-error sd; rho is one
number per (relationship, position pair): same team, opponents, or 0 for unrelated players,
estimated from earlier gameweeks. Justified and validated in preliquisites.ipynb.
"""

from pathlib import Path

import numpy as np
import pandas as pd

MIN_HISTORY = 3  # gameweeks of errors a player needs for his own variance / standardisation
POS_ORDER = ["GK", "DEF", "MID", "FWD"]

_SRC = Path(__file__).resolve().parents[2] / "inputs/sources/vaastav_23-24"


def load_opponents(src=_SRC):
    """One row per (round, team, opp): two rows for a double gameweek, none for a blank."""
    team_names = pd.read_csv(src / "teams.csv").set_index("id")["name"]
    fixtures = pd.read_csv(src / "fixtures.csv").dropna(subset=["event"])
    fixtures["event"] = fixtures["event"].astype(int)
    home, away = fixtures["team_h"].map(team_names), fixtures["team_a"].map(team_names)
    return pd.concat(
        [
            pd.DataFrame({"round": fixtures["event"], "team": home, "opp": away}),
            pd.DataFrame({"round": fixtures["event"], "team": away, "opp": home}),
        ],
        ignore_index=True,
    )


opponents = load_opponents()


def pos_pair(a, b):
    return tuple(sorted((a, b), key=POS_ORDER.index))


def standardised_errors(hist):
    """Each player's errors (total_points - pred), standardised by his own mean and sd."""
    e = hist[["round", "fpl_id", "pos", "team"]].copy()
    e["error"] = hist["total_points"] - hist["pred"]
    g = e.groupby("fpl_id")["error"]
    e["z"] = (e["error"] - g.transform("mean")) / g.transform("std")
    return e[(g.transform("size") >= MIN_HISTORY) & np.isfinite(e["z"])]


def correlation_table(hist):
    """rho per (relationship, position pair), pooled over every same-team / opposing pair."""
    e = standardised_errors(hist)
    same = e.merge(e, on=["round", "team"], suffixes=("_i", "_j"))
    same = same[same["fpl_id_i"] < same["fpl_id_j"]].assign(rel="same")
    opp = e.merge(opponents, on=["round", "team"]).merge(
        e, left_on=["round", "opp"], right_on=["round", "team"], suffixes=("_i", "_j")
    )
    opp = opp[opp["fpl_id_i"] < opp["fpl_id_j"]].assign(rel="opp")
    pairs = pd.concat([same, opp], ignore_index=True)
    pairs["pos_pair"] = [pos_pair(a, b) for a, b in zip(pairs["pos_i"], pairs["pos_j"])]
    return (pairs["z_i"] * pairs["z_j"]).groupby([pairs["rel"], pairs["pos_pair"]]).agg(
        rho="mean", n_pairs="size"
    )


def nearest_psd_corr(corr, min_eig_frac=1e-4):
    vals, vecs = np.linalg.eigh((corr + corr.T) / 2)
    vals = np.clip(vals, max(min_eig_frac * vals.max(), 1e-8), None)
    c = vecs @ np.diag(vals) @ vecs.T
    d = np.sqrt(np.diag(c))
    return c / np.outer(d, d)


def player_std(players, hist):
    """sd of each player's past errors; his position's median with fewer than MIN_HISTORY
    gameweeks, or the position's pooled error variance when no player has enough yet."""
    errors = hist["total_points"] - hist["pred"]
    var = errors.groupby(hist["fpl_id"]).agg(["var", "size"])
    var = var.loc[var["size"] >= MIN_HISTORY, "var"].reindex(players.index)
    pos_median = var[var > 0].groupby(players["pos"]).median()
    pooled = errors.groupby(hist["pos"]).var()
    pos_median = pos_median.reindex(pooled.index).fillna(pooled)
    return np.sqrt(var.where(var > 0, players["pos"].map(pos_median)).to_numpy())


def build_mu_sigma(gw, preds):
    """(players, mu_delta, Sigma_delta) for gameweek gw, from gameweeks before gw only."""
    players = preds[preds["round"] == gw].drop_duplicates("fpl_id").set_index("fpl_id")
    hist = preds[preds["round"] < gw]
    rho = correlation_table(hist)["rho"]
    std = player_std(players, hist)

    pos, team = players["pos"].to_numpy(), players["team"].to_numpy()
    week = opponents[opponents["round"] == gw]
    plays = set(zip(week["team"], week["opp"]))
    corr = np.eye(len(players))
    for a in range(len(players)):
        for b in range(a + 1, len(players)):
            if team[a] == team[b]:
                rel = "same"
            elif (team[a], team[b]) in plays:
                rel = "opp"
            else:
                continue
            corr[a, b] = corr[b, a] = rho.get((rel, pos_pair(pos[a], pos[b])), 0.0)
    sigma = nearest_psd_corr(corr) * np.outer(std, std)
    return players, players["pred"].to_numpy(), (sigma + sigma.T) / 2
