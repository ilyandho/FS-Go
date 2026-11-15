"""
Sigma_delta with correlations by position and relationship (Haugh & Singal, Appendix C), for
2024-25; the 2023-24 version is 23-24/optimizer/MVT/sigma.py.

Players' residuals (xP - pred) are standardised and pooled over every pair of teammates in the
earlier gameweeks, giving one correlation per position pair. Those are fitted by a team factor,
rho_same(p, q) = a_p a_q, one loading a_p per position, so

    Sigma = B B' + diag(sigma^2 (1 - a_pos^2)) + diag(noise),   B[i, team(i)] = sigma_i a_pos(i),

with sigma_i the player's residual sd and noise the points - xP scatter pooled per position (as in
optimization.ipynb's sample-covariance version). B has one column per team, so Sigma minus its
diagonal has rank <= 20: the MIQP keeps every component and takes the diagonal as its exact
linear term (`diag_var`). Opposing players' correlations are near zero in 2024-25 (|rho| <= 0.03,
see correlation_table) and are left out. Gameweeks without xP (22, 32, 34) drop out.
"""

from pathlib import Path

import numpy as np
import pandas as pd

MIN_HISTORY = 3  # gameweeks of residuals a player needs for his own variance / standardisation
POS_ORDER = ["GK", "DEF", "MID", "FWD"]
PLAYER_DATA_PATH = Path(__file__).resolve().parent.parent / "inputs/rolled_data_24-25_xp_fixed.csv"


def load_opponents(path=PLAYER_DATA_PATH):
    """One row per (round, team, opp), from the player data's fixtures."""
    d = pd.read_csv(path, usecols=["round", "team", "opponent_team"])
    return (
        d.rename(columns={"opponent_team": "opp"})
        .dropna()
        .drop_duplicates()
        .reset_index(drop=True)
    )


opponents = load_opponents()


def pos_pair(a, b):
    return tuple(sorted((a, b), key=POS_ORDER.index))


def residuals(hist):
    """xP - pred per player-gameweek, where xP exists."""
    r = hist[["round", "element", "pos", "team"]].copy()
    r["resid"] = hist["xP"] - hist["pred"]
    return r.dropna(subset=["resid"])


def standardised_residuals(hist):
    e = residuals(hist)
    g = e.groupby("element")["resid"]
    e["z"] = (e["resid"] - g.transform("mean")) / g.transform("std")
    return e[(g.transform("size") >= MIN_HISTORY) & np.isfinite(e["z"])]


def correlation_table(hist):
    """rho per (relationship, position pair), pooled over every same-team / opposing pair."""
    e = standardised_residuals(hist)
    same = e.merge(e, on=["round", "team"], suffixes=("_i", "_j"))
    same = same[same["element_i"] < same["element_j"]].assign(rel="same")
    opp = e.merge(opponents, on=["round", "team"]).merge(
        e, left_on=["round", "opp"], right_on=["round", "team"], suffixes=("_i", "_j")
    )
    opp = opp[opp["element_i"] < opp["element_j"]].assign(rel="opp")
    pairs = pd.concat([same, opp], ignore_index=True)
    pairs["pos_pair"] = [pos_pair(a, b) for a, b in zip(pairs["pos_i"], pairs["pos_j"])]
    return (pairs["z_i"] * pairs["z_j"]).groupby([pairs["rel"], pairs["pos_pair"]]).agg(
        rho="mean", n_pairs="size"
    )


def player_std(elements, position, hist):
    """sd of each player's past residuals; his position's median with fewer than MIN_HISTORY,
    or the position's pooled residual variance when no player has enough yet."""
    r = residuals(hist)
    var = r.groupby("element")["resid"].agg(["var", "size"])
    var = var.loc[var["size"] >= MIN_HISTORY, "var"].reindex(elements).to_numpy(dtype=float, copy=True)
    known = np.isfinite(var) & (var > 0)
    for pos in np.unique(position):
        here = position == pos
        fill = np.median(var[known & here]) if (known & here).any() else r.loc[r["pos"] == pos, "resid"].var()
        var[~known & here] = fill if np.isfinite(fill) else 1.0
    return np.sqrt(np.maximum(var, 1e-4))


def team_loadings(table):
    """a_p per position, least squares (weighted by pairs) fit of rho_same(p, q) = a_p a_q."""
    from scipy.optimize import least_squares

    if "same" not in table.index.get_level_values("rel"):
        # too early in the season: no player has MIN_HISTORY gameweeks of residuals yet,
        # so there are no teammate pairs and Sigma has no correlations (a_p = 0)
        return dict.fromkeys(POS_ORDER, 0.0)
    same = table.loc["same"]
    i = np.array([POS_ORDER.index(p) for p, _ in same.index])
    j = np.array([POS_ORDER.index(q) for _, q in same.index])
    w = np.sqrt(same["n_pairs"].to_numpy() / same["n_pairs"].sum())
    rho = same["rho"].to_numpy()
    fit = least_squares(lambda a: w * (a[i] * a[j] - rho), x0=np.full(4, 0.4), bounds=(0, 0.95))
    return dict(zip(POS_ORDER, fit.x))


def build_mu_sigma_delta(GW, preds, outcome_noise=True):
    """Same outputs as optimization.ipynb's build_mu_sigma_delta:
    (elements, mu_delta, Sigma_delta, diag_var), from gameweeks before GW only. diag_var is
    Sigma's diagonal part (idiosyncratic + points - xP noise), Sigma - diag(diag_var) = B B'."""
    mu_row = preds[preds["round"] == GW].drop_duplicates("element")
    hist = preds[preds["round"] < GW]
    elements = sorted(mu_row["element"])
    rows = mu_row.set_index("element").loc[elements]
    mu_delta = rows["pred"].to_numpy(dtype=float)
    position, team = rows["pos"].to_numpy(), rows["team"].to_numpy()

    loadings = team_loadings(correlation_table(hist))
    a = np.array([loadings[p] for p in position])
    std = player_std(elements, position, hist)
    teams = sorted(set(team))
    B = np.zeros((len(elements), len(teams)))
    B[np.arange(len(elements)), [teams.index(t) for t in team]] = std * a
    diag_var = std**2 * (1 - a**2)
    if outcome_noise:
        noise = (hist["total_points"] - hist["xP"]).groupby(hist["pos"]).var()
        diag_var = diag_var + noise.reindex(position).fillna(noise.median()).to_numpy()
    return elements, mu_delta, B @ B.T + np.diag(diag_var), diag_var
