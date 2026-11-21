"""Structural xP model: predict each FPL scoring component, then combine them with the scoring rules.

For every player with a fixture in gameweek gw, every model is fitted on gameweeks < gw only:

  minutes state     P(plays), then       gradient boosting      all rows, every position pooled; uses minutes
                    P(60+ | plays)       (plays stage monotonic history and the pre-deadline FPL status
                                         in the FPL status)     and chance of playing (build_availability.py)
  goals conceded    GA ~ Poisson(lam)    Poisson GLM            team-fixture rows (all teams, DGWs too)
  goals, assists,   per appearance,      Poisson GLMs           appearance rows of the position; goals and
  saves (GK)        given 1-59 or 60+                           assists also use pre-deadline set-piece duties
  bonus             per appearance       Poisson GLM            appearance rows, every position pooled
  pens saved (GK), cards, own goals, missed pens: training mean per appearance

  E[pts] = sum over s in {1-59, 60+} of P(s) * E[pts | s], with
  E[pts | 60+]  = 2 + cs*e^-lam - gcp*E[floor(GA/2)] + goal*g + 3*a + b + save*E[floor(S/3)] + other
  E[pts | 1-59] = 1            - gcp*E[floor(GA'/2)] + goal*g + 3*a + b + save*E[floor(S/3)] + other
  where GA' ~ Poisson(lam * share of the match a substitute plays).

Assumptions: the minutes state is independent of the match result, and a player on for 60+ minutes is on
for the whole match (team clean sheet = player clean sheet).

targets="expected" (default) trains the goal, assist and goals-conceded models on xG, xA and team xGA
instead of the realised counts (less noise; predictions are still scored against total_points). It scored
slightly better than "realised" for every position in 2023-24.

Works for any season in config.SEASONS (--season; inputs from build_data.py and build_availability.py).

Writes paths(season).output, one row per player and gameweek: season, pos, round, fpl_id, name, team,
opponent_team, fixture, value, total_points, minutes, played, has_target, the component probabilities (p_play, p60, lam_ga, p_cs),
the expected points per component (e_* columns, which sum to pred), pred, pred_if_plays and history (the
earlier seasons in the training data, empty if none).
pred is unconditional (it includes the chance of not playing): use it to pick a team before the deadline.
pred_if_plays = E[pts | plays], for scoring on the players who played.

Model selection metric: RMSE (squared error is minimised by the mean, which is what the optimiser needs;
MAE is minimised by the median).

Usage (from src/):
  python structural.py                                    # 23-24 (config.DEFAULT_SEASON), all positions, GW 5-38
  python structural.py --season 24-25
  python structural.py --pos GK DEF --gws 20-38 --targets realised
  python structural.py --gws 21                           # one gameweek
  python structural.py --no-availability                  # without the FPL status features
  python structural.py --no-set-pieces                    # without the set-piece duties
  python structural.py --season 24-25 --history 23-24     # also train on all of 23-24; predicts GW1-38
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats
from sklearn import ensemble, impute, linear_model, metrics, preprocessing
from sklearn.pipeline import make_pipeline

import config

FIRST_GW, LAST_GW = config.FIRST_GW, config.LAST_GW

# FPL scoring, 2023-24. Every position: 1 pt for 1-59 min, 2 for 60+, 3 per assist, bonus as awarded,
# -1 yellow, -3 red, -2 own goal, -2 penalty missed, 5 per penalty saved.
SCORING = {
    "GK": {"goal": 6, "cs": 4, "gc_per2": 1, "save_per3": 1},
    "DEF": {"goal": 6, "cs": 4, "gc_per2": 1, "save_per3": 0},
    "MID": {"goal": 5, "cs": 1, "gc_per2": 0, "save_per3": 0},
    "FWD": {"goal": 4, "cs": 0, "gc_per2": 0, "save_per3": 0},
}
ASSIST, PEN_SAVE = 3, 5

# ---------------------------------------------------------------------------- features
TEAM_FEATS = ["elo", "opp_elo", "elo_diff", "home",
              "t_xga_5", "t_xga_10", "t_ga_10", "o_xgf_5", "o_xgf_10", "o_gf_10"]
MIN_FEATS = ["minutes_rolling_1", "minutes_rolling_3", "minutes_rolling_5", "minutes_3_ewm", "minutes_5_ewm",
             "starts_rolling_1", "starts_rolling_3", "starts_rolling_5", "value", "selected_by_percent",
             "log_transfers_in", "log_transfers_out"]
# Pre-deadline FPL status and chance of playing (build_availability.py). unavailable = status u or n.
# stale_flag = (1 - chance) * snapshot age in days: lets older injury news count for less.
AVAIL_FEATS = ["chance", "injured", "doubtful", "suspended", "unavailable", "in_snapshot",
               "snapshot_age_days", "stale_flag"]
# Position flags for the models pooled across positions (minutes, bonus); FWD is the baseline.
POSITION_FEATS = ["is_GK", "is_DEF", "is_MID"]
CONTEXT = ["elo_diff", "home", "full"]  # full = 1 if the appearance is 60+ minutes
GOAL_FEATS = ["expected_goals_rolling_5", "npxG_rolling_5", "xG_rolling_5", "threat_rolling_5",
              "shots_rolling_5", "minutes_rolling_3", "t_xgf_10", "o_xga_10"] + CONTEXT
ASSIST_FEATS = ["expected_assists_rolling_5", "xA_rolling_5", "creativity_rolling_5", "key_passes_rolling_5",
                "minutes_rolling_3", "t_xgf_10", "o_xga_10"] + CONTEXT
# Pre-deadline set-piece duties (FPL's taker order, build_availability.py): 1 = first choice.
SETPIECE_GOAL_FEATS = ["pen_first", "pen_backup", "freekick_first"]
SETPIECE_ASSIST_FEATS = ["corner_first", "freekick_first"]
BONUS_FEATS = ["bps_rolling_5", "bps_3_ewm", "bonus_rolling_5", "expected_goal_involvements_rolling_5",
               "minutes_rolling_3", "t_xga_10", "o_xgf_10"] + CONTEXT
SAVE_FEATS = ["saves_rolling_5", "saves_3_ewm", "minutes_rolling_3", "t_xga_10", "o_xgf_10"] + CONTEXT

# Components fitted as a constant rate for a position (too few events for a GLM).
CONSTANT = {"GK": {"goals", "assists"}}


def points(df, pos):
    """FPL points from the realised components (reproduces total_points exactly; a check)."""
    s = SCORING[pos]
    app = np.where(df.minutes >= 60, 2, np.where(df.minutes > 0, 1, 0))
    return (app + s["goal"] * df.goals_scored + ASSIST * df.assists
            + s["cs"] * df.clean_sheets * (df.minutes >= 60) - s["gc_per2"] * (df.goals_conceded // 2)
            + s["save_per3"] * (df.saves // 3) + PEN_SAVE * df.penalties_saved + df.bonus
            - df.yellow_cards - 3 * df.red_cards - 2 * df.own_goals - 2 * df.penalties_missed)


def team_fixtures(players, p):
    """One row per team and fixture: goals and xG for and against, and form going into the fixture.

    Form features use matches in earlier gameweeks only (also the second fixture of a double gameweek).
    """
    m = pd.read_csv(f"{p.vaastav_cache}/merged_gw.csv")
    m = m[m.position.isin(config.POSITIONS)]  # not 2024-25's Assistant Manager chip managers ("AM")
    fx = (m.groupby(["fixture", "was_home"])
          .agg(team=("team", "first"), round=("round", "first"), kickoff=("kickoff_time", "first"),
               xga=("expected_goals_conceded", "max"), hs=("team_h_score", "first"),
               as_=("team_a_score", "first"))
          .reset_index())
    fx["ga"] = np.where(fx.was_home, fx.as_, fx.hs)
    opp = fx[["fixture", "was_home", "team", "xga", "ga"]].rename(columns={"team": "opp", "xga": "xgf", "ga": "gf"})
    opp["was_home"] = ~opp.was_home
    fx = fx.merge(opp, on=["fixture", "was_home"]).sort_values(["team", "kickoff"])

    roll = []
    for c in ["xga", "ga", "xgf", "gf"]:
        for w in (5, 10):
            name = f"t_{c}_{w}"
            fx[name] = fx.groupby("team")[c].transform(lambda s: s.shift(1).rolling(w, min_periods=1).mean())
            roll.append(name)
    fx[roll] = fx.groupby(["team", "round"])[roll].transform("first")  # DGW: as of the gameweek's start
    fx[roll] = fx[roll].fillna(fx[roll].mean())  # first match of the season

    o = fx[["fixture", "team"] + roll].rename(columns={"team": "opp", **{c: "o" + c[1:] for c in roll}})
    fx = fx.merge(o, on=["fixture", "opp"])
    elo = players.groupby(["team", "fixture"])[["elo", "opp_elo"]].first().reset_index()
    fx = fx.merge(elo, on=["team", "fixture"], how="left")
    fx["elo_diff"] = fx.elo - fx.opp_elo
    fx["home"] = fx.was_home.astype(int)
    return fx


def load(season=config.DEFAULT_SEASON):
    """Every player with a fixture in the season (also those who did not play), with the team features.

    Returns (players, fx). Both carry has_target (False in config.SEASONS[season]["missing_target_gws"]).
    """
    p = config.paths(season)
    players = pd.read_csv(p.players, low_memory=False)
    players = players[players.position.isin(config.POSITIONS)]
    fx = team_fixtures(players, p)
    fx["has_target"] = ~fx["round"].isin(p.missing_target_gws)
    t = [c for c in fx.columns if c.startswith(("t_", "o_")) and c not in players.columns] + ["elo_diff", "home"]
    players = players.merge(fx[["team", "fixture"] + t], on=["team", "fixture"], how="left").copy()
    state = np.select([players.minutes >= 60, players.minutes > 0], [2, 1], 0)
    players = players.assign(
        state=state,
        full=(state == 2).astype(int),
        other=(PEN_SAVE * players.penalties_saved - players.yellow_cards - 3 * players.red_cards
               - 2 * players.own_goals - 2 * players.penalties_missed),
        log_transfers_in=np.log1p(players.transfers_in.clip(lower=0)),
        log_transfers_out=np.log1p(players.transfers_out.clip(lower=0)),
        **{f: (players.position == f[3:]).astype(int) for f in POSITION_FEATS},
        season=season,
        has_target=~players["round"].isin(p.missing_target_gws),
    )
    return add_availability(players, p), fx


def add_availability(players, p):
    """Join each player's last pre-deadline FPL status; players missing from the snapshot count as available."""
    av = pd.read_csv(p.availability)[["fpl_id", "round", "status", "chance_of_playing", "snapshot_age_hours",
                                      "penalties_order", "corners_order", "freekicks_order"]]
    players = players.merge(av, on=["fpl_id", "round"], how="left")
    found = players.status.notna()
    chance = players.chance_of_playing.fillna(100).where(found, 100) / 100
    age = (players.snapshot_age_hours / 24).fillna(av.snapshot_age_hours.median() / 24)
    return players.assign(
        chance=chance,
        injured=(players.status == "i").astype(int),
        doubtful=(players.status == "d").astype(int),
        suspended=(players.status == "s").astype(int),
        unavailable=players.status.isin(["u", "n"]).astype(int),
        in_snapshot=found.astype(int),
        snapshot_age_days=age,
        stale_flag=(1 - chance) * age,
        pen_first=(players.penalties_order == 1).astype(int),
        pen_backup=(players.penalties_order >= 2).astype(int),
        corner_first=(players.corners_order == 1).astype(int),
        freekick_first=(players.freekicks_order == 1).astype(int),
    )


# ---------------------------------------------------------------------------- models
class Constant:
    """Mean of the target, as a model with fit/predict."""

    def fit(self, X, y):
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_)


class MinutesModel:
    """P(1-59 min) and P(60+ min) in two stages: P(plays), then P(60+ | plays), both gradient boosting.

    The plays stage is monotonic in the FPL status features: a lower chance of playing, or an injury,
    suspension, unavailability or doubt flag, never raises P(plays). Without that (and without the
    interactions boosting can fit), a regular starter's minutes history outweighs an injury flag.
    """

    MONOTONIC = {"chance": 1, "injured": -1, "doubtful": -1, "suspended": -1, "unavailable": -1, "stale_flag": -1}

    def __init__(self, feats):
        self.feats = feats

    @staticmethod
    def _hgb(monotonic_cst=None):
        return ensemble.HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40, l2_regularization=1.0,
            monotonic_cst=monotonic_cst, random_state=42)

    def _fit_binary(self, model, X, y):
        return model.fit(X, y) if y.nunique() > 1 else Constant().fit(X, y)  # e.g. every early GK app is 60+

    def fit(self, train):
        cst = [self.MONOTONIC.get(f, 0) for f in self.feats]
        self.plays_ = self._fit_binary(self._hgb(cst), train[self.feats], train.state > 0)
        apps = train[train.state > 0]
        self.full_ = self._fit_binary(self._hgb(), apps[self.feats], apps.state == 2)
        return self

    @staticmethod
    def _proba(model, X):
        return model.predict(X) if isinstance(model, Constant) else model.predict_proba(X)[:, 1]

    def predict(self, test):
        """(P(1-59), P(60+)) for the rows of test."""
        X = test[self.feats]
        plays, full = self._proba(self.plays_, X), self._proba(self.full_, X)
        return plays * (1 - full), plays * full


def glm(est):
    return make_pipeline(impute.SimpleImputer(strategy="median"), preprocessing.StandardScaler(), est)


def poisson(alpha=1e-2):
    return glm(linear_model.PoissonRegressor(alpha=alpha, max_iter=1000))


def e_floor(lam, k, kmax=30):
    """E[floor(X / k)] for X ~ Poisson(lam), elementwise."""
    x = np.arange(kmax + 1)
    return (stats.poisson.pmf(x[None, :], np.asarray(lam)[:, None]) * (x // k)).sum(1)


def target(name, targets):
    """Training column for a component: realised count, or its expected counterpart."""
    expected = {"goals": "expected_goals", "assists": "expected_assists", "ga": "xga"}
    realised = {"goals": "goals_scored", "assists": "assists", "ga": "ga"}
    return (expected if targets == "expected" else realised)[name]


def fit_team(fx_train, targets="expected"):
    """Goals conceded per team-fixture: Poisson GLM, shared by every position."""
    return poisson(1e-3).fit(fx_train[TEAM_FEATS], fx_train[target("ga", targets)])


def fit_pooled(train, availability=True):
    """Minutes and bonus models, fitted on every position at once with position flags (gameweeks < gw).

    Pooling cut the minutes log loss by 3-23% (most for GK, the smallest position) and the GK bonus
    deviance by 12% against one model per position in 2023-24; goals and assists gained nothing, so they stay
    per position.
    """
    min_feats = MIN_FEATS + (AVAIL_FEATS if availability else []) + POSITION_FEATS
    apps = train[train.state > 0]
    bonus_feats = BONUS_FEATS + POSITION_FEATS
    return {"minutes": MinutesModel(min_feats).fit(train),
            "bonus": (bonus_feats, poisson().fit(apps[bonus_feats], apps.bonus))}


def fit_position(train, pos, pooled, targets="expected", set_pieces=True):
    """The per-position component models, fitted on train (gameweeks < gw), plus the pooled ones."""
    apps = train[train.state > 0]
    goal_feats = GOAL_FEATS + (SETPIECE_GOAL_FEATS if set_pieces else [])
    assist_feats = ASSIST_FEATS + (SETPIECE_ASSIST_FEATS if set_pieces else [])
    rate = {
        "goals": (goal_feats, target("goals", targets)),
        "assists": (assist_feats, target("assists", targets)),
    }
    if SCORING[pos]["save_per3"]:
        rate["saves"] = (SAVE_FEATS, "saves")
    fitted = dict(pooled)
    for name, (feats, col) in rate.items():
        model = Constant() if name in CONSTANT.get(pos, ()) else poisson()
        fitted[name] = (feats, model.fit(apps[feats], apps[col]))
    fitted["other"] = apps.other.mean()
    subs = apps[apps.state == 1]
    fitted["sub_share"] = subs.minutes.mean() / 90 if len(subs) else 1 / 3
    return fitted


def predict(fitted, team_model, test, pos):
    """Component probabilities and expected points per component for the rows of test."""
    s = SCORING[pos]
    p_sub, p_full = fitted["minutes"].predict(test)
    lam = team_model.predict(test[TEAM_FEATS])

    def rate(name, full):
        if name not in fitted:
            return np.zeros(len(test))
        feats, model = fitted[name]
        return model.predict(test[feats].assign(full=full))

    out = pd.DataFrame({
        "season": test.season.values, "pos": pos, "round": test["round"].values, "fpl_id": test.fpl_id.values, "name": test.name.values,
        "team": test.team.values, "opponent_team": test.opponent_team.values, "fixture": test.fixture.values,
        "value": test.value.values, "total_points": test.total_points.values, "minutes": test.minutes.values,
        "played": (test.minutes > 0).values, "has_target": test.has_target.values,
        "p_play": p_sub + p_full, "p60": p_full, "lam_ga": lam, "p_cs": np.exp(-lam),
    }, index=test.index)
    out["e_app"] = p_sub * 1 + p_full * 2
    out["e_cs"] = p_full * s["cs"] * out.p_cs
    out["e_gc"] = -s["gc_per2"] * (p_full * e_floor(lam, 2) + p_sub * e_floor(lam * fitted["sub_share"], 2))
    for name, pts in [("goals", s["goal"]), ("assists", ASSIST), ("bonus", 1)]:
        out[f"e_{name}"] = pts * (p_sub * rate(name, 0) + p_full * rate(name, 1))
    out["e_saves"] = s["save_per3"] * (p_sub * e_floor(rate("saves", 0), 3) + p_full * e_floor(rate("saves", 1), 3))
    out["e_other"] = out.p_play * fitted["other"]
    out["pred"] = out.filter(like="e_").sum(axis=1)
    # every component is weighted by P(1-59) or P(60+), so this is exactly E[pts | plays]
    out["pred_if_plays"] = out.pred / out.p_play.clip(lower=1e-6)
    return out


def walk_forward(positions=config.POSITIONS, gws=range(FIRST_GW, LAST_GW + 1), targets="expected",
                 availability=True, data=None, season=config.DEFAULT_SEASON, set_pieces=True, history=()):
    """Predict every player of the positions for each gameweek, from models fitted on earlier gameweeks.

    data: the output of load(season), to avoid reloading it (its season wins over `season`).
    history: earlier seasons (e.g. ["23-24"]) whose every gameweek is added to the training data of each gameweek.
      Each history season is either a season name, loaded with load(), or a (players, fx) pair from load().
      In 2024-25, adding 2023-24 cut RMSE from 1.877 to 1.871 (most of it in GW5-10) and makes GW1-4 predictable.
    """
    players, fx = data if data is not None else load(season)
    current = players.season.iloc[0]
    fx = fx[fx.has_target]
    hist_p, hist_fx, names = [], [], []
    for h in history:
        hp, hf = load(h) if isinstance(h, str) else h
        if hp.season.iloc[0] == current:
            raise ValueError(f"history season {current} is the season being predicted")
        names.append(hp.season.iloc[0])
        hist_p.append(hp[hp.has_target])
        hist_fx.append(hf[hf.has_target])
    out = []
    for gw in gws:
        train_all = pd.concat(hist_p + [players[players.has_target & (players["round"] < gw)]], ignore_index=True)
        fx_train = pd.concat(hist_fx + [fx[fx["round"] < gw]], ignore_index=True)
        if train_all.empty:
            raise ValueError(f"no training data before GW{gw} of {current}: start later or pass a history season")
        team_model = fit_team(fx_train, targets)
        pooled = fit_pooled(train_all, availability)  # always on every position, whatever `positions` is
        for pos in positions:
            train, test = train_all[train_all.position == pos], players[(players.position == pos) & (players["round"] == gw)]
            if test.empty:
                continue
            out.append(predict(fit_position(train, pos, pooled, targets, set_pieces), team_model, test, pos))
    return pd.concat(out, ignore_index=True).assign(history="+".join(names))


# ---------------------------------------------------------------------------- scoring
def score(df, pred_cols=("pred",), by="pos", played_only=True):
    """RMSE (the selection metric), R2, Spearman and MAE of each prediction column against total_points,
    per group and overall."""
    df = df[df.played] if played_only else df
    df = df[df.has_target]
    rows = []
    for group, g in list(df.groupby(by)) + [("all", df)]:
        for col in pred_cols:
            rows.append({by: group, "model": col, "n": len(g),
                         "RMSE": metrics.root_mean_squared_error(g.total_points, g[col]),
                         "R2": metrics.r2_score(g.total_points, g[col]),
                         "Spearman": stats.spearmanr(g.total_points, g[col])[0],
                         "MAE": metrics.mean_absolute_error(g.total_points, g[col])})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--season", default=config.DEFAULT_SEASON, choices=list(config.SEASONS))
    parser.add_argument("--pos", nargs="+", default=config.POSITIONS, choices=config.POSITIONS)
    parser.add_argument("--gws", help=f"e.g. 5-38 or 21; default {FIRST_GW}-{LAST_GW}, or 1-{LAST_GW} with --history")
    parser.add_argument("--history", nargs="+", default=[], choices=list(config.SEASONS),
                        help="earlier seasons added to the training data of every gameweek")
    parser.add_argument("--targets", default="expected", choices=["realised", "expected"])
    parser.add_argument("--no-availability", action="store_true", help="leave FPL status out of the minutes model")
    parser.add_argument("--no-set-pieces", action="store_true", help="leave set-piece duties out of goals/assists")
    parser.add_argument("--out", help="default: config.output_path(season, history)")
    args = parser.parse_args()
    out = args.out or config.output_path(args.season, args.history)
    lo, _, hi = (args.gws or (f"1-{LAST_GW}" if args.history else f"{FIRST_GW}-{LAST_GW}")).partition("-")
    preds = walk_forward(args.pos, range(int(lo), int(hi or lo) + 1), args.targets, not args.no_availability,
                         season=args.season, set_pieces=not args.no_set_pieces, history=args.history)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    preds.to_csv(out, index=False)
    print(f"Wrote {out} ({len(preds)} rows)\n")
    print("Every player with a fixture (pred):")
    print(score(preds, ("pred",), played_only=False).round(3).to_string(index=False))
    print("\nPlayers who played (pred_if_plays):")
    print(score(preds, ("pred_if_plays",)).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
