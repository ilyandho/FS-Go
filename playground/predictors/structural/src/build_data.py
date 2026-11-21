"""Build one row per player and gameweek, for every player with a fixture, with lagged features.

Adapted from 23-24/src/build_rolled_data.py (same sources and the same features, so the seasons line up), for
any season in config.SEASONS. Sources, downloaded once into config.SOURCES:

  vaastav/Fantasy-Premier-League   per-gameweek player stats, price, ownership and transfers (gws/merged_gw.csv),
                                   fixture difficulty (fixtures.csv), team names, web names (players_raw.csv)
                                   and Understat match logs (shots, key passes, xG, xA, ...)
  FPL-Optimal-Transfer             FPL id -> Understat id (id_dict_{yy}_{zz}.csv)
  football-data.co.uk              Premier League results and Bet365 odds (win_prob), and the results of the
                                   five seasons before, for the Elo ratings

  * one row per player and gameweek, also players who did not play; in a double gameweek the first fixture is kept
  * players only (GK, DEF, MID, FWD): the managers of 2024-25's Assistant Manager chip ("AM") are left out
  * win_prob, draw_prob, lose_prob: Bet365 odds as probabilities, normalised to sum to 1
  * lagged features (<stat>_rolling_k, <stat>_k_ewm for k = 1, 3, 5): from the gameweeks the player played,
    shifted by one gameweek (0 for a gameweek without minutes, and before the first appearance)
  * xP (FPL's own figure) is blanked in gameweeks where it is 0 for every player, and has no lagged features
  * Elo: computed from results, K = 20, home advantage 65, scaled by goal difference, starting at 1500 five
    seasons back; a promoted team starts at the mean rating of the teams that went down

Writes paths(season).players.

Usage (from src/): python build_data.py --season 24-25
"""

import argparse
import difflib
import html
import io
import json
import os
import unicodedata
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

import config

FOOTBALL_DATA = "https://www.football-data.co.uk/mmz4281/{code}/E0.csv"
ELO_SEASONS = 5  # seasons of results before the season used to warm up the Elo ratings
WINDOWS = [1, 3, 5]
HEADERS = {"User-Agent": "fpl-research-script"}

# football-data team name -> FPL team name, where they differ
TEAM_NAMES = {"Man United": "Man Utd", "Tottenham": "Spurs", "Sheffield United": "Sheffield Utd"}

# Stats with rolling and ewm features
LAGGED_STATS = [
    "points", "total_points", "minutes", "goals_scored", "assists", "clean_sheets", "goals_conceded",
    "own_goals", "yellow_cards", "red_cards", "saves", "bonus", "bps", "influence", "creativity", "threat",
    "ict_index", "starts", "expected_goals", "expected_assists", "expected_goal_involvements",
    "expected_goals_conceded", "value", "selected_by_percent", "transfers_in", "transfers_out",
    "shots", "key_passes", "xG", "xA", "npg", "npxG", "xGChain", "xGBuildup",
]
UNDERSTAT_STATS = ["shots", "key_passes", "xG", "xA", "npg", "npxG", "xGChain", "xGBuildup"]


class Source:
    """Downloads for one season, cached under config.SOURCES."""

    def __init__(self, p):
        self.p = p
        yy, zz = p.season.split("-")
        self.vaastav = f"https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/{p.season_dir}/"
        self.understat_list = ("https://api.github.com/repos/vaastav/Fantasy-Premier-League/contents/data/"
                               f"{p.season_dir}/understat")
        self.id_dict = ("https://raw.githubusercontent.com/ilyandho/FPL-Optimal-Transfer/refs/heads/FantasyGo/"
                        f"FPL%20Data/data/vaastav/data/id_dict_{yy}_{zz}.csv")

    @staticmethod
    def fetch(url, file):
        """Contents of url, downloaded once into file."""
        path = Path(file)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=60) as response:
                path.write_bytes(response.read())
        return path.read_bytes()

    def csv(self, url, file, **kwargs):
        return pd.read_csv(io.BytesIO(self.fetch(url, file)), **kwargs)

    def vaastav_csv(self, name):
        return self.csv(self.vaastav + name, f"{self.p.vaastav_cache}/{name.rsplit('/', 1)[-1]}")


def football_data(src, season_start):
    """Premier League results and odds for the season starting in season_start (a year), FPL team names."""
    code = f"{season_start % 100:02d}{(season_start + 1) % 100:02d}"
    df = src.csv(FOOTBALL_DATA.format(code=code), f"{config.SOURCES}/football-data/E0_{code}.csv",
                 encoding="latin-1")
    df = df.dropna(subset=["HomeTeam"]).copy()
    df["date"] = pd.to_datetime(df["Date"], dayfirst=True)
    df["team_h"] = df["HomeTeam"].replace(TEAM_NAMES)
    df["team_a"] = df["AwayTeam"].replace(TEAM_NAMES)
    return df.sort_values("date", kind="stable")


def elo_ratings(src, season_start, k=20, home_advantage=65):
    """Pre-match Elo of both teams for every match of the season, keyed by (team_h, team_a)."""
    ratings, pre_match, previous = {}, {}, set()
    for start in range(season_start - ELO_SEASONS, season_start + 1):
        df = football_data(src, start)
        teams = set(df["team_h"]) | set(df["team_a"])
        if previous:
            down = previous - teams
            start_rating = np.mean([ratings[t] for t in down]) if down else 1500.0
            for t in teams - previous:
                ratings[t] = start_rating
        for t in teams:
            ratings.setdefault(t, 1500.0)
        for m in df.itertuples():
            rh, ra = ratings[m.team_h], ratings[m.team_a]
            if start == season_start:
                pre_match[(m.team_h, m.team_a)] = (rh, ra)
            expected = 1 / (1 + 10 ** (-(rh + home_advantage - ra) / 400))
            result = 1.0 if m.FTHG > m.FTAG else 0.5 if m.FTHG == m.FTAG else 0.0
            gd = abs(m.FTHG - m.FTAG)
            scale = 1 if gd <= 1 else 1.5 if gd == 2 else (11 + gd) / 8
            change = k * scale * (result - expected)
            ratings[m.team_h] += change
            ratings[m.team_a] -= change
        previous = teams
    return pre_match


def odds_probabilities(src, season_start):
    """Bet365 home, draw and away probabilities, normalised to sum to 1, keyed by (team_h, team_a)."""
    df = football_data(src, season_start)
    inv = (1 / df[["B365H", "B365D", "B365A"]]).round(5)
    probs = inv.div(inv.sum(axis=1), axis=0).round(3)
    probs.columns = ["home_win_prob", "draw_prob", "away_win_prob"]
    probs.index = pd.MultiIndex.from_frame(df[["team_h", "team_a"]])
    return probs


def name_similarity(fpl, understat):
    """How well the words of an Understat name ("Martin_Odegaard") match an FPL full name, 0 to 1."""

    def words(name):
        name = unicodedata.normalize("NFKD", html.unescape(name).replace("_", " "))
        return "".join(c for c in name if not unicodedata.combining(c)).lower().split()

    fpl_words = words(fpl)
    return np.mean([max(difflib.SequenceMatcher(None, w, f).ratio() for f in fpl_words) for w in words(understat)])


def understat_matches(src, appearances):
    """Understat match stats per fpl_id and date, and the fpl_id -> Understat id map.

    Players come from id_dict when it lists them. The others are matched on their appearances: an Understat
    player matches an appearance if they played the same match (date and score) for about as long (minutes
    within 15). The candidate who matches most appearances is taken, ties broken by name similarity, if they
    match at least 70% of the appearances.
    """
    p = src.p
    listing = json.loads(src.fetch(src.understat_list, f"{p.vaastav_cache}/understat_files.json"))
    files = {int(f["name"].rsplit("_", 1)[1][:-4]): f["name"]
             for f in listing if f["name"].rsplit("_", 1)[-1][:-4].isdigit()}

    def load(item):
        understat, name = item
        df = src.csv(src.vaastav + "understat/" + urllib.request.quote(name), f"{p.vaastav_cache}/understat/{name}")
        return df.assign(understat_id=understat)

    with ThreadPoolExecutor(16) as pool:
        logs = pd.concat(pool.map(load, files.items()), ignore_index=True)
    logs = logs[logs["date"] >= f"{p.start_year}-07-01"]  # the files hold earlier seasons too
    logs["date"] = pd.to_datetime(logs["date"]).dt.date.astype(str)

    ids = src.csv(src.id_dict, f"{p.vaastav_cache}/id_dict.csv")
    understat_id = {f: u for f, u in zip(ids["FPL_ID"], ids["Understat_ID"]) if u in files}

    unmapped = appearances[~appearances["fpl_id"].isin(understat_id)]
    pairs = unmapped.merge(logs[["understat_id", "date", "time", "h_goals", "a_goals"]],
                           left_on=["date", "team_h_score", "team_a_score"], right_on=["date", "h_goals", "a_goals"])
    pairs = pairs[(pairs["minutes"] - pairs["time"]).abs() <= 15]
    pairs = pairs[~pairs["understat_id"].isin(understat_id.values())]
    hits = pairs.groupby(["fpl_id", "understat_id"]).size().rename("hits").reset_index()
    hits["share"] = hits["hits"] / hits["fpl_id"].map(unmapped.groupby("fpl_id").size())
    fpl_name = unmapped.drop_duplicates("fpl_id").set_index("fpl_id")["name"]
    hits["similarity"] = [name_similarity(fpl_name[f], files[u].rsplit("_", 1)[0])
                          for f, u in zip(hits["fpl_id"], hits["understat_id"])]
    best = hits.sort_values(["hits", "similarity"], ascending=False).drop_duplicates("fpl_id")
    best = best[best["share"] >= 0.7].drop_duplicates("understat_id")
    understat_id.update(zip(best["fpl_id"], best["understat_id"]))

    logs["fpl_id"] = logs["understat_id"].map({u: f for f, u in understat_id.items()})
    logs = logs.dropna(subset=["fpl_id"]).astype({"fpl_id": int})
    logs = logs.groupby(["fpl_id", "date"], as_index=False)[UNDERSTAT_STATS].sum()
    return logs, understat_id


def lagged_features(df, stat):
    """Rolling and ewm features of `stat` for every player and gameweek, from the gameweeks before it."""
    rounds = sorted(df["round"].unique())
    grid = (df.pivot_table(index="round", columns="fpl_id", values=stat, aggfunc="sum")
            .reindex(rounds).fillna(0.0))  # no row for the player that gameweek
    out = {}
    for k in WINDOWS:
        out[f"{stat}_rolling_{k}"] = grid.rolling(k, min_periods=k).mean().shift(1).fillna(0.0)
        out[f"{stat}_{k}_ewm"] = grid.ewm(span=k, adjust=False).mean().shift(1).fillna(0.0)
    return pd.concat({n: f.stack() for n, f in out.items()}, axis=1).reset_index()


def build(season):
    """Every player with a fixture, one row per gameweek (first fixture of a double gameweek)."""
    p = config.paths(season)
    src = Source(p)
    gw = src.vaastav_csv("gws/merged_gw.csv")
    fixtures = src.vaastav_csv("fixtures.csv")
    teams = src.vaastav_csv("teams.csv")
    players = src.vaastav_csv("players_raw.csv")
    team_name = dict(zip(teams["id"], teams["name"]))

    df = gw.rename(columns={"element": "fpl_id"}).drop(columns=["round"]).rename(columns={"GW": "round"})
    df = df[df["position"].isin(config.POSITIONS)]  # 2024-25 lists the Assistant Manager chip's managers as "AM"
    df["opponent_team"] = df["opponent_team"].map(team_name)

    # Ownership as a percentage of all managers: every manager owns 15 players. Players without a fixture have
    # no row, so blank gameweeks undercount; the number of managers only grows.
    managers = (df.drop_duplicates(["fpl_id", "round"]).groupby("round")["selected"].sum() / 15).cummax()
    df["selected_by_percent"] = (df["selected"] / df["round"].map(managers) * 100).round(1)

    # Fixture difficulty, home and away teams
    fx = fixtures.rename(columns={"id": "fixture"})[["fixture", "team_h", "team_a", "team_h_difficulty",
                                                     "team_a_difficulty"]]
    fx["team_h"] = fx["team_h"].map(team_name)
    fx["team_a"] = fx["team_a"].map(team_name)
    df = df.merge(fx, on="fixture", how="left", validate="many_to_one")

    # Odds and Elo, from the player's side
    home = df["was_home"].astype(bool)
    match = pd.MultiIndex.from_frame(df[["team_h", "team_a"]])
    probs = odds_probabilities(src, p.start_year).reindex(match)
    df["win_prob"] = np.where(home, probs["home_win_prob"], probs["away_win_prob"])
    df["draw_prob"] = probs["draw_prob"].to_numpy()
    df["lose_prob"] = np.where(home, probs["away_win_prob"], probs["home_win_prob"])
    elo = pd.DataFrame(elo_ratings(src, p.start_year), index=["elo_h", "elo_a"]).T.reindex(match)
    df["elo_h"] = elo["elo_h"].round(2).to_numpy()
    df["elo_a"] = elo["elo_a"].round(2).to_numpy()
    df["elo"] = np.where(home, df["elo_h"], df["elo_a"])
    df["opp_elo"] = np.where(home, df["elo_a"], df["elo_h"])
    df["opp_elo_h"], df["opp_elo_a"] = df["elo_h"], df["elo_a"]

    # One row per player and gameweek; first fixture of a double gameweek
    df = df.sort_values(["fpl_id", "round", "kickoff_time"])
    df = df.drop_duplicates(["fpl_id", "round"], keep="first").reset_index(drop=True)
    played = df["minutes"] > 0

    # Understat match stats, joined on the match date
    df["date"] = pd.to_datetime(df["kickoff_time"]).dt.date.astype(str)
    logs, understat_id = understat_matches(
        src, df.loc[played, ["fpl_id", "name", "date", "minutes", "team_h_score", "team_a_score"]])
    df = df.merge(logs, on=["fpl_id", "date"], how="left", validate="one_to_one")
    df[UNDERSTAT_STATS] = df[UNDERSTAT_STATS].fillna(0.0)
    df["understat_id"] = df["fpl_id"].map(understat_id)

    # FPL's xP is missing (0 for every player) in some gameweeks, e.g. GW26 of 2023-24: blank it there. It is
    # kept only for reference (the same-gameweek xP was recorded after the match) and has no lagged features.
    no_xp = df.groupby("round")["xP"].transform(lambda s: (s == 0).all())
    df.loc[no_xp, "xP"] = np.nan

    df["points"] = df["total_points"] - df["bonus"]
    df["fpl_name"] = df["fpl_id"].map(players.set_index("id")["web_name"])
    df["season"] = p.start_year

    # Lagged features, from the gameweeks players played (0 for a gameweek without minutes)
    lags = [lagged_features(df[played], stat) for stat in LAGGED_STATS]
    lagged = lags[0]
    for f in lags[1:]:
        lagged = lagged.merge(f, on=["round", "fpl_id"])
    df = df.merge(lagged, on=["fpl_id", "round"], how="left", validate="one_to_one")
    lag_cols = [c for c in lagged.columns if c not in ("round", "fpl_id")]
    df[lag_cols] = df[lag_cols].fillna(0.0)  # players who have not played yet

    df = df.drop(columns=[c for c in df.columns if c.startswith("mng_")])  # Assistant Manager stats, empty for players
    first = ["fpl_id", "round", "name", "fpl_name", "position", "team", "opponent_team", "was_home", "fixture",
             "kickoff_time", "date", "season", "xP", "total_points"]
    return df[first + [c for c in df.columns if c not in first]]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--season", default=config.DEFAULT_SEASON, choices=list(config.SEASONS))
    args = parser.parse_args()
    p = config.paths(args.season)
    df = build(args.season)
    os.makedirs(os.path.dirname(p.players), exist_ok=True)
    df.to_csv(p.players, index=False)
    played = df[df.minutes > 0]
    missing = {c: int(n) for c, n in played.isna().sum().items() if n}
    print(f"{len(df)} rows ({len(played)} with minutes), {df['fpl_id'].nunique()} players, "
          f"gameweeks {df['round'].min()}-{df['round'].max()}")
    print("Teams:", sorted(df["team"].unique()))
    print("Columns with empty values (rows with minutes):", missing or "none")
    print(f"Wrote {p.players}")


if __name__ == "__main__":
    main()
