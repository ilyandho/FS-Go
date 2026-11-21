"""Build the rolled player data for config.SEASON from public sources, in the 2024-25 column format.

The 2024-25 file was built by FPL-Optimal-Transfer's data.ipynb. Its 2023-24 file is a copy of the
2024-25 data, so this script rebuilds 2023-24 from the same kind of sources:

  vaastav/Fantasy-Premier-League   per-gameweek player stats, xP, price, ownership and transfers
                                   (gws/merged_gw.csv), fixture difficulty (fixtures.csv), team names,
                                   web names (players_raw.csv) and Understat match logs (xGChain, ...)
  FPL-Optimal-Transfer             FPL id -> Understat id (id_dict_{yy_yy}.csv)
  football-data.co.uk              Premier League results and Bet365 odds (win_prob), and the results
                                   of the five seasons before for the Elo ratings

As in the 2024-25 file:
  * one row per player and gameweek, only for players with minutes; in a double gameweek the first
    fixture is kept. config.ALL_PLAYERS_DATA has every player with a fixture, also those who did not
    play, with the same columns; its lagged features are the same as in the main file (built from
    the gameweeks players played)
  * win_prob, draw_prob and lose_prob are the Bet365 odds as probabilities, normalised to sum to 1
  * the lagged features are built like the source (fix_missing_xp.lagged_features): shifted by one
    gameweek, 0 for a gameweek without a row, rolling means over 1, 3 and 5 gameweeks and ewm spans
  * xP is 0 in the gameweeks without a target (config.MISSING_TARGET_GWS); fix_missing_xp.py blanks them

Not available for 2023-24: tackles and the other defensive actions (FPL started publishing them in
2025-26), so there are no tackles_* or cbitr_* features.

Elo: ClubElo's API was unavailable, so the ratings are computed from the results: K = 20, home advantage
65 points, scaled by goal difference, starting at 1500 in 2018-19. A promoted team starts at the mean
rating of the teams that went down. Only differences between teams matter to the models.

Downloads are cached in ../inputs/sources/.

Writes config.RAW_DATA and config.ALL_PLAYERS_DATA.

Usage: python build_rolled_data.py
"""

import difflib
import html
import io
import json
import unicodedata
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import config
import fix_missing_xp
import numpy as np
import pandas as pd

YEARS = config.SEASON.split("-")  # "23-24" -> ["23", "24"]
VAASTAV = (
    "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/"
    f"20{YEARS[0]}-{YEARS[1]}/"
)
UNDERSTAT_LIST = (
    "https://api.github.com/repos/vaastav/Fantasy-Premier-League/contents/data/"
    f"20{YEARS[0]}-{YEARS[1]}/understat"
)
ID_DICT = (
    "https://raw.githubusercontent.com/ilyandho/FPL-Optimal-Transfer/refs/heads/FantasyGo/"
    f"FPL%20Data/data/vaastav/data/id_dict_{YEARS[0]}_{YEARS[1]}.csv"
)
FOOTBALL_DATA = "https://www.football-data.co.uk/mmz4281/{code}/E0.csv"
ELO_SEASONS = (
    5  # seasons of results before config.SEASON used to warm up the Elo ratings
)

CACHE = Path("../inputs/sources")

# football-data team name -> FPL team name, where they differ
TEAM_NAMES = {
    "Man United": "Man Utd",
    "Tottenham": "Spurs",
    "Sheffield United": "Sheffield Utd",
    "Nott'm Forest": "Nott'm Forest",
}

# Stats with rolling and ewm features, as in the 2024-25 file (where available)
LAGGED_STATS = [
    "points",
    "xP",
    "total_points",
    "minutes",
    "goals_scored",
    "assists",
    "clean_sheets",
    "goals_conceded",
    "own_goals",
    "yellow_cards",
    "red_cards",
    "saves",
    "bonus",
    "bps",
    "influence",
    "creativity",
    "threat",
    "ict_index",
    "starts",
    "expected_goals",
    "expected_assists",
    "expected_goal_involvements",
    "expected_goals_conceded",
    "value",
    "selected_by_percent",
    "transfers_in",
    "transfers_out",
    "shots",
    "key_passes",
    "xG",
    "xA",
    "npg",
    "npxG",
    "xGChain",
    "xGBuildup",
]
UNDERSTAT_STATS = [
    "shots",
    "key_passes",
    "xG",
    "xA",
    "npg",
    "npxG",
    "xGChain",
    "xGBuildup",
]


def fetch(url, name):
    """Contents of url, downloaded once into CACHE/name."""
    path = CACHE / name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=60) as response:
            path.write_bytes(response.read())
    return path.read_bytes()


def read_csv(url, name, **kwargs):
    return pd.read_csv(io.BytesIO(fetch(url, name)), **kwargs)


def football_data(season_start):
    """Premier League results and odds for the season starting in 20{season_start}, FPL team names."""
    code = f"{season_start:02d}{season_start + 1:02d}"
    df = read_csv(
        FOOTBALL_DATA.format(code=code),
        f"football-data/E0_{code}.csv",
        encoding="latin-1",
    )
    df = df.dropna(subset=["HomeTeam"])
    df["date"] = pd.to_datetime(df["Date"], dayfirst=True)
    df["team_h"] = df["HomeTeam"].replace(TEAM_NAMES)
    df["team_a"] = df["AwayTeam"].replace(TEAM_NAMES)
    return df.sort_values("date", kind="stable")


def elo_ratings(season_start, k=20, home_advantage=65):
    """Pre-match Elo of both teams for every match of the season, keyed by (team_h, team_a)."""
    ratings, pre_match = {}, {}
    previous = set()
    for start in range(season_start - ELO_SEASONS, season_start + 1):
        df = football_data(start)
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


def odds_probabilities(season_start):
    """Bet365 home, draw and away probabilities, normalised to sum to 1, keyed by (team_h, team_a)."""
    df = football_data(season_start)
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
    return np.mean(
        [
            max(difflib.SequenceMatcher(None, w, f).ratio() for f in fpl_words)
            for w in words(understat)
        ]
    )


def understat_matches(appearances):
    """Understat match stats (UNDERSTAT_STATS) per fpl_id and date, and the fpl_id -> Understat id map.

    appearances has fpl_id, name, date, minutes, team_h_score and team_a_score. Players come from
    id_dict when it lists them. The others are matched on their appearances: an Understat player
    matches an appearance if they played the same match (date and score) for about as long (minutes
    within 15). The candidate who matches most appearances is taken, ties broken by name similarity,
    if they match at least 70% of the appearances.
    """
    listing = json.loads(
        fetch(UNDERSTAT_LIST, f"vaastav_{config.SEASON}/understat_files.json")
    )
    files = {
        int(f["name"].rsplit("_", 1)[1][:-4]): f["name"]
        for f in listing
        if f["name"].rsplit("_", 1)[-1][:-4].isdigit()
    }

    def load(item):
        understat, name = item
        df = read_csv(
            VAASTAV + "understat/" + urllib.request.quote(name),
            f"vaastav_{config.SEASON}/understat/{name}",
        )
        return df.assign(understat_id=understat)

    with ThreadPoolExecutor(16) as pool:
        logs = pd.concat(pool.map(load, files.items()), ignore_index=True)
    logs = logs[
        logs["date"] >= f"20{YEARS[0]}-07-01"
    ]  # the files hold earlier seasons too
    logs["date"] = pd.to_datetime(logs["date"]).dt.date.astype(str)

    ids = read_csv(ID_DICT, f"vaastav_{config.SEASON}/id_dict.csv")
    understat_id = dict(zip(ids["FPL_ID"], ids["Understat_ID"]))
    understat_id = {f: u for f, u in understat_id.items() if u in files}

    unmapped = appearances[~appearances["fpl_id"].isin(understat_id)]
    pairs = unmapped.merge(
        logs[["understat_id", "date", "time", "h_goals", "a_goals"]],
        left_on=["date", "team_h_score", "team_a_score"],
        right_on=["date", "h_goals", "a_goals"],
    )
    pairs = pairs[(pairs["minutes"] - pairs["time"]).abs() <= 15]
    pairs = pairs[~pairs["understat_id"].isin(understat_id.values())]
    hits = pairs.groupby(["fpl_id", "understat_id"]).size().rename("hits").reset_index()
    hits["share"] = hits["hits"] / hits["fpl_id"].map(unmapped.groupby("fpl_id").size())
    fpl_name = unmapped.drop_duplicates("fpl_id").set_index("fpl_id")["name"]
    hits["similarity"] = [
        name_similarity(fpl_name[f], files[u].rsplit("_", 1)[0])
        for f, u in zip(hits["fpl_id"], hits["understat_id"])
    ]
    best = hits.sort_values(["hits", "similarity"], ascending=False).drop_duplicates(
        "fpl_id"
    )
    best = best[best["share"] >= 0.7].drop_duplicates("understat_id")
    understat_id.update(zip(best["fpl_id"], best["understat_id"]))

    logs["fpl_id"] = logs["understat_id"].map({u: f for f, u in understat_id.items()})
    logs = logs.dropna(subset=["fpl_id"]).astype({"fpl_id": int})
    logs = logs.groupby(["fpl_id", "date"], as_index=False)[UNDERSTAT_STATS].sum()
    return logs, understat_id


def build():
    """(rows of the players who played, rows of every player with a fixture)"""
    season_start = int(YEARS[0])
    gw = read_csv(
        VAASTAV + "gws/merged_gw.csv", f"vaastav_{config.SEASON}/merged_gw.csv"
    )
    fixtures = read_csv(
        VAASTAV + "fixtures.csv", f"vaastav_{config.SEASON}/fixtures.csv"
    )
    teams = read_csv(VAASTAV + "teams.csv", f"vaastav_{config.SEASON}/teams.csv")
    players = read_csv(
        VAASTAV + "players_raw.csv", f"vaastav_{config.SEASON}/players_raw.csv"
    )
    team_name = dict(zip(teams["id"], teams["name"]))

    df = (
        gw.rename(columns={"element": "fpl_id"})
        .drop(columns=["round"])
        .rename(columns={"GW": "round"})
    )
    df["opponent_team"] = df["opponent_team"].map(team_name)

    # Ownership as a percentage of all managers: every manager owns 15 players. Players without a
    # fixture have no row, so blank gameweeks undercount; the number of managers only grows.
    managers = (
        df.drop_duplicates(["fpl_id", "round"]).groupby("round")["selected"].sum() / 15
    )
    managers = managers.cummax()
    df["selected_by_percent"] = (
        df["selected"] / df["round"].map(managers) * 100
    ).round(1)

    # Fixture difficulty, home and away teams
    fx = fixtures.rename(columns={"id": "fixture"})[
        ["fixture", "team_h", "team_a", "team_h_difficulty", "team_a_difficulty"]
    ]
    fx["team_h"] = fx["team_h"].map(team_name)
    fx["team_a"] = fx["team_a"].map(team_name)
    df = df.merge(fx, on="fixture", how="left", validate="many_to_one")

    # Odds and Elo, from the player's side
    home = df["was_home"].astype(bool)
    match = pd.MultiIndex.from_frame(df[["team_h", "team_a"]])
    probs = odds_probabilities(season_start).reindex(match)
    df["win_prob"] = np.where(home, probs["home_win_prob"], probs["away_win_prob"])
    df["draw_prob"] = probs["draw_prob"].to_numpy()
    df["lose_prob"] = np.where(home, probs["away_win_prob"], probs["home_win_prob"])
    elo = pd.DataFrame(elo_ratings(season_start), index=["elo_h", "elo_a"]).T.reindex(
        match
    )
    df["elo_h"] = elo["elo_h"].round(2).to_numpy()
    df["elo_a"] = elo["elo_a"].round(2).to_numpy()
    df["elo"] = np.where(home, df["elo_h"], df["elo_a"])
    df["opp_elo"] = np.where(home, df["elo_a"], df["elo_h"])
    df["opp_elo_h"], df["opp_elo_a"] = df["elo_h"], df["elo_a"]  # copies, as in 2024-25

    # One row per player and gameweek; first fixture of a double gameweek
    df = df.sort_values(["fpl_id", "round", "kickoff_time"])
    df = df.drop_duplicates(["fpl_id", "round"], keep="first").reset_index(drop=True)
    played = df["minutes"] > 0

    # Understat match stats, joined on the match date
    df["date"] = pd.to_datetime(df["kickoff_time"]).dt.date.astype(str)
    logs, understat_id = understat_matches(
        df.loc[
            played,
            ["fpl_id", "name", "date", "minutes", "team_h_score", "team_a_score"],
        ]
    )
    df = df.merge(logs, on=["fpl_id", "date"], how="left", validate="one_to_one")
    df[UNDERSTAT_STATS] = df[UNDERSTAT_STATS].fillna(0.0)
    df["understat_id"] = df["fpl_id"].map(understat_id)

    df["points"] = df["total_points"] - df["bonus"]
    df["fpl_name"] = df["fpl_id"].map(players.set_index("id")["web_name"])
    df["season"] = 2000 + season_start

    # Lagged features, from the gameweeks players played (0 for a gameweek without minutes)
    lags = [fix_missing_xp.lagged_features(df[played], stat) for stat in LAGGED_STATS]
    lagged = lags[0]
    for f in lags[1:]:
        lagged = lagged.merge(f, on=["round", "fpl_id"])
    df = df.merge(lagged, on=["fpl_id", "round"], how="left", validate="one_to_one")
    lag_cols = [c for c in lagged.columns if c not in ("round", "fpl_id")]
    df[lag_cols] = df[lag_cols].fillna(0.0)  # players who have not played yet

    first = [
        "fpl_id",
        "round",
        "name",
        "fpl_name",
        "position",
        "team",
        "opponent_team",
        "was_home",
        "fixture",
        "kickoff_time",
        "date",
        "season",
        "xP",
        "total_points",
    ]
    df = df[first + [c for c in df.columns if c not in first]]
    return df[played].reset_index(drop=True), df


if __name__ == "__main__":
    df, everyone = build()
    missing = {c: int(n) for c, n in df.isna().sum().items() if n}
    df.to_csv(config.RAW_DATA, index=False)
    everyone.to_csv(config.ALL_PLAYERS_DATA, index=False)
    print(
        f"{len(df)} rows, {df['fpl_id'].nunique()} players, gameweeks {df['round'].min()}-{df['round'].max()}"
    )
    print("Teams:", sorted(df["team"].unique()))
    print("Columns with empty values:", missing or "none")
    print(f"Wrote {config.RAW_DATA}")
    print(
        f"Wrote {config.ALL_PLAYERS_DATA} ({len(everyone)} rows, every player with a fixture)"
    )
