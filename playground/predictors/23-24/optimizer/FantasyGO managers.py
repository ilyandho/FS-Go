"""Organise the FantasyGo manager data: the players each manager picked, their points, captain and prize.

The FantasyGo data is from the 2023-24 season (config.SEASON). Player ids are the FPL ids of that season,
the same as fpl_id in the pipeline's data, so the picks join to predictions.csv on (fpl_id, round).
The FPL data comes from vaastav, through src/build_rolled_data.py's download cache (inputs/sources/).

Writes
  data/fg_managers.csv       one row per manager entry and gameweek: points, prize, squad, captain, vice
  data/fg_manager_picks.csv  one row per player picked

Usage: python "FantasyGO managers.py"
"""

import ast
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import build_rolled_data as source
import config

source.CACHE = ROOT / "inputs" / "sources"  # its own path is relative to src/

FG_MANAGERS_URL = (
    "https://raw.githubusercontent.com/ilyandho/FPL-Optimal-Transfer/refs/heads/"
    "FantasyGo/FPL%20predictors/Fantasy%20Go/fgo_managers.csv"
)
OUT_DIR = Path(__file__).parent / "data"
POSITIONS = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}

# Misspelt names in the FantasyGo picks -> FPL web_name
NAME_FIXES = {
    "Bradely": "Bradley",
    "De Bryune": "De Bruyne",
    "Isak.": "Isak",
    "Mitooma": "Mitoma",
    "N.Semendo": "N.Semedo",
    "Vinicius": "Vinícius",
}

FG_managers = source.read_csv(FG_MANAGERS_URL, "fantasygo/fgo_managers.csv")
players = source.read_csv(
    source.VAASTAV + "players_raw.csv", f"vaastav_{config.SEASON}/players_raw.csv"
)
teams = source.read_csv(
    source.VAASTAV + "teams.csv", f"vaastav_{config.SEASON}/teams.csv"
)
# Points per player and gameweek, double gameweeks summed (the picks score both matches)
gw_points = (
    source.read_csv(
        source.VAASTAV + "gws/merged_gw.csv", f"vaastav_{config.SEASON}/merged_gw.csv"
    )
    .groupby(["element", "GW"])["total_points"]
    .sum()
)

players["full_name"] = players["first_name"] + " " + players["second_name"]
players["team_name"] = players["team"].map(teams.set_index("id")["name"])
players["position"] = players["element_type"].map(POSITIONS)
players = players.set_index("id")
candidates = players.reset_index().groupby("web_name")["id"].apply(list)


def to_number(text):
    # "R 8,180.18" -> 8180.18, "81.5 Points" -> 81.5
    if pd.isna(text):
        return 0.0
    return float(
        str(text).replace("R", "").replace("Points", "").replace(",", "").strip()
    )


def resolve_id(name, gameweek, points, multiplier):
    # Some web_names are shared (Onana, Wilson, ...): pick the player whose FPL points
    # that gameweek match the FantasyGo points, otherwise the higher scorer over the season
    ids = candidates.get(NAME_FIXES.get(name, name), [])
    if len(ids) <= 1:
        return ids[0] if ids else None
    base = points / multiplier
    matching = [i for i in ids if gw_points.get((i, gameweek), 0) == base]
    return max(matching or ids, key=lambda i: players.at[i, "total_points"])


# One row per player picked
pick_rows = []
for row in FG_managers.itertuples():
    for pick in ast.literal_eval(row.pick):
        points = float(pick["points"])
        multiplier = 2 if pick["is_captin"] else 1.5 if pick["is_vice"] else 1
        player_id = resolve_id(pick["player_name"], row.gameweek, points, multiplier)
        player = players.loc[player_id] if player_id is not None else None
        pick_rows.append(
            {
                "row": row.Index,
                "gameweek": row.gameweek,
                "manager": row.manager,
                "entry": row.entry,
                "player_id": player_id,
                "player_name": player["full_name"]
                if player is not None
                else pick["player_name"],
                "fg_name": pick["player_name"],
                "team": player["team_name"] if player is not None else None,
                "position": player["position"] if player is not None else None,
                "points": points,
                "is_captain": pick["is_captin"],
                "is_vice": pick["is_vice"],
            }
        )
manager_picks = pd.DataFrame(pick_rows).astype({"player_id": "Int64"})

# One row per manager entry per gameweek
managers = FG_managers[["gameweek", "contestant_no", "manager", "entry"]].copy()
managers["points"] = FG_managers["points"].map(to_number)
managers["prize"] = FG_managers["prize"].map(to_number)
managers["prize_pool"] = FG_managers["prize_pool"].map(to_number)

# Picks are tied back to their FantasyGo row, since a manager can enter the same gameweek more than once
picks_by_row = manager_picks.set_index("row")
squads = picks_by_row.groupby(level="row").agg(
    player_ids=("player_id", list),
    player_names=("player_name", list),
)
captains = picks_by_row[picks_by_row["is_captain"]][
    ["player_id", "player_name", "points"]
].rename(columns=lambda c: "captain_" + c.replace("player_", ""))
vices = picks_by_row[picks_by_row["is_vice"]][["player_id", "player_name"]].rename(
    columns=lambda c: "vice_" + c.replace("player_", "")
)
# A couple of FantasyGo rows list two vice captains, keep the first
vices = vices[~vices.index.duplicated()]
managers = managers.join([squads, captains, vices])

if __name__ == "__main__":
    OUT_DIR.mkdir(exist_ok=True)
    managers.to_csv(OUT_DIR / "fg_managers.csv", index=False)
    manager_picks.to_csv(OUT_DIR / "fg_manager_picks.csv", index=False)

    unmatched = manager_picks[manager_picks["player_id"].isna()]["fg_name"].unique()
    print(f"{len(managers)} manager entries, {len(manager_picks)} picks")
    print("Unmatched player names:", list(unmatched) or "none")
    print(managers.head())
