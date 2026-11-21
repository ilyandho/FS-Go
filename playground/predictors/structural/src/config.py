"""Settings for the structural xP model, per season. Paths are relative to src/ (they resolve the same from notebooks/).

Every script takes the season (--season, default DEFAULT_SEASON) and gets its paths from paths(season).
To add a season: add it to SEASONS, then run build_data.py and build_availability.py for it.
"""

from types import SimpleNamespace

POSITIONS = ["GK", "DEF", "MID", "FWD"]
DEFAULT_SEASON = "23-24"

# missing_target_gws: gameweeks with no target (total_points = 0 for every player); left out of training and
# of every score, still predicted.
SEASONS = {
    "23-24": {"missing_target_gws": []},
    "24-25": {"missing_target_gws": []},
}

FIRST_GW, LAST_GW = 5, 38  # gameweeks predicted by default

DATA = "../data"
SOURCES = f"{DATA}/sources"  # download caches, shared by the seasons where they overlap (football-data)
OUTPUTS = "../outputs"


def paths(season):
    """Every path and setting of a season."""
    if season not in SEASONS:
        raise ValueError(f"unknown season {season!r}; add it to config.SEASONS (have {list(SEASONS)})")
    yy, zz = season.split("-")
    return SimpleNamespace(
        season=season,
        season_dir=f"20{yy}-{zz}",  # vaastav's folder name, e.g. 2023-24
        start_year=2000 + int(yy),
        missing_target_gws=SEASONS[season]["missing_target_gws"],
        # build_data.py
        vaastav_cache=f"{SOURCES}/vaastav_{season}",  # merged_gw.csv, fixtures.csv, teams.csv, understat/, ...
        players=f"{DATA}/{season}/players.csv",  # every player with a fixture, with lagged features
        # build_availability.py
        snapshot_dir=f"{SOURCES}/players_raw_snapshots_{season}",
        availability=f"{DATA}/{season}/availability.csv",
        # structural.py
        output=output_path(season),
    )


def output_path(season, history=()):
    """structural.py's predictions for a season, trained on the season alone or with earlier seasons (history)."""
    suffix = f"_history_{'+'.join(history)}" if history else ""
    return f"{OUTPUTS}/{season}/structural_predictions{suffix}.csv"
