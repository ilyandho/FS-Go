"""Settings for the structural xP model. Paths are relative to src/ (they resolve the same from notebooks/).

The large inputs are shared with the rest of 23-24/ and read from ../../inputs/ (built by 23-24/src/build_rolled_data.py
and fix_missing_xp.py, see 23-24/README.md). Everything this model creates is kept inside this folder.
"""

POSITIONS = ["GK", "DEF", "MID", "FWD"]

# Season of the data: "23-24" or "24-25".
SEASON = "23-24"
SEASON_DIR = {"23-24": "2023-24", "24-25": "2024-25"}[SEASON]  # vaastav's folder name

# Gameweeks with no target (total_points = 0 for every player): left out of training and of every score,
# still predicted.
MISSING_TARGET_GWS = {"23-24": [], "24-25": []}[SEASON]

# Shared inputs (not written by this folder)
SHARED_INPUTS = "../../inputs"
PLAYERS_FILE = f"{SHARED_INPUTS}/rolled_player_data_{SEASON}_all_players_xp_fixed.csv"  # every player with a fixture
VAASTAV = f"{SHARED_INPUTS}/sources/vaastav_{SEASON}"  # merged_gw.csv, fixtures.csv

# This folder's own inputs and outputs
SNAPSHOT_DIR = f"../inputs/sources/players_raw_snapshots_{SEASON}"  # build_availability.py download cache
AVAILABILITY = f"../inputs/availability_{SEASON}.csv"  # build_availability.py
OUTPUT = f"../outputs/structural_predictions_{SEASON}.csv"  # structural.py

FIRST_GW, LAST_GW = 5, 38  # gameweeks predicted by default
