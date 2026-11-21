"""Build each player's pre-deadline availability (FPL status and chance of playing) for every gameweek.

vaastav's players_raw.csv is a snapshot of the FPL API, overwritten each time the scraper runs, so the
season's file only holds the final state. Its git history keeps the earlier snapshots. For each gameweek
this script takes the last snapshot committed before the deadline (first kickoff - 90 minutes).

Downloads (cached in SNAPSHOT_DIR, so only the first run needs the network):
  - the commit list of data/{season}/players_raw.csv from the GitHub API (unauthenticated)
  - players_raw.csv at each commit that is some gameweek's latest pre-deadline snapshot

Writes AVAILABILITY, one row per player and gameweek present in a snapshot:
  fpl_id, round, status, chance_of_playing (chance_of_playing_next_round, empty = no news = 100),
  news, snapshot_time, snapshot_age_hours (deadline - snapshot time)

Usage (from src/): python build_availability.py
"""

import json
import os
import time
import urllib.request

import pandas as pd

import config

REPO = "vaastav/Fantasy-Premier-League"
SEASON_DIR = config.SEASON_DIR
SNAPSHOT_DIR = config.SNAPSHOT_DIR
COMMITS = f"{SNAPSHOT_DIR}/commits.csv"
AVAILABILITY = config.AVAILABILITY
DEADLINE_BEFORE_KICKOFF = pd.Timedelta(minutes=90)
HEADERS = {"User-Agent": "fpl-research-script"}


def get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=60) as r:
        return r.read()


def commits():
    """sha and commit time of every commit of the season's players_raw.csv (cached)."""
    if os.path.exists(COMMITS):
        return pd.read_csv(COMMITS, parse_dates=["time"])
    path = f"data/{SEASON_DIR}/players_raw.csv"
    rows, page = [], 1
    while True:
        url = f"https://api.github.com/repos/{REPO}/commits?path={path}&per_page=100&page={page}"
        batch = json.loads(get(url))
        rows += [{"sha": c["sha"], "time": c["commit"]["committer"]["date"]} for c in batch]
        if len(batch) < 100:
            break
        page += 1
    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df.time, utc=True)
    df = df.sort_values("time").reset_index(drop=True)
    df.to_csv(COMMITS, index=False)
    return df


def deadlines():
    f = pd.read_csv(f"{config.VAASTAV}/fixtures.csv")
    first = pd.to_datetime(f.kickoff_time, utc=True).groupby(f.event).min()
    return (first - DEADLINE_BEFORE_KICKOFF).rename("deadline").rename_axis("round").reset_index()


def snapshot(sha):
    """players_raw.csv at commit sha (cached)."""
    file = f"{SNAPSHOT_DIR}/players_raw_{sha[:10]}.csv"
    if not os.path.exists(file):
        url = f"https://raw.githubusercontent.com/{REPO}/{sha}/data/{SEASON_DIR}/players_raw.csv"
        with open(file, "wb") as fh:
            fh.write(get(url))
        time.sleep(0.5)
    return pd.read_csv(file)


def main():
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    c = commits()
    c["time"] = pd.to_datetime(c.time, utc=True)
    gw = pd.merge_asof(deadlines().sort_values("deadline"), c, left_on="deadline", right_on="time",
                       direction="backward").dropna(subset=["sha"])
    out = []
    for g in gw.itertuples():
        p = snapshot(g.sha)
        out.append(pd.DataFrame({
            "fpl_id": p["id"], "round": g.round, "status": p["status"],
            "chance_of_playing": p["chance_of_playing_next_round"], "news": p["news"],
            "snapshot_time": g.time, "snapshot_age_hours": (g.deadline - g.time).total_seconds() / 3600,
        }))
    av = pd.concat(out, ignore_index=True)
    av.to_csv(AVAILABILITY, index=False)
    age = gw.assign(h=(gw.deadline - gw.time).dt.total_seconds() / 3600)
    print(f"Wrote {AVAILABILITY}: {len(av)} rows, {av['round'].nunique()} gameweeks, "
          f"{gw.sha.nunique()} snapshots; snapshot age (hours) median {age.h.median():.0f}, max {age.h.max():.0f}")


if __name__ == "__main__":
    main()
