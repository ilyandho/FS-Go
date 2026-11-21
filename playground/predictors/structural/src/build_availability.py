"""Build each player's pre-deadline FPL status, chance of playing and set-piece orders for every gameweek.

vaastav's players_raw.csv is a snapshot of the FPL API, overwritten each time the scraper runs, so the
season's file only holds the final state. Its git history keeps the earlier snapshots. For each gameweek
this script takes the last snapshot committed before the deadline (first kickoff - 90 minutes).

Downloads (cached in paths(season).snapshot_dir, so only the first run needs the network):
  - the commit list of data/{season}/players_raw.csv from the GitHub API (unauthenticated)
  - players_raw.csv at each commit that is some gameweek's latest pre-deadline snapshot
Needs the season's fixtures.csv, cached by build_data.py (run it first).

Writes paths(season).availability, one row per player and gameweek present in a snapshot:
  fpl_id, round, status, chance_of_playing (chance_of_playing_next_round, empty = no news = 100),
  news, penalties_order, corners_order, freekicks_order (FPL's set-piece taker order, 1 = first choice,
  empty = not listed), snapshot_time, snapshot_age_hours (deadline - snapshot time)

Usage (from src/): python build_availability.py --season 24-25
"""

import argparse
import json
import os
import time
import urllib.request

import pandas as pd

import config

REPO = "vaastav/Fantasy-Premier-League"
DEADLINE_BEFORE_KICKOFF = pd.Timedelta(minutes=90)
HEADERS = {"User-Agent": "fpl-research-script"}


def get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=60) as r:
        return r.read()


def commits(p):
    """sha and commit time of every commit of the season's players_raw.csv (cached)."""
    file = f"{p.snapshot_dir}/commits.csv"
    if os.path.exists(file):
        df = pd.read_csv(file)
    else:
        path = f"data/{p.season_dir}/players_raw.csv"
        rows, page = [], 1
        while True:
            url = f"https://api.github.com/repos/{REPO}/commits?path={path}&per_page=100&page={page}"
            batch = json.loads(get(url))
            rows += [{"sha": c["sha"], "time": c["commit"]["committer"]["date"]} for c in batch]
            if len(batch) < 100:
                break
            page += 1
        df = pd.DataFrame(rows)
        df.to_csv(file, index=False)
    df["time"] = pd.to_datetime(df.time, utc=True)
    return df.sort_values("time").reset_index(drop=True)


def deadlines(p):
    f = pd.read_csv(f"{p.vaastav_cache}/fixtures.csv")
    first = pd.to_datetime(f.kickoff_time, utc=True).groupby(f.event).min()
    return (first - DEADLINE_BEFORE_KICKOFF).rename("deadline").rename_axis("round").reset_index()


def snapshot(p, sha):
    """players_raw.csv at commit sha (cached)."""
    file = f"{p.snapshot_dir}/players_raw_{sha[:10]}.csv"
    if not os.path.exists(file):
        url = f"https://raw.githubusercontent.com/{REPO}/{sha}/data/{p.season_dir}/players_raw.csv"
        with open(file, "wb") as fh:
            fh.write(get(url))
        time.sleep(0.5)
    return pd.read_csv(file)


def build(season):
    p = config.paths(season)
    os.makedirs(p.snapshot_dir, exist_ok=True)
    gw = pd.merge_asof(deadlines(p).sort_values("deadline"), commits(p), left_on="deadline", right_on="time",
                       direction="backward").dropna(subset=["sha"])
    out = []
    for g in gw.itertuples():
        snap = snapshot(p, g.sha)
        out.append(pd.DataFrame({
            "fpl_id": snap["id"], "round": g.round, "status": snap["status"],
            "chance_of_playing": snap["chance_of_playing_next_round"], "news": snap["news"],
            "penalties_order": snap["penalties_order"],
            "corners_order": snap["corners_and_indirect_freekicks_order"],
            "freekicks_order": snap["direct_freekicks_order"],
            "snapshot_time": g.time, "snapshot_age_hours": (g.deadline - g.time).total_seconds() / 3600,
        }))
    return pd.concat(out, ignore_index=True), gw


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--season", default=config.DEFAULT_SEASON, choices=list(config.SEASONS))
    args = parser.parse_args()
    p = config.paths(args.season)
    av, gw = build(args.season)
    os.makedirs(os.path.dirname(p.availability), exist_ok=True)
    av.to_csv(p.availability, index=False)
    age = (gw.deadline - gw.time).dt.total_seconds() / 3600
    stale = gw.loc[age > 7 * 24, "round"].tolist()
    print(f"Wrote {p.availability}: {len(av)} rows, {av['round'].nunique()} gameweeks, {gw.sha.nunique()} snapshots; "
          f"snapshot age (hours) median {age.median():.0f}, max {age.max():.0f}; older than a week: GW {stale}")


if __name__ == "__main__":
    main()
