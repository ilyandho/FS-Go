"""
The 2024-25 benchmark-rank (r) sweep for Chapter 4.

Runs optimization.ipynb's rolling season (run_rolling_optimization) at each R_PCT, and the
baseline once, without editing the notebook. The notebook's code cells are executed as they are,
except cell 17 (run_optimization_pipeline), which is re-executed with three recording-only lines
added; `check` confirms that this copy returns exactly the notebook function's result.

Extra fields per gameweek (results column `extra`):
    sense (branch of eq. 3.17), mu_Yw_current, mu_G_r, sd_G_r, |W_o| (unique squads and accepted
    draws), r_index, G_r_is_max, the chosen line-up and the chosen team's mean and variance,
    and the frontier over lambda (P(beat), mean, variance, SCIP status). The baseline gets no
    P(beat) fields.

Usage (from anywhere):
    python run_r_sweep_2425.py check                  # checks before the full runs
    python run_r_sweep_2425.py run mvt 0.6            # one MVT season at R_PCT = 0.6
    python run_r_sweep_2425.py run base               # the baseline season
Results go to 24-25/optimizer/notebooks/rolling_optimization_results_mvt_r{pct}.joblib and
rolling_optimization_results_base_rsweep.joblib. An existing file is never overwritten.
"""
import json
import os
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

warnings.filterwarnings("ignore")
NOTEBOOK_DIR = Path(__file__).resolve().parent.parent / "24-25/optimizer/notebooks"
SETUP_CELLS = [2, 3, 5, 7, 9, 11, 13, 15, 17, 19]  # every code cell but the run (21) and compare (23)
PIPELINE_CELL = 17
WORKERS = 3  # seasons run at the same time; the cores are split between them
MVT_TIME_LIMIT = 60  # seconds per SCIP solve, for every MVT run (the saved runs used 30)
SAVED_TIME_LIMIT = 30
# the saved r = 0.5 runs' other arguments (optimization.ipynb, cell 21)
RUN_KWARGS = dict(N_SAMPLES_OPP=20_000, N_MC=2000, seed=42)

os.chdir(NOTEBOOK_DIR)
nb = json.load(open("optimization.ipynb"))
g = {"__name__": "__main__"}
for i in SETUP_CELLS:
    exec("".join(nb["cells"][i]["source"]), g)
notebook_pipeline = g["run_optimization_pipeline"]

# --- the recording copy of run_optimization_pipeline -----------------------------------------
src = "".join(nb["cells"][PIPELINE_CELL]["source"])
injections = [
    (
        "    best_prob, best_w, best_v, best_lam = -1.0, None, None, None\n",
        "    _frontier = []\n    best_prob, best_w, best_v, best_lam = -1.0, None, None, None\n",
    ),
    (
        "        prob_beat = float((delta_samples_ @ v_lam - G_r_ > 0).mean())\n",
        "        prob_beat = float((delta_samples_ @ v_lam - G_r_ > 0).mean())\n"
        "        _frontier.append((lam, prob_beat, w_lam, v_lam))\n",
    ),
]
for old, new in injections:
    assert src.count(old) == 1, old
    src = src.replace(old, new)
head, tail = src.rsplit("    return result", 1)
src = head + '    result["extra"] = _capture(locals())\n    return result' + tail
SOLVE_LOG, EXTRAS = [], {}
notebook_solve = g["solve_for_lambda"]
notebook_build = g["build_mean_variance_model"]


def build_without_perspective(*args, **kwargs):
    """The notebook's model with SCIP's perspective nonlinear handler switched off. With it on,
    SCIP 10 stops with "error in input data" on the nonconvex branch (e.g. GW6 at r = 1.0,
    lambda = 2.5) and the process then segfaults; off, every solve of that week is optimal.
    It is a solver setting, not a model change: a solve that reaches optimality returns the same
    optimum either way."""
    out = notebook_build(*args, **kwargs)
    out[0].setParam("nlhdlr/perspective/enabled", False)
    return out


def logged_solve(m, wv, lin, q, lin_cov, P, lam, sense=1):
    """The notebook's solve_for_lambda, with its SCIP status logged. A SCIP error at one lambda
    is logged (status "error") and that lambda skipped, where the notebook would lose the whole
    gameweek; the other lambdas are still solved."""
    try:
        sol = notebook_solve(m, wv, lin, q, lin_cov, P, lam, sense=sense)
        status = m.getStatus()
    except Exception as e:
        print(f"SCIP error at lambda = {lam}: {e!r}", flush=True)
        sol, status = None, "error"
    SOLVE_LOG.append({"lambda": float(lam), "status": status, "solved": sol is not None})
    return sol


def _capture(L):
    mu, Sigma = L["mu_u_"], L["Sigma_u_"]
    v = L["best_v"] if L["made_transfer_"] else L["current_v_"]
    starters, captain, vice, bench = L["lineup_star_"]
    status = {s["lambda"]: s["status"] for s in SOLVE_LOG}
    extra = {
        "sense": L["sense_"],
        "mu_Yw_current": L["mu_Yw_check_"],
        "mu_G_r": float(L["mu_G_r_"]),
        "sd_G_r": float(L["G_r_"].std()),
        "n_unique_Wo": int(L["Wo_"].shape[0]),
        "n_draws_Wo": int(sum(L["Wo_weights_"])),
        "r_index_from_top": int(L["r_index_from_top_"]),
        "G_r_is_max": bool(np.allclose(L["G_r_"], L["G_all_"].max(axis=1))),
        "lineup": {"starters": starters, "captain": captain, "vice": vice, "bench": bench},
        "team_mean": float(mu @ v),
        "team_var": float(v @ Sigma @ v),
        "solve_log": list(SOLVE_LOG),
        "frontier": [
            {
                "lambda": float(lam),
                "prob_beat": prob,
                "mean": float(mu @ v_lam),
                "var": float(v_lam @ Sigma @ v_lam),
                "n_transfers": int(round((w_lam * (1 - L["owned_vec_"])).sum())),
                "status": status.get(float(lam)),
            }
            for lam, prob, w_lam, v_lam in L["_frontier"]
        ],
    }
    if L["method"] == "base":  # the baseline doesn't use the opponent model
        for key in ["mu_Yw_current", "mu_G_r", "sd_G_r", "G_r_is_max", "r_index_from_top"]:
            extra.pop(key)
        for f in extra["frontier"]:
            f.pop("prob_beat")
    return extra


g["_capture"] = _capture
g["solve_for_lambda"] = logged_solve
g["build_mean_variance_model"] = build_without_perspective
exec(src, g)
recording_pipeline = g["run_optimization_pipeline"]


def recorded(GW, *args, **kwargs):
    """The recording copy; keeps the extra fields aside so the result is the notebook's."""
    SOLVE_LOG.clear()
    res = recording_pipeline(GW, *args, **kwargs)
    if res is None:
        EXTRAS[GW] = {"status": "skipped", "solve_log": list(SOLVE_LOG)}
    else:
        EXTRAS[GW] = res.pop("extra")
    return res


g["run_optimization_pipeline"] = recorded  # what run_rolling_optimization calls
START_GW = max(min(fits) for fits in g["LEAGUE_FITS"].values())


def rolling(method, r, start=START_GW, end=38, time_limit=MVT_TIME_LIMIT):
    EXTRAS.clear()
    df = g["run_rolling_optimization"](
        start, end, g["OWN_TEAM_ID"], R_PCT=r, method=method,
        mip_time_limit=time_limit, **RUN_KWARGS,
    )
    df["extra"] = df["GW"].map(dict(EXTRAS))
    return df


def decisions(df):
    cols = ["GW", "status", "made_transfer", "lambda_star", "prob_beat_opponent", "squad",
            "sim_optimized_points", "sim_optimized_xP"]
    return df[[c for c in cols if c in df]].reset_index(drop=True)


def check():
    t = time.time()
    with threadpool_limits(max(1, (os.cpu_count() or 2) // WORKERS)):
        # 1. the recording copy returns the notebook function's result (GW12, convex branch)
        kw = dict(R_PCT=0.5, N_SAMPLES_OPP=20_000, N_MC=2000, seed=42, verbose=False,
                  mip_time_limit=SAVED_TIME_LIMIT, method="mvt")
        opp = g["opponents_by_round"][12]
        a = notebook_pipeline(12, g["OWN_TEAM_ID"], opp, **kw)
        b = recorded(12, g["OWN_TEAM_ID"], opp, **kw)
        same = joblib.hash(a) == joblib.hash(b)
        print(f"[check 1] recording copy == notebook function at GW12: {same} "
              f"({time.time() - t:.0f} s for both)", flush=True)
        print(f"          GW12 extra: sense {EXTRAS[12]['sense']}, |W_o| {EXTRAS[12]['n_unique_Wo']} "
              f"unique of {EXTRAS[12]['n_draws_Wo']} draws, statuses "
              f"{sorted({s['status'] for s in EXTRAS[12]['solve_log']})}", flush=True)
        # 2. reproducibility: the first three gameweeks of the saved r = 0.5 MVT season
        t2 = time.time()
        new = rolling("mvt", 0.5, START_GW, START_GW + 2, time_limit=SAVED_TIME_LIMIT)
        per_gw = (time.time() - t2) / 3
        saved = joblib.load("rolling_optimization_results_mvt.joblib")
        saved = saved[saved["GW"].between(START_GW, START_GW + 2)]
        dn, ds = decisions(new), decisions(saved)
        print(f"[check 2] GW{START_GW}-{START_GW + 2} at r = 0.5, 30 s: "
              f"{per_gw / 60:.1f} min per gameweek", flush=True)
        for (_, x), (_, y) in zip(dn.iterrows(), ds.iterrows()):
            diffs = [c for c in dn.columns if joblib.hash(x[c]) != joblib.hash(y[c])]
            ex = EXTRAS.get(x["GW"], {})
            limits = sorted({s["status"] for s in ex.get("solve_log", [])})
            print(f"          GW{x['GW']}: {'match' if not diffs else 'DIFFER in ' + str(diffs)}"
                  f" | transfer {x['made_transfer']} lambda* {x['lambda_star']} "
                  f"P(beat) {x['prob_beat_opponent']:.3f} (saved {y['prob_beat_opponent']:.3f})"
                  f" | SCIP {limits} | fallback {x.get('used_fallback_benchmark')}", flush=True)
    # 3. R_PCT = 1.0 selects the maximum over W_o: index ceil((1 - 1.0) * O) = 0, clamped to 1,
    #    i.e. G_sorted_[:, -1]; each run records G_r_is_max per week to confirm it
    O = 7
    idx = min(max(int(np.ceil((1 - 1.0) * O)), 1), O)
    G = np.random.default_rng(0).normal(size=(5, O))
    print(f"[check 3] R_PCT = 1.0 -> index {idx} from the top; equals the max: "
          f"{np.allclose(np.sort(G, axis=1)[:, -idx], G.max(axis=1))}", flush=True)
    print(f"checks done in {(time.time() - t) / 60:.0f} min", flush=True)


def run(method, r=None):
    name = (f"rolling_optimization_results_mvt_r{round(r * 100)}.joblib" if method == "mvt"
            else "rolling_optimization_results_base_rsweep.joblib")
    out = NOTEBOOK_DIR / name
    if out.exists():
        sys.exit(f"{out} exists; not overwriting it")
    t = time.time()
    with threadpool_limits(max(1, (os.cpu_count() or 2) // WORKERS)):
        if method == "mvt":
            df = rolling("mvt", r)
        else:  # the saved settings: decisions must match the saved baseline
            df = rolling("base", 0.5, time_limit=SAVED_TIME_LIMIT)
    if out.exists():
        sys.exit(f"{out} appeared during the run; not overwriting it")
    joblib.dump(df, out)
    skipped = df.loc[df["status"] != "ok", "GW"].tolist()
    fallback = df.loc[df.get("used_fallback_benchmark", False) == True, "GW"].tolist()
    print(f"{name}: {len(df) - len(skipped)} / {len(df)} gameweeks solved, skipped {skipped}, "
          f"single-squad fallback {fallback}, {(time.time() - t) / 60:.0f} min", flush=True)
    if method == "base":
        saved = decisions(joblib.load("rolling_optimization_results_base.joblib"))
        new = decisions(df)
        cols = ["GW", "status", "made_transfer", "squad", "sim_optimized_points"]
        bad = [int(x["GW"]) for (_, x), (_, y) in zip(new.iterrows(), saved.iterrows())
               if any(joblib.hash(x[c]) != joblib.hash(y[c]) for c in cols)]
        print(f"baseline vs saved: {'identical decisions' if not bad else 'DIFFER in GW ' + str(bad)}",
              flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "check":
        check()
    elif sys.argv[1] == "run":
        run(sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else None)
    print("done", flush=True)
