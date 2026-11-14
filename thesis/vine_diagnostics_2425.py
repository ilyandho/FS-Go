"""
Vine diagnostics for Chapter 4 (2024-25): for every gameweek of the rolling season and that
week's opponent, the sparse C-vine used by optimization.ipynb and the Chapter 3 C-vine
(§3.2.2: six families, AIC, truncated at tree 15, over the whole pool), each on the real and on
the column-permuted latent uniforms U, plus held-out log-likelihoods.

The history (pool, overhaul filtering, data augmentation, eqs. 3.6-3.7) is built by the
notebook's own run_copula_pipeline (cell 11), run as in the rolling season with diagnostics=True;
the only addition is one line that keeps a copy of U. Writes thesis/outputs/results/vine_diagnostics_2425.csv.
"""
import json
import multiprocessing
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyvinecopulib as pv
from threadpoolctl import threadpool_limits

warnings.filterwarnings("ignore")
THESIS = Path(__file__).resolve().parent
NOTEBOOK_DIR = THESIS.parent / "24-25/optimizer/notebooks"
OUT = THESIS / "outputs/results/vine_diagnostics_2425.csv"
SEED = 42  # the rolling season's seed
HOLDOUT_WEEKS = (4, 6, 8)  # as vine_diagnostics in the notebook
WORKERS = 4

# the Chapter 3 vine: the original settings in copula_estimation.ipynb (git HEAD)
CH3_FAMILIES = [pv.BicopFamily.indep, pv.BicopFamily.gaussian, pv.BicopFamily.clayton,
                pv.BicopFamily.gumbel, pv.BicopFamily.frank, pv.BicopFamily.joe]
CH3_TRUNC = 15

os.chdir(NOTEBOOK_DIR)
nb = json.load(open("optimization.ipynb"))
g = {"__name__": "__main__"}
for i in [2, 3, 5, 7, 9]:
    exec("".join(nb["cells"][i]["source"]), g)
src = "".join(nb["cells"][11]["source"])
old = "    U_vals = latent_uniforms(X_hist, P_hist, rng).values\n"
assert src.count(old) == 1
src = src.replace(old, old + "    _U_CAPTURE[TEAM_ID, GW] = (U_vals.copy(), X_hist.values.sum(axis=0) > 0)\n")
g["_U_CAPTURE"] = {}
exec(src, g)


def fit_ch3_vine(U):
    order = g["kendall_cvine_order"](U)
    structure = pv.CVineStructure(order=order, trunc_lvl=min(CH3_TRUNC, U.shape[1] - 1))
    controls = pv.FitControlsVinecop(family_set=CH3_FAMILIES, selection_criterion="aic",
                                     trunc_lvl=CH3_TRUNC)
    return pv.Vinecop.from_data(U, structure=structure, controls=controls)


def n_dependent(vine):
    return sum(f != pv.BicopFamily.indep for tree in vine.families for f in tree)


def holdout(fit, U):
    """Mean log-likelihood of the last k weeks under a vine fit on the earlier ones (0 =
    independence), over HOLDOUT_WEEKS, as the notebook's vine_diagnostics."""
    scores = [fit(U[:-k]).loglik(U[-k:]) for k in HOLDOUT_WEEKS if U.shape[0] - k >= 5]
    return float(np.mean(scores)) if scores else np.nan


def one_week(gw):
    t = time.time()
    with threadpool_limits(2):
        opp = g["opponents_by_round"][gw]
        res = g["run_copula_pipeline"](gw, opp, N_SAMPLES=20_000, seed=SEED, verbose=False,
                                       diagnostics=True)
        row = {"GW": gw, "opponent": opp}
        if res is None or (opp, gw) not in g["_U_CAPTURE"]:
            row["status"] = "no history"
            return row
        U, informative = g["_U_CAPTURE"][opp, gw]
        row |= {
            "status": "ok",
            "n_history_weeks": res["n_history_weeks"],
            "overhaul_rounds": res["overhaul_rounds"],
            "M": res["M"],
            "T_limit": res["T_LIMIT"],
            "n_draws_feasible": res["n_feasible"],
            "n_unique_squads": res["n_unique_feasible_squads"],
            # the sparse vine (notebook settings, informative players only)
            "sparse_dim": res["n_informative"],
            "sparse_dep_pairs": res["n_dep_pairs"],
            "sparse_loglik": res["vine_loglik"],
            "sparse_null_loglik": res["null_loglik"],
            "sparse_holdout_loglik": res["holdout_loglik"],
        }
        # the Chapter 3 vine over the whole pool, on real and on column-permuted U
        rng = np.random.default_rng(SEED + gw)
        U_null = np.column_stack([rng.permutation(U[:, k]) for k in range(U.shape[1])])
        real, null = fit_ch3_vine(U), fit_ch3_vine(U_null)
        row |= {
            "ch3_dim": U.shape[1],
            "ch3_pairs": sum(len(tree) for tree in real.families),
            "ch3_dep_pairs": n_dependent(real),
            "ch3_loglik": real.loglik(U),
            "ch3_null_dep_pairs": n_dependent(null),
            "ch3_null_loglik": null.loglik(U_null),
            "ch3_holdout_loglik": holdout(fit_ch3_vine, U),
            "seconds": round(time.time() - t),
        }
    print(f"GW{gw}: " + ", ".join(f"{k}={v}" for k, v in row.items() if k not in ("overhaul_rounds",)),
          flush=True)
    return row


if __name__ == "__main__":
    if OUT.exists():
        raise SystemExit(f"{OUT} exists; not overwriting it")
    OUT.parent.mkdir(exist_ok=True)
    start = max(min(f) for f in g["LEAGUE_FITS"].values())
    with multiprocessing.get_context("fork").Pool(WORKERS) as pool:
        rows = pool.map(one_week, range(start, 39), chunksize=1)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"wrote {OUT}", flush=True)
