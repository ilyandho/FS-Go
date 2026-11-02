# Role
You are co-authoring my MSc (Mathematics) dissertation at Mbarara University of Science and
Technology: *"An Optimization Problem for a Head-to-Head Competition of the Fantasy Football
League"*. You are an academic writer and research engineer in operations research and Bayesian
statistics, and you have full access to the project repo (`FS-Go/`). You get every number by
reading or running the project's code and saved results. You never invent or estimate numbers.

# Task
1. Run the missing benchmark-rank (r) sweep for the 2024-25 H2H season.
2. Write **Chapter 4: Simulation and Results** from scratch, about **4,800 words**.

# Read first
The dissertation PDF in the repo root (`An Optimization Problem for a Head-to-Head Competition of
the Fantasy Football League.pdf`). **Read Chapters 1-3 only. Ignore the existing Chapter 4
completely**: don't reuse its structure, text, numbers or figures.
- **Chapter 1, §1.3.2**: the objectives the results must answer.
- **Chapter 2**: it positions Haugh & Singal [1] as the framework adapted to H2H.
- **Chapter 3**: the notation and numbering you must reuse:
  - W, 𝕎_o, G^{(r')}, r and r'
  - eqs. (3.2)-(3.5): the league baseline, its posterior, α_{j,m,t,k}, θ_{j,m} = {Δβ_{j,m}, γ_m}
  - (3.6)-(3.7): data augmentation
  - (3.8)-(3.9): Kendall's τ and the hub score
  - (3.10)-(3.14): the truncated C-vine and pair-copula selection by AIC
  - (3.15): the inverse PIT
  - (3.16): the double-up problem
  - Theorem 3.2.3.1 / (3.17), Lemma 3.2.2, Algorithm 1

# The framing (follow it throughout)
1. **H2H as a special case of the double-up, solved with Haugh & Singal's methods [1].** We take
   the formulation and solution procedure of Haugh & Singal, "How to play fantasy sports
   strategically (and win)" [1]:
   - solve max_{w∈W} P(wᵀδ > G^{(r')}(𝕎_o, δ)) (eq. 3.16)
   - reduce it to a mean-variance problem (Theorem 3.2.3.1)
   - solve a parametric MIQP over λ, with (μ_{G^{(r')}}, σ²_{G^{(r')}}, σ_{δ,G^{(r')}})
     estimated by Monte Carlo (Lemma 3.2.2)
   - pick λ* = argmax P(Y_{w(λ)} > 0) (Algorithm 1)

   Credit [1] as the source of the method, and cite Morton et al. [12] only where [1] itself
   relies on them. In H2H there is one opponent, so 𝕎_o is the set of squads that opponent could
   plausibly field, and G^{(r')} is the r-th order statistic over those squads.
2. **The benchmark rank r.** r = 100% (beat the opponent's best possible squad) is the
   winner-takes-all case, and lower r (e.g. 50%) is the double-up case. The chapter must show
   how r changes λ*, team variance and outcomes **in both seasons**.
3. **Non-strategic opponents.** As in [1], opponents are assumed not to react to our selection:
   𝕎_o comes only from their own historical selection behaviour. State this assumption where the
   model is set up, and discuss its consequences in the Discussion.
4. **In each season, the comparison is MVT vs. the baseline optimizer.** The baseline applies
   the optimizer directly to the player data and the predicted xP: each week it picks the team
   (and captain) with the highest total predicted xP under that season's rules, with no opponent
   model, no 𝕎_o and no Monte Carlo, and it is judged on the players' actual performance
   (points). Both are scored on the same opponents or field. Real history and real opponents are
   reference points only.

# The two stages
**Stage 1: Replication of [1] on FantasyGo, 2023-24 (`23-24/`).** FantasyGo round contests are
[1]'s native pool-style setting, so this stage validates our implementation.
- Team: 11 players, 1 GK, 3-5 DEF, 2-5 MID, 1-3 FWD, budget 80, at most 3 per club, captain
  ×2, vice ×1.5, unlimited changes each round.
- 𝕎_o: either simulated from a league-wide Dirichlet-Multinomial model, or the real field of
  the previous gameweek.
- Evaluation: against the real contest by points, rank and prize. The r sweep has already been
  run: `mvt_results_simulated_r{50,60,80,90,95,99,100}.pkl`.

**Stage 2: Application to the 2024-25 FPL H2H league (`24-25/`).** This is the contribution:
- one known opponent per week in a 20-manager round-robin
- 15-man squads with free transfers, bank and squad value carried forward
- the hierarchical selection model (eqs. 3.2-3.5) with Wildcard/Free Hit overhaul detection
- C-vine opponent squads
- a rolling season simulation of MVT vs. the baseline
So far only r = 50% has been run (see Step 2).

# Opponent squads: explain the sparse vine
Chapter 3 (§3.2.2, eqs. 3.10-3.14) describes a C-vine truncated at tree 15, with pair-copula
families chosen by AIC. The implementation uses a **sparse** vine instead:
- Gaussian pair-copulas only
- selected by BIC
- one tree
- a |τ| threshold
- only players with ownership history enter the vine

**The authority for the vine is cell 11 of `24-25/optimizer/notebooks/optimization.ipynb`**
(`fit_sparse_cvine`, `vine_diagnostics`, `run_copula_pipeline`). Take only the *rationale* from
the comments above the vine settings there and in
`24-25/optimizer/beta estimators/opponent betas/copula_estimation.ipynb`. That notebook is out
of date with the optimizer (it predates the squad and target-week α fixes) and contains other
code; do not describe its code or reuse its outputs.

**No diagnostics are recorded anywhere, so compute them fresh** with a new script,
`thesis/vine_diagnostics_2425.py`, that uses the optimizer's functions:
- for every gameweek of the rolling season (GW6-38) and that week's opponent
  (`opponents_by_round`), build the history exactly as `run_copula_pipeline` does
  (pool, overhaul filtering, data augmentation) and record:
  - the sparse vine's dimension, its number of non-independent pairs, and its log-likelihood
  - `null_loglik` (a refit on column-permuted U) and `holdout_loglik` (from `vine_diagnostics`)
- for the same weeks, fit the Chapter 3 vine (all six families, AIC, truncated at tree 15, over
  the whole pool) on the real and on the column-permuted U, and record the number of
  non-independent pairs and the held-out log-likelihood of each
- save the table to `thesis/results/vine_diagnostics_2425.csv`; use a fixed seed and log the run

Explain the result with these numbers in a dedicated subsection: whether the Chapter 3 vine
fits as many non-independent pairs on permuted data as on real data, whether it scores below
independence (0) on held-out weeks, and how the sparse vine's log-likelihood compares with its
null and its holdout. **Ignore `estimates/copula_results_all.joblib`**: it is from an earlier
version of the pipeline (it shows empty ensembles in GW30, 35 and 36 that have since been
fixed) and has no null or holdout log-likelihoods.

If the sparse vine also reduces to (near) independence, report that plainly as a finding about
the data, and say what it means for §3.2.2.

**Do not mention any alternative squad sampler (e.g. a "swap" sampler) anywhere.** The vine is
the only method used to build 𝕎_o. Ignore `optimization_swap.ipynb` and any swap/vine results.

# Step 1: Read the code (it is the authority on what was done)
**Stage 1**
- `23-24/README.md`, `23-24/src/`, `23-24/notebooks/model_comparison.ipynb` and
  `23-24/models/*.csv`
- `23-24/optimizer/FantasyGO managers.py` and `optimizer/data/*.csv`
- `23-24/optimizer/optimizer.ipynb`: the baseline optimizer.
- `23-24/optimizer/MVT/beta estimators.ipynb` and `league_models.pkl`
- `23-24/optimizer/MVT/mvt.ipynb`: the main experiment. Its "Adapted to FantasyGo" and "Caveats"
  sections list the departures from [1].
- The `mvt_results_*.pkl` files.

**Stage 2**
- `24-25/README.md`, `24-25/predictor/src/` and `24-25/predictor/models/*.csv`
- `24-25/optimizer/beta estimators/league betas/*` and `utils.py`
- `24-25/optimizer/beta estimators/opponent betas/longitudinal.ipynb` and
  `estimates/manager_fits_*.joblib` (see the opponent-squads section for `copula_estimation.ipynb`
  and `copula_results_all.joblib`)
- `24-25/optimizer/notebooks/optimization.ipynb` and
  `rolling_optimization_results_{base,mvt}.joblib` (the latter is r = 50%).

Take every hyperparameter, seed, manager ID and gameweek range from the code.

**Baselines** (see framing point 4: the optimizer on predicted xP, judged on actual points).
- **Stage 1:** `23-24/optimizer/optimizer.ipynb` (`best_xi`, PuLP): each gameweek from 6, the
  11 players, captain and vice with the highest predicted xP under the FantasyGo rules, scored
  on `total_points` against the real contest (rank, percentile, prize). The λ = 0 team in
  `mvt.ipynb` (`res["teams"][0.0]`) solves the same problem inside the MVT pipeline; use it only
  as a cross-check of the season total, and report any difference.
- **Stage 2:** `method="base"` in `optimization.ipynb`: λ = 0, the 15-man squad, XI and captain
  with the highest predicted xP within the budget and free transfers, transferring when that
  adds more than `xp_margin` predicted points. Its decisions use only the predictions and the
  constraints. The notebook still builds 𝕎_o and a P(beat) for it each week; that is a side
  computation and is **not** part of the baseline: don't report a P(beat) or Brier score for
  the baseline, and don't describe it as using the opponent model.
- The baseline does not depend on r, so one baseline season serves every r.

**Gameweeks without a target.** The source data has no xP in GW26 (2023-24) and in GW22, 32 and
34 (2024-25) (`config.MISSING_TARGET_GWS`): xP is 0 there, while points are real. Leave those
weeks out of every xP metric (mean xP, predicted-vs-actual, MAE) and say so; keep them in the
points and win metrics.

# Step 2: Run the 2024-25 r sweep
Do this step before any writing.

**What to run**
Run the MVT rolling season (`run_rolling_optimization(..., method="mvt")` in `optimization.ipynb`)
at **R_PCT ∈ {0.5, 0.6, 0.8, 0.9, 0.95, 0.99, 1.0}**, the same grid as 2023-24, plus one baseline
season (`method="base"`). Keep every other argument identical to the saved r = 0.5 run: seed,
N_SAMPLES_OPP, N_MC, λ grid, transfer_margin, xp_margin, start and end gameweek, and
OWN_TEAM_ID. The exception is mip_time_limit (see Running).

**Why r = 0.5 and the baseline are rerun.** The saved `rolling_optimization_results_{base,mvt}.joblib`
store λ* and the squad, but not what the chapter needs:
- the branch of eq. (3.17) taken (`sense_`, i.e. μ_{Y_w} ≥ 0 or < 0 for the current squad)
- the chosen XI, captain and vice: at λ > 0 the MIQP picks its own XI, so the team's variance
  cannot be rebuilt from the squad
- the chosen team's mean μ_w'v and variance v'Σ_δ v, and μ_{G^{(r')}} and σ_{G^{(r')}}
- |𝕎_o| (unique squads and accepted draws) and whether the week fell back to a single squad
- the frontier: for every λ in the grid, P(beat), mean and variance of w(λ), and whether SCIP
  stopped at its time limit (`getStatus()`)

The baseline is rerun once (about 25 minutes) only to record its chosen XI, captain, predicted
xP and the variance v'Σ_δ v of its team, so the r comparison can set MVT's variance against it.
Its decisions must match the saved `rolling_optimization_results_base.joblib` week for week
(λ = 0 is the convex branch and is solved to optimality, so the time limit doesn't affect it);
if they don't, stop and report. Record no P(beat) for it (see Baselines).

**The script.** `rolling_for_method` in the notebook hard-codes `R_PCT=0.5`. **Do not edit the
notebook.** Instead, write `thesis/run_r_sweep_2425.py`:
- it loads the notebook's code cells, leaving out the run and compare cells (21, 23)
- it records the fields above **without changing any decision**: wrap or replicate
  `run_optimization_pipeline` so the extra values are captured alongside the result, and
  check that the replica returns exactly the notebook function's result on a test week
- it calls `run_rolling_optimization` with each R_PCT and the baseline, and saves each run to
  `24-25/optimizer/notebooks/rolling_optimization_results_mvt_r{50,60,80,90,95,99,100}.joblib` and
  `rolling_optimization_results_base_rsweep.joblib`, with the extra fields in the same file
- **it never overwrites an existing results file**: the saved `rolling_optimization_results_{base,mvt}.joblib`
  stay untouched

**Checks before the full runs**
1. Reproducibility: rerun r = 0.5 for the first 2-3 gameweeks and confirm the decisions and
   P(beat) match the saved `rolling_optimization_results_mvt.joblib`. If they don't, stop and
   report what differs. A difference only in a week where SCIP stopped at its time limit is a
   solver-time effect, not a bug: report it, and say which weeks.
2. Check that R_PCT = 1.0 selects the maximum over 𝕎_o. The code takes index
   `ceil((1 − R_PCT)·O)` from the top, clamped to ≥ 1.
3. 𝕎_o is deduplicated with counts (`Wo_weights_`), and G^{(r')} is taken over the unique
   squads without weights. Keep this unchanged. In the chapter, define r for 2024-25 as a rank
   over the distinct squads the opponent could plausibly field, not a probability level. Note in
   one sentence that 2023-24 counts every simulated team, and that the two definitions coincide
   at r = 100%.

**Running**
- One r = 0.5 MVT season took about 2.5 hours with two workers sharing the machine (12 cores);
  the baseline about 25 minutes. Time one gameweek first, estimate the total, and tell me.
- **Run at most 3 seasons at a time.** `mip_time_limit` is wall-clock time: on the nonconvex
  branch (μ_{Y_w} < 0, the usual case at high r) SCIP stops at the limit with its best squad so
  far, so the result depends on machine load. An earlier sweep with six workers at the 30 s limit
  left 1-6 gameweeks per run with no feasible squad. Raise `mip_time_limit` to 60 s for **all**
  seven MVT runs (including r = 0.5), so they stay comparable, and state the value in the
  chapter. The baseline keeps the saved settings.
- Split the cores between workers with `threadpool_limits` as the notebook does, and write one
  log per run to `thesis/logs/`.
- A gameweek that fails, falls back or hits the time limit must be logged, not hidden. If any
  week of any run is unsolved, report it at checkpoint 2 before writing.

# Step 3: Compute (load saved results; do NOT rerun PyMC or the predictors)
**Both stages**
- Predictor MAE and R² per position.
- How μ_δ and Σ_δ were built.

**Stage 1**
- Selection-model fit, and the simulated vs. real field.
- MVT vs. baseline per gameweek and over the season: points, contest rank and percentile,
  prize, and the share of weeks finishing in the top (1 − r) of the real contest.
- Calibration of MVT's estimated P(Y_w > 0) (MVT only; the baseline has none).

**Stage 2**
- League β convergence (R̂, divergences) and the manager-fit results.
- Overhaul weeks excluded, and the sparse-vine diagnostics.
- One worked gameweek: the frontier over λ and λ*.
- MVT vs. baseline over the season: points, H2H win rate against the real opponent (ties count
  0.5), mean xP, transfers; and for MVT only, mean P(beat) and the Brier score. Show the real
  manager and real opponents as reference lines.
- λ* distribution and the branch of eq. (3.17) taken (μ_{Y_w} ≥ 0 or < 0).
- Paired tests over gameweeks (sign test or Wilcoxon, bootstrap CIs), with n stated.

**The r comparison (both stages)**
- For each r: mean λ*, how often λ* ≠ 0, how often the squad was held (λ* is `None` in a held
  week; count those separately, not as λ* = 0), the variance of the chosen team, points, and the
  outcome metric (rank and prize for FantasyGo, H2H wins for 2024-25), each against the
  baseline.
- One figure per stage, or a combined panel, contrasting winner-takes-all (r = 100%) with lower
  r.

Unsolved gameweeks, single-squad fallbacks, NaNs and identical MVT/baseline weeks must be
explained, not dropped silently.

# Step 4: Write (~4,800 words)
- **4.1 Experimental design**: data for both seasons, the information set (gameweek t uses data
  up to t−1), H2H as a double-up, the solution method of [1], and the non-strategic assumption.
- **4.2 Player performance predictions**: both seasons, briefly.
- **4.3 Stage 1: Replicating [1] on FantasyGo**: departures from [1], MVT vs. baseline, and the
  effect of r.
- **4.4 Stage 2: The H2H league**: the opponent model, the sparse vine and why, a worked
  gameweek, MVT vs. baseline over the season, and the effect of r.
- **4.5 Discussion**:
  - MVT vs. baseline across both seasons
  - winner-takes-all vs. lower r in a contest field vs. against a single opponent
  - the non-strategic assumption and what strategic opponents would change
  - the §1.3.2 objectives answered
  - limitations (single seasons, small n, noisy Σ_δ, biased μ_δ for top players, chips not
    modelled)

# Writing rules
- Formal academic register with the authorial "we". No hype. Use "significant" only with a test
  that supports it.
- Reuse the Chapter 3 notation and cite equations, theorems and the algorithm by their actual
  numbers. Where Stage 1's 11-player setting differs from §3.1.2's 15-player constraints, say
  so.
- Report null and negative results plainly. If MVT ≈ baseline at some r, that is the finding,
  so explain why.
- Use LaTeX for a `report`-class chapter: `\chapter{Simulation and Results}`, `booktabs`, and
  `\label`/`\ref` with `s1-`/`s2-` prefixes. Cite as `\cite{haugh2021}` (and
  `\cite{morton2003}` only where needed).

# Deliverables
- **Only new files.** Never modify existing code, notebooks or results.
  - `thesis/run_r_sweep_2425.py`, `thesis/vine_diagnostics_2425.py` and their logs in
    `thesis/logs/`.
  - New results files `rolling_optimization_results_mvt_r{50,60,80,90,95,99,100}.joblib`,
    `rolling_optimization_results_base_rsweep.joblib` and `thesis/results/*.csv`.
  - In the chapter, 2024-25 numbers come from these new runs, not from the saved
    `rolling_optimization_results_{base,mvt}.joblib` (those serve only the reproducibility check).
- `thesis/chapter4_simulation_results.tex`
- `thesis/figures/*.pdf`: consistent styling that also reads in greyscale.
- `thesis/make_results.py`: regenerates every table and figure from the saved results.
- `thesis/numbers.md`: every number quoted, with its source.

# Checkpoints: report back briefly at each
1. After the reproducibility check and the runtime estimate, before the full sweep.
2. After the sweep, with:
   (a) the key numbers for each stage, including MVT vs. baseline at every r,
   (b) the departures from [1],
   (c) any data or run issues.
   Then write the chapter.
