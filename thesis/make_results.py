"""
Regenerates every table, figure and quoted number of Chapter 4 from the saved results.

Nothing here refits a model: the selection models, predictors and optimizer runs are loaded from
their saved files. The notebooks' own functions are reused by executing their code cells (the
definitions and the cached loads), so every metric is computed as in the notebooks.

    python make_results.py            # everything
    python make_results.py s1         # only one part: pred, s1, s2fits, s2

Writes thesis/outputs/results/*.csv, thesis/outputs/figures/*.pdf and thesis/numbers.md (every quoted number
with its source).
"""
import importlib
import json
import os
import sys
import warnings
from contextlib import contextmanager
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon

warnings.filterwarnings("ignore")
THESIS = Path(__file__).resolve().parent
ROOT = THESIS.parent
RESULTS, FIGURES = THESIS / "outputs/results", THESIS / "outputs/figures"
S1_DIR = ROOT / "23-24/optimizer/MVT"
S2_NB_DIR = ROOT / "24-25/optimizer/notebooks"
R_GRID = [50, 60, 80, 90, 95, 99, 100]
MISSING_TARGET = {"23-24": [26], "24-25": [22, 32, 34]}  # config.MISSING_TARGET_GWS
BOOT, SEED = 10_000, 0
NUMBERS = []  # (key, value, source) for numbers.md

# greyscale-safe styling: identity by line style and marker as well as tone
plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8, "lines.linewidth": 1.4,
    "lines.markersize": 5, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "0.88", "grid.linewidth": 0.6, "figure.dpi": 150,
})
STYLE = {
    "MVT": dict(color="black", linestyle="-", marker="o"),
    "Baseline": dict(color="0.45", linestyle="--", marker="s"),
    "Real manager": dict(color="0.25", linestyle=":", marker=None),
    "Real opponents": dict(color="0.6", linestyle="-.", marker=None),
}


def note(key, value, source):
    NUMBERS.append((key, value, source))
    return value


@contextmanager
def cwd(path):
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def notebook_ns(path, cells, split=None):
    """Execute the given code cells of a notebook (run from its folder) into one namespace.
    `split` = {cell: marker} keeps only the part of that cell before the marker."""
    nb = json.load(open(path))
    g = {"__name__": "__main__"}
    with cwd(Path(path).parent):
        for i in cells:
            src = "".join(nb["cells"][i]["source"])
            if split and i in split:
                src = src.split(split[i])[0]
            exec(src, g)
    return g


def boot_ci(x, rng, stat=np.mean):
    x = np.asarray(x, dtype=float)
    idx = rng.integers(len(x), size=(BOOT, len(x)))
    s = stat(x[idx], axis=1)
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def paired(a, b, rng):
    """Paired comparison over gameweeks of a against b: mean difference with a bootstrap CI and
    a Wilcoxon signed-rank test (zero differences dropped, as scipy's default)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    nz = d[d != 0]
    p = float(wilcoxon(nz).pvalue) if len(nz) >= 1 else np.nan
    lo, hi = boot_ci(d, rng)
    return {"n": len(d), "n_nonzero": len(nz), "mean_diff": d.mean(), "ci_lo": lo, "ci_hi": hi,
            "wilcoxon_p": p}


def sign_test(wins_a, wins_b):
    """Sign test on the weeks where the two outcomes differ (a better vs b better)."""
    a_better, b_better = int((wins_a > wins_b).sum()), int((wins_a < wins_b).sum())
    n = a_better + b_better
    p = float(binomtest(a_better, n).pvalue) if n else np.nan
    return {"a_better": a_better, "b_better": b_better, "sign_p": p}


def save(df, name, index=False):
    RESULTS.mkdir(exist_ok=True)
    df.to_csv(RESULTS / name, index=index)
    return df


# ---------------------------------------------------------------------------------------------
# Player performance predictions (both seasons)
# ---------------------------------------------------------------------------------------------
def predictors():
    rows, bias_rows, cand_rows, pop_rows, trend_rows, deployed = [], [], [], [], [], {}
    for season, src in [("23-24", ROOT / "23-24/src"), ("24-25", ROOT / "24-25/predictor/src")]:
        for m in ["config", "compare"]:
            sys.modules.pop(m, None)
        sys.path.insert(0, str(src))
        with cwd(src):
            config = importlib.import_module("config")
            compare = importlib.import_module("compare")
            res, dep = compare.load_results(), compare.load_deploy()
            for pos in config.POSITIONS:
                table, picks = compare.summarize(res, dep, pos)
                # MAE by gameweek: validation MAE of the model picked at each selection point, then
                # that model's test MAE on the gameweeks it is used for, and the dummy's for reference
                for s_, m in picks.items():
                    v = res[(res.pos == pos) & (res.label == m) & (res.fold == s_)]
                    trend_rows.append({"season": season, "position": pos, "set": "validation",
                                       "selected_at": s_, "model": m, "gw": s_,
                                       "mae": v["mae"].iloc[0]})
                    for lab, kind in [(m, "test"), ("dummy", "test dummy")]:
                        for _, x in dep[(dep.pos == pos) & (dep.selected_at == s_)
                                        & (dep.label == lab)].iterrows():
                            trend_rows.append({"season": season, "position": pos, "set": kind,
                                               "selected_at": s_, "model": lab, "gw": x.gw,
                                               "mae": x.mae})
                for label, x in table.iterrows():
                    cand_rows.append({"season": season, "position": pos, "candidate": label,
                                      "use_mae": x["use_mae"], "use_r2": x["use_r2"]})
                fixed = table.drop(index=[compare.SELECTED, "dummy"], errors="ignore")
                best = fixed["use_mae"].idxmin()
                gws = sorted(dep[dep.pos == pos].gw.unique())
                rows.append({
                    "season": season, "position": pos,
                    "procedure_mae": table.loc[compare.SELECTED, "use_mae"],
                    "procedure_r2": table.loc[compare.SELECTED, "use_r2"],
                    "best_fixed": best, "best_fixed_mae": fixed.loc[best, "use_mae"],
                    "best_fixed_r2": fixed.loc[best, "use_r2"],
                    "dummy_mae": table.loc["dummy", "use_mae"] if "dummy" in table.index else np.nan,
                    "gameweeks": len(gws), "first_gw": gws[0], "last_gw": gws[-1],
                    "selection_points": " ".join(map(str, config.VALID_FOLDS)),
                    "picks": " ".join(f"{s}:{m}" for s, m in picks.items()),
                })
            preds = pd.read_csv(config.MODEL_OUTPUT + "predictions.csv")
            if "played" not in preds:  # 2024-25: whether the player had minutes that gameweek
                mins = pd.read_csv(config.FULL_DATA, usecols=["element", "round", "minutes"],
                                   low_memory=False).rename(columns={"element": "fpl_id"})
                preds = preds.merge(mins, on=["fpl_id", "round"], how="left", validate="m:1")
                preds["played"] = preds["minutes"] > 0
        sys.path.remove(str(src))
        # accuracy of the deployed predictions on two populations: every player in the pool, and
        # only those who played (the 2023-24 models were fitted and selected on the latter)
        q = preds[(preds["round"] >= 6) & preds["xP"].notna() & ~preds["in_selection"]
                  & ~preds["round"].isin(MISSING_TARGET[season])]
        deployed[season] = q
        for pos in config.POSITIONS:
            for label, x in [("all", q[q.pos == pos]), ("played", q[(q.pos == pos) & q["played"]])]:
                err = x["pred"] - x["xP"]
                pop_rows.append({
                    "season": season, "position": pos, "population": label, "n": len(x),
                    "per_gw": len(x) / x["round"].nunique(), "mae": err.abs().mean(),
                    "r2": 1 - (err ** 2).sum() / ((x["xP"] - x["xP"].mean()) ** 2).sum(),
                    "zero_xP_share": (x["xP"] == 0).mean(),
                })
        # bias of the predictions for the players predicted highest (top decile per gameweek),
        # on the gameweeks used for decisions (not in_selection, target available)
        p = preds[~preds["in_selection"] & ~preds["round"].isin(MISSING_TARGET[season])].copy()
        p["top"] = p.groupby("round")["pred"].rank(pct=True) > 0.9
        for label, x in [("top decile", p[p["top"]]), ("rest", p[~p["top"]])]:
            bias_rows.append({
                "season": season, "group": label, "n": len(x),
                "bias_vs_xP": (x["pred"] - x["xP"]).mean(),
                "bias_vs_points": (x["pred"] - x["total_points"]).mean(),
            })
    t = save(pd.DataFrame(rows), "predictor_mae.csv")
    b = save(pd.DataFrame(bias_rows), "predictor_bias.csv")
    c = save(pd.DataFrame(cand_rows), "predictor_candidates.csv")
    pop = save(pd.DataFrame(pop_rows), "predictor_population.csv")
    fig_pred_mae_trend(save(pd.DataFrame(trend_rows), "predictor_mae_trend.csv"))
    for _, r in c.iterrows():
        note(f"pred_{r.season}_{r.position}_cand_{r.candidate}_mae", round(r.use_mae, 3),
             "outputs/results/predictor_candidates.csv")
    for _, r in pop.iterrows():
        for k in ["n", "per_gw", "mae", "r2", "zero_xP_share"]:
            v = r[k]
            note(f"pred_pop_{r.season}_{r.position}_{r.population}_{k}",
                 round(float(v), 3) if k != "n" else int(v), "outputs/results/predictor_population.csv")
    picks = pd.DataFrame([{"season": r.season, "position": r.position, "point": int(s_), "model": m}
                          for _, r in t.iterrows() for s_, m in (x.split(":") for x in r.picks.split())])
    for season, x in picks.groupby("season"):
        for m, n in x["model"].value_counts().items():
            note(f"pred_{season}_picks_{m}", int(n), "outputs/results/predictor_mae.csv (picks)")
    for _, r in t.iterrows():
        src = f"outputs/results/predictor_mae.csv ({r.season} compare.summarize)"
        note(f"pred_{r.season}_{r.position}_mae", round(r.procedure_mae, 3), src)
        note(f"pred_{r.season}_{r.position}_r2", round(r.procedure_r2, 3), src)
        note(f"pred_{r.season}_{r.position}_best", f"{r.best_fixed} {r.best_fixed_mae:.3f}", src)
    for _, r in b.iterrows():
        note(f"pred_bias_{r.season}_{r.group.replace(' ', '_')}_xP", round(r.bias_vs_xP, 3),
             "outputs/results/predictor_bias.csv")
        note(f"pred_bias_{r.season}_{r.group.replace(' ', '_')}_points", round(r.bias_vs_points, 3),
             "outputs/results/predictor_bias.csv")
    fig_pred_players(deployed)
    fig_pred_scatter(deployed)
    print(t.round(3).to_string(), "\n", b.round(3).to_string())


POSITION_NAMES = {"GK": "Goalkeeper", "DEF": "Defender", "MID": "Midfielder", "FWD": "Forward"}


def fig_pred_mae_trend(tr):
    """MAE by gameweek, by season and position: validation MAE at each selection point (the score
    the pick is made on) and the picked model's test MAE on the gameweeks that follow."""
    FIGURES.mkdir(exist_ok=True)
    src = "outputs/results/predictor_mae_trend.csv"
    for (season, pos), x in tr.groupby(["season", "position"]):
        v, t = x[x.set == "validation"], x[x.set == "test"]
        note(f"pred_trend_{season}_{pos}_valid_mean", round(v.mae.mean(), 3), src)
        note(f"pred_trend_{season}_{pos}_test_mean", round(t.mae.mean(), 3), src)
        note(f"pred_trend_{season}_{pos}_test_range", f"{t.mae.min():.2f}-{t.mae.max():.2f}", src)
        tm, dm = t.set_index("gw").mae, x[x.set == "test dummy"].set_index("gw").mae
        note(f"pred_trend_{season}_{pos}_test_beats_dummy_weeks", f"{int((tm < dm).sum())}/{len(t)}",
             src)
        note(f"pred_trend_{season}_{pos}_test_gw6-10", round(tm[tm.index <= 10].mean(), 2), src)
        note(f"pred_trend_{season}_{pos}_test_gw11+", round(tm[tm.index > 10].mean(), 2), src)
        note(f"pred_trend_{season}_{pos}_test_corr_dummy", round(tm.corr(dm), 2), src)
    seasons = sorted(tr.season.unique())
    fig, ax = plt.subplots(len(POSITION_NAMES), len(seasons), figsize=(6.3, 7.0), sharex=True)
    for j, season in enumerate(seasons):
        for i, pos in enumerate(POSITION_NAMES):
            x = tr[(tr.season == season) & (tr.position == pos)]
            a = ax[i, j]
            for s_ in sorted(x.selected_at.unique()):
                a.axvline(s_, color="0.8", linewidth=0.6, linestyle=":", zorder=0)
            t = x[x.set == "test"].sort_values("gw")
            d = x[x.set == "test dummy"].sort_values("gw")
            v = x[x.set == "validation"].sort_values("gw")
            # break the lines at gameweeks with no target, rather than joining across them
            full = pd.RangeIndex(t.gw.min(), t.gw.max() + 1)
            a.plot(full, d.set_index("gw").mae.reindex(full), label="test, dummy", color="0.6",
                   linestyle="--", linewidth=1.0)
            a.plot(full, t.set_index("gw").mae.reindex(full), label="test, selected model",
                   **STYLE["MVT"] | {"marker": None})
            a.plot(v.gw, v.mae, label="validation, selected model", color="black",
                   linestyle="none", marker="D", markersize=3.5, markerfacecolor="white")
            a.set_title(f"{season} {POSITION_NAMES[pos]}")
            a.set_ylim(bottom=0)
    for a in ax[-1]:
        a.set_xlabel("gameweek")
    for a in ax[:, 0]:
        a.set_ylabel("MAE")
    h, l = ax[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIGURES / "pred_mae_trend.pdf")
    plt.close(fig)


def fig_pred_players(deployed, per_position=2, min_played=25):
    """Per season, `per_position` players per position drawn at random (seed SEED) from those who
    played in at least `min_played` of the deployed gameweeks: prediction, xP and points by week."""
    FIGURES.mkdir(exist_ok=True)
    rows = []
    for season, q in deployed.items():
        rng = np.random.default_rng(SEED)
        fig, ax = plt.subplots(4, per_position, figsize=(6.3, 7.6), sharex=True)
        for i, pos in enumerate(POSITION_NAMES):
            x = q[q.pos == pos]
            regular = x.groupby("fpl_id")["played"].sum()
            pool = np.sort(regular[regular >= min_played].index.to_numpy())
            for j, pid in enumerate(rng.choice(pool, size=per_position, replace=False)):
                p = x[x.fpl_id == pid].sort_values("round")
                a = ax[i, j]
                a.plot(p["round"], p["pred"], label="prediction", **STYLE["MVT"] | {"marker": None})
                a.plot(p["round"], p["xP"], label="xP", color="0.45", linestyle="--", marker="s",
                       markersize=2.5)
                a.plot(p["round"], p["total_points"], label="points", color="0.6", linestyle="none",
                       marker="o", markerfacecolor="none", markersize=3)
                name = p["name"].iloc[0]
                a.set_title(f"{POSITION_NAMES[pos]}: {name}")
                mae = (p["pred"] - p["xP"]).abs().mean()
                a.text(0.02, 0.95, f"MAE {mae:.2f}", transform=a.transAxes, va="top", fontsize=7)
                rows.append({"season": season, "position": pos, "fpl_id": int(pid), "name": name,
                             "weeks": len(p), "played": int(p["played"].sum()), "mae": mae})
        for a in ax[-1]:
            a.set_xlabel("gameweek")
        for a in ax[:, 0]:
            a.set_ylabel("points")
        ax[0, 0].legend(frameon=False, fontsize=7, loc="upper right")
        fig.tight_layout()
        fig.savefig(FIGURES / f"pred_players_{season.replace('-', '')}.pdf")
        plt.close(fig)
    sp = save(pd.DataFrame(rows), "pred_players_sampled.csv")
    for _, r in sp.iterrows():
        note(f"pred_{r.season}_sampled_{r.position}_{r.fpl_id}", f"{r['name']} MAE {r.mae:.2f}",
             "outputs/results/pred_players_sampled.csv")


def fig_pred_scatter(deployed):
    """Prediction against xP for every deployed player-gameweek, by season and position."""
    FIGURES.mkdir(exist_ok=True)
    for season, q in deployed.items():
        hi, zero = q[q["xP"] >= 8], q[q["xP"] == 0]
        src = "deployed predictions (predictor predictions.csv), as predictor_population.csv"
        note(f"pred_{season}_xP8_n", len(hi), src)
        note(f"pred_{season}_xP8_gap", round((hi["pred"] - hi["xP"]).mean(), 2), src)
        note(f"pred_{season}_xP0_mean_pred", round(zero["pred"].mean(), 2), src)
        note(f"pred_{season}_xP0_share_above1", round((zero["pred"] > 1).mean(), 2), src)
    fig, ax = plt.subplots(2, 4, figsize=(6.3, 3.6), sharex=True, sharey=True)
    for i, (season, q) in enumerate(deployed.items()):
        for j, pos in enumerate(POSITION_NAMES):
            x = q[q.pos == pos]
            a = ax[i, j]
            a.hexbin(x["xP"], x["pred"], gridsize=30, bins="log", cmap="Greys", mincnt=1,
                     extent=(-1, 16, -1, 16), linewidths=0)
            a.plot([-1, 16], [-1, 16], color="black", linewidth=0.7, linestyle=":")
            a.set(xlim=(-1, 16), ylim=(-1, 16))
            a.grid(False)
            if i == 0:
                a.set_title(POSITION_NAMES[pos])
            if i == 1:
                a.set_xlabel("xP")
            if j == 0:
                a.set_ylabel(f"{season}\nprediction")
    fig.tight_layout()
    fig.savefig(FIGURES / "pred_scatter.pdf")
    plt.close(fig)


def coef_frame(fits, names):
    """Posterior mean and 94% HDI of every coefficient of every per-gameweek fit.
    `fits` = {position: {round: fit}}; `names(fit)` maps the summary rows to feature names."""
    rows = []
    for pos, fr in fits.items():
        for rnd, f in fr.items():
            sm = f["summary_final"]
            for idx, feat in names(f).items():
                rows.append({"position": pos, "round": rnd, "feature": feat, "mean": sm.loc[idx, "mean"],
                             "lo": sm.loc[idx, "hdi_3%"], "hi": sm.loc[idx, "hdi_97%"]})
    return pd.DataFrame(rows)


COEF_STYLES = [dict(color="black", marker="o"), dict(color="0.45", marker="s"),
               dict(color="0.25", marker="^"), dict(color="0.6", marker="D"), dict(color="0.35", marker="v")]


def note_coefs(cf, prefix, src, min_share=0.3):
    """Per position, the features chosen in at least `min_share` of the gameweeks: how often, the
    mean, minimum and maximum posterior mean, and the share of fits whose 94% HDI excludes 0."""
    for (pos, feat), x in cf.groupby(["position", "feature"]):
        if len(x) < min_share * cf.loc[cf.position == pos, "round"].nunique():
            continue
        excl = ((x["lo"] > 0) | (x["hi"] < 0)).mean()
        note(f"{prefix}_{pos}_{feat}", f"{len(x)} fits, mean {x['mean'].mean():.2f} "
             f"[{x['mean'].min():.2f}, {x['mean'].max():.2f}], HDI excludes 0 in {excl:.2f}", src)


def fig_coefs(cf, name, min_share=0.3, max_features=4):
    """Coefficient paths over the season, one panel per position: the features chosen in at least
    `min_share` of the gameweeks (at most `max_features`, the most frequent first)."""
    FIGURES.mkdir(exist_ok=True)
    fig, ax = plt.subplots(2, 2, figsize=(6.3, 5.6))
    for a, (pos, x) in zip(ax.flat, cf.groupby("position", sort=False)):
        n = x["round"].nunique()
        freq = x["feature"].value_counts()
        keep = freq[freq >= min_share * n].index[:max_features]
        for k, feat in enumerate(keep):
            y = x[x.feature == feat].sort_values("round")
            off = (k - (len(keep) - 1) / 2) * 0.15
            a.errorbar(y["round"] + off, y["mean"], yerr=[y["mean"] - y["lo"], y["hi"] - y["mean"]],
                       linestyle="none", markersize=3, elinewidth=0.6, capsize=0, label=feat,
                       **COEF_STYLES[k])
        a.axhline(0, color="0.3", linewidth=0.6)
        a.set_title(POSITION_NAMES.get(pos, pos))
        a.set_xlabel("target gameweek")
        a.legend(frameon=False, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    for a in ax[:, 0]:
        a.set_ylabel("coefficient (standardised)")
    fig.tight_layout()
    fig.savefig(FIGURES / name)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------
# Stage 1: FantasyGo 2023-24
# ---------------------------------------------------------------------------------------------
def s1_selection_model():
    g = notebook_ns(S1_DIR / "beta estimators.ipynb", [1, 3, 5, 7, 11, 14])
    fits = g["league_models"]
    conv = pd.DataFrame([
        {"position": pos, "round": r, "features": len(f["feats"]),
         "max_r_hat": f["summary_final"]["r_hat"].max(), "min_ess_bulk": f["summary_final"]["ess_bulk"].min()}
        for pos, fr in fits.items() for r, f in fr.items()
    ])
    save(conv, "s1_selection_convergence.csv")
    cf = coef_frame(fits, lambda f: {feat: feat for feat in f["feats"]})
    save(cf, "s1_selection_coefs.csv")
    note_coefs(cf, "s1_coef", "outputs/results/s1_selection_coefs.csv")
    fig_coefs(cf, "s1_selection_coefs.pdf")
    # predicted against real field shares of every player in every round (the notebook's check loop)
    shares = []
    data = g["data"]
    for (pos, rnd), x in data[data["round"] >= min(g["FIT_ROUNDS"])].groupby(["position", "round"]):
        fit = g["league_fit_for_round"](fits[pos], rnd)
        alpha = g["alpha_league"](fit, x[fit["feats"]].to_numpy(dtype=float))
        shares.append(pd.DataFrame({"position": pos, "round": rnd, "pred": alpha / alpha.sum(),
                                    "real": x["fg_share"].to_numpy(dtype=float)}))
    shares = save(pd.concat(shares, ignore_index=True), "s1_selection_shares.csv")
    fig_s1_shares(shares)
    check = g["check"].groupby(["position", "predictor"])[["log_loss", "spearman", "top10"]].mean()
    save(check.reset_index(), "s1_selection_oos.csv")
    note("s1_sel_fits", len(conv), "outputs/results/s1_selection_convergence.csv")
    note("s1_sel_max_rhat", round(conv["max_r_hat"].max(), 3), "outputs/results/s1_selection_convergence.csv")
    note("s1_sel_min_ess", int(conv["min_ess_bulk"].min()), "outputs/results/s1_selection_convergence.csv")
    for (pos, pred), r in check.iterrows():
        note(f"s1_sel_logloss_{pos}_{pred}", round(r.log_loss, 3), "outputs/results/s1_selection_oos.csv")
    print(check.round(3).unstack("position").to_string())


def fig_s1_shares(d):
    """Stage 1: predicted against real share of the FantasyGo field for every player and round,
    pooled over gameweeks 6-38, one panel per position (log scales; players never picked at the
    bottom)."""
    FIGURES.mkdir(exist_ok=True)
    fig, ax = plt.subplots(1, 4, figsize=(6.3, 1.95), sharex=True, sharey=True)
    floor = 1e-4
    for a, pos in zip(ax, POSITION_NAMES):
        x = d[d.position == pos]
        a.hexbin(np.log10(x["pred"].clip(lower=floor)), np.log10(x["real"].clip(lower=floor)),
                 gridsize=25, bins="log", cmap="Greys", mincnt=1, extent=(-4, 0, -4, 0), linewidths=0)
        a.plot([-4, 0], [-4, 0], color="black", linewidth=0.7, linestyle=":")
        a.set(title=POSITION_NAMES[pos], xlabel="predicted share")
        a.set_xticks([-4, -2, 0], ["$10^{-4}$", "$10^{-2}$", "1"])
        a.set_yticks([-4, -2, 0], ["$\\leq10^{-4}$", "$10^{-2}$", "1"])
        a.grid(False)
    ax[0].set_ylabel("real share")
    fig.tight_layout()
    fig.savefig(FIGURES / "s1_selection_shares.pdf")
    plt.close(fig)
    for pos, x in d.groupby("position"):
        top = x[x["pred"] >= 0.05]
        note(f"s1_shares_{pos}_top_pred_mean", round(top["pred"].mean(), 3), "outputs/results/s1_selection_shares.csv")
        note(f"s1_shares_{pos}_top_real_mean", round(top["real"].mean(), 3), "outputs/results/s1_selection_shares.csv")
        note(f"s1_shares_{pos}_top_n", len(top), "outputs/results/s1_selection_shares.csv")
    note("s1_shares_rows", len(d), "outputs/results/s1_selection_shares.csv")
    note("s1_shares_unpicked", round(float((d["real"] == 0).mean()), 3), "outputs/results/s1_selection_shares.csv")
    note("s1_shares_unpicked_pred_mean", round(float(d.loc[d["real"] == 0, "pred"].mean()), 4),
         "outputs/results/s1_selection_shares.csv")
    note("s1_shares_unpicked_pred_above_1pct", round(float((d.loc[d["real"] == 0, "pred"] >= 0.01).mean()), 3),
         "outputs/results/s1_selection_shares.csv")


def s1_field():
    g = notebook_ns(S1_DIR / "mvt.ipynb", [1, 3, 5, 7, 9])
    fields = g["fields"]
    save(fields, "s1_field_by_gw.csv")
    summ = fields.groupby("field")[["cost", "cut_points", "mean_points", "share_corr", "accept_rate"]].mean()
    save(summ.reset_index(), "s1_field_summary.csv")
    for f, r in summ.iterrows():
        for c in ["cost", "cut_points", "mean_points", "share_corr", "accept_rate"]:
            if pd.notna(r[c]):
                note(f"s1_field_{f}_{c}", round(r[c], 2), "outputs/results/s1_field_summary.csv (mvt.ipynb cell 9, R_PCT 0.8)")
    print(summ.round(2).to_string())


def s1_runs():
    g = notebook_ns(S1_DIR / "mvt.ipynb", [1, 3], )
    nb = json.load(open(S1_DIR / "mvt.ipynb"))
    with cwd(S1_DIR):
        exec("".join(nb["cells"][23]["source"]).split("rows = []")[0], g)
        if "fg_managers" not in g:
            g["fg_managers"] = pd.read_csv("../data/fg_managers.csv")
    team_points, contest_rank = g["team_points"], g["contest_rank"]
    # the baseline: optimizer.ipynb (best_xi on predicted xP), scored on actual points
    base = notebook_ns(ROOT / "23-24/optimizer/optimizer.ipynb", [0, 3, 4])["summary"]
    rows = []
    for pct in R_GRID:
        res = pickle_load(S1_DIR / f"mvt_results_simulated_r{pct}.pkl")
        for gw, r in res.items():
            fr = r["frontier"].set_index("lambda")
            lam = r["lambda_star"]
            pts = team_points(r["teams"][lam], gw)
            rank, n, prize = contest_rank(pts, gw)
            target = max(int(np.ceil((1 - pct / 100) * n)), 1)
            b_pts = float(base.at[gw, "points"])
            b_rank, _, b_prize = contest_rank(b_pts, gw)
            rows.append({
                "r": pct, "gw": gw, "entries": n, "target_rank": target, "G_mean": r["G_mean"],
                "G_sd": r["G_sd"], "lambda_star": lam, "prob_beat": fr.loc[lam, "prob_beat"],
                "team_mean": fr.loc[lam, "mean"], "team_sd": fr.loc[lam, "sd"],
                "points": pts, "rank": rank, "pct_rank": rank / n, "prize": prize,
                "hit": rank <= target,
                "lam0_points": team_points(r["teams"][0.0], gw), "lam0_sd": fr.loc[0.0, "sd"],
                "base_points": b_pts, "base_rank": b_rank, "base_pct_rank": b_rank / n,
                "base_prize": b_prize, "base_hit": b_rank <= target,
                "base_pred": float(base.at[gw, "pred"]),
            })
    d = save(pd.DataFrame(rows), "s1_runs_by_gw.csv")
    rng = np.random.default_rng(SEED)
    summ = []
    for pct, x in d.groupby("r"):
        win = x["hit"].astype(float)
        pp = paired(x["points"], x["base_points"], rng)
        st = sign_test(x["hit"].astype(int), x["base_hit"].astype(int))
        summ.append({
            "r": pct, "weeks": len(x), "mean_lambda": x["lambda_star"].mean(),
            "share_lambda_nonzero": (x["lambda_star"] > 0).mean(),
            "mean_team_sd": x["team_sd"].mean(), "mean_lam0_sd": x["lam0_sd"].mean(),
            "points": x["points"].sum(), "base_points": x["base_points"].sum(),
            "median_pct_rank": x["pct_rank"].median(), "base_median_pct_rank": x["base_pct_rank"].median(),
            "prize": x["prize"].sum(), "base_prize": x["base_prize"].sum(),
            "paid_weeks": (x["prize"] > 0).sum(), "base_paid_weeks": (x["base_prize"] > 0).sum(),
            "top3": (x["rank"] <= 3).sum(), "base_top3": (x["base_rank"] <= 3).sum(),
            "hits": int(x["hit"].sum()), "base_hits": int(x["base_hit"].sum()),
            "mean_prob": x["prob_beat"].mean(), "brier": ((x["prob_beat"] - win) ** 2).mean(),
            "points_diff_mean": pp["mean_diff"], "points_diff_lo": pp["ci_lo"], "points_diff_hi": pp["ci_hi"],
            "points_wilcoxon_p": pp["wilcoxon_p"], "hit_sign_p": st["sign_p"],
            "hit_mvt_only": st["a_better"], "hit_base_only": st["b_better"],
            "lam0_points": x["lam0_points"].sum(),
        })
    s = save(pd.DataFrame(summ), "s1_r_summary.csv")
    src = "outputs/results/s1_r_summary.csv"
    for _, r in s.iterrows():
        for c in s.columns:
            if c != "r":
                v = r[c]
                note(f"s1_r{int(r.r)}_{c}", round(v, 4) if isinstance(v, float) else v, src)
    note("s1_base_points", s["base_points"].iloc[0], src + " (optimizer.ipynb)")
    note("s1_base_pred", round(d[d.r == 80]["base_pred"].sum(), 1), "outputs/results/s1_runs_by_gw.csv (optimizer.ipynb pred)")
    # calibration of MVT's P(Y_w > 0), pooled over r and gameweeks
    bins = [0, 0.2, 0.4, 0.6, 0.8, 0.95, 1.0001]
    d["bin"] = pd.cut(d["prob_beat"], bins, right=False)
    cal = d.groupby("bin", observed=True).agg(n=("hit", "size"), predicted=("prob_beat", "mean"),
                                              observed=("hit", "mean")).reset_index()
    cal["bin"] = cal["bin"].astype(str)
    save(cal, "s1_calibration.csv")
    print(s.round(3).T.to_string(), "\n", cal.round(3).to_string())
    return d, s


def pickle_load(path):
    import pickle
    return pickle.loads(Path(path).read_bytes())


# ---------------------------------------------------------------------------------------------
# Stage 2: selection-model fits and overhauls (2024-25)
# ---------------------------------------------------------------------------------------------
def s2_fits():
    lb = ROOT / "24-25/optimizer/beta estimators/league betas"
    rows, league_fits = [], {}
    for short, pos in [("gk", "Goalkeeper"), ("def", "Defender"), ("mid", "Midfielder"), ("fwd", "Forward")]:
        fits = league_fits[pos] = joblib.load(lb / f"league_models_{short}")
        for rnd, f in fits.items():
            t = f["trace_final"]
            rows.append({"position": pos, "round": rnd, "features": len(f["feats"]),
                         "max_r_hat": f["summary_final"]["r_hat"].max(),
                         "min_ess_bulk": f["summary_final"]["ess_bulk"].min(),
                         "divergences": int(t.sample_stats["diverging"].sum()),
                         "draws": int(t.posterior.sizes["chain"] * t.posterior.sizes["draw"])})
    league = save(pd.DataFrame(rows), "s2_league_convergence.csv")
    est = ROOT / "24-25/optimizer/beta estimators/opponent betas/estimates"
    mrows = []
    for gw in range(6, 39):
        p = est / f"manager_fits_{gw}.joblib"
        if not p.exists():
            continue
        for pos, f in joblib.load(p)[gw].items():
            t = f["trace"]
            ss = getattr(t, "sample_stats", None)
            mrows.append({"gw": gw, "position": pos, "team_id": f["team_id"], "n_weeks": f["n_weeks"],
                          "gamma_mean": f["gamma_mean"], "gamma_lo": f["summary"].loc["gamma_m", "hdi_3%"],
                          "gamma_hi": f["summary"].loc["gamma_m", "hdi_97%"],
                          "max_r_hat": f["summary"]["r_hat"].max(),
                          "min_ess_bulk": f["summary"]["ess_bulk"].min(),
                          "divergences": int(ss["diverging"].sum()) if ss is not None else np.nan})
    man = save(pd.DataFrame(mrows), "s2_manager_fits.csv")
    cf = coef_frame(league_fits, lambda f: {f"beta[{i}]": feat for i, feat in enumerate(f["feats"])})
    save(cf, "s2_league_coefs.csv")
    note_coefs(cf, "s2_coef", "outputs/results/s2_league_coefs.csv")
    fig_coefs(cf, "s2_league_coefs.pdf")
    del league_fits
    fig_s2_gamma(man)
    for k, v, s in [
        ("s2_league_fits", len(league), "s2_league_convergence.csv"),
        ("s2_league_max_rhat", round(league["max_r_hat"].max(), 3), "s2_league_convergence.csv"),
        ("s2_league_min_ess", int(league["min_ess_bulk"].min()), "s2_league_convergence.csv"),
        ("s2_league_divergences", int(league["divergences"].sum()), "s2_league_convergence.csv"),
        ("s2_manager_fits", len(man), "s2_manager_fits.csv"),
        ("s2_manager_max_rhat", round(man["max_r_hat"].max(), 3), "s2_manager_fits.csv"),
        ("s2_manager_min_ess", int(man["min_ess_bulk"].min()), "s2_manager_fits.csv"),
        ("s2_manager_divergences", man["divergences"].sum(), "s2_manager_fits.csv"),
        ("s2_gamma_median", round(man["gamma_mean"].median(), 2), "s2_manager_fits.csv"),
        ("s2_gamma_min", round(man["gamma_mean"].min(), 2), "s2_manager_fits.csv"),
        ("s2_gamma_max", round(man["gamma_mean"].max(), 2), "s2_manager_fits.csv"),
    ]:
        note(k, v, "outputs/results/" + s)
    gam = man.groupby("position")["gamma_mean"].describe()[["mean", "50%", "min", "max"]]
    save(gam.reset_index(), "s2_gamma_by_position.csv")
    for pos, r in gam.iterrows():
        note(f"s2_gamma_median_{pos}", round(r["50%"], 2), "outputs/results/s2_gamma_by_position.csv")
    note("s2_gamma_weeks_spearman", round(man[["gamma_mean", "n_weeks"]].corr(method="spearman").iloc[0, 1], 2),
         "outputs/results/s2_manager_fits.csv")
    # overhaul (Wildcard / Free Hit) weeks detected league-wide, as the notebook's detect_squad_overhaul
    g = notebook_ns(S2_NB_DIR / "optimization.ipynb", [2, 3, 5, 7])
    ls = g["league_selections"]
    ov = []
    for team in ls["team_id"].unique():
        for rnd in sorted(ls.loc[ls["team_id"] == team, "round"].unique()):
            out = g["detect_squad_overhaul"](team, rnd, ls)
            if out[1]:
                ov.append({"team_id": team, "round": rnd, "detail": out[0]})
    ov = save(pd.DataFrame(ov), "s2_overhauls.csv")
    note("s2_overhaul_weeks", len(ov), "outputs/results/s2_overhauls.csv (detect_squad_overhaul, threshold 8)")
    note("s2_overhaul_managers", ov["team_id"].nunique() if len(ov) else 0, "outputs/results/s2_overhauls.csv")
    note("s2_league_managers", ls["team_id"].nunique(), "league_selections_df.csv")
    vd = pd.read_csv(RESULTS / "vine_diagnostics_2425.csv")
    for c in ["sparse_dim", "sparse_dep_pairs", "sparse_loglik", "sparse_null_loglik", "sparse_holdout_loglik",
              "ch3_dep_pairs", "ch3_null_dep_pairs", "ch3_loglik", "ch3_null_loglik",
              "n_unique_squads", "n_draws_feasible", "n_history_weeks"]:
        x = vd[c].replace(-np.inf, np.nan)
        note(f"s2_vine_{c}_mean", round(x.mean(), 2), "outputs/results/vine_diagnostics_2425.csv")
        note(f"s2_vine_{c}_median", round(x.median(), 2), "outputs/results/vine_diagnostics_2425.csv")
        note(f"s2_vine_{c}_min", round(x.min(), 2), "outputs/results/vine_diagnostics_2425.csv")
        note(f"s2_vine_{c}_max", round(x.max(), 2), "outputs/results/vine_diagnostics_2425.csv")
    note("s2_vine_weeks", len(vd), "outputs/results/vine_diagnostics_2425.csv")
    note("s2_vine_sparse_zero_dep_weeks", int((vd.sparse_dep_pairs == 0).sum()), "outputs/results/vine_diagnostics_2425.csv")
    note("s2_vine_sparse_beats_null_weeks", int((vd.sparse_loglik > vd.sparse_null_loglik).sum()), "outputs/results/vine_diagnostics_2425.csv")
    note("s2_vine_ch3_holdout_below0_weeks", int((vd.ch3_holdout_loglik < 0).sum()), "outputs/results/vine_diagnostics_2425.csv")
    note("s2_vine_ch3_holdout_ninf_weeks", int(np.isneginf(vd.ch3_holdout_loglik).sum()), "outputs/results/vine_diagnostics_2425.csv")
    note("s2_vine_ch3_holdout_median", round(vd.ch3_holdout_loglik.median(), 1), "outputs/results/vine_diagnostics_2425.csv")
    note("s2_vine_ch3_zero_pair_weeks", int((vd.ch3_dep_pairs == 0).sum()), "outputs/results/vine_diagnostics_2425.csv")
    note("s2_vine_ch3_more_on_null_weeks", int((vd.ch3_null_dep_pairs >= vd.ch3_dep_pairs).sum()), "outputs/results/vine_diagnostics_2425.csv")
    s2_opponent_check()
    print(league.describe().round(3).to_string(), "\n", gam.round(2).to_string(), "\n overhauls", len(ov))


def fig_s2_gamma(man):
    """The retention coefficient gamma_m of every manager fit: by position, and against the
    length of the manager's history."""
    FIGURES.mkdir(exist_ok=True)
    order = ["Goalkeeper", "Defender", "Midfielder", "Forward"]
    fig, ax = plt.subplots(1, 2, figsize=(6.3, 2.7))
    a = ax[0]
    a.boxplot([man.loc[man.position == p, "gamma_mean"] for p in order], tick_labels=order, widths=0.5,
              medianprops=dict(color="black"), flierprops=dict(marker="o", markersize=3))
    a.set(title="(a) Posterior mean of $\\gamma_m$", ylabel="$\\gamma_m$")
    a.tick_params(axis="x", labelsize=7)
    a = ax[1]
    for k, p in enumerate(order):
        x = man[man.position == p]
        a.plot(x["n_weeks"] + (k - 1.5) * 0.15, x["gamma_mean"], linestyle="none", markersize=3,
               label=p, **COEF_STYLES[k])
    a.set(title="(b) $\\gamma_m$ against history", xlabel="weeks of the manager's history",
          ylabel="$\\gamma_m$")
    a.legend(frameon=False, fontsize=6.5)
    fig.tight_layout()
    fig.savefig(FIGURES / "s2_gamma.pdf")
    plt.close(fig)


def s2_opponent_check(seed=SEED):
    """Out of sample check of the opponent model: for every gameweek's opponent, the target-week
    marginal probability of each player being in the squad (position_alpha_target, the inclusion
    probabilities of marginal_inclusion_probabilities), against the squad actually fielded."""
    g = notebook_ns(S2_NB_DIR / "optimization.ipynb", [2, 3, 5, 7, 9])
    rng = np.random.default_rng(seed)
    rows = []
    for gw in range(6, 39):
        opp = g["opponents_by_round"][gw]
        with cwd(S2_NB_DIR):  # the notebook's paths are relative to its folder
            mfits = joblib.load(g["MANAGER_FITS_PATH"].format(GW=gw))[gw]
        for pos in g["POSITIONS"]:
            lfit = g["get_league_fit_for_round"](g["LEAGUE_FITS"][pos], gw)
            alpha_map, owned = g["position_alpha_target"](opp, pos, mfits[pos], lfit, gw)
            el = np.array(list(alpha_map))
            p = g["marginal_inclusion_probabilities"](np.array(list(alpha_map.values())),
                                                      g["SQUAD_SLOTS"][pos], rng)
            picked = g["squad_in_position"](opp, gw, pos)
            if not picked:
                continue
            for e, pe in zip(el.tolist(), p.tolist()):
                rows.append({"gw": gw, "opponent": opp, "position": pos, "element": e, "p": pe,
                             "owned_prev": e in owned, "picked": e in picked})
    d = pd.DataFrame(rows)
    # overhaul weeks: the opponent dropped 8 or more of last week's 15 (the threshold of
    # detect_squad_overhaul); the selection model is not trained on such weeks
    dropped = d[d.owned_prev & ~d.picked].groupby("gw").size().reindex(d["gw"].unique(), fill_value=0)
    d["overhaul"] = d["gw"].isin(dropped[dropped >= 8].index)
    d["rank_new"] = d[~d.owned_prev].groupby(["gw", "position"])["p"].rank(ascending=False)
    save(d, "s2_opponent_check.csv")
    src = "outputs/results/s2_opponent_check.csv"
    note("s2_opp_rows", len(d), src)
    note("s2_opp_overhaul_gws", " ".join(map(str, sorted(d.loc[d.overhaul, "gw"].unique()))), src)
    summ = []
    for label, x in [("all", d), ("normal", d[~d.overhaul])]:
        kept, r = x[x.owned_prev], x.loc[~x.owned_prev & x.picked, "rank_new"]
        row = {"weeks": label, "n_weeks": x["gw"].nunique(),
               "brier": ((x["p"] - x["picked"]) ** 2).mean(),
               "brier_last_week": ((x["owned_prev"].astype(float) - x["picked"]) ** 2).mean(),
               "keep_p": kept["p"].mean(), "keep_rate": kept["picked"].mean(),
               "other_p": x.loc[~x.owned_prev, "p"].mean(), "other_rate": x.loc[~x.owned_prev, "picked"].mean(),
               "buys": len(r), "buys_top5": (r <= 5).mean(), "buys_top10": (r <= 10).mean(),
               "buys_top30": (r <= 30).mean(), "buys_median_rank": r.median()}
        summ.append(row)
        for k, v in row.items():
            if k != "weeks":
                note(f"s2_opp_{label}_{k}", round(float(v), 5 if "brier" in k else 3), src)
    for (pos, own), x in d[~d.overhaul].groupby(["position", "owned_prev"]):
        note(f"s2_opp_normal_{pos}_{'kept' if own else 'other'}_p", round(x["p"].mean(), 3), src)
        note(f"s2_opp_normal_{pos}_{'kept' if own else 'other'}_rate", round(x["picked"].mean(), 3), src)
    summ = save(pd.DataFrame(summ), "s2_opponent_check_summary.csv")
    fig_s2_opponent_check(d[~d.overhaul])
    print(summ.round(4).T.to_string())


def fig_s2_opponent_check(d):
    """(a) calibration of the opponent marginals, pooled over the gameweeks given, separately for
    the previous week's squad and the rest; (b) where the players the opponent bought ranked
    among the players it did not own, by predicted probability."""
    FIGURES.mkdir(exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(6.3, 2.8))
    a = ax[0]
    edges = np.array([0, 3e-4, 1e-3, 3e-3, 0.01, 0.03, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0001])
    for k, (own, label) in enumerate([(True, "last week's squad"), (False, "other players")]):
        x = d[d.owned_prev == own].copy()
        x["bin"] = pd.cut(x["p"], edges, include_lowest=True)
        b = x.groupby("bin", observed=True).agg(p=("p", "mean"), rate=("picked", "mean"), n=("p", "size"))
        b = b[b.n >= 20]
        a.plot(b["p"], b["rate"], label=label, linestyle="-", markersize=4, **COEF_STYLES[k])
    a.plot([1e-4, 1], [1e-4, 1], color="0.3", linewidth=0.7, linestyle=":")
    a.set(title="(a) Calibration", xlabel="predicted probability in squad", ylabel="observed share in squad",
          xscale="log", yscale="log", xlim=(1e-4, 1.3), ylim=(1e-4, 1.3))
    a.legend(frameon=False, fontsize=7)
    a = ax[1]
    r = d.loc[~d.owned_prev & d.picked, "rank_new"]
    a.hist(r.clip(upper=60), bins=np.arange(0.5, 61.5, 2), color="0.5", edgecolor="white")
    a.set(title="(b) Rank of bought players", xlabel="rank by predicted probability (60 = 60+)",
          ylabel="players bought")
    fig.tight_layout()
    fig.savefig(FIGURES / "s2_opponent_check.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------------------------
# Stage 2: the rolling seasons (2024-25)
# ---------------------------------------------------------------------------------------------
S2_SETUP = [2, 3, 5, 7, 9, 11, 13, 15, 17, 19]


def s2_load_runs():
    runs = {}
    for pct in R_GRID:
        p = S2_NB_DIR / f"rolling_optimization_results_mvt_r{pct}.joblib"
        if p.exists():
            runs[pct] = joblib.load(p)
    p = S2_NB_DIR / "rolling_optimization_results_base_rsweep.joblib"
    base = joblib.load(p) if p.exists() else None
    return runs, base


def s2_rows(df, label, r=None):
    rows = []
    for _, x in df.iterrows():
        ex = x["extra"] if isinstance(x.get("extra"), dict) else {}
        ok = x["status"] == "ok"
        log = ex.get("solve_log", [])
        rows.append({
            "run": label, "r": r, "gw": x["GW"], "status": x["status"],
            "points": x.get("sim_optimized_points") if ok else np.nan,
            "xP": x.get("sim_optimized_xP") if ok else np.nan,
            "made_transfer": x.get("made_transfer") if ok else np.nan,
            "n_transfers": x["n_transfers"], "lambda_star": x.get("lambda_star") if ok else np.nan,
            "prob_beat": x.get("prob_beat_opponent") if (ok and label == "mvt") else np.nan,
            "fallback": x.get("used_fallback_benchmark") if ok else np.nan,
            "sense": ex.get("sense", np.nan), "team_mean": ex.get("team_mean", np.nan),
            "team_var": ex.get("team_var", np.nan), "mu_G_r": ex.get("mu_G_r", np.nan),
            "sd_G_r": ex.get("sd_G_r", np.nan), "n_unique_Wo": ex.get("n_unique_Wo", np.nan),
            "n_draws_Wo": ex.get("n_draws_Wo", np.nan), "G_r_is_max": ex.get("G_r_is_max", np.nan),
            "n_solves": len(log), "n_timelimit": sum(s["status"] == "timelimit" for s in log),
            "n_failed_solves": sum(not s["solved"] for s in log),
            "solve_errors": sum(s["status"] == "error" for s in log),
        })
    return rows


def s2_runs():
    runs, base = s2_load_runs()
    if base is None or not runs:
        print("s2: sweep results not complete yet; skipped")
        return
    g = notebook_ns(S2_NB_DIR / "optimization.ipynb", S2_SETUP + [23], split={23: "compare_rows = []"})
    ref = []
    for gw in range(int(base["GW"].min()), int(base["GW"].max()) + 1):
        rx, rp = g["real_squad_actuals"](g["OWN_TEAM_ID"], gw)
        ox, op = g["real_squad_actuals"](g["opponents_by_round"][gw], gw)
        ref.append({"gw": gw, "opponent": g["opponents_by_round"][gw], "real_points": rp, "real_xP": rx,
                    "opp_points": op, "opp_xP": ox})
    ref = pd.DataFrame(ref)
    rows = s2_rows(base, "base")
    for pct, df in runs.items():
        rows += s2_rows(df, "mvt", pct)
    d = pd.DataFrame(rows).merge(ref, on="gw")
    d["win"] = (d["points"] > d["opp_points"]).astype(float) + 0.5 * (d["points"] == d["opp_points"])
    d.loc[d["points"].isna(), "win"] = np.nan
    d["xP_valid"] = ~d["gw"].isin(MISSING_TARGET["24-25"])
    save(d, "s2_runs_by_gw.csv")
    b = d[d["run"] == "base"].set_index("gw")
    rng = np.random.default_rng(SEED)
    summ = []
    real_win = (ref["real_points"] > ref["opp_points"]) + 0.5 * (ref["real_points"] == ref["opp_points"])
    for pct in sorted(runs):
        m = d[(d["run"] == "mvt") & (d["r"] == pct)].set_index("gw")
        solved = m.index[(m["status"] == "ok") & (b.loc[m.index, "status"] == "ok")]
        mm, bb = m.loc[solved], b.loc[solved]
        held = mm["made_transfer"] == False
        tr = mm[mm["made_transfer"] == True]
        pp = paired(mm["points"], bb["points"], rng)
        st = sign_test(mm["win"].to_numpy(), bb["win"].to_numpy())
        lo, hi = boot_ci(mm["win"].to_numpy() - bb["win"].to_numpy(), rng)
        xv = mm["xP_valid"]
        summ.append({
            "r": pct, "weeks": len(m), "solved_both": len(solved),
            "skipped": int((m["status"] != "ok").sum()), "fallback_weeks": int((m["fallback"] == True).sum()),
            "timelimit_weeks": int((m["n_timelimit"] > 0).sum()), "solve_error_weeks": int((m["solve_errors"] > 0).sum()),
            "G_r_is_max_all": bool(m["G_r_is_max"].dropna().all()) if pct == 100 else np.nan,
            "behind_branch_weeks": int((mm["sense"] == -1).sum()),
            "held_weeks": int(held.sum()), "transfer_weeks": int((mm["made_transfer"] == True).sum()),
            "mean_lambda_transfer": tr["lambda_star"].mean(),
            "lambda_nonzero_weeks": int((tr["lambda_star"] > 0).sum()),
            "mean_team_sd": np.sqrt(mm["team_var"]).mean(), "base_mean_team_sd": np.sqrt(bb["team_var"]).mean(),
            "points": mm["points"].sum(), "base_points": bb["points"].sum(),
            "wins": mm["win"].sum(), "base_wins": bb["win"].sum(),
            "win_rate": mm["win"].mean(), "base_win_rate": bb["win"].mean(),
            "mean_xP": mm.loc[xv, "xP"].mean(), "base_mean_xP": bb.loc[xv, "xP"].mean(), "xP_weeks": int(xv.sum()),
            "n_transfers": int(mm["n_transfers"].sum()), "base_n_transfers": int(bb["n_transfers"].sum()),
            "mean_prob": mm["prob_beat"].mean(), "brier": ((mm["prob_beat"] - mm["win"]) ** 2).mean(),
            "points_diff_mean": pp["mean_diff"], "points_diff_lo": pp["ci_lo"], "points_diff_hi": pp["ci_hi"],
            "points_wilcoxon_p": pp["wilcoxon_p"], "points_n_nonzero": pp["n_nonzero"],
            "win_diff_lo": lo, "win_diff_hi": hi, "win_mvt_better": st["a_better"],
            "win_base_better": st["b_better"], "win_sign_p": st["sign_p"],
            "identical_points_weeks": int((mm["points"] == bb["points"]).sum()),
            "real_points": ref.set_index("gw").loc[solved, "real_points"].sum(),
            "opp_points": ref.set_index("gw").loc[solved, "opp_points"].sum(),
            "real_wins": real_win.set_axis(ref["gw"]).loc[solved].sum(),
        })
    s = save(pd.DataFrame(summ), "s2_r_summary.csv")
    for _, r in s.iterrows():
        for c in s.columns:
            if c != "r":
                v = r[c]
                note(f"s2_r{int(r.r)}_{c}", round(v, 4) if isinstance(v, float) else v, "outputs/results/s2_r_summary.csv")
    bs = b[b["status"] == "ok"]
    for k, v in [("s2_base_points", bs["points"].sum()), ("s2_base_wins", bs["win"].sum()),
                 ("s2_base_weeks", len(bs)), ("s2_base_transfer_weeks", int((bs["made_transfer"] == True).sum())),
                 ("s2_base_mean_xP", round(bs.loc[bs["xP_valid"], "xP"].mean(), 2)),
                 ("s2_real_points", ref["real_points"].sum()), ("s2_opp_points", ref["opp_points"].sum()),
                 ("s2_real_wins", real_win.sum())]:
        note(k, v, "outputs/results/s2_runs_by_gw.csv")
    # the worked gameweek: at r = 0.5, the week where lambda* > 0 gains most P(beat) over lambda = 0
    if 50 in runs:
        best, gain = None, -1
        for _, x in runs[50].iterrows():
            ex = x.get("extra")
            if x["status"] != "ok" or not isinstance(ex, dict) or not ex.get("frontier"):
                continue
            fr = pd.DataFrame(ex["frontier"]).set_index("lambda")
            if 0.0 in fr.index and fr["prob_beat"].idxmax() > 0:
                gn = fr["prob_beat"].max() - fr.loc[0.0, "prob_beat"]
                if gn > gain:
                    best, gain = x, gn
        if best is not None:
            ex = best["extra"]
            fr = pd.DataFrame(ex["frontier"])
            fr["sd"] = np.sqrt(fr["var"])
            fr.insert(0, "gw", best["GW"])
            save(fr, "s2_worked_frontier.csv")
            for k, v in [("s2_worked_gw", best["GW"]), ("s2_worked_opponent", best["opponent"]),
                         ("s2_worked_sense", ex["sense"]), ("s2_worked_mu_G_r", round(ex["mu_G_r"], 2)),
                         ("s2_worked_sd_G_r", round(ex["sd_G_r"], 2)), ("s2_worked_unique_Wo", ex["n_unique_Wo"]),
                         ("s2_worked_draws_Wo", ex["n_draws_Wo"]), ("s2_worked_lambda_star", best["lambda_star"]),
                         ("s2_worked_prob_star", round(best["prob_beat_opponent"], 3)),
                         ("s2_worked_made_transfer", best["made_transfer"]),
                         ("s2_worked_mu_Yw_current", round(ex["mu_Yw_current"], 2))]:
                note(k, v, "outputs/results/s2_worked_frontier.csv (r = 0.5)")
            fig_s2_frontier(fr, best)
    # mu_delta / Sigma_delta as built for one gameweek (the worked one, else GW20)
    gw = int(best["GW"]) if 50 in runs and best is not None else 20
    elements, mu, Sigma, noise = g["build_mu_sigma_delta"](gw, g["preds"])
    note("s2_sigma_gw", gw, "build_mu_sigma_delta (optimization.ipynb cell 13)")
    note("s2_sigma_players", len(elements), "build_mu_sigma_delta")
    sd = np.sqrt(np.diag(Sigma))
    note("s2_sigma_sd_median", round(float(np.median(sd)), 2), "build_mu_sigma_delta")
    note("s2_sigma_noise_sd_median", round(float(np.median(np.sqrt(noise))), 2), "build_mu_sigma_delta")
    fig_s2_r(s)
    fig_s2_season(d, ref)
    print(s.round(3).T.to_string())


def fig_s2_frontier(fr, best):
    FIGURES.mkdir(exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(6.3, 2.6))
    a = ax[0]
    a.plot(fr["lambda"], fr["prob_beat"], **STYLE["MVT"])
    lam = best["lambda_star"]
    star = fr.loc[fr["lambda"] == lam]
    a.plot(star["lambda"], star["prob_beat"], marker="*", markersize=12, color="black", linestyle="none",
           label="$\\lambda^*$")
    a.set_xscale("symlog", linthresh=0.01)
    a.set(title=f"(a) GW{best['GW']}: $P(Y_{{w(\\lambda)}}>0)$", xlabel="$\\lambda$", ylabel="probability")
    a.legend(frameon=False)
    a = ax[1]
    a.plot(fr["sd"], fr["mean"], **STYLE["MVT"])
    a.plot(star["sd"], star["mean"], marker="*", markersize=12, color="black", linestyle="none")
    a.set(title="(b) Mean against s.d. of $w(\\lambda)$", xlabel="$\\sigma_w$ (points)", ylabel="$\\mu_w$ (points)")
    fig.tight_layout()
    fig.savefig(FIGURES / "s2_frontier.pdf")
    plt.close(fig)


def fig_s2_r(s):
    FIGURES.mkdir(exist_ok=True)
    r = s["r"] / 100
    fig, ax = plt.subplots(2, 2, figsize=(6.3, 4.6))
    a = ax[0, 0]
    a.plot(r, s["points"], label="MVT", **STYLE["MVT"])
    a.plot(r, s["base_points"], label="Baseline", **STYLE["Baseline"])
    a.plot(r, s["real_points"], label="Real manager", **STYLE["Real manager"])
    a.plot(r, s["opp_points"], label="Real opponents", **STYLE["Real opponents"])
    a.set(title="(a) Season points (solved weeks)", xlabel="$r$", ylabel="points")
    a.legend(frameon=False)
    a = ax[0, 1]
    a.plot(r, s["win_rate"], label="MVT", **STYLE["MVT"])
    a.plot(r, s["base_win_rate"], label="Baseline", **STYLE["Baseline"])
    a.plot(r, s["real_wins"] / s["solved_both"], label="Real manager", **STYLE["Real manager"])
    a.set(title="(b) H2H win rate", xlabel="$r$", ylabel="share of gameweeks", ylim=(0, 1))
    a = ax[1, 0]
    a.plot(r, s["mean_team_sd"], label="MVT", **STYLE["MVT"])
    a.plot(r, s["base_mean_team_sd"], label="Baseline", **STYLE["Baseline"])
    a.set(title="(c) Mean s.d. of the chosen team", xlabel="$r$", ylabel="$\\sigma_w$ (points)")
    a = ax[1, 1]
    a.plot(r, s["mean_prob"], label="MVT predicted $P(Y_w>0)$", **STYLE["MVT"])
    a.plot(r, s["win_rate"], label="MVT observed win rate", color="black", linestyle="--", marker="^")
    a.set(title="(d) Predicted against observed", xlabel="$r$", ylabel="probability", ylim=(0, 1))
    a.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "s2_r_effect.pdf")
    plt.close(fig)


def fig_s2_season(d, ref):
    FIGURES.mkdir(exist_ok=True)
    fig, a = plt.subplots(figsize=(6.3, 2.8))
    m = d[(d["run"] == "mvt") & (d["r"] == 50)].sort_values("gw")
    b = d[d["run"] == "base"].sort_values("gw")
    ref = ref.sort_values("gw")
    for label, x, y in [("MVT ($r=0.5$)", m["gw"], m["points"]), ("Baseline", b["gw"], b["points"]),
                        ("Real manager", ref["gw"], ref["real_points"]), ("Real opponents", ref["gw"], ref["opp_points"])]:
        key = "MVT" if label.startswith("MVT") else label
        st = dict(STYLE[key], marker=None)
        a.plot(x, np.cumsum(np.nan_to_num(y.to_numpy(dtype=float))), label=label, **st)
    a.set(xlabel="gameweek", ylabel="cumulative points", title="Cumulative points, 2024-25 (GW6-38)")
    a.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(FIGURES / "s2_season.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------------------------
# Figures and numbers.md
# ---------------------------------------------------------------------------------------------
def fig_s1_r(s):
    FIGURES.mkdir(exist_ok=True)
    r = s["r"] / 100
    fig, ax = plt.subplots(2, 2, figsize=(6.3, 4.6))
    a = ax[0, 0]
    a.plot(r, s["points"], label="MVT", **STYLE["MVT"])
    a.plot(r, s["base_points"], label="Baseline", **STYLE["Baseline"])
    a.set(title="(a) Season points", xlabel="$r$", ylabel="points")
    a.legend(frameon=False)
    a = ax[0, 1]
    a.plot(r, s["prize"], label="MVT", **STYLE["MVT"])
    a.plot(r, s["base_prize"], label="Baseline", **STYLE["Baseline"])
    a.set(title="(b) Season prize money", xlabel="$r$", ylabel="prize")
    a = ax[1, 0]
    a.plot(r, s["mean_team_sd"], label="MVT", **STYLE["MVT"])
    a.plot(r, s["mean_lam0_sd"], label="Baseline ($\\lambda=0$)", **STYLE["Baseline"])
    a.set(title="(c) Mean s.d. of the chosen team", xlabel="$r$", ylabel="$\\sigma_w$ (points)")
    a.legend(frameon=False)
    a = ax[1, 1]
    a.plot(r, s["mean_prob"], label="MVT predicted $P(Y_w>0)$", **STYLE["MVT"])
    a.plot(r, s["hits"] / s["weeks"], label="MVT observed", color="black", linestyle="--", marker="^")
    a.plot(r, s["base_hits"] / s["weeks"], label="Baseline observed", **STYLE["Baseline"])
    a.set(title="(d) Weeks in the top $1-r$", xlabel="$r$", ylabel="share of gameweeks", ylim=(0, 1.02))
    a.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "s1_r_effect.pdf")
    plt.close(fig)


def write_numbers():
    lines = ["# Numbers quoted in Chapter 4", "",
             "Generated by `thesis/make_results.py`. Every number in the chapter appears here with its source.", "",
             "| key | value | source |", "|---|---|---|"]
    lines += [f"| {k} | {v} | {s} |" for k, v, s in NUMBERS]
    (THESIS / "numbers.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    parts = sys.argv[1:] or ["pred", "s1", "s2fits", "s2"]
    if "pred" in parts:
        predictors()
    if "s1" in parts:
        s1_selection_model()
        s1_field()
        d1, s1 = s1_runs()
        fig_s1_r(s1)
    if "s2fits" in parts:
        s2_fits()
    if "s2" in parts and "s2_runs" in globals():
        s2_runs()
    write_numbers()
