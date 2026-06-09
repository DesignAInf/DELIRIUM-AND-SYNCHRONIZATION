"""
analysis/reviewer_response.py
=============================
Re-analysis addressing Reviewer 1 (Point 1: Pearson r is inappropriate for a
thresholded, non-linear relationship; use a logistic model).

WHAT THIS SCRIPT DOES
---------------------
1. Uses the genuine BINARY outcome `delirium_declared` (the sustained
   joint-threshold rule in loop.py) instead of `p_delirium_trace[-1]`.
   This matters: p_delirium_trace is, by construction in loop.py, a rescaled
   and clipped transform of the same desync that produces mean_sync, so
   correlating the two is partly circular. The binary `delirium_declared`
   flag is a genuinely different (sustained-threshold) function of the traces,
   so logistic(delirium_declared ~ mean_sync) is both the statistically
   appropriate model AND breaks the circularity.

2. Fits logistic regression P(delirium_declared) ~ mean_sync per model
   variant, reporting beta1, 95% CI, and AUC (discrimination).

3. Reports Spearman rho (monotonic, robust to the ceiling cluster) alongside
   Pearson r, so the reader can see the contrast the reviewer pointed to.

4. Runs the M0-M3 ablation across seeds by calling run_cohort with the
   existing flags, recomputing Del% per phenotype, rho (mean +/- SD), and the
   Grad fraction (seeds preserving A < B,C,D).

5. Re-plots Figure 2 with the binary outcome on the y-axis and the fitted
   logistic curve overlaid.

HOW TO RUN
----------
Place this file in your repo (e.g. analysis/reviewer_response.py) so that the
existing imports below resolve, then:

    pip install statsmodels        # optional but recommended for CIs
    python -m analysis.reviewer_response

All numbers it prints are computed from YOUR simulation. Paste them into the
paper as-is. Nothing here is hard-coded.

ADJUST THESE IF YOUR PAPER USED DIFFERENT SETTINGS
--------------------------------------------------
- MAIN_N_PER, MAIN_CYCLES, MAIN_SEED  -> the Figure 2 / Table 2 cohort
- ABLATION_SEEDS, ABLATION_N_PER, ABLATION_CYCLES -> Table 3
- The flag combinations in VARIANTS map to M0-M3 exactly as documented in
  run_cohort's docstring.
"""

import numpy as np
from scipy import stats

# --- existing project imports (names taken from your repo) -----------------
from simulation.cohort import run_cohort, CohortResults
from world.spaces import ALL_PHENOTYPES

# Optional: statsmodels gives analytic CIs; fall back to sklearn + bootstrap.
try:
    import statsmodels.api as sm
    _HAVE_SM = True
except Exception:
    _HAVE_SM = False
    from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_auc_score


# ---------------------------------------------------------------------------
# Settings -- EDIT to match the cohort sizes reported in the paper
# ---------------------------------------------------------------------------
MAIN_N_PER   = 30      # Figure 2 / Table 2: 30 per phenotype (120 total)
MAIN_CYCLES  = 200
MAIN_SEED    = 42

ABLATION_SEEDS  = [1, 7, 42]          # Table 3 seeds
ABLATION_N_PER  = 12                  # Table 3: 12 per phenotype
ABLATION_CYCLES = 150

# M0-M3 flag combinations (see run_cohort docstring)
VARIANTS = {
    "M0_random_room":   dict(single_agent=True,  use_theta_learning=True,  use_causal_feedback=True),
    "M1_no_theta":      dict(single_agent=False, use_theta_learning=False, use_causal_feedback=True),
    "M2_no_feedback":   dict(single_agent=False, use_theta_learning=True,  use_causal_feedback=False),
    "M3_full":          dict(single_agent=False, use_theta_learning=True,  use_causal_feedback=True),
}

# Phenotype ordering for the gradient check A < B, C, D
PHENO_ORDER = [ph.name for ph in ALL_PHENOTYPES]   # assumes A,B,C,D order


# ---------------------------------------------------------------------------
# Core statistics
# ---------------------------------------------------------------------------

def extract_xy(cohort: CohortResults):
    """Return (mean_sync array, binary delirium array) over all patients."""
    x = np.array([r.mean_sync for r in cohort.results], dtype=float)
    y = np.array([1 if r.delirium_declared else 0 for r in cohort.results], dtype=int)
    return x, y


def logistic_fit(x, y, n_boot=2000, seed=0):
    """
    Fit P(y=1) ~ x via logistic regression.
    Returns dict with beta0, beta1, ci_low, ci_high (for beta1), auc, and the
    fitted curve sampler. Uses statsmodels if available, else sklearn+bootstrap.
    """
    out = {}
    # AUC is model-agnostic discrimination of mean_sync for the binary outcome.
    # (Only defined if both classes present.)
    if len(np.unique(y)) == 2:
        out["auc"] = float(roc_auc_score(y, -x))  # lower sync -> higher delirium
    else:
        out["auc"] = float("nan")

    if _HAVE_SM:
        X = sm.add_constant(x)
        try:
            model = sm.Logit(y, X).fit(disp=0)
            out["beta0"] = float(model.params[0])
            out["beta1"] = float(model.params[1])
            ci = model.conf_int(alpha=0.05)
            out["ci_low"]  = float(ci[1][0])
            out["ci_high"] = float(ci[1][1])
            out["pvalue"]  = float(model.pvalues[1])
            out["_predict"] = lambda grid: model.predict(sm.add_constant(grid))
            return out
        except Exception:
            pass  # fall through to sklearn

    # sklearn fallback with bootstrap CI on beta1
    clf = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
    clf.fit(x.reshape(-1, 1), y)
    out["beta0"] = float(clf.intercept_[0])
    out["beta1"] = float(clf.coef_[0][0])
    rng = np.random.default_rng(seed)
    betas = []
    n = len(x)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        xb, yb = x[idx], y[idx]
        if len(np.unique(yb)) < 2:
            continue
        try:
            c = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
            c.fit(xb.reshape(-1, 1), yb)
            betas.append(c.coef_[0][0])
        except Exception:
            continue
    if betas:
        out["ci_low"], out["ci_high"] = np.percentile(betas, [2.5, 97.5])
    else:
        out["ci_low"] = out["ci_high"] = float("nan")
    out["pvalue"] = float("nan")
    out["_predict"] = lambda grid: clf.predict_proba(np.asarray(grid).reshape(-1, 1))[:, 1]
    return out


def correlations(x, y):
    """Pearson r and Spearman rho between mean_sync and binary outcome."""
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    r, _   = stats.pearsonr(x, y)
    rho, _ = stats.spearmanr(x, y)
    return float(r), float(rho)


def gradient_preserved(cohort: CohortResults):
    """True if delirium-rate gradient A < B, C, D holds (A strictly lowest)."""
    rates = [cohort.delirium_rate(name) for name in PHENO_ORDER]
    a = rates[0]
    return all(a < rates[i] for i in range(1, len(rates)))


# ---------------------------------------------------------------------------
# Main analyses
# ---------------------------------------------------------------------------

def analyse_main():
    print("=" * 72)
    print("MAIN COHORT (Figure 2 / Table 2 replacement)")
    print(f"  n_per={MAIN_N_PER}, cycles={MAIN_CYCLES}, seed={MAIN_SEED}, full model")
    print("=" * 72)
    cohort = run_cohort(
        n_per_phenotype=MAIN_N_PER, n_cycles=MAIN_CYCLES,
        seed=MAIN_SEED, verbose=False, **VARIANTS["M3_full"],
    )
    x, y = extract_xy(cohort)
    fit = logistic_fit(x, y)
    r, rho = correlations(x, y)

    print(f"\n  Patients: {len(y)}   Delirium (binary): {y.sum()} "
          f"({100*y.mean():.0f}%)")
    print(f"\n  Logistic  P(delirium_declared) ~ mean_sync")
    print(f"    beta1     = {fit['beta1']:.3f}  "
          f"(95% CI {fit['ci_low']:.3f}, {fit['ci_high']:.3f})")
    print(f"    AUC       = {fit['auc']:.3f}")
    print(f"  Spearman rho = {rho:.3f}   [report THIS]")
    print(f"  Pearson  r   = {r:.3f}   [old metric, for contrast only]")

    # Per-phenotype binary delirium rates (Table 2 col Del%)
    print(f"\n  Per-phenotype delirium rate (binary, declared):")
    for name in PHENO_ORDER:
        print(f"    {name:<24} {100*cohort.delirium_rate(name):>5.1f}%")

# --- Table 2 mean ± SD per phenotype (reviewer point 25) ---
    print(f"\n  Table 2 (mean +/- SD across patients within phenotype, ddof=1):")
    print(f"    {'Phenotype':<24} {'SI':>14} {'S_R->P':>14} "
          f"{'S_P->R':>14} {'ell':>14}")
    for name in PHENO_ORDER:
        rs = cohort.by_phenotype(name)
        si  = np.array([r.mean_sync     for r in rs], dtype=float)
        srp = np.array([r.mean_s_r2p    for r in rs], dtype=float)
        spr = np.array([r.mean_s_p2r    for r in rs], dtype=float)
        ell = np.array([r.mean_learning for r in rs], dtype=float)
        def ms(a):
            return f"{a.mean():.2f}+/-{a.std(ddof=1):.2f}"
        print(f"    {name:<24} {ms(si):>14} {ms(srp):>14} "
              f"{ms(spr):>14} {ms(ell):>14}   (n={len(rs)})")

    _plot_figure2(x, y, fit, rho, r)
    return cohort


def analyse_ablation():
    print("\n" + "=" * 72)
    print("ABLATION (Table 3 replacement)  -- binary outcome, Spearman rho")
    print(f"  n_per={ABLATION_N_PER}, cycles={ABLATION_CYCLES}, "
          f"seeds={ABLATION_SEEDS}")
    print("=" * 72)

    header = (f"\n  {'Model':<16} "
              + " ".join(f"{n.split('_')[0]+'Del%':>8}" for n in PHENO_ORDER)
              + f"{'rho(m±SD)':>16}{'Grad':>7}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    for vname, flags in VARIANTS.items():
        per_pheno_rates = {name: [] for name in PHENO_ORDER}
        rhos = []
        grad_count = 0
        for sd in ABLATION_SEEDS:
            cohort = run_cohort(
                n_per_phenotype=ABLATION_N_PER, n_cycles=ABLATION_CYCLES,
                seed=sd, verbose=False, **flags,
            )
            for name in PHENO_ORDER:
                per_pheno_rates[name].append(100 * cohort.delirium_rate(name))
            x, y = extract_xy(cohort)
            _, rho = correlations(x, y)
            if not np.isnan(rho):
                rhos.append(rho)
            if gradient_preserved(cohort):
                grad_count += 1

        rate_strs = []
        for name in PHENO_ORDER:
            vals = per_pheno_rates[name]
            rate_strs.append(f"{np.mean(vals):>4.0f}±{np.std(vals):>2.0f}")
        rho_m  = np.mean(rhos) if rhos else float("nan")
        rho_sd = np.std(rhos)  if rhos else float("nan")
        print(f"  {vname:<16} "
              + " ".join(f"{s:>8}" for s in rate_strs)
              + f"  {rho_m:>6.3f}±{rho_sd:<5.3f}"
              + f"{grad_count}/{len(ABLATION_SEEDS):>3}")


def _plot_figure2(x, y, fit, rho, r):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    # jitter the binary outcome slightly for visibility
    rng = np.random.default_rng(0)
    yj = y + rng.uniform(-0.02, 0.02, size=len(y))
    ax.scatter(x, yj, s=45, alpha=0.6, edgecolors="white", lw=0.4,
               color="#555555")

    grid = np.linspace(x.min(), x.max(), 200)
    ax.plot(grid, fit["_predict"](grid), color="#C0392B", lw=2.2,
            label="Logistic fit")

    ax.set_xlabel("Mean synchronization index", fontsize=11)
    ax.set_ylabel("Declared delirium (0/1) and fitted probability", fontsize=11)
    ax.set_title("Delirium vs synchronization (binary outcome, logistic fit)",
                 fontsize=11)
    ax.text(0.05, 0.55,
            f"Spearman rho = {rho:.3f}\nAUC = {fit['auc']:.3f}\n"
            f"(Pearson r = {r:.3f})",
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
    ax.set_ylim(-0.08, 1.08)
    ax.legend(fontsize=9, loc="center right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("figure2_logistic.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("\n  -> Saved figure2_logistic.png")

def analyse_robustness():
    """Table 5 replacement: 5-seed robustness on the binary outcome, Spearman rho."""
    SEEDS, N_PER, CYCLES = [1, 7, 13, 42, 99], 20, 200
    print("\n" + "=" * 72)
    print("ROBUSTNESS (Table 5 replacement) -- binary outcome, Spearman rho")
    print(f"  n_per={N_PER}, cycles={CYCLES}, seeds={SEEDS}")
    print("=" * 72)
    per_pheno = {name: [] for name in PHENO_ORDER}
    overall_rates, rhos = [], []
    for sd in SEEDS:
        cohort = run_cohort(n_per_phenotype=N_PER, n_cycles=CYCLES,
                            seed=sd, verbose=False, **VARIANTS["M3_full"])
        for name in PHENO_ORDER:
            per_pheno[name].append(100 * cohort.delirium_rate(name))
        overall_rates.append(100 * cohort.delirium_rate())
        x, y = extract_xy(cohort)
        _, rho = correlations(x, y)
        if not np.isnan(rho):
            rhos.append(rho)
    def row(label, vals):
        a = np.array(vals, float)
        print(f"  {label:<34} {a.mean():>6.1f} {a.std():>5.1f} {a.min():>5.1f} {a.max():>5.1f}")
    print(f"\n  {'Metric':<34} {'Mean':>6} {'SD':>5} {'Min':>5} {'Max':>5}")
    print("  " + "-" * 58)
    for name in PHENO_ORDER:
        row(f"{name} -- Del%", per_pheno[name])
    row("Overall delirium rate (%)", overall_rates)
    r = np.array(rhos, float)
    print(f"  {'rho(SI, declared delirium)':<34} {r.mean():>6.3f} {r.std():>5.3f} {r.min():>5.3f} {r.max():>5.3f}")

if __name__ == "__main__":
    analyse_main()
    analyse_ablation()
    analyse_robustness()
    print("\nDone. Paste the printed numbers into the paper "
          "(see the response-letter text).")
