"""
analysis/plots.py — Two-Agent POMDP delirium model (Level-3 room agent)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from world.spaces import ALL_PHENOTYPES
from simulation.cohort import CohortResults

PHENOTYPE_COLORS = {
    "A_young_healthy":    "#2E6DB4",
    "B_elderly_frail":    "#E8A838",
    "C_septic_ventilated":"#C0392B",
    "D_dementia_MCI":     "#8E44AD",
}
PHENOTYPE_SHORT = {
    "A_young_healthy":    "A: Young",
    "B_elderly_frail":    "B: Elderly",
    "C_septic_ventilated":"C: Septic",
    "D_dementia_MCI":     "D: Dementia",
}

EPS = 1e-16


def _shade(ax, t, mean, std, color, alpha=0.13):
    ax.fill_between(t, mean - std, mean + std, color=color, alpha=alpha)


def plot_sync_traces(cohort, save_path=None):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ph in ALL_PHENOTYPES:
        c, lbl = PHENOTYPE_COLORS[ph.name], PHENOTYPE_SHORT[ph.name]
        rs = cohort.by_phenotype(ph.name)
        if not rs: continue
        si_m  = cohort.sync_trace_mean(ph.name)
        r2p_m = cohort.s_r2p_trace_mean(ph.name)
        p2r_m = cohort.s_p2r_trace_mean(ph.name)
        t = np.arange(len(si_m))
        si_std  = np.array([r.sync_trace  for r in rs]).std(axis=0)
        r2p_std = np.array([r.s_r2p_trace for r in rs]).std(axis=0)
        p2r_std = np.array([r.s_p2r_trace for r in rs]).std(axis=0)
        for ax, m, sd in [(axes[0], si_m, si_std),
                           (axes[1], r2p_m, r2p_std),
                           (axes[2], p2r_m, p2r_std)]:
            ax.plot(t, m, color=c, lw=2.0, label=lbl)
            _shade(ax, t, m, sd, c)

    axes[0].axhline(0.35, color="gray", lw=1.0, ls="--", alpha=0.6, label="Expl. threshold")
    for ax in axes[1:]:
        ax.axhline(3.0, color="red", lw=1.0, ls="--", alpha=0.6, label="Del. threshold")
    for ax, title, ylabel in zip(axes,
        ["Composite SI (1=perfect)",
         r"Room→Patient  $S_{R\to P}=-\ln Q_R(s^*)$",
         r"Patient→Room  $S_{P\to R}=\mathcal{F}_P$"],
        ["Sync index [0,1]", "Surprisal (nats)", "VFE_P (nats)"]):
        ax.set_title(title, fontsize=10); ax.set_xlabel("Cycle", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9); ax.legend(fontsize=8); ax.grid(alpha=0.28)
    fig.suptitle("Predictive synchronization over time — Level-3 Room", fontsize=12)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    return fig


def plot_delirium_vs_sync(cohort, save_path=None):
    fig, ax = plt.subplots(figsize=(8, 6))
    for ph in ALL_PHENOTYPES:
        rs = cohort.by_phenotype(ph.name)
        ax.scatter([r.mean_sync for r in rs],
                   [float(r.p_delirium_trace[-1]) for r in rs],
                   c=PHENOTYPE_COLORS[ph.name], label=PHENOTYPE_SHORT[ph.name],
                   s=60, alpha=0.75, edgecolors="white", lw=0.5)

    all_sync = [r.mean_sync for r in cohort.results]
    all_pdel = [float(r.p_delirium_trace[-1]) for r in cohort.results]
    if len(set(all_pdel)) > 1:
        corr = float(np.corrcoef(all_sync, all_pdel)[0, 1])
        ax.text(0.05, 0.92, f"r = {corr:.3f}", transform=ax.transAxes,
                fontsize=11, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))

    ax.axhline(0.70, color="gray", lw=1.0, ls="--", alpha=0.7, label="Del. threshold")
    ax.axvline(0.35, color="gray", lw=1.0, ls=":", alpha=0.7, label="Sync alarm")
    ax.set_xlabel("Mean synchronization index", fontsize=11)
    ax.set_ylabel("P(delirium)", fontsize=11)
    ax.set_title("Hypothesis: delirium ← de-synchronization\n"
                 "Expected: negative r(sync, P_del)", fontsize=11)
    ax.set_xlim(0, 1.05); ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    return fig


def plot_vfe_traces(cohort, save_path=None):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    axes = axes.flatten()
    for i, ph in enumerate(ALL_PHENOTYPES):
        ax    = axes[i]
        c     = PHENOTYPE_COLORS[ph.name]
        rs    = cohort.by_phenotype(ph.name)
        if not rs: continue
        vp    = cohort.vfe_patient_trace_mean(ph.name)
        vr    = cohort.vfe_room_trace_mean(ph.name)
        t     = np.arange(len(vp))
        vp_sd = np.array([r.vfe_patient_trace for r in rs]).std(axis=0)
        ax.plot(t, vp, color=c, lw=2.0, label="VFE patient")
        ax.plot(t, vr, color=c, lw=2.0, ls="--", label="VFE room")
        _shade(ax, t, vp, vp_sd, c)
        ax.axhline(3.0, color="red", lw=0.8, ls=":", alpha=0.6, label="Del. threshold")
        ax.set_title(PHENOTYPE_SHORT[ph.name], fontsize=10, color=c, fontweight="bold")
        ax.set_ylabel("VFE (nats)", fontsize=9); ax.set_ylim(bottom=0)
        ax.legend(fontsize=8); ax.grid(alpha=0.25)
    for ax in axes: ax.set_xlabel("Decision cycle", fontsize=9)
    fig.suptitle("VFE: patient (solid) vs room (dashed)\n"
                 "Divergence = de-synchronization", fontsize=12)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    return fig


def plot_exploration(cohort, save_path=None):
    """
    Level-3 room agent summary figure — three panels:
      1. θ learning progress (bar) per phenotype
      2. S_{R→P} distribution (boxplot) per phenotype
      3. MAP θ scatter: θ_cog vs θ_emo per patient, coloured by phenotype
    Replaces the trivial exploration-rate bar chart which is always 0%
    with the Level-3 architecture.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # ── Panel 1: θ learning progress ─────────────────────────────────────
    ax = axes[0]
    learns = [cohort.mean_learning(ph.name) for ph in ALL_PHENOTYPES]
    names  = [PHENOTYPE_SHORT[ph.name]       for ph in ALL_PHENOTYPES]
    colors = [PHENOTYPE_COLORS[ph.name]      for ph in ALL_PHENOTYPES]
    bars   = ax.bar(names, learns, color=colors, alpha=0.82,
                    edgecolor="white", width=0.55)
    for bar, val in zip(bars, learns):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.0005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10,
                fontweight="bold")
    ax.set_ylabel("Mean learning progress  ℓ", fontsize=10)
    ax.set_title("Room learning about patient θ\n"
                 "ℓ = 0: uniform prior  |  ℓ = 1: θ fully identified",
                 fontsize=10)
    ax.set_ylim(0, max(learns) * 2.0 + 0.005)
    ax.grid(axis="y", alpha=0.3)
    # Annotation explaining what learning means
    ax.text(0.5, 0.85,
            "Higher ℓ → room has built\na patient-specific model",
            transform=ax.transAxes, ha="center", fontsize=9,
            color="gray", style="italic")

    # ── Panel 2: S_{R→P} distribution per phenotype ───────────────────────
    ax = axes[1]
    bplot_data  = []
    bplot_cols  = []
    bplot_names = []
    for ph in ALL_PHENOTYPES:
        rs = cohort.by_phenotype(ph.name)
        if not rs:
            continue
        # Collect all per-cycle S_R2P values across patients
        all_vals = np.concatenate([r.s_r2p_trace for r in rs])
        bplot_data.append(all_vals)
        bplot_cols.append(PHENOTYPE_COLORS[ph.name])
        bplot_names.append(PHENOTYPE_SHORT[ph.name])

    bp = ax.boxplot(bplot_data, patch_artist=True, widths=0.5,
                    medianprops=dict(color="white", lw=2.0),
                    whiskerprops=dict(lw=1.2),
                    capprops=dict(lw=1.2),
                    flierprops=dict(marker=".", ms=3, alpha=0.3))
    for patch, col in zip(bp["boxes"], bplot_cols):
        patch.set_facecolor(col)
        patch.set_alpha(0.78)
    ax.axhline(3.0, color="red", lw=1.2, ls="--", alpha=0.7,
               label="Delirium threshold (3.0 nats)")
    ax.set_xticks(range(1, len(bplot_names) + 1))
    ax.set_xticklabels(bplot_names, fontsize=9)
    ax.set_ylabel("$S_{R\\to P}$ = $-\\ln Q_R(s^*)$  (nats)", fontsize=10)
    ax.set_title("Room surprisal at patient's true state\n"
                 "Lower = room predicts patient better",
                 fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    # ── Panel 3: MAP θ scatter ────────────────────────────────────────────
    ax = axes[2]
    for ph in ALL_PHENOTYPES:
        rs  = cohort.by_phenotype(ph.name)
        c   = PHENOTYPE_COLORS[ph.name]
        lbl = PHENOTYPE_SHORT[ph.name]
        tcs = [getattr(r, '_theta_cog_final', 0.5) for r in rs]
        tes = [getattr(r, '_theta_emo_final', 0.5) for r in rs]
        ax.scatter(tcs, tes, c=c, s=65, alpha=0.80,
                   edgecolors="white", lw=0.6, label=lbl)
        # Mark true θ for this phenotype
        true_tc = [1.00, 0.65, 0.50, 0.35][ALL_PHENOTYPES.index(ph)]
        true_te = [1.00, 1.30, 1.70, 1.45][ALL_PHENOTYPES.index(ph)]
        ax.scatter([true_tc], [true_te], c=c, s=180, marker="*",
                   edgecolors="black", lw=0.8, zorder=5)

    ax.axhline(0.5, color="gray", lw=0.8, ls="--", alpha=0.5)
    ax.axvline(0.5, color="gray", lw=0.8, ls="--", alpha=0.5)
    ax.set_xlabel("Inferred $\\theta_{\\mathrm{cog}}$ (cognitive plasticity)",
                  fontsize=10)
    ax.set_ylabel("Inferred $\\theta_{\\mathrm{emo}}$ (emotional reactivity)",
                  fontsize=10)
    ax.set_title("Room MAP estimate of patient parameters\n"
                 "Dots = inferred  |  Stars = true θ",
                 fontsize=10)
    ax.set_xlim(0.0, 1.1); ax.set_ylim(0.0, 1.1)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.28)

    fig.suptitle("Level-3 Room Agent — Hierarchical Inference about the Patient\n"
                 "Room learns θ = (θ_cog, θ_emo) from physiological observations",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    return fig


def plot_learning(cohort, save_path=None):
    """θ learning progress and inferred parameters by phenotype."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1: mean learning progress per phenotype
    learns = [cohort.mean_learning(ph.name) for ph in ALL_PHENOTYPES]
    colors = [PHENOTYPE_COLORS[ph.name]      for ph in ALL_PHENOTYPES]
    names  = [PHENOTYPE_SHORT[ph.name]       for ph in ALL_PHENOTYPES]
    bars   = axes[0].bar(names, learns, color=colors, alpha=0.8, edgecolor="white", width=0.55)
    for bar, val in zip(bars, learns):
        axes[0].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.001,
                     f"{val:.3f}", ha="center", va="bottom", fontsize=10)
    axes[0].set_ylim(0, max(learns) * 1.5 + 0.01)
    axes[0].set_ylabel("Mean learning progress (0→1)", fontsize=10)
    axes[0].set_title("Room θ learning progress by phenotype\n"
                       "(0 = uniform prior, 1 = θ fully identified)", fontsize=10)
    axes[0].grid(axis="y", alpha=0.3)

    # Panel 2: per-patient scatter θ_cog vs θ_emo
    for ph in ALL_PHENOTYPES:
        c  = PHENOTYPE_COLORS[ph.name]
        rs = cohort.by_phenotype(ph.name)
        tcs = [getattr(r, '_theta_cog_final', 0.5) for r in rs]
        tes = [getattr(r, '_theta_emo_final', 0.5) for r in rs]
        axes[1].scatter(tcs, tes, c=c, s=60, alpha=0.75,
                        edgecolors="white", lw=0.5,
                        label=PHENOTYPE_SHORT[ph.name])

    axes[1].axhline(0.5, color="gray", lw=0.8, ls="--", alpha=0.5)
    axes[1].axvline(0.5, color="gray", lw=0.8, ls="--", alpha=0.5)
    axes[1].set_xlabel("Inferred θ_cog (cognitive plasticity)", fontsize=10)
    axes[1].set_ylabel("Inferred θ_emo (emotional reactivity)", fontsize=10)
    axes[1].set_title("MAP estimates of patient parameters\n"
                       "Dashed = uninformative prior", fontsize=10)
    axes[1].set_xlim(0, 1.05); axes[1].set_ylim(0, 1.05)
    axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)

    fig.suptitle("Level-3 hierarchical inference: room learning patient θ\n"
                 "θ = (θ_cog, θ_emo) inferred from physiological observations",
                 fontsize=12)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    return fig


def plot_dashboard(cohort, save_path=None):
    fig = plt.figure(figsize=(16, 11))
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.42, wspace=0.35)

    ax_sync  = fig.add_subplot(gs[0, :2])
    ax_r2p   = fig.add_subplot(gs[0, 2])
    ax_sc    = fig.add_subplot(gs[1, :2])
    ax_learn = fig.add_subplot(gs[1, 2])
    ax_del   = fig.add_subplot(gs[2, :2])
    ax_txt   = fig.add_subplot(gs[2, 2])

    for ph in ALL_PHENOTYPES:
        c     = PHENOTYPE_COLORS[ph.name]
        lbl   = PHENOTYPE_SHORT[ph.name]
        si_m  = cohort.sync_trace_mean(ph.name)
        r2p_m = cohort.s_r2p_trace_mean(ph.name)
        pdm   = cohort.p_delirium_trace_mean(ph.name)
        t     = np.arange(len(si_m))
        ax_sync.plot(t, si_m,  color=c, lw=1.8, label=lbl)
        ax_r2p.plot(t, r2p_m, color=c, lw=1.8)
        ax_del.plot(t, pdm,   color=c, lw=1.8, label=lbl)
        rs = cohort.by_phenotype(ph.name)
        ax_sc.scatter([r.mean_sync for r in rs],
                      [float(r.p_delirium_trace[-1]) for r in rs],
                      c=c, s=35, alpha=0.7, edgecolors="white", lw=0.4, label=lbl)

    all_sync = [r.mean_sync for r in cohort.results]
    all_pdel = [float(r.p_delirium_trace[-1]) for r in cohort.results]
    if len(set(all_pdel)) > 1:
        corr = float(np.corrcoef(all_sync, all_pdel)[0, 1])
        ax_sc.text(0.05, 0.88, f"r = {corr:.3f}", transform=ax_sc.transAxes,
                   fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))

    ax_sync.axhline(0.35, color="gray", lw=1.0, ls="--", alpha=0.7)
    ax_sync.set_title("Synchronization index", fontsize=10)
    ax_sync.set_ylim(0, 1.05); ax_sync.legend(fontsize=8); ax_sync.grid(alpha=0.25)
    ax_r2p.axhline(5.0, color="red", lw=1.0, ls="--", alpha=0.7)
    ax_r2p.set_title("S_R→P", fontsize=10); ax_r2p.grid(alpha=0.25)
    ax_sc.axhline(0.70, color="gray", lw=0.8, ls="--", alpha=0.6)
    ax_sc.axvline(0.35, color="gray", lw=0.8, ls=":", alpha=0.6)
    ax_sc.set_xlabel("Mean sync", fontsize=9); ax_sc.set_ylabel("P(delirium)", fontsize=9)
    ax_sc.set_title("P(delirium) vs sync", fontsize=10)
    ax_sc.legend(fontsize=7); ax_sc.grid(alpha=0.25)
    ax_del.axhline(0.70, color="gray", lw=1.0, ls="--", alpha=0.7, label="Threshold")
    ax_del.set_title("Mean P(delirium) over time", fontsize=10)
    ax_del.set_xlabel("Decision cycle", fontsize=9)
    ax_del.set_ylim(0, 1.05); ax_del.legend(fontsize=8); ax_del.grid(alpha=0.25)

    learns = [cohort.mean_learning(ph.name) for ph in ALL_PHENOTYPES]
    colors = [PHENOTYPE_COLORS[ph.name]      for ph in ALL_PHENOTYPES]
    bars   = ax_learn.bar(range(4), learns, color=colors, alpha=0.8, width=0.6)
    for bar, val in zip(bars, learns):
        ax_learn.text(bar.get_x() + bar.get_width()/2,
                      bar.get_height() + 0.001,
                      f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    ax_learn.set_xticks(range(4))
    ax_learn.set_xticklabels(["A","B","C","D"], fontsize=9)
    ax_learn.set_title("θ learning progress", fontsize=10)
    ax_learn.set_ylim(0, max(learns) * 1.5 + 0.01); ax_learn.grid(axis="y", alpha=0.3)

    ax_txt.axis("off")
    lines = ["Summary (Level-3)\n"]
    for ph in ALL_PHENOTYPES:
        lines.append(f"{PHENOTYPE_SHORT[ph.name]}")
        lines.append(f"  Del:{cohort.delirium_rate(ph.name):.0%} "
                     f"SI:{cohort.mean_sync(ph.name):.2f}")
        lines.append(f"  R2P:{cohort.mean_s_r2p(ph.name):.2f} "
                     f"P2R:{cohort.mean_s_p2r(ph.name):.2f}")
        lines.append(f"  θ-learn:{cohort.mean_learning(ph.name):.3f}\n")
    ax_txt.text(0.05, 0.95, "\n".join(lines), transform=ax_txt.transAxes,
                fontsize=9, va="top", fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    fig.suptitle("ICU Delirium — Two-Agent POMDP (Level-3 Hierarchical Room)\n"
                 "Room infers θ = (θ_cog, θ_emo) — patient-specific generative model",
                 fontsize=13, fontweight="bold")
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    return fig
