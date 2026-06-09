"""
main.py — ICU Delirium Two-Agent POMDP Model
=============================================
Author: Luca M. Possati

Two bidirectionally coupled POMDP agents:
    PatientAgent: maintains belief over (γ_cog, γ_emo, circadian) hidden states
                  minimises own VFE from room observations
    RoomAgent:    maintains belief about patient state from physiological sensors
                  minimises EFE with preferences over healthy patient states
                  uses VFE_R as early-warning signal for de-synchronization

Core hypothesis: ICU delirium emerges from failure of synchronization
between patient and environment generative models.

Run:
    python main.py                    # standard run (48 patients, 30 cycles)
    python main.py --fast             # quick test (16 patients, 15 cycles)
    python main.py --n 24 --cycles 20
    python main.py --verbose          # print cycle-by-cycle output for 1 patient
    python main.py --no-plots         # text only
"""

import argparse
import os
import sys
import time
import numpy as np

OUTPUT_DIR = "results"
FIG_DIR    = os.path.join(OUTPUT_DIR, "figures")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast",      action="store_true",
                        help="Quick run: 16 patients, 15 cycles")
    parser.add_argument("--n",         type=int, default=12,
                        help="Patients per phenotype (default 12)")
    parser.add_argument("--cycles",    type=int, default=30,
                        help="Decision cycles per simulation (default 30)")
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--no-plots",  action="store_true")
    parser.add_argument("--verbose",   action="store_true",
                        help="Print one verbose single-patient trace")
    # Model variant flags for ablation / model comparison
    parser.add_argument("--no-theta",    action="store_true",
                        help="M1: disable Level-3 theta learning")
    parser.add_argument("--no-feedback", action="store_true",
                        help="M2: disable causal feedback loop")
    parser.add_argument("--single-agent", action="store_true",
                        help="M0: room acts randomly (no inference)")
    args = parser.parse_args()

    if args.fast:
        n_per_ph = 4
        n_cycles = 15
    else:
        n_per_ph = args.n
        n_cycles = args.cycles

    os.makedirs(FIG_DIR, exist_ok=True)

    print("=" * 70)
    print("  ICU Delirium — Two-Agent POMDP Simulation")
    print("  Author: Luca M. Possati")
    print("=" * 70)
    print(f"  Patients per phenotype : {n_per_ph}")
    print(f"  Total patients         : {n_per_ph * 4}")
    print(f"  Decision cycles        : {n_cycles}")
    print(f"  Hypothesis             : delirium ← de-synchronization")
    print("=" * 70)

    # Optional: verbose single-patient demo
    if args.verbose:
        _run_verbose_demo(n_cycles=n_cycles, seed=args.seed)

    # Full cohort
    from simulation.cohort import run_cohort
    print(f"\n{'─'*70}\n  Running cohort simulation\n{'─'*70}")
    t0     = time.time()
    # Determine model variant
    use_theta    = not args.no_theta
    use_feedback = not args.no_feedback
    use_single   = args.single_agent

    variant_label = "M3: Full model"
    if use_single:
        variant_label = "M0: Single agent (room random)"
    elif not use_theta and not use_feedback:
        variant_label = "M1+M2: No theta, no feedback"
    elif not use_theta:
        variant_label = "M1: No theta learning"
    elif not use_feedback:
        variant_label = "M2: No causal feedback"

    print(f"  Model variant: {variant_label}")

    cohort = run_cohort(
        n_per_phenotype=n_per_ph,
        n_cycles=n_cycles,
        seed=args.seed,
        verbose=True,
        use_theta_learning=use_theta,
        use_causal_feedback=use_feedback,
        single_agent=use_single,
    )
    elapsed = time.time() - t0
    print(f"\n  Completed in {elapsed:.1f}s")

    # Summary
    _print_summary(cohort)

    # Figures
    if not args.no_plots:
        print(f"\n{'─'*70}\n  Generating figures → {FIG_DIR}/\n{'─'*70}")
        _generate_figures(cohort)

    print(f"\n{'='*70}")
    print("  Simulation complete.")
    print(f"  Results → {os.path.abspath(OUTPUT_DIR)}/")
    print("=" * 70)


def _run_verbose_demo(n_cycles: int, seed: int):
    """Run and print a single patient trace (phenotype C: worst case)."""
    from world.spaces import PHENOTYPE_C
    from simulation.cohort import build_agents
    from simulation.loop import run_simulation

    print(f"\n{'─'*70}")
    print("  Verbose demo: one septic/ventilated patient (Phenotype C)")
    print(f"{'─'*70}")

    rng = np.random.default_rng(seed)
    patient, room, B_P, B_R = build_agents(PHENOTYPE_C, seed=seed)
    result = run_simulation(
        phenotype=PHENOTYPE_C,
        patient=patient, room=room,
        B_P_true=B_P, B_R_true=B_R,
        n_cycles=n_cycles, rng=rng, verbose=True,
    )

    print(f"\n  Outcome: {'DELIRIUM' if result.delirium_declared else 'no delirium'}")
    if result.delirium_declared:
        print(f"  Onset cycle: {result.delirium_onset_cycle}")
    print(f"  Mean sync (SI):  {result.mean_sync:.3f}")
    print(f"  Mean S_R2P:      {result.mean_s_r2p:.3f}  (room predicts patient)")
    print(f"  Mean S_P2R:      {result.mean_s_p2r:.3f}  (patient predicts room)")
    print(f"  Exploration rate: {result.exploration_trace.mean():.0%}")
    print()


def _print_summary(cohort):
    from world.spaces import ALL_PHENOTYPES
    from simulation.cohort import CohortResults

    print(f"\n{'─'*70}")
    print("  RESULTS SUMMARY")
    print(f"{'─'*70}")
    print(f"\n  {'Phenotype':<30} {'Del%':>5} {'SI':>6} "
          f"{'S_R2P':>7} {'S_P2R':>7} {'Learn':>7} {'Explore':>8}")
    print(f"  {'-'*75}")

    for ph in ALL_PHENOTYPES:
        rs     = cohort.by_phenotype(ph.name)
        dr     = cohort.delirium_rate(ph.name)
        ms     = cohort.mean_sync(ph.name)
        mr2p   = cohort.mean_s_r2p(ph.name)
        mp2r   = cohort.mean_s_p2r(ph.name)
        ml     = cohort.mean_learning(ph.name)
        er     = cohort.exploration_rate(ph.name)
        print(f"  {ph.name:<30} {dr:>5.0%} {ms:>6.2f} "
              f"{mr2p:>7.2f} {mp2r:>7.2f} {ml:>7.2f} {er:>8.0%}")

    # Correlation: sync index vs P(delirium)
    all_sync = [r.mean_sync for r in cohort.results]
    all_pdel = [float(r.p_delirium_trace[-1]) for r in cohort.results]
    all_desync = [r.mean_desync for r in cohort.results]
    if len(all_sync) > 2:
        r_sync  = float(np.corrcoef(all_sync,   all_pdel)[0, 1])
        r_desync = float(np.corrcoef(all_desync, all_pdel)[0, 1])
        print(f"\n  Correlation SI    ↔ P(delirium): r = {r_sync:.3f}")
        print(f"  Correlation desync ↔ P(delirium): r = {r_desync:.3f}")
        if r_sync < -0.3:
            print("  ✓ Hypothesis supported: lower sync → higher P(delirium)")

    overall_del = cohort.delirium_rate()
    overall_sync = cohort.mean_sync()
    print(f"\n  Overall delirium rate : {overall_del:.0%}")
    print(f"  Overall mean sync     : {overall_sync:.2f}")


def _generate_figures(cohort):
    from analysis.plots import (
        plot_sync_traces, plot_delirium_vs_sync,
        plot_vfe_traces, plot_exploration, plot_dashboard, plot_learning,
    )

    figs = [
        ("dashboard",         plot_dashboard),
        ("sync_traces",       plot_sync_traces),
        ("delirium_vs_sync",  plot_delirium_vs_sync),
        ("vfe_traces",        plot_vfe_traces),
        ("exploration",       plot_exploration),
        ("learning",          plot_learning),
    ]

    for name, fn in figs:
        path = os.path.join(FIG_DIR, f"{name}.png")
        try:
            fn(cohort, save_path=path)
            kb = os.path.getsize(path) // 1024
            print(f"  ✓ {name:<25} → {path}  ({kb} KB)")
        except Exception as e:
            print(f"  ✗ {name}: {e}")


if __name__ == "__main__":
    main()
