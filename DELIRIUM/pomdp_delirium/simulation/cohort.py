"""
simulation/cohort.py
====================
Runs a full cohort of virtual patients across all phenotypes,
collects results, and computes group-level statistics on
synchronization, delirium onset, and room adaptation.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import time

from world.spaces import ALL_PHENOTYPES, Phenotype, N_STATES
from world.generative_models import (
    build_A_patient, build_B_patient,
    build_A_room,    build_B_room,
    build_C_patient, build_C_room,
    build_D_patient, build_D_room,
)
from agents.agents import PatientAgent, RoomAgent
from simulation.loop import run_simulation, SimulationResult


# ---------------------------------------------------------------------------
# Build agents for a given phenotype
# ---------------------------------------------------------------------------

def build_agents(phenotype: Phenotype,
                 seed: int = 0) -> tuple:
    """
    Construct PatientAgent and RoomAgent for a given phenotype.
    Returns (patient, room, B_P_true, B_R_true).
    """
    A_P = build_A_patient()
    B_P = build_B_patient(phenotype)
    C_P = build_C_patient()
    D_P = build_D_patient(phenotype)

    A_R = build_A_room()
    B_R = build_B_room(phenotype)
    C_R = build_C_room()
    D_R = build_D_room()

    patient = PatientAgent(
        phenotype=phenotype,
        A_P=A_P, B_P=B_P, C_P=C_P,
        qs=D_P.copy(),
    )
    room = RoomAgent(
        A_R=A_R, B_R=B_R, C_R=C_R,
        qs=D_R.copy(),
    )
    return patient, room, B_P, B_R
# Cohort simulation
# ---------------------------------------------------------------------------

@dataclass
class CohortResults:
    results:     List[SimulationResult]
    phenotypes:  List[str]

    def by_phenotype(self, name: str) -> List[SimulationResult]:
        return [r for r in self.results if r.phenotype == name]

    def delirium_rate(self, phenotype: Optional[str] = None) -> float:
        rs = self.by_phenotype(phenotype) if phenotype else self.results
        if not rs: return 0.0
        return float(np.mean([r.delirium_declared for r in rs]))

    def mean_sync(self, phenotype: Optional[str] = None) -> float:
        rs = self.by_phenotype(phenotype) if phenotype else self.results
        if not rs: return 0.0
        return float(np.mean([r.mean_sync for r in rs]))

    def mean_desync(self, phenotype: Optional[str] = None) -> float:
        rs = self.by_phenotype(phenotype) if phenotype else self.results
        if not rs: return 0.0
        return float(np.mean([r.mean_desync for r in rs]))

    def mean_s_r2p(self, phenotype: Optional[str] = None) -> float:
        """Mean room surprisal at patient's true state (lower = room predicts better)."""
        rs = self.by_phenotype(phenotype) if phenotype else self.results
        if not rs: return 0.0
        return float(np.mean([r.mean_s_r2p for r in rs]))

    def mean_s_p2r(self, phenotype: Optional[str] = None) -> float:
        """Mean patient VFE (lower = patient predicts room better)."""
        rs = self.by_phenotype(phenotype) if phenotype else self.results
        if not rs: return 0.0
        return float(np.mean([r.mean_s_p2r for r in rs]))

    def mean_learning(self, phenotype: Optional[str] = None) -> float:
        """Mean θ learning progress (0=no learning, 1=fully identified)."""
        rs = self.by_phenotype(phenotype) if phenotype else self.results
        if not rs: return 0.0
        return float(np.mean([r.mean_learning for r in rs]))

    def sync_trace_mean(self, phenotype: Optional[str] = None) -> np.ndarray:
        rs = self.by_phenotype(phenotype) if phenotype else self.results
        if not rs: return np.array([])
        return np.mean([r.sync_trace for r in rs], axis=0)

    def s_r2p_trace_mean(self, phenotype: Optional[str] = None) -> np.ndarray:
        rs = self.by_phenotype(phenotype) if phenotype else self.results
        if not rs: return np.array([])
        return np.mean([r.s_r2p_trace for r in rs], axis=0)

    def s_p2r_trace_mean(self, phenotype: Optional[str] = None) -> np.ndarray:
        rs = self.by_phenotype(phenotype) if phenotype else self.results
        if not rs: return np.array([])
        return np.mean([r.s_p2r_trace for r in rs], axis=0)

    def p_delirium_trace_mean(self, phenotype: Optional[str] = None) -> np.ndarray:
        rs = self.by_phenotype(phenotype) if phenotype else self.results
        if not rs: return np.array([])
        return np.mean([r.p_delirium_trace for r in rs], axis=0)

    def vfe_patient_trace_mean(self, phenotype: Optional[str] = None) -> np.ndarray:
        rs = self.by_phenotype(phenotype) if phenotype else self.results
        if not rs: return np.array([])
        return np.mean([r.vfe_patient_trace for r in rs], axis=0)

    def vfe_room_trace_mean(self, phenotype: Optional[str] = None) -> np.ndarray:
        rs = self.by_phenotype(phenotype) if phenotype else self.results
        if not rs: return np.array([])
        return np.mean([r.vfe_room_trace for r in rs], axis=0)

    def exploration_rate(self, phenotype: Optional[str] = None) -> float:
        rs = self.by_phenotype(phenotype) if phenotype else self.results
        if not rs: return 0.0
        return float(np.mean([r.exploration_trace.mean() for r in rs]))


def run_cohort(n_per_phenotype: int = 12,
               n_cycles: int = 30,
               T_cycle:  int = 60,
               seed:     int = 42,
               verbose:  bool = True,
               use_theta_learning:  bool = True,
               use_causal_feedback: bool = True,
               single_agent:        bool = False,
               ) -> CohortResults:
    """
    Simulate n_per_phenotype patients for each of the 4 phenotypes.

    Model variant flags (for ablation / model comparison):
        use_theta_learning:  False → room uses fixed B_R, no Level-3  [M1]
        use_causal_feedback: False → desync does not degrade gamma_cog [M2]
        single_agent:        True  → room acts randomly, no inference   [M0]
    """
    rng_master = np.random.default_rng(seed)
    all_results = []
    phenotype_names = []

    t0 = time.time()
    total = len(ALL_PHENOTYPES) * n_per_phenotype
    done  = 0

    for phenotype in ALL_PHENOTYPES:
        if verbose:
            print(f"\n  Phenotype: {phenotype.name} (n={n_per_phenotype})")

        for i in range(n_per_phenotype):
            patient_seed = int(rng_master.integers(0, 2**31))
            rng = np.random.default_rng(patient_seed)

            patient, room, B_P_true, B_R_true = build_agents(phenotype, seed=patient_seed)

            result = run_simulation(
                phenotype=phenotype,
                patient=patient,
                room=room,
                B_P_true=B_P_true,
                B_R_true=B_R_true,
                n_cycles=n_cycles,
                T_cycle=T_cycle,
                rng=rng,
                verbose=False,
                use_theta_learning=use_theta_learning,
                use_causal_feedback=use_causal_feedback,
                single_agent=single_agent,
            )
            all_results.append(result)
            phenotype_names.append(phenotype.name)
            done += 1

            if verbose:
                status = "DEL" if result.delirium_declared else "ok "
                print(f"    [{i+1:2d}/{n_per_phenotype}] {status} | "
                      f"SI={result.mean_sync:.2f} "
                      f"S_R2P={result.mean_s_r2p:.2f} "
                      f"S_P2R={result.mean_s_p2r:.2f} "
                      f"learn={result.mean_learning:.2f} "
                      f"P(del)={result.p_delirium_trace[-1]:.2f}")

    elapsed = time.time() - t0
    if verbose:
        print(f"\n  Completed {total} simulations in {elapsed:.1f}s")
        print(f"\n  {'Phenotype':<30} {'Del%':>5} {'SI':>6} {'S_R2P':>7} {'S_P2R':>7} {'Explore':>8}")
        print(f"  {'-'*65}")
        for ph in ALL_PHENOTYPES:
            cr = CohortResults(all_results, phenotype_names)
            print(f"  {ph.name:<30} "
                  f"{cr.delirium_rate(ph.name):>5.0%} "
                  f"{cr.mean_sync(ph.name):>6.2f} "
                  f"{cr.mean_s_r2p(ph.name):>7.2f} "
                  f"{cr.mean_s_p2r(ph.name):>7.2f} "
                  f"{cr.exploration_rate(ph.name):>8.0%}")

    return CohortResults(all_results, phenotype_names)
