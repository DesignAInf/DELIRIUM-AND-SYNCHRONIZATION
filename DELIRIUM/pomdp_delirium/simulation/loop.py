"""
simulation/loop.py
==================
Bidirectional coupling loop between PatientAgent and RoomAgent.

One simulation cycle (T_CYCLE seconds, default 60s):
    1. Room selects action a_R (light/sound configuration)
    2. Patient receives observation o_P = f(true_state, a_R)
    3. Patient updates belief Q_P via VFE minimisation
    4. Patient selects micro-action a_P
    5. True patient state transitions: s' ~ B_R(s'|s, a_R) [room drives]
                                    and s' ~ B_P(s'|s, a_P) [patient drives]
    6. Room receives physiological observation o_R = f(true_state, a_P)
    7. Room updates belief Q_R and computes synchronization metrics
    8. Record all quantities

Delirium detection:
    Delirium onset is declared when P(delirium | Q_P) > DELIRIUM_THRESHOLD
    for >= DELIRIUM_SUSTAIN_CYCLES consecutive cycles.
    This is purely inferential — no threshold on a continuous variable.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

from world.spaces import (
    N_STATES, DELIRIUM_STATES, HEALTHY_STATES,
    index_to_state, state_label, room_action_label,
    Phenotype,
)
from world.observations import (
    sample_patient_obs, sample_room_obs, sample_true_next_state,
)
from agents.agents import PatientAgent, RoomAgent
from inference.core import (
    sync_room_to_patient, sync_patient_to_room,
    sync_patient_to_room_true,
    synchronization_index, bidirectional_desync,
)
from inference.causal_feedback import feedback_step

# Delirium criteria — predictive synchronization definition:
#
#   An agent is synchronized with another when it can predict the
#   other's states. Delirium emerges when BOTH directions fail:
#
#   S_{R->P} = -ln Q_R(true_state)   room fails to predict patient
#   S_{P->R} = VFE_P                 patient fails to predict room
#
# Thresholds (in nats):
S_R2P_THRESHOLD      = 3.0   # room surprisal: room cannot predict patient
S_P2R_THRESHOLD      = 3.0   # patient VFE: patient cannot predict room
DELIRIUM_SUSTAIN_CYCLES = 4  # both must hold for ~4 consecutive cycles


# ---------------------------------------------------------------------------
# Single simulation result
# ---------------------------------------------------------------------------

@dataclass
class SimulationResult:
    """Complete record of one patient × room simulation."""
    phenotype:          str
    n_cycles:           int
    T_cycle:            int

    # Delirium outcome
    delirium_onset_cycle: Optional[int]   # cycle of onset, or None
    delirium_declared:    bool

    # Synchronization traces (one value per cycle)
    # S_{R->P} = -ln Q_R(true_state): room's surprisal predicting patient
    # S_{P->R} = VFE_P:               patient's surprisal predicting room
    s_r2p_trace:    np.ndarray   # room predicts patient (lower = better)
    s_p2r_trace:    np.ndarray   # patient surprisal at room obs (TRUE, symmetric)
    s_p2r_proxy_trace: np.ndarray  # patient VFE proxy (deprecated, kept for comparison)
    sync_trace:     np.ndarray   # composite sync index [0,1]
    desync_trace:   np.ndarray   # bidirectional de-sync (s_r2p + s_p2r)

    # Free energy traces
    vfe_patient_trace: np.ndarray
    vfe_room_trace:    np.ndarray

    # Belief traces (expected cog and emo under Q_P)
    expected_cog_trace: np.ndarray
    expected_emo_trace: np.ndarray

    # Room behavior
    room_action_trace:   np.ndarray   # int, index into room actions
    exploration_trace:   np.ndarray   # bool, whether room was in exploration mode

    # Patient behavior
    patient_action_trace: np.ndarray
    p_delirium_trace:     np.ndarray   # P(delirium) under Q_P per cycle
    p_delirium_room_trace: np.ndarray  # room's inferred P(delirium) per cycle

    # True state trace (ground truth for analysis)
    true_state_trace:    np.ndarray

    # Desynchronization pressure trace (causal feedback)
    desync_pressure_trace: np.ndarray   # pressure applied each cycle [0,1]

    @property
    def mean_learning(self) -> float:
        """Mean θ learning progress of the room agent."""
        return getattr(self, '_mean_learning', 0.0)

    @property
    def mean_sync(self) -> float:
        return float(np.mean(self.sync_trace))

    @property
    def mean_desync(self) -> float:
        return float(np.mean(self.desync_trace))

    @property
    def mean_s_r2p(self) -> float:
        return float(np.mean(self.s_r2p_trace))

    @property
    def mean_s_p2r(self) -> float:
        return float(np.mean(self.s_p2r_trace))

    @property
    def sync_at_onset(self) -> Optional[float]:
        if self.delirium_onset_cycle is None:
            return None
        onset  = self.delirium_onset_cycle
        window = max(0, onset - 3)
        return float(np.mean(self.sync_trace[window:onset+1]))


# ---------------------------------------------------------------------------
# Main simulation loop
# ---------------------------------------------------------------------------

def run_simulation(phenotype:    Phenotype,
                   patient:      PatientAgent,
                   room:         RoomAgent,
                   B_P_true:     np.ndarray,   # true patient dynamics
                   B_R_true:     np.ndarray,   # true room-driven dynamics
                   n_cycles:     int = 30,
                   T_cycle:      int = 60,
                   rng:          Optional[np.random.Generator] = None,
                   verbose:      bool = False,
                   # Model variant flags
                   use_theta_learning:  bool = True,
                   use_causal_feedback: bool = True,
                   single_agent:        bool = False,
                   ) -> SimulationResult:
    """
    Run the full bidirectional POMDP simulation.

    Model variant flags:
        use_theta_learning:  False → room skips Level-2 θ update each cycle
        use_causal_feedback: False → desynchronization pressure not applied
        single_agent:        True  → room selects actions uniformly at random
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # Sample initial true state from phenotype prior
    D_true = _build_true_initial_state(phenotype)
    true_state = int(rng.choice(N_STATES, p=D_true))

    # Pre-allocate trace arrays
    s_r2p_trace         = np.zeros(n_cycles)
    s_p2r_trace         = np.zeros(n_cycles)
    s_p2r_proxy_trace   = np.zeros(n_cycles)
    sync_trace          = np.zeros(n_cycles)
    desync_trace        = np.zeros(n_cycles)
    vfe_patient_trace   = np.zeros(n_cycles)
    vfe_room_trace      = np.zeros(n_cycles)
    expected_cog_trace  = np.zeros(n_cycles)
    expected_emo_trace  = np.zeros(n_cycles)
    room_action_trace   = np.zeros(n_cycles, dtype=int)
    exploration_trace   = np.zeros(n_cycles, dtype=bool)
    patient_action_trace = np.zeros(n_cycles, dtype=int)
    p_delirium_trace    = np.zeros(n_cycles)
    p_delirium_room_trace = np.zeros(n_cycles)
    true_state_trace    = np.zeros(n_cycles, dtype=int)
    desync_pressure_trace = np.zeros(n_cycles)   # causal feedback pressure

    delirium_onset_cycle = None
    delirium_counter     = 0
    delirium_declared    = False
    s_r2p_history: List[float] = []   # tracks recent S_R->P for causal feedback

    for t in range(n_cycles):
        # ── 1. Room selects action ────────────────────────────────────────
        # M0 (single_agent): room picks uniformly at random — no inference
        if single_agent:
            room_action = int(rng.integers(0, room.n_actions))
        else:
            room_action = room.select_action(rng, temperature=1.0)

        # ── 2. Patient receives observation from room ─────────────────────
        obs_p = sample_patient_obs(true_state, room_action, patient.A_P, rng)

        # ── 3. Patient updates belief (VFE minimisation) ──────────────────
        vfe_p = patient.observe_and_update(obs_p)

        # ── 4. Patient selects micro-action ───────────────────────────────
        patient_action = patient.select_action(rng, temperature=1.2)

        # ── 5. True state transitions ─────────────────────────────────────
        # M2 (no feedback): causal feedback disabled.
        # M3 (full model):  feedback_step sharpens patient's prior belief
        #                   proportionally to recent S_{R->P} history.
        #                   This implements precision rigidity: the patient
        #                   becomes hypercertain about its current state,
        #                   discounting incoming observations.
        if use_causal_feedback:
            pressure = feedback_step(patient, s_r2p_history)
        else:
            pressure = 0.0

        # Room-driven transition (dominant)
        true_state = sample_true_next_state(true_state, B_R_true, room_action, rng)
        # Patient-driven transition (B_P_true unchanged — only prior sharpened)
        true_state = sample_true_next_state(true_state, B_P_true, patient_action, rng)

        # Propagate agent beliefs forward
        patient.step_belief(patient_action)
        room.step_belief(room_action)

        # ── 6. Room receives physiological observation ────────────────────
        obs_r = sample_room_obs(true_state, patient_action, room.A_R, rng)

        # ── 7. Room updates belief + synchronization metrics ──────────────
        # M0 (single_agent):       room does not update beliefs at all
        # M1 (no theta learning):  room updates state belief but skips Level-2
        # M2/M3:                   full update including theta inference
        if single_agent:
            vfe_r = 0.0
        else:
            vfe_r = room.observe_and_update(
                obs_r,
                true_state=true_state,
                vfe_p=vfe_p,
                update_theta=use_theta_learning,
            )

        # ── 8. Compute synchronization (predictive definition) ────────────
        # S_{R->P}: room surprisal at patient true state (genuine surprisal)
        # S_{P->R}: patient surprisal at room observation
        #   - TRUE version (symmetric):  -ln P(o_R | Q_P)
        #   - PROXY version (deprecated): VFE_P (conflates accuracy+complexity)
        # Delirium detection and SI use the TRUE symmetric metric.
        # The proxy is retained for backward compatibility and comparison.
        s_r2p      = sync_room_to_patient(room.qs, true_state)
        s_p2r_true = sync_patient_to_room_true(patient.qs, room.A_R, obs_r)
        s_p2r_proxy= sync_patient_to_room(vfe_p)          # deprecated proxy
        s_p2r      = s_p2r_true                            # use true for detection
        desync = bidirectional_desync(s_r2p, s_p2r)
        si     = synchronization_index(s_r2p, s_p2r)

        # Update S_R->P history for causal feedback next cycle
        s_r2p_history.append(s_r2p)

        # ── 9. Delirium detection (predictive synchronization criterion) ──
        # DELIRIUM when BOTH agents fail to predict each other:
        #   room cannot predict patient (S_{R->P} > threshold)
        #   patient cannot predict room (S_{P->R} > threshold)
        room_fails    = s_r2p > S_R2P_THRESHOLD
        patient_fails = s_p2r > S_P2R_THRESHOLD

        if room_fails and patient_fails:
            delirium_counter += 1
            if (delirium_counter >= DELIRIUM_SUSTAIN_CYCLES
                    and delirium_onset_cycle is None):
                delirium_onset_cycle = t - DELIRIUM_SUSTAIN_CYCLES + 1
                delirium_declared    = True
        else:
            delirium_counter = max(0, delirium_counter - 1)

        # Continuous P(delirium) proxy: normalized bidirectional desync
        p_del = float(min(1.0, desync / (S_R2P_THRESHOLD + S_P2R_THRESHOLD)))

        # ── 10. Record ────────────────────────────────────────────────────
        s_r2p_trace[t]          = s_r2p
        s_p2r_trace[t]          = s_p2r
        s_p2r_proxy_trace[t]    = s_p2r_proxy
        sync_trace[t]           = si
        desync_trace[t]         = desync
        vfe_patient_trace[t]    = vfe_p
        vfe_room_trace[t]       = vfe_r
        expected_cog_trace[t]   = patient.expected_cog
        expected_emo_trace[t]   = patient.expected_emo
        room_action_trace[t]    = room_action
        exploration_trace[t]    = room.exploration_mode
        patient_action_trace[t] = patient_action
        p_delirium_trace[t]     = p_del
        p_delirium_room_trace[t]= room.p_delirium_inferred
        true_state_trace[t]     = true_state
        desync_pressure_trace[t]= pressure

        if verbose:
            cog, emo, circ = index_to_state(true_state)
            print(f"  t={t:3d} | {room_action_label(room_action)[:38]:<38} | "
                  f"cog={cog} emo={emo} | "
                  f"S_R2P={s_r2p:.2f} S_P2R={s_p2r:.2f} "
                  f"SI={si:.2f} "
                  f"{'[EXPLORE]' if room.exploration_mode else '         '}"
                  f"{'[DELIRIUM]' if delirium_declared else ''}")

        if delirium_declared:
            for tt in range(t+1, n_cycles):
                s_r2p_trace[tt]          = s_r2p_trace[t]
                s_p2r_trace[tt]          = s_p2r_trace[t]
                s_p2r_proxy_trace[tt]    = s_p2r_proxy_trace[t]
                sync_trace[tt]           = sync_trace[t]
                desync_trace[tt]         = desync_trace[t]
                vfe_patient_trace[tt]    = vfe_patient_trace[t]
                vfe_room_trace[tt]       = vfe_room_trace[t]
                expected_cog_trace[tt]   = expected_cog_trace[t]
                expected_emo_trace[tt]   = expected_emo_trace[t]
                room_action_trace[tt]    = room_action_trace[t]
                exploration_trace[tt]    = exploration_trace[t]
                patient_action_trace[tt] = patient_action_trace[t]
                p_delirium_trace[tt]     = p_delirium_trace[t]
                p_delirium_room_trace[tt]= p_delirium_room_trace[t]
                true_state_trace[tt]     = true_state_trace[t]
                desync_pressure_trace[tt]= desync_pressure_trace[t]
            break

    result = SimulationResult(
        phenotype=phenotype.name,
        n_cycles=n_cycles,
        T_cycle=T_cycle,
        delirium_onset_cycle=delirium_onset_cycle,
        delirium_declared=delirium_declared,
        s_r2p_trace=s_r2p_trace,
        s_p2r_trace=s_p2r_trace,
        s_p2r_proxy_trace=s_p2r_proxy_trace,
        sync_trace=sync_trace,
        desync_trace=desync_trace,
        vfe_patient_trace=vfe_patient_trace,
        vfe_room_trace=vfe_room_trace,
        expected_cog_trace=expected_cog_trace,
        expected_emo_trace=expected_emo_trace,
        room_action_trace=room_action_trace,
        exploration_trace=exploration_trace,
        patient_action_trace=patient_action_trace,
        p_delirium_trace=p_delirium_trace,
        p_delirium_room_trace=p_delirium_room_trace,
        true_state_trace=true_state_trace,
        desync_pressure_trace=desync_pressure_trace,
    )
    result._mean_learning = room.mean_learning
    tc_final, te_final = room.theta_belief.map_theta
    result._theta_cog_final = tc_final
    result._theta_emo_final = te_final
    return result


def _build_true_initial_state(phenotype: Phenotype) -> np.ndarray:
    """Sample initial true state from phenotype priors."""
    from world.spaces import N_STATES, N_COG, N_EMO, N_CIRC, state_index
    D = np.zeros(N_STATES)
    for cog in range(N_COG):
        for emo in range(N_EMO):
            for circ in range(N_CIRC):
                s = state_index(cog, emo, circ)
                D[s] = (phenotype.cog_prior[cog] *
                        phenotype.emo_prior[emo] *
                        phenotype.circ_prior[circ])
    D /= D.sum()
    return D
