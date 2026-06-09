"""
agents/patient.py  &  agents/room.py
=====================================
Two POMDP agents that couple bidirectionally.

PatientAgent:
    - Maintains belief Q_P(s_P) over its own hidden states
    - Receives observations from the room (light, sound)
    - Selects micro-actions (passive, orient, startle)
    - Objective: minimise own VFE

RoomAgent:
    - Maintains belief Q_R(s_R) about the patient's hidden state
    - Receives physiological observations from sensors
    - Selects room configurations (light × sound)
    - Objective: minimise own EFE with preferences over patient states
    - Meta-level: monitors VFE_R as early-warning signal and switches
      to high-epistemic-value policy when de-synchronization detected
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List

from world.spaces import (
    N_STATES, N_OBS_PATIENT, N_OBS_ROOM,
    N_ACTIONS_PATIENT, N_ACTIONS_ROOM,
    DELIRIUM_STATES, NEAR_DELIRIUM_STATES, HEALTHY_STATES,
    index_to_state, state_label, room_action_label,
    Phenotype,
)
from inference.core import (
    belief_update, predict_next_state,
    compute_vfe, compute_efe, select_action,
    sync_room_to_patient, sync_patient_to_room,
    synchronization_index, bidirectional_desync,
)

EPS = 1e-16


# ---------------------------------------------------------------------------
# Patient Agent
# ---------------------------------------------------------------------------

@dataclass
class PatientAgent:
    """
    Active inference agent representing the ICU patient.
    Maintains a belief over its own (cog, emo, circ) state and
    selects micro-actions to minimise VFE.
    """
    phenotype:    Phenotype
    A_P:          np.ndarray   # (N_OBS_PATIENT, N_STATES)
    B_P:          np.ndarray   # (N_STATES, N_STATES, N_ACTIONS_PATIENT)
    C_P:          np.ndarray   # (N_OBS_PATIENT,)
    qs:           np.ndarray   # current belief Q_P(s), shape (N_STATES,)

    # History for analysis
    vfe_history:  List[float]  = field(default_factory=list)
    action_history: List[int]  = field(default_factory=list)
    obs_history:  List[int]    = field(default_factory=list)
    qs_history:   List[np.ndarray] = field(default_factory=list)

    def observe_and_update(self, obs: int) -> float:
        """
        Receive observation from room, update belief Q_P(s).
        Returns VFE of this observation.
        """
        qs_prior = self.qs.copy()
        self.qs  = belief_update(obs, self.A_P, qs_prior)
        vfe      = compute_vfe(obs, self.qs, self.A_P, qs_prior)

        self.obs_history.append(obs)
        self.vfe_history.append(vfe)
        self.qs_history.append(self.qs.copy())
        return vfe

    def select_action(self, rng: np.random.Generator,
                      temperature: float = 1.0) -> int:
        """
        Select micro-action by minimising EFE.
        Intubated patients have high probability of passive action;
        startle is triggered stochastically by phenotype.
        """
        # Stochastic startle — phenotype-dependent
        if rng.random() < self.phenotype.startle_prob:
            action = 2   # startle
        else:
            action = select_action(
                self.A_P, self.B_P, self.qs, self.C_P,
                N_ACTIONS_PATIENT, temperature=temperature, rng=rng,
                use_epistemic=True,
            )

        self.action_history.append(action)
        return action

    def step_belief(self, action: int) -> None:
        """Propagate belief forward using B_P after action selection."""
        self.qs = predict_next_state(self.qs, self.B_P, action)

    def sharpen_prior(self, pressure: float, strength: float = 0.10) -> None:
        """
        Increase precision of prior beliefs proportional to
        de-synchronization pressure.

        This implements the correct causal mechanism for precision rigidity:
        when the environment persistently fails to predict the patient,
        the patient's prior beliefs become increasingly peaked — not flat.
        A peaked prior means the patient is hypercertain about its current
        state and discounts incoming observations, regardless of their content.

        Mechanism:
            Q_prior(s) ← Q_prior(s)^(1 + pressure * strength)
            then renormalized.

        Raising a distribution to a power > 1 sharpens it: high-probability
        states become relatively more probable, low-probability states become
        relatively less probable. As pressure → 1, the prior approaches a
        delta function on the MAP state, making Bayesian updating impossible.

        Args:
            pressure: de-synchronization pressure ρ(t) ∈ [0, 1]
            strength: exponent scaling per unit pressure (default 0.10)
        """
        if pressure < 1e-6:
            return
        exponent = 1.0 + pressure * strength
        sharpened = np.power(self.qs + 1e-16, exponent)
        total = sharpened.sum()
        if total > 1e-16:
            self.qs = sharpened / total
        # Track prior precision as entropy of current belief
        # (lower entropy = higher precision = more rigidity)

    # --- Derived quantities ---

    @property
    def p_delirium(self) -> float:
        """Probability mass on delirium states."""
        return float(sum(self.qs[s] for s in DELIRIUM_STATES))

    @property
    def p_healthy(self) -> float:
        """Probability mass on healthy states."""
        return float(sum(self.qs[s] for s in HEALTHY_STATES))

    @property
    def map_state(self) -> int:
        """Maximum a posteriori state."""
        return int(np.argmax(self.qs))

    @property
    def mean_vfe(self) -> float:
        return float(np.mean(self.vfe_history)) if self.vfe_history else 0.0

    @property
    def expected_cog(self) -> float:
        """Expected cog level (0–3) under current belief."""
        cog_vals = np.array([index_to_state(s)[0] for s in range(N_STATES)])
        return float(self.qs @ cog_vals)

    @property
    def expected_emo(self) -> float:
        """Expected emo level (0–3) under current belief."""
        emo_vals = np.array([index_to_state(s)[1] for s in range(N_STATES)])
        return float(self.qs @ emo_vals)


# ---------------------------------------------------------------------------
# Room Agent — Level 3: Hierarchical active inference
# ---------------------------------------------------------------------------

# S_R2P threshold: above this, room fails to predict patient → exploration
S_R2P_ALARM = 5.0   # raised from 3.0 to avoid chronic exploration mode


class RoomAgent:
    """
    Level-3 hierarchical active inference agent for the ICU room.

    Two-level inference:

        Level 2 (slow):  Q_R(θ) over latent patient parameters
                         θ = (θ_cog, θ_emo) — cognitive plasticity
                         and emotional reactivity.
                         Updated via Bayes' rule every cycle using
                         observed state transitions.

        Level 1 (fast):  Q_R(s) over current patient state.
                         Updated every cycle using A_R and the
                         θ-marginalised effective B_R(θ).

    The room's generative model improves over time as it accumulates
    evidence about θ. Early cycles: high uncertainty, generic actions.
    Later cycles: sharp θ estimate, personalised B_R, better sync.

    Synchronization metrics:
        S_{R->P} = -ln Q_R(true_state)    room predicts patient state
        S_{P->R} = VFE_P                   patient predicts room observations
    """

    def __init__(self,
                 A_R: np.ndarray,
                 B_R: np.ndarray,
                 C_R: np.ndarray,
                 qs:  np.ndarray):
        # Core generative model
        self.A_R   = A_R
        self.B_R   = B_R        # reference matrix (θ = 0.5, 0.5)
        self.C_R   = C_R
        self.qs    = qs.copy()  # Level-1: Q_R(s)

        # Level-2: θ belief state
        from inference.hierarchical import ThetaBeliefState
        self.theta_belief = ThetaBeliefState()

        # Effective B_R (θ-marginalised), recomputed after θ update
        self._B_R_eff: np.ndarray = B_R.copy()
        self._prev_map_state: int = int(np.argmax(qs))

        # n_actions for single_agent random selection
        self.n_actions: int = B_R.shape[2]

        # Meta-level
        self.vfe_alarm_count:  int  = 0
        self.exploration_mode: bool = False

        # History
        self.vfe_history:       List[float] = []
        self.action_history:    List[int]   = []
        self.obs_history:       List[int]   = []
        self.sync_history:      List[float] = []
        self.s_r2p_history:     List[float] = []
        self.s_p2r_history:     List[float] = []
        self.desync_history:    List[float] = []
        self.exploration_flags: List[bool]  = []
        self.theta_cog_history: List[float] = []
        self.theta_emo_history: List[float] = []
        self.theta_entropy_history: List[float] = []
        self.learning_history:  List[float] = []

    def observe_and_update(self,
                           obs:          int,
                           true_state:   Optional[int]   = None,
                           vfe_p:        Optional[float] = None,
                           update_theta: bool            = True) -> float:
        """
        Full Level-3 update cycle:
            1. Update Level-1 Q_R(s) from observation
            2. Update Level-2 Q_R(θ) from observed state transition
               (skipped when update_theta=False → M1 ablation)
            3. Recompute effective B_R = Σ_θ Q_R(θ) B_R(θ)
            4. Compute synchronization metrics
            5. Update exploration mode

        Returns VFE_R (Level-1 free energy).
        """
        from inference.hierarchical import clear_cache

        # ── Level 1: state inference ──────────────────────────────────────
        qs_prior = self.qs.copy()
        self.qs  = belief_update(obs, self.A_R, qs_prior)
        vfe      = compute_vfe(obs, self.qs, self.A_R, qs_prior)

        self.obs_history.append(obs)
        self.vfe_history.append(vfe)

        # ── Level 2: θ inference ──────────────────────────────────────────
        # Skipped when update_theta=False (M1 ablation: no Level-3 learning)
        curr_map = int(np.argmax(self.qs))

        if update_theta:
            # Reliability weight: lower when Q_R(s) is very uncertain
            qs_entropy = -float(np.sum(self.qs * np.log(self.qs + EPS)))
            max_entropy = np.log(len(self.qs))
            obs_weight  = 1.0 - min(qs_entropy / max_entropy, 0.9)

            self.theta_belief.update(
                s_prev    = self._prev_map_state,
                s_curr    = curr_map,
                action    = self.action_history[-1] if self.action_history else 0,
                B_R_base  = self.B_R,
                obs_weight= obs_weight,
            )
            # Recompute effective B_R with updated θ belief
            self._B_R_eff = self.theta_belief.effective_B_R(self.B_R)

        self._prev_map_state = curr_map

        # Record θ estimates
        tc, te = self.theta_belief.mean_theta
        self.theta_cog_history.append(tc)
        self.theta_emo_history.append(te)
        self.theta_entropy_history.append(self.theta_belief.entropy)
        self.learning_history.append(self.theta_belief.learning_progress)

        # ── Synchronization metrics ───────────────────────────────────────
        if true_state is not None and vfe_p is not None:
            s_r2p  = sync_room_to_patient(self.qs, true_state)
            s_p2r  = sync_patient_to_room(vfe_p)
            desync = bidirectional_desync(s_r2p, s_p2r)
            si     = synchronization_index(s_r2p, s_p2r)

            self.s_r2p_history.append(s_r2p)
            self.s_p2r_history.append(s_p2r)
            self.desync_history.append(desync)
            self.sync_history.append(si)

            # Exploration: triggered when room fails to predict patient
            # AND has already learned enough to distinguish (learning > 0.1)
            if s_r2p > S_R2P_ALARM and self.theta_belief.learning_progress > 0.1:
                self.vfe_alarm_count += 1
            else:
                self.vfe_alarm_count = max(0, self.vfe_alarm_count - 1)
        else:
            if vfe > S_R2P_ALARM:
                self.vfe_alarm_count += 1
            else:
                self.vfe_alarm_count = max(0, self.vfe_alarm_count - 1)

        self.exploration_mode = self.vfe_alarm_count >= 3
        self.exploration_flags.append(self.exploration_mode)
        return vfe

    def select_action(self, rng: np.random.Generator,
                      temperature: float = 1.0) -> int:
        """
        Select room action using the θ-personalised effective B_R.
        Early cycles (low learning): higher temperature → more exploration.
        Later cycles (high learning): lower temperature → more exploitation.
        """
        # Adaptive temperature: higher when θ is uncertain
        learning = self.theta_belief.learning_progress
        adaptive_temp = temperature * (2.0 - learning)  # 2x at start, 1x when learned

        B_for_action = self._B_R_eff

        if self.exploration_mode:
            action = select_action(
                self.A_R, B_for_action, self.qs, self.C_R,
                N_ACTIONS_ROOM, temperature=adaptive_temp * 1.5,
                use_epistemic=True, rng=rng,
            )
        else:
            action = select_action(
                self.A_R, B_for_action, self.qs, self.C_R,
                N_ACTIONS_ROOM, temperature=adaptive_temp, rng=rng,
                use_epistemic=True,
            )

        self.action_history.append(action)
        return action

    def step_belief(self, action: int) -> None:
        """Propagate belief using θ-personalised effective B_R."""
        self.qs = predict_next_state(self.qs, self._B_R_eff, action)

    # --- Derived quantities ---

    @property
    def p_delirium_inferred(self) -> float:
        return float(sum(self.qs[s] for s in DELIRIUM_STATES))

    @property
    def mean_sync(self) -> float:
        return float(np.mean(self.sync_history)) if self.sync_history else 0.5

    @property
    def mean_s_r2p(self) -> float:
        return float(np.mean(self.s_r2p_history)) if self.s_r2p_history else 0.0

    @property
    def mean_s_p2r(self) -> float:
        return float(np.mean(self.s_p2r_history)) if self.s_p2r_history else 0.0

    @property
    def mean_desync(self) -> float:
        return float(np.mean(self.desync_history)) if self.desync_history else 0.0

    @property
    def mean_vfe(self) -> float:
        return float(np.mean(self.vfe_history)) if self.vfe_history else 0.0

    @property
    def mean_learning(self) -> float:
        """Mean learning progress (0=no learning, 1=fully identified θ)."""
        return float(np.mean(self.learning_history)) if self.learning_history else 0.0

    @property
    def final_theta(self):
        """Final MAP estimate of (θ_cog, θ_emo)."""
        return self.theta_belief.map_theta
