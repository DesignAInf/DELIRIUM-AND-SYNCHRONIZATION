"""
inference/hierarchical.py
=========================
Level-3 hierarchical inference for the RoomAgent.

The room maintains a two-level generative model:

    Level 2 (slow):  θ = (θ_cog, θ_emo) ∈ [0,1]²
                     Latent patient parameters:
                       θ_cog: cognitive plasticity (how easily γ_cog improves)
                       θ_emo: emotional reactivity (how strongly γ_emo spikes)
                     Prior: uniform over 10×10 grid → Q_R(θ) ∈ R^100

    Level 1 (fast):  s_t ∈ {0,...,47}
                     Current patient state, inferred every cycle
                     using B_R(θ) = Σ_θ Q_R(θ) · B_R^(θ)

The room never observes θ directly. It accumulates evidence across
cycles and progressively sharpens Q_R(θ), building a patient-specific
generative model that improves synchronization over time.

Bayesian update for θ:
    Q_R(θ) ∝ P(o_R | θ, Q_R(s)) · Q_R(θ)_prior

where P(o_R | θ, Q_R(s)) is computed by marginalising over states:
    P(o_R | θ) = Σ_s A_R[o_R, s] · Q_R(s)

then reweighted by how consistent the observed state transitions are
with the dynamics implied by θ.
"""

import numpy as np
from typing import Tuple

EPS = 1e-16

# Grid resolution for θ space
N_THETA = 10                          # points per dimension
THETA_GRID = np.linspace(0.1, 1.0, N_THETA)   # avoid 0 (degenerate)
N_THETA_TOTAL = N_THETA * N_THETA     # 100 grid points


def theta_index(i: int, j: int) -> int:
    """Flat index from (i_cog, j_emo) grid coordinates."""
    return i * N_THETA + j


def index_to_theta(idx: int) -> Tuple[float, float]:
    """Recover (θ_cog, θ_emo) values from flat grid index."""
    i = idx // N_THETA
    j = idx  % N_THETA
    return float(THETA_GRID[i]), float(THETA_GRID[j])


def build_B_R_theta(B_R_base: np.ndarray,
                    theta_cog: float,
                    theta_emo: float) -> np.ndarray:
    """
    Construct a θ-personalised transition matrix B_R(θ).

    θ_cog modulates how strongly room actions improve cognitive states:
        high θ_cog → room actions are more effective at raising γ_cog
        low  θ_cog → room actions have little effect on γ_cog (rigid patient)

    θ_emo modulates emotional reactivity:
        high θ_emo → threatening sounds spike γ_emo more strongly
        low  θ_emo → patient is less emotionally reactive

    Implementation: interpolate between B_R_base (θ=0.5 reference)
    and two extreme versions (fully plastic / fully rigid).

    Args:
        B_R_base:  reference transition matrix (N_STATES, N_STATES, N_ACTIONS)
        theta_cog: cognitive plasticity ∈ [0.1, 1.0]
        theta_emo: emotional reactivity ∈ [0.1, 1.0]

    Returns:
        B_R_theta: personalised matrix, same shape as B_R_base
    """
    from world.spaces import (
        N_STATES, N_ACTIONS_ROOM, N_COG, N_EMO, N_CIRC,
        state_index, index_to_state, index_to_room_action,
    )

    B = np.zeros_like(B_R_base)

    for a in range(N_ACTIONS_ROOM):
        light, sound = index_to_room_action(a)

        # Determine base action effects (same logic as build_B_room,
        # but now scaled by θ instead of phenotype multipliers)
        cog_delta  = [0.0, +1.0, +0.5, -1.0][light]
        circ_delta = [0.0, +1.0, +0.5, -0.5][light]
        emo_delta  = [0.0, -1.0, -0.3, +1.5][sound]

        # θ personalisation: scale by inferred patient parameters
        cog_delta  *= theta_cog          # plastic patient benefits more
        emo_delta  *= theta_emo          # reactive patient is more affected

        for s in range(N_STATES):
            cog, emo, circ = index_to_state(s)
            col = np.zeros(N_STATES)
            col[s] = 0.50   # inertia

            # Cognitive transition
            if cog_delta > 0 and cog < 3:
                p = min(0.30 * abs(cog_delta), 0.35)
                col[state_index(min(cog+1, 3), emo, circ)] += p
            elif cog_delta < 0 and cog > 0:
                p = min(0.30 * abs(cog_delta), 0.35)
                col[state_index(max(cog-1, 0), emo, circ)] += p

            # Emotional transition
            if emo_delta < 0 and emo > 0:
                p = min(0.30 * abs(emo_delta), 0.40)
                col[state_index(cog, max(emo-1, 0), circ)] += p
            elif emo_delta > 0 and emo < 3:
                p = min(0.25 * abs(emo_delta), 0.45)
                col[state_index(cog, min(emo+1, 3), circ)] += p

            # Circadian
            if circ_delta > 0 and circ < 2:
                p = min(0.20 * abs(circ_delta), 0.25)
                col[state_index(cog, emo, min(circ+1, 2))] += p
            elif circ_delta < 0 and circ > 0:
                p = min(0.15 * abs(circ_delta), 0.20)
                col[state_index(cog, emo, max(circ-1, 0))] += p

            # Self-reinforcing attractors
            if cog == 0 and emo == 3:
                col[s] += 0.20
            elif cog >= 3 and emo == 0:
                col[s] += 0.10

            total = col.sum()
            col   = col / total if total > EPS else (np.zeros(N_STATES) + EPS)
            col[s] = max(col[s], EPS)
            col  /= col.sum()
            B[:, s, a] = col

    return B


# Pre-compute B_R for every θ grid point at import time.
# Stored as (N_THETA_TOTAL, N_STATES, N_STATES, N_ACTIONS_ROOM).
# This costs ~100 × 48 × 48 × 16 × 8 bytes ≈ 28 MB — acceptable.
_B_CACHE: dict = {}   # populated lazily on first use


def get_B_R_for_theta(idx: int, B_R_base: np.ndarray) -> np.ndarray:
    """Return cached B_R(θ) for grid point idx, computing if needed."""
    if idx not in _B_CACHE:
        tc, te = index_to_theta(idx)
        _B_CACHE[idx] = build_B_R_theta(B_R_base, tc, te)
    return _B_CACHE[idx]


def clear_cache() -> None:
    """Clear precomputed B_R cache (call between independent runs)."""
    _B_CACHE.clear()


class ThetaBeliefState:
    """
    Maintains Q_R(θ) — the room's belief over latent patient parameters.

    Updated every cycle via Bayes' rule:
        Q_R(θ) ∝ L(θ | o_R, s_prev, s_curr, a_R) · Q_R(θ)_prior

    where the likelihood measures how well the transition s_prev → s_curr
    under action a_R is explained by dynamics B_R(θ).

    Also tracks:
        - MAP estimate of θ over time
        - Entropy of Q_R(θ) (decreasing entropy = learning)
    """

    def __init__(self):
        # Uniform prior over θ grid
        self.q_theta = np.ones(N_THETA_TOTAL) / N_THETA_TOTAL
        self.theta_history:   list = []
        self.entropy_history: list = []

    def update(self,
               s_prev:    int,
               s_curr:    int,
               action:    int,
               B_R_base:  np.ndarray,
               obs_weight: float = 1.0) -> None:
        """
        Bayesian update of Q_R(θ) given observed state transition.

        The likelihood of each θ is the probability that B_R(θ)
        would produce the observed transition s_prev → s_curr under action:
            L(θ) = B_R(θ)[s_curr, s_prev, action]

        Args:
            s_prev:      state at previous cycle (room's MAP estimate)
            s_curr:      state at current cycle  (room's MAP estimate)
            action:      room action taken
            B_R_base:    reference B_R matrix
            obs_weight:  reliability weight for this observation (0-1)
        """
        log_likelihood = np.zeros(N_THETA_TOTAL)

        for idx in range(N_THETA_TOTAL):
            B_theta = get_B_R_for_theta(idx, B_R_base)
            # Transition probability under this θ
            p_trans = float(B_theta[s_curr, s_prev, action])
            log_likelihood[idx] = np.log(max(p_trans, EPS)) * obs_weight

        # Log-space Bayes update for numerical stability
        log_q = np.log(self.q_theta + EPS) + log_likelihood
        log_q -= log_q.max()
        self.q_theta  = np.exp(log_q)
        self.q_theta /= self.q_theta.sum()

        # Record MAP and entropy
        self.theta_history.append(self.map_theta)
        h = -float(np.sum(self.q_theta * np.log(self.q_theta + EPS)))
        self.entropy_history.append(h)

    def effective_B_R(self, B_R_base: np.ndarray) -> np.ndarray:
        """
        Compute the θ-marginalised effective transition matrix:
            B_R_eff = Σ_θ Q_R(θ) · B_R(θ)

        This is the matrix actually used for state inference —
        it embeds the room's current uncertainty about the patient type.
        """
        from world.spaces import N_STATES, N_ACTIONS_ROOM
        B_eff = np.zeros((N_STATES, N_STATES, N_ACTIONS_ROOM))
        for idx in range(N_THETA_TOTAL):
            if self.q_theta[idx] < EPS:
                continue
            B_eff += self.q_theta[idx] * get_B_R_for_theta(idx, B_R_base)
        return B_eff

    @property
    def map_theta(self) -> Tuple[float, float]:
        """Maximum a posteriori estimate of (θ_cog, θ_emo)."""
        best = int(np.argmax(self.q_theta))
        return index_to_theta(best)

    @property
    def mean_theta(self) -> Tuple[float, float]:
        """Posterior mean of (θ_cog, θ_emo)."""
        tc = sum(self.q_theta[idx] * index_to_theta(idx)[0]
                 for idx in range(N_THETA_TOTAL))
        te = sum(self.q_theta[idx] * index_to_theta(idx)[1]
                 for idx in range(N_THETA_TOTAL))
        return float(tc), float(te)

    @property
    def entropy(self) -> float:
        """Current entropy of Q_R(θ) in nats."""
        return float(self.entropy_history[-1]) if self.entropy_history else np.log(N_THETA_TOTAL)

    @property
    def learning_progress(self) -> float:
        """
        Normalised entropy reduction from initial to current:
            0.0 = no learning (still at max entropy)
            1.0 = fully identified θ (entropy = 0)
        """
        h_max = np.log(N_THETA_TOTAL)
        h_cur = self.entropy
        return float(max(0.0, 1.0 - h_cur / h_max))
