"""
inference/core.py
=================
Core active inference computations for both agents.

Functions:
    belief_update      -- VFE minimization via softmax (variational inference)
    compute_vfe        -- Variational Free Energy F = -accuracy + complexity
    compute_efe        -- Expected Free Energy G for policy selection
    select_action      -- Softmax policy selection from EFE values
    kl_divergence      -- KL(Q_P || Q_R) synchronization metric
    mutual_information -- MI between two belief distributions
"""

import numpy as np
from typing import List, Optional

EPS = 1e-16


# ---------------------------------------------------------------------------
# Belief updating (VFE minimization)
# ---------------------------------------------------------------------------

def belief_update(obs: int,
                  A: np.ndarray,
                  qs_prior: np.ndarray,
                  n_iter: int = 16) -> np.ndarray:
    """
    Update belief Q(s) given observation obs and prior Q(s).

    Implements variational message passing: iteratively updates Q(s)
    to minimise VFE = -E_Q[ln P(o|s)] + KL[Q(s)||prior].

    This is the standard active inference belief update, equivalent to
    a softmax on log-likelihood + log-prior (single-step approximation).

    Args:
        obs:      observed index (integer)
        A:        likelihood matrix P(o|s), shape (n_obs, n_states)
        qs_prior: prior belief Q(s), shape (n_states,)
        n_iter:   VMP iterations (16 is sufficient for these matrix sizes)

    Returns:
        qs_post: posterior belief Q(s), shape (n_states,)
    """
    # Log-likelihood of observation under each state
    ln_likelihood = np.log(A[obs] + EPS)           # shape (n_states,)
    ln_prior      = np.log(qs_prior + EPS)          # shape (n_states,)

    # Iterative VMP (converges fast for discrete POMDP)
    qs = qs_prior.copy()
    for _ in range(n_iter):
        ln_qs = ln_likelihood + ln_prior
        # Softmax normalization
        ln_qs -= ln_qs.max()
        qs     = np.exp(ln_qs)
        qs    /= qs.sum()

    return qs


def predict_next_state(qs: np.ndarray,
                       B: np.ndarray,
                       action: int) -> np.ndarray:
    """
    Predictive prior for next time step:
        Q(s') = sum_s B(s'|s,a) Q(s)

    Args:
        qs:     current belief, shape (n_states,)
        B:      transition matrix, shape (n_states, n_states, n_actions)
        action: chosen action index

    Returns:
        qs_pred: predicted belief at t+1, shape (n_states,)
    """
    qs_pred = B[:, :, action] @ qs
    qs_pred = np.clip(qs_pred, EPS, None)
    qs_pred /= qs_pred.sum()
    return qs_pred


# ---------------------------------------------------------------------------
# Free energy quantities
# ---------------------------------------------------------------------------

def compute_vfe(obs: int,
                qs: np.ndarray,
                A: np.ndarray,
                qs_prior: np.ndarray) -> float:
    """
    Variational Free Energy:
        F = -E_Q[ln P(o|s)]     (negative accuracy)
          + KL[Q(s) || Q_prior]  (complexity)

    Lower F = better fit between model and observation.
    High F = the agent is surprised — cannot explain the observation.

    Args:
        obs:      observed index
        qs:       posterior belief Q(s)
        A:        likelihood P(o|s)
        qs_prior: prior belief

    Returns:
        F: scalar free energy
    """
    # Accuracy: expected log-likelihood
    accuracy = float(qs @ np.log(A[obs] + EPS))

    # Complexity: KL divergence Q(s) || prior
    complexity = float(np.sum(qs * (np.log(qs + EPS) - np.log(qs_prior + EPS))))

    return -accuracy + complexity


def compute_efe(A: np.ndarray,
                B: np.ndarray,
                qs: np.ndarray,
                C: np.ndarray,
                action: int,
                use_epistemic: bool = True) -> float:
    """
    Expected Free Energy for a single action:
        G(a) = -E[ln P(o)] - MI(o; s | a)   [pragmatic + epistemic]

    Decomposes into:
        Pragmatic value:  E_Q(s')[E_Q(o|s')[C(o)]]   (preference satisfaction)
        Epistemic value:  E_Q(s')[H[P(o|s')]] - H[E_Q(s')[P(o|s')]]
                          (expected information gain about hidden states)

    Lower G = better action (more likely to be selected).

    Args:
        A:              likelihood P(o|s), shape (n_obs, n_states)
        B:              transition matrix, shape (n_states, n_states, n_actions)
        qs:             current belief Q(s), shape (n_states,)
        C:              log-preference vector, shape (n_obs,) or (n_states,)
        action:         action index to evaluate
        use_epistemic:  if True, include epistemic (information gain) term

    Returns:
        G: scalar EFE (lower = preferred)
    """
    # Predicted state at next step under this action
    qs_next = predict_next_state(qs, B, action)   # shape (n_states,)

    # Predicted observation distribution: P(o) = sum_s A(o|s) Q(s')
    po = A @ qs_next   # shape (n_obs,)
    po = np.clip(po, EPS, None)
    po /= po.sum()

    # Pragmatic value: expected preference satisfaction
    if C.shape[0] == A.shape[0]:   # C over observations
        pragmatic = float(po @ C)
    else:                           # C over states (room agent)
        pragmatic = float(qs_next @ C)

    # Epistemic value: expected information gain (Bayesian surprise reduction)
    epistemic = 0.0
    if use_epistemic:
        # H[P(o)] — entropy of predicted observation distribution
        H_po = -float(np.sum(po * np.log(po + EPS)))

        # E_Q(s')[H[P(o|s')]] — expected entropy of likelihood columns
        H_Aos = -np.sum(A * np.log(A + EPS), axis=0)   # shape (n_states,)
        E_H_Aos = float(qs_next @ H_Aos)

        # MI = H[P(o)] - E[H[P(o|s')]]
        epistemic = H_po - E_H_Aos

    # G = -(pragmatic + epistemic): lower is better
    return -(pragmatic + epistemic)


def select_action(A: np.ndarray,
                  B: np.ndarray,
                  qs: np.ndarray,
                  C: np.ndarray,
                  n_actions: int,
                  temperature: float = 1.0,
                  use_epistemic: bool = True,
                  rng: Optional[np.random.Generator] = None) -> int:
    """
    Select action by computing EFE for all actions and sampling
    from softmax distribution (stochastic policy).

    Args:
        temperature: controls exploration (higher = more uniform)
        rng:        random generator (if None, uses argmin — greedy)

    Returns:
        chosen action index
    """
    G = np.array([
        compute_efe(A, B, qs, C, a, use_epistemic=use_epistemic)
        for a in range(n_actions)
    ])

    if rng is None:
        return int(np.argmin(G))

    # Softmax policy: pi(a) ∝ exp(-G(a) / temperature)
    log_pi = -G / temperature
    log_pi -= log_pi.max()
    pi      = np.exp(log_pi)
    pi     /= pi.sum()

    return int(rng.choice(n_actions, p=pi))


# ---------------------------------------------------------------------------
# Synchronization metrics
# ---------------------------------------------------------------------------

def sync_room_to_patient(Q_R: np.ndarray, true_state: int) -> float:
    """
    Synchronization of room toward patient:
        S_{R->P} = -ln Q_R(true_state)

    Measures how well the room predicts the patient's true hidden state.
    Low surprisal = good prediction = room is synchronized with patient.
    High surprisal = room model has diverged from patient's true state.
    """
    p = float(np.clip(Q_R[true_state], EPS, None))
    return -np.log(p)


def sync_patient_to_room(vfe_p: float) -> float:
    """
    Synchronization of patient toward room (PROXY):
        S_{P->R} = VFE_P

    DEPRECATED proxy: conflates accuracy and complexity terms.
    Use sync_patient_to_room_true() for the commensurable version.
    """
    return vfe_p


def sync_patient_to_room_true(Q_P: np.ndarray,
                               A_R: np.ndarray,
                               obs_r: int) -> float:
    """
    True symmetric S_{P->R}: patient surprisal at room observation.

        S_{P->R}^{true} = -ln P(o_R | Q_P)
                        = -ln sum_s Q_P(s) * A_R[o_R, s]

    Commensurable with S_{R->P} = -ln Q_R(s*):
        Both are negative log-probabilities of an observation
        under a belief distribution.

    Args:
        Q_P:   patient belief over hidden states (n_states,)
        A_R:   room likelihood matrix P(o_R|s), shape (n_obs_room, n_states)
        obs_r: room observation index at this cycle

    Returns:
        S_{P->R}^{true} in nats (>= 0)
    """
    p_obs = float(np.dot(Q_P, A_R[obs_r]))
    p_obs = max(p_obs, EPS)
    return -np.log(p_obs)


def synchronization_index(s_r2p: float, s_p2r: float,
                           s_r2p_max: float = 6.0,
                           s_p2r_max: float = 6.0) -> float:
    """
    Composite synchronization index in [0, 1].

    Synchronization is defined as the capacity of each agent to predict
    the states of the other:
        - Room predicts patient: S_{R->P} = -ln Q_R(true_state)
        - Patient predicts room: S_{P->R} = VFE_P

    Both terms are surprisal measures: lower = better prediction = more sync.
    The index inverts and normalizes both so that 1.0 = perfect sync.

        SI = 0.5 * (1 - S_{R->P}/max) + 0.5 * (1 - S_{P->R}/max)
    """
    r2p_norm = 1.0 - min(s_r2p / s_r2p_max, 1.0)
    p2r_norm = 1.0 - min(s_p2r / s_p2r_max, 1.0)
    return 0.5 * r2p_norm + 0.5 * p2r_norm


def bidirectional_desync(s_r2p: float, s_p2r: float) -> float:
    """
    Raw bidirectional de-synchronization score (lower = more synchronized):
        D = S_{R->P} + S_{P->R}

    Used as primary delirium criterion input.
    """
    return s_r2p + s_p2r
