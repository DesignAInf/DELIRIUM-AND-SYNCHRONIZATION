"""
inference/causal_feedback.py
============================
Implements the causal feedback loop at the heart of the
predictive synchronization hypothesis:

    persistent de-synchronization → precision rigidity → delirium

Mechanism (v2 — correct implementation):
    When S_{R->P} is persistently high, the patient's prior beliefs
    become increasingly PEAKED (high precision), not flat.
    A peaked prior means the patient is hypercertain about its current
    state and discounts incoming observations — regardless of their content.
    This is precision rigidity in the correct active inference sense:
    the patient is not confused by noise, it is locked into a prior
    that no observation can update.

    At each cycle, desynchronization pressure ρ(t) ∈ [0,1] is computed
    from the recent history of S_{R->P}. This pressure is passed to
    PatientAgent.sharpen_prior(), which raises the current belief
    distribution to a power > 1, sharpening it proportionally.

    The sharpening is applied to the patient's POSTERIOR (which becomes
    the next cycle's prior via step_belief). The patient's generative
    model (A_P, B_P) is NOT modified — the patient's model of the world
    stays the same, but its confidence in its current state becomes so
    high that the model stops being updated by observations.

    This is the distinction from v1:
        v1: modified B_P_true → degraded γ_cog (wrong direction)
        v2: sharpens Q_P prior → precision rigidity (correct direction)

Parameters:
    DESYNC_WINDOW     — cycles of S_{R->P} history used for pressure (10)
    DESYNC_THRESHOLD  — S_{R->P} above which pressure accumulates (3.0 nats)
    FEEDBACK_STRENGTH — exponent increment per unit pressure (0.10)
                        At pressure=1.0: exponent = 1.10
                        At pressure=1.0 for 20 cycles: prior approaches spike

Author: Luca M. Possati
Version: 2.0.0
"""

import numpy as np
from typing import List

EPS = 1e-16

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

DESYNC_WINDOW     = 10    # cycles of S_R2P history
DESYNC_THRESHOLD  = 3.0   # nats — above this, pressure accumulates
FEEDBACK_STRENGTH = 0.05  # exponent scaling per unit pressure


# ---------------------------------------------------------------------------
# Desynchronization pressure
# ---------------------------------------------------------------------------

def compute_desync_pressure(s_r2p_history: List[float],
                             window: int = DESYNC_WINDOW,
                             threshold: float = DESYNC_THRESHOLD) -> float:
    """
    Compute normalised desynchronization pressure from recent S_{R->P}.

    Pressure ∈ [0, 1]:
        0.0 — all recent cycles below threshold (no pressure)
        1.0 — all recent cycles maximally above threshold

    Args:
        s_r2p_history: list of S_{R->P} values, most recent last
        window:        number of recent cycles to consider
        threshold:     S_{R->P} level above which pressure accumulates

    Returns:
        pressure: float in [0, 1]
    """
    if not s_r2p_history:
        return 0.0

    recent = s_r2p_history[-window:]
    pressures = []
    for v in recent:
        if v > threshold:
            p = min((v - threshold) / 3.0, 1.0)
            pressures.append(p)
        else:
            pressures.append(0.0)

    return float(np.mean(pressures))


# ---------------------------------------------------------------------------
# Convenience: full feedback step
# ---------------------------------------------------------------------------

def feedback_step(patient,
                  s_r2p_history: List[float],
                  window: int = DESYNC_WINDOW,
                  threshold: float = DESYNC_THRESHOLD,
                  strength: float = FEEDBACK_STRENGTH) -> float:
    """
    Compute pressure and apply prior sharpening to patient in one call.

    This is the main entry point called from the simulation loop.
    Unlike v1, it does NOT modify B_P_true — it calls
    patient.sharpen_prior() directly.

    Args:
        patient:       PatientAgent instance (modified in place)
        s_r2p_history: recent S_{R->P} history
        window:        pressure window
        threshold:     pressure threshold in nats
        strength:      sharpening strength per unit pressure

    Returns:
        pressure: float in [0, 1] — for recording in trace
    """
    pressure = compute_desync_pressure(s_r2p_history, window, threshold)
    if pressure > 1e-6:
        patient.sharpen_prior(pressure, strength)
    return pressure
