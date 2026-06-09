"""
world/observations.py
=====================
Stochastic observation generators.

For each agent, given the true hidden state, sample an observation
from the corresponding likelihood matrix A.

Also: generate patient observations from room actions (what the
patient perceives given what the room is doing).
"""

import numpy as np
from world.spaces import (
    N_OBS_PATIENT, N_OBS_ROOM,
    obs_patient_index, obs_room_index,
    index_to_room_action,
    LIGHT_ACTIONS, SOUND_ACTIONS,
    index_to_state,
)

EPS = 1e-16


def sample_patient_obs(true_state: int,
                        room_action: int,
                        A_P: np.ndarray,
                        rng: np.random.Generator) -> int:
    """
    Sample what the patient perceives given their true hidden state
    and the current room action.

    The room action directly shapes the patient's observation distribution:
    it modulates A_P by biasing toward the observation consistent with
    the current light/sound configuration.

    Args:
        true_state:  patient's true hidden state index
        room_action: current room action index
        A_P:         patient likelihood matrix
        rng:         random generator

    Returns:
        obs_patient index
    """
    # Base likelihood from A_P
    probs = A_P[:, true_state].copy()

    # Room action modulates the observation: the physical environment
    # directly shapes what the patient can perceive
    light, sound = index_to_room_action(room_action)

    # Light observation bias: the room physically produces a light level
    # that anchors the patient's perception (regardless of their internal state)
    # This implements the coupling: room action → patient observation
    light_anchor = [0, 1, 2, 3][light]   # dark→0, warm→1, neutral→2, blue→3
    sound_anchor = [0, 1, 2, 1][sound]   # silence→0, nature→1, incoherent→2, incoh_loud→2

    # Blend A_P column with action-consistent observation
    # (precision-weighted: high cog = more veridical; low cog = more random)
    cog, emo, circ = index_to_state(true_state)
    precision = [0.15, 0.45, 0.80, 1.00][cog]

    # Anchor distribution: peaked on action-consistent observation
    anchor = np.full(N_OBS_PATIENT, EPS)
    anchor[obs_patient_index(light_anchor, sound_anchor)] = 1.0
    anchor /= anchor.sum()

    # Mix: precision controls how much the real environment "gets through"
    probs = precision * anchor + (1.0 - precision) * probs
    probs = np.clip(probs, EPS, None)
    probs /= probs.sum()

    return int(rng.choice(N_OBS_PATIENT, p=probs))


def sample_room_obs(true_state: int,
                     patient_action: int,
                     A_R: np.ndarray,
                     rng: np.random.Generator) -> int:
    """
    Sample what the room's sensors detect given the patient's true state
    and the patient's micro-action.

    Patient actions modulate the observable signal:
        passive:  no extra movement signal
        orient:   slight increase in movement detection
        startle:  HR spike + agitation visible to sensors

    Args:
        true_state:    patient's true hidden state
        patient_action: patient's micro-action index
        A_R:           room likelihood matrix
        rng:           random generator

    Returns:
        obs_room index
    """
    probs = A_R[:, true_state].copy()

    # Patient action modulates observable physiology
    from world.spaces import N_HR, N_MOV, N_EEG, obs_room_index

    if patient_action == 2:   # startle: HR spike + agitation
        # Boost probability of (elevated/very_high HR) × (agitated movement)
        for h in [2, 3]:        # elevated, very_high HR
            for m in [2]:       # agitated
                for e in range(N_EEG):
                    o = obs_room_index(h, m, e)
                    probs[o] *= 2.5

    elif patient_action == 1:   # orient: micro-movement
        for h in range(N_HR):
            for m in [1]:       # micro-movement
                for e in range(N_EEG):
                    o = obs_room_index(h, m, e)
                    probs[o] *= 1.5

    probs = np.clip(probs, EPS, None)
    probs /= probs.sum()

    return int(rng.choice(N_OBS_ROOM, p=probs))


def sample_true_next_state(true_state: int,
                            B: np.ndarray,
                            action: int,
                            rng: np.random.Generator) -> int:
    """
    Sample the true next hidden state from the transition distribution.
    Used to advance the "ground truth" state of the patient.

    Args:
        true_state: current true state
        B:          transition matrix (N_STATES, N_STATES, N_ACTIONS)
        action:     action index
        rng:        random generator

    Returns:
        next true state index
    """
    probs = B[:, true_state, action].copy()
    probs = np.clip(probs, EPS, None)
    probs /= probs.sum()
    return int(rng.choice(len(probs), p=probs))
