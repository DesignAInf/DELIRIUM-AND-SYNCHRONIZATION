"""
world/spaces.py
===============
Shared state, observation, and action spaces for the two-agent
POMDP model of ICU delirium.

Design principle: all spaces are fully discrete and small enough
to allow exact inference (no sampling approximations needed).

Patient hidden state s_P:
    Cartesian product of gamma_cog × gamma_emo × circadian
    4 × 4 × 3 = 48 states

Room hidden state s_R:
    Same 48-dimensional space — the room maintains an internal
    model of the patient's state, which it infers from sensors.

Patient observations o_P:
    What the patient perceives from the room.
    light_level × sound_type = 4 × 4 = 16 observations

Room observations o_R:
    Physiological signals detected by ICU sensors.
    hr_proxy × movement × eeg_proxy = 4 × 3 × 2 = 24 observations

Patient actions a_P:
    Micro-actions available to an intubated patient.
    3 actions: passive, orient, startle

Room actions a_R:
    Light × sound configurations.
    4 × 4 = 16 actions
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple


# ---------------------------------------------------------------------------
# Patient hidden state space  (48 states)
# ---------------------------------------------------------------------------

# gamma_cog: cognitive precision (cholinergic gating)
COG_LEVELS = ["collapsed", "low", "normal", "high"]
N_COG = len(COG_LEVELS)   # 4

# gamma_emo: emotional precision (amygdala-HPA)
EMO_LEVELS = ["minimal", "moderate", "elevated", "spiked"]
N_EMO = len(EMO_LEVELS)   # 4

# circadian alignment
CIRC_LEVELS = ["misaligned", "partial", "aligned"]
N_CIRC = len(CIRC_LEVELS)   # 3

N_STATES = N_COG * N_EMO * N_CIRC   # 48


def state_index(cog: int, emo: int, circ: int) -> int:
    """Flat index from (cog, emo, circ) bin triple."""
    return cog * (N_EMO * N_CIRC) + emo * N_CIRC + circ


def index_to_state(idx: int) -> Tuple[int, int, int]:
    """Recover (cog, emo, circ) from flat index."""
    circ = idx % N_CIRC;        idx //= N_CIRC
    emo  = idx % N_EMO;         idx //= N_EMO
    cog  = idx
    return cog, emo, circ


# Delirium state: collapsed cog + spiked emo + misaligned circadian
DELIRIUM_STATES = frozenset(
    state_index(cog=0, emo=3, circ=c) for c in range(N_CIRC)
)
# Near-delirium: any collapsed-cog + elevated/spiked emo
NEAR_DELIRIUM_STATES = frozenset(
    state_index(cog=0, emo=e, circ=c)
    for e in [2, 3] for c in range(N_CIRC)
)
# Healthy states: normal/high cog + minimal/moderate emo
HEALTHY_STATES = frozenset(
    state_index(cog=c, emo=e, circ=circ)
    for c in [2, 3] for e in [0, 1] for circ in range(N_CIRC)
)


# ---------------------------------------------------------------------------
# Patient observation space  (16 observations)
# ---------------------------------------------------------------------------

# What the patient perceives from the room
LIGHT_OBS  = ["dark", "dim_warm", "bright_neutral", "bright_blue"]
SOUND_OBS  = ["silence", "nature_coherent", "noise_incoherent", "alarm"]

N_LIGHT_OBS = len(LIGHT_OBS)   # 4
N_SOUND_OBS = len(SOUND_OBS)   # 4
N_OBS_PATIENT = N_LIGHT_OBS * N_SOUND_OBS   # 16


def obs_patient_index(light: int, sound: int) -> int:
    return light * N_SOUND_OBS + sound


def index_to_obs_patient(idx: int) -> Tuple[int, int]:
    sound = idx % N_SOUND_OBS
    light = idx // N_SOUND_OBS
    return light, sound


# ---------------------------------------------------------------------------
# Room observation space  (24 observations)
# ---------------------------------------------------------------------------

# Physiological signals from ICU sensors
HR_PROXY   = ["low", "normal", "elevated", "very_high"]     # 4
MOVEMENT   = ["still", "micro", "agitated"]                  # 3
EEG_PROXY  = ["normal", "altered"]                           # 2

N_HR  = len(HR_PROXY)    # 4
N_MOV = len(MOVEMENT)    # 3
N_EEG = len(EEG_PROXY)   # 2
N_OBS_ROOM = N_HR * N_MOV * N_EEG   # 24


def obs_room_index(hr: int, mov: int, eeg: int) -> int:
    return hr * (N_MOV * N_EEG) + mov * N_EEG + eeg


def index_to_obs_room(idx: int) -> Tuple[int, int, int]:
    eeg = idx % N_EEG;       idx //= N_EEG
    mov = idx % N_MOV;       idx //= N_MOV
    hr  = idx
    return hr, mov, eeg


# ---------------------------------------------------------------------------
# Patient action space  (3 actions)
# ---------------------------------------------------------------------------

PATIENT_ACTIONS = ["passive", "orient", "startle"]
N_ACTIONS_PATIENT = len(PATIENT_ACTIONS)   # 3

PATIENT_ACTION_PASSIVE  = 0
PATIENT_ACTION_ORIENT   = 1
PATIENT_ACTION_STARTLE  = 2


# ---------------------------------------------------------------------------
# Room action space  (16 actions = 4 light × 4 sound)
# ---------------------------------------------------------------------------

LIGHT_ACTIONS = ["off_dark", "dim_warm_2700K", "neutral_4000K", "blue_6500K"]
SOUND_ACTIONS = ["silence", "nature_35dB_coherent",
                  "noise_50dB_coherent", "noise_65dB_incoherent"]

N_LIGHT_ACT = len(LIGHT_ACTIONS)   # 4
N_SOUND_ACT = len(SOUND_ACTIONS)   # 4
N_ACTIONS_ROOM = N_LIGHT_ACT * N_SOUND_ACT   # 16


def room_action_index(light: int, sound: int) -> int:
    return light * N_SOUND_ACT + sound


def index_to_room_action(idx: int) -> Tuple[int, int]:
    sound = idx % N_SOUND_ACT
    light = idx // N_SOUND_ACT
    return light, sound


# Canonical "best" and "worst" room actions for reference
ROOM_ACTION_BEST  = room_action_index(light=1, sound=1)   # dim_warm + nature
ROOM_ACTION_WORST = room_action_index(light=3, sound=3)   # blue + incoherent


# ---------------------------------------------------------------------------
# State labels (for printing and figures)
# ---------------------------------------------------------------------------

def state_label(idx: int) -> str:
    cog, emo, circ = index_to_state(idx)
    return f"cog={COG_LEVELS[cog]},emo={EMO_LEVELS[emo]},circ={CIRC_LEVELS[circ]}"


def room_action_label(idx: int) -> str:
    light, sound = index_to_room_action(idx)
    return f"{LIGHT_ACTIONS[light]} | {SOUND_ACTIONS[sound]}"


# ---------------------------------------------------------------------------
# Clinical phenotypes (discrete parameter sets, no ODE)
# ---------------------------------------------------------------------------

@dataclass
class Phenotype:
    name:        str
    description: str
    prevalence:  float

    # Initial belief over cog states: P(cog=0..3) at admission
    cog_prior:   np.ndarray   # shape (4,)
    # Initial belief over emo states: P(emo=0..3) at admission
    emo_prior:   np.ndarray   # shape (4,)
    # Initial belief over circ states
    circ_prior:  np.ndarray   # shape (3,)

    # How strongly room actions shift cog/emo states (multiplier on B_P)
    cog_plasticity: float   # 1.0 = normal, <1 = harder to improve
    emo_reactivity: float   # 1.0 = normal, >1 = more reactive to threats

    # Probability of spontaneous startle action per cycle
    startle_prob: float


PHENOTYPE_A = Phenotype(
    name="A_young_healthy",
    description="Young adult, full cholinergic reserve",
    prevalence=0.15,
    cog_prior  = np.array([0.05, 0.15, 0.50, 0.30]),
    emo_prior  = np.array([0.50, 0.35, 0.10, 0.05]),
    circ_prior = np.array([0.10, 0.30, 0.60]),
    cog_plasticity=1.00,
    emo_reactivity=1.00,
    startle_prob=0.05,
)

PHENOTYPE_B = Phenotype(
    name="B_elderly_frail",
    description="Age >65, reduced ACh reserve, fragile circadian",
    prevalence=0.40,
    cog_prior  = np.array([0.10, 0.30, 0.45, 0.15]),
    emo_prior  = np.array([0.25, 0.40, 0.25, 0.10]),
    circ_prior = np.array([0.30, 0.40, 0.30]),
    cog_plasticity=0.65,
    emo_reactivity=1.30,
    startle_prob=0.10,
)

PHENOTYPE_C = Phenotype(
    name="C_septic_ventilated",
    description="Sepsis + MV >48h, HPA hyperactivation",
    prevalence=0.30,
    cog_prior  = np.array([0.20, 0.35, 0.35, 0.10]),
    emo_prior  = np.array([0.10, 0.25, 0.35, 0.30]),
    circ_prior = np.array([0.50, 0.35, 0.15]),
    cog_plasticity=0.50,
    emo_reactivity=1.70,
    startle_prob=0.20,
)

PHENOTYPE_D = Phenotype(
    name="D_dementia_MCI",
    description="Dementia/MCI, structural ACh depletion",
    prevalence=0.15,
    cog_prior  = np.array([0.30, 0.40, 0.25, 0.05]),
    emo_prior  = np.array([0.15, 0.30, 0.35, 0.20]),
    circ_prior = np.array([0.55, 0.30, 0.15]),
    cog_plasticity=0.35,
    emo_reactivity=1.45,
    startle_prob=0.15,
)

ALL_PHENOTYPES = [PHENOTYPE_A, PHENOTYPE_B, PHENOTYPE_C, PHENOTYPE_D]
