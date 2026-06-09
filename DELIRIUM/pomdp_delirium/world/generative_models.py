"""
world/generative_models.py
==========================
Constructs the A (likelihood) and B (transition) matrices for both agents,
and the C (preference) vectors.

All matrices are hand-specified from neurobiological priors and then
normalized. This is transparent and auditable — every entry has a
documented reason. No MLE from synthetic data (which was circular in
the previous model).

Patient generative model:
    A_P : P(o_P | s_P)   shape (N_OBS_PATIENT, N_STATES)
    B_P : P(s_P'| s_P, a_P)  shape (N_STATES, N_STATES, N_ACTIONS_PATIENT)
    C_P : log-preferences over o_P  shape (N_OBS_PATIENT,)
    D_P : prior over initial states  shape (N_STATES,)

Room generative model:
    A_R : P(o_R | s_R)   shape (N_OBS_ROOM, N_STATES)
    B_R : P(s_R'| s_R, a_R)  shape (N_STATES, N_STATES, N_ACTIONS_ROOM)
    C_R : log-preferences over s_R  shape (N_STATES,)  [over inferred states]
    D_R : prior over patient states  shape (N_STATES,)
"""

import numpy as np
from world.spaces import (
    N_STATES, N_COG, N_EMO, N_CIRC,
    N_OBS_PATIENT, N_OBS_ROOM,
    N_ACTIONS_PATIENT, N_ACTIONS_ROOM,
    N_LIGHT_OBS, N_SOUND_OBS,
    N_HR, N_MOV, N_EEG,
    N_LIGHT_ACT, N_SOUND_ACT,
    state_index, index_to_state, index_to_room_action,
    obs_patient_index, obs_room_index,
    room_action_index,
    HEALTHY_STATES, DELIRIUM_STATES, NEAR_DELIRIUM_STATES,
    Phenotype,
)

EPS = 1e-6   # Dirichlet smoothing


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _normalize_cols(M: np.ndarray) -> np.ndarray:
    """Column-normalize a matrix (each column sums to 1)."""
    col_sums = M.sum(axis=0, keepdims=True)
    col_sums = np.where(col_sums < EPS, 1.0, col_sums)
    return M / col_sums


def _normalize_rows(M: np.ndarray) -> np.ndarray:
    """Row-normalize a matrix (each row sums to 1)."""
    row_sums = M.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums < EPS, 1.0, row_sums)
    return M / row_sums


# ---------------------------------------------------------------------------
# A_P : Patient likelihood  P(o_P | s_P)
# shape (N_OBS_PATIENT=16, N_STATES=48)
# Rows = observations, Cols = hidden states
# ---------------------------------------------------------------------------

def build_A_patient() -> np.ndarray:
    """
    Each column gives P(o_P | s_P=s).
    Logic:
      - collapsed cog → patient cannot distinguish observations reliably
        (likelihood flattens toward uniform — the formal implementation
        of precision-weighted likelihood from the active inference framework)
      - high cog → sharp, reliable perception
      - spiked emo → patient tends to perceive threat signals (alarm/incoherent)
        even when absent (prior dominates perception)
      - circadian misalignment → dark/silence perceived as disorienting
    """
    A = np.full((N_OBS_PATIENT, N_STATES), EPS)

    for s in range(N_STATES):
        cog, emo, circ = index_to_state(s)

        # Precision weight: how sharply the patient perceives the environment
        # At cog=collapsed, likelihood is near-uniform (cannot update on evidence)
        precision = [0.10, 0.40, 0.80, 1.00][cog]

        # Base likelihood per light observation given true state
        # (will be modulated by precision below)
        # Assume room is in a "neutral" state for A construction;
        # B matrix handles how actions change states.
        # A encodes the perceptual mapping, not the environmental action.

        # Light perception
        # High cog → correct perception; low cog → confused
        light_probs = np.array([0.25, 0.25, 0.25, 0.25])   # uniform baseline

        if cog >= 2:   # normal/high — veridical perception
            # Sane patient can distinguish all light levels
            # The "true" light is embedded in the state via circadian:
            # aligned circ → more likely to be in bright-neutral environment
            if circ == 2:    # aligned
                light_probs = np.array([0.05, 0.25, 0.50, 0.20])
            elif circ == 1:  # partial
                light_probs = np.array([0.15, 0.35, 0.35, 0.15])
            else:            # misaligned — dark/dim expected
                light_probs = np.array([0.40, 0.35, 0.15, 0.10])
        else:
            # Low/collapsed cog: perception unreliable
            # Collapsed → cannot distinguish; slight bias to confuse
            noise = 1.0 - precision
            light_probs = precision * light_probs + noise * np.array([0.25]*4)

        # Sound perception
        sound_probs = np.array([0.25, 0.25, 0.25, 0.25])

        if cog >= 2:
            if emo <= 1:   # low emotional precision → calm perception
                sound_probs = np.array([0.20, 0.45, 0.25, 0.10])
            elif emo == 2: # elevated emo → more sensitive to threat signals
                sound_probs = np.array([0.10, 0.25, 0.35, 0.30])
            else:          # spiked emo → alarm dominates perception
                sound_probs = np.array([0.05, 0.10, 0.25, 0.60])
        else:
            # Low cog + spiked emo: alarm perceived even in silence
            threat_bias = [0.0, 0.0, 0.15, 0.40][emo]
            sound_probs = np.array([
                0.25 - threat_bias * 0.5,
                0.25 - threat_bias * 0.3,
                0.25 + threat_bias * 0.3,
                0.25 + threat_bias * 0.5,
            ])
            sound_probs = np.clip(sound_probs, EPS, 1.0)

        # Normalize sub-distributions
        light_probs /= light_probs.sum()
        sound_probs /= sound_probs.sum()

        # Fill A column: outer product of marginal light × sound
        for l in range(N_LIGHT_OBS):
            for snd in range(N_SOUND_OBS):
                o = obs_patient_index(l, snd)
                A[o, s] = light_probs[l] * sound_probs[snd]

    return A   # already column-normalized by construction


# ---------------------------------------------------------------------------
# B_P : Patient transition  P(s_P' | s_P, a_P)
# shape (N_STATES, N_STATES, N_ACTIONS_PATIENT)
# ---------------------------------------------------------------------------

def build_B_patient(phenotype: Phenotype) -> np.ndarray:
    """
    Transition matrix for the patient under each of their 3 micro-actions.
    Key biological logic:
      - passive:  slow drift toward equilibrium based on current state
                  (spontaneous recovery limited by cholinergic reserve)
      - orient:   attempt to re-align circadian; small cog benefit
      - startle:  emo spike; possible cog drop if near threshold

    Phenotype modulates plasticity (how easily cog recovers) and
    reactivity (how strongly emo spikes).
    """
    B = np.zeros((N_STATES, N_STATES, N_ACTIONS_PATIENT))

    for a in range(N_ACTIONS_PATIENT):
        for s in range(N_STATES):
            cog, emo, circ = index_to_state(s)
            col = np.zeros(N_STATES)

            # Base: stay in current state
            col[s] = 0.50

            # --- Action-specific transitions ---
            if a == 0:   # passive
                # Slow spontaneous recovery of cog (if not collapsed)
                if cog < 3:
                    s_up = state_index(min(cog+1, 3), emo, circ)
                    col[s_up] += 0.10 * phenotype.cog_plasticity
                # Slow emo drift toward moderate
                if emo > 1:
                    s_down = state_index(cog, max(emo-1, 0), circ)
                    col[s_down] += 0.08
                # Slight circadian drift toward misaligned (ICU disrupts sleep)
                if circ > 0:
                    s_circ = state_index(cog, emo, circ-1)
                    col[s_circ] += 0.05

            elif a == 1:   # orient — patient tries to make sense of environment
                # Circadian alignment benefit
                if circ < 2:
                    s_circ = state_index(cog, emo, min(circ+1, 2))
                    col[s_circ] += 0.15 * phenotype.cog_plasticity
                # Cog benefit if circadian helps
                if cog < 3 and circ >= 1:
                    s_cog = state_index(min(cog+1, 3), emo, circ)
                    col[s_cog] += 0.10 * phenotype.cog_plasticity
                # Emo cost if misaligned (disorientation increases threat)
                if circ == 0 and emo < 3:
                    s_emo = state_index(cog, min(emo+1, 3), circ)
                    col[s_emo] += 0.10 * phenotype.emo_reactivity

            elif a == 2:   # startle — involuntary threat response
                # Emo spike
                if emo < 3:
                    s_emo = state_index(cog, min(emo+1, 3), circ)
                    col[s_emo] += 0.30 * phenotype.emo_reactivity
                # Possible cog drop if already low
                if cog <= 1:
                    s_cog = state_index(max(cog-1, 0), emo, circ)
                    col[s_cog] += 0.15

            # Normalize
            total = col.sum()
            if total < EPS:
                col[s] = 1.0
            else:
                col /= total

            B[:, s, a] = col

    return B


# ---------------------------------------------------------------------------
# A_R : Room likelihood  P(o_R | s_R)
# shape (N_OBS_ROOM=24, N_STATES=48)
# Maps patient hidden states to observable physiological signals
# ---------------------------------------------------------------------------

def build_A_room() -> np.ndarray:
    """
    Each column gives P(o_R | s_R=s) — the physiological signal
    the room's sensors observe given the patient's true state.

    HR proxy:
        collapsed cog + spiked emo → very_high HR (autonomic storm)
        minimal emo + high cog     → normal HR
    Movement:
        spiked emo → agitated
        collapsed cog → still (unresponsive) OR agitated (hyperactive)
        healthy → micro-movements only
    EEG proxy:
        collapsed cog → altered (theta dominance, reduced alpha)
        high cog       → normal
    """
    A = np.full((N_OBS_ROOM, N_STATES), EPS)

    for s in range(N_STATES):
        cog, emo, circ = index_to_state(s)

        # HR proxy distribution
        hr = np.array([0.25, 0.25, 0.25, 0.25])
        if emo == 0:
            hr = np.array([0.20, 0.60, 0.15, 0.05])
        elif emo == 1:
            hr = np.array([0.10, 0.55, 0.25, 0.10])
        elif emo == 2:
            hr = np.array([0.05, 0.25, 0.45, 0.25])
        else:   # spiked
            hr = np.array([0.02, 0.08, 0.30, 0.60])

        # Movement distribution
        mov = np.array([0.33, 0.33, 0.34])
        if cog == 0 and emo == 3:   # delirium: agitated OR still
            mov = np.array([0.30, 0.10, 0.60])
        elif cog == 0:              # collapsed, quiet
            mov = np.array([0.60, 0.30, 0.10])
        elif emo >= 2:              # elevated emo: restless
            mov = np.array([0.10, 0.30, 0.60])
        else:                       # healthy range
            mov = np.array([0.20, 0.65, 0.15])

        # EEG proxy
        eeg = np.array([0.5, 0.5])
        if cog <= 1:
            eeg = np.array([0.15, 0.85])   # altered
        elif cog >= 2:
            eeg = np.array([0.80, 0.20])   # normal

        # Normalize
        hr  /= hr.sum()
        mov /= mov.sum()
        eeg /= eeg.sum()

        # Fill A column
        for h in range(N_HR):
            for m in range(N_MOV):
                for e in range(N_EEG):
                    o = obs_room_index(h, m, e)
                    A[o, s] = hr[h] * mov[m] * eeg[e]

    return A


# ---------------------------------------------------------------------------
# B_R : Room transition  P(s_R' | s_R, a_R)
# shape (N_STATES, N_STATES, N_ACTIONS_ROOM)
# How room actions shift patient states
# ---------------------------------------------------------------------------

def build_B_room(phenotype: Phenotype) -> np.ndarray:
    """
    For each room action, encodes how patient states are expected to evolve.
    The room's B matrix is its generative model of how its actions affect the patient.

    Light actions:
        dim_warm_2700K   → cog protection (circadian support), emo calm
        blue_6500K       → cog cost (melanopsin suppression), emo neutral
        off_dark         → circadian disruption if prolonged
        neutral_4000K    → moderate cog support

    Sound actions:
        nature_35dB_coherent   → emo reduction (predictable, localizable)
        noise_50dB_coherent    → moderate emo (localizable but not calming)
        noise_65dB_incoherent  → emo spike (unresolvable spatial PE)
        silence                → neutral (context-dependent)
    """
    B = np.zeros((N_STATES, N_STATES, N_ACTIONS_ROOM))

    for a in range(N_ACTIONS_ROOM):
        light, sound = index_to_room_action(a)

        # Determine action effects
        # Light effects on cog and circadian
        cog_delta  = [0.0, +1.0, +0.5, -1.0][light]   # dark=0,warm=+,neut=+,blue=-
        circ_delta = [0.0, +1.0, +0.5, -0.5][light]   # warm supports circadian

        # Sound effects on emo
        emo_delta  = [0.0, -1.0, -0.3, +1.5][sound]   # nature=calm,incoh=threat

        # Scale by phenotype
        cog_delta  *= phenotype.cog_plasticity
        emo_delta  *= phenotype.emo_reactivity

        for s in range(N_STATES):
            cog, emo, circ = index_to_state(s)
            col = np.zeros(N_STATES)
            col[s] = 0.50   # inertia: half the time state does not change

            # Cognitive transition
            new_cog = cog
            if cog_delta > 0 and cog < 3:
                # Probability of improving cog by 1 level
                p_improve = min(0.30 * abs(cog_delta), 0.35)
                s_better  = state_index(min(cog+1, 3), emo, circ)
                col[s_better] += p_improve
            elif cog_delta < 0 and cog > 0:
                p_worsen = min(0.30 * abs(cog_delta), 0.35)
                s_worse  = state_index(max(cog-1, 0), emo, circ)
                col[s_worse] += p_worsen

            # Emotional transition
            if emo_delta < 0 and emo > 0:
                p_calm = min(0.30 * abs(emo_delta), 0.40)
                s_calm = state_index(cog, max(emo-1, 0), circ)
                col[s_calm] += p_calm
            elif emo_delta > 0 and emo < 3:
                p_spike = min(0.25 * abs(emo_delta), 0.45)
                s_spike = state_index(cog, min(emo+1, 3), circ)
                col[s_spike] += p_spike

            # Circadian transition
            if circ_delta > 0 and circ < 2:
                p_align = min(0.20 * abs(circ_delta), 0.25)
                s_align = state_index(cog, emo, min(circ+1, 2))
                col[s_align] += p_align
            elif circ_delta < 0 and circ > 0:
                p_misalign = min(0.15 * abs(circ_delta), 0.20)
                s_mis = state_index(cog, emo, max(circ-1, 0))
                col[s_mis] += p_misalign

            # Self-reinforcing dynamics at extremes (structural ICU feature)
            # Once collapsed, harder to recover without external intervention
            if cog == 0 and emo == 3:
                # Absorbing tendency: most probability stays or worsens
                col[s] += 0.20
            elif cog >= 3 and emo == 0:
                # Healthy attractor: slight pull to stay
                col[s] += 0.10

            # Normalize
            total = col.sum()
            if total < EPS:
                col[s] = 1.0
            else:
                col /= total

            B[:, s, a] = col

    return B


# ---------------------------------------------------------------------------
# Preference vectors
# ---------------------------------------------------------------------------

def build_C_patient() -> np.ndarray:
    """
    Log-preferences over patient observations o_P.
    Patient prefers: dim_warm light, nature_coherent sound.
    Patient dislikes: alarm, bright_blue, silence-in-dark (disorienting).
    """
    C = np.zeros(N_OBS_PATIENT)
    for l in range(N_LIGHT_OBS):
        for snd in range(N_SOUND_OBS):
            o = obs_patient_index(l, snd)
            # Light preference: dim_warm=+2, neutral=0, dark=-1, blue=-2
            lp = [-1.0, +2.0, 0.0, -2.0][l]
            # Sound preference: nature=+2, silence=0, coherent_noise=-0.5, alarm=-3
            sp = [0.0, +2.0, -0.5, -3.0][snd]
            C[o] = lp + sp
    return C


def build_C_room() -> np.ndarray:
    """
    Log-preferences over inferred patient states for the room agent.
    Room prefers patient to be in healthy states, dislikes delirium states.
    """
    C = np.zeros(N_STATES)
    for s in range(N_STATES):
        cog, emo, circ = index_to_state(s)
        # Prefer high cog, low emo, aligned circadian
        C[s] = (
            [- 3.0, -1.0, +1.0, +3.0][cog]
          + [0.0, -0.5, -2.0, -4.0][emo]
          + [-0.5, 0.0, +1.0][circ]
        )
    return C


# ---------------------------------------------------------------------------
# Initial state priors
# ---------------------------------------------------------------------------

def build_D_patient(phenotype: Phenotype) -> np.ndarray:
    """
    Prior over patient's initial hidden state, drawn from phenotype distribution.
    """
    D = np.zeros(N_STATES)
    for cog in range(N_COG):
        for emo in range(N_EMO):
            for circ in range(N_CIRC):
                s = state_index(cog, emo, circ)
                D[s] = (phenotype.cog_prior[cog] *
                        phenotype.emo_prior[emo] *
                        phenotype.circ_prior[circ])
    D += EPS
    D /= D.sum()
    return D


def build_D_room() -> np.ndarray:
    """
    Room starts with a uniform prior over patient states — it knows nothing
    about this specific patient at admission.
    """
    D = np.ones(N_STATES) / N_STATES
    return D
