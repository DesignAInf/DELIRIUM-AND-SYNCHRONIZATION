# ICU Delirium — Two-Agent POMDP Model

**Author:** Luca M. Possati
**Version:** 3.0.0
**License:** MIT

## Overview

This repository implements a computational model of ICU delirium grounded in the
active inference framework (Friston et al., 2015; Parr et al., 2022). The model
represents the ICU as two bidirectionally coupled POMDP agents — the **patient** and
the **room** — each maintaining a generative model and continuously attempting to
predict the other.

The "room agent" is a modeling construct: it denotes the environment-side
sensing–inference–actuation loop (physiological sensors, an inferential process that
maintains a belief about the patient's hidden state, and actuators that set light and
sound), whether instantiated by clinical staff or by an automated monitoring system.
It is not a claim that a physical room holds beliefs.

The accompanying paper is *ICU Delirium as a Failure of Predictive Synchronization:
A Two-Agent Active Inference Model* (Possati, 2026).

## Core hypothesis

ICU delirium emerges when persistent de-synchronization between patient and
environment induces **precision rigidity** in the patient: the patient's prior beliefs
become so precise that incoming observations can no longer update them, and belief
updating stalls.

Crucially, the causal feedback does **not** degrade the patient's generative model:
`A_P` and `B_P` remain unchanged. Instead, persistent de-synchronization progressively
**sharpens the patient's prior** (raises a probability distribution to a power > 1),
concentrating mass on the current MAP state until the posterior becomes insensitive to
observations. The causal chain is:

\```
persistent de-synchronization  (S_{R→P} high)
        ↓
desynchronization pressure ρ accumulates
        ↓
prior sharpening:  Q_P^sharp ∝ (Q_P)^{1 + ρ·α}  →  Q_P^sharp ≈ δ_{s*}
        ↓
precision rigidity:  posterior Q*(s) ≈ Q_prior(s)  regardless of observation
        ↓
delirium
\```

The patient's generative model (`A_P`, `B_P`, `C_P`) is intact throughout; only its
confidence in the current state becomes pathologically elevated. This is consistent
with the clinical picture of hypoactive delirium (withdrawn, unresponsive — not
agitated or hallucinating).

## Synchronization metrics

Synchronization is the capacity of each agent to predict the other. Both directions
are **commensurable** negative log-probabilities of an observed event under a belief:

| Direction | Metric | Interpretation |
|---|---|---|
| Room → Patient | `S_{R→P} = −ln Q_R(s*)` | Room surprisal at the patient's true state |
| Patient → Room | `S_{P→R} = −ln Σ_s Q_P(s) · P(o_R | s)` | Patient surprisal at the room's observation |

Note: `S_{P→R}` is **not** the patient's variational free energy `F_P`. An earlier
implementation used `F_P` as a proxy, but `F_P` adds a non-negative divergence term to
the surprisal and is therefore not commensurable with `S_{R→P}`. The current
implementation uses the true surprisal form above for both directions.

`S_{R→P}` is anchored to the patient's true hidden state `s*`, which exists in the
simulator but is not available to the room during inference (the room forms `Q_R(s)`
only from physiological signals); `s*` is used only to *score* the room's belief.
`S_{P→R}` is anchored to the room's actual observation `o_R`, which is measurable from
sensors.

Delirium is declared when both directions fail simultaneously for ≥ 4 consecutive
cycles:

\```
S_{R→P} > 3.0 nats   AND   S_{P→R} > 3.0 nats
\```

## Architecture

### Two agents

\```
PatientAgent                          RoomAgent (Level-3 hierarchical)
────────────                          ────────────────────────────────
States:  (γ_cog × γ_emo × circ)       Level 1:  Q_R(s) — current patient state
         4 × 4 × 3 = 48               Level 2:  Q_R(θ) — latent patient parameters
                                                θ = (θ_cog, θ_emo) on 10×10 grid
Obs:     light × sound (16)           Obs:      HR × movement × EEG proxy (24)
Actions: passive/orient/startle (3)   Actions:  light × sound (16)
Goal:    minimise VFE_P               Goal:     minimise EFE_R (personalised B_R(θ))
\```

Here "precision" is used in its standard active-inference sense (the inverse variance
governing how strongly a signal is weighted): `γ_cog` indexes the precision of the
patient's sensory likelihood `A_P`, and `γ_emo` the gain on threat-related signals.

### Level-3: the room learns the patient

The room infers the patient's latent parameters progressively:

\```
θ = (θ_cog, θ_emo)     ← cognitive plasticity, emotional reactivity
       ↓ Bayes update each cycle
  Q_R(θ) sharpens over time (entropy decreases = learning)
       ↓
  B_R_eff = Σ_θ Q_R(θ) · B_R(θ)   ← patient-specific transition matrix
       ↓
  Better state inference → better synchronization → lower desynchronization pressure
\```

## Installation

\```bash
git clone https://github.com/DesignAInf/DELIRIUM-AND-SYNCHRONIZATION.git
cd DELIRIUM-AND-SYNCHRONIZATION/DELIRIUM
pip install -r requirements.txt
\```

Dependencies: numpy, scipy, scikit-learn, matplotlib.

## Usage

\```bash
python3 main.py --n 30 --cycles 200          # main cohort (120 patients, paper-quality)
python3 main.py --n 12 --cycles 150          # ablation-scale run
python3 main.py --fast                        # quick validation
python3 main.py --fast --verbose              # single-patient trace
\```

Reviewer-response analyses (Figure 2, Tables 2–4) are reproduced with:

\```bash
python3 -m analysis.reviewer_response
\```

## Results (main cohort: 120 patients, 200 cycles, seed 42, full model)

Surprisal and learning values are mean ± SD across the 30 patients per phenotype
(sample SD, ddof = 1). Delirium is the binary declared-delirium outcome.

| Phenotype | Del% | SI | S_{R→P} | S_{P→R} | θ-learn (ℓ) |
|---|---|---|---|---|---|
| A: Young, healthy | 26.7 | 0.54 ± 0.10 | 2.75 ± 1.58 | 2.97 ± 0.34 | 0.09 ± 0.05 |
| B: Elderly, frail | 40.0 | 0.51 ± 0.12 | 2.93 ± 1.01 | 3.05 ± 0.43 | 0.10 ± 0.06 |
| C: Septic, ventilated | 53.3 | 0.48 ± 0.13 | 3.78 ± 2.96 | 3.13 ± 0.46 | 0.09 ± 0.05 |
| D: Dementia / MCI | 53.3 | 0.46 ± 0.13 | 3.37 ± 1.18 | 3.18 ± 0.56 | 0.07 ± 0.05 |

Overall: 120 patients, 52 delirious (43%). Logistic regression
`P(delirium) ∼ mean SI`: β₁ = −45.56 (95% CI [−64.52, −26.60]), AUC = 0.97.
**Spearman ρ(SI, delirium) = −0.81** (Pearson r = −0.78, for comparison).

The association is summarized with Spearman's ρ (a distribution-free measure of
monotonic association) and a logistic fit, rather than Pearson's r: the
synchronization–delirium relationship is thresholded and saturating, so a linear
coefficient is not an appropriate summary.

The within-phenotype SD of `S_{R→P}` is comparable to or larger than the
between-phenotype spread of its mean, so the phenotypic ordering is a difference in
means rather than a separation at the individual-patient level. The robust, replicable
claim — phenotype A develops less delirium than phenotype D — is preserved across all
seeds tested (ρ ≈ −0.80, SD 0.03 across 5 seeds).

## Repository structure

\```
DELIRIUM-AND-SYNCHRONIZATION/
└── DELIRIUM/
    ├── main.py
    ├── README.md
    ├── requirements.txt
    ├── world/          spaces, generative models, observations
    ├── inference/      core.py (VFE/EFE), hierarchical.py (θ inference)
    ├── agents/         PatientAgent, RoomAgent
    ├── simulation/     loop.py, cohort.py
    └── analysis/       plots.py (figures), reviewer_response.py (Fig. 2, Tables 2–4)
\```

## Citation

\```bibtex
@software{possati2026icu,
  author  = {Possati, Luca M.},
  title   = {ICU Delirium as a Failure of Predictive Synchronization:
             A Two-Agent Active Inference Model},
  year    = {2026},
  url     = {https://github.com/DesignAInf/DELIRIUM-AND-SYNCHRONIZATION},
  version = {3.0.0}
}
\```

## License

MIT
