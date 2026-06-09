
# ICU Delirium — Two-Agent POMDP Model (Level-3 + Causal Feedback)

**Author:** Luca M. Possati
**Version:** 3.0.0
**License:** MIT

---

## Overview

This repository implements a computational model of ICU delirium grounded
in the **active inference framework** (Friston et al., 2015). The model
represents the ICU environment as two bidirectionally coupled POMDP agents.

### Core hypothesis

> ICU delirium emerges when **persistent de-synchronization between patient
> and environment causes precision rigidity**, which blocks belief updating
> and produces the clinical syndrome.

The causal chain is:

```
persistent de-synchronization (S_{R→P} high)
        ↓
γ_cog degradation  (causal feedback loop)
        ↓
precision rigidity  (A_P flattens → belief updating blocked)
        ↓
delirium
```

Synchronization is defined as the **capacity of each agent to predict
the other's states**:

| Direction | Metric | Interpretation |
|-----------|--------|----------------|
| Room → Patient | $S_{R\to P} = -\ln Q_R(s^*)$ | Room surprisal at patient's true state |
| Patient → Room | $S_{P\to R} = \mathcal{F}_P$ | Patient VFE from room observations |

**Delirium** is declared when both fail simultaneously for ≥ 4 cycles:
```
S_{R→P} > 3.0 nats   AND   S_{P→R} > 3.0 nats
```

---

## Architecture

### Two agents

```
PatientAgent                          RoomAgent (Level-3 hierarchical)
────────────                          ────────────────────────────────
States:  (γ_cog × γ_emo × circ)      Level 1:  Q_R(s) — current patient state
         4 × 4 × 3 = 48              Level 2:  Q_R(θ) — latent patient parameters
                                                θ = (θ_cog, θ_emo) on 10×10 grid
Obs:     light × sound (16)          Obs:      HR × movement × EEG proxy (24)
Actions: passive/orient/startle (3)   Actions:  light × sound (16)
Goal:    minimise VFE_P               Goal:     minimise EFE_R (personalised B_R(θ))
```

### Level-3: room learns the patient

The room infers latent patient parameters progressively:

```
θ = (θ_cog, θ_emo)     ← cognitive plasticity, emotional reactivity
       ↓ Bayes update each cycle
  Q_R(θ) sharpens over time (entropy decreases = learning)
       ↓
  B_R_eff = Σ_θ Q_R(θ) · B_R(θ)   ← patient-specific transition matrix
       ↓
  Better state inference → better synchronization → delirium prevention
```

---

## Installation

```bash
git clone https://github.com/possati/icu-delirium-pomdp.git
cd icu-delirium-pomdp
pip install -r requirements.txt   # numpy, matplotlib only
```

## Usage

```bash
python3 main.py --n 12 --cycles 60    # standard run (48 patients, ~13s)
python3 main.py --fast                 # quick validation (~2s)
python3 main.py --n 30 --cycles 120   # paper-quality run
python3 main.py --fast --verbose       # single-patient trace
```

---

## Results (v2.0, 48 patients, 60 cycles)

| Phenotype | Del% | SI | S_R→P | S_P→R | θ-learn |
|-----------|------|----|-------|-------|---------|
| A: Young | 0% | 0.59 | 2.20 | 2.74 | 0.03 |
| B: Elderly | 8% | 0.56 | 2.52 | 2.81 | 0.05 |
| C: Septic | 8% | 0.55 | 2.67 | 2.87 | 0.04 |
| D: Dementia | **17%** | 0.55 | 2.65 | 2.87 | 0.04 |

**r(sync ↔ P_del) = −0.452** — lower synchronization predicts higher delirium probability.

---

## Repository structure

```
pomdp_delirium/
├── main.py
├── world/          spaces, generative models, observations
├── inference/      core.py (VFE/EFE), hierarchical.py (θ inference)
├── agents/         PatientAgent, RoomAgent
├── simulation/     loop.py, cohort.py
└── analysis/       plots.py (6 figures)
```

---

## Citation

```bibtex
@software{possati2025icu,
  author  = {Possati, Luca M.},
  title   = {ICU Delirium: A Two-Agent POMDP Model with Hierarchical Room Agent},
  year    = {2025},
  url     = {https://github.com/possati/icu-delirium-pomdp},
  version = {2.0.0}
}
```

**License:** MIT
