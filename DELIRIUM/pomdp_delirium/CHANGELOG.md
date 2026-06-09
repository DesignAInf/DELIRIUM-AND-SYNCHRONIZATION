# Changelog

## [3.0.0] — 2025

### Major: Causal Feedback Loop — De-synchronization → Precision Rigidity

This release implements the core causal mechanism of the predictive
synchronization hypothesis. In previous versions, precision rigidity
(low γ_cog) was a consequence of phenotypic parameters — the causal
arrow ran from precision rigidity to de-synchronization. This version
closes the loop in the correct direction:

    persistent de-synchronization → γ_cog degradation → delirium

- **New:** `inference/causal_feedback.py` — computes desynchronization
  pressure from recent S_{R→P} history and modifies the true patient
  transition matrix B_P accordingly. The patient's generative model
  (used for inference) is NOT modified — the patient experiences the
  degradation without being able to attribute it to de-synchronization.
- **Changed:** `simulation/loop.py` — integrates causal feedback at
  every cycle (step 5). S_{R→P} history is tracked and fed into
  `feedback_step()` before each true state transition.
- **New trace:** `desync_pressure_trace` in `SimulationResult` — records
  the causal pressure applied at each cycle.
- **Parameters:** `DESYNC_WINDOW=10`, `DESYNC_THRESHOLD=3.0`,
  `FEEDBACK_STRENGTH=0.15` (tunable in `causal_feedback.py`).



### Major: Level-3 Hierarchical Room Agent

- **New:** `inference/hierarchical.py` — Bayesian inference over latent patient
  parameters θ = (θ_cog, θ_emo) on a 10×10 grid. Room builds a patient-specific
  generative model progressively over time via Q_R(θ) updates.
- **New:** `ThetaBeliefState` class — maintains Q_R(θ), computes MAP/mean θ,
  tracks entropy reduction (learning progress) and effective B_R = Σ_θ Q_R(θ) B_R(θ).
- **Changed:** `RoomAgent` fully rewritten as a non-dataclass with two-level inference:
  Level-1 (state, fast) and Level-2 (θ, slow). Adaptive temperature scales with
  learning progress.
- **Changed:** Exploration mode threshold raised (S_R2P > 5.0) and conditioned on
  learning progress > 0.1 to avoid chronic exploration before θ is identified.
- **New figure:** `learning.png` — θ learning progress and MAP (θ_cog, θ_emo) scatter.
- **New metric:** `mean_learning` in CohortResults and SimulationResult.
- **Changed:** Synchronization redefined as predictive capacity:
  S_{R→P} = −ln Q_R(s*), S_{P→R} = VFE_P.

### Results (v2, 48 patients, 60 cycles)

| Phenotype | Del% | SI | r(SI↔del) |
|-----------|------|----|-----------|
| A: Young  |  0%  | 0.59 | |
| B: Elderly|  8%  | 0.56 | |
| C: Septic |  8%  | 0.55 | |
| D: Dementia| 17% | 0.55 | |
| **All**   | **8%** | 0.56 | **−0.452** |

## [1.0.0] — 2025

### Initial release

- Two-agent POMDP architecture: PatientAgent and RoomAgent (Level-1)
- Bidirectional coupling loop with KL divergence and mutual information
- Four clinical phenotypes
- Delirium criterion: ΔVFE < ε AND KL > θ
- Five publication-ready figures
