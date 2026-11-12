# Stepwise Compositional Alignment Dynamics in Reasoning Models

## Motivation

Existing alignment analyses, including the Platonic Representation Hypothesis work of the Idea Cards, treat cross-modal alignment as a static per-layer quantity computed from a single forward pass, and therefore cannot represent how alignment evolves across the many reasoning steps that RL-induced reasoning models actually generate. This static assumption is structurally wrong for reasoning models because their intermediate representations are produced by a sequential inference process whose depth changes what the model is computing, so any per-layer average silently mixes early perceptual-grounded steps with late abstract steps and reports a number that corresponds to no actual reasoning state.

## Key Insight

Alignment between a reasoning model's intermediate states and perceptual features is a state variable of the reasoning trajectory whose step-to-step evolution is governed by a low-dimensional compositional recurrence, so the trajectory of step-wise alignments — not any single layer statistic — is the identifiable object that explains final alignment.

## Method

**What it is.** We introduce **CADENCE** (Compositional Alignment Dynamics Estimator), which takes as input a reasoning model's per-step hidden states from a DeepSeek-R1-style trajectory and a frozen perceptual encoder's feature bank, and outputs (i) a step-wise alignment curve and (ii) a fitted compositional aggregator that predicts the final-step alignment from the curve.

**Load-bearing assumption (stated explicitly).** CADENCE assumes that the vector-valued, layer-resolved step-wise alignment state — obtained by stacking CKA values across probe layers after fixed random projections — is a sufficient statistic of the reasoning trajectory's representational state, such that a first-order d=8 tanh recurrence fitted to this vector sequence captures the compositional dynamics that determine final alignment. This assumption is not derived; it is verified empirically via the diagnostics below (effective-rank check, second-order recurrence comparison, and phase-randomized surrogate control). If the vector state does not beat the d=1 mean-of-steps ablation, the compositional-dynamics claim is dropped and the contribution is recast as a descriptive step-wise alignment measurement.

**How it works.**

```pseudocode
# Inputs: reasoning model M with L layers; perceptual encoder E (frozen, e.g., DINOv2)
# Input query q with paired perceptual stimulus x (image/caption set)
# Hyperparameters: probe layer set S (every 4th layer), alignment metric = CKA, recurrence dim d=8

1. Encode perceptual features: F = E(x)                      # [n_tokens, d_E]
2. Run M on q, capturing hidden states H_t^l for each reasoning step t=1..T
   and each probe layer l in S. (A reasoning step = one delimited CoT segment,
   delimited by newline tokens; segmenter = split on '\n\n' or 'Step N:' markers.)
3. For each (t, l): a_{t,l} = CKA( P_l(H_t^l), P_E(F) )
   where P_l, P_E are fixed random projections to a shared dim (seeded, no training).
   Diagnostic: compute effective rank of P_l(H_t^l) per step; flag steps with
   effective rank < 0.1 * min(n_tokens, d_E) as degenerate (a_{t,l} reflects
   projection noise, not grounding).
4. Form vector state: a_t = [a_{t,l} for l in S]   # layer-resolved vector, not mean
5. Fit compositional recurrence over the trajectory {a_t}:
      z_t = tanh( W_a * a_t + W_z * z_{t-1} + b )     # d=8 latent bottleneck
      a_hat_{t} = v^T z_t
      a_hat_final = v^T z_T
   Train (W_a, W_z, b, v) by ridge-regularized least squares on a held-out
   split of trajectories, minimizing sum_t (a_hat_t - a_t)^2 + lambda*||W||^2, lambda=1e-2.
   Diagnostic: also fit a second-order recurrence (z_t depends on z_{t-1}, z_{t-2})
   and compare held-out R^2 to test the Markov-in-latent assumption.
6. Output: alignment curve {a_t}, fitted recurrence, predicted final alignment a_hat_final.
7. Compare a_hat_final against static per-layer baseline (mean over layers of final-layer alignment)
   via held-out R^2 and variance decomposition of residual.
```

**Why this design.** We chose a fixed random projection to a shared dimension over a learned alignment head because a learned head would absorb the very step-to-step variation we want to measure, accepting the cost that the projection is not optimal for any single step. We chose a low-dimensional (d=8) recurrent aggregator over a full attention aggregator over all steps because the hypothesis is that alignment dynamics are governed by a small number of latent factors, and a full attention aggregator would fit noise across T steps, accepting the cost that we may underfit genuinely high-dimensional dynamics. We chose ridge-regularized least squares over end-to-end gradient training because the recurrence must be identifiable from ~hundreds of trajectories without overfitting, accepting the cost that the fit is linear-in-latent rather than fully expressive. A domain expert would not describe CADENCE as a variant of layer-wise CKA probing (the Idea Cards' Platonic baseline) because that baseline produces one number per layer from one forward pass, whereas CADENCE fits a dynamical system whose state is the sequence of step-wise alignment vectors; nor is it a variant of process reward models, which score step correctness rather than representational alignment.

**Why it measures what we claim.** The step-wise CKA quantity a_{t,l} measures perceptual grounding of reasoning step t because CKA between a hidden state and a frozen perceptual feature bank is equivalent to linear representational similarity under the assumption that the random projection preserves the top-k principal subspace; this assumption fails when a step's representation is near-degenerate (rank collapse), in which case a_{t,l} reflects projection noise rather than grounding (diagnosed by the effective-rank check in step 3). The recurrence latent z_t measures the compositional state of alignment because a first-order recurrence with tanh nonlinearity is equivalent to a finite-memory aggregation of past step alignment vectors under the assumption that alignment dynamics are Markov in the latent; this assumption fails when a single late step discontinuously re-grounds the representation, in which case z_t reflects a lagged average rather than the current state (diagnosed by comparing first-order vs second-order recurrence fits in step 5). The predicted final alignment a_hat_final measures the compositional contribution of the trajectory to final alignment because it is the terminal state of a fitted recurrence, which is equivalent to a step-attributed decomposition under the assumption that the fitted recurrence captures all step-to-step dependence; this assumption fails when alignment depends on interactions between non-adjacent steps, in which case a_hat_final reflects only the first-order component (diagnosed by the second-order recurrence comparison).

## Contribution

(1) CADENCE, a compositional alignment-dynamics estimator that fits a low-dimensional recurrent model to step-wise cross-modal alignment curves and predicts final alignment from trajectory structure rather than static per-layer statistics. (2) An empirical characterization of non-monotonic alignment dynamics in reasoning models, with early reasoning steps more perceptually grounded and later steps diverging, and a variance decomposition showing this trajectory structure explains residual variance that per-layer baselines leave unexplained. (3) A released set of step-wise alignment trajectories and fitted recurrences across reasoning-model families and perceptual stimuli, enabling reuse for alignment-dynamics analysis.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | R1-style CoT trajectories on image-grounded QA (e.g., ScienceQA-style) | Provides step-delimited reasoning with paired perceptual stimuli. |
| Primary metric | Held-out R² of a_hat_final predicting observed final-step CKA | Directly tests whether recurrence captures compositional alignment dynamics. |
| Baseline | Static layer-wise CKA probing (Platonic baseline) | Tests whether step dynamics add predictive power beyond one-shot probing. |
| Baseline | Process reward model step scores | Tests whether correctness scoring substitutes for alignment dynamics. |
| Baseline | Phase-randomized surrogate recurrence (order-destroyed control) | Tests whether temporal order, not just marginal autocorrelation, carries signal. |
| Baseline | Temporal-conv aggregator over the same {a_t} curve | Tests whether recurrence is more than a convenient temporal re-parameterization. |
| Ablation-of-ours | CADENCE with d=1 and no recurrence (mean-of-steps predictor) | Isolates contribution of low-dim recurrence vs simple averaging. |
| Ablation-of-ours | CADENCE with second-order recurrence | Tests Markov-in-latent assumption via first-order vs second-order fit. |

### Why this setup validates the claim

The central claim is that step-wise representational alignment evolves as a low-dimensional compositional dynamical system, so a fitted recurrence over the alignment curve should predict final-step alignment better than any static or order-blind summary. The static layer-wise CKA baseline tests the sub-claim that dynamics add information beyond a single forward-pass probe: if CADENCE only matches it, the recurrence is superfluous. The process reward model baseline tests the sub-claim that alignment is not merely step correctness: if PRM scores predict final alignment equally well, our quantity is redundant with correctness. The phase-randomized surrogate control tests the sub-claim that temporal order matters: it preserves the marginal distribution and autocorrelation of the alignment curve while destroying the specific lag structure, so if it performs equally, the "dynamics" are just marginal statistics. The temporal-conv aggregator baseline tests whether the recurrence is merely a convenient re-parameterization of known temporal aggregation. The d=1 mean-of-steps ablation isolates whether the low-dimensional latent recurrence contributes anything beyond averaging. The second-order recurrence ablation tests the Markov-in-latent assumption: if a second-order fit significantly outperforms first-order, the first-order recurrence is misspecified. Held-out R² is the right primary metric because the claim is explicitly predictive — the recurrence must generalize to unseen trajectories, and R² decomposes into exactly the variance the compositional aggregator claims to capture. A metric like correlation would not penalize a predictor that captures only the mean; R² does.

### Expected outcome and causal chain

**vs. Static layer-wise CKA probing** — On a trajectory where early steps are weakly grounded but a mid-trajectory step sharply re-grounds the representation, the static baseline produces a single final-layer number that conflates the re-grounding with the preceding drift, because it collapses all step information into one forward pass. Our method instead fits a recurrence that assigns the re-grounding step a distinct latent transition, because z_t updates on each a_t and propagates forward, so we expect a clear R² advantage concentrated on trajectories with large step-to-step alignment swings and near-parity on flat trajectories.

**vs. Process reward model step scores** — On a case where a step is logically correct but its hidden state stays in a language-only subspace (correct reasoning, ungrounded representation), the PRM baseline produces a high step score and predicts strong final alignment, because it scores correctness rather than representational similarity. Our method instead reads the CKA curve directly, so the ungrounded-but-correct step contributes a low a_t and the recurrence carries that deficit forward, and we expect a visible gap on the correctness-alignment decoupled subset with parity where correctness and grounding co-vary.

**vs. Phase-randomized surrogate recurrence** — On a case where alignment rises monotonically then plateaus, the phase-randomized surrogate produces a fitted recurrence that predicts the trajectory mean but misses the terminal value, because destroying the specific lag structure removes the dependence structure the recurrence relies on while preserving marginal autocorrelation. Our method instead preserves temporal order, so the latent integrates the monotone rise into a terminal state near the plateau, and we expect the order-preserving R² to exceed the surrogate control by a margin that grows with the autocorrelation of the alignment curve.

**vs. Temporal-conv aggregator** — On a trajectory with a sharp mid-trajectory re-grounding, the temporal-conv aggregator can capture local patterns but lacks the recurrent state that propagates the re-grounding forward, because it aggregates a fixed window rather than maintaining a latent state. Our method instead maintains z_t across all steps, so the re-grounding influences all subsequent predictions, and we expect CADENCE to outperform the temporal-conv aggregator on trajectories with long-range dependence, with parity on short trajectories.

**vs. d=1 mean-of-steps ablation** — On a trajectory where alignment is roughly constant, the mean-of-steps predictor matches the final alignment, because averaging is sufficient when there is no dynamics. Our method instead fits a recurrence that can capture step-to-step variation, so we expect CADENCE to outperform the mean-of-steps predictor primarily on trajectories with high step-to-step alignment variance, and to match it on flat trajectories. If CADENCE does not beat the mean-of-steps ablation, the compositional-dynamics claim is dropped.

**vs. Second-order recurrence ablation** — On a trajectory where a late step discontinuously re-grounds the representation, the first-order recurrence may lag, because it only uses z_{t-1}. The second-order recurrence can capture the discontinuity via z_{t-2}, so we expect the second-order fit to outperform first-order on such trajectories. If the second-order fit does not significantly improve held-out R², the Markov-in-latent assumption is supported.

### What would falsify this idea

If CADENCE's held-out R² advantage over the static baseline is uniform across trajectories rather than concentrated on those with high step-to-step alignment variance, or if the phase-randomized surrogate control matches the order-preserving recurrence, or if the d=1 mean-of-steps ablation matches CADENCE, then the claimed compositional dynamics are not what drives final alignment and the central claim is wrong.

## References

1. Where Does Alignment Emerge and What Does It Mean? A Closer Look at the Platonic Representation Hypothesis
