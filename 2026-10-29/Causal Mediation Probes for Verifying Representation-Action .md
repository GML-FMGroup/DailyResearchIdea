# Causal Mediation Probes for Verifying Representation-Action Coupling in Dual-System VLAs

## Motivation

Existing dual-system VLA architectures such as π0 and GR00T N1 assume that tight joint training between a VLM backbone and a diffusion/flow action head produces a causally necessary coupling between semantic representation and motion generation, but the training-free S² metric from Sensorimotor Alignment only measures static geometric alignment between representations and motion trajectories and cannot distinguish correlation from causation. This is a structural failure of the evaluation paradigm: alignment scores computed offline are invariant to whether the action head actually reads the aligned subspace, so a model can score highly on S² while its action head draws on an orthogonal, task-irrelevant feature direction. The field therefore lacks any interventional instrument that can establish whether representation-motion alignment is causally necessary (and sufficient) for action generation rather than merely correlated with it.

## Key Insight

Because the action head is a deterministic function of the backbone representation, a linear probe that predicts the action head's output from the backbone's hidden states has a well-defined mediation effect under ablation, and the drop in probe-predicted action fidelity when the aligned subspace is projected out is a direct interventional estimate of the causal path from aligned representation to action — a quantity that static alignment metrics are mathematically blind to.

## Method

**What it is.** We introduce CAP (Causal Alignment Probe), a training-light interventional protocol that takes a frozen dual-system VLA (VLM backbone f + diffusion/flow action head g) and outputs a causal mediation score: the fraction of the action head's output variance that is attributable to the backbone's motion-aligned subspace, measured by ablating that subspace and observing the induced change in action outputs against a probe that predicts those outputs.

**Load-bearing assumption (stated explicitly).** CAP assumes that the aligned subspace U (top-k right singular vectors of the ridge map from backbone hidden states to motion targets) is the same subspace the action head actually reads, so that projecting U out of the backbone features and re-running the head constitutes a valid do-intervention on the causal path from representation to action. This assumption is not guaranteed: U is defined by a least-squares fit to raw action targets, not by the head's Jacobian, and linear-probe directions are known to be non-identifiable and to differ from the directions a downstream nonlinear readout uses (Belinkov, 2022; Ravichandran et al.). If U and the head's read subspace are near-orthogonal, N and S both collapse toward zero for reasons unrelated to coupling, and CM becomes a measure of subspace misalignment between two independently-fit linear maps rather than of causal mediation. To make this assumption testable, CAP estimates the head's own input-sensitivity subspace J (top-k right singular vectors of the action head's Jacobian ∂g/∂h averaged over D) and reports the overlap fraction O = ||U^T J||_F^2 / k as a primary diagnostic; when O is low, CM is reported with a subspace-mismatch flag rather than as a point estimate. Additionally, to avoid off-manifold interventions (Hase et al., 2023), the ablation is constrained to remain on-manifold by projecting onto the complement of the data covariance's top-k directions rather than by raw U U^T subtraction (see Phase 3).

**How it works.**

```pseudocode
Input: frozen VLM backbone f, action head g, calibration dataset D (state, image, action), k=32
Output: causal mediation score CM, sufficiency score S, necessity score N, overlap O, probe R^2

# Phase 1: extract aligned subspace
1. Collect backbone hidden states H = {f(o_i)} and motion targets M = {a_i} for (o_i,a_i) in D
2. Compute S^2-style alignment: fit linear map W_align = argmin ||W H - M||^2 (ridge, lambda=1e-3)
3. Take top-k (k=32) right singular vectors of W_align -> aligned subspace basis U (d x k)

# Phase 1b: estimate head sensitivity subspace and overlap
4. For each o_i, compute Jacobian J_i = ∂g/∂h at h_i = f(o_i) via one backward pass (or finite differences with eps=1e-3 if g is non-differentiable)
5. Average J = mean_i J_i; take top-k right singular vectors of J -> sensitivity subspace basis V (d x k)
6. Report overlap O = ||U^T V||_F^2 / k (fraction of aligned subspace captured by head sensitivity; O in [0,1])

# Phase 2: train lightweight action-output probe
7. Freeze f, g. For each (o_i,a_i): record action head output y_i = g(f(o_i))
8. Train linear probe P: P(f(o_i)) -> y_i (ridge regression, closed form, no gradient)
9. Report probe R^2 on held-out split (probe fidelity check; if R^2 < 0.8, coupling is non-linear, flag)

# Phase 3: interventional ablation (necessity), on-manifold
10. For each o_i: compute h_i = f(o_i); compute data covariance Sigma = cov(H); take top-k eigenvectors of Sigma -> C (d x k)
11. Ablated h_i^perp = h_i - C C^T h_i (project out top-k covariance directions, preserving feature norm and covariance structure)
12. Recompute action output y_i^abl = g(h_i^perp)  [requires action head to accept arbitrary h; if head is end-to-end, ablate at the interface layer]
13. Necessity N = 1 - E[||y_i^abl - y_i||^2] / E[||y_i - mean(y)||^2]  (normalized output degradation)

# Phase 4: sufficiency via matched-subspace swap (sufficiency)
14. For each o_i: h_i^aligned = U U^T h_i (keep only aligned component), feed to g -> y_i^align
15. Sufficiency S = 1 - E[||y_i^align - y_i||^2] / E[||y_i - mean(y)||^2]

# Phase 5: causal mediation score
16. CM = N x S  (both required: aligned subspace must be both necessary and sufficient)
17. Report CM with flags: if O < 0.3, flag 'subspace mismatch'; if R^2 < 0.8, flag 'non-linear head'
```

**Why this design.** We chose a closed-form ridge probe over a gradient-trained MLP probe because the probe must be identifiable from a small calibration set (a few hundred trajectories) and because a closed-form solution lets us report exact probe fidelity as a gate on the mediation estimate, accepting the cost that non-linear action heads may be under-predicted and flagged rather than scored. We chose subspace projection ablation (U U^T) over random-direction ablation because projecting out a fixed k-dimensional aligned subspace holds the intervention's dimensionality constant across models, accepting the cost that the k=32 choice is a hyperparameter that must be reported and swept. We chose the product CM = N × S over an additive combination because necessity without sufficiency (ablating something the head never used) and sufficiency without necessity (keeping a redundant subspace) are both degenerate, accepting the cost that CM is sensitive to either factor being near zero. This differs from the S² proxy of Sensorimotor Alignment, which is a static correlation between representation geometry and motion geometry and has no interventional step; a domain expert would not describe CAP as a variant of S² because S² never touches the action head's outputs, whereas CAP's entire signal comes from the action head's response to a representation-space intervention. We added the head-sensitivity subspace V and overlap O because the load-bearing assumption that U equals the head's read subspace is not guaranteed, and O provides a direct empirical check: when O is low, CM is flagged as a subspace-mismatch rather than interpreted as causal mediation. We constrained the ablation to the complement of the data covariance's top-k directions (rather than raw U U^T) to keep g(h^perp) an in-distribution query, accepting the cost that this ablation is not exactly the aligned subspace but a covariance-matched surrogate.

**Why it measures what we claim.** The necessity term N measures causal necessity of the aligned subspace because projecting out C C^T (covariance-matched complement) and observing action-head degradation is equivalent to a do-intervention on the representation under the assumption that the action head reads its input linearly in the neighborhood of the data manifold and that the covariance-matched projection preserves the data distribution; this assumption fails when the head's Jacobian rotates across the data distribution, in which case N reflects local sensitivity rather than global necessity. The sufficiency term S measures causal sufficiency because retaining only the aligned component and observing preserved action outputs is equivalent to a sufficient-statistic claim under the assumption that the aligned subspace carries all motion-relevant information; this assumption fails when motion-relevant information lives in a subspace orthogonal to the S²-aligned directions, in which case S reflects redundancy rather than sufficiency. The probe fidelity R² measures the linear-mediation assumption because a high R² licenses treating the probe as a faithful surrogate for the action head under the assumption that the head is approximately linear in backbone features; this assumption fails for highly non-linear heads, in which case the mediation score is reported with a non-linearity flag rather than as a point estimate. The overlap O measures the subspace-identification assumption because a high O licenses treating U as the head's read subspace under the assumption that the top-k Jacobian directions capture the head's input sensitivity; this assumption fails when the head's sensitivity is distributed across more than k directions or when the Jacobian is rank-deficient, in which case O reflects only partial overlap and CM is flagged as a subspace-mismatch.

## Contribution

(1) CAP, a training-light causal mediation protocol that produces necessity and sufficiency scores for the representation-to-action coupling in dual-system VLAs by intervening on the backbone's motion-aligned subspace and measuring action-head response. (2) An empirical finding characterizing when S² alignment is causally load-bearing versus merely correlated, obtained by applying CAP across multiple VLM backbones and action-head families and comparing CAP scores to S² scores and closed-loop success. (3) A reusable diagnostic that flags non-linear coupling between backbone and action head via probe fidelity, giving practitioners a signal for when interventional mediation estimates are trustworthy.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | DROID multi-institution manipulation trajectories (calibration set: 512 trajectories; held-out: 128) | Real-world diversity stresses alignment subspace generalization |
| Primary metric | Spearman rank correlation of CM with closed-loop success (10 rollouts per model, 5 models) | Tests whether causal score predicts policy outcome |
| Baseline | S² static alignment proxy | Tests whether interventional step adds predictive signal |
| Baseline | Random-subspace ablation score (k=32 random directions, 10 seeds) | Tests whether aligned directions specifically matter |
| Baseline | Probe R² alone | Tests whether mediation adds beyond linear fidelity |
| Ablation-of-ours | CAP with k swept (k=8,16,32,64) and shuffled-U control | Tests sensitivity to subspace dimensionality and identity |
| Diagnostic | Overlap O = ||U^T V||_F^2 / k (head-sensitivity vs. aligned subspace) | Tests load-bearing subspace-identification assumption |
| Diagnostic | Jacobian-variation score: mean cosine distance between J_i and J_j for i≠j | Tests local-linearity assumption |

### Why this setup validates the claim

The central claim is that CAP's interventional mediation score tracks downstream policy success better than static representation-geometry proxies because it measures what the action head actually reads, not merely how features correlate with motion. The S² baseline directly tests this: both scores derive from the same backbone features, so any predictive advantage must come from CAP's ablation step touching action outputs. The random-subspace baseline isolates whether the *aligned* directions specifically carry the causal signal or whether any k-dimensional projection degrades outputs equally — if random matches aligned, the alignment step is doing no work. The probe-R² baseline tests whether mediation adds information beyond simple linear fidelity. The overlap O diagnostic directly tests the load-bearing assumption that U equals the head's read subspace: if O is low across models, CM is flagged as a subspace-mismatch and the rank-correlation claim is conditioned on high-O models. The Jacobian-variation diagnostic tests the local-linearity assumption: high variation indicates that N reflects local sensitivity rather than global necessity. Closed-loop success rank correlation is the right metric because the claim is predictive about behavior, not about representation geometry; a score that ranks policies correctly on real rollouts is exactly what the proxy is supposed to deliver, and rank correlation is invariant to the score's arbitrary scale. The calibration set of 512 trajectories and 10 rollouts per model are pre-registered to scope the study to what is runnable on 8 A100 GPUs (estimated 48 GPU-hours total).

### Expected outcome and causal chain

**vs. S² static alignment proxy** — On a case where two backbones have near-identical S² geometry but one yields a VLA whose action head ignores the aligned subspace (e.g., the head learned to read orthogonal features), the baseline ranks them as equivalent because it never queries the head. Our method ablates C Cᵀ and observes the action head's degradation, so the ignoring model gets low N and low CM. We expect a noticeable rank-correlation gap concentrated on model pairs with similar S² but divergent head behavior, and near-parity on pairs where S² and head usage agree. The overlap O diagnostic should be low for the ignoring model, confirming the subspace-mismatch interpretation.

**vs. Random-subspace ablation score** — On a case where the aligned subspace is genuinely motion-relevant, random k-dimensional ablations will degrade outputs only mildly and inconsistently because they mostly hit directions the head ignores. Our aligned ablation removes the specific directions the head reads, producing larger and more consistent degradation. We expect CM to exceed the random-ablation score by a clear margin on high-alignment models, with the gap shrinking toward zero on models where alignment is weak — the diagnostic signature that the aligned directions, not dimensionality, drive the effect. The random-subspace baseline is run with 10 seeds and we report mean ± std.

**vs. Probe R² alone** — On a case where the head is linear in features but the aligned subspace is a small fraction of what the probe uses, probe R² stays high while the aligned subspace is neither necessary nor sufficient. The baseline reports confident fidelity and ranks the model well, missing the mediation failure. Our method's N and S drop, pulling CM down. We expect CM to disagree with probe R² specifically on models with high fidelity but low mediation, and to agree elsewhere — a divergence pattern that would confirm mediation is not redundant with fidelity. The overlap O diagnostic should be high for these models (head reads some subspace, but not the aligned one), distinguishing this failure from the subspace-mismatch case.

### What would falsify this idea

If CAP's rank correlation with closed-loop success is no better than S²'s, or if the random-subspace ablation matches the aligned ablation across models, the claim that interventional mediation on the aligned subspace captures something S² misses would be wrong. A uniform gain across model pairs rather than a gain concentrated where S² and head usage diverge would likewise indicate the mechanism is not doing what we claim. Additionally, if the overlap O is uniformly low across all models, the load-bearing subspace-identification assumption is violated and CM cannot be interpreted as causal mediation.

## References

1. Sensorimotor Alignment: A Platonic Proxy for Embodied Policy Success
2. RoboArena: Distributed Real-World Evaluation of Generalist Robot Policies
3. Robot Learning as an Empirical Science: Best Practices for Policy Evaluation
4. π0: A Vision-Language-Action Flow Model for General Robot Control
5. BridgeData V2: A Dataset for Robot Learning at Scale
6. Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware
7. RT-1: Robotics Transformer for Real-World Control at Scale
