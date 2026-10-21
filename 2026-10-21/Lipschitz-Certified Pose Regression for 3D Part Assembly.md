# Lipschitz-Certified Pose Regression for 3D Part Assembly

## Motivation

Across the convergent branches of 3D part assembly, pose regression is trained on clean, complete targets and robustness to partial or noisy inputs is hoped for via data augmentation rather than enforced structurally. PART's set-prediction pose head and Rectified Point Flow's velocity transport both assume unposed, complete inputs, so partiality silently corrupts the pose estimate with no mechanism tying error growth to corruption magnitude. The field lacks a stability invariant that makes pose error monotone in input corruption, which is the minimal structural condition for trustworthy assembly under imperfect perception.

## Key Insight

Pose error is guaranteed monotone in a measurable corruption statistic because the pose-regression map is architecturally constrained to a certified local Lipschitz constant, turning robustness from a data-dependent hope into a structural property that also yields a calibrated per-part error certificate at inference.

## Method

**What it is.** We introduce LCPR (Lipschitz-Certified Pose Regression), a wrapper around a PART-style transformer pose head. LCPR takes a target point cloud, a retrieved part library, and a per-part corruption statistic (fraction of missing points, noise sigma) and outputs 6-DoF poses together with a per-part error certificate that is a closed-form function of the corruption statistic and the certified Lipschitz constant.

**How it works.** (B) The mechanism is a constrained training and inference procedure with three coupled components:

```pseudocode
# Inputs: target cloud X (possibly partial/noisy), part library L, per-part corruption stats c_i
# Hyperparameters: Lipschitz target L_target=2.0, consistency weight lambda=0.5, corruption grid K=8

1. PART backbone (frozen after warmup) produces per-part features f_i for retrieved parts.
2. Pose head H: f_i -> 6-DoF pose p_i. Constrain H to have spectral norm <= L_target:
     after every optimizer step, project each weight matrix W of H via W <- W * min(1, L_target / ||W||_2).
     (||W||_2 computed by power iteration, 5 iterations.)
3. Corruption-consistency loss: for each training shape, generate K corrupted variants X^(k)
   by masking a random fraction r_k ~ U[0, 0.6] of points and adding Gaussian noise sigma_k ~ U[0, 0.02].
   For each variant, compute measured corruption statistic c_k = r_k + 10*sigma_k.
   Let p_i^(0) be the pose predicted on the clean cloud. Enforce:
     ||p_i^(k) - p_i^(0)||_2  <=  L_target * c_k   for all i,k   (hinge loss with margin 0.1)
   plus a consistency term that minimizes the variance of p_i^(k) across k after rescaling by c_k.
   Total loss = L_pose (PART set-prediction loss) + lambda * mean_{i,k} max(0, ||p_i^(k)-p_i^(0)|| - L_target*c_k).
4. At inference: given input X and measured corruption c_i per part (from segmentation confidence
   and point-density statistics), the certificate is E_i = L_target * c_i. This is reported alongside p_i.
```

**Why this design.** We chose spectral normalization with a fixed target L_target over an unconstrained head with a learned penalty because the projection step gives a hard architectural bound on the pose map's sensitivity, accepting the cost that the head's expressivity is reduced (it cannot fit arbitrarily sharp pose jumps). We chose a corruption-consistency hinge loss over standard data augmentation because augmentation only samples corruptions without tying the resulting pose error to the corruption magnitude, accepting the cost that the loss requires generating K corrupted variants per training step (≈8× forward passes). We chose to compute the corruption statistic c = r + 10σ from measurable input properties rather than from a learned confidence head because a learned head would reintroduce the log-prob calibration anti-pattern that PART and Rectified Point Flow implicitly rely on; accepting the cost that c is a hand-designed proxy. We chose to project weights via power iteration rather than use a Lipschitz-regularized optimizer because projection is exact per step, accepting the cost of 5 extra matrix-vector products per layer per step. A domain expert would not describe this as "PART + spectral norm" because the corruption-consistency loss ties the pose error to a measured corruption statistic, which no prior assembly method does; PART's set-prediction head has no such coupling.

**Why it measures what we claim.** The certified bound E_i = L_target * c_i measures the true pose error under corruption because the assumption that the pose map's local Lipschitz constant is bounded by L_target makes the output perturbation bounded by L_target times the input perturbation; this assumption fails when the corruption statistic c_i under-estimates the true input perturbation (e.g., structured occlusion that removes an entire part's points but yields a small r), in which case E_i reflects only the measured point-level corruption and not the semantic ambiguity. The hinge loss term measures the consistency of pose predictions across corruption levels because the assumption that clean-pose prediction is an anchor makes the deviation ||p_i^(k)-p_i^(0)|| a direct proxy for corruption sensitivity; this assumption fails when the clean pose itself is wrong (e.g., symmetric parts), in which case the hinge loss enforces consistency around an incorrect anchor. The spectral projection measures the architectural Lipschitz constant because the assumption that the pose head is the only Lipschitz-relevant component (backbone frozen, retrieval fixed) makes the product of spectral norms an upper bound on the map's sensitivity; this assumption fails when the backbone's features change non-smoothly with input corruption, in which case the bound is loose and the certificate becomes conservative.

**(D) Assumption A (anchor validity) and failure mode F.** The quantity ||p_i^(k) - p_i^(0)|| measures true pose error under corruption only under assumption A: the clean-input pose p_i^(0) is correct (anchor validity). This assumption fails (failure mode F) for symmetric or ambiguous parts where the clean pose is itself wrong; in that case ||p_i^(k) - p_i^(0)|| measures deviation from an incorrect anchor, not true error, and the hinge loss enforces consistency around that incorrect anchor. We operationalize A by stratifying calibration results by anchor correctness (using ground-truth clean pose error as a covariate) and reporting the bound conditional on A.

## Contribution

['A Lipschitz-certified pose regression framework (LCPR) that enforces a hard spectral-norm constraint on the pose head and a corruption-consistency hinge loss, producing per-part error certificates that are closed-form functions of a measured corruption statistic.', 'A design principle showing that tying pose error to a measurable corruption statistic via a hinge loss yields certificates that are monotone in corruption by construction, avoiding the need for post-hoc calibration of confidence scores.', "An empirical demonstration on PART's 80K-shape dataset that LCPR maintains pose accuracy under partial and noisy inputs while providing certificates whose empirical coverage tracks the certified bound."]

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | PART 80K-shape assembly benchmark | Standard retrieval-assembly setting with part libraries |
| Primary metric | Pose error vs. measured corruption c (calibration curve) | Directly tests whether E_i tracks true error |
| Baseline | PART transformer pose head | Unconstrained head, no certificate |
| Baseline | PART + data augmentation only | Tests whether coupling error to c matters |
| Baseline | Rectified Point Flow | Strong SOTA pose regressor, no corruption certificate |
| Baseline | Generic Lipschitz-constrained regressor (spectral norm on head, no corruption-consistency hinge) | Isolates whether spectral norm alone suffices for calibration |
| Baseline | Learned confidence head (MLP on features predicting per-part error) | Tests whether a learned head matches the certified bound |
| Ablation-of-ours | LCPR without spectral projection | Isolates the architectural Lipschitz bound |
| Ablation-of-ours | LCPR with c_i = ||Δf_i|| (measured feature-space perturbation) | Tests whether the certificate holds when c_i is a measured feature-space statistic |
| Downstream metric | Certificate-gated assembly success rate (re-scan/re-query when E_i > threshold) | Quantifies end-to-end benefit of the certificate |

### Why this setup validates the claim

The central claim is that a certified bound E_i = L_target * c_i tracks true pose error under measured corruption, and that this tracking comes from coupling pose sensitivity to a measurable corruption statistic rather than from better clean-input accuracy. The PART benchmark supplies the retrieval-assembly distribution where corruption statistics are naturally measurable (missing points from partial scans, sensor noise), so the calibration curve of true pose error against E_i becomes a falsifiable object: a well-calibrated method produces a curve whose slope stays under L_target and whose ordering by c_i predicts which parts fail. PART's unconstrained head tests the sub-claim that hard spectral bounding is needed for the bound to hold; PART with augmentation-only tests whether merely sampling corruptions, without tying deviation to c, suffices; Rectified Point Flow tests whether a strong but certificate-free SOTA regressor already exhibits the calibration we claim to provide. The generic Lipschitz-constrained regressor (spectral norm on head, no hinge) tests whether spectral norm alone yields calibration, isolating the corruption-statistic coupling as the novel contribution. The learned confidence head tests whether a learned predictor of error matches the certified bound, addressing the anti-pattern we avoid. The no-projection ablation isolates the architectural bound from the consistency loss, so any calibration gain can be attributed to the mechanism we name. The feature-space c_i ablation tests whether replacing the hand-designed c with a measured feature-space perturbation ||Δf_i|| yields a tighter, valid certificate. The calibration metric, not raw MSE, is the right detector because the claim is about the relationship between predicted error and true error, not about minimizing either alone. The downstream certificate-gated assembly metric quantifies the practical benefit: using E_i to trigger a re-scan/re-query when E_i exceeds a threshold should reduce catastrophic misassemblies compared to a fixed-confidence threshold, demonstrating that the certificate is actionable.

### Expected outcome and causal chain

**vs. PART transformer pose head** — On a case where a retrieved part is heavily occluded (high missing-point fraction, so large c_i), the baseline produces a confident pose whose error is uncorrelated with the amount of missing input because its unconstrained head has no architectural reason to keep sensitivity bounded, so it can jump sharply on small feature changes. Our method instead projects the head to spectral norm ≤ L_target and trains a hinge that forces deviation to scale with c_i, so we expect the baseline to show a flat, non-monotone error-vs-c curve while ours is monotone with slope near but under L_target, with the gap concentrated in the high-c tail.

**vs. PART + data augmentation only** — On a case where the same part is presented at several corruption levels, the baseline produces poses whose errors vary with corruption but are not ordered by the measured c_i, because augmentation samples corruptions without ever enforcing a bound relating pose deviation to corruption magnitude. Our method instead adds the corruption-consistency hinge and variance term, so the pose deviation is explicitly tied to c_k, and we expect the augmentation-only baseline to have comparable clean-input accuracy but a visibly worse calibration slope and higher variance of error within a fixed c bin, while ours stays close to the L_target line.

**vs. Rectified Point Flow** — On a case where a symmetric or partially ambiguous part is corrupted, the baseline produces a low-error pose on clean inputs but an error that under corruption does not track any reported quantity, because it regresses poses with ad-hoc symmetry handling and emits no corruption-linked certificate. Our method instead reports E_i and we expect it to be conservative: slightly higher clean-input error than the SOTA regressor but a calibration curve that stays under the bound, so the observable signal is a small clean-accuracy deficit for us paired with a large advantage in the fraction of parts whose true error falls under E_i at high c.

**vs. Generic Lipschitz-constrained regressor (spectral norm on head, no corruption-consistency hinge)** — On a case where corruption level varies across parts, the baseline has a bounded head but no training signal tying pose deviation to c_i, so its error-vs-c curve may be flat or non-monotone despite the spectral bound. Our method adds the hinge loss that explicitly enforces ||p_i^(k)-p_i^(0)|| ≤ L_target * c_k, so we expect the baseline to show poor calibration (slope far from L_target, high variance within c bins) while ours is well-calibrated, demonstrating that the corruption-statistic coupling, not spectral norm alone, produces the certificate.

**vs. Learned confidence head (MLP on features predicting per-part error)** — On a case where the learned head is trained to predict error, the baseline may achieve low mean absolute error in its predictions but can be overconfident on high-c parts because it is not constrained by a certified bound. Our method instead provides a closed-form certificate E_i = L_target * c_i that is guaranteed to upper-bound true error under assumption A, so we expect the learned head to have occasional large underestimates (true error > predicted) while our certificate has a lower violation rate (fraction of parts with true error > E_i) at high c.

**vs. LCPR without spectral projection** — On a case where the head is unconstrained but trained with the corruption-consistency hinge, the baseline may fit the hinge on training corruptions but lack a hard architectural bound, so at test time with unseen corruption patterns its sensitivity can exceed L_target. Our method projects weights to spectral norm ≤ L_target, so we expect the no-projection ablation to show calibration curves that exceed the L_target line at high c, while ours stays under, confirming the architectural bound is necessary.

**vs. LCPR with c_i = ||Δf_i|| (measured feature-space perturbation)** — On a case where the hand-designed c_i = r + 10σ under-estimates the true feature-space perturbation (e.g., structured occlusion), the baseline certificate E_i = L_target * c_i may be violated. Our method with c_i = ||Δf_i|| uses the actual L2 change in per-part backbone features between clean and corrupted inputs, which is a direct measure of the perturbation seen by the pose head, so we expect the feature-space certificate to have a lower violation rate and tighter bound (smaller E_i while still upper-bounding true error) than the hand-designed c_i, especially under structured occlusion.

**Downstream certificate-gated assembly** — On a case where a part has high E_i (uncertain pose), a fixed-confidence threshold may accept a bad pose and cause catastrophic misassembly. Our method uses E_i to gate a re-scan/re-query action: if E_i > threshold, the system requests additional views or re-retrieves the part. We expect certificate-gated assembly to reduce catastrophic misassemblies (defined as pose error > 0.1 rad or > 5 cm) by at least 30% relative to a fixed-confidence threshold at matched re-scan budget, demonstrating the end-to-end utility of the certificate.

### What would falsify this idea

If the calibration curve of true error against E_i is no better for LCPR than for the augmentation-only baseline, or if the no-projection ablation matches LCPR's calibration, then the spectral bound and corruption coupling are not doing the work we claim and the central claim is wrong. A uniform gain across all corruption levels rather than a gain concentrated in the high-c tail would likewise indicate the effect is not produced by the proposed mechanism. Additionally, if the feature-space c_i = ||Δf_i|| does not yield a valid upper bound (i.e., true error frequently exceeds L_target * ||Δf_i||), then the certificate is not sound even with a measured perturbation, falsifying the core claim.

## References

1. PART: Learning 3D Part Assembly and Retrieval with Transformers
2. Rectified Point Flow: Generic Point Cloud Pose Estimation
3. PartNeXt: A Next-Generation Dataset for Fine-Grained and Hierarchical 3D Part Understanding
4. PuzzleFusion++: Auto-agglomerative 3D Fracture Assembly by Denoise and Verify
5. Scaling Rectified Flow Transformers for High-Resolution Image Synthesis
6. Point-SAM: Promptable 3D Segmentation Model for Point Clouds
7. SAMPart3D: Segment Any Part in 3D Objects
8. Boosting Latent Diffusion with Flow Matching
