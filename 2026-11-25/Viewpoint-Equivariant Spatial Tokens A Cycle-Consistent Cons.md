# Viewpoint-Equivariant Spatial Tokens: A Cycle-Consistent Constraint for Multi-View Geometry in MLLMs

## Motivation

S-Mind and related methods supervise multi-view fusion and viewpoint re-expression as separate objectives, so the compact spatial tokens that are supposed to encode a fused 3D scene need not be consistent with the tokens the model emits when it re-expresses that same scene from a new viewpoint. This structural gap recurs across the lineage: S-Mind reports strong fusion (65.11%) and transformation (82.17%) probe accuracies yet trains the two without any equivariance constraint, while drawing-based (Spark) and visualization-based (MVoT) approaches likewise treat visual manipulation as an auxiliary output rather than a geometry-preserving map on the internal representation. The consequence is that downstream spatial QA can be answered correctly through local heuristics even when the internal scene representation is not invertibly tied to the viewpoints the reasoner adopts.

## Key Insight

If the fused spatial-token set is required to be a group-equivariant representation of the 3D scene — re-emitting under viewpoint A then composing to B must equal re-emitting directly under the composed transformation — then the token map becomes invertible and geometry-preserving, which no amount of separate fusion/re-emission supervision can guarantee.

## Method

**A. What it is.** We introduce **VEST** (Viewpoint-Equivariant Spatial Tokens), a training-time cycle-consistency constraint layered on S-Mind's compact spatial-token architecture. VEST takes as input the fused spatial-token set `Z` produced from a multi-view observation bundle and two pose-annotated viewpoints `(v_A, v_B)` drawn from the training data's camera-pose annotations (not from the model's own reasoning traces), and outputs a scalar equivariance loss added to the existing fusion and re-emission losses. The load-bearing assumption is stated explicitly: the re-emission head `R(Z; v)` is treated as a deterministic, pose-parameterized function whose output lives in the same coordinate frame as the closed-form rigid transform `T(v_A→v_B)`, so that the residual `||T(v_A→v_B)·R(Z;v_A) − R(Z;v_B)||` is zero exactly when the token map is a group action and nonzero only when equivariance is violated. Because S-Mind's own transformation-probe accuracy is 82.17% (not 100%), we do not treat the raw residual as a pure equivariance signal; instead we calibrate it against a held-out estimate of the head's own error floor (step 5b) and we verify the metric's validity with a linear pose-recovery probe and a residual-vs-reprojection-error correlation (block D).

**B. How it works.**

```pseudocode
Input: multi-view images X = {x_1..x_K}; spatial-token encoder E; re-emission head R; 
       pose-annotated viewpoint pairs (v_A, v_B) from training-data camera poses; 
       group action T(v) on token space; margin m=0.1; weight lambda=0.5; 
       calibration set C of 512 held-out (X, v_A, v_B) triples with frozen S-Mind

1. Z = E(X)                                  # fused spatial-token set (S-Mind encoder)
2. Sample (v_A, v_B) from training-data camera-pose annotations   # not from CoT traces
3. Path-1 (direct):  Z_B_direct = R(Z ; v_B)
4. Path-2 (composed): Z_A = R(Z ; v_A);  Z_B_comp = T(v_A -> v_B) · Z_A
   # T(v_A -> v_B) is the closed-form rigid transform on token coordinates
   # induced by the pose difference between v_A and v_B
5a. Raw residual: r_i = || Z_B_comp[i] - Z_B_direct[i] ||_2  per token position i
5b. Calibrated residual: r_i_cal = max(0, r_i - m - b)   # b = head/pose noise floor
    # b is estimated on frozen S-Mind over calibration set C as the 90th percentile
    # of r_i under identity transforms (v_A = v_B), giving a scalar per token position
5c. L_equiv = mean over token positions of (r_i_cal)^2
6. L_cycle = mean || T(v_B -> v_A) · R(R(Z; v_A); v_B) - Z ||_2^2   # invertibility back to fused set
7. L_total = L_fusion + L_reemit + lambda * (L_equiv + L_cycle)
8. Update E, R by gradient descent on L_total
```

Hyperparameters: margin `m=0.1` on the per-token residual (tokens within margin contribute zero loss to tolerate sampling noise); `lambda=0.5` balances equivariance against the original S-Mind objectives; viewpoint pairs are sampled uniformly from the pose-annotated viewpoints in the training data (4–8 per multi-view bundle), not from the model's reasoning traces. The group action `T` is computed in closed form from the camera poses associated with the sampled viewpoints, so no learned transform module is introduced. The noise floor `b` is a scalar per token position estimated once on frozen S-Mind over a calibration set of 512 held-out triples with `v_A = v_B` (identity transform), taken as the 90th percentile of the raw per-token residual; this subtracts the head's own error floor so the loss measures deviation beyond it rather than absolute residual. Training cost: VEST adds two extra forward passes through `R` per step (Path-2 and the cycle term) on top of S-Mind's existing fusion and re-emission passes; on 8×A100-80GB with the Qwen3.5 backbone this is estimated at ~1.6× S-Mind's per-step wall-clock and ~1.4× its peak memory, i.e. roughly 1.6× the GPU-hours of the S-Mind training run.

**C. Why this design.** We chose a closed-form group action `T` over a learned viewpoint-transformation module because the composition law of rigid 3D transforms is exact and a learned module would reintroduce the very inconsistency we are trying to remove, accepting the cost that `T` requires camera-pose metadata at training time. We chose cycle consistency back to the fused set `Z` (step 6) in addition to the pairwise equivariance in step 5 because pairwise consistency alone admits a degenerate solution where all re-emissions collapse to a constant token set, accepting the extra forward passes through `R` that this requires. We chose to sample viewpoints from the training data's pose annotations rather than from the model's own reasoning traces because the equivariance constraint must bind on ground-truth poses; trace-sampled viewpoints are themselves unreliable (VSI-Bench shows linguistic CoT fails to improve spatial reasoning), so anchoring to them would penalize a pose distribution the model hallucinates. We accept the cost that this restricts the constraint to viewpoints present in the training annotations rather than every viewpoint the reasoner might adopt. A domain expert would not describe VEST as a variant of Spark or MVoT: those methods add visual manipulation as an auxiliary output channel (drawing operations, image visualizations) without constraining the internal token representation to be invertible, whereas VEST places the constraint directly on the internal spatial-token set and ties it to the re-emission map, which is a different surface and a different invariant.

**D. Why it measures what we claim.** The calibrated per-token residual `max(0, || Z_B_comp[i] - Z_B_direct[i] ||_2 - m - b)` measures **viewpoint equivariance of the internal scene representation** because the assumption that `R(Z; v)` is a deterministic, pose-parameterized function whose output lives in the same coordinate frame as `T` makes the two paths mathematically equal iff the token map is a group action; this assumption fails when the re-emission head is stochastic, when `T` is mis-estimated from noisy poses, or when the token space is not a linear/affine representation of 3D coordinates on which `T` acts exactly, in which case the residual reflects head error, pose noise, or representation mismatch rather than genuine equivariance violation. We verify the linear/affine assumption empirically with a linear pose-recovery probe: a ridge regression from `Z` to camera pose (rotation as quaternion, translation as 3-vector) fit on a held-out split, reporting R²; if R² is low, the token space is not metrically meaningful and the residual is not a clean equivariance signal. We further validate that the residual tracks geometry by correlating the per-token calibrated residual with ground-truth pose-reprojection error across the calibration set (Pearson r), reporting the correlation; a near-zero correlation would indicate the residual shrinks without geometric consistency (tokens encode appearance/texture rather than geometry). The cycle residual `|| T(v_B->v_A)·R(R(Z;v_A);v_B) - Z ||_2^2` measures **invertibility of the fused encoding** because the assumption that a faithful fused scene can be recovered from any two re-expressions makes the cycle identity a necessary condition for invertibility; this assumption fails when the scene contains occluded regions not observed in any input view, in which case the residual reflects missing information rather than a representation defect. The margin `m` and noise floor `b` operationalize **tolerance to viewpoint-sampling and head noise** because the assumption that pose estimation error and head error are bounded by `m` and `b` respectively makes sub-threshold residuals uninformative; this assumption fails under large pose error or a miscalibrated `b`, in which case the margin suppresses real violations or the loss penalizes head error rather than equivariance.

## Contribution

['A viewpoint-equivariance and cycle-consistency training objective (VEST) that constrains S-Mind-style compact spatial tokens to form an invertible, geometry-preserving map under the closed-form rigid-transform group action induced by reasoning viewpoints.', 'A design principle: tying fusion and re-expression supervision through a shared token-space invariant, rather than supervising them as separate heads, is what converts a downstream-QA-validated representation into a geometrically consistent one.', 'A diagnostic protocol that separates equivariance violation from occlusion-induced information loss by comparing sub-margin and super-margin cycle residuals across scenes with varying view coverage.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | Four spatial benchmarks from S-Mind | Same suite where S-Mind reported gains |
| Primary metric | Viewpoint-transformation probe accuracy | Directly measures the invariant VEST constrains |
| Secondary metric | Linear pose-recovery probe R² from `Z` (ridge regression, held-out split) | Validates that token space is a linear/affine representation of 3D coordinates on which `T` acts |
| Secondary metric | Pearson r between per-token calibrated residual and ground-truth pose-reprojection error | Validates that token-space L2 residual tracks geometric equivariance |
| Baseline | S-Mind (Qwen3.5 backbone) | Isolates VEST's added constraint |
| Baseline | Spark (drawing-based spatial reasoning) | Tests internal-token vs. output-channel surface |
| Baseline | MVoT (visualization-of-thought) | Tests constraint vs. auxiliary visual reasoning |
| Ablation-of-ours | VEST without cycle term (L_equiv only) | Tests whether invertibility term prevents collapse |
| Ablation-of-ours | VEST with `T` computed from ground-truth poses vs. estimated poses | Bounds sensitivity to pose noise before committing to full training run |
| Ablation-of-ours | VEST with `b=0` (no noise-floor calibration) | Tests whether calibration is necessary for the residual to isolate equivariance |

### Why this setup validates the claim

The central claim is that imposing a closed-form viewpoint-equivariance plus cycle-consistency constraint on S-Mind's internal spatial tokens yields a more stable, invertible scene representation and therefore better viewpoint-transformation accuracy. S-Mind is the correct primary baseline because it shares the exact token architecture, so any gain isolates the constraint rather than architecture or backbone; Spark and MVoT test the competing hypothesis that spatial ability comes from adding an output channel (drawing, visualization) rather than constraining the internal representation. The viewpoint-transformation probe is the right metric because VEST's loss is defined on re-emission under viewpoint change, so the probe is the closest behavioral readout of the constrained quantity; multi-view fusion accuracy acts as a control that should move less, since VEST does not constrain fusion directly. The linear pose-recovery probe and the residual-vs-reprojection-error correlation are the validity checks the reviewer requested: they test the load-bearing assumption that token-space L2 distance is monotone in 3D geometric discrepancy, so that the residual measures equivariance rather than appearance drift. The ablation (equivariance without cycle consistency) is the falsification lever for the degenerate-collapse concern: if the cycle term matters, removing it should degrade the transformation probe specifically while leaving fusion roughly intact. The ground-truth-vs-estimated-pose ablation bounds the sensitivity to pose noise, and the `b=0` ablation tests whether the noise-floor calibration is necessary. Together this forms a falsifiable test: the predicted pattern is a gain concentrated on the transformation probe, not uniform across all probes, and a positive residual-vs-reprojection-error correlation.

### Expected outcome and causal chain

**vs. S-Mind (Qwen3.5 backbone)** — On a case where the reasoning trace re-expresses the fused scene from a viewpoint far from any input view, the baseline produces a re-emission that drifts from the pose-implied geometry because nothing ties successive re-expressions to a consistent group action. Our method instead penalizes the calibrated residual between direct and composed re-emission via the closed-form transform, so the re-emission map is forced toward an actual group action. We expect a noticeable gap on the viewpoint-transformation probe and near-parity on the fusion probe, since fusion is not directly constrained. We also expect the linear pose-recovery probe to show a higher R² for VEST than for S-Mind, and the residual-vs-reprojection-error correlation to be positive (Pearson r > 0.3) on the calibration set, confirming the residual tracks geometry.

**vs. Spark** — On a case requiring precise continuous spatial tracking, the baseline produces drawing operations that are locally plausible but not invertible back to a coherent scene, because its mechanism adds visual manipulation as an output channel without constraining internal tokens. Our method instead constrains the internal token set to be recoverable through the cycle term, so the representation itself carries the invariant. We expect the gap to be largest on video-based and multi-view reasoning subsets where tracking must persist across steps, and small on static single-view subsets.

**vs. MVoT** — On a case where the verbal chain-of-thought fails and visualization is invoked, the baseline produces an image trace that improves coherence but does not guarantee the underlying token map is viewpoint-equivariant, because its mechanism complements verbal reasoning without an invertibility constraint. Our method instead binds the constraint directly on the fused token set, so the improvement should not depend on whether a visualization is generated. We expect a modest but consistent gain on the transformation probe even on instances where no visualization is emitted, distinguishing representation-level from output-level gains.

### What would falsify this idea

If the gain over S-Mind is uniform across the fusion and transformation probes rather than concentrated on the transformation probe, the constraint is not doing what we claim and the improvement is a generic fine-tuning effect. Likewise, if the equivariance-only ablation matches the full VEST on the transformation probe, the cycle-consistency term is not preventing the predicted degeneration and the central mechanism argument is wrong; and if the linear pose-recovery probe yields low R² or the residual-vs-reprojection-error correlation is near zero, the token-space residual does not measure geometric equivariance and the measurement claim fails.

## References

1. S-Mind: Internalizing Multi-View Geometry for Spatial Intelligence
