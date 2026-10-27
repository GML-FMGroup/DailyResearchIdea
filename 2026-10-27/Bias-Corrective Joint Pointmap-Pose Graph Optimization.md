# Bias-Corrective Joint Pointmap-Pose Graph Optimization

## Motivation

HiP3R's hierarchical optimizer consumes MASt3R-style pointmaps as fixed inputs, so any systematic pointmap bias — a consistent depth/scale distortion from the feed-forward regressor — is baked into the constraint set and cannot be compensated by the global optimizer; the chunk-level Sim(3) and frame-level SE(3) edges can only rearrange already-biased residuals. This is a structural violation of the property that global optimization must compensate for local estimator bias, inherited from DUSt3R/MASt3R's assumption that pointmaps are reliable observable inputs rather than estimable latent quantities. The gap persists because no prior work backpropagates the pose-graph objective into the pointmap that generated its constraints.

## Key Insight

Treating pointmaps as latent variables whose bias is identifiable from the same cycle-consistency residuals that the pose graph already minimizes makes the pointmap correction and the graph edges two views of one shared objective, so the optimizer's global residuals become the supervision signal that fixes local pointmap bias.

## Method

**What it is.** We introduce **BPGO** (Bias-corrective Pointmap-Pose Graph Optimization), a differentiable layer that takes per-frame MASt3R pointmaps and produces jointly refined pointmaps, frame-level SE(3) poses, and chunk-level Sim(3) edges, all minimizing one hierarchical pose-graph energy. Inputs: overlapping chunks of pointmaps from the frozen feed-forward backbone. Outputs: corrected pointmaps and a globally consistent pose graph.

**How it works.**

```pseudocode
# Frozen backbone: MASt3R produces per-frame pointmaps X_i (HxWx3) and confidences c_i
# Learnable bias field: a per-frame low-rank residual R_i = A_i B_i^T + b_i
#   A_i: (H*W x r), B_i: (r x 3), r=8; b_i: (H*W x 3)
# Gauge fixing: R_i is constrained to zero mean and zero scale per frame:
#   sum_pixels R_i = 0, and ||X_i + R_i||_F = ||X_i||_F (enforced by projection after each Adam step)
# Global params: frame poses T_i in SE(3); chunk transforms S_k in Sim(3)
# Identifiability phase: for the first 50 iterations, T_i and S_k are frozen at their
#   MASt3R-initialized values (or strongly anchored with weight 1e3), so that dL/dtheta
#   is the only active gradient; after iteration 50, all parameters are optimized jointly.

for each optimization step:
    # 1. Apply learned pointmap correction (differentiable)
    Xhat_i = X_i + R_i(theta)            # theta = {A_i, B_i, b_i}

    # 2. Build frame-level residuals from corrected pointmaps
    for each overlapping frame pair (i,j):
        # project Xhat_j into frame i via relative pose
        r_ij = c_i * c_j * || pi(T_i^{-1} T_j Xhat_j) - Xhat_i ||

    # 3. Build chunk-level Sim(3) residuals (held-out anchor frames)
    for each chunk pair (k,l) sharing overlap frames:
        # align chunk pointmaps with Sim(3) S_k^{-1} S_l
        r_kl = sum over shared frames || S_k^{-1} S_l Xhat - Xhat ||

    # 4. Single hierarchical energy, jointly minimized
    L = lambda_f * sum r_ij + lambda_c * sum r_kl + lambda_r * ||theta||^2
    # lambda_f=1.0, lambda_c=0.1, lambda_r=1e-4

    # 5. Backprop through constraint generation into theta AND poses
    #    (poses frozen for first 50 iters as above)
    theta, T, S = Adam_step(L)           # lr=1e-3, 200 iters, cosine decay

return Xhat, T, S
```

**Why this design.** We chose a low-rank per-frame residual field over a full per-pixel free variable because the pointmap bias is spatially smooth (depth/scale drift), accepting that high-frequency per-pixel noise is not corrected. We chose to keep the backbone frozen and only learn the residual over fine-tuning MASt3R because the residual is identifiable from the graph residuals alone, avoiding catastrophic forgetting of the matching features that generate the constraints. We chose a single joint energy over HiP3R's two-stage (chunk-then-frame) schedule because the two-stage schedule holds the chunk layout fixed, which would block backpropagation from reaching the pointmap parameters through the chunk-level residuals; accepting a larger joint optimization problem in exchange for a single gradient path. We chose to regularize theta toward zero rather than initialize from a learned prior because the prior would reintroduce the bias we are correcting. A domain expert would not describe this as HiP3R plus a residual head: HiP3R's residual is applied to poses only and its pointmaps remain fixed inputs, whereas here the pointmap itself is a latent variable in the same energy that determines the poses. The gauge-fixing constraints (zero mean/scale per frame) and the initial frozen-pose phase are included to break the (theta, T) reparameterization degeneracy: without them, the energy can be reduced by moving T alone, leaving theta at its zero-regularized prior. The frozen-pose phase ensures that dL/dtheta is the only active gradient for the first 50 iterations, so theta is identified from the graph residuals before poses are allowed to absorb them.

**Why it measures what we claim.** The per-frame residual `R_i(theta)` measures *systematic pointmap bias* because the low-rank parameterization captures only spatially coherent depth/scale distortion, under the assumption that MASt3R bias is smooth and frame-consistent; this assumption fails when bias is high-frequency (e.g., thin structures, reflective surfaces), in which case `R_i` absorbs matching noise rather than bias. The frame-level residual `r_ij` measures *pose-graph consistency* because it is the reprojection error of corrected pointmaps under relative SE(3), under the assumption that overlapping frames observe the same rigid scene; this fails under dynamic objects, in which case `r_ij` reflects motion rather than geometric error. The chunk-level Sim(3) residual `r_kl` measures *global scale/layout agreement* because it enforces alignment of shared anchor frames across chunks, under the assumption that chunks overlap on static structure; this fails on degenerate overlaps, in which case `r_kl` is underconstrained. The joint gradient `dL/dtheta` measures *bias correctability* because it is the sensitivity of the global objective to the pointmap, under the assumption that the graph residuals are dominated by pointmap bias rather than pose error; when pose error dominates, the gradient corrects poses instead of pointmaps, and the pointmap correction reflects pose misalignment. The frozen-pose phase and gauge-fixing constraints are designed to make this assumption hold during the initial optimization, so that `dL/dtheta` identifies pointmap bias before pose error can dominate.

## Contribution

(1) A differentiable joint pointmap-pose-graph layer (BPGO) that makes MASt3R-style pointmaps latent variables optimized by the same hierarchical Sim(3)/SE(3) energy that determines poses, enabling backpropagation through constraint generation. (2) A low-rank per-frame residual parameterization of pointmap bias that is identifiable from pose-graph cycle-consistency residuals without backbone fine-tuning. (3) The design principle that a single joint energy — rather than HiP3R's two-stage fixed-layout schedule — is required for the pointmap gradient path to reach the chunk-level residuals.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Tanks and Temples (intermediate) | Long sequences with scale drift stress-test global consistency. |
| Dataset | Synthetic bias-injection (controlled) | Known smooth bias injected into MASt3R pointmaps; tests recovery. |
| Primary metric | ATE after Sim(3) alignment | Directly measures global pose accuracy without scale confound. |
| Secondary metric | Parameter recovery error (synthetic) | Measures whether BPGO recovers injected bias field. |
| Baseline | HiP3R (two-stage) | Tests claim that joint energy beats fixed chunk layout. |
| Baseline | MASt3R + global alignment | Tests claim that learned bias correction beats frozen pointmaps. |
| Baseline | DUSt3R pointmap matching | Tests claim that hierarchical Sim(3) edges beat pairwise matching. |
| Baseline | Low-rank residual supervised by direct pointmap loss | Isolates joint-energy gradient path vs. residual parameterization. |
| Ablation-of-ours | BPGO with full-rank residual | Tests claim that low-rank residual captures systematic bias only. |
| Ablation-of-ours | BPGO without frozen-pose phase | Tests whether gauge-fixing phase is necessary for identifiability. |

### Why this setup validates the claim

This design isolates each sub-claim of BPGO with a matched counterfactual. Tanks and Temples provides long, overlapping sequences where pointmap bias accumulates into global scale/drift, so ATE after Sim(3) alignment is the right metric: it penalizes the exact failure mode (global inconsistency) that the chunk-level Sim(3) residuals are designed to correct, without letting a single global scale factor mask drift. The synthetic bias-injection dataset directly tests the load-bearing assumption: by injecting a known smooth bias field into MASt3R pointmaps, we can measure whether BPGO's learned residual `R_i(theta)` recovers that field (parameter recovery error), which would confirm that the joint gradient identifies pointmap bias rather than being absorbed by pose parameters. HiP3R tests the joint-energy claim because its two-stage schedule freezes chunk layout, blocking gradient flow from chunk residuals to pointmaps; if our single energy is the cause of improvement, we should beat HiP3R specifically where chunk layout is wrong, not uniformly. MASt3R + global alignment tests whether learning a residual field adds anything over optimizing poses with frozen pointmaps; if bias correction is real, the gap should widen with sequence length. DUSt3R matching tests whether hierarchical Sim(3) edges are necessary versus pairwise pointmap matching. The low-rank residual supervised by direct pointmap loss (against GT depth/pointmaps) isolates whether the novelty is the joint-energy gradient path or merely the residual parameterization: if this baseline matches BPGO's ATE, then the joint energy is not the source of improvement. The full-rank ablation is the key falsifier: if a free per-pixel residual performs equally well, then low-rank smoothness is not what captures the bias, and our mechanism story is wrong. The ablation without frozen-pose phase tests whether the gauge-fixing phase is necessary for identifiability; if removing it collapses performance to HiP3R levels, the frozen-pose phase is load-bearing.

### Expected outcome and causal chain

**vs. HiP3R (two-stage)** — On a long sequence where early chunks accumulate scale drift, HiP3R produces a globally inconsistent pose graph because its fixed chunk layout prevents chunk-level residuals from backpropagating into pointmap parameters. Our method instead jointly minimizes frame and chunk residuals in one energy, so gradient from `r_kl` reaches `theta` and corrects the pointmap scale before it hardens into a chunk transform. We expect a noticeable ATE gap on long sequences with many chunk boundaries, but near-parity on short sequences with few boundaries.

**vs. MASt3R + global alignment** — On a scene with smoothly varying depth bias (e.g., a long corridor), the baseline produces systematic scale drift because its pointmaps are frozen and pose optimization can only rotate/translate, not correct the underlying pointmap distortion. Our method instead learns a low-rank per-frame residual that absorbs the smooth depth/scale bias, so the frame-level reprojection residuals shrink and poses align to corrected geometry. We expect the gap to grow with sequence length and to be concentrated on frames with large depth range, with parity on short, texture-rich sequences.

**vs. DUSt3R pointmap matching** — On a scene with weak pairwise overlap between non-adjacent frames, DUSt3R matching produces fragmented pose estimates because pairwise matching cannot enforce global scale/layout agreement across distant chunks. Our method instead builds chunk-level Sim(3) edges over shared anchor frames, propagating global scale consistency through the graph. We expect a clear ATE improvement on sequences with sparse pairwise overlap but dense chunk overlap, and near-parity where pairwise overlap is already dense.

**vs. Low-rank residual supervised by direct pointmap loss** — On sequences where the joint-energy gradient path is the key novelty, this baseline (which learns the same low-rank residual but supervised by a direct pointmap reconstruction loss against GT depth/pointmaps) should underperform BPGO because it lacks the global pose-graph residuals that identify bias from cycle consistency. We expect BPGO to achieve lower ATE, especially on long sequences where global consistency matters, while the direct-loss baseline may overfit to local pointmap errors. If the baseline matches BPGO, the joint-energy gradient path is not the source of improvement.

**vs. BPGO full-rank residual (ablation)** — On a scene with high-frequency pointmap noise (thin structures, reflective surfaces), the full-rank residual can absorb matching noise into `theta`, producing overfit corrections that reduce training residuals but degrade held-out pose accuracy. Our low-rank residual instead captures only spatially coherent bias, so it should generalize better. We expect the full-rank variant to show lower training energy but higher ATE on reflective/thin-structure subsets, while matching our ATE on smooth scenes.

**vs. BPGO without frozen-pose phase (ablation)** — On sequences where pose error is large at initialization, removing the frozen-pose phase allows the gradient to be absorbed by pose parameters, leaving `theta` near zero and reducing BPGO to HiP3R-like behavior. We expect the ablation to show ATE close to HiP3R on long sequences, confirming that the frozen-pose phase is necessary for identifiability of pointmap bias.

### What would falsify this idea

If BPGO's gain over HiP3R and MASt3R is uniform across sequence lengths and scene types rather than concentrated where chunk-boundary drift and smooth depth bias occur, then the claimed causal chain (joint gradient path and low-rank bias correction) is not what produces the improvement. Likewise, if the full-rank ablation matches BPGO's ATE on reflective/thin-structure subsets, the low-rank smoothness assumption is wrong. Additionally, if the synthetic bias-injection experiment shows high parameter recovery error (i.e., BPGO fails to recover the injected bias field), the identifiability claim is falsified.

## References

1. HiP3R: Hierarchical Pose Graphs for Long-Horizon 3D Reconstruction
2. Grounding Image Matching in 3D with MASt3R
3. DUSt3R: Geometric 3D Vision Made Easy
4. RoMa: Robust Dense Feature Matching
