# Injective Segment-to-Part Assignment for Retrieval-Based 3D Part Assembly

## Motivation

Across the part-assembly and part-segmentation literature, segmentation labels are treated as reliable supervision rather than as an ambiguous latent variable: PART jointly trains segmentation with retrieval and can therefore converge to a self-consistent but wrong segment-to-part map, because nothing in its objective forces the map from segments to library parts to be injective under the library. The same structural assumption recurs in segmentation-centric branches (SAMPart3D, Point-SAM, PartNeXt), where discovered parts are assumed to correspond one-to-one with library parts, so a wrong-but-consistent assignment is invisible to every training signal these methods use.

## Key Insight

Injectivity of the segment-to-part map under a fixed library is a library-injectivity constraint that can be enforced as a differentiable permutation-matching objective, turning a hidden failure mode of joint segmentation-retrieval training into a constrained inference problem whose optimum is identifiable on the observed data.

## Method

**What it is.** Library-Injective Assignment (LIA) is a training-time constraint layer placed on top of a PART-style transformer that retrieves library parts and regresses 6-DoF poses from a target shape. Inputs: target point cloud, part library, and target segmentation labels (treated as an ambiguous latent variable). Outputs: a set of retrieved library parts with poses, plus a segment-to-part assignment matrix that is injective by construction on the observed data, and a pre-projection ambiguity score used at test time to flag ambiguous segmentation.

**How it works.**

```pseudocode
# Phase 1: forward pass (PART-style backbone)
S = point_encoder(target_cloud)              # per-point features
Z = segment_encoder(seg_labels)              # per-segment features (|segments| = K)
Q = library_encoder(part_library)            # per-library-part features (|library| = M)
A_logits = cross_attention(Z, Q)             # K x M assignment logits
P_hat = pose_head(transformer(Z, Q))         # K x 6-DoF poses

# Phase 2: library-injective assignment (Sinkhorn with hard one-to-one projection)
A = sinkhorn(A_logits / tau, num_iters=20)   # tau = 0.1, doubly-stochastic K x M
A_hard = hungarian_projection(A)             # binary, row/col sums = 1 where feasible

# Phase 2b: pre-projection ambiguity score (test-time detector)
# For each row i, compute margin = top1_logit - top2_logit over A_logits[i, :]
# ambiguity_score = mean_i( 1 / (margin_i + 1e-6) )  # high when rows are near-tied
# Also compute transport_gap = cost(best_perm) - cost(second_best_perm) using A_logits as costs

# Phase 3: joint loss with injectivity residual
L_pose     = chamfer(A_hard @ library_parts_transformed(P_hat), target_cloud)
L_seg      = CE(A_hard @ library_part_labels, seg_labels)
L_inj      = || A_hard^T A_hard - I_M ||_F^2   # injectivity residual (0 iff injective)
L_total    = L_pose + lambda_seg * L_seg + lambda_inj * L_inj
# lambda_seg = 1.0, lambda_inj = 0.5 (annealed from 0 over first 5k steps)

# Phase 4: test-time constrained inference
A_test = hungarian_projection(sinkhorn(A_logits / tau, 20))
# reject assembly if ambiguity_score > epsilon_amb (epsilon_amb = 2.0, calibrated on held-out split)
# flag ambiguous segmentation
```

**Why this design.** We chose a Sinkhorn-plus-Hungarian projection over a learned assignment head because the injectivity condition must hold *by construction* on the observed data, not be encouraged by a soft penalty that a learned head can satisfy self-consistently; the cost is that hard Hungarian projection is non-differentiable, so we use Sinkhorn during training and Hungarian only at test time. We chose to anneal `lambda_inj` over the first 5k steps rather than apply it from step 0 because an early hard injectivity constraint on a randomly initialized backbone collapses the assignment to identity, accepting slower early convergence in exchange for avoiding a degenerate local optimum. We chose to keep segmentation labels as an ambiguous latent variable (a soft assignment matrix `A` over library parts) rather than as hard supervision because PART's joint segmentation objective can reinforce a wrong-but-consistent map; treating `A` as latent and injecting the injectivity residual makes the wrong map detectable. We chose to reject assemblies at test time when the pre-projection ambiguity score exceeds a calibrated threshold rather than always outputting a prediction, accepting a coverage drop in exchange for flagging the exact failure mode (ambiguous segmentation) that prior work silently absorbs. Unlike PART, which optimizes segmentation and retrieval jointly with no injectivity term, LIA makes the segment-to-part map's one-to-one property a measured quantity rather than an assumed one.

**Why it measures what we claim.** The quantity `L_inj = || A_hard^T A_hard - I_M ||_F^2` measures *library-injectivity of the segment-to-part map* because the assumption that a binary assignment matrix with unit column norms is injective holds exactly when each library part is assigned to at most one segment; this assumption fails when the library contains near-duplicate parts (e.g., two symmetric bolts), in which case `L_inj` reflects library redundancy rather than assignment error. The quantity `A_hard` from Hungarian projection measures *the identified segment-to-part correspondence* because the assumption that the Sinkhorn marginals converge to a doubly-stochastic optimum holds when `tau` is small enough that the entropy-regularized optimum is near-integral; this assumption fails when `A_logits` has near-tied rows, in which case `A_hard` reflects an arbitrary tie-break rather than the maximum-likelihood correspondence. The quantity `L_seg` with `A_hard` substituted measures *segmentation consistency under the injective map* because the assumption that library part labels are complete for the observed segments holds when the library covers the target's parts; this assumption fails when the target has a part absent from the library, in which case `L_seg` reflects library incompleteness rather than segmentation error. The quantity `ambiguity_score` (mean inverse margin of `A_logits` rows) measures *pre-projection assignment ambiguity* because the assumption that a confident correct assignment yields large top1-top2 margins holds when the backbone is trained; this assumption fails when the backbone is under-confident on symmetric segments, in which case `ambiguity_score` reflects model uncertainty rather than true ambiguity, and we calibrate `epsilon_amb` on a held-out split to separate these cases.

## Contribution

(1) A library-injective assignment layer (LIA) that treats segmentation labels as an ambiguous latent variable and enforces one-to-one segment-to-library-part correspondence via a Sinkhorn-plus-Hungarian projection with an explicit injectivity residual, turning a hidden failure mode of joint segmentation-retrieval training into a constrained inference problem. (2) A design principle: injectivity of the segment-to-part map under a fixed library is a training-time measurable quantity, and annealing its weight avoids the degenerate identity-assignment optimum that a hard constraint from step 0 produces. (3) A test-time rejection rule that flags ambiguous segmentations via the injectivity residual, giving a coverage-accuracy trade-off curve that prior retrieval-based assembly methods cannot produce.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | PART 80K-shape assembly benchmark | Large, retrieval-based, matches method assumptions |
| Primary metric | Assembly success rate under ambiguity-score rejection | Directly measures the predicted failure-mode flagging |
| Baseline | PART transformer | Same backbone, no injectivity constraint |
| Baseline | Rectified Point Flow | Strong pose regression with symmetry handling |
| Baseline | Hungarian-only assignment (no Sinkhorn) | Isolates effect of differentiable soft projection |
| Ablation-of-ours | LIA without `L_inj` term | Tests whether injectivity residual is load-bearing |
| Ablation-of-ours | LIA with `L_inj` as test-time detector (instead of `ambiguity_score`) | Tests whether pre-projection score is necessary |

### Why this setup validates the claim

The central claim is that making the segment-to-part map injective-by-construction, and measuring its injectivity residual, converts a silently-absorbed failure mode (ambiguous segmentation) into a detectable and rejectable one. The PART transformer baseline shares our backbone and training data but lacks the injectivity layer, so any gap between PART and LIA isolates the injectivity contribution rather than confounding it with architecture or data. Rectified Point Flow tests whether the gain is merely from better pose regression: if LIA beats it on ambiguous-segmentation subsets but not on clean ones, the effect is attributable to assignment, not pose quality. The Hungarian-only baseline tests whether the Sinkhorn-relaxation step matters for training stability, and the no-`L_inj` ablation tests whether the residual term is actually load-bearing versus decorative. The additional ablation using `L_inj` as the test-time detector directly tests the reviewer's concern that `L_inj` is structurally near-zero for any feasible Hungarian assignment; if this ablation shows no rejection capability (i.e., `L_inj` never exceeds `epsilon` on ambiguous cases), it confirms that the pre-projection `ambiguity_score` is necessary. Success rate *under ambiguity-score rejection* is the right metric because it rewards correct flagging: a method that silently outputs wrong assemblies scores lower than one that rejects them, which is exactly the behavior the design predicts. If LIA's gain were uniform across all subsets, the metric would not discriminate the mechanism; the design only validates the claim if gains concentrate where ambiguous segmentation is predicted to occur.

### Expected outcome and causal chain

**vs. PART transformer** — On a case where two target segments are near-symmetric and the library contains two similar parts, PART produces a plausible but swapped assignment because its joint segmentation objective reinforces a wrong-but-consistent map with no term penalizing non-injectivity. Our method instead computes `ambiguity_score` from the pre-projection logits, detects the near-tied rows, and rejects the assembly, so we expect a noticeable success-rate gap concentrated on symmetric/near-duplicate subsets with parity on unambiguous subsets.

**vs. Rectified Point Flow** — On a case where segmentation is ambiguous but poses are otherwise easy, Rectified Point Flow regresses accurate poses yet commits to whichever assignment its symmetry handling prefers, because it has no mechanism to flag assignment ambiguity as distinct from pose error. Our method separates the two: pose error stays low while `ambiguity_score` rises, triggering rejection. We expect LIA and Rectified Point Flow to be near-parity on pose-only metrics but for LIA to show a clear advantage on the rejection-aware success metric on ambiguous subsets.

**vs. Hungarian-only assignment (no Sinkhorn)** — On cases with near-tied assignment logits, Hungarian-only training receives non-differentiable hard assignments and its backbone never learns to sharpen ambiguous rows, so it produces arbitrary tie-breaks that the pre-projection ambiguity score cannot distinguish from genuine errors. Our Sinkhorn-during-training path gives gradient signal that separates near-tied rows before projection, so we expect Hungarian-only to show higher `ambiguity_score` variance and more spurious rejections, while LIA shows rejections concentrated on genuinely ambiguous cases.

### What would falsify this idea

If LIA's success-rate gain over PART is uniform across ambiguous and unambiguous subsets rather than concentrated on the symmetric/near-duplicate cases where `ambiguity_score` is predicted to spike, then the injectivity layer is not detecting the claimed failure mode and the central claim is wrong. Equivalently, if the no-`L_inj` ablation matches LIA on rejection-aware success, the residual term is not load-bearing.

## References

1. PART: Learning 3D Part Assembly and Retrieval with Transformers
2. Rectified Point Flow: Generic Point Cloud Pose Estimation
3. PartNeXt: A Next-Generation Dataset for Fine-Grained and Hierarchical 3D Part Understanding
4. PuzzleFusion++: Auto-agglomerative 3D Fracture Assembly by Denoise and Verify
5. Scaling Rectified Flow Transformers for High-Resolution Image Synthesis
6. Point-SAM: Promptable 3D Segmentation Model for Point Clouds
7. SAMPart3D: Segment Any Part in 3D Objects
8. Boosting Latent Diffusion with Flow Matching
