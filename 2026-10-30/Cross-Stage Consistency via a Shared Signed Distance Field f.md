# Cross-Stage Consistency via a Shared Signed Distance Field for 3D Scene Generation

## Motivation

Staged 3D scene generation pipelines, such as Agentic 3D Scene Generation with Spatially Contextualized VLMs, produce a chain of explicit spatial intermediates (point clouds, BEV layouts, lifted 3D boxes, refined meshes) that are not tied to a shared metric-grounded invariant. Without such an invariant, errors introduced in one stage are undetectable from another, and the composed pipeline has no guarantee that its final output is consistent with its first-stage input—a cross-stage identifiability failure that recurs across layout-based and agentic generation methods.

## Key Insight

A single globally-optimized signed distance field (SDF) provides a metric-grounded invariant whose differentiable residuals against any intermediate representation enable stage-local repair by projection, making cross-stage inconsistencies detectable and correctable without a learned verifier.

## Method

**What it is**

We introduce **SDF-Stitch**, a framework that maintains a single globally-optimized signed distance field (SDF) over the scene as a shared metric anchor. Every explicit intermediate produced by a staged 3D scene generator—point cloud, BEV layout, lifted 3D boxes, refined mesh—is treated as a queryable projection of this SDF. SDF-Stitch detects cross-stage inconsistencies by measuring each stage's residual against the SDF and repairs the offending stage by projecting its output back onto the SDF manifold.

**Load-bearing assumption (stated explicitly):** SDF-Stitch assumes that the SDF f_θ, after being updated on stage i's intermediate X_i, remains a metric-grounded reference against which later stages can be validly judged—i.e., that the residual r_j = mean|f_θ(x)| for j > i is dominated by stage-j error rather than by SDF estimation error absorbed from X_i. This assumption is load-bearing because the update L_sdf fits f_θ to d_i(x) computed by nearest-neighbor from X_i itself; if X_i is noisy or self-intersecting, f_θ learns a smoothed version of that error, and r_i is near zero by construction. We make this assumption explicit and add a calibration diagnostic (step 4 below) that separates SDF estimation error from stage residual, so that the detector is not tautologically calibrated to the error it is supposed to catch.

**How it works**

```pseudocode
# Input: staged generator G with stages s_1..s_K (e.g., point cloud, BEV, lifted 3D, refined layout)
# Output: consistent intermediate chain and final scene

1. Initialize SDF f_θ as a coordinate MLP (8 layers, 256 hidden units, positional encoding) with θ random.
2. For each stage i = 1..K:
   a. Run stage i to obtain intermediate X_i (points, BEV cells, boxes, or mesh vertices).
   b. Compute residual r_i = mean_{x in X_i} |f_θ(x)| (or task-specific residual for boxes: sum of |f_θ(corner)|).
   c. If r_i > τ_i (τ_i = 0.05 for points, 0.1 for boxes):
        Repair X_i by gradient descent on L_proj = sum_{x in X_i} f_θ(x)^2 for 50 steps with lr=1e-3,
        projecting each element onto the zero level set of f_θ.
   d. Update f_θ by minimizing L_sdf = sum_{x in X_i} (|f_θ(x)| - d_i(x))^2 + λ * E_{x~uniform}[(||∇f_θ(x)|| - 1)^2],
      where d_i(x) is the distance from x to the nearest surface implied by X_i (computed via nearest-neighbor),
      λ = 0.1, for 200 iterations. This aligns the SDF with the repaired intermediate.
3. After all stages, return the final scene and the globally consistent SDF f_θ.
4. Calibration diagnostic (runs alongside steps 2–3, does not alter the training procedure):
   a. Hold out a random 10% of X_i before step 2d; fit f_θ on the remaining 90% for the same 200 iterations.
   b. Compute held-out residual r_i^heldout = mean_{x in held-out 10%} |f_θ(x)| and self-fit residual r_i^self = mean_{x in X_i} |f_θ(x)| after step 2d.
   c. Report normalized residual r_i / σ_SDF, where σ_SDF = std of |f_θ(x)| over 512 uniform samples in the scene bounding box (a proxy for SDF estimation uncertainty).
   d. If r_i^self < 0.1 * r_i^heldout, flag that the SDF has absorbed X_i's error (self-consistency failure) and report r_i^heldout as the stage residual instead of r_i^self.
```

**Why this design**

We chose a single globally-optimized SDF over per-stage learned verifiers because the SDF provides a differentiable, metric-grounded object that can be queried by any intermediate representation without requiring stage-specific calibration, accepting the cost that the SDF must be re-optimized as new stages are added. We chose projection-based repair over retraining the generator because projection is a closed-form geometric operation that directly enforces consistency on the SDF manifold, accepting the cost that repair may slightly distort the stage's output when the SDF is inaccurate. We chose to update the SDF after each stage rather than only at the end because early stages should anchor the SDF to coarse geometry, accepting the cost that early errors can propagate if the first stage is severely wrong. We added the held-out calibration diagnostic (step 4) because the update L_sdf fits f_θ to X_i itself, so r_i is near zero by construction; the held-out residual r_i^heldout and the normalized residual r_i/σ_SDF provide a check that the SDF has not simply absorbed the stage error it is supposed to detect. This design is not a variant of Agentic 3D Scene Generation with Spatially Contextualized VLMs, which uses a VLM agent to iteratively update a spatial context but lacks a shared metric invariant; SDF-Stitch introduces a globally-optimized SDF that ties all intermediates together and enables projection-based repair, a mechanism absent from that work.

**Why this measures what we claim**

The residual r_i = mean |f_θ(x)| measures cross-stage inconsistency because the SDF is optimized to be a globally consistent distance field, so any intermediate that deviates from the zero level set indicates a stage-local error; this assumption fails when the SDF itself is inaccurate due to sparse or noisy stage outputs, in which case r_i reflects SDF estimation error rather than stage inconsistency—the held-out residual r_i^heldout and normalized residual r_i/σ_SDF (step 4) are the diagnostics that separate these two cases. The projection repair measures consistency restoration because gradient descent on f_θ(x)^2 moves points to the nearest zero-crossing, which is the closest consistent configuration under the SDF metric; this assumption fails when the SDF has multiple zero-crossings near a point, in which case projection may choose a locally consistent but globally wrong surface. The SDF update loss L_sdf measures alignment between the SDF and the intermediate because it fits the SDF to the distance-to-surface implied by the stage output while enforcing the Eikonal constraint; this assumption fails when the stage output is itself inconsistent (e.g., self-intersecting boxes), in which case the SDF learns a smoothed version that may not reflect the true geometry—the held-out calibration (step 4d) flags this case when r_i^self < 0.1 * r_i^heldout.

## Contribution

['A framework, SDF-Stitch, that introduces a single globally-optimized signed distance field as a shared metric-grounded invariant across all stages of a 3D scene generation pipeline, enabling differentiable residual-based detection and projection-based repair of cross-stage inconsistencies.', 'A projection-based repair mechanism that corrects stage outputs by projecting them onto the SDF manifold, avoiding learned verifiers and their calibration issues.', 'An empirical demonstration that SDF-Stitch reduces cross-stage inconsistency and improves final scene quality compared to staged pipelines without a shared invariant.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | Multi-stage indoor scenes with per-stage intermediates, constructed from 200 synthetic scenes (procedural room layouts with known ground-truth SDF) and 50 real indoor scenes from ScanNet with posed depth for ground-truth SDF fitting | Enables residual measurement across stages and provides a held-out ground-truth SDF for calibration |
| Primary metric | Cross-stage SDF residual reduction, reported as both self-fit residual r_i^self and held-out residual r_i^heldout (step 4), plus normalized residual r_i/σ_SDF | Directly measures claimed consistency gain while separating SDF estimation error from stage residual |
| Secondary metrics | Collision/penetration rate (fraction of object pairs with mesh intersection), physical plausibility score (VLM preference on final renders, 2AFC), and wall-clock cost per scene | Tests whether the invariant changes practical outcomes, not just an internal residual |
| Baseline | Agentic 3D Scene Generation (VLM agent) | Tests claim of shared metric invariant |
| Baseline | Direct Numerical Layout Generation | Tests claim of geometric repair over constraint optimization |
| Baseline | Naive staged generator, no stitching | Isolates contribution of SDF anchoring |
| Baseline | TSDF volumetric fusion of all stage outputs (Curless & Levoy) | Tests whether SDF-Stitch does anything a standard TSDF fusion of point intermediates would not, especially for non-point intermediates (boxes, BEV) via task-specific residuals |
| Ablation-of-ours | SDF-Stitch without projection repair | Tests whether repair drives consistency |
| Ablation-of-ours | SDF-Stitch with self-fit residual only (no held-out calibration) | Tests whether the calibration diagnostic is necessary to avoid tautological residuals |

### Why this setup validates the claim

The central claim is that a single globally-optimized SDF, used both as a residual detector and a projection-repair target, reduces cross-stage inconsistency more than either an agentic VLM loop or constraint-based layout optimization. The Agentic baseline tests the sub-claim that a spatially contextualized VLM without a shared metric invariant cannot enforce geometric consistency: if SDF-Stitch's gain over it is concentrated on stages where the VLM's spatial context drifts, the invariant is doing the work. The Direct Numerical Layout baseline tests the sub-claim that projection onto a metric manifold beats predefined-constraint optimization when stages conflict. The TSDF fusion baseline tests whether SDF-Stitch's task-specific residuals for boxes and BEV cells provide a measurable advantage over standard volumetric fusion of point intermediates. The no-stitch baseline isolates whether the SDF anchoring itself, rather than the underlying generator, causes the improvement. The no-repair ablation isolates whether projection repair, versus mere residual detection, is the causal mechanism. The no-calibration ablation isolates whether the held-out residual is necessary to avoid the self-consistency failure where r_i^self is near zero by construction. Cross-stage SDF residual is the right primary metric because it is the exact quantity the method optimizes and the quantity that operationalizes "consistency" as a falsifiable, stage-agnostic number; reporting both self-fit and held-out residuals, plus the normalized residual r_i/σ_SDF, ensures the metric reflects stage inconsistency rather than SDF estimation error. The synthetic scenes with known ground-truth SDF allow direct measurement of SDF estimation error and validation that the held-out residual is predictive of true geometric error.

### Expected outcome and causal chain

**vs. Agentic 3D Scene Generation (VLM agent)** — On a case where the VLM agent's spatial context drifts after several edit rounds, the baseline produces a scene whose later stages contradict earlier ones because its spatial context is a text/image memory without a metric invariant. Our method instead measures each stage's residual against a globally optimized SDF and projects drifting stages back onto its zero level set, so we expect a noticeable held-out residual gap on multi-edit scenes but near-parity on single-pass scenes where drift cannot accumulate.

**vs. Direct Numerical Layout Generation** — On a case where open-vocabulary instructions force a layout that conflicts with predefined constraints, the baseline produces a layout that satisfies constraints but misaligns with lifted geometry because constraint optimization has no differentiable geometric anchor. Our method instead repairs the conflicting stage via gradient descent on the SDF, so we expect a clear held-out residual advantage on open-vocabulary conflict cases and parity on constrained in-distribution layouts.

**vs. Naive staged generator, no stitching** — On a case where an early point-cloud stage is noisy, the baseline propagates that noise into BEV and box stages because nothing re-anchors later stages to a global metric. Our method instead re-optimizes the SDF after each stage and repairs downstream stages against it, so we expect the held-out residual gap to grow with stage depth, being small at stage one and largest at the final refined stage.

**vs. TSDF volumetric fusion** — On a case with non-point intermediates (boxes, BEV cells), the TSDF baseline must first convert them to points, losing task-specific residuals; our method computes box residuals as sum of |f_θ(corner)| and BEV residuals directly, so we expect a held-out residual advantage on box and BEV stages, with parity on point-cloud stages.

### What would falsify this idea

If the no-repair ablation matches SDF-Stitch's held-out residual reduction, then projection repair is not the causal mechanism and the central claim is wrong. If SDF-Stitch's gain is uniform across single-pass and multi-edit subsets rather than concentrated where the predicted drift/conflict failures occur, the invariant is not doing the claimed work. If the held-out residual r_i^heldout is not predictive of ground-truth geometric error on synthetic scenes (e.g., correlation < 0.5), then the SDF residual measures SDF estimation error rather than stage inconsistency, and the calibration diagnostic fails to rescue the load-bearing assumption.

## References

1. Agentic 3D Scene Generation with Spatially Contextualized VLMs
2. Direct Numerical Layout Generation for 3D Indoor Scene Synthesis via Spatial Reasoning
