# Capability-Matched Causal Interventions for Testing Monotonicity of Representation Alignment

## Motivation

Prior work on the Platonic Representation Hypothesis (e.g., 'Where Does Alignment Emerge and What Does It Mean?') reports capability-alignment trends that are purely correlational: models at different capability levels differ in depth, width, attention pattern, and training objective simultaneously, so any observed monotonic trend cannot be attributed to capability itself. This structural confound means the field's central claim — that better models align more — rests on an untested assumption that capability is comparable across architectures, and no existing study has held architectural inductive bias fixed while varying only capability, so the trend could equally be produced by architecture-specific biases in representation geometry.

## Key Insight

Alignment monotonicity is only identifiable as a capability effect when architectural inductive bias is a fixed point of the model family, so that capability is the sole degree of freedom along which representational geometry can vary; a matched-family intervention design converts the correlational trend into a causal one by construction.

## Method

**A. What it is.** We introduce **CAPFIX** (Capability-Architecture-Fixed Intervention eXperiment), a controlled training-and-measurement protocol that produces a matched family of models sharing an identical architectural inductive bias (same sliding-window attention pattern, same depth/width ratio, same training objective, same tokenizer) but spanning a capability axis (compute scale × data quality), then measures whether cross-modal and same-modality representation alignment increases monotonically along that axis. Inputs: a base architecture specification and a capability ladder; outputs: an alignment-vs-capability curve with an architecture-held-fixed guarantee, plus a confound-decomposition of any residual non-monotonicity.

**B. How it works.**

```pseudocode
# Phase 1: Fix the inductive bias
arch = {attention: sliding_window(w=4096), depth_width_ratio: r*=1.0, objective: next_token, tokenizer: T}
# w, r*, T frozen across ALL runs; only capability knobs move

# Phase 2: Build the capability ladder (matched family)
# 5 levels, 10M-300M params, single shared tokenizer/objective checkpoint family
capability_levels = [C1: 10M params, 2.5e18 FLOPs, caption_score>=0.8,
                     C2: 30M params, 7.5e18 FLOPs, caption_score>=0.8,
                     C3: 70M params, 1.75e19 FLOPs, caption_score>=0.8,
                     C4: 150M params, 3.75e19 FLOPs, caption_score>=0.8,
                     C5: 300M params, 7.5e19 FLOPs, caption_score>=0.8]
# data_quality held fixed at caption_score>=0.8 across all levels to orthogonalize compute vs. data-quality
for c in capability_levels:
    model[c] = train(arch_fixed=arch,
                     compute_scale=c.compute,      # FLOPs
                     data_quality=c.caption_score, # fixed at 0.8
                     seed=c.seed)                  # 3 seeds per level
    # Measured invariance certificate (replaces syntactic arch_hash check)
    erf_ratio[c] = effective_receptive_field(model[c]) / context_length
    residual_dim[c] = residual_stream_dimension(model[c])
    cka_geometry[c] = CKA(R[c], R_random_matched)  # geometry covariate
    scaling_optimality[c] = |log(params[c]) - log(params_optimal(FLOPs[c]))|  # distance from Chinchilla-optimal
    assert abs(erf_ratio[c] - erf_ratio[c-1]) < 0.05
    assert abs(residual_dim[c] - residual_dim[c-1]) < 0.05 * residual_dim[c-1]
    assert scaling_optimality[c] < 0.1  # within 10% of compute-optimal params

# Phase 3: Extract representations on a FIXED probe set
R[c] = {layer_l: extract(model[c], probe_set) for l in layers}
# probe_set: 512 fixed image-text pairs, same across all levels

# Phase 4: Alignment measurement (CKA) with matched-dimension baselines
for c, c' in consecutive_pairs(capability_levels):
    A[c,c'] = CKA(R[c], R[c'])              # cross-modal AND same-modality
    B[c,c'] = CKA(R[c], R_random_matched)   # depth/width-matched random baseline
    delta[c] = A[c,c'] - B[c,c']            # alignment above chance
    # Validate null baseline: permutation-based null
    B_perm[c,c'] = mean over 100 permutations of CKA(R[c], R_permuted[c'])
    null_gap[c] = |B[c,c'] - B_perm[c,c']|  # residual geometry gap, reported as measured quantity

# Phase 5: Monotonicity test with architecture-invariance guarantee
# Fit partial effect of capability after regressing out measured geometry covariates
for c in levels:
    delta_resid[c] = delta[c] - beta1*erf_ratio[c] - beta2*residual_dim[c] - beta3*cka_geometry[c]
monotone = all(delta_resid[c+1] >= delta_resid[c] for c in levels)
# If non-monotone: attribute residual to capability-vs-architecture via ablation
```

**C. Why this design.** We chose to freeze the full architectural inductive bias (sliding-window width, depth/width ratio, objective, tokenizer) across the entire capability ladder rather than following the common practice of comparing off-the-shelf models of different sizes, because off-the-shelf comparisons confound capability with architectural differences — accepting the cost that our family is smaller and less diverse than the frontier and cannot speak to architectures we did not train. We chose a CKA-based alignment metric with a depth/width-matched random baseline rather than raw CKA, because raw CKA is sensitive to representational dimensionality and depth, and the baseline subtracts the geometry-induced chance alignment that would otherwise masquerade as a capability effect; the cost is that the baseline is an approximation of the null geometry and may over- or under-correct. We chose a consecutive-pair monotonicity test rather than a single correlation across all models, because monotonicity is a stronger, more falsifiable condition than a positive slope and directly tests the field's implicit claim; the cost is reduced statistical power with few capability levels. A domain expert would not describe CAPFIX as a variant of prior alignment studies (e.g., the Platonic Representation Hypothesis analyses) because those compare independently trained, architecturally heterogeneous models, whereas CAPFIX's load-bearing property is the architecture-invariance assertion in Phase 2 — a design constraint no prior alignment study enforces, and without which the monotonicity claim is not identifiable.

**D. Why it measures what we claim.** The measured invariance certificate (erf_ratio, residual_dim, scaling_optimality) measures architecture-invariance because identical parameterized architecture and objective imply the only remaining degree of freedom is capability; this assumption fails when capability knobs (e.g., compute scale) force a width change that alters the residual stream dimension, in which case the certificate catches it and the run is rejected rather than silently confounded. The baseline-subtracted alignment `delta[c] = CKA(R[c],R[c']) - CKA(R[c],R_random_matched)` measures capability-induced alignment because subtracting a depth/width-matched random baseline removes geometry-induced chance similarity; this assumption fails when the random baseline's geometry differs systematically from trained-model geometry, in which case delta reflects residual geometric mismatch rather than capability — we report `null_gap[c] = |B[c,c'] - B_perm[c,c']|` as a measured quantity to quantify this failure mode. The consecutive-pair monotonicity predicate `all(delta_resid[c+1] >= delta_resid[c])` measures the causal capability-alignment trend because, given architecture invariance and orthogonalized compute vs. data-quality axes, any monotone increase in delta_resid along the ladder can only be produced by the capability axis; this assumption fails when capability levels are not ordered by a single scalar (compute and data quality move jointly and non-monotonically), in which case the predicate reflects a composite of two capability axes rather than a clean capability effect — we hold data_quality fixed at caption_score>=0.8 to orthogonalize the axes.

## Contribution

(1) CAPFIX, a controlled intervention protocol that enforces architecture-invariance across a capability ladder and thereby makes the capability-alignment monotonicity claim causally identifiable rather than correlational. (2) A design principle — baseline-subtracted CKA with a depth/width-matched random null — that separates capability-induced alignment from geometry-induced chance alignment, yielding a falsifiable monotonicity condition. (3) A confound-decomposition procedure that attributes residual non-monotonicity to either capability-axis confounding or architecture leakage, applicable to any matched model family.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | Fixed synthetic probe set (512 image-text pairs) + curated caption corpus (caption_score>=0.8) | Same probe set isolates capability from data drift; fixed data quality orthogonalizes compute vs. data-quality axes |
| Primary metric | Baseline-subtracted CKA delta (delta_resid after regressing out erf_ratio, residual_dim, cka_geometry), monotonicity across 5-level ladder | Removes geometry chance-alignment; directly tests trend along a single scalar |
| Baseline | Off-the-shelf heterogeneous model pairs | Tests whether architecture control matters |
| Baseline | Raw CKA without matched random baseline | Tests whether chance-geometry correction matters |
| Baseline | Unimodal vision-vision and language-language pairs | Tests whether cross-modal gain exceeds same-modality gain |
| Ablation-of-ours | CAPFIX with width-varying capability knobs (certificate-rejected) | Tests whether architecture invariance is load-bearing |
| Null validation | Permutation-based null: 100 permutations of CKA(R[c], R_permuted[c']) | Empirically validates the random-baseline null; reports null_gap as measured quantity |

### Why this setup validates the claim

The central claim is that once architectural inductive bias is held fixed, representation alignment rises monotonically with capability. Each baseline isolates one sub-claim. The off-the-shelf heterogeneous pair baseline tests whether architecture control is what makes the trend identifiable: if heterogeneous pairs show noisier or non-monotone alignment, architecture is a confound. The raw-CKA baseline without the matched random correction tests whether the trend survives removal of geometry-induced chance similarity; if raw CKA trends but delta does not, the effect is dimensional geometry, not capability. The unimodal baseline tests whether the cross-modal alignment gain is genuinely cross-modal rather than a generic depth-correlated geometry artifact. The ablation, which deliberately violates architecture invariance by letting compute knobs change width, is the load-bearing control: if the monotone trend persists even when the certificate fails, then architecture invariance is not doing the causal work we claim. Baseline-subtracted CKA is the right metric because it converts an ordinal similarity into a chance-corrected quantity whose monotonicity is a strong, falsifiable predicate rather than a mere positive slope. The permutation-based null validation reports null_gap as a measured quantity, so the correction is not assumed exact but empirically checked.

### Expected outcome and causal chain

**vs. Off-the-shelf heterogeneous model pairs** — On a case where two frontier models differ in sliding-window width and depth/width ratio, the baseline produces a non-monotone or noisy alignment curve because its mechanism conflates capability with inductive-bias differences. Our method instead holds the measured invariance certificate (erf_ratio, residual_dim, scaling_optimality) constant across the ladder because only compute scale moves (data quality fixed at caption_score>=0.8), so we expect a clean monotone delta_resid curve within our family, with the heterogeneous baseline showing visibly larger variance and at least one rank inversion across capability levels.

**vs. Raw CKA without matched random baseline** — On a case where a shallow and a deep model are compared, raw CKA produces inflated similarity because representational dimensionality and depth induce chance geometry alignment. Our method instead subtracts a depth/width-matched random baseline because that baseline captures the geometry-induced chance floor, so we expect raw CKA to trend upward even along our fixed-architecture ladder while delta_resid stays flat or rises modestly, revealing that part of the apparent capability effect is geometric. We additionally report null_gap = |B[c,c'] - B_perm[c,c']| to quantify residual geometry mismatch.

**vs. Unimodal vision-vision and language-language pairs** — On a case where two same-modality models are compared, the unimodal baseline produces high alignment that could be driven by shared tokenizer or modality statistics rather than capability. Our method instead measures cross-modal and same-modality alignment on the same fixed probe set with the same architecture, so we expect cross-modal delta_resid to rise with capability at least as steeply as same-modality delta_resid, with the gap between them shrinking along the ladder — the signature of convergence rather than modality-specific geometry.

### What would falsify this idea

If baseline-subtracted delta_resid is flat or non-monotone across the capability ladder while the measured invariance certificate passes, or if the width-varying ablation reproduces the same monotone trend as the invariant family, then architecture invariance is not the causal lever and the central claim is wrong. A uniform gain across all capability levels and probe subsets, rather than a trend concentrated where the predicted capability axis moves, would equally falsify the idea.

## References

1. Where Does Alignment Emerge and What Does It Mean? A Closer Look at the Platonic Representation Hypothesis
