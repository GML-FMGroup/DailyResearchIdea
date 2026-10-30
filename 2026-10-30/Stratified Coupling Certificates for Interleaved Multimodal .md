# Stratified Coupling Certificates for Interleaved Multimodal Benchmarks

## Motivation

Benchmarks for interleaved multimodal generation such as IntReal-100K evaluate seven capability dimensions but provide no reproducible mechanism tying their sample distribution to the trillion-token interleaved corpora used for pretraining (e.g., the BAGEL data protocol), so a measured capability gap cannot be attributed to any data-curation variable. Structurally this fails because evaluation and training distributions are documented in disjoint statistic spaces: IntReal reports expert-model capability scores, while pretraining pipelines report only token counts and coarse source mixes, leaving no shared coordinates on which 'train-test closure' could even be defined. The result is that the field's persistent gap between benchmark improvement and data-scaling interventions is unidentifiable rather than merely unmeasured.

## Key Insight

If every evaluation dimension is mapped to a data-quality statistic that is computable identically on benchmark samples and on training shards, then the benchmark's sample distribution can be certified as a stratified subsample of the training manifold, which makes any capability gap attributable to a specific curation variable instead of to an opaque distribution shift.

## Method

## Coupling Certificates (CoC)

**What it is.** CoC is a reproducible data-curation and reporting protocol. Its inputs are (i) the IntReal-100K evaluation set, (ii) a manifest of MINT-1T-scale training shards, and (iii) a labeled calibration split of IntReal-100K (20% of samples, stratified by dimension, with expert-model capability scores). Its outputs are (a) a seven-dimensional quality-statistic vector per sample, (b) a per-dimension statistic-sufficiency score R²_d from a held-out calibration mapping, (c) a coupling certificate stating that the evaluation set is a stratified subsample of the training shard manifold under those statistics, restricted to dimensions marked certifiable (R²_d ≥ 0.30), and (d) a per-dimension 'curation lever' report linking capability gaps to shard-level statistic shifts for certifiable dimensions only.

**Load-bearing assumption (stated explicitly).** The seven closed-form statistics s_d are sufficient coordinates for their corresponding capability dimensions: distributional containment and distance in this 7-D statistic space is equivalent to containment and distance in capability space, making a capability gap attributable to a specific shard-level statistic shift. This assumption is not taken for granted; it is measured per dimension by R²_d and the certificate marks dimensions with R²_d < 0.30 as 'uncertifiable' rather than emitting a lever for them.

**How it works.**

```pseudocode
# Phase 0: Calibration split (statistic sufficiency)
# IntReal-100K expert scores are already available; split 80/20 stratified by dimension.
# For each dimension d, fit a held-out mapping f_d: R^7 -> R (capability score)
#   f_d = ridge regression, alpha=1.0, on the 7-D statistic vector s(x)
#   R2_d = 1 - SS_res/SS_tot on the 20% held-out split
#   certifiable_d = (R2_d >= 0.30)   # stated threshold; dimensions below are 'uncertifiable'

# Phase 1: Statistic bank S = {s_1..s_7}, one per IntReal dimension
for dim d in {coherence, alignment_density, entity_consistency,
              narrative_flow, cross_modal_reference, style_stability, factual_grounding}:
    s_d = fixed_statistic(d)   # closed-form, no learned parameters
    # e.g. s_alignment_density = #CLIP>0.28 image-text pairs / #images
    #      s_coherence = mean PMI of adjacent text spans across image boundaries
    #      s_entity_consistency = 1 - (#distinct entity strings / #mentions)

# Phase 2: Compute statistic vectors on both sides
E = { s(x) : x in IntReal-100K }            # eval manifold
T = { s(y) : y in stratified_sample(MINT-1T shards, n=200k) }  # train manifold

# Phase 3: Stratified-subsample test (the certificate), per certifiable dimension d
for each d in 1..7 with certifiable_d == True:
    bin E and T into deciles of s_d
    compute coverage c_d = min_bin( |E_bin| / |T_bin| )      # mass coverage
    compute two-sample energy distance D_d(E, T)
    compute per-bin likelihood ratio w_d(bin) = P_E(bin)/P_T(bin)
CERTIFY iff for all certifiable d: c_d >= 0.5 and D_d <= tau_d (tau_d = 0.05 * std(T_d))
# Dimensions with certifiable_d == False are reported as 'uncertifiable' with their R2_d,
# and no lever is emitted for them.

# Phase 4: Curation-lever attribution (certifiable dimensions only)
for each certifiable dimension d with a reported capability gap g_d:
    regress g_d ~ sum_beta_dk * (shard-level mean of s_k) + controls
    report beta_dk as the certified curation lever for dimension d

# Phase 5: Emit certificate JSON {statistic_bank, R2_d, certifiable_d, c_d, D_d, w_d, beta_dk, seed, shard hashes}
```

**Why this design.** We chose closed-form statistics over learned quality scorers because learned scorers cannot be recomputed identically on a 200k shard sample and on 100k eval samples without a shared training set, which would reintroduce the distributional coupling problem the protocol is meant to solve; the cost is that closed-form statistics miss semantic quality dimensions, which is why Phase 0 measures per-dimension statistic sufficiency R²_d and marks dimensions below 0.30 as uncertifiable rather than emitting a lever. We chose decile stratification with a min-bin coverage criterion over a single global two-sample test because a global test can pass while the eval set is concentrated in a few high-quality bins, which is exactly the failure that makes benchmark scores non-diagnostic; the cost is that deciles with fewer than 50 eval samples produce noisy coverage estimates, so we require a minimum bin count and report it. We chose energy distance over KL divergence because KL is undefined when eval and train supports are disjoint (common when a shard lacks a modality), whereas energy distance remains finite; the cost is lower sensitivity to tail mismatch. We chose shard-level regression for lever attribution over per-sample attribution because curation interventions act on shards, not individual samples; the cost is that within-shard heterogeneity is invisible. This differs from retrieval-style descriptor matching (e.g., Toolformer-style nearest-neighbor selection) in that CoC does not retrieve training samples for eval items; it tests a distributional containment property and attributes capability gaps to shard-level statistic shifts, which no retrieval baseline computes.

**Why it measures what we claim.** The min-bin coverage `c_d` measures the claim 'the evaluation distribution is a stratified subsample of the training manifold' because the assumption that decile bins of a quality statistic are the right strata makes equal bin mass equivalent to containment; this assumption fails when a statistic is multimodal within a decile (e.g., two distinct sub-populations with the same mean alignment density), in which case `c_d` reflects marginal mass agreement rather than manifold containment. The energy distance `D_d` measures distributional distance under the same statistic because the assumption that the statistic captures the dimension's relevant variation makes distance in statistic space equivalent to distance in capability space; this fails when a capability dimension is driven by a statistic not in the bank, in which case `D_d` reflects a lower bound on true distance. The calibration R²_d measures statistic sufficiency because the assumption that a linear ridge mapping from the 7-D statistic vector to the expert capability score captures the dimension's variation makes R²_d an upper bound on how much of the capability is recoverable from the bank; this fails when the mapping is misspecified (e.g., non-linear), in which case R²_d underestimates sufficiency and may mark a certifiable dimension as uncertifiable. The regression coefficient `beta_dk` measures the curation lever for dimension `d` because the assumption that shard-level statistic shifts are exogenous to the model makes the coefficient a causal effect estimate; this fails under confounding (e.g., shards with high alignment density also have more tokens), in which case `beta_dk` reflects an association confounded with scale.

## Contribution

(1) A reproducible data-curation protocol, Coupling Certificates (CoC), that maps each of IntReal-100K's seven evaluation dimensions to a closed-form data-quality statistic and tests stratified-subsample containment against MINT-1T-scale training shards. (2) A design principle: benchmark reports must include per-dimension coverage, energy distance, and shard-level lever coefficients, making capability gaps attributable to curation variables rather than to opaque distribution shift. (3) An open certificate schema and reference implementation with fixed seeds and shard hashes enabling independent replication of the train-test coupling claim.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | IntReal-100K eval + MINT-1T shard manifest | The protocol's native inputs; tests containment claim directly |
| Primary metric | Certification pass rate across 7 dimensions, reported separately for certifiable (R²_d ≥ 0.30) and uncertifiable dimensions | Directly measures the containment claim, not downstream accuracy |
| Baseline (global test) | Single global two-sample energy test (no decile stratification) | Tests whether min-bin coverage is necessary for diagnostic certificates |
| Baseline (learned scorer) | CLIP-score-only quality filter on shards | Tests whether closed-form bank beats a learned quality proxy |
| Baseline (retrieval match) | Toolformer-style nearest-neighbor descriptor matching | Tests whether retrieval containment substitutes for distributional containment |
| Ablation-of-ours | CoC with 3 of 7 statistics (drop coherence, entity, grounding) | Tests whether the full statistic bank is needed for lever attribution |
| Ablation-of-ours | CoC without Phase 0 calibration (fixed bank, all dimensions certifiable) | Tests whether the R²_d sufficiency gate changes certification decisions |

### Why this setup validates the claim

The central claim is that CoC's certificate identifies when an evaluation set is a stratified subsample of the training shard manifold, and that its lever report attributes capability gaps to shard-level statistic shifts. The global-test baseline isolates the value of min-bin coverage: if a single global energy test passes while CoC fails on the same eval/train pair, then the certificate's diagnostic power comes specifically from stratification, confirming the mechanism. The learned-scorer baseline tests the closed-form design choice: if a CLIP-only filter certifies pairs that CoC rejects and those pairs show larger downstream capability gaps, the closed-form bank is doing work a learned proxy cannot. The retrieval baseline tests the distinctness claim: if nearest-neighbor descriptor matching produces the same lever rankings as CoC, then CoC adds nothing over retrieval. The ablation tests whether all seven statistics are load-bearing for lever attribution. The Phase 0 calibration ablation tests whether the R²_d sufficiency gate changes certification decisions: if CoC without calibration certifies dimensions that CoC with calibration marks uncertifiable, and those dimensions show no downstream capability gap, the gate is doing work. The primary metric — certification pass rate — is the right one because the claim is about the certificate's validity, not about model accuracy; a method that certifies everything or nothing is uninformative, so we expect CoC to pass on genuinely contained pairs and fail on deliberately shifted pairs, with uncertifiable dimensions reported separately so the certificate is honest about its coordinate coverage.

### Expected outcome and causal chain

**vs. Single global two-sample energy test** — On a case where the eval set is concentrated in the top decile of alignment density while the training shards span all deciles, the global test produces a pass because aggregate energy distance is small, masking the concentration. Our method instead computes per-decile coverage and flags the min-bin coverage below threshold because the eval mass is absent from low-alignment bins. We expect CoC to reject this pair while the global test certifies it, with the rejection concentrated in the alignment-density dimension and parity on dimensions where eval spans all deciles.

**vs. CLIP-score-only quality filter** — On a case where a shard has high CLIP alignment but low entity consistency (e.g., repeated generic objects with varied captions), the CLIP-only filter certifies the shard as high quality because its single statistic is high, missing the entity-consistency gap. Our method instead uses the full seven-statistic bank and detects the entity-consistency decile mismatch, producing a lever report that flags entity consistency as the curation target. We expect a noticeable certification disagreement on entity-heavy shards and parity on shards where CLIP and entity consistency co-vary.

**vs. Toolformer-style nearest-neighbor descriptor matching** — On a case where an eval item's nearest training descriptor is close but the eval set's marginal distribution over coherence is shifted, retrieval matching reports high containment because individual neighbors exist, while the distributional mass is misaligned. Our method instead tests marginal bin coverage and rejects containment, attributing the gap to the coherence lever. We expect CoC to reject pairs that retrieval certifies, with the disagreement concentrated on dimensions where local neighbor distance is small but global mass is shifted.

**vs. CoC without Phase 0 calibration** — On a dimension where the closed-form statistic has low R²_d (e.g., factual_grounding, where PMI-based statistics may not predict expert grounding scores), the uncalibrated CoC emits a lever and certifies containment, while the calibrated CoC marks the dimension uncertifiable and emits no lever. We expect the calibrated CoC to report a lower certification pass rate on uncertifiable dimensions and to avoid lever attributions that the uncalibrated version would make, with parity on dimensions where R²_d ≥ 0.30.

### What would falsify this idea

If CoC's certification decisions are indistinguishable from the global energy test across all pairs, or if its lever rankings match nearest-neighbor retrieval on every dimension, then the stratification and statistic-bank mechanisms are not doing the work claimed and the central claim is wrong. Additionally, if the ablation with three statistics produces identical lever rankings to the full bank, the seven-dimensional claim is unsupported; and if the Phase 0 calibration gate does not change any certification decision (i.e., all R²_d ≥ 0.30), the sufficiency assumption is untested rather than verified.

## References

1. IntReal: A Real-World Dataset and Benchmark for Multimodal Interleaved Generation
2. Emerging Properties in Unified Multimodal Pretraining
3. Causal Diffusion Transformers for Generative Modeling
