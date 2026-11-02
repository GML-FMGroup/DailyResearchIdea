# Motion-Decomposed Recapture Residual Learning for Document Forgery Detection

## Motivation

Document recapture detection in Venus-DefakerOne2 inherits the single-global-motion-field assumption from optical-flow-based physical-spoof detectors: Tide-Style-Tuning and Entropy-based Test Adaptation operate on whole-image forensic features without ever separating camera-induced global document motion from local recapture residuals. When a document is recaptured under mixed dynamics (hand tremor plus screen refresh plus page curl), the residual that actually reveals the spoof is entangled with a dominant global warp, so the foundation model's implicit single-motion prior cannot isolate it — a structural failure that scales of pretraining data (tens of millions of samples) do not repair because the ancestor's motion model is baked into the representation, not the data volume.

## Key Insight

Recapture residuals become identifiable only when the global document motion is first estimated and removed as a low-dimensional parametric field, because the residual then lives in a subspace that is orthogonal to the motion manifold — a separation property that a single-flow prior cannot express.

## Method

**What it is.** We introduce MoDeR (Motion-Decomposed Recapture), a motion-decomposition estimator inserted into Venus-DefakerOne2's Tide-Style-Tuning pipeline. Inputs: a short burst of document frames (or a single frame plus its estimated global homography) and the foundation model's multi-scale forensic feature map; outputs: a global motion field G (8-DoF homography + low-order polynomial warp) and a residual motion field R (dense local displacement), plus a residual-identifiability score used as an auxiliary forensic head.

**Load-bearing assumption (stated explicitly).** MoDeR assumes that the recapture-revealing residual motion field R is spatially sparse and geometrically orthogonal to the low-dimensional parametric global document motion G_poly, so that subtracting an 8-DoF homography plus second-order polynomial warp isolates spoof signal from genuine document motion. This assumption is not derived; it is the premise the method tests. The method therefore includes an explicit calibration step (Stage 0) that measures the separation margin on synthetic bursts with known ground-truth global fields, and a texture-conditioned calibration of s_id (Stage 4) that flags when the assumption is likely to fail (low-texture regions where G is under-estimated).

**How it works.**
```pseudocode
# Stage 0: separation calibration (synthetic bursts with GT global field)
# For each synthetic burst b with known G_gt and known recapture label y:
#   compute R_gt = Flow_gt - warp_by_field(G_gt)
#   measure margin m(b) = ||R_gt||_1 / ||G_gt||_1 and orthogonality o(b) = 1 - cos(R_gt, G_gt)
#   fit texture-conditioned calibration curve s_id_cal = f(o(b), texture_score(b))
#   (texture_score = mean local gradient magnitude in document quad)
#   Store f as a 2D lookup table (o in [0,1] x texture_score in [0,1], 10x10 bins).

# Stage 1: global field estimation (per document)
G = HomographyNet(features[0], features[-1])        # 8-DoF, trained with robust M-estimator loss (Huber delta=0.1)
G_poly = PolyWarp(G, order=2)                        # absorb page curl / lens distortion
# Texture-robust fallback: if texture_score < 0.2, G is re-estimated from document
# boundary/quad tracking (4 corner points tracked by a 2-layer MLP on features,
# hidden=256, GeLU) and fused with HomographyNet output by weighted average (w=0.5).

# Stage 2: residual field via motion subtraction
R_raw = FlowNet(features)                            # dense optical flow, RAFT-style backbone
R = R_raw - warp_by_field(G_poly)                    # residual displacement

# Stage 3: decomposition loss (explicit multi-field supervision)
L_dec = ||R_raw - G_poly||_1 + lambda_orth * ||<R, G_poly>||_F^2   # orthogonality term, lambda_orth=0.1
L_cons = ||R - R_gt_local||_1                        # local residual ground truth from synthetic bursts
L_forg = BCE(ForensicHead(R), y)                     # recapture vs. genuine
L_total = L_dec + L_cons + L_forg + L_tide           # L_tide = Tide-Style-Tuning loss, unchanged

# Stage 4: identifiability score at test time
s_id_raw = 1 - cos(R, G_poly)                        # residual-motion alignment
s_id = f(s_id_raw, texture_score)                    # texture-conditioned calibration from Stage 0
# Entropy-based Test Adaptation selects modality using s_id + forensic entropy
```

**Why this design.** We chose an explicit parametric global field (8-DoF homography plus second-order polynomial) over a single learned flow because the global document motion under hand tremor and page curl is genuinely low-dimensional, and forcing it into a dense flow lets the network absorb residual signal into the global branch — accepting the cost that unusual rigid-body dynamics (e.g., a document on a rotating turntable) may fall outside the polynomial basis. We chose an orthogonality penalty between R and G_poly over a mutual-information independence head because the two fields are geometrically coupled by construction (R = R_raw - G_poly), so independence must be enforced in the displacement subspace rather than in feature space — accepting the cost that the penalty is a soft constraint rather than a hard projection. We chose to keep Tide-Style-Tuning's LoRA/full-parameter alternation intact and only add MoDeR as a side branch over replacing the backbone, because the foundation model's forensic features are the only source of semantic priors at this scale — accepting the cost that MoDeR inherits any feature-level blind spots of Venus-DefakerOne2. A domain expert would not describe this as 'optical-flow PAD plus a foundation model': optical-flow PAD (e.g., the ancestor of Venus-DefakerOne2) fits a single global flow and thresholds its magnitude, whereas MoDeR estimates a multi-field decomposition and uses the residual's orthogonality to the global field as the forensic signal, which is a property the single-flow prior cannot produce.

**Why it measures what we claim.** The residual motion field R measures local recapture residual because the orthogonality assumption makes R equivalent to the component of motion that cannot be explained by the document's rigid/parametric global motion; this assumption fails when the document undergoes non-parametric global deformation (e.g., crumpled paper under uneven lighting), in which case R reflects unmodeled global warp rather than recapture residual. The identifiability score s_id measures motion-residual separability because low alignment is assumed to imply the residual is dominated by local spoof dynamics rather than by global motion leakage; this assumption fails when the global field is under-estimated (low-texture document regions), in which case s_id reflects estimation error rather than true separability. The decomposition loss L_dec measures the explicit multi-field modeling claim because the L1 term forces R_raw to be split into a smooth parametric component and a sparse residual, with the assumption that genuine recapture residuals are spatially sparse relative to global motion; this fails when the recapture artifact is itself spatially smooth (e.g., uniform screen moiré), in which case the split is ambiguous and L_dec reflects reconstruction rather than decomposition. The texture-conditioned calibration f(s_id_raw, texture_score) measures the equivalence assumption A (global-field estimation error is negligible relative to true residual magnitude) because it maps raw alignment to a calibrated score using the texture score as a proxy for estimation reliability; this assumption fails when texture_score is low and G is under-estimated, in which case the calibration curve flattens s_id to avoid inflating it on estimator noise.

## Contribution

(1) MoDeR, a motion-decomposition estimator with an explicit global-plus-residual field decomposition and an orthogonality-constrained decomposition loss, integrated into Venus-DefakerOne2's Tide-Style-Tuning pipeline. (2) The residual-identifiability score s_id, a test-time quantity that operationalizes motion-residual separability and is used alongside Entropy-based Test Adaptation to select prediction modality under mixed recapture dynamics. (3) A synthetic burst-generation protocol producing paired global-motion and local-residual ground truth for document recapture, enabling supervised decomposition training without manual annotation.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Recapture-vs-genuine document burst benchmark (Venus-DefakerOne2 test split, 2,000 bursts; 500 per motion regime: static, hand-tremor, page-curl, rotating-turntable) | Bursts expose global motion vs local residual separation |
| Primary metric | AUC on recapture detection, stratified by motion regime and by texture quartile (mean gradient magnitude) | Detects gain concentrated where decomposition should help |
| Baseline | Venus-DefakerOne2 (single-flow PAD ancestor) | Tests whether multi-field beats single-flow thresholding |
| Baseline | Dense-flow PAD without parametric global branch | Tests whether explicit parametric field matters |
| Baseline | Learned two-branch motion-separation network (single network outputs smooth and sparse flow components, no explicit parametric basis) | Tests whether parametric decomposition is more than re-parameterization |
| Baseline | MoDeR w/o orthogonality penalty (ablation) | Isolates the residual-identifiability mechanism |
| Ablation-of-ours | MoDeR with orthogonality penalty removed | Same as above, named once for clarity |
| Ablation-of-ours | MoDeR with over-fit global basis (order=4 polynomial) | Tests whether spoof cue is absorbable by global basis |
| Ablation-of-ours | MoDeR with texture-conditioned calibration disabled (s_id = s_id_raw) | Tests whether calibration corrects low-texture inflation |

### Why this setup validates the claim

The central claim is that decomposing document motion into a low-dimensional parametric global field and a sparse local residual produces a forensic signal that single-flow PAD cannot reach. The dataset choice matters because the claim is inherently about the *regime* where global motion is strong and structured: a burst benchmark lets us stratify by motion regime (static, hand-tremor, page-curl, rotating-turntable) so that a uniform gain would be uninformative while a gain concentrated in the structured-global-motion strata directly confirms the mechanism. The single-flow PAD baseline tests sub-claim one (multi-field beats single-flow), the dense-flow-without-parametric baseline tests sub-claim two (explicit low-dimensional basis beats letting the network absorb global motion into dense flow), and the orthogonality ablation tests sub-claim three (the residual-identifiability score is what carries the forensic signal, not merely having two branches). The learned two-branch baseline tests whether the parametric basis is a genuine inductive bias or merely a re-parameterization of known motion-separation methods. The over-fit global basis ablation directly tests the load-bearing assumption: if the spoof cue is absorbable by a higher-order polynomial, the forensic head's signal will drop when order=4 is used, falsifying the separation premise. The texture-conditioned calibration ablation tests the equivalence assumption A: if disabling calibration inflates s_id on low-texture subsets, the calibration is doing the work we attribute to it. AUC stratified by motion regime and texture quartile is the right metric because the predicted effect is a *differential* one — the method should be roughly at parity on static or low-texture cases and separate sharply where the global field is well-conditioned; a flat aggregate metric would hide exactly the pattern that distinguishes our mechanism from a generic capacity increase. Additionally, a controlled synthetic experiment (Stage 0 calibration) directly measures residual-motion orthogonality to the ground-truth global field for genuine vs. recaptured bursts, quantifying the separation margin and its dependence on texture and deformation type, to substantiate the subspace-orthogonality premise.

### Expected outcome and causal chain

**vs. Venus-DefakerOne2 (single-flow PAD)** — On a case where a genuine document is filmed while the phone pans under hand tremor, the baseline fits a single global flow whose magnitude threshold is exceeded by the pan, producing a false recapture alarm because its single-field prior cannot separate rigid global motion from local spoof residual. Our method instead subtracts the fitted 8-DoF-plus-polynomial field and scores the orthogonality of the remaining residual, so the pan is absorbed into G and R stays small and misaligned with G, yielding a low recapture score. We expect a pronounced AUC gap on the hand-tremor and page-curl strata but near-parity on the static stratum.

**vs. Dense-flow PAD without parametric global branch** — On a case where a page is curled so that its surface undergoes a smooth second-order warp, the baseline's dense flow absorbs the curl into its residual channel and reads it as local spoof dynamics, producing a false positive because it has no low-dimensional basis to attribute smooth deformation to global motion. Our method's PolyWarp(G, order=2) captures the curl parametrically, so R retains only the genuinely local, spatially sparse component and the forensic head sees a clean residual. We expect the largest gap on the page-curl stratum and a smaller but still positive gap on the turntable stratum, where the polynomial basis is stressed.

**vs. Learned two-branch motion-separation network** — On a case where a document is recaptured under mixed dynamics (hand tremor plus screen refresh), the baseline's single network outputs smooth and sparse flow components without an explicit parametric basis, so it may entangle the global warp with the residual because the smooth branch is free to absorb residual signal. Our method's explicit 8-DoF-plus-polynomial basis constrains the global branch to a low-dimensional manifold, forcing the residual to live in the orthogonal complement. We expect MoDeR to outperform this baseline on the hand-tremor and page-curl strata, with the gap largest where the global motion is well-approximated by the parametric basis; on the turntable stratum, where the basis is stressed, the two may be closer.

**vs. MoDeR w/o orthogonality penalty** — On a case where low-texture document regions leave the global field under-estimated, the ablated model lets R leak into G_poly directions and the forensic head fires on estimation error, producing unstable scores because nothing forces the two fields into separate displacement subspaces. Our method's lambda_orth term softly pushes R orthogonal to G_poly, so leakage is suppressed and the identifiability score s_id reflects true separability rather than estimator noise. We expect the ablation to show a higher false-positive rate specifically on low-texture subsets and a compressed spread of s_id, while the full model preserves a bimodal s_id distribution separating genuine from recapture.

**vs. MoDeR with over-fit global basis (order=4)** — On a case where the recapture cue is spatially smooth (e.g., uniform screen moiré), a higher-order polynomial basis could absorb the spoof signal into G_poly, leaving R empty. If the forensic head's AUC drops when order=4 is used, the load-bearing assumption that the spoof cue is not absorbable by the global basis is falsified. We expect a modest AUC drop on the moiré-heavy subset but not a collapse, indicating that the second-order basis is a reasonable trade-off.

**vs. MoDeR with texture-conditioned calibration disabled** — On a low-texture subset, disabling calibration leaves s_id inflated by estimation error, producing false positives. We expect the calibrated model to show a lower false-positive rate and a more bimodal s_id distribution on the low-texture quartile, confirming that the calibration corrects the equivalence assumption A.

### What would falsify this idea

If MoDeR's AUC gain over the single-flow PAD baseline is uniform across all motion-regime strata — including static and low-texture cases where the decomposition has nothing to exploit — rather than concentrated on the structured-global-motion strata, the claim that multi-field decomposition is the causal driver would be wrong. Likewise, if removing the orthogonality penalty leaves performance unchanged on the low-texture subset where leakage is predicted, or if the over-fit global basis (order=4) absorbs the spoof cue and collapses AUC, the residual-identifiability mechanism is not doing the work we attribute to it.

## References

1. Venus-DefakerOne2: A Vision-Forensics Foundation Model with Broader Coverage
