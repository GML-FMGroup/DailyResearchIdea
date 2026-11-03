# Physics-Law Attribution from Latent Prediction Error Signatures

## Motivation

The framework of 'When Does Latent Prediction Error Support Physical Surprise Claims in Video World Models?' shows that latent prediction error on IntPhys2 and Physion++ is only modestly aligned with physical possibility and is strongly driven by non-physical factors such as frame repetition, color changes, and incorrect actions. Because the error signal is a scalar magnitude without directional structure tied to individual laws, it cannot support explanatory attribution of which physical law was violated; it records surprise but not its cause. This is a structural limitation of treating surprise as a single scalar rather than a law-conditioned pattern.

## Key Insight

The pattern of latent prediction error across spatial regions and time steps is a law-specific signature, because different physical laws constrain different spatiotemporal derivatives of the scene, so a classifier trained on error patterns can recover the violated law even when the scalar error magnitude cannot.

## Method

**What it is.** LAW-SIG (Law-Attributed World-model Surprise) takes a video world model's per-patch, per-timestep latent prediction errors and outputs a probability distribution over a fixed set of physical laws (gravity, collision/contact, object permanence, continuity, solidity). It is a two-stage mechanism: first it computes a law-conditioned error signature via derivative-consistency residuals, then it attributes the law with a linear classifier trained on counterfactually generated violations.

**How it works.**

```pseudocode
Input: video V = {I_t}, world model f, latent z_t = f.encode(I_t), predictor g
Output: law attribution p(law | V)

# Stage 1: law-conditioned error signature
for each t, each patch p:
    e_{t,p} = || z_{t+1,p} - g(z_{<=t})_p ||_2          # scalar latent error
    # derivative-consistency residuals per law family
    r_grav_{t,p}   = || d2(y_{t,p})/dt2 - g_const ||     # vertical acceleration residual
    r_contact_{t,p}= | d(area_{t,p})/dt | * 1[boundary_touching]  # contact change residual
    r_perm_{t,p}   = || z_{t,p} - z_{t-1,p} || * 1[object_disappeared]  # permanence residual
    r_cont_{t,p}   = || d(z_{t,p})/dt - d(z_{t-1,p})/dt ||  # continuity residual
    r_solid_{t,p}  = | overlap_{t,p} |                       # solidity residual
    s_{t,p} = [e_{t,p}, r_grav, r_contact, r_perm, r_cont, r_solid]

# Stage 2: counterfactual violation augmentation
for each training video V and each law l:
    V_l = apply_law_violation(V, l)   # e.g., reverse gravity, teleport object, pass-through
    S_l = aggregate(signature(V_l))    # mean-pool s_{t,p} over patches and time -> vector

# Stage 3: attribution classifier
Train linear softmax classifier C: S -> law label (including 'none')
At test: p(law | V) = C(aggregate(signature(V)))
```

**Why this design.** We chose a linear classifier over a deep attribution head because the signature vector is already law-conditioned by construction (each residual targets a specific spatiotemporal derivative), so a linear head suffices to test whether the signature linearly separates laws; this accepts the cost that nonlinear interactions between laws (e.g., gravity and contact co-occurring) may be underfit. We chose counterfactual violation augmentation over natural video labels because natural videos rarely contain labeled single-law violations, and counterfactuals give exact law labels; this accepts the cost that synthetic violations may not match the distribution of real impossible events. We chose derivative-consistency residuals over raw latent error because raw error conflates appearance and physics, as shown by the modest decoding of physical possibility in the prior framework; this accepts the cost that residuals require estimating object masks and vertical axes, which introduces preprocessing errors. We chose mean-pooling over patches and time over a learned temporal attention because mean-pooling preserves interpretability of which residual drives attribution; this accepts the cost of losing temporal localization of the violation. A domain expert would not describe this as a variant of the prior framework's linear probing, because the prior work probes frozen representations for a binary possible/impossible label, whereas LAW-SIG probes the error signal itself for a multi-class law label using counterfactually generated training data that the prior work does not construct.

**Why it measures what we claim.** The derivative-consistency residual r_grav measures violation of gravity because it assumes that under gravity, vertical acceleration is constant and any deviation from that constant is due to a violated law; this assumption fails when camera motion changes the vertical axis, in which case r_grav reflects camera-induced acceleration rather than physical violation. The contact residual r_contact measures violation of contact laws because it assumes that object area changes only at contact boundaries; this assumption fails when an object is occluded, in which case r_contact reflects occlusion rather than contact violation. The permanence residual r_perm measures object permanence because it assumes that an object disappearing without occlusion is a permanence violation; this assumption fails when an object exits the frame, in which case r_perm reflects frame-boundary effects. The classifier's output p(law | V) measures explanatory attribution because it assumes that the counterfactual violation distribution covers the same law space as natural violations; this assumption fails when a natural violation is a composition of laws not present in the counterfactual set, in which case p(law | V) reflects the closest counterfactual law rather than the true composed violation.

**Confounder-controlled residual construction (explicit assumption and calibration).** The load-bearing assumption is stated plainly: the residuals r_grav, r_contact, r_perm, r_cont, r_solid computed from estimated masks, vertical axes, and boundary/occlusion flags are law-specific signatures whose variation is driven by the violated physical law rather than by confounders (camera motion, occlusion, frame exit, mask/preprocessing error). To operationalize this, we compute residuals only on patches where the relevant confounder is measured to be absent, and we add a confounder-invariance constraint to the signature construction itself. Concretely:

- Camera motion: estimate per-frame global motion via dense optical flow (Farneback, 3-level pyramid, window 15) and fit a homography; compute r_grav only on patches where the residual flow after homography compensation is below 0.5 px/frame, and regress out the camera-induced vertical acceleration component from r_grav by subtracting the homography-predicted vertical acceleration.
- Occlusion: estimate occlusion masks via forward-backward flow consistency (threshold 1.0 px); set r_contact to zero on patches where occlusion is detected, and compute r_perm only on patches where the object is not occluded and not within 5% of the frame boundary.
- Frame exit: detect frame-boundary events by tracking object masks; set r_perm to zero for objects whose centroid is within 5% of the frame boundary or whose mask touches the boundary.
- Mask/axis error: estimate mask uncertainty via test-time augmentation (horizontal flip) and discard patches where mask IoU between original and flipped prediction is below 0.7; estimate vertical axis from the homography-compensated gravity direction and discard frames where the axis estimate variance exceeds 5 degrees.

We validate law-specificity by requiring each residual to fire on its target law while remaining invariant under matched confounder-only perturbations applied to physically possible videos. Specifically, we construct a calibration set of 512 physically possible clips from Physion++ (with ground-truth masks and gravity axis) and apply confounder-only perturbations: camera pan (5-15 degrees), synthetic occlusion (random rectangle covering 10-30% of object), and frame exit (object moved to boundary). We measure the residual response to these perturbations and require that the mean residual magnitude under confounder-only perturbation is less than 20% of the mean residual magnitude under the corresponding law violation (e.g., r_grav under camera pan < 0.2 * r_grav under reversed gravity). If this invariance threshold is not met, we adjust the gating thresholds (e.g., tighten flow threshold to 0.3 px/frame) until it is, and report the final thresholds and invariance ratios. This calibration is performed on the training split only and frozen for test evaluation.

We also report a confounder-balance diagnostic across law classes: for each law label in the counterfactual training set, we compute the distribution of confounder magnitudes (camera motion energy, occlusion fraction, frame-boundary proximity) and report the maximum mean difference between law classes. If any confounder co-varies with law label (e.g., gravity clips systematically have more camera tilt), we flag it and report attribution accuracy conditioned on confounder magnitude to separate law signal from confounder artifact.

## Contribution

['A law-conditioned error signature that decomposes latent prediction error into derivative-consistency residuals targeting specific physical laws, enabling multi-class attribution rather than binary surprise.', 'A counterfactual violation augmentation procedure that generates labeled single-law violations for training an attribution classifier without requiring natural labeled impossible events.', "An empirical design principle: attribution accuracy depends on the alignment between the residual's assumed invariance and the law's actual constraint, which we operationalize as a per-law diagnostic."]

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | IntPhys2 + Physion++ violation clips | Natural and synthetic impossible events with law-type labels |
| Primary metric | Law-attribution accuracy on single-law violations | Directly tests multi-class law discrimination, not binary possibility |
| Baseline | Frozen V-JEPA2 linear probe (possible/impossible) | Prior framework's binary probe; tests sub-claim of multi-class gain |
| Baseline | Raw latent-error signature + linear classifier | Tests whether derivative residuals beat raw error |
| Baseline | Deep attribution head on latent errors | Tests whether linear head on law-conditioned signature suffices |
| Ablation-of-ours | LAW-SIG without counterfactual augmentation | Tests whether counterfactual labels drive attribution |
| Ablation-of-ours | LAW-SIG without confounder-controlled gating | Tests whether confounder invariance drives law-specificity |
| Diagnostic | Confounder-invariance ratio per residual | Measures residual response to confounder-only perturbations vs. law violations |
| Diagnostic | Confounder-balance table across law classes | Checks whether confounders co-vary with law label |

### Why this setup validates the claim

The central claim is that law-conditioned derivative residuals plus counterfactual augmentation yield linearly separable multi-class law attribution where raw-error probes cannot. IntPhys2 and Physion++ supply clips with known law violations (gravity, contact, permanence, continuity, solidity), so accuracy on single-law violations is the right metric: it directly measures whether the classifier recovers the violated law rather than merely detecting impossibility. The frozen V-JEPA2 binary probe tests the sub-claim that multi-class attribution exceeds binary possibility decoding; if LAW-SIG only matches it, the law-conditioning adds nothing. The raw-latent-error classifier isolates whether the derivative residuals, not just the error magnitude, carry the law signal. The deep attribution head tests whether the linear head is sufficient given a law-conditioned signature, and the no-counterfactual ablation tests whether the augmentation, rather than the residuals alone, drives attribution. The no-confounder-gating ablation and the two diagnostics directly test the load-bearing assumption: if attribution accuracy drops when gating is removed, or if the confounder-invariance ratio exceeds 0.2, or if the confounder-balance table shows large co-variation, then the residuals are not law-specific signatures but confounder artifacts. Together these form a falsifiable test: the gain must concentrate on law-type discrimination and vanish when residuals, counterfactuals, or confounder controls are removed.

### Expected outcome and causal chain

**vs. Frozen V-JEPA2 linear probe (possible/impossible)** — On a clip where an object floats upward against gravity, the binary probe outputs "impossible" but cannot say which law, because its label space collapses all violations into one bit. Our method instead computes r_grav and attributes the violation to gravity because the vertical-acceleration residual spikes while others stay near baseline, so we expect a large gap in law-attribution accuracy on gravity clips but parity on binary possibility detection, with attribution accuracy well above chance across all five laws.

**vs. Raw latent-error signature + linear classifier** — On a clip where an object is briefly occluded, the raw-error classifier confuses occlusion with a permanence violation because its scalar error conflates appearance change with physics, producing misattribution to permanence. Our method instead suppresses this via the occlusion-gated permanence residual and the contact residual's boundary condition, so we expect our attribution to be correct on occlusion-heavy clips where the raw-error baseline drops noticeably, while both perform similarly on clean single-law violations.

**vs. Deep attribution head on latent errors** — On a composed clip where gravity and contact co-occur, the deep head overfits spurious appearance cues and mislabels the dominant law because it lacks law-conditioned structure, scrambling attribution. Our method instead separates the two residuals and assigns the law with the larger normalized residual, so we expect our linear head to match or exceed the deep head on single-law clips and show a clear advantage on composed-law clips, confirming the signature is already linearly separable.

**vs. LAW-SIG without confounder-controlled gating** — On a clip with camera pan, the ungated version computes r_grav from camera-induced vertical acceleration and misattributes the clip to gravity even when no law is violated. Our method instead regresses out the homography-predicted vertical acceleration and computes r_grav only on low-residual-flow patches, so we expect the gated version to maintain high attribution accuracy on camera-pan clips where the ungated version drops to near chance, and we expect the confounder-invariance ratio for r_grav to be below 0.2 only with gating enabled.

### What would falsify this idea

If LAW-SIG's attribution gain over the raw-error baseline is uniform across all law types rather than concentrated on the law whose residual is designed to fire, or if removing counterfactual augmentation leaves accuracy unchanged, or if the confounder-invariance ratio exceeds 0.2 for any residual or the confounder-balance table shows large co-variation with law label, the claim that derivative residuals and counterfactual labels carry the law signal would be wrong.

## References

1. When Does Latent Prediction Error Support Physical Surprise Claims in Video World Models?
