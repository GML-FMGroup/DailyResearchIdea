# Metric-Anchored Spatial Foresight for Dexterous Policies

## Motivation

DexForesight learns an implicit spatial world model whose 3D hand-point predictions are only defined up to an unknown similarity transform, because the offline training objective (depth reconstruction + kinematic consistency) never ties predicted coordinates to a physical unit. This scale ambiguity is not a nuisance: the policy consumes aligned latents whose implied hand trajectories can be uniformly shrunk or stretched without changing the training loss, so the foresight signal is systematically mis-calibrated relative to the true workspace geometry and manipulation precision degrades exactly where millimeter-level finger placement matters. The root cause is that scale is a gauge freedom of the world model's latent space, and no existing component of DexForesight fixes that gauge.

## Key Insight

Metric scale becomes identifiable — not merely regularized — once the world model's latent gauge is fixed by a per-trajectory scale parameter that is simultaneously constrained by known object dimensions and forward-kinematic link lengths, because these two independent metric measurements over-determine the same single gauge degree of freedom.

## Method

**What it is.** We introduce **MAGIC** (Metric-Anchored Gauge-Identified Calibration), a training-time modification to DexForesight's spatial world model. MAGIC takes as input the same offline tuples (depth frames, kinematic hand joints, object bounding boxes with known class-level dimensions, action trajectories) and outputs 3D hand-point predictions that live in a physically calibrated coordinate frame; at deployment the frozen world model and the policy are unchanged in architecture, only their latent gauge is metric.

**Load-bearing assumption (stated explicitly).** MAGIC's identifiability claim rests on the assumption that the two metric anchors — class-level object dimensions and URDF bone lengths — provide *independent, unbiased* measurements of the same single scale factor, so their intersection uniquely identifies the world model's latent gauge. This assumption is load-bearing and is not automatically true: both anchors are computed from the same predicted hand points `P_hat` produced by the same depth encoder, so if the depth encoder carries a systematic scale bias they are two functions of one biased estimate and the over-determination collapses to a single equation. MAGIC therefore adds one *external* anchor measured in a frame the world model does not produce — the absolute metric values of the observed depth map — and a verification step that reports the correlation between the two prediction-derived anchor residuals to demonstrate they are not collinear.

**How it works.**

```pseudocode
# Per training trajectory i, world model f_theta predicts hand points P_hat in an arbitrary gauge.
# MAGIC adds ONE learnable scalar s_i > 0 per trajectory (a gauge fixer), not per frame.

for each offline trajectory i:
    # 1. World-model forward pass (unchanged from DexForesight)
    P_hat_i, F_hat_i = world_model(depth_i, kin_i, actions_i)   # P_hat in arbitrary units

    # 2. Apply the per-trajectory gauge fixer
    P_metric_i = s_i * P_hat_i                                  # now claimed metric

    # 3. Metric anchor A: object-size calibration loss
    #    For each manipulated object o with known class dimensions d_o (from a lookup table
    #    of 12 object classes in DexJoCo, e.g. cube edge, mug radius):
    #    Touching set T_o = { hand points p in P_metric_i : ||p - c_o|| < 0.02 m },
    #    where c_o is the object center from the bounding box in the *kinematic* frame
    #    (kinematic frame is scale-fixed by URDF, so the 2cm threshold is scale-independent).
    #    Diameter estimate: max_{p,q in T_o} ||p - q||  (farthest-point pair).
    L_obj = mean_o | diameter(P_metric_i restricted to T_o) - d_o |

    # 4. Metric anchor B: kinematic link-length calibration loss
    #    Known bone lengths b_k from the robot hand URDF (constant, not learned):
    L_kin = mean_k | ||P_metric_i[joint k] - P_metric_i[joint k+1]|| - b_k |

    # 5. Metric anchor C (external, breaks depth-encoder correlation):
    #    Absolute-depth anchor. For each frame, take the observed depth map D_i (metric,
    #    from the calibrated camera) and the predicted hand points P_metric_i projected
    #    through the known camera intrinsics K. The anchor compares the predicted depth
    #    of each hand point to the observed depth at its pixel:
    L_depth = mean_{p in P_metric_i} | (K^{-1} p)_z - D_i[pi(K p)] |
    #    This constraint is external to P_hat: D_i is measured, not predicted.

    # 6. Gauge-consistency loss across frames of the same trajectory
    #    s_i must be constant within a trajectory (this is what makes it a gauge, not a fudge):
    L_gauge = 0   # enforced structurally: s_i is a single scalar per trajectory

    # 7. Total loss (DexForesight's original loss L_wm plus anchors)
    L = L_wm + lambda_obj * L_obj + lambda_kin * L_kin + lambda_depth * L_depth
    # lambda_obj = 1.0, lambda_kin = 0.5 (kinematic lengths are noisier due to occlusion),
    # lambda_depth = 1.0 (external anchor, same weight as object anchor)

    # 8. Optimize theta AND {s_i} jointly with Adam, lr 3e-4, s_i initialized to 1.0
    #    After convergence, freeze s_i; at test time s_test is estimated by a single
    #    closed-form least-squares fit of the three anchor losses on 5 calibration frames
    #    from the target scene (no gradient steps at deployment).
    #    Closed-form: minimize over s of sum_a || s * a_hat - a_true ||^2, where a_hat are
    #    the predicted anchor quantities and a_true the known metric values; solution
    #    s* = sum_a a_hat * a_true / sum_a a_hat^2.
```

**Why this design.** We chose a *single per-trajectory scalar* gauge fixer over a per-frame or per-point scale because scale ambiguity is a global similarity gauge of the world model's latent space, and a per-frame scale would reintroduce the ambiguity as a temporal degree of freedom that the anchors cannot over-determine; the cost is that a trajectory containing a genuine camera-zoom event would violate the constant-scale assumption, which we accept because DexJoCo uses fixed cameras. We chose *two independent metric anchors* (object dimensions and kinematic link lengths) over either alone because a single anchor leaves a one-parameter family of scales consistent with it whenever the anchor measurement is noisy, whereas two anchors with independent noise over-determine the same scalar and make it identifiable; the cost is that object-class dimension tables must be available, which we accept because DexJoCo's 12 object classes have standardized dimensions. We chose to *freeze the gauge after world-model training and estimate s_test by closed-form least squares at deployment* rather than fine-tuning online, because online gradient steps on a frozen world model would violate DexForesight's design contract that the world model is not adapted at test time; the cost is that a scene with an out-of-distribution object class whose dimensions are unknown cannot be calibrated, which we accept as a stated scope boundary. A domain expert would not describe MAGIC as "DexForesight + a calibration loss," because the load-bearing mechanism is the *identifiability argument*: the per-trajectory scalar plus two independent anchors is a gauge-fixing construction that removes a latent symmetry, not a regularizer that merely penalizes scale drift — the distinction is that a regularizer leaves the gauge free and only biases it, while MAGIC's construction makes the maximum-likelihood scale unique. The external depth anchor is included precisely because the two prediction-derived anchors share the depth encoder; it provides a constraint on s_i that is measured in a frame the world model does not produce, so the identifiability argument does not rest on an unverified independence assumption.

**Why it measures what we claim.** The per-trajectory scalar `s_i` measures the *latent gauge freedom* of the world model because the assumption that DexForesight's 3D predictions are defined only up to a global similarity transform makes `s_i` exactly the unknown similarity factor; this assumption fails when the world model's depth encoder has already partially fixed scale through absolute-depth supervision, in which case `s_i` reflects residual depth-encoder bias rather than a pure gauge. The object-size residual `L_obj` measures *metric identifiability from object priors* because the assumption that class-level object dimensions are accurate to within 5% makes the residual zero only when the predicted hand points are metrically correct; this assumption fails for deformable or occluded objects whose apparent diameter differs from the class prior, in which case `L_obj` reflects object-prior mismatch rather than scale error. The kinematic residual `L_kin` measures *metric identifiability from robot geometry* because the assumption that URDF bone lengths are exact makes the residual zero only at the correct scale; this assumption fails under finger occlusion where predicted joint positions are interpolated, in which case `L_kin` reflects prediction noise rather than scale error. The depth residual `L_depth` measures *external metric anchoring* because the assumption that the observed depth map is metrically calibrated makes the residual zero only when the predicted hand points are at the correct metric depth; this assumption fails under depth-sensor noise or missing depth at specular/transparent surfaces, in which case `L_depth` reflects sensor error rather than scale error. The joint optimization over `(theta, {s_i})` measures *gauge-fixed world-model training* because the assumption that the three anchors' errors are independent makes their intersection the unique scale; this assumption fails when the anchors are corrupted by a shared bias, in which case the optimum reflects that shared bias rather than a metric fix — which is why we report the pairwise residual correlation between anchors as a diagnostic.

## Contribution

['A gauge-fixing calibration framework (MAGIC) for implicit spatial world models that makes metric scale identifiable via a per-trajectory scale parameter jointly constrained by object-dimension and kinematic-link-length anchors, with a closed-form test-time scale estimator that requires no online adaptation.', 'A design principle for spatial world models in manipulation: scale ambiguity should be treated as a latent gauge symmetry to be fixed by over-determining anchors, not as a quantity to be regularized, because regularization leaves the gauge free while gauge-fixing makes the maximum-likelihood scale unique.', 'An analysis of which anchor pairs over-determine scale in dexterous manipulation settings, identifying object-class dimension priors and URDF link lengths as the minimal sufficient pair for DexJoCo-style tasks.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | DexJoCo, 11 dexterous tasks | Standard benchmark; fixed cameras, 12 known object classes |
| Primary metric | Task success rate | Directly reflects whether metric gauge improves policy foresight |
| Secondary metric | Per-trajectory scale-drift diagnostic: fit a single scale to ground-truth metric hand points and report its variance across trajectories | Empirically establishes the gauge is free before claiming MAGIC fixes it |
| Secondary metric | Anchor residual correlation (pairwise Pearson r between L_obj, L_kin, L_depth residuals across trajectories) | Verifies the anchors are not collinear, supporting the identifiability claim |
| Baseline | DexForesight (unmodified) | Tests whether gauge freedom actually harms downstream control |
| Baseline | DexForesight + per-frame scale regressor | Tests whether a single per-trajectory scalar is the right gauge |
| Baseline | DexForesight + object-anchor-only loss | Tests whether two anchors are needed for identifiability |
| Baseline | DexForesight + standard scale-regularizer (L2 penalty on s_i toward 1.0, no anchors) | Contrasts gauge-fixing against a regularizer that leaves the gauge free |
| Ablation-of-ours | MAGIC without kinematic anchor (L_obj + L_depth only) | Isolates contribution of the second prediction-derived anchor |
| Ablation-of-ours | MAGIC without external depth anchor (L_obj + L_kin only) | Tests whether the external anchor is load-bearing |
| Ablation-of-ours | MAGIC with proximity threshold varied over {1cm, 2cm, 3cm, 5cm} | Tests sensitivity of L_obj to the touching-set threshold |
| Sanity check | Closed-form s_test least-squares fit on held-out trajectories vs. training-time s_i | Verifies the deployment calibration recovers the training gauge |

### Why this setup validates the claim

The central claim is that DexForesight's latent 3D predictions carry an unfixed similarity gauge, and that pinning a single per-trajectory scalar against independent metric anchors makes scale identifiable and thereby improves downstream control. Unmodified DexForesight tests whether the gauge is actually free and harmful: if its success rate already matches MAGIC, the gauge is either already fixed by absolute-depth supervision or irrelevant to control, and the idea fails. The per-trajectory scale-drift diagnostic directly measures the variance of the fitted scale across trajectories; if that variance is near zero, the gauge is not free and the premise is false. The anchor residual correlation diagnostic tests the load-bearing independence assumption: if L_obj, L_kin, and L_depth residuals are highly correlated (r > 0.8), the over-determination story collapses. The per-frame-scale baseline tests the specific design choice — if per-frame scaling matches or beats our per-trajectory scalar, the gauge-fixing argument is wrong and the ambiguity is being exploited as useful temporal freedom. The object-anchor-only baseline, the kinematic-dropout ablation, and the external-depth-dropout ablation test the identifiability argument: if one anchor suffices, the multi-anchor over-determination story is decorative. The scale-regularizer baseline tests whether MAGIC is merely a regularizer: if an L2 penalty toward s_i=1.0 matches MAGIC, the gauge-fixing construction is not load-bearing. Success rate is the right primary metric because the claim is that a metric gauge improves the world model's usefulness to the frozen policy, not that it lowers a training loss; a loss-only improvement with no success-rate change would not validate the causal claim.

### Expected outcome and causal chain

**vs. DexForesight (unmodified)** — On tasks requiring precise hand-object contact, such as cube reorientation or mug grasping, unmodified DexForesight produces hand-point forecasts in an arbitrary scale, so its implicit spatial foresight is systematically mis-scaled relative to the true object geometry and the policy commits to contacts that are too shallow or too deep. Our method instead pins the trajectory's scale against object diameters, URDF bone lengths, and observed depth, so the forecast geometry is metric and contact planning is consistent. We expect a noticeable success-rate gap concentrated on contact-critical tasks, with near-parity on coarse reach-and-transport tasks where scale barely matters. The scale-drift diagnostic is expected to show substantial per-trajectory variance in the fitted scale for unmodified DexForesight, establishing that the gauge is free.

**vs. DexForesight + per-frame scale regressor** — On a trajectory where the world model's per-frame predictions drift in scale (e.g., mid-trajectory occlusion of the hand), the per-frame regressor absorbs that drift into a time-varying scale, reintroducing the gauge as a free degree of freedom and letting the anchors be satisfied frame-by-frame while the trajectory remains internally inconsistent. Our method instead holds one scalar per trajectory, so drift must be explained by the anchors as a genuine model error rather than hidden in the scale. We expect the per-frame baseline to sit between unmodified DexForesight and MAGIC, with its gap largest on long trajectories with occlusion events.

**vs. DexForesight + object-anchor-only loss** — On tasks where the manipulated object is small or partially occluded, the object-diameter residual is noisy, and a single anchor leaves a one-parameter family of scales consistent with it, so the fitted scale is under-determined and biased toward whichever object happens to dominate the loss. Our method instead over-determines the same scalar with the kinematic and external-depth anchors, whose measurements are noise-free (URDF) and external to the depth encoder (observed depth), so the optimum is unique. We expect the object-only baseline to show high variance in success across object classes and to underperform MAGIC specifically on small-object tasks while matching it on large-object tasks.

**vs. DexForesight + standard scale-regularizer** — A regularizer that penalizes s_i toward 1.0 leaves the gauge free and only biases it, so the fitted scale can still drift with the depth-encoder bias. We expect the regularizer to provide little or no success-rate gain over unmodified DexForesight, confirming that the gauge-fixing construction (not the mere presence of a scale penalty) is what matters.

**vs. MAGIC without external depth anchor** — If the two prediction-derived anchors are collinear under depth-encoder bias, dropping the external anchor should degrade success on contact-critical tasks, and the anchor residual correlation diagnostic should show high r between L_obj and L_kin. We expect the no-external-anchor variant to underperform full MAGIC, with the gap largest on tasks where the depth encoder's scale bias is strongest.

**vs. MAGIC with varied proximity threshold** — If the 2cm threshold is scale-dependent or misidentifies touching points, success should vary with the threshold. We expect stable performance across 1–3cm and degradation at 5cm (where non-touching points enter the diameter estimate), confirming the threshold is not a hidden tuning knob.

### What would falsify this idea

If MAGIC's success-rate gain over unmodified DexForesight is uniform across all 11 tasks rather than concentrated on contact-critical, small-object, and occlusion-heavy tasks where the gauge is predicted to bite, then the improvement is not coming from gauge fixing and the central identifiability claim is wrong. Likewise, if the per-frame-scale baseline matches MAGIC, or if the anchor residual correlation diagnostic shows the anchors are collinear (r > 0.8), the single-per-trajectory-scalar design and the over-determination argument are not load-bearing.

## References

1. DexForesight: Grounding Dexterous Policies with Implicit Spatial Foresight
