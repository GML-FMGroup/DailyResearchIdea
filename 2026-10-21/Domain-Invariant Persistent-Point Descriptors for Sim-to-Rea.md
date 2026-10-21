# Domain-Invariant Persistent-Point Descriptors for Sim-to-Real Multi-View Video Prediction

## Motivation

Existing point-trajectory forecasting models such as VPM learn persistent-point representations exclusively from simulated data, so their predicted trajectories drift under real sensor noise and appearance shift because the representation has no mechanism to align simulated and real feature distributions. This is a meta-level structural failure: the field's 4D world models treat the persistent-point descriptor as a domain-agnostic quantity, yet it is trained only on synthetic renders, and no current branch—including HOT3D's fixed-capture tracking—provides a domain-invariance constraint on that descriptor. Closing this gap requires enforcing sim-to-real invariance directly on the persistent-point representation rather than post-hoc filtering of trajectories.

## Key Insight

If the persistent-point descriptor is trained so that its similarity structure is preserved across simulated and real multi-view observations of the same physical point, then the trajectory head—which reads only these descriptors—inherits a representation whose geometry is invariant to sensor noise and appearance shift, making forecast drift a property of the descriptor's domain sensitivity rather than an irreducible sim-to-real gap.

## Method

**What it is.** We introduce **DIP-VPM** (Domain-Invariant Persistent-point VPM), a training procedure that augments the VPM point-trajectory head with a domain-invariance constraint on its persistent-point descriptor. Inputs are multi-view video clips from VPM-Data (simulated, with point trajectories) and a small unlabeled set of real multi-view clips; outputs are the same video and trajectory predictions as VPM, but with a descriptor whose sim-to-real similarity structure is aligned.

**Load-bearing assumption (stated explicitly).** The contrastive term `L_con` treats the same physical point index in a simulated clip and a real clip as a valid positive pair. This requires that VPM's own trajectory tracker, run on the unlabeled real clips, produces cross-domain point correspondences that are consistent and correct enough that the positive pairs are not dominated by tracker noise. This assumption is not verified by construction; it is measured and gated as described below.

**How it works.**

```pseudocode
# Phase 1: Supervised VPM training (unchanged backbone)
for each simulated clip (V_sim, T_sim) in VPM-Data:
    z_sim = PointDescriptor(WorldModel(V_sim))          # per-point descriptors
    loss_traj = TrajectoryLoss(TrajHead(z_sim), T_sim)
    loss_video = DiffusionLoss(VideoHead(WorldModel(V_sim)), V_sim)
    update(loss_traj + loss_video)

# Phase 2: Contrastive domain-invariance fine-tuning
for each pair (V_sim, V_real) from a shared physical scene:
    z_sim  = PointDescriptor(WorldModel(V_sim))         # (N, d)
    z_real = PointDescriptor(WorldModel(V_real))        # (N, d)
    # Cross-domain positive pairs: same physical point index across domains,
    # accepted only if the tracker confidence gate passes (see below).
    # Negative pairs: different points within each domain.
    L_con = InfoNCE(z_sim, z_real, tau=0.07)            # symmetric over both directions
    # Domain-adversarial term on descriptor statistics
    L_adv = -CE(DomainClassifier(GradReverse(z_sim, z_real)), domain_label)
    # Trajectory consistency: forecast from real history must match forecast from sim history
    T_hat_sim  = TrajHead(z_sim)
    T_hat_real = TrajHead(z_real)
    L_cons = ||T_hat_sim - T_hat_real||_1
    update(L_con + lambda_adv * L_adv + lambda_cons * L_cons)   # lambda_adv=0.1, lambda_cons=0.5
```

**Correspondence confidence gate (specification of the load-bearing assumption).** For each real clip, VPM's tracker outputs, for every point index, a confidence score `c_i` (the tracker's per-point visibility/uncertainty score) and a forward-backward cycle-consistency error `e_i` (L2 distance between the point's tracked location when tracked forward then backward). A point index `i` is admitted as a positive pair only if `c_i >= 0.5` and `e_i <= 2.0` pixels; otherwise it is excluded from `L_con` for that pair. The fraction of admitted points per clip is logged as `frac_admit`. If `frac_admit < 0.3` for a clip, that clip is skipped for `L_con` (but still used for `L_adv` and `L_cons`, which do not require point-index correspondence). This gate is the concrete operationalization of the assumption that tracker correspondences are valid; it does not change the contrastive objective, only which positives are used.

**Why this design.** We chose a contrastive InfoNCE loss over a simple L2 feature-matching loss because the persistent-point descriptor must preserve *relative* similarity structure (which point is which) rather than absolute feature values, accepting the cost that InfoNCE requires matched point correspondences between simulated and real clips, which we obtain from VPM's own trajectory tracker on the unlabeled real videos and admit only through the confidence gate above. We chose a gradient-reversal domain classifier over an explicit distribution-matching loss (e.g., MMD) because the adversarial term directly penalizes any linearly decodable domain information in the descriptor, accepting the cost of unstable adversarial training that we stabilize with a 0.1 weight and a warm-up of 500 steps. We chose to add a trajectory-consistency term over descriptor-only alignment because a descriptor can be domain-invariant while still yielding different trajectory-head outputs if the head is sensitive to residual domain-specific variance, accepting the cost that this term requires running the trajectory head on real clips, which is cheap relative to video generation. A domain expert would not describe this as "VPM + contrastive learning": the contrastive term is applied to the *persistent-point descriptor*—the exact quantity VPM's trajectory head consumes—and is coupled to a trajectory-consistency loss, so the invariance is enforced on the causal pathway from representation to forecast rather than on an auxiliary feature.

**Why it measures what we claim.** The InfoNCE loss `L_con` measures *domain-invariant point identity* because it assumes that the same physical point index across a simulated and real clip of the same scene should be closer in descriptor space than any different point; this assumption fails when the real clip's tracker assigns inconsistent point indices under occlusion, in which case `L_con` reflects tracker noise rather than domain invariance. The confidence gate above is the concrete check on this assumption: we report `frac_admit` per clip and the cross-domain correspondence accuracy of the tracker on a held-out set of 200 manually annotated real-clip point pairs, and we condition the interpretation of `L_con` on these measured quantities. The adversarial term `L_adv` measures *absence of linearly decodable domain information* because it assumes that domain shift is captured by a linear probe on the descriptor; this assumption fails when domain shift is nonlinear (e.g., specular highlights), in which case `L_adv` reflects only the linear component of domain sensitivity. The trajectory-consistency term `L_cons` measures *forecast stability under domain shift* because it assumes that the trajectory head is Lipschitz in its descriptor input, so descriptor alignment implies forecast alignment; this assumption fails when the head has a domain-specific bias (e.g., learned noise scale), in which case `L_cons` reflects head bias rather than representation invariance. We measure the empirical Lipschitz constant of the trajectory head on a calibration set of 512 descriptor pairs (perturbation magnitude 0.01–0.1 in L2) and report it alongside `L_cons`. Together these quantities operationalize the motivation's claim that sim-to-real invariance of the persistent-point representation is what controls forecast drift.

## Contribution

['A domain-invariance training procedure (DIP-VPM) that applies contrastive consistency and gradient-reversal adversarial losses directly to the persistent-point descriptor of a point-trajectory forecasting world model, coupled with a trajectory-consistency loss that ties descriptor invariance to forecast stability.', 'A design principle showing that sim-to-real invariance for 4D world models should be enforced on the representation consumed by the trajectory head rather than on the generated video or on post-hoc trajectory filtering, because the descriptor is the sufficient statistic for trajectory prediction.', "A lightweight fine-tuning protocol that requires only a small set of unlabeled real multi-view videos with pseudo-correspondences from the model's own tracker, avoiding the need for real point-trajectory annotations."]

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | VPM-Data sim + unlabeled real clips (VP-WorldBench) | Matched sim/real scenes with point trajectories. |
| Primary metric | Point-trajectory forecasting error on real clips | Directly measures the causal pathway we align. |
| Baseline | Standard multi-view video prediction model | Tests whether explicit 3D trajectory supervision matters. |
| Baseline | VPM without domain-invariance fine-tuning | Tests whether sim-to-real descriptor alignment matters. |
| Baseline | VPM + L2 feature-matching (no InfoNCE) | Tests whether preserving relative similarity structure matters. |
| Ablation-of-ours | DIP-VPM minus trajectory-consistency term | Isolates the causal-pathway coupling claim. |
| Diagnostic | Cross-domain correspondence accuracy of VPM's tracker on 200 annotated real-clip point pairs | Validates the load-bearing assumption behind L_con. |
| Diagnostic | Empirical Lipschitz constant of TrajHead on 512 descriptor pairs | Validates the assumption behind L_cons. |

### Why this setup validates the claim

The central claim is that sim-to-real drift in VPM's forecasts is caused by domain-specific variance in the persistent-point descriptor that feeds the trajectory head, and that constraining that descriptor's similarity structure plus its downstream forecast will reduce drift. The metric—real-clip trajectory forecasting error—is the right detector because it is the exact quantity the causal pathway produces; if the descriptor is genuinely invariant on the causal pathway, this error should drop, and if the gain is an artifact it will not. The standard video-prediction baseline tests the sub-claim that explicit trajectory supervision is necessary at all; the untuned-VPM baseline tests the sub-claim that sim-to-real alignment (not just capacity) drives the gain; the L2-matching baseline tests the sub-claim that preserving relative point-identity structure, not absolute feature values, is what matters. The ablation isolates the trajectory-consistency coupling: if removing it collapses the gain, the invariance claim must be about the representation-to-forecast pathway, not the descriptor alone. The two diagnostics make the load-bearing assumptions explicit and measurable: cross-domain correspondence accuracy conditions whether L_con is measuring domain invariance or tracker noise, and the empirical Lipschitz constant conditions whether L_cons is measuring forecast stability or head bias. Together these form a falsifiable test because each baseline removes exactly one mechanism the argument invokes, and each diagnostic measures one assumption the argument relies on.

### Expected outcome and causal chain

**vs. Standard multi-view video prediction model** — On a case where a real clip contains a moving object under a lighting shift absent from simulation, the baseline produces trajectory forecasts that drift in depth because it lacks any explicit persistent-point representation to anchor 3D motion. Our method instead forecasts from a descriptor whose point-identity structure is aligned across domains because the InfoNCE term ties the same physical point index across sim and real (admitted through the confidence gate), so we expect a clear gap on real clips with appearance shift but near-parity on clean simulated clips.

**vs. VPM without domain-invariance fine-tuning** — On a case where a real clip's specular highlights push the descriptor manifold away from the simulated manifold, the untuned VPM produces forecasts biased toward simulated appearance statistics because its descriptor retains domain-specific variance that the trajectory head reads as motion. Our method instead suppresses linearly decodable domain information through the adversarial term and stabilizes the forecast through consistency, so we expect the gain to concentrate on real clips with strong appearance shift and to be small on simulated clips.

**vs. VPM + L2 feature-matching (no InfoNCE)** — On a case where two distinct points share similar absolute descriptor values after L2 matching, the baseline produces point-identity swaps and trajectory crossings because L2 matching collapses relative structure. Our method instead preserves which-point-is-which via InfoNCE's relative comparison, so we expect fewer identity-swap errors on crowded scenes and a noticeably lower trajectory error there than on sparse scenes.

**Diagnostic: cross-domain correspondence accuracy** — We report the fraction of admitted positive pairs (`frac_admit`) and the tracker's cross-domain correspondence accuracy on 200 manually annotated real-clip point pairs. If accuracy is below 70%, the L_con gain is interpreted as tracker-noise-limited, and the method's claim is conditioned on the confidence gate's filtering.

**Diagnostic: empirical Lipschitz constant** — We report the maximum ratio of trajectory-head output change to descriptor perturbation (L2) over 512 calibration pairs. If the constant exceeds 5.0, the L_cons term is interpreted as potentially reflecting head bias rather than representation invariance, and the ablation-of-ours result is read accordingly.

### What would falsify this idea

If the gain over untuned VPM is uniform across real clips rather than concentrated on clips with strong appearance shift, or if removing the trajectory-consistency term leaves the gain unchanged, then the claim that descriptor-level sim-to-real invariance on the forecast pathway controls drift is wrong. Additionally, if the cross-domain correspondence accuracy is below 70% and the confidence gate does not recover a gain, the load-bearing assumption behind L_con is falsified.

## References

1. Learning the 4D World through Point Trajectory Forecasting
