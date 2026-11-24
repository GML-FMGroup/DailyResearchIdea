# Morphology-Conditioned Content-Slot Initialization with Distribution-Shift Regularization for Long-Horizon Humanoid Motion

## Motivation

Long-horizon terminal control in Motion Mamba 2.0 is demonstrated only on retargeted HiPHI, with short-horizon comparisons on BONES Seed; the persistent content memory (PCM) slots that carry history are initialized and regularized as if the target morphology and instruction distribution match training, so cross-morphology transfer is an untested assumption rather than a structural property. This is the field-level pattern: long-horizon controllable motion generators (Motion Mamba 2.0, TIMotion for human-human interaction, Sitcom-Crafter for scene-aware plots) each acquire a history-carrying state — recurrent slots, causal interaction sequences, or SDF collision memory — whose distribution is implicitly tied to the training bodies/scenes/plots, and none makes the state's transfer to unseen morphologies or instruction distributions a tested property. The root cause is structural: PCM slots are computed from Mamba recurrent states and text features without any morphology-conditioned anchor, so a body with different limb proportions or actuator ranges produces slots that lie off the training manifold and terminal control degrades without any signal that it has.

## Key Insight

If PCM content slots are initialized from a morphology-conditioned anchor and constrained by a distribution-shift regularizer that bounds their deviation from the anchor under instruction reweighting, then the slot subspace becomes an invariant shared across morphologies and instruction distributions, so long-horizon terminal control transfers to held-out bodies as a bounded-degradation structural property rather than a training-set coincidence.

## Method

**What it is.** We introduce MARS-PCM (Morphology-Anchored Regularized Slot PCM), an extension of Motion Mamba 2.0 that replaces the free-form initialization of persistent content memory slots with a morphology-conditioned anchor and adds an explicit distribution-shift regularizer over the slot trajectory. Inputs: a morphology descriptor m (joint count, limb-length ratios, actuator range vector), a text instruction x, and the Mamba recurrent state h_t. Outputs: an initialized slot bank S_0 and, at each step, a regularized slot update S_t used by local attention to produce the root-body diffusion denoising target.

**Load-bearing assumption (stated explicitly).** The method assumes that the 12-d morphology descriptor m (joint count, limb-length ratios, actuator ranges) is a sufficient statistic for the target body's kinematic reachable set, so that the anchor A = MLP_anchor(m) spans the morphology-conditioned slot subspace and the orthogonal-complement projection proj_A(delta) preserves exactly the morphology-relevant directions while admitting task adaptation. This assumption is load-bearing and is not derived; it is verified empirically in the calibration step below. It fails when a held-out morphology differs in joint topology (e.g., additional spine joints) rather than proportions, or when MLP_anchor underfits and its span is low-rank; in those cases the anchor may be systematically biased or the projection may discard task-relevant adaptation. The calibration step gates the projection strength by a per-morphology reliability score so that when the descriptor is insufficient the regularizer relaxes toward L2-only behavior rather than over-constraining.

**How it works.**
```pseudocode
# Phase 0: anchor calibration (once, on training morphology distribution)
# Collect slot trajectories {S_t} from a short unregularized warm-up (10k steps) across training morphologies.
# Fit PCA on the flattened slot vectors; record explained-variance spectrum and effective rank r_eff (number of components to reach 95% variance).
# For each training morphology m_i, compute reliability score rho(m_i) = 1 - (||S_t - A_i||_F^2 / ||S_t - mean(S)||_F^2) averaged over its trajectories, clipped to [0,1].
# Fit a small regression (2-layer MLP, hidden=64) mapping m -> rho, used at test time to predict reliability for held-out morphologies.

# Phase 1: morphology anchor (once per rollout)
m       = encode_morphology(joint_count, limb_ratios, actuator_range)   # 12-d descriptor
A       = MLP_anchor(m)                       # A in R^{K x d}, K=16 slots, d=256; 2-layer MLP, hidden=512, GeLU
g       = MLP_shift(m)                        # g in R^{K x d}, shift basis; 2-layer MLP, hidden=512, GeLU
S_0     = A + g * eps,  eps ~ N(0, I)          # anchored stochastic init; g initialized with std=0.01
rho     = MLP_reliability(m)                   # predicted reliability in [0,1]; 2-layer MLP, hidden=64, sigmoid

# Phase 2: per-step PCM update with shift regularization and reliability gating
for t in 1..T:
    h_t     = MambaStep(h_{t-1}, prev_motion_token)     # resumable recurrent state
    c_t     = TextCond(x)                                # text-conditioned feature
    delta   = SlotUpdate(h_t, c_t, S_{t-1})              # candidate slot update
    # distribution-shift regularizer: project delta onto anchor-orthogonal complement
    delta_perp = delta - proj_A(delta)                   # remove component along anchor span
    # reliability gating: when rho is low (descriptor insufficient or anchor low-rank), relax projection toward L2-only
    lambda_perp_eff = lambda_perp * rho                  # lambda_perp=0.3 base
    lambda_par_eff  = lambda_par + (1 - rho) * lambda_perp  # lambda_par=1.0 base; shifts weight to parallel component when unreliable
    S_t     = S_{t-1} + lambda_perp_eff * delta_perp + lambda_par_eff * proj_A(delta)
    # explicit shift penalty (training loss term)
    L_shift = || S_t - A ||_F^2 / (K*d)                  # deviation from morphology anchor
    L_total = L_diff + beta * L_shift                    # beta=0.05
    root_body_token = LocalAttention(S_t, h_t)
    prev_motion_token = DiffusionDenoise(root_body_token)
```

**Why this design.** We chose a morphology-conditioned anchor over a learned disentanglement head because the morphology descriptor is a measured physical quantity (joint count, limb ratios, actuator ranges) that we hypothesize is a sufficient statistic for the body's kinematic reachable set, whereas a disentanglement head would have to manufacture independence between morphology and motion that the data does not contain; the cost is that the descriptor must be available at test time, which holds for retargeted humanoid platforms, and the sufficiency hypothesis is explicitly verified in Phase 0 rather than assumed. We chose an orthogonal-complement projection of the slot update over a simple L2 penalty on slots because the projection preserves the anchor span exactly while allowing off-anchor adaptation, accepting the cost that lambda_perp must be tuned per morphology family; the reliability gate rho(m) makes this tuning data-calibrated rather than hand-set, relaxing the projection when the anchor span is low-rank or the descriptor is insufficient. We chose a stochastic anchored initialization (A + g*eps) over deterministic A because terminal control requires exploration of history-dependent slot configurations within the anchor neighborhood, accepting the cost that the variance of g must be small enough not to wash out the anchor (initialized at std=0.01). A domain expert would not describe this as "Motion Mamba 2.0 + a regularizer": Motion Mamba 2.0's PCM slots are unconstrained functions of recurrent state and text, and the contribution here is the structural constraint that the slot subspace is anchored to a morphology-sufficient statistic, which changes what the slots can represent, not just how they are penalized.

**Why it measures what we claim.** The morphology descriptor m measures the target body's kinematic reachable set because we assume joint count, limb-length ratios, and actuator ranges are a sufficient statistic for the set of achievable root-body trajectories; this assumption fails when a held-out morphology differs in joint topology (e.g., additional spine joints) rather than proportions, in which case m reflects only a coarse reachability envelope and the anchor may be systematically biased. The anchor deviation ||S_t - A||_F^2 measures distribution shift of the history-carrying state because we assume the anchor A is the conditional mean of slots under the training morphology distribution; this assumption fails when the training morphology distribution is multi-modal (e.g., bipeds and quadrupeds mixed), in which case the deviation reflects distance to a mixture mean rather than shift from the target body. The orthogonal projection proj_A(delta) measures preservation of morphology-relevant slot directions because we assume the anchor span captures the morphology-conditioned subspace; this assumption fails when the anchor MLP underfits and the span is low-rank, in which case the projection discards task-relevant adaptation and the regularizer over-constrains rather than anchors. The reliability score rho(m) measures the trustworthiness of the anchor for a given morphology because we assume it is calibrated against measured reachable-set violations on the training distribution; this assumption fails when the held-out morphology lies outside the training distribution's support, in which case rho is an extrapolation and the gate may be miscalibrated.

## Contribution

(1) A morphology-anchored persistent content memory (MARS-PCM) that initializes and regularizes PCM slots from a kinematic-sufficient morphology descriptor, making the slot subspace an explicit function of the target body rather than of the training distribution. (2) An orthogonal-complement slot-update rule with a distribution-shift penalty that separates morphology-preserving from task-adapting slot directions, providing a design principle for constraining history-carrying state in autoregressive diffusion generators. (3) A held-out evaluation protocol spanning unseen morphologies, interaction scenarios, and instruction distributions, turning cross-morphology transfer from an assumption into a measured property of long-horizon terminal control.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | Retargeted HiPHI (long-horizon rollouts) | Matches Motion Mamba 2.0 terminal-control protocol; morphology varies |
| Primary metric | Terminal root-body control error at long horizon | Directly measures morphology-conditioned reachability preservation |
| Baseline | Motion Mamba 2.0 (unconstrained PCM slots) | Isolates the effect of anchoring slots |
| Baseline | Learnable-slot init (no morphology descriptor) | Tests whether morphology conditioning matters vs generic init |
| Baseline | L2-penalty-only variant (no orthogonal projection) | Tests whether projection matters vs simple shift penalty |
| Ablation-of-ours | MARS-PCM with deterministic A (no stochastic eps) | Tests contribution of anchored stochastic exploration |
| Ablation-of-ours | MARS-PCM with fixed rho=1 (no reliability gating) | Tests whether the calibration gate contributes beyond a fixed projection strength |
| Diagnostic | Descriptor-distance vs terminal-error correlation across held-out morphologies | Validates the sufficiency claim: if m is sufficient, descriptor distance should predict error |
| Diagnostic | Anchor-span rank/conditioning (effective rank r_eff from Phase 0 PCA) | Confirms A is not low-rank before relying on proj_A |
| Diagnostic | Training slot distribution modality estimate (e.g., dip test or Gaussian mixture BIC on slot vectors) | Tests the conditional-mean assumption A (unimodal training morphology distribution) |

Morphology-mismatched vs in-distribution split: we pre-specify the split by computing the Mahalanobis distance of each held-out morphology descriptor from the training descriptor mean; the top quartile (largest distance) is the mismatched subset, the bottom quartile is in-distribution. Actuator-range vectors are recovered from the retargeted HiPHI assets where available; if not recoverable, we use a synthetic morphology-perturbation protocol (limb ratios scaled by factors in {0.85, 1.15}, actuator ranges clipped to {0.8, 1.2} of nominal) so the experiment is runnable without new data collection.

### Why this setup validates the claim

The central claim is that anchoring PCM slots to a morphology-sufficient statistic changes what the slots can represent, improving long-horizon terminal control on bodies whose morphology differs from training. Retargeted HiPHI is the right dataset because it already exhibits morphology variation and long-horizon rollouts, so the predicted failure mode of unconstrained slots (drift into morphology-inconsistent configurations) has room to appear. Motion Mamba 2.0 tests the core sub-claim: if anchoring matters, the gap should appear specifically at long horizons where history-carrying state accumulates. The learnable-init baseline tests whether the morphology descriptor itself is doing work rather than merely providing a stable initialization; if that baseline matches ours, the descriptor is decorative. The L2-only variant tests whether the orthogonal-complement projection contributes beyond a generic shift penalty. The deterministic-A ablation tests whether stochastic exploration within the anchor neighborhood is needed for terminal control. The fixed-rho ablation tests whether the calibration gate contributes beyond a fixed projection strength. The descriptor-distance vs terminal-error diagnostic directly tests the load-bearing sufficiency assumption: if m is sufficient, descriptor distance should correlate with terminal error across held-out morphologies. The anchor-span rank diagnostic confirms A is not low-rank before relying on proj_A. The modality diagnostic tests the conditional-mean assumption A. Terminal control error is the right primary metric because it is exactly the quantity the anchor is supposed to protect: a method that anchors correctly should hold the reachable-set envelope over long rollouts, while unanchored slots should drift. The ablation is diagnostic, not decorative: it isolates which ingredient produces the predicted subset-specific gain.

### Expected outcome and causal chain

**vs. Motion Mamba 2.0** — On a rollout where the target morphology has limb-length ratios outside the training mode and the horizon exceeds ~30 s, the baseline's unconstrained PCM slots absorb history and text features without any morphological reference, so the root-body trajectory gradually leaves the target's reachable set and terminal control error grows with horizon. Our method instead constrains slot evolution to an anchor span derived from the measured morphology descriptor, so off-anchor adaptation is admitted only through the orthogonal complement, and terminal error stays bounded. We expect a noticeable gap concentrated at long horizons and near-parity at short horizons.

**vs. Learnable-slot init (no morphology descriptor)** — On a case where two target bodies share similar motion statistics but differ in actuator range, the baseline's learned initialization cannot distinguish them and produces a configuration that is plausible for the average body but infeasible for the narrower one, yielding late-horizon drift. Our method conditions the anchor on the actuator range vector, so the slot bank starts inside the correct reachable envelope. We expect our gain to be largest on the morphology-mismatched subset and near-zero on in-distribution morphologies.

**vs. L2-penalty-only variant** — On a case where the slot update has a large component along the anchor span, the L2-only variant shrinks all directions uniformly, suppressing task-relevant adaptation along morphology-relevant axes and producing sluggish or over-regularized motion. Our method projects the update, preserving the anchor span exactly while allowing full adaptation off-span, so motion remains responsive while morphology consistency holds. We expect a gap on dynamic, fast-changing motion segments rather than on static poses.

**vs. MARS-PCM with fixed rho=1 (no reliability gating)** — On a held-out morphology where the descriptor is insufficient (e.g., topology change or low-rank anchor), the fixed-rho variant over-constrains the slot update and discards task-relevant adaptation, producing higher terminal error than the gated version. We expect the gated version to match fixed-rho on in-distribution morphologies and outperform it on the mismatched subset where rho is predicted low.

**Diagnostic expectations** — If the sufficiency assumption holds, descriptor distance should show a positive correlation (Spearman rho > 0.4) with terminal error across held-out morphologies for the baseline, and a weaker correlation for our method. The anchor-span effective rank should be at least 8 (out of 16 slots) to justify proj_A. The modality diagnostic should indicate unimodal training slot distribution (dip test p > 0.05) for the conditional-mean assumption to hold; if multi-modal, we report this as a limitation and the deviation metric reflects distance to a mixture mean.

### What would falsify this idea

If the gain over Motion Mamba 2.0 is uniform across in-distribution and morphology-mismatched subsets rather than concentrated where morphology mismatch is largest, the anchor is not doing the claimed work. Likewise, if the deterministic-A ablation matches the full method, the stochastic anchored initialization contributes nothing and the central mechanism story is wrong. If the descriptor-distance vs terminal-error correlation is near zero for the baseline, the sufficiency assumption is falsified and the anchor cannot be doing the claimed work.

## References

1. Motion Mamba 2.0: Autoregressive Diffusion Mamba for Long-Horizon Humanoid Control
