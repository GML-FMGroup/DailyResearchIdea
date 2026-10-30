# Joint Factor-Graph Calibration of Primitive Semantics and Spatial Belief from Execution Feedback

## Motivation

Existing training-free multimodal controllers (e.g., the minimal-interface agent of Embodied Agents Take Control) treat primitive semantics and spatial state as given, using execution feedback only for re-planning. Structurally, action effects and spatial drift are confounded in monocular RGB, so neither can be estimated independently—yet no prior work imposes a joint identifiability constraint. This meta-gap recurs across frontier VLA (π0), map-based (MapGPT), and language-based (LangNav) agents, all of which assume a VLM backbone can ground percepts to actions zero-shot without ever calibrating either quantity from closed-loop data.

## Key Insight

A shared latent referent frame makes the per-primitive effect model and the persistent spatial belief jointly identifiable from the same execution-feedback stream, because the frame absorbs the gauge freedom that otherwise confounds action effects with spatial drift.

## Method

**What it is.** JF-CAL (Joint Factor-graph Calibration) is a training-free online estimator that maintains a joint posterior over (a) per-primitive continuous effect models—expected displacement, collision probability, occlusion footprint—and (b) a persistent landmark/obstacle map, both updated only from execution feedback (collision, drift, occlusion) and coupled by a shared latent referent frame. Inputs: monocular RGB, primitive execution logs (action, observed outcome). Output: a calibrated joint posterior used by the controller to select primitives via marginalization.

**Load-bearing assumption (stated explicitly).** The shared latent referent frame ξ is the only coupling between the per-primitive effect likelihood and the landmark-projection likelihood: conditioning on ξ renders θ_p and L conditionally independent, so the joint posterior factorizes as q(ξ) ∏_p q(θ_p|ξ) ∏_j q(l_j|ξ). This factorization is identifiable from execution feedback alone only when ξ is anchored—i.e., when the gauge freedom is fixed by an external constraint rather than estimated as a free latent. We therefore fix the gauge explicitly: ξ is anchored to a fixed world frame via a known-metric landmark subset (the subset of landmarks whose metric positions are provided by the simulator's ground-truth map at episode start, |L_anchor| = 5 landmarks) and an odometry prior (the agent's wheel/IMU odometry, available in R2R, provides a metric displacement prior with covariance Σ_odo = diag(0.01, 0.01, 0.001) in (x, y, θ)). Primitive effects are re-parameterized in this anchored frame: μ_p is the displacement in the world frame, not the body frame. When a primitive's effect is body-frame-defined (turns, strafes), we apply the known body-to-world transform from the current heading estimate before updating μ_p, so ξ and θ_p are not separable along the body-frame direction—this is the known gauge degeneracy, and we handle it by (i) the anchor constraint above and (ii) an explicit observability check: at each step we compute the rank of the joint Fisher information matrix for (ξ, θ_p) using the current residuals; if the smallest eigenvalue falls below 1e-3 (indicating an unobservable direction), we fall back to a coupled (non-factorized) update for that primitive—i.e., we update the joint q(ξ, θ_p) via a single Kalman update on the stacked state rather than coordinate ascent. This fallback is logged per episode as a drift-observability diagnostic (fraction of steps with rank deficiency).

**How it works.**
```pseudocode
# State: for each primitive p, effect params θ_p = (μ_p, Σ_p, c_p, o_p)
#        landmark map L = {l_j}, shared referent frame ξ ∈ SE(2) (anchored)
#        joint posterior q(θ, L, ξ) factorized as q(ξ) ∏_p q(θ_p|ξ) ∏_j q(l_j|ξ)
#        anchor set L_anchor ⊂ L with known metric positions (|L_anchor| = 5)
#        odometry prior: ξ_odo ~ N(ξ_prev + Δ_odo, Σ_odo), Σ_odo = diag(0.01, 0.01, 0.001)

Initialize q with broad priors: θ_p ~ N(0, 10·I), c_p ~ Beta(1,1), o_p ~ Beta(1,1);
  L empty except L_anchor initialized at known positions; ξ ~ N(0, 1·I) anchored to world frame.

for each execution step t:
    observe RGB I_t, execute primitive p_t, receive feedback f_t = (collision_t, drift_t, occlusion_t)
    # E-step: compute expected sufficient statistics under current q
    for each landmark j visible in I_t:
        compute predicted projection using ξ and L; residual r_j = observed − predicted
    # Observability check: compute smallest eigenvalue λ_min of joint Fisher information
    #   for (ξ, θ_p) from residuals {r_j} and drift_t; if λ_min < 1e-3, flag rank-deficient.
    # M-step: coordinate ascent on factor graph
    if not rank-deficient:
        update q(ξ) via Gaussian message passing on SE(2) using all residuals {r_j}, θ_p predictions,
          and odometry prior (ξ_odo, Σ_odo); anchor landmarks in L_anchor contribute fixed-position factors.
        for each primitive p seen in window W (last 50 steps):
            update q(θ_p|ξ) via Kalman filter with process noise Q=0.01, observation noise R from feedback
            update c_p, o_p via Beta-Bernoulli conjugacy
        for each landmark j:
            update q(l_j|ξ) via Kalman filter with drift_t as observation
    else:
        # coupled fallback: single Kalman update on stacked state (ξ, θ_p) for the affected primitive
        update q(ξ, θ_p) jointly via Kalman filter on stacked state with same Q, R
        update c_p, o_p via Beta-Bernoulli conjugacy
        update q(l_j|ξ) for landmarks as above
    # Marginalize for control
    for each candidate primitive p:
        compute expected displacement E[μ_p], collision prob E[c_p], occlusion E[o_p] under q
        score(p) = task_reward(p) − λ_c·E[c_p] − λ_o·E[o_p]  (λ_c=λ_o=0.5)
    select primitive with max score
    log drift-observability diagnostic: rank-deficient flag, λ_min, fraction of steps rank-deficient per episode
```

**Why this design.** We chose a factor-graph with a shared referent frame over independent per-primitive and per-landmark estimators because the shared frame is the minimal structure that breaks the action-effect/spatial-drift confound—without it, a collision could be attributed equally to a bad primitive or a drifted map, and neither posterior is identifiable; the cost is that the frame must be re-estimated online, which we handle via coordinate ascent. We chose conjugate Kalman/Beta updates over gradient-based inference because the feedback stream is sparse and non-stationary, and conjugacy gives closed-form updates that are robust to the small per-primitive sample sizes (often <10 executions); the trade-off is reduced expressiveness for non-Gaussian effects, which we accept because displacement and collision are well-modeled as Gaussian/Bernoulli. We chose marginalization over the joint posterior for control rather than MAP primitive selection because marginalization propagates uncertainty—a primitive with high expected reward but high collision variance is correctly penalized—accepting the cost of a slightly more expensive per-step computation (O(|P|·|L|) marginal updates). This design is not a variant of retrieval-based tool selection (Toolformer/Gorilla) because it computes a counterfactual posterior over primitive effects that retrieval cannot represent, nor is it a two-module router: the controller's decision rule is a closed-form marginalization of a single joint posterior, not a learned scheduler. The anchor constraint and observability check are not new mechanisms—they are the explicit gauge-fixing and diagnostic that make the stated factorization identifiable, as required by the load-bearing assumption.

**Why it measures what we claim.** The shared referent frame ξ measures the gauge freedom between action effects and spatial drift because we assume the frame is the only latent that couples the two likelihoods (displacement residuals and landmark projection residuals) and that ξ is anchored to a fixed world frame via known-metric landmarks and odometry; this assumption fails when a primitive's effect is itself frame-dependent (e.g., a turning primitive whose displacement is defined relative to the agent's heading), in which case ξ absorbs part of the effect and the posterior reflects a confounded estimate—we detect this via the rank-deficiency diagnostic and fall back to a coupled update. The per-primitive collision probability c_p measures the primitive's calibrated semantics because we assume Beta-Bernoulli conjugacy with a fixed prior is a sufficient model of collision outcomes conditional on the primitive; this fails when collisions are correlated across steps (e.g., repeated collisions with the same obstacle), in which case c_p overestimates the primitive's intrinsic collision rate. The marginalization score measures task-appropriate primitive selection because we assume the expected displacement and collision probability are sufficient statistics for the downstream reward; this fails when the reward depends on higher moments (e.g., risk-sensitive tasks), in which case the score reflects a risk-neutral approximation. The drift-observability diagnostic measures whether drift_t is a sufficiently informative observation to identify ξ independently of θ_p because we assume drift_t provides a metric displacement observation with covariance Σ_odo; this fails in featureless corridors where drift is unobservable, in which case the diagnostic flags rank deficiency and the ablation without ξ degrades uniformly rather than differentially.

## Contribution

['A joint factor-graph estimator (JF-CAL) that couples per-primitive effect models and a persistent spatial belief through a shared latent referent frame, making both identifiable from execution feedback alone without gradient updates to the VLM.', 'A closed-form marginalization-based primitive selection rule that propagates posterior uncertainty into control decisions, replacing the standard MAP/argmax selection used by training-free agents.', 'An online calibration procedure that requires only monocular RGB and primitive execution logs, compatible with any frontier multimodal controller.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|-----------------------|
| Dataset | R2R with execution feedback | Tests online calibration under realistic drift/collision |
| Primary metric | Success rate weighted by collision penalty | Directly measures calibrated primitive selection |
| Baseline | π0 (zero-shot) | Tests whether our online calibration beats static VLA |
| Baseline | MapGPT (zero-shot) | Tests whether joint factor-graph beats map-only prompting |
| Baseline | Embodied Agents (frontier VLM) | Tests whether execution feedback beats pure vision-language |
| Baseline | π0 + LoRA online fine-tuning | Tests whether training-free joint calibration is the right regime |
| Ablation-of-ours | JF-CAL w/o shared referent frame ξ | Isolates the frame's role in breaking confound |
| Ablation-of-ours | JF-CAL w/o anchor constraint (ξ free) | Tests whether anchoring is necessary for identifiability |
| Diagnostic | Drift-observability log (rank deficiency fraction) | Operationalizes the concentrated-gains prediction |

### Why this setup validates the claim

This setup tests the central claim that a joint factor-graph with a shared referent frame breaks the action-effect/spatial-drift confound and yields better primitive selection under sparse, non-stationary feedback. R2R with execution feedback provides the exact conditions where the confound arises: collisions and drift are ambiguous without a latent frame. The primary metric—success rate weighted by collision penalty—directly reflects the marginalization score's ability to penalize high-variance, high-collision primitives. π0 tests whether online calibration beats a strong static policy; MapGPT tests whether joint inference beats map-only prompting; the frontier VLM baseline tests whether execution feedback adds value beyond vision-language reasoning. The π0 + LoRA online fine-tuning baseline tests whether the training-free joint calibration is the right regime rather than just a cheaper alternative: if LoRA fine-tuning on the same execution-feedback stream matches or exceeds JF-CAL, the training-free claim is weakened. The ablation without ξ isolates the frame's causal role: if the frame is necessary to break the confound, removing it should cause uniform degradation across all subsets, not just where drift is high. The ablation without the anchor constraint (ξ free) tests whether the anchor is necessary for identifiability: if the free-ξ variant drifts along the unobservable direction, its performance should degrade specifically on episodes with high drift, confirming the load-bearing assumption. The drift-observability diagnostic logs per-episode rank deficiency, allowing us to test the prediction that gains are concentrated on episodes where drift is observable (low rank deficiency) and that the no-ξ ablation degrades uniformly on episodes where drift is unobservable. Together, these baselines and ablations form a falsifiable test: the method must show a specific pattern of gains concentrated where the confound is predicted to occur, and the ablations must show the opposite pattern.

### Expected outcome and causal chain

**vs. π0 (zero-shot)** — On a case where the agent repeatedly collides with a transparent obstacle and drifts, π0 produces a static action sequence because its policy is frozen and cannot update from execution feedback. Our method instead maintains a joint posterior over per-primitive collision probabilities and landmark positions, updating both from feedback, so we expect a noticeable gap on episodes with high drift/collision frequency but parity on clean episodes.

**vs. MapGPT (zero-shot)** — On a case where a landmark is mislocalized due to occlusion, MapGPT produces a wrong navigation plan because its map is updated only from vision-language prompts without probabilistic drift correction. Our method instead uses the shared referent frame to jointly re-estimate landmark positions and primitive effects, so we expect higher success on occluded-landmark subsets but similar performance on clear-view subsets.

**vs. Embodied Agents (frontier VLM)** — On a case where a primitive's effect is ambiguous (e.g., a turn that could be due to heading drift or a bad turn primitive), the frontier VLM produces a heuristic choice because it lacks a counterfactual posterior over primitive effects. Our method instead marginalizes over the joint posterior to penalize high-variance primitives, so we expect a concentrated gain on ambiguous-effect episodes but no gain on unambiguous ones.

**vs. π0 + LoRA online fine-tuning** — On episodes with sparse feedback (fewer than 10 executions per primitive), LoRA fine-tuning overfits to the small sample and degrades, because gradient-based updates require more data to converge. Our method instead uses conjugate closed-form updates that are robust to small sample sizes, so we expect JF-CAL to outperform LoRA on sparse-feedback episodes but to be comparable on episodes with dense feedback (more than 50 executions per primitive).

**Ablation: JF-CAL w/o shared referent frame ξ** — Without ξ, the per-primitive effect model and landmark map are estimated independently, so collisions and drift are confounded. We expect uniform degradation across all subsets, not just high-drift ones, because the confound affects all episodes equally. If degradation is concentrated on high-drift subsets instead, the frame's causal role is not isolated.

**Ablation: JF-CAL w/o anchor constraint (ξ free)** — With ξ estimated as a free latent, the gauge freedom is not fixed, so ξ and θ_p drift along the unobservable direction. We expect degradation specifically on episodes with high drift and body-frame primitives, and the drift-observability diagnostic should show high rank deficiency. If the free-ξ variant performs as well as the anchored version, the anchor constraint is unnecessary and the load-bearing assumption is weakened.

### What would falsify this idea

If the gain over baselines is uniform across all subsets rather than concentrated on episodes with high drift, collision ambiguity, or occlusion—and if the ablation without the shared referent frame shows no differential degradation—then the central claim that the frame breaks the confound is wrong. Additionally, if the free-ξ ablation matches the anchored version on high-drift episodes, the anchor constraint is not necessary for identifiability, falsifying the load-bearing assumption.

## References

1. Embodied Agents Take Control: Minimal-Interface Zero-Shot Agents Rival Industrial-Scale Policies in Vision-and-Language Navigation
2. π0: A Vision-Language-Action Flow Model for General Robot Control
3. MapGPT: Map-Guided Prompting with Adaptive Path Planning for Vision-and-Language Navigation
4. Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware
5. BridgeData V2: A Dataset for Robot Learning at Scale
6. Visual Instruction Tuning
7. LangNav: Language as a Perceptual Representation for Navigation
8. March in Chat: Interactive Prompting for Remote Embodied Referring Expression
