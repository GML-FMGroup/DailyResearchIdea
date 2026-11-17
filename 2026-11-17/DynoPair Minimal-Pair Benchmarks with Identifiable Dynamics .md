# DynoPair: Minimal-Pair Benchmarks with Identifiable Dynamics via Interactive Rollouts and Marginalized Outcome Scoring

## Motivation

DominoBench (arxiv:2611.20385) diagnoses frontier models as physically unreasonable, but its single-still-image observation model makes the very quantities that decide each minimal pair — velocity, force, mass — structurally unidentifiable rather than merely hard to infer, so a correct answer cannot be attributed to physical reasoning over geometric pattern matching, and the four-rung hint ladder cannot repair this because no rung supplies physical measurements. This is the meta-level bottleneck recurring across physical-reasoning benchmarks: PhysBench (E3) and CLEVR-AVR/PhysVLM-AVR (E2, E7) likewise evaluate physical understanding on observation channels that either omit or passively withhold the latent dynamic state, so low scores conflate model failure with benchmark-level non-identifiability.

## Key Insight

Pairing each DominoBench-style minimal pair with a matched interactive rollout that makes the deciding dynamic quantity observable turns outcome prediction into an identifiable estimation problem, so marginalizing over the residual unobserved dynamics yields a proper scoring rule whose calibration discriminatingly separates genuine physical reasoning from geometric pattern matching.

## Method

**What it is.** DynoPair is a benchmark construction and evaluation protocol. Input: a DominoBench-style minimal pair (two 2D rigid-body scenes differing by 1–3 grid steps in one deciding quantity) plus a physics simulator. Output: for each pair, a *matched rollout pair* (short trajectory or interactive probe sequence) in which the deciding quantity is either directly observable or manipulable, and per-model scores consisting of a marginalized outcome log-likelihood and an *identifiability margin* between the still-image and rollout conditions.

**Load-bearing assumption (stated explicitly).** IM(M) = score(C2) − score(C0) is attributable to the model resolving the simulator-labeled deciding quantity only if the model's p_hat(outcome | s) is a genuine belief over the latent state s and not a function of the observation channel itself. This assumption fails when the C2 probe hands the model the deciding quantity as prompt text (oracle leak) or when p_hat is uncalibrated so that any channel change inflates likelihood. The protocol below is constructed so that (i) the deciding quantity is never presented as text in C2 — only as a noisy observation the model must integrate — and (ii) IM is reported only on pairs passing an explicit calibration and placebo-probe check.

**How it works.**

```pseudocode
# Phase 1 — pair extraction and deciding-quantity labeling
for pair P = (A, B) in DominoBench_pairs:            # 427 pairs inherited
    # q = deciding_quantity(P): sensitivity sweep over each scalar in {pos, vel, mass, force}
    #   with all others held fixed; q is the scalar whose 1-3 grid-step change flips the
    #   fate-flip indicator I[outcome_A != outcome_B]; report the Jacobian dI/dq at the flip.
    q = deciding_quantity(P)
    tau = simulator.trajectory(A), simulator.trajectory(B)   # ground-truth latent state s_t = (pos, vel, mass, force)
    label P with q, tau, and the deciding step t* (first step where A,B diverge)

# Phase 2 — observation-channel ladder (four matched conditions per pair)
C0_still   = render(A, t=0), render(B, t=0)                       # DominoBench baseline
C1_frames  = render(A, t in {0..T}), render(B, t in {0..T})       # T = 8 frames, fixed stride
C2_interact= agent may issue k in {1..3} probe actions per twin   # e.g. nudge, drop marker, query velocity at a chosen body
             # probe returns a NOISY, DISCRETIZED measurement of the queried body's
             # (vel, force, mass): vel quantized to 0.1 m/s bins, force to 0.5 N bins,
             # mass to 0.1 kg bins, each with additive Gaussian noise sigma = 0.05 * range.
             # The deciding quantity is NEVER returned as exact text; the model receives
             # only the binned noisy reading and must integrate it into its posterior.
C2p_placebo= identical probe budget to C2, but probes are restricted to a NON-deciding body
             # (the body whose sensitivity Jacobian dI/dq is smallest); bounds the generic
             # interaction-affordance confound.

# Phase 3 — marginalized outcome scoring (same marginal for C0, C1, C2, C2p)
for each model M and condition C in {C0, C1, C2, C2p}:
    for each pair P:
        p_hat = M.predict_outcome(C(P))               # distribution over {A-survives, B-survives, both, neither}
        # analytic marginalization over residual unobserved dynamics:
        #   for C0: integrate over prior over initial velocities (uniform over simulator-reachable range)
        #   for C1: integrate over velocities consistent with finite-difference of observed frames;
        #           propagate finite-difference uncertainty as sigma_v = sqrt(2)*sigma_pos/dt
        #   for C2: integrate over only the quantities the agent did NOT probe, using the
        #           binned-noisy probe reading as a likelihood term (not as a point value)
        #   for C2p: same as C2 but the probed body is non-deciding, so the deciding-quantity
        #           posterior is unchanged from C0
        score(P, C) = log sum_{s in posterior(C)} p_hat(outcome | s) * w(s)
    # Identifiability margin:
    IM(M) = mean_P [ score(P, C2) - score(P, C0) ]
    # Placebo-corrected margin:
    IM_placebo(M) = mean_P [ score(P, C2) - score(P, C2p) ]

# Phase 4 — calibration gate and chance-corrected attribution
# Calibration check: ECE on a held-out 20% of pairs (split by pair, not by twin).
#   Pairs where the C0 posterior is degenerate (entropy < 0.1 nats) are excluded from IM.
#   IM is reported only on pairs passing ECE <= 0.10 for the model under C0.
# Geometric-pattern baseline: ignores dynamics, predicts from still image via hint-ladder geometry only.
report per-model: marginalized log-likelihood, IM, IM_placebo, ECE, and the geometric-pattern baseline.
```

**Why this design.** We chose a *matched-pair rollout* over simply adding frames because DominoBench's scientific claim rests on controlled geometric perturbation, so the rollout must perturb the same quantity by the same 1–3 grid steps to keep the causal contrast intact, accepting the cost that rendering and simulator bookkeeping triple per item. We chose *analytic marginalization over simulator-reachable priors* over a learned uncertainty head because the deciding quantity's support is simulator-defined, so the marginal is a well-posed integral rather than a calibration guess, accepting that it is only as good as the simulator's prior. We chose *four matched observation conditions (still / frames / interactive / placebo-interactive)* over a single richer channel because the identifiability margin only becomes measurable when the same pair is scored under all conditions and the placebo bounds the interaction-affordance confound, accepting the extra evaluation budget. We chose *noisy, discretized probe returns* over simulator-exact returns because exact returns would leak the deciding quantity as prompt text, turning IM into a measure of 'did we tell the model the answer' rather than physical reasoning. A domain expert would not describe this as 'DominoBench plus video' or as 'PhysVLM-AVR applied to DominoBench': PhysVLM-AVR (E2, E7) trains an embodied MLLM on CLEVR-AVR and *reports* active-reasoning gaps, whereas DynoPair *constructs* the identifiability bridge — the deciding quantity is simulator-labeled per pair and the marginalization is over a known support — so the contribution is the benchmark's identifiability property, not a new policy.

**Why it measures what we claim.** The computational quantity `IM(M) = score(C2) − score(C0)` measures *whether a model's correctness is attributable to physical reasoning rather than geometric pattern matching* because the marginalization in C0 and C2 differs only in which dynamic quantities are unobserved, so any score gap must come from resolving those quantities; this assumption fails when a model's still-image prediction is already driven by a geometric shortcut that happens to correlate with the deciding quantity, in which case IM reflects shortcut robustness rather than reasoning. The placebo-corrected `IM_placebo(M) = score(C2) − score(C2p)` measures *whether the gain is specific to the labeled deciding quantity* because C2p uses the same probe budget on a non-deciding body, so any residual gap must come from resolving the deciding quantity rather than generic interaction affordance; this assumption fails when the non-deciding body is itself informative about the deciding quantity, in which case IM_placebo reflects probe informativeness rather than labeled identifiability. The per-pair `deciding_quantity(P)` label measures *causal necessity of the perturbation* because the simulator's flip of fate under a 1–3 grid-step change is the operational definition of the pair's minimality; this assumption fails when two quantities co-vary within the grid-step budget, in which case the label reflects a confounded perturbation. The `log sum_s p_hat(outcome|s) w(s)` term measures *calibrated outcome belief under explicit uncertainty* because the weights w(s) are the simulator-induced posterior over unobserved dynamics; this assumption fails when the model's p_hat is itself uncalibrated (Anti-Pattern 3 risk), in which case the score reflects likelihood fit to a miscalibrated head rather than physical belief. We therefore report IM alongside raw accuracy and an ECE gate (IM reported only on pairs where C0 ECE ≤ 0.10) so that the attribution claim does not rest on calibration alone.

## Contribution

(1) A benchmark construction method, DynoPair, that upgrades DominoBench-style minimal pairs into matched still/frames/interactive observation conditions with simulator-labeled deciding quantities and analytic marginalization over unobserved dynamics. (2) An evaluation protocol introducing the identifiability margin IM(M) = score(C2) − score(C0), a chance-corrected quantity that separates physical-reasoning attribution from geometric pattern matching. (3) A reusable labeling pipeline that extracts deciding quantity, deciding step, and trajectory for any 2D rigid-body minimal pair from a simulator.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | DominoBench 427 minimal pairs, stratified subset of 100 (balanced dynamic vs. statically-visible deciding quantities) for first experiment | Controlled geometric perturbation isolates deciding quantity |
| Primary metric | Identifiability margin IM(M) and placebo-corrected IM_placebo(M) | Gap between interactive and still/placebo conditions isolates dynamic resolution |
| Baseline | Frontier VLM unaided (C0) | Tests whether still-image accuracy is shortcut-driven |
| Baseline | Frontier VLM with hint ladder | Tests whether geometry hints substitute for dynamics |
| Baseline | PhysVLM-AVR-style active policy | Tests whether generic active probing resolves the quantity |
| Baseline | Placebo-probe control C2p (non-deciding body) | Bounds generic interaction-affordance confound |
| Ablation-of-ours | DynoPair without C1 frames (C0 vs C2 only) | Isolates value of matched rollout channel |
| Calibration gate | ECE ≤ 0.10 on held-out 20% of pairs; IM reported only on passing pairs | Prevents uncalibrated head from inflating C2 likelihood |

### Why this setup validates the claim

The design is falsifiable because each baseline attacks a distinct sub-claim of the central thesis: that outcome correctness on minimal pairs is attributable to physical reasoning rather than geometric pattern matching. The unaided C0 baseline tests the shortcut sub-claim — if a model already scores well on C0, any IM gain must come from genuinely resolving unobserved dynamics, not from the still image. The hint-ladder baseline tests whether explicit geometric information substitutes for dynamic quantities; if hints close the gap, the deciding quantity is not the bottleneck and IM should collapse. The PhysVLM-AVR-style active policy tests whether generic active exploration transfers to the deciding quantity, distinguishing our simulator-labeled identifiability bridge from a learned probing heuristic. The placebo-probe control C2p tests whether any C2 advantage is specific to the labeled deciding quantity or merely a generic interaction affordance; IM_placebo = score(C2) − score(C2p) must be positive and concentrated on dynamic-decided pairs for the attribution to hold. The ECE gate ensures that IM is not an artifact of an uncalibrated head. The ablation removes the intermediate frame condition, so any IM that survives without frames but vanishes with them indicates the margin is driven by finite-difference velocity cues rather than interactive resolution. IM is the right metric because it holds the pair fixed and varies only which dynamic quantities are observed, so the marginalization support differs exactly where the claim locates the effect.

### Expected outcome and causal chain

**vs. Frontier VLM unaided (C0)** — On a pair where the deciding quantity is an initial speed the still image cannot show, the unaided model produces a fate prediction driven by ramp geometry and object arrangement, because its visual prior encodes static configuration, not hidden velocity. Our method instead marginalizes over the simulator-reachable velocity prior in C0 and over only unprobed quantities in C2, so the same model's C2 prediction is scored against a narrower, correct support once the agent queries velocity. We expect a pronounced IM gap concentrated on pairs whose deciding quantity is dynamic (velocity, force) and near-zero IM on pairs whose deciding quantity is statically visible (mass, position).

**vs. Frontier VLM with hint ladder** — On a pair where the deciding step is pointed out and exact geometry is given as text, the hinted baseline still mispredicts because text geometry supplies positions but not the speeds the outcome depends on. Our method instead lets the C2 agent probe the queried body and receive a noisy, discretized velocity reading, so the deciding quantity enters the posterior as a likelihood term rather than as exact text. We expect hint-condition accuracy to improve over C0 but IM to remain large, since hints narrow the geometric support without supplying the dynamic support — the diagnostic is a persistent C2-minus-hinted gap on velocity-decided pairs.

**vs. PhysVLM-AVR-style active policy** — On a pair where the agent must choose among nudge, marker-drop, and velocity-query probes, the generic active policy produces inefficient or off-target probes because its information-gain heuristic was trained on CLEVR-AVR scenes, not on deciding-quantity identification. Our method instead labels the deciding quantity by simulator introspection and restricts probes to channels that resolve it, so probe sequences converge on the deciding body. We expect the AVR-style policy to show positive but smaller IM than DynoPair, with the gap concentrated on pairs where the deciding quantity is not the most visually salient object.

**vs. Placebo-probe control C2p** — On a pair where the agent probes a non-deciding body, the deciding-quantity posterior is unchanged from C0, so score(C2p) should be statistically indistinguishable from score(C0). We expect IM_placebo(M) ≈ IM(M) for models that genuinely resolve the deciding quantity, and IM_placebo(M) ≈ 0 for models whose C2 gain is a generic interaction affordance. A model with positive IM but zero IM_placebo is flagged as affordance-driven, not reasoning-driven.

### What would falsify this idea

If IM is uniform across pairs regardless of whether the deciding quantity is dynamic or statically visible — rather than concentrated on velocity- and force-decided pairs — then the margin reflects general probe-condition advantage, not resolution of the labeled deciding quantity, and the identifiability claim is wrong. Likewise, if the hint-ladder baseline closes the C2-minus-C0 gap entirely, or if IM_placebo is not positive on dynamic-decided pairs, the deciding quantity is not the bottleneck and the central attribution fails.

## References

1. DominoBench: Evaluating Multi-Step Physical Reasoning from Still Images
