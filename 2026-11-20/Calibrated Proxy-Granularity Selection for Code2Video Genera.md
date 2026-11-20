# Calibrated Proxy-Granularity Selection for Code2Video Generation

## Motivation

Task-adaptive selection of proxy granularity, physics simulation, and prompting in Coda-DaVinci is heuristic: the framework chooses between proxy-guided and direct generation based on task characteristics and assigns granularity per entity, but the paper itself reports that 'how to construct proxies and prompts for optimal downstream quality remains underexplored' and that trade-offs are 'task-dependent' (coarse proxies favor natural motion, medium-grained favor consistency). The root cause is that the selection criterion is a qualitative reading of task type rather than a measurable function of the configuration to the two competing quality axes, so the agent cannot know which configuration maximizes trajectory alignment under a scene-consistency constraint without running full generation for every candidate. FloVD and IM-Zero exhibit the same residual gap: both expose control knobs (flow representation, keyframe/mask layouts) whose effect on alignment versus consistency is asserted qualitatively rather than predicted.

## Key Insight

Because trajectory alignment and scene consistency respond to proxy granularity in opposing, monotone directions, a small calibrated regressor trained on cheap low-resolution rollouts can predict both axes from a configuration descriptor, turning an uncalibrated heuristic choice into a constrained maximization whose argmax is identifiable from a handful of probes rather than full generations.

## Method

**What it is.** CGSP (Calibrated Granularity-Selection Predictor) takes a scene specification and a candidate configuration set — each configuration is a tuple (proxy granularity level g ∈ {coarse, medium, fine}, physics-simulation flag and substep count, prompting mode) — and outputs the selected configuration that maximizes predicted trajectory-alignment score subject to predicted scene-consistency ≥ τ. It sits in front of Coda-DaVinci's generator and replaces its heuristic mode/granularity choice.

**How it works.**

```pseudocode
Input: scene spec S, candidate configs C = {c_1..c_K} (K ≤ 18), constraint τ=0.80, probe budget B=6, calibration set D_cal of 30 scenes with cached full-generation scores for all K configs

# Phase 0: calibration of probe-to-full rank preservation (pre-registered, run once)
for each scene s in D_cal:
    for each config c in C:
        v_full = full_generate(s, c)          # cached
        a_full = trajectory_alignment(v_full, s.trajectories)
        s_full = scene_consistency(v_full, s.assets)
        v_probe = low_res_generate(s, c)      # 8 frames, 4 denoise steps, 1/40 cost
        a_probe = trajectory_alignment(v_probe, s.trajectories)
        s_probe = scene_consistency(v_probe, s.assets)
    compute Spearman ρ_a = rank_corr(a_probe, a_full) over all c
    compute Spearman ρ_s = rank_corr(s_probe, s_full) over all c
    if ρ_a < 0.7 or ρ_s < 0.7: abort and report rank-preservation failure
    fit per-granularity empirical error bounds: for each g in {coarse, medium, fine},
        σ_a(g) = std(a_full - a_probe) over c with granularity g
        σ_s(g) = std(s_full - s_probe) over c with granularity g

# Phase 1: descriptor construction (no generation)
for each c in C:
    d(c) = [ g_onehot(c) (3) , physics_flag, log(1+substeps),
             prompt_mode_onehot (2), entity_count(S),
             mean_entity_scale(S), motion_magnitude(S) ]   # 9-dim
    # reduce to 5-dim via fixed low-rank basis: PCA fit on D_cal descriptors, keep 5 components
    d_red(c) = W^T d(c)   # W: 9x5, fixed from calibration

# Phase 2: cheap probe rollouts (low-res, 8 frames, 4 denoise steps)
P = stratified_sample(C, B)                      # cover each granularity level
for c in P:
    v_c = low_res_generate(S, c)                 # ~1/40 cost of full gen
    a_c = trajectory_alignment(v_c, S.trajectories)   # tracks vs. spec
    s_c = scene_consistency(v_c, S.assets)            # DINO/CLIP asset match

# Phase 3: fit two calibrated ridge regressors on probe residuals
f_a(d_red) -> a_hat ; f_s(d_red) -> s_hat          # closed-form ridge, λ=1e-2
# uncertainty from per-granularity empirical bounds, not homoscedastic residual
for each c in C:
    g = granularity(c)
    sigma_a(c) = σ_a(g) ; sigma_s(c) = σ_s(g)

# Phase 4: constrained selection with uncertainty margin
for c in C: predict (a_hat, s_hat, sigma_a, sigma_s)
c* = argmax_c a_hat(c) - 1.0*sigma_a(c)
     s.t.  s_hat(c) - 1.0*sigma_s(c) >= τ
return c*   # passed to Coda-DaVinci's multi-agent harness for full generation
```

The selection rule is a single constrained argmax over a closed-form regressor; there is no router choosing between sub-systems and no log-prob gating.

**Why this design.** We chose a low-resolution probe rollout over full-generation scoring because full scoring makes selection cost identical to exhaustive search, accepting the cost that probe statistics are noisier than full-video statistics and must be corrected by a residual-variance term. We chose a closed-form ridge regressor over a learned neural predictor because the candidate set is tiny (K ≤ 18) and probe count is budgeted at B=6, so a high-capacity model would overfit the probes; the cost is that the predictor is only valid within the descriptor range spanned by probes, which we mitigate by stratified probe sampling across all three granularity levels. We chose a constrained argmax with an explicit uncertainty margin over a soft weighted-sum objective because the motivation treats alignment and consistency as an invariant (consistency is a floor, not a tradeable currency), accepting the cost that when no configuration clears τ the agent must fall back to the highest-consistency candidate and flag the scene. We chose a 9-dimensional hand-specified descriptor over a learned embedding because the selection must be auditable and reproducible across scenes. A domain expert would not describe this as a variant of Coda-DaVinci's task-adaptive heuristic: Coda-DaVinci reads task type and assigns granularity, whereas CGSP fits a calibrated mapping from configuration to both quality axes and solves a constrained program whose feasibility is checked before generation.

**Why it measures what we claim.** The probe-derived trajectory-alignment score `a_c` measures downstream trajectory alignment because we assume low-resolution, short-horizon rollouts preserve the rank ordering of alignment across configurations — i.e., alignment is dominated by proxy geometry rather than pixel detail; this assumption fails when fine-grained prompting changes trajectory adherence only at high resolution, in which case `a_c` reflects coarse motion fidelity rather than final alignment. The probe-derived scene-consistency score `s_c` measures downstream scene consistency because we assume asset-preservation failures (identity drift, object disappearance) manifest already in 8-frame low-res probes; this assumption fails for slow-drift identities that only diverge after many frames, in which case `s_c` reflects short-horizon stability. The ridge predictor `f_a(d_red)` measures the configuration-to-alignment function because we assume the mapping is locally smooth and approximately linear in the granularity and physics descriptors within the probe span; this assumption fails at granularity boundaries where coarse-to-medium transitions are discontinuous, in which case `f_a` reflects an interpolated trend that may misrank boundary candidates. The per-granularity empirical error bound `sigma_a(g)` measures predictive uncertainty because we assume probe residuals are homoscedastic within each granularity level; this assumption fails in descriptor regions with no probes, in which case `sigma_a(g)` understates true error and the margin is too small.

## Contribution

(1) A calibrated configuration-to-quality predictor (CGSP) that maps proxy granularity, physics-simulation settings, and prompting mode to predicted trajectory-alignment and scene-consistency scores, replacing Coda-DaVinci's heuristic task-adaptive selection with a constrained argmax over a closed-form regressor fitted on cheap low-resolution probes. (2) A design principle for controllable video generation: alignment and consistency respond to proxy granularity in opposing monotone directions, so a consistency-floor constraint with an uncertainty margin yields identifiable selections from a budget of six probes rather than exhaustive full generation. (3) A probe-budget protocol and descriptor specification that makes configuration selection auditable and reproducible across scenes.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Coda-DaVinci VTR and R2M task suites | Same scenes where adaptive granularity was studied |
| Primary metric | Trajectory-alignment score subject to scene-consistency ≥ τ | Directly tests constrained selection objective |
| Baseline | Coda-DaVinci task-adaptive heuristic | Current heuristic mode/granularity choice to be replaced |
| Baseline | Direct generation (no proxy) | Tests whether proxy selection matters at all |
| Baseline | Uniform medium-granularity fixed config | Tests whether adaptivity beats a strong fixed default |
| Ablation-of-ours | CGSP without uncertainty margin (point-estimate argmax) | Isolates contribution of calibrated residual variance |
| Ablation-of-ours | CGSP with homoscedastic residual std (original) | Tests per-granularity empirical bounds |
| Calibration study | Spearman rank correlation between probe and full scores | Validates rank-preservation assumption |

### Why this setup validates the claim

The central claim is that a calibrated, probe-based constrained selection over configurations beats Coda-DaVinci's task-type heuristic at maximizing trajectory alignment while keeping scene consistency above a floor. The VTR and R2M suites are the right testbed because they are exactly the scenes where Coda-DaVinci's own controlled study showed task-dependent trade-offs between proxy granularity, physics simulation, and prompting — so any gain must come from better selection, not from a new generator. The Coda-DaVinci heuristic baseline tests the sub-claim that reading task type and assigning granularity is insufficient; direct generation tests whether proxies help at all; the fixed medium-granularity baseline tests whether adaptivity per se is the source of gain rather than merely using a proxy. The no-margin ablation is the diagnostic: if the uncertainty margin is doing real work, removing it should degrade consistency-floor satisfaction specifically on scenes near the τ boundary, where probe residuals are largest. The homoscedastic-residual ablation tests whether per-granularity empirical bounds improve calibration over a single residual std. The calibration study directly measures Spearman rank correlation between probe and full scores on a held-out set of 30 scenes, providing the empirical basis for the rank-preservation assumption; if correlation falls below 0.7, the probe-based selection is invalid and the method must be revised. Trajectory alignment under a consistency floor is the right primary metric because it directly instantiates the constrained objective — a method that wins alignment while dropping below τ has not solved the stated problem, and a method that wins consistency trivially by picking coarse proxies will lose alignment.

### Expected outcome and causal chain

**vs. Coda-DaVinci task-adaptive heuristic** — On a scene mixing a fast-moving foreground entity with a slowly drifting background asset, the heuristic assigns granularity from task type alone and lands on a medium proxy whose coarse motion is natural but whose identity drift only surfaces late, producing consistency just under τ or alignment that ignores the fast entity. Our method instead probes all three granularity levels cheaply, fits the alignment and consistency surfaces, and picks the configuration whose lower-confidence consistency still clears τ while maximizing alignment. We expect a noticeable alignment gap on mixed-motion scenes but near-parity on homogeneous scenes where the heuristic's default already happens to be right.

**vs. Direct generation (no proxy)** — On a scene with strong physical dynamics and multiple interacting entities, direct generation produces plausible-looking but trajectory-inaccurate motion because it has no explicit structural handle, so alignment stays low regardless of prompting. Our method routes through Coda-DaVinci's proxy-guided generator with a selected physics flag and substep count, so alignment is structurally anchored. We expect a large alignment gap concentrated on high-motion, multi-entity scenes and smaller gaps on static or single-entity scenes where direct generation already tracks well.

**vs. Uniform medium-granularity fixed config** — On a scene with one large, finely detailed asset and one small fast object, the fixed medium config favors consistency for the large asset but under-resolves the fast object's trajectory, so alignment drops. Our method selects a per-entity adaptive configuration whose probe-measured alignment is higher while consistency stays above τ. We expect the gain to appear specifically on heterogeneous-scale scenes and to vanish on scenes where all entities share similar scale and motion, which is the signature that adaptivity, not proxy use, drives the improvement.

**vs. CGSP without uncertainty margin** — On scenes near the τ boundary, the point-estimate argmax ignores probe noise and may select a configuration whose true consistency falls below τ. Our method uses per-granularity empirical error bounds to add a safety margin, reducing consistency-floor violations. We expect the no-margin ablation to have higher violation rate specifically on boundary scenes, with similar alignment on scenes far from τ.

**vs. CGSP with homoscedastic residual std** — The original homoscedastic assumption underestimates uncertainty for fine granularity where probe residuals are larger. Our per-granularity bounds correct this. We expect the homoscedastic version to have more violations on fine-granularity selections, while our method maintains the consistency floor.

**Calibration study** — We expect Spearman ρ_a and ρ_s to exceed 0.7 on the 30-scene calibration set, validating the rank-preservation assumption. If ρ falls below 0.7, the probe-based selection is invalid and the method must be revised.

### What would falsify this idea

If CGSP's gain over the Coda-DaVinci heuristic is uniform across scene subsets rather than concentrated on mixed-motion, heterogeneous-scale scenes where the heuristic's task-type default is demonstrably wrong, then the calibrated selection is not doing what we claim and the improvement comes from some unrelated confound. Likewise, if the no-margin ablation matches CGSP on consistency-floor violations near τ, the residual-variance calibration is inert and the central mechanism is unsupported. Additionally, if the calibration study yields Spearman rank correlation below 0.7 between probe and full scores, the rank-preservation assumption fails and the probe-based selection is invalid.

## References

1. Coda-DaVinci: Reimagining Video Generation with Code2Video
