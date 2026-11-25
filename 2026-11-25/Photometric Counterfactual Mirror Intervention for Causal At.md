# Photometric Counterfactual Mirror Intervention for Causal Attribution of Reflection-Induced Manipulation Failures

## Motivation

RoboDojo-Mirror (the ancestor paper) reports that mirror conditions reduce manipulation success rates by 10.8–62.5 percentage points and that 50–88% of mirror-specific failures occur without robot–panel contact, but it attributes failures to reflections via trajectory and contact analysis rather than a controlled intervention on the visual input — so the reflection-to-failure pathway remains correlational. Structurally, the benchmark's conditions co-vary reflection with scene layout, occlusion geometry, and physical obstacle presence, so no condition-level contrast can isolate the photometric reflection channel as the causal factor; the same confound recurs across robustness benchmarks that vary whole scenes instead of single input channels (e.g., viewpoint-shift studies like ReViWo also entangle camera pose with occlusion and appearance).

## Key Insight

Because a mirror's reflected-content term is a purely photometric additive component over otherwise identical geometry, rendering the same scene with and without that term yields an intervention that is exogenous to scene layout, occlusion, and contact — making the reflection-to-failure effect identifiable rather than merely correlated.

## Method

**What it is.** We introduce **PhCF-Mirror** (Photometric Counterfactual Mirror protocol), a rendering-level intervention harness plus a decision-point representational probe. Inputs: a task scene with a mirror asset, a fixed policy checkpoint, and a fixed robot initial state. Outputs: a per-policy paired causal effect estimate of the reflected-content channel on task success, a per-policy paired estimate of the geometric/occlusion component, and a per-layer representation-divergence profile at the decision timestep.

**Load-bearing assumption (stated explicitly).** The causal estimand is the *reflected-content channel*: the semantic content of the image the mirror reflects. We assume this channel can be varied while holding geometry, occlusion, shadowing, and global illumination fixed, by replacing the reflected scene content with a content-swapped reflection of a different scene region rendered through the same mirror surface. We do **not** assume that a mirror's reflection term is a separable additive shading component that can be zeroed; zeroing it would also remove the mirror's cast shadow and indirect-illumination contribution. The three-way decomposition below is designed so that the estimand survives even when the mirror casts shadows or occludes the scene.

**How it works.**

```pseudocode
# Phase 0: renderer and coverage pre-commitment
RENDERER = Mitsuba 3 path tracer, max_depth=8, spp=512, denoiser=OIDN
# reflection/indirect-bounce toggle exposed as a scene-graph parameter
TASKS = RoboDojo-Mirror tasks whose mirror asset is present in the scene graph
POLICIES_WHITEBOX = checkpoints exposing .hidden_states() (report count up front)
POLICIES_BLACKBOX = checkpoints exposing only actions (behavioral tau only)

# Phase 1: three-way paired counterfactual rendering (geometry/occlusion/shadow/GI fixed)
for task in TASKS:
    scene = load_scene(task, mirror_asset)          # geometry, lighting, camera fixed
    for seed in range(N_SEEDS=20):                   # paired seeds, same init state
        I_a = render(scene, reflected_content=TRUE_SCENE)      # full mirror
        I_b = render(scene, reflected_content=DIFFUSE_MATCH)   # matched diffuse, same geometry/albedo stats
        I_c = render(scene, reflected_content=SCRAMBLED)       # phase-scrambled / content-swapped reflection
        # sanity gate: pixel-diff support must lie inside mirror_surface_mask
        assert pixel_diff_support(I_a, I_c) ⊆ mirror_surface_mask
        assert pixel_diff_support(I_a, I_b) ⊆ mirror_surface_mask ∪ shadow_mask
        # Phase 2: policy rollout under paired inputs
        for policy in POLICIES:
            s_a, traj_a = rollout(policy, I_a, seed)
            s_c, traj_c = rollout(policy, I_c, seed)
            s_b, traj_b = rollout(policy, I_b, seed)
            record_paired(task, policy, seed, s_a, s_c, s_b, traj_a, traj_c, traj_b)

# Phase 3: causal effect estimation
tau[policy]      = paired_mean_diff(s_a, s_c)          # reflected-content channel (primary)
tau_geom[policy] = paired_mean_diff(s_a, s_b)          # geometric/occlusion component (reported separately)
tau_free[policy] = paired_mean_diff(s_a, s_c, subset=no_contact_events)
first_div[policy] = fraction of pairs with action divergence at t < t_first_contact

# Phase 4: decision-point representation probe (white-box policies only)
for policy in POLICIES_WHITEBOX:
    for pair in pairs:
        t_dec = align_decision_timestep(pair)   # action-chunk boundary alignment; DTW fallback
        h_a, h_c = policy.hidden_states(I_a, I_c, at=t_dec)
        D[policy, layer] = cosine_distance(h_a, h_c)
    corr[policy] = spearman(D[policy, layer*], |tau[policy]|)
    # chunking-aware control: recompute D at chunk boundaries only
    D_chunk[policy, layer] = cosine_distance at chunk-boundary timesteps
```

**Why this design.** We chose a *three-way reflected-content decomposition (a: true scene, b: matched diffuse, c: scrambled content) over a single reflection-term ON/OFF toggle* because zeroing a mirror's reflection term in a path tracer also removes its cast shadow and indirect-illumination contribution, so an ON/OFF contrast would mix photometric and geometric effects; the (a)-(c) contrast holds geometry, occlusion, shadowing, and global illumination fixed and varies only the semantic content of the reflected image, while the (a)-(b) contrast is reported separately as the geometric/occlusion component. We chose *paired seeds with identical initial states* over independent sampling because pairing cancels task-difficulty variance and turns a between-condition comparison into a within-pair difference, accepting that the estimate is only defined for scenes where all three renders are valid. We chose *a decision-point hidden-state probe over post-hoc trajectory saliency* because trajectory analysis in RoboDojo-Mirror cannot distinguish a policy that is fooled at perception from one that is fooled at control; the cost is that hidden states are only accessible for white-box checkpoints, so black-box VLA policies receive only the behavioral effect estimate. We chose *a contact-free subset decomposition* over conditioning on contact in a regression because contact is a post-treatment variable — conditioning on it would reintroduce collider bias — so subsetting by the pre-treatment absence of contact preserves the causal contrast. A domain expert would not describe this as a variant of RoboDojo-Mirror's benchmark comparison, because the load-bearing operation is a rendering intervention that holds geometry, occlusion, shadowing, and global illumination fixed while varying only reflected content, not a new condition in a condition sweep.

**Why this measures what we claim.** The paired mean difference `tau[policy]` measures the causal effect of the reflected-content channel on task success because the only quantity varied between renders (a) and (c) is the semantic content of the reflected image, under the assumption that the renderer's content-swap preserves geometry, occlusion, shadowing, and global illumination; this assumption fails when the content swap changes the mirror's indirect-illumination contribution to the scene, in which case `tau` reflects a mixture of reflected-content and indirect-lighting effects. The contrast `tau_geom[policy]` measures the geometric/occlusion component because renders (a) and (b) differ in surface reflectance and shadowing, under the assumption that the matched diffuse surface has identical geometry and albedo statistics; this fails when the diffuse match alters the mirror's cast shadow, in which case `tau_geom` conflates occlusion with reflectance. The contact-free subset effect `tau_free` measures the reflected-content pathway net of physical obstruction because restricting to pairs with no recorded contact removes the collision mediator, under the assumption that contact events are recorded without false negatives; this fails when contact is missed by the sensor, in which case `tau_free` still contains collision-driven failures. The first-divergence fraction `first_div` measures whether reflected content acts at perception-before-action because it checks action divergence prior to the first contact timestep, under the assumption that the policy's action at t depends only on observations up to t; this fails for policies with action chunking that commit to multi-step plans, in which case early divergence may reflect a later committed plan. The layer-wise cosine distance `D[policy, layer]` measures representational sensitivity to the reflected-content channel because it compares hidden states across the paired renders at the decision timestep, under the assumption that the decision timestep is temporally aligned across the pair (via action-chunk boundary alignment, with DTW fallback); this fails when the two rollouts diverge in timing, in which case `D` reflects temporal misalignment rather than reflected-content sensitivity. The chunk-boundary control `D_chunk` measures the same sensitivity net of chunking-induced misalignment because it recomputes divergence only at chunk boundaries, under the assumption that chunk boundaries are comparable across the pair; this fails when chunk boundaries themselves shift between renders, in which case `D_chunk` still contains alignment artifacts.

## Contribution

(1) A rendering-level photometric counterfactual intervention protocol (PhCF-Mirror) that isolates the mirror reflection channel from scene layout, occlusion, and physical-contact confounds, converting the correlational attribution in RoboDojo-Mirror into a paired causal effect estimate. (2) A decision-point representational probe that links per-layer hidden-state divergence under the photometric intervention to the magnitude of the behavioral effect, providing a white-box diagnostic of whether reflection acts at perception or control. (3) A contact-free subset decomposition that separates the photometric reflection pathway from the physical-obstruction pathway without conditioning on a post-treatment variable.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | RoboDojo-Mirror tasks whose mirror asset is present in the scene graph and passes the three-way sanity gate | Only scenes where reflected-content swap is valid |
| Primary metric | Paired success-rate difference tau = (a)-(c) per policy | Within-pair causal contrast cancels task difficulty |
| Secondary metric | Paired success-rate difference tau_geom = (a)-(b) per policy | Separates geometric/occlusion component |
| Baseline | RoboDojo-Mirror panel-swap comparison | Geometry-confounded attribution to beat |
| Baseline | Independent-seed mirror vs mirror-free sampling | Unpaired estimate to show pairing gain |
| Baseline | Post-hoc trajectory saliency attribution | Behavioral-only attribution to show probe value |
| Ablation-of-ours | Unpaired rendering toggle without contact-free subset | Isolates pairing and mediation decomposition |
| Ablation-of-ours | Single ON/OFF reflection toggle (no three-way decomposition) | Shows the geometric confound the decomposition removes |
| Validation | Mirror facing away from all task-relevant objects (tau≈0 expected) | Tests that intervention isolates intended channel |
| Validation | Synthetic injected reflection as only difference (tau>0 expected) | Tests sensitivity to reflected content alone |

### Why this setup validates the claim

The central claim is that the reflected-content channel exerts a causal, perception-level effect on manipulation success that is separable from geometric obstruction, and that this effect is visible in policy representations at the decision point. The panel-swap baseline tests the first sub-claim: it is the strongest existing attribution, but it changes geometry and occlusion alongside appearance, so if our paired reflected-content contrast yields a materially different effect estimate on the same scenes, the confound is real and our estimate is the cleaner one. The independent-seed baseline tests the pairing sub-claim: if within-pair differences are tighter and larger in magnitude than between-condition differences, pairing is doing the variance-cancellation work we claim. The trajectory-saliency baseline tests the probe sub-claim: it attributes failure behaviorally, so if our layer-wise divergence correlates with effect size on contact-free pairs where behavior alone is ambiguous, the representational probe adds discriminating information. The ablation removes pairing and the contact-free decomposition, so any gap between it and the full method localizes the gain to the causal machinery rather than to rendering novelty. The single-toggle ablation tests the load-bearing assumption directly: if the (a)-(c) and ON/OFF estimates diverge on scenes where the mirror casts shadows or occludes the scene, the three-way decomposition is doing the work we claim. The two validation scenes test identifiability: a mirror facing away from all task-relevant objects should yield tau≈0, and a scene where a synthetic injected reflection is the only difference should yield tau>0. The primary metric is the paired success difference because it is the only quantity that directly estimates the counterfactual contrast the idea is about; success rate alone would be confounded by task difficulty, and representation divergence alone would be unfalsifiable without a behavioral anchor.

### Expected outcome and causal chain

**vs. RoboDojo-Mirror panel-swap comparison** — On a scene where the mirror also occludes part of the workspace, the panel-swap baseline attributes failure to reflection because the panel changes occlusion and contact surfaces along with appearance, inflating the apparent photometric effect. Our method instead holds geometry, occlusion, shadowing, and global illumination fixed and varies only the reflected content, so the paired difference on contact-free pairs isolates the reflected-content pathway. We expect the panel-swap estimate to exceed our tau on occlusion-heavy tasks, with the two converging on tasks where the mirror is a purely decorative surface.

**vs. Independent-seed mirror vs mirror-free sampling** — On a high-variance task where initial object pose dominates success, independent sampling produces a noisy between-condition difference that can flip sign across draws because task difficulty is not controlled. Our method instead pairs seeds with identical initial states, so the within-pair difference cancels the difficulty term and the residual reflects the reflected-content channel. We expect our effect estimate to be more stable across resampling and to show a consistent sign for policies where the unpaired estimate is ambiguous.

**vs. Post-hoc trajectory saliency attribution** — On a case where the policy fixates on the reflected object and reaches toward it, trajectory saliency flags the reach as the failure locus but cannot say whether perception or control was fooled, since both produce similar trajectories. Our method instead probes hidden states at the decision timestep, before any contact, and finds divergence concentrated in early visual layers for perception-fooled policies. We expect first-divergence to occur before first contact for most failing pairs, and layer-wise divergence to correlate with effect size on contact-free pairs, with the chunk-boundary control D_chunk confirming that the correlation is not an alignment artifact.

**vs. Single ON/OFF reflection toggle ablation** — On a scene where the mirror casts a shadow onto the tabletop, the ON/OFF toggle removes the shadow along with the reflected content, so its effect estimate conflates photometric and geometric components. Our method instead reports tau and tau_geom separately, so the two components are distinguishable. We expect the ON/OFF estimate to exceed tau on shadow-casting scenes and to converge with tau on scenes where the mirror casts no shadow.

**vs. Validation scenes (mirror facing away; synthetic injected reflection)** — On the mirror-facing-away scene, the reflected content is causally irrelevant to the task, so a valid intervention should yield tau≈0; on the synthetic-injection scene, the injected reflection is the only difference, so a valid intervention should yield tau>0. We expect both outcomes, confirming that the intervention isolates the intended channel rather than merely being a new condition.

### What would falsify this idea

If the paired reflected-content effect tau is statistically indistinguishable from the panel-swap effect on occlusion-heavy tasks, then the geometric confound we claim to remove is not actually driving the attribution gap, and the rendering intervention buys nothing over the existing benchmark. Likewise, if layer-wise divergence fails to correlate with effect size on contact-free pairs after chunk-boundary alignment, the decision-point probe does not measure reflected-content sensitivity and the representational half of the claim collapses. If the mirror-facing-away validation yields tau materially different from zero, the intervention does not isolate the intended channel.

## References

1. Embodied AI Fails the Mirror Test: How a Simple Mirror Breaks Robot Policies
