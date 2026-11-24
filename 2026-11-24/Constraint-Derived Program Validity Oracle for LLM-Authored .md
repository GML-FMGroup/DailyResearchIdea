# Constraint-Derived Program Validity Oracle for LLM-Authored Parametric Motion Programs

## Motivation

The Agentic Motion Framework of 3AGameFactory-Motion (E1, E5) assumes the LLM agent authors semantically valid parametric skeleton and motion programs and that the deterministic executor is correct, so any program-authoring error — an infeasible joint limit, a disconnected part chain, or a timing/angle constraint that cannot be satisfied by the target morphology — propagates silently into rigging and kinematic solving with no verification step in between. This is a structural gap shared across agentic motion authoring: the authoring surface (compact parametric programs) and the execution surface (rigging + IK) are separated by an unchecked assumption of validity, and no rule exists that closes the loop before execution. The same unverified handoff appears in plot-driven systems such as Sitcom-Crafter (E2, E6), where motion modules are fused and retargeted without a static check that the requested motion is realizable for the target rig.

## Key Insight

A parametric motion program is a finite constraint system over joint-limit, connectivity, and timing/angle/trajectory predicates, so its validity for a given asset can be decided statically by constraint propagation and interval arithmetic over the program's parameter ranges — producing a diagnostic that is causally tied to the exact program edit needed, rather than a scalar confidence that must be guessed.

## Method

**What it is.** We introduce COVE (Constraint-Over-Program Validity Oracle), a morphology-conditioned static checker that takes as input (i) a target asset's kinematic constraint bundle K = {joint-limit intervals, part-connectivity graph, per-part DOF}, (ii) the LLM-authored parametric skeleton program P_s and motion program P_m (compact parameter blocks with explicit timing, angle, repetition, and trajectory fields), and (iii) the DivDynBench-style constraint annotation schema C, and outputs a structured verdict V = {definitely_invalid | unresolved, violated_predicate_id, parameter_slice, suggested_interval_repair} that the agent consumes to revise its program before the deterministic executor runs. **Load-bearing assumption (stated explicitly):** COVE's interval propagation over forward kinematics is sound but over-approximating (dependency problem and wrapping effect), so an empty interval is a *proof* of infeasibility, but a non-empty interval does *not* certify feasibility. Therefore COVE never certifies 'valid' and never treats a suggested repair as authoritative: every non-empty-interval case and every suggested repair is routed through a cheap executor probe (a single IK solve on the projected slice) before being returned to the agent. The contribution is a sound pre-filter that prunes provably-infeasible slices and proposes candidate repairs, not a validity oracle with minimal-repair guarantees.

**How it works.**

```pseudocode
Input: asset kinematic bundle K, skeleton program P_s, motion program P_m, annotation schema C
Output: V = {definitely_invalid | unresolved, violated_predicate_id, parameter_slice, suggested_interval_repair}
tau = 1e-6          # tolerance on interval emptiness (width < tau => empty)
depth_max = 4       # propagation rounds before declaring unresolved
probe_budget = 1    # single IK solve per suggested repair

1. PARSE(P_s, P_m) -> AST with typed parameter blocks
     each block b has fields: part_id, phase, angle_range, timing_range,
     repetition_count, trajectory_waypoints
2. GROUND(AST, K) -> grounded predicates
     for each block b: bind part_id to K.connectivity graph;
     instantiate joint-limit predicate J_b: angle_range(b) subseteq K.limits(part)
     instantiate connectivity predicate N_b: path exists in K.connectivity from
        parent(part) to part with DOF >= required DOF(b)
     instantiate timing predicate T_b: timing_range(b) nonempty and ordered
        across phases (phase_i.end <= phase_{i+1}.start)
     instantiate trajectory predicate R_b: every waypoint in trajectory_waypoints(b)
        lies within reachable workspace of part given K.limits along the chain
3. PROPAGATE(grounded predicates) -> interval narrowing
     repeat up to depth_max rounds:
        for each J_b: intersect angle_range(b) with K.limits(part)
        for each R_b: propagate narrowed angle intervals through forward kinematics
           interval arithmetic to shrink reachable workspace
        for each T_b: intersect timing_range with downstream phase windows
        if any interval becomes empty (width < tau): emit violated_predicate_id
           and mark verdict = definitely_invalid (sound: empty interval proves infeasibility)
4. DIAGNOSE(violated_predicate_id, parameter_slice)
     look up the violated predicate's parameter slice in AST
     compute candidate interval repair: nearest feasible sub-interval of the
        original parameter range that satisfies the predicate (projection onto
        K.limits or onto reachable workspace)
     PROBE: run a single IK solve on the candidate repair slice
        if IK solve succeeds: return V = {definitely_invalid, violated_predicate_id,
           parameter_slice, suggested_interval_repair}
        else: return V = {unresolved, violated_predicate_id, parameter_slice,
           suggested_interval_repair = null}  # do not claim a repair
5. If V.definitely_invalid: return V to LLM agent; agent revises parameter_slice; goto step 1.
   If V.unresolved: hand AST to deterministic executor (rigging + kinematic solving).
   Loop cap: 3 repair rounds; on cap, escalate to full-program rewrite.
```

The oracle is *learned* in the sense that the predicate templates and the parameter-slice-to-repair mapping are fit from DivDynBench-style constraint annotations: a predicate-template classifier assigns each annotation sentence to one of {joint-limit, connectivity, timing, trajectory} and extracts the parameter slice it constrains; a repair-projector is trained on (violated slice, feasible slice) pairs synthesized by perturbing valid DivDynBench programs until the executor's kinematic solver rejects them, then recording the nearest feasible projection. At test time the oracle applies these learned templates and projectors statically, without invoking the executor except for the single IK probe per suggested repair. The predicate-template classifier is a 2-layer MLP (hidden=256, GeLU) over frozen sentence embeddings from a pre-trained language model (e.g., all-MiniLM-L6-v2, 384-d), trained on 2,000 annotation sentences from DivDynBench (80/10/10 train/val/test split), with expected held-out accuracy of 85–90% on the four predicate types. The repair projector is a 3-layer MLP (hidden=512, ReLU) that takes the violated slice's interval bounds and the asset's K features and outputs a projected interval; it is trained on 10,000 synthesized (violated, feasible) pairs generated by perturbing valid DivDynBench programs with Gaussian noise (σ=0.1 of range) until executor rejection, then recording the nearest feasible projection via a deterministic interval-arithmetic projection. A deterministic baseline (pure interval-arithmetic nearest-feasible projection, no training) is also implemented for ablation.

**Why this design.** We chose a *static constraint-propagation oracle over interval arithmetic* rather than a learned scalar validity classifier because the motivation-level claim is that program-authoring errors are *localized* to specific parameter slices, and a scalar score cannot name the slice; the cost is that interval propagation may report 'unresolved' when ranges are wide, which we accept because unresolved cases fall back to the executor rather than blocking. We chose *structured diagnostics with a suggested interval repair* rather than a binary valid/invalid verdict because the agent's repair loop needs a gradient-like signal pointing at the editable parameter, and a binary verdict forces blind resampling; the cost is that the repair projector must be trained on synthesized violations, which may not cover all real authoring errors. We chose a *bounded repair loop (3 rounds) with escalation to full rewrite* rather than unbounded iteration because DivDynBench instructions contain multi-phase constraints whose interactions can cycle; the cost is that some repairable programs are rewritten unnecessarily. We chose to *learn predicate templates from DivDynBench annotations* rather than hand-code a fixed grammar because the annotation schema C is open-vocabulary across six morphology categories; the cost is classifier error on out-of-schema instructions. This is not a variant of retrieval-based tool selection (Toolformer/Gorilla): those systems retrieve a matching program from a library, whereas COVE *intervenes on the program* by projecting its parameter intervals onto the asset's feasible set — a counterfactual repair that retrieval cannot produce because the repaired program does not exist in any library.

**Why it measures what we claim.** The quantity `violated_predicate_id` measures *the specific authoring error that would propagate into execution* because the grounding step binds each predicate to a concrete asset constraint in K, and the propagation step declares violation only when the interval becomes empty under that binding; this assumption fails when K itself is under-specified (e.g., a morphology category with unannotated joint limits), in which case the predicate reflects a missing constraint rather than an authoring error. The quantity `suggested_interval_repair` measures *the minimal program edit that restores validity* because the repair projector is trained on nearest-feasible projections of synthesized violations, and the projection is defined as the closest sub-interval satisfying the violated predicate; this assumption fails when the violated predicate is coupled to a downstream predicate that the projection does not re-check, in which case the repair reflects local rather than global feasibility. The quantity `repair_rounds_to_valid` measures *the closure of the authoring–executor loop* because each round is a full re-check of the revised program against K before execution; this assumption fails when the agent's revision ignores the suggested slice, in which case the count reflects agent compliance rather than oracle quality. To empirically test the agent-compliance assumption, we log per-round agent-revision-vs-suggestion agreement: the fraction of rounds where the agent's revised parameter slice overlaps the suggested interval by at least 50%. We report this agreement rate alongside repair rounds, and in the ablation we replace the agent with a deterministic revision policy that always applies the suggested slice exactly, so that round count reflects oracle precision rather than agent stochasticity.

## Contribution

["A morphology-conditioned static validity oracle (COVE) that grounds LLM-authored parametric skeleton and motion programs against an asset's kinematic constraint bundle and returns structured, slice-localized diagnostics with suggested interval repairs, inserted between the agent's authoring step and the deterministic executor.", 'A learned predicate-template and repair-projection procedure fit from DivDynBench-style constraint annotations and synthesized executor-rejected violations, enabling open-vocabulary constraint checking across six morphology categories without invoking the executor at check time.', 'A bounded authoring–execution repair loop with escalation, together with an analysis of which DivDynBench constraint types (joint-limit, connectivity, timing, trajectory) are statically decidable before execution versus which remain executor-dependent.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | DivDynBench (six morphology categories) | Matches target: multi-phase, multi-morphology constraint programs |
| Primary metric | Repair rounds to valid | Directly measures authoring–executor loop closure |
| Secondary metric | Agent-revision-vs-suggestion agreement rate | Tests whether round count reflects oracle vs agent |
| Baseline | Specialized motion generation | Tests whether static pre-check beats monolithic generator |
| Baseline | Video generation + motion recovery | Tests whether program-space repair beats pixel-space recovery |
| Baseline | Direct agent-driven generation (no oracle) | Tests whether structured diagnostics beat blind resampling |
| Baseline | Classical interval-arithmetic IK feasibility checker (no learned classifier/projector) | Tests whether learned components add value over hand-coded checker |
| Ablation-of-ours | COVE with binary verdict (no repair slice) | Isolates value of structured repair signal |
| Ablation-of-ours | COVE with deterministic nearest-feasible projection (no learned projector) | Isolates value of learned repair projector |
| Ablation-of-ours | COVE with deterministic agent revision policy (always applies suggested slice) | Isolates oracle precision from agent stochasticity |

### Why this setup validates the claim

The central claim is that authoring errors are localized to parameter slices, so a static constraint-propagation oracle that names the slice and projects it onto the feasible set will close the authoring–executor loop faster and more reliably than generators that never inspect the program's constraint structure. DivDynBench is the right dataset because its instructions carry explicit timing, angle, repetition, and trajectory constraints across six morphologies — exactly the predicate types COVE grounds, and exactly where the competing paradigms are documented to struggle on control and cross-morphology adaptability. Repair rounds to valid is the correct primary metric because it is the only quantity that directly indexes loop closure: a method could produce a valid motion by luck after many blind resamples, but few rounds to validity can only arise from a diagnostic signal that points at the editable parameter. The three baselines each probe a different sub-claim: specialized generation tests whether monolithic synthesis already internalizes constraints (if it does, our oracle adds nothing); video-then-recovery tests whether pixel-space generation sidesteps program-space errors (if it does, our program-space intervention is misdirected); direct agent-driven generation without the oracle tests whether the diagnostics themselves — not merely the agent loop — carry the benefit. The classical interval-arithmetic IK feasibility checker baseline tests whether the learned predicate-template classifier and repair projector add value beyond a hand-coded checker over the same grounded predicates. The binary-verdict ablation is the linchpin: it keeps the static checker and the agent loop but strips the parameter-slice and suggested-repair output, so any gap between it and full COVE is attributable specifically to the structured repair signal rather than to checking per se. The deterministic-projection ablation isolates the learned repair projector's marginal value, and the deterministic-agent-revision ablation tests whether round count reflects oracle precision or agent stochasticity. Together these form a falsifiable test: the claim predicts a gain concentrated where slice-localized errors occur, not a uniform lift.

### Expected outcome and causal chain

**vs. Specialized motion generation** — On a DivDynBench instruction requiring multi-phase part coordination with explicit angle and repetition constraints on a non-humanoid morphology, the specialized generator produces a motion that satisfies the broad motion type but violates a joint-limit or phase-ordering constraint, because its mechanism maps instruction to motion directly without grounding the instruction against the asset's kinematic bundle. Our method instead parses the program into typed parameter blocks, grounds each against K, and narrows intervals until the empty interval names the violated predicate, so the agent revises exactly that slice before execution. We expect a noticeable gap on the constraint-dense multi-phase subsets and near-parity on simple single-phase locomotion, where the specialized generator's priors already suffice.

**vs. Video generation + motion recovery** — On a case where the requested trajectory waypoints leave the reachable workspace of a limb given the asset's joint limits, the video-then-recovery pipeline produces a visually plausible clip whose recovered motion drifts outside the feasible set, because its mechanism optimizes pixel-space plausibility and only recovers kinematics afterward, so the infeasibility is baked in before any constraint is consulted. Our method instead checks trajectory predicates against interval-propagated reachable workspace before any execution, and the repair projector returns the nearest feasible waypoint sub-interval. We expect a clear gap on trajectory-constrained and morphology-atypical subsets, with parity on unconstrained free-form motion where recovery has nothing to violate.

**vs. Direct agent-driven generation (no oracle)** — On a case with interacting timing constraints across three phases, the direct agent produces a program whose phase windows overlap, then resamples blindly after executor rejection, cycling through near-identical invalid programs because it receives no signal about which parameter caused the failure. Our method instead returns violated_predicate_id plus a suggested interval repair, giving the agent a gradient-like direction, so we expect markedly fewer repair rounds to valid and fewer escalations to full rewrite, with the largest separation on the multi-phase timing subsets and convergence within the three-round cap on most repairable cases.

**vs. Classical interval-arithmetic IK feasibility checker** — On the same grounded predicates, the hand-coded checker can detect empty intervals but lacks the learned predicate-template classifier for open-vocabulary annotations and the learned repair projector for suggesting slices. We expect COVE to match the classical checker on definitely-invalid detection (both are sound) but to outperform on repair rounds to valid due to the structured repair signal, with the largest gap on out-of-schema instructions where the learned classifier generalizes.

**vs. COVE with binary verdict (no repair slice)** — This ablation keeps the static checker and agent loop but returns only valid/invalid. We expect full COVE to require fewer repair rounds and fewer escalations, with the gap concentrated on multi-phase timing and trajectory-constrained subsets where the slice-localized repair signal is most informative.

**vs. COVE with deterministic nearest-feasible projection (no learned projector)** — This ablation replaces the learned repair projector with pure interval-arithmetic projection. We expect the learned projector to yield slightly fewer repair rounds on average, with the largest benefit on morphologies with complex joint-limit interactions, but we also expect the deterministic baseline to be competitive, indicating that the core benefit comes from the static checker and structured diagnostics rather than the learned projector alone.

**vs. COVE with deterministic agent revision policy** — This ablation replaces the LLM agent's revision with a deterministic policy that always applies the suggested slice exactly. We expect repair rounds to be lower and less variable than with the LLM agent, and we will report the agent-revision-vs-suggestion agreement rate to quantify how much of the round count is attributable to agent compliance versus oracle precision.

### What would falsify this idea

If the gain over the no-oracle agent is uniform across subsets rather than concentrated on the constraint-dense, multi-phase, and atypical-morphology subsets where slice-localized violations are predicted to occur, then the diagnostics are not doing the causal work and the central localization claim is wrong. Likewise, if the binary-verdict ablation matches full COVE on repair rounds to valid, the structured repair signal — the idea's core mechanism — is not what produces the benefit.

## References

1. 3AGameFactory-Motion: Agentic Rigging and Reliable Complex Motion Generation for Diverse 3D Assets
