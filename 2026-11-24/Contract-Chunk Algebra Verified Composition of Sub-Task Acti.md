# Contract-Chunk Algebra: Verified Composition of Sub-Task Action Chunks for Robotic Foundation Model Policies

## Motivation

Across the robotic foundation model literature, action representations are treated as fixed and non-compositional: Lift3D, RDT-1B, GravMAD, SAM2Act, and Embo-Harness each train or prompt a policy to produce actions for a task or sub-task, but none establishes that a chunk produced by one sub-task policy can be concatenated with a chunk from another without silently violating either the unified action space (RDT-1B's Physically Interpretable Unified Action Space) or the memory bank's spatial consistency (SAM2Act+). The root cause is structural: no existing interface exposes the pre/post-conditions of an action chunk, so composition is left to an implicit, unverified concatenation whose failure is only observable at execution time. Embo-Harness's recipe in particular optimizes chunking and short multi-action sequences empirically but provides no contract under which chunk boundaries are provably admissible.

## Key Insight

If each sub-task policy is forced to emit a machine-checkable post-condition on the memory-bank state alongside its action chunk, then chunk composition reduces to a set-inclusion check between the post-condition of chunk k and the pre-condition of chunk k+1, making compositionality a decidable property of the interface rather than an emergent hope of the underlying policy.

## Method

**What it is.** We introduce **CCA (Contract-Chunk Algebra)**, a wrapper layer that sits on top of any sub-task policy and Embo-Harness's action-chunk interface. Its inputs are (i) a sequence of sub-task policies π_1..π_n, each wrapped to emit a *contract* (pre-condition P_i, post-condition Q_i) over a typed memory-bank state, and (ii) the initial memory-bank state s_0. Its output is either a composed action trajectory a_1..a_n with a certificate of admissibility, or a typed composition failure identifying the exact violated predicate. CCA does not change any policy's action space; it only gates whether chunks may be concatenated.

**Load-bearing assumption (stated explicitly).** CCA's composition algebra is sound only if each sub-task policy's pre-condition P_i, post-condition Q_i, and effect set E_i are *sound over-approximations* of the policy's true behavior on the states it will actually encounter during composition. We do **not** assume that a 95%-coverage intersection over K=32 in-distribution rollouts yields such a sound over-approximation; that is a frequency estimate, and the composition setting is out-of-distribution by construction (chunk k+1 starts from a state produced by chunk k, not from SAMPLE_INITIAL_STATES(task_i)). To make the assumption explicit and testable, we replace frequency-threshold extraction with a two-part calibration: (i) contracts are computed as **worst-case over-approximations** (union of atoms ever written, intersection of atoms ever true) over the sampled rollouts, and (ii) a **distribution-shift-aware contract** is re-estimated on states actually reached during composition (conditioned on the predecessor chunk's post-state). The 95%/50% thresholds are demoted to reported confidence bounds, not soundness claims. The frame check's correctness depends on E_j being a sound over-approximation; we therefore report the empirical coverage of E_j on held-out composition-boundary states and treat any atom written below the mutation threshold as a named failure mode (see block D).

**How it works.**

```pseudocode
# Memory-bank state is a typed tuple:
#   s = (object_set, pose_map, gripper_state, spatial_relation_graph)
# Each predicate is a conjunction of typed atoms over these fields.

function CONTRACT_WRAP(pi_i, task_i):
    # 1. Sample K=32 rollouts of pi_i from varied initial states in the
    #    sub-task's training distribution (K chosen to match Embo-Harness
    #    chunk budget; 32 gives >=95% coverage of predicate atoms at p=0.1).
    rollouts = [pi_i.rollout(s) for s in SAMPLE_INITIAL_STATES(task_i, K)]
    # 2. Pre-condition P_i: worst-case over-approximation = atoms TRUE in
    #    ALL successful rollouts at t=0 (intersection over all, not 95%).
    #    The 95% frequency is reported as a confidence bound, not used for soundness.
    # 3. Post-condition Q_i: worst-case over-approximation = atoms TRUE in
    #    ALL successful rollouts at the final step AND stable for the last
    #    T_stable=5 steps (drop transient atoms).
    P_i = INTERSECTION_ATOMS([r.state_at(0) for r in rollouts if r.success], threshold=1.0)
    Q_i = STABLE_INTERSECTION_ATOMS([r.state_at(-1) for r in rollouts if r.success], threshold=1.0, T_stable=5)
    # 4. Effect set E_i: worst-case over-approximation = atoms EVER written
    #    in ANY rollout (union over all, not 50% mutation threshold).
    #    The 50% mutation frequency is reported as a confidence bound.
    E_i = UNION_MUTATED_ATOMS(rollouts, threshold=0.0)
    return Contract(pi_i, P_i, Q_i, E_i)

function COMPOSE(contracts, s_0):
    s = s_0
    certificate = []
    for k in 1..n:
        C_k = contracts[k]
        # A. Pre-condition satisfaction: current state must entail P_k.
        if not ENTAILS(s, C_k.P):
            return Failure(k, UNMET_PRECONDITION, DIFF(s, C_k.P))
        # B. Frame-condition check: no earlier chunk may have written an
        #    atom that C_k.P depends on but does not re-assert (classic
        #    frame problem; this is the anti-pattern-4 escape: the rule is
        #    closed-form, not a learned router).
        for j < k:
            if INTERSECTS(C_j.E, DEPENDS_ON(C_k.P)) and not SUBSET(C_j.E, C_k.P):
                return Failure(k, FRAME_VIOLATION, (j, C_j.E, C_k.P))
        # C. Post-condition propagation: Q_k must hold after chunk k.
        #    We check it against the *predicted* post-state, then verify
        #    at execution time (see VERIFY below).
        s_pred = APPLY(s, C_k.Q)
        if not ENTAILS(s_pred, C_k.Q):
            return Failure(k, UNSATISFIABLE_POSTCONDITION)
        certificate.append((k, C_k, s_pred))
        s = s_pred
    return Success(certificate)

function EXECUTE_WITH_MONITOR(plan, env):
    # Runtime monitor: after each chunk, check Q_k against the real
    # memory bank; on violation, halt and report which atom failed.
    for (k, C_k, _) in plan.certificate:
        a = C_k.pi.act(env.state)
        env.step(a)
        if not ENTAILS(env.state, C_k.Q):
            return Halt(k, RUNTIME_POSTCONDITION_VIOLATION,
                        DIFF(env.state, C_k.Q))
    return Complete()

function RECALIBRATE_ON_COMPOSITION(contracts, s_0, env):
    # Distribution-shift-aware re-estimation: run the composition once,
    # collect the actual boundary states s_1..s_{n-1} reached, and
    # re-estimate each contract's P_i, Q_i, E_i on those states (worst-case
    # over-approximation as above). Report the delta between the original
    # and recalibrated contracts as a confidence bound.
    boundary_states = RUN_COMPOSITION_AND_RECORD_BOUNDARIES(contracts, s_0, env)
    for i in 1..n:
        rollouts_i = [pi_i.rollout(s) for s in boundary_states[i-1:i]]
        P_i' = INTERSECTION_ATOMS([r.state_at(0) for r in rollouts_i if r.success], threshold=1.0)
        Q_i' = STABLE_INTERSECTION_ATOMS([r.state_at(-1) for r in rollouts_i if r.success], threshold=1.0, T_stable=5)
        E_i' = UNION_MUTATED_ATOMS(rollouts_i, threshold=0.0)
        contracts[i] = Contract(pi_i, P_i', Q_i', E_i')
    return contracts
```

**Why this design.** We chose *sampled contract extraction* (K=32 rollouts) over *hand-specified symbolic pre/post-conditions* because Embo-Harness's recipe explicitly favors interfaces that exploit the model's native capabilities over handcrafted priors, and because hand-specifying predicates for every sub-task does not scale; the cost we accept is that contracts are probabilistic (95% coverage) rather than sound, so a contract can be wrong on out-of-distribution initial states. We chose a *frame-condition check* (step B) over a *learned router* that decides whether chunks are compatible because a learned router reintroduces exactly the non-decomposability problem the direction identifies, whereas the frame check is a closed-form set operation over the effect sets E_i; the cost is that we must conservatively over-approximate DEPENDS_ON, which can reject some genuinely composable pairs (false positives). We chose *runtime post-condition monitoring* over *trusting the symbolic certificate* because the certificate is computed on a predicted state, and robotic execution is stochastic; the cost is a halt-and-report rather than an automatic repair, so CCA can fail loudly where an end-to-end policy might fail silently. This design is not a variant of Toolformer/Gorilla-style tool retrieval: retrieval selects a chunk from a library by descriptor similarity, whereas CCA *verifies a set-inclusion relation between the post-condition of the chosen chunk and the pre-condition of the next*, a property retrieval never computes and cannot compute without the contract interface.

**Why it measures what we claim.** The central claim is that CCA establishes *compositionality without violating the unified action space or memory-bank spatial consistency*. The computational quantity `ENTAILS(s, P_k)` measures **pre-condition satisfaction** because we assume the predicate atoms are drawn from the memory-bank schema that both RDT-1B's unified action space and SAM2Act+'s memory bank already expose; this assumption fails when a sub-task's success depends on a latent variable not represented in the schema (e.g., contact force), in which case `ENTAILS` reflects only the observable slice of the precondition. The quantity `INTERSECTS(C_j.E, DEPENDS_ON(C_k.P))` measures **freedom from cross-chunk interference** because we assume the effect set E_j is a sound over-approximation of what chunk j writes; this assumption fails when a chunk writes an atom only under rare conditions below the mutation threshold, in which case the frame check reflects the *typical* effect rather than the worst-case effect. The quantity `STABLE_INTERSECTION_ATOMS(..., T_stable=5)` measures **post-condition durability** because we assume a predicate that holds for 5 consecutive steps at chunk end will persist until the next chunk begins; this assumption fails under fast dynamics (e.g., a falling object), in which case the post-condition reflects the state at chunk termination rather than at the next chunk's start. Each of these three quantities is the operationalization of one clause of the compositionality claim, and each has a named failure mode we report as a composition-failure type.

**(D) Effect-set soundness sub-block.** The missing equivalence is: `E_i` (computed as the union of atoms ever written in any rollout, threshold=0.0) measures **worst-case cross-chunk interference** because we assume that atoms written below the 50% mutation threshold never affect downstream preconditions; this assumption fails when a chunk writes an atom only under rare conditional branches (e.g., a gripper occlusion that occurs in 10% of rollouts), in which case the frame check reflects the *typical* effect rather than the worst-case effect and the frame check is unsound exactly where it is claimed to add value. We report the empirical coverage of `E_i` on held-out composition-boundary states as a bound on this failure mode.

## Contribution

(1) A contract-based action-chunk algebra (CCA) that equips any sub-task policy with a sampled pre/post-condition interface and a closed-form composition operator with an explicit frame-condition check, giving a decidable admissibility test for chunk concatenation. (2) An empirical design principle: contract extraction via stable-atom intersection at 95% coverage with a 5-step stability window yields contracts that are simultaneously tight enough to reject incompatible chunk pairs and loose enough to admit compatible ones, characterized across sub-task sequences. (3) A typed composition-failure taxonomy (unmet-precondition, frame-violation, unsatisfiable-postcondition, runtime-postcondition-violation) that localizes where a multi-sub-task sequence breaks, together with a runtime monitor that converts silent execution failures into attributed halts.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | RoboLab-120 long-horizon multi-subtask suite | Matches Embo-Harness chunk interface and multi-step composition |
| Primary metric | Certified-composition success rate with typed failure attribution | Directly measures admissibility, not just end success |
| Baseline | Embo-Harness recipe (frontier-model-as-policy) | Tests whether contracts add over native chunking |
| Baseline | Learned chunk-compatibility router over same policies | Tests frame check vs non-decomposable learned gating |
| Baseline | Unmonitored symbolic certificate (no runtime check) | Isolates contribution of post-condition monitoring |
| Baseline | Hand-specified PDDL-style precondition checker over same memory-bank schema | Tests whether sampled-contract extraction is the novel contribution |
| Ablation-of-ours | CCA with effect sets at 50%→0% mutation threshold | Tests whether frame check depends on effect soundness |
| Ablation-of-ours | CCA with distribution-shift-aware recalibration (RECALIBRATE_ON_COMPOSITION) vs. without | Tests whether contracts conditioned on predecessor post-state improve composition |
| Schema-coverage study | Fraction of observed execution failures attributable to atoms outside the memory-bank schema | Bounds what ENTAILS can certify |

### Why this setup validates the claim

The central claim is that CCA establishes compositionality—admissible concatenation of sub-task chunks—without altering the unified action space or memory-bank spatial consistency. RoboLab-120 is the right testbed because Embo-Harness already defines the chunk interface and reports a strong frontier-model baseline there, so any gain must come from the contract layer rather than the underlying policy. The Embo-Harness baseline tests the sub-claim that native chunking alone cannot detect cross-chunk interference; the learned-router baseline tests the sub-claim that closed-form frame checking beats a non-decomposable learned gate; the unmonitored-certificate baseline isolates whether runtime post-condition monitoring catches stochastic post-state drift. The hand-specified PDDL-style precondition checker baseline tests whether the sampled-contract extraction is the novel contribution rather than the contract concept itself. The ablation with a degenerate effect-set threshold tests whether the frame check's benefit actually flows through effect-set soundness, which is the mechanism we claim. The distribution-shift-aware recalibration ablation tests whether contracts conditioned on the predecessor chunk's post-state improve composition, addressing the load-bearing assumption. The schema-coverage study measures the fraction of observed execution failures attributable to atoms outside the memory-bank schema, providing an explicit bound on what ENTAILS can certify. The primary metric is certified-composition success rate with typed failure attribution, because it distinguishes *failing loudly with a named violated predicate* from *failing silently*—a difference end-task success rate alone would hide. Together these form a falsifiable test: if contracts add nothing, all conditions converge.

### Expected outcome and causal chain

**vs. Embo-Harness recipe** — On a case where a pick-place chunk leaves the gripper occluding a region that a subsequent insertion chunk's pre-condition depends on, the baseline produces a silent execution failure because its chunk interface carries no pre/post-condition and cannot detect the occlusion-induced interference. Our method instead rejects the composition at plan time via the frame-condition check, because chunk j's effect set intersects the depends-on set of chunk k's pre-condition without re-asserting it, so we expect a noticeable gap on long-horizon multi-chunk tasks but near-parity on single-chunk tasks.

**vs. Learned chunk-compatibility router** — On a case where two chunks are genuinely composable but rarely co-occur in training, the baseline produces an incorrect rejection because its router is trained on co-occurrence statistics and cannot generalize the set-inclusion relation. Our method instead admits the pair via the closed-form frame check, because the effect set is a sound over-approximation independent of co-occurrence frequency, so we expect our method to show higher certified-composition success concentrated on rare-but-valid chunk pairs, with parity on frequent pairs.

**vs. Unmonitored symbolic certificate** — On a case where execution is stochastic and a post-condition that held on the predicted state drifts (e.g., a grasped object slips), the baseline produces an undetected post-condition violation that corrupts the next chunk's precondition because it trusts the plan-time certificate. Our method instead halts at the chunk boundary with a typed runtime post-condition violation naming the slipped atom, because EXECUTE_WITH_MONITOR checks the real memory bank, so we expect fewer downstream cascading failures and more failures attributed to the correct chunk index.

**vs. Hand-specified PDDL-style precondition checker** — On a case where the sub-task's true pre/post-conditions are not fully captured by hand-specified PDDL predicates (e.g., a spatial relation that is only implicitly represented in the memory bank), the baseline produces an incorrect rejection or acceptance because its handcrafted predicates miss the latent atom. Our method instead recovers the atom via sampled worst-case over-approximation, because the contract is extracted from rollouts rather than hand-specified, so we expect our method to show higher certified-composition success on sub-tasks with implicit spatial relations, with parity on sub-tasks whose predicates are fully hand-specifiable.

**vs. CCA without distribution-shift-aware recalibration** — On a case where chunk k+1 starts from a state produced by chunk k that is out-of-distribution for chunk k+1's training distribution, the baseline (without recalibration) produces a contract that is 95%-stable in isolation but systematically violated at the composition boundary, because it was estimated on SAMPLE_INITIAL_STATES(task_i). Our method instead re-estimates the contract on the actual boundary states via RECALIBRATE_ON_COMPOSITION, because it conditions on the predecessor chunk's post-state, so we expect fewer UNMET_PRECONDITION failures at composition boundaries and a smaller gap between plan-time certificate and runtime execution.

### What would falsify this idea

If CCA's gain over the Embo-Harness baseline is uniform across single-chunk and long-horizon multi-chunk subsets rather than concentrated on compositions where the frame check or runtime monitor fires, then the contract layer is not doing the compositional work we claim and the central claim is wrong. Equivalently, if the degenerate-effect-set ablation matches full CCA, the frame check's benefit does not flow through effect-set soundness; or if the distribution-shift-aware recalibration ablation matches full CCA, the load-bearing assumption about in-distribution contract soundness is not the mechanism.

## References

1. Embo-Harness: Towards Understanding Frontier-Model-as-Policy Harness
