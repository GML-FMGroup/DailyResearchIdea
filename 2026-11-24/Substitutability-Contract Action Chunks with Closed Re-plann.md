# Substitutability-Contract Action Chunks with Closed Re-planning for Frontier-Model-as-Policy Harnesses

## Motivation

The Embo-Harness recipe establishes that frontier multimodal models can serve directly as policies inside a closed-loop robotic harness, and it identifies action representation, chunking, and policy refinement as effective design choices, but it treats each action chunk as an atomic primitive emitted by a frozen policy — so the recipe offers no structural rule by which a chunk that fails during execution can be replaced by a re-planned alternative while preserving the harness's observation-action consistency. This is a meta-level gap: across every frontier-model-as-policy tree (Embo-Harness's recipe, Lift3D's 3D policy, RDT-1B's diffusion foundation model, SAM2Act's memory policy), the chunk is an opaque unit whose failure carries no contract about what world state it assumed or what it guaranteed, so error recovery can only be handled by re-prompting from scratch, which discards the closed-loop state the harness had accumulated.

## Key Insight

If every action chunk is emitted together with a substitutability contract — a pre-condition over the observation at chunk start and a post-condition over the observation at chunk end, both drawn from a finite, pre-enumerated predicate vocabulary — then failure detection becomes predicate violation rather than a confidence score, and a failed chunk can be replaced by any alternative chunk in the closed action set whose pre-condition is satisfied by the current observation, so substitution is a typed rewrite in a finite algebra rather than a free-form re-prompt.

## Method

**What it is.** SCAC (Substitutability-Contract Action Chunks) is a harness wrapper that takes as input the frontier model's action-chunk emissions and produces, for each chunk, a machine-checkable substitutability contract plus a closed substitution step at failure time. Inputs: observation stream `o_t`, instruction `l`, frontier model `M`, a finite action-chunk library `A = {a_1..a_N}` with contracts, and a predicate vocabulary `P`. Outputs: a dispatched chunk, a violation verdict, and a substituted chunk with a preserved closed-loop state.

**How it works.**

```pseudocode
# Phase 0: Contract vocabulary construction (offline, once)
P = {predicate templates over object poses, gripper state, contact, region membership}
# each predicate is a finite-domain check on a structured observation parser
# parser: deterministic rule-based segmentation of RGB-D into object poses (6-DoF),
# gripper open/closed, binary contact flags, and axis-aligned region membership;
# predicates are evaluated on this parsed structure, not on raw pixels.

# Phase 1: Contracted chunk library construction (offline, per task family)
for each demonstration trajectory d in D:
    segment d into chunks by gripper-state change points
    for each chunk c:  # chunk = (action sequence, duration)
        pre(c)  = conjunction of predicates in P true at chunk start in ALL demos of c
        post(c) = conjunction of predicates in P true at chunk end   in ALL demos of c
        # predicates that are not stable across demos are dropped -> contract is sound-by-construction
A = { (c, pre(c), post(c)) }  # closed, pre-enumerated set

# Phase 2: Online harness loop
state = o_0
for step in episode:
    # 2a. Frontier model proposes a chunk ID + free-form rationale
    proposal = M(o_t, l, A.describe())        # M selects from A, not free-form actions
    c = A[proposal.chunk_id]

    # 2b. Pre-condition gate (closed-loop consistency check)
    if not satisfies(pre(c), o_t):
        c = substitute(c, o_t)                # see 2d

    # 2c. Execute and monitor post-condition
    for k in range(c.duration):
        dispatch(c.action[k]); o_t = observe()
    if satisfies(post(c), o_t):
        continue                              # chunk closed successfully

    # 2d. Substitution operator (failure-substitution closure)
    def substitute(c_failed, o_t):
        # Guard: require downstream admissibility if next chunk is known
        next_pre = pre(next_chunk) if next_chunk is not None else None
        candidates = { c' in A : satisfies(pre(c'), o_t)
                                and post(c') implies post(c_failed)
                                and (next_pre is None or post(c') implies next_pre) }
        # 'implies' is checked on the finite predicate lattice P
        if candidates empty: return replan_from_scratch(o_t)   # documented fallback
        return argmin_{c' in candidates} cost(c')              # cost = expected duration
```

**Why this design.** We chose a *finite, pre-enumerated* chunk library over free-form chunk generation because substitution closure requires that the candidate set be closed under the post-condition implication relation, and a generative policy's output set is not enumerable at check time, accepting the cost that the harness cannot express chunks outside the library. We chose *sound-by-construction* contracts (predicates kept only if stable across all demonstrations of a chunk) over learned per-chunk confidence heads because a contract that can be violated is exactly what makes failure detection a predicate check rather than a calibration problem — this is the deliberate departure from the Embo-Harness recipe, which selects action representation and chunking by empirical ablation but never attaches a checkable contract, and from SAM2Act's memory-bank policy, which stores past observations but not chunk-level pre/post-conditions. We chose the post-condition implication check `post(c') implies post(c_failed)` as the substitution criterion over a similarity match between chunks because implication on a finite predicate lattice is decidable and transitively closed, whereas similarity is not — accepting the cost that the candidate set can be empty when the predicate vocabulary is too coarse. We chose to keep the frontier model as the *selector* over `A` rather than as the *generator* of actions, because this preserves Embo-Harness's finding that native visual reasoning should drive the interface while making the selected unit contract-bearing; the cost is that the model's expressivity is bounded by `A`.

**Why this measures what we claim.** The computational quantity `satisfies(pre(c), o_t)` measures *substitutability readiness* because we assume the structured observation parser is faithful to the predicates in `P` (every predicate is a deterministic function of the parsed observation); this assumption fails when the parser mis-segments an occluded object, in which case the gate reflects parser error rather than world-state mismatch. The quantity `post(c') implies post(c_failed)` measures *closed-loop consistency preservation* because we assume the predicate lattice `P` is rich enough that any two chunks with the same post-condition leave the harness in observationally equivalent states for downstream chunk selection; this assumption fails when a downstream chunk's pre-condition depends on a predicate outside `P`, in which case the implication check certifies a substitution that later violates the next chunk's pre-condition. The quantity `candidates = {c' : pre(c') holds and post(c') implies post(c_failed)}` measures *failure-substitution closure* because we assume the library `A` was segmented at gripper-state change points so that chunk boundaries align with the predicates' evaluation points; this assumption fails when a demonstration contains a mid-chunk regrasp, in which case the closure holds only over the segmented library and not over the true action space. The `cost(c')` tie-break measures *recovery efficiency* because we assume duration is a monotone proxy for recovery risk; this fails when a short chunk is high-variance, in which case the substitution prefers a fast but unreliable alternative.

**Load-bearing assumption (explicit).** The substitution operator's soundness depends on the predicate vocabulary `P` being rich enough that `post(c') implies post(c_failed)` on the lattice certifies observational equivalence for every downstream chunk's pre-condition. This is a closed-world assumption: a finite hand-enumerated `P` cannot anticipate all frame conditions that later chunks depend on. To make the assumption testable and to guard against its failure, the substitution operator in Phase 2d is guarded: it requires that the substituted chunk's post-condition also implies the pre-condition of the currently pending next chunk (when known), and any predicate referenced by a downstream pre-condition but absent from `P` is treated as a hard block that triggers the documented `replan_from_scratch` fallback rather than a certified substitution. The assumption is measured directly by the downstream pre-condition violation rate after substitution (see experiment).

## Contribution

(1) A substitutability-contract action-chunk interface in which every chunk in a finite library carries a sound-by-construction pre/post-condition pair over a finite predicate vocabulary, turning failure detection into a decidable predicate check rather than a confidence threshold. (2) A failure-substitution closure operator — a candidate-set construction defined by pre-condition satisfaction plus post-condition implication on a finite predicate lattice — that replaces a failed chunk with a contract-compatible alternative while preserving the harness's observation-action state. (3) A contracted chunk library construction procedure that segments demonstrations at gripper-state change points and retains only predicates stable across all demonstrations of a chunk, yielding contracts that are violated only by genuine world-state mismatch.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | RoboLab-120 plus 20 real-world zero-shot trials | Matches Embo-Harness protocol; includes unseen-task generalization |
| Primary metric | Success rate stratified by contract-violation subset | Isolates recovery where predicted mechanism acts |
| Secondary metric | Downstream pre-condition violation rate after substitution | Directly tests the load-bearing assumption that implication on `P` certifies downstream admissibility |
| Baseline | Embo-Harness recipe (frontier-model-as-policy) | Tests whether contracts add value over free-form harness |
| Baseline | SAM2Act+ memory-bank policy | Tests whether chunk contracts beat observation memory alone |
| Baseline | Lift3D 3D-lifted policy | Tests whether substitution closure beats 3D representation gains |
| Ablation-of-ours | SCAC without post-condition implication (similarity match) | Tests whether implication, not substitution per se, drives gains |
| Ablation-of-ours | SCAC without downstream-admissibility guard | Tests whether the guard is necessary to preserve closed-loop consistency |
| Ablation-of-ours | SCAC with parser-error injection (Gaussian noise on object poses, σ ∈ {0.01, 0.05, 0.1} m) | Quantifies how much of the gain survives under controlled parser noise |

### Why this setup validates the claim

The central claim is that attaching machine-checkable substitutability contracts to a finite chunk library converts frontier-model failures into recoverable, closed-loop substitutions rather than episode-ending errors. RoboLab-120 and the real-world trials are the right testbed because they contain the multi-step manipulation tasks where chunk-level failures accumulate, and because Embo-Harness already establishes a strong frontier-model baseline there, so any gain cannot be attributed to model quality. Stratifying success rate by contract-violation subset is the key design move: the mechanism predicts a gain concentrated exactly on episodes where a pre- or post-condition is violated, not a uniform uplift. The downstream pre-condition violation rate after substitution is the direct test of the load-bearing assumption: if implication on `P` certifies downstream admissibility, this rate should be near zero; if it is high, the guard is doing the work and the assumption is false. Embo-Harness tests the sub-claim that contracts add value beyond native visual reasoning; SAM2Act+ tests whether chunk-level pre/post-conditions outperform stored-observation memory; Lift3D tests whether substitution closure beats representation-level robustness. The similarity-match ablation isolates implication-on-the-lattice as the causal ingredient, since a similarity-matched substitute is otherwise identical. The no-guard ablation isolates the downstream-admissibility guard as the mechanism that preserves closed-loop consistency. The parser-error injection quantifies the parser-faithfulness assumption: if the gain collapses under small pose noise, the gate reflects parser error rather than world-state mismatch.

### Expected outcome and causal chain

**vs. Embo-Harness recipe** — On a case where the frontier model emits a chunk whose pre-condition is silently false (e.g., the target object slipped out of the gripper region after a prior chunk), the baseline produces a confident but mis-grounded action sequence because its recipe selects action representation and chunking empirically but attaches no checkable contract, so the mismatch is only discovered at final task failure. Our method instead gates on `satisfies(pre(c), o_t)` before dispatch and substitutes a chunk whose pre-condition holds and whose post-condition implies the failed chunk's, so we expect a noticeable success gap concentrated on the contract-violation subset with near-parity on episodes where no predicate is violated.

**vs. SAM2Act+** — On a case where an early chunk leaves the scene in a state the memory bank has seen but the next chunk's pre-condition does not hold (e.g., a regrasp left the gripper occluded), the baseline produces actions conditioned on remembered observations but without a chunk-level state contract, so it cannot detect that the selected unit is inadmissible and fails downstream. Our method instead checks the pre-condition against the parsed observation and, on post-condition failure, substitutes via lattice implication, so we expect a gap on memory-dependent and regrasp-heavy tasks but parity on single-chunk tasks where memory alone suffices.

**vs. Lift3D policy** — On a case where 3D geometry is well perceived but the chosen chunk is still the wrong unit for the current state (e.g., the object is in the right region but the gripper approach is blocked), the baseline produces geometrically well-grounded but semantically mis-sequenced actions because stronger 3D representations do not by themselves certify chunk admissibility. Our method instead treats the chunk as a contract-bearing unit and closes the failure by substitution, so we expect a modest gap on spatially simple tasks and a larger gap on tasks requiring re-sequencing after a mid-episode perturbation.

### What would falsify this idea

If the success-rate gain is uniform across contract-violation and no-violation subsets rather than concentrated on episodes where a pre- or post-condition actually fails, the mechanism is not doing the work and the claim is wrong. Likewise, if the similarity-match ablation matches implication-based substitution, then implication on the predicate lattice is not the causal ingredient; and if the downstream pre-condition violation rate after substitution is high even with the guard, the load-bearing assumption is false and the substitution operator does not preserve closed-loop consistency.

## References

1. Embo-Harness: Towards Understanding Frontier-Model-as-Policy Harness
