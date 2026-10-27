# Self-Expanding Failure Constraints for Multi-Agent Orchestration via Counterfactual Trace Perturbation

## Motivation

Existing failure taxonomies for multi-agent systems are closed-world: MAST derives 14 failure modes from 150 human-annotated traces, and DynaFlow compiles that fixed enumeration into executable constraints, so the constraint space is a proper subset of possible failures. When novel sub-agent compositions introduce coordination failures absent from the original trace corpus, both the taxonomy and DynaFlow's constraint compiler have no predicate to express them, and no mechanism exists to admit one — the representation cannot grow. This is a structural closure failure, not a coverage gap that more annotation would fix: the admission criterion itself is missing, so unseen failure modes are undetectable by construction.

## Key Insight

A candidate failure predicate is admissible only if it is causally necessary for the failure — perturbing the trace to remove the predicate's violation must restore successful execution while preserving all other trace properties — so admission is grounded in an intervention on execution traces rather than in agreement with an LLM's judgment or similarity to known failures.

## Method

## FARA: Failure-Admission by Reversible Ablation

**(A) What it is.** FARA takes as input a set of novel sub-agent compositions with execution traces (successes and failures), the existing MAST-derived constraint set, and a held-out set of distribution-shifted compositions. It outputs an expanded constraint set: each admitted predicate is an executable Python constraint over trace events, admitted only if counterfactual perturbation shows it is causally necessary for failure and it improves detection on held-out shift without regressing on the original distribution.

**(B) How it works.**

```pseudocode
INPUT: novel_traces T (labeled success/fail), base_constraints C0 (from DynaFlow/MAST),
       heldout_shift_traces H, original_dist_traces O
HYPER: n_candidates=4, n_perturb=3, tau_necessity=0.8, tau_delta=0.05,
       tau_locality=1.0  # fraction of non-descendant events whose observable outputs must be unchanged

# Phase 0: typed event schema and dependency declaration (enables locality check)
for each trace t in T:
    events(t) = list of typed events: {event_id, agent_role, event_type, fields: dict,
                 read_set: set of event_ids whose fields this event consumes,
                 write_set: set of field names this event produces}
    # read_set/write_set are extracted from the orchestration harness's message-passing log
    # (each sub-agent call records which prior event fields it reads and which fields it writes)

# Phase 1: LLM constraint synthesis (open-world proposal)
for each failed trace t in T:
    contexts = extract_composition_context(t)   # sub-agent roles, message edges, verify steps
    cands = LLM.propose(contexts, C0, n=n_candidates)
        # each cand is executable predicate: trace -> {violated, witness_event_ids, target_field}
        # target_field must be a (event_id, field_name) pair present in events(t)

# Phase 2: typed counterfactual intervention with locality check
for each candidate p in cands:
    for each failed trace t where p(t).violated:
        witness = p(t).witness_event_ids
        target = p(t).target_field  # (event_id, field_name)
        # 2a: substitute a value at the target field that satisfies p
        t_pert = substitute_field(t, target, value_satisfying(p))
        # 2b: re-execute downstream agents from the checkpoint at target.event_id
        #      using the fixed tool-calling orchestration with mocked sub-agents
        reexecute_from_checkpoint(t_pert, start_event=target.event_id) -> success or fail
        # 2c: locality check: for every event e not in descendants(target.event_id),
        #      compare observable outputs (message payloads, tool return values) pre/post;
        #      locality_pass = (fraction unchanged == tau_locality)
        if not locality_pass:
            mark p as dropped_no_locality; break
        # 2d: record outcome
        outcomes[k] = success or fail
    necessity[p] = fraction of perturbed traces with locality_pass that now succeed
    # also compute necessity_raw[p] = fraction over all perturbed traces (no locality filter)

# Phase 3: held-out admission gate
for each candidate p with necessity[p] >= tau_necessity:
    delta_shift = F1(H, C0 + p) - F1(H, C0)      # detection on shifted distribution
    delta_orig  = F1(O, C0 + p) - F1(O, C0)      # regression on original distribution
    if delta_shift >= tau_delta and delta_orig >= 0:
        C0 = C0 + p                               # admit
    else:
        reject p                                  # archive as rejected with reason
RETURN C0  # expanded, distribution-aware constraint set
```

**(C) Why this design.** We chose *counterfactual trace re-execution* over LLM-judge agreement as the necessity signal because an LLM judge inherits the same closed-world bias that produced MAST's fixed enumeration, accepting the cost that re-execution requires a replayable environment and is more expensive per candidate. We chose *minimal witness-event rewriting* over full trace regeneration because regeneration changes the composition graph and confounds the causal attribution, accepting the cost that some predicates have no minimally editable witness and are dropped. We chose a *held-out shift gate with an explicit non-regression floor* over pure precision improvement because a predicate that fires on everything would inflate shift detection while destroying original-distribution precision, accepting the cost that stricter admission yields a slower-growing constraint set. This is not a retrieval-augmented constraint library (which would retrieve nearest MAST mode) nor an LLM-as-judge extension of MAST: retrieval cannot perform the intervention, and a judge cannot certify causal necessity, so a domain expert would not describe FARA as a variant of either.

**(D) Why it measures what we claim.** `necessity[p]` measures *causal necessity* of the predicate's violation for failure because the assumption is that minimal witness-event rewriting is a valid intervention — the rewritten trace differs from the original only in the events the predicate names, so any success/failure flip is attributable to that violation; this assumption fails when the trace editor's rewrite is not behaviorally equivalent outside the witness (e.g., an LLM rewrite silently changes a downstream message), in which case necessity reflects rewrite artifacts rather than causal structure. To make this assumption explicit and testable, we require that each candidate predicate name a typed event field with a declared read/write dependency set, perform the intervention by substituting a value at that field and re-running the actual downstream agents from a checkpoint, and verify locality by checking that no non-descendant event's observable outputs change (a do-calculus-style descendant check with threshold `tau_locality=1.0`). Candidates whose witness cannot be expressed as a typed field substitution with a verified descendant set are dropped, and necessity is computed only over interventions that pass the locality check; we also report `necessity_raw` (without the locality filter) to quantify the effect of rewrite noise. `delta_shift` measures *distribution-aware detectability* because the assumption is that held-out shifted compositions are representative draws from the novel-failure distribution; this fails when the shift set is too small or correlated with the synthesis traces, in which case delta_shift reflects overfitting to the held-out set. `delta_orig >= 0` measures *non-regression on known failures* because the assumption is that F1 on the original corpus is a stable proxy for existing constraint quality; this fails when original-corpus labels are noisy, in which case the floor admits a predicate that merely fits label noise.

## Contribution

(1) FARA, a self-expanding failure-constraint mechanism that admits new executable failure predicates via counterfactual trace perturbation and a held-out distribution-shift gate, extending DynaFlow's fixed MAST compiler to an open-world constraint space. (2) The design principle that causal necessity of a predicate, established by reversible witness ablation, is a viable admission criterion for failure taxonomies where LLM-judge agreement and taxonomy similarity both inherit closed-world bias. (3) A rejected-candidate archive with perturbation witnesses, providing an auditable record of which failure modes were proposed, tested, and refused, and why.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | Novel MAS traces with held-out shifted compositions (MAST-annotated), generated from a single deterministic, fully replayable MAS harness: a fixed tool-calling orchestration with mocked sub-agents (each sub-agent is a deterministic Python function with a fixed message-passing log; no live LLM calls during re-execution) | Provides both synthesis pool and distribution-shift test set, and enables exact checkpoint re-execution for the typed intervention |
| Primary metric | F1 on held-out shifted failures | Directly measures distribution-aware detection gain |
| Baseline | Static MAST constraint set (C0 only) | Tests whether expansion adds anything beyond fixed enumeration |
| Baseline | LLM-as-Judge constraint admission | Tests whether causal necessity beats judge agreement |
| Baseline | Retrieval-augmented MAST library | Tests whether nearest-mode retrieval substitutes for intervention |
| Baseline | Invariant causal prediction (ICP) over trace features | Tests whether trace-level intervention yields predicates that feature-based causal discovery cannot express |
| Ablation-of-ours | FARA minus counterfactual perturbation (synthesis + shift gate only) | Isolates the causal-necessity phase's contribution |
| Ablation-of-ours | FARA with `necessity_raw` (no locality filter) | Quantifies the effect of rewrite noise on the necessity signal |
| Ablation-of-ours | FARA with `tau_locality=0.9` (relaxed locality) | Sensitivity analysis of necessity under rewrite noise |

### Why this setup validates the claim

The central claim is that admitted predicates are causally necessary for novel-composition failures and generalize to distribution shift without regressing on known failures. The static MAST baseline tests the sub-claim that fixed enumeration misses novel failure modes: if FARA's expanded set does not beat C0 on shifted compositions, expansion is pointless. The LLM-as-Judge baseline tests the sub-claim that causal necessity, not judge agreement, is the right admission signal: judge agreement inherits the closed-world bias that produced MAST, so it should admit plausible-but-non-causal predicates that fire on shifted traces without precision. The retrieval baseline tests the sub-claim that intervention is irreducible: retrieval cannot rewrite witness events, so it cannot certify necessity and should admit nearest-mode predicates that overfit. The ICP baseline tests the sub-claim that trace-level intervention yields predicates that feature-based causal discovery cannot express: ICP operates on a fixed feature vector per trace and cannot name a typed event field or perform a checkpoint re-execution, so it should miss predicates whose causal variable is a specific message payload or verification step. The ablation isolates Phase 2: without counterfactual perturbation, synthesis plus the shift gate should admit more predicates, inflating shift recall but regressing original-distribution precision, since the gate alone cannot filter non-causal candidates. The locality-filter ablation (`necessity_raw`) tests the load-bearing assumption directly: if necessity computed without the locality check diverges from necessity with it, rewrite artifacts are contaminating the causal signal. F1 on held-out shift is the right metric because it jointly penalizes missed novel failures and false positives that a permissive gate would introduce; a precision-only or recall-only metric would let a predicate that fires on everything pass. The non-regression floor on the original corpus is a guardrail, not the primary signal, so we expect the ablation to show the predicted precision/recall tradeoff most sharply. A power analysis with n_candidates=4, n_perturb=3 on the deterministic harness shows that a shift-F1 gap of 0.10 is detectable at 80% power with 200 held-out shifted traces (assuming per-trace F1 variance ≤ 0.25, two-sided α=0.05).

### Expected outcome and causal chain

**vs. Static MAST constraint set (C0 only)** — On a shifted composition where a sub-agent silently drops a verification step before handoff, the baseline produces a missed detection because its fixed enumeration has no predicate naming verification-before-handoff as a failure precondition. Our method instead synthesizes a candidate from the failed trace, substitutes a value at the typed witness field (the verification-step flag) and re-executes downstream agents from the checkpoint, observes success, verifies locality (no non-descendant event outputs change), and admits the predicate because necessity clears the threshold and shift F1 improves. We expect a noticeable F1 gap concentrated on shifted subsets whose failure modes are absent from C0, with near-parity on subsets already covered by MAST modes.

**vs. LLM-as-Judge constraint admission** — On a case where a plausible-sounding predicate ("agent expresses low confidence") co-occurs with failure but is not causally necessary, the baseline produces a false-positive admission because judge agreement rewards surface plausibility and inherits the closed-world bias that generated MAST. Our method instead attempts the typed field substitution, finds that satisfying the predicate does not flip the trace to success, records low necessity, and rejects it. We expect the judge baseline to show higher shift recall but lower shift precision, so F1 lags, and the gap should widen as the shift set grows more distant from MAST's enumeration.

**vs. Retrieval-augmented MAST library** — On a case where a novel composition resembles a known mode structurally but fails for a different reason, the baseline produces a misattributed detection because retrieval returns the nearest MAST mode and cannot intervene to test whether the retrieved predicate's violation actually causes the failure. Our method instead runs the typed perturbation and either confirms the retrieved-style predicate is necessary or synthesizes a new one, admitting only the causally certified version. We expect the retrieval baseline to match FARA on compositions close to known modes but to fall behind sharply on structurally novel compositions, producing a characteristic gap that grows with distance from the MAST library.

**vs. Invariant causal prediction (ICP) over trace features** — On a case where the causal variable is a specific message payload (e.g., a handoff message missing a required field) rather than a trace-level aggregate feature, the baseline produces a missed detection because ICP's feature vector cannot represent the typed event field and its intervention is a feature-level do-operation that does not correspond to a checkpoint re-execution. Our method instead names the typed field, substitutes a value, and re-executes, admitting the predicate only if the intervention flips the outcome. We expect ICP to match FARA on failures whose cause is a trace-level aggregate (e.g., total number of turns) but to lag on failures whose cause is a specific event field, with the gap concentrated on the latter subset.

**vs. FARA minus counterfactual perturbation** — On a case where a synthesized predicate fires on many shifted traces but is a correlate rather than a cause, the ablation produces an over-admitted set because the shift gate alone rewards detection gain and cannot distinguish correlation from necessity. Our method instead filters it at Phase 2, keeping the set smaller but more precise. We expect the ablation to show higher shift recall but a visible original-distribution precision drop, so its F1 either lags or ties while its constraint set is larger — the diagnostic signature that the necessity phase, not the gate, is doing the causal work.

**vs. FARA with `necessity_raw` (no locality filter)** — On a case where the typed field substitution passes the locality check, both variants agree; on a case where the substitution leaks (e.g., a downstream agent's output changes even though it is not a descendant of the target event), the raw variant counts the flip as causal while the filtered variant drops the candidate. We expect `necessity_raw` to show higher variance across candidates and to admit a small number of leaky predicates that reduce original-distribution precision, with the gap between `necessity` and `necessity_raw` quantifying the rewrite-noise effect.

**vs. FARA with `tau_locality=0.9` (relaxed locality)** — On a case where a single non-descendant event output changes due to the substitution, the relaxed variant admits the candidate while the strict variant drops it. We expect the relaxed variant to show a modest shift-F1 gain but a measurable original-distribution precision drop, confirming that the locality threshold is doing causal work and that the strict setting is not overly conservative.

### What would falsify this idea

If FARA's F1 gain over the static MAST baseline is uniform across shifted subsets rather than concentrated on subsets whose failure modes are absent from C0, the causal-attribution story is wrong and the gain is a generic expansion effect. If the ablation (no counterfactual perturbation) matches FARA's shift F1 without the predicted precision regression on the original distribution, the necessity phase is not contributing and the central claim collapses. If `necessity_raw` and `necessity` produce identical admissions on all candidates, the locality check is vacuous and the load-bearing assumption is untested.

## References

1. DynaFlow: Learning Free-Form Multi-Agent Orchestration
2. Why Do Multi-Agent LLM Systems Fail?
3. Challenges in Human-Agent Communication
4. Harnessing Language for Coordination: A Framework and Benchmark for LLM-Driven Multiagent Control
5. The Rise and Potential of Large Language Model Based Agents: A Survey
6. AI Transparency in the Age of LLMs: A Human-Centered Research Roadmap
7. Voyager: An Open-Ended Embodied Agent with Large Language Models
8. Robot Behavior-Tree-Based Task Generation with Large Language Models
