# Belief-Indexed Memory with Refinement-Aware Retraction for Continual Learning Agents

## Motivation

Memory-centric continual learning frameworks such as MCCL (Memory-Centric Continual Learning for AI Agents) treat memory as a static store managed by a two-loop scheduler, so when the agent's world model refines — a previously held belief being corrected by new evidence — stored entries tagged under the old belief silently contradict the refined model, and no mechanism exists to detect or retract them. The root cause is structural: the frontier's memory schema has no representation of the belief state under which an entry was valid, so semantic invariants across entries cannot be enforced as beliefs change. This gap recurs across any memory-centric agent design that schedules writes without a formal consistency operator.

## Key Insight

If every memory entry is tagged with the belief state under which it was stored and a refinement operator re-validates entries against the refined model, then monotone consistency of retained memory is maintained as a structural invariant of the update rule rather than as an emergent property of scheduling.

## Method

**What it is.** We introduce BIM-R (Belief-Indexed Memory with Refinement-aware retraction), a memory schema and update operator for continual-learning agents. Inputs: a stream of (observation, task-feedback) pairs and an evolving world model W_t (a set of grounded propositions with truth values). Outputs: a belief-indexed memory store M where every retained entry is consistent with the current world model, and a recorded retraction/refinement log.

**Load-bearing assumption (stated explicitly).** BIM-R assumes that the agent's world model W_t is a sound and complete propositional representation over the vocabulary that memory entries assert, so that CONSISTENT(e, W) reliably distinguishes genuine contradiction from mere unverifiability and that entries can be re-derived from W' alone without new observations. This is the closed-world/complete-knowledge assumption; it fails when an entry asserts a proposition outside W's vocabulary (unverifiability, not contradiction) and when entries are episodic/perceptual facts not derivable from W'. To make this assumption testable rather than hidden, we replace the binary entailment-based CONSISTENT/DERIVABLE predicates with a three-valued consistency operator (entailed / contradicted / unknown) and add a provenance link from each entry to the specific propositions and observation that supported it. Re-validation checks contradiction against the cited support rather than requiring derivability from W'; entries whose support propositions are untouched by a refinement are left valid without needing to be re-derived, and 'unknown' entries are quarantined rather than conservatively retained or retracted. The world model W is a hand-specified propositional ontology with a known entailment oracle (a finite set of propositional atoms and Horn clauses; entailment computed by forward chaining in time linear in the number of clauses). We report the measured soundness/completeness of W on the stream (fraction of entry-asserted propositions that are in W's vocabulary and correctly valued) as an empirical grounding of the CONSISTENT predicate.

**How it works.**

```pseudocode
# State: world model W (set of propositions p with truth value), memory M (entries e = (content, b_e, support, prov)), refinement log L
# prov = provenance link: the set of propositions and the observation id that supported e at write time
# Hyperparameters: consistency threshold tau (default 0.9), support decay gamma (default 0.95), quarantine flag q (default False)

function WRITE(content, b_current, W, prov):
    e = (content, b_current, support=1.0, prov=prov)
    M.add(e)
    return e

function REFINE(W, new_evidence):
    # Phase 1: propose candidate refinements to W from new evidence
    W' = W
    for (p, v) in new_evidence:
        if p in W and W[p] != v:
            W' = W' with p := v          # belief revision
    # Phase 2: re-validate every memory entry against W' using three-valued consistency
    for e in M:
        status = CONSISTENT_3(e, W')   # returns ENTAILED, CONTRADICTED, or UNKNOWN
        if status == CONTRADICTED:
            if SUPPORT(e) > tau:
                e.support *= gamma          # degrade, do not yet retract
            else:
                M.remove(e); L.record(retract(e, reason=inconsistent_with(W')))
        elif status == UNKNOWN:
            e.q = True                      # quarantine: exclude from retrieval, do not retract
        else:  # ENTAILED
            e.q = False
    # Phase 3: re-derive entries whose support fell but whose content is still derivable
    for e in M where e.support < tau and not e.q:
        if DERIVABLE(e.content, W'):
            e.b_e = W'; e.support = 1.0     # re-validate under refined belief
    W = W'
    return W, M

function CONSISTENT_3(e, W):
    # three-valued consistency against the entry's cited support, not full derivability from W
    # check each proposition p in e.content against W
    statuses = []
    for p in e.content.propositions:
        if p not in W.vocabulary:
            statuses.append(UNKNOWN)
        elif W.entails(p):
            statuses.append(ENTAILED)
        elif W.entails(not p):
            statuses.append(CONTRADICTED)
        else:
            statuses.append(UNKNOWN)
    if any(s == CONTRADICTED for s in statuses):
        return CONTRADICTED
    if all(s == ENTAILED for s in statuses):
        return ENTAILED
    return UNKNOWN

function DERIVABLE(content, W):
    # entry is derivable iff every proposition it asserts is entailed by W
    return all(W.entails(p) for p in content.propositions)

function DUAL_LOOP_STEP(observation, feedback):
    # inner loop: memory write
    b_cur = BELIEF_STATE(W)
    prov = extract_provenance(observation, feedback, W)  # propositions and observation id supporting the write
    WRITE(extract(observation, feedback), b_cur, W, prov)
    # outer loop: schedule a refinement step when feedback signals belief change
    if BELIEF_SHIFT(feedback, W):
        REFINE(W, extract_evidence(feedback))
    return M, W
```

**Why this design.** We chose a belief-indexed tag per entry over a global timestamp because consistency is a property of the belief state at write time, not of recency; the cost is that entries written under stale beliefs must be re-checked on every refinement, which we accept because refinement steps are sparse relative to writes. We chose a support-decay-then-retract rule over immediate deletion because immediate deletion discards entries that may be re-derivable under the refined model, losing transferable knowledge; the cost is a small window of retained-but-degraded entries that must be excluded from retrieval. We chose to re-derive entries from the refined world model rather than re-query the environment because re-derivation is cheap and does not require new observations; the cost is that entries whose content is not logically derivable from the world model are retracted rather than repaired. A domain expert would not describe BIM-R as a variant of MCCL's dual-loop scheduler: MCCL schedules memory writes and policy updates, whereas BIM-R's outer loop is a formal refinement operator that mutates the memory store itself under a consistency invariant, and its inner loop tags entries with belief state — a mechanism absent from MCCL's static taxonomy. The three-valued operator and provenance link make the load-bearing assumption explicit: entries whose support propositions are untouched by a refinement are left valid without needing to be re-derived, and 'unknown' entries are quarantined rather than conservatively retained or retracted.

**Why this measures what we claim.** The consistency predicate `CONSISTENT_3(e, W)` measures semantic non-contradiction of a memory entry because we assume the world model W is a sound and complete representation of the agent's current beliefs over the propositions entries assert; this assumption fails when an entry asserts a proposition outside W's vocabulary, in which case CONSISTENT_3 returns UNKNOWN (unverifiability rather than contradiction) and the entry is quarantined rather than conservatively retained or retracted. The support-decay quantity `support(e)` measures the evidential grounding of an entry because we assume support is initialized at write time and decays only under inconsistency; this assumption fails when an entry is written under a belief that is later refined without contradiction (e.g., a refinement that adds a proposition), in which case support overstates grounding and re-derivation under Phase 3 restores it. The refinement log L measures the rate of belief-driven memory revision because we assume every retraction is caused by a belief change recorded in the log; this assumption fails when retraction is triggered by support decay from repeated weak inconsistencies, in which case L reflects accumulated drift rather than discrete belief revision. The provenance link `prov` measures the evidential basis of an entry because we assume the cited propositions and observation are the actual support at write time; this assumption fails when the observation is noisy or the provenance extraction misses a supporting proposition, in which case re-validation may miss a contradiction or falsely flag one.

## Contribution

['A belief-indexed memory schema in which every entry carries the belief state under which it was stored, together with a formal refinement operator that enforces monotone consistency by re-validating, degrading, re-deriving, or retracting entries when the world model refines.', 'A dual-loop controller that schedules belief refinement steps alongside memory writes, making semantic consistency an invariant of the update rule rather than an emergent property of scheduling.', 'A retraction/refinement log that records the causal chain from belief change to memory revision, enabling audit of which retained memories depend on which beliefs.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | Interference-and-revision agent stream (retention/adaptation/transfer splits) | Directly exercises belief change and stale-memory interference |
| Dataset (non-synthetic) | Text-world navigation stream with genuine proposition flips (e.g., object moved between rooms) | Tests whether the consistency invariant transfers beyond the bespoke dataset |
| Primary metric | Retention-under-interference with post-refinement task transfer | Isolates genuine behavioral gain from context expansion |
| Secondary metric | Retrieval audit: fraction of failing cases where a stale entry was actually returned | Verifies that transfer loss is caused by retrieved stale entries, not retrieval ranking or policy stochasticity |
| Baseline | Passive retrieval-cache memory | Tests whether belief-indexing beats recency-based storage |
| Baseline | Recency-timestamped memory store | Tests whether belief tags beat global timestamps |
| Baseline | Immediate-delete-on-inconsistency memory | Tests whether decay-then-rederive preserves transferable knowledge |
| Baseline | AGM entrenchment ordering over the same proposition set | Tests whether belief-indexed schema yields gains the standard truth-maintenance approach does not |
| Ablation-of-ours | BIM-R without Phase-3 re-derivation | Isolates the re-derivation mechanism's contribution |
| Ablation-of-ours | BIM-R with binary CONSISTENT (no three-valued operator, no provenance) | Isolates the contribution of the three-valued consistency and provenance link |

### Why this setup validates the claim

The central claim is that indexing memory by belief state and refining memory under a consistency invariant yields better retention and transfer than recency- or cache-based memory when the underlying world model changes. The dataset must therefore contain genuine belief revisions mid-stream (propositions that flip truth value) interleaved with writes, so that stale entries become inconsistent. The primary metric—retention under interference combined with post-refinement transfer—is the right detector because it separates two effects: pure recall (which passive caches can score well on) from behavior after a belief shift (where stale entries actively mislead). The passive-cache baseline tests the sub-claim that consistency-at-write-time matters more than recency; the recency-timestamped baseline isolates belief-indexing from mere temporal tagging; the immediate-delete baseline tests the sub-claim that decay-then-rederive preserves knowledge that immediate deletion destroys. The AGM entrenchment baseline tests whether the belief-indexed schema yields gains the standard truth-maintenance approach does not. The Phase-3 ablation is the decisive control: if removing re-derivation erases our advantage, the gain is attributable to the refinement operator rather than to memory size or retrieval heuristics. The three-valued ablation tests whether the provenance link and quarantine mechanism are necessary for the consistency invariant to hold under incomplete vocabulary. The retrieval audit addresses the missing equivalence that post-refinement transfer measures consistency-driven retention: it logs which entries were actually returned on failing cases, so we can verify that observed transfer loss is caused by retrieved stale entries (not retrieval ranking or policy stochasticity) and detect the failure mode where a stale entry is never retrieved yet transfer still drops. Together these form a falsifiable test: a real mechanism should show a subset-specific gap, not a uniform lift.

### Expected outcome and causal chain

**vs. Passive retrieval-cache memory** — On a case where the agent learns "container A holds object X," then later evidence refines the model so A is emptied and X moves to B, the passive cache retrieves the stale "X in A" fact because it ranks by recency/recall and never checks consistency against the current world model. Our method instead re-validates the entry during REFINE, finds it inconsistent with W', degrades support, and re-derives the corrected placement under the refined belief, so retrieval returns the consistent fact. We expect a noticeable gap on the interference-and-revision subset but near parity on the pure-recall subset. The retrieval audit should show that on failing cases the passive cache returned the stale entry, confirming the causal chain.

**vs. Recency-timestamped memory store** — On a case where a belief revision arrives shortly after a burst of writes, the timestamped store keeps the most recent entries even though several were written under a now-falsified belief, because timestamps capture when, not what was believed. Our method tags each entry with its write-time belief state, so entries written under the stale belief are flagged and re-checked on refinement rather than trusted for recency. We expect our advantage to concentrate on entries written just before a belief shift, with the gap shrinking as the time between writes and revision grows. The retrieval audit should show that on failing cases the timestamped store returned entries written under the stale belief.

**vs. Immediate-delete-on-inconsistency memory** — On a case where a proposition is temporarily contradicted by weak evidence and later re-derived under a refined model, the immediate-delete store permanently discards the entry and loses transferable knowledge that would have been recoverable. Our method degrades support with decay gamma and only retracts below threshold, then Phase 3 re-derives entries whose content follows from the refined W'. We expect the immediate-delete baseline to show a sharp drop in post-refinement transfer on recoverable-knowledge cases, while our method retains most of it. The retrieval audit should show that on failing cases the immediate-delete store had no entry to return (it was deleted), confirming that the loss is due to deletion rather than retrieval ranking.

**vs. AGM entrenchment ordering** — On a case where a belief revision requires retracting a proposition that is deeply entrenched (highly supported) but contradicted by new evidence, the AGM ordering may retain the entrenched proposition and instead revise a less entrenched one, leading to a different set of retractions than our belief-indexed schema. Our method tags each entry with its write-time belief state and re-validates against the refined W', so retraction is driven by consistency with the current model rather than by an entrenchment ordering. We expect our advantage to concentrate on cases where entrenchment and consistency diverge (e.g., a highly supported but now-false belief), with near parity on cases where they coincide. The retrieval audit should show that on failing cases the AGM baseline returned an entry that was consistent with its entrenchment ordering but inconsistent with the refined W'.

### What would falsify this idea

If our gain over the passive-cache and recency baselines is uniform across all subsets rather than concentrated on the interference-and-revision and recoverable-knowledge subsets, the claimed causal mechanism is not what produces the effect. Likewise, if removing Phase-3 re-derivation leaves our advantage intact, the refinement operator is not doing the work we attribute to it. If the retrieval audit shows that failing cases do not involve retrieved stale entries, the transfer loss is not caused by the consistency mechanism.

## References

1. Memory-Centric Continual Learning for AI Agents
