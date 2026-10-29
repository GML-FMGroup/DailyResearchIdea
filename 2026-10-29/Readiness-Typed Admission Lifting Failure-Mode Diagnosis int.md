# Readiness-Typed Admission: Lifting Failure-Mode Diagnosis into a Scheduler Feasibility Predicate

## Motivation

Current agent-serving schedulers and failure taxonomies live on incompatible surfaces: TAO scores waiting sessions by a monotone (tool-readiness, cache-credit, token-footprint) lattice and solves an exact joint admission/placement DP, while MAST classifies the same sessions' pathologies into 14 flat categorical failure modes that carry no order — so a diagnosed 'tool-call loop' or 'context overflow' cannot enter the DP's feasibility region and is only usable post-hoc. The structural root cause is that MAST's taxonomy was built as a descriptive label set, never as a precondition algebra, so a naive port feeding MAST labels into TAO's scorer destroys the dominance order the DP's exactness depends on. This is the recurring field-level bottleneck: diagnostic signal and control signal are produced by separate computations that cannot be composed, as also seen when InferCept's interception state and SGLang's RadixAttention prefix state are consumed only as heuristics rather than as typed scheduling constraints.

## Key Insight

Each MAST failure mode is a lower-bound precondition on the session's tool-state vector, so re-typing the 14 labels as readiness types embeds them into TAO's existing dominance lattice and turns diagnosis and admission into one feasibility computation rather than a pipeline.

## Method

**What it is.** FARA (Failure-Algebra Readiness Admission) takes as input a waiting agent session's observed tool-state vector `s = (r, c, t, e)` — near-term tool readiness `r`, time-decayed cache credit `c`, token footprint `t`, and trajectory repetition entropy `e` (Shannon entropy over the last 8 tool-call transition types, computed from the session's tool-call log) — plus a MAST-style classifier's per-mode posterior, and outputs a single boolean admission/placement decision by evaluating a typed feasibility predicate inside TAO's dominance-ordered DP. Diagnosis and scheduling are the same evaluation, not two stages.

**Load-bearing assumption (stated explicitly).** FARA assumes that each MAST failure mode `m` can be represented as a monotone axis-aligned box `tau[m] = (r_min[m], c_min[m], t_max[m], e_max[m])` in the augmented tool-state space, such that box membership `s in box_m` is equivalent to 'the pathology of mode m is currently satisfied', and that the 14 estimated boxes form a partial order compatible with TAO's dominance lattice. This assumption is audited per mode (Phase 1b) and is not assumed silently: modes whose empirical monotonicity fails are flagged and their box is replaced by the empirical monotone projection, with the audit reported.

**How it works.**

```pseudocode
# Phase 1: readiness typing (offline, once per taxonomy)
for each MAST failure mode m in {1..14}:
    tau[m] = (r_min[m], c_min[m], t_max[m], e_max[m])   # a box in augmented tool-state space
    # estimation: per-mode quantile regression of each coordinate on the annotated traces
    #   r_min[m] = 10th percentile of r over traces labeled m
    #   c_min[m] = 10th percentile of c over traces labeled m
    #   t_max[m] = 90th percentile of t over traces labeled m
    #   e_max[m] = 90th percentile of e over traces labeled m
    # 95% bootstrap CIs (1000 resamples) reported per bound; modes with <30 traces
    #   use pooled quantiles from the nearest mode in the MAST hierarchy and are flagged 'rare'
    # tau is a partial order: m1 <= m2  iff  box(m1) is dominated by box(m2)

# Phase 1b: monotonicity audit (offline, once per taxonomy)
for each mode m:
    fit logistic regression P(label=m | r, c, t, e) on annotated traces
    check sign of each coefficient; if any coefficient contradicts the box direction
      (e.g., positive coefficient on e for a mode with e_max), flag m as 'non-monotone'
    report per-mode audit table: coefficient signs, AUC, #traces, CI width
    # lattice-consistency check: verify transitivity and antisymmetry of the estimated
    #   box order over all 14 modes; report any violations (incomparable or inconsistent pairs)

# Phase 2: online, per scheduling epoch
for each waiting session j:
    p_j = classifier_posterior(modes)          # 14-vector, LLM-as-Judge head
    m*_j = argmax_m p_j[m]                     # diagnosed mode (single label)
    feasible_j = dominates(s_j, tau[m*_j])     # readiness-typed feasibility predicate
    # dominates(a,b) = a.r>=b.r_min and a.c>=b.c_min and a.t<=b.t_max and a.e<=b.e_max

# Phase 3: joint admission + placement DP over the dominance lattice
# value of session j if admitted on node n:
V(j,n) = w_r * min(r_j, tau[m*_j].r_min)      # readiness credit, capped at type
       + w_c * decay(c_j; n)                  # node-local cache credit
       - w_t * t_j                            # footprint penalty
# DP state = (sorted feasible sessions by dominance, node capacities)
# transition = admit next dominance-maximal feasible session, place on argmax_n V
return DP.solve({j : feasible_j}, capacities, V)
```

The three phases are one program: Phase 2's `feasible_j` prunes the DP's action set to sessions whose diagnosed failure type is dominated by their current tool-state, and Phase 3's value uses the same `tau[m*_j]` to cap readiness credit so a session cannot be over-rewarded for readiness beyond what its failure type requires. Weights `w_r, w_c, w_t` are set by a one-dimensional grid search on a held-out 10% of traces (grid: 0.1 to 2.0 step 0.1, 20 values per weight, coordinate-wise); `decay` uses TAO's time-aware exponential with half-life 200ms. The classifier is an off-the-shelf LLM-as-Judge (GPT-4o-mini) with reported per-mode accuracy on a 200-trace held-out split; the joint DP is solved exactly for ≤8 nodes and ≤32 waiting sessions, and falls back to a documented greedy dominance-maximal heuristic (admit next dominance-maximal feasible session, place on argmax_n V) when either bound is exceeded, with wall-clock reported per phase.

**Why this design.** We chose to re-type MAST's 14 modes as boxes in TAO's tool-state space rather than feed the 14-vector directly into TAO's scalar scorer, because categorical labels have no order and would collapse the DP's dominance lattice into an unordered set — accepting the cost that the box bounds `tau[m]` must be estimated from annotated traces and may be loose for rare modes. We chose a closed-form `dominates()` predicate over a learned feasibility head because the predicate is what makes the DP exact over the lattice, accepting the cost that mis-typed modes become hard (not soft) infeasibilities. We chose to cap readiness credit at `tau[m*_j].r_min` rather than use raw `r_j`, because a session diagnosed as tool-call-loop should not be rewarded for readiness far beyond its required threshold, accepting the cost that a genuinely-ready session with a misdiagnosis is under-valued. We added the trajectory repetition entropy coordinate `e` because sequence-defined modes like tool-call-loop are properties of repeated transitions, not of instantaneous readiness, and without a coordinate in which they can be monotone the box typing would be unsupported for those modes. A domain expert would not describe FARA as 'TAO plus a MAST classifier': the classifier's output is not a feature but the feasibility predicate itself, and the DP's action set is defined by that predicate — the composition TAO+MAST that a reviewer might expect would feed labels as features and lose exactness, which is precisely the failure FARA's typing prevents.

**Why it measures what we claim.** The quantity `dominates(s_j, tau[m*_j])` measures schedulability-under-diagnosis because the assumption that each MAST mode is a monotone lower-bound precondition on the augmented tool-state makes the box membership equivalent to 'this session's pathology is currently satisfied'; this assumption fails when a mode is non-monotone (e.g., a failure that is triggered by too-high readiness), in which case `dominates` reflects only the monotone projection of the mode and admits sessions the true pathology would block — the Phase 1b audit reports exactly which modes this applies to. The quantity `V(j,n)`'s readiness cap `min(r_j, tau[m*_j].r_min)` measures marginal schedulable value because the assumption that credit beyond the type threshold is wasted makes the cap equivalent to the DP's true marginal contribution; this assumption fails when a session's downstream tool chain needs more readiness than its diagnosed mode's threshold, in which case the cap underestimates value and may deprioritize a session that would unblock others. The cache-credit term `decay(c_j; n)` measures reusable KV locality because TAO's time-decay assumption makes node-local credit equivalent to recomputation savings; this assumption fails under cross-node migration, in which case the term reflects stale locality.

## Contribution

["A readiness-typed failure algebra that lifts MAST's 14 flat failure modes into a partial order of tool-state boxes, making each diagnosed mode a feasibility predicate rather than a label.", 'A joint admission/placement DP (FARA) whose action set is defined by the typed feasibility predicate, so diagnosis and scheduling are one computation and the DP remains exact over the dominance lattice.', 'An empirical design principle: typing failure modes as lower-bound preconditions is what restores TAO-style DP exactness, whereas feeding the same labels as features provably loses the order and degrades scheduling.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | MAST-annotated 1600-trace MAS corpus, replayed with tool-state logs | Provides per-mode labels plus tool-state for typing |
| Primary metric | Admitted-session success rate at fixed node capacity | Directly measures whether feasibility typing admits schedulable sessions |
| Baseline | TAO scalar scorer | Tests whether typed feasibility beats untyped scoring |
| Baseline | TAO + MAST labels as features | Tests whether typing beats label-as-feature composition |
| Baseline | FIFO admission with MAST-argmax gating | Tests whether dominance ordering adds value over gating |
| Baseline | Greedy dominance-maximal heuristic (no DP) | Tests whether exact DP contributes beyond greedy ordering |
| Ablation-of-ours | FARA without readiness cap (raw r_j) | Isolates the value of capping credit at tau |
| Ablation-of-ours | FARA without trajectory entropy coordinate e | Isolates the value of the augmented coordinate for sequence-defined modes |
| Ablation-of-ours | FARA with soft posterior-weighted feasibility (sum_m p_j[m] * 1[s_j in box_m] > 0.5) | Tests whether hard argmax predicate is load-bearing vs. soft mixture |

### Why this setup validates the claim

The central claim is that re-typing MAST failure modes as boxes in TAO's augmented tool-state space, and using that typing as the feasibility predicate inside the dominance-ordered DP, admits more genuinely schedulable sessions than either untyped scoring or label-as-feature composition. Each baseline isolates one sub-claim. TAO scalar scorer tests whether typing matters at all: if labels carry no order, its dominance lattice collapses and it should admit sessions whose pathology is not yet satisfied. TAO+MAST-as-features tests whether the specific mechanism (predicate, not feature) matters. FIFO-with-gating tests whether the DP's dominance ordering contributes beyond argmax gating. The greedy dominance-maximal heuristic tests whether exact DP contributes beyond greedy ordering. The readiness-cap ablation tests whether the cap is load-bearing. The entropy-coordinate ablation tests whether the augmented coordinate is necessary for sequence-defined modes. The soft-posterior ablation tests whether the hard argmax predicate is load-bearing versus a soft mixture. Admitted-session success rate at fixed capacity is the right metric because it directly reflects the admission decision's correctness—not throughput, which can rise while admitting wrong sessions. If FARA's advantage is concentrated on trace subsets where the diagnosed mode's box membership differs from raw readiness ranking, the causal story is confirmed; uniform gains would suggest the classifier, not the typing, is doing the work. The Phase 1b monotonicity audit and lattice-consistency check are reported as preconditions: modes flagged non-monotone are listed, and any transitivity/antisymmetry violations in the estimated box order are reported, so the DP's exactness claim is scoped to the verified lattice.

### Expected outcome and causal chain

**vs. TAO scalar scorer** — On a case where a session is diagnosed with tool-call-loop but has readiness far above its threshold while a context-overflow session sits just below its token ceiling, TAO's scalar scorer rewards raw readiness and admits the loop session, which then re-enters the same loop and consumes a node slot. Our method's `dominates()` predicate rejects the loop session's over-readiness as irrelevant (and its high repetition entropy `e` exceeds `e_max` for the loop mode) and caps its credit, admitting the overflow session instead. We expect a noticeable success-rate gap concentrated on traces with high-readiness-but-pathological sessions, with near-parity on traces where diagnosis is unambiguous and readiness is monotone in true schedulability.

**vs. TAO + MAST labels as features** — On a case where a session's mode posterior is split between two adjacent modes with different box bounds, feeding the 14-vector as features produces a smooth score that admits the session under the looser mode's implicit threshold, even though its tool-state satisfies only the stricter mode's box. Because features do not define the action set, the DP cannot prune this session, and it occupies capacity while blocked. Our method uses argmax mode as a hard predicate, so the session is excluded until its state dominates the diagnosed box. We expect the gap to widen as posterior entropy increases, and to shrink to parity on low-entropy traces.

**vs. FIFO admission with MAST-argmax gating** — On a case where two feasible sessions compete for one slot, one dominance-maximal and one not, FIFO admits by arrival order and may pick the dominated session, which then blocks a downstream tool that the maximal session would have unblocked. Our DP's dominance ordering selects the maximal session first because the lattice transition admits next dominance-maximal feasible sessions. We expect a gap on traces with contention among feasible sessions and little gap when at most one session is feasible per epoch, since ordering is then irrelevant.

**vs. Greedy dominance-maximal heuristic** — On a case where two feasible sessions compete for two nodes with different cache-credit decay, greedy admits the dominance-maximal session first but places it on the node with highest immediate `V`, which may be the node where the second session would have had higher cache credit; the exact DP considers the joint placement and swaps. We expect a small gap concentrated on traces with heterogeneous node cache states, and near-parity when nodes are homogeneous.

**vs. FARA without readiness cap** — On a case where a session diagnosed as tool-call-loop has readiness far above its threshold, the uncapped value rewards it as if all readiness were schedulable, so it outcompetes a context-overflow session whose readiness is just at threshold. The capped value admits the overflow session. We expect the gap concentrated on traces with high-readiness-pathological sessions.

**vs. FARA without trajectory entropy coordinate** — On a case where a tool-call-loop session has moderate readiness and low token footprint, the box without `e` cannot distinguish it from a healthy session with the same (r, c, t), so it is admitted and loops. With `e`, the high repetition entropy exceeds `e_max` and the session is excluded. We expect the gap concentrated on sequence-defined modes (tool-call-loop, step repetition) and near-parity on resource-defined modes (context overflow).

**vs. FARA with soft posterior-weighted feasibility** — On a case where the posterior is split between two adjacent modes with different box bounds, the soft mixture admits the session if the weighted box membership exceeds 0.5, even though the argmax mode's box is not satisfied. The hard predicate excludes it. We expect the hard predicate to outperform on high-posterior-entropy traces and the soft mixture to be competitive on low-entropy traces, confirming the argmax collapse is load-bearing.

### What would falsify this idea

If FARA's success-rate gain over TAO+MAST-as-features is uniform across trace subsets rather than concentrated on high-posterior-entropy and high-readiness-pathology subsets, the typing mechanism is not the cause and the central claim is wrong. Likewise, if removing the readiness cap (the ablation) leaves the gain unchanged, the cap is not load-bearing and the claimed value mechanism fails; if the Phase 1b audit finds the estimated box order violates transitivity or antisymmetry for a majority of mode pairs, the dominance-lattice precondition fails and the DP's exactness claim is unsupported.

## References

1. TAO: Tool-Aware Joint Scheduling for LLM-Based Agentic Serving
2. Why Do Multi-Agent LLM Systems Fail?
3. InferCept: Efficient Intercept Support for Augmented Large Language Model Inference
4. SGLang: Efficient Execution of Structured Language Model Programs
5. SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills
6. Efficient Memory Management for Large Language Model Serving with PagedAttention
7. DeepSpeed- Inference: Enabling Efficient Inference of Transformer Models at Unprecedented Scale
