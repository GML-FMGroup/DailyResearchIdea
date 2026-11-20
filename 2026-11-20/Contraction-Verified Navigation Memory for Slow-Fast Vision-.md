# Contraction-Verified Navigation Memory for Slow-Fast Vision-Language Navigation

## Motivation

Across the vision-language navigation literature, memory is built as an append-and-retrieve graph with no revision operator: LH-VLN's MGDM (E1) integrates short-term blurring with long-term retrieval but never contracts a stored node when a new observation contradicts it, and GSA-R2R's GR-DUET (E2) retains navigation graphs across environment-specific training without any consistency rule that would force conflicting edges to be removed. ZeroVLN (E6) inherits this same unverified memory substrate inside its slow-fast split, so the VLM brain can re-issue a tool call toward a node the agent's own observations have already falsified. The structural root cause is that memory is treated as a monotone accumulator, not as a belief state that must remain internally satisfiable.

## Key Insight

If the navigation graph is treated as a belief state constrained by a contraction invariant — every stored edge must remain consistent with the union of observations that produced it — then a contradictory observation forces a unique minimal revision before any tool call, and the slow brain's requests become checkable against that revised graph rather than against an unreconciled accumulation.

## Method

**What it is.** CVMN (Contraction-Verified Memory Navigation) is a wrapper around ZeroVLN's slow-fast split. Inputs: the current navigation graph G_t (nodes = visited/frontier waypoints, edges = traversed or hypothesized transitions with observation signatures), the new RGB-D observation o_t, and the VLM brain's proposed natural-language movement request r_t. Outputs: a contracted graph G'_t, a boolean consistency verdict on r_t, and either a dispatched tool call or a re-issued request.

**How it works.**

```pseudocode
# Phase 1: observation ingestion
observe(o_t) -> signature s_t = (landmark set, free-space directions, depth range)
# Per-edge evidence accumulator (replaces single-frame hard threshold):
#   each edge e stores a log-likelihood ratio LLR_e over 'edge valid' vs 'edge invalid'
#   updated on every observation of e's endpoints:
#       LLR_e <- LLR_e + log( p(s_t | edge valid) / p(s_t | edge invalid) )
#   p(s_t | edge valid) and p(s_t | edge invalid) are Gaussian densities fit on a
#   held-out calibration set of 512 VLN-CE episodes (see calibration below).
for each edge e in G_t incident to current node:
    predicted = edge_signature_prediction(e)      # stored at insertion
    delta = conflict(s_t, predicted)              # same signature distance as before
    LLR_e <- LLR_e + log_gaussian_ratio(delta)    # calibrated per-edge evidence
    if LLR_e < tau_llr:                           # tau_llr calibrated to FPR=0.05
        mark e as CONTRADICTED

# Calibration of tau_llr (done once, before evaluation):
#   On 512 held-out VLN-CE episodes, collect all (delta, ground-truth edge-valid) pairs
#   from re-observed edges. Fit Gaussian densities for valid/invalid. Choose tau_llr
#   as the LLR threshold yielding false-positive rate 0.05 on this set.
#   tau_conflict=0.6 is retained only as the per-frame delta cap; the decision
#   to contract is made on accumulated LLR, not on a single frame.

# Phase 2: contraction operator (unique minimal revision)
C = set of CONTRADICTED edges
if C nonempty:
    # rank by observation support count; remove lowest-support edges first
    # until remaining graph has no CONTRADICTED edge
    for e in sort_by_support_ascending(C):
        remove e from G_t
        # cascade: any node whose only incident edges were removed
        # becomes UNREACHABLE and is demoted to frontier-hypothesis
        if node_unreachable(e.child):
            demote_to_frontier(e.child)
    G'_t = G_t
else:
    G'_t = G_t

# Phase 3: pre-tool-call consistency invariant
propose r_t from brain(G'_t, instruction)
target_node = parse_target(r_t)
if target_node in G'_t and path_exists(G'_t, current, target_node):
    dispatch r_t to action expert
else:
    # re-issue: brain sees the contraction report and must re-propose
    brain.receive(contraction_report(C, G'_t))
    r_t' = brain.repropose()
    dispatch r_t'   # logged as a re-issue event
```

**Why this design.** We chose a deterministic minimal-revision contraction operator over a learned memory-editing head because the failure we target is structural (contradictory edges accumulate) and a learned editor would itself need supervision on contradictions that no existing VLN dataset provides, accepting the cost that our operator cannot recover from a wrongly-contracted edge without re-observation. We chose to rank contradicted edges by observation-support count rather than by recency because recency is exactly the append-and-retrieve heuristic that MGDM (E1) and GR-DUET (E2) already use, and support-count gives a stable order that does not flip under a single noisy frame; the cost is that an early, well-supported but wrong edge is expensive to remove. We chose to gate the action expert on a path-existence invariant rather than on the brain's own confidence because ZeroVLN's tool-calling interface (E6) provides a discrete, checkable target node — unlike a geometric goal — so the invariant can be evaluated exactly; the cost is that free-form requests not naming a graph node must be mapped to one by a fixed parser, which we restrict to the closed vocabulary of node labels. A domain expert would not describe this as a variant of MGDM or GR-DUET: those methods add memory, whereas CVMN subtracts memory under a satisfiability rule and uses the subtraction as a verification signal for the slow brain.

**Why it measures what we claim.** The conflict score `conflict(s_t, predicted)` measures observational contradiction because we assume the stored edge signature is a sufficient statistic of the observation that created it, so a large divergence implies the world state changed or the edge was wrong; this assumption fails when the observation is corrupted by occlusion or lighting, in which case the score reflects sensor noise rather than contradiction. The per-edge accumulated LLR measures evidence of edge invalidity because we assume observations of the same edge are conditionally independent given the edge's true state, so log-likelihood ratios add; this assumption fails when observations are temporally correlated (e.g., a repeated viewpoint), in which case the LLR overstates evidence and the operator may remove a correct edge. The contraction order by support count measures minimal revision because we assume independent observations are exchangeable evidence, so removing the least-supported edge yields the smallest belief change; this fails when observations are temporally correlated, in which case support count overstates evidence and the operator may remove a correct edge. The path-existence invariant measures whether the brain's request respects contracted memory because we assume the parser maps every meaningful request onto a graph node, so a non-resolvable request is genuinely inconsistent; this fails for requests that legitimately reference unexplored space, in which case the invariant reflects parser coverage rather than memory violation, and we log these as frontier-hypothesis requests rather than counting them as re-issues.

## Contribution

['A contraction operator on the navigation memory graph that enforces a monotone consistency invariant — every stored edge must remain supported by the observations that produced it — with a deterministic minimal-revision rule and cascade demotion of unreachable nodes.', "A pre-tool-call consistency check that verifies the slow VLM brain's re-issued movement requests against the contracted graph, turning ZeroVLN's semantic tool-calling interface into a checkable surface rather than an unverified one.", 'A contraction-report feedback channel that lets the VLM brain re-propose after a contradiction, providing a concrete mechanism for memory revision in slow-fast VLN without navigation-domain fine-tuning of the brain.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | VLN-CE (continuous environments) | ZeroVLN's standard benchmark; supports graph-node parsing |
| Primary metric | Success Rate on contradiction-heavy episodes | Directly targets structural memory failure |
| Baseline | ZeroVLN (slow-fast, no contraction) | Tests whether contraction adds value over tool-calling alone |
| Baseline | GR-DUET (memory-graph, recency retrieval) | Tests whether support-order contraction beats append-retrieve |
| Baseline | MGDM (multi-granularity dynamic memory) | Tests whether subtraction beats memory addition |
| Ablation-of-ours | CVMN minus contraction (invariant gate only) | Isolates the contraction operator's contribution |
| Ablation-of-ours | CVMN with single-frame tau_conflict=0.6 (no LLR) | Tests whether calibrated multi-observation evidence is necessary |
| Ablation-of-ours | CVMN with synthetic occlusion injection (sensor-noise stress) | Quantifies false-positive contraction rate under noise |

### Why this setup validates the claim

The central claim is that contradictory edges accumulate in the navigation graph and that a deterministic minimal-revision contraction operator, used as a pre-tool-call verification signal, recovers performance that memory-addition methods cannot. VLN-CE is the right dataset because ZeroVLN already reports there, so the slow-fast split is held fixed and only the memory-revision layer varies. The primary metric is success rate restricted to episodes where the stored graph accumulates a contradiction (a revisited waypoint whose observation signature diverges from its stored prediction); on such episodes the causal mechanism should fire, whereas on contradiction-free episodes all methods should be near parity. ZeroVLN tests whether the invariant gate alone (without contraction) suffices; GR-DUET tests whether recency-ordered retrieval, which E2 uses, is dominated by support-ordered removal; MGDM tests whether subtracting memory under a satisfiability rule beats adding memory, which is E1's approach. The ablation CVMN-minus-contraction isolates the operator: if the gain survives without contraction, the claim that contraction is the active ingredient is false. The single-frame ablation tests whether the calibrated LLR is necessary to avoid false-positive contractions under noise. The occlusion-injection ablation directly measures how often the operator removes a correct edge when the observation is corrupted, quantifying the failure mode stated in the method. Falsifiability comes from the subset structure: the prediction is a gap concentrated on contradiction-heavy episodes, not a uniform lift.

### Expected outcome and causal chain

**vs. ZeroVLN** — On a case where the agent revisits a waypoint whose RGB-D observation no longer matches the stored edge signature (a moved obstacle or a lighting shift), ZeroVLN's brain receives the stale graph unchanged and issues a tool call toward a node whose connecting edge is contradicted, so the action expert executes a request that the observation already refuted. Our method instead accumulates per-edge LLR evidence, marks that edge CONTRADICTED once LLR falls below the calibrated tau_llr, contracts it, demotes the unreachable child to frontier-hypothesis, and the path-existence invariant forces a re-issue rather than a dispatch. We expect a noticeable success-rate gap on contradiction-heavy episodes with near-parity on contradiction-free episodes, and a measurable re-issue rate that is near zero for ZeroVLN by construction.

**vs. GR-DUET** — On a case where an early, frequently observed edge is wrong and a later, sparsely observed edge is correct, GR-DUET's recency-weighted retrieval keeps the well-trodden wrong edge prominent because its memory graph is retained and append-biased, so the agent loops back through the contradicted transition. Our method ranks contradicted edges by observation-support ascending and removes the least-supported first, so the sparse-but-correct edge survives contraction while the heavily supported wrong edge is removed only after its support is outweighed. We expect GR-DUET to show loop-back failures concentrated on episodes with high revisit counts, and our method to show a gap that grows with revisit frequency rather than shrinking.

**vs. MGDM** — On a case where short-term blurring suppresses a just-observed contradiction because the blur window treats it as noise, MGDM's memory addition keeps both the stale and the new signature and lets the brain arbitrate, which can produce a confident but inconsistent request. Our method instead treats the contradiction as a satisfiability violation and subtracts the edge deterministically, so the brain never sees a graph in which both signatures coexist. We expect MGDM to show high brain-confidence but low invariant-pass rates on contradiction episodes, and our method to show the opposite pattern: lower confidence variance and a higher fraction of dispatched requests that pass the path-existence check.

**vs. CVMN minus contraction** — On a contradiction-heavy episode, the invariant gate alone cannot remove the contradicted edge, so the brain still sees a graph in which the stale edge is present; the gate may block a dispatch but cannot propose an alternative path through the contracted graph. We expect the full CVMN to outperform this ablation on contradiction-heavy episodes, with the gap vanishing on contradiction-free episodes.

**vs. CVMN with single-frame tau_conflict=0.6** — On an episode with a transient occlusion (e.g., a passing object), the single-frame test marks a correct edge CONTRADICTED and the support-ascending removal deletes it, whereas the LLR accumulator requires sustained evidence before contraction. We expect the single-frame variant to show a higher false-positive contraction rate and lower success on episodes with transient noise, while matching full CVMN on clean episodes.

**vs. CVMN with synthetic occlusion injection** — On episodes where we inject synthetic occlusion (randomly mask 20% of RGB-D frames), the LLR accumulator should still avoid contracting correct edges most of the time, but we expect a measurable increase in false-positive contractions relative to clean episodes. This ablation quantifies the failure mode stated in the method: when the observation is corrupted, the conflict score reflects sensor noise, and we report the fraction of correct edges removed under this stress.

### What would falsify this idea

If the success-rate gain over ZeroVLN, GR-DUET, and MGDM is uniform across contradiction-heavy and contradiction-free episodes rather than concentrated on the contradiction-heavy subset, then contraction is not the active ingredient and the central claim is wrong. A second falsifier: if CVMN-minus-contraction matches full CVMN on contradiction-heavy episodes, the operator adds nothing beyond the invariant gate; a third: if the single-frame variant matches full CVMN under occlusion injection, the calibrated LLR is unnecessary.

## References

1. ZeroVLN: Learning to Navigate without Supervised Fine-Tuning of the VLM Brain
