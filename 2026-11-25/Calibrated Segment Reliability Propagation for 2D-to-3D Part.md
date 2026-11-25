# Calibrated Segment Reliability Propagation for 2D-to-3D Part Segmentation

## Motivation

Both branches of 2D-to-3D part segmentation propagate 2D foundation-model semantic cues as if every SAM segment or 2D-expert region carried uniform trust: SegGraph builds a segment graph whose nodes modulate 2D features with no per-node confidence, and PartAgent's Structural-Semantic Expert Integration fuses multiview 2D evidence into compositional partitions while its Observation-Driven Closed-Loop Refinement verifies changes without any quantified reliability variable to certify that refinement is monotone. The root cause is structural: the propagation operators (graph message passing, image-to-surface fusion) are unweighted with respect to cue quality, so ambiguous or occluded regions inject errors that are indistinguishable from confident regions and can only be fixed post hoc by heuristics rather than measured.

## Key Insight

Because a segment's cross-view agreement is a sufficient statistic for its semantic reliability when views are conditionally independent given the true part label, temperature-scaling that agreement yields a per-segment confidence that is calibrated in the same units as label correctness and can therefore serve as a conserved weight through graph propagation and refinement, making ambiguous-region trust an explicit, checkable quantity rather than an implicit uniform assumption.

## Method

## CaliSeg: Calibrated Segment-Reliability Propagation

**(A) What it is.** CaliSeg takes as input a 3D object with a set of SAM/2D-expert segments (each with multiview 2D semantic evidence mapped to surface points) and outputs a per-point part labeling together with a per-segment calibrated reliability score in [0,1] that is propagated through a segment graph and used as the evidence variable in a closed-loop refinement. It replaces uniform 2D-cue trust in SegGraph-style propagation and PartAgent-style expert integration with a calibrated, propagated reliability.

**(B) How it works.**

```pseudocode
INPUT: surface points P, segments S = {s_1..s_n} (SAM masks lifted to surface),
       multiview 2D evidence E = {e_{i,v}} (per segment i, per view v: part-logit vector),
       per-segment view-coverage c_i in [0,1] (fraction of surface points in s_i visible in >=1 view),
       per-segment occlusion o_i in [0,1] (fraction of views where s_i is occluded by another segment),
       held-out calibration split C (segments with ground-truth part labels)
OUTPUT: part labels y(P), per-segment reliability r_i

# Phase 1: cross-view agreement statistic per segment
for each segment s_i:
    a_i = mean_v [ 1 - TV( softmax(e_{i,v}), mean_{v'!=v} softmax(e_{i,v'}) ) ]   # total-variation agreement, in [0,1]
    # TV = total variation distance between per-view part distributions

# Phase 2: temperature-scaled calibration on held-out split, conditioned on coverage/occlusion
fit scalar T > 0 and linear coefficients (beta_a, beta_c, beta_o) on C by minimizing negative log-likelihood of
    z_i = beta_a * (a_i - 0.5) + beta_c * c_i + beta_o * o_i
    p_i = sigmoid( z_i / T )        # calibrator: agreement+coverage+occlusion -> correctness probability
# T, beta_a, beta_c, beta_o are the four calibration hyperparameters; fitted once, frozen at test time
# segments with c_i < 0.2 are marked 'unknown' (p_i = 0.5) and excluded from graph message weights

# Phase 3: reliability-weighted segment graph (SegGraph-style)
build graph G=(S, E_g) with edges from mutual overlap/adjacency (as in SegGraph)
node feature f_i = p_i * (view-fused 2D feature of s_i)
for L=3 rounds of GNN message passing:
    f_i <- f_i + (1/deg(i)) * sum_{j in N(i)} p_j * f_j     # neighbor influence scaled by its reliability
    p_i <- p_i * (1 + alpha * agreement_of(f_i with neighbors))  # alpha=0.5, renormalize to [0,1]

# Phase 4: reliability as refinement evidence (PartAgent-style)
partition <- argmax over part labels of reliability-weighted feature vote per point
loop until no change or max 5 iterations:
    for each region in partition:
        if mean r in region < tau (tau=0.4):  # low-trust region
            inspect: recompute a_i from additional views if available
            repair: reassign region to neighbor part with highest reliability-weighted overlap
            verify: accept change only if it increases sum_i p_i * agreement(region, part)
    update partition

RETURN y(P) = partition labels, r_i = p_i
```

**(C) Why this design.** We chose to calibrate a cross-view agreement statistic via a single temperature on a held-out split rather than to learn a free-form reliability head over features, because a learned head would re-introduce the coupling between feature quality and confidence that the motivation diagnoses, accepting the cost that calibration requires a small labeled calibration split and assumes views are conditionally independent given the part label. We chose to propagate reliability multiplicatively through the GNN message weights (p_j scaling neighbor contributions) rather than to average neighbor features uniformly as SegGraph does, because uniform averaging lets a low-agreement segment dominate a high-agreement neighbor, accepting the cost that a mis-calibrated p_j can over-suppress a genuinely informative but noisy segment. We chose to make refinement acceptance a reliability-weighted agreement test rather than a heuristic reselection as PartAgent does, because the refinement then has an explicit quantity that must increase, accepting the cost that refinement can stall on regions where all views disagree and no repair raises the weighted agreement. A domain expert would not describe CaliSeg as 'SegGraph + PartAgent' because the load-bearing object — a calibrated per-segment correctness probability that is both a graph weight and a refinement acceptance criterion — does not exist in either prior system; it is a new conserved quantity, not a module swap.

**(D) Why it measures what we claim.** The cross-view total-variation agreement a_i measures per-segment semantic reliability because the assumption that distinct views are conditionally independent given the true part label makes disagreement a direct signal of label uncertainty; this assumption fails when views share a systematic viewpoint bias (e.g., a part always occluded from one side), in which case a_i reflects view-coverage rather than label correctness. The temperature-scaled p_i = sigmoid((a_i-0.5)/T) measures calibrated correctness probability because the assumption that a monotone link exists between agreement and correctness (verified by the held-out NLL fit) makes the sigmoid a proper calibrator; this assumption fails when the agreement-correctness relation is non-monotone (high agreement on a confidently wrong label), in which case p_i reflects consensus rather than correctness. The reliability-weighted message weight p_j measures neighbor trust because the assumption that a segment's reliability transfers to its spatial neighbors via the graph edges makes p_j a valid influence coefficient; this assumption fails at part boundaries where adjacent segments belong to different parts, in which case p_j suppresses a correct cross-boundary cue. The refinement acceptance test (increase in sum_i p_i * agreement) measures monotone evidence improvement because the assumption that the weighted agreement is an unbiased estimate of partition quality makes its increase equivalent to refinement progress; this assumption fails when the calibration split distribution differs from test objects, in which case the test certifies improvement under a shifted reliability scale.

## Contribution

['A calibrated per-segment reliability estimator that converts cross-view SAM/2D-expert agreement into a temperature-scaled correctness probability and propagates it as a multiplicative weight through both segment-graph message passing and closed-loop refinement acceptance.', 'A design principle that the same calibrated reliability variable can serve simultaneously as a graph edge weight and as the evidence variable certifying refinement monotonicity, unifying two previously separate propagation and refinement stages under one quantity.', 'An explicit characterization of the conditional-independence and monotonicity assumptions under which agreement-based reliability is valid, with the failure modes (view bias, confident-wrong consensus, boundary leakage) that bound its applicability.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | PartObjaverse-Tiny-WT | Same benchmark as PartAgent; enables direct comparison |
| Dataset | PartNet-E | Same benchmark as SegGraph; tests propagation sub-claim on a second benchmark |
| Primary metric | mIoU | Standard part-segmentation quality; sensitive to boundary errors |
| Secondary metric | Calibration NLL of p_i on held-out test segments | Directly measures whether the reliability is calibrated in the same units as correctness |
| Baseline | PartAgent | Tests whether calibrated reliability beats heuristic refinement |
| Baseline | SegGraph | Tests whether reliability-weighted propagation beats uniform averaging |
| Baseline | Uniform-trust variant of CaliSeg | Isolates the calibrated reliability as the load-bearing component |
| Ablation-of-ours | CaliSeg without calibration (T fixed at 1.0, beta_c=beta_o=0) | Tests whether the NLL-fit temperature and coverage/occlusion conditioning matter |
| Ablation-of-ours | CaliSeg without coverage/occlusion terms (beta_c=beta_o=0, T fitted) | Isolates the conditioning on view coverage and occlusion |
| Ablation-of-ours | CaliSeg with 'unknown' gating disabled (c_i<0.2 segments assigned p_i from sigmoid) | Tests whether marking low-coverage segments as unknown rather than low-trust matters |

### Why this setup validates the claim

The central claim is that a calibrated per-segment correctness probability, propagated as graph weight and used as refinement acceptance, improves part segmentation over both uniform-trust propagation (SegGraph) and heuristic closed-loop refinement (PartAgent). PartObjaverse-Tiny-WT is the right stage because it is the benchmark where PartAgent already reports a strong number, so any gain over it is a clean test of the refinement sub-claim, and it contains the small-part and boundary cases where the reliability mechanism is supposed to matter. PartNet-E is added because it is the benchmark where SegGraph reports its gains, so it provides an independent test of the propagation sub-claim on a different object distribution. SegGraph is the baseline that tests the propagation sub-claim: if uniform neighbor averaging is genuinely the weakness, replacing it with reliability-scaled messages should shift mIoU specifically on segments with heterogeneous view agreement. The uniform-trust CaliSeg variant and the fixed-temperature ablation together isolate the calibration step: if the gain survives with a frozen T but vanishes with uniform trust, the load-bearing object is the reliability weighting; if the gain vanishes only when T is frozen, the calibration fit is doing the work. The calibration NLL metric directly checks the load-bearing assumption: if p_i is calibrated, its NLL on held-out test segments should be low and should not degrade under view-coverage shift; if NLL degrades, the reliability scale is shifted and the refinement acceptance test is certifying improvement under a shifted scale. mIoU is the right metric because it penalizes exactly the under-segmentation and boundary inconsistency that the motivation diagnoses, and it aggregates over the small-part subsets where the predicted effect should concentrate.

### Expected outcome and causal chain

**vs. PartAgent** — On an object where a part is visible from only one side and the 2D expert is confidently wrong on the remaining views, PartAgent's heuristic reselection can accept a region move that raises local semantic score without any quantity tracking whether the supporting evidence is trustworthy, so it locks in a wrong part label on that region. CaliSeg instead assigns that segment a low calibrated reliability because cross-view TV agreement is low and coverage c_i is low, and its acceptance test requires an increase in reliability-weighted agreement, so the low-trust region is repaired toward the neighbor part rather than reselected on raw score. We expect a noticeable mIoU gap concentrated on objects with strongly asymmetric view coverage, with near-parity on well-covered objects.

**vs. SegGraph** — On a case where a small high-agreement segment sits adjacent to a large low-agreement segment, SegGraph's uniform message averaging lets the large noisy neighbor dominate the small segment's fused feature, producing under-segmentation and label bleed across the boundary. CaliSeg scales each neighbor's contribution by its calibrated p_j, so the high-agreement small segment retains its feature identity and the noisy neighbor is down-weighted. We expect the gain to appear as improved mIoU on small components and part boundaries, matching SegGraph's own reported strength areas, with parity on large monolithic parts.

**vs. Uniform-trust variant of CaliSeg** — On an object where view agreement is uniformly high but one segment is confidently wrong across all views, the uniform-trust variant treats every segment as equally reliable, so the confidently-wrong segment propagates its error through the graph and the refinement has no signal to reject it. CaliSeg's calibrated p_i is low for that segment precisely because the agreement-correctness link fitted on the held-out split predicts low correctness despite consensus, so the graph suppresses it and the acceptance test refuses repairs that lean on it. We expect a modest but consistent gap on the confidently-wrong subset, and near-identical results elsewhere.

**vs. CaliSeg without coverage/occlusion conditioning** — On objects where a part is systematically occluded from one side, the unconditioned calibrator assigns low p_i to that segment because agreement is low, even though the segment is correct; the conditioned calibrator uses c_i and o_i to separate low agreement due to occlusion from low agreement due to label uncertainty, so it assigns higher p_i to the correctly occluded segment and lower p_i to the confidently-wrong segment. We expect the conditioned variant to improve mIoU on the occluded-part subset and to reduce calibration NLL on that subset, with parity elsewhere.

### What would falsify this idea

If CaliSeg's gain over SegGraph and PartAgent is uniform across well-covered and poorly-covered objects, and across small-part and large-part subsets, rather than concentrated where cross-view agreement is heterogeneous, then the calibrated reliability is not the causal mechanism and the central claim is wrong. Likewise, if the fixed-temperature ablation matches full CaliSeg, the calibration fit is not load-bearing and the idea reduces to reliability-weighted propagation with an arbitrary monotone link.

## References

1. PartAgent: Agentic 3D Part Segmentation via Expert Integration and Closed-Loop Refinement
