# Monotone Partition-Energy Control for Agentic 3D Part Segmentation

## Motivation

Agentic refinement loops for 3D part segmentation, such as PartAgent (arXiv:2611.32554), select inspection, repair, composition, or reselection actions from observations without any invariant that each accepted change improves the partition, so the loop can cycle, oscillate between granularities, or silently degrade part-boundary consistency and intra-part semantic coherence. The root cause is structural: the action selector is a free-running policy over a state space of partitions with no scalar potential tying proposed changes to a verifiable non-degradation criterion; SegGraph (NeurIPS 2025) likewise assumes a single forward pass with no convergence criterion, and the evolution to an agentic loop added action repertoire but no admission test. Across the paper-tree this recurs as a missing monotone potential in iterative partition refinement, not a missing expert or a missing action.

## Key Insight

If every admissible action must strictly decrease or preserve a single partition-energy functional that decomposes into boundary-consistency and intra-part-coherence terms, then the accepted-partition ledger is a monotone sequence in a finite partition lattice, so cycling and degradation are excluded by construction rather than by heuristic stopping.

## Method

**What it is.** We introduce MECo (Monotone Energy Controller), a training-free admission layer wrapped around PartAgent's closed-loop refinement. Its inputs are the current partition P_t over surface points, the expert evidence state (3D region proposals, multiview 2D semantic maps, image-to-surface correspondences), and a candidate action a from {inspect, repair, compose, reselect}; its output is either an accepted successor partition P_{t+1} recorded in a ledger L, or a rejection that forces the next action, terminating when no candidate yields verified non-degradation.

**Load-bearing assumption (stated explicitly).** The scalar partition energy E(P) = 0.5*B(P) + 0.5*S(P), computed from a frozen expert-feature cache on a fixed surface graph, is a valid global potential whose non-increase is causally aligned with mIoU improvement, so that greedily descending it via argmin over admitted successors monotonically improves actual segmentation quality rather than merely optimizing a proxy. This assumption is load-bearing because the admission rule is only as good as the potential it descends. To keep the assumption from being violated by the degenerate single-part solution, both energy terms are volume-normalized (Ncut-style for B, Davies-Bouldin-style ratio for S) so that merging distinct parts increases energy; and the expert feature cache is frozen at P_0 and inspect actions are forbidden from mutating it within a run, so the potential is computed on a genuinely fixed graph and feature set.

**How it works.**

```pseudocode
Input: initial partition P_0 from Structural-Semantic Expert Integration
       surface graph G=(V,E_adj) with edge weights w_uv = 1 / geodesic(u,v)
       frozen expert-feature cache F_0 = {f_2d(v), f_3d(v) for v in V}  # frozen at P_0
       energy E(P) = lambda_b * B(P) + lambda_s * S(P)   # lambda_b=0.5, lambda_s=0.5
       tolerance eps = 1e-4, max_steps T = 20
Ledger L = [(P_0, E(P_0), null)]
for t in 0..T-1:
    obs = Observe(P_t, evidence_state)          # PartAgent observation module
    candidates = ProposeActions(obs)             # {inspect, repair, compose, reselect}
    admitted = []
    for a in candidates:
        P' = Simulate(a, P_t, evidence_state)    # PartAgent's verifier produces P'
        if P' is None or not IsValidPartition(P'): continue
        delta = E(P') - E(P_t)
        if delta <= eps:                          # non-degradation, eps absorbs float noise
            admitted.append((a, P', delta))
    if admitted is empty: break                   # certified local termination
    (a*, P*, _) = argmin over admitted of E(P*)  # strictly best admissible successor
    L.append((P*, E(P*), a*)); P_t = P*
return L, P_t

# Energy terms, computed on the surface graph G=(V,E_adj) with frozen features F_0:
# B(P) = Ncut-style normalized boundary cut:
#        B(P) = sum_{k} cut(P_k, V\P_k) / vol(P_k)
#        where cut(P_k, V\P_k) = sum_{u in P_k, v not in P_k, (u,v) in E_adj} w_uv
#              vol(P_k) = sum_{u in P_k} sum_{v: (u,v) in E_adj} w_uv
#        (so the all-one-label partition has B=0 only if the graph is disconnected;
#         for a connected surface graph, merging distinct parts increases B)
# S(P) = Davies-Bouldin-style dispersion-to-separation ratio:
#        S(P) = (1/|P|) * sum_{k} max_{l != k} [ (disp(P_k) + disp(P_l)) / sep(P_k, P_l) ]
#        where disp(P_k) = mean_{v in P_k} (1 - cos(f_2d(v), mu_2d(P_k)))
#                        + mean_{v in P_k} (1 - cos(f_3d(v), mu_3d(P_k)))
#              mu_2d(P_k), mu_3d(P_k) = mean 2D/3D expert feature in part k
#              sep(P_k, P_l) = 1 - cos(mu_2d(P_k), mu_2d(P_l))
#                            + 1 - cos(mu_3d(P_k), mu_3d(P_l))
#        (so merging two distinct parts increases S because sep drops toward 0)
```

The controller never re-runs the experts; it only re-uses the evidence state PartAgent already maintains, and the feature cache F_0 is frozen at P_0. The ledger L is the audit trail: each entry stores the accepted partition, its energy, and the action that produced it, so any later inspection can replay the sequence.

**Why this design.** We chose a single scalar energy with fixed convex weights (lambda_b = lambda_s = 0.5) over a lexicographic or Pareto rule because a scalar potential is what makes the admission test a total order on candidate successors, accepting the cost that a genuinely better partition on one term but worse on the other is rejected; the ledger records the rejected delta so a human can audit the trade-off. We chose strict non-degradation with a tolerance eps = 1e-4 rather than strict decrease because segmentations have ties (two actions may produce identical partitions), and requiring strict decrease would terminate prematurely on plateaus, accepting the cost that eps must be set above numerical noise in the feature-similarity terms. We chose argmin over admitted successors rather than first-accepted, because first-accepted makes the trajectory order-dependent and lets a weak action block a stronger one, accepting the cost of simulating all candidates per step (bounded by |actions| = 4). We chose a fixed step budget T = 20 and termination on empty admitted set rather than a learned stopping head because the whole point is a verifiable criterion, accepting the cost that pathological evidence states may exhaust the budget before reaching a fixed point. We chose volume-normalized B (Ncut-style) and dispersion-to-separation S (Davies-Bouldin-style) rather than unnormalized cut and dispersion because the unnormalized forms are jointly minimized by the degenerate single-part partition, which would make argmin-descent monotone toward under-segmentation; the normalized forms make merging distinct parts increase energy, so the descent direction is aligned with the motivation-level objective. We freeze the expert feature cache at P_0 and forbid inspect actions from mutating it within a run so that the potential is computed on a genuinely fixed graph and feature set, making the ledger's monotonicity well-defined per run. A domain expert would not describe MECo as a variant of PartAgent's observation-driven refinement: PartAgent selects actions by observation and verifies them locally, whereas MECo changes the semantics of 'verified' from 'the action executed' to 'the action is admissible under a global potential', and the matched prior method (PartAgent) has no potential at all.

**Why it measures what we claim.** The boundary-cut term B(P) measures part-boundary consistency because it equals the volume-normalized total weight of surface-graph edges crossing part labels, an assumption that adjacent surface points with different labels are exactly the boundary and that edge weights w_uv (inverse geodesic distance) reflect true geometric adjacency; this assumption fails when the surface graph is noisy or the 3D expert's region proposals are over-segmented, in which case B(P) reflects proposal granularity rather than boundary quality. The intra-part feature-dispersion term S(P) measures intra-part semantic coherence because the Davies-Bouldin-style ratio of within-part cosine dispersion to between-part separation equals semantic variance relative to inter-part distinctness under the assumption that the expert feature spaces are locally metric-consistent with part identity (view-invariant and scale-calibrated); this assumption fails when the 2D expert's multiview features are view-inconsistent or the 3D proposals are over-segmented, in which case S(P) reflects view disagreement or proposal granularity rather than semantic incoherence. The admission rule delta <= eps measures non-degradation of the motivation-level objective because a monotone sequence in a finite partition lattice with a bounded energy cannot cycle, under the assumption that the energy is computed on a fixed surface graph and a frozen expert feature cache within one run; this assumption fails if the evidence state is mutated by an inspect action in a way that changes the feature cache, in which case the ledger's monotonicity is with respect to a drifting energy and must be recomputed on the stored features — which is why we freeze the cache at P_0 and forbid inspect actions from mutating it within a run.

## Contribution

['A monotone admission controller (MECo) for agentic 3D part segmentation that wraps any action-proposing refinement loop with a partition-energy functional and an accepted-partition ledger, converting free-running action selection into a non-degradation-filtered search.', 'A design principle: a two-term partition energy (surface-graph boundary cut plus expert-feature dispersion) is sufficient to make action admission order-robust and to give a replayable audit trail, without retraining or modifying the underlying experts.', 'An analysis protocol over the ledger that separates actions which improve boundary consistency from those which improve semantic coherence, enabling diagnosis of which expert (3D proposal vs 2D semantic) is the bottleneck in a given scene.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | PartObjaverse-Tiny-WT | Same benchmark as PartAgent; enables direct comparison |
| Primary metric | mIoU | Part-level overlap; sensitive to boundary and label errors |
| Baseline | PartAgent (no admission control) | Tests whether monotone admission itself adds value |
| Baseline | First-accepted refinement | Tests whether argmin selection beats order-dependent acceptance |
| Baseline | Strict-decrease admission (eps=0) | Tests whether tolerance prevents premature plateau termination |
| Ablation-of-ours | MECo with boundary term only (lambda_s=0) | Isolates contribution of the semantic dispersion term |
| Ablation-of-ours | MECo with unnormalized B and S (original unnormalized forms) | Tests whether volume normalization prevents degenerate under-segmentation |
| Calibration experiment | Correlation of B(P) and S(P) with ground-truth boundary and part-identity signals on a 512-instance calibration split of PartObjaverse-Tiny-WT | Validates that the energy terms measure what they claim (boundary consistency and intra-part coherence) rather than a drifting proxy |
| Overhead measurement | Wall-clock per instance and number of expert-free simulations per step, against PartAgent | Confirms argmin-over-all-candidates cost is acceptable |

### Why this setup validates the claim

The central claim is that replacing PartAgent's local "action executed" verification with a global monotone potential admission rule yields strictly better partitions without any expert retraining. PartObjaverse-Tiny-WT is the right dataset because it is exactly where PartAgent already reports strong numbers (68.41 mIoU), so any gain must come from refinement quality rather than from a weaker baseline or an easier domain. The unmodified PartAgent baseline tests the core sub-claim: that a global potential is doing work that observation-driven local verification does not. The first-accepted and strict-decrease baselines are diagnostic controls that isolate two specific design decisions (argmin selection and eps tolerance) rather than the whole controller; if MECo only beats PartAgent but loses to first-accepted, the claim about argmin is falsified even though the headline number improves. The boundary-only ablation tests whether the semantic dispersion term S(P) contributes anything or whether boundary cut alone explains the gain, which is the difference between "a good energy" and "the energy we claimed." The unnormalized-energy ablation tests the load-bearing assumption directly: if the unnormalized form collapses toward the degenerate single-part partition (visible as a drop in mIoU and an increase in under-segmentation rate), then volume normalization is necessary and the normalized energy is the one that aligns with mIoU. The calibration experiment on a 512-instance split tests the proxy-validity assumption: we compute the Spearman correlation between B(P) and the ground-truth boundary F1 (or boundary distance) and between S(P) and ground-truth part-identity agreement (e.g., purity of predicted parts with respect to ground-truth part labels), and require positive correlation for the energy to be a valid proxy. The overhead measurement confirms the practical cost of simulating all four actions per step is acceptable. mIoU is the correct primary metric because the predicted effect is concentrated on boundary regions and small parts, both of which mIoU penalizes directly; a metric like Chamfer distance would be blind to label swap errors that the controller is specifically designed to reject.

### Expected outcome and causal chain

**vs. PartAgent (no admission control)** — On a case where the 3D expert over-segments a thin handle into two proposals and the 2D expert's multiview evidence is ambiguous, PartAgent executes the repair action because its verifier only checks that the action ran, producing a partition with a spurious cut across the handle. Our method instead simulates the repair, computes the normalized boundary-cut increase on the surface graph, and rejects the successor because delta > eps, then tries compose or reselect. We expect a noticeable mIoU gap concentrated on thin/small-part subsets and near parity on large convex parts, since the failure mode we correct is boundary-localized.

**vs. First-accepted refinement** — On a case where an early inspect action yields a plateau successor (identical partition) and a later compose action would have merged two correctly-adjacent proposals, first-accepted absorbs the plateau step and burns budget, sometimes exhausting T before reaching the merge. Our method instead admits the plateau (delta <= eps) but selects argmin over all admitted successors, so the compose candidate is chosen whenever it lowers energy more. We expect the gap to widen with the number of candidate actions per step and to show up as higher variance in first-accepted's final energy across instances with identical evidence states.

**vs. Strict-decrease admission (eps=0)** — On a case where two valid actions produce numerically identical partitions differing only in float noise of the feature-similarity term, strict decrease rejects both and terminates the loop early, freezing a suboptimal partition. Our method instead admits ties under eps=1e-4 and continues, allowing a subsequent action to escape the plateau. We expect strict-decrease to terminate in fewer steps and to underperform specifically on instances whose optimal partition requires crossing a plateau, giving a gap that correlates with the number of plateau transitions in the ledger.

**vs. MECo with unnormalized B and S** — On a case where the initial partition is over-segmented, the unnormalized energy is minimized by merging all parts into one, so argmin-descent drives the controller toward the degenerate single-part partition. Our method instead uses volume-normalized B and Davies-Bouldin-style S, so merging distinct parts increases energy and the controller rejects the degenerate merge. We expect the unnormalized variant to show a much higher under-segmentation rate and lower mIoU, confirming that normalization is necessary for the energy to align with mIoU.

**Calibration experiment** — We expect B(P) to have a positive Spearman correlation (target > 0.3) with ground-truth boundary F1 and S(P) to have a positive Spearman correlation (target > 0.3) with ground-truth part-identity purity on the 512-instance calibration split. If these correlations are near zero or negative, the energy terms are not measuring the claimed quantities and the proxy-validity assumption fails.

**Overhead measurement** — We expect MECo's wall-clock per instance to be at most 1.5× PartAgent's, with the number of expert-free simulations per step bounded by |actions| = 4. If overhead exceeds 2×, the argmin-over-all-candidates design is impractical and the first-accepted variant would be preferable despite its order-dependence.

### What would falsify this idea

If MECo's gain over PartAgent is uniform across small-part, boundary, and large-convex subsets rather than concentrated on the boundary and small-part subsets where the potential is designed to bite, then the improvement is not coming from the admission mechanism and the central claim is wrong. Equally fatal: if the boundary-only ablation matches full MECo, the semantic dispersion term S(P) is not contributing and the two-term energy claim is unsupported; if the unnormalized-energy ablation matches full MECo, the normalization is not doing work and the load-bearing assumption about avoiding the degenerate partition is unsupported; and if the calibration correlations are near zero, the energy terms are not valid proxies for boundary consistency and intra-part coherence.

## References

1. PartAgent: Agentic 3D Part Segmentation via Expert Integration and Closed-Loop Refinement
