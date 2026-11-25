# Nested Part Partitions: Refinement-Closed Multi-Granularity 3D Part Segmentation

## Motivation

Multi-granularity part segmentation methods construct partitions at several scales but provide no guarantee that these partitions are mutually consistent: a 'wheel' at fine granularity may straddle two coarse parts, or two fine parts sharing a coarse label may receive contradictory labels. PartAgent obtains its multi-granularity partitions by heuristic expert fusion, so coarser and finer partitions are produced independently and can violate the refinement relation; SegGraph builds only a single segment graph at one SAM granularity and therefore cannot even express cross-granularity consistency. The root cause is that granularity is treated as a set of independent fusion runs rather than as a lattice in which finer partitions must be refinements of coarser ones.

## Key Insight

If every multi-granularity partition is constrained to lie in a refinement lattice — each fine part contained in exactly one coarse part with label consistency under the refinement map — then cross-granularity inconsistency becomes a detectable, repairable violation of a closure invariant rather than an unobservable fusion artifact, and the invariant is checkable purely from partition geometry without any extra learned model.

## Method

**(A) What it is.** NestedPart is a training-free, invariant-constrained composition layer that wraps PartAgent's expert fusion. Its inputs are a set of candidate partitions at granularities g_1 < g_2 < ... < g_K (coarse to fine) produced by the 3D expert's region proposals fused with 2D multiview semantic evidence, plus per-region evidence scores. Its outputs are a refinement-closed partition family, a log of repairs performed, and a first-class attribution report that separates violations caused by fine-part inconsistency from violations caused by coarse-proposal noise. **Load-bearing assumption (stated explicitly):** the coarse and fine candidate partitions are noisy versions of a common underlying part hierarchy, so that a fine region's containment ratio rho(f) and parent-label agreement measure cross-granularity inconsistency rather than the coarse expert's own proposal error or a vocabulary mismatch between granularities. NestedPart does not assume this holds; it estimates per-region reliability and attributes each violation to the higher-confidence side before repairing, and it reports the fraction of violations attributed to coarse-proposal noise so that the invariant's diagnostic claim is separable from the repair's accuracy claim.

**(B) How it works.**

```pseudocode
INPUT: candidate partitions P_1..P_K (coarse→fine), regions r with
       evidence score s(r) in [0,1] (PartAgent's 2D multiview agreement
       score for r), overlap graph O (fraction of shared surface area),
       cross-granularity label map M: labels(P_{k-1}) -> labels(P_k)
       (identity on shared vocabulary; pairs absent from M are not
       checked for LABEL_CONFLICT)
PARAM: containment threshold tau = 0.85, max repair rounds R = 3,
       coarse-reliability margin delta = 0.10

# Phase 0: region representation and reliability
# Regions are point sets with noisy boundaries. For fine region f and
# coarse region c, |f ∩ c| is the surface-area-weighted count of f's
# points lying in c (each point weighted by its Voronoi surface area),
# and |f| is the total surface area of f. This makes rho(f) robust to
# boundary jitter.
# Coarse-region reliability rel(c) is estimated as the mean evidence
# score of c's constituent fine regions that are contained in c with
# rho > tau, i.e. rel(c) = mean_{f in P_k : parent(f)=c, rho(f)>tau} s(f);
# if no such f exists, rel(c) = s(c).

# Phase 1: build refinement relation
for k = 2..K:
    for each fine region f in P_k:
        parent(f) = argmax_{c in P_{k-1}} |f ∩ c| / |f|
        rho(f) = |f ∩ parent(f)| / |f|        # containment ratio
        rel(parent(f)) = mean_{f' in P_k : parent(f')=parent(f),
                              rho(f')>tau} s(f')   # coarse reliability

# Phase 2: detect and attribute closure violations
violations = []
for each f in P_k, k>=2:
    if rho(f) < tau:                       # fine part straddles coarse parts
        # two-sided attribution: which side is higher-confidence?
        if s(f) >= rel(parent(f)) + delta:
            violations.append((STRADDLE_FINE, f))      # fine side is reliable
        elif rel(parent(f)) >= s(f) + delta:
            violations.append((STRADDLE_COARSE, f))    # coarse side is reliable
        else:
            violations.append((STRADDLE_AMBIGUOUS, f)) # neither dominates
    if label(f) != M(label(parent(f))) and s(f) > s(parent(f)):
        violations.append((LABEL_CONFLICT, f, parent(f)))
    # sibling merge check: two fine parts with same parent and same label
    if exists f' != f with parent(f')==parent(f) and label(f')==label(f)
       and O(f,f') > 1 - tau:
        violations.append((REDUNDANT_SPLIT, f, f'))

# Phase 3: targeted repair (reuses PartAgent action vocabulary)
for round in 1..R:
    for v in violations sorted by severity = |1 - rho| + evidence gap:
        if STRADDLE_FINE:   # fine side reliable -> repair upward
            reselect parent(f) as the coarse region maximizing
            |f ∩ c|/|f| among candidates with s(c) >= s(parent(f));
            if none, demote f to P_{k-1} (merge into parent)
        if STRADDLE_COARSE: # coarse side reliable -> repair downward
            split parent(f) along the boundary induced by f's points
            (assign each point of parent(f) to the fine region whose
            parent it is, or to a new coarse region if unassigned)
        if STRADDLE_AMBIGUOUS:
            leave f unchanged and log it as an unresolved coarse-proposal
            defect (counted in the attribution report, not repaired)
        if LABEL_CONFLICT:  # only for pairs present in M
            reassign label of the lower-evidence region to the
            higher-evidence one (evidence repair)
        if REDUNDANT_SPLIT:
            compose f and f' into one region (composition)
    recompute rho, rel, and violations; break if empty

OUTPUT: refinement-closed partitions P_1..P_K, repair log,
        attribution report: fraction of violations by type
        (STRADDLE_FINE / STRADDLE_COARSE / STRADDLE_AMBIGUOUS /
        LABEL_CONFLICT / REDUNDANT_SPLIT) and fraction of
        coarse-proposal defects (STRADDLE_COARSE + STRADDLE_AMBIGUOUS)
```

**(C) Why this design.** We chose a hard containment threshold tau=0.85 over a soft consistency penalty because the refinement relation is a discrete lattice property — a fine part either has a unique parent or it does not — and a soft penalty would let PartAgent's fusion drift into inconsistent states, accepting the cost that legitimate boundary regions near part junctions are occasionally demoted. We chose targeted reselection/merge repairs over global re-fusion because global re-fusion is exactly the heuristic PartAgent already performs and is what produces violations, accepting the cost that some violations require multiple rounds. We chose to reuse PartAgent's existing action vocabulary (inspection, evidence repair, composition, reselection) rather than introduce a new learned repair policy, because the invariant itself specifies which action applies to which violation type, accepting the cost that repair quality is bounded by the pretrained experts' proposal set. We added the two-sided attribution step (Phase 2) and the coarse-reliability estimate rel(c) because the load-bearing assumption — that coarse and fine partitions are noisy versions of a common hierarchy — is not guaranteed by PartAgent's independent per-granularity fusion; without attribution, a STRADDLE violation dominated by coarse-proposal error would be repaired in the wrong direction (demoting a correct fine region). The attribution step does not add a new mechanism; it routes the existing repair actions to the higher-confidence side and reports the residual as a coarse-proposal defect. A domain expert would not describe NestedPart as a variant of PartAgent: PartAgent fuses experts to *produce* partitions, whereas NestedPart imposes a lattice closure *constraint* on whatever partitions are produced and can reject PartAgent's own output.

**(D) Why it measures what we claim.** The containment ratio rho(f) measures cross-granularity consistency because the refinement relation is equivalent to rho(f)=1 for every fine part under the assumption that the coarse partition is a true coarsening of the fine one; this assumption fails when the coarse expert's proposals are themselves noisy, in which case rho reflects coarse-proposal quality rather than fine-part inconsistency — which is why Phase 2 attributes each STRADDLE to the higher-confidence side using rel(parent(f)) and reports STRADDLE_COARSE and STRADDLE_AMBIGUOUS separately from STRADDLE_FINE. The label conflict count measures semantic consistency across granularities because label agreement between a fine part and its parent is equivalent to label consistency under refinement when labels are inherited top-down; this assumption fails when the fine and coarse label vocabularies differ, in which case the count reflects vocabulary mismatch — which is why LABEL_CONFLICT is checked only for label pairs present in the cross-granularity map M. The redundant-split overlap O(f,f') measures whether two fine parts should have been one because high surface overlap between same-parent same-label siblings is equivalent to a spurious split when the 3D expert over-segments; this assumption fails when two genuinely distinct parts are adjacent, in which case O reflects adjacency rather than redundancy. The attribution report measures whether closure violations are caused by fine-part inconsistency or by coarse-proposal noise because it partitions violations by the higher-confidence side; this assumption fails when both sides have similar evidence (STRADDLE_AMBIGUOUS), in which case the report reflects irreducible proposal ambiguity rather than a repairable inconsistency.

## Contribution

['A refinement-closure invariant (nesting/containment with label consistency) for multi-granularity 3D part partitions, together with a constrained composition operator that rejects and repairs expert-fusion compositions violating the relation via targeted reselection, composition, and evidence repair.', 'A violation taxonomy (straddle, label conflict, redundant split) with a severity ordering that makes cross-granularity inconsistency detectable and repairable without any additional learned model, contrasting with heuristic fusion that cannot express such inconsistencies.', "A training-free integration into PartAgent's closed-loop refinement that preserves its action vocabulary while adding a lattice-level acceptance criterion for partition families."]

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | PartObjaverse-Tiny-WT | Same benchmark as PartAgent; permits direct comparison |
| Primary metric | mIoU | Standard part-segmentation accuracy; sensitive to both under- and over-segmentation |
| Secondary metric | Violation rate (violations per object) and attribution fractions (STRADDLE_FINE / STRADDLE_COARSE / STRADDLE_AMBIGUOUS / LABEL_CONFLICT / REDUNDANT_SPLIT) | Directly measures the closure invariant and separates diagnostic from repair claims |
| Baseline | PartAgent (unconstrained) | Tests whether lattice closure adds value over raw expert fusion |
| Baseline | PartAgent + global re-fusion | Tests whether targeted repair beats the heuristic that produces violations |
| Baseline | PartGen | Tests against an independent multi-view part pipeline |
| Ablation-of-ours | NestedPart w/o closure (tau=0) | Isolates the contribution of the invariant constraint |
| Ablation-of-ours | NestedPart w/o attribution (single-sided STRADDLE repair, no rel(c)) | Isolates the contribution of the two-sided attribution step |
| Stratification | Objects binned by violation count (low/mid/high) and by proposal-noise level (low/high, measured as mean evidence score of coarse regions) | Tests whether mIoU gain tracks violation count while controlling for proposal noise |

### Why this setup validates the claim

The central claim is that imposing a refinement-lattice closure on PartAgent's expert-fused partitions reduces cross-granularity inconsistency and thereby improves part segmentation, and that targeted repair outperforms the global re-fusion heuristic PartAgent already performs. PartAgent (unconstrained) is the direct test of the first sub-claim: if closure helps, our output should beat PartAgent's own on the same benchmark, and the gain should be concentrated where closure violations occur. PartAgent + global re-fusion tests the second sub-claim: it is the exact heuristic our design rejects, so if targeted repair is genuinely better, our method should dominate it specifically on objects with many violations rather than uniformly. PartGen provides an external sanity check that gains are not an artifact of sharing PartAgent's expert stack. The ablation with tau=0 removes the constraint while keeping the repair machinery, so any remaining gain must come from repair mechanics rather than closure; if the ablation matches full NestedPart, the invariant is doing no work. The attribution ablation (single-sided STRADDLE repair) tests whether the two-sided attribution step is necessary: if full NestedPart beats it, the load-bearing assumption is not automatically satisfied and attribution is doing measurable work. The stratified analysis by violation count and proposal-noise level tests the missing equivalence that violation-count reduction measures segmentation-error reduction: the assumption is that closure violations are the dominant contributor to mIoU loss, not a byproduct of noisy proposals; the failure mode is that high-violation objects also have high proposal noise, so repair cannot recover. If mIoU gain tracks violation count within each proposal-noise bin, the equivalence holds; if gain is flat or tracks proposal noise instead, it fails. mIoU is the right primary metric because closure violations manifest as both under-segmentation (merged straddling parts) and over-segmentation (redundant splits), and mIoU penalizes both, unlike purity or boundary-F1 which each capture only one side.

### Expected outcome and causal chain

**vs. PartAgent (unconstrained)** — On an object where the coarse expert merges two limbs into one proposal while the fine expert splits them correctly, PartAgent's fusion accepts the coarse merge because it has no mechanism to check that the fine partition refines the coarse one, producing a straddling fine part with low containment. Our method detects rho(f) < tau, attributes the violation to the fine side (s(f) >= rel(parent(f)) + delta), reselects a higher-evidence parent or demotes the fine part, and restores the lattice, so we expect a noticeable mIoU gap concentrated on objects with high STRADDLE_FINE counts and near-parity on objects where PartAgent already produces closed partitions. On objects where the coarse side is the reliable one (STRADDLE_COARSE), our method repairs upward (splits the coarse region) rather than demoting the fine region, so we expect a smaller but positive gap there as well; on STRADDLE_AMBIGUOUS objects we expect near-parity because the violation is logged as an unresolved coarse-proposal defect.

**vs. PartAgent + global re-fusion** — On an object with a single localized label conflict between a fine part and its parent, global re-fusion perturbs the entire partition, degrading regions that were already correct because it re-runs the fusion heuristic everywhere rather than at the violation. Our method applies a single evidence-repair action to the conflicted region and leaves the rest untouched, so we expect our gain over this baseline to scale with the fraction of an object's regions that are already correct — largest on objects with few, isolated violations and shrinking on globally noisy objects. The attribution report lets us check this directly: on objects where most violations are STRADDLE_FINE or LABEL_CONFLICT (repairable), our gain over global re-fusion should be largest; on objects dominated by STRADDLE_COARSE or STRADDLE_AMBIGUOUS, the gain should shrink because the repair is bounded by coarse-proposal quality.

**vs. PartGen** — On an object with occluded parts, PartGen's multi-view diffusion must hallucinate or complete invisible regions and can produce view-inconsistent part boundaries, because its segmentation comes from generative priors rather than grounded expert proposals. Our method inherits PartAgent's spatially grounded 3D proposals and only repairs their cross-granularity consistency, so we expect higher mIoU on occlusion-heavy objects and rough parity on fully visible ones, with the gap tracking occlusion severity.

**vs. NestedPart w/o attribution** — On an object where the coarse expert is noisy and the fine expert is correct, the single-sided ablation demotes the correct fine region (because it only checks rho < tau and always repairs downward), producing a wrong fine label. Our full method attributes the violation to the coarse side (rel(parent(f)) >= s(f) + delta) and splits the coarse region instead, so we expect full NestedPart to beat the ablation specifically on objects with high STRADDLE_COARSE counts, with near-parity on objects where STRADDLE_FINE dominates.

### What would falsify this idea

If the gain over unconstrained PartAgent is uniform across all objects rather than concentrated on objects with high closure-violation counts, or if the stratified analysis shows mIoU gain tracks proposal-noise level rather than violation count within each noise bin, the closure constraint is not the causal driver and the improvement must come from something else. Likewise, if the tau=0 ablation matches full NestedPart, the invariant is doing no measurable work, and if the attribution ablation matches full NestedPart, the two-sided attribution step is not necessary and the load-bearing assumption is automatically satisfied on this benchmark.

## References

1. PartAgent: Agentic 3D Part Segmentation via Expert Integration and Closed-Loop Refinement
