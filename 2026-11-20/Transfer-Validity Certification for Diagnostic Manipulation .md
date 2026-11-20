# Transfer-Validity Certification for Diagnostic Manipulation Benchmarks via Paired Sim-to-Real Task Subsets

## Motivation

Diagnostic manipulation benchmarks such as Mind2Act World (arxiv:2611.26444) decompose failure into Mind, Act, and Mind2Act views, but every view is scored exclusively in simulation, so the benchmark's capability attributions are never checked against the physical execution they claim to diagnose. The structural gap is not a missing task or metric but a missing invariant: no benchmark in this lineage carries a property that ties simulated success rankings to real-robot execution validity, so a system can improve its Mind2Act score while its physical capability attribution is unconstrained. The root cause is that simulation-only evaluation has no observable that can falsify a capability ranking, and the diagnostic harness inherits this blindness because it attributes failures using the same simulated rollouts it is supposed to validate.

## Key Insight

A benchmark's capability decomposition is only evidence-bearing when the simulated view ranking is rank-consistent with physically executed counterparts on a paired task subset, which turns transfer validity from an unverifiable assumption into a per-view invariant that the harness can certify or reject.

## Method

**What it is.** We introduce TVC (Transfer-Validity Certification), a benchmark invariant and harness extension for Mind2Act World. Inputs: the 30-task, 180-configuration simulated suite, a designated paired subset P of tasks instantiated on a physical arm, and a set of evaluated policies. Outputs: a per-view transfer-validity score τ_view and a certification flag that gates whether the harness's Mind/Act/Mind2Act attributions are reported as embodied-capability evidence.

**Load-bearing assumption (stated plainly).** The certification flag is only meaningful if τ_view is a stable, estimable quantity — i.e., if the per-view rank correlation between simulated and physical success rates on P is not dominated by sampling noise from single-task flips. We make this assumption explicit and test it directly: we report the bootstrap distribution of τ_view and require the lower bound to exceed 0.5 before any view can be certified. If the bootstrap interval is wide (e.g., width > 0.4), the view is reported as UNVERIFIED regardless of the point estimate, because the flag would reflect sampling noise rather than transfer validity.

**How it works.**

```pseudocode
# Phase 1: paired-subset construction (design-time, once per benchmark release)
P = select_paired_subset(tasks=30, |P|=10, strata=5 domains x {Mind,Act,Mind2Act})
for t in P:
    for lvl in {1,2,3}:
        for scene in {Clean,Random}:
            assert physical_instantiation_exists(t, lvl, scene)   # reject task if not realizable
            register_equivalence_map(t, lvl, scene)                # object poses, tolerances, success predicate

# Phase 2: paired evaluation (per evaluated policy pi)
for t in P, lvl, scene:
    s_sim[t,lvl,scene] = run_sim(pi, t, lvl, scene, n_rollouts=50)   # success rate
    s_real[t,lvl,scene] = run_physical(pi, t, lvl, scene, n_rollouts=20)

# Phase 3: per-view transfer-validity score
tau_view = spearman_rank_correlation(
    {s_sim[t] for t in P_view}, {s_real[t] for t in P_view}, min_n=10)
bootstrap_CI = bootstrap_interval(tau_view, B=1000, resample_tasks=True)
certified_view = (tau_view >= tau_min=0.7) and (bootstrap_CI.lower > 0.5) and (bootstrap_CI.width <= 0.4)

# Phase 4: harness gating
for attribution a in harness_attributions(pi):
    if not certified_view(a.view):
        a.status = "UNVERIFIED"   # reported but not evidence-bearing
    else:
        a.status = "CERTIFIED"
```

**Why this design.** We chose a stratified non-random paired subset over a random sample because rank correlation is only meaningful if every view is represented with enough tasks to estimate a rank; a random draw could leave Mind2Act with two tasks and produce an unstable τ, accepting the cost that the subset is hand-curated and may not cover all task families. We chose Spearman rank correlation over absolute success-rate agreement because the benchmark's diagnostic claim is ordinal (which system is better at Mind vs Act), not cardinal, accepting the cost that τ is insensitive to uniformly biased success rates. We chose a minimum of 20 physical rollouts per configuration over a larger budget because real-arm time is the binding resource, accepting wider confidence intervals and a conservative certification threshold. We chose a hard certification gate over a soft warning because the point is to make the invariant falsifiable, accepting the cost that some policies will be reported as UNVERIFIED. This differs from a retrieval or descriptor-matching scheme: τ is not a similarity between task descriptions but a rank statistic over two independently executed measurement channels, and it differs from the SAM2Act/MemoryBench line (venue:icml:2025:89787e1cfd56), which introduces new simulated memory tasks but never pairs them with physical counterparts.

**Why it measures what we claim.** The Spearman rank correlation τ_view measures transfer validity of a capability view because of the assumption that rank order of success on the paired subset is preserved under the sim-to-real gap for that view; this assumption fails when the physical instantiation introduces a confound absent in simulation (e.g., a perception failure on a textured object), in which case τ reflects the confound rather than the view's intrinsic transferability. The certification threshold τ_min=0.7 with a bootstrap lower bound measures evidence-bearing status because of the assumption that a policy ordering stable at τ≥0.7 is stable under resampling of the paired subset; this assumption fails when the subset is too small for the bootstrap to be informative, in which case the flag reflects sampling noise rather than genuine validity. The gating of harness attributions measures the claim that attributions are embodied-capability evidence because of the assumption that the harness's simulated failure attribution mechanism is itself view-consistent with the physical runs; this assumption fails when the harness attributes failures using simulation-only features, in which case the UNVERIFIED flag correctly reflects an unvalidated attribution rather than a validated one.

## Contribution

['A transfer-validity certification protocol (TVC) that defines a per-view rank-correlation invariant over a stratified paired sim-to-real task subset and gates diagnostic harness attributions on its satisfaction.', 'A concrete paired-subset construction procedure with equivalence mapping (object poses, tolerances, success predicates) that makes the 180-configuration Mind2Act World suite physically instantiable on a real arm for a designated subset.', 'An augmented harness reporting layer that labels each Mind/Act/Mind2Act attribution as CERTIFIED or UNVERIFIED, turning benchmark scores into conditional evidence.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | Mind2Act World paired subset P (10 tasks) | Stratified across 5 domains and 3 views |
| Primary metric | Per-view Spearman τ (sim vs. physical) | Ordinal diagnostic claim, not cardinal agreement |
| Baseline | Unpaired sim-only attribution reporting | Tests whether pairing is needed at all |
| Baseline | SAM2Act/MemoryBench-style simulated benchmark | Tests view-level transfer without physical counterpart |
| Baseline | Random-subset τ (no stratification) | Tests whether stratification stabilizes τ |
| Ablation-of-ours | TVC without hard certification gate | Tests whether gating changes reported evidence |

### Why this setup validates the claim

The central claim is that Mind2Act World's Mind/Act/Mind2Act attributions only count as embodied-capability evidence when a per-view rank order survives a sim-to-real paired check. The unpaired sim-only baseline tests the sub-claim that pairing is necessary: if reporting simulated attributions alone already predicts physical ordering, τ adds nothing. The SAM2Act/MemoryBench-style baseline tests the sub-claim that a view-specific transfer statistic is meaningful at all, since that line introduces simulated memory tasks without physical counterparts and thus cannot produce τ. The random-subset baseline isolates the design choice of stratification: if τ computed on a random draw is as stable as on the stratified P, our curation rationale is wrong. The no-gate ablation isolates the gating mechanism: if UNVERIFIED and CERTIFIED attributions are equally predictive of physical failures, the gate is decorative. Spearman τ is the right metric because the benchmark's diagnostic claim is ordinal — which system is better at Mind versus Act — so absolute success-rate agreement would certify policies that are uniformly biased but correctly ordered, and would reject correctly calibrated but differently scaled ones. To ground the τ_min=0.7 threshold and the 20-rollout budget, we include a pilot study on 2–3 paired tasks with 1 policy, reporting measured sim-real rank agreement and per-configuration variance; the pilot informs whether the bootstrap interval width criterion (≤0.4) is achievable at |P|=10.

### Expected outcome and causal chain

**vs. Unpaired sim-only attribution reporting** — On a case where a policy fails Mind2Act tasks in simulation due to a reasoning-execution composition error, the baseline reports that attribution as embodied evidence because it never measures the physical channel. Our method instead computes τ over the paired subset for the Mind2Act view and gates the attribution, because the rank order must be reproduced on the physical arm before it counts. We expect the baseline to show attributions that are confidently wrong on the physical arm, while our method flags a noticeable fraction of Mind2Act attributions as UNVERIFIED, with the gap concentrated on the composition view rather than uniform across views.

**vs. SAM2Act/MemoryBench-style simulated benchmark** — On a memory-dependent manipulation instance, the baseline reports high simulated success and treats that as capability evidence because its benchmark has no physical instantiation of the same task. Our method instead registers an equivalence map for the paired task and runs 20 physical rollouts, because spatial-memory success in simulation can hinge on simulator-specific object permanence cues absent on the real arm. We expect the baseline's simulated ordering to diverge from physical ordering on memory-heavy tasks, producing a low τ_view for that view and a certified-versus-uncertified split that tracks memory dependence rather than task difficulty.

**vs. Random-subset τ (no stratification)** — On a draw that leaves Mind2Act with two or three tasks, the baseline produces an unstable τ whose bootstrap interval straddles the threshold, because rank correlation over a handful of tasks is dominated by single-task flips. Our method instead stratifies P across five domains and three views so every view has enough tasks to estimate a rank, because the certification decision depends on the lower bootstrap bound. We expect the random-subset τ to fluctuate across draws while the stratified τ stays near the threshold, with the certification flag flipping under resampling for the random subset but not for P.

### What would falsify this idea

If τ_view computed on the paired subset fails to predict which harness attributions are borne out on the physical arm — that is, if UNVERIFIED and CERTIFIED attributions are equally likely to match physical failure modes — then the certification gate is not measuring transfer validity. Likewise, if the stratified-subset τ is no more stable than a random-subset τ, the claimed benefit of curation is absent.

## References

1. Mind2Act World: A Diagnostic Benchmark for Embodied Mind, Physical Action, and Mind-Action Composition in Robotic Manipulation
