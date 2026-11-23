# Closure-Certified Repair for Recursive Aerial Vision-Language Navigation

## Motivation

Recursive self-improvement frameworks such as AeroRSI verify each candidate repair only by continuing flight from the repaired state, so a repair is certified against the single continuation it was sampled from. This is a local, context-bound criterion: a repair that succeeds in isolation can silently invalidate a longer multi-stage plan once it is substituted into a different sub-task slot. The same residual gap recurs across the VLN literature — LH-VLN's MGDM (E1, E7) provides no cross-subtask decision-consistency guarantee, GSA-VLN (E2, E12) adapts per-scene without certifying that adapted behavior composes, and OpenUAV (E3, E10) supplies long-horizon aerial trajectories but no composition-closure criterion — so recursive self-improvement inherits a verification criterion that is structurally blind to sub-task composition.

## Key Insight

A repair is sound for recursive self-improvement exactly when it is a fixed point of a sub-task substitution algebra — its continuation-validity is preserved under re-embedding into every longer plan in a compositional closure set — and this closure property is checkable by re-running continuation verification on substituted plans rather than by any new learned signal.

## Method

**What it is.** We introduce **CCR (Closure-Certified Repair)**, a repair operator for AeroRSI-style recursive self-improvement. Its input is a candidate repair $r$ (a local policy edit proposed at a failure state $s_f$), the current policy $\pi_k$, and a compositional closure test set $\mathcal{C}$ of longer multi-stage plans built from the sub-task library; its output is a binary certificate $\mathrm{cert}(r) \in \{\text{accept}, \text{reject}\}$ plus a stored repair-composition certificate recording which substitutions were tested and which coverage partition cells were exercised.

**Load-bearing assumption (stated explicitly).** CCR assumes that the closure set $\mathcal{C}$ is a representative and sufficiently covering sample of the slot-contexts the repaired sub-policy will actually face during deployment. This assumption fails when the sub-task library omits rare slot-contexts (e.g., a repair valid only when the preceding sub-task ends at a corridor junction), in which case the certificate reflects validity on the sampled closure rather than on the deployment distribution. To make this assumption checkable rather than asserted, we construct $\mathcal{C}$ by coverage-aware partitioning (step 2a) and report the certificate as coverage-conditional (step 3).

**How it works.**

```pseudocode
INPUT: policy pi_k, failure state s_f, candidate repair r, sub-task library L,
       budget B=64, repeat threshold k=2, coverage partition function part(.)
# C is constructed by coverage-aware partitioning of the slot-context space:
#   part(P, j, tau) = (preceding_subtask_terminal_geometry, plan_length_bucket, subtask_type)
#   plan_length_bucket in {short: <10 steps, medium: 10-50, long: >50}
#   subtask_type in {goto, search, verify, return}
#   terminal_geometry in {open_room, corridor_junction, doorway, outdoor_open}
# C contains at least one (P, j, tau) per non-empty partition cell, then random fill to B.

1. LOCAL CHECK (inherited from AeroRSI):
     if continuation_rollout(pi_k + r, s_f) does not improve the failure:
         return REJECT

2. CLOSURE CHECK (new):
     cert = {}                       # repair-composition certificate
     coverage = {}                   # partition cell -> tested?
     # 2a. Coverage-aware construction of C (done once per round, reused for all repairs):
     C = build_closure_set(L, B=64, part=part)   # one sample per non-empty cell, then random fill
     for (P, j, tau) in C:
         pi_sub = substitute(pi_k + r, slot=j, sub_task=tau)
         # validity = does the repaired sub-policy still complete P
         # when re-embedded at slot j of the longer plan?
         # noise-robust: require k=2 repeated failing rollouts before REJECT
         fails = 0
         for trial in 1..k:
             if plan_rollout(pi_sub, P).success == False:
                 fails += 1
         v = (fails < k)             # True = valid, False = counterexample
         cert[(P, j, tau)] = v
         coverage[part(P, j, tau)] = True
         if v == False:
             return REJECT, cert, coverage   # counterexample found

3. CERTIFICATE STORE:
     store cert and coverage alongside r in the repair bank
     # certificate is coverage-conditional: it records which partition cells were tested
     return ACCEPT, cert, coverage

# Recursive loop (unchanged from AeroRSI except for step 2):
for k in 1..K:                               # K = 3 rounds
    failures = identify_failures(pi_k)
    for (s_f, r) in propose_repairs(failures): # rule-based + sampled proposals
        if CCR(pi_k, s_f, r, L, B=64, k=2) == ACCEPT:
            distill(pi_k, r)                 # keep behavior elsewhere fixed
    pi_{k+1} = pi_k
```

**Why this design.** We chose a *substitution-based counterexample search* over a learned validity predictor because the latter would require a training signal for composition-failure that does not exist before the first closure check, accepting the cost that each repair costs $B=64$ extra plan rollouts (plus $k=2$ repeats per sampled plan, so up to 128 plan rollouts per repair in the worst case). We chose to *reject on the first counterexample* rather than average validity scores because a single failing substitution is a concrete soundness violation and averaging would let a repair with a hidden composition failure pass, accepting the cost of higher rejection rates and slower self-improvement. We chose to *store the certificate* rather than re-verify every round because the sub-task library is fixed at round start, accepting the cost that a repair accepted under one library cannot be reused if the library is expanded mid-run. We chose *coverage-aware partitioning* over uniform random sampling because the slot-context space grows multiplicatively with plan length and library size, so a fixed budget of 64 uniform samples cannot be representative for long-horizon plans; partitioning by structural features (preceding sub-task terminal geometry, plan length bucket, sub-task type) guarantees at least one test per non-empty cell, accepting the cost that the certificate is coverage-conditional rather than a global accept/reject. We chose *k=2 repeated failing rollouts* before REJECT because a single stochastic failure is not a sound witness of invalidity; requiring two consecutive failures reduces false rejects from rollout noise, accepting the cost of up to 2x rollouts per sampled plan. A domain expert would not describe CCR as a variant of AeroRSI's continuation verification because AeroRSI's criterion is a single-context rollout with no substitution operator and no closure set — CCR's load-bearing computation is the re-embedding of the repaired sub-policy into plans it was never sampled from, which AeroRSI never performs.

**Why it measures what we claim.** The computational quantity `plan_rollout(substitute(pi_k+r, slot=j, tau), P).success` measures *compositional repair validity* because the named assumption is that the closure set $\mathcal{C}$ is a representative sample of the substitutions the repair will actually face during deployment; this assumption fails when the sub-task library omits rare slot-contexts (e.g., a repair valid only when the preceding sub-task ends at a corridor junction), in which case the quantity reflects validity on the sampled closure rather than on the deployment distribution. The quantity `cert[(P,j,tau)] == False` measures *existence of a composition counterexample* because the assumption is that k=2 repeated failing plan rollouts is a sound witness of invalidity (no rollout noise); this fails when plan rollouts are stochastic and two failures are a sampling artifact, in which case the quantity reflects rollout variance rather than a true composition failure. The stored `certificate` measures *repair soundness provenance* because the assumption is that the sub-task library is static within a round; this fails under mid-round library expansion, in which case the certificate reflects soundness only relative to the original library. The `coverage` record measures *coverage-conditional soundness* because the assumption is that the partition function `part(.)` captures the structural features that determine slot-context validity; this fails when two slot-contexts in the same partition cell differ in a feature not captured by `part(.)`, in which case the coverage record reflects tested partition cells rather than tested slot-contexts.

## Contribution

(1) A closure-certified repair operator (CCR) that replaces AeroRSI's continuation-only verification with a sub-task substitution algebra over a compositional closure test set, producing a repair-composition certificate for every accepted repair. (2) A design principle for recursive self-improvement in VLN: repairs should be certified as fixed points of substitution, not as local continuation successes, which changes which repairs are distilled and therefore which failure modes the next round can expose. (3) A reusable compositional closure test-set construction procedure over a sub-task library, applicable to TravelUAV-style aerial benchmarks and to LH-VLN-style long-horizon benchmarks.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | TravelUAV + multi-stage plan compositions | Aerial VLN with sub-task library for closure tests |
| Primary metric | Success rate on held-out multi-stage plans | Detects composition failures single-context checks miss |
| Secondary metric | False-accept rate vs. exhaustive closure oracle (small library) | Calibrates certificate soundness as function of B |
| Secondary metric | Coverage diagnostic: held-out plan slot-contexts vs. C partition cells | Measures distribution shift between certification and deployment |
| Baseline | AeroRSI continuation-only verification | Isolates value of closure check over local check |
| Baseline | AeroRSI with random-repair acceptance | Tests whether closure filtering or repair volume drives gains |
| Baseline | No-repair frozen 2B policy | Lower bound; confirms self-improvement actually helps |
| Ablation-of-ours | CCR with B=1 closure sample | Tests whether breadth of substitution search matters |
| Ablation-of-ours | CCR with uniform random C (no coverage partitioning) | Tests whether coverage-aware construction is load-bearing |
| Ablation-of-ours | CCR with k=1 (single failing rollout) | Tests whether noise-robust rejection matters |

### Why this setup validates the claim

The central claim is that AeroRSI's single-context continuation check admits repairs that break compositional structure, and that substitution-based closure certification removes exactly those repairs. The AeroRSI continuation-only baseline tests the sub-claim that local verification is insufficient: if our method beats it on multi-stage success while matching it on single-stage success, the gap is attributable to composition failures that only closure checking catches. The random-acceptance baseline tests whether our gain comes from filtering quality or merely from accepting fewer repairs; if random acceptance with matched repair volume underperforms CCR, filtering—not volume—is causal. The frozen policy anchors that any gain is from self-improvement at all. The B=1 ablation tests whether the closure set's breadth is load-bearing: if B=1 matches B=64, the substitution search adds nothing. The uniform-random-C ablation tests whether coverage-aware partitioning (the smallest repair to the load-bearing assumption) is doing work beyond random sampling. The k=1 ablation tests whether noise-robust rejection prevents false rejects from stochastic rollouts. The false-accept calibration against an exhaustive closure oracle on a reduced sub-task library (e.g., 5 sub-tasks, 3 plan lengths, 4 terminal geometries, so the full slot-context space is enumerable) directly measures the certificate's soundness as a function of B in {1,8,16,64}, converting the asserted representativeness of C into a measured coverage curve. The coverage diagnostic reports, for each held-out plan, whether its slot-context partition cell was tested in C; this measures the distribution shift between certification and deployment that the load-bearing assumption denies. Multi-stage success rate on held-out plans is the right primary metric because it directly measures the deployment quantity CCR claims to protect—completion of plans the repair was never sampled from—rather than the single-context success AeroRSI already optimizes.

### Expected outcome and causal chain

**vs. AeroRSI continuation-only verification** — On a case where a repair fixes a sub-policy at a corridor-junction failure but breaks it when the preceding sub-task ends in an open room, the baseline accepts the repair because its single-context rollout at the failure state improves, then fails during multi-stage execution. Our method rejects it because a sampled substitution (plan P with slot j re-embedded) yields a failing plan rollout, so we expect a noticeable gap on multi-stage success concentrated on junction-adjacent subtasks, with parity on single-stage success.

**vs. AeroRSI with random-repair acceptance** — On a case where many proposed repairs are locally plausible but compositionally unsound, the random-acceptance baseline admits them at the same rate as sound repairs, producing drift that surfaces as cascading failures in longer plans. Our method instead filters via counterexample search, so we expect CCR to show higher multi-stage success at comparable repair counts, with the gap widening as plan length grows and narrowing on short plans where composition failures are rare.

**vs. No-repair frozen 2B policy** — On a case where the base policy repeatedly fails a target-oriented search requiring a detour, the frozen baseline never improves because it has no repair mechanism. Our method accumulates certified repairs across rounds, so we expect monotone improvement across the three rounds on multi-stage success, with the largest per-round gain in round one and diminishing returns thereafter as remaining failures are harder to repair soundly.

**vs. CCR with B=1 closure sample** — On a case where the single sampled substitution happens to be benign but other slot-contexts are broken, the B=1 ablation accepts an unsound repair. Our method with B=64 samples enough substitutions to surface the counterexample, so we expect CCR to show fewer composition failures on held-out plans than the B=1 ablation, with the gap concentrated on repairs whose validity is slot-context-dependent. The false-accept calibration against the exhaustive oracle is expected to show false-accept rate decreasing with B (e.g., from ~0.4 at B=1 to <0.1 at B=64 on the reduced library), and the coverage diagnostic is expected to show that held-out plans whose partition cells were all tested in C have higher success than those with untested cells.

**vs. CCR with uniform random C (no coverage partitioning)** — On a case where the slot-context space is dominated by short-plan cells, uniform random sampling of B=64 draws few or no samples from rare long-plan cells, so a repair that breaks only in long plans passes. Our coverage-aware construction guarantees at least one sample per non-empty cell, so we expect CCR to show higher multi-stage success on long plans than the uniform-random ablation, with the gap concentrated on long-plan subsets and on rare terminal-geometry cells.

**vs. CCR with k=1 (single failing rollout)** — On a case where plan rollouts are stochastic and a valid repair occasionally fails a single rollout, the k=1 ablation rejects it as a false counterexample, reducing the number of accepted repairs and slowing self-improvement. Our k=2 threshold requires two consecutive failures, so we expect CCR to accept more sound repairs (higher repair acceptance rate) while maintaining comparable multi-stage success, with the gap in acceptance rate concentrated on repairs near the decision boundary.

### What would falsify this idea

If CCR's gain over AeroRSI continuation-only verification is uniform across single-stage and multi-stage subsets rather than concentrated on multi-stage plans with slot-context-sensitive subtasks, then the closure check is not doing the compositional work we claim and the central mechanism is wrong. Equivalently, if the B=1 ablation matches B=64 on multi-stage success, or if the false-accept calibration shows no decrease with B, the substitution search's breadth is not load-bearing.

## References

1. AeroRSI: Recursive Self-Improvement for Aerial Vision-Language Navigation
