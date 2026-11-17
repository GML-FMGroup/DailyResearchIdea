# Order-Invariant Requirement Belief Revision Under Contradictory Colleague Testimony

## Motivation

Organizational coding benchmarks such as SWE-CoWork present colleague statements as a fixed pre-authored pool, so an agent's recovered requirement set is measured only once and only under one testimony order; the benchmark therefore cannot observe whether an agent's requirement estimator is order-sensitive or non-convergent. This is a structural limitation of the benchmark class, not of any single model: because testimony is static, the environment never reacts to the agent's evolving belief, and the reported 'requirement recovery' score conflates genuine belief revision with the accident of which colleague was queried first. Real organizational requirements arrive as a revising, sometimes contradictory stream, so an agent whose estimator converges to different requirement sets under permuted testimony orders is unreliable as a digital employee even when its one-shot score is high.

## Key Insight

If colleague testimony is generated conditionally on the agent's currently recovered belief and contradictory testimony is injected only where the belief is still under-determined, then a correct requirement estimator is forced to be a contraction mapping on belief states, so its recovered requirement set is invariant to testimony order — an invariant that static testimony pools cannot test and that order-permutation consistency directly exposes.

## Method

**A. What it is.** We introduce BICON (Belief-Conditioned Organizational Narrator), a reactive benchmark harness that wraps the SWE-CoWork organization simulator. Its inputs are a repository assignment, a colleague roster with bounded knowledge, and an agent's requirement estimator; its outputs are (i) a sequence of belief-conditioned colleague utterances and (ii) an order-permutation consistency score over the requirement sets the agent recovers under K permuted testimony orders, paired with a correctness term so that invariance is not rewarded in isolation.

**B. How it works.**

```pseudocode
BICON(assignment A, colleagues C, estimator E, permutations K=6, eps=0.02):
  # Phase 1: seed belief
  b0 = E.initialize(A.brief_handoff)          # initial recovered requirement set
  R  = A.private_requirement_set              # hidden ground truth, never shown

  # Phase 2: belief-conditioned testimony generation (ONCE per atom, not per permutation)
  # Generate the testimony multiset T once, conditioned on the agent's belief trajectory
  # under a canonical order pi_0, so that all K permutations deliver the SAME multiset.
  T = {}                                      # map: requirement atom u -> testimony item
  b = b0
  for colleague c in pi_0:                    # canonical order, fixed seed
    U = underdetermined(b, R)                 # requirements not yet pinned by b
    if c.knowledge ∩ U == ∅: continue
    u = sample(U ∩ c.knowledge, seed=hash(pi_0,c))
    t = c.speak(u, b, R)                      # LLM-generated, grounded in c.records
    T[u] = t
    b = E.update(b, t)

  # Phase 3: order-permutation consistency over the FIXED testimony multiset
  for each permutation pi of C:
    b = b0
    for colleague c in pi:
      for u in T where speaker(u)==c:         # deliver pre-generated item, no regeneration
        b = E.update(b, T[u])
    B_pi = b

  for i,j in pairs(K):
    cons[i,j] = Jaccard(B_pi_i, B_pi_j)       # over requirement atoms
  OPCS = mean(cons)                            # order-permutation consistency score

  # Phase 4: convergence check
  conv = 1[|B_pi_final ⊖ B_pi_prev| <= eps]   # eps = 0.02

  # Phase 5: correctness term (pairs OPCS with recovery)
  acc = Jaccard(B_pi_final, R)                # requirement-atom accuracy vs hidden truth
  return OPCS, conv, acc, {B_pi}
```

Colleague `speak(u, b, R)` is implemented as a constrained LLM call: the prompt contains the colleague's bounded knowledge record, the requirement atom `u`, the agent's current belief `b` restricted to `u`, and the hidden truth `R[u]`; the colleague is instructed to agree, contradict, or decline according to whether `b[u]` matches `R[u]`. Contradiction injection is therefore *conditional on belief state*, not pre-authored. The estimator `E` is the agent under test (any SWE agent with a requirement-recovery interface); BICON does not modify it.

**Load-bearing assumption (stated explicitly).** OPCS measures belief-revision quality only under the assumption that a correct requirement estimator's recovered requirement set is a commutative aggregator over testimony — i.e., its output depends only on the evidence multiset and not on presentation order. This assumption fails when the estimator uses order-dependent heuristics (e.g., last-writer-wins, or recency-weighted context truncation), in which case OPCS reflects context-window recency/positional bias rather than belief-revision quality. To decouple the metric from the estimator's positional sensitivity, Phase 2 generates the testimony multiset `T` once under a canonical order `pi_0`, and Phase 3 permutes only the *delivery order* of that fixed multiset; the harness's own generation is therefore held constant across permutations. To calibrate what OPCS looks like when commutativity holds versus when it is violated by recency alone, we include two control estimators: (i) a **set-aggregation baseline** that is provably order-invariant (it accumulates testimony into an unordered set and returns the set), which establishes the OPCS ceiling under commutativity; and (ii) a **recency-weighted baseline** that applies an explicit exponential decay to testimony position, which establishes the OPCS floor under known positional bias. A third control, **oracle-speak mode**, replaces the LLM `speak` call with a deterministic rule-based agree/contradict/decline function read directly from `R`, isolating LLM-generation noise from belief-conditioning effects.

**C. Why this design.** We chose belief-conditioned testimony generation over the fixed pre-authored pool used in SWE-CoWork because a static pool cannot distinguish an estimator that genuinely revises beliefs from one that happens to query colleagues in a lucky order, accepting the cost that testimony must be generated at inference time (slower, LLM-dependent). We chose to inject contradictions only on under-determined requirement atoms rather than uniformly across all atoms because uniform injection would swamp the estimator with redundant contradictions and measure noise tolerance rather than belief revision, accepting the cost that the harness must maintain an explicit under-determination mask over `R`. We chose order-permutation consistency (Jaccard over requirement atoms across K=6 permutations) over a single-pass accuracy metric because accuracy conflates recovery with order luck, accepting the cost that K permutations multiply evaluation cost by K. We chose to generate the testimony multiset once and permute only its delivery order because regenerating testimony per permutation would confound the harness's own order-dependent generation with the estimator's positional bias, accepting the cost that the testimony stream is no longer fully reactive within each permutation. A domain expert would not describe BICON as a variant of SWE-CoWork: SWE-CoWork is a static benchmark of one-shot requirement recovery, whereas BICON is a reactive harness whose testimony distribution is a function of the agent's own belief trajectory, making the estimator a dynamical system under test rather than a single query.

**D. Why it measures what we claim.** Order-permutation consistency (OPCS, mean Jaccard across K=6 permuted testimony orders over a fixed testimony multiset) measures belief-revision order-invariance because we assume that a correct requirement estimator's output is a function only of the evidence set, not of its presentation order — i.e., the estimator is a commutative aggregator over testimony; this assumption fails when the estimator uses order-dependent heuristics (e.g., last-writer-wins, or recency-weighted context truncation), in which case OPCS reflects context-window recency bias rather than belief revision quality. The convergence flag `conv` measures iterative-estimator stability because we assume that a convergent estimator's belief change between successive testimony items shrinks toward zero as under-determined atoms are resolved; this assumption fails when the estimator's belief is dominated by a single late-arriving contradiction, in which case `conv` reflects a single update rather than asymptotic stability. The under-determination mask `U` measures the harness's ability to target informative testimony because we assume that requirements already pinned by `b` are uninformative to query; this assumption fails when the estimator's belief is confidently wrong on a pinned atom, in which case `U` reflects the estimator's self-reported certainty rather than true ambiguity. The correctness term `acc` (Jaccard between the final recovered set and hidden `R`) measures recovery quality because we assume that an order-invariant estimator is also accurate — i.e., invariance implies convergence to `R`; this assumption fails for a degenerate constant estimator that returns the same fixed set under every permutation, achieving OPCS=1 while recovering nothing, in which case OPCS reflects trivial invariance rather than belief-revision quality and must be read jointly with `acc`.

## Contribution

["BICON, a reactive organizational benchmark harness that generates colleague testimony conditionally on the agent's current recovered-belief state and injects contradictions only on under-determined requirement atoms, replacing SWE-CoWork's fixed pre-authored testimony pool.", 'An order-permutation consistency metric (OPCS) over recovered requirement sets across K permuted testimony orders, together with a convergence flag, that operationalizes belief-revision order-invariance as a measurable property of a requirement estimator rather than a one-shot accuracy score.', 'A design principle: requirement estimators that are commutative aggregators over testimony are order-invariant, and OPCS separates such estimators from order-sensitive ones even when their one-shot recovery accuracy is identical.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | SWE-CoWork organization assignments (101 repos) | Real multi-colleague requirement recovery with executable verification |
| Primary metric | Order-permutation consistency score (OPCS, mean Jaccard) paired with requirement-atom accuracy `acc` | Isolates order-invariance of belief revision from recovery luck |
| Baseline | SWE-CoWork static pre-authored testimony | Tests whether reactive testimony reveals order sensitivity |
| Baseline | Single-pass requirement accuracy (complete-requirements condition) | Tests whether accuracy conflates recovery with order luck |
| Baseline | Uniform contradiction injection harness | Tests whether targeted under-determined injection beats noise flooding |
| Baseline | Set-aggregation estimator (provably order-invariant) | Calibrates OPCS ceiling when commutativity holds |
| Baseline | Recency-weighted estimator (explicit positional decay) | Calibrates OPCS floor under known positional bias |
| Ablation-of-ours | BICON with fixed pre-authored testimony pool | Isolates belief-conditioning as the causal ingredient |
| Ablation-of-ours | BICON with oracle-speak (rule-based agree/contradict from R) | Isolates LLM-generation noise from belief-conditioning |
| Ablation-of-ours | BICON with eps ∈ {0.01, 0.02, 0.05} and K ∈ {3, 6, 12} | Sensitivity of convergence flag and OPCS to hyperparameters |

### Why this setup validates the claim

The central claim is that a requirement estimator's belief-revision quality is only visible when testimony is a function of the estimator's own evolving belief, and that order-permutation consistency is the metric that exposes this. SWE-CoWork's static pool gives us the natural control: same assignments, same colleagues, same hidden requirement set, but testimony that cannot react to the agent. If OPCS under BICON separates estimators that the static pool cannot distinguish, the belief-conditioning mechanism is doing the work. The single-pass accuracy baseline tests the second sub-claim — that accuracy conflates recovery with order luck — because if accuracy and OPCS rank estimators identically, the permutation machinery is unnecessary. The uniform-injection baseline tests the third sub-claim, that targeting under-determined atoms measures belief revision rather than noise tolerance. The set-aggregation and recency-weighted baselines calibrate the OPCS scale: the former establishes the ceiling when commutativity holds, the latter the floor when positional bias dominates, so a BICON OPCS value can be interpreted against known endpoints rather than in isolation. The oracle-speak ablation isolates LLM-generation noise from belief-conditioning effects. The fixed-pool ablation is the decisive control: it keeps the reactive harness, the permutation protocol, and the metric, and removes only belief-conditioned contradiction placement, so any gap between it and full BICON is attributable to conditioning rather than to the evaluation apparatus. Because the testimony multiset is generated once and only its delivery order is permuted, any OPCS spread is attributable to the estimator's order sensitivity rather than to the harness's own order-dependent generation.

### Expected outcome and causal chain

**vs. SWE-CoWork static pre-authored testimony** — On a case where the handoff brief under-specifies an authentication requirement that two colleagues each partially know, the static baseline produces a fixed testimony sequence regardless of what the agent has already inferred, so a lucky ordering yields high recovery and an unlucky one yields low recovery for the same estimator. Our method instead injects contradiction only on atoms the agent's belief has not yet pinned, because the under-determination mask recomputes after every utterance, so the testimony stream tracks the agent's actual ambiguity. We expect a visible spread in per-permutation Jaccard for the static baseline that collapses under BICON for strong estimators, and a persistent spread for weak ones.

**vs. Single-pass requirement accuracy (complete-requirements condition)** — On a case where the agent recovers 60% of fail-to-pass tests but completes the assignment only under one particular colleague ordering, the accuracy baseline reports a single number that hides the ordering dependence entirely, because its mechanism aggregates one transcript with no counterfactual order. Our method instead runs six permuted orders and scores pairwise Jaccard over requirement atoms, because order-invariance is the property being measured. We expect estimators that look equivalent on single-pass accuracy to separate by a clear margin on OPCS, with the gap concentrated on assignments where requirements are distributed across three or more colleagues.

**vs. Uniform contradiction injection harness** — On a case where the agent has already correctly pinned a requirement from the brief, uniform injection produces a contradiction on that atom anyway, so the agent must spend belief-revision capacity re-defending a settled belief. Our method instead skips pinned atoms via the under-determination mask, because testimony is conditioned on the belief state, so contradiction budget is spent only where the agent is genuinely ambiguous. We expect uniform injection to depress OPCS roughly uniformly across easy and hard assignments, while BICON shows parity on easy assignments and a marked advantage on assignments with many under-determined atoms.

**vs. Set-aggregation estimator (provably order-invariant)** — This baseline accumulates testimony into an unordered set and returns it, so by construction it has OPCS=1.0. We expect it to establish the OPCS ceiling and to show that a high OPCS alone is insufficient: its `acc` will be low because it performs no belief revision, demonstrating the degenerate-estimator failure mode and motivating the paired correctness term.

**vs. Recency-weighted estimator (explicit positional decay)** — This baseline applies an exponential decay to testimony position, so it is deliberately order-sensitive. We expect it to establish the OPCS floor and to show that BICON's OPCS for strong estimators sits well above this floor, confirming that the metric is not merely re-measuring recency bias.

**vs. BICON with oracle-speak (rule-based agree/contradict from R)** — This ablation replaces the LLM `speak` call with a deterministic rule-based function read directly from `R`. We expect its OPCS and `acc` to be close to full BICON for strong estimators, with any gap attributable to LLM-generation noise rather than to belief-conditioning, confirming that the conditioning mechanism rather than generation stochasticity drives the result.

**vs. BICON with eps and K sensitivity** — We expect OPCS to be stable across K ∈ {3, 6, 12} (with variance decreasing as K grows) and `conv` to be stable across eps ∈ {0.01, 0.02, 0.05}, confirming that the reported conclusions are not artifacts of hyperparameter choice.

### What would falsify this idea

If BICON's OPCS gains over the fixed-pool ablation are uniform across assignments rather than concentrated on those with many under-determined requirement atoms distributed across colleagues, the belief-conditioning mechanism is not the cause and the central claim is wrong. Likewise, if OPCS ranks estimators identically to single-pass accuracy across the full assignment set, or if the set-aggregation baseline's OPCS=1.0 is not accompanied by low `acc` (i.e., the degenerate-estimator failure mode does not appear), the order-invariance framing adds nothing and the metric should be abandoned.

## References

1. SWE-CoWork: Benchmarking Coding Agents as Digital Employees in Organizations
