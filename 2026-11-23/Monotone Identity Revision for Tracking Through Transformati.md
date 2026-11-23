# Monotone Identity Revision for Tracking Through Transformations

## Motivation

Across the video object tracking literature, identity is almost universally treated as a per-frame or per-proposal decision rather than a globally coherent belief that must remain consistent as evidence accumulates. MOTIP decodes ID labels for current detections directly from in-context trajectories (E1, E6), and TIPO generates split/merge proposals and queries an MLLM per top proposal without any mechanism that forces earlier identity or state-graph assignments to remain coherent when later evidence contradicts them (E3, E7). The root cause is structural: both branches lack a shared invariant on how beliefs may be revised, so a contradiction triggers either an independent re-decode (MOTIP) or an unconstrained state-graph edit (TIPO), and neither can distinguish a justified correction from an identity flip.

## Key Insight

A single monotone belief over trajectories and transformation states, revised only through a globally-scored and explicitly justified operation, makes identity persistence a checkable invariant rather than an emergent property of per-frame decoding.

## Method

```pseudocode
INPUT: video frames F_1..F_T; detector D; mask decoder M (SAM2-style);
       ID decoder Phi (MOTIP-style, in-context); MLLM V (verifier)
OUTPUT: identity belief Beta_T = (Tracks, StateGraph, RevisionLog)

INITIALIZE:
  Beta_0 = (empty Tracks, empty StateGraph, empty RevisionLog)
  overlap_prior weights (w_in, w_ext) = (0.6, 0.4)
  # Calibration (replaces fixed tau=0.35): fit on a held-out split of 512
  # VISTA clips (disjoint from test). For each candidate revision r in R,
  # compute raw cost delta delta_r = cost(r) - cost(retained alternative).
  # Fit a temperature-scaled logistic map P_correct(delta_r) = sigmoid(a*delta_r + b)
  # by minimizing cross-entropy on (delta_r, y_r) where y_r = 1 if r matches
  # ground-truth revision. Set target error rate alpha = 0.05. Compute
  # conformal quantile q_alpha on a held-out calibration subset (256 clips)
  # such that P_correct(delta_r) >= 1 - alpha for admitted retractions.
  # Store (a, b, q_alpha) as calibration parameters.
  K = 3   # freeze window for rejected retractions

FOR t = 1..T:
  # 1. Per-frame evidence (no commitment yet)
  dets_t   = D(F_t)
  masks_t  = M(F_t, dets_t)
  id_pred_t = Phi(tracks_in(Beta_{t-1}), dets_t)   # candidate ID labels

  # 2. Transformation proposals from disjoint mask components
  props_t = generate_split_merge_props(masks_t)          # TIPO-style
  score each p in props_t:
     s(p) = w_in * inner_overlap(p, masks_t)
          + w_ext * external_overlap(p, masks_t)

  # 3. Build candidate revisions, each a (operation, evidence, cost)
  R = []
  FOR each det d with id_pred_t[d] = c:
     R.append( op=ASSIGN(d, c), cost = id_consistency_cost(d, c, Beta_{t-1}) )
     FOR each alternative id c' in candidate_ids(Beta_{t-1}, d):
        R.append( op=REASSIGN(d, c'), cost = id_consistency_cost(d, c', Beta_{t-1})
                  + lambda * revision_penalty(c, c') )   # lambda = 1.0
  FOR each proposal p with s(p) > 0.5:   # proposal gate (was tau=0.35; now 0.5)
     verdict = V(p, top_k=1)          # MLLM: verify + semantic label
     R.append( op=APPLY_TRANSFORM(p, verdict),
               cost = (1 - s(p)) + transform_consistency_cost(p, Beta_{t-1}) )

  # 4. Constrained global revision: pick the set of revisions that
  #    minimizes total cost subject to the monotonicity invariant
  chosen = argmin_{S subset R, S mutually consistent}
              sum_{(op,cost) in S} cost
           subject to  monotone_ok(S, Beta_{t-1})

  # 5. Apply and log
  Beta_t = apply(S=chosen, Beta_{t-1})
  RevisionLog.append( (t, chosen, justification(chosen, Beta_{t-1})) )
RETURN Beta_T
```

**Monotonicity invariant (`monotone_ok`).** For every prior assignment (track id → entity, or state-graph edge) already in Beta_{t-1}, a revision S is admissible only if it either (a) leaves the assignment untouched, or (b) retracts it together with a justification record J = (triggering evidence id, cost delta, verifier verdict if any). An assignment may be retracted only when the calibrated probability that the alternative is correct, P_correct(delta_r) from the temperature-scaled logistic map, is at least 1 - alpha (alpha = 0.05) and delta_r exceeds the conformal quantile q_alpha; otherwise S is rejected and the assignment is frozen for the next K=3 frames before re-evaluation. This makes retraction a *justified, cost-gated* operation rather than a free per-frame re-decode.

**Concrete cost functions.**
- `id_consistency_cost(d, c, Beta_{t-1})` = negative log-likelihood of the decoded ID token for detection d under Phi, conditioned on the retained trajectory of identity c: `-log Phi(c | tracks_in(Beta_{t-1}), d)`.
- `transform_consistency_cost(p, Beta_{t-1})` = graph-edit distance between the state graph implied by proposal p and the retained state graph in Beta_{t-1}, computed as the minimum number of node/edge insertions and deletions to transform one into the other, normalized by the total number of nodes and edges in Beta_{t-1}.
- `revision_penalty(c, c')` = 1 if c != c' else 0 (identity-change indicator), scaled by lambda = 1.0.

**(A) What it is.** MIR (Monotone Identity Revision) is a tracking framework that takes frames, a detector, a mask decoder, an in-context ID decoder, and an MLLM verifier, and outputs a single persistent belief Beta_T = (Tracks, StateGraph, RevisionLog) in which every identity and transformation assignment is either retained or retracted with a logged justification.

**(C) Why this design.** We chose a *single shared belief* over separate MOTIP-style per-detection ID decoding and TIPO-style state-graph edits because the two surfaces must be revised under the same admissibility test; otherwise a track reassignment and a split/merge edit can each be locally plausible while jointly violating identity persistence, accepting the cost that the joint optimization is solved by greedy set selection over R rather than exact combinatorial search. We chose a *calibrated cost-gated retraction threshold* over an unconstrained argmin because unconstrained revision is exactly what MOTIP and TIPO already do (E1, E3), and a calibrated threshold converts 'contradiction' into an explicit, auditable event, accepting the cost that a genuinely fast-changing target may be frozen for K frames. We chose *MLLM verification only on top-scoring proposals* over verifying every proposal because TIPO already shows single-top-proposal MLLM queries are effective and cheap, accepting the cost that a mis-scored proposal is never rescued. We chose an *explicit RevisionLog* over implicit belief state because it is the only way the monotonicity invariant is externally checkable, accepting the cost of extra bookkeeping. A domain expert would not describe MIR as 'MOTIP + TIPO': the load-bearing object is the joint admissibility predicate `monotone_ok` over a shared belief, which neither MOTIP (no state graph) nor TIPO (no global revision constraint) possesses.

**(D) Why it measures what we claim.** The quantity `id_consistency_cost(d, c, Beta_{t-1})` measures *identity persistence* under the assumption that a correct ID assignment yields lower in-context decoding cost against the retained trajectory than any alternative; this assumption fails when two targets are appearance-identical for several frames, in which case the cost reflects appearance ambiguity rather than true persistence. The quantity `transform_consistency_cost(p, Beta_{t-1})` measures *state-graph coherence* under the assumption that a valid split/merge proposal is consistent with the retained entity relations; this fails when the state graph is stale, in which case the cost reflects staleness rather than proposal invalidity. The `revision_penalty(c, c')` measures *justified revision* under the assumption that larger identity changes require stronger evidence; this fails when an entire trajectory was initialized wrong, in which case the penalty reflects initialization error rather than unjustified flips. The `monotone_ok` predicate measures the *non-retraction invariant* under the assumption that calibrated cost-gated retraction with a logged justification is equivalent to a monotone belief update; this fails when the calibration map is miscalibrated or the log is incomplete, in which case admissibility reflects calibration error or logging completeness rather than belief monotonicity.

## Contribution

['A joint identity-and-transformation belief representation with an explicit monotone admissibility predicate (`monotone_ok`) that unifies MOTIP-style ID decoding and TIPO-style state-graph edits under one revision constraint.', 'A cost-gated, justification-logged revision algorithm that converts contradictions into auditable retraction events rather than per-frame re-decodes, with a frozen-frame fallback for ambiguous evidence.', 'A RevisionLog artifact that makes identity persistence externally checkable and supports post-hoc auditing of when and why identities changed.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | VISTA (assembly/disassembly events) | Contains transformations where identity persistence is genuinely contested |
| Dataset | DanceTrack (identity switches) | Tests invariant on occlusion re-identification without physical transformation |
| Primary metric | Identity persistence accuracy through transformations | Directly measures retained-vs-retracted assignment correctness |
| Secondary metric | Identity switch rate (DanceTrack) | Measures unjustified flips under occlusion |
| Baseline | TIPO (training-free proposals + MLLM) | Tests whether global admissibility beats per-proposal edits |
| Baseline | SAM 2/3 memory tracker | Tests whether cost-gated revision beats appearance-only tracklets |
| Baseline | MOTIP (in-context ID prediction) | Tests whether shared belief beats per-detection ID decoding |
| Ablation-of-ours | MIR without monotone_ok (unconstrained argmin) | Isolates the admissibility predicate as load-bearing |
| Ablation-of-ours | MIR with fixed tau=0.35 (no calibration) | Tests whether calibration is necessary for the gate |

### Why this setup validates the claim

The central claim is that a single shared belief revised under a calibrated cost-gated admissibility predicate is what produces correct identity persistence through physical transformations — not the ID decoder, not the proposal generator, and not the MLLM alone. VISTA is the right dataset because it is precisely the regime where identity is contested: an object that splits or merges has no stable appearance cue, so any method that relies on appearance continuity alone must fail, and any method that edits state without a retention constraint must over-revise. DanceTrack adds a second domain where identity is contested by occlusion rather than physical transformation, testing whether the invariant generalizes beyond VISTA. Each baseline isolates one sub-claim. SAM 2/3 tests whether appearance-robust tracklets suffice — they should not, because they cannot relabel a semantically new entity. TIPO tests whether per-proposal generate-evaluate-judge edits suffice — they should not, because each edit is locally plausible while the sequence of edits drifts. MOTIP tests whether strong per-detection ID decoding suffices — it should not, because it has no state graph to revise and no retraction gate. The ablation without `monotone_ok` is the decisive one: removing the admissibility predicate keeps every other component identical, so any drop in persistence accuracy is attributable to the predicate itself. The fixed-tau ablation tests whether the calibration step is necessary: if fixed tau matches calibrated gating, the calibration story is not load-bearing. Identity persistence accuracy is the right metric because it is exactly the quantity the invariant is designed to protect; a metric like MOTA or HOTA would blend detection quality and mask quality into the signal and could mask a revision failure as a detection miss.

### Expected outcome and causal chain

**vs. TIPO** — On a case where a tracked object is disassembled into two components and one component is later re-merged with a third object, TIPO produces a chain of locally-scored split and merge edits, each accepted because its overlap prior is high, so the state graph drifts and the original entity's lineage is lost. Our method instead routes each proposal through `monotone_ok`, so a merge that would retract an existing entity edge without a calibrated probability of correctness exceeding 1 - alpha and a logged justification is frozen rather than applied, and the lineage survives. We expect a noticeable gap on the multi-transformation subset (assembly-then-disassembly chains) and near-parity on single-event clips, where no retraction is contested.

**vs. SAM 2/3 memory tracker** — On a case where an object is split and the resulting components are appearance-similar to a distractor elsewhere in the frame, the SAM memory tracker produces an identity swap because its memory buffer attends to appearance and has no mechanism to declare the old identity retired. Our method instead treats the split as a transformation proposal whose transform consistency cost is checked against the retained state graph, so the old identity is retracted with a justification and the new entities are labeled by the verifier rather than silently inherited. We expect the gap to concentrate on distractor-heavy frames and to shrink on clean single-object clips.

**vs. MOTIP** — On a case where a merged entity must inherit partial identity from two prior tracks, MOTIP produces a single decoded ID for the merged detection because its in-context decoder assigns one label per detection and has no representation for a many-to-one identity event. Our method instead carries the merge as a state-graph operation with an explicit revision cost, so the merged entity is recorded as a new node with two justified parent retractions. We expect a gap on merge events specifically, with parity on pure detection-and-association frames where no transformation occurs.

**vs. MIR without monotone_ok** — On a case where a correct revision requires retracting an assignment that has been stable for several frames, the unconstrained argmin accepts the revision immediately because no gate blocks it, but on a case where a spurious revision is locally cheaper, it also accepts that, leading to identity flips. Our method freezes the assignment for K=3 frames unless the calibrated probability of correctness exceeds 1 - alpha, so spurious flips are suppressed while justified corrections are admitted after the freeze. We expect the ablation to show higher identity switch rate and lower persistence accuracy on contested retractions, with parity on uncontested frames.

**vs. MIR with fixed tau=0.35** — On a case where the raw cost delta is miscalibrated (e.g., overconfident decoder scores), the fixed threshold either freezes correct revisions (if delta < tau despite high true correctness) or admits unjustified flips (if delta > tau despite low true correctness). Our calibrated gate uses the temperature-scaled logistic map and conformal quantile to set the admission boundary per-track, so it should reduce both false freezes and false admissions. We expect the fixed-tau ablation to underperform on clips with appearance-identical targets, where raw costs are unreliable.

### What would falsify this idea

If the ablation without `monotone_ok` matches full MIR on the multi-transformation subset, then the admissibility predicate is not load-bearing and the central claim is wrong. Likewise, if MIR's gain over TIPO is uniform across single-event and multi-event clips rather than concentrated on contested retractions, the improvement comes from something other than calibrated cost-gated revision.

## References

1. Taking Apart and Putting Together: Tracking Objects Through Physical Transformations
