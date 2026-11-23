# Intent-Gated Supervision Pruning for Text-to-Image Prompt Enhancers

## Motivation

Offline prompt-enhancer training pipelines such as P2E refine dense-caption supervision by heuristically deleting redundant sentences under a per-example deletion bound, but the bound is set on caption redundancy alone with no invariant tying the pruned supervision to user-stated constraints. As a result, refined SFT pairs can drift from user intent: the enhancer is trained to reproduce whatever survives the heuristic, and intent preservation is treated as emergent rather than enforced. This is a structural failure of the supervision-construction stage, not a model-capacity failure, and it recurs across both the offline-pruning lineage (P2E) and the online-optimization lineage (MinorityPrompt), where intent preservation is likewise a soft objective rather than a binding constraint.

## Key Insight

If every accepted pruning step must preserve entailment of the user-stated constraints by the surviving supervision, then the refined SFT pair is confined to the intent-satisfying equivalence class by construction, so intent drift cannot be introduced during supervision refinement regardless of how aggressive the deletion is.

## Method

**What it is.** IGSP (Intent-Gated Supervision Pruning) is a text-only supervision-refinement procedure that replaces P2E's fixed per-example deletion bound with a constrained optimization over deletion sets. Its inputs are a dense caption `c`, a short user-like prompt `u`, and an extracted constraint set `K(u)`; its output is a pruned caption `c*` and a training pair `(u, c*)` admitted to the enhancer SFT set.

**How it works.**

```pseudocode
IGSP(c, u, K(u), verifier V, budget B, lambda, tau):
  S = sentences(c)                                  # candidate deletion units
  K = extract_constraints(u)                        # atomic user constraints (entities, counts, relations, styles)
  # V is a constraint-typed verifier: each k in K is routed by type to a type-specific checker
  #   entity  -> exact entity-string match (case-insensitive, alias table)
  #   count   -> numeral/count parser + exact integer match
  #   relation-> spatial-relation parser (spaCy dependency parse) + relation-tuple match
  #   style   -> frozen NLI model (DeBERTa-v3-large-MNLI) queried per (caption, style-constraint) pair
  # V returns a binary label and a calibrated score in [0,1] per (caption, k)
  c0 = c
  for each sentence s in S ordered by redundancy score r(s) descending:
      c_try = c_current \ {s}
      # (i) hard intent gate: every constraint must remain entailed by its type-specific checker
      if not all(V(c_try, k).label == ENTAIL for k in K):
          reject s; continue
      # (ii) redundancy gain vs. intent-margin cost (binary gate + calibrated margin)
      gain = r(s)
      margin_cost = sum_k [ V(c_current, k).score - V(c_try, k).score ]
      if gain - lambda * margin_cost > 0 and deletions < B:
          c_current = c_try
          record (s, gain, margin_cost)
  c* = c_current
  # (iii) sufficiency check: the pruned supervision must still entail each constraint
  if not all(V(c*, k).label == ENTAIL for k in K): return REJECT_PAIR
  return (u, c*)
```

Hyperparameters: redundancy score `r(s)` from a sentence-level TF-IDF/embedding novelty score; deletion budget `B` set per-example to `ceil(0.5 * |S|)` (replacing P2E's fixed bound); intent-margin weight `lambda = 0.1`; verifier `V` is a constraint-typed verifier whose style branch is a frozen NLI model (DeBERTa-v3-large-MNLI) and whose entity/count/relation branches are rule-based parsers as specified above; the NLI branch's decision threshold `tau` is calibrated on a held-out set of 512 (caption, style-constraint) pairs annotated for constraint preservation, chosen to maximize F1 of the ENTAIL label; constraints `K(u)` are extracted by a rule-based parser over the user prompt (noun phrases, numerals, spatial prepositions, style adjectives), yielding 3-8 atomic constraints per prompt.

**Why this design.** We replaced P2E's fixed per-example deletion bound with a per-step entailment gate rather than a post-hoc filter because a post-hoc filter can only accept or reject an already-drifted pair, whereas a per-step gate keeps the search inside the intent-satisfying region and lets aggressive deletion continue where it is safe. We chose a constraint-typed verifier over a single generic NLI checker because generic MNLI models are known to fail on composition, counting, and spatial reasoning (McCoy et al. 2019; Gururangan et al. 2018), which are exactly the atomic constraint types IGSP must protect; routing each constraint type to a sound checker removes the load-bearing dependence on generic NLI soundness while keeping the per-step gating mechanism. We chose a redundancy-ordered greedy search over an exhaustive subset search because exhaustive search is exponential in `|S|` and the greedy order makes the intent-margin cost interpretable per step, accepting the cost that the greedy order can miss deletions that only become safe after other deletions. A domain expert would not describe this as "P2E with a filter" because the gate changes the optimization object itself: the deletion set is now the argmax of a constrained objective whose feasible region is defined by type-specific entailment, not a heuristic bound followed by rejection.

**Why this measures what we claim.** The binary label `V(c_try, k).label` measures constraint preservation because each type-specific checker is assumed to be a sound approximation of the user-intent entailment relation on its constraint type (exact match for entities/counts, dependency-parse relation match for spatial relations, calibrated NLI for style); this assumption fails when a constraint is implicit or pragmatic (e.g., "a cozy room" where coziness is not lexically entailed), in which case the style branch's label reflects surface lexical overlap rather than intent satisfaction. The intent-margin cost `V(c_current,k).score - V(c_try,k).score` measures the marginal intent risk of a deletion because the calibrated score difference is assumed monotone in the amount of constraint-relevant content removed; this assumption fails when the checker's scores saturate near 1.0 for both captions, in which case the margin reflects checker saturation rather than true risk. The redundancy gain `r(s)` measures caption redundancy because sentence-level novelty is assumed to be inversely related to the information already present in the surviving caption; this assumption fails for sentences that are lexically novel but semantically redundant, in which case `r(s)` reflects lexical novelty rather than redundancy. The sufficiency check on `c*` measures training-pair admissibility because a pair is assumed useful for SFT only if its supervision still entails the user constraints; this assumption fails when the enhancer's target distribution requires detail beyond the constraints, in which case admissibility reflects constraint coverage rather than training utility.

## Contribution

["A constrained supervision-refinement procedure (IGSP) that replaces P2E's heuristic per-example deletion bound with a per-step entailment gate and a redundancy-versus-intent-margin objective, producing refined SFT pairs whose intent satisfaction is enforced rather than emergent.", 'A design principle: intent preservation in offline prompt-enhancer training should be implemented as a feasibility constraint on the supervision-construction search, not as a post-hoc filter or a soft likelihood objective, because the constraint changes which deletions are reachable.', 'A per-example deletion budget derived from the constraint set size and redundancy distribution, replacing the fixed bound, with the intent-margin weight `lambda` as the single knob trading redundancy removal against constraint coverage.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Prompt+caption SFT pairs from P2E's caption-derived construction | Same supervision source as the baseline under test |
| Primary metric | Human-preference alignment on generated images | Directly targets the outcome P2E optimizes |
| Baseline | Conventional PE (unpruned caption pairs) | Tests whether pruning helps at all |
| Baseline | P2E fixed-bound deletion | Tests whether gating beats a fixed bound |
| Baseline | Post-hoc NLI rejection filter | Tests whether per-step gating beats post-filtering |
| Ablation-of-ours | IGSP without intent-margin term (lambda=0) | Isolates the margin-cost contribution |
| Ablation-of-ours | IGSP with binary entailment-margin (logit margin) | Tests saturation failure mode of score-difference margin |
| Calibration study | 512 held-out (caption, constraint) pairs annotated for constraint preservation | Quantifies verifier soundness per constraint type |

### Why this setup validates the claim

The central claim is that replacing P2E's fixed per-example deletion bound with a per-step entailment gate yields supervision that better preserves user intent while still pruning redundancy, so the trained enhancer produces images more aligned with human preference. The unpruned conventional PE baseline establishes that pruning is beneficial at all; if IGSP cannot beat it, the pruning premise is dead. The fixed-bound P2E baseline is the head-to-head test of the specific mechanism swap: same redundancy objective, different feasibility region. The post-hoc filter baseline isolates whether the gain comes from gating during search versus merely rejecting drifted pairs afterward, which is exactly the design distinction we argue matters. The lambda=0 ablation tests whether the intent-margin cost term contributes anything beyond the hard gate. The binary entailment-margin ablation tests whether the score-difference margin is corrupted by saturation, as predicted. The calibration study measures the verifier's precision/recall against human constraint-preservation judgments, stratified by constraint type (entity, count, relation, style), so the soundness assumption underlying the gate is empirically scoped rather than asserted. Human-preference alignment is the right metric because it is the outcome P2E itself optimizes and because constraint preservation is a perceptual, pragmatic property that lexical scores only proxy; a metric on verifier scores would be circular with our own gate. Together these form a falsifiable test: the mechanism claim predicts gains concentrated where fixed-bound pruning violates constraints, not uniform gains.

### Expected outcome and causal chain

**vs. Conventional PE (unpruned caption pairs)** — On a case where the dense caption carries image-specific detail weakly related to the short user prompt, the baseline trains the enhancer to reproduce that detail, so generated images drift toward caption idiosyncrasies and lose compositional focus. Our method deletes those redundant sentences while the gate guarantees the surviving supervision still entails the user constraints, so the enhancer learns to expand intent rather than imitate captions. We expect a moderate but consistent preference-alignment gain, largest on prompts with short, sparse user intent and long dense captions.

**vs. P2E fixed-bound deletion** — On a case where a constraint-bearing sentence (e.g., a spatial relation or a count) falls within P2E's fixed deletion bound and its redundancy score ranks it high, P2E deletes it and produces supervision that no longer entails the constraint, so the enhancer under-specifies that attribute and images violate it. Our per-step gate rejects that deletion because the type-specific checker (exact count match or relation-tuple match) no longer entails the constraint, and the margin cost further penalizes near-threshold deletions, so the surviving pair keeps the constraint. We expect a noticeable gap concentrated on constraint-dense prompts (counts, relations, styles) and near-parity on constraint-sparse prompts.

**vs. Post-hoc NLI rejection filter** — On a case where aggressive deletion is safe only after several other sentences are already removed, the post-hoc filter sees only the final drifted pair and rejects it wholesale, discarding otherwise-usable supervision and shrinking the SFT set. Our per-step gate keeps the search inside the intent-satisfying region, so it can delete aggressively where safe and retain pairs the filter throws away. We expect higher admitted-pair yield at comparable constraint preservation, showing as a preference gain that grows as deletion aggressiveness increases.

**vs. IGSP without intent-margin (lambda=0)** — On a case where the verifier's scores sit near 1.0 for both the current and trial caption, the hard gate alone admits deletions that shave constraint-relevant content because entailment still holds barely. The margin term penalizes those near-threshold deletions, keeping supervision that more robustly implies the constraints. We expect a small but non-zero gain, concentrated on prompts whose constraints are only weakly lexically entailed, confirming the margin term is doing real work rather than acting as a tiebreaker.

**vs. IGSP with binary entailment-margin (logit margin)** — On a case where the score-difference margin saturates near 1.0, the binary logit margin (difference in NLI logits before softmax) provides a non-saturated signal of marginal intent risk. We expect this ablation to match or slightly exceed the score-difference margin on style constraints, confirming that saturation is a real failure mode and that the calibrated binary gate is the safer design.

### What would falsify this idea

If IGSP's preference gain over fixed-bound P2E is uniform across constraint-dense and constraint-sparse prompt subsets rather than concentrated on the constraint-dense subset where the predicted failure mode lives, the mechanism story is wrong and the gain is attributable to something other than intent-gated search. Equivalently, if the post-hoc filter matches IGSP at equal admitted-pair yield, the per-step gate adds nothing over rejection; and if the calibration study shows the type-specific verifier has near-chance precision/recall on count and relation constraints, the soundness assumption fails and the gate cannot confine the search to the intent-satisfying region.

## References

1. Prune to Expand: Training Prompt Enhancers with Less to Imitate
