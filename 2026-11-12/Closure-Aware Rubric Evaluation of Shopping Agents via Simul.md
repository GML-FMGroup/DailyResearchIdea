# Closure-Aware Rubric Evaluation of Shopping Agents via Simulated Substitution

## Motivation

Rubric-based shopping-advice benchmarks such as CARTBench score each criterion against a single frozen product record set and a fixed reply, so a model that names an unavailable product and stops still earns partial credit for the criterion it nominally addressed. The structural root cause is that rubrics are evaluated pointwise over a static action set: no criterion forces the action set to remain closed when a recommended item is removed, so dynamic recovery — the ability to substitute an alternative satisfying the same need — is never measured. This makes the reported understanding-to-execution gap (none of the evaluated models exceeding 45% on Excellence) silent about the failure mode most consequential in real shopping advice: the recommended product is out of stock.

## Key Insight

A shopping agent's competence is a property of its action set under intervention, not of its single realized recommendation: scoring the same rubric on the model's response to a product-removal intervention, and requiring the substitute to re-satisfy the exact criteria the removed product satisfied, converts a static pointwise rubric into a closure test that no single-response rubric can express.

## Method

**What it is.** CASR (Closure-Aware Substitution Rubric) takes a CARTBench-style case (dialogue, product records, frozen rubric criteria) plus an agent's initial reply, and outputs a rubric score plus a *closure score*: the fraction of rubric-satisfying recommendations for which the agent, when the recommended product is marked unavailable, proposes a substitute that re-satisfies the same criteria under the same frozen rubric.

**Load-bearing assumption (stated explicitly).** CASR assumes that the frozen CARTBench rubric criteria credited to a removed product are *need-complete and product-agnostic enough* that re-crediting those same criteria to a substitute product is a valid measure of need preservation — i.e., that criterion re-satisfaction is equivalent to genuine substitution. This assumption is not taken on faith: it is made operational by a per-criterion substitutability partition (step 4a below) and calibrated against human-judged substitute pairs (step 0 below). When a criterion is product-bound (names a brand, size, or SKU), re-crediting it to a substitute is impossible by construction, so closure(p) would be forced to 0 for reasons unrelated to the agent's recovery policy; the partition excludes such criteria from the closure mean and reports them separately as an explicit *non-substitutable* category.

**How it works.**

```pseudocode
Input: case c (dialogue D, product records P, frozen criteria R = Core ∪ Excellence ∪ Safety)
       agent A, substitution budget k=3, availability-perturbation set U (single-item removal)
       per-criterion substitutability flag sub(r) ∈ {need-level, product-bound}

0. CALIBRATION (held-out, before scoring):
   - Sample 200 (removed product p, credited criterion r, human-proposed substitute q) triples
     from a held-out set of CARTBench cases disjoint from the evaluation split.
   - Two annotators label each triple: does q serve the same need as p under r? (binary)
   - Report Cohen's κ between annotators; retain only criteria whose triples reach κ ≥ 0.6
     as need-level; flag the rest product-bound. This yields sub(r) for every criterion r.
   - Report the agreement rate as an upper bound on closure validity.

1. r0 = A(D)                                    # initial reply
2. S0 = extract_products(r0) ∩ P                # products named in r0
3. base_score(r0) = rubric_score(r0, R)         # standard CARTBench scoring
4. for each p in S0 that is credited by some criterion r in R:
4a.    R_p = { r ∈ R : r credited to p in r0 }
       R_p^need = { r ∈ R_p : sub(r) = need-level }
       R_p^bound = { r ∈ R_p : sub(r) = product-bound }
       if R_p^need = ∅: record p as non-substitutable; skip closure(p) (excluded from mean)
5.     P_p = P \ {p}                            # intervene: remove p from catalog
6.     D_p = D ⊕ "note: product p is now unavailable"   # single-turn availability notice
7.     r_p = A(D_p)                             # agent must re-recommend
8.     S_p = extract_products(r_p) ∩ P_p
9.     closure(p) = 1 if (S_p ≠ ∅ and every criterion r in R_p^need
                          is credited to some q in S_p under rubric R) else 0
10. closure_score = mean over p with R_p^need ≠ ∅ of closure(p)
    non_substitutable_rate = |{p : R_p^need = ∅}| / |{p : p credited}|  (reported separately)
11. final = (base_score, closure_score, non_substitutable_rate, per-criterion closure breakdown)
```

Hyperparameters: substitution budget k=3 (max retries before scoring closure(p)=0), single-item removal only (no combinatorial unavailability), availability notice appended as a single user turn, calibration sample size 200 triples, κ threshold 0.6 for need-level retention.

**Why this design.** We chose *counterfactual catalog intervention* over prompt-only "what if it were unavailable" questions because a prompt-only probe lets the model answer hypothetically without committing to a product record, which is exactly the failure CARTBench already cannot see; the cost is that we must re-run the agent once per credited product, multiplying inference cost by |S0|. We chose *criterion-level re-satisfaction* over embedding similarity between the original and substitute product because similarity is a proxy that rewards near-duplicates (same brand, different size) that may violate a Core criterion (e.g., budget), while criterion re-satisfaction is scored on the same frozen rubric and therefore inherits its validity; the cost is reliance on the frozen rubric being complete enough to distinguish substitutes, which we address by the substitutability partition and calibration in steps 0 and 4a. We chose *single-item removal* over multi-item or probabilistic stock simulation because single-item removal yields an unambiguous per-product closure bit that can be attributed to one decision, whereas multi-item removal confounds failures across products; the cost is that we under-measure closure under correlated stockouts. Unlike Toolformer/Gorilla-style retrieval, which selects the nearest catalog item to a query, CASR does not retrieve a substitute for the model — it removes an item and scores whether the model's *own* re-recommendation re-satisfies the rubric, so the measured quantity is the agent's recovery policy, not a nearest-neighbor lookup.

**Why it measures what we claim.** `closure(p)` measures *dynamic recovery ability* because removing p from P makes p infeasible while holding the need fixed, so any rubric-credited substitute q is evidence the agent conditioned on the intervention rather than replaying its prior answer; this assumes the availability notice is the only information change between r0 and r_p, and the assumption fails when the agent's re-recommendation is driven by conversational priming (it repeats p's attributes) rather than the removal, in which case closure(p) reflects answer-copying rather than recovery. `criterion re-satisfaction` measures *need preservation* because the frozen rubric criteria credited to p encode the need p was serving, so re-crediting them to q is equivalent to q serving the same need; this assumes the rubric criteria are need-complete and product-agnostic, and the assumption fails when a criterion is product-specific (e.g., names a brand), in which case the metric reflects literal criterion matching rather than need substitution — which is why step 4a partitions R_p into need-level and product-bound subsets and excludes the latter from the closure mean. `mean closure over credited products` measures *action-set closure under failure substitution* because it averages the per-product closure bits that each represent one removal event; this assumes removals are independent, and the assumption fails under correlated stockouts, in which case the mean reflects single-failure robustness rather than full closure.

## Contribution

(1) A closure criterion and scoring procedure (CASR) that augments rubric-based shopping evaluation with catalog-intervention closure tests, producing a per-product closure score alongside the standard rubric score. (2) A design principle: substitution validity should be defined by re-satisfaction of the same frozen rubric criteria rather than by product similarity, which makes recovery measurable without a new annotation pass. (3) An analysis of which rubric categories (Core vs. Excellence vs. Safety) are most often violated by substitutes, exposing where models recover nominally but break safety or budget constraints.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | CARTBench shopping-advice cases with frozen rubrics | Rubric-credited products enable per-product removal intervention |
| Primary metric | Closure score (mean per-product closure bits over need-level criteria only) | Directly measures recovery, not static advice quality |
| Secondary metric | Non-substitutable rate (fraction of credited products with only product-bound criteria) | Reports rubric expressiveness separately from agent recovery |
| Baseline | GPT-6.1-Sol under standard CARTBench scoring | Tests whether scale alone yields closure |
| Baseline | Claude-Opus-5 under standard CARTBench scoring | Tests whether a second frontier model closes by default |
| Baseline | Prompt-only "what if unavailable" probe | Tests counterfactual-intervention necessity |
| Ablation-of-ours | CASR with embedding-similarity substitutes | Tests criterion re-satisfaction versus near-duplicate proxy |
| Control | Sham-removal (remove a non-credited product) | Certifies closure is driven by removal, not priming |
| Validation | Human-judged substitute pairs (200 triples, 2 annotators, κ ≥ 0.6) | Calibrates need-level vs product-bound criteria |

### Why this setup validates the claim

The central claim is that static rubric scores hide a recovery gap, and that only a committed catalog intervention with criterion-level re-satisfaction exposes it. CARTBench is the right dataset because its frozen Core/Excellence/Safety criteria give us both the products to remove and the criteria to re-credit, so every closure bit is attributable to a single decision. The two frontier baselines test the sub-claim that high base_score does not imply high closure: if strong models already recover, closure will track base_score and our framing collapses. The prompt-only probe isolates the mechanism claim — that hypothetical questioning lets the model answer without committing to a product record, so it should overstate closure relative to actual removal. The similarity ablation isolates the scoring claim: if near-duplicate substitutes score as well as criterion re-satisfaction, the rubric-based closure bit adds nothing over a cheap proxy. The sham-removal control isolates the priming confound: removing a non-credited product should leave closure near 1, certifying that the availability notice — not answer-copying — drives the measured closure. The human-judged calibration study bounds the load-bearing assumption: if need-level criteria cannot be reliably identified (κ < 0.6), the closure mean is uninterpretable and we report the non-substitutable rate as the honest ceiling. Because closure is a per-product bit averaged over credited products with need-level criteria, the predicted effect is a specific divergence between base_score rank and closure rank, concentrated on cases whose credited products carry Core criteria that near-duplicates would violate — exactly the pattern that would be invisible to a single aggregate score.

### Expected outcome and causal chain

**vs. GPT-6.1-Sol (standard CARTBench scoring)** — On a case where the agent recommends a budget-constrained laptop and the rubric credits a Core price criterion, removing that laptop leaves the model replaying its prior attributes (same brand, same tier) and naming a near-duplicate that breaches the budget. Its mechanism gap is that static scoring never forces a re-commitment under infeasibility, so the failure is invisible. Our method removes the item and re-scores the model's own substitute against the same frozen criteria, so a budget-breaching near-duplicate yields closure 0. We expect a noticeable base_score-versus-closure gap on Core-heavy cases, with parity on cases whose criteria are product-agnostic.

**vs. Claude-Opus-5 (standard CARTBench scoring)** — On a case where the request is ambiguous and the agent hedges across two products, the baseline's high rubric score comes from partial credit spread over both, and removing either one produces a reply that restates the removed product's virtues rather than re-grounding in the catalog. Its mechanism gap is that conversational priming dominates the availability notice, so closure reflects answer-copying. Our method attributes closure per credited product and flags the copy pattern as closure 0; the sham-removal control confirms the copy pattern is specific to removing a credited product. We expect Opus to show high base_score but a closure gap concentrated on multi-product, ambiguous cases, with strong closure on single-product Safety cases.

**vs. Prompt-only "what if unavailable" probe** — On a case where the model is asked hypothetically whether its pick could be substituted, it answers fluently with a plausible alternative that is never checked against the catalog, so the probe reports near-ceiling closure. Its mechanism gap is that no product record is removed, so the model can satisfy the question without committing. Our method physically removes the item and requires a re-recommendation from the surviving catalog, so ungrounded substitutes score closure 0. We expect the probe to overstate closure by a wide margin, with the largest divergence on cases where the hypothetical substitute is out-of-catalog or violates a Core criterion.

**vs. CASR with embedding-similarity substitutes (ablation)** — On a case where the credited product is a specific size or brand, similarity scoring rewards a same-brand different-size item that breaches a Core criterion, so the ablation reports closure where none exists. Its mechanism gap is that similarity is a proxy that cannot see rubric violations. Our criterion re-satisfaction scores the substitute on the same frozen rubric, so the breach is caught; the need-level partition ensures this comparison is made only on criteria where substitution is well-defined. We expect the ablation to inflate closure specifically on product-specific-criterion cases while matching ours elsewhere, confirming that the gain comes from criterion-level scoring.

**vs. Sham-removal control** — On a case where a non-credited product is removed, the agent's reply should be unchanged in its credited recommendations, so closure should be near 1. Its mechanism gap is that if closure drops under sham removal, the measured closure reflects priming or generic perturbation rather than recovery from the credited product's removal. Our method reports the sham-removal closure as a control; a large gap between sham and real removal certifies the intervention is doing the work. We expect sham closure near 1 across models, with the real-removal closure gap concentrated on Core-heavy and product-specific-criterion cases.

### What would falsify this idea

If closure scores are uniformly high across all baselines and track base_score almost perfectly, with no concentration of the gap on Core-heavy and product-specific-criterion cases, then the claimed recovery gap does not exist and the intervention adds nothing over static rubric scoring. Likewise, if the prompt-only probe matches actual-removal closure, or if sham-removal closure is as low as real-removal closure, the counterfactual-intervention mechanism is unnecessary; and if human-judged calibration yields κ < 0.6 for most criteria, the closure mean is uninterpretable and the non-substitutable rate is the honest ceiling.

## References

1. CARTBench: Rubric-Based Evaluation of Shopping Advice for Complex and Ambiguous Requests with Timelines
