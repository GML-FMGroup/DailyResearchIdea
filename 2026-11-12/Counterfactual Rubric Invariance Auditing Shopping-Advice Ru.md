# Counterfactual Rubric Invariance: Auditing Shopping-Advice Rubrics for Attribute-Sensitivity

## Motivation

Rubric-based evaluation such as CARTBench freezes expert criteria and scores advice against them, but a rubric score is only meaningful if it reflects the substance of the advice rather than the demographic attributes of the advice-seeker. Because CARTBench's 1,328 criteria are authored against single instantiations of a request (one gender, one age, one household configuration), the rubric silently encodes which attributes were present at authoring time, so advice that is substantively equivalent but tailored to a different user can score differently — a fairness failure that the benchmark cannot see because it never varies the attribute. The structural root cause is that rubric scores are point estimates on one request realization, with no intervention on user attributes and hence no invariance requirement.

## Key Insight

A rubric score is a valid measure of advice substance only if it is invariant under attribute interventions that leave the request's decision-relevant content fixed, so the residual score variance across a matched counterfactual set is a direct, rubric-level estimate of attribute bias that no single-request evaluation can expose.

## Method

**What it is.** CRI (Counterfactual Rubric Invariance) is an evaluation procedure that takes a request from CARTBench, a candidate advice response, and the frozen rubric, and returns an attribute-bias-penalized rubric score. Its inputs are the original request, a designated set of interchangeable user attributes, and the response; its output is a scalar score plus a per-attribute sensitivity profile.

**How it works.**

```pseudocode
INPUT: request r, response a, rubric R (Core/Excellence/Safety criteria),
       attribute slots A = {gender, age, household}, interchangeability map I
OUTPUT: CRI score S, sensitivity profile {s_x for x in A}

# Phase 0: decision-equivalence certification (replaces entailment gate)
# For each attribute x and value v, a rewrite r_xv is certified decision-equivalent
# iff it passes BOTH:
#   (i) a constraint-preservation annotation: a human annotator (or a held-out
#       set of 200 rewrites verified by two domain experts, Cohen's kappa >= 0.8)
#       confirms that budget, product category, and occasion are unchanged;
#   (ii) a no-attribute control rewrite r_ctrl (paraphrase of r with no attribute
#        change) yields a score delta |score[r_ctrl] - score[r]| <= tau_ctrl,
#        where tau_ctrl = 0.05 (5% of rubric weight).
# Rewrites failing either check are excluded from C and reported as
# 'constraint-shift' cases; s_x is computed only on certified pairs.

# Phase 1: build matched counterfactual request set
C = {r}
for each attribute x in A:
    for each value v in I[x] \ {value of x in r}:
        r_xv = rewrite(r, x -> v)          # LLM rewrite, then
        if certified_decision_equivalent(r, r_xv):  # Phase 0 gate
            C.add(r_xv)

# Phase 2: rubric scoring of the SAME response under each counterfactual
for each r' in C:
    score[r'] = rubric_score(a, R, r')     # per-criterion pass/fail, weighted

# Phase 3: invariance test per attribute
for each x in A:
    s_x = max_{r', r'' in C differing only in x} |score[r'] - score[r'']|
S_base = score[r]
S = S_base - lambda * mean_x(s_x)          # lambda = 0.5, penalty weight

return S, {s_x}
```

Attribute values are drawn from an interchangeability map `I` that lists values which the request's decision-relevant constraints (budget, product category, occasion) do not depend on; the decision-equivalence certification (Phase 0) rejects rewrites that accidentally change those constraints. `rubric_score` reuses CARTBench's frozen criteria verbatim, so the only thing that changes across the set is the attribute.

**Why this design.** We chose a *matched counterfactual set with a decision-equivalence certification* over simply prompting the judge to "ignore demographics" because an instruction-following judge has no way to report the advice it would have given to a different user, accepting the cost that each request requires several rewrite calls plus a human-verified certification step. We chose a *max-over-pairs* sensitivity statistic over a mean or variance because bias is a worst-case property — a single attribute value that flips a safety criterion matters more than small average drift — accepting higher variance in the estimate for small sets. We chose to *penalize the original score subtractively* rather than re-rank responses by invariance alone, because a response that is invariant but substantively poor should not be rewarded, accepting that `lambda` becomes a tunable that trades substance against fairness. A domain expert would not describe this as a variant of CARTBench itself: CARTBench is a static rubric-scoring benchmark, whereas CRI is a *test-time intervention procedure* whose output is a property (invariance) that the CARTBench rubric, by construction, never measures.

**Why it measures what we claim.** The computational quantity `s_x = max |score[r'] - score[r'']|` measures *attribute bias in the rubric's verdict* because it assumes that the decision-equivalence certification (Phase 0) guarantees the counterfactual requests are decision-equivalent for the advice task; this assumption fails when a rewrite preserves surface entailment but changes an implicit constraint (e.g., "gift for my father" rewritten to "gift for my mother" shifts a plausible price range), in which case `s_x` reflects legitimate context-sensitivity rather than bias. The quantity `S_base` measures *advice substance* because it assumes the frozen rubric criteria are attribute-neutral as authored; this assumption fails when a criterion itself references a demographic (e.g., "age-appropriate"), in which case the penalty is applied to a rubric that is already biased. The final `S = S_base - lambda * mean_x(s_x)` measures *bias-adjusted advice quality* because it assumes attribute sensitivity and substance are separable penalties; this fails when a genuinely better response is also more attribute-tailored, in which case the penalty under-rewards correct personalization.

## Contribution

(1) A counterfactual-invariance evaluation procedure, CRI, that augments frozen rubric scoring with attribute interventions and an entailment-gated matched counterfactual set, producing a bias-penalized score and per-attribute sensitivity profile. (2) A design principle: rubric scores should be reported as invariance-tested intervals rather than point estimates, with the max-over-pairs statistic as the operative bias signal. (3) An attribute-interchangeability map and rewrite-gate pipeline for shopping requests that makes counterfactual rubric testing reproducible across the CARTBench retail domains.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | CARTBench cases with interchangeable attribute slots (n=500 cases, 3 attribute slots each) | Frozen rubrics plus ambiguous requests expose attribute sensitivity |
| Primary metric | CRI score S (bias-penalized rubric score) | Combines substance and invariance in one falsifiable scalar |
| Baseline | CARTBench frozen-rubric score (no intervention) | Tests whether invariance is already captured |
| Baseline | Judge-prompted "ignore demographics" scoring | Tests whether instruction suffices to remove bias |
| Baseline | Mean-sensitivity penalty variant | Tests whether max-over-pairs is the right statistic |
| Ablation-of-ours | CRI without decision-equivalence certification (Phase 0 removed) | Tests whether certification is load-bearing for validity |
| Calibration set | 200 hand-labeled rewrite pairs (2 domain experts, Cohen's kappa ≥ 0.8) for Phase 0 certification | Bounds false-accept rate of decision-equivalence check |

### Why this setup validates the claim

The central claim is that CRI detects and penalizes attribute bias in rubric verdicts that CARTBench's static scoring cannot see. The vanilla CARTBench baseline tests the sub-claim that frozen rubric scores are already attribute-invariant: if they were, CRI would subtract nothing and rankings would match, so any divergence between CRI and CARTBench rankings is direct evidence of unmeasured bias. The "ignore demographics" judge tests whether a naive instruction-following fix substitutes for matched counterfactuals: since such a judge cannot report what it would have said to a different user, its failure isolates the mechanism gap that motivates the counterfactual set. The decision-equivalence certification ablation tests whether the certification is load-bearing: without it, rewrites that shift implicit constraints should inflate sensitivity and misrank responses, so the ablation should degrade agreement with human bias judgments specifically on gate-relevant cases. The max-sensitivity variant tests the worst-case statistic choice. CRI score is the right metric because it is a scalar that both preserves substance ordering and encodes invariance, letting one number falsify the joint claim. The calibration set of 200 hand-labeled rewrite pairs provides an empirical bound on the false-accept rate of the decision-equivalence check, ensuring that the load-bearing assumption is measured rather than asserted.

### Expected outcome and causal chain

**vs. CARTBench frozen-rubric score** — On a case where a response gives safety-critical advice that flips when the user is described as elderly versus young, the baseline produces a single high rubric score because its rubric is evaluated once against one request and never varies the attribute. Our method instead re-scores the same response under matched counterfactuals and takes the max pairwise gap, so the safety-criterion flip surfaces as a large `s_x` and a penalized `S`. We expect a noticeable CRI-versus-CARTBench ranking divergence concentrated on cases with attribute-sensitive safety or excellence criteria, with near-parity elsewhere.

**vs. Judge-prompted "ignore demographics" scoring** — On a case where the judge is instructed to ignore demographics but the response's advice is tailored to a gendered price expectation, the baseline produces an unchanged score because the instruction changes the judge's stated policy, not the response being judged, and the judge cannot synthesize the counterfactual user. Our method instead materializes the counterfactual requests and re-scores the identical response, so the tailoring becomes observable as score drift. We expect the instruction baseline to show small or zero measured sensitivity overall while CRI shows a clear gap on the subset where tailoring is strongest.

**vs. Mean-sensitivity penalty variant** — On a case where one attribute value flips a safety criterion while other rewrites cause only minor drift, the baseline produces a small averaged penalty that lets the safety flip slip through because averaging dilutes a single worst-case pair. Our method instead takes the max over pairs, so the safety flip dominates the penalty. We expect the mean variant to under-rank responses with localized worst-case bias relative to CRI, with the gap concentrated on cases having one extreme attribute pair.

**vs. CRI without decision-equivalence certification** — On a case where a rewrite passes surface entailment but shifts an implicit constraint (e.g., father-to-mother price range), the uncertified variant includes the rewrite in the counterfactual set, so the resulting `s_x` conflates legitimate context-sensitivity with bias. Our method excludes such rewrites via the constraint-preservation annotation and the no-attribute control delta (tau_ctrl = 0.05), so `s_x` is computed only on certified pairs. We expect the ablation to show inflated sensitivity and misranking on gate-relevant cases, with degradation measured against the 200-pair human-annotated calibration set.

### What would falsify this idea

If CRI's gain over CARTBench were uniform across all subsets rather than concentrated on cases with attribute-sensitive criteria and on certified rewrites, the claimed causal chain would be wrong. Likewise, if removing the decision-equivalence certification left rankings unchanged, the certification would not be load-bearing and the invariance claim would collapse.

## References

1. CARTBench: Rubric-Based Evaluation of Shopping Advice for Complex and Ambiguous Requests with Timelines
