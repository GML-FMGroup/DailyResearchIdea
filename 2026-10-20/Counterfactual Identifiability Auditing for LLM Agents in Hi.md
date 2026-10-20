# Counterfactual Identifiability Auditing for LLM Agents in Hidden-Objective Operations Research

## Motivation

Existing benchmarks for LLM agents in operations research evaluate only the final proposed objective and therefore cannot distinguish a correct inference of the client's latent objective from a lucky guess that happens to produce an acceptable solution. OR-License introduced hidden client objectives and a hidden feasibility check, but its DAQ score conflates inference quality with solution quality: an agent that never elicits the discriminating constraint can still score highly if its default model coincides with the expert solution. The root cause is structural — no existing benchmark verifies that the information the agent elicited is sufficient to uniquely determine the true objective, so the inference process itself is never tested.

## Key Insight

A proposed objective is evidence of correct inference only if no alternative objective consistent with all elicited information is also consistent with the hidden client constraints; identifiability is a property of the elicited information set, not of the final solution, and can therefore be tested by counterfactual enumeration against the simulator's hidden constraint set.

## Method

# CACHE: Counterfactual Auditing of Client-inferred Hidden-objective Equivalence
# Input: ORSim instance (hidden constraints H, hidden expert objective f*, DSL grammar G), agent trajectory T
# Output: identifiability verdict I in {UNIQUE, AMBIGUOUS, UNIDENTIFIABLE, GRAMMAR_MISS}, witness set W, coverage certificate

1. EXTRACT(T) -> (Q_asked, E_elicited, f_hat)
   # Q_asked: set of question/answer pairs with ORSim
   # E_elicited: the business facts the simulator revealed (each tagged with the constraint it activates)
   # f_hat: agent's proposed objective, parsed into the instance's DSL grammar G

2. GRAMMAR_COVERAGE_CHECK(f*, G) -> coverage_flag in {IN_GRAMMAR, OUT_OF_GRAMMAR}
   # Simulator-side check: does the true expert objective f* parse into G?
   # If OUT_OF_GRAMMAR, emit GRAMMAR_MISS verdict and skip steps 3-5 (report grammar-coverage diagnostic)

3. BUILD_CONSISTENT_SET(E_elicited, H, G) -> C
   # Enumerate objectives f in G such that:
   #   (a) every revealed fact in E_elicited is satisfied by f's feasible region
   #   (b) f is realizable within the instance's parameter bounds (demand, capacity, cost ranges)
   # Use constraint propagation over G with enumeration budget B=2000 (default);
   # prune by dominance (objectives with identical argmin on all feasible data are collapsed to one equivalence class)

4. FILTER_BY_HIDDEN(C, H) -> C_star
   # Keep only f in C whose optimum also satisfies every hidden constraint in H
   # (H is the simulator's ground-truth constraint set never revealed to the agent)

5. TEST_UNIQUENESS(C_star, f_hat) -> (I, W)
   # I = UNIQUE iff the equivalence class of f_hat is the only member of C_star
   # I = AMBIGUOUS iff |C_star| > 1 and f_hat's class is in C_star but not alone
   # I = UNIDENTIFIABLE iff f_hat's class not in C_star or C_star empty (agent's info does not pin any objective)
   # W = up to 3 witness objectives from C_star \ {f_hat} that agree on E_elicited but differ on H
   # For each witness w in W, tag difference_type(w, f_hat) in {CONSTRAINT_ACTIVATION, PARAMETER_BOUND}

6. BUDGET_SENSITIVITY(C_star, B) -> stability_flag
   # Re-run step 3 with B in {500, 2000, 8000}; if |C_star| changes across budgets, mark UNIQUE as unstable
   # Report stability_flag alongside I; only assert UNIQUE when stable under budget doubling

7. SCORE(I, W, T, coverage_flag, stability_flag) -> audit report
   # Report per-instance identifiability class, witness count with difference_type tags,
   # coverage_flag, stability_flag, and the minimal set of unasked discriminating questions
   # (computed by re-running step 3 with each candidate question's answer added, measuring entropy reduction in C_star)

**Why this design.** We chose counterfactual enumeration over a learned discriminator because the simulator already exposes the hidden constraint set H, which gives an exact ground-truth filter that a learned head would only approximate; the cost is that enumeration is bounded by B=2000 and may miss rare objectives outside the DSL grammar. We chose equivalence-class pruning by dominance over exact objective comparison because many distinct parameterizations produce identical argmins on the instance's data, and treating them as distinct would inflate the ambiguous count; the cost is that two objectives that differ only on out-of-distribution data are collapsed, which is acceptable because the benchmark evaluates in-distribution behavior. We chose to report the minimal set of unasked discriminating questions rather than a scalar identifiability score because a scalar would hide which piece of information was missing, and the audit's value is diagnostic; the cost is that the report is multi-dimensional and harder to aggregate across instances. This differs from OR-License's DAQ, which averages expert-relative solution quality and cannot distinguish a lucky guess from a correct inference; a domain expert would not describe CACHE as a variant of DAQ because DAQ scores the output artifact while CACHE scores the sufficiency of the elicitation process.

**Why this measures what we claim.** The size of C_star (the set of objectives consistent with elicited information and hidden constraints) measures identifiability of the latent objective because the assumption that the DSL grammar G spans the space of plausible client objectives makes enumeration exhaustive over that space; this assumption fails when the true objective uses a constraint form outside G, in which case C_star is empty and the verdict reflects grammar coverage rather than agent inference (we detect this via GRAMMAR_COVERAGE_CHECK and emit GRAMMAR_MISS). The uniqueness test on f_hat's equivalence class measures correct inference versus lucky guessing because the assumption that equivalence-class pruning preserves behavioral distinguishability makes class-membership equivalent to behavioral identity on the instance data; this assumption fails when two objectives are behaviorally identical on all feasible data but differ in the client's stated preference, in which case the verdict reflects behavioral non-identifiability rather than a failure of the agent. The witness set W measures the discriminating power of the unasked questions because the assumption that each witness differs from f_hat only on constraints activated by unanswered questions makes the witness-to-question mapping one-to-one; this assumption fails when a witness differs due to parameter bounds rather than constraint activation, in which case W reflects parameter ambiguity rather than information gap (we tag each witness with difference_type to distinguish these).

## Contribution

["CACHE, a counterfactual identifiability audit for LLM agents in hidden-objective settings that classifies each agent trajectory as UNIQUE, AMBIGUOUS, or UNIDENTIFIABLE by enumerating objectives consistent with elicited information and filtering against the simulator's hidden constraint set.", 'A diagnostic report that names the minimal set of unasked discriminating questions per instance, turning the audit from a scalar score into an actionable signal about which information the agent failed to elicit.', "An empirical design principle: identifiability is separable from solution quality, so benchmarks that score only the final objective (such as OR-License's DAQ) systematically over-credit agents whose default models happen to coincide with the expert solution."]

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | ORSim instances from OR-License | Hidden constraints and expert objective are ground-truth exposed |
| Primary metric | Identifiability-verdict accuracy vs. ground truth (UNIQUE/AMBIGUOUS/UNIDENTIFIABLE/GRAMMAR_MISS) | Directly tests sufficiency of elicitation, not solution luck; includes grammar-coverage diagnostic |
| Baseline | DAQ (expert-relative solution quality) | Tests whether output-quality scoring misses lucky guesses |
| Baseline | DAQ + feasibility-only check | Tests whether feasibility alone conflates inference with guessing |
| Baseline | Elicitation-coverage heuristic (question count) | Tests whether more questions implies better inference |
| Ablation-of-ours | CACHE without hidden-constraint filter (C instead of C_star) | Isolates contribution of ground-truth H filtering |
| Ablation-of-ours | CACHE without dominance pruning (exact objective comparison) | Quantifies verdict inflation from equivalence-class collapsing |
| Ablation-of-ours | CACHE with fixed B=2000 only (no budget sensitivity) | Tests whether truncation-induced false uniqueness affects verdict accuracy |

### Why this setup validates the claim

The central claim is that CACHE distinguishes correct hidden-objective inference from lucky guessing, and that the identifiability verdict tracks the sufficiency of the elicitation process rather than the quality of the produced artifact. DAQ tests the opposite hypothesis: if output-quality scoring were sufficient, DAQ would correlate with identifiability verdicts and CACHE would add nothing. The feasibility-only baseline tests whether mere solution acceptance conflates a lucky-feasible guess with a correctly inferred objective, which is exactly the failure mode CACHE is designed to expose. The question-count heuristic tests whether raw elicitation volume predicts inference, isolating whether the *content* of elicited facts matters. The ablation without H-filtering removes the hidden-constraint filter, so if CACHE's verdict accuracy collapses to the ablated level, the gain is attributable to H-filtering rather than to enumeration or equivalence pruning. The ablation without dominance pruning tests whether equivalence-class collapsing inflates the ambiguous count, isolating the behavioral-distinguishability assumption. The fixed-budget ablation tests whether truncation at B=2000 produces false UNIQUE verdicts, isolating the enumeration-exhaustiveness assumption. Identifiability-verdict accuracy is the right primary metric because it is the only quantity that directly measures whether the agent's inferred objective is the unique consistent one, which is the paper's stated target.

### Expected outcome and causal chain

**vs. DAQ (expert-relative solution quality)** — On a case where the agent asks few questions but happens to land on a feasible solution matching the expert's argmin, DAQ produces a high score because it compares the output artifact to the expert solution and cannot see the elicitation gap. Our method instead reports AMBIGUOUS with witnesses in C_star, because the elicited facts leave multiple objectives consistent with H, so we expect a noticeable DAQ-accuracy divergence concentrated on instances with high DAQ but low identifiability, and parity on instances where the agent asked discriminating questions.

**vs. DAQ + feasibility-only check** — On a case where the agent's solution is feasible under hidden constraints but the agent's stated objective f_hat is not in C_star (it fits the revealed facts only by coincidence), the feasibility-only baseline accepts the solution because feasibility is the only gate. Our method reports UNIDENTIFIABLE because f_hat's equivalence class is absent from C_star, so we expect the feasibility-only baseline to over-accept on the subset where f_hat is behaviorally off but feasible, with our method showing a clear gap there and parity on instances where f_hat is genuinely in C_star.

**vs. Elicitation-coverage heuristic (question count)** — On a case where the agent asks many low-information questions that all probe the same constraint, the question-count heuristic predicts high identifiability because it treats volume as a proxy for information. Our method reports AMBIGUOUS because the elicited facts do not shrink C_star, and the minimal discriminating-question set identifies the unasked high-entropy-reduction questions, so we expect the heuristic to be near-chance on high-volume/low-discrimination instances while our verdict accuracy stays high, with parity on instances where question count happens to track discrimination.

**vs. CACHE without hidden-constraint filter (C instead of C_star)** — On a case where the agent's elicited facts are consistent with many objectives, the ablated version reports AMBIGUOUS because it lacks the ground-truth H filter that would eliminate objectives whose optima violate hidden constraints. Full CACHE uses H to shrink the set to C_star, potentially yielding UNIQUE when the hidden constraints disambiguate, so we expect the ablation to over-report AMBIGUOUS and under-report UNIQUE, with the gap concentrated on instances where H is informative.

**vs. CACHE without dominance pruning (exact objective comparison)** — On a case where many parameterizations produce identical argmins on the instance data, the ablated version treats each as distinct, inflating |C_star| and causing false AMBIGUOUS verdicts. Full CACHE collapses these into one equivalence class, so we expect the ablation to show inflated ambiguous counts and lower verdict accuracy, with the gap proportional to the number of behaviorally identical parameterizations in the instance.

**vs. CACHE with fixed B=2000 only (no budget sensitivity)** — On a case where the true objective lies near the enumeration boundary, the fixed-budget version may miss it and falsely report UNIQUE due to truncation. Full CACHE re-runs with B in {500, 2000, 8000} and only asserts UNIQUE when stable, so we expect the fixed-budget version to show false UNIQUE verdicts on instances where C_star changes with budget, and full CACHE to correct these, with the gap concentrated on instances with large C_star.

### What would falsify this idea

If CACHE's verdict accuracy gain over DAQ is uniform across all instances rather than concentrated on the subset where DAQ is high but identifiability is low, then the identifiability signal is not capturing the guessing-vs-inference distinction the claim rests on. Additionally, if the ablation without H-filtering matches full CACHE on verdict accuracy, then the hidden-constraint filter is not doing the work attributed to it and the mechanism claim fails.

## References

1. OR-License: Evaluating LLM Agents as Real-World Operations Research Practitioners
