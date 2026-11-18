# Monotone Belief Tracking for Binary-Oracle Research Ideation

## Motivation

Binary-oracle ideation benchmarks such as Re-Search (E3, E6) treat each yes/no answer as an independent constraint on the evolving idea hypothesis, so the benchmark cannot distinguish genuine refinement from oscillation caused by inconsistent or redundant oracle feedback; the field's trajectory toward iterative research agents (E2, E4, E5) inherits this unmodeled sequential-feedback assumption, meaning no existing branch imposes a consistency invariant on how belief over candidate ideas updates across turns. The root cause is structural: the ancestor single-shot protocol never modeled sequential feedback, so neither the benchmark nor the LLM agents it evaluates maintain an explicit belief state whose support is constrained to shrink under each new binary observation.

## Key Insight

If each yes/no oracle answer is treated as a hard constraint on a maintained hypothesis set, then a monotone-support invariant turns every answer into a checkable shrinkage-or-identity operation, making order-consistency violations a direct, model-agnostic diagnostic that separates genuine ideation progress from feedback-driven oscillation.

## Method

**What it is.** We introduce **MBT** (Monotone Belief Tracker), a wrapper around the Re-Search Twenty-Questions protocol that maintains an explicit candidate-hypothesis set H_t ⊆ H_0 and, after every binary oracle answer a_t, applies a constraint filter that must satisfy `|H_{t+1}| ≤ |H_t|`. MBT inputs the research problem, the LLM's question stream, the oracle answers, and a hypothesis generator; it outputs a per-turn belief state, a monotone-consistency log, and a diagnostic score separating progress from oscillation.

**Load-bearing assumption (stated explicitly).** MBT's diagnostic interpretation rests on the assumption A: *the binary oracle's yes/no answers are sound, stationary, context-independent predicates over a fixed hypothesis set H_0*, so that each answer genuinely partitions H_t and the surviving set's shrinkage (and its order-independence under shuffled replay) reflects ideation progress rather than artifacts of the oracle or of the seed generator's coverage. Assumption A fails when the same question means different things at different turns t (context-dependent question semantics), when the oracle is noisy, or when the hypothesis space is misspecified; under failure mode F, the order-consistency violation count V reflects oracle non-stationarity rather than model oscillation, and `|H_t|` reflects constraint-filtering artifacts rather than genuine narrowing. MBT is designed to make A checkable rather than assumed: the predicate vocabulary below fixes question semantics at t=0, and the calibration arm in the experiment measures how much of V is attributable to oracle non-stationarity versus model behavior.

**How it works.**

```pseudocode
Input: research problem P, hypothesis generator G, oracle O, horizon T=100
Output: belief trace {H_t}, consistency log V, diagnostic scores

0. Predicate vocabulary: at t=0, fix a set of K=16 atomic predicates
   Π = {π_1,...,π_16} declared by G(P) (e.g., "uses retrieval",
   "requires training", "targets low-resource setting", ...).
   Each hypothesis h ∈ H_0 is represented as a bit-vector b(h) ∈ {0,1}^16
   over Π. Each question q_t must be a predicate π_k ∈ Π or a boolean
   combination of predicates drawn from Π; questions outside Π are
   rejected and logged as out-of-vocabulary (OOV) turns.

1. H_0 ← G(P, n=32)                  # seed candidate ideas, n=32
   # each seed is converted to b(h) by a fixed template: for each π_k,
   # the generator answers "does this idea satisfy π_k?" with yes/no;
   # the resulting 16-bit vector is validated on a 64-idea held-out set
   # (agreement with human bit-vectors ≥ 0.85 required before use).
2. for t = 1..T:
3.     q_t ← LLM(P, H_t)            # question chosen by the model under test;
   #   if q_t ∉ vocabulary(Π), log OOV and skip to next t
4.     a_t ← O(q_t) ∈ {yes,no}       # binary oracle answer
5.     H_t^yes ← {h ∈ H_t : b(h) ⊨ q_t}    # bit-vector check, exact
6.     H_t^no  ← {h ∈ H_t : b(h) ⊭ q_t}    # bit-vector check, exact
7.     H_{t+1} ← H_t^yes if a_t=yes else H_t^no
8.     if |H_{t+1}| > |H_t|: log violation (should be empty by construction)
9.     if |H_{t+1}| = |H_t|: log stall turn
10.    if H_{t+1} = ∅: log contradiction; reset H_{t+1} ← H_t, flag q_t as inconsistent
11. V ← order-consistency check: replay answers in shuffled order (5 random
   permutations), compare final H_T across permutations; V = number of
   permutations whose final support differs from the forward-pass support.
12. Stationarity test: for each turn t, re-ask q_t at turn t+10 (if t+10 ≤ T)
   and record whether a_{t+10}(q_t) = a_t(q_t); non-stationary turns are
   flagged and excluded from the V interpretation.
13. return {H_t}, V, progress=|H_0|/|H_T|, stall_rate, contradiction_rate,
   oov_rate, nonstationary_rate
```

Step 5–7 is the monotone filter: each answer partitions H_t via exact bit-vector entailment, and we keep only the side the oracle selected. Step 11 is the diagnostic: if the oracle's answers are mutually consistent, any permutation of the same answers must yield the same final support; violations quantify how much the observed trajectory was driven by inconsistent feedback rather than by the model's questions. Step 12 is the stationarity test that flags turns where the same question re-asked yields a different answer, operationalizing the check on assumption A.

**Why this design.** We chose a hard set-membership filter over a probabilistic posterior because hard constraints make the monotone invariant checkable exactly (`|H_{t+1}| ≤ |H_t|` holds by construction), accepting the cost that hypotheses must be representable as checkable predicates rather than free text; we chose an explicit hypothesis generator G with n=32 seeds over letting the LLM under test maintain its own belief, because the diagnostic must be model-agnostic and not confounded by the evaluated model's self-evaluation failures that E2 and E7 document, accepting the cost that G's coverage bounds the achievable progress; we chose order-shuffled replay (step 11) over a single forward pass because order-consistency is the only observable that distinguishes oscillation from refinement without ground-truth idea labels, accepting the cost of O(T) extra oracle calls per trace. A reviewer might describe this as a Bayesian filter or a constraint-satisfaction wrapper, but neither maintains the monotone-support invariant as an explicit, logged, model-agnostic diagnostic — standard Bayesian updating can expand support under noisy likelihoods, and CSP solvers do not log order-consistency violations as an ideation diagnostic.

**Why it measures what we claim.** The quantity `|H_t|` measures ideation progress because the assumption that each oracle answer is a sound constraint on the true idea implies that the surviving set shrinks monotonically as information accumulates; this assumption fails when the oracle is noisy or the hypothesis space is misspecified, in which case `|H_t|` reflects constraint-filtering artifacts rather than genuine narrowing. The order-consistency violation count V measures feedback consistency because the assumption that a consistent oracle induces an order-independent constraint system implies that any permutation of answers yields the same final support; this assumption fails when the oracle's answers are context-dependent (e.g., the same question means different things at different t), in which case V reflects oracle non-stationarity rather than model oscillation. The stall rate (fraction of turns with `|H_{t+1}|=|H_t|`) measures question informativeness because the assumption that an informative question partitions H_t non-trivially implies that a stall indicates a redundant or uninformative question; this assumption fails when H_t is already a singleton, in which case stall rate reflects convergence rather than poor questioning. The non-stationary rate (step 12) measures the extent to which assumption A holds on a given trace, because the assumption that a stationary oracle returns the same answer to the same question at different turns implies that re-asking q_t at t+10 should yield a_t; this assumption fails when question semantics drift across turns, in which case the non-stationary rate flags the trace as one where V cannot be interpreted as model oscillation.

## Contribution

['MBT, a monotone belief tracker that wraps the Re-Search binary-oracle protocol with an explicit candidate-hypothesis set and a hard constraint filter guaranteeing `|H_{t+1}| ≤ |H_t|` by construction.', 'An order-consistency diagnostic that replays oracle answers in shuffled order to separate genuine ideation progress from oscillation driven by inconsistent binary feedback, yielding three per-trace scores: progress ratio, stall rate, and contradiction rate.', "A model-agnostic evaluation protocol that decouples the evaluated LLM's question-asking from belief maintenance, enabling comparison of ideation progress across models without confounding from self-evaluation failures."]

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Re-Search 200-paper ICML problem set | Same protocol MBT wraps; isolates tracker effect |
| Dataset (calibration) | Synthetic 200-problem set with known ground-truth idea and injected oracle noise at rates {0%, 5%, 10%, 20%} | Measures how V and |H_t| shrinkage degrade with oracle noise |
| Primary metric | Order-consistency violation count V (5 random permutations) | Directly diagnoses oscillation vs refinement without labels |
| Secondary metric | Non-stationary rate (step 12) | Flags traces where assumption A fails |
| Baseline | Raw Re-Search LLM question stream | Tests whether belief tracking adds signal beyond generation |
| Baseline | Probabilistic Bayesian belief filter | Tests whether hard monotone filter beats soft updating |
| Baseline | Self-evaluating LLM belief tracker | Tests whether model-agnostic tracking avoids self-eval failure |
| Ablation-of-ours | MBT without order-shuffled replay | Isolates diagnostic value of order-consistency logging |
| Ablation-of-ours | MBT with free-text entailment (LLM-judged h ⊨ q_t) instead of bit-vector vocabulary | Isolates the effect of exact predicate representation |

### Why this setup validates the claim

The central claim is that an explicit monotone-support filter, wrapped around the Re-Search protocol, yields a model-agnostic diagnostic that separates genuine ideation progress from oscillation, and that this separation is not obtainable from the raw question stream or from soft updating. The Re-Search corpus is the right dataset because it supplies a fixed problem, a binary oracle, and a 100-question horizon — exactly the inputs MBT consumes — so any measured difference is attributable to the tracker rather than to problem difficulty. The synthetic calibration set is the right complement because it supplies a known ground-truth idea and controlled oracle noise, so the correlation between V / |H_t| shrinkage and true progress can be measured directly rather than inferred; calibration curves across noise rates show how the diagnostic degrades as assumption A is violated. The raw-stream baseline tests the sub-claim that tracking adds information beyond what the model's questions alone reveal; the Bayesian baseline tests the sub-claim that hard set-membership is necessary for an exactly checkable invariant; the self-evaluating baseline tests the sub-claim that model-agnostic tracking is required given documented self-evaluation failures. The ablation removing order-shuffled replay isolates the diagnostic mechanism itself; the ablation replacing bit-vector vocabulary with free-text entailment isolates the contribution of exact predicate representation to the monotone invariant. Order-consistency violations are the right primary metric because they are the only observable in this protocol that distinguishes oscillation from refinement without ground-truth idea labels — the very gap the Re-Search analysis identifies — and the non-stationary rate is the right secondary metric because it flags the traces on which that interpretation is licensed.

### Expected outcome and causal chain

**vs. Raw Re-Search LLM question stream** — On a case where the model over-exploits one direction after turn 30, the baseline produces a flat progress trace that looks like convergence because it has no explicit belief state to reveal that |H_t| has stopped shrinking. Our method instead logs stall turns and contradiction resets against a concrete candidate set, so we expect a clear separation on the stall-rate and contradiction-rate curves concentrated in the post-30-question regime, with parity in the first 30 turns where nearly all progress occurs.

**vs. Probabilistic Bayesian belief filter** — On a case where the oracle gives a noisy or context-dependent answer, the Bayesian baseline produces support that expands or drifts because soft likelihood updates permit non-monotone posterior mass, masking the inconsistency as legitimate uncertainty. Our method instead enforces |H_{t+1}| ≤ |H_t| by construction and flags the offending question in the consistency log, so we expect MBT to show near-zero monotone violations while the Bayesian filter accumulates them, with the gap widening as oracle noise increases; on the synthetic calibration set we expect the Bayesian filter's V to rise monotonically with injected noise rate while MBT's V stays near zero until the noise rate exceeds the fraction of H_0 excluded by the noisy predicate.

**vs. Self-evaluating LLM belief tracker** — On a case where the model must judge whether its own candidate idea is still viable, the baseline produces self-confirming belief updates because it inherits the self-evaluation failure documented in the ideation literature, so its progress trace overstates narrowing. Our method instead derives belief purely from oracle answers against an external generator, so we expect MBT's progress estimate to diverge from the self-evaluating tracker specifically on turns where the model's questions are uninformative, visible as a higher stall rate for MBT that the self-evaluator conceals.

**vs. MBT with free-text entailment (ablation)** — On a case where a question is ambiguous in free text (e.g., "does it involve retrieval?" where some hypotheses use retrieval at training time and others at inference), the free-text entailment baseline produces inconsistent partitions because the LLM's judgment of h ⊨ q_t varies across calls, so its monotone invariant is violated by construction. Our method instead uses exact bit-vector checks against the fixed vocabulary Π, so we expect the free-text ablation to show non-zero monotone violations and higher V than full MBT, with the gap concentrated on OOV and ambiguous-predicate turns.

### What would falsify this idea

If MBT's order-consistency violation count is indistinguishable from the ablation's single forward pass across all subsets, or if its advantage over the raw stream is uniform rather than concentrated in the post-30-question oscillation regime, then the monotone filter is not measuring what we claim and the central diagnostic claim is wrong. Additionally, if on the synthetic calibration set V does not rise with injected oracle noise rate while the non-stationary rate does, then V is not measuring feedback consistency as claimed.

## References

1. Re-Search: Dissecting LLM Ideation Through a Two-Player Questioning Game
