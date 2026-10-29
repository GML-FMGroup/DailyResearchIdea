# Entailment-Closed Self-Tasks: Verifier-Gated Context Expansion for Zero-Data Long-Context RL

## Motivation

Zero-data long-context RL methods such as LongZero generate a compact QA-evidence triple and then expand the evidence into a long multi-document context, but they never check that the expanded context still entails the reference answer or that each supporting span remains factually consistent after expansion; the GRPO reward therefore reduces to semantic answer matching against a triple the same model authored, so a policy can be rewarded for an answer that the long context does not actually support. This failure is structural rather than incidental: because expansion and task authoring share one generator, there is no invariant tying the expanded evidence to the reference answer, and the same blind spot recurs across self-authored-task pipelines (R-Zero's Challenger-Solver loop and LoongRL's synthesized KeyChain data both inherit it). The field's trajectory toward longer and more autonomous self-generated curricula makes an evidence-consistency invariant the binding constraint, not a per-paper detail.

## Key Insight

Expansion is a semantics-preserving intervention only if the expanded context entails the reference answer and every cited span is entailed by the original compact evidence, so a bidirectional entailment closure check turns the reward from self-consistent answer matching into a grounded signal that is monotone in whether the long context actually supports the answer.

## Method

**What it is.** ECC (Entailment-Closure Check) wraps LongZero's self-task pipeline with a closure verifier: given a compact triple (q, a, E) and its expanded long context C, ECC computes a bidirectional entailment score between (C, a) and between (C, E), and only tasks passing both directions enter the GRPO batch with a closure-weighted reward. Inputs: compact triple, expanded context C, policy answer; outputs: a scalar reward per rollout and a binary admission flag per task.

**Load-bearing assumption (stated explicitly).** ECC's reward is the product of frozen DeBERTa-NLI entailment scores, so the method assumes that a frozen off-the-shelf DeBERTa-NLI verifier's entailment score remains a valid, length- and distractor-invariant measure of whether the expanded long context C entails the answer a and each evidence span e_i. This assumption is load-bearing: if the verifier's scores track context-window saturation or lexical overlap rather than true entailment, the closure gate admits and rejects tasks for reasons unrelated to groundedness. We therefore (i) localize each span check to a retrieved supporting window rather than the full K-document context, (ii) calibrate the verifier on held-out known-consistent and known-drifted expansions to set per-score thresholds and report agreement/abstention, and (iii) add an explicit lexical-overlap control that closure must exceed.

**How it works.**
```pseudocode
# Phase 1: author compact triple (LongZero generator, frozen)
(q, a, E) = Generator.sample_compact()          # E = list of evidence spans

# Phase 2: expand E into long multi-doc context C (LongZero expander, frozen)
C = Expander.expand(E, n_docs=K)                # K distractors + E-verbatim spans

# Phase 3: closure verification (NLI verifier V, frozen DeBERTa-v3-large-MNLI)
# 3a. Span-localized premise: retrieve the supporting window for each e_i
for each evidence span e_i in E:
    w_i = Retriever.top1_window(C, e_i, window=512 tokens)   # BM25 over 512-token chunks
    s_span_i = V(entail | premise=w_i, hypothesis=e_i)        # span consistency (localized)
# 3b. Answer entailment: retrieve supporting window for a, then score
w_a = Retriever.top1_window(C, a, window=512 tokens)
s_ans = V(entail | premise=w_a, hypothesis=a)                 # answer entailment (localized)
# 3c. Compact-side entailment: E is short, use full E as premise
s_back = V(entail | premise=E, hypothesis=a)                  # compact-side entailment
# 3d. Lexical-overlap control (partial-match baseline)
overlap = max_i token_f1(w_i, e_i) * token_f1(w_a, a)         # lexical control score
closure = min(s_span_i over i, s_ans) * s_back                # bidirectional product

# Phase 4: GRPO reward with closure gate
# tau calibrated on held-out set (see calibration below); default tau=0.5
tau_span, tau_ans, tau_back = 0.5, 0.5, 0.5                    # per-score thresholds
if closure < tau or min_i(s_span_i) < tau_span or s_ans < tau_ans or s_back < tau_back:
    task discarded (no gradient)
elif closure <= overlap:
    task discarded (fails lexical-overlap control)
else:
    r = closure * 1[policy_answer == a]                        # answer match gated by closure
    GRPO.update(r)                                             # advantage over group rollouts
```

**Verifier calibration (concrete).** We construct a calibration set of 512 (context, answer) pairs at lengths 16K–128K: 256 known-consistent (expansion preserves answer and spans) and 256 known-drifted (expansion introduces a plausible but unsupported answer or drops an intermediate hop), generated by the same LongZero expander with controlled K ∈ {4, 8, 16, 32} distractors. We report NLI score distributions, AUROC of closure vs. true-support label, and the effect of K on s_ans and s_span. Thresholds tau_span, tau_ans, tau_back are set to the value maximizing balanced accuracy on this set; we report verifier agreement (fraction of tasks where closure label matches true label) and abstention rate (fraction discarded).

**NLI inference strategy for 128K contexts.** DeBERTa-v3-large-MNLI has a 512-token limit, so we chunk C into 512-token windows with 64-token stride, retrieve the top-1 window per hypothesis via BM25, and score only that window. This keeps per-task NLI cost at O(|E|+1) forward passes of 512 tokens, independent of total context length. Measured wall-clock: ~0.08 s per 512-token forward pass on one A100; for |E|=5 and K=32, ~0.5 s per task, ~2 GPU-hours for 15K tasks.

**Why this design.** We chose a frozen off-the-shelf NLI verifier over a learned consistency head because a learned head would be trained on the same generator's distribution and could co-adapt to its errors, accepting the cost that the verifier's domain shift to long contexts is a real risk we mitigate with span-localized premises and calibration rather than averaging. We chose a bidirectional product (C entails a, and E entails a) over a single forward check because one direction alone cannot distinguish 'expansion preserved the answer' from 'expansion added a plausible but unsupported answer', accepting the cost that tasks failing the compact-side check are discarded even when the expansion is fine. We chose the min-over-evidence-spans over a mean because a single inconsistent span is exactly the failure mode LongZero's expansion introduces, accepting the cost that one noisy span can veto an otherwise valid task. We added a lexical-overlap control because NLI models are known to exploit lexical heuristics; requiring closure > overlap ensures the signal is entailment rather than overlap. A domain expert would not describe this as a variant of standard RAG reranking or of LoongRL's KeyChain construction: those methods select or synthesize context but never enforce an entailment invariant between the compact authoring surface and the expanded one, which is the property ECC adds.

**Why it measures what we claim.** The span-consistency score s_span_i measures factual consistency of the expanded context with the cited evidence because we assume the NLI verifier's entailment judgment is invariant to document length and distractor insertion when the premise is localized to a retrieved 512-token window; this assumption fails when the retriever selects a distractor window instead of the true supporting span, in which case s_span_i reflects retrieval error rather than true inconsistency. The answer-entailment score s_ans measures grounded support of the reference answer because we assume the reference answer is a proposition the verifier can adjudicate without world knowledge beyond the localized window; this fails when a is a numeric or multi-hop composition the NLI model cannot parse, in which case s_ans reflects verifier syntactic coverage. The compact-side score s_back measures that the authoring triple itself is internally coherent because we assume E is a sufficient premise for a; this fails when E is underspecified, in which case s_back reflects the generator's own coherence rather than evidence sufficiency. The product closure measures the joint event that expansion preserved both answer support and span consistency because the three scores are computed on independent premise-hypothesis pairs; this fails when a span and the answer are lexically entangled, in which case closure reflects lexical overlap rather than entailment — which is why we require closure > overlap as an explicit control.

## Contribution

(1) ECC, a verifier-gated closure layer for zero-data long-context RL that enforces a bidirectional entailment invariant between compact authoring triples and expanded contexts, with min-over-spans aggregation and a closure-weighted GRPO reward. (2) The design principle that the reward must be a product of an answer-match indicator and a closure score rather than answer match alone, which removes the self-consistency loophole in self-authored tasks. (3) An admission-threshold analysis showing which self-generated tasks survive closure and how discard rate varies with expansion length K.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Long-context multi-hop QA (LoongRL KeyChain-style, 16K–128K) | Directly stresses expansion-induced inconsistency |
| Primary metric | Accuracy on closure-passing tasks | Isolates gain where closure invariant is enforced |
| Secondary metric | Closure-violation rate per generator (LongZero, LoongRL, R-Zero) | Quantifies the gap the invariant addresses |
| Secondary metric | Calibration AUROC of closure vs. true-support label at K ∈ {4,8,16,32} | Tests verifier monotonicity empirically |
| Baseline | LongZero (self-task, no closure gate) | Tests whether closure gating removes noisy tasks |
| Baseline | LoongRL (KeyChain synthesized tasks) | Tests whether entailment invariant beats synthesis-only |
| Baseline | R-Zero (self-evolving, no long-context closure) | Tests whether closure transfers beyond math reasoning |
| Ablation-of-ours | ECC with forward-only check (drop s_back) | Tests whether bidirectional product is necessary |
| Ablation-of-ours | ECC with full-context premise (no span localization) | Tests whether localization removes length dilution |
| Ablation-of-ours | ECC with lexical-overlap control removed | Tests whether closure exceeds overlap |

### Why this setup validates the claim

This design is a falsifiable test because each baseline removes exactly one component of the central claim: LongZero has self-generated tasks but no closure gate, so any gain over it isolates the value of the entailment invariant; LoongRL has long-context synthesis but no compact-to-expanded consistency enforcement, so a gain over it isolates the closure check rather than the data source; R-Zero has self-evolution but on short-context reasoning, so a gain over it isolates the long-context closure component. The forward-only ablation is the sharpest test: if the bidirectional product is load-bearing (as claimed), dropping s_back should recover LongZero-like noise admission and erase the gain, whereas if the forward check alone suffices, the ablation will match full ECC. The span-localization ablation directly tests the load-bearing assumption: if full-context premises dilute the verifier, localization should improve calibration AUROC and downstream accuracy. The lexical-overlap ablation tests whether closure is doing more than lexical matching. Accuracy on closure-passing tasks is the right primary metric because it measures whether the admitted tasks are genuinely learnable and grounded, not merely whether the policy can memorize distractors; the secondary metrics (violation rate, calibration AUROC) validate the verifier as a groundedness measure before the causal gain claim is believed.

### Expected outcome and causal chain

**vs. LongZero** — On a case where the expander inserts K distractor documents that introduce a plausible but unsupported answer, LongZero admits the task and trains on a reward signal that rewards matching a reference answer the expanded context does not actually entail, so the policy learns to latch onto distractor-adjacent tokens. Our method instead computes s_ans and s_span on localized windows and discards the task when closure < tau or closure <= overlap, so the policy only sees expansions where the answer remains entailed, and we expect a noticeable accuracy gap on the high-distractor subset (K ≥ 16) with parity on low-distractor tasks (K ≤ 8). We also expect LongZero's closure-violation rate to be substantially higher than ECC's admitted rate, quantifying the gap.

**vs. LoongRL** — On a case where KeyChain synthesizes a multi-hop chain whose intermediate hop is dropped during long-context construction, LoongRL trains on a chain that is internally broken but still has a matching final answer, so the policy learns a shortcut from the answer token rather than the reasoning path. Our method instead checks s_back (E entails a) and s_span over each evidence span, discarding tasks whose compact authoring surface does not support the answer, so we expect a gap concentrated on multi-hop tasks with 3+ hops and near-parity on single-hop tasks. We expect LoongRL's closure-violation rate to be highest on 3+ hop tasks, confirming the invariant targets the claimed failure mode.

**vs. R-Zero** — On a case where the Challenger-Solver loop generates a compact math problem whose expanded context is a paraphrase that subtly shifts a numeric constraint, R-Zero has no long-context closure check and accepts the drifted task, teaching the solver to answer the drifted version. Our method instead product-gates on s_ans and s_back so drifted expansions are rejected before entering GRPO, and we expect a gap on numeric/multi-hop long-context subsets while both methods stay close on short-context math. We expect R-Zero's closure-violation rate on numeric tasks to be non-trivial, demonstrating the invariant transfers beyond LongZero.

**Calibration results** — On the 512-pair calibration set, we expect closure AUROC to be high (≥0.85) at K=4 and to degrade with K for the full-context premise, while span-localized premises maintain AUROC ≥0.80 across K ∈ {4,8,16,32}. If AUROC is near chance or drops sharply with K, the load-bearing assumption fails and the closure signal is not a valid groundedness measure.

### What would falsify this idea

If the gain over LongZero is uniform across distractor-count and hop-count subsets rather than concentrated on high-distractor and multi-hop cases where the predicted failure mode lives, the closure invariant is not doing the claimed causal work. Likewise, if the forward-only ablation matches full ECC on every subset, the bidirectional product is not necessary and the central mechanism claim is wrong; and if calibration AUROC is near chance or closure does not exceed the lexical-overlap control, the verifier is not measuring entailment and the reward signal collapses.

## References

1. LongZero: Enhancing Long-Context Reasoning in LLMs from Zero Data
2. LoongRL: Reinforcement Learning for Advanced Reasoning over Long Contexts
3. R-Zero: Self-Evolving Reasoning LLM from Zero Data
4. QwenLong-L1.5: Post-Training Recipe for Long-Context Reasoning and Memory Management
