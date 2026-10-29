# Author-Solver Adversarial Gating for Non-Trivial Self-Generated Long-Context Tasks

## Motivation

LongZero collapses the Challenger and Solver into a single self-writing model, so the model that authors a QA-evidence triple is also the one that judges its solvability, and nothing forces the task to be non-trivially solvable by that same model; R-Zero avoids this only by instantiating two separately optimized models, which is expensive and still lacks a per-task solvability check. The structural root cause is that authorship and solvability judgment share one set of weights, so a task is accepted whenever the author finds it plausible rather than when an independent solver-role pass can actually recover the answer from the evidence without shortcuts.

## Key Insight

A task is only useful for training if the same base model, when placed in a solver role with the evidence but not the author's hidden answer, can recover the answer through multi-hop evidence grounding while failing a surface-level shortcut, and this two-sided condition can be enforced per-task by an adversarial gate without training two separate models.

## Method

**What it is.** We introduce ASAG (Author-Solver Adversarial Gating), a task-generation and filtering loop for zero-data long-context RL. Input: a base LLM and a document pool. Output: a filtered set of (question, answer, evidence) triples that are *non-trivially solvable* by the same base model, which are then expanded into long multi-document contexts and used for GRPO training.

**How it works.**

```pseudocode
# Phase 1: Author role (same base LLM, author prompt)
for each seed document d in pool:
    triple = LLM.author(d)            # {q, a, e} where e is compact evidence
    # Author is asked to produce a multi-hop question whose answer
    # requires at least k=2 evidence spans and whose answer string
    # does not appear verbatim in e.

# Phase 2: Expand evidence into long context
triple.context = expand(triple.e, pool, n_distractors=20)

# Phase 3: Solver-role adversarial check (same base LLM, solver prompt)
# Four solver passes on the SAME expanded context:
full_pass        = LLM.solve(triple.context, triple.q)          # full context
mask_pass        = LLM.solve(mask(triple.context, triple.e), triple.q)  # evidence spans removed
random_mask_pass = LLM.solve(mask(triple.context, random_spans(len(triple.e))), triple.q)  # equal-length random spans removed
closed_book_pass = LLM.solve("", triple.q)                      # no context

# Phase 4: Solvability-but-nontriviality gate
solvable    = (normalize(full_pass) == normalize(triple.a))
nontrivial  = (normalize(mask_pass) != normalize(triple.a)) and (normalize(random_mask_pass) != normalize(triple.a))
no_memorization = (normalize(closed_book_pass) != normalize(triple.a))
if solvable and nontrivial and no_memorization:
    accept(triple)
else:
    reject(triple)   # trivially solvable (mask_pass or random_mask_pass correct), memorized (closed_book_pass correct), or unsolvable (full_pass wrong)

# Phase 5: GRPO training on accepted triples
for batch of accepted triples:
    trajectories = LLM.rollout(triple.context, triple.q, G=8)
    rewards      = exact_match(trajectories, triple.a)
    update(LLM, GRPO(rewards))
```

The key computational objects are: (i) the author pass produces a candidate triple; (ii) the full-context solver pass tests whether the evidence is *sufficient* for the answer; (iii) the masked-evidence solver pass tests whether the answer is *recoverable without the evidence*, i.e., whether the question is trivially answerable from surface cues or distractor leakage; (iv) the random-mask solver pass tests whether the answer is recoverable from *any* long context of equal length, i.e., whether the question is trivially answerable from the presence of context rather than the specific evidence; (v) the closed-book solver pass tests whether the answer is recoverable from parametric knowledge alone. A triple is accepted only when (ii) succeeds and (iii), (iv), and (v) all fail. The same base LLM is used for all roles; no second model is trained.

**Why this design.** We chose to reuse one base LLM with role-specific prompts rather than instantiate two separately optimized models as in R-Zero, because the gate only needs a *per-task* solvability signal at generation time, not a co-evolving curriculum, accepting the cost that the solver-role pass is not adversarially trained and may be weaker than a dedicated solver. We chose a *masked-evidence* counterfactual rather than a log-probability or self-confidence threshold because LLM log-probs are not calibrated for task correctness (Kadavath et al.), accepting the cost that masking is a coarse intervention that can leave distractor leakage; we mitigate this by adding a random-mask pass and a closed-book pass to rule out context-presence and parametric-recall shortcuts. We chose exact-match on the normalized answer string rather than an LLM judge because the answer is a short span produced by the author, accepting the cost that paraphrased correct answers are counted as failures and reduce yield. We chose to gate *before* GRPO rather than shaping rewards with a solvability term, because a per-task binary gate keeps the GRPO advantage estimator unchanged and avoids reward hacking on the gate itself, accepting the cost that rejected tasks are discarded rather than used as negative examples. A domain expert would not describe ASAG as a variant of R-Zero because R-Zero's separation is *across training runs* (two co-evolving models) whereas ASAG's separation is *within a single forward pass over one task* and is used as a filter, not as a curriculum signal.

**Why it measures what we claim.** The full-context solver exact-match `solvable` measures *evidence sufficiency* because we assume that if the evidence spans are the unique minimal support for the answer, then a solver with access to them can recover the answer and a solver without them cannot; this assumption fails when the context contains a paraphrase of the evidence outside the masked spans, in which case `solvable` reflects distractor leakage rather than evidence grounding. The masked-evidence solver exact-match `nontrivial` measures *non-triviality* because we assume that removing the author-identified evidence spans is a valid intervention that destroys the only sufficient support for the answer; this assumption fails when the question is answerable from world knowledge or from a single distractor span, in which case `nontrivial` reflects memorization rather than multi-hop reasoning. The random-mask solver exact-match `nontrivial` (second clause) measures *dependence on the specific evidence* because we assume that removing random spans of equal length does not remove the answer's support; this assumption fails when the question is answerable from any long context, in which case `nontrivial` reflects context-presence bias rather than evidence integration. The closed-book solver exact-match `no_memorization` measures *absence of parametric recall* because we assume that the base model cannot answer the question without any context; this assumption fails when the answer is a well-known fact, in which case `no_memorization` reflects memorization rather than evidence grounding. The conjunction `solvable AND NOT nontrivial AND NOT no_memorization` measures *non-trivially solvable by the same base model* because we assume that the author and solver roles share the same knowledge but differ only in whether the evidence is visible; this assumption fails when the author prompt leaks the answer into the question phrasing, in which case the gate accepts tasks whose difficulty comes from prompt artifacts rather than evidence integration.

## Contribution

['ASAG, a per-task adversarial gating mechanism that uses a masked-evidence counterfactual solver pass from the same base LLM to reject self-authored tasks that are either trivially solvable or unsolvable, without training a second model.', 'A design principle for zero-data long-context RL: authorship and solvability judgment can be separated by role-conditioned prompts plus a counterfactual evidence mask, and the resulting gate changes the difficulty distribution of accepted tasks relative to unfiltered self-generation.', 'An analysis of the yield and difficulty profile of accepted versus rejected tasks under ASAG, identifying the failure modes (distractor leakage, world-knowledge shortcuts, prompt leakage) that the gate catches.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Long-context multi-hop QA (16K–128K) | Where evidence integration is the bottleneck. |
| Primary metric | Exact-match accuracy on held-out long-context QA | Directly tests evidence-grounded answer recovery. |
| Baseline | Base LLM, no RL | Isolates gain from ASAG-filtered GRPO. |
| Baseline | LongZero-style self-generated RL | Tests whether trivial-task filtering matters. |
| Baseline | R-Zero-style co-evolution RL | Tests whether within-task gating beats cross-model curriculum. |
| Ablation-of-ours | ASAG without masked-evidence gate | Tests whether non-triviality filter is the causal ingredient. |
| Ablation-of-ours | ASAG without random-mask gate | Tests whether context-presence shortcut is ruled out. |
| Ablation-of-ours | ASAG without closed-book gate | Tests whether parametric-recall shortcut is ruled out. |
| Diagnostic | Gate-precision metrics (leakage, world-knowledge) | Directly measures the two linking assumptions. |

### Why this setup validates the claim

The central claim is that gating on *evidence sufficiency AND masked-evidence failure AND random-mask failure AND closed-book failure* yields training tasks whose difficulty comes from multi-hop evidence integration, and that GRPO on such tasks improves long-context reasoning more than RL on unfiltered self-generated tasks. The base-LLM baseline establishes the untrained ceiling and shows whether any gain is attributable to training at all. The LongZero-style baseline tests the sub-claim that *filtering* matters: it trains on self-generated tasks without the masked-evidence counterfactual, so if ASAG beats it, the nontriviality gate is doing work beyond mere self-generation. The R-Zero-style baseline tests the sub-claim that *within-task gating* is sufficient relative to a cross-model co-evolution curriculum. The ablation (ASAG minus the masked gate) isolates the counterfactual as the causal ingredient: if removing it collapses the gain toward the LongZero-style baseline, the gate — not the author-solver loop per se — is responsible. The additional ablations (ASAG minus random-mask gate, ASAG minus closed-book gate) test whether the two new solver passes are necessary to rule out context-presence and parametric-recall shortcuts, respectively. The diagnostic gate-precision metrics directly measure the two linking assumptions: for accepted triples, we compute the fraction where a paraphrase of the evidence exists outside masked spans (distractor leakage) and the fraction answerable from world knowledge (closed-book success), reporting these as gate-precision metrics rather than only asserting them as failure modes. Exact-match on long-context multi-hop QA is the right metric because the claim is specifically about recovering a short evidence-grounded answer from a long context; a judge-based or partial-credit metric would blur whether failures are evidence-integration failures or surface-form mismatches, which is exactly the distinction the gate is designed to control.

### Expected outcome and causal chain

**vs. Base LLM, no RL** — On a case where the answer requires chaining two evidence spans separated by 20 distractor documents, the base model produces a plausible but ungrounded answer because it has no training signal rewarding evidence retrieval over parametric recall. Our method instead trains GRPO on triples where the masked-evidence solver *fails*, the random-mask solver *fails*, the closed-book solver *fails*, and the full-context solver *succeeds*, so the only trajectories that receive reward are those that actually consult the evidence. We expect a noticeable accuracy gap on multi-hop subsets and near-parity on single-hop subsets, since the gate preferentially admits multi-hop tasks.

**vs. LongZero-style self-generated RL** — On a case where the author produces a question whose answer is leaked by a distractor paraphrase, the LongZero-style baseline accepts it and trains on a task solvable without evidence integration, producing a model that learns to pattern-match question phrasing rather than retrieve. Our method rejects that triple because the masked-evidence solver still answers correctly, so the accepted set is enriched for evidence-dependent tasks. We expect ASAG to show a larger gain on tasks with high distractor density and a smaller, possibly null, gain on low-distractor tasks — the signature of filtering rather than raw self-generation.

**vs. R-Zero-style co-evolution RL** — On a case where the challenger and solver co-evolve but the solver's weakness is prompt-format sensitivity rather than evidence integration, the R-Zero-style baseline produces a curriculum that drifts toward format-hard tasks, leaving multi-hop evidence tasks under-trained. Our method instead fixes the difficulty axis to evidence visibility via the masked counterfactual, so every accepted task is hard *for the reason we care about*. We expect ASAG to match or trail R-Zero-style on short-context reasoning but to lead on long-context multi-hop QA, with the gap widening as context length grows from 16K to 128K.

**vs. ASAG without masked-evidence gate** — On a case where the author produces a question whose answer is leaked by a distractor paraphrase, the ablation without the masked gate accepts it and trains on a task solvable without evidence integration, producing a model that learns to pattern-match question phrasing rather than retrieve. Our method rejects that triple because the masked-evidence solver still answers correctly, so the accepted set is enriched for evidence-dependent tasks. We expect the full ASAG to show a larger gain on tasks with high distractor density and a smaller, possibly null, gain on low-distractor tasks — the signature of filtering rather than raw self-generation.

**vs. ASAG without random-mask gate** — On a case where the question is answerable from any long context (e.g., the answer is a common entity mentioned in many documents), the ablation without the random-mask gate accepts it and trains on a task solvable without the specific evidence, producing a model that learns to exploit context-presence bias rather than evidence integration. Our method rejects that triple because the random-mask solver still answers correctly, so the accepted set is enriched for evidence-specific tasks. We expect the full ASAG to show a larger gain on tasks with high context length and high distractor diversity, and a smaller gain on tasks with low context length — the signature of ruling out context-presence shortcuts.

**vs. ASAG without closed-book gate** — On a case where the answer is a well-known fact (e.g., "What is the capital of France?"), the ablation without the closed-book gate accepts it and trains on a task solvable from parametric knowledge, producing a model that learns to rely on memorization rather than evidence retrieval. Our method rejects that triple because the closed-book solver answers correctly, so the accepted set is enriched for evidence-dependent tasks. We expect the full ASAG to show a larger gain on tasks with low parametric-knowledge coverage (e.g., recent or niche facts) and a smaller gain on tasks with high parametric-knowledge coverage — the signature of ruling out memorization shortcuts.

### What would falsify this idea

If ASAG's gain over the LongZero-style baseline is uniform across distractor-density and hop-count subsets rather than concentrated on high-distractor, multi-hop tasks, then the masked-evidence gate is not selecting the tasks we claim it selects and the central mechanism is wrong. Likewise, if the ablation without the masked gate matches full ASAG, the counterfactual is not the causal ingredient. Additionally, if the gate-precision metrics show high leakage or world-knowledge fractions among accepted triples, the linking assumptions are violated and the gate's selectivity is unverified.

## References

1. LongZero: Enhancing Long-Context Reasoning in LLMs from Zero Data
2. LoongRL: Reinforcement Learning for Advanced Reasoning over Long Contexts
3. R-Zero: Self-Evolving Reasoning LLM from Zero Data
4. QwenLong-L1.5: Post-Training Recipe for Long-Context Reasoning and Memory Management
