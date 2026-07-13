# Causal-Permutation-Guided Span Selection for Test-Time Training of Long-Context LLMs

## Motivation

Self-guided test-time training methods such as S-TTT rely on imperfect intrinsic signals (e.g., attention or perplexity) to select evidence spans, which provides no correctness guarantee and can lead to adaptation on irrelevant or noisy content, especially under domain shift. Existing attention-based selection (e.g., QRHead, ICR) also fails to establish causal necessity of spans, as attention can be biased by recency or spurious correlations. This structural gap—relying on correlational rather than causal relevance signals—hinders reliable adaptation in long-context reasoning tasks.

## Key Insight

A permutation test on the model's own output distribution measures the causal necessity of each candidate span, providing an invariant that is robust to the spurious correlations that mislead attention-based selection.

## Method

**Causal-Permutation-guided Span Selection (CPS)**

**(A) What it is:** CPS is a test-time training method that integrates a permutation-based causal importance test to select evidence spans, ensuring only causally necessary spans are used for adaptation. Input: test input with long context, question, and predicted answer. Output: adapted model parameters.

**(B) How it works:**
```pseudocode
Input: model M, test sample (x, q) where x is long context, q is question
1. Obtain answer a ← argmax M(x, q)  // initial prediction
2. Identify candidate spans S using attention heads (top-20% of tokens by average attention to q). Span size: contiguous token sequences of length L=10 (min) to 50 (max), with overlap stride 5.
3. For each span s in S:
     - Create permuted input x_perm by randomly shuffling tokens within s (keep tokenization intact)
     - Compute p = softmax(M(x, q)) and p_perm = softmax(M(x_perm, q))
     - Importance I(s) = KL(p || p_perm)   // measure distribution change
4. Select top-m spans E with highest I(s) (m=5)
5. Fine-tune M on E with language modeling objective (cross-entropy next-token prediction) for T=10 steps, learning rate 1e-5, optimizer: AdamW with linear warmup (first 2 steps). Use gradient clipping norm 1.0.
6. Use adapted M for final prediction
```

**(C) Why this design:** We chose permutation test over attention scores (e.g., as in QRHead or ICR) because attention can be biased by recency or position (as shown in Attention Sorting paper) and does not measure causal necessity; permutation is a direct intervention that quantifies the span's impact on the output, at the cost of O(|S|) additional forward passes. We selected top-20% candidates via attention to reduce computational overhead, accepting that some low-attention but causally important spans may be missed. We used KL divergence rather than log-probability change because KL captures the full distribution shift, making it more robust when the model is uncertain about the correct answer, at the expense of slight extra computation. Finally, we fine-tune with standard LM loss rather than directly optimizing the permutation test loss because the latter could lead to overfitting to the perturbation; this choice prioritizes stable adaptation over direct reinforcement of causal selection.

**(D) Why it measures what we claim:** The importance score I(s) = KL(p || p_perm) measures causal necessity of span s for the model's prediction, under the assumption that the model is sensitive to token order within the span. This assumption fails when the model uses a bag-of-words representation or relies on positional cues that are invariant to local shuffling (e.g., with long spans where order is not critical). In that case, I(s) reflects the strength of lexical correlation rather than true causal necessity. To partially mitigate this, we restrict candidate spans to attention-selected tokens (top-20%), which are more likely to be in the model's reasoning path, and we use a small shuffle window (within span) to avoid breaking global positional encodings. However, this is not a full guarantee. Additionally, we perform a calibration step before test-time training: on a held-out set of 100 examples with known relevant spans (sampled from the training set), we verify that I(s) is significantly higher for true relevant spans than for random spans (t-test p < 0.05). If the calibration fails (i.e., no significant difference), we fall back to attention-only selection. This ensures the permutation test is valid for the given model and domain.

## Contribution

(1) A novel test-time training method that uses permutation tests to causally select evidence spans, providing a statistical guarantee absent in prior self-guided selection approaches. (2) Empirical demonstration that CPS improves long-context reasoning accuracy over S-TTT and attention-based selection baselines, validated on LongBench-v2 and a synthetic causal QA task. (3) An analysis showing that the permutation importance measure is more robust than attention to recency and positional biases in long contexts.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|----------|
| Dataset | LongBench-v2 (all subsets, 1K samples per subset to manage compute) | Long-context multi-task QA benchmark with diverse reasoning types; subsets: single-doc QA, multi-doc QA, summarization, few-shot, synthetic. |
| Primary metric | Accuracy (exact match on answer) | Measures correct answer rate; standard for QA. |
| Baseline1 | Random Span TTT | Random selection of m=5 spans from context, then same fine-tuning as CPS. |
| Baseline2 | Attention-Only Span Selection | Select top-m spans by average attention to q (from last layer), no causal test; same fine-tuning. |
| Baseline3 | CPS with Integrated Gradients (IG) | Replace permutation test with IG: for each candidate span, compute IG attributions of answer logit w.r.t. token embeddings, sum absolute attributions over span tokens; select top-m spans. IG baseline: 50 integration steps. |
| Ablation1 | CPS w/ attention importance (worse-case) | Instead of permutation test, use average attention to q as importance; select top-m spans. |
| Ablation2 | CPS w/ deletion test | Replace permutation with deletion: mask span tokens with [MASK] and compute KL(p || p_masked); same candidate selection and fine-tuning. |

### Why this setup validates the claim

This setup tests the central claim that causal permutation tests improve span selection for test-time training over non-causal alternatives. LongBench-v2 provides long-context questions requiring multi-step reasoning, where spurious correlations (e.g., recency bias in attention) can mislead selection. Comparing against random span TTT establishes the necessity of any selection; attention-only selection isolates the added value of causal testing. The ablation using attention importance pinpoints the permutation test as the key innovation. The addition of Integrated Gradients baseline tests whether permutation uniquely captures causal necessity compared to gradient-based attribution. The deletion-test ablation validates the permutation assumption by comparing to a more direct intervention (masking). Accuracy as a metric directly reflects whether the adapted model produces correct answers; if better causal selection improves adaptation, accuracy will increase. The dataset has subsets with varying reasoning types, allowing us to see if gains concentrate where causal necessity matters most (e.g., questions requiring distant evidence), or if improvements are uniform (which would weaken the claim).

### Expected outcome and causal chain

**vs. Random Span TTT** — On a case where the answer depends on a single sentence buried in the middle of a long context, random span TTT may select irrelevant tokens (e.g., from the beginning or end) and fine-tune on them, potentially overwriting useful knowledge and decreasing accuracy. Our CPS method identifies that sentence via the permutation test (shuffling it causes large KL divergence) and fine-tunes only on that sentence, preserving other parameters. Therefore, we expect a noticeable gap: CPS outperforms random by at least 5-10 percentage points on subsets requiring pinpoint evidence retrieval (e.g., multi-hop QA), with parity on simpler tasks.

**vs. Attention-Only Span Selection** — On a case where the model's attention is dominated by recent tokens (e.g., the last sentence in the context) due to positional bias, attention-only selection chooses that recent span even when it is irrelevant. The permutation test for that span yields low importance because shuffling it does not affect the output (the answer depends on an earlier span), so CPS selects the earlier span instead. Thus, on subsets where answers rely on early or mid-context information, we expect CPS to show a 3-8 percentage point improvement over attention-only, while on recency-dependent tasks (e.g., last-sentence QA) both perform similarly.

**vs. CPS with Integrated Gradients** — Integrated Gradients (IG) may attribute importance to tokens that are not causally necessary (e.g., tokens that are highly correlated but not relied upon by the model). CPS's permutation test directly measures the effect of perturbing the span, thus is expected to be more robust. We expect CPS to outperform IG-based selection by 1-3 percentage points on average, with larger gaps on subsets where causal necessity is subtle (e.g., multi-doc QA with conflicting evidence). However, IG may be more computationally efficient (single forward pass per span, vs. two for permutation); we will report runtime.

**vs. CPS with attention importance (ablation)** — On a case where a span has high attention but is not causally necessary (e.g., a repeated word that attracts attention but is irrelevant), the ablation selects it and fine-tunes on it, potentially degrading performance. CPS avoids this by requiring causal impact. We expect CPS to outperform its ablation on reasoning-heavy subsets, with a gap of 2-5 points, confirming that the permutation test adds value beyond attention.

**vs. CPS with deletion test (ablation)** — The deletion test (masking) may overestimate causal importance because masking removes content entirely, while permutation preserves lexical content but disrupts order. In models that are order-sensitive, both should agree; in models that are order-insensitive, deletion may still show importance while permutation fails. We expect both to perform similarly overall, but on subsets where local order is critical (e.g., date arithmetic), permutation may yield better selection. This ablation validates the permutation assumption; if deletion outperforms permutation significantly, it indicates the permutation test is missing some causally important spans.

### What would falsify this idea

If CPS's accuracy is not significantly higher than attention-only selection on subsets where attention is known to be biased (e.g., early-context QA), or if the ablation using attention importance performs comparably to CPS overall, then the causal permutation test does not provide the expected benefit and the central claim is falsified. Additionally, if the deletion-test ablation substantially outperforms the permutation test (e.g., >5 points on any subset), it would indicate that the permutation test fails to capture causal necessity, undermining the method's foundation.

## References

1. Self-Guided Test-Time Training for Long-Context LLMs
2. Test-Time Training Done Right
3. Query-Focused Retrieval Heads Improve Long-Context Reasoning and Re-ranking
4. Attention in Large Language Models Yields Efficient Zero-Shot Re-Rankers
5. Understanding Synthetic Context Extension via Retrieval Heads
6. Titans: Learning to Memorize at Test Time
7. Attention Sorting Combats Recency Bias In Long Context Language Models
