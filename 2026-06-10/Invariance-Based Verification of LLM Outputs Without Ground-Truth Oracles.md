# Invariance-Based Verification of LLM Outputs Without Ground-Truth Oracles

## Motivation

Both role-playing evaluation (e.g., RPEval) and LLM-based exploration (e.g., PSRL agents) assume that LLM-generated outputs—such as evaluation rubrics or posterior inferences—are correct without external validation. This assumption is structurally fragile because LLMs can produce plausible but incorrect outputs. For example, RPEval uses rubrics scored by an LLM without verifying that the rubrics are invariant under dialogue paraphrases. Similarly, PSRL agents rely on LLM posterior samples without checking consistency under permuted observation orders. We propose a verification method that tests invariance under causally irrelevant input transformations, flagging unreliable outputs without needing ground-truth oracles.

## Key Insight

The structural property that causally irrelevant input transformations (e.g., paraphrasing, permuting exchangeable data) must leave the target output unchanged provides a necessary condition for correctness that an LLM output violates only when it is unreliable.

## Method

(A) **What it is**: Invariance-Based Output Verification (IOV) is a zero-shot verification method that takes an LLM output (e.g., a rubric score or posterior sample) and a set of causally irrelevant input transformations, and returns a reliability score (the fraction of transformations under which the output remains consistent). The output is flagged as unreliable if the consistency falls below a threshold τ.

(B) **How it works** (pseudocode):
```pseudocode
Input: LLM f, input context C, output O (scalar or distribution), 
       transformation set T = {t_i : i=1..N} where each t_i is a function 
       that maps C to C' such that t_i is causally irrelevant w.r.t. the target.
Output: reliability score r ∈ [0,1], flag if r < τ.

1. For each t_i in T:
   a. Compute transformed input C_i = t_i(C)
   b. Compute LLM output O_i = f(C_i) (same output format as O, e.g., scalar score or posterior)
   c. Compute consistency metric d(O, O_i) ∈ [0,1] (e.g., for scalar: 1 - |O - O_i|/range; for distribution: 1 - JS-divergence)
2. Compute r = mean_i d(O, O_i)
3. If r < τ, flag O as unreliable.
```

For the consistency metric, we choose for scalar outputs: normalized absolute difference; for distributional outputs: 1 - Jensen-Shannon divergence (JS < 0.5 implies high consistency). The threshold τ is set to 0.8 based on validation on a small held-out set.

(C) **Why this design**: We chose a transformation-based consistency check over an LLM-based critic (e.g., asking "is this correct?") because the latter suffers from the same verification crisis—the critic itself may be unreliable. By leveraging causal irrelevance, we obtain a formal invariant that does not require a meta-model. We chose a fixed threshold τ rather than a learned adaptive threshold to avoid overfitting and maintain simplicity, accepting that a single threshold may not be optimal for all tasks. We chose a simple average consistency rather than a weighted scheme because we have no prior knowledge of which transformations are most informative; uniform weighting ensures no transformation dominates. We used JS divergence for distributional outputs because it is symmetric and bounded, unlike KL divergence which can be infinite. The trade-off is that JS may be less sensitive to certain differences, but the boundedness allows a direct interpretation as a similarity score.

(D) **Why it measures what we claim**: The computational quantities (d(O, O_i) and r) measure the "causal necessity" property of the LLM output: an output that is causally necessary (i.e., determined solely by the target quantity) should be invariant under transformations that do not affect that target quantity. Specifically, d(O, O_i) measures the degree of invariance under transformation t_i; this measures causal necessity under the assumption that t_i is indeed causally irrelevant to the target—this assumption fails when the transformation inadvertently changes the target (e.g., paraphrasing alters the meaning of the dialogue in a way that affects the rubric), in which case invariance is not expected and the metric reflects an irrelevant variation. The average r then measures the fraction of transformations for which the output is consistent, providing a robustness score; this measures the overall causal necessity under the assumption that the transformations are a representative sample of causally irrelevant variations—this assumption fails when the transformation set is biased (e.g., only easy paraphrases), in which case r may overestimate reliability. The threshold τ operationalizes an accept/reject decision; it measures the minimum acceptable degree of necessity, under the assumption that outputs with r below τ are likely incorrect—this assumption fails when a correct output is sparsely sampled by transformations (e.g., due to noise), leading to false flags.

## Contribution

(1) A zero-shot verification method, IOV, that detects unreliable LLM outputs by testing invariance under causally irrelevant input transformations, applicable to both scalar outputs (e.g., rubrics) and distributional outputs (e.g., posteriors). (2) A concrete instantiation of the verification for role-playing evaluation and LLM-based exploration, showing how to construct transformation sets (paraphrasing and observation permutation) and consistency metrics. (3) A demonstration that the invariance condition provides a necessary condition for correctness without ground-truth oracles, addressing the convergent gap across evaluation and exploration.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|-----------------------|
| Dataset | GSM8K (math word problems) | Standard benchmark with ground truth answers. |
| Primary metric | F1 for flagging incorrect outputs | Balances precision and recall of verification. |
| Baseline1 | LLM Self-Consistency (k=5) | Standard method for output consistency. |
| Baseline2 | LLM-as-Critic (same LLM) | Typical verification using the model itself. |
| Baseline3 | No verification (raw accuracy) | Lower bound on performance. |
| Ablation-of-ours | Random transformation set | Removes causal irrelevance, tests necessity. |

### Why this setup validates the claim

This design directly tests the central claim that invariance under causally irrelevant transformations reliably detects unreliable LLM outputs. GSM8K provides clear ground truth for correctness, enabling quantitative comparison. Self-consistency and LLM-as-Critic baselines represent common verification strategies; if IOV outperforms them, it demonstrates the advantage of formal invariance over sampling or self-assessment. The ablation with random transformations isolates whether causal irrelevance is the key property—if it degrades performance, the claim is supported. The F1 metric captures both precision and recall of flagging incorrect outputs, directly measuring how well each method distinguishes reliable from unreliable outputs. A successful validation would show IOV achieving higher F1, especially on cases where baselines fail due to systematic bias or overconfidence.

### Expected outcome and causal chain

**vs. LLM Self-Consistency** — On a math problem where the LLM consistently outputs the same wrong answer due to systematic error (e.g., misapplying a formula), self-consistency shows high agreement and fails to flag it because it only checks random variation. Our method applies transformations like reordering numbers or paraphrasing that do not affect the math, causing the biased answer to change, thus detecting inconsistency. We expect IOV to achieve significantly higher recall on such cases, leading to a noticeable F1 gap (e.g., +15%) on problems with common misconceptions.

**vs. LLM-as-Critic** — On a problem where the LLM confidently gives a wrong answer and is then used to judge itself, the critic often shares the same blind spot (e.g., flawed reasoning is reaffirmed). Our method's invariance check is independent of the model's self-assessment, so it can flag these cases. For an instance where the LLM is overconfident on a tricky word problem, the critic says "correct" but our transformations reveal output change. We expect IOV to have higher precision on false positives (fewer false flags on correct answers) because transformations are causally irrelevant, while the critic may also misjudge correct answers, resulting in an overall F1 improvement of 10-20% relative.

**vs. No verification** — No verification achieves raw accuracy (e.g., 80% on GSM8K). Our method will reject some correct answers (false positives) but catch more wrong ones, improving F1 for detecting incorrect outputs. Expected relative improvement of 15-25% in F1, with a slight drop in overall accuracy due to false flags, but a better trade-off for reliability.

**Ablation (random transformations)** — Without causal irrelevance, random transformations (e.g., random word substitutions) may alter the problem's meaning, causing false flags on correct answers or missing real errors. We expect IOV with proper transformations to outperform this ablation by 10-15% F1, particularly on cases where correct answers are robust to irrelevant changes, confirming that causal irrelevance is essential.

### What would falsify this idea

If IOV's F1 is not significantly better than the LLM-as-Critic baseline, or if the gain is uniform across all subsets rather than concentrated on cases where the model's bias is invariant under irrelevant transformations (e.g., systematic errors), then the central claim of causal necessity as a reliable signal is undermined.

## References

1. Role-Playing Evaluation for Large Language Models
2. Large Language Models are Superpositions of All Characters: Attaining Arbitrary Role-play via Self-Alignment
3. Character-LLM: A Trainable Agent for Role-Playing
4. Llama 2: Open Foundation and Fine-Tuned Chat Models
5. Efficiently Scaling Transformer Inference
6. Scaling Instruction-Finetuned Language Models
7. Toward Efficient Exploration by Large Language Model Agents
8. Efficient Reinforcement Learning with Large Language Model Priors
