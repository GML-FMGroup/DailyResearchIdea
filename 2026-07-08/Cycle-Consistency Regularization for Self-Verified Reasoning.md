# Cycle-Consistency Regularization for Self-Verified Reasoning Traces

## Motivation

Current multimodal language models (e.g., Gemma 4's thinking mode) generate reasoning traces before answers but lack per-step feedback, leading to potential spurious or incorrect intermediate steps. The key bottleneck is verifying trace correctness without expensive human annotations or process reward models. While RLVR (TÜLU 3) uses verifiable constraints for reinforcement learning, it is limited to tasks with automatic verification, leaving most reasoning tasks unaddressed. Cycle-consistency offers a self-supervised alternative, but prior work has not applied it to enforce that reasoning traces are uniquely determined by the input-answer pair, thereby eliminating spurious steps.

## Key Insight

Cycle-consistency forces the reasoning trace to be a deterministic function of the input-answer pair, making spurious steps (those not causally necessary) unstable under reconstruction and naturally penalized by reconstruction loss.

## Method

We propose Cycle-Consistent Reasoning Regularization (CCRR), a training procedure that adds two cycle-consistency losses to the standard language modeling objective. The method takes a dataset of (question, answer) pairs and a language model M (a 7B-parameter multimodal transformer, e.g., Gemma 4 thinking mode architecture: 32 layers, 16 heads, 50k vocabulary), and outputs a model that produces reasoning traces that are faithful to the answer. The core idea is to enforce that (1) the reasoning trace must be sufficient to reconstruct the correct answer, and (2) the trace must be consistent when generated twice from the same input-answer pair.

**(B) How it works:** The following pseudocode describes the training loop:
```python
for each batch (Q, A):
    # Forward: generate reasoning trace T from Q
    T = M.generate(Q, max_length=256, temperature=0.8)  # explore diverse traces
    # Reconstruct answer A' from reasoning trace T ONLY (no Q)
    A' = M.generate("Reasoning: " + T + "\n Therefore, the answer is", max_length=10)
    # Cycle-consistency loss: cross-entropy between A and A'
    L_cycle = cross_entropy(A, A')
    # Backward consistency: generate T' from (Q, A) and compare distributions
    T' = M.generate("Question: " + Q + "\n Answer: " + A + "\n Reasoning steps:", max_length=256, temperature=0.8)
    # Compute KL divergence between logits of T and T' (averaged over tokens)
    L_consistency = KL_divergence(logits(T) || logits(T'))
    # Total loss = standard LM loss + lambda1 * L_cycle + lambda2 * L_consistency
    L_lm = cross_entropy over whole sequence (Q, T, A) with teacher forcing
    loss = L_lm + 0.1 * L_cycle + 0.05 * L_consistency
    # Update model parameters
    optimizer.step(loss)
```
Hyperparameters: lambda1=0.1, lambda2=0.05, max trace length=256, temperature=0.8. Training on GSM8K (7,473 examples) with batch size 16 on 4x A100 GPUs (80GB each) takes approximately 0.5 hours per epoch; total training for 10 epochs uses ~5 GPU-hours.

**(C) Why this design:** We chose to use two complementary consistency losses rather than a single one. The forward cycle loss (L_cycle) directly ensures that the trace contains enough information to recover the answer; omitting critical steps increases reconstruction error. The backward consistency loss (L_consistency) encourages the trace to be a deterministic function of (Q,A), reducing variability that could allow spurious steps. We selected cross-entropy for L_cycle because it directly measures answer prediction accuracy, while KL divergence for L_consistency preserves the token-level distribution. A key trade-off: generating T' from (Q,A) doubles the compute per batch, which we accepted to enforce uniqueness; an alternative would be to use a single generation with dropout, which would be cheaper but less stable. Another design decision: we use temperature sampling (tau=0.8) rather than greedy decoding to explore diverse traces during training, accepting noisy gradients in exchange for better coverage. Finally, we set lambda1 larger than lambda2 because the forward cycle is the primary signal, with consistency serving as a regularizer. This design differs from TÜLU 3's RLVR, which relies on external verifiers; our method is fully self-supervised and does not require task-specific verifiers.

**(D) Why it measures what we claim:** L_cycle measures the extent to which the reasoning trace T causally determines the answer A, because if T contains all necessary reasoning, then A' should match A; spurious steps (e.g., irrelevant calculations) do not increase A' accuracy but may still be present. The assumption that high reconstruction accuracy implies faithful reasoning requires that the model cannot cheat by ignoring T (e.g., by memorizing the answer from the question); this assumption is addressed by conditioning reconstruction only on T. L_consistency measures the uniqueness of the trace given (Q,A); if the model generates different traces for the same input-answer pair, it indicates that the trace is not a deterministic function, which may allow spurious variability. The assumption that deterministic traces are faithful relies on the premise that for each correct input-answer pair, there exists a unique correct reasoning path; this assumption fails when multiple equally valid reasoning strategies exist (e.g., different algebraic manipulations), in which case L_consistency would penalize legitimate diversity. In that scenario, the metric reflects diversity rather than correctness, a known limitation we address by only enforcing strong consistency for tasks with a single standard reasoning path.

Explicit assumption: L_cycle measures sufficiency of trace for answer, but not necessity. We assume that uniqueness from L_consistency eliminates spurious steps, which holds only when a single correct reasoning path exists. Failure mode F: when multiple correct reasoning paths exist, L_consistency penalizes legitimate diversity, and sufficiency+uniqueness does not imply necessity.

## Contribution

(1) A cycle-consistency training framework for self-verification of reasoning traces without per-step feedback, applicable to any task with ground-truth answers. (2) Empirical demonstration that cycle-consistency reduces spurious steps and improves downstream task accuracy on math and reasoning benchmarks. (3) Open-source implementation of the training procedure, including code for trace generation and consistency losses.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|------------------------|
| Dataset | GSM8K, ScienceQA, BIG-bench (multi-step) | Cover math, science, diverse reasoning |
| Primary metric | Answer accuracy | Directly measures correct reasoning outcome |
| Secondary metric | Step faithfulness (human rating on 200 samples) | Measures trace necessity directly |
| Baseline 1 | Standard SFT | No cycle-consistency regularization |
| Baseline 2 | RLVR (external verifier) | State-of-the-art with external reward |
| Baseline 3 | Trace dropout (randomly drop 20% of trace tokens during reconstruction) | Tests if simple perturbation helps |
| Baseline 4 | Random permutation (shuffle trace token order before reconstruction) | Tests need for logical order |
| Ablation of ours | CCRR w/o backward loss | Tests importance of backward consistency |

### Why this setup validates the claim

GSM8K provides a controlled setting where reasoning steps are critical for correct answers. ScienceQA tests generalization to multimodal (diagram) questions. BIG-bench multi-step tasks evaluate chain-of-thought length. Comparing against standard SFT isolates the effect of cycle consistency: if SFT fails on problems with spurious correlations but CCRR succeeds, the mechanism is validated. The RLVR baseline tests whether our self-supervised approach can match or exceed methods that use expensive external verifiers. The trace dropout and random permutation baselines test whether the cycle consistency design is uniquely beneficial over simpler regularizations. The ablation (removing backward consistency) disentangles the contributions of the two loss terms. Accuracy is chosen as the primary metric because it reflects the ultimate goal of correct reasoning; the secondary human rating directly assesses trace faithfulness.

### Expected outcome and causal chain

**vs. Standard SFT** — On a case where a spurious correlation exists (e.g., certain numbers frequently appear in answers), SFT might generate a reasoning trace that ignores actual steps and still predicts correctly by memorization. Our method forces the trace to reconstruct the answer; a spurious trace would fail reconstruction. Thus we expect a noticeable gap (e.g., 5–10% higher accuracy) on problems with such correlations, but parity on simple problems.

**vs. RLVR** — On a case where an external verifier is imperfect (e.g., a math problem with novel phrasing), RLVR might accept a plausible-sounding but flawed trace. Our method uses inherent cycle consistency, which does not rely on a potentially brittle verifier. We expect CCRR to achieve comparable or slightly higher accuracy (e.g., 2–5% improvement) on out-of-distribution problems, with no regressions on standard splits.

**vs. Trace dropout** — Trace dropout may help generalization but does not enforce consistency of the trace with the answer. On problems where the trace contains irrelevant steps, dropout might still allow reconstruction via masked tokens. CCRR penalizes such traces through L_cycle, so we expect a 3–5% accuracy gain.

**vs. Random permutation** — Random permutation destroys logical order; if the trace order matters for reconstructing the answer, this baseline will fail. CCRR preserves order, so we expect a clear advantage (e.g., 5–8%) on multi-step problems.

**vs. CCRR w/o backward loss** — On a case where multiple valid reasoning paths exist but only one is fully faithful, the ablation might generate diverse traces that still reconstruct the answer, but include irrelevant steps. Our full method enforces uniqueness via backward consistency, reducing such noise. Thus we expect a modest gain (e.g., 2–3%) on problems with multiple valid strategies, with no loss on single-path problems.

### What would falsify this idea
If our gains are uniform across all problem subsets rather than concentrated on predicted failure modes (e.g., spurious correlations or multiple strategies), then the improvement likely stems from increased model capacity rather than the specific consistency regularization, refuting the central claim of improved faithfulness. Also, if the trace-dropout baseline matches CCRR performance, the cycle-consistency design is not uniquely beneficial.

## References

1. Gemma 4 Technical Report
2. Gemini 2.5: Pushing the Frontier with Advanced Reasoning, Multimodality, Long Context, and Next Generation Agentic Capabilities
3. Generalizing Verifiable Instruction Following
4. Qwen2.5 Technical Report
5. TÜLU 3: Pushing Frontiers in Open Language Model Post-Training
