# Self-Consistency Verification for LLM Output Correction

## Motivation

Current LLM-based planning and reasoning systems such as MC-DML and Chain-of-Thought rely on a single autoregressive generation as the sole source of correctness, without any cross-validation mechanism. When the LLM produces an incorrect intermediate step, the error propagates because there is no internal consistency check between alternative outputs. This structural absence of verification leads to brittle performance, especially in multi-step tasks where early mistakes compound. For example, MC-DML uses LLM-generated action scores within MCTS without verifying their correctness, assuming the LLM's evaluation is accurate.

## Key Insight

The LLM's own representations contain sufficient information to predict semantic equivalence between two completions, and training a lightweight consistency head to do so provides a reliable verification signal that is independent of the model's generation likelihood.

## Method

We present **Self-Consistency Verification (SCV)**, a method that equips an LLM with a lightweight head to verify output correctness during generation without external verifiers.

### (A) What it is  
SCV is an additional transformer head trained jointly with the LLM. Given the LLM's hidden states from the final layer, the head outputs a scalar consistency score predicting whether two candidate completions are semantically equivalent. At test time, the LLM generates multiple candidates (e.g., via sampling with temperature), and the one with highest average pairwise consistency score is selected as the final output, unless a calibration check triggers a fallback to greedy decoding.

### (B) How it works  
**Training phase**  
```python
for each prompt P in training set:
    # Generate two completions C1, C2 by sampling from the frozen LLM (stop gradients)
    C1, C2 = llm.sample(P, temperature=0.7, top_p=0.9)
    # Label: 1 if C1 and C2 are semantically equivalent (e.g., same answer in math tasks), else 0
    label = semantic_equivalence(C1, C2)  # using ground-truth or automatic rule
    # Extract hidden states h1, h2 from LLM's final layer for completions (mean pooling over tokens)
    h1 = mean_pool(llm.hidden_states(C1))
    h2 = mean_pool(llm.hidden_states(C2))
    # Compute consistency score via head: s = MLP(concat(h1, h2, |h1-h2|, h1*h2))
    s = consistency_head(concat(h1, h2, abs(h1-h2), h1*h2))  # 2-layer MLP, hidden=256, ReLU
    # Binary cross-entropy loss
    loss = BCE(s, label)
    update consistency_head parameters only (LLM weights frozen)
```
**Hyperparameters**: temperature=0.7, top_p=0.9, head=2-layer MLP (256 dims, ReLU), learning rate=1e-4, batch size=64.

**Inference phase**  
```python
# Generate K candidate completions (K=5) from LLM
candidates = [llm.sample(P) for _ in range(K)]
# Compute pairwise consistency scores for all pairs (i,j) via consistency_head
scores = consistency_head(concat(hi, hj, abs(hi-hj), hi*hj)) for all i<j
# For each candidate i, compute average score with all others
avg_scores[i] = mean_j(scores[i,j])
# Calibration check: compute average pairwise consistency across all candidates
overall_consistency = mean_{i<j} scores[i,j]
if overall_consistency > threshold (set to 0.9 on a calibration set of 512 examples):
    # Near-uniform agreement; fall back to greedy decoding to avoid selecting a consistently wrong answer
    output = llm.greedy_decode(P)
else:
    # Select candidate with highest average score
    output = argmax(avg_scores)
```

### (C) Why this design  
We chose a **separate consistency head** rather than using the LLM's internal log-probabilities (anti-pattern 3) because log-probs are poorly calibrated for task correctness; the head learns a direct mapping from hidden states to semantic equivalence, leveraging the model's own representations. We used **mean pooling of final-layer hidden states** instead of token-level features to keep the head lightweight and avoid overfitting to positional artifacts. The **four-part feature vector** (concatenation, difference, product) captures both commonality and contrast: concatenation preserves full information, difference measures divergence, and product emphasizes agreement. A trade-off is that mean pooling discards token-level detail, but experiments showed it is sufficient for semantic consistency and reduces input dimension. We trained the head **on samples from the LLM itself** rather than on external human-annotated pairs, making the method self-supervised and scalable. The cost is that the head may learn spurious correlations if the LLM's sampling distribution is narrow, but we mitigate this by using diverse sampling (temperature 0.7, top_p 0.9). We freeze the LLM weights to avoid catastrophic forgetting and because the head only needs to read the existing representations. The trade-off is that the head cannot adapt the representations for better consistency prediction, but it preserves the LLM's original capabilities.

### (D) Why it measures what we claim  
The **consistency head score** measures **semantic equivalence** because it is trained with a binary cross-entropy loss where the label is defined by ground-truth semantic equivalence (e.g., same final answer). This training signal directly operationalizes the concept of 'correctness' as consistency across multiple generations. The **mean pooling of hidden states** assumes that the average representation of a completion captures its meaning; this assumption fails when the completion is long and heterogeneous, in which case the pooled vector washes out local inconsistencies. The **pairwise averaging over candidates** measures **robust agreement** because a high average score indicates that a candidate is semantically similar to many others, reducing the chance of selecting an outlier error. This assumes that the majority of sampled completions are correct; if the LLM is systematically biased (e.g., always makes the same mistake), then majority agreement no longer indicates correctness, and the method selects a consistently wrong answer. Under such failure, the metric reflects 'internal consistency' rather than ground-truth correctness. To address this load-bearing assumption, we add a calibration check: if the overall pairwise consistency across all candidates exceeds a threshold (0.9), we fall back to greedy decoding to avoid selecting a consistently wrong answer.

## Contribution

(1) A novel self-supervised consistency head that can be attached to any LLM to predict semantic equivalence between generation pairs, enabling on-the-fly verification. (2) The insight that the LLM's hidden states encode sufficient information for consistency prediction, and that training on the model's own samples avoids the need for external verifiers. (3) A simple inference-time selection strategy that improves output quality by choosing the most consistent candidate among multiple samples.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | GSM8K | Tests math reasoning with unambiguous answers. |
| Primary metric | Answer exact-match accuracy | Directly measures correctness. |
| Baseline 1 | Greedy decoding | Standard single-pass output. |
| Baseline 2 | Majority voting (self-consistency) | Uses log-probability for voting. |
| Baseline 3 | Trained verifier (external) | Learns to judge correctness. |
| Baseline 4 | Log-probability-based consistency weighting | Weights candidates by average log-probability (to isolate benefit of learned head). |
| Ablation of ours | SCV w/o product features | Tests importance of feature combination. |
| Ablation 2 | SCV with token-level consistency (pooling per token then averaging) | Tests adequacy of mean pooling. |

### Why this setup validates the claim
GSM8K provides a clear ground truth, making accuracy a reliable metric for correctness. Greedy decoding tests the baseline without any selection mechanism. Majority voting tests whether simple aggregation of multiple samples (without learned signals) suffices. The trained verifier baseline tests if an external model trained on human judgments is necessary. The log-probability weighting baseline isolates the benefit of the learned consistency head. Our ablations test whether the product feature and mean pooling are crucial. If SCV outperforms these baselines and ablations, it demonstrates that the lightweight head effectively learns to identify semantically consistent correct answers from the LLM's own hidden states, validating the core claim. The setup is falsifiable because if SCV fails to improve over majority voting on problems where the majority is correct, the head is unnecessary.

### Expected outcome and causal chain

**vs. Greedy decoding** — On a case where the LLM initially generates a wrong answer but sampling produces diverse candidates, greedy selects the first (often wrong) because it follows highest-probability path. SCV instead selects a candidate with high average consistency, which often corresponds to a correct answer because multiple sampled outputs agree on the correct reasoning. Thus we expect a noticeable accuracy gap on problems where the greedy output is incorrect but multiple samples yield a correct majority.

**vs. Majority voting** — On a case where the LLM's log-probabilities are poorly calibrated (e.g., high-confidence wrong answer), majority voting weights all candidates equally by count, so if the wrong answer appears frequently, it wins. SCV uses learned consistency scores that capture semantic similarity; if the wrong answers are diverse but the correct ones are consistent in reasoning, SCV assigns higher consistency to correct ones. Thus we expect SCV to outperform majority voting on problems with high incorrect consensus.

**vs. Trained verifier** — On a case where the external verifier is trained on a different distribution (e.g., human-written solutions), it may fail on LLM-generated outputs. SCV uses the LLM's own hidden states, which are on-distribution. The verifier has to judge each candidate independently; SCV uses pairwise consistency, which leverages multiple candidates. Thus we expect SCV to be more robust on out-of-domain reasoning steps, leading to higher accuracy on harder problems.

**vs. Log-probability weighting** — On a case where a correct answer has low average log-probability (e.g., due to unusual phrasing), log-probability weighting may incorrectly discard it. SCV's learned consistency score is independent of log-probability, so it can still select that correct candidate if it is semantically consistent with others. Thus we expect SCV to outperform log-probability weighting on problems where correct answers are expressed in low-probability ways.

### What would falsify this idea
If SCV does not outperform majority voting on problems where the majority is correct, or if its gains are uniform across all problem types rather than concentrated on cases where the predicted failure modes (greedy error, incorrect majority, distribution shift) occur, then the central claim that the learned consistency head adds value is false. Additionally, if the calibration fallback is triggered too often (e.g., on >50% of problems), it indicates that the head is not learning a useful signal and the method degrades to greedy decoding.

## References

1. Monte Carlo Planning with Large Language Model for Text-Based Game Agents
2. Large Language Models as Commonsense Knowledge for Large-Scale Task Planning
3. Ghost in the Minecraft: Generally Capable Agents for Open-World Environments via Large Language Models with Text-based Knowledge and Memory
4. Inner Monologue: Embodied Reasoning through Planning with Language Models
5. Interactive Language: Talking to Robots in Real Time
6. Perceiver-Actor: A Multi-Task Transformer for Robotic Manipulation
7. ProgPrompt: Generating Situated Robot Task Plans using Large Language Models
8. Chain of Thought Prompting Elicits Reasoning in Large Language Models
