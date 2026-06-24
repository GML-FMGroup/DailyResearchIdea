# Self-Consistent Knowledge Distillation for Autonomous Knowledge Acquisition in Scientific Reasoning Agents

## Motivation

Existing scientific reasoning agents such as DelveAgent assume fixed knowledge and lack mechanisms to learn from new data without external labels, a structural limitation that prevents autonomous adaptation. Self-consistency across reasoning paths offers a self-supervised signal: correct reasoning tends to converge, while errors diverge, enabling agents to bootstrap knowledge from their own outputs.

## Key Insight

Self-consistency across diverse reasoning paths acts as a natural verifier because the correct answer is more likely to be the attractor of consistent reasoning, making it a structurally reliable pseudo-ground-truth for fine-tuning.

## Method

### Self-Consistent Knowledge Distillation (SCKD)
#### (A) What it is
Self-Consistent Knowledge Distillation (SCKD) takes an unlabeled query set Q and a pre-trained LLM agent, and produces an improved agent by fine-tuning on self-generated consistent reasoning traces.

#### (B) How it works
```python
def SCKD(agent, Q, num_paths=5, consistency_threshold=0.8):
    # Load-bearing assumption: majority vote consistency across reasoning paths correlates with correctness.
    # Calibrate threshold on a small held-out set (n=512) to maximize precision at 90% recall.
    consistent_traces = []
    for q in Q:
        paths = [agent.generate(q) for _ in range(num_paths)]
        answers = [extract_answer(p) for p in paths]
        if majority_vote(answers) >= consistency_threshold:
            consistent_traces.append((q, majority_answer, paths))
    # Fine-tune agent on consistent traces
    for epoch in range(3):
        for q, answer, paths in consistent_traces:
            # Select path with highest path_consistency
            best_path = max(paths, key=lambda p: path_consistency(p, paths))
            loss = CrossEntropyLoss(agent(q, output_logits=True), best_path)
            loss.backward(); optimizer.step()
    return agent
```

#### (C) Why this design
We chose majority vote consistency over model confidence (anti-pattern 3) because confidence is not calibrated for task correctness, while answer agreement is a structural property of the reasoning space. We select the most consistent path for fine-tuning to avoid averaging noise, accepting reduced data volume. The consistency threshold is a hyperparameter; a learned threshold could adapt but would require labeled validation data, which we avoid to maintain autonomy. We use CrossEntropyLoss to maximize likelihood of the consistent path, a standard choice that weighs all tokens equally; a token-level weight based on path consensus could reduce noise but adds complexity.

#### (D) Why it measures what we claim
The quantity `majority_vote(answers) >= consistency_threshold` measures answer correctness under the assumption that correct reasoning paths are more likely to agree than incorrect ones; this assumption fails when the model has systematic bias or the problem has multiple correct answers, in which case consistency may reflect spurious agreement. `path_consistency(p, paths)` is defined as the average token-level agreement between path p and all other paths (computed via cosine similarity of logit distributions at each token). It measures reasoning quality under the assumption that the most consistent path contains the most reliable reasoning steps; this assumption fails when all paths are poor, in which case it selects the least bad path and the learning signal is weak.

## Contribution

(1) Introduces SCKD, a self-supervised method for knowledge acquisition that leverages self-consistency as a training signal, enabling agents to learn from their own reasoning traces without external labels. (2) Establishes a design principle that reasoning consistency can serve as a reliable pseudo-ground-truth for fine-tuning, validated through ablation studies. (3) The method closes the autonomous acquisition loop, addressing the meta-gap identified across prior agent frameworks.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | PhySciBench | Tests scientific multi-step reasoning. |
| Primary metric | Accuracy on correct final answer | Directly measures reasoning success. |
| Baseline 1 | Vanilla LLM agent | No self-improvement; lower bound. |
| Baseline 2 | Confidence-based filtering | Uses model confidence instead of consistency. |
| Baseline 3 | Inference-time self-consistency | Majority vote without fine-tuning. |
| Baseline 4 | Self-training without path selection (majority-vote only) | Ablates path selection to test its necessity. |
| Ablation | SCKD without path selection | Fine-tunes on all paths uniformly. |

### Why this setup validates the claim
This combination tests the central hypothesis that filtering by answer consistency and distilling the most consistent path improves reasoning over alternatives. The vanilla baseline establishes the starting performance. Confidence-based filtering isolates the effect of using consistency vs. confidence. Inference-time self-consistency shows whether fine-tuning on consistent traces provides additional gains beyond voting. The self-training baseline (majority-vote only) tests whether path selection adds value over simply fine-tuning on all filtered paths. The ablation (no path selection) tests whether selecting the single best path is crucial. Accuracy on PhySciBench directly captures correct reasoning outcomes; if the gain is concentrated on tasks requiring multi-step logical deduction (where consistency is informative), the claim is supported. If gains are uniform across trivial and hard subsets, the method may just be averaging noise. To assess compounding benefits, we also evaluate multi-round self-improvement (2 additional rounds) using the same SCKD procedure on the growing consistent trace set.

### Expected outcome and causal chain

**vs. Vanilla LLM agent** — On a question requiring multiple reasoning steps (e.g., calculating orbital period), the vanilla agent may produce a plausible but wrong answer due to one erroneous step (e.g., misapplying Kepler's law). SCKD generates several reasoning paths; most converge to the correct answer while the wrong path is inconsistent. By fine-tuning on the most consistent path, SCKD reinforces the correct chain of reasoning. We expect SCKD to outperform vanilla by 8-12 percentage points on complex multi-step questions, with negligible gain on single-step factual queries.

**vs. Confidence-based filtering** — On a question where the model is overconfident about a wrong answer (e.g., common misconception about force), the confidence-based agent incorrectly keeps the flawed path because its confidence is high. SCKD uses answer agreement among multiple paths; since the wrong answer is not consistently reached across all paths, it is filtered out. Thus SCKD avoids reinforcing systematic biases. We expect SCKD to outperform confidence-based by 5-10 points on questions where known misconceptions lead to high-confidence errors.

**vs. Inference-time self-consistency** — On a question where the correct answer appears in multiple paths but no single path is perfectly correct (e.g., each path has one minor mistake but the final answers agree), inference-time self-consistency yields the correct answer via majority vote but does not improve the underlying reasoning. SCKD fine-tunes on the most consistent path, which—although not flawless—is the best available; this strengthens the model's internal reasoning for future queries. We expect SCKD to match self-consistency on the current test set but achieve higher accuracy after further rounds of self-improvement (not evaluated here). In a single round, SCKD should equal or slightly exceed self-consistency.

**vs. Self-training without path selection** — On a question where some paths contain noisy intermediate steps despite agreeing on the final answer, fine-tuning on all paths (as in self-training) may dilute the signal. SCKD's path selection picks the most coherent reasoning chain, leading to stronger improvement. We expect SCKD to outperform this baseline by 3-6 points on multi-step tasks.

### What would falsify this idea
If SCKD's gain over self-consistency inference is negligible or uniform across all question difficulties, the central claim that fine-tuning on consistent traces drives improvement would be undermined. Similarly, if the ablation (no path selection) matches SCKD, then selecting the best path is unnecessary. If multi-round improvement saturates after one round, the compounding benefit hypothesis fails.

## References

1. Deep Research in Physical Sciences: A Multi-Agent Framework and Comprehensive Benchmark
2. Autonomous chemical research with large language models
3. Augmenting large language models with chemistry tools
4. Training language models to follow instructions with human feedback
5. Measuring and Narrowing the Compositionality Gap in Language Models
