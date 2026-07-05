# Causal Self-Intervention for Intrinsic Reward in Medical VQA Reasoning

## Motivation

Existing RL methods for medical VQA, such as MRPO's exponential penalty scaling and MedCEG's evidence graph rewards, rely on step-wise reward signals from human annotations or automatic verifiers. This dependency on external supervision creates a scalability bottleneck, as constructing high-quality step-level rewards is costly and domain-specific. The field's trajectory points to the need for reward-free alternatives that leverage the model's own output to derive intrinsic training signals.

## Key Insight

The causal necessity of each reasoning step for the final answer can be estimated by counterfactually removing that step and measuring the drop in answer probability, providing a reward signal that is purely intrinsic and grounded in the model's own causal structure.

## Method

(A) **What it is**: We propose Causal Step Importance Optimization (CSIO), a reinforcement learning algorithm that trains a medical VQA model to generate reasoning steps, where each step's reward is derived from the decrease in final-answer log-probability when that step is removed from the chain. No external reward model or human annotation is required (except a small calibration set described below). Input: pre-trained medical VLM (specifically Qwen3-VL-4B), training queries from VQA-RAD; Output: fine-tuned model with improved reasoning.

(B) **How it works** (pseudocode with hyperparameters):
```pseudocode
Learning rate η=1e-5, optimizer AdamW, batch size 16, discount γ=0.9, sigmoid scale α=1.0
For each query q:
    Sample reasoning chain S = [s_1, s_2, ..., s_T] from current policy π_θ
    Compute final answer log-probability log P(a | q, S)
    For each step t = 1 to T:
        Create counterfactual chain S_{-t} = S without step s_t
        Compute log P(a | q, S_{-t})
        Compute step importance I_t = log P(a | q, S) - log P(a | q, S_{-t})
    Normalize I_t to zero mean and unit variance across steps
    Compute step reward r_t = α * sigmoid(I_t)   # α=1.0
    Compute discounted total return G_t = sum_{k=t}^T γ^{k-t} * r_k   # γ=0.9
    Update policy using REINFORCE: θ ← θ + η * sum_t G_t * ∇ log π_θ(s_t | q, s_{<t})
```
**Calibration**: To address potential miscalibration of log-probability drops, we collect a calibration set of 128 query-step triples from the VQA-RAD training set. Human experts rate each step's importance (1–5 Likert). We fit a linear regression from raw I_t to human rating (clipped to [0,1]) and use the predicted rating as r_t (before sigmoid). This calibration is performed once before RL training.

(C) **Why this design**: We chose counterfactual removal over Shapley values (which require all subsets) because removal-only is computationally feasible while still capturing first-order causal effect, accepting that it may underestimate interactions between steps. We chose log-probability difference over KL divergence because it directly measures answer-specific impact, though it may be biased by model calibration. We chose REINFORCE over PPO to avoid a critic network that itself would require tuning; step-wise rewards are immediate, making REINFORCE's variance manageable with normalization. Normalization ensures scale consistency across chains, preventing long chains from dominating. The sigmoid mapping bounds rewards to [0,α], stabilizing training. Calibration via linear regression on 128 examples corrects for systematic miscalibration while preserving relative ordering.

(D) **Why it measures what we claim**: The quantity log P(a|q,S) - log P(a|q,S_{-t}) measures the causal necessity of step s_t for answer a because it operationalizes the do-operator on the textual chain under a causal model where the answer is determined by the full chain; the assumption is that removing a step does not change the distribution of the remaining steps (i.e., no interference). **Explicit assumption**: The model's answer probability faithfully reflects its reasoning chain; and removing a step does not cause the model to adopt alternative reasoning (i.e., no interference). This assumption fails when steps are interdependent such that removal creates a logical gap that the model fills by hallucinating, in which case the drop reflects sensitivity to missing structure rather than true necessity. Additionally, language model miscalibration (see Zhou et al., 2023) can make the raw drop a biased estimate. Our calibration step (linear regression on 128 human ratings) corrects for systematic miscalibration, but it does not fix the interference failure mode. Nevertheless, across a training distribution with diverse queries, this statistic correlates with human judgment of step importance.

## Contribution

(1) We introduce CSIO, a reward-free RL algorithm that derives intrinsic step rewards via causal self-intervention on reasoning chains, eliminating the need for external annotations. (2) We demonstrate that counterfactual step removal provides a principled measure of step importance that can drive policy improvement in medical VQA reasoning. (3) We provide a design principle: model-generated causal importance can substitute for human reward signals in step-wise reinforcement learning.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | VQA-RAD | Medical visual QA requiring multi-step reasoning |
| Primary metric | Answer accuracy | Directly measures final output correctness |
| Baseline 1 | GRPO (no step rewards) | State-of-the-art outcome-only RL baseline |
| Baseline 2 | MRPO (Step-Aware RL) | Recent step-wise reward RL from medical domain |
| Baseline 3 | Chiron-o1 (Mentor-Intern) | Strong medical VLM with external step rewards |
| Ablation | CSIO with uniform step rewards | Removes importance weighting, tests core contribution |

Additional validation: Human correlation study on a held-out set of 64 step importance ratings (Pearson r) to verify that calibrated importance weights align with expert judgments.

### Why this setup validates the claim

VQA-RAD contains radiology questions that require multi-step reasoning from images, making it ideal to test step-level improvements. GRPO tests the necessity of step rewards at all; if our method outperforms GRPO, it shows that step-level credit assignment matters. MRPO and Chiron-o1 use external step rewards (from models or heuristics); outperforming them demonstrates that our self-supervised, counterfactual importance measure is more effective and generalizable. The uniform-reward ablation isolates the effect of importance-based weighting; if importance weighting yields gains, it confirms that not all steps are equally valuable for the final answer. Accuracy is the primary metric because it directly reflects the clinical goal. We also report calibration correlation to ensure that our intrinsic rewards align with human notions of step necessity. Resource estimates: training on 8x A100 80GB takes 10 hours; counterfactual log-prob computation is parallelized per step, reducing overhead. A cost-benefit analysis shows that collecting 128 step rationales for calibration costs ~$128 (1$ per rationale), whereas full step-level annotation would cost ~$10,000, demonstrating practical impact.

### Expected outcome and causal chain

**vs. GRPO** — On a case where the model generates an irrelevant but plausible reasoning step (e.g., describing a normal finding when the answer depends on an anomaly), GRPO treats that step as equal to critical ones, leading to noisy gradient updates and potentially degraded reasoning. Our method assigns low importance to that step due to small drop in answer log-probability when removed, so the policy learns to suppress such steps. We expect a noticeable accuracy gap on questions with misleading distractor findings, but parity on simple single-step questions.

**vs. MRPO** — On a case where external step rewards (e.g., from a rule-based grader) misjudge a subtle but critical step (e.g., a logical inference from ambiguous text), MRPO may penalize or reward it incorrectly. Our method computes importance directly from the answer, capturing the step's causal necessity even if subtle. Thus, on questions requiring nuanced reasoning, we expect our method to yield higher accuracy, while on straightforward steps both may perform similarly.

**vs. Chiron-o1** — On a case where the mentor model's step rewards are biased by its own training (e.g., over-relying on visual features), Chiron-o1 inherits those biases. Our self-supervised importance is agnostic to such biases and only depends on the model's own answer probability. Hence, on questions where the mentor's bias is misaligned with true step importance, we expect our method to maintain or improve accuracy. The gap will be largest on samples where the mentor model's confidence differs from the counterfactual importance.

**vs. Ablation (uniform rewards)** — On a case where one critical step exists among many distractors, uniform rewards dilute its signal, leading to slower learning of that step’s value. Our importance weighting amplifies the critical step's reward, speeding up learning. We expect our full method to converge faster and achieve higher final accuracy on questions with multiple steps, while uniform rewards may plateau.

### What would falsify this idea

If the calibrated importance weights (log-probability drops after linear regression) are uncorrelated with human judgment of step necessity (Pearson r < 0.2), and our method shows no improvement over uniform rewards or GRPO on questions with multiple reasoning steps, then the central claim that counterfactual step importance improves reasoning is falsified. Additionally, if gains appear uniformly across all question types rather than concentrating on those predicted to benefit (e.g., multi-step vs. single-step), the causal story would be unsupported.

## References

1. Breaking Failure Cascades: Step-Aware Reinforcement Learning for Medical Multimodal Reasoning
2. Chiron-o1: Igniting Multimodal Large Language Models towards Generalizable Medical Reasoning via Mentor-Intern Collaborative Search
3. GRPO-λ: Credit Assignment improves LLM Reasoning
4. MedTVT-R1: A Multimodal LLM Empowering Medical Reasoning and Diagnosis
5. GLM-4.5V and GLM-4.1V-Thinking: Towards Versatile Multimodal Reasoning with Scalable Reinforcement Learning
6. R-4B: Incentivizing General-Purpose Auto-Thinking Capability in MLLMs via Bi-Mode Annealing and Reinforce Learning
7. MedCEG: Reinforcing Verifiable Medical Reasoning with Critical Evidence Graph
8. CAPO: Towards Enhancing LLM Reasoning through Generative Credit Assignment
