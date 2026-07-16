# Self-Consistency Driven Exploration for Open-Ended Reasoning

## Motivation

Existing post-training methods for LLMs, such as the Proxy Exploration and Reusable Guidance (PUST) framework, rely on an external reward signal (e.g., final answer correctness) to guide exploration. This assumption limits their applicability to tasks where ground-truth rewards are unavailable or costly. The root cause is that exploration is driven by an external proxy reward, which inherits the structural limitation of requiring a reward model or verifier. A reward-free exploration capability is needed to enable open-ended reasoning without predefined correctness criteria.

## Key Insight

Self-consistency among multiple independent reasoning trajectories serves as a reliable intrinsic reward signal because correct reasoning in well-defined domains produces consistent outputs, while incorrect reasoning yields diverse errors, making agreement a natural proxy for correctness.

## Method

# Self-Consistency Driven Exploration (SCDE)

(A) **What it is:** SCDE is a reinforcement learning algorithm for fine-tuning language models using an intrinsic reward computed from the self-consistency of multiple sampled reasoning paths, eliminating the need for external reward signals. Input is a pretrained LLM; output is an LLM fine-tuned for open-ended reasoning tasks. SCDE relies on the assumption that correct reasoning yields consistent outputs while incorrect reasoning yields diverse errors.

(B) **How it works:**

```pseudocode
# Hyperparameters: N (number of trajectories, default=8), λ (diversity bonus weight, default=0.1), α (learning rate, default=1e-5)
# Calibration set: optionally, a small labeled set of 512 examples with ground-truth answers

for each input prompt x in training set:
  1. Sample N reasoning trajectories {τ_i} ~ π_θ(·|x) using temperature sampling (T=0.8).
  2. For each τ_i, extract final answer a_i (e.g., last line or special token).
  3. Compute self-consistency reward:
     R_consistency = (2/(N*(N-1))) * Σ_{i < j} sim(a_i, a_j)
     where sim is lexical Jaccard similarity (token overlap of answer strings).
  4. (Optional) Correct for systematic errors: if a calibration set exists, fit a linear regressor to map R_consistency to expected accuracy on that set, then use predicted accuracy as reward instead of raw R_consistency.
  5. Compute diversity bonus:
     R_diversity = λ * (1 - (2/(N*(N-1))) * Σ_{i < j} sim(τ_i, τ_j))
     where sim(τ_i, τ_j) is cosine similarity of average token embeddings.
  6. Total reward R_total = R_consistency + R_diversity.
  7. Update policy π_θ using REINFORCE:
     ∇θ J ≈ (1/N) * Σ_i [R_total * ∇θ log π_θ(τ_i | x)]
  8. (Optional) Clip gradients to max_norm=1.0 and apply weight decay 0.01.
```

(C) **Why this design:** We chose REINFORCE over PPO because the variance of our intrinsic reward is manageable due to trajectory averaging, and the simplicity reduces implementation overhead; the trade-off is lower sample efficiency, but the reward is computed online without a critic. We used lexical Jaccard similarity for answer agreement instead of embedding-based similarity because it is interpretable and computationally cheap, accepting that it may miss semantically equivalent paraphrases (e.g., "x=5" vs. "five"). We added a diversity bonus in step 4 to prevent the model from collapsing to a single token-motif, trading off some self-consistency pressure for broader exploration. The diversity bonus uses embedding similarity instead of lexical to capture semantic diversity, acknowledging that it may over-penalize valid paraphrases. These three decisions collectively maintain a balance between exploration breadth and convergence to consistent reasoning.

(D) **Why it measures what we claim:** The self-consistency reward (R_consistency) measures the agent's exploration progress towards correct reasoning because high agreement among independent trajectories indicates that the model has discovered a stable pattern; this assumes that for tasks like math or logic, correct reasoning yields an unambiguous answer while errors are diverse. This assumption fails when multiple incorrect but consistent outputs exist (e.g., a systematic bias in addition), in which case R_consistency reflects internal coherence rather than correctness. To test this failure mode, we inject systematic errors (e.g., always adding 2) and measure the correlation between R_consistency and accuracy; a low correlation would indicate the assumption is violated. The diversity bonus (R_diversity) measures exploration breadth because it penalizes trajectory population similarity; this assumes that diverse trajectories cover distinct reasoning strategies, but when the model's prior is very narrow, diversity may be superficial. The REINFORCE update operationalizes the reward as a gradient signal that pushes the policy toward trajectories with high self-consistency and diversity. The lexical answer similarity operationalizes the notion of "consistency" as token-level overlap, assuming that correct answers are lexically canonical; this fails when correct answers have multiple phrasings, leading to underrewarding valid alternatives.

## Contribution

(1) A novel intrinsic reward mechanism, Self-Consistency Driven Exploration (SCDE), that uses the agreement among multiple sampled reasoning trajectories as the sole training signal for RL fine-tuning of LLMs, removing the dependence on external reward signals. (2) Empirical findings showing that self-consistency correlates with correctness on multiple reasoning benchmarks, and that SCDE enables effective exploration without ground-truth labels, achieving performance competitive with supervised methods. (3) An analysis of the relationship between self-consistency and correctness, revealing that the reward assumption holds for structured tasks but degrades in open-ended generation domains.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset 1 | GSM8K | Tests multi-step math reasoning. |
| Dataset 2 | StoryCloze | Tests open-ended story endpoint selection. |
| Dataset 3 | Synthetic Arithmetic (e.g., 2-digit addition with systematic errors) | Known ground truth; tests reward-accuracy correlation. |
| Primary metric | Answer exact-match accuracy (for GSM8K and Synthetic); accuracy of selected ending (for StoryCloze) | Directly measures reasoning correctness. |
| Baseline 1 | PPO with reward model (PPO-RM) | Standard RL fine-tuning with external reward. |
| Baseline 2 | Supervised fine-tuning (SFT) | Trains on golden reasoning chains. |
| Baseline 3 | Self-consistency decoding (SC) | No fine-tuning, multiple samples. |
| Ablation 1 | SCDE without diversity bonus (SCDE-NoDiv) | Isolates effect of diversity bonus. |
| Ablation 2 | SCDE with calibration (SCDE-Cal) | Tests if calibration corrects systematic errors. |

### Why this setup validates the claim

This setup directly tests whether SCDE's intrinsic self-consistency reward can replace external rewards or supervision. Comparing to PPO-RM shows if SCDE avoids reward model pitfalls. Comparing to SFT tests if exploration compensates for scarce demonstrations. The SC baseline establishes a lower bound from the pretrained model alone. The ablations isolate the diversity bonus's contribution and the calibration's effect. The Synthetic Arithmetic dataset provides controlled ground truth to directly measure correlation between self-consistency reward and true accuracy. StoryCloze tests generalization to open-ended tasks where the answer is not a single number. Using a single primary metric (accuracy) on standard benchmarks ensures clear, falsifiable outcomes.

### Expected outcome and causal chain

**vs. PPO-RM** — On a case where the reward model assigns high reward to a plausible but wrong answer chain (e.g., arithmetic slips), PPO-RM reinforces that chain because it optimizes the proxy reward. Our method instead relies on self-consistency across multiple trajectories; the wrong chain rarely agrees with others, so it receives low reward. We expect a noticeable gap (e.g., 5-10% accuracy) on subsets with high reward noise, but parity on clean ones.

**vs. SFT** — On a novel multi-step problem not represented in the training demonstrations, SFT may collapse to a single memorized pattern and fail because it lacks diverse strategies. Our method uses a diversity bonus to encourage exploration of different reasoning paths, making it more likely to find a correct one. We expect SCDE to outperform SFT on hard or rare problem types (e.g., problems requiring unusual operations), while SFT may excel on common ones.

**vs. SC decoding** — On a problem where the pretrained model produces inconsistent answers due to sampling temperature (e.g., a math problem with multiple possible misinterpretations), SC decoding achieves moderate accuracy by voting but still suffers from inherent unreliability. Our method fine-tunes the policy to increase self-consistency, directly raising the probability of agreeing trajectories. Thus, we expect SCDE to consistently outperform SC decoding across all difficulty levels, with a larger margin on problems where the pretrained model is most erratic.

**vs. SCDE-NoDiv** — On problems where diverse strategies are needed to avoid local optima, SCDE (with diversity bonus) should outperform SCDE-NoDiv; on simple problems where a single consistent path suffices, the diversity bonus may slightly degrade performance by encouraging unnecessary variation. Expected gap: moderate (e.g., 2-5%) on hard problems.

**vs. SCDE-Cal** — On problems where systematic errors produce consistent but wrong answers (e.g., Synthetic Arithmetic with forced addition bias), calibration should realign reward with accuracy, so SCDE-Cal outperforms standard SCDE. On clean problems, both should perform similarly.

### What would falsify this idea

The idea would be falsified if SCDE shows no improvement over SC decoding (i.e., self-consistency reward fails to shift the policy) or if the diversity bonus degrades performance on problems where consistent but narrow strategies suffice (e.g., simple arithmetic), indicating that the bonus unduly penalizes valid correct paths. Additionally, if on Synthetic Arithmetic the correlation between R_consistency and accuracy is low (<0.5), the core assumption is violated.

## References

1. Proxy Exploration and Reusable Guidance: A Modular LLM Post-Training Paradigm via Proxy-Guided Update Signals
2. Incentivizing Strong Reasoning from Weak Supervision
3. Weak-to-Strong Generalization beyond Accuracy: a Pilot Study in Safety, Toxicity, and Legal Reasoning
4. LegalBench: A Collaboratively Built Benchmark for Measuring Legal Reasoning in Large Language Models
5. Weak-to-Strong Generalization: Eliciting Strong Capabilities With Weak Supervision
6. Discovering Language Model Behaviors with Model-Written Evaluations
7. Constitutional AI: Harmlessness from AI Feedback
8. Measuring Progress on Scalable Oversight for Large Language Models
