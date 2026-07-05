# Consistency-Driven Policy Optimization for Self-Supervised Medical Visual Question Answering

## Motivation

Current RL methods for medical VQA, such as MRPO and MedCEG, rely on external reward signals—either final answer correctness or human-annotated process rewards—which limits scalability to domains without annotated reasoning chains. The structural problem is that credit assignment is tied to an external oracle, preventing self-supervised improvements. This dependency creates a bottleneck because medical reasoning often lacks explicit step-wise annotations.

## Key Insight

Consistency across multiple independent reasoning paths is a reliable proxy for correctness in medical VQA because the same answer typically requires a limited set of clinically valid reasoning steps, so steps that appear frequently in convergent paths are likely necessary and sufficient.

## Method

## Consistency-Driven Policy Optimization (CDPO)

### (A) What it is
**Consistency-Driven Policy Optimization (CDPO)** is a self-supervised RL method for medical VQA that uses step-level consistency across multiple stochastic reasoning paths as an internal reward, eliminating the need for external reward signals. Input: a medical image and question; Output: a policy model fine-tuned to generate correct reasoning paths. A small labeled calibration set of 512 examples (random subset of training data) is used to verify majority answer correctness during training.

### (B) How it works
```python
# Pseudocode for one training iteration
for each (image, question) in training set:
    # Generate N stochastic reasoning paths using current policy π
    paths = [π(question, image) for _ in range(N)]  # N=8
    # Step-level parsing: split each path into steps using spaCy sentencizer
    for each path, decompose into list of steps via sentence segmentation
    # Determine final answers for each path
    answers = [extract_answer(path) for path in paths]
    majority_answer = mode(answers)

    # Calibration: if sample is in calibration set and majority answer is wrong, skip update
    if sample_id in calibration_set and majority_answer != calibration_set[sample_id].correct_answer:
        continue  # skip this sample (no gradient update)

    # Build step frequencies across convergent and divergent paths
    convergent_steps = defaultdict(int)
    divergent_steps = defaultdict(int)
    for path in paths:
        answer = extract_answer(path)
        for step in path.steps:
            if answer == majority_answer:
                convergent_steps[step] += 1
            else:
                divergent_steps[step] += 1
    # Compute step reward
    def step_reward(step):
        freq_conv = convergent_steps[step]
        freq_div = divergent_steps[step]
        return (freq_conv - freq_div) / N  # normalized to [-1, 1]
    # Aggregate step rewards to path-level reward (mean)
    path_rewards = [mean(step_reward(s) for s in path.steps) for path in paths]
    # Compute baseline (mean reward across paths)
    baseline = mean(path_rewards)
    # Policy gradient update (REINFORCE with baseline)
    for path, reward in zip(paths, path_rewards):
        advantage = reward - baseline
        log_prob = path.log_probability()
        loss = -advantage * log_prob
    total_loss = mean(loss for all paths)
    optimize(π, total_loss)
```

### (C) Why this design
We chose step-level rewards over path-level rewards because step-level granularity provides finer credit assignment, crucial for identifying which reasoning steps are actually valid; the trade-off is increased computational cost from step parsing and frequency counting. We used REINFORCE with baseline instead of PPO or GRPO because REINFORCE is simpler and avoids the need for a critic network, which would introduce additional components that could destabilize training in the absence of external rewards; the cost is higher variance, which we mitigate via the baseline and by averaging over N paths. We set N=8 as a hyperparameter balancing statistical reliability and training efficiency; smaller N yields noisier frequency estimates, while larger N increases generation cost. The frequency-based reward (freq_conv - freq_div) directly operationalizes consistency: steps appearing more in convergent paths are rewarded, and steps in divergent paths are penalized. This design differs from prior work like MedCEG, which requires pre-constructed evidence graphs, and MRPO, which relies on external step-wise rewards; our key difference is that the reward is derived entirely from the model's own generations, making it self-supervised. A domain expert would not describe our method as a variant of MedCEG or MRPO because those methods depend on external oracles, while CDPO's reward is generated autonomously from consistency patterns.

### (D) Why it measures what we claim
The computational quantities in (B) operationalize the motivation-level concepts as follows: `freq_conv[step]` measures **causal necessity** of a step under the assumption that valid steps are those common to many paths leading to the correct answer; this assumption fails when the model systematically repeats a spurious step across paths due to initialization bias, in which case the metric reflects frequency of co-occurrence rather than necessity. `freq_div[step]` measures **causal harmfulness** under the assumption that steps in divergent paths are erroneous; this assumption fails when a correct step is present in a divergent path by chance (e.g., noise from answer extraction), in which case the metric over-penalizes a valid step. The `majority_answer` serves as a **proxy for ground truth** under the assumption that consistency implies correctness; this assumption fails when the majority answer is itself wrong (e.g., all paths make the same mistake), in which case the entire reward signal becomes misleading. The `baseline` captures the **average consistency** of the batch, ensuring that advantage reflects relative step quality; this assumption fails when path rewards are not independent, a violation unlikely in practice due to stochastic generation.

## Contribution

(1) A self-supervised reinforcement learning framework for medical VQA that uses step-level consistency across multiple stochastic reasoning paths as an internal reward signal, eliminating the need for external annotations. (2) A credit assignment mechanism that rewards steps appearing in convergent paths and penalizes steps in divergent paths, providing fine-grained process rewards without human supervision. (3) A training pipeline integrating consistency-based rewards with REINFORCE policy optimization, demonstrating that purely self-supervised signals can drive RL improvement in medical reasoning tasks.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|-----------------------|
| Dataset | VQA-RAD | Standard medical VQA dataset |
| Primary metric | Answer accuracy | Directly measures final answer correctness |
| Baseline 1 | GRPO | Outcome-only RL without step rewards |
| Baseline 2 | MRPO | Step-aware RL with external rewards |
| Ablation-of-ours | CDPO without step-level rewards | Tests importance of step granularity |

### Why this setup validates the claim
This experimental design tests the central claim that CDPO's self-supervised step-level consistency reward improves reasoning over outcome-only RL (GRPO) and external-step RL (MRPO). Comparing to GRPO isolates the benefit of step-level credit assignment. Comparing to MRPO tests whether internal consistency can rival external supervision. The ablation of removing step-level rewards measures whether granularity is crucial. Accuracy on VQA-RAD directly reflects reasoning quality. A falsifiable pattern emerges: if CDPO consistently outperforms GRPO on multi-step questions but matches on simple ones, the step-level mechanism is validated.

### Expected outcome and causal chain

**vs. GRPO** — On a multi-step diagnostic question (e.g., "Given these imaging findings, what is the most likely diagnosis?"), GRPO's outcome-only reward may reinforce a reasoning path that reaches the correct answer by chance via a flawed intermediate step. Since GRPO lacks step-level credit assignment, it cannot distinguish causally necessary steps. Our method rewards steps that appear consistently across paths leading to the majority answer, thus penalizing spurious steps. We expect CDPO to achieve higher accuracy (e.g., +5-10%) on multi-step questions, while performing similarly on single-step questions.

**vs. MRPO** — On a case where an external step reward is noisy or unavailable (e.g., rare disease with few annotations), MRPO's reward may be misaligned, penalizing correct steps or rewarding incorrect ones. Our method derives reward from internal consistency, which is robust to annotation noise. For such cases, CDPO is expected to outperform MRPO (e.g., +3-7% on rare disease subsets), while on well-annotated data, MRPO may slightly lead due to external supervision.

**Ablation (path-level only)** — On a complex reasoning chain where only a subset of steps are correct, the path-level reward gives uniform reward to all steps in a path, failing to isolate valid sub-steps. Our step-level version can assign differential rewards, leading to better reasoning improvements. We expect step-level to outperform path-level by a noticeable margin (e.g., 2-4% overall accuracy).

### What would falsify this idea
If CDPO's gain over GRPO is uniform across all question types, or if the step-level ablation matches step-level performance, then consistency reward is not providing the claimed benefit. Also, if MRPO with external reward significantly outperforms CDPO on all subsets, then self-supervised reward is insufficient.

## References

1. Breaking Failure Cascades: Step-Aware Reinforcement Learning for Medical Multimodal Reasoning
2. Chiron-o1: Igniting Multimodal Large Language Models towards Generalizable Medical Reasoning via Mentor-Intern Collaborative Search
3. GRPO-λ: Credit Assignment improves LLM Reasoning
4. MedTVT-R1: A Multimodal LLM Empowering Medical Reasoning and Diagnosis
5. GLM-4.5V and GLM-4.1V-Thinking: Towards Versatile Multimodal Reasoning with Scalable Reinforcement Learning
6. R-4B: Incentivizing General-Purpose Auto-Thinking Capability in MLLMs via Bi-Mode Annealing and Reinforce Learning
7. MedCEG: Reinforcing Verifiable Medical Reasoning with Critical Evidence Graph
8. CAPO: Towards Enhancing LLM Reasoning through Generative Credit Assignment
