# ARDaG: Adaptive Reward Dataset Generation for Long-Context Reinforcement Learning Post-Training

## Motivation

LongStraw uses a static pre-collected dataset of long-context tasks, which leads to overfitting to the synthetic data distribution and poor generalization to diverse long-context scenarios. The root cause is that static datasets cannot adapt to the policy's evolving weaknesses, causing the policy to exploit spurious patterns. Prior work like DAPO uses dynamic sampling of existing data but does not generate new tasks tailored to the policy's current deficiencies.

## Key Insight

By generating tasks that are specifically challenging for the current policy—where reward variance is high—we create a curriculum that forces the policy to develop robust long-context reasoning, avoiding overfitting to any fixed data distribution.

## Method

## Method

### (A) What it is
ARDaG is an online data generation framework that uses a separate generative model (pre-trained LLaMA-7B, frozen during generation) to produce long-context tasks conditioned on the current policy's weaknesses, as estimated by reward variance across rollouts (calibrated against a noise threshold). It outputs a continuously updated dataset for RL training. Training requires approximately 100 GPU hours on 8×A100 GPUs for a 7B generator and 7B policy.

### (B) How it works
```pseudocode
# ARDaG: Adaptive Reward Dataset Generator
# Input: Policy π, Generator G (LLaMA-7B, frozen), Reward model ensemble R (3 models), Buffer B of (task, reward)
# Hyperparameters: gen_budget = 10 per iteration, diversity_coeff = 0.5, calibration_threshold = 0.1

# Calibration: compute noise floor from a held-out set of 500 tasks with known easy/difficult labels
calibration_variance = compute_reward_variance_on_calibration_set(R, 500)
noise_floor = 1.5 * calibration_variance

for iteration in range(num_iterations):
    # 1. Evaluate policy weaknesses (only tasks with variance > noise_floor)
    tasks = sample_from_buffer(B, 100)
    variances = {task: variance(R(π(task)), n=10) for task in tasks}
    weak_tasks = [task for task in sorted(variances, key=lambda x: variances[x], reverse=True) if variances[task] > noise_floor][:gen_budget]
    
    # 2. Generate new tasks via G conditioned on weak tasks
    for weak_task in weak_tasks:
        new_tasks = G.generate(num_candidates=5, context=weak_task, diversity_coeff)
        for new_task in new_tasks:
            reward = average(R(π(new_task)))  # average over ensemble
            B.push((new_task, reward))
    
    # 3. Update policy using RL on sampled tasks from B
    π.update(using_grpo(batch=sample_from_B(B, batch_size=256), learning_rate=1e-5))
    
    # 4. Update generator G to maximize policy improvement
    # Signal = difference between current reward and average reward of similar tasks (embedding cosine similarity > 0.8)
    G.update(using_reinforce_with_diversity_penalty, learning_rate=1e-5)
```

### (C) Why this design
We chose an adversarial generator over fixed augmentation or random generation because it directly targets the policy's weaknesses, ensuring training data focuses on areas of high learning potential. This requires maintaining a separate generator model and a reward-based training signal, which increases computational cost but avoids the inefficiency of uniform data generation. We opted for a separate generator rather than using the policy itself (as in self-play) because the policy tends to generate tasks it can already solve, reducing learning gains. Using reward variance as a proxy for weakness is a trade-off: it is computationally cheaper than full counterfactual evaluation but may be noisy for low-frequency tasks; we mitigate this by sampling multiple rollouts and calibrating against a noise floor. The diversity penalty in the generator's loss prevents mode collapse into a narrow set of task types, sacrificing some generation focus for broader coverage. This design contrasts with standard adversarial training (e.g., red-teaming) where the adversary maximizes loss; here we maximize policy improvement, which provides a more direct curriculum signal.

### (D) Why it measures what we claim
The reward variance of a task measures **policy uncertainty** because high variance across multiple rollouts reflects inconsistent performance under stochastic policy outputs; this assumption holds when the reward model reliably captures task success, but fails when reward noise dominates, in which case variance reflects measurement error rather than uncertainty. We mitigate this by calibrating the variance threshold using a held-out set of tasks with known difficulty, ensuring only tasks with variance significantly above the noise floor are considered weak. The generator's objective of maximizing policy improvement (difference between current reward and average reward for that task type) measures **learning potential** because it directly quantifies how much the policy stands to gain from training on that task; this assumption fails if the improvement is short-lived (e.g., the policy memorizes the specific instance rather than generalizing), in which case the generated task's benefit is transient. The diversity penalty, computed as cosine similarity between task embeddings (from a pretrained Sentence-BERT model), measures **task distinctiveness** under the assumption that embedding distance correlates with functional diversity; this fails for tasks that are semantically similar but require different reasoning strategies, leading to redundancy despite low embedding similarity.

## Contribution

(1) ARDaG, an online framework that generates long-context tasks conditioned on policy weaknesses for RL post-training, using a separate generative model with an adversarial objective. (2) A curriculum learning principle for RL post-training: generating tasks that maximize expected policy improvement yields better generalization than static or uniformly sampled datasets. (3) An analysis of how reward variance can serve as a proxy for task difficulty, with identified failure modes.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|----------------------|
| Dataset 1 | AIME 2024 | Long-context math benchmark requiring reasoning |
| Dataset 2 | Needle-in-Haystack (long retrieval) | Diverse long-context task beyond math |
| Primary metric | Solve rate | Direct measure of task success |
| Baseline 1 | DAPO | Strong open-source RL baseline |
| Baseline 2 | GRPO (no adaptive gen) | Tests benefit of adaptive generation |
| Baseline 3 | Random Generation | Tests conditioning on weaknesses |
| Baseline 4 | Ensemble Uncertainty (reward variance from ensemble) | Tests novelty of single-model variance signal |
| Ablation | ARDaG w/o diversity | Tests diversity penalty impact |

### Why this setup validates the claim
This setup provides a falsifiable test of ARDaG's central claim: that adaptive generation targeting policy weaknesses improves learning efficiency. AIME 2024 is a challenging long-context reasoning benchmark where weaknesses are diverse; Needle-in-Haystack tests retrieval over extremely long contexts (up to 100k tokens), requiring robust attention. DAPO represents a strong fixed-data RL baseline; outperforming it demonstrates that online adaptive generation adds value beyond scale. GRPO (without adaptive generation) isolates the benefit of the generator by keeping the same RL algorithm, testing whether the data itself drives gains. Random Generation (same generator but not conditioned on weaknesses) tests if conditioning on reward variance is the key factor. The Ensemble Uncertainty baseline uses a reward model ensemble to compute disagreement (variance across ensemble members) instead of rollout variance, isolating the novelty of the specific uncertainty signal. The ablation (no diversity penalty) checks whether diversity is necessary to avoid mode collapse. The solve rate metric directly captures final reasoning performance. If ARDaG outperforms DAPO and GRPO while Random Generation and the ablation lag, we confirm the causal chain: targeting weaknesses yields focused improvement; diversity prevents overfitting.

### Expected outcome and causal chain

**vs. DAPO** — On a case where the policy struggles with multi-step arithmetic, DAPO's fixed dataset may contain few such examples, so it improves slowly across all tasks. Our method detects high variance on such tasks via reward rollouts, generates similar tasks, and trains the policy repeatedly, leading to faster mastery. We expect ARDaG to show a 5-10% higher solve rate on complex multi-step problems, with parity on simple tasks.

**vs. GRPO (no adaptive gen)** — On a case where the policy has a systematic error in geometry, GRPO learns from a static dataset and may never see enough geometry examples to correct it. Our method identifies that geometry tasks have high variance, generates novel geometry variants, and provides a focused curriculum. We expect ARDaG to have a 10-15% higher solve rate on geometry, while GRPO plateaus.

**vs. Random Generation** — On a case where random generation produces tasks the policy already solves (low variance), it wastes training budget. Our method filters for high-variance tasks, concentrating compute on weak areas. We expect ARDaG to achieve the same solve rate as Random Generation but with 30% fewer training steps, or a higher final score given fixed steps.

**vs. Ensemble Uncertainty** — On a case where the reward model is noisy (e.g., ambiguous instruction), rollout variance may be high even when policy is confident; ensemble disagreement better captures epistemic uncertainty. We expect ARDaG's calibration threshold to mitigate this, yielding comparable performance to Ensemble Uncertainty on noisy tasks but maintaining the advantage on clean tasks where rollout variance is more efficient. If Ensemble Uncertainty outperforms ARDaG on a majority of tasks, it would suggest that our simpler variance estimate is insufficient and the method's core assumption is fragile.

### What would falsify this idea
If ARDaG's gain over GRPO is uniform across all task subsets rather than concentrated in high-variance tasks (as measured by pre-training variance), then the claim that targeting weaknesses drives improvement is false. Similarly, if the ablation without diversity penalty performs equally well, then diversity is not necessary.

## References

1. LongStraw: Long-Context RL Beyond 2M Tokens under a Fixed GPU Budget
2. DAPO: An Open-Source LLM Reinforcement Learning System at Scale
3. Qwen2.5 Technical Report
4. HybridFlow: A Flexible and Efficient RLHF Framework
5. ReMax: A Simple, Effective, and Efficient Reinforcement Learning Method for Aligning Large Language Models
