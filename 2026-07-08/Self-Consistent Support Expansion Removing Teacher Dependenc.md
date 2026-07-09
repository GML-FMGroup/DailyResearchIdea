# Self-Consistent Support Expansion: Removing Teacher Dependence via Cycle-Consistent Trajectory Verification

## Motivation

TREK requires a high-quality teacher model (DeepSeek-V4) to generate verified trajectories for hard prompts, limiting applicability when no such teacher is available. The core bottleneck is that support expansion relies on an external verification signal; we need a self-supervised verification that matches the teacher's ability to identify valid trajectories.

## Key Insight

Cycle-consistency between a trajectory and its originating prompt acts as a self-supervised verification because a trajectory that can accurately reconstruct the prompt must have captured the prompt's essential constraints, serving as a proxy for correctness without an external teacher.

## Method

## Self-Consistent Support Expansion (SCSE)

### (A) What it is
We propose Self-Consistent Support Expansion (SCSE), a method that trains a forward policy (the student) and a backward prompt-reconstruction model jointly. During exploration, the forward policy generates candidate trajectories; each trajectory is filtered by a cycle-consistency score from the backward model. Only self-verified trajectories are used for forward-KL support expansion.

### (B) How it works
```pseudocode
Initialize: Student forward policy π_θ (Mistral 7B), backward model B_φ (Bidirectional LSTM, hidden=256, 2 layers, linear projection to token logits).
Pre-train B_φ on (prompt, trajectory) pairs from a seed dataset (e.g., 1000 correct trajectory-prompt pairs from GSM8K training set, filtered by ground truth answer).

for each iteration:
  1. Sample a batch of hard prompts P (where π_θ has low pass rate or confidence, e.g., based on self-consistency).
  2. For each prompt p in P:
     a. Forward Generation: Sample K candidate trajectories {τ_k} ~ π_θ(·|p) using temperature T (e.g., T=1.0).
     b. Cycle-Consistency Verification:
        For each τ_k, compute reconstruction score: s_k = log B_φ(p | τ_k).
        Select top r trajectories {τ_k} with highest s_k (e.g., r=5).
  3. Support Expansion: Perform a short forward-KL training phase on the selected trajectories:
     L = E_{p, τ in selected} [ -log π_θ(τ | p) ].
     Update π_θ with a small learning rate (3e-5) for M steps (e.g., M=50).
  4. Backward Model Update: Periodically (every N=10 iterations), update B_φ on a replay buffer of 10000 (prompt, trajectory) pairs, where trajectories are from the current π_θ's generations that pass cycle-consistency. Use standard supervised learning:
     L_B = E_{p, τ in buffer} [ -log B_φ(p | τ) ].
     Also optionally add negative examples (trajectories with low s_k) to improve discrimination.
  5. Calibration: Every 100 iterations, reset B_φ using the seed dataset of 100 human-verified (prompt, trajectory) pairs to prevent drift. Then continue training.
  6. Return to step 1, increasing temperature or number of samples if needed.
```
Hyperparameters: K=16, r=5, T=1.0, M=50, N=10, replay buffer size=10000, learning rate for forward policy: 3e-5, backward model: 1e-4.

### (C) Why this design
We chose cycle-consistency over a learned discriminator (as in GAD) because an adversarial discriminator requires careful tuning and may suffer from mode collapse; cycle-consistency provides a deterministic, self-supervised signal based on reconstruction likelihood. We chose a separate backward model instead of using the forward model's own log-probs (anti-pattern 3) because forward log-probs are calibrated to next-token prediction, not to trajectory-prompt faithfulness; a dedicated backward model explicitly learns the mapping from trajectory to prompt. We chose to update the backward model periodically on self-consistent trajectories rather than on all generations to avoid reinforcing unreliable backward predictions; this trades off some data efficiency for stability. The forward-KL expansion uses a small number of M steps to avoid overfitting to the filtered trajectories while still incorporating new modes. Accepting the cost of needing a seed set for initial backward model training, but this can be minimal (e.g., from a single teacher demonstration in TREK's initial phase).

### (D) Why it measures what we claim
Load-bearing assumption: The cycle-consistency score log B_φ(p|τ) reliably distinguishes correct trajectories from incorrect ones for hard prompts. We calibrate this by periodically resetting the backward model using a small set of 100 human-verified trajectory-prompt pairs (e.g., correct solutions from the seed dataset) every 100 iterations to prevent confirmation bias. The cycle-consistency score s_k = log B_φ(p | τ) measures the **causal fidelity** of the trajectory τ to the prompt p because we assume that the backward model learns a generative process that inverts the forward generation; this assumption fails when the backward model is poorly trained or when the trajectory is ambiguous (i.e., multiple different prompts could produce the same τ), in which case s_k reflects reconstruction likelihood under the learned distribution rather than true causal necessity. The selection of top r trajectories based on s_k operationalizes **support expansion** because we assume that trajectories with high s_k are more likely to be valid (i.e., correctly solve the task); this assumption fails when the backward model is biased toward short or common trajectories, causing it to reject correct but novel solutions. The forward-KL training on selected trajectories directly increases the student's likelihood on those self-verified trajectories, which we claim expands the student's support on hard prompts; the causal chain is that high s_k trajectories are valid, and increasing their likelihood incorporates them into the student's distribution.

## Contribution

(1) A self-supervised trajectory verification mechanism using cycle-consistency that eliminates the need for an external teacher in support expansion. (2) A scalable method for joint training of forward and backward models that enables continuous improvement without quality degradation. (3) A demonstration that self-verified trajectories can achieve comparable support expansion to teacher-verified trajectories.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Datasets | GSM8K (math), HumanEval (code) | Two diverse benchmarks to test generality |
| Primary metric | Accuracy (Pass@1), Pass@5, solution diversity (unique final answers) | Measures task completion, coverage, and mode collapse |
| Baseline 1 | TREK | Teacher-guided exploration without self-consistency |
| Baseline 2 | GAD | Learned discriminator for trajectory filtering |
| Baseline 3 | GRPO | Standard on-policy RL baseline |
| Baseline 4 | CycleConsistency-baseline | Uses a pre-trained T5 model fine-tuned on paraphrasing to compute cycle-consistency (text-to-text mapping) |
| Baseline 5 | SCSE-zeroshot | SCSE without any seed data; backward model pre-trained on 1000 synthetic (prompt, trajectory) pairs from a rule-based generator (e.g., randomly sampled arithmetic problems) |
| Ablation of ours | SCSE w/o cycle-consistency | Removes cycle-consistency filtering |

Additionally, before the main experiment, we conduct a correlation experiment on a simplified domain (Gridworld, 8x8, with 1000 prompts: "reach (x,y)" with a sequence of moves as trajectory) to verify that cycle-consistency scores correlate with ground-truth correctness (Pearson r > 0.6). This validates the core assumption.

### Why this setup validates the claim
This combination tests the central claim that cycle-consistency filtering improves support expansion by selecting faithful trajectories. GSM8K and HumanEval offer prompts with multiple valid solution paths, making faithfulness crucial. TREK tests whether a teacher oracle is necessary; GAD tests whether an adversarial discriminator suffices; GRPO tests whether any filtering is beneficial; the CycleConsistency-baseline tests whether a generic cycle-consistency model (without task-specific adaptation) works; SCSE-zeroshot tests the method's robustness to absence of seed data; and the ablation isolates the contribution of cycle-consistency. Accuracy and diversity directly reflect whether the student learns correct solutions, and a gap on hard prompts would confirm that filtering helps where the policy is uncertain.

### Expected outcome and causal chain
**vs. TREK** — On a case where the student generates a correct but novel solution path unknown to the teacher, TREK fails to include it because the teacher filter rejects it. Our method uses cycle-consistency with a learned backward model which can verify novel paths if they faithfully reconstruct the prompt. Thus we expect a noticeable gap on prompts requiring out-of-distribution reasoning.

**vs. GAD** — On a case where the discriminator overfits to spurious features (e.g., trajectory length), GAD may filter out correct long trajectories. Our cycle-consistency is less prone to such bias because it measures prompt reconstruction. Therefore we expect our method to maintain higher diversity and accuracy on longer solutions.

**vs. GRPO** — On a case with ambiguous prompts leading to many plausible but incorrect trajectories, GRPO trains on all generated trajectories, reinforcing errors. Our method filters via cycle-consistency, selecting only trajectories that likely correspond to the prompt. Thus we expect significantly higher accuracy on hard prompts.

**vs. CycleConsistency-baseline** — On a case where the generic cycle-consistency model (T5) fails to capture domain-specific nuances (e.g., mathematical notation), our task-specific backward model should outperform. Thus we expect our method to have higher accuracy on math prompts.

**vs. SCSE-zeroshot** — On a case where the synthetic seed data does not cover the distribution of real prompts, the backward model may be poorly calibrated initially. However, with periodic calibration using human-verified pairs (100 examples), the method should recover. We expect a moderate gap between full SCSE and SCSE-zeroshot, narrowing over iterations.

**vs. SCSE w/o cycle-consistency** — On a case where the student initially produces many incorrect trajectories, the ablation would train on all of them, propagating errors. Our full method filters out incorrect ones, so we expect improved accuracy specifically on prompts with low initial success rate.

### What would falsify this idea
If our method does not outperform GRPO and the CycleConsistency-baseline, or if the improvement is uniform across all prompts rather than concentrated on hard prompts (where cycle-consistency matters), the central claim is wrong. Additionally, if the correlation experiment on Gridworld yields Pearson r < 0.6, the core assumption that cycle-consistency scores correlate with correctness is falsified.

## References

1. TREK: Distill to Explore, Reinforce to Refine
2. Black-Box On-Policy Distillation of Large Language Models
3. DAPO: An Open-Source LLM Reinforcement Learning System at Scale
4. VinePPO: Refining Credit Assignment in RL Training of LLMs
5. On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes
6. Large Language Models Are Reasoning Teachers
7. GEAR: A GPU-Centric Experience Replay System for Large Reinforcement Learning Models
8. PowerInfer: Fast Large Language Model Serving with a Consumer-grade GPU
