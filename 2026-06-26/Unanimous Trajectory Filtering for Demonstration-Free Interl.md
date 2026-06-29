# Unanimous Trajectory Filtering for Demonstration-Free Interleaved SFT in Tool-Use Reinforcement Learning

## Motivation

Current interleaved SFT for tool-use RL, as studied in 'Why Multi-Step Tool-Use Reinforcement Learning Collapses,' requires high-quality golden demonstration sequences that are costly to curate. This creates a data bottleneck; dependence on human annotation or external supervision prevents scaling. The structural problem is that while deterministic environments guarantee that correct tool calls lead to consistent outcomes, prior methods do not exploit this to generate certified demonstrations automatically.

## Key Insight

In deterministic tool-use environments, unanimous final outcomes across multiple independent trajectories imply that all intermediate actions are correct, enabling automatic certification of demonstrations without human annotation.

## Method

### Unanimous Trajectory Filtering (UTF)

**Assumption:** In deterministic environments, unanimous final outcomes across K=5 independent rollouts imply all intermediate actions are correct for the purpose of SFT. This is verified via calibration (see block D).

```pseudocode
Algorithm: Interleaved SFT+RL with UTF

Given: environment E (deterministic), initial policy π, reward function R (binary final outcome), number of rollouts per task K=5, SFT data buffer D (capacity C=10000), RL batch size B, SFT batch size S, entropy coefficient η=0.2.

For each training iteration t:
  1. Sample a batch of tasks T_b from training set.
  2. For each task τ in T_b:
     a. Generate K=5 independent rollouts using current policy π. Each rollout: sequence of actions (tool calls and final answer) leading to final outcome o_k.
     b. Compute final outcome o_k ∈ {success, fail} (e.g., correct final answer or correct tool output).
     c. If all o_k are identical (i.e., unanimous outcome), then:
        - For each rollout, if o_k==success, add whole trajectory to D as high-quality demonstration.
        - If o_k==fail, discard.
     d. Else (non-unanimous): discard all rollouts.
  3. Update D with new trajectories (FIFO buffer, capacity C=10000).
  4. Sample B rollouts from current policy (without unanimous filter) for RL update: compute PPO gradient on those rollouts using final outcome reward.
  5. Sample S trajectories from D for SFT update: compute cross-entropy loss on demonstration trajectories.
  6. Interleave updates: use SRFT-style entropy-aware weighting of RL and SFT losses.

Hyperparameters: K=5, C=10000, η=0.2, B=512, S=256.
```

**Why this design:** We chose unanimous filtering (K=5 independent rollouts) over a single rollout with high confidence because deterministic environments guarantee that correct trajectories yield identical outcomes, but confidence thresholds on single trajectories are unreliable due to stochastic policy variance. This accepts the cost of 5× rollouts but ensures certified correctness of demonstrations. We calibrate K=5 by verifying on a held-out set of 100 tasks that the false positive rate of incorrect intermediate actions in unanimous successful trajectories is below 5%. We use outcome-based reward (binary success/fail) rather than process rewards because tool-use environments provide exact final correctness, and process rewards would require manual annotation, defeating the goal of demonstration independence. We interleave SFT and RL updates (like SRFT) rather than two-stage SFT-then-RL because interleaving prevents catastrophic collapse as shown in the target paper; the trade-off is increased computational overhead but stable convergence. We populate D exclusively from unanimous successful trajectories, avoiding external golden data. This may miss some correct but divergent trajectories, but in deterministic settings, a correct policy should consistently succeed; if not, the trajectory is likely suboptimal or the policy is unstable.

**Why it measures what we claim:** Unanimous outcomes measure causal correctness because in deterministic environments, any trajectory achieving the correct final outcome must have executed a minimal set of correct tool calls; if multiple independent rollouts all achieve the same outcome, the intermediate actions are causally necessary for that outcome. This assumption fails when the environment is stochastic or when task symmetry allows redundant but correct actions. In that case, unanimous outcomes reflect empirical consistency rather than causal correctness. However, for deterministic tool-use environments (calculators, code executors), the assumption holds, and unanimous outcomes across K=5 runs, due to low probability of chance agreement, ensure high fidelity. The SFT loss on these trajectories then reinforces correct tool-use behavior.

## Contribution

(1) Unanimous Trajectory Filtering (UTF), a mechanism to generate certified demonstrations from self-generated rollouts in deterministic environments, eliminating the need for human-annotated golden sequences. (2) A design principle that interleaved SFT for tool-use RL can be bootstrapped from autonomous data by exploiting environment determinism, reducing the data curation bottleneck. (3) An empirical demonstration on a tool-use benchmark that UTF+interleaved SFT matches or exceeds performance of human-demonstration-based methods.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Multi-step Arithmetic with Calculator | Deterministic tool-use environment matches assumption. |
| Primary metric | Task success rate | Directly measures correct tool-use completion. |
| Baseline 1 | RL-only | Shows collapse without supervision. |
| Baseline 2 | Two-stage SFT+RL | Tests stability of separate SFT then RL. |
| Baseline 3 | Interleaved SFT+RL (no filter) | Shows effect of unanimous filtering. |
| Ablation-of-ours | UTF with K=1 | Tests necessity of multiple rollouts. |
| Additional ablation | UTF with K=2,3,4 | Tests sensitivity of demonstration quality to K. |

### Why this setup validates the claim

The deterministic dataset ensures that unanimous outcomes across K=5 rollouts certify correct intermediate steps, making UTF's core assumption valid. The RL-only baseline tests whether our method prevents the catastrophic collapse observed in prior work. Two-stage SFT+RL examines if interleaving is truly beneficial over separated phases. Interleaved without filter isolates the impact of filtering out non-unanimous trajectories. The ablation with varying K (1 to 5) tests the necessity of unanimity and the calibration of K=5. Task success rate directly captures the final outcome that rewards are based on, providing a precise measure of tool-use proficiency. Together, these components form a falsifiable test: if UTF works, it should consistently outperform baselines on complex multi-step tasks where collapse is predicted. Additionally, we measure the quality of demonstrations added to D by checking agreement with a held-out set of 100 golden trajectories (manually curated for this verification only); we compute the false positive rate of incorrect intermediate actions in unanimous successful trajectories, which should be below 5% for K=5.

### Expected outcome and causal chain

**vs. RL-only** — On a five-step arithmetic problem requiring successive calculator calls, RL-only suffers from credit assignment over long horizons, leading to erratic action probabilities and ultimate collapse to repetitive tool calls or meaningless outputs. Our method interleaves SFT on unanimous trajectories, which provide clean demonstrations of the correct action sequence, stabilizing the policy. We expect RL-only to achieve near-zero success on such problems, whereas our method maintains >70% success, with a clear gap on tasks requiring ≥3 steps.

**vs. Two-stage SFT+RL** — After initial SFT on static expert demonstrations, the policy performs well on seen tasks, but when RL fine-tuning begins, it can over-optimize for reward and drift into out-of-distribution actions, causing performance degradation. Our interleaved approach continuously refreshes the SFT buffer with current high-quality trajectories, preventing reward hacking. We expect two-stage to show an initial success rate of ~80% that then drops to ~50% after RL, while ours stays stable around 80%.

**vs. Interleaved SFT+RL (no filter)** — Without unanimous filtering, many rollouts contain correct final answers but noisy intermediate steps (e.g., redundant tool calls), which are added to the SFT buffer. These noisy demonstrations dilute the signal and can misguide learning. Our UTF ensures only trajectories where all K=5 rollouts agree are used, guaranteeing high-quality demonstrations. We expect the no-filter variant to achieve moderate success (~60%) but plateau earlier due to noisy data, whereas ours climbs to ~80%.

**vs. UTF with K=1** — With K=1, any single successful rollout is added, likely including many suboptimal trajectories. This leads to noisier SFT data, resulting in lower final success (~50%) and higher variance. UTF with K=5 shows clear improvement, demonstrating the value of unanimity.

### What would falsify this idea
If our method's success rate is not significantly higher than interleaved without filter, or if the improvement is uniform across task difficulty instead of concentrated on complex multi-step tasks where the predicted failure modes occur, then the central claim that unanimous filtering provides certified high-quality demonstrations is wrong. Additionally, if the false positive rate of intermediate action errors in unanimous successful trajectories exceeds 10% on the held-out verification set, the assumption of causal correctness is violated.

## References

1. Why Multi-Step Tool-Use Reinforcement Learning Collapses and How Supervisory Signals Fix It
2. Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning
3. SRFT: A Single-Stage Method with Supervised and Reinforcement Fine-Tuning for Reasoning
4. Boosting MLLM Reasoning with Text-Debiased Hint-GRPO
5. SFT Memorizes, RL Generalizes: A Comparative Study of Foundation Model Post-training
6. Grounding by Trying: LLMs with Reinforcement Learning-Enhanced Retrieval
7. Training Language Models to Self-Correct via Reinforcement Learning
8. Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution
