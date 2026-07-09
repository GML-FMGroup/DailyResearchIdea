# Meta-Adaptive Budget Control for On-Policy Distillation

## Motivation

TurnOPD (Turn-Aware On-Policy Distillation) introduces adaptive rollout-depth and loss-weight budgeting, but its threshold and schedule parameters still require manual tuning per task, limiting robustness across varying task horizons and supervision quality. For example, on long-horizon search tasks, a fixed rollout depth may waste computation on uninformative tail turns or truncate too early—a problem that TurnOPD inherits from Search-R1's manual tuning and does not fully resolve. We need a mechanism that automatically adapts these budgets to task-specific statistics without human intervention.

## Key Insight

The optimal budget parameters (rollout depth, loss weight) are a deterministic function of per-turn probe statistics (e.g., teacher-student KL divergence) that can be learned across a distribution of tasks via meta-reinforcement learning, enabling zero-shot adaptation to unseen tasks at test time.

## Method

### Meta-Adaptive Budget Controller (MABC)

**(A) What it is**: MABC—a lightweight recurrent neural network that observes a sequence of per-turn KL divergence values and per-turn teacher confidence (as a measure of task progress) from probes, and outputs two scalars: a rollout depth fraction λ ∈ [0,1] and a loss-temperature τ ∈ [0.5,2.0]. These are used to truncate rollouts and weight the turn-level KL loss during OPD training. MABC is meta-trained via REINFORCE on a distribution of agent training tasks, using final validation accuracy as reward.

**Load-bearing assumption**: Per-turn KL divergence between student and teacher policies is a sufficient statistic for determining optimal rollout depth and loss weighting. To mitigate potential failures, we augment input with teacher confidence (probability of greedy action per turn) as a progress measure. During meta-training, we compute the Pearson correlation between KL divergence and teacher confidence over a held-out set of turns; if |correlation| < 0.3, we increase the weight of teacher confidence in the controller input by 0.1.

**(B) How it works**:
```pseudocode
Initialize meta-controller φ (2-layer RNN, hidden=128, tanh activation), student policy π, teacher π_teacher
Pre-train φ on a subset of 10% of tasks for 10K inner steps to speed convergence
for meta-iteration = 1 to M:
  Sample task T with max horizon H
  Initialize lists KL_seq = [], Conf_seq = []
  for inner_step = 1 to K:
    // Run probes to collect per-turn KL and teacher confidence
    Run student for full horizon H, record KL per turn: kl[1..H] and teacher confidence conf[1..H]
    Append moving average of kl and conf to KL_seq and Conf_seq (window size W=5)
    // Get budgets from meta-controller
    s = encoder(concatenate(KL_seq, Conf_seq))  # 2-channel RNN state
    (λ, τ) = f_φ(s)            # λ ∈ [0,1] via sigmoid, τ ∈ [0.5,2] via sigmoid scaling
    // Truncate rollout
    depth = floor(λ * H)
    Collect trajectories truncated at depth
    // Compute turn-normalized loss with temperature
    L = Σ_t (log τ * KL_t) / (τ * T)   # progressive turn weighting
    Update π via gradient on L
  // Evaluate student on validation set
  R = accuracy(π, T_val)
  // Update meta-controller via REINFORCE
  ∇_φ log π_φ(λ,τ|s) * (R - baseline)
```
Hyperparameters: W=5, K=100, baseline = running average of recent rewards, correlation threshold=0.3, weight adjustment=0.1.

**(C) Why this design**: We chose a recurrent meta-controller over a feedforward one because probe sequences have variable length across tasks and steps—the RNN's hidden state captures temporal dynamics of supervision quality, enabling adaptation to non-stationary statistics within a single task (e.g., early turns may have high KL that fades later). We output a continuous λ and τ rather than discrete actions to allow finer-grained budgets; the cost is increased variance in REINFORCE due to continuous action spaces, but we mitigate this with a baseline. We use a simple moving average of KL and confidence as input instead of raw per-step values to reduce noise sensitivity—this smooths outlier probes but may mask rapid changes in supervision signal. We train the meta-controller on a distribution of tasks (by varying horizon, reward noise, teacher quality) to encourage generalization, at the expense of requiring a multi-task environment during meta-training.

**(D) Why it measures what we claim**: The rollout depth λ measures the concept of "sufficient horizon" because it truncates rollouts at a point where per-turn KL divergence indicates diminishing returns (i.e., low KL relative to early turns); this assumes KL is a monotonic indicator of supervision quality, which fails when later turns contain critical task steps that are distinct from early turns despite high KL—in that case, λ reflects only similarity to teacher behavior, not task importance. The addition of teacher confidence mitigates this by capturing task progress. The loss temperature τ measures the relative emphasis between early and late turns because it scales the turn-normalized KL loss; this assumes that a single scalar can capture the optimal turn-weighting schedule, which fails when different turns require different emphasis across training phases—then τ reflects an average trade-off that may underweight informative turns. Together, these components map budget decisions to probe statistics, but the mapping is only as valid as the assumption that KL divergence is a sufficient statistic for supervision informativeness; our calibration step verifies this assumption and adjusts input weights when it fails.

## Contribution

(1) A meta-learning framework (MABC) that automatically adapts rollout-depth and loss-weight budgets for on-policy distillation, eliminating manual tuning. (2) The discovery that per-turn KL statistics are sufficient predictors of optimal budget parameters across a task distribution, enabling zero-shot adaptation. (3) A training procedure that interleaves inner-loop OPD updates with outer-loop REINFORCE updates on a multi-task environment.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | ALFWorld | Long-horizon tasks with variable complexity |
| Primary metric | Validation accuracy | Direct measure of task success |
| Baseline 1 | Vanilla OPD | No turn or budget adaptation |
| Baseline 2 | TurnOPD | Turn-aware but static budget |
| Baseline 3 | Random Budget | Random λ, τ to test meta-learning |
| Ablation-of-ours | MABC-FF | Isolates RNN benefit |

### Why this setup validates the claim
This experimental design forms a falsifiable test of the central claim that meta-adaptive budget control improves on-policy distillation. ALFWorld provides diverse long-horizon tasks where KL dynamics vary, challenging static budgets. Comparing against Vanilla OPD tests the value of any adaptation; against TurnOPD tests the benefit of dynamic over static turn-awareness; and against Random Budget tests whether meta-learning provides meaningful guidance. The ablation (MABC-FF) isolates the RNN's temporal modeling. Validation accuracy directly measures task success, the ultimate goal. If MABC outperforms baselines specifically on tasks with non-stationary KL, the claim is supported; uniform gains would instead suggest confounding factors.

### Expected outcome and causal chain

**vs. Vanilla OPD** — On a case where later turns have low KL (e.g., simple manipulation after complex reasoning), vanilla OPD uses full-horizon rollouts, wasting compute and potentially diluting gradient signal from early informative turns. Our method instead truncates rollouts based on live KL, focusing compute on early high-KL turns while preserving performance. We expect MABC to achieve higher validation accuracy with fewer environment steps, especially on tasks where early turns are critical.

**vs. TurnOPD** — On a case where KL dynamics shift mid-task (e.g., sudden increase in exploration need), TurnOPD with its fixed turn-weighting cannot adjust, leading to suboptimal gradient emphasis. Our method outputs λ and τ per step via the RNN, capturing temporal patterns and adapting budgets in real time. Thus, we expect MABC to outperform TurnOPD on tasks with non-stationary supervision, while matching it on stationary tasks.

**vs. Random Budget** — On a case where optimal budgets are task-specific (e.g., long vs. short horizons), random λ and τ yield noisy rollouts and unstable learning. Our meta-trained controller learns from validation reward to output budgets that maximize performance. We expect MABC to consistently outperform Random Budget, with the gap larger on tasks where the optimal budget deviates from random expectation.

### What would falsify this idea
If MABC performs similarly to Vanilla OPD on tasks where early turns dominate, or if its gains are uniform across all task types rather than concentrated on those with non-stationary KL dynamics, then the central claim that adaptive budget control via KL probes is beneficial would be falsified.

## References

1. TurnOPD: Making On-Policy Distillation Turn-Aware for Efficient Long-Horizon Agent Training
2. OWL: Optimized Workforce Learning for General Multi-Agent Assistance in Real-World Task Automation
3. AgentMath: Empowering Mathematical Reasoning for Large Language Models via Tool-Augmented Agent
4. MALT: Improving Reasoning with Multi-Agent LLM Training
5. Magentic-One: A Generalist Multi-Agent System for Solving Complex Tasks
6. Coevolving with the Other You: Fine-Tuning LLM with Sequential Cooperative Multi-Agent Reinforcement Learning
7. Llama 2: Open Foundation and Fine-Tuned Chat Models
8. A social path to human-like artificial intelligence
