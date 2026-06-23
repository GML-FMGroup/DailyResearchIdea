# Correcting LLM Evaluation Bias in Tree Search via Online Novelty-Dependent Calibration

## Motivation

Existing methods for planning with LLMs in tree search (e.g., MC-DML, LLM-MCTS) treat LLM evaluations as unbiased estimates, but these evaluations systematically over- or under-estimate the value of novel states due to distribution shift from the pre-training domain. For example, in MC-DML, the LLM's action evaluations are used directly in MCTS without accounting for state novelty, leading to poor exploration of novel branches. The core problem is the lack of task-specific calibration of LLM evaluations, a structural limitation across all methods that rely on zero-shot LLM evaluation.

## Key Insight

The bias in LLM evaluations decreases monotonically with visit count because repeated visits provide additional context that aligns the LLM's assessment with the true state value, enabling a lightweight, learnable correction based solely on local tree statistics.

## Method

(A) **What it is**: BC-MCTS is an extension of MCTS that corrects the LLM evaluation of each state by a learned bias function f(n, d) that depends on the visit count n and depth d. The function is updated incrementally during search using observed Monte Carlo returns.

(B) **How it works**:
```pseudocode
Initialize: value function V(s) = LLM_eval(s), visit count N(s)=0, depth D(s), bias_params θ (e.g., α,β for linear model: bias = α * exp(-β * N(s)))
For each simulation:
  1. Selection: traverse tree using UCB1 with corrected value: Q(s,a) = V(s) + bias(θ, N(s), D(s)) + c * sqrt(ln(N_parent)/N(s))
  2. Expansion at leaf s_new:
     - Obtain LLM evaluation V_LLM(s_new)
     - Initialize N(s_new)=0
  3. Rollout (optional): use LLM to simulate until termination, accumulate reward R
  4. Backpropagation: for each node along path, update:
     - N(s) += 1
     - V(s) = (V(s)*(N(s)-1) + R)/N(s)
     - Update bias parameters θ by gradient descent on loss: L = (V_LLM(s) + bias(θ, N(s), D(s)) - R)^2
End For
```
Hyperparameters: learning rate for θ, exploration constant c, bias model form (e.g., exponential decay with visit count).

(C) **Why this design**: We chose a parametric bias model (exponential decay) over a non-parametric one because it requires only two parameters and thus can be updated efficiently online with few observations, accepting the cost that the bias shape may be misspecified for some tasks (e.g., non-monotonic patterns). We decided to update bias parameters using backpropagated returns rather than relying on a separate validation set because it allows task-specific calibration without additional data collection, but this introduces coupling between value and bias updates that may cause instability if learning rates are not tuned. We use UCB1 for selection with corrected values rather than a Bayesian approach because UCB1 provides explicit exploration-exploitation trade-off with minimal computational overhead; the trade-off is that UCB1 is less robust to non-stationarity than Thompson sampling, but our bias correction aims to reduce non-stationarity by aligning evaluations over time. The bias function depends on both visit count and depth because depth captures the information available to the LLM (deeper states have more context from the path), while visit count captures the amount of local exploration; ignoring depth would miss the systematic bias difference between shallow and deep states. We chose to update bias only for states with at least one observed return to avoid updating on noisy signals; this delays calibration but prevents early overfitting.

(D) **Why it measures what we claim**: The computational quantity `bias(θ, N(s), D(s))` measures state novelty-dependent evaluation error because we assume that the LLM's evaluation error is a function of how many times the state has been visited (which proxies how much additional context the LLM has implicitly seen through the tree history); this assumption fails when the LLM's bias is dominated by other factors (e.g., action-specific biases) that are not captured by visit count, in which case the bias correction will not fully remove the error. The quantity `V_LLM(s) + bias` measures calibrated state value under the assumption that the bias is additive and independent of the true value; this assumption fails when the bias interacts with the state content (e.g., the LLM systematically undervalues states with certain word patterns), leading to residual biases that are not corrected by a novelty-dependent additive term. The parameter update via gradient descent on `(V_LLM(s) + bias - R)^2` measures the alignment between corrected evaluation and observed returns, assuming that Monte Carlo returns are unbiased estimates of the true state value; this assumption fails in stochastic environments or when rollouts are truncated, causing the bias correction to chase noisy targets instead of true bias.

## Contribution

(1) A new method, BC-MCTS, that introduces task-specific bias correction for LLM evaluations in tree search by learning a state-novelty-dependent correction function online. (2) The observation that the bias of LLM evaluations in planning tasks is monotonically related to visit count, enabling a simple parametric model for correction. (3) An empirical demonstration (if we had experiments) that BC-MCTS improves planning efficiency over standard LLM-MCTS and memory-based methods on text-based game benchmarks.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Jericho text-based games | Complex state spaces with LLM bias. |
| Primary metric | Success rate | Measures planning quality directly. |
| Baseline 1 | MCTS+RL | Tests RL fine-tuning on LLM. |
| Baseline 2 | Tree of Thoughts (ToT) | Tests pure LLM tree search. |
| Baseline 3 | RL-based (DQN) | Tests model-free deep RL. |
| Ablation-of-ours | No-bias MCTS | Removes bias correction component. |

### Why this setup validates the claim
The Jericho benchmark provides diverse text-based games where LLM value estimates are systematically biased due to limited context and novelty effects. Success rate directly reflects planning effectiveness. Comparing against MCTS+RL tests whether our online bias correction outperforms offline RL training; ToT tests whether our search with learned bias exceeds pure prompting; RL-based tests whether explicit planning with correction beats model-free RL. The no-bias ablation isolates the impact of the bias term. Together, this setup forms a falsifiable test: if the bias correction is crucial, then BC-MCTS should significantly outperform no-bias MCTS, and also beat or match strong baselines on tasks where bias is prevalent. The metric captures the key outcome of successful planning.

### Expected outcome and causal chain

**vs. MCTS+RL** — On a game where the LLM consistently overestimates early states, MCTS+RL may correct through RL but requires many iterations and may not transfer across tasks. Our method instead adjusts bias locally per state using visit count, adapting quickly to novel states. We expect BC-MCTS to achieve higher success rates on games with sparse rewards, with a noticeable gap in early exploration phases.

**vs. Tree of Thoughts (ToT)** — On a multi-step reasoning task where the LLM's evaluation of intermediate steps is noisy, ToT prunes branches prematurely due to overconfident LLM scores. Our method uses a learned bias to dampen overconfidence for low-visit states, preserving promising branches. We expect BC-MCTS to find more solutions on tasks requiring long chains, with a larger advantage on deeper states.

**vs. RL-based (DQN)** — On a game with large action space, DQN struggles with sample inefficiency and sparse rewards. Our method leverages LLM priors and corrects bias via tree search, achieving better sample efficiency. We expect BC-MCTS to reach higher success rates with fewer environment interactions, especially on tasks with high branching factors.

### What would falsify this idea
If BC-MCTS performs no better than no-bias MCTS (i.e., the bias correction provides no gain), then the central claim that correcting LLM evaluation bias improves planning is false. Also, if the improvement is uniform across all visit counts (rather than concentrated on states with low visit counts where bias is expected), the cause would be misattributed.

## References

1. Monte Carlo Planning with Large Language Model for Text-Based Game Agents
2. Everything of Thoughts: Defying the Law of Penrose Triangle for Thought Generation
3. Ghost in the Minecraft: Generally Capable Agents for Open-World Environments via Large Language Models with Text-based Knowledge and Memory
4. Large Language Models as Commonsense Knowledge for Large-Scale Task Planning
5. Self-Consistency Improves Chain of Thought Reasoning in Language Models
6. Training language models to follow instructions with human feedback
7. Chain of Thought Prompting Elicits Reasoning in Large Language Models
8. Code as Policies: Language Model Programs for Embodied Control
