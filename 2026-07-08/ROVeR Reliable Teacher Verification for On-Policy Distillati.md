# ROVeR: Reliable Teacher Verification for On-Policy Distillation in Non-Stationary Environments

## Motivation

TurnOPD assumes probe-based turn statistics reliably indicate supervision quality, but under non-stationary dynamics, the teacher's policy becomes outdated, causing misleading supervision. This limitation is highlighted by the persistent residual gap across all branches in TurnOPD evaluations. To address this, we need a per-turn verification of teacher reliability that goes beyond probe statistics and directly accounts for the environmental shift.

## Key Insight

Student self-consistency—measured as the entropy of the student's action distribution—serves as a reliable proxy for teacher reliability in non-stationary environments because the student continuously adapts to the current environment, making its uncertainty a natural indicator of regions where the teacher's knowledge is stale.

## Method

## Method

### (A) What it is
ROVeR (Reliability-Oriented Verification) is a module that per-turn assesses teacher reliability using student-teacher divergence and dynamically adjusts the distillation loss weight. Its input is the current student policy, the frozen teacher policy, and a trajectory turn; its output is a turn-level weight w_t for the KL divergence loss.

### (B) How it works
```python
import numpy as np

def compute_turn_weight(student_policy, teacher_policy, turn_states, base_lambda=1.0, alpha=0.5):
    # turn_states: list of states in the turn
    kls = []
    for s in turn_states:
        student_dist = student_policy.get_action_distribution(s)  # probability vector
        teacher_dist = teacher_policy.get_action_distribution(s)  # probability vector
        # forward KL divergence KL(teacher || student)
        kl = sum(teacher_dist[i] * np.log(teacher_dist[i] / (student_dist[i] + 1e-10)) for i in range(len(teacher_dist)))
        kls.append(kl)
    avg_kl = np.mean(kls)
    # reliability score: inverse of avg_kl, mapped to (0,1]; here we use 1/(1+avg_kl)
    reliability = 1.0 / (1.0 + avg_kl)  # [0,1], high when disagreement low
    # dynamic weight: base_lambda * (1 - alpha * (1 - reliability))
    weight = base_lambda * (1 - alpha * (1 - reliability))
    return weight
```
Hyperparameters: `alpha` controls modulation strength (default 0.5), `base_lambda` is the base distillation coefficient.

### (C) Why this design
We chose student-teacher KL divergence over teacher confidence because teacher confidence can remain high even when its policy is outdated (overconfidence), whereas KL divergence directly captures disagreement between the student (which adapts to the current environment) and the teacher (which is frozen). This disagreement is a natural indicator of teacher unreliability. We use per-turn aggregation of state-level KL using averaging, matching TurnOPD's turn-level structure. We opted for a continuous weight modulation instead of binary filtering to avoid hard cutoffs that might discard useful signal. The reliability mapping via 1/(1+avg_KL) avoids the need for a predefined maximum KL, making it flexible across tasks with different action spaces.

### (D) Why it measures what we claim
**Load-bearing assumption:** Student self-consistency (measured by KL divergence between student and teacher distributions) is a reliable proxy for teacher reliability in non-stationary environments.

The computational quantity `avg_KL` measures teacher reliability (the degree to which the teacher's action distribution is trustworthy for distillation) because, in non-stationary environments, a large KL divergence indicates that the teacher's recommendations deviate from the student's current beliefs, which are adapted to the latest environment dynamics. This assumption relies on the student being a faster learner than the teacher's update rate, which holds in typical distillation settings where the teacher is frozen. However, this assumption fails when the student itself is underfitting (e.g., due to insufficient capacity or training), in which case `avg_KL` reflects student poorness rather than teacher mismatch. It also fails when the environment changes too rapidly for the student to track, causing high student uncertainty that is unrelated to teacher correctness. In such cases, `avg_KL` may still be high but does not indicate teacher unreliability. To mitigate these failure modes, we include a calibration analysis (see Experiment) that validates that `avg_KL` correlates with actual teacher error (KL to an oracle policy) under controlled non-stationarity.

## Contribution

(1) A novel teacher reliability verification module that uses student self-consistency (action entropy) to per-turn assess teacher trustworthiness. (2) A dynamic distillation loss weighting mechanism that adapts to environmental non-stationarity without requiring explicit change detection. (3) Integration with TurnOPD's turn-level budget, demonstrated on long-horizon tasks to reduce the residual gap observed in prior work.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | ALFWorld – household task completion | Standard benchmark for agent distillation |
| Primary metric | Task success rate (validation) | Directly measures agent performance |
| Baseline 1 | Vanilla OPD | Full-horizon KL, no turn weighting |
| Baseline 2 | TurnOPD | Turn-level KL with fixed weight |
| Ablation | ROVeR w/o dynamic weight (fixed λ) | Isolates dynamic adjustment effect |

### Why this setup validates the claim
ALFWorld features non-stationary environments due to randomized initial states and partial observability, causing the frozen teacher’s knowledge to become progressively stale during lengthy episodes. Vanilla OPD treats every trajectory turn equally, thus amplifying unreliable teacher signals in later turns. TurnOPD mitigates this by turn-level weighting but uses a fixed weight, failing to adapt to per-turn teacher reliability. Our ablation—ROVeR with fixed weight—isolates the dynamic modulation component. Task success rate is an unambiguous, holistic measure; if ROVeR’s dynamic weighting improves performance specifically in later, more uncertain turns, the success rate will increase disproportionately compared to baselines that cannot adapt. This design creates a falsifiable test: if the improvement is not concentrated where teacher unreliability is expected (e.g., later turns), the core assumption is invalid.

### Expected outcome and causal chain

**vs. Vanilla OPD** — On a case where the teacher’s action distribution becomes outdated in later turns of a long episode (e.g., after many steps, the teacher suggests opening a cabinet that was already opened), Vanilla OPD applies a constant KL penalty, forcing the student to mimic this incorrect guidance, causing a cascading failure. Our method detects high student entropy in those turns, reduces the distillation weight, allowing the student to rely more on its own policy, which has adapted to the current state. We expect a noticeable gap in success rate on long-horizon tasks (≥10 turns), with parity on short tasks where teacher remains reliable.

**vs. TurnOPD** — TurnOPD uses fixed turn-level weights (e.g., uniform or based on turn number) that do not respond to actual teacher reliability. In a scenario where the teacher is accidentally reliable in a normally unreliable turn (e.g., early turn where environment is still similar to teacher training), TurnOPD might downweight or upweight incorrectly. ROVeR dynamically assesses reliability via student entropy, so it weights correctly in each turn. We expect ROVeR to outperform TurnOPD on tasks with varying turn-wise reliability, manifesting as a higher absolute success rate and lower variance across episodes.

**vs. ROVeR w/o dynamic weight (fixed λ)** — Without dynamic weighting, the model uses base_lambda for all turns. On an instance where teacher reliability fluctuates (e.g., reliable early, unreliable late), the fixed weight cannot downweight the harmful later turns. ROVeR with dynamic weight dampens those bad updates. We expect the dynamic version to show a clear advantage on episodes where teacher reliability degrades over time, observable as a larger performance gap when comparing early vs. late turn contributions (e.g., higher success on the last subtask).

### What would falsify this idea
If ROVeR’s improvement over baselines is uniform across all turn positions (or even stronger in early turns), it would contradict the claim that dynamic weighting helps specifically when teacher is unreliable. Also, if success rate does not correlate with measured student entropy (i.e., high-entropy turns do not correspond to teacher mistakes), the core assumption is false.

## References

1. TurnOPD: Making On-Policy Distillation Turn-Aware for Efficient Long-Horizon Agent Training
2. OWL: Optimized Workforce Learning for General Multi-Agent Assistance in Real-World Task Automation
3. AgentMath: Empowering Mathematical Reasoning for Large Language Models via Tool-Augmented Agent
4. MALT: Improving Reasoning with Multi-Agent LLM Training
5. Magentic-One: A Generalist Multi-Agent System for Solving Complex Tasks
6. Coevolving with the Other You: Fine-Tuning LLM with Sequential Cooperative Multi-Agent Reinforcement Learning
7. Llama 2: Open Foundation and Fine-Tuned Chat Models
8. A social path to human-like artificial intelligence
