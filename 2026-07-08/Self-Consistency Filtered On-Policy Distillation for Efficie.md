# Self-Consistency Filtered On-Policy Distillation for Efficient Agent Training

## Motivation

AgentMath assumes automatically generated trajectories are reliable, but teacher models can produce noisy actions in long-horizon tasks. TurnOPD improves efficiency via adaptive rollout and loss budgeting but still uses all teacher supervision without quality control, allowing noisy signals to accumulate and degrade learning. We propose filtering teacher supervision using self-consistency, a method validated in reasoning (STaR), to discard unreliable targets before distillation.

## Key Insight

The agreement among multiple independent teacher samples (self-consistency) acts as a reliable proxy for supervision quality because high agreement indicates confident correct decisions, while low agreement signals uncertainty or error—filtering the latter removes harmful noise without requiring ground-truth labels.

## Method

**Self-Consistency Filtered On-Policy Distillation (SCF-OPD)**

**(A) What it is:** SCF-OPD augments TurnOPD with a state-level filter that uses the teacher's own cross-sample agreement to decide whether to use its action distribution as a target for the student. The filter operates under the load-bearing assumption that high self-consistency among teacher samples reliably indicates that the teacher's action is correct and beneficial for student learning. Input: student policy π_s, teacher policy π_t, rollout depth budget B, loss budget L, self-consistency samples K=5, agreement threshold τ (calibrated on a held-out set of 1000 states to maximize F1 score for filtering correct actions). Output: updated π_s.

**(B) How it works (pseudocode):**
```pseudocode
Algorithm: SCF-OPD training iteration
Input: π_s, π_t, B, L, K, τ
For each episode in batch:
    # Phase 1: Adaptive rollout (TurnOPD)
    Run probes to estimate turn-level statistics.
    for turn t = 1..B (adaptive):
        a_s ~ π_s(·|s_t)
        s_{t+1} = env.step(s_t, a_s)
        store state s_t
    # Phase 2: Self-consistency filter
    For each stored state s:
        samples = [π_t(·|s).sample() for _ in range(K)]  # K=5 teacher samples
        majority_action = mode(samples)
        agreement = count(majority_action) / K
        if agreement >= τ:  # τ calibrated on held-out set
            p_target = one-hot(majority_action)  # or smoothed empirical distribution
            loss_state = KL(p_target || π_s(·|s))
        else:
            loss_state = 0  # skip
    # Phase 3: TurnOPD loss budgeting
    Apply turn-level weighting from TurnOPD to aggregate loss_state into total loss.
    Update π_s via gradient descent.
```
Hyperparameters: K=5, τ calibrated on held-out set.

**(C) Why this design:** We chose self-consistency over a learned reward model to avoid training an additional model and to leverage the teacher's own sampling distribution for free. A threshold-based hard filter (rather than soft weighting) provides a clean binary signal, avoiding the difficult problem of calibrating continuous weights. We filter at the state level (rather than trajectory level) because in long-horizon tasks, only a fraction of states may be noisy; partial filtering preserves useful supervision. The trade-off is that generating K samples per state increases computation, but this is done offline before the gradient update and does not affect rollout cost. We also accept the risk that a confidently wrong teacher (systematic bias) will pass the filter, but we mitigate by combining with TurnOPD's adaptive budgeting which already reduces reliance on weak turns. We use a majority vote with threshold rather than entropy because entropy does not capture the magnitude of agreement on a single action; a teacher could have low entropy but still be wrong, whereas majority agreement directly measures consistency of the chosen action.

**(D) Why it measures what we claim:** The quantity `agreement = count(majority_action)/K` measures supervision reliability because it operationalizes the assumption that a teacher that consistently produces the same action across independent samples is more likely to be correct (the self-consistency assumption). This assumption fails when the teacher has a systematic bias (e.g., always picks the same wrong action), in which case high agreement reflects spurious consistency rather than correctness. In that scenario, the metric reflects the teacher's confidence but not accuracy. To bound this risk, we calibrate τ using a held-out set of 1000 states from the teacher's training distribution, where we compute the agreement-correctness correlation (Pearson r) and choose τ that maximizes F1 for filtering correct actions (where correctness is measured by a small amount of ground-truth labels from an oracle, e.g., human demonstrations or sparse reward). Additionally, we track the agreement distribution during training to detect shifts that might indicate systematic bias.

## Contribution

(1) A self-consistency filter for on-policy distillation that uses teacher cross-sample agreement to discard unreliable supervision before student training. (2) Integration of this filter with TurnOPD's adaptive rollout and loss budgeting, creating a combined algorithm (SCF-OPD) that maintains efficiency while improving supervision quality. (3) The design principle that state-level filtering based on empirical agreement is an effective proxy for reliability in the absence of ground truth, applicable to other distillation settings.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|-----------------------|
| Dataset 1 | ALFWorld | Long-horizon interactive reasoning tasks |
| Dataset 2 | Meta-World (10 tasks) | Diverse robotic manipulation tasks |
| Dataset 3 | DMControl (Humanoid, Walker) | Continuous control with long horizons |
| Primary metric | Success rate | Measures final task completion accuracy |
| Secondary metric | Agreement-correctness correlation (Pearson r) | Quantifies self-consistency assumption validity |
| Baseline 1 | Vanilla OPD | Full rollout without turn awareness |
| Baseline 2 | TurnOPD | Turn-aware but no consistency filter |
| Baseline 3 | Entropy-based filtering (state-level, threshold on teacher entropy) | Alternative state-level filter |
| Baseline 4 | Learned confidence filtering (small MLP trained on held-out states to predict correctness) | Alternative learned filter |
| Ablation-of-ours | SCF-OPD w/o filter (i.e., TurnOPD) | Isolates effect of consistency filter |
| Synthetic experiment | Teacher with controlled noise (add label noise to states) | Directly measure agreement vs. correctness |

### Why this setup validates the claim

ALFWorld provides long-horizon tasks where teacher actions may be unreliable in certain states. Comparing against TurnOPD isolates the contribution of the self-consistency filter; vanilla OPD demonstrates the value of on-policy distillation itself. Success rate directly measures task completion, which is the ultimate goal. The ablation (SCF-OPD without filter) tests whether the filter alone drives improvement. The inclusion of Meta-World and DMControl ensures that the benefits generalize across diverse long-horizon domains (robotic manipulation and continuous control). The alternative baselines (entropy-based and learned confidence) directly address the reviewer's novelty concern by showing that self-consistency outperforms other state-level filtering methods. The synthetic experiment with controlled teacher noise provides a direct test of the load-bearing assumption: we can vary the noise level and measure the correlation between agreement and correctness, which should be high for the method to hold. Together, these components create a falsifiable test: if the filter is beneficial, SCF-OPD should outperform TurnOPD primarily on subtasks with high teacher disagreement, while parity is expected where teacher is already consistent.

### Expected outcome and causal chain

**vs. Vanilla OPD** — On a case where early turns are critical and the teacher is weak, vanilla OPD wastes the entire rollout budget on noisy supervision, leading to poor student performance. Our method adaptively budgets turns and filters unreliable states using self-consistency, so we expect a noticeable gap in success rate (e.g., 10-15% absolute improvement across all datasets).

**vs. TurnOPD** — On a case where the teacher is confidently wrong on some states (e.g., systematic bias due to spurious correlations), TurnOPD still uses these states because it only budgets at the turn level. Our state-level filter skips states with low self-consistency, so we expect SCF-OPD to outperform TurnOPD on tasks where such biased states occur, with a gap of 5-10% on those subsets and parity elsewhere.

**vs. Entropy-based filtering** — Entropy measures uncertainty but not agreement on a specific action; a teacher with low entropy may still be wrong if it is consistently wrong. Self-consistency directly measures agreement on the chosen action, so we expect SCF-OPD to outperform entropy-based filtering, especially in settings with systematic bias.

**vs. Learned confidence filtering** — Learned confidence requires additional training data and may overfit to the held-out set. Self-consistency is free and leverages the teacher's own distribution, so SCF-OPD should achieve comparable or better performance with lower computational overhead.

**vs. Synthetic noise control** — In the synthetic experiment, we expect the agreement-correctness correlation to be above 0.8 when teacher noise is low, validating the assumption. As noise increases, agreement drops and the filter correctly discards more states, maintaining student performance.

### What would falsify this idea

If SCF-OPD's improvement over TurnOPD is uniform across all episodes rather than concentrated on episodes with high teacher disagreement (low self-consistency), then the central claim that self-consistency filtering is the cause would be falsified. Additionally, if the agreement-correctness correlation in the synthetic experiment is below 0.5 under low noise, the load-bearing assumption is invalid.

## References

1. TurnOPD: Making On-Policy Distillation Turn-Aware for Efficient Long-Horizon Agent Training
2. OWL: Optimized Workforce Learning for General Multi-Agent Assistance in Real-World Task Automation
3. AgentMath: Empowering Mathematical Reasoning for Large Language Models via Tool-Augmented Agent
4. MALT: Improving Reasoning with Multi-Agent LLM Training
5. Magentic-One: A Generalist Multi-Agent System for Solving Complex Tasks
6. Coevolving with the Other You: Fine-Tuning LLM with Sequential Cooperative Multi-Agent Reinforcement Learning
7. Llama 2: Open Foundation and Fine-Tuned Chat Models
8. A social path to human-like artificial intelligence
