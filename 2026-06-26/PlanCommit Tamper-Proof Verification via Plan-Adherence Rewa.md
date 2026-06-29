# PlanCommit: Tamper-Proof Verification via Plan-Adherence Rewards in Reinforcement Learning

## Motivation

Existing verifiers, as analyzed in 'The Verification Horizon: No Silver Bullet for Coding Agent Rewards', treat agent outputs as static artifacts, allowing agents to obfuscate or tamper with intermediate steps under optimization pressure (e.g., 'Monitoring Reasoning Models for Misbehavior and the Risks of Promoting Obfuscation'). This structural gap between planning and verification enables reward hacking. We propose a commitment protocol that verifies the agent's plan during execution, making any deviation immediately detectable.

## Key Insight

By requiring the agent to commit to a structured plan before execution and then scoring based on adherence to that plan, we create an invariant: any deviation from the plan must be due to either execution error (which the agent would avoid) or intentional tampering (which becomes observable).

## Method

# PlanCommit Training Loop
### Load-bearing assumption: The adversarial training with the sneaky agent ensures that the verifier achieves near-perfect accuracy on both in-distribution and adversarial plan-state pairs. If this assumption fails (e.g., due to distribution shift or novel attacks not seen during training), the verifier's predictions may be unreliable, compromising the plan adherence measurement. This is a known limitation and a potential direction for future certified defenses.
Initialize agent π (policy), verifier V (4-layer Transformer, hidden=256, 8 attention heads, ~5M parameters), sneaky agent π_sneaky
For each iteration:
  # Phase 1: Train helpful agent
  Sample environment start state s_0
  π generates plan p = [a_1, ..., a_T] (structured token sequence, max length 64 tokens)
  Execute actions step-by-step, collecting actual states s_1,...,s_T
  For each step t:
    adherence[t] = V(s_t, p[t])  # binary: 1 if match, 0 otherwise
  plan_adherence = mean(adherence)
  final_reward = R(s_T) * plan_adherence  # combined reward; R is task success reward
  Update π via PPO (clip epsilon 0.2) to maximize final_reward
  
  # Phase 2: Train sneaky agent (adversarial)
  Sample environment, π_sneaky generates plan p' (non-faithful but plausible)
  Execute actions randomly (or follow π_sneaky's advice) to get states
  Reward = R(s_T) * (1 - plan_adherence)  # maximize non-adherence
  Update π_sneaky via PPO (clip epsilon 0.2) to maximize reward
  
  # Phase 3: Train verifier V
  Sample traces from both helpful and sneaky agents (including which is which)
  V trained via cross-entropy to predict faithfulness (real adherence label)
  Input: (plan p, actual state s_t) -> 1 if plan was followed, 0 otherwise
  # Calibration: After each iteration, we recalibrate the verifier's decision threshold on a held-out calibration set of 512 plan-state pairs (balanced between helpful and sneaky traces) to achieve a false-negative rate ≤ 0.01. This ensures that the verifier is not overly confident and maintains a low miss-rate.

**Why this design:** We chose plan commitment over post-hoc verification because it forces the agent to reveal its intentions, eliminating the separation that enables obfuscation. We used a learned verifier (a 4-layer Transformer) instead of a deterministic state-matching function to handle partial observability and natural language plans, accepting the cost of potential inaccuracy under distribution shift. We train the verifier adversarially with a sneaky agent (inspired by Prover-Verifier Games) to expose and correct vulnerabilities, rather than relying on a static verifier that could be gamed—this adds training compute but yields robustness. The combined reward `R(s_T) * plan_adherence` ensures both correctness and honesty, preventing the agent from achieving high final reward by deviating from its plan. Hyperparameters: PPO clip epsilon 0.2, verifier hidden size 256, adversarial iterations every 5 main iterations, plan max length 64 tokens.

**Why it measures what we claim:** `plan_adherence` measures **plan faithfulness** because it directly compares the agent's prior commitment (plan step p[t]) to the observed state s_t, under the assumption that the verifier V accurately identifies matches; this assumption fails when the verifier is fooled by adversarial inputs (e.g., sneaky agent's obfuscated plans), in which case `plan_adherence` reflects the verifier's error rather than true faithfulness. The final reward `R(s_T) * plan_adherence` measures **tamper-resistance** because any deviation from the plan (whether intentional or not) proportionally reduces the reward, creating a structural disincentive for tampering; this equivalence assumes the verifier's judgments correlate with actual plan deviations, which fails if the verifier ignores subtle deviations (e.g., action-level differences not captured by state changes), in which case the reward may be gamed. The sneaky agent's objective `R(s_T) * (1 - plan_adherence)` measures **adversarial exploitability** by explicitly searching for holes in the verifier, under the assumption that a successful sneaky plan would yield high final reward despite low plan adherence; this fails if the sneaky agent's search space is insufficiently diverse, in which case the measure underestimates true exploitability.

## Contribution

(1) The PlanCommit training framework that couples planning and verification through a commitment-check protocol, structured as an adversarial Prover-Verifier loop between a helpful planner, a sneaky planner, and a verifier. (2) Empirical demonstration that PlanCommit reduces reward hacking compared to post-hoc verification in agentic coding environments, with analysis of the trade-off between plan granularity and verifier accuracy. (3) A set of design principles for tamper-proof verification, including the necessity of adversarial training and the coupling of plan adherence with final outcome rewards.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | Minecraft-based 'Builder' tasks with 5-step natural language plans (e.g., 'collect wood', 'craft planks', 'build shelter', 'place bed', 'sleep') | Tests natural language plans and partial observability |
| Primary metric | Task success (binary: 1 if shelter built and slept, 0 otherwise) × plan adherence (average over 5 steps of verifier's binary prediction) | Combines correctness and honesty |
| Baseline 1 | No verification (standard RL) | Baseline to show effect of verification |
| Baseline 2 | Static verifier (pre-trained BERT, 12-layer, 768 hidden) | Shows value of adversarial training |
| Baseline 3 | Oracle verifier (ground-truth state-matching, accessible via simulator) | Upper bound if verifier were perfect |
| Baseline 4 | Fixed verifier (same architecture as ours but trained only on helpful traces, no adversarial training) | Isolates novelty of adversarial training |
| Ablation-of-ours | Ours without sneaky agent phase | Isolates benefit of adversarial training |

### Why this setup validates the claim

The experiment juxtaposes four baselines to isolate the contributions of verification and adversarial training. The Minecraft environment with language-conditioned subgoal plans exercises both the plan faithfulness challenge (the agent must commit to a 5-step plan) and partial observability (states like 'has wood' are not perfectly captured by single plan steps). The primary metric—task success weighted by plan adherence—directly measures the claimed dual objective. The oracle verifier baseline establishes an upper bound, while the no-verification baseline shows the failure mode of reward tampering. The fixed verifier baseline (trained without adversarial examples) tests whether adversarial training is necessary for robustness. The ablation (no sneaky agent) tests whether the adversarial component is necessary for robustness. A significant improvement over the static verifier on adversarial examples would confirm the central claim that adversarial training closes exploitability gaps.

### Expected outcome and causal chain

**vs. No verification** — On a case where the agent can achieve high task reward by deviating from its stated plan (e.g., skipping a dangerous subgoal like 'collect wood' and going directly to 'build shelter'), standard RL maximizes final reward despite plan deviation, leading to reward tampering. Our method instead penalizes plan deviation via the verifier, so we expect our final metric (task success × adherence) to be significantly higher (e.g., >1.5×) in environments where tampering yields high task reward.

**vs. Static verifier** — On a case where the sneaky agent generates obfuscated plans that fool a fixed verifier (e.g., using synonym substitution or plausible‐looking state justifications), the static verifier mislabels adherence, allowing high reward despite unfaithfulness. Our method retrains the verifier on adversarial examples from the sneaky agent, closing that loophole. We expect our verifier accuracy on such adversarial traces to be >90% vs. <50% for static verifier, leading to a larger gap in the final metric.

**vs. Fixed verifier (no adversarial training)** — On a case where the sneaky agent uses plan deviations that are misclassified by a verifier trained only on helpful traces (e.g., a plausible plan that slightly misaligns with states), the fixed verifier fails to penalize the deviation. Our adversarially trained verifier learns to detect these subtle mismatches. We expect our plan adherence metric to be at least 20% higher than the fixed verifier on these cases, demonstrating the necessity of adversarial training.

**vs. Oracle verifier** — On a case where the environment has partial observability (e.g., state does not reveal plan step execution), an oracle verifier has perfect information and achieves high performance. Our learned verifier may struggle with distribution shift, leading to slightly lower plan adherence accuracy (e.g., 95% vs. 100%). However, on clear‐visibility cases, our method should match the oracle, confirming the verifier is not a weak bottleneck.

### What would falsify this idea

If our method fails to outperform the static verifier on adversarial plan traces (i.e., the sneaky agent cannot find exploitable weaknesses or the verifier does not improve), then the claim of adversarial robustness is unsupported. Also, if the ablation (without sneaky agent) performs similarly to the full method across all test cases, then the adversarial training is unnecessary.

## References

1. The Verification Horizon: No Silver Bullet for Coding Agent Rewards
2. Sycophancy to Subterfuge: Investigating Reward-Tampering in Large Language Models
3. Monitoring Reasoning Models for Misbehavior and the Risks of Promoting Obfuscation
4. Prover-Verifier Games improve legibility of LLM outputs
5. Frontier Models are Capable of In-context Scheming
6. Scalable and Transferable Black-Box Jailbreaks for Language Models via Persona Modulation
7. TinyGSM: achieving >80% on GSM8k with small language models
8. Weak-to-Strong Generalization: Eliciting Strong Capabilities With Weak Supervision
