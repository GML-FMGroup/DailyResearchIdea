# Trust-Region Constrained Log-Ratio Reward Distillation for Reliable Weak-to-Strong Generalization

## Motivation

Direct on-policy distillation using log-ratio rewards (Direct-OPD) achieves weak-to-strong generalization but lacks theoretical guarantees: the log-ratio reward's validity degrades under student distribution shifts and teacher quality variance, leading to potentially misleading training signals. This limitation is structural because the reward depends on the teacher's policy divergence, which is not controlled during student training. We ground our approach in the observation that the teacher's pre-RL policy provides a natural anchor; constraining the student's divergence from this anchor yields a monotonic improvement bound, ensuring reward reliability.

## Key Insight

Constraining the student's policy within a trust region around the teacher's pre-RL policy bounds the bias of the log-ratio reward by the teacher's policy shift, making the reward provably indicative of true improvement.

## Method

### Trust-Region Constrained Log-Ratio Reward Distillation (TRCLR)

**Input:** Teacher policies π_pre, π_post (frozen); student policy π_stu (initialized to π_pre); trust region threshold δ=0.1; learning rate η=3e-5; number of iterations T=10000; PPO clip ε=0.2; initial penalty coefficient β₀=0.01; PID controller for β (Kp=1, Ki=0.1, Kd=0.01); reward validation threshold τ=0.2; validation interval C=500 steps; calibration set size N_cal=512.

**Pseudocode:**

```
For iteration t = 1 to T:
  1. Sample on-policy rollouts τ ~ π_stu (student generates sequences, batch_size=512, max_length=1024).
  2. For each token in each rollout, compute log-ratio reward:
     r(s,a) = log(π_post(a|s) / π_pre(a|s))
  3. Compute advantage estimates A_t using GAE (λ=0.95) with rewards r.
  4. Update student policy by maximizing clipped surrogate objective:
     L = E[ min(ratio * A, clip(ratio, 1-ε, 1+ε) * A) ]
     where ratio = π_stu(a|s) / π_old(a|s)
  5. Enforce trust region constraint:
     Compute D_KL(π_pre || π_stu) via Monte Carlo estimate using states from rollouts.
     If D_KL > δ, then:
       L_total = L - β * (D_KL - δ)
     else:
       L_total = L
     Update β via PID: β = β + Kp*(D_KL-δ) + Ki*Σ(D_KL-δ) + Kd*(D_KL_prev - D_KL)
     Clip β to [0, 10].
  6. Update student policy via Adam (β1=0.9, β2=0.999, weight_decay=0.01).
  7. [Reward Validation] Every C steps:
     - Sample N_cal trajectories from π_stu.
     - Compute Pearson correlation ρ between log-ratio reward r and true environment reward (available for calibration only).
     - If ρ < τ, set β = 10 (aggressive constraint) and reduce η by factor 0.5.
Output: π_stu
```

(C) **Why this design:** [unchanged from original]
We chose a KL constraint relative to the teacher's pre-RL policy (π_pre) rather than the student's previous policy (as in standard PPO) because π_pre is the anchor that guarantees reward validity: the log-ratio reward is defined relative to π_pre, so constraining divergence from it directly bounds the distribution shift that could invalidate the reward. We opted for a soft penalty with adaptive β over a hard constraint (e.g., projection) because it integrates naturally with gradient-based optimization and allows tuning the trade-off between reward maximization and trust region satisfaction; the cost is that the constraint may be violated transiently, requiring careful β adaptation. We used PPO-style clipping rather than natural gradient (TRPO) because per-step KL control is computationally simpler and empirically effective, accepting that clipping may not guarantee a strict monotonic improvement at every step but the trust region constraint provides an overall bound. Finally, we compute D_KL on the student's on-policy states rather than the teacher's, because the constraint is meant to limit the student's divergence from π_pre on the student's own distribution, which directly affects reward bias. The optional reward validation step ensures that if the log-ratio reward becomes uncorrelated with true reward (e.g., due to unmodeled distribution shift), the method falls back to a conservative regime (stronger constraint, lower learning rate), preserving reliability.

(D) **Why it measures what we claim:**  
- `r(s,a) = log(π_post(a|s) / π_pre(a|s))` measures the teacher's relative improvement at state s. **Assumption:** This reward is indicative of true improvement when the state distribution is close to that of π_pre. **Failure mode:** When the student visits states where π_pre assigns low probability (large distribution shift), the log-ratio may be dominated by sampling error or be irrelevant, leading to misleading signals.  
- `D_KL(π_pre || π_stu)` measures the student's distribution shift from π_pre. **Assumption:** A small D_KL implies that the reward bias is bounded (monotonic improvement bound). **Failure mode:** If the teacher's pre-to-post shift (π_pre vs π_post) is itself non-informative (e.g., spurious correlations), even a small D_KL does not guarantee reward validity; the bound becomes vacuous.  
- The penalty `β * max(0, D_KL - δ)` operationalizes the constraint. **Assumption:** Adaptive β via PID maintains D_KL near δ. **Failure mode:** Under non-stationary dynamics (e.g., rapidly changing student distribution), PID may overshoot or oscillate, causing constraint violations. The reward validation step mitigates this by detecting reward corruption and applying conservative adjustments.

## Contribution

(1) A trust-regularized distillation algorithm (TRCLR) that augments log-ratio reward training with a KL constraint to the teacher's pre-RL policy, ensuring theoretical validity. (2) Derivation of a monotonic improvement bound for the student's true reward under the trust region, showing that updates satisfying the constraint guarantee non-degradation. (3) Empirical design insight that the teacher's pre-RL policy serves as an effective anchor for reward reliability, bridging weak-to-strong generalization with theoretical safety.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | MATH | Standard reasoning benchmark for distribution shift analysis |
| Dataset 2 | GSM8K | Additional benchmark to test generalizability |
| Primary metric | Accuracy (pass@1) | Directly measures task success |
| Baseline 1 | Direct Distillation | Baseline without reward signals |
| Baseline 2 | Student RL from scratch | Upper bound with full reward (PPO with true reward) |
| Baseline 3 | SeqKD | Common distillation baseline |
| Ablation | TRCLR w/o trust region | Isolates effect of constraint |
| Ablation 2 | TRCLR with constraint w.r.t. student's previous policy (π_old) | Tests novelty of using π_pre as anchor |
| Resource estimate | 100 GPU hours on 4x A100 (100K trajectories, 7B model) | Feasible for typical academic lab |

### Why this setup validates the claim
This experimental design tests the central claim that the trust-region constraint in TRCLR prevents reward bias from the log-ratio reward, enabling effective distillation without costly RL. The dataset MATH provides a challenging reasoning benchmark where distribution shift matters; GSM8K adds generalizability. Comparing against Direct Distillation (no reward) and Student RL (full reward) isolates the benefit of reward guidance and the cost of exploration. SeqKD represents a standard distillation baseline. The ablation without trust region tests whether the constraint is responsible for performance gains; the ablation with constraint on π_old tests whether the anchor on π_pre is crucial. The accuracy metric directly measures the predicted improvement. The reward validation step (correlation check) ensures that if the log-ratio reward becomes unreliable, the method adapts, further testing robustness.

### Expected outcome and causal chain

**vs. Direct Distillation** — On a problem requiring multi-step reasoning beyond the teacher's pre-RL capability, Direct Distillation copies the teacher's weak behavior and fails. Our method instead uses the log-ratio reward to reinforce steps that improved after teacher RL, so we expect a noticeable accuracy gap (e.g., +10-20%) on hard problems, but parity on simple recall problems.

**vs. Student RL from scratch** — On a problem with sparse verifiable rewards (e.g., long chains), student RL struggles due to exploration difficulty. Our method leverages the teacher's relative reward to provide dense guidance, so we expect faster convergence and higher final accuracy (e.g., +5-10%) on such problems, with comparable performance on problems where rewards are dense.

**vs. SeqKD** — On a problem where the teacher's post-RL policy is noisy, SeqKD inherits the noise. Our method uses the log-ratio reward (relative improvement) which filters out absolute noise, so we expect higher robustness, e.g., smaller variance and better worst-case accuracy.

**Ablation: w/o trust region** — Without the KL constraint, the student may drift far from π_pre, causing reward bias and degraded performance. We expect a significant drop in accuracy on distribution-shift-heavy problems (e.g., -10-15% on multi-step reasoning).

**Ablation: constraint on π_old** — Constraining to π_old does not directly bound reward bias (since the reward is defined relative to π_pre), so we expect worse performance than TRCLR, especially on problems where the student's previous policy diverges from π_pre.

### What would falsify this idea
If TRCLR's accuracy gains are uniform across all problem types rather than concentrated on problems where distribution shift from the teacher's pre-RL policy is large (e.g., multi-step reasoning problems), then the trust region constraint is not addressing the intended reward bias mechanism. Additionally, if the ablation with constraint on π_old matches TRCLR, then using π_pre as anchor is not critical.

## References

1. Weak-to-Strong Generalization via Direct On-Policy Distillation
2. Black-Box On-Policy Distillation of Large Language Models
3. RLKD: Distilling LLMs' Reasoning via Reinforcement Learning
4. Teaching Large Language Models to Reason with Reinforcement Learning
5. On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes
6. WizardMath: Empowering Mathematical Reasoning for Large Language Models via Reinforced Evol-Instruct
7. Reinforced Self-Training (ReST) for Language Modeling
8. Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback
