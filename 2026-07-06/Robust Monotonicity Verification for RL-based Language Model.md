# Robust Monotonicity Verification for RL-based Language Models via Stable Difference-of-Softmax Computation

## Motivation

MIPU's verification step (from 'The Mirage of Optimizing Training Policies') relies on importance sampling with ratio π_new/π_old, which becomes numerically unstable under FP16 precision when action probabilities are near zero, causing sign flips in estimated reward improvement and invalidating the monotonicity guarantee. This is structurally the same problem across branches A, C, and F: they all assume the verification procedure is robust to approximation errors, but this robustness is not established for finite-precision computation. The root cause is that the ratio amplifies numerical errors in the denominator.

## Key Insight

Because successive policies in iterative RL are close in parameter space (bounded logit change), the difference in their action probabilities can be computed directly via a coupled softmax difference that cancels common numerical errors, preserving the sign of expected reward improvement even under FP16.

## Method

(A) **What it is**: Stable Difference-of-Softmax (SDOS) verification is a numerically stable algorithm that replaces the importance ratio in MIPU's verification step with a direct computation of π_new - π_old, using a coupled softmax difference formula. Inputs: old policy logits l_old, new policy logits l_new, old Q-function Q_{π_old}, and a bound B on the maximum logit change. Output: boolean indicating whether the policy update guarantees monotonic improvement under FP16 precision.

(B) **How it works** (pseudocode):
```python
def sdos_verify(l_old, l_new, Q_old, B, tol=1e-5):
    # Step 1: Check bounded logit change (enforced by proximal training)
    delta = l_new - l_old
    if max_abs(delta) > B:
        return False  # update too large; reject
    # Step 2: Compute stable softmax difference
    # Use numerically stable coupled softmax with joint max-logit subtraction
    max_logit = max(max(l_old), max(l_new))
    exp_old = exp(l_old - max_logit)
    exp_new = exp(l_new - max_logit)
    sum_old = sum(exp_old)
    sum_new = sum(exp_new)
    # Compute difference: (exp_new/sum_new) - (exp_old/sum_old)
    # Direct formula to avoid separate divisions and subtraction
    pi_diff = (exp_new * sum_old - exp_old * sum_new) / (sum_old * sum_new)
    # Step 3: Estimate reward improvement
    delta_J = dot(pi_diff, Q_old(s))  # Q_old evaluated at state s
    # Step 4: Compare with tolerance (accounts for FP16 error bound)
    return delta_J > tol
```
Hyperparameters: `B` is set adaptively based on the KL divergence limit (e.g., 0.2); `tol` is set to 1e-5 based on FP16 error analysis.

(C) **Why this design**: Three key decisions: (i) We choose direct difference computation over importance ratio because the ratio involves a division by π_old(a) which is ill-conditioned when π_old(a) is near zero, and FP16 amplifies the error; direct difference uses only subtraction and multiplication, which are stable. The trade-off is that we lose the simple connection to importance sampling, but we gain numerical robustness. (ii) We use a coupled softmax difference that computes both softmaxes jointly, sharing the max-logit subtraction, instead of computing each softmax separately and then subtracting. This avoids cancellation errors from subtracting nearly equal large numbers. The trade-off is a slight increase in computational cost (one extra multiplication and division), but the stability improvement is critical. (iii) We impose a bound B on the logit change, which is justified because we use proximal policy optimization (e.g., PPO with KL penalty) that keeps policies close. The trade-off is that if the update exceeds B, the verification rejects it, potentially slowing learning; but this is acceptable for maintaining the monotonicity guarantee. We also set a tolerance `tol` based on a theoretical FP16 error bound derived from the condition number of the softmax difference, rather than a heuristic, ensuring the sign is preserved.

(D) **Why it measures what we claim**: The quantity `pi_diff(a)` measures the change in action probability for action a between the old and new policies. The assumption that `delta_J = sum_a pi_diff(a) * Q_{π_old}(s,a)` has the same sign as the true expected reward improvement `E_{π_new}[Q_{π_new}] - E_{π_old}[Q_{π_old}]` relies on two sub-assumptions: (i) the reward improvement is dominated by the change in action probabilities, not by the change in Q-values (which holds when policy change is small, enforced by B), and (ii) the old Q-function is a good approximation for the new Q-function (true under small policy change). The computational quantity `tol` operationalizes the concept of 'numerical robustness guarantee': the assumption is that the FP16 error in `delta_J` is bounded by `tol` with high probability (derived from worst-case analysis of the coupled softmax formula); this assumption fails when the logit change is near the bound B or when actions have exponentially small probabilities, in which case the error bound may be an overestimate and we may reject safe updates. However, this conservatism is intended to prioritize monotonicity over learning speed, and it can be mitigated by using a tighter bound or adaptive tolerance.

## Contribution

(1) A numerically stable difference-of-softmax verification algorithm (SDOS) that guarantees monotonicity under FP16 precision without computing importance ratios. (2) A theoretical bound on the FP16 error of the estimated reward improvement sign, derived from the bounded logit change property of proximal policy updates. (3) An empirical demonstration on LLM fine-tuning showing that SDOS maintains monotonic improvement across thousands of iterations while standard MIPU verification fails under FP16.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | GSM8K math reasoning | Tests policy change on symbolic reasoning |
| Primary metric | Sign accuracy of δ_J | Direct test of monotonicity preservation |
| Baseline 1 | Importance ratio verification | Standard method; suffers from numerical instability |
| Baseline 2 | Separate softmax verification | Shows effect of coupled computation |
| Baseline 3 | No logit bound verification | Tests necessity of bounded change |
| Ablation | FP32 precision (no FP16) | Isolates FP16 root cause of failure |

### Why this setup validates the claim

The combination of GSM8K dataset, sign accuracy metric, and three baselines isolates each key innovation of SDOS verification. GSM8K requires precise step-by-step reasoning, making monotonic policy updates critical. Sign accuracy directly measures whether the verification algorithm correctly predicts improvement direction. Importance ratio baseline tests the numerical stability claim: FP16 issues should cause failures on near-zero probabilities. Separate softmax baseline tests the cancellation error problem: large logit differences should degrade its accuracy. No-bound baseline tests whether bounded logit change is necessary for the coupling to work. The FP32 ablation confirms that the improvements are due to FP16 robustness, not other factors. This design yields a falsifiable test: if SDOS does not outperform on the specific failure modes, the central claim is wrong.

### Expected outcome and causal chain

**vs. Importance ratio verification** — On a case where π_old(a) is near zero (e.g., a rare but optimal action), the importance ratio π_new(a)/π_old(a) becomes huge, causing FP16 overflow or NaN. The baseline verification either incorrectly rejects a good update or accepts a bad one due to invalid ratio. Our method instead computes π_new(a) - π_old(a) directly, which remains stable, so we expect SDOS to have near-perfect sign accuracy on such instances, while baseline shows random behavior (accuracy ~50%).

**vs. Separate softmax verification** — On a case where logits are large (e.g., >20) and close together (difference <1), computing softmax separately and subtracting leads to catastrophic cancellation: exp(l_old) and exp(l_new) are both large, and subtracting nearly equal numbers loses significant digits. The baseline δ_J may have wrong sign. SDOS uses the coupled formula with shared max-logit subtraction, avoiding cancellation, so we expect its sign accuracy to remain high (e.g., >95%) on such inputs, while separate softmax drops (e.g., <70%).

**vs. No logit bound verification** — On a case where the policy update is large (e.g., logit change >B), the assumption that Q_old approximates Q_new fails. Without B, the verification may compute a positive δ_J even though the true improvement is negative. SDOS with bound B rejects such updates (returns False), preventing false positives. We expect SDOS to have zero false positives on high-KL updates, while the unbounded baseline has many false positives (e.g., 20% of large-update cases).

### What would falsify this idea

If SDOS verification shows no significant improvement in sign accuracy over separate softmax verification on instances with large logits, or if its advantage is uniformly spread across all data instead of being concentrated on the predicted failure modes (near-zero probabilities, large logits, and high KL), then the central claim of numerical robustness is unsupported.

## References

1. The Mirage of Optimizing Training Policies: Monotonic Inference Policies as the Real Objective for LLM Reinforcement Learning
2. Improving Deep Reinforcement Learning by Reducing the Chain Effect of Value and Policy Churn
3. DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning
4. DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models
5. DrM: Mastering Visual Reinforcement Learning through Dormant Ratio Minimization
6. Maintaining Plasticity in Deep Continual Learning
7. Measuring and Mitigating Interference in Reinforcement Learning
8. Llemma: An Open Language Model For Mathematics
