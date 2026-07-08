# Intrinsic Consistency Certificate: Reward-Free Monotonic Policy Improvement for Pretrained LLMs

## Motivation

Current monotonic verification methods for LLM reinforcement learning, such as MIPU, require a two-step procedure that constructs a candidate update and then verifies it via an additional forward pass, incurring significant computational overhead. Moreover, they depend on external reward signals which may be unavailable or expensive to obtain. This dual bottleneck prevents scalable deployment on pretrained LLMs.

## Key Insight

The gradient and Fisher information of the policy on a batch of prompts contain sufficient curvature information to derive a closed-form lower bound on policy improvement, and for pretrained LLMs, the average increase in top-1 token log-probability on held-out prompts serves as a consistent monotonic proxy for reward due to the statistical regularity of language model outputs.

## Method

We propose Intrinsic Consistency Certificate (ICC), a method that computes a closed-form lower bound on policy improvement using only the gradient and Fisher information matrix from the current training batch, and replaces external reward with an intrinsic consistency score derived from the increase in top-1 token log-probability on a held-out prompt set. ICC outputs a binary certificate indicating whether the proposed policy update guarantees monotonic improvement, without additional forward passes or reward computation.

**Assumption (load-bearing):** The mean increase in top-1 token log-probability on held-out prompts is a consistent monotonic proxy for reward in language modeling. This is justified by the statistical regularity of language model outputs, but we verify it empirically before each update using a calibration set of 512 held-out prompts drawn from the same distribution. We compute the Spearman correlation between per-prompt log-probability change and a small reward signal (e.g., accuracy on a subset of 128 calibration prompts). If the correlation is ≥ 0.3, we proceed with the consistency certificate; otherwise, we skip the update and log a warning. This calibration step ensures the proxy remains valid under distribution shift.

**Pseudocode (with hyperparameters specified):**
```python
def icc_certificate(policy, batch_prompts, heldout_prompts, calib_prompts, calib_rewards, epsilon=0.01, lambda_=0.5, correlation_threshold=0.3):
    # Step 1: Compute gradient and diagonal Fisher on batch
    grad = compute_gradient(policy, batch_prompts)  # shape [d]
    F = compute_diag_fisher(policy, batch_prompts)  # shape [d]

    # Step 2: Natural gradient direction (elementwise division)
    d = epsilon * grad / (F + 1e-8)

    # Step 3: Compute intrinsic consistency on held-out prompts
    heldout_grads = compute_gradient_per_sample(policy, heldout_prompts)  # [M, d]
    delta_logprobs = (heldout_grads @ d)  # linear approximation, shape [M]
    consistency = delta_logprobs.mean()

    # Step 4: Calibration check: verify proxy correlation on calibration set
    calib_delta_logprobs = (compute_gradient_per_sample(policy, calib_prompts) @ d).mean(axis=1)  # [C,]
    correlation = spearmanr(calib_delta_logprobs, calib_rewards).correlation
    if correlation < correlation_threshold:
        return False  # skip update, proxy unreliable

    # Step 5: Closed-form lower bound on batch improvement (second-order Taylor)
    L = grad @ d - 0.5 * d @ (F * d)  # diagonal: elementwise product

    # Step 6: Certificate condition
    certificate = (L + lambda_ * consistency) >= 0
    return certificate, L, consistency
```
Hyperparameters: epsilon=0.01 (natural gradient step size), lambda_=0.5 (trade-off between batch and held-out), correlation_threshold=0.3 (minimum Spearman correlation for proxy validity). The natural gradient step uses diagonal Fisher to keep computation O(d) and feasible for LLMs. We compare to MIPU, which requires an extra forward pass and a reward model; ICC avoids both.

**Comparison to prior monotonic verification (MIPU):** MIPU constructs a candidate update and then runs an additional forward pass to verify monotonic improvement via a KL constraint, requiring a reward model. ICC instead computes a closed-form lower bound from gradient and Fisher, eliminating the extra forward pass, and replaces reward with the intrinsic consistency score, removing the need for external reward signals.

**Resource estimates:** For a 1B-parameter model using LoRA (rank=8, applied to all linear layers), the diagonal Fisher computation on a batch of 16 prompts with sequence length 512 takes approximately 4 GB of GPU memory and 2 seconds per update on an A100. The calibration step (512 prompts) adds about 1 second. Total per-update cost is ~3 seconds, which is less than MIPU's two-forward-pass cost (~5 seconds).

**Why this design:** (preserved and refined) We chose natural gradient (elementwise division by diagonal Fisher) over vanilla gradient because the Fisher diagonal provides per-parameter curvature, enabling a tighter second-order lower bound that accounts for parameter sensitivity; this prevents the certificate from being overly optimistic under high curvature. We opted for a diagonal Fisher approximation over full matrix to keep the computation O(d) and feasible for LLMs with billions of parameters, at the cost of ignoring off-diagonal interactions—acceptable because the lower bound remains valid (though looser) and off-diagonals are empirically small in large models. We selected held-out prompts as a separate set from the training batch to measure generalization: the consistency score approximates how the update transfers to unseen prompts, avoiding overfitting signals from the batch. The additive combination L + lambda_*consistency was chosen over a product or threshold condition because it allows both terms to contribute independently; lambda_=0.5 was set via a small validation sweep on a calibration set to balance batch-level curvature and held-out extrapolation, accepting the cost of an additional hyperparameter to tune. We added a calibration check on the consistency-reward correlation before each update to guard against proxy failure when the held-out distribution shifts.

**Why it measures what we claim:** (preserved and expanded) The lower bound L measures guaranteed improvement on the training batch because it is derived from a second-order Taylor expansion of the policy's objective under a natural gradient step, assuming the step is small enough that higher-order terms are negligible (common in trust-region methods); this assumption fails when epsilon is too large (e.g., >0.05) or the curvature changes rapidly, in which case L may overestimate the actual improvement (no longer a lower bound). The consistency score measures the expected increase in top-1 log-probability on held-out prompts because it uses a first-order linear approximation of the update's effect, assuming that held-out prompts are drawn from the same distribution as training and that the linear model holds for small steps; this assumption fails when the held-out distribution shifts (e.g., different prompt styles) or the update is large (epsilon>0.05), in which case consistency may misrepresent the true held-out change. The certificate condition L + lambda_*consistency >= 0 operationalizes monotonic improvement (defined as non-negative combined improvement) because it ensures that even a negative batch-level bound can be compensated by positive held-out gains—a proxy for generalization that reflects the field's goal of reward-free improvement; however, this condition relies on the additive surrogate being monotonic in true overall improvement, an assumption that fails when the two terms are on incompatible scales or when held-out consistency is not actually monotonic with respect to deployment performance, potentially admitting updates that degrade actual task performance. The calibration check partially addresses this by verifying the proxy before each update.

## Contribution

(1) A closed-form lower bound for policy improvement using gradient and diagonal Fisher information, eliminating the need for two-step verification (e.g., MIPU) and its associated computational overhead. (2) The intrinsic consistency score, a reward-free signal computed from held-out log-probability increases, which substitutes external reward signals for pretrained LLMs. (3) A combined certificate that guarantees monotonic improvement without additional forward passes or reward models, enabling efficient and autonomous LLM fine-tuning.

## Experiment

### Evaluation Setup
| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | GSM8K (7485 training, 1319 test) | Math word problems test generalization. Held-out set of 512 prompts used for calibration, another 512 for consistency computation. |
| Primary metric | Task accuracy on test set | Directly measures final performance. Also track held-out accuracy for monotonicity check. |
| Baseline 1 | Supervised learning (SL) on training set (3 epochs, AdamW lr=1e-5) | Standard approach for comparison. |
| Baseline 2 | PPO without certificate (2 updates per batch, reward = accuracy on a separate reward set of 256 prompts, clip epsilon=0.2) | Classic RL baseline for LLMs. Uses learned reward model (distilled from ground truth) to isolate effect of certificate. |
| Ablation-of-ours | ICC w/o consistency term (lambda_=0) | Isolates batch lower bound effect. |
| Additional ablation | ICC without calibration (skip correlation check) | Tests the necessity of the calibration step. |
| Compute budget | All experiments run on 8x A100 80GB GPUs; total ~200 GPU hours for ICC, ~250 for PPO. |

### Why this setup validates the claim
This experimental design tests whether ICC's certificate guarantees monotonic improvement without external rewards. GSM8K provides a held-out prompt set to compute consistency, and accuracy measures actual task success. The held-out prompts for consistency are disjoint from the calibration set, ensuring no data leakage. Comparing against SL isolates the benefit of RL, while PPO without certificate tests the value of the monotonic guarantee. The ablation (ICC minus consistency) isolates the contribution of the held-out term. The additional ablation (no calibration) tests the calibration step's role in preventing premature updates. We also include a correlation analysis: for every 10 updates, we compute the Spearman correlation between consistency and true held-out accuracy (on the 512 held-out prompts) to validate the proxy assumption. If ICC outperforms both baselines and the ablations, it confirms that the certificate and consistency together improve learning efficiency and final performance. The primary metric, accuracy, directly reflects the policy's output quality; a monotonic improvement guarantee should translate to higher accuracy with less variance. Together, these choices form a falsifiable test: if ICC does not yield better accuracy than PPO without certificate, or if the ablation matches ICC, then the central claim is unsupported.

### Expected outcome and causal chain
**vs. Supervised learning (SL)** — On a case where the policy must handle a new problem type unseen in SL data (e.g., problems with larger numbers), SL produces incorrect answers because it lacks exploration to adapt to novel reasoning patterns. Our method instead explores via natural gradient updates guided by the consistency certificate, which ensures each step improves held-out prompts, so we expect ICC to achieve higher accuracy on the held-out subset (e.g., >5% gap on problems with large numbers) while performing similarly on in-distribution problems (within 1%).

**vs. PPO without certificate** — On a case where a policy update would overfit to a training batch (e.g., repeated similar prompts from a single skill), PPO without certificate accepts the update because it only uses batch reward (which may be high due to overfitting), causing a drop in held-out accuracy of up to 5%. Our method rejects such updates via the certificate (L + lambda*consistency < 0), so we expect ICC to show a monotonic improvement in held-out accuracy across training steps (no dips >0.5%), while PPO without certificate exhibits non-monotonic behavior (e.g., oscillations of ≥3% accuracy).

**vs. ICC without consistency term** — On a case where batch improvement L is high but held-out generalization is poor (e.g., memorizing training patterns), ICC without consistency still accepts the update, leading to held-out degradation. Our method with full certificate rejects it because consistency term is negative, so we expect full ICC to maintain or improve held-out accuracy, while the ablation shows occasional drops of >2% accuracy on held-out prompts.

**vs. ICC without calibration** — On a case where the held-out distribution shifts (e.g., after 100 updates, the model's log-probability changes become less correlated with accuracy), ICC without calibration may incorrectly accept updates that degrade performance. Full ICC with calibration detects low correlation and skips such updates, preventing degradation. We expect full ICC to maintain monotonic improvement while the no-calibration variant shows occasional drops (>1% accuracy).

### What would falsify this idea
If ICC's held-out accuracy does not exhibit a clear monotonic trend (e.g., it drops by >2% after any update the certificate accepts) or if the advantages over PPO without certificate are negligible (<0.5% gap) on both in-distribution and held-out subsets, then the certificate's guarantee is not translating into empirical monotonic improvement. Additionally, if the correlation between consistency and true held-out accuracy falls below 0.2 for more than 20% of updates (even after calibration), the proxy assumption is empirically invalid.

## References

1. The Mirage of Optimizing Training Policies: Monotonic Inference Policies as the Real Objective for LLM Reinforcement Learning
2. Improving Deep Reinforcement Learning by Reducing the Chain Effect of Value and Policy Churn
3. DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning
4. DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models
5. DrM: Mastering Visual Reinforcement Learning through Dormant Ratio Minimization
6. Maintaining Plasticity in Deep Continual Learning
7. Measuring and Mitigating Interference in Reinforcement Learning
8. Llemma: An Open Language Model For Mathematics
