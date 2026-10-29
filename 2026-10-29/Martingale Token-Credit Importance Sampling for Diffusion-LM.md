# Martingale Token-Credit Importance Sampling for Diffusion-LM Policy Gradients

## Motivation

Existing surrogate-based policy-gradient methods for diffusion language models (e.g., DTCA's confidence-improvement credit, GDPO's sequence-level ELBO) inherit an importance-sampling correction that was only ever analyzed at sequence granularity; when credit is assigned per token, the IS ratio becomes a product of per-token ratios that are correlated with the commitment order, so the inherited correction is neither unbiased nor variance-bounded. This structural gap is shared across the ELBO-based (GDPO), confidence-based (DTCA), and process-reward branches, none of which re-derive the IS variance when credit moves from sequence to token level.

## Key Insight

The token-level IS ratio is a martingale with respect to the denoising filtration if and only if the surrogate gradient is conditionally unbiased at every token commitment, and this conditional-unbiasedness condition is exactly the credit-consistency functional that simultaneously quantifies the surrogate-vs-true gradient gap and bounds the IS variance.

## Method

**What it is.** We introduce **Martingale Token-Credit IS** (MTC-IS), a training-time reweighting scheme that turns any surrogate token-credit signal (ELBO delta, confidence improvement, or process reward) into a token-level importance-sampling estimator whose bias and variance are controlled by a single computable functional. Inputs: a diffusion-LM policy πθ, a denoising trajectory with T commitment steps, per-token surrogate credits c_t, and a target reward. Output: a reweighted policy-gradient update.

**Load-bearing assumption (stated explicitly).** MTC-IS assumes that the token-likelihood gradient estimator g_t^true used in steps 2–3 is a faithful target for the surrogate credit at every commitment step t, i.e. that the surrogate's conditional mean equals the true token-likelihood gradient. In masked diffusion this target is the Rao-Blackwellized ELBO derivative, which is a variational bound, not the exact marginal-likelihood gradient; its bias is the KL gap between the variational posterior and the true reverse process, and this gap is largest at early, high-uncertainty commitment steps. Consequently, the residual (g_t − g_t^true) is a well-defined, estimable bias term only up to this ELBO KL gap, and the martingale property of M_t = ∏_{s≤t} r_s holds under the exact-policy filtration; with an approximate g_t^true the conditional-unbiasedness condition is satisfied only up to the gap. The method therefore treats the martingale reweighting as a variance-controlled correction whose bias is bounded by the credit-consistency functional C plus the ELBO KL gap, and it reports both quantities separately. When no unbiased token-gradient estimator is available (black-box sampler), the method drops the martingale claim and presents a_t as a heuristic variance-reduction weight with empirical justification only.

**How it works.**

```pseudocode
# Given: trajectory x_{0:T}, commitments x_t, surrogate credits c_t, reward R
# Hyperparameters: beta (credit-consistency temperature, init 1.0), lambda (variance penalty, init 0.5),
#   clip_eps (0.2), eps (1e-8), C_max (0.1), dual lr for beta (0.01)

1. Compute per-token IS ratios:
   r_t = pi_theta(x_t | x_{<t}) / pi_old(x_t | x_{<t})        # token-level likelihood ratio

2. Compute the credit-consistency functional at each t:
   # g_t = surrogate gradient contribution from c_t
   # g_t^true = token-likelihood gradient estimate (masked-diffusion Rao-Blackwellized ELBO derivative;
   #            single-sample ELBO-derivative estimate with a variance-reduction baseline when the
   #            Rao-Blackwellized form is unavailable)
   # delta_t is a debiased two-sample estimate of surrogate bias, separable from Monte-Carlo noise:
   #   draw two independent ELBO-derivative samples g_t^true(1), g_t^true(2);
   #   delta_t = || g_t - mean(g_t^true(1), g_t^true(2)) ||^2 / (|| mean(g_t^true(1), g_t^true(2)) ||^2 + eps)
   #   minus the noise floor || g_t^true(1) - g_t^true(2) ||^2 / (2*(|| mean ||^2 + eps))
   delta_t = max(0, delta_t_raw - noise_floor_t)
   C = sum_t delta_t * w_t,   w_t = softmax(-beta * delta_t)   # credit-consistency score

3. Martingale reweighting (enforce conditional unbiasedness up to ELBO KL gap):
   # M_t = prod_{s<=t} r_s is a martingale under the denoising filtration when g_t^true is exact;
   # with the ELBO-derivative estimate the correction is unbiased up to the KL gap, reported separately.
   g_hat_t = g_t - (1 - 1/M_t) * (g_t - g_t^true)            # conditional-unbiased residual
   # Equivalent: g_hat_t = g_t^true + (g_t - g_t^true)/M_t

4. Variance-penalized token weights:
   a_t = clip( M_t * exp(-lambda * delta_t), 1-clip_eps, 1+clip_eps )
   # a_t is the IS weight that makes the reweighted credit a martingale (up to ELBO KL gap)

5. Policy-gradient update:
   L = - E[ sum_t a_t * g_hat_t * (R - baseline) ]
   theta <- theta - eta * grad_theta L

6. (Optional) Update beta via dual ascent on C <= C_max:
   beta <- beta + dual_lr * (C - C_max), clipped to [0.1, 10.0]
```

**Why this design.** We chose a martingale-based reweighting over the standard sequence-level IS correction because the sequence-level ratio only controls variance at the trajectory endpoint, whereas token-level credit assignment multiplies per-token ratios that are correlated through the commitment order; enforcing the martingale property at every commitment step (step 3) is the minimal structural condition that makes the surrogate conditionally unbiased at each token, accepting the cost that we must compute the token-likelihood gradient g_t^true (available in masked diffusion via the Rao-Blackwellized ELBO derivative, but not in black-box samplers). We chose a softmax temperature beta on the credit-consistency functional over a hard threshold because delta_t is noisy early in training and a smooth weighting prevents premature collapse onto a few tokens; the trade-off is that beta must be tuned or dual-ascended, adding a hyperparameter. We chose to clip the martingale weight a_t over unclipped IS because the product M_t can explode when the policy drifts, and clipping bounds the variance at the cost of a small bias that is controlled by clip_eps. This design is not a variant of DTCA or GDPO: DTCA uses confidence improvement as an uncalibrated token credit without any IS correction, and GDPO applies a sequence-level ELBO IS correction; neither enforces conditional unbiasedness at token commitment, which is the structural property MTC-IS establishes.

**Why it measures what we claim.** The credit-consistency functional C measures the surrogate-vs-true gradient gap because delta_t is the normalized squared difference between the surrogate credit gradient and the token-likelihood gradient estimate, debiased by subtracting the two-sample noise floor, under the assumption that the ELBO-derivative estimate is a faithful target up to its KL gap; this assumption fails when the sampler is black-box or the ELBO derivative is unavailable, in which case delta_t reflects Monte-Carlo noise rather than true bias. The martingale weight a_t measures the token-level IS variance because M_t is the product of per-token likelihood ratios and is a martingale under the denoising filtration if and only if the surrogate is conditionally unbiased; this assumption fails when the policy update violates the filtration (e.g., off-policy replay) or when g_t^true carries an ELBO KL gap, in which case a_t reflects stale-ratio drift or the KL gap rather than variance. The softmax weight w_t measures which tokens dominate the credit-consistency score because it upweights tokens with small delta_t, under the assumption that delta_t is a reliable per-token signal; this assumption fails when delta_t is dominated by a few high-variance tokens, in which case w_t reflects noise concentration rather than credit consistency.

## Contribution

['A token-level importance-sampling estimator for diffusion-LM policy gradients that enforces a martingale property with respect to the denoising filtration, yielding a reweighting scheme whose bias and variance are controlled by a single computable credit-consistency functional.', 'A credit-consistency functional that couples the surrogate signal (ELBO delta, confidence improvement, or process reward) to the exact token-likelihood gradient, enabling training-time detection and correction of surrogate-gradient gaps at token granularity.', 'An empirical demonstration that martingale reweighting improves reasoning performance over sequence-level IS corrections (GDPO) and uncalibrated token credits (DTCA) on diffusion-LM benchmarks.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | GSM8K + MATH reasoning splits | Multi-step reasoning exposes token-credit misattribution |
| Primary metric | Pass@1 accuracy vs. token-budget curve | Detects credit-quality gains, not just endpoint reward |
| Baseline | DTCA | Uncalibrated token credit, no IS correction |
| Baseline | GDPO | Sequence-level ELBO IS, no per-token unbiasedness |
| Baseline | diffu-GRPO | One-step unmasking likelihood, no martingale correction |
| Ablation-of-ours | MTC-IS w/o martingale step (a_t≡1) | Isolates the conditional-unbiasedness contribution |
| Ablation-of-ours | MTC-IS with high-variance g_t^true (single-sample ELBO derivative, no baseline) | Tests failure mode F: delta_t reflects noise not bias |
| Ablation-of-ours | MTC-IS with exact g_t^true replaced by score-function/REINFORCE surrogate (black-box setting) | Tests heuristic-weight fallback when martingale claim dropped |

### Why this setup validates the claim

The central claim is that enforcing conditional unbiasedness at each commitment step (the martingale reweighting) reduces the surrogate-credit bias that corrupts token-level policy gradients, and that this reduction converts into better reasoning accuracy per unit of token budget. DTCA tests the sub-claim that a token credit *without* any IS correction suffers from uncalibrated credit: if MTC-IS beats DTCA specifically on multi-step problems where early-token credit errors compound, the calibration story is supported. GDPO tests the sub-claim that *sequence-level* correction is insufficient: because it only controls endpoint variance, it should lag on problems with long denoising trajectories where per-token ratios drift. diffu-GRPO tests whether the martingale structure adds anything beyond a likelihood estimator. The a_t≡1 ablation (keeping g_hat_t) is the sharpest test: it retains the surrogate-correction term but removes the martingale weight, so any gain attributable to the reweighting functional alone is isolated. The high-variance g_t^true ablation directly tests failure mode F: if delta_t is dominated by Monte-Carlo noise, the debiased two-sample estimator should show inflated C and degraded accuracy, confirming that the credit-consistency functional is measuring bias rather than noise. The black-box ablation tests the heuristic fallback: with a score-function surrogate for g_t^true, the martingale claim is dropped and a_t is presented as a variance-reduction weight; if accuracy is preserved, the mechanism generalizes beyond masked diffusion. Pass@1 as a function of token budget is the right metric because the predicted effect is a *sample-efficiency* gain — better credit assignment should reach a given accuracy with fewer denoising steps — which a single endpoint number would mask.

### Expected outcome and causal chain

**vs. DTCA** — On a multi-step GSM8K problem where an early commitment (e.g. choosing the wrong intermediate quantity) is locally plausible but globally wrong, DTCA assigns that token high confidence-improvement credit because it never checks the credit against the token-likelihood gradient, so the policy reinforces the erroneous early commitment and the error compounds. Our method computes delta_t against g_t^true (debiased by the two-sample noise floor) and the martingale weight a_t suppresses tokens whose surrogate diverges from the gradient estimate, so the early wrong commitment receives attenuated credit. We expect a noticeable pass@1 gap concentrated on problems with ≥3 reasoning hops, with near-parity on single-hop problems where no compounding occurs.

**vs. GDPO** — On a long denoising trajectory where the policy drifts mid-generation, GDPO's sequence-level ELBO ratio stays near 1 at the endpoint even though intermediate per-token ratios have drifted substantially, so its gradient is effectively uncorrected for the tokens that actually caused the drift. Our method multiplies per-token ratios into M_t and clips a_t, so drift is detected and bounded at the commitment step where it occurs. We expect MTC-IS to show lower gradient variance (visible as smoother accuracy-vs-budget curves) and a larger accuracy gain on the longest trajectories, with the gap shrinking as trajectory length decreases.

**vs. diffu-GRPO** — On a problem where the one-step unmasking likelihood estimate is biased because the commitment order is non-sequential, diffu-GRPO's token likelihood is misestimated, so its policy gradient points in a slightly wrong direction and learning plateaus early. Our method does not rely on the one-step estimate for the correction; it uses the masked-diffusion ELBO derivative for g_t^true and the martingale weight for variance control, so the update direction stays aligned. We expect diffu-GRPO to plateau at lower accuracy while MTC-IS continues improving with more updates, with the divergence appearing after the initial easy-problem gains saturate.

**vs. MTC-IS with high-variance g_t^true** — When g_t^true is replaced by a single-sample ELBO-derivative estimate without a baseline, delta_t conflates surrogate bias with Monte-Carlo noise, so the debiased two-sample estimator should show inflated C and the martingale weight a_t should over-suppress tokens. We expect this ablation to underperform full MTC-IS, with the gap largest on early, high-uncertainty commitment steps where the ELBO KL gap and estimator variance are largest, confirming that the credit-consistency functional is measuring bias rather than noise.

**vs. MTC-IS with score-function surrogate (black-box)** — When g_t^true is replaced by a REINFORCE-style score-function surrogate, the martingale claim is dropped and a_t is a heuristic variance-reduction weight. We expect accuracy to be preserved or slightly lower than full MTC-IS but still above DTCA and GDPO, establishing that the mechanism generalizes beyond masked diffusion to black-box samplers.

### What would falsify this idea

If MTC-IS's gain over DTCA and GDPO is uniform across reasoning-hop counts and trajectory lengths rather than concentrated where the predicted failure modes (early-commitment compounding, mid-trajectory drift) occur, then the martingale conditional-unbiasedness mechanism is not the cause of any improvement. Equally falsifying: if the a_t≡1 ablation matches full MTC-IS, the reweighting functional contributes nothing and the claim collapses to the surrogate-correction term alone; and if the high-variance g_t^true ablation matches full MTC-IS, then delta_t is not separable from Monte-Carlo noise and the credit-consistency functional is not measuring bias.

## References

1. DTCA: Denoising Token Credit Assignment for Reinforcement Learning in Diffusion Language Models
2. Advancing Reasoning in Diffusion Language Models with Denoising Process Rewards
3. Improving Reasoning for Diffusion Language Models via Group Diffusion Policy Optimization
4. Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations
5. Simple and Effective Masked Diffusion Language Models
6. Simplified and Generalized Masked Diffusion for Discrete Data
7. Likelihood-Based Diffusion Language Models
8. Bayesian Flow Networks
