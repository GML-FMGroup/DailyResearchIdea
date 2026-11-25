# Step-Aware Discretization-Error Weighting for Finite-Step iTDM Policy Gradients

## Motivation

Across the diffusion-RL lineage, the diffusion sampler inside the RL loop is treated as an exact generative oracle, so no branch establishes how finite-step ODE discretization error propagates into the policy-gradient estimator. iTDM (E1) explicitly anchors its sample-level gradient at the behavior velocity and relies on an online auxiliary denoiser to track it, but the abstract treats this tracking as exact and does not account for the discretization error of the auxiliary denoiser across the ODE steps; under CFG rollouts and offline/off-policy updates the behavior–policy mismatch already breaks baseline invariance, and finite-step discretization compounds this by perturbing the reward-tilted velocity field in a step-dependent way that no existing method weights for.

## Key Insight

The discretization error of the auxiliary denoiser is a step-indexed perturbation of the reward-tilted velocity field whose contribution to the policy-gradient estimator is multiplicatively separable per ODE step, so reweighting the denoiser loss by the local error magnitude makes the finite-step gradient estimator baseline-invariant even when the behavior–policy mismatch is nonzero.

## Method

**What it is.** We propose SADW-iTDM (Step-Aware Discretization-Weighted iTDM), a finite-step correction to iTDM's policy-gradient estimator. Inputs: a pretrained diffusion policy v_theta (Stable Diffusion v1.5 UNet), a reward function (CLIP score on the decoded image), a deterministic ODE sampler with K steps (Euler; K ∈ {10, 20, 50, 100}), and an online auxiliary denoiser v_phi (a copy of the UNet velocity network, same architecture, trained from v_theta at initialization). Outputs: a step-weighted auxiliary-denoiser loss and a finite-step policy-gradient estimator that is invariant to baseline subtraction under nonzero behavior–policy mismatch.

**Load-bearing assumption (stated explicitly).** The method assumes that the per-step residual r_k = ||v_theta^K(x_k, t_k) - v_theta^{K_ref}(x_k, t_k)||, computed on a shared probe trajectory against a reference high-K velocity (K_ref = 4K, same solver), is a faithful, sample-comparable measure of step k's contribution to policy-gradient bias, and that the closed-form projection w_k = K * r_k / sum_j r_j makes sum_k w_k = K identically for every sample, so that baseline subtraction is invariant under nonzero behavior–policy mismatch. This assumption is explicit and is tested in the diagnostic experiment (see experiment section); it is not claimed as a theorem.

**How it works.**

```pseudocode
# K = number of ODE steps; t_k = k/K; beta = weighting sharpness; lambda = base LR
# K_ref = 4K (reference high-K velocity for error calibration)
# probe trajectory: shared across all samples in a batch, generated once per iteration
for each RL iteration:
    # 1. Rollout with deterministic ODE sampler, K steps
    x_K ~ N(0,I); for k=K..1: x_{k-1} = ODE_step(x_k, v_theta, t_k)
    # 2. Per-step local discretization-error magnitude (externally calibrated)
    #    Generate a shared probe trajectory with K_ref steps from the same x_K
    #    (or from a fixed probe batch of 8 prompts) and evaluate both velocities at x_k
    for k in 1..K:
        # residual between K-step and K_ref-step velocity at the same x_k
        r_k = || v_theta^{K}(x_k, t_k) - v_theta^{K_ref}(x_k, t_k) ||
        # closed-form projection: weight sum is exactly K for every sample
        w_k = K * r_k / sum_j r_j
        # optional cap for numerical stability (applied after projection, then renormalized)
        w_k = min(w_k, w_max); w_k = K * w_k / sum_j w_k
    # 3. Step-weighted auxiliary denoiser loss
    L_aux = sum_k w_k * || v_phi(x_k, t_k) - v_theta(x_k, t_k) ||^2
    update v_phi by gradient descent on L_aux
    # 4. Finite-step policy-gradient estimator anchored at behavior velocity
    #    (iTDM sample-level gradient, but with per-step weight w_k)
    g = sum_k w_k * ( v_phi(x_k, t_k) - v_theta(x_k, t_k) ) * grad_theta log pi_theta(x_{k-1}|x_k)
    # 5. Baseline-invariant update: subtract reward baseline before weighting
    g = g - (sum_k w_k) * b   # b = running mean of rewards; invariance holds because sum_k w_k = K for all samples
    update theta by -lambda * g
```

Hyperparameters: K (ODE steps, e.g., 20), K_ref = 4K (reference steps, e.g., 80), beta (weighting sharpness, e.g., 1.0; note beta is absorbed into the projection and set to 1.0 in all experiments), w_max (cap, e.g., 5.0, applied after projection and renormalized), b (running reward baseline, EMA with decay 0.99), lambda (base LR, e.g., 1e-5 for SD v1.5 fine-tuning).

**Why this design.** We chose to weight the auxiliary-denoiser loss by the per-step discretization-error magnitude rather than use a uniform loss because the perturbation of the reward-tilted velocity field is step-dependent (early steps have larger local truncation error under CFG rollouts), and a uniform loss lets high-error steps dominate the gradient estimator; the cost is that r_k must be recomputed each iteration against a K_ref reference, adding one forward pass per ODE step on a shared probe trajectory (8 prompts, not the full batch). We chose to cap w_k at w_max rather than use unbounded weights because a single outlier step with large r_k would otherwise destabilize the auxiliary denoiser, accepting that genuinely large discretization errors are under-corrected; the cap is applied after the projection and renormalized so that sum_k w_k = K is preserved. We chose to subtract the reward baseline after weighting rather than before because baseline invariance requires the weight sum to be sample-independent, and the closed-form projection w_k = K * r_k / sum_j r_j makes sum_k w_k = K identically for every sample; subtracting before weighting would reintroduce the baseline-induced drift that iTDM (E1) identifies; the cost is that b must be tracked as a running mean rather than a per-batch statistic. Unlike the signed reward-weighted regression (RWR) baseline that iTDM (E1) critiques, and unlike standard iTDM which treats the auxiliary denoiser as an exact oracle, SADW-iTDM makes the finite-step error explicit in the gradient; a domain expert would not describe this as a variant of RWR because the weighting is on the denoiser loss, not on the reward, and the invariance argument uses the weight sum rather than the reward sign.

**Why it measures what we claim.** The per-step residual r_k measures the local discretization error of the ODE sampler because it compares the K-step velocity to a K_ref-step reference velocity at the same x_k, so it is an externally calibrated error signal rather than a solver-specific algebraic remainder; this assumption fails when the K_ref reference itself has non-negligible error (e.g., K_ref not large enough), in which case r_k reflects the reference's residual rather than the K-step truncation error and w_k under-weights the true error. The step weight w_k measures the contribution of step k to the policy-gradient bias because the gradient estimator is a weighted sum of per-step velocity residuals and the bias from discretization is linear in the per-step residual under the assumption that the auxiliary denoiser error is independent across steps; this assumption fails when the online denoiser is updated with correlated batches, in which case w_k reflects batch correlation rather than per-step bias. The weight sum sum_k w_k measures the total effective number of steps contributing to the gradient because baseline invariance requires this sum to be sample-independent, which holds by construction under the closed-form projection w_k = K * r_k / sum_j r_j (sum_k w_k = K identically); this construction fails when r_k is computed from different probe trajectories for different samples (e.g., under off-policy updates where trajectories come from different behavior policies), in which case the projection must be applied per-sample and the invariance is only approximate if the probe trajectory is shared across samples with different behavior policies.

## Contribution

["A step-aware discretization-error weighting scheme (SADW-iTDM) that modifies iTDM's auxiliary-denoiser loss and policy-gradient estimator with per-step weights derived from the ODE local truncation residual, yielding a finite-step baseline-invariant gradient estimator.", 'A design principle: baseline invariance under finite-step sampling requires the weight sum to be sample-independent, which is achieved by computing weights from a shared ODE trajectory and subtracting the reward baseline after weighting.', 'An empirical protocol for measuring discretization-error propagation by varying K and beta and comparing gradient-estimator variance against uniform-weight iTDM.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Text2Image prompts (SD v1.5 fine-tuning), 512 held-out prompts | Matches iTDM's forward-diffusion RL setting directly |
| Primary metric | Held-out prompt reward alignment (CLIP score) | Directly measures policy-gradient bias from finite K |
| Baseline | iTDM (uniform auxiliary loss) | Tests whether step-weighted denoiser loss reduces bias |
| Baseline | Signed RWR forward-diffusion RL | Tests against the criticized baseline iTDM replaces |
| Baseline | iTDM with large K (K_ref = 4K) | Isolates finite-step error as the causal factor |
| Baseline | SNR-based timestep weighting | Tests whether w_k is a reparameterization of known schemes |
| Baseline | Uniform timestep-sampling | Tests whether w_k is a reparameterization of known schemes |
| Ablation-of-ours | SADW-iTDM with constant w_k=1 | Tests whether weighting itself, not extra compute, drives gain |
| Ablation-of-ours | SADW-iTDM with self-referential residual (original r_k) | Tests whether external calibration matters |
| Diagnostic | Per-step residual profile r_k vs k and CFG scale | Verifies early-step dominance empirically |
| Diagnostic | Measured gradient bias vs w_k (ground-truth large-K reference) | Validates that w_k tracks bias |

Compute budget: 8× A100 80GB, 4 days total (pre-registered). K-sweep at {10, 20, 50, 100}, CFG at {1, 3, 7}. Per-step residual logging released.

### Why this setup validates the claim

The central claim is that finite-step discretization error is step-dependent and that weighting the auxiliary-denoiser loss by an externally calibrated per-step residual reduces policy-gradient bias under behavior–policy mismatch. The dataset and metric must expose that mismatch: fine-tuning Stable Diffusion v1.5 on 512 text prompts with a CLIP reward creates exactly the CFG-rollout and offline/on-policy mixture where iTDM's baseline-invariance assumption is violated. The primary metric, held-out prompt reward alignment, is the right detector because it measures the downstream consequence of gradient bias rather than any internal quantity. The baselines form a five-way test: signed RWR tests whether SADW-iTDM beats the method iTDM critiques; iTDM with uniform auxiliary loss tests whether step weighting matters at fixed K; iTDM with large K tests whether the gain disappears when discretization error is driven toward zero; SNR-based and uniform timestep-sampling baselines test whether w_k is merely a reparameterization of known weighting schemes. The constant-weight ablation isolates the weighting mechanism from the extra forward-pass cost, and the self-referential-residual ablation isolates the effect of external calibration from the original residual. The diagnostic experiments measure the actual per-step residual profile r_k across k and CFG scale to verify the claimed early-step dominance, and plot measured gradient bias (via a ground-truth large-K reference) against w_k to validate the equivalence. If SADW-iTDM's advantage is concentrated where per-step residuals are large and vanishes at large K, the causal story is confirmed; if it is uniform across K, the mechanism is wrong.

### Expected outcome and causal chain

**vs. Signed RWR forward-diffusion RL** — On a case where CFG rollouts create a behavior–policy velocity mismatch at early ODE steps, the baseline produces baseline-induced gradient drift because its signed reward weighting assumes policy velocity equals behavior marginal velocity. Our method instead anchors the sample-level gradient at the behavior velocity with externally calibrated per-step residual weights, so early high-error steps are down-weighted rather than allowed to dominate, and we expect a visible reward-alignment gap that grows with CFG guidance scale.

**vs. iTDM (uniform auxiliary loss)** — On a case where early ODE steps have larger local truncation residuals than late steps, the baseline produces a gradient estimator dominated by high-error steps because its uniform denoiser loss lets those steps contribute disproportionately. Our method instead scales each step's contribution by w_k, so the effective gradient is balanced across the trajectory, and we expect a moderate but consistent improvement that is largest when K is small (e.g., K=10–20) and shrinks as K grows.

**vs. iTDM with large K** — On a case where K is large enough that discretization error is negligible, the baseline produces nearly the same gradient as our method because per-step residuals r_k become small and roughly uniform. Our method instead reduces to approximately uniform weighting in this regime, so we expect parity between the two methods at large K, which is the predicted null result confirming that finite-step error is the causal factor.

**vs. SNR-based timestep weighting** — On a case where per-step residuals vary strongly across the trajectory, the baseline produces a gradient estimator that is not baseline-invariant under nonzero mismatch because its weight sum is sample-independent but its per-step contributions are misallocated. Our method instead reallocates weight toward steps with larger bias contribution, so we expect the ablation to underperform specifically on high-CFG, low-K subsets while matching our method elsewhere.

**vs. Uniform timestep-sampling** — On a case where per-step residuals vary strongly across the trajectory, the baseline produces a gradient estimator that is not baseline-invariant under nonzero mismatch because its weight sum is sample-independent but its per-step contributions are misallocated. Our method instead reallocates weight toward steps with larger bias contribution, so we expect the ablation to underperform specifically on high-CFG, low-K subsets while matching our method elsewhere.

**vs. SADW-iTDM with constant w_k=1** — On a case where per-step residuals vary strongly across the trajectory, the ablation produces a gradient estimator that is not baseline-invariant under nonzero mismatch because its weight sum is sample-independent but its per-step contributions are misallocated. Our method instead reallocates weight toward steps with larger bias contribution, so we expect the ablation to underperform specifically on high-CFG, low-K subsets while matching our method elsewhere.

**vs. SADW-iTDM with self-referential residual** — On a case where the self-referential residual is dominated by velocity-field curvature and CFG-induced mismatch, the ablation produces a weight w_k that is a monotone function of the very mismatch term it claims to correct for, so the weighting is circular and the gain over uniform iTDM is small or absent. Our method instead uses an externally calibrated residual against a K_ref reference, so w_k tracks truncation error rather than mismatch, and we expect a clear improvement over the self-referential ablation, especially at high CFG.

### What would falsify this idea

If SADW-iTDM's gain over uniform iTDM is uniform across K and CFG scales rather than concentrated at small K and high CFG, then the step-weighted correction is not addressing finite-step discretization bias and the central claim is wrong. Similarly, if the constant-weight ablation matches SADW-iTDM everywhere, or if the diagnostic plot shows no correlation between w_k and measured gradient bias, the weighting mechanism itself is not responsible for any observed improvement.

## References

1. Reinforcing Diffusion Models by Instantaneous Trajectory Distribution Matching
