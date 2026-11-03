# Contractive Latent Dynamics for Retrieval-Augmented Autoregressive Video World Models

## Motivation

Existing retrieval-augmented autoregressive video world models, such as VRAG in 'Learning World Models for Interactive Video Generation', assume that conditioning on past frames, actions, and retrieved content suffices to control compounding error, but they provide no structural invariant on the error recursion; consequently, long-horizon rollouts drift under distribution shift because the latent error can amplify arbitrarily across steps. This meta-level gap recurs across multiple paper-trees (e.g., GameNGen's conditioning augmentations and GAIA-1's token-level autoregression), all of which rely on richer conditioning as an empirical stabilizer rather than a bounded error propagation mechanism.

## Key Insight

A learned, architecturally constrained contraction metric that is certified per-step against counterfactual trajectories sharing the same retrieved context yields a state-dependent error bound whose growth is independent of observable-history length, converting conditioning from an empirical stabilizer into a structural error-control mechanism.

## Method

**What it is:** We introduce CLD-VWM (Contractive Latent Dynamics for Video World Models), a retrieval-augmented autoregressive video world model whose latent transition is certified to be contractive in a learned metric. Inputs: past frames, actions, retrieved video clips, and an explicit global state; outputs: next-frame latents and a per-step contraction certificate that bounds the divergence from any counterfactual latent trajectory that shares the same retrieved context.

**Load-bearing assumption (stated explicitly):** The per-step contraction coefficient κ_t computed from the Jacobian of the latent transition f_θ with respect to z_t, in a metric G(z,c) that is itself a function of z, is a valid local Lipschitz constant whose product bounds the accumulated latent error e_T, i.e., that the error recursion e_{t+1} ≤ κ_t e_t + δ_t holds exactly for counterfactual trajectories sharing the same retrieved context c. This assumption requires three conditions to hold: (i) the metric G(z,c) is uniformly bounded above and below along rollouts, so the metric-change term ||L(z_{t+1},c)^{-1} L(z_t,c)|| is bounded by M/m and can be folded into an effective contraction rate κ̃_t = (M/m)·κ_t < 1; (ii) the reported κ_t is an upper bound on the true spectral norm, not a lower bound from a k=8 sketch; and (iii) the retrieved context c is identical for the factual and counterfactual trajectories, enforced by a retrieval-consistency check at inference. We enforce (i) via a hard normalization on L, (ii) via a certified upper bound using a Hutchinson estimator with an explicit high-probability correction factor, and (iii) via a re-retrieval check that rejects steps where c changes.

**How it works:**

```pseudocode
# Phase 1: Metric-constrained encoder-decoder
# Encoder E maps frames to latent z; decoder D reconstructs frames.
# Metric network M_ψ(z, c) outputs a positive-definite matrix G(z, c) = L(z,c) L(z,c)^T,
# where L is lower-triangular with positive diagonal (Cholesky parameterization).
# Hard normalization: we constrain the diagonal of L to lie in [sqrt(m), sqrt(M)]
# with m=0.5, M=2.0, via a scaled sigmoid activation on the diagonal entries.
# This ensures G(z,c) has eigenvalues in [m, M] for all z, c.
# Retrieved context c = R(past frames, actions) via a frozen retriever.

# Phase 2: Transition model f_θ with contraction constraint
# f_θ(z_t, a_t, c) predicts z_{t+1}.
# We enforce a local contraction condition in the metric G:
#   || J_f(z_t) ||_G = || L(z_t,c)^{-1} J_f(z_t) L(z_t,c) ||_2 ≤ 1 - ε,
# where J_f is the Jacobian of f_θ w.r.t. z_t, and ε > 0 is a margin (ε=0.05).
# This is implemented as a soft penalty during training:
#   L_contr = E[ max(0, || L(z_t,c)^{-1} J_f(z_t) L(z_t,c) ||_2 - (1 - ε))^2 ].
# The Jacobian spectral norm is estimated via a Hutchinson estimator with k=8
# random Gaussian vectors v_i ~ N(0, I), computing J_f v_i via forward-mode autodiff (JVP).
# The estimator is: ||J_f||_2 ≈ sqrt( (1/k) Σ_i ||J_f v_i||_2^2 ).
# To obtain a certified upper bound, we multiply by a correction factor
# C(k, δ) = (1 + sqrt(2 log(1/δ)/k)) with δ=0.01, giving a high-probability upper bound
# on the spectral norm (by matrix concentration inequalities).
# Thus the reported κ_t = C(k, δ) * sqrt( (1/k) Σ_i || L(z_t,c)^{-1} J_f(z_t) L(z_t,c) v_i ||_2^2 ).

# Phase 3: Certificate computation at inference
# For each step t, given current latent z_t, action a_t, retrieved context c:
#   Compute local Lipschitz estimate L_t = || L(z_t,c)^{-1} J_f(z_t) L(z_t,c) ||_2
#   using the Hutchinson estimator with correction factor as above.
#   Certificate: κ_t = L_t. If κ_t ≤ 1 - ε, the step is certified contractive.
#   The per-step bound on latent error e_t = || z_t - z'_t ||_{G(z_t,c)} for any counterfactual z'_t
#   sharing the same c satisfies: e_{t+1} ≤ κ̃_t e_t + δ_t, where κ̃_t = (M/m) * κ_t,
#   and δ_t is the model bias (estimated via a held-out probe set as the maximum
#   reconstruction error in G-metric).
#   Accumulated bound: e_T ≤ (∏_{t=1}^T κ̃_t) e_0 + Σ_{t=1}^T (∏_{s=t+1}^T κ̃_s) δ_t.
#   Since κ̃_t ≤ (M/m)(1 - ε) < 1 (with M/m=4, ε=0.05, so κ̃_t ≤ 3.8, which is >1;
#   to ensure κ̃_t < 1 we require κ_t ≤ (1 - ε) * (m/M) = 0.95 * 0.25 = 0.2375.
#   Thus we set the contraction target to κ_t ≤ 0.2375 to guarantee κ̃_t < 1.
#   The first term decays geometrically and the second is bounded by δ_max / (1 - κ̃_max),
#   independent of T (observable-history length).
#   Retrieval-consistency check: before applying the bound, we re-retrieve using the
#   counterfactual past frames and verify that the retrieved context c' equals c.
#   If c' ≠ c, the step is marked as 'context-inconsistent' and the certificate is invalid;
#   we then fall back to a conservative bound using the maximum possible κ_t over the
#   validation set (κ_max_val) and δ_max, which may be vacuous but is reported.

# Phase 4: Training objective
# L_total = L_recon + λ_1 L_contr + λ_2 L_cert (certificate consistency)
# L_cert = E[ max(0, κ_t - (1 - ε))^2 ] on a held-out probe set to calibrate δ_t.
# Hyperparameters: ε=0.05, λ_1=1.0, λ_2=0.1, sketch size k=8, δ=0.01, m=0.5, M=2.0.
```

**Why this design:** We chose a learned metric network with Cholesky parameterization over a fixed Euclidean metric because the latent space of video world models is highly anisotropic—different directions (e.g., motion vs. appearance) have vastly different sensitivity to compounding error—and a fixed metric would either over-constrain or under-constrain; the cost is that the metric must be trained jointly with the transition model, which we handle by alternating updates every 100 steps. We chose a soft penalty on the Jacobian spectral norm over a hard architectural constraint (e.g., spectral normalization) because hard constraints can excessively restrict the model's capacity to represent complex dynamics; the trade-off is that the soft penalty requires careful tuning of λ_1 and may not enforce contraction everywhere, which we mitigate by monitoring the certificate violation rate on a validation set. We chose a per-step certificate computed via forward-mode autodiff over a low-rank sketch (k=8) rather than full Jacobian computation because full Jacobians are O(d^2) in latent dimension (d=1024), infeasible for real-time video generation; the sketch introduces a small approximation error that we bound empirically. This design is not a variant of standard retrieval-augmented generation (e.g., VRAG) because VRAG retrieves context to condition generation but provides no structural invariant on the error recursion; our method introduces a certified contraction metric that bounds error propagation by construction, a property that retrieval alone cannot produce.

**Why it measures what we claim:** The computational quantity κ_t = || L(z_t,c)^{-1} J_f(z_t) L(z_t,c) ||_2 measures the contraction coefficient of the latent transition in the learned metric G because the assumption that the metric G is a valid Riemannian metric on the latent manifold (i.e., positive-definite and smoothly varying) makes κ_t equivalent to the local Lipschitz constant of f_θ in that metric; this assumption fails when the metric network is poorly trained (e.g., G becomes ill-conditioned), in which case κ_t reflects numerical instability rather than true contraction. The quantity δ_t, estimated as the maximum reconstruction error on a held-out probe set, measures the model bias in the G-metric because the assumption that the probe set is representative of the state distribution encountered during rollouts makes the maximum error a valid upper bound on per-step bias; this assumption fails under distribution shift where the probe set does not cover rare states, in which case δ_t underestimates the true bias. The accumulated bound e_T ≤ (∏ κ̃_t) e_0 + δ_max / (1 - κ̃_max) measures the compounding error bound because the assumption that the counterfactual trajectory shares the same retrieved context c ensures that the error recursion e_{t+1} ≤ κ̃_t e_t + δ_t holds exactly; this assumption fails when the retriever returns different contexts for the factual and counterfactual trajectories (e.g., due to small input perturbations), in which case the bound does not apply and the certificate is invalid—we detect this via the retrieval-consistency check and report the step as context-inconsistent.

## Contribution

['A novel contractive latent dynamics framework (CLD-VWM) for retrieval-augmented autoregressive video world models, comprising a learned Riemannian metric network with Cholesky parameterization and a Jacobian spectral-norm penalty that enforces per-step contraction.', 'A per-step contraction certificate computed via forward-mode autodiff over a low-rank sketch, yielding a state-dependent error bound that is independent of observable-history length under the stated assumptions.', 'An empirical design principle: contraction margin ε and metric conditioning jointly control the trade-off between model expressiveness and error stability, validated on long-horizon interactive video generation benchmarks.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Interactive long-horizon rollouts with action-conditioned clips | Exposes compounding error over many steps |
| Primary metric | Long-horizon latent drift in G-metric | Directly measures the certified error recursion |
| Baseline | Naive autoregressive + extended context | Tests whether contraction beats context length alone |
| Baseline | Retrieval-augmented generation (VRAG) | Tests whether retrieval alone bounds error propagation |
| Baseline | Spectral-normalized transition (hard constraint) | Tests whether soft metric constraint preserves capacity |
| Ablation-of-ours | CLD-VWM with fixed Euclidean metric | Isolates contribution of learned anisotropic metric |
| Ablation-of-ours | CLD-VWM with fixed diagonal metric (learned per-dimension scales) | Isolates off-diagonal anisotropy vs. any learned metric |
| Ablation-of-ours | CLD-VWM without retrieval-consistency check | Tests whether context-invariance enforcement is necessary |

### Why this setup validates the claim

The central claim is that a learned contractive metric on the latent transition bounds error compounding independently of rollout length, and that this bound—not retrieval quality or context length—is what sustains long-horizon spatiotemporal consistency. The naive autoregressive baseline tests the sub-claim that context extension alone cannot stop geometric error growth; VRAG tests the sub-claim that retrieving relevant clips conditions generation but leaves the error recursion structurally unconstrained; the spectral-normalized baseline tests the sub-claim that a hard architectural contraction over-restricts dynamics and sacrifices fidelity, so the soft metric penalty should win on fidelity at comparable drift. The fixed-metric ablation is the decisive control: if contraction in a learned anisotropic metric is doing the work, removing the metric should degrade drift specifically on motion-heavy, high-sensitivity subsets while leaving appearance-dominated subsets near parity. The fixed diagonal metric ablation further isolates whether the off-diagonal anisotropy—not just any learned metric—is necessary for the claimed gains. The ablation without the retrieval-consistency check tests whether enforcing the context-invariance precondition is necessary for the certificate to be valid. Long-horizon latent drift in the G-metric is the right primary metric because it is exactly the quantity the certificate bounds—if the certificate is meaningful, drift should stay flat in horizon length for our method while growing for baselines, a pattern no other metric would reveal as directly.

### Expected outcome and causal chain

**vs. Naive autoregressive + extended context** — On a rollout where the action sequence drives sustained motion (e.g., repeated turning in a scene with fast parallax), the baseline accumulates small per-step latent errors that compound geometrically because nothing in its transition bounds the Jacobian, so frames drift into blur and inconsistent geometry after a few dozen steps. Our method instead applies a certified contraction (κ_t ≤ 0.2375, ensuring κ̃_t < 1) each step, so latent divergence from the counterfactual trajectory is damped rather than amplified, and the accumulated bound saturates at δ_max / (1 - κ̃_max) regardless of horizon. We expect near-parity on short rollouts and a widening gap that grows with horizon length, concentrated on motion-heavy segments.

**vs. Retrieval-augmented generation (VRAG)** — On a case where the retriever returns a plausible but not exactly matching clip (small input perturbation shifts the nearest neighbor), the baseline's conditioning changes discontinuously and its error recursion is unconstrained, so a single mismatched retrieval injects a jump that propagates and compounds. Our method shares the same retrieved context but its contraction certificate bounds the divergence from any counterfactual sharing that context, so a retrieval mismatch enters only as a bounded bias term δ_t rather than an amplifying jump. We expect VRAG to show spiky, retrieval-correlated drift while our drift stays smooth and horizon-flat, with the gap largest on retrieval-mismatch subsets. Additionally, our retrieval-consistency check flags steps where the counterfactual context differs, and we report the fraction of such steps; on those steps the certificate is invalid and we fall back to a conservative bound, which we expect to be rare (<5% of steps) on the evaluation distribution.

**vs. Spectral-normalized transition (hard constraint)** — On a case requiring sharp, high-frequency dynamics (fast appearance changes, rapid camera cuts), the hard spectral constraint caps the transition's expressiveness everywhere, so the baseline under-fits these dynamics and produces over-smoothed, laggy frames even though its drift is bounded. Our method imposes contraction softly in a learned metric that can stretch capacity along dynamics-sensitive directions, so it retains fidelity on sharp transitions while still certifying contraction. We expect the hard-constraint baseline to match us on drift but lose clearly on reconstruction fidelity on high-frequency subsets, while our method holds both.

**vs. CLD-VWM with fixed Euclidean metric** — On motion-heavy segments where the latent sensitivity to error is highly anisotropic, the fixed Euclidean metric cannot adapt, so the contraction penalty either over-constrains motion directions or under-constrains appearance directions, leading to either blur or drift. Our learned metric stretches capacity along dynamics-sensitive directions, so we expect the fixed-metric ablation to show higher drift on motion-heavy subsets and near-parity on appearance-dominated subsets.

**vs. CLD-VWM with fixed diagonal metric** — On segments where error propagation couples different latent dimensions (e.g., motion and appearance interact), a diagonal metric cannot capture the off-diagonal correlations, so the contraction certificate is looser and drift may grow. We expect the diagonal-metric ablation to perform between the Euclidean and full learned metric, with the full metric showing the lowest drift on coupled-dynamics subsets.

**vs. CLD-VWM without retrieval-consistency check** — On retrieval-mismatch subsets, the baseline without the check may apply the certificate even when the counterfactual context differs, leading to invalid bounds and potentially larger drift. We expect the full method to show lower drift on those subsets and to report a higher fraction of context-inconsistent steps, validating the necessity of the check.

### What would falsify this idea

If long-horizon drift for our method grows with horizon length at the same rate as the naive autoregressive baseline—i.e., the certificate's predicted geometric decay does not appear as horizon-flat drift—then the contraction claim is false. Likewise, if the fixed-metric ablation shows no degradation concentrated on motion-heavy subsets relative to appearance subsets, the learned anisotropic metric is not doing the causal work we claim.

## References

1. Learning World Models for Interactive Video Generation
2. Diffusion Models Are Real-Time Game Engines
3. SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis
4. GAIA-1: A Generative World Model for Autonomous Driving
5. MAGVIT: Masked Generative Video Transformer
6. Phenaki: Variable Length Video Generation From Open Domain Textual Description
7. StreamEval: Evaluating Interactive Streaming Video Generation Models
