# Monotone Routing Regularization with Per-Sample Return Calibration for Looped Mixture-of-Experts Diffusion Transformers

## Motivation

Dynamic-compute diffusion transformers such as DyLoMo-DiT and DyDiT allocate loop passes, expert width, or spatial tokens by heuristics (e.g., more passes at low noise) without any guarantee that extra routing compute yields a quality return; DyLoMo-DiT's abstract reports only aggregate FID gains at matched budgets, leaving the per-sample capacity–compute relationship unverified, and DyDiT similarly tunes width/token sparsity by timestep and region with no monotonicity constraint. Structurally, this is because the router is optimized jointly with the denoising objective as an unconstrained free variable, so the loss surface admits routing decisions that add FLOPs while worsening the denoising loss on some samples — the shared residual gap across both branches is the absence of an invariant tying routing to quality.

## Key Insight

If the router is trained under a pairwise ranking constraint that any higher-compute routing configuration must achieve no worse denoising loss than any lower-compute configuration on the same sample, then the compute–quality curve becomes monotone by construction, converting routing from an unconstrained predictor into a certified quality-return signal.

## Method

**A. What it is.** We introduce MORC (Monotone Routing with Calibrated return), a fine-tuning procedure for DyLoMo-DiT-style diffusion transformers that adds (i) a pairwise ranking loss over nested compute levels per sample and (ii) a per-sample calibration head that predicts the denoising-loss return of a routing configuration. Inputs: a pretrained DiT with weight-shared loops and sparse MoE layers, a batch of (x_t, t, c), and a discrete compute ladder. Outputs: updated router weights, updated calibration head, and a per-sample monotone compute–quality profile. **Load-bearing assumption (stated explicitly):** the per-sample denoising MSE difference between a higher-compute and lower-compute routing configuration, loss_hi_i - loss_lo_i, is a low-variance, causally meaningful signal of that sample's quality return, so a pairwise ranking loss with an adaptive margin and a calibration head regressing this quantity can be trained to certify non-negative returns at inference. This assumption is tested directly in the experiment (per-sample MSE-return vs. decoded-image-quality correlation) and is the target of the variance-reduction and confidence-bound machinery below.

**B. How it works.**
```pseudocode
# Compute ladder: L = [1,2,3,4] loop passes; expert width W = [0.5, 0.75, 1.0]
# Each level ell = (loops, width) with strictly increasing total FLOPs.
# Main experiment uses a reduced two-level ladder ell_lo=(1,1.0), ell_hi=(4,1.0);
# the full 4x3 ladder is an appendix ablation.
# Pretrained DyLoMo-DiT gives router R_theta and denoiser D_phi.
# Calibration head C_psi: 2-layer MLP, hidden=256, GeLU, input = [x_t pooled to 64-d, t embedding 64-d, c embedding 64-d, ell one-hot 12-d], output = scalar.
# Variance-reduced return estimator: K=4 independent eps draws per (x_t,t,c).
# Per-timestep SNR weight w(t) = alpha_bar_t / (1 - alpha_bar_t) (Nichol & Dhariwal 2021).

for each fine-tuning step:
    sample batch {(x_t, t, c)} ~ data; draw levels ell_lo < ell_hi from ladder
    for each sample i:
        for k in 1..K:
            eps_k ~ N(0,I)
            eps_lo_k = D_phi(x_t^i, t^i, c^i; R_theta(x_t^i,t^i,c^i)=ell_lo)
            eps_hi_k = D_phi(x_t^i, t^i, c^i; R_theta(...)=ell_hi)
            loss_lo_ik = w(t^i) * ||eps_lo_k - eps_k||^2
            loss_hi_ik = w(t^i) * ||eps_hi_k - eps_k||^2
        loss_lo_i = mean_k loss_lo_ik ; loss_hi_i = mean_k loss_hi_ik
        # per-sample standard error of the return estimator
        se_i = sqrt( var_k(loss_lo_ik - loss_hi_ik) / K )
    # (1) denoising objective at the sampled levels (teacher-free)
    L_den = mean_i (loss_lo_i + loss_hi_i)
    # (2) pairwise monotone ranking loss with adaptive margin m_i = 2 * se_i
    L_rank = mean_i max(0, loss_hi_i - loss_lo_i + m_i)
    # (3) per-sample calibration head regresses realized loss return
    return_i = loss_lo_i - loss_hi_i            # realized quality return of ell_hi over ell_lo
    L_cal  = mean_i ( C_psi(x_t^i,t^i,c^i,ell_hi) - C_psi(x_t^i,t^i,c^i,ell_lo) - return_i )^2
    # (4) router is updated ONLY through L_rank + lambda * L_cal (stop-grad on L_den for router)
    total = L_den + alpha * L_rank + beta * L_cal     # alpha=1.0, beta=0.5
    update R_theta via grad(total) with stop-gradient on D_phi in L_rank, L_cal
    update D_phi via grad(L_den); update C_psi via grad(L_cal)

# Inference: router selects smallest ell whose calibrated lower confidence bound
# C_psi(...) - se_hat(...) >= tau, where se_hat is the calibration head's predicted
# standard error (a second scalar output head trained on se_i), and tau=0.0.
# Extra compute is spent only where the return is certified above the noise floor.
```

**C. Why this design.** We chose a pairwise ranking loss over a pointwise penalty on loss_hi because monotonicity is a relational property between compute levels on the same sample, and a pointwise penalty on loss_hi alone would push the denoiser toward the low-compute solution rather than enforcing ordering, accepting the cost that each step needs two forward passes through the denoiser. We chose a discrete compute ladder (four loop counts × three widths) over a continuous relaxation because DyLoMo-DiT's routing is already discrete (loop pass count and expert activation), and continuous relaxation would introduce a straight-through estimator that decouples the certified ordering from the deployed routing, accepting the cost of coarser control. We chose a per-sample calibration head over a global scalar because the return of extra compute is sample-dependent (low-noise vs high-noise, easy vs hard images), and a global scalar cannot express the per-sample variation that the router needs, accepting the cost of an extra small network and its own regression loss. We stop gradients from L_rank and L_cal into the denoiser D_phi so that the ranking signal cannot be satisfied by degrading the high-compute branch's denoiser — a failure mode that would make the monotonicity vacuous; the trade-off is a two-timescale optimization that requires more fine-tuning steps. We use K=4 independent eps draws and per-timestep SNR weighting w(t) to reduce the variance of the realized-return estimator, because a single-draw difference of squared errors is dominated by irreducible posterior variance that grows with t; the adaptive margin m_i = 2*se_i and the inference lower confidence bound C_psi - se_hat are the concrete calibration that makes the certified-return claim testable rather than vacuous. A domain expert would not describe MORC as a variant of DyDiT or DyLoMo-DiT: those methods tune the router with the denoising objective only, whereas MORC adds an explicit ordering constraint and a return predictor that make the router's compute decisions verifiable per sample, which is not present in either prior method.

**D. Why it measures what we claim.** The quantity `loss_hi_i - loss_lo_i` (realized return, computed with K=4 draws and SNR weighting) measures the quality return of extra compute on sample i because we assume the denoising MSE at a fixed (x_t, t, c) is a sufficient statistic for sample-level generation quality at that timestep; this assumption fails when the MSE is dominated by irreducible noise at high t, in which case the realized return reflects noise-level variance rather than a genuine quality difference — we test this directly by correlating per-sample MSE return with decoded-image quality proxies (per-sample FID/CLIP) stratified by timestep. The quantity `C_psi(x_t, t, c, ell_hi) - C_psi(x_t, t, c, ell_lo)` measures the router's expected quality return because we assume the calibration head is trained on the same distribution of (x_t, t, c) as inference and that its residual error is small relative to the return magnitude; this assumption fails when the fine-tuning data under-covers a noise level or class, in which case the head's prediction reflects extrapolation error rather than a certified return. The adaptive margin `m_i = 2*se_i` measures the tolerance for noise in the realized return because we assume differences smaller than the per-sample standard error are not meaningful for generation quality; this assumption fails when the denoiser's per-sample loss variance exceeds the between-level effect, in which case the ranking loss can be satisfied by noise rather than by a genuine ordering — the lower confidence bound at inference (C_psi - se_hat >= tau) is the concrete guard against this failure.

## Contribution

['A fine-tuning framework, MORC, that adds a pairwise monotone ranking loss over a discrete compute ladder and a per-sample calibration head to the router of a looped mixture-of-experts diffusion transformer, with stop-gradient separation between the ranking signal and the denoiser.', 'A concrete design principle: routing compute decisions become certifiable per sample via a calibration head whose predicted return is used at inference as a gate (tau threshold), replacing heuristic timestep-based pass allocation with a learned, sample-conditional return estimate.', 'An empirical protocol for measuring capacity–compute monotonicity per sample (realized return distribution, calibration error of the return head, and the fraction of samples where extra compute is certified non-negative) that can be applied to any dynamic-compute DiT.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | ImageNet 256×256 class-conditional | Matches DyLoMo-DiT baseline regime and FID comparison |
| Primary metric | FID at matched average FLOPs | Directly measures compute–quality tradeoff the router controls |
| Baseline | DyLoMo-DiT (denoising-only router) | Tests whether explicit ordering beats implicit routing |
| Baseline | DiT-XL/2 fixed compute | Tests whether adaptive routing beats static allocation |
| Baseline | DyDiT (timestep/space dynamic) | Tests whether per-sample return certification beats heuristic dynamism |
| Baseline | Cost-aware MoE router (load-balancing + FLOP penalty) | Tests whether ranking+calibration is more than a compute penalty |
| Ablation-of-ours | MORC minus calibration head (rank-only) | Isolates contribution of return prediction to routing |
| Ablation-of-ours | MORC with fixed margin m=0.01 and tau=0.0 (no variance reduction) | Tests whether variance reduction and confidence bound are necessary |
| Diagnostic | Per-sample MSE-return vs. decoded-image-quality correlation | Directly tests the load-bearing sufficiency assumption |

### Why this setup validates the claim

The central claim is that a router trained with an explicit pairwise monotone ranking constraint plus a per-sample return predictor will allocate compute more effectively than routers trained only on the denoising objective. ImageNet 256×256 is the right testbed because DyLoMo-DiT already establishes FID at matched parameter and compute budgets there, giving a directly comparable reference point for the exact architecture MORC fine-tunes. FID at matched average FLOPs is the correct primary metric because it forces the comparison onto the compute–quality frontier rather than raw quality: a router that spends more compute on average would trivially lower FID, so only matched-FLOP comparison isolates routing quality. The DyLoMo-DiT baseline tests the sub-claim that explicit ordering constraints improve over implicit denoising-only routing on the same backbone. The DiT-XL/2 baseline tests whether adaptivity itself matters at all. The DyDiT baseline tests whether MORC's per-sample certification beats DyDiT's heuristic timestep/space adjustment, which lacks a return predictor. The cost-aware MoE router baseline (load-balancing loss plus an explicit FLOP penalty on the router) tests whether the ranking+calibration mechanism is merely a reparameterization of existing compute-penalty routing. The rank-only ablation is the critical control: if the calibration head contributes nothing, rank-only should match full MORC, and any observed gain would be attributable to the ranking loss alone rather than the certified-return mechanism we claim is essential. The fixed-margin/no-variance-reduction ablation tests whether the variance-reduction and confidence-bound machinery is necessary for the certified-return claim to hold. The per-sample correlation diagnostic directly tests the load-bearing sufficiency assumption: for a held-out set of 512 samples, we compute the per-sample MSE return (loss_lo - loss_hi) and correlate it with decoded-image quality proxies (per-sample FID and CLIP score against the class label), stratified by timestep quartile, and report Pearson and Spearman coefficients with bootstrap 95% confidence intervals; a near-zero correlation at high t would confirm the failure mode we guard against and justify the SNR weighting and confidence bound.

### Expected outcome and causal chain

**vs. DyLoMo-DiT (denoising-only router)** — On samples where the optimal compute level is ambiguous — mid-noise timesteps on hard classes — the baseline router, trained only on denoising loss, drifts toward whichever level happens to reduce average loss, producing near-uniform compute allocation that under-serves hard samples and over-serves easy ones. Our method instead enforces a per-sample ordering via the ranking loss and certifies returns via the calibration head, so the router can distinguish samples where extra loops genuinely help. We expect a noticeable FID gap concentrated on high-difficulty class subsets at matched average FLOPs, with parity on easy subsets.

**vs. DiT-XL/2 fixed compute** — On samples that are intrinsically easy (low-noise, simple texture), the fixed-compute baseline wastes FLOPs on all samples uniformly, so at any matched average budget it must either under-serve hard samples or over-spend on easy ones. Our method instead routes down on easy samples where the calibration head predicts near-zero return and up on hard ones, reallocating the same budget. We expect MORC to beat fixed-compute DiT-XL/2 at equal average FLOPs, with the gain driven by the hard-sample tail rather than uniform improvement.

**vs. DyDiT (timestep/space dynamic)** — On samples where the difficulty is driven by content rather than timestep or spatial location — a hard class at a moderate noise level — DyDiT's timestep-wise width and spatial token pruning cannot express that the whole sample needs more compute, so it prunes based on location heuristics that miss content-driven difficulty. Our per-sample calibration head conditions on (x_t, t, c) jointly, so it can certify extra compute for content-hard samples regardless of spatial pattern. We expect a gap on content-hard subsets where DyDiT's heuristics mis-allocate, and rough parity where difficulty aligns with timestep or spatial sparsity.

**vs. Cost-aware MoE router (load-balancing + FLOP penalty)** — On samples where the FLOP penalty is nearly flat across levels (mid-noise, ambiguous difficulty), the cost-aware router allocates compute to satisfy the penalty rather than to certify a quality return, so it can spend extra FLOPs on samples where the return is negative. Our method instead uses the pairwise ranking loss and the calibrated lower confidence bound to certify non-negative return before spending compute. We expect MORC to outperform the cost-aware router at matched average FLOPs, with the gap concentrated on samples where the calibration head predicts a return near the noise floor (i.e., where the FLOP penalty alone is uninformative).

### What would falsify this idea

If MORC's FID gain over DyLoMo-DiT at matched average FLOPs is uniform across easy and hard subsets rather than concentrated where the calibration head predicts non-trivial returns, then the ranking and calibration signals are not actually driving per-sample compute decisions and the central claim fails. Equally fatal: if the rank-only ablation matches full MORC, the calibration head adds nothing and the certified-return mechanism is not doing the work we attribute to it; and if the per-sample MSE-return vs. decoded-image-quality correlation is near zero at high t even after SNR weighting and K=4 averaging, the load-bearing sufficiency assumption is false and the certified-return claim is vacuous.

## References

1. DyLoMo-DiT: Dynamic Looped Mixture-of-Experts Diffusion Transformers
