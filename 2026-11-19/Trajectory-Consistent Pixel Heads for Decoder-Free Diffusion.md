# Trajectory-Consistent Pixel Heads for Decoder-Free Diffusion Transformers

## Motivation

Existing decoder-free acceleration work such as Decoding Without a Decoder (DWD) validates its linear pixel head only at the final denoising step, leaving the head unconstrained along the sampling trajectory; because the denoising head is anchored and the DWD head is trained only on terminal latents, the pixel-prediction mapping is never required to be consistent with the latent states actually visited during sampling. This is a structural gap, not a tuning issue: the separation of denoising and pixel-prediction roles is exercised at exactly one timestep, so any error the head makes on intermediate latents is invisible to training and can compound silently across the trajectory.

## Key Insight

A pixel head trained at every denoising step against both ground-truth pixels and the frozen decoder's output on the same latent becomes a trajectory-wide consistent readout of the anchored latent path, so its terminal prediction error is bounded by its agreement with the pretrained decoder rather than by final-step-only fitting.

## Method

**What it is.** We introduce TCPH (Trajectory-Consistent Pixel Head), a training scheme for the DWD linear pixel head that supervises it at every denoising step t ∈ {1..T} using (i) a pixel reconstruction loss against the clean target and (ii) a consistency regularizer against the frozen pretrained decoder applied to the same latent. The anchored denoising head and the frozen decoder are both held fixed; only the DWD head and the last K transformer blocks are updated. **Load-bearing assumption (stated explicitly):** the frozen pretrained decoder D is a faithful, low-bias pixel reference for every latent z_t visited along the anchored sampling trajectory, so that supervising h_phi toward D(x_t) at all timesteps yields a readout whose terminal error is bounded by decoder agreement rather than by final-step-only fitting. This assumption is load-bearing and is tested empirically in the per-timestep diagnostic described below; it is expected to weaken at high noise where D's output is prior-dominated.

**How it works.**

```pseudocode
Input: pretrained diffusion transformer eps_theta (frozen), frozen decoder D,
       DWD linear head h_phi (trainable, single linear layer, output dim = 3*H*W),
       last K=2 transformer blocks (trainable),
       clean image x0, weights lambda_rec=1.0, lambda_con=0.5, K=2, T=50

# Precompute-and-cache anchored trajectories (feasibility spec):
#   for a fixed set of N_cache=4096 (z_T, t) pairs, run AnchoredSample once
#   with no grad and store (z_t, x_t_denoised) on disk. Each training
#   iteration loads a cached pair instead of re-running the sampler.

for each training step:
    (z_t, x_t_denoised) ~ CachedTrajectory   # t ~ Uniform{1..T} at cache time
    # DWD head reads the intermediate latent z_t (the latent actually visited)
    x_hat_t = h_phi(z_t)
    # frozen decoder reference is applied to the DENOISED latent x_t_denoised,
    # not to the raw noisy z_t: D was trained on the clean-end latent
    # distribution, so x_t_denoised is the closest on-support input to D.
    x_ref_t = D(x_t_denoised)
    # losses
    L_rec = || x_hat_t - x0 ||_2^2
    L_con = || x_hat_t - x_ref_t ||_2^2
    L = lambda_rec * L_rec + lambda_con * L_con
    update h_phi and last K blocks by gradient descent on L

# Per-timestep diagnostic (empirical test of the load-bearing assumption):
#   after training, for each t in {1..T}, log mean ||h_phi(z_t) - D(x_t_denoised)||
#   and mean ||h_phi(z_t) - x0|| on a held-out set of 512 cached trajectories.
```

At inference, sampling proceeds exactly as in DWD: the anchored denoising head drives all T steps, and only at the final step is h_phi applied to z_1 to produce pixels — no decoder is invoked.

**Why this design.** We chose to supervise the DWD head at every timestep rather than only the final step (as in DWD) because the anchored denoising head fixes the latent trajectory, so intermediate latents are a free, already-computed training signal; the cost is that each training iteration must run a partial forward pass of the anchored sampler, which we accept because it requires no gradients through eps_theta and because we cache N_cache=4096 anchored trajectories to amortize this cost. We chose a consistency regularizer against the frozen decoder rather than a purely pixel-space loss because the decoder defines the reference pixel geometry the latent space was trained for, giving a dense target even at high-noise timesteps where x0 reconstruction is ill-posed; the cost is that the head is pulled toward decoder outputs that may themselves be imperfect, which we accept because the decoder is the strongest available pixel reference and is only used during training. We chose to keep the denoising head frozen and finetune only the last K=2 blocks rather than joint optimization, because joint training would perturb the sample trajectory and destroy the inference-cost benefit that motivates DWD; the cost is reduced capacity for the head, which we accept because the head is linear and the last blocks already carry high-level semantic features. A domain expert would not describe this as 'DWD plus an extra loss': DWD trains a terminal readout, whereas TCPH turns the head into a trajectory-consistent estimator whose error is anchored to the frozen decoder at every step, a property no terminal-only scheme can produce.

**Why it measures what we claim.** The consistency loss ||h_phi(z_t) − D(x_t_denoised)||^2 measures trajectory-wide pixel-readout consistency because we assume the frozen decoder D is a faithful pixel reference for latents on the anchored trajectory; this assumption fails when the anchored sampler visits off-manifold latents (e.g., at very high noise where D's output is dominated by prior rather than signal), in which case the regularizer reflects decoder bias rather than true pixel error. The reconstruction loss ||h_phi(z_t) − x0||^2 measures per-step pixel accuracy because we assume x0 is the unique clean target for every z_t on the trajectory; this assumption fails when the trajectory is stochastic and multiple x0 are compatible with z_t, in which case the loss reflects an average over plausible targets rather than a single ground truth. The final-step application of h_phi measures decoder-free inference cost because we assume h_phi is a linear map whose cost is negligible relative to D; this assumption fails if h_phi is widened beyond a single linear layer, in which case the cost saving shrinks accordingly. The per-timestep diagnostic curves of ||h_phi(z_t) − D(x_t_denoised)|| and ||h_phi(z_t) − x0|| against t measure whether the load-bearing assumption holds across the trajectory because we assume these two curves track each other where D is faithful; this assumption fails at high t where the curves diverge, in which case the diagnostic localizes the timesteps at which the consistency target is decoder bias rather than pixel truth.

## Contribution

['A trajectory-consistent training objective (TCPH) for decoder-free pixel heads that supervises the DWD linear head at every denoising step using paired intermediate latents and frozen-decoder references, without modifying the anchored denoising head.', "A design principle: anchoring the pixel head's error to a frozen pretrained decoder at all timesteps converts a terminal readout into a trajectory-wide consistent estimator, which we operationalize as a lightweight consistency regularizer that requires no decoder at inference.", 'A reproducible training recipe (partial anchored sampling, last-K block finetuning, dual reconstruction-plus-consistency loss) for extending DWD-style decoder removal to latent image and video generators.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | Text-to-image (SD-2 or SDXL) as primary; video (Wan-class) as secondary confirmation | Matches DWD's evaluated latent generators; primary scoped to one image model per feasibility |
| Primary metric | Pixel fidelity (LPIPS / warping error) vs. decoder baseline, reported at step counts 10/20/50 | Directly measures readout quality claim and cost-quality frontier |
| Secondary metric | Wall-clock time and MACs per generated sample at 10/20/50 steps | Tests whether TCPH shifts the Pareto frontier, not just a single LPIPS number |
| Diagnostic metric | Per-timestep curves of mean ||h_phi(z_t) − D(x_t_denoised)|| and mean ||h_phi(z_t) − x0|| on 512 held-out cached trajectories | Empirically tests the load-bearing assumption that D is faithful across the trajectory |
| Baseline | DWD (terminal-only head) | Tests trajectory-wide vs. final-step supervision |
| Baseline | Frozen-decoder latent pipeline (no DWD) | Tests decoder-free parity claim |
| Baseline | Pixel-space diffusion (no decoder) | Tests head quality without latent geometry |
| Ablation-of-ours | TCPH without consistency regularizer (rec-only) | Isolates decoder-anchoring contribution |
| Ablation-of-ours | TCPH with consistency target D(z_t) instead of D(x_t_denoised) | Tests the latent-reference choice in the pseudocode |

### Why this setup validates the claim

The central claim is that supervising the DWD head across the whole anchored trajectory, anchored to the frozen decoder, yields a better pixel readout than a terminal-only head at the same decoder-free inference cost. DWD is the correct primary baseline because it isolates exactly the supervision-scope variable: same architecture, same inference path, differing only in whether the head sees intermediate latents. The frozen-decoder pipeline tests whether TCPH actually recovers decoder quality without invoking the decoder at inference, which is the cost claim. The pixel-space diffusion baseline tests whether the gain comes from latent geometry rather than from the extra training signal. The rec-only ablation is essential: if TCPH's advantage survives removing the consistency term, then the decoder-anchoring story is wrong and the effect is just more supervision. The D(z_t)-vs-D(x_t_denoised) ablation pins down the pseudocode's latent-reference ambiguity. LPIPS and warping error are the right metrics because they detect the specific failure mode DWD reports (detail distortion, flickering) rather than coarse semantic similarity, which would be blind to the readout-level differences we predict. The per-timestep diagnostic curves are the direct empirical test of the load-bearing assumption: if ||h_phi(z_t) − D(x_t_denoised)|| and ||h_phi(z_t) − x0|| track each other across t, D is a faithful reference; if they diverge at high t, the diagnostic localizes where the consistency target is decoder bias.

### Expected outcome and causal chain

**vs. DWD (terminal-only head)** — On a case where high-frequency texture must be reconstructed (e.g., fine hair or foliage in a generated frame), DWD produces detail distortion and flickering because its head is trained only on the final latent and never learns the intermediate-latent-to-pixel mapping, so it must extrapolate a readout it was never calibrated for. Our method instead trains the head on intermediate latents drawn from the same anchored trajectory, so it learns the full latent-to-pixel geometry, and we expect a concentrated LPIPS gap on high-frequency subsets with near-parity on smooth regions. The per-timestep diagnostic should show TCPH's ||h_phi(z_t) − x0|| curve staying closer to the D(x_t_denoised) reference curve than DWD's at intermediate t.

**vs. Frozen-decoder latent pipeline** — On a case where the decoder introduces its own artifacts (e.g., over-smoothed faces or color drift at low step counts), the baseline produces decoder-inherited distortions because its pixel output passes through D's learned prior. Our method instead learns a readout anchored to D during training but bypasses D at inference, so it can inherit D's geometry without inheriting its inference-time artifacts, and we expect parity on average fidelity but a visible advantage on frames where decoder artifacts are strongest.

**vs. Pixel-space diffusion (no decoder)** — On a case where the model must produce texture detail from a latent-free denoising path, the baseline produces weaker texture because it lacks the latent space's structured geometry that D was trained against. Our method instead leverages the frozen decoder as a dense pixel reference during training, giving the head a well-conditioned target even at high noise, so we expect a noticeable texture-detail gain over the pixel-space baseline while both remain decoder-free at inference.

**vs. TCPH rec-only ablation** — On a case where the decoder-anchoring term is doing the work, the rec-only variant should underperform full TCPH on high-frequency subsets because it lacks the dense pixel-geometry target that D provides at intermediate t. If the rec-only variant matches full TCPH across all subsets, the decoder-anchoring mechanism is not doing the work and the claim collapses to 'more supervision helps.'

**vs. TCPH with D(z_t) target** — On a case where the raw noisy latent z_t is off D's training support, the D(z_t) variant should underperform the D(x_t_denoised) variant because D was trained on clean-end latents, so D(x_t_denoised) is the on-support reference. This ablation directly tests the pseudocode's latent-reference choice.

### What would falsify this idea

If the rec-only ablation matches full TCPH across all subsets, the decoder-anchoring mechanism is not doing the work and the claim collapses to 'more supervision helps.' Likewise, if TCPH's gain over DWD is uniform across smooth and high-frequency regions rather than concentrated where terminal-only readout should fail, the trajectory-consistency story is not supported. If the per-timestep diagnostic shows ||h_phi(z_t) − D(x_t_denoised)|| diverging from ||h_phi(z_t) − x0|| at high t, the load-bearing assumption that D is a faithful reference across the trajectory is falsified at those timesteps.

## References

1. Decoding Without a Decoder: Direct Pixel Prediction from Diffusion Transformers
