# Boundary-Owned Drift Correction via a Shared Reconstructive Sufficient Statistic in Script-to-Video Pipelines

## Motivation

Decomposed streaming script-to-video pipelines—exemplified by the three-module framework in Yan's Foundational Interactive Video Generation—optimize each module against a local objective (3D-VAE reconstruction for the simulator, denoising likelihood for the generator, edit fidelity for the editing module), so no module's loss penalizes divergence of the composed latent state from the simulator's own trajectory. Benchmarks like StreamEval confirm the symptom: models retain visual fidelity yet fail sustained state consistency across incremental requests (e.g., 36.90 'Following' score), because the drift accumulates at the simulator–generator boundary where it is structurally unowned. The root cause is that the boundary carries an unconstrained, unmeasured quantity: the generator samples freely from its conditional distribution without any term that ties its output distribution to the simulator's latent trajectory, so even a perfect simulator and a perfect generator compose into a drifting system.

## Key Insight

If both modules are jointly required to reconstruct the same low-dimensional sufficient statistic of the simulator's latent trajectory, then the composed system's state divergence is bounded by the simulator's own local reconstruction error rather than by the generator's unconstrained sampling variance—a conservation law that no single module can satisfy alone and that is checkable exactly at the boundary.

## Method

**What it is.** We introduce BOSS (Boundary-Owned Sufficient Statistic), a cross-module drift-correction invariant for simulator–generator streaming pipelines. Inputs: the simulator's latent trajectory z_{1:t} (from the 3D-VAE/KV-cache simulator) and the generator's rendered latent trajectory z'_{1:t}. Outputs: a boundary correction applied to the generator's conditioning, plus a scalar invariant residual r_t reported at every frame boundary.

**How it works.**

```pseudocode
# Phase 1: statistic definition (offline, once per scene)
Given simulator latents z_{1:t}, fit a fixed random projection P (d_z -> d_s, d_s=16, seed-fixed)
s_t = P @ pool(z_{t-k+1:t})          # k=8, the shared sufficient statistic target
# s_t is a low-dim linear sufficient statistic of the local latent trajectory

# Phase 2: dual reconstruction heads (trained jointly, frozen at deploy)
head_S: z_{t-k+1:t} -> s_hat_t^S    # simulator-side reconstructor (2-layer MLP, hidden=256, GeLU)
head_G: z'_{t-k+1:t} -> s_hat_t^G   # generator-side reconstructor (2-layer MLP, hidden=256, GeLU)
loss = ||s_hat_t^S - s_t||^2 + ||s_hat_t^G - s_t||^2 + lambda * ||s_hat_t^S - s_hat_t^G||^2   # lambda=1.0

# Phase 3: streaming boundary correction (per frame, online)
for each frame t:
    s_hat_t^G = head_G(z'_{t-k+1:t})
    r_t = ||s_hat_t^G - s_t||            # invariant residual, reported
    if r_t > tau:                        # tau = 2 * sigma_S (simulator local recon std)
        delta = alpha * (s_t - s_hat_t^G)    # alpha=0.5
        z'_{t} <- z'_{t} + P^+ @ delta        # project correction back to latent space
    emit z'_t, r_t
```

**Why this design.** We chose a *fixed random projection* of the pooled latent trajectory over a *learned encoder* because a fixed linear map is invertible in expectation and cannot itself absorb drift, so the invariant residual r_t reflects cross-module divergence rather than the statistic-extractor's own capacity—accepting the cost that the statistic is not task-optimal and may miss nonlinear trajectory structure. We chose a *dual reconstruction loss with an explicit cross-head coupling term* (lambda) over two independent local losses because independent heads can each be individually accurate while disagreeing at the boundary, which is exactly the unowned-drift failure of Yan's decomposed pipeline; the coupling term forces the two heads onto a shared target, accepting that training now requires paired simulator/generator latents. We chose a *hard threshold tau tied to the simulator's own reconstruction std sigma_S* over a learned gate because it makes the correction budget attributable to the simulator's local error (the conservation-law claim) rather than to a learned confidence signal, accepting that sigma_S must be estimated per scene. This differs from a controller/router (Anti-Pattern 4) in that there is no decision about which module to invoke: the correction is a closed-form projection whose magnitude is set by the invariant residual, and the two heads live on the same statistic surface. It also differs from a simple reconstruction autoencoder: the load-bearing object is the *cross-head disagreement*, not either head's accuracy.

**Why this measures what we claim.** The residual r_t = ||s_hat_t^G − s_t|| measures *cross-boundary state divergence* because the assumption that P is an injective linear sufficient statistic of the local trajectory makes equality of s_hat^G and s_t equivalent to equality of the generator and simulator trajectory moments; this assumption fails when the trajectory has strong nonlinear or multimodal structure that the linear projection collapses, in which case r_t reflects only the projected-moment divergence and under-reports drift in collapsed directions. The threshold tau = 2·sigma_S measures *attributable drift budget* because the assumption that the simulator's local reconstruction error is the dominant noise source makes sigma_S an upper bound on acceptable boundary divergence; this assumption fails when the generator introduces high-frequency rendering noise that inflates head_G's variance, in which case tau under-triggers and corrections are suppressed. The correction delta = alpha·(s_t − s_hat_t^G) measures *drift correction* because the assumption that P^+ is a stable left inverse makes back-projection of the statistic residual equivalent to a latent-space correction; this assumption fails when P is ill-conditioned, in which case the correction amplifies noise in near-null directions.

**Assumption A (explicit).** The load-bearing assumption is that the fixed random projection P (d_s=16) is an injective linear sufficient statistic of the simulator's local latent trajectory, so that equality of the projected statistics s_hat^G and s_t is equivalent to equality of the underlying generator and simulator trajectory states. This assumption fails when the trajectory has strong nonlinear or multimodal structure that the linear projection collapses; in that case r_t measures only the 16-dimensional linear shadow of drift and under-reports divergence in the (d_z − 16)-dimensional near-null space of P, and the correction z' ← z' + P^+ delta cannot touch those collapsed directions because P^+ P is a rank-16 projector. We therefore report the singular-value spectrum of P and the fraction of trajectory variance captured by P on a held-out calibration set (see experiment), and we do not claim the conservation law holds in collapsed directions.

## Contribution

["BOSS, a boundary-owned drift-correction invariant that couples the simulator and generator through a shared reconstructive sufficient statistic and exposes a per-frame invariant residual with a closed-form correction projection, requiring no change to either module's internal objective.", "A design principle—cross-head disagreement on a shared sufficient statistic—that converts an unowned cross-module drift into an attributable, thresholdable quantity whose budget is set by the simulator's own local reconstruction error rather than by generator sampling variance.", 'An analysis of when the linear sufficient-statistic assumption under-reports drift (nonlinear/multimodal trajectories) and when the sigma_S-derived threshold under-triggers (generator rendering noise), yielding concrete guidance on statistic dimensionality and threshold calibration for streaming pipelines.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | StreamEval 300-case streaming suite | Incremental requests stress cross-boundary drift directly |
| Primary metric | Cross-boundary state consistency score | Directly measures the divergence BOSS claims to bound |
| Baseline | Yan decomposed pipeline (uncoupled) | Tests whether cross-head coupling is load-bearing |
| Baseline | Streaming model + naive latent L2 clamp | Tests whether statistic-space residual beats pixel-space heuristics |
| Baseline | Streaming model, no correction | Tests whether any correction helps vs. drift accumulation |
| Ablation-of-ours | BOSS with lambda=0 (independent heads) | Isolates the cross-head coupling term's contribution |
| Diagnostic metric | Fraction of trajectory variance captured by P (held-out calibration set) | Tests injectivity/sufficiency assumption A |
| Diagnostic metric | Singular-value spectrum of P | Justifies P^+ stability |

**Paired-latent collection protocol.** For each of the 300 StreamEval cases, we run the frozen simulator and generator to collect paired latents (z_{1:T}, z'_{1:T}) over the full stream length T (mean 120 frames, max 240). We hold out 30 cases (10%) as a calibration set for estimating sigma_S and for computing the variance-captured diagnostic; the remaining 270 cases are used for training the dual heads (Phase 2) and for evaluation. Training uses 8-frame windows with stride 4, yielding approximately 270 × (T−7)/4 ≈ 7,600 windows.

**sigma_S estimation.** sigma_S is estimated per scene as the standard deviation of the simulator's local reconstruction error ||z_t − decode(encode(z_t))|| over the first 16 frames of that scene (a warm-up window not used for correction). We report both per-scene and global (pooled across calibration cases) sigma_S; the main results use per-scene sigma_S, and the global value is reported as a sensitivity check.

**Pilot run.** Before the full 300-case suite, we run a 30-case pilot (drawn from the calibration set) to confirm that the correction fires at a non-degenerate rate: we report the fraction of frames with r_t > tau and require it to lie in [0.05, 0.50] for the pilot to be considered valid; otherwise we adjust alpha (0.5 → 0.25 or 0.75) and re-run the pilot.

### Why this setup validates the claim

The central claim is that the *cross-head disagreement* on a shared linear sufficient statistic is the load-bearing signal for correcting simulator–generator drift at frame boundaries, and that a fixed random projection plus a sigma_S-tied threshold makes the correction budget attributable to the simulator's own local error. StreamEval's incremental multimodal requests are the right stress because each request is a fresh boundary where the generator must re-anchor to the simulator's evolving latent state — exactly the "unowned drift" failure Yan's decomposed pipeline exhibits. The primary metric, cross-boundary state consistency, is the direct observable of r_t: if BOSS's residual is a valid invariant, consistency should track r_t and improve precisely when corrections fire. The three baselines form a ladder: no-correction establishes the drift floor, the naive L2 clamp tests whether the *statistic* (not just any) correction matters, and the uncoupled Yan pipeline tests whether the coupling term (lambda) — not merely the presence of two heads — is what makes the invariant meaningful. The lambda=0 ablation is the sharpest test: it holds architecture and data fixed and removes only the coupling, so any gain attributable to BOSS must vanish or collapse there. The two diagnostic metrics directly test assumption A: if P captures a low fraction of trajectory variance (e.g., < 0.5) or has a highly skewed singular-value spectrum, then r_t is measuring only a 16-dimensional shadow and the conservation-law claim is unsupported; we report these alongside consistency so the reader can judge whether gains are attributable to the statistic or to the rank-16 clamp.

### Expected outcome and causal chain

**vs. Yan decomposed pipeline (uncoupled)** — On a case where the simulator's latent trajectory shifts mid-stream (e.g., a scene cut or object entering frame) and the generator's local reconstruction stays locally accurate but disagrees with the simulator at the boundary, the Yan pipeline produces a visible discontinuity and post-boundary drift because its two modules are optimized independently and nothing forces them onto a shared target. Our method instead projects both trajectories through the same fixed P and drives the coupled heads toward a shared s_t, so the disagreement becomes a measurable residual that triggers a closed-form correction. We expect a noticeable consistency gap on scene-cut and object-entry subsets, with near-parity on static subsets where no boundary drift exists. We also expect the variance-captured diagnostic to be moderate (0.6–0.8) on these subsets, indicating that P captures most but not all of the diverged variance.

**vs. Streaming model + naive latent L2 clamp** — On a case where the drift is concentrated in a low-variance direction of the pooled trajectory (e.g., slow camera pan with subtle parallax), the naive L2 clamp produces over-correction or under-correction because its penalty is isotropic in raw latent space and blind to which moments actually diverged. Our method instead corrects only along the projected statistic residual and back-projects through P^+, so the correction targets the diverged moment directions. We expect the naive clamp to show higher variance in per-frame correction magnitude and worse consistency on slow-motion subsets, while BOSS shows tighter correction magnitude and a consistency gain concentrated exactly where r_t exceeds tau. However, if the diverged moment falls in the near-null space of P (variance-captured < 0.5 on that subset), BOSS may under-correct and show no advantage over the clamp; we report this as a boundary condition on the claim.

**vs. Streaming model, no correction** — On a long stream (many incremental requests), the no-correction baseline produces monotonically accumulating boundary divergence because nothing re-anchors the generator to the simulator's latent state after each request. Our method instead fires corrections whenever r_t exceeds 2·sigma_S, so divergence is bounded by the simulator's local reconstruction error rather than growing with stream length. We expect the consistency gap between BOSS and no-correction to widen with stream length — small on short streams, pronounced on long ones — which is the signature of a drift-bounding mechanism rather than a one-shot quality boost. The pilot run will confirm that the correction fires at a non-degenerate rate (5–50% of frames) before we commit to the full suite.

### What would falsify this idea

If the lambda=0 ablation matches BOSS's consistency gains across all subsets — i.e., the cross-head coupling term contributes nothing — then the central claim that *disagreement on a shared statistic* is load-bearing is wrong. Likewise, if BOSS's gain is uniform across static and boundary-stressed subsets rather than concentrated where r_t exceeds tau, the residual is not measuring the drift it claims to; additionally, if the variance-captured diagnostic is low (< 0.5) on the subsets where BOSS shows gains, the gains are attributable to a rank-16 clamp rather than to a sufficient statistic, falsifying assumption A.

## References

1. Yan: Foundational Interactive Video Generation
2. StreamEval: Evaluating Interactive Streaming Video Generation Models
3. Video-Bench: Human-Aligned Video Generation Benchmark
4. DreamBench++: A Human-Aligned Benchmark for Personalized Image Generation
5. T2V-CompBench: A Comprehensive Benchmark for Compositional Text-to-video Generation
6. VBench: Comprehensive Benchmark Suite for Video Generative Models
7. FETV: A Benchmark for Fine-Grained Evaluation of Open-Domain Text-to-Video Generation
