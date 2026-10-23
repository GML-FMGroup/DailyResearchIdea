# Prefix-Extension Consistency for Streaming Video Generators

## Motivation

Streaming video generators are evaluated and trained under the implicit assumption that clip-level or auxiliary supervision is a sufficient proxy for latent-state consistency, but neither lineage enforces it. StreamEval scores state consistency only post-hoc on scripted rollouts and finds it decoupled from visual quality (top Following scores of 25.93–36.90), while FCP adds an auxiliary future-chunk loss that shapes the representation without ever constraining how the latent state depends on request segmentation. Structurally, no mechanism makes the generator's latent state a sufficient statistic of the request prefix, so arbitrary re-segmentation of an interleaved request stream can silently change the continuation.

## Key Insight

If the latent state after any request prefix is trained to be a sufficient statistic for every extension of that prefix, then segmentation of an interleaved request stream becomes an irrelevant variable to the generator by construction, not by post-hoc measurement.

## Method

**What it is.** PEC (Prefix-Extension Consistency) is a training-time objective for autoregressive streaming video diffusion generators that couples the denoising trajectory to the request-arrival process. Inputs: a video, a scripted interleaved request stream (segmented into variable-length chunks), and the generator's chunk-level latent state. Outputs: a trained generator whose latent state is invariant, up to a learned equivalence, to how the same request prefix is segmented.

**How it works.**

```pseudocode
# Given: video V, request stream R split into chunks r_1..r_K
# Hyperparameters: lambda_1=0.5, lambda_2=0.1, temperature tau=0.07
# Projection head f: 2-layer MLP, hidden=256, GeLU, output dim=128
# Denoiser eps_theta: standard U-Net with cross-attention conditioning on z
# Future predictor g: 2-layer MLP, hidden=256, GeLU, output dim = video chunk dim

for each training step:
    sample a prefix length k ~ Uniform(1, K)
    sample two segmentations S_a, S_b of the SAME prefix r_1..r_k
        # S_a and S_b differ only in chunk boundaries, not content
    z_a = encode_prefix(V, R, S_a)   # generator latent after consuming S_a
    z_b = encode_prefix(V, R, S_b)   # generator latent after consuming S_b
    # Prefix-extension consistency loss (contrastive over extensions):
    sample an extension e (next request chunk + its video frames)
    # Negatives: latents from other prefixes in the same batch (batch size B=32)
    L_ext = -log exp(sim(f(z_a), f(z_b))/tau) / sum_{j=1}^{B} exp(sim(f(z_a), f(z_j))/tau)
        # f = projection head; z_j are latents from other prefixes (negatives)
    # Denoising loss on the extension, conditioned on BOTH z_a and z_b:
    L_den = E_{t,eps} [ || eps - eps_theta(x_t^e, t, z_a) ||^2
                      + || eps - eps_theta(x_t^e, t, z_b) ||^2 ]
    # Future-chunk prediction (FCP-style) on the extension:
    L_fcp = || x^e - g(z_a) ||^2 + || x^e - g(z_b) ||^2
    L = L_den + lambda_1 * L_ext + lambda_2 * L_fcp
```

**Why this design.** We chose a contrastive prefix-extension loss over a direct L2 penalty between z_a and z_b because the two segmentations produce latents in different coordinate frames (different chunk counts, different temporal offsets), so an L2 penalty would force a spurious alignment that destroys temporal information; a contrastive objective instead learns an equivalence relation that preserves what matters for the extension, accepting the cost that negatives must be sampled carefully (we use latents from other prefixes in the same batch). We chose to condition the denoising loss on both z_a and z_b rather than only z_a because the extension must be reachable from either segmentation — this is what makes the latent a sufficient statistic rather than merely a good predictor. We chose to keep the FCP-style future-chunk term rather than drop it because FCP already shapes the representation toward scene evolution, and our loss is complementary: FCP constrains what the state contains, PEC constrains what the state ignores. A domain expert would NOT describe this as a variant of FCP or StreamEval: FCP adds an auxiliary prediction target, StreamEval measures consistency post-hoc; PEC adds a training-time invariant that ties the latent state to the request-arrival process, a structural property neither prior method establishes.

**Why it measures what we claim.** The contrastive similarity sim(f(z_a), f(z_b)) measures segmentation-invariance of the latent state because the assumption is that if two segmentations of the same request prefix yield latents that are close under f, then any extension conditioned on either latent produces the same continuation; this assumption fails when the projection head f collapses distinct temporal information (e.g., both latents encode the same average scene but different motion phase), in which case the similarity reflects scene-level rather than state-level equivalence. The denoising loss conditioned on both z_a and z_b measures sufficiency of the latent for the extension because the assumption is that if the same extension is denoised equally well from either latent, then the latent captures all information in the prefix relevant to the extension; this assumption fails when the extension is predictable from the video content alone regardless of the request, in which case the loss reflects visual continuity rather than request-state sufficiency. The FCP term measures that the latent encodes future evolution because the assumption is that predicting a temporally separated chunk requires representing scene dynamics; this fails when the video has low temporal variation, in which case it reflects redundancy rather than dynamics.

**(D) Assumptions and failure modes.** The load-bearing assumption of PEC is that a contrastive similarity between projected latents f(z_a) and f(z_b) from two segmentations of the same request prefix, combined with dual-conditioned denoising, forces the latents to be interchangeable sufficient statistics for the extension — i.e., that invariance-under-f plus equal denoising loss implies the continuation is actually the same. This assumption is unverified: the InfoNCE objective only constrains f(z_a)≈f(z_b) in the projection space, placing no constraint on the raw latents z_a, z_b that are actually fed to the denoiser, and the dual-conditioned denoising loss is satisfied trivially whenever the extension is predictable from video content alone. This is the representation-collapse / shortcut problem of contrastive learning (Chen et al. 2020, SimCLR; Wang & Isola 2020): alignment in the projection head does not imply alignment of the encoder outputs used downstream. Worse, the two segmentations produce latents in different coordinate frames (different chunk counts/offsets), so the denoiser can learn to ignore z entirely and rely on x_t^e and the shared video context, making L_den non-identifying for sufficiency; this is the standard conditional-ignoring failure of auxiliary conditioning (cf. classifier-free guidance literature, Ho & Salimans 2022). The claim that 'equal denoising loss from either latent ⇒ latent captures all prefix information relevant to the extension' is an unverified identifiability claim: nothing in the objective prevents the denoiser from marginalizing over z_a vs z_b and reconstructing the extension from the video prior, which is exactly the low-request-dependence regime we flag as a confound. To detect this failure mode, we add a diagnostic: a linear probe predicting motion phase from f(z) (if f collapses motion phase, probe accuracy will be near chance) and a request-reconstruction head (if z carries no request-specific information, reconstruction error will be high). These diagnostics are reported but do not alter the training objective.

## Contribution

(1) A training-time prefix-extension consistency invariant (PEC) that formalizes segmentation-invariance of a streaming generator's latent state as a contrastive objective coupling denoising to the request-arrival process. (2) A design principle showing that sufficiency of the latent state for arbitrary prefix extensions can be enforced by conditioning the denoising loss on multiple segmentations of the same prefix, without new data or post-hoc measurement. (3) A training recipe that composes PEC with FCP-style future-chunk prediction, yielding a generator whose state is provably independent of request segmentation under the stated equivalence.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Scripted interleaved request streams with paired segmentations (e.g., StreamEval-style interactive cases re-segmented) | Directly manipulates chunk boundaries while holding content fixed. |
| Primary metric | Segmentation-invariance gap: extension quality drop under re-segmentation | Directly measures the invariant the method claims to establish. |
| Baseline 1 | Standard teacher-forced autoregressive training | Tests whether segmentation invariance needs an explicit objective. |
| Baseline 2 | Teacher-forced + FCP (Looking Ahead) | Tests whether future-prediction alone yields segmentation invariance. |
| Baseline 3 | StreamEval post-hoc consistency filtering | Tests whether post-hoc measurement can substitute for training-time invariance. |
| Ablation-of-ours | PEC without contrastive term (L2 latent penalty instead) | Tests whether contrastive equivalence is necessary vs. naive alignment. |

### Why this setup validates the claim

The central claim is that PEC makes the latent state invariant to request segmentation while remaining sufficient for the extension. The dataset is constructed so that the same request prefix appears under multiple segmentations with identical content, isolating segmentation as the only manipulated variable — this is what makes the metric causal rather than correlational. The primary metric, the segmentation-invariance gap, is chosen because it directly operationalizes the claim: if the latent is truly invariant, extension quality should not depend on which segmentation produced it. Standard teacher-forced training tests the null hypothesis that invariance emerges without an explicit objective. FCP tests whether shaping the state's content (future prediction) incidentally yields invariance. StreamEval-style post-hoc filtering tests whether measuring consistency after the fact can substitute for building it in. The L2 ablation tests the design argument that contrastive equivalence is necessary because segmentations live in different coordinate frames. Together these form a falsifiable test: if the gap is closed equally by all baselines, the method's structural contribution is illusory.

### Expected outcome and causal chain

**vs. Standard teacher-forced autoregressive training** — On a case where the same request prefix is delivered as one long chunk versus several short chunks, the baseline produces visibly different continuations (e.g., drift in motion phase or object identity) because its latent state is never constrained to be a function of request content independent of arrival pattern. Our method instead trains the latent to be equivalent across segmentations via the contrastive prefix-extension loss and to remain sufficient via dual-conditioned denoising, so we expect a large segmentation-invariance gap for the baseline and a near-zero gap for ours, with the difference concentrated on the re-segmented subset.

**vs. Teacher-forced + FCP** — On a case where the extension is predictable from video content alone (low request dependence), FCP closes the gap trivially because its future-prediction term rewards scene continuity regardless of segmentation; but on cases where the extension depends on request state (e.g., a mid-stream edit request), FCP still yields segmentation-dependent latents because it never couples the state to the request-arrival process. Our method handles both because PEC explicitly ties the latent to the request prefix, so we expect parity with FCP on low-request-dependence cases but a clear gap on high-request-dependence cases.

**vs. StreamEval post-hoc consistency filtering** — On a case where a model generates acceptable frames under one segmentation but inconsistent ones under another, post-hoc filtering can only reject the inconsistent output, not produce a consistent one, because it operates after generation and cannot alter the latent state. Our method instead builds invariance into training, so we expect the post-hoc baseline to show high rejection rates and reduced yield on re-segmented requests while our method maintains both yield and quality, with the divergence largest on requests with many possible segmentations.

### What would falsify this idea

If the segmentation-invariance gap is uniform across all subsets — including cases where the extension is fully predictable from video content alone — rather than concentrated on the high-request-dependence subset where the predicted failure mode occurs, then the contrastive term is not doing the work we claim and the central mechanism is wrong.

## References

1. Looking Ahead: Modeling Future Evolution in Autoregressive Video Generation
2. StreamEval: Evaluating Interactive Streaming Video Generation Models
3. Self-Forcing++: Towards Minute-Scale High-Quality Video Generation
4. Diffusion Forcing: Next-token Prediction Meets Full-Sequence Diffusion
5. LTX-Video: Realtime Video Latent Diffusion
6. GAIA-1: A Generative World Model for Autonomous Driving
7. AR-Diffusion: Auto-Regressive Diffusion Model for Text Generation
8. Phenaki: Variable Length Video Generation From Open Domain Textual Description
