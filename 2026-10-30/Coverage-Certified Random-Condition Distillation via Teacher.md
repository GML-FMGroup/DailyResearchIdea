# Coverage-Certified Random-Condition Distillation via Teacher Score-Field Coverage Functionals

## Motivation

Random-condition distillation (Random Conditioning with Distillation for Data-Efficient Diffusion Model Compression) transfers a conditional teacher's knowledge by pairing noised images with randomly sampled text conditions, but the objective it optimizes — a denoising/score-matching loss averaged over sampled conditions — is a mean over a finite random draw and therefore provides no certificate that every region of the teacher's concept space receives non-negligible student support. The same structural gap recurs across the data-free distillation tree (DKDM) and distribution-matching tree (DMD/DMD2, One-Step DMD): all branches assume the teacher's conditional knowledge transfers once the average loss is small, so random pairing can silently leave concept regions unsupervised, and because no quantity computable from the teacher alone measures per-region coverage, the failure is invisible until generation on rare concepts. The root cause is that the distillation loss is a scalar expectation over conditions with no invariant tying it to the geometry of the condition space.

## Key Insight

A teacher's own conditional score field induces a per-condition coverage functional — the teacher-conditional Fisher divergence integrated against the teacher's marginal — whose level sets certify where the student has support, so coverage becomes a computable invariant that can be coupled to the distillation loss rather than an unobservable property of random sampling.

## Method

**What it is.** Coverage-Certified Distillation (CCD) augments random-condition distillation with a teacher-only coverage functional `C(θ)` over the condition space and a dual-form objective that couples the standard denoising loss with `C(θ)`, producing per-condition weights and a certificate that the student's conditional output distribution has non-negligible support on every teacher concept region. Inputs: pretrained conditional teacher `ε_T(x_t, t, c)`, student `ε_S(x_t, t, c; θ)`, noised images `x_t` (no captions needed), and a condition sampler over `c`. Outputs: student parameters `θ` and a coverage certificate `{r_k, C_k}` over a partition of the condition space.

**Load-bearing assumption (stated explicitly).** CCD's certificate is sound only if the per-region teacher-conditional Fisher divergence `D_k` computed over noised latents `x_t ~ q_t` is a faithful, monotone proxy for per-region student support across the *full* reverse trajectory, not merely on the high-noise latents where `q_t` concentrates mass. Because denoising score-matching weights each noise level by `sigma_t^2` (Vincent 2011), a single `q_t`-averaged `D_k` is dominated by high-noise latents and can be near zero while the student collapses modes on a region at low noise. CCD therefore does *not* claim full-trajectory coverage from a single-band `D_k`; instead it computes `D_k` as a noise-schedule-stratified quantity (defined in Phase 1 below) and the certificate is explicitly a statement about support across the stratified bands, with the low-noise band carrying the mode-proportion signal. The calibration study in the experiment section quantifies the correlation between high-noise and low-noise `D_k` and, if that correlation is weak, restricts the certificate claim to the bands actually measured.

**How it works.**

```pseudocode
# Phase 0: teacher-only concept partition (no real images, no captions)
Sample M = 2000 text conditions c_1..c_M from the random-condition sampler
For each c_m, draw N = 16 noised latents x_t ~ q_t (pure noise schedule)
Compute teacher conditional score s_T(x_t,t,c_m) = -eps_T(x_t,t,c_m)/sigma_t
Cluster {c_m} in the space of teacher score-field statistics
    (mean and covariance of s_T over the N latents) into K regions r_1..r_K
    via k-means with K = ceil(sqrt(M)) = 45; store region centroids mu_k

# Phase 1: coverage functional from teacher alone (noise-schedule-stratified)
# Noise bands: J = 4 log-spaced bands over sigma_t in [sigma_min, sigma_max],
#   e.g. sigma in {0.02, 0.1, 0.5, 5.0} (EDM-style), each band b has weight pi_b = 1/J.
def coverage_functional(theta):
    for each region k:
        for each noise band b in 1..J:
            sample B = 32 conditions c ~ r_k ; sample B latents x_t ~ q_t restricted to band b
            # teacher-conditional Fisher divergence in band b, student vs teacher
            D_kb = mean_{b} [ || s_S(x_t,t,c;theta) - s_T(x_t,t,c) ||^2 ]
            # per-band normalization by running median over regions
            Dhat_kb = D_kb / (median_k D_kb + eps)
        # stratified region divergence: geometric mean over bands (worst-case variant reported too)
        D_k = exp( (1/J) * sum_b log(Dhat_kb + eps) )
        # region coverage = exp(-D_k / tau), tau = running median of D_k over regions
        C_k = exp(-D_k / tau)
    C = (1/K) * sum_k log(C_k + eps)      # geometric-mean coverage
    return C, {C_k}

# Phase 2: dual-form distillation step
def train_step(theta):
    # (a) standard random-condition distillation loss, per-condition weighted
    sample batch of noised x_t and random conditions c
    L_distill = mean_i [ w(c_i) * || eps_S(x_t,t,c_i;theta) - eps_T(x_t,t,c_i) ||^2 ]
    # weights from coverage: uncovered regions automatically up-weighted
    w(c_i) = 1 / (C_{k(c_i)} + eps)        # k(c_i) = region of c_i
    # (b) coverage functional and its gradient (dual form)
    C, Ck = coverage_functional(theta)
    L = L_distill - lambda_cov * C          # lambda_cov = 0.1, tuned on held-out prompts
    theta <- theta - lr * grad_theta L
    # (c) certificate: regions with C_k < delta (delta = 0.5) are flagged
    certificate = {k : Ck[k] for k in 1..K}
    return theta, certificate

# Loop: alternate (a) weighted distillation and (b) coverage ascent every step;
# re-estimate region centroids every 500 steps from teacher only.
# Cost control: update C_k with an EMA (decay 0.99) over a random subsample of
#   K_sub = 8 regions per step; full K-region sweep every 100 steps.
```

**Why this design.** We chose a teacher-only score-field clustering over a caption-embedding partition because random-condition distillation deliberately has no matched captions, so the concept space must be defined by what the teacher *knows* (its conditional score statistics) rather than by text embeddings, accepting the cost that two semantically different conditions with identical score statistics collapse into one region. We chose a Fisher-divergence coverage functional `C_k = exp(-D_k/tau)` over a discriminator-based coverage score because the teacher's conditional score is exactly the object the distillation loss already matches, so coverage and distillation live on the same surface and cannot be gamed independently, accepting the cost that `tau` must be estimated online (running median) and can drift. We chose the dual form `L_distill - lambda_cov * C` with inverse-coverage weights `w(c)=1/(C_k+eps)` over a hard region-sampling curriculum because the dual form makes uncovered regions *automatically* up-weighted through the same gradient step that reduces the average loss, rather than requiring a separate scheduler — this is the structural property that distinguishes CCD from DMD (One-Step Diffusion with Distribution Matching Distillation), which matches a global distribution and has no per-region term, and from DKDM, which distills time-domain knowledge with no condition-space geometry at all; a domain expert would not describe CCD as a variant of either because neither carries a coverage invariant computed from the teacher's conditional score field. We chose geometric-mean aggregation `(1/K) sum log C_k` over arithmetic mean because a single uncovered region should not be masked by K-1 covered ones, accepting the cost of noisier gradients when K is large. We chose the noise-schedule-stratified `D_k` (geometric mean over J=4 bands, per-band normalized) over a single `q_t`-averaged Fisher divergence because the single-band version is dominated by high-noise latents and cannot certify low-noise mode proportions; the stratified version costs J× more teacher/student score evaluations per region, which we offset with the EMA subsampling above.

**Why it measures what we claim.** The quantity `D_k = exp((1/J) sum_b log(Dhat_kb))` measures *per-region student support across the reverse trajectory* because each band `b` isolates a noise level, the per-band normalization `Dhat_kb = D_kb / median_k D_kb` removes the intrinsic scale differences across bands, and the geometric mean penalizes a region that is matched at high noise but mismatched at low noise; this assumption fails when the student matches the score on the sampled latents within each band but diverges between bands (e.g., at band boundaries), in which case `D_k` reflects support at the J sampled noise levels rather than the full continuum. The inverse weight `w(c)=1/(C_k+eps)` measures *coverage deficit* because it is monotone decreasing in region coverage under the assumption that `C_k` is a faithful proxy for support mass; this assumption fails when a region is covered but with wrong mode proportions, in which case the weight reflects presence rather than correctness. The certificate threshold `delta=0.5` measures *non-negligible support* because `C_k = exp(-D_k/tau)` crosses 0.5 exactly when `D_k = tau log 2` under the assumption that `tau` tracks the median region divergence, so a region below threshold has divergence well above the typical region; this assumption fails when the region distribution is heavy-tailed, in which case the threshold flags rare-but-covered regions instead of uncovered ones — which is why the experiment reports the full per-region `C_k` profile alongside the thresholded certificate.

## Contribution

(1) A teacher-only coverage functional over the condition space, derived from the teacher's conditional score field, that yields a per-region certificate of student support without real images or matched captions. (2) A dual-form distillation objective that couples the standard random-condition denoising loss with the coverage functional through inverse-coverage per-condition weights, so uncovered concept regions are automatically up-weighted within the same gradient step. (3) A teacher-only concept partition procedure that defines the condition-space regions by clustering conditional score-field statistics, making the certificate architecture-agnostic and compatible with data-free distillation pipelines such as DKDM.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | Random-condition distillation benchmark (no captions) | Matches the caption-free regime CCD assumes |
| Downstream rare-concept benchmark | Counting prompts (e.g., "three red cubes") and compositional prompts (e.g., "a blue sphere left of a green cylinder") drawn from a held-out set of 256 prompts | Tests whether flagged regions correspond to actual generation failures, not just synthetic worst-region divergence |
| Primary metric | Worst-region teacher Fisher divergence (max over regions of the stratified `D_k`) | Directly tests per-region coverage, not average quality |
| Secondary metric | Per-region `C_k` profile and number of regions below `delta=0.5` | Tests the certificate's soundness (no false "covered" regions) |
| Secondary metric | Downstream rare-concept generation accuracy (CLIP-score on held-out prompts, counting accuracy) | Tests practical significance of the certificate |
| Baseline | Random Conditioning distillation | Isolates value of teacher-only coverage term |
| Baseline | DMD2 | Tests whether global distribution matching misses region deficits |
| Baseline | DKDM | Tests data-free distillation without condition-space geometry |
| Baseline | Hard-example mining (per-cluster teacher-student divergence used only for reweighting, no dual-form ascent) | Tests whether the certificate adds something beyond standard reweighting |
| Ablation-of-ours | CCD without coverage functional (inverse weights only) | Isolates dual-form gradient from weighting |
| Ablation-of-ours | CCD with single-band `D_k` (original `q_t` average) | Tests whether noise-schedule stratification is necessary for the certificate to be sound |
| Ablation-of-ours | CCD with arithmetic-mean aggregation instead of geometric mean | Tests the masking claim |

### Why this setup validates the claim

The central claim is that CCD gives non-negligible student support on every teacher concept region, so the decisive metric is the *worst-region* Fisher divergence, not average generation quality: a method that improves the mean while leaving one region uncovered is exactly what CCD claims to prevent, and a mean-based metric would hide that. Random Conditioning distillation is the right first baseline because it shares CCD's caption-free setup and differs only in lacking the coverage functional and inverse weights, so any gap isolates the coverage mechanism rather than the data regime. DMD2 tests the sub-claim that global distribution matching does not carry a per-region invariant: if CCD's advantage over DMD2 is concentrated on low-coverage regions, that confirms the region term is doing the work. DKDM tests the sub-claim that condition-space geometry matters at all, since it distills without any condition-space structure. The hard-example-mining baseline tests whether the dual-form ascent adds anything beyond reweighting, and the single-band ablation tests whether the stratified `D_k` is necessary for the certificate to be sound. The downstream rare-concept benchmark tests whether flagged regions correspond to actual generation failures, establishing practical significance beyond the synthetic worst-region metric. The ablation (inverse weights but no coverage gradient) separates the two halves of the dual form, so a gain attributable to weighting alone would be caught rather than credited to coverage ascent.

### Expected outcome and causal chain

**vs. Random Conditioning distillation** — On a case where the random-condition sampler draws a rare concept region (e.g. an unusual style combination) only a handful of times, the baseline produces a student whose conditional score in that region drifts far from the teacher because its loss is an unweighted average over all conditions and rare regions contribute almost no gradient. Our method instead assigns that region a large inverse weight and an explicit coverage-ascent term because `C_k` is computed per region and enters the same gradient step, so we expect a noticeable gap in worst-region Fisher divergence with near-parity on the most frequent regions — a gap concentrated exactly where the sampler under-draws.

**vs. DMD2** — On a case where the student matches the teacher's global output distribution well but collapses two semantically distinct conditions onto one mode, DMD2 produces high average sample quality while one region's conditional score is badly mismatched, because its distribution-matching objective is global and has no per-region term to penalize the collapse. Our method instead keeps a per-region `C_k` that drops when any region's Fisher divergence rises, and the geometric-mean aggregation prevents K-1 covered regions from masking the uncovered one, so we expect DMD2 and CCD to be close on overall FID-like quality but CCD to show a clear advantage on the flagged-region divergence, with CCD's certificate flagging fewer regions below `delta`.

**vs. DKDM** — On a case where the teacher's conditional score field partitions the condition space into regions with very different divergence scales, DKDM produces a student that is uniformly mediocre across regions because it distills in the time domain with no notion of which condition regions are under-covered. Our method instead clusters conditions by teacher score statistics and normalizes each region's divergence by the running median `tau`, so regions with intrinsically larger divergence are not silently ignored, and we expect DKDM to show a flatter, higher worst-region divergence while CCD's per-region profile is more even — the signature of coverage rather than average fit.

**vs. Hard-example mining** — On a case where a rare region is under-drawn, hard-example mining up-weights it but has no per-region certificate and no coverage-ascent term, so it can over-fit the few sampled conditions in that region without guaranteeing support across the region's score field. Our method instead couples the weight to a stratified `C_k` that is recomputed from the teacher and enters the gradient, so we expect CCD to match hard-example mining on frequent regions but to show lower worst-region divergence and fewer flagged regions on the rare regions, isolating the certificate's contribution beyond reweighting.

**vs. CCD with single-band `D_k`** — On a case where the student matches the teacher at high noise but collapses modes at low noise, the single-band variant reports high `C_k` (false "covered") because its `D_k` is dominated by high-noise latents, while the stratified variant's low-noise band drops `C_k` below `delta`. We expect the single-band variant to show a larger gap between its certificate and actual downstream rare-concept accuracy, and the stratified variant to close that gap — the direct test of the load-bearing assumption.

### What would falsify this idea

If CCD's improvement over Random Conditioning distillation is uniform across all regions rather than concentrated in the low-coverage regions the certificate flags, the coverage mechanism is not what produces the gain and the central claim is wrong. Likewise, if the ablation without coverage ascent matches full CCD on worst-region divergence, the dual-form term is inert and the certificate is not measuring student support; and if the single-band `D_k` variant matches the stratified variant on downstream rare-concept accuracy, the noise-schedule stratification is unnecessary and the load-bearing assumption is not repaired.

## References

1. Random Conditioning with Distillation for Data-Efficient Diffusion Model Compression
2. Improved Distribution Matching Distillation for Fast Image Synthesis
3. DKDM: Data-Free Knowledge Distillation for Diffusion Models with Any Architecture
4. One-Step Diffusion with Distribution Matching Distillation
5. DreamFusion: Text-to-3D using 2D Diffusion
6. On Distillation of Guided Diffusion Models
