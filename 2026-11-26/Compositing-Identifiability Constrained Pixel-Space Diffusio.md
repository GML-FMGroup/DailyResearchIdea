# Compositing-Identifiability Constrained Pixel-Space Diffusion Matting

## Motivation

Pixel-space diffusion matting methods such as PixelMatte (E1) achieve high-fidelity alpha prediction by transferring generative priors and refining with post-hoc calibration modules (LayerShift, High-Pass Refiner), but they never enforce the alpha-compositing equation that defines matting itself: I = αF + (1−α)B. This structural omission means the predicted alpha and foreground are only implicitly constrained by synthetic supervision and real-image fine-tuning, so opacity-identifiability under real compositing distributions is not guaranteed—the calibration modules act as post-hoc refiners rather than identifiability-constrained estimators. The same gap recurs in video matting (MatAnyone, E4) and instance matting (MP-Mat, E6), where matte consistency is pursued through memory or multiplane representations without an explicit compositing constraint tying alpha to observed pixels.

## Key Insight

Enforcing a differentiable compositing-consistency residual between the observed image and the alpha-composited foreground/background during fine-tuning turns the matting network's output from a generatively plausible alpha into an alpha that is identifiable with respect to the actual compositing equation, because the residual is zero exactly on the identifiability manifold of the matting model.

## Method

**What it is.** We introduce CIC-Matte (Compositing-Identifiability Constrained Matting), a fine-tuning framework that augments PixelMatte's pixel-space diffusion matting pipeline with an explicit differentiable compositing-consistency loss. Inputs: an image I, a trimap T, and PixelMatte's predicted alpha α and foreground F (and an estimated background B). Output: a refined alpha α' and foreground F' that satisfy the compositing equation I ≈ α'F' + (1−α')B while retaining PixelMatte's boundary fidelity.

**Load-bearing assumption (stated explicitly).** The compositing residual r = I − (αF + (1−α)B) is zero on a high-dimensional family of (α, F, B) triples, not at a unique point (3 equations per pixel, 7 unknowns). CIC-Matte therefore does NOT claim that driving r to zero alone identifies α. Instead, it relies on the assumption that the residual is evaluated only where B is observed (the known-background trimap region) and that F is constrained by a local color-line prior, so that the zero set is reduced to a unique (α, F) per pixel. This assumption fails when the trimap provides no background samples in a mixed-pixel neighborhood, in which case the residual reflects background estimation error rather than alpha error.

**How it works.**

```pseudocode
# Phase 1: PixelMatte forward pass (frozen backbone, trainable heads)
α, F = PixelMatte(I, T)            # single forward pass, pixel-space
B = estimate_background(I, α)      # mask-weighted color propagation from known-B region

# Phase 2: Compositing-consistency residual (restricted to known-B region)
M_known = trimap == 'background'   # binary mask of observed background pixels
I_hat = α * F + (1 - α) * B        # differentiable compositing model
r = (I - I_hat) * M_known          # per-pixel residual, 3 channels, zero outside known-B
L_comp = mean( ||r||^2 * w(α) )    # w(α) = 1 + λ_boundary * |∇α|, λ_boundary = 5.0

# Phase 2b: Foreground-ambiguity penalty (local color-line prior on F)
# For each 3x3 window, fit a color line to F; penalize deviation from the line.
L_colorline = mean( ||F - Proj_line(F)||^2 * w(α) )

# Phase 3: Joint calibration with LayerShift and High-Pass Refiner
α_c, F_c = LayerShift(α, F)         # regional opacity calibration
α_r, F_r = HighPassRefiner(α_c, F_c) # boundary fidelity calibration

# Phase 4: Identifiability-constrained fine-tuning objective
L_total = L_alpha_sup + β * L_comp(α_r, F_r, B, I) + δ * L_colorline(F_r) + γ * L_reg(α_r, F_r)
# β = 0.5, δ = 0.1, γ = 0.01, L_alpha_sup = synthetic alpha supervision
# L_reg = total-variation on α_r + smoothness on F_r

# Phase 5: Single-forward inference
α_out, F_out = CIC-Matte(I, T)      # all modules fused, no extra passes
```

**Why this design.** We chose to attach the compositing-consistency loss directly to the calibrated outputs (α_r, F_r) rather than to the raw PixelMatte outputs because the calibration modules are where regional and boundary errors accumulate, so constraining the post-calibration surface forces LayerShift and High-Pass Refiner to act as identifiability-constrained estimators rather than post-hoc refiners; the cost is that gradients must flow through both lightweight modules, which adds memory but keeps the backbone frozen. We chose a mask-weighted residual w(α) that upweights boundary pixels (via |∇α|) over a uniform residual because matting errors concentrate at mixed pixels, accepting that the loss is less sensitive to uniform-region opacity bias. We chose to estimate the background B by color propagation from the known-background trimap region rather than jointly optimizing B, because joint optimization of B, F, and α is underdetermined without additional constraints (the classic matting ill-posedness), accepting that B estimation errors propagate into the residual; to mitigate this, the residual is restricted to the known-B region (M_known) where B is observed, not estimated. We chose to add a local color-line prior on F (L_colorline) because the raw compositing residual alone is zero on a degenerate family of (α, F) pairs; the color-line prior reduces this family to a unique (α, F) per pixel, accepting that the prior may be violated at sharp color edges. We chose to keep the backbone frozen and fine-tune only the calibration heads plus a small residual adapter, rather than fine-tuning the full diffusion transformer, because the pretrained pixel-space prior already encodes strong image statistics and full fine-tuning risks catastrophic forgetting; the cost is that the identifiability constraint cannot reshape the backbone's internal representation. A domain expert would not describe this as 'PixelMatte + a loss term' because the compositing residual is not an auxiliary regularizer—it is a constraint that changes what the calibration modules are trained to estimate, converting them from post-hoc refiners into identifiability-constrained estimators, a role no prior matting calibration module (including PixelMatte's LayerShift and High-Pass Refiner) plays.

**Why it measures what we claim.** The per-pixel compositing residual r = (I − (αF + (1−α)B)) · M_known measures alpha-identifiability because the matting compositing equation is exact for any true (α, F, B) triple, so a zero residual on the known-B region is equivalent to the predicted triple lying on the identifiability manifold; this assumption fails when the background estimate B is wrong (e.g., in regions where the trimap gives no background samples), in which case the residual reflects background estimation error rather than alpha error. The boundary-weighted mask w(α) = 1 + λ|∇α| measures boundary-sensitive identifiability because |∇α| is large exactly at mixed pixels where alpha is hardest to identify; this assumption fails when the alpha gradient is large due to noise rather than true mixing, in which case the loss overweights noisy regions. The color-line penalty L_colorline measures foreground-ambiguity because a true foreground F lies on a local color line within a small window; this assumption fails at sharp color edges, in which case the penalty reflects color-line violation rather than foreground error. The LayerShift and HighPassRefiner outputs (α_r, F_r) measure the calibrated matte because these modules are the only trainable pathway between the frozen backbone and the loss; this assumption fails when the frozen backbone's features are systematically biased, in which case the calibration modules cannot correct the bias and the loss reflects calibration capacity rather than backbone identifiability.

## Contribution

["A compositing-identifiability constrained fine-tuning framework (CIC-Matte) that adds a differentiable, boundary-weighted compositing-consistency loss tying predicted alpha and foreground to the observed image, jointly optimized with PixelMatte's LayerShift and High-Pass Refiner calibration modules.", 'A design principle showing that calibration modules in pixel-space diffusion matting can be converted from post-hoc refiners into identifiability-constrained estimators by constraining their outputs on the compositing manifold, rather than adding new network capacity.', 'A background-estimation and mask-weighting scheme that makes the compositing residual robust to trimap sparsity, enabling the identifiability constraint to be applied during real-image fine-tuning without paired ground-truth foreground.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | Composition-1k | Standard mixed-pixel benchmark with ground-truth alpha |
| Primary metric | Gradient error (SAD on boundaries) | Detects boundary identifiability, our claimed effect |
| Baseline | PixelMatte (frozen) | Tests whether constraint beats post-hoc calibration |
| Baseline | PixelMatte + uniform residual loss | Tests boundary-weighting sub-claim |
| Baseline | Trimap-free latent diffusion matting | Tests pixel-space vs compressed-prior advantage |
| Baseline | CIC-Matte with F frozen during L_comp | Tests whether alpha alone improves, isolating F-absorption |
| Ablation-of-ours | CIC-Matte without L_comp | Isolates compositing-constraint contribution |
| Ablation-of-ours | CIC-Matte without L_colorline | Isolates foreground-ambiguity penalty contribution |

### Why this setup validates the claim

The central claim is that attaching a compositing-consistency residual to PixelMatte's calibrated outputs converts LayerShift and High-Pass Refiner from post-hoc refiners into identifiability-constrained estimators, yielding better boundary alpha than calibration alone. Composition-1k is the right testbed because its ground-truth alpha and synthetic composites make the compositing equation exactly satisfiable, so any residual we drive down is genuine alpha error, not background ambiguity. PixelMatte frozen tests the sub-claim that the constraint, not the backbone, does the work; the uniform-residual variant isolates the boundary-weighting sub-claim by removing |∇α| upweighting while keeping everything else identical. The latent-diffusion baseline tests whether pixel-space prediction plus an explicit identifiability constraint beats compressed-prior matting on high-frequency boundaries. The F-frozen baseline tests the load-bearing assumption directly: if the residual is absorbed by the unconstrained foreground, freezing F during L_comp should prevent alpha improvement; if alpha still improves, the constraint is acting on alpha. The no-L_comp and no-L_colorline ablations are the falsification levers: if removing either does not degrade boundary gradient error, the corresponding mechanism claim collapses. Gradient error is the correct primary metric because our predicted effect is concentrated at mixed pixels; SAD over the whole image would be dominated by easy uniform regions where all methods agree and would mask the boundary signal.

### Expected outcome and causal chain

**vs. PixelMatte (frozen)** — On a case with fine hair against a textured background, the frozen PixelMatte produces a matte whose boundary is slightly displaced and whose mixed-pixel alpha is biased because its calibration modules were trained only against alpha supervision, never asked to explain the observed composite. Our method instead backpropagates the compositing residual through LayerShift and High-Pass Refiner, so those modules must produce an (alpha, F) pair that reconstructs I on the known-B region, correcting the boundary bias. We expect a concentrated gradient-error reduction at mixed pixels with near-parity on uniform interiors.

**vs. PixelMatte + uniform residual loss** — On a case with a thin translucent strand, the uniform-residual variant spreads correction effort across easy interior pixels, so its boundary alpha barely improves because the loss gradient is dominated by large uniform regions where alpha is already correct. Our method instead weights the residual by 1 + λ|∇α|, concentrating identifiability pressure exactly at mixed pixels, so we expect a visible gap on boundary gradient error but similar SAD on uniform regions.

**vs. Trimap-free latent diffusion matting** — On a case with high-frequency edge detail, the latent baseline produces blurred boundaries because its VAE compression discards the high-frequency precision matting requires. Our method instead predicts in pixel space and enforces the compositing constraint directly on full-resolution pixels, so we expect a marked gradient-error advantage on high-frequency subsets and smaller differences on smooth boundaries.

**vs. CIC-Matte with F frozen during L_comp** — On a case where the foreground color is ambiguous (e.g., a dark object on a dark background), the F-frozen variant cannot adjust F to absorb the residual, so if the residual is truly identifying alpha, alpha should still improve relative to frozen PixelMatte. If instead the F-frozen variant shows no alpha improvement, the residual is being absorbed by F and the identifiability claim is falsified. We expect the F-frozen variant to show a smaller but still positive boundary gradient-error reduction, confirming that alpha alone improves.

### What would falsify this idea

If CIC-Matte's gain over frozen PixelMatte is uniform across boundary and interior regions rather than concentrated at mixed pixels, or if removing L_comp yields no boundary-gradient degradation, or if the F-frozen variant shows no alpha improvement over frozen PixelMatte, then the compositing constraint is not acting as an identifiability constraint and the central claim is wrong.

## References

1. High-Fidelity Image Matting with Pixel-Space Diffusion Transformers
