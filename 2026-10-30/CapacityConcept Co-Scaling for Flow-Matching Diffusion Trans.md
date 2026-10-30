# Capacity–Concept Co-Scaling for Flow-Matching Diffusion Transformers via Loss-Curvature Probing

## Motivation

Across multiple paper-trees, from-scratch generative models trained on narrow curated corpora — Oboro's flow-matching diffusion transformer on copyright-cleared images, and the fixed-size assumptions in the Gemma and sigmoid-dual-encoder lineages — are reported as single points with no parameter count, no scaling law, and no accounting for whether results are capacity-limited or data-limited. The structural failure is that 'dataset size' is treated as raw sample count, so the capacity–data coupling is unformalized: a small corpus with many distinct image-text concepts and a same-sized corpus with few concepts are indistinguishable to the practitioner, and Oboro cannot tell whether its quality ceiling comes from model width or concept scarcity. This turns an unmeasured confound into an uncontrolled axis that no amount of additional samples within the corpus resolves.

## Key Insight

The ratio of trainable parameters to the intrinsic concept dimensionality of a fixed corpus is held within a measurable band, and that band is identifiable from the curvature of the flow-matching loss with respect to model width at fixed data — because curvature peaks where capacity first saturates the available concept directions, removing the need for expensive multi-model scaling sweeps.

## Method

**What it is.** COSCA (Concept–Capacity Co-Scaling Analyzer) takes as input a fixed, copyright-cleared image-text corpus, a family of flow-matching diffusion transformers of varying width, and a target loss tolerance; it outputs an estimate of the corpus's intrinsic concept dimensionality $d_c$, the optimal parameter count $N^*$ for that corpus, and a co-scaling band $[\rho_{lo},\rho_{hi}]$ on the ratio $N/d_c$ within which trained models are neither capacity-limited nor data-limited.

**How it works.**

```pseudocode
# Phase 1: estimate intrinsic concept dimensionality d_c from the fixed corpus
for each image-text pair (x_i, t_i):
    embed x_i with frozen DINOv2 (ViT-B/14, 768-d), t_i with frozen CLIP text encoder (ViT-B/32, 512-d)
    form joint vector z_i = [img_emb ; txt_emb]   # 1280-d
Z = stack(z_i)                      # shape (M, 1280)
center Z; compute covariance C = Z^T Z / M
run eigendecomposition; retain eigenvalues lambda_k
# effective concept count: participation ratio, robust to noise floor
d_c = (sum_k lambda_k)^2 / sum_k lambda_k^2
# cross-check with a second estimator: number of eigenvalues above
# the Marchenko-Pastur upper edge for the (M, D) aspect ratio

# Phase 2: width sweep at fixed data (K = 10 widths, log-spaced from 64 to 1024)
for width w in {w_1,...,w_K}:
    train flow-matching DiT of width w on the SAME corpus for T = 50k steps (batch size 64, AdamW lr=1e-4, cosine schedule)
    record loss-vs-noise curve L_w(sigma) on a held-out 5% split
    # convergence check: loss-vs-step plateau (last 5k steps slope < 1e-4) and seed variance (3 seeds)
    fit per-width curvature kappa_w = d^2 L_w / d w^2 at the optimum width

# Phase 3: locate the co-scaling band from curvature
# curvature is maximal at the width where capacity first saturates concepts
w_star = argmax_w kappa_w
N_star = params(w_star)
rho_star = N_star / d_c
band = [rho_star * (1 - delta), rho_star * (1 + delta)]   # delta = 0.15

# Phase 4: report and validate the invariant
return d_c, N_star, band
# validation: train widths inside vs. outside the band, compare
# held-out FID and concept-recall; the band predicts which are capacity-limited
```

**Why this design.** We estimate concept dimensionality from a frozen-feature participation ratio rather than from raw sample count because a copyright-cleared corpus can contain thousands of near-duplicate images that inflate sample count without adding concept directions, accepting the cost that the estimator is sensitive to the frozen encoder's feature geometry. We use a log-spaced width sweep of ten models rather than a full scaling-law grid because the limited-data setting cannot afford dozens of training runs, accepting the cost that curvature is estimated from a coarse grid and the band width $\delta$ must absorb interpolation error. We locate the band from the curvature peak $\arg\max_w \kappa_w$ rather than from held-out loss minima because loss minima are flat and noisy at small data, whereas curvature has a sharp, identifiable maximum where capacity first saturates the available concept directions. A domain expert would not describe this as a variant of Chinchilla-style scaling laws or of Oboro's single-model report: Chinchilla regresses loss on raw token count with no concept-dimensionality term, and Oboro reports one fixed-size model with no curvature probe — COSCA's load-bearing quantity is the curvature-vs-width peak normalized by $d_c$, which neither method computes.

**Why it measures what we claim.** The participation ratio $d_c$ measures the effective number of distinct image-text concepts because the assumption that concept directions correspond to the leading eigen-directions of the joint embedding covariance makes the participation ratio equal to the effective rank of the concept manifold; this assumption fails when the frozen encoder entangles style with content, in which case $d_c$ reflects stylistic variance rather than semantic concept count. The curvature $\kappa_w = d^2 L_w / dw^2$ measures capacity saturation because the assumption that flow-matching loss is locally quadratic in width near the saturation point makes the second derivative peak exactly where additional width stops reducing loss on existing concepts; this assumption fails when training is under-converged at some widths, in which case $\kappa_w$ reflects optimization noise rather than saturation. The ratio $\rho = N/d_c$ measures the co-scaling invariant because the assumption that the optimal parameter count scales linearly with intrinsic concept dimensionality makes $\rho$ constant within the band; this assumption fails when the corpus has heavy concept imbalance, in which case $\rho$ drifts with the tail-concept count and the band must be widened.

## Contribution

(1) COSCA, a curvature-probing framework that estimates the intrinsic concept dimensionality of a fixed corpus and locates the parameter-to-concept co-scaling band without held-out multi-model scaling sweeps. (2) The empirical finding that flow-matching loss curvature with respect to model width peaks at the capacity–concept saturation point, yielding an identifiable co-scaling band on the parameter-to-concept ratio. (3) A reproducible protocol and released diagnostic for auditing whether a from-scratch limited-data diffusion transformer is capacity-limited or data-limited.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Copyright-cleared limited image-text corpus (oboro-style) | Matches the limited-data regime the claim targets |
| Primary metric | Held-out concept-recall vs. FID gap across widths | Detects capacity-limited vs. data-limited regimes directly |
| Baseline | Oboro fixed-width DiT | Single-model report with no curvature probe |
| Baseline | Chinchilla-style loss-vs-parameter regression | Predicts optimum from raw counts, no concept term |
| Baseline | Width sweep without d_c normalization | Tests whether the ratio, not width alone, is load-bearing |
| Ablation-of-ours | COSCA with d_c replaced by raw sample count | Isolates the concept-dimensionality estimator's contribution |

### Why this setup validates the claim

The central claim is that the curvature-vs-width peak, normalized by intrinsic concept dimensionality, identifies a co-scaling band that predicts which model widths are capacity-limited versus data-limited. Each baseline probes a distinct sub-claim. Oboro's single fixed-width model tests whether a static report can even locate a saturation point; Chinchilla-style regression tests whether raw-count scaling laws, which omit a concept term, misplace the optimum on a corpus whose sample count is inflated by near-duplicates; the unnormalized width sweep tests whether the ratio N/d_c is load-bearing or whether absolute width alone suffices. The ablation replacing d_c with sample count isolates the estimator's contribution. Held-out concept-recall is the right primary metric because capacity saturation should manifest first as failure to cover tail concepts, not as aggregate FID degradation; a method that only improves FID without concentrating gains on the predicted capacity-limited widths would not confirm the causal story.

### Expected outcome and causal chain

**vs. Oboro fixed-width DiT** — On a case where the corpus contains many near-duplicate images, the baseline trains one fixed-width model whose capacity is chosen without reference to concept count, so it either over-parameterizes relative to the effective concept directions and FID plateaus flat, or under-parameterizes and misses tail concepts. Our method instead estimates d_c from the participation ratio and locates w_star from curvature, because the curvature peak marks where width first saturates available concept directions. We expect the baseline to sit near but not at the band, showing a concept-recall deficit on tail-concept subsets while matching on head concepts.

**vs. Chinchilla-style loss-vs-parameter regression** — On a corpus where sample count is inflated by near-duplicates, the baseline regresses loss on raw count and predicts an optimum shifted toward larger widths, because it has no term for effective concept dimensionality. Our method instead normalizes by d_c, which is far below the raw sample count, because the participation ratio discounts redundant directions. We expect the baseline's predicted optimum to fall outside our band on the high side, with the baseline's chosen width showing no concept-recall gain over band-interior widths despite more parameters.

**vs. Unnormalized width sweep** — On a case where two corpora differ in concept richness but share sample count, the baseline's curvature peak lands at similar widths for both, because absolute width ignores concept count. Our method instead places the band at different ratios for the two corpora, because rho = N/d_c is the invariant. We expect the baseline to misclassify the richer corpus as capacity-limited when it is concept-limited, while our band correctly separates the two regimes.

### What would falsify this idea

If the concept-recall gain from band-interior widths over band-exterior widths is uniform across head and tail concept subsets rather than concentrated on tail concepts, then the curvature peak is not tracking concept saturation and the central claim fails. Equivalently, if the d_c ablation (sample count) produces the same band and the same recall pattern, the concept-dimensionality estimator is not load-bearing.

## References

1. oboro: Text-to-Image Synthesis on Limited Data using Flow-based Diffusion Transformer with MMH Attention
2. SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis
3. eDiff-I: Text-to-Image Diffusion Models with an Ensemble of Expert Denoisers
