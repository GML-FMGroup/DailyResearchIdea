# Factor-Exchangeable Reliability Fields for Noise-Robust Conditional Image Generation

## Motivation

Conditional image generation methods, such as the disentangled text supervision of DeCLIP in Language-Guided Visual Perception Disentanglement for Image Quality Assessment and Conditional Image Generation, treat conditioning signals (text factors, frozen embeddings) as noise-free and complete definitions of the target factors. This structural assumption is shared across multiple paper-trees: when the text factor is inaccurate or the embedding is corrupted, the generator is forced to fit noise, and no mechanism exists to estimate per-factor reliability or to expand coverage to unseen distortions or classes without new annotations. The root cause is that conditioning is consumed as a sufficient statistic, with no reliability field identified from the data itself.

## Key Insight

A per-factor reliability field is identifiable end-to-end from a factor-wise exchangeability invariant—requiring the generator to produce identical outputs when a factor's supervision is resampled from its own conditional distribution—because this invariance holds if and only if the reliability field correctly down-weights factors whose supervision is inconsistent with the target, without needing ground-truth noise labels.

## Method

```pseudocode
# FERF: Factor-Exchangeable Reliability Field
Input: image x, factor supervision set S = {s_1,...,s_K} (text factors, encoder embeddings)
Output: generated image x_hat, reliability field r = (r_1,...,r_K) ∈ [0,1]^K

# --- Load-bearing assumption (stated explicitly) ---
# The exchangeability loss ||G(z_x,c) - G(z_x,c')||^2 is minimized if and only if r_k correctly
# identifies which factors are corrupted, i.e., the generator's output sensitivity to resampling
# a factor from its conditional distribution is a faithful, invertible proxy for that factor's
# supervision reliability. This holds only if (i) R_k approximates the true conditional p(s_k|x)
# for all corruption types, (ii) G is not allowed to trivially ignore conditioning, and (iii) r_k
# is anchored to an external reliability signal. We enforce (i)-(iii) below.

1. Encode x -> z_x via generator encoder E.
2. For each factor k:
   a. Encode s_k -> e_k via factor encoder F_k.
   b. Predict reliability r_k = sigma(MLP_k(e_k, z_x))  # sigma = sigmoid; MLP_k: 2-layer MLP, hidden=256, GeLU
3. Form conditioning c = sum_k r_k * e_k / sum_k r_k  (reliability-weighted aggregation)
4. Generate x_hat = G(z_x, c).
5. Exchangeability training step (with stop-gradient on G):
   a. For a random subset M ⊆ {1..K}, resample s_k' ~ p(s_k | x) for k in M using a factor-specific conditional resampler R_k (trained via denoising score matching on paired data augmented with Gaussian noise σ=0.1).
   b. Compute c' using r_k and e_k' for k in M.
   c. Loss_exch = || sg(G(z_x, c)) - sg(G(z_x, c')) ||^2  # sg = stop-gradient; G frozen for this loss
6. Reconstruction loss: Loss_rec = || x - x_hat ||^2 + LPIPS(x, x_hat).
7. Reliability anchor loss (replaces weak L2 prior):
   a. For a held-out clean calibration set C of 512 examples (factors known uncorrupted), compute r_k^clean.
   b. Loss_anchor = -[ log r_k^clean + log(1 - r_k^perturbed) ] where s_k^perturbed is drawn from a deliberately perturbed conditional (e.g., add Gaussian noise σ=0.5 to s_k).
   c. Total anchor: Loss_anchor = mean over k and calibration examples.
8. Total loss: Loss = Loss_rec + lambda_exch * Loss_exch + lambda_anchor * Loss_anchor  (lambda_exch=0.5, lambda_anchor=0.1)
9. Update E, F_k, MLP_k, G, and R_k jointly (G updated only via Loss_rec; stop-gradient in Loss_exch).
```

**Why this design**
We chose a factor-wise exchangeability invariant over a separate noise model or confidence head because the invariant is defined directly on the generator's output distribution: resampling a factor from its conditional distribution should not change the output if that factor is reliable, so the reliability field is identified by the generator's own sensitivity, not by an auxiliary classifier that could be fooled by spurious correlations. We chose a conditional resampler R_k trained via denoising score matching over a fixed Gaussian noise model because it provides a concrete, trainable distribution p(s_k | x) that can generate plausible alternative supervisions without requiring ground-truth noise labels, accepting the cost that R_k may be imperfect for rare distortions. We chose reliability-weighted aggregation over attention-based fusion because the weighted sum makes the gradient of the exchangeability loss with respect to r_k directly proportional to the output change under resampling, giving a clear training signal; the trade-off is that the aggregation is linear and may underrepresent complex interactions between factors. We chose to regularize r_k toward 1 (lambda_reg=0.01) to prevent the trivial solution of setting all reliabilities to zero, accepting a small bias toward trusting supervision. This design is not a variant of DeCLIP's disentangled text supervision: DeCLIP uses fixed text encoders to separate perceptual and semantic factors but never estimates per-factor reliability or uses an exchangeability invariant; our method introduces a reliability field that is identified by the generator's invariance to factor resampling, which DeCLIP does not produce.

**Why it measures what we claim**
The exchangeability loss ||G(z_x, c) - G(z_x, c')||^2 measures the reliability of factor k because under the assumption that the conditional resampler R_k draws from the true conditional distribution p(s_k | x), a reliable factor's supervision is exchangeable with any other draw from that distribution, so the output should be invariant; this assumption fails when R_k is misspecified for unseen distortions, in which case the loss reflects the mismatch between R_k and the true conditional rather than the factor's reliability. The reliability weight r_k measures the inverse sensitivity of the generator to factor k because the gradient of the exchangeability loss with respect to r_k is proportional to the output change under resampling, under the assumption that the generator's output is differentiable and the aggregation is linear; this assumption fails when factors interact nonlinearly, in which case r_k reflects a first-order approximation of reliability. The anchor loss (replacing the L2 prior) measures the prior belief that supervision is mostly reliable because it penalizes low r_k on clean calibration factors and high r_k on perturbed factors, under the assumption that the calibration set is representative; this assumption fails in heavily corrupted datasets, in which case the anchor biases reliabilities upward.

## Contribution

['A novel framework, FERF, that learns a per-factor reliability field over conditioning signals jointly with a conditional image generator via a factor-wise exchangeability invariant, eliminating the need for ground-truth noise labels or a separate confidence head.', "A design principle showing that the exchangeability invariant identifies reliability through the generator's own sensitivity to factor resampling, enabling automatic coverage expansion to unseen distortions and classes because the invariant is defined over the factor's conditional distribution rather than a fixed label set.", 'A conditional resampler trained via denoising score matching that provides a concrete mechanism for generating alternative factor supervisions during training, making the exchangeability invariant computable without paired clean/noisy data.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | I&2T with synthetic factor corruptions | Provides paired factors and controlled reliability shifts |
| Primary metric | AUROC of r_k against ground-truth corruption | Directly measures reliability discrimination, not circular |
| Secondary metric | FID and CLIP-score on corrupted subsets | Measures downstream generation quality under corruption |
| Baseline | DeCLIP | Tests whether disentangled text supervision alone suffices |
| Baseline | Standard conditional diffusion (LDM) | Tests whether reliability weighting is needed for robustness |
| Baseline | DisQUE | Tests whether appearance-content disentanglement without reliability fails |
| Baseline | Learned auxiliary confidence head (noise-prediction classifier) | Tests whether exchangeability yields reliability a supervised head cannot |
| Ablation-of-ours | FERF w/o exchangeability loss | Isolates the role of the invariance training signal |
| Ablation-of-ours | FERF w/o anchor loss | Tests whether anchor prevents r_k collapse |
| Ablation-of-ours | FERF with frozen R_k (trained offline) | Tests necessity of joint R_k training |

### Why this setup validates the claim

The central claim is that reliability is identified by the generator’s invariance to factor resampling, not by auxiliary classifiers. The I&2T dataset with synthetic factor corruptions gives us ground-truth reliability labels per factor, so we can measure whether the learned reliability field r_k aligns with true corruption levels via AUROC. The primary metric, AUROC of r_k against ground-truth corruption, directly quantifies how well r_k separates corrupted from clean factors; a higher AUROC is the predicted signature. The secondary metrics, FID and CLIP-score on corrupted subsets, measure downstream generation quality under corruption. DeCLIP tests the sub-claim that disentangled text supervision alone cannot estimate per-factor reliability: it should produce similar outputs regardless of factor corruption because it lacks a reliability mechanism. Standard conditional diffusion (LDM) tests whether naive conditioning without reliability weighting is sensitive to corrupted factors. DisQUE tests whether appearance-content disentanglement without a reliability field also fails to isolate factor reliability. The learned auxiliary confidence head (noise-prediction classifier trained on the same synthetic corruptions) tests whether the exchangeability invariant yields reliability estimates that a supervised head cannot, isolating the novel mechanism’s contribution. The ablation (FERF without exchangeability loss) tests whether the exchangeability invariant is necessary to identify reliability; without it, r_k should collapse to the anchor prior and AUROC should not discriminate corrupted from clean factors. The ablation (FERF without anchor loss) tests whether the anchor prevents r_k collapse. The ablation (FERF with frozen R_k) tests whether joint R_k training is necessary. Thus, the combination forms a falsifiable test: if our method’s AUROC does not separate corrupted from clean factors better than baselines, or if the ablation matches our performance, the claim that exchangeability identifies reliability is wrong.

### Expected outcome and causal chain

**vs. DeCLIP** — On a case where a text factor (e.g., “blurry”) is corrupted to “sharp” for a blurry image, DeCLIP produces an output that follows the corrupted factor because its fixed text encoder has no mechanism to estimate per-factor reliability. Our method instead assigns a low r_k to the corrupted factor via the exchangeability loss, so the reliability-weighted aggregation downweights it, and the output remains close to the original. We expect a noticeable gap in AUROC on the corrupted-factor subset (our method much higher) but parity on the clean-factor subset.

**vs. Standard conditional diffusion (LDM)** — On a case where a segmentation mask factor is noisy (e.g., boundary errors), LDM generates an image that propagates the noise because it conditions on the mask directly without reliability weighting. Our method estimates a low reliability for the noisy mask via resampling and exchangeability, and the weighted aggregation reduces its influence, producing a cleaner output. We expect our AUROC to be substantially higher on the noisy-mask subset, with similar performance on clean masks.

**vs. DisQUE** — On a case where appearance factors (e.g., lighting) are corrupted but content is intact, DisQUE’s disentanglement without reliability may still mix corrupted appearance into the output because it lacks a per-factor reliability estimate. Our method assigns low reliability to corrupted appearance factors through exchangeability, preserving content while ignoring the corrupted appearance. We expect a clear advantage on the corrupted-appearance subset, with parity on clean subsets.

**vs. Learned auxiliary confidence head** — On a case where a text factor is corrupted in a way not seen during training of the confidence head, the supervised head may overfit to the training corruption types and fail to generalize. Our method, using the exchangeability invariant, does not rely on labeled corruption types and should assign low reliability based on output sensitivity. We expect our AUROC to be higher on unseen corruption types, demonstrating the advantage of the invariant over a supervised head.

### What would falsify this idea

If the AUROC of r_k against ground-truth corruption for our method does not exceed that of the baselines and the learned auxiliary confidence head, or if the ablation without the exchangeability loss achieves the same AUROC, then the central claim that exchangeability identifies reliability is false.

## References

1. Language-Guided Visual Perception Disentanglement for Image Quality Assessment and Conditional Image Generation
2. Joint Quality Assessment and Example-Guided Image Processing by Disentangling Picture Appearance from Content
3. AesExpert: Towards Multi-modality Foundation Model for Image Aesthetics Perception
4. OCTFlow: Unified conditional generation for balanced classification and annotated segmentation in retinal OCT.
