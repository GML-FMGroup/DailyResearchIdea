# Iterative Layout Decomposition for Diffusion Models via Structured Sparsity

## Motivation

Existing unified diffusion models like DesignDiffusion generate layouts implicitly in a single forward pass, causing errors in complex multi-object compositions because they lack explicit spatial reasoning over component relationships. This root limitation stems from the structural assumption that a single denoising step can adequately disentangle overlapping text and image regions without iterative refinement, a problem that recurs across multiple branches transitioning from modular pipelines to end-to-end models.

## Key Insight

By decomposing the latent into sparse component maps at each denoising step via a differentiable structured sparsity constraint, the model enforces that each region's noise prediction is conditioned on its own spatial context, directly addressing the missing spatial reasoning in single-pass unified models.

## Method

We propose **Compositional Layout Diffusion (CLD)**, an end-to-end diffusion model that replaces the single-forward-pass layout with an iterative refinement loop containing a differentiable Layout Parsing Module (LPM).

### (A) What it is
CLD takes a noisy latent z_t, text embedding, and timestep t; outputs a predicted noise epsilon_pred. At each denoising step, LPM generates K soft masks that decompose the latent into component latents, which are then separately denoised by a shared U-Net and recombined.

### (B) How it works (pseudocode)
```python
def denoising_step(z_t, text_embedding, t, LPM, denoiser, K=10, lambda_sparse=0.1):
    # (1) Generate component masks via LPM
    raw_masks = LPM(z_t, text_embedding, t)  # (B, K, H, W)
    masks = softmax(raw_masks, dim=1)        # spatial assignment per pixel

    # (2) Enforce structured sparsity (training only)
    # entropy = -sum_k masks_k * log(masks_k), minimized when one-hot
    L_sparse = -torch.mean(torch.sum(masks * torch.log(masks + 1e-8), dim=1))

    # (3) Decompose latent into component latents
    z_t_k = masks.unsqueeze(2) * z_t.unsqueeze(1)  # (B, K, C, H, W)

    # (4) Predict noise per component using shared denoiser
    epsilon_pred_k = denoiser(z_t_k.reshape(B*K, C, H, W), t, text_embedding)
    epsilon_pred_k = epsilon_pred_k.reshape(B, K, C, H, W)

    # (5) Aggregate noise predictions using masks as weights
    epsilon_pred = (masks.unsqueeze(2) * epsilon_pred_k).sum(dim=1)  # (B, C, H, W)

    return epsilon_pred, L_sparse
```
During training, total loss = L_diffusion + lambda_sparse * L_sparse.

### (C) Why this design
We chose softmax masks over hard thresholding because differentiability is essential for end-to-end training; we accept that soft assignments may introduce blur across regions, but counter this with an entropy penalty that sharpens assignments toward one-hot distributions. We used a shared denoiser across components (rather than per-component denoisers) to keep parameters manageable and force the model to learn region-agnostic features, risking that the same network may not specialize to diverse regions; we accept this because conditioning on the component latent provides necessary context. We added the sparsity penalty only during training (not inference) to avoid extra compute at test time, accepting that the hyperparameter lambda must be tuned to balance decomposition granularity without mode collapse. Unlike DesignDiffusion’s single-pass layout, our iterative decomposition explicitly models spatial relationships at each step, mitigating compositional errors without external layout supervision.

### (D) Why it measures what we claim
The masks measure spatial decomposition into distinct regions because each pixel’s softmax assignment normalizes over components; the assumption is that each component captures a coherent text region or object, which fails when components split a single object (then masks reflect arbitrary grouping). The per-component noise prediction measures region-specific denoising because the denoiser operates on the masked latent; the assumption is that noise in each region is independent given the content, which fails when cross-region dependencies exist (e.g., lighting or shadows). The aggregated noise prediction via mask-weighted sum measures compositional combination; the assumption is that global noise is a convex combination of regional noises, which fails for non-additive interactions (e.g., occlusion). These operational definitions link computational quantities to the claimed spatial reasoning, with explicit failure modes to prevent unstated proxy assumptions.

## Contribution

(1) A novel iterative layout decomposition method for diffusion models, CLD, that introduces a differentiable sparsity-constrained parsing module to decompose latents into component maps at each denoising step, enabling explicit spatial reasoning without external layout supervision. (2) The design principle that enforcing per-component noise prediction via shared denoising and mask-weighted aggregation reduces compositional errors in end-to-end text-to-image generation. (3) An ablation analysis (in experiments) demonstrating that the sparsity penalty is critical for mask interpretability and that the iterative loop improves layout coherence over single-pass baselines.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|------------------------|
| Dataset | CLEVR | Tests multi-object spatial composition. |
| Primary metric | FID | Evaluates overall image quality and diversity. |
| Baseline 1 | Stable Diffusion | Standard text-to-image diffusion baseline. |
| Baseline 2 | DesignDiffusion | Single-pass layout diffusion baseline. |
| Baseline 3 | TextDiffuser-2 | Region-conditioned text generation baseline. |
| Ablation | Ours w/o L_sparse | Tests impact of sparsity penalty on masks. |

### Why this setup validates the claim
The combination of CLEVR (controlled multi-object scenes) and FID (holistic quality) provides a clear test: if our method improves compositional generation, FID on CLEVR should be lower than baselines. DesignDiffusion tests whether iterative refinement beats single-pass layout; TextDiffuser-2 tests whether implicit decomposition beats explicit region conditioning; Stable Diffusion tests whether any layout helps. The ablation w/o L_sparse tests whether sharpening masks is necessary. This setup isolates each component, making it possible to falsify the idea if our method does not beat all baselines on this specific dataset.

### Expected outcome and causal chain

**vs. Stable Diffusion** — On a case where the prompt describes "a blue cube to the left of a red sphere", Stable Diffusion often produces a single object or incorrect spatial arrangement because it lacks explicit layout modeling. Our method generates separate masks for each object and denoises them independently, ensuring both are present in correct positions. Thus we expect a noticeable FID improvement on multi-object prompts, but parity on single-object prompts.

**vs. DesignDiffusion** — On a case where object boundaries are ambiguous (e.g., "a striped cat and a plain dog"), DesignDiffusion's single-pass layout may assign wrong regions due to early commitment. Our iterative refinement adjusts masks at each step, allowing correction. Hence we expect our method to achieve lower FID on scenes with spatially ambiguous objects, but similar performance on simple distinct arrangements.

**vs. TextDiffuser-2** — On a case requiring free-form composition without region text prompts (e.g., "a green apple on a wooden table"), TextDiffuser-2 cannot condition on spatial layout. Our method automatically decomposes the scene, enabling proper placement. We thus expect our method to outperform on scenes without explicit region instructions, with improved FID.

### What would falsify this idea
If the FID of our method is not significantly lower than DesignDiffusion on multi-object CLEVR scenes, the claim that iterative decomposition improves composition is falsified. Additionally, if the ablation w/o L_sparse matches full CLD, the importance of mask sharpening is disconfirmed.

## References

1. DesignDiffusion: High-Quality Text-to-Design Image Generation with Diffusion Models
2. Diffusion Model Alignment Using Direct Preference Optimization
3. Improving alignment of dialogue agents via targeted human judgements
