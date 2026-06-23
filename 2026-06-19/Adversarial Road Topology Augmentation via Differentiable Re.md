# Adversarial Road Topology Augmentation via Differentiable Rendering for Robust Video World Models

## Motivation

Existing video world models for driving, such as DriveDreamer-2, are trained on fixed datasets that under-represent rare road topologies (e.g., multiple roundabouts, irregular intersections), causing poor generalization to novel structures. The root cause is that all current approaches rely on learned distributions from static datasets without an explicit mechanism to synthesize physically plausible rare topologies during training, leaving a distribution gap that cannot be closed by scaling data alone.

## Key Insight

Differentiable neural rendering of road layouts enables gradient-based adversarial generation of topology perturbations in a low-dimensional latent space, directly targeting the model's failure modes by backpropagating the video prediction error into the topology parameters.

## Method

# Adversarial Road Topology Augmentation via Differentiable Rendering for Robust Video World Models

## Method

**(A) What it is:** ART (Adversarial Road Topology augmentation) integrates a differentiable renderer into the video diffusion decoder and adversarially perturbs a low-dimensional topology code during training to generate rare road topologies. Input: a pretrained video world model (e.g., DriveDreamer-2) with encoder E, diffusion decoder D, and a differentiable renderer R that maps a topology code z ∈ ℝ^d (d=16) to a top-down road layout image. Output: a robust world model trained on adversarial topology examples.

**(B) How it works (pseudocode):**
```python
# ART training loop
# R: neural renderer with 2-layer MLP (hidden=256, ReLU) predicting 32 control points of cubic Bézier splines per lane centerline,
#     then differentiable rasterizer (SoftRas) to 256x256 top-down layout image.
# Prior: Multivariate Gaussian fitted on z0 from training set; mean μ, diagonal covariance Σ.
# lr = 0.01, K = 5, eps = 0.1

given: pretrained world model W (E, D, R)
given: dataset of video clips x and initial topology codes z0 (from HDMap generator)
for each iteration:
  sample real video x, initial topology code z = z0
  # Step 1: Compute loss with current z
  layout = R(z)                     # differentiable, e.g., spline-based rendering
  latent = E(x)
  loss = diffusion_loss(D(latent, condition=layout), target_noise)
  # Step 2: Adversarial perturbation (PGD with K=5, eps=0.1)
  z_adv = z
  for k in range(K):
    # gradient ascent on loss w.r.t z, clamp to prior range via tanh
    z_adv = z_adv + 0.01 * sign(gradient(loss, z_adv))
    z_adv = tanh((z_adv - μ) / sqrt(diag(Σ))) * sqrt(diag(Σ)) + μ   # clamp to μ ± 3σ
    layout_adv = R(z_adv)
    loss_adv = diffusion_loss(D(latent, condition=layout_adv), target_noise)
    # track best adversarial code that maximizes loss
  z_best = argmax over steps of loss_adv
  # Step 3: Update world model on adversarial condition
  layout_best = R(z_best)
  train_step(W, x, condition=layout_best, loss=diffusion_loss)
  # Every 1000 iterations, update prior: recompute μ, Σ from last 100 successful adversarial codes (loss_adv > 0.5) + original codes
```

**(C) Why this design:** We made three key design decisions. First, using a low-dimensional topology code (d=16) instead of full layout images enables efficient adversarial search via gradient ascent, accepting the trade-off that very intricate topologies (e.g., irregular lane markings) may not be representable. Second, we employ projected gradient descent (PGD) with a tanh clamp to a prior distribution learned from real road data, rather than random sampling; this directly targets model weaknesses, but risks generating unrealistic codes if the prior is too restrictive—we mitigate by periodically updating the prior with successful adversarial codes (every 1000 iterations, using codes with loss_adv > 0.5). Third, we update the world model on the adversarial condition combined with real data, rather than only augmenting the dataset, to force online adaptation; this may cause catastrophic forgetting, which we counteract by mixing real and adversarial samples in a 4:1 ratio.

**(D) Why it measures what we claim:** The diffusion loss on adversarial topology codes (loss_adv) measures the model's vulnerability to rare road structures because it quantifies prediction error under conditions unseen in training; this assumption fails when the adversarial topology is physically unrealistic (e.g., disconnected lanes), in which case the loss reflects out-of-distribution input rather than genuine topology rarity. The gradient of the loss w.r.t. z (sign-gradient in PGD) measures the sensitivity of the model to topology changes, because it indicates which parameter perturbations increase error most; this assumption fails when the renderer R is not smooth or gradients are noisy, leading to misleading step directions. The clamp-to-prior (tanh) ensures physical plausibility by restricting z to values observed in real road designs; this assumption fails when the prior is too narrow, in which case it prevents generation of truly novel topologies, limiting the augmentation diversity. Each component operationalizes a concept: loss_adv → vulnerability, gradient → sensitivity, prior clamp → plausibility. To further validate the code coverage, we compute reconstruction error of R on a held-out set of 1000 layout images from Waymo: report mean IoU and FID; if IoU < 0.7 or FID > 100, the 16-d code may not capture rare topologies, and we would increase d to 128.

## Contribution

(1) A novel framework for adversarial data augmentation in video world models via differentiable rendering of road topologies, enabling automatic generation of rare but physically plausible road structures. (2) A demonstration that adversarially perturbing a low-dimensional topology code produces challenging scenarios that improve downstream perception tasks (e.g., 3D detection) when used for training. (3) An open-source differentiable road renderer and adversarial training pipeline for driving world models.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | nuScenes | Standard driving video benchmark; with 28130 training clips (20s each) and 6019 validation clips. |
| Primary metric | MSE on adversarially perturbed topology frames (layout from R) | Directly measures robustness on failure modes. Also report FID between generated and real video frames on held-out topologies. |
| Baseline 1 | DriveDreamer-2 | Original pretrained model, no augmentation. |
| Baseline 2 | Stable Video Diffusion (SVD) | Pretrained general video model (version 1.1). |
| Baseline 3 | GAIA-1 | Autoregressive world model (350M parameters). |
| Ablation-of-ours | ART w/o adversarial (random only) | Same ART training but replace PGD step with random sampling of z from prior (uniform in [-3σ, 3σ]). |
| Additional ablation | Non-differentiable rendering baseline | Replace R with a fixed pretrained semantic segmentation network (DeepLabV3+) that maps z to layout via nearest neighbor retrieval from a codebook of 512 layout images from nuScenes; no gradient flows to z. Adversarial search uses finite differences (5 random perturbations per step). |
| Held-out city dataset | Carla Town 05 (20 video clips) + Waymo Open (50 clips) | Not seen during training; test generalization. |
| Diagnostic | Gradient smoothness | Compute cos-similarity between ∇loss(z) and ∇loss(z+δ) for δ~N(0,0.01I); report proportion of 100 steps where similarity < 0.5. |

### Why this setup validates the claim

This experimental design forms a falsifiable test of the claim that adversarial topology augmentation improves robustness to rare road structures. The nuScenes dataset provides a diverse set of real driving videos with varying topologies but is known to lack extremely rare configurations, making it ideal for testing generalization. Using MSE on video frames conditioned on adversarially perturbed topology codes directly measures prediction error under precisely the failure mode we aim to address. The baselines isolate different mechanisms: DriveDreamer-2 shows the impact of no augmentation; SVD tests if a large-scale pretrained model still struggles with driving-specific rare topologies; GAIA-1 examines autoregressive models. The ablation (random augmentation) determines whether the adversarial search is essential or if random variety suffices. The additional non-differentiable rendering baseline isolates the benefit of gradient-based search. The held-out city dataset verifies generalization to new domains. The diagnostic measures gradient smoothness to interpret when adversarial search fails. The metric detects improvements specifically on rare topologies if our method works as intended, providing a clear causal signal.

### Expected outcome and causal chain

**vs. DriveDreamer-2** — On a case where the road contains an unusually sharp, multi-lane merge (rare in nuScenes), the baseline produces high frame prediction error because its diffusion decoder never saw such a topology during training and fails to extrapolate. Our method handles this correctly because the adversarial training explicitly generated and trained on topologies near the decision boundary, so the model learns to accommodate such structures. We expect a noticeable gap on the highest-loss adversarial samples (e.g., 30% lower MSE) but parity on common topologies.

**vs. Stable Video Diffusion (SVD)** — On a case with a sudden road narrowing (uncommon pattern), SVD generates blurry or unstable future frames because its general video knowledge does not capture driving-specific dynamics under unusual road layouts. Our method maintains sharpness because the adversarial topology augmentation forces the renderer to produce layouts that challenge the diffusion process, and the model adapts. We expect SVD to show high variance on rare topologies, while ART yields consistently low MSE, with a 20% advantage on the adversarial subset.

**vs. GAIA-1** — On a case involving a non-standard intersection (e.g., five-way fork), GAIA-1 produces inconsistent vehicle trajectories due to its autoregressive nature compounding errors from unseen layout tokens. Our method avoids this because the differentiable renderer directly conditions the diffusion process on the topology code, making the model robust to the whole layout at once. We expect GAIA-1 to have 15% higher MSE on samples with the most extreme topology codes, while ART maintains performance close to that on in-distribution conditions.

### What would falsify this idea
If ART shows uniform MSE improvement across all topology subsets (including common ones) rather than concentrated gains where the predicted failure mode occurs (i.e., on the most extreme adversarial codes), then the central claim that adversarial augmentation specifically targets rare topologies is wrong. Alternatively, if the ablation (random augmentation) performs as well as ART, then the adversarial search is unnecessary. If the non-differentiable baseline matches ART, gradient-based search is not beneficial. If gradient smoothness is low (<0.5) in >20% of steps, the adversarial directions may be unreliable.

## References

1. DriveDreamer-2: LLM-Enhanced World Models for Diverse Driving Video Generation
2. Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets
3. VideoPoet: A Large Language Model for Zero-Shot Video Generation
4. SparseFusion: Distilling View-Conditioned Diffusion for 3D Reconstruction
5. Latent Video Diffusion Models for High-Fidelity Long Video Generation
