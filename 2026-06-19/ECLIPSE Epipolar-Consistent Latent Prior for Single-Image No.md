# ECLIPSE: Epipolar-Consistent Latent Prior for Single-Image Novel View Synthesis

## Motivation

Existing single-image novel view synthesis methods, such as SceneCompleter, treat the input image as complete and attempt to inpaint occluded regions in 2D, leading to inconsistent 3D geometry because they lack an explicit representation of the distribution of occluded scene parts. This meta-gap arises across multiple approaches that condition generation solely on a single reference image without modeling unobserved content.

## Key Insight

Epipolar geometry provides a deterministic constraint between corresponding points in two views, so by learning a structured latent prior over occluded regions and enforcing epipolar consistency during decoding, we can generate novel views that are both plausible and geometrically coherent.

## Method

(A) What it is: ECLIPSE is a conditional variational autoencoder (VAE) that takes a single reference image I_ref and a target camera pose T as input, and outputs a novel view I_nov. It explicitly models the distribution of occluded scene parts via a latent variable z, which is decoded into a representation of the unseen geometry and appearance.

(B) How it works (pseudocode):
```python
# Training loop for ECLIPSE
for (I_ref, I_nov_gt, T_ref, T_nov, K, depth_ref) in dataset:
    # Encode visible scene from reference
    visible_features = encoder(I_ref)  # shape: (H/8, W/8, D), D=512

    # Sample latent for occluded parts
    prior_input = concat(visible_features_flat, relative_pose_vec(T_ref, T_nov))  # flatten features, concatenate 6-DoF pose
    z_mean, z_logvar = prior_net(prior_input)  # 3-layer MLP, hidden=256, ReLU
    z = reparameterize(z_mean, z_logvar)  # shape: (latent_dim=128,)

    # Decode latent into occluded representation (3D feature volume)
    occluded_volume = decoder(z)  # 3D ResNet-18 upscaling to (H/8, W/8, D)

    # Compute visibility mask: project reference depth into target view, threshold
    target_coords = project(depth_ref, T_ref, T_nov, K)  # (2, H, W)
    occlusion_mask = (depth_ref > 0) & (target_coords within bounds)  # binary mask: 1 where visible
    occlusion_mask = resize(occlusion_mask, (H/8, W/8))  # match feature resolution

    # Combine visible and occluded: occluded contribution zeroed where visible
    full_volume = visible_features + occluded_volume * (1 - occlusion_mask.unsqueeze(-1))
    # Apply cross-attention: query=full_volume, key=visible_features, value=full_volume, 8 heads
    full_volume = cross_attention(full_volume, visible_features, visible_features) + full_volume

    # Render novel view using differentiable ray marching
    I_nov_pred = render(full_volume, T_nov, K)  # via NeRF-style MLP, 4-layer, hidden=256

    # Losses
    recon_loss = L1(I_nov_pred, I_nov_gt) + 0.1 * LPIPS(I_nov_pred, I_nov_gt)
    kl_loss = KL_divergence(z_mean, z_logvar)  # prior: N(0,I)
    # Epipolar consistency: for each pixel p in I_nov_pred, compute its epipolar line in I_ref
    # Sample 16 points along line, extract DINOv2 features (layer 6), compute cosine distance
    epipolar_loss = epipolar_consistency(I_ref, I_nov_pred, T_ref, T_nov, K, feature_net=DINOv2_vitb8)

    total_loss = recon_loss + 0.01 * kl_loss + 1.0 * epipolar_loss
    optimize(total_loss, lr=1e-4, Adam)
```
Hyperparameters: beta=0.01, gamma=1.0; prior_net is a 3-layer MLP with hidden dim 256, ReLU; decoder is a 3D ResNet-18 with transposed convolutions, output channels 512; encoder is pretrained DINOv2 ViT-B/8 (frozen, only visible_features from last layer); render MLP: 4-layer, hidden=256, ReLU, output RGB; latent_dim=128; training: batch size 8, 200k steps, 4 A100 GPUs, ~72 hours.

(C) Why this design: We chose a VAE over a GAN because the VAE's explicit likelihood modeling allows us to enforce a structured latent prior that captures multiple plausible completions, whereas GANs often mode-collapse and produce deterministic inpaintings. We chose an epipolar consistency loss over a cross-view discriminator because the epipolar constraint is a geometric invariant that must hold exactly for correct 3D structure, whereas a discriminator only captures statistical consistency and may overlook fine-grained geometric errors. We chose to separate visible and occluded representations explicitly using a visibility-gated mask rather than a single decoder, because this forces the latent variable to contribute only to occluded regions, preventing it from wasting capacity on already-observed content. The trade-off is that our method requires known camera poses, intrinsics, and depth for masking, limiting applicability to datasets where these are available; we obtain depth from a pretrained model (MiDaS) during training. Compared to SceneCompleter, which uses a dual-stream diffusion without explicit latent prior for occluded parts, ECLIPSE directly models the distribution of unseen content and enforces geometric consistency during training.

(D) Why it measures what we claim: The KL divergence measures the information content of z relative to N(0,I) under the assumption that the true prior is N(0,I). This assumption fails when the true prior is non-Gaussian, in which case KL still regularizes but may suppress multimodal completions. We test this assumption by fitting a mixture-of-Gaussians prior (MoG, 5 components) and comparing LPIPS (see experiment). The epipolar consistency loss measures geometric correctness via DINOv2 feature distance, assuming that DINOv2 features are invariant to viewpoint but sensitive to shape. Failure mode: if DINOv2 features are scene-agnostic (e.g., uniform), the loss may not penalize geometric errors. We add a synthetic experiment with known geometric error (random warping) to verify correlation with depth RMSE (see experiment). The visibility-gated mask ensures that z exclusively models occluded content, under the assumption that projected depth accurately identifies visible pixels; when depth is noisy, the mask may be incorrect, leading to leakage. We use depth from MiDaS and verify with ground-truth depth on a subset.

## Contribution

(1) A novel conditional VAE framework for single-image novel view synthesis that explicitly models the distribution of occluded scene parts via a structured latent prior. (2) The introduction of an epipolar consistency loss that enforces geometric coherence between the reference and generated views, bridging the gap between generative inpainting and 3D consistency. (3) A demonstration that explicitly separating visible and occluded representations improves generalization and reduces artifacts in novel views.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|-----------------------|
| Dataset | RealEstate10K | Large, diverse, known camera poses |
| Primary metric | LPIPS | Perceptual quality sensitive to occlusions |
| Baseline | SceneCompleter | Dual-stream diffusion; lacks explicit latent |
| Baseline | latentSplat | Variational Gaussians; no epipolar constraint |
| Baseline | PixelNeRF | Single-view NeRF; ambiguous for occluded parts |
| Ablation-of-ours | ECLIPSE w/o epipolar loss | Tests importance of geometric consistency |
| Ablation-of-ours | ECLIPSE w/o visibility mask | Tests effect of mask on latent disentanglement |
| Ablation-of-ours | Unified VAE (no separate occluded volume) | Isolates benefit of separating visible/occluded (reviewer request) |
| Ablation-of-ours | ECLIPSE with MoG prior | Tests sensitivity to prior assumption (reviewer request) |
| Additional experiment | Noisy pose (add Gaussian noise σ=5° rotation, 0.1 trans) | Measures robustness to pose errors (reviewer request) |
| Additional experiment | Synthetic geometric error (random warp 5px) | Verifies epipolar loss correlates with depth RMSE (reviewer request) |
| Supplementary | Toy dataset (ShapeNet cars, 10 categories) | Lowers entry barrier for verification (reviewer request) |

### Why this setup validates the claim
RealEstate10K provides diverse scenes with known camera poses, enabling evaluation of novel view synthesis under various occlusions. LPIPS measures perceptual fidelity, directly reflecting the quality of generated occluded content. SceneCompleter tests the sub-claim that explicit latent modeling is beneficial; latentSplat tests the role of epipolar consistency; PixelNeRF tests the advantage of variational inference over deterministic prediction. The ablations isolate the effects of the epipolar loss, visibility mask, and separation of representations. The MoG prior ablation tests the assumption that KL divergence under a Gaussian prior captures multimodal occluded distributions. The noisy pose experiment tests robustness, and the synthetic geometric error experiment validates that epipolar loss penalizes geometric inaccuracies. The toy dataset provides a fast verification of the method's core components.

### Expected outcome and causal chain

**vs. SceneCompleter** — On a scene with a large occluded region (e.g., a car hiding part of a building), SceneCompleter’s joint diffusion averages over plausible completions, yielding blurry textures. Our VAE samples a latent z that captures the occluded distribution, producing sharp details. We expect LPIPS >0.05 lower on heavily occluded views but similar on near-unoccluded ones. The unified VAE ablation (no separate occluded volume) will perform worse than ECLIPSE but better than SceneCompleter, confirming that separation provides marginal gain.

**vs. latentSplat** — When the target view requires extrapolating beyond the reference field of view (e.g., >30° rotation), latentSplat’s learned Gaussians fail to represent unseen geometry, causing foggy artifacts. Our epipolar-constrained model infers correct structure, yielding sharper renderings. Expect LPIPS gap of ~0.04 on wide-baseline pairs, parity on narrow. The w/o epipolar loss ablation will show degraded geometry but similar texture, confirming the loss's role.

**vs. PixelNeRF** — On a scene with complex occluders (e.g., a lattice), PixelNeRF’s deterministic prediction collapses to a mean, losing fine details. Our VAE samples plausible completions, preserving high-frequency information. Expect LPIPS improvement of ~0.03 on such patterns, less on simple ones.

**Ablation (w/o epipolar loss)** — Without epipolar consistency, the model may produce geometrically implausible views (e.g., misaligned edges) with low LPIPS but high geometric error. Expect a noticeable LPIPS drop (e.g., 0.02) on large motions, indicating the loss enforces structural correctness.

**Ablation (w/o visibility mask)** — Without the mask, z may encode visible content, reducing quality on occluded regions. Expect LPIPS increase of ~0.02 on heavily occluded views compared to full model.

**Ablation (Unified VAE)** — A single encoder-decoder without separate occluded volume will have higher LPIPS (~0.03) on occluded regions, supporting the benefit of separation.

**Ablation (MoG prior)** — Using a mixture-of-Gaussians prior (5 components) improves LPIPS by ~0.01 on multimodal scenes (e.g., multiple plausible completions), but KL loss increases slightly. This confirms that Gaussian prior assumption limits expressivity but does not break performance.

**Additional: Noisy pose** — Adding Gaussian noise (σ=5° rotation, 0.1 translation) degrades LPIPS by ~0.02 for ECLIPSE and baselines; ECLIPSE degrades less due to epipolar constraint's robustness. Expect ECLIPSE to maintain advantage.

**Additional: Synthetic geometric error** — Random warping (5px) of the generated view increases epipolar loss by 0.15 and depth RMSE by 0.3, confirming that epipolar loss correlates with geometric accuracy (Pearson r>0.8).

**Toy dataset (ShapeNet)** — On ShapeNet cars, ECLIPSE achieves LPIPS 0.08 vs. SceneCompleter 0.12, demonstrating feasibility with 50K training steps (4 hours on 1 GPU).

### What would falsify this idea
If the LPIPS gain of ECLIPSE over baselines is uniform across all image subsets (e.g., constant 0.01) rather than concentrated on scenes with large occlusions or wide viewpoint changes, the claim that explicit occlusion modeling drives improvement would be falsified. Additionally, if the ablation without visibility mask performs similarly to the full model, the load-bearing assumption about latent disentanglement would be invalid.

## References

1. SceneCompleter: Dense 3D Scene Completion for Generative Novel View Synthesis
2. latentSplat: Autoencoding Variational Gaussians for Fast Generalizable 3D Reconstruction
3. CameraCtrl: Enabling Camera Control for Text-to-Video Generation
4. ReconX: Reconstruct Any Scene From Sparse Views With Video Diffusion Model
5. 3D Reconstruction with Spatial Memory
6. Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets
7. MagicVideo: Efficient Video Generation With Latent Diffusion Models
8. Latent Video Diffusion Models for High-Fidelity Long Video Generation
