# Direct3DLatent: Learning 3D-Native Latent Representations for Single-View Reconstruction

## Motivation

Existing feedforward methods like FLAT rely on pre-trained 2D video diffusion latents, which inherit a pixel-space bias and lack geometric precision. This structural mismatch forces the decoder to learn a mapping from a 2D latent to 3D primitives, limiting geometric accuracy and requiring large-scale video pretraining. We propose to replace the frozen 2D latent encoder with a trainable 3D-native encoder that directly outputs a set of 3D Gaussian primitives from a single image, enabling end-to-end geometric supervision via differentiable rendering.

## Key Insight

Encoding directly into a 3D primitive space (Gaussian parameters) eliminates the domain gap between 2D latents and 3D geometry because the renderer's gradient provides a direct signal to enforce multi-view consistency without relying on a separate diffusion prior.

## Method

Direct3DLatent is a feedforward model that takes a single RGB image and outputs a set of 3D Gaussian primitives. It uses a transformer-based encoder to predict Gaussian parameters directly, without relying on a pre-trained video diffusion model. The output Gaussians can be rendered from arbitrary viewpoints using a differentiable rasterizer (3D Gaussian Splatting rasterizer, adapted from Kerbl et al. 2023, with depth and normal rendering).

**Forward pass:**
```python
import torch

def forward(input_image, target_cameras=None):
    # Encoder: extract features using a CNN backbone + transformer
    features = backbone(input_image)  # ResNet-50 (pretrained on ImageNet, frozen up to block3, fine-tuned block4), output (B, 1024, H/32, W/32)
    # Set prediction: use learned queries (like DETR)
    queries = torch.randn(N_QUERIES, D_MODEL)  # N_QUERIES=1024, D_MODEL=256; learnable sinusoidal positional encoding
    encoder_output = transformer_encoder(features, queries)  # 6-layer, 8 heads, feedforward dim 1024, dropout 0.1; output (B, N_QUERIES, D_MODEL)
    # Decode to Gaussian parameters: mean (3), scale (3), rotation (4 quaternion), opacity (1), color (3)
    gaussians = gaussian_head(encoder_output)  # 3-layer MLP (hidden 512, ReLU), output (B, N_QUERIES, 14)
    
    # If training with multi-view supervision: render from target cameras
    if target_cameras is not None:
        rendered_views = differentiable_rendering(gaussians, target_cameras)  # RGB and depth images (B, H, W, 3) and (B, H, W, 1)
        # Losses: photometric L1 + depth L1
        loss = L1_loss(rendered_views['rgb'], ground_truth_views['rgb']) + lambda_depth * L1_loss(rendered_views['depth'], ground_truth_views['depth'])
        # lambda_depth=0.1
        return loss
    else:
        return gaussians
```
**Training details:** Batch size 8, Adam optimizer (β1=0.9, β2=0.999), learning rate 1e-4 with cosine decay and 1000-step warmup, weight decay 1e-4. Train for 200k iterations. Bipartite matching (Hungarian algorithm) with matching cost: L2 distance on means (λ=1.0), L1 on scales (λ=0.1), angular difference on rotation quaternions (λ=0.1), L1 on opacity (λ=0.01), L1 on RGB (λ=0.1). **Load-bearing assumption:** The bipartite matching provides stable gradients for learning continuous Gaussian parameters from scratch without auxiliary losses or pretraining. **Mitigation:** We use a matching cost warm-up for the first 1000 iterations, where ground-truth Gaussians are matched to a fixed set of reference positions (uniformly sampled on a unit sphere) instead of predictions, gradually transitioning to learned parameters over 500 iterations.

**Why this design:** We chose a set prediction head over a pixel-aligned dense output because Gaussians are discrete primitives requiring variable spatial distribution; the trade-off is that set prediction requires careful bipartite matching during training (like DETR) which can be unstable but allows arbitrary topology. We selected Gaussian primitives over triangle splats (as in FLAT) because Gaussians provide smoother gradients via volumetric rendering, easing training from scratch; the cost is reduced surface sharpness for thin structures. We opted for a direct reconstruction loss (L1 + depth) instead of a generative objective (e.g., diffusion) because it provides stronger geometric gradients; the downside is dependence on multi-view supervision, which we assume is available during training (e.g., from real-world scans or synthetic datasets). This design explicitly avoids the 2D latent bottleneck by producing 3D primitives in a single forward pass.

**Why it measures what we claim:** The reconstruction loss on rendered views directly measures geometric accuracy because any error in 3D Gaussian positions, scales, or rotations translates to pixel misalignment in the rendered image. This measures **geometric consistency** under the assumption that the differentiable renderer's gradients are informative enough to correct errors; this assumption fails when the initial predictions are far from the optimum (e.g., due to poor encoder initialization), in which case the loss reflects trivial image difference rather than meaningful geometry. The depth consistency term measures **depth accuracy** under the assumption that depth from the renderer aligns with ground truth; this fails when there are occlusions or transparencies, where rendered depth may be ambiguous. By combining L1 and depth losses, we ensure that both appearance and geometry are optimized, with the depth term providing a direct proxy for 3D structure.

## Contribution

(1) A novel 3D-native encoder that predicts a set of 3D Gaussian parameters directly from a single image, bypassing the need for pre-trained 2D latents. (2) A training pipeline that uses differentiable rendering and multi-view reconstruction losses to enforce geometric accuracy, achieving state-of-the-art geometry on single-view reconstruction benchmarks. (3) Analysis showing that the learned 3D latents are more geometrically consistent than 2D diffusion latents, as measured by novel view synthesis accuracy.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | ShapeNet (13 categories, 80/20 split, 24 views per object) | Single-view with multi-view GT |
| Primary metric | Chamfer Distance (L2, 10k points) and F-score (τ=0.05) | Measures geometric accuracy directly |
| Baseline 1 | Splatter Image | Feedforward Gaussian baseline |
| Baseline 2 | FLAT | Feedforward triangle baseline |
| Baseline 3 | 3D Gaussian Splatting (3D GS) | Optimization-based multi-view (per-scene, 1000 iters) |
| Ablation-of-ours | No depth loss | Removes depth consistency term |

### Why this setup validates the claim

The central claim is that Direct3DLatent's direct 3D Gaussian prediction achieves high geometric accuracy via a set prediction head and multi-view supervision. ShapeNet provides ground truth 3D shapes, and Chamfer Distance directly evaluates point cloud fidelity, avoiding appearance confounds. F-score captures completeness and precision. Comparing against feedforward baselines Splatter Image and FLAT tests whether our transformer-based set prediction and Gaussian representation improve over existing pixel-aligned or triangle-based feedforward approaches. Including optimization-based 3D Gaussian Splatting (which uses an explicit per-scene optimization from multiple views) quantifies the gap between feedforward and optimization-based methods, revealing if our direct approach captures sufficient 3D structure without iterative refinement. The ablation (removing depth loss) isolates the contribution of geometric supervision. This combination forms a falsifiable test: if our method does not significantly outperform Splatter Image on geometry, or if the depth loss is redundant, the claim that set prediction and direct geometric loss are key to accuracy is invalidated.

### Expected outcome and causal chain

**vs. Splatter Image** — On an object with a concave interior (e.g., a bowl), Splatter Image predicts per-pixel Gaussians aligned to visible surfaces, failing to represent the interior because the encoder sees only front faces; this produces a flat depth map with missing geometry. Our method uses set prediction with global context, allowing Gaussians to occupy interior space, so the rendered depth more accurately captures concavities. We expect a noticeable gap (e.g., Chamfer distance 30% lower on concave objects, F-score 10% higher) but parity on convex shapes.

**vs. FLAT** — On a thin wireframe (e.g., a bicycle wheel), FLAT's triangle splats produce sharp edges but suffer from discrete normals, causing alignment errors during rasterization; the network struggles to fit sparse triangles to fine structures due to gradient instability. Our Gaussian representation yields smooth gradients, enabling stable optimization of scales and rotations, leading to better coverage of thin parts. We expect our Chamfer distance to be lower by ~20% on thin structures, while rendering quality may be similar.

**vs. 3D Gaussian Splatting** — On a scene with specular highlights (e.g., a chrome sphere), optimization-based 3D GS leverages multi-view images to disambiguate geometry from appearance, producing accurate depth. Our single-view feedforward method must infer geometry without multi-view consistency, so it may misinterpret highlights as geometry changes. We expect our Chamfer distance to be higher (e.g., 1.5× worse on shiny objects, F-score 15% lower) but comparable on diffuse materials, indicating the feedforward trade-off.

### What would falsify this idea

If Direct3DLatent's Chamfer distance is not significantly better than Splatter Image across object categories, or if the ablation without depth loss performs equally well on thin structures, then the central claim that set prediction and depth consistency improve geometric accuracy is falsified. Specifically, uniform gains across all subsets rather than concentration on concave/thin cases would contradict the hypothesized causal mechanism.

## References

1. FLAT: Feedforward Latent Triangle Splatting for Geometrically Accurate Scene Generation
2. Uni3C: Unifying Precisely 3D-Enhanced Camera and Human Motion Controls for Video Generation
3. Lyra: Generative 3D Scene Reconstruction via Video Diffusion Model Self-Distillation
4. I2VControl-Camera: Precise Video Camera Control with Adjustable Motion Strength
5. Motion Prompting: Controlling Video Generation with Motion Trajectories
6. HunyuanVideo: A Systematic Framework For Large Video Generative Models
7. Wonderland: Navigating 3D Scenes From a Single Image
8. AC3D: Analyzing and Improving 3D Camera Control in Video Diffusion Transformers
