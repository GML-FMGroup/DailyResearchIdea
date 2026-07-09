# 3D-Aware Autoregressive Vision-Language Model via Voxel Token Prediction

## Motivation

Existing vision-language models (e.g., Qwen2-VL) achieve strong 2D understanding but lack single-view 3d attribute prediction because they do not incorporate a 3D output module. While G2VLM added a geometry module, it required separate training stages and could not generate 3D outputs autoregressively within the same sequence, limiting its use for instruction-following tasks that require 3D predictions. The root cause is that these models treat 3D as an auxiliary module rather than integrating it into the generative token space, preventing the model from learning joint representations that align visual cues with 3D structures.

## Key Insight

The visual encoder's latent representations are already predictive of 3D shape; by introducing a discrete voxel token vocabulary and training the model to autoregressively generate these tokens from image features, we can leverage the existing language modeling objective to learn 3D reconstruction without task-specific decoders.

## Method

### (A) What it is
**Voxel-Token VLM (VT-VLM)** extends a pretrained autoregressive VLM (LLaVA-7B, visual encoder CLIP ViT-L/14, language decoder 7B-parameter transformer) by adding a fixed set of special tokens representing 3D voxel occupancy at a compressed resolution via a pre-trained VQ-VAE. The model is trained to generate sequences of voxel latent tokens conditioned on an input image and optional text, enabling single-view 3D prediction within the same autoregressive framework.

### (B) How it works
```python
# Stage 1: Prepare voxel token vocabulary.
# Pre-train a 3D VQ-VAE on ShapeNet to compress voxel grids (32^3) into token sequences of length M=512 with codebook size K=512.
# Load-bearing assumption: VQ-VAE latent codes preserve sufficient detail (reconstruction IoU >= 90% on validation set; otherwise increase latent resolution to 16^3 or codebook to 1024).
Vocab: original VLM tokens + K special tokens representing VQ-VAE latent codes.

# Stage 2: Training data format.
Each sample: (image I, text prompt P, target sequence S_target).
For 3D reconstruction: P = "Generate the 3D voxel shape of the object.", S_target = [latent_code_1, ..., latent_code_M] (each from codebook).
For language tasks: standard text tokens.

# Stage 3: Unified autoregressive training.
# Freeze visual encoder? No, fine-tune entire model but weight 3D token loss by lambda=0.5.
# Training hyperparameters: batch size 64, learning rate 1e-5, AdamW, 50k steps, mix ratio 1:1 3D:language data.
logits = model(image, text_prefix)  # forward through VLM
loss = cross_entropy(logits, target_sequence, reduction='none')
loss = loss * mask  # mask padding
loss = loss.sum() / mask.sum()
backward()

# Inference: For single-view image, prompt with "Generate the 3D shape.", decode latent tokens, then VQ-VAE decoder to get voxel grid.
```

### (C) Why this design
We chose to compress 3D voxel grids via a VQ-VAE (with codebook size 512 and latent grid 8x8x8) over direct autoregressive generation of binary voxel tokens (V=32768) because the latter would produce excessively long sequences (32K tokens) and disrupt the model's ability to maintain coherence across text and 3D segments, accepting the cost of extra pretraining and potential reconstruction error. The VQ-VAE is pre-validated to achieve reconstruction IoU >= 90% on ShapeNet; if not, we increase latent resolution to 16^3 or codebook to 1024. We chose to fine-tune the full VLM (rather than freezing the visual encoder) because freezing reduces the model's capacity to align visual features with the new 3D token space, but at the risk of forgetting some language capabilities; we mitigate this by mixing a large portion of text-only data (1:1 ratio). We chose a uniform cross-entropy loss on both text and 3D tokens (with a scaling coefficient lambda=0.5) over a two-stage training to encourage joint representations and avoid catastrophic forgetting. The VQ-VAE compression ratio (8x8x8 from 32^3) reduces the 3D token sequence to 512, which is comparable to typical sentence lengths, making integration seamless.

### (D) Why it measures what we claim
The cross-entropy loss on 3D latent tokens measures how well the model predicts the correct compressed shape representation given the image, because the loss directly compares predicted latent distribution to the ground-truth latent index. This operationalizes 3D understanding as successful reconstruction in a compact space. The assumption is that the VQ-VAE latent code captures sufficient geometric detail to evaluate single-view 3D reasoning; this assumption is validated by requiring VQ-VAE reconstruction IoU >= 90%, otherwise the metric would reflect compression error rather than model reasoning. The language cross-entropy on text tokens preserves the original modality coverage and prevents forgetting; this measures general VLM capability via next-token prediction, relying on the assumption that the text data distribution is representative; this assumption fails when the text data distribution shifts away from the test domain, leading to overestimated language retention. We additionally measure VQ-VAE reconstruction IoU on test voxels as a sanity check to disentangle compression error from model prediction error.

## Contribution

(1) A novel integration of 3D information into autoregressive VLMs by introducing a compressed voxel token vocabulary via VQ-VAE, enabling single-view 3D prediction within the same generation framework. (2) A unified training strategy that jointly optimizes language and 3D token prediction using a single cross-entropy loss with a balancing coefficient, showing that joint training improves 3D accuracy without degrading language performance. (3) Empirical evidence that a pretrained visual encoder already encodes 3D cues, as the model can learn to generate 3D tokens from image-only input with minimal architectural changes.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | ShapeNet renderings (single-view RGB, ground truth 32^3 voxels) | Controlled evaluation of single-view 3D reconstruction. |
| Primary metric | 3D IoU on voxel grid (decoded from VQ-VAE latent) | Measures geometric accuracy directly; we also report VQ-VAE reconstruction IoU as sanity check. |
| Baseline 1 | Pix2Vox (task-specific) | Strong 3D reconstruction baseline. |
| Baseline 2 | Unified-IO (direct voxel generation without compression) | Unified model without compression tokens. |
| Baseline 3 (new) | VT-VLM with continuous latent regression (regress a 512-dim vector instead of discrete tokens) | Isolates benefit of discrete token prediction over regression in autoregressive framework. |
| Ablation 1 | VT-VLM w/o language mixture (trained only on 3D data) | Isolates benefit of joint training. |
| Ablation 2 (new) | VQ-VAE reconstruction quality: compute IoU between ground-truth voxels and VQ-VAE decoded voxels from ground-truth latents | Disentangles compression error from model prediction error. |

### Why this setup validates the claim
This design creates a falsifiable test of two sub-claims: (1) that compressed 3D latent tokens enable accurate single-view reconstruction within an autoregressive VLM, and (2) that joint training with language preserves general capabilities. ShapeNet provides controlled single-view images with high-quality ground-truth voxels, making IoU a direct measure of 3D understanding. Pix2Vox serves as a strong task-specific baseline: if our method matches or exceeds it, the compressed token approach is viable for 3D. Unified-IO tests whether our VQ-VAE compression provides an advantage over direct voxel sequence generation. The continuous latent regression baseline (Baseline 3) tests whether discrete tokens are beneficial over a continuous regression head within the same autoregressive framework. The ablation without language mixture isolates the effect of joint training. The VQ-VAE reconstruction quality ablation ensures that the primary IoU metric reflects model reasoning rather than compression artifacts. Additionally, we conduct a probing experiment to measure mutual information between visual features (CLIP embeddings) and ground-truth voxel grids using a linear probe, directly testing the claim that visual encoders contain 3D shape information (groundedness). Language perplexity on a held-out text corpus (e.g., WikiText-2) is monitored to ensure retention of language capabilities.

### Expected outcome and causal chain

**vs. Pix2Vox** — On a shape with fine details (e.g., a chair with slender legs and a curved back), Pix2Vox often produces a smooth, missing-leg volume because its deterministic encoder-decoder lacks global context to infer occluded parts. Our method instead leverages the VLM's ability to reason about the entire image and the compressed latent code that captures holistic shape features, enabling it to reconstruct the thin legs. Thus, we expect a noticeable gap on complex shapes (IoU gain >10%) but near-parity on simple, symmetric objects (e.g., sphere).

**vs. Unified-IO** — On a scene where the object occupies a small fraction of the image (e.g., a remote control on a table), Unified-IO's direct 32^3 voxel output suffers from the long sequence (32K tokens) diluting attention to the relevant region, causing missing or distorted predictions. Our method's VQ-VAE compression reduces the sequence to 512 tokens, focusing the autoregressive process on compact latent codes that represent the object robustly. Therefore, we expect a significant gap on such cases (IoU gain >15%), while on centered large objects performance is comparable.

**vs. Continuous latent regression (Baseline 3)** — On shapes with sharp edges or fine details, the continuous regression baseline may produce blurry or averaged predictions because the MSE loss on latent vectors encourages mean predictions, whereas discrete token classification preserves sharp mode selection. We expect a moderate gap (IoU gain ~5%) on such shapes, validating the benefit of discrete tokens over regression in capturing multi-modal distributions.

**Ablation: VQ-VAE reconstruction quality** — We expect VQ-VAE reconstruction IoU (from ground-truth latents) to be >90% on average. If it falls below 90%, the model's IoU may be capped by compression error, and we would need to increase VQ-VAE capacity before claiming 3D reasoning improvements.

### What would falsify this idea
If the IoU gains over Pix2Vox are uniform across all shape complexities (not concentrated on fine-detail cases) or if the continuous regression baseline matches or exceeds VT-VLM on fine details, then the hypothesized benefit of discrete tokens is unsupported. If the ablation without language mixture achieves nearly identical 3D accuracy and language perplexity, then joint training is unnecessary, falsifying the claim of synergistic representations. If the VQ-VAE reconstruction IoU is below 90%, the primary metric is confounded by compression error, invalidating the claim that model performance reflects 3D reasoning.

## References

1. Vision as Unified Multimodal Generation
2. Scaling Spatial Intelligence with Multimodal Foundation Models
3. Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution
