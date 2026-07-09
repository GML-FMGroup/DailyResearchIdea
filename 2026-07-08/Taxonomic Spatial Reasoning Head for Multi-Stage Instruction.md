# Taxonomic Spatial Reasoning Head for Multi-Stage Instruction Tuning of Vision-Language Models

## Motivation

Existing vision-language models (e.g., Qwen2-VL, SenseNova-Vision) rely on implicit spatial reasoning learned from large-scale data, but perform poorly on complex spatial tasks that require explicit relational understanding. Scaling Spatial Intelligence shows that data curation with a spatial taxonomy improves performance, yet the model still lacks a dedicated architectural component for spatial reasoning, relying on the same unified head for both semantic and spatial outputs. This causes interference and limits the ability to capture fine-grained spatial relationships, as the model must compress spatial cues into generic token predictions.

## Key Insight

By constructing a dedicated spatial reasoning head that operates on a disentangled geometric representation of visual tokens, we can enforce a compositional inductive bias that separates spatial computation from semantic content, enabling efficient and accurate spatial generalization.

## Method

### (A) What it is
We propose **Spatial Reasoning Head (SRH)**, a lightweight transformer module that takes token-level visual features and text-conditioned query vectors, and outputs explicit spatial relation predictions (e.g., relative positions, distances, orientations). The input is a set of visual patch tokens from a dynamic-resolution encoder (e.g., Qwen2-VL) and a text embedding from the language model; the output is a set of relation logits for predefined spatial categories.

### (B) How it works
Pseudocode for the training of SRH and multi-stage procedure:

```pseudocode
# Stage 1: Verify geometric encoding (calibration)
# Train a linear probe on frozen V_geo to predict normalized patch coordinates (x,y).
# Require probe accuracy > 70% on held-out patches to proceed.

# Stage 2: Train Spatial Reasoning Head (SRH)
Input: Vision tokens V (from frozen encoder), text embedding T (from frozen LLM), 
       spatial taxonomy S (e.g., 'left', 'right', 'above', 'near', 'far', 'touching')
Hyperparameters: SRH_layers=4, SRH_dim=512, lr=1e-4, batch_size=256, 
                 probe_lr=1e-3, probe_epochs=50

# 1. Compute geometric tokens
# Instead of linear projection, concatenate normalized patch coordinates (x,y) to each token.
# Patch coordinates: grid positions normalized to [0,1] based on image height/width.
V_geo = concat(V, patch_coords)  # patch_coords shape: (num_patches, 2)
# Project to SRH_dim via linear layer (no bias)
V_geo = Linear(SRH_dim, bias=False)(V_geo)

# 2. Text-conditioned spatial queries
Q = linear(T)  # project T to SRH_dim, linear with bias

# 3. Cross-attention between queries and geometric tokens
for layer in 1..SRH_layers:
    Q = MHA(Q, V_geo) + FFN(Q)  # standard transformer, MHA has 8 heads, FFN hidden dim = 2048, GeLU

# 4. Predict relations: aggregate Q tokens to one vector via mean pooling, then linear head to |S| logits.
relation_logits = Linear(|S|, bias=True)(mean_pool(Q))

# Loss: cross-entropy with ground truth relation labels (from spatial taxonomy)
loss = CrossEntropy(relation_logits, y)

# Stage 3: Joint fine-tuning
Unfreeze encoder and LLM (lr=1e-5), continue training with spatial data + general instruction data 
(mix ratio 1:1) to integrate SRH features into language decoder. Use gradient checkpointing, batch_size=128.

Estimated compute: 128 GPU hours on A100 for full training (stages 2+3).
```

### (C) Why this design
We chose a separate lightweight head over integrating spatial relations directly into the language model because decoupling spatial computation avoids interfering with the language model's semantic capabilities; the cost is additional parameters (SRH: ~6M params) and inference latency that scales linearly with the number of relation types (inference adds ~2ms per image on A100). We chose cross-attention over concatenation to allow dynamic alignment between text queries and visual regions, accepting that attention computes pairwise interactions which may be quadratic in token count (but token count is typically 256-1024). We chose multi-stage training (freeze base, train head, then joint fine-tune) over end-to-end training from scratch because the base model's pretrained features are crucial and we want to prevent catastrophic forgetting; the trade-off is that the first stage may not perfectly align with the head. Unlike prior work like Scaling Spatial Intelligence which relies solely on data curation, our method adds a structural inductive bias that explicitly computes spatial relations, avoiding the need for the base model to learn implicit spatial reasoning from scratch.

### (D) Why it measures what we claim
The cross-entropy loss on relation_logits measures the model's ability to predict explicit spatial relations from geometric tokens because we assume that the concatenation of normalized patch coordinates to visual tokens (V_geo) captures sufficient geometric information (e.g., relative positions of tokens) for relation identification; this assumption fails when the image contains fine-grained textures that dominate token embeddings, in which case the loss reflects texture-based shortcuts rather than genuine spatial reasoning. The linear probe calibration (Stage 1) verifies that V_geo encodes spatial geometry: if probe accuracy on patch coordinate prediction is low, we increase the weight of coordinates or retrain encoder with spatial loss. The text-conditioned queries Q encode the question's spatial intent, and the cross-attention weights measure the relevance of each visual region to that intent; we assume that the attention distribution is a faithful proxy for the region's contribution to the spatial relation, but this fails when the question is ambiguous or the model attends to spurious correlations (verified by attention rollout analysis). The multi-stage training ensures that the SRH learns spatial features without overriding the base model's general knowledge; we assume that the frozen base model's representations are sufficiently rich for spatial tasks, but this assumption fails when the base model's encoder lacks necessary geometric granularity (e.g., low-resolution tokens), in which case the SRH compensates by learning from limited spatial cues rather than truly understanding 3D structure.

## Contribution

(1) A novel Spatial Reasoning Head (SRH) that explicitly predicts spatial relations from token-level visual features, incorporating a geometric projection and text-conditioned cross-attention. (2) A multi-stage instruction-tuning procedure that preserves base model capabilities while adding dedicated spatial reasoning, with a taxonomy-driven data curation step. (3) Design principle: decoupling spatial computation from semantic content via a separate head with compositional inductive bias prevents interference and improves spatial generalization.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | VSI-Bench | Covers diverse spatial relation types |
| Primary metric | Spatial relation accuracy | Directly measures explicit spatial reasoning |
| Baseline 1 | Qwen2-VL | Strong VLM without explicit spatial head |
| Baseline 2 | Scaling Spatial Intelligence | Data-only approach, no architectural bias |
| Ablation-of-ours | SRH w/o multi-stage training | Tests necessity of staged training |
| Additional analysis | Linear probe on V_geo | Verifies geometric content in frozen tokens |

Estimated compute: 128 GPU hours on A100 for full training (stages 2+3).

### Why this setup validates the claim
This experimental design isolates the effect of our explicit spatial head. VSI-Bench provides a focused test of spatial relation understanding across varied scenarios. The primary metric (accuracy on spatial relation prediction) directly measures the skill we aim to improve. Comparing against Qwen2-VL tests whether our architectural addition outperforms a strong generic VLM; comparing against Scaling Spatial Intelligence tests whether our structural inductive bias is superior to data-only augmentation. The ablation (training from scratch) tests the importance of multi-stage training in preserving base model capabilities. Additionally, we include a linear probe analysis on frozen V_geo (using 512 held-out patches) to verify that our geometric token representation indeed encodes spatial layout independent of semantics; we expect probe accuracy >70% on coordinate prediction. Together, these choices form a falsifiable test: if our method outperforms baselines on spatial accuracy but not on general VQA, the claim that explicit spatial computation helps is supported; if the ablation matches our full method, multi-stage is unnecessary.

### Expected outcome and causal chain

**vs. Qwen2-VL** — On a case where the required spatial relation involves relative distances (e.g., "is the car left of the tree?"), Qwen2-VL may produce incorrect relation because its implicit architecture can confuse semantic cues with spatial ones. Our method instead computes geometric tokens from patch coordinates and uses cross-attention conditioned on the text query, enabling precise spatial reasoning. We expect a noticeable gap on relation accuracy for spatial queries (e.g., >10%) but parity on general VQA tasks.

**vs. Scaling Spatial Intelligence** — On a case where the spatial arrangement is rare in training data (e.g., objects stacked in an unusual orientation), Scaling Spatial Intelligence fails because it relies solely on data distribution without structural priors. Our method generalizes via explicit geometric token representation that encodes relative positions. We expect a larger gain on out-of-distribution spatial subsets (e.g., >15%) compared to in-distribution ones, where data augmentation also helps.

### What would falsify this idea
If our method's accuracy improvement is uniform across all spatial relation types, or if the ablation (no multi-stage) achieves similar gains, then the central claim that explicit spatial reasoning heads are beneficial would be falsified. Additionally, if the linear probe on V_geo shows low accuracy (<50%), the geometric encoding assumption is violated.

## References

1. Vision as Unified Multimodal Generation
2. Scaling Spatial Intelligence with Multimodal Foundation Models
3. Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution
