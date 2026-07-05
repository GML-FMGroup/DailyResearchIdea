# Self-Aligned Autoregression for Encoder-Independent Image Synthesis

## Motivation

Existing methods like GEAR rely on a frozen pre-trained vision encoder (e.g., DINOv2) to compute the alignment loss between the tokenizer's continuous features and the generator's predictions. This introduces bias from the encoder's pretraining dataset and limits the model's ability to adapt to new domains. The root cause is that the alignment space is fixed by the external encoder, which does not co-evolve with the autoregressive generator and tokenizer during training.

## Key Insight

The autoregressive model's hidden states naturally encode semantic information that is dynamically updated during training, so using them as the alignment target eliminates the need for a fixed external encoder and enables the representation to co-adapt.

## Method

**Self-Aligned Autoregression (SAA)**

**(A) What it is:** Self-Aligned Autoregression (SAA) replaces the external vision encoder (e.g., DINOv2) in the alignment loss with the autoregressive model's own hidden state. Input: image x, VQ tokenizer (encoder E: ResNet-50 with 4x downsampling, codebook size=1024, d_codebook=256, decoder D: matching architecture), AR transformer (12-layer GPT-2, d_model=768). Output: trained tokenizer and AR generator with co-adapted representations.

**(B) How it works:**
```pseudocode
Input: image x
Hyperparameters: λ_ar=1.0, λ_align=0.1

# Forward pass through tokenizer
z_e = E(x)                              # continuous features [batch, d_codebook, H, W]
z_q, indices = quantize(z_e)            # hard discrete tokens
z_soft = soft_quantize(z_e)             # differentiable weighted sum [batch, d_codebook, H, W]

# Autoregressive branch (hard targets)
logits = AR(indices[:-1])               # predict next token
loss_ar = cross_entropy(logits, indices[1:])

# Alignment branch: use AR hidden state at last step
h_ar = AR.get_last_hidden_state()       # shape [batch, d_model]
h_proj = Linear(d_model=768, d_codebook=256)  # project to soft feature dimension
loss_align = MSE(h_proj(h_ar), stop_grad(z_soft))  # stop-gradient on tokenizer features

# Total loss and update
total_loss = λ_ar * loss_ar + λ_align * loss_align
total_loss.backward()
update(E, codebook, decoder, AR, Linear)
```

**(C) Why this design:** We chose to use the AR model's last hidden state as the alignment target (instead of an external encoder like DINOv2) because it creates a closed-loop consistency between the two core components. The trade-off is that the alignment signal depends on the AR's current state, which may be noisy early in training; however, this noise diminishes as training progresses and forces co-adaptation. We opted for a linear projection from the AR hidden state (768-dim) to the soft feature dimension (256-dim) rather than a more complex predictor) to keep the alignment loss simple and avoid overfitting. This choice accepts that the linear projection may not capture all nuances, but it preserves gradient flow and computational efficiency. We used MSE as the alignment loss (instead of cosine similarity) because MSE penalizes scale differences, which encourages the AR to learn a representation with similar magnitude to the tokenizer's features; the trade-off is that the AR must also match the average scale, which may slow convergence. To prevent representation collapse, we apply a stop-gradient on the tokenizer's soft features (z_soft) in the alignment loss, ensuring that gradients flow only into the AR branch and the linear projection, analogous to BYOL. Overall, these design decisions prioritize simplicity and direct gradient flow over expressive power.

**(D) Why it measures what we claim:** The computational quantity `loss_align` (MSE between the AR hidden state projection and the tokenizer's soft features with stop-gradient) measures the consistency between the AR's internal representation and the tokenizer's continuous output. This consistency operationalizes the concept of *encoder-independent alignment* because the AR's hidden state serves as the sole alignment reference, replacing the external encoder. The underlying assumption (A) is that the AR's hidden state projection spans the same space as the tokenizer's semantic features, enabling them to be compared via MSE. This assumption fails (F) early in training when the AR is randomly initialized or if the linear layer has insufficient capacity, in which case `loss_align` reflects a noisy mismatch rather than meaningful alignment. As training proceeds, the gradient from `loss_align` shapes the AR's hidden states to become a reliable proxy for the tokenizer's continuous features, thereby achieving encoder independence.

## Contribution

(1) We introduce Self-Aligned Autoregression (SAA), a method that replaces external vision encoders with the autoregressive model's own hidden states as the alignment target, enabling encoder-independent feature alignment in autoregressive image generation. (2) We show that using the AR's hidden state as a dynamic alignment reference co-adapts the tokenizer and generator, leading to improved domain adaptability without relying on frozen pretrained features. (3) We provide a training framework that jointly optimizes the tokenizer and AR with a consistency loss between the AR's hidden representation and the soft quantized features.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | ImageNet 256x256, train split (1.2M images) | Standard benchmark for class-conditional generation. |
| Primary metric | FID | Measures distributional fidelity of generated images. |
| Secondary metric | Inception Score (IS) | Captures image quality and diversity. |
| Baseline 1 | Two-stage VQ-AR + DINOv2 (external encoder) | Tests external encoder alignment vs. self-alignment. |
| Baseline 2 | Two-stage VQ-AR (no alignment loss) | Isolates effect of alignment loss. |
| Ablation | SAA without alignment loss (λ_align=0) | Verifies that alignment loss drives improvement. |
| Compute budget | 8× V100 GPUs, ~72 hours per run | Standard hardware for ImageNet-scale generation. |

### Why this setup validates the claim
This setup isolates the effect of replacing the external encoder with the AR's hidden state while controlling for alignment loss. Comparing SAA to the standard two-stage VQ-AR with DINOv2 alignment tests whether our self-alignment yields better or equivalent consistency. Comparing to two-stage without alignment tests if any alignment is beneficial. The ablation removes the alignment loss from SAA, degenerating to a standard autoregressive model; if SAA outperforms it, the gain is attributable to the alignment branch. FID and IS are chosen because they capture both diversity and fidelity, and are standard for comparing image generation. Additionally, we probe the AR hidden state by training a linear classifier on h_ar for 10-class ImageNet classification, measuring accuracy every 10 epochs to verify that h_ar encodes semantic content. We also compute the cosine similarity between h_proj(h_ar) and z_soft (with stop_grad) on a held-out validation set every 5 epochs to track alignment evolution. If the cosine similarity reaches above 0.5 by epoch 50, it indicates that the AR representation is becoming a reliable proxy for the tokenizer's features. These quantitative analyses directly test the underlying assumption that the AR hidden state can capture semantic features aligned with the tokenizer.

### Expected outcome and causal chain

**vs. Two-stage VQ-AR + DINOv2** — On a novel combination of shapes and textures (e.g., a zebra with a car's pattern), the two-stage method fails because the external encoder's features are not perfectly aligned with the tokenizer's codebook, causing reconstruction artifacts that propagate to generation. Our SAA method handles this because the AR's hidden state is directly targetted to match the tokenizer's soft features, forcing co-adaptation that yields more consistent representations. We expect SAA to achieve a ~5-10 point lower FID on such out-of-distribution categories, while performing similarly on common ones. The cosine similarity between h_ar and z_soft should steadily increase from ~0.2 to >0.6 over 100 epochs, indicating progressive alignment.

**vs. Two-stage VQ-AR (no align)** — On fine-grained details like fur texture or hair strands, the no-alignment baseline produces overly smooth outputs because the AR has no constraint to be consistent with the tokenizer's continuous features, leading to loss of high-frequency information. Our SAA method preserves details because the alignment loss forces the AR's hidden states to encode those fine-grained patterns present in the soft features. We expect SAA to show a noticeable FID improvement (e.g., 2-3 points) on datasets with high texture complexity, while being comparable on simple shapes. The probing accuracy on 10-class fine-grained subsets (e.g., dog breeds) should be >5% higher for SAA than the no-alignment baseline.

**Ablation: SAA without alignment loss** — Without the alignment loss, the AR only receives cross-entropy on hard tokens; the tokenizer remains independently trained. This fails when the codebook is not perfectly trained because the AR cannot correct tokenizer errors. With alignment, the AR's gradient helps shape the tokenizer, leading to better reconstruction and generation. We expect the full SAA to outperform the ablation by a clear margin (e.g., 3-5 FID points), confirming the alignment branch's contribution. The ablation's AR hidden state correlation with z_soft should stay low (<0.3) throughout training.

### What would falsify this idea
If the FID of SAA is not significantly lower than the two-stage baseline with DINOv2 alignment (or even higher), and the ablation performs similarly, then the self-alignment mechanism is not beneficial. Specifically, if the cosine similarity between h_ar and z_soft does not exceed 0.5 by epoch 50, or if the probing accuracy on a held-out 10-class set does not improve over the no-alignment baseline, the claim of co-adaptation addressing representation mismatch would be unsupported.

## References

1. GEAR: Guided End-to-End AutoRegression for Image Synthesis
2. X-Omni: Reinforcement Learning Makes Discrete Autoregressive Image Generative Models Great Again
3. VQRAE: Representation Quantization Autoencoders for Multimodal Understanding, Generation and Reconstruction
4. Orthus: Autoregressive Interleaved Image-Text Generation with Modality-Specific Heads
5. TokenFlow: Unified Image Tokenizer for Multimodal Understanding and Generation
6. mPLUG-OwI2: Revolutionizing Multi-modal Large Language Model with Modality Collaboration
7. Generative Multimodal Models are In-Context Learners
8. NExT-GPT: Any-to-Any Multimodal LLM
