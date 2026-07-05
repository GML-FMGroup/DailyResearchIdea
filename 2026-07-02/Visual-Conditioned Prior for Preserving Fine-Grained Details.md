# Visual-Conditioned Prior for Preserving Fine-Grained Details in Multimodal Variational Alignment

## Motivation

Existing variational multimodal alignment methods, such as AMVL (Multimodal Continuous Reasoning via Asymmetric Mutual Variational Learning), optimize for distribution matching in a shared latent space but do not explicitly preserve fine-grained visual features. This is a meta-level structural problem: the alignment objective itself prioritizes distribution matching over pixel-level retention, causing systematic information loss across multiple approaches. Our method addresses this by making visual preservation a structural property of the prior, so the variational objective inherently penalizes loss of visual details.

## Key Insight

By making the prior distribution a function of the visual input that reconstructs visual features, the variational objective inherently penalizes any posterior that loses visual information, because the KL divergence forces the posterior to match a prior that already encodes those details.

## Method

### (A) What it is
ViCP (Visual-Conditioned Prior) is a variational framework that replaces the standard fixed prior with a visual-conditioned prior p(z|v) trained to reconstruct the visual input. Its inputs are a visual signal v and a text signal t; its output is a latent z that preserves fine-grained visual details during multimodal alignment.

### (B) How it works (pseudocode)
```python
# Architecture specifications:
# visual_encoder: ResNet-18 (frozen, output 512-dim feature) + learned projection to 256-dim
# text_encoder: BERT-base-uncased (frozen, [CLS] token 768-dim) + learned projection to 256-dim
# posterior_network: 2-layer MLP, input 512 (concat visual+text), hidden 256, GeLU, output mean+logvar (latent_dim=128)
# prior_network: 2-layer MLP, input 256 (visual features), hidden 128, GeLU, output mean+logvar (latent_dim=128)
# visual_decoder: 3-layer deconvolution (transposed conv) upsampling to original RGB (e.g., 224x224)
# text_decoder: single linear layer mapping latent to vocabulary logits (for bag-of-words)
# Kl_loss uses closed-form Gaussian KL (diagonal covariance)

# Training loop
for each batch (v, t):
    # Encode
    visual_features = visual_encoder(v)  # 256-dim
    text_features = text_encoder(t)      # 256-dim
    # Posterior
    q_params = posterior_network(concat(visual_features, text_features))  # q(z|v,t), dim 128
    z = reparameterize(q_params)
    # Prior from visual input only
    p_params = prior_network(visual_features)  # p(z|v), dim 128
    # Reconstruct visual from z
    v_recon = visual_decoder(z)  # output same size as v
    # Reconstruct text from z (bag-of-words)
    t_recon = text_decoder(z)  # output logits over vocabulary
    # Compute losses
    recon_loss_v = MSE(v, v_recon)  # visual reconstruction loss
    recon_loss_t = CE(t, t_recon)   # cross-entropy text reconstruction
    recon_loss = recon_loss_v + recon_loss_t
    kl_loss = KL(q_params || p_params)   # closed-form Gaussian KL (diagonal)
    # Additional visual preservation loss to train prior_network
    z_prior = reparameterize(p_params)
    v_prior_recon = visual_decoder(z_prior)
    prior_recon_loss = MSE(v, v_prior_recon)
    # Total loss
    beta = 1.0, gamma = 0.1
    loss = recon_loss + beta * kl_loss + gamma * prior_recon_loss
    # Calibration: after each epoch, compute SSIM on validation set; if SSIM < 0.95, flag warning
    update all networks
```

### (C) Why this design
We chose to condition the prior exclusively on visual input (rather than on both modalities) because this forces the prior to encode visual details independently, preventing the KL from collapsing to a trivial match with a multi-modal prior. This design accepts the trade-off that the prior may not capture text-relevant variations, but those are handled by the posterior via the text input. Second, we use a separate visual decoder for prior training (shared with the main reconstruction decoder) to ensure the prior is grounded in pixel-level reconstruction, not just latent similarity. The alternative of using a fixed Gaussian prior loses visual information entirely. Third, we set the KL weight beta and prior reconstruction weight gamma as hyperparameters: beta balances detail preservation vs. alignment flexibility, while gamma ensures the prior is a good visual model. This explicit two-weight scheme avoids the need for auxiliary contrastive losses that would decouple the prior from the variational objective.

### (D) Why it measures what we claim
The KL divergence KL(q(z|v,t) || p(z|v)) measures visual detail retention because it quantifies how much the posterior deviates from a prior that is trained to reconstruct visual features; under the assumption that p(z|v) captures all visual details (validated by the prior reconstruction loss converging to a low error, quantified by SSIM > 0.95 on the validation set), a high KL indicates the posterior has lost information. This assumption fails when prior reconstruction is poor (e.g., limited decoder capacity, blurry outputs as noted in VAE literature), in which case KL reflects mismatch unrelated to visual detail. We address this by explicitly checking the SSIM of prior reconstructions on the validation set after each epoch; only if SSIM > 0.95 do we interpret KL as a measure of visual detail loss. The prior reconstruction loss ||v - v_prior_recon||^2 measures the prior's ability to encode visual details; the hyperparameter gamma controls how tightly the prior is tied to visual reconstruction, and if gamma is too low the prior may become detached and no longer serve as a faithful visual memory.

## Contribution

(1) A visual-conditioned prior p(z|v) that integrates visual reconstruction into the variational objective, ensuring fine-grained visual detail preservation during multimodal latent alignment. (2) A training framework that jointly optimizes the prior to reconstruct visual input, creating a structural coupling between alignment and visual fidelity without separate additive losses. (3) A design principle that internalizes detail retention within the prior, applicable to existing variational multimodal models.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | VQA v2.0 fine-grained subset (questions requiring spatial/color details), RefCOCO (visual grounding), MS COCO Captions (fine-grained captioning) | Tests visual detail preservation across multiple tasks |
| Primary metric | VQA: Fine-grained Accuracy; RefCOCO: grounding accuracy (Intersection over Union > 0.5); Captions: CIDEr, BLEU-4, METEOR | Directly measures detail preservation in different modalities |
| Baseline 1 | Standard MVAE (fixed prior, e.g., standard Gaussian) | Tests benefit of visual conditioning |
| Baseline 2 | Symmetric MVAE (joint prior p(z|v,t)) | Tests asymmetric design necessity |
| Baseline 3 | Learnable prior (p(z) with trainable mean and logvar, not conditioned on v) | Isolates effect of conditioning on v |
| Ablation | ViCP w/o prior recon loss (gamma=0) | Tests prior grounding importance |

### Why this setup validates the claim
This setup isolates the central claim—that ViCP preserves fine-grained visual details during multimodal alignment—by focusing on a VQA subset where questions require pinpointing subtle visual features, additionally on visual grounding (requires precise object localization) and captioning (requires detailed visual descriptions). The baseline comparisons are structured to test each sub-mechanism: fixed prior (ignores visual input) vs. symmetric prior (may cause posterior collapse) vs. learnable prior (learns a generic distribution) all fail to retain details for different reasons, while our ablation removes the prior reconstruction loss to test grounding. The fine-grained accuracy, grounding accuracy, and caption metrics directly capture the predicted effect because they reward correct answers/descriptions only when visual details survive the variational bottleneck—generic accuracy would blur the signal. We also enforce a calibration: we only consider runs where the prior reconstruction SSIM on the validation set exceeds 0.95 after training; otherwise, we flag the run as a failure (limited decoder capacity). This quantitative check ensures that KL divergence is a valid proxy for detail loss. Together, this design forms a falsifiable test: if ViCP's improvement is concentrated on detail-oriented tasks and not on coarse ones, the causal chain is supported.

### Expected outcome and causal chain

**vs. Standard MVAE (fixed prior)** — On a question like "What color is the cat's left eye?", the fixed prior loses pixel-level color information because it ignores visual input; the posterior collapses toward a generic Gaussian that cannot reconstruct subtle hues. ViCP instead conditions the prior on visual features via pixel-level reconstruction, so the posterior preserves eye color even after KL regularization. We expect a noticeable gap (e.g., 10-15% accuracy difference) on fine-grained VQA questions, but parity on questions answerable from global features. On RefCOCO, standard MVAE will struggle with precise bounding boxes; ViCP should yield higher IoU. On captioning, ViCP should produce more detailed descriptions (higher CIDEr) while standard MVAE yields generic ones.

**vs. Symmetric MVAE (joint prior)** — On the same question, the symmetric prior uses both modalities, but this can cause posterior collapse: the prior already captures text-relevant variations (e.g., shape, location) and the KL term becomes trivial, dropping visual texture details. ViCP's asymmetric design forces the prior to encode only visual details and the posterior to inject text info separately, avoiding collapse. We expect a moderate gap (e.g., 5-10%) on fine-grained questions, with symmetric MVAE performing well on coarse questions but struggling on subtle visual distinctions. On grounding, symmetric MVAE may rely on language cues (e.g., "left") but fail on visual details; ViCP is more robust.

**vs. Learnable prior (not conditioned on v)** — The learnable prior can adapt to the overall data distribution but does not encode instance-specific visual details. On fine-grained questions, it will behave similarly to a fixed prior because it cannot capture the pixel-level information of each specific image. ViCP with visual conditioning should outperform by at least 8-12% on fine-grained VQA. This comparison isolates the effect of conditioning on the visual input.

**vs. Ablation (ViCP w/o prior recon loss)** — Without the prior reconstruction loss, the prior network is trained only by the KL term and may drift toward a fixed prior, losing its visual encoding. On a fine-grained question, the prior becomes uninformative, leading to similar failure as standard MVAE but less extreme. We expect the ablation to underperform full ViCP by 5-8% on fine-grained subset, but still beat standard MVAE due to visual conditioning. Additionally, we expect the prior reconstruction SSIM to drop below 0.95 in this ablation, which would invalidate the assumption that KL measures detail loss.

### What would falsify this idea
If ViCP's accuracy improvement is uniform across both fine-grained and coarse question subsets (e.g., no differential advantage), then the claim that it preserves fine-grained visual details is not supported; the method may benefit from some other factor like a better variational objective, not from the visual-conditioned prior specifically. Also, if prior reconstruction SSIM cannot reach 0.95 on the validation set, the underlying assumption fails and the method's intended mechanism cannot be verified.

## References

1. Multimodal Continuous Reasoning via Asymmetric Mutual Variational Learning
2. Multimodal Chain of Continuous Thought for Latent-Space Reasoning in Vision-Language Models
3. Variational Reasoning for Language Models
4. RAVR: Reference-Answer-guided Variational Reasoning for Large Language Models
5. Training Large Language Models to Reason in a Continuous Latent Space
6. STaR: Bootstrapping Reasoning With Reasoning
7. Guiding Language Model Reasoning with Planning Tokens
8. Think before you speak: Training Language Models With Pause Tokens
