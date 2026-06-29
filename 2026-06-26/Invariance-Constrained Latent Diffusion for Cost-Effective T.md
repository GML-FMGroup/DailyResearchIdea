# Invariance-Constrained Latent Diffusion for Cost-Effective Text-to-Image Generation

## Motivation

Existing text-to-image models such as Scaling Rectified Flow Transformers and PixArt-α achieve high quality by scaling transformer size and training compute, implicitly assuming that parameter count is the primary lever for fidelity. This assumption creates a structural dependency on massive resources, with no mechanism to decouple generation quality from model scale. The root cause is the model's need to memorize the large variation in linguistic surface forms of the same semantic concept, wasting capacity on irrelevant textual variations.

## Key Insight

Enforcing semantic invariance of the latent space to paraphrastic variations reduces the effective sample complexity, enabling a small generator to focus its capacity on visual semantics rather than linguistic diversity.

## Method

## (A) What it is
We propose **InvarDiT**, a training framework that adds a semantic invariance regularization to a small DiT-based diffusion model. The method takes as input a text prompt and outputs an image. During training, each prompt is paired with an automatically generated paraphrase; the regularization encourages the latent representations of the original and paraphrase to be close in the latent space, while keeping different prompts separated.

## (B) How it works
```python
# Training pseudocode for InvarDiT
# Hyperparameters: lambda_inv=0.1, margin=0.2, batch_size=64

for batch in dataloader:
    # batch contains (prompt, image, paraphrase)
    # Encode original prompt
    z_orig = text_encoder(prompt)          # frozen larger LLM (e.g., T5-large)
    # Encode paraphrase
    z_para = text_encoder(paraphrase)
    
    # Diffusion denoising steps (standard DDPM)
    noise = torch.randn_like(latent)
    t = torch.randint(0, T, (batch_size,))
    noisy_latent = forward_diffusion(latent, noise, t)
    # Predict noise using DiT conditioned on z_orig
    noise_pred = DiT(noisy_latent, t, z_orig)
    loss_mse = F.mse_loss(noise_pred, noise)
    
    # Invariance loss: triplet with anchor=z_orig, positive=z_para, negative from other prompts
    # Randomly sample negative z_neg from same batch
    z_neg = z_orig[torch.randperm(batch_size)]
    # Cosine distance
    d_pos = 1 - F.cosine_similarity(z_orig, z_para)
    d_neg = 1 - F.cosine_similarity(z_orig, z_neg)
    loss_inv = F.relu(d_pos - d_neg + margin).mean()
    
    # Total loss
    loss = loss_mse + lambda_inv * loss_inv
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

The text encoder (T5-large) is frozen, and the DiT generator has 300M parameters (6 layers, 512 hidden dim). Paraphrases are generated offline by a frozen GPT-3.5 with high-temperature decoding and filtered for semantic similarity (CLIP score > 0.9).

## (C) Why this design
We chose a contrastive triplet loss over direct MSE between latents because MSE would collapse all prompts to a single point, losing discriminability. The triplet formulation ensures invariance within equivalence classes while preserving separation across different concepts. We chose cosine distance over L2 because it is scale-invariant and more stable across different text encoder outputs. We selected a frozen larger LLM (T5-large) as text encoder to provide more robust representations for invariance without increasing training cost; the cost is that we cannot adapt the encoder to the latent space, but the invariance regularization indirectly shapes the encoder's outputs. Using a frozen LLM for paraphrase generation introduces data preparation overhead but guarantees diversity; we accept that approximate semantic equivalence may introduce noise, but we filter by CLIP score to mitigate. The small DiT (6 layers, 512 hidden dim) was chosen to demonstrate scalability; the trade-off is reduced capacity for detail, but the invariance loss compensates by reducing the variance that the model must memorize.

## (D) Why it measures what we claim
The triplet loss (d_pos vs. d_neg) measures **semantic invariance** because it forces the latent representations of semantically equivalent texts to be closer than those of different texts. **Assumption A**: paraphrases are exact semantic equivalents. **Failure mode F**: when a paraphrase introduces semantic drift (e.g., adds a new object), the loss incorrectly treats a non-equivalent pair as close, which may degrade generation accuracy. The MSE loss (noise prediction) measures **generation fidelity** as it directly optimizes the denoising objective. The combination ensures that the model maintains high image quality while learning an invariant latent space. The margin hyperparameter operationalizes the **tightness of invariance**; a small margin forces stronger collapse but risks overfitting to paraphrase noise, while a larger margin allows more variance but reduces the decoupling effect. Together, these quantities close the causal chain: reducing the effect of linguistic surface form (invariance) lowers the sample complexity, allowing a small model to match the quality of larger models that must memorize many paraphrases.

## Contribution

(1) A training framework that enforces latent invariance to semantic paraphrases via a contrastive regularization, enabling small diffusion models to achieve competitive generation quality. (2) Empirical demonstration that a 300M-parameter DiT with invariance regularization achieves FID and CLIP scores comparable to 1B-parameter models on standard benchmarks (e.g., MS-COCO), breaking the scaling assumption. (3) A curated dataset of 10M prompt-paraphrase pairs generated by LLM and filtered by CLIP similarity for training invariance.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | COCO Captions (30K) | Standard benchmark with diverse prompts. |
| Primary metric | FID (↓) | Measures overall image quality and diversity. |
| Baseline 1 | Small DiT (no inv.) | Same architecture without invariance loss. |
| Baseline 2 | Large DiT (no inv.) | Larger capacity to show size advantage. |
| Baseline 3 | Prompt augmentation + MSE | Uses varied prompts without triplet loss. |
| Ablation of ours | Ours w/ MSE invariance | Replace triplet with MSE on latents. |

We will pre-train the paraphrase generation pipeline using GPT-3.5 with high temperature and release the generated paraphrase dataset to facilitate reproducibility. Additionally, we will report a cost vs. quality Pareto curve: training hours vs. FID for our method and all baselines.

### Why this setup validates the claim
This combination tests the central claim that semantic invariance allows a small DiT to rival larger models. Comparing to Small DiT (no invariance) isolates the effect of the triplet loss. Large DiT shows the target quality we aim to match with a smaller model. Prompt augmentation tests whether simply feeding varied paraphrases during training suffices, while our explicit triplet loss should provide stronger regularization. The ablation with MSE invariance verifies that the triplet formulation is crucial to avoid latent collapse. FID is chosen because it captures both fidelity and diversity, and the invariance regularization should improve both by reducing overfitting to linguistic surface forms. If our method yields lower FID than Small DiT and on par with Large DiT, the claim is supported; a specific failure pattern (e.g., gain only on easy prompts) would challenge it.

### Expected outcome and causal chain

**vs. Small DiT (no inv.)** — On a prompt like "a cat on a mat" whose paraphrase is "a cat sitting on a rug", the baseline produces different images because it overfits to surface words. Our invariant latent space maps both to similar representations, so generated images are consistent. We expect a noticeable FID gap (e.g., 10-15% lower) on prompts with multiple valid paraphrases, but parity on fixed-form prompts.

**vs. Large DiT (no inv.)** — On rare prompts (e.g., "a blue giraffe with polka dots"), the large model memorizes specific examples, while the small baseline struggles. Our invariance lets the small model generalize by aggregating semantic content from paraphrases, reducing sample complexity. We expect the small model's FID to approach that of the large model on rare concepts, with a gap of <5%.

**vs. Prompt augmentation + MSE** — On prompts where paraphrases vary widely in wording but not meaning (e.g., "a happy dog" vs. "a joyful canine"), augmentation without a targeted invariance loss may not effectively regularize the latent space. Our triplet loss explicitly pulls positives together and pushes negatives apart, leading to better decoupling. We expect a moderate FID improvement (e.g., 5-10%) on such prompts.

We will also evaluate the training cost (GPU hours) vs. FID trade-off: our method should achieve FID close to Large DiT at a fraction of training time (estimated 30% less due to smaller DiT size).

### What would falsify this idea
If our method's FID is no better than Small DiT on any subset, or if the improvement is uniform across all prompt types (rather than concentrated on paraphrase-rich or rare prompts), then the invariance regularization is not providing the hypothesized benefit. Also, if the MSE invariance ablation matches or outperforms our triplet loss, the design choice is invalidated.

## References

1. Qwen-Image-Agent: Bridging the Context Gap in Real-World Image Generation
2. DraCo: Draft as CoT for Text-to-Image Preview and Rare Concept Generation
3. FLUX.1 Kontext: Flow Matching for In-Context Image Generation and Editing in Latent Space
4. ELLA: Equip Diffusion Models with LLM for Enhanced Semantic Alignment
5. TokenFlow: Unified Image Tokenizer for Multimodal Understanding and Generation
6. Scaling Rectified Flow Transformers for High-Resolution Image Synthesis
7. Ranni: Taming Text-to-Image Diffusion for Accurate Instruction Following
8. PixArt-α: Fast Training of Diffusion Transformer for Photorealistic Text-to-Image Synthesis
