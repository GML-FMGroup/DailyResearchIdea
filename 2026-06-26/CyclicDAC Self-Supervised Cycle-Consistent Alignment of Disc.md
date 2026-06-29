# CyclicDAC: Self-Supervised Cycle-Consistent Alignment of Discrete Visual and Textual Codes from Unpaired Data

## Motivation

ViQ achieves text-aligned discrete visual codes but relies on large-scale paired image-text data and a large pretrained language model, creating a new bottleneck in data and model efficiency. Current self-supervised methods require paired data or a strong pretrained LM, limiting scalability. We address this by enforcing cycle consistency between discrete codes from unpaired images and text, eliminating the need for any paired data or external pretrained models.

## Key Insight

A shared discrete codebook provides a common latent space where the cycle-consistency constraint (image→code→text→code and text→code→image→code) forces the two modalities to align semantically without paired data, because the coupling through the codebook's discrete indices acts as a pivot that makes the mapping invertible.

## Method

**CyclicDAC (Cyclic Discrete Alignment via Codebook)** is a self-supervised method that aligns visual and textual modalities using a shared discrete codebook. Input: unpaired images and text. Output: a visual encoder and a text encoder that map to the same discrete codebook, enabling text-aligned visual representations.

**(B) How it works:**
```pseudocode
// Architecture specifications:
// Shared codebook: C = {c_k in R^d}, size K=8192, d=256
// Image encoder: E_img = ResNet-18 (output dim d) followed by a linear projection to d
// Text encoder: E_txt = 2-layer BiLSTM (hidden=256) followed by a linear projection to d
// Quantizer: Q(z) = argmin_k ||z - c_k||, return one-hot index and c_k (straight-through estimator for gradients)
// Image decoder: D_img = 5-layer transposed CNN (channels: 256->128->64->32->3, kernel=4, stride=2, padding=1)
// Text decoder: D_txt = 2-layer LSTM (hidden=256) with learned embedding of size d

// Hyperparameters:
// lambda_cyc=1.0, lambda_rec=0.5, lambda_div=0.1, lambda_commit=0.25
// Optimizer: AdamW, lr=1e-4, weight_decay=1e-5
// Batch size: 64 (32 images + 32 texts per batch, unpaired)
// Training steps: 200k
// Image resolution: 128x128 (cropped and resized)
// Text max length: 20 tokens (BPE tokenization)
// Codebook update: exponential moving average (EMA) with decay 0.99

for each batch of unpaired images and texts:
  // Image cycle
  z_img = E_img(x_img)
  idx_img, c_img = Q(z_img)
  x_cycled_img = D_img(c_img)            // reconstruct image
  t_cycled = D_txt(c_img)                // generate text from image code
  z_cycled_txt = E_txt(t_cycled)
  idx_cycled, c_cycled_txt = Q(z_cycled_txt)
  L_cyc_img = ||c_img - c_cycled_txt||^2  // cycle consistency on codes

  // Text cycle
  z_txt = E_txt(x_txt)
  idx_txt, c_txt = Q(z_txt)
  t_cycled_txt = D_txt(c_txt)            // reconstruct text
  x_cycled = D_img(c_txt)                // generate image from text code
  z_cycled_img = E_img(x_cycled)
  idx_cycled2, c_cycled_img = Q(z_cycled_img)
  L_cyc_txt = ||c_txt - c_cycled_img||^2

  // Reconstruction losses for decoders (image reconstruction and text reconstruction)
  L_rec_img = L1(x_img, x_cycled_img)
  L_rec_txt = cross_entropy(x_txt, t_cycled_txt)

  // Codebook divergence: encourage used codes to be distinct (contrastive)
  // For each code in the batch, maximize its cosine similarity to itself vs others
  L_div = -log(exp(sim(c_i,c_i)/tau) / (sum_(j!=i) exp(sim(c_i,c_j)/tau)))  // tau=0.07, averaged over batch codes

  // Commitment loss to keep encoder outputs close to chosen code
  L_commit = ||z_img - c_img.detach()||^2 + ||z_txt - c_txt.detach()||^2

  Total loss = lambda_cyc*(L_cyc_img+L_cyc_txt) + lambda_rec*(L_rec_img+L_rec_txt) + lambda_div*L_div + lambda_commit*L_commit
```

**(C) Why this design:** We chose a shared discrete codebook over continuous alignment because discrete indices enforce a hard bottleneck that makes cycle-consistency non-trivial; continuous alignment could collapse to trivial identity mappings. We chose cycle-consistency over contrastive methods because cycle-consistency directly enforces invertibility, which is a stronger signal for alignment than similarity; the trade-off is that cycle-consistency requires two decoders (image and text), adding computational cost. We chose to train the text decoder on unpaired texts from scratch, accepting that it may generate lower-quality text initially, but it avoids reliance on a large pretrained language model. We chose a contrastive diversity loss on codebook entries to prevent code collapse; without it, the entire batch could use the same code, invalidating the cycle. We chose a commitment loss to stabilize training, a standard choice from VQ-VAE, but we detach the code to prevent the encoder from chasing a moving target. The image decoder is a simple convolutional upsampler; we accept that reconstruction quality may be lower than specialized models, but it suffices for alignment.

**Load-bearing assumption:** The text decoder D_txt, trained from scratch on unpaired text, must generate text that is semantically consistent with the input code and lies within the distribution of the text encoder E_txt. If D_txt produces degenerate or out-of-distribution text, the cycle-consistency loss will reflect generation error rather than misalignment. To verify this assumption, we will measure semantic similarity between generated text and original captions for a held-out set of 512 image-text pairs (used only for monitoring, not training). Specifically, after every 10k training steps, we compute BERTScore (F1) between t_cycled and ground-truth captions for those 512 images that were encoded to codes and decoded to text. We track BERTScore over time; if it remains below 0.5 (random baseline ~0.3), we flag that the text decoder is not preserving semantics. Additionally, we compute the cycle-consistency loss on this held-out set separately to see if low BERTScore correlates with high cycle loss, confirming the assumption's impact. As a mitigation, if BERTScore does not exceed 0.5 after 50k steps, we switch to initializing D_txt with a pretrained GPT-2 (small) and keep it frozen for the remainder of training.

**(D) Why it measures what we claim:** The cycle-consistency loss ||c_img - c_cycled_txt||^2 measures semantic alignment between image and text representations because it quantifies how well the discrete code from an image, when decoded to text and re-encoded, returns to the same code; this equivalence holds under the assumption that the text decoder and text encoder are sufficiently trained to map the code to a semantically similar text and back. The assumption fails when the text decoder generates text that is out-of-distribution for the text encoder, causing a different code; then the loss reflects reconstruction errors rather than misalignment. The diversity loss L_div measures codebook utilization because it maximizes the distance between used code embeddings; if this term is omitted, the trivial solution of all inputs mapping to the same code yields low cycle loss but no alignment. The commitment loss measures the correspondence between encoder outputs and codebook entries; without it, the encoder outputs can drift arbitrarily, making the codebook indexing unstable. Together, these components operationalize the concept of cycle-consistent alignment in a self-supervised, unpaired setting.

## Contribution

(1) A novel self-supervised method, CyclicDAC, that aligns visual and textual modalities using cycle-consistency on a shared discrete codebook, without requiring paired data or a large pretrained language model. (2) The demonstration that cycle-consistency across discrete codes is a sufficient signal for cross-modal alignment, with theoretical grounding in the invertibility constraint. (3) A training framework that jointly optimizes reconstruction, cycle consistency, and code diversity, enabling text-aligned discrete visual representations from unpaired data.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|-----------------------|
| Dataset | COCO images + captions (unpaired) | Standard cross-modal retrieval benchmark |
| Primary metric | Image-to-text & text-to-image Recall@1 | Measures cross-modal alignment |
| Baseline | VQ-VAE (no cross-modal) | Tests effect of discretization alone |
| Baseline | Continuous cycle-consistency (shared bottleneck, InfoNCE loss) | Tests importance of discretization |
| Baseline | CLIP (paired oracle) | Upper bound with paired supervision |
| Baseline | CyclicDAC with frozen GPT-2 text decoder | Tests reliance on pretrained LM |
| Ablation-of-ours | CyclicDAC w/o cycle loss | Isolates cycle consistency contribution |

### Why this setup validates the claim
Using unpaired COCO images and captions as separate pools directly tests cross-modal alignment without paired supervision. VQ-VAE (no cross-modal) shows discretization alone fails to align modalities. Continuous cycle-consistency with a shared continuous bottleneck (using InfoNCE loss to align image and text embeddings in a continuous space) determines whether discretization prevents trivial solutions like semantic collapse. CLIP provides an oracle upper bound, quantifying the gap from paired supervision. The frozen GPT-2 baseline tests whether a pretrained language model is necessary; if our scratch-trained decoder performs comparably, it validates our resource advantage. Ablating the cycle loss isolates its contribution, confirming that cycle consistency drives alignment. Recall@1 measures whether the codebook yields matching image-text pairs, a direct test of semantic alignment.

### Expected outcome and causal chain

**vs. VQ-VAE (no cross-modal)** — On a case where an image of a dog and a caption "a dog playing fetch" are semantically similar, VQ-VAE produces no cross-modal alignment because it only reconstructs within modality, so retrieval fails (Recall@1 ≈ random). Our method instead aligns them via cycle consistency: the image's code decodes to a caption that, when re-encoded, returns the same code. Thus we expect Recall@1 > 30%, demonstrating alignment.

**vs. Continuous cycle-consistency** — On a case where two visually different images depict the same concept (e.g., a golden retriever and a black lab), continuous cycle-consistency may collapse their representations because continuous space allows trivial identity mappings, hurting discrimination. Our method's discrete bottleneck forces distinct codes for distinct semantics, so we expect a large gap (e.g., 30% vs 5%) on varied retrieval sets.

**vs. CLIP (paired oracle)** — On any instance, CLIP uses paired supervision to achieve high Recall@1 (≈60-70%). Our method, using only unpaired data, will be lower but should close the gap within 15% (e.g., ≈50%), showing unpaired alignment is feasible. If not, the central claim is weakened.

**vs. CyclicDAC with frozen GPT-2 text decoder** — Using a frozen GPT-2 as text decoder (initialized from pretrained weights) may improve text generation quality, raising Recall@1 by up to 5-10%. If the gap is large (e.g., >15%), it suggests that our scratch-trained decoder is the bottleneck, partially supporting the load-bearing assumption but still requiring verification. We report BERTScore for generated text in both variants; if the scratch decoder's BERTScore is >0.6 (fair), the assumption holds reasonably well.

### What would falsify this idea
If CyclicDAC's Recall@1 is no better than the ablation without cycle loss (near random), or if performance is uniform across semantically varied subsets rather than concentrated on cases where cycle consistency forces alignment, then the method fails to align modalities as claimed. Additionally, if the BERTScore of generated text remains below 0.5 after 100k steps, the load-bearing assumption is violated and the cycle loss likely reflects generation errors, not alignment.

## References

1. ViQ: Text-Aligned Visual Quantized Representations at Any Resolution
2. Patch n' Pack: NaViT, a Vision Transformer for any Aspect Ratio and Resolution
3. Multimodal Autoregressive Pre-training of Large Vision Encoders
4. Unified-IO 2: Scaling Autoregressive Multimodal Models with Vision, Language, Audio, and Action
5. PaLI: A Jointly-Scaled Multilingual Language-Image Model
6. Pix2Struct: Screenshot Parsing as Pretraining for Visual Language Understanding
