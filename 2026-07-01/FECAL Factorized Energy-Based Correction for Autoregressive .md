# FECAL: Factorized Energy-Based Correction for Autoregressive Image Generation

## Motivation

Autoregressive image models like GEAR suffer from non-monotonic error propagation because they commit to each token without any backtracking mechanism. This structural limitation is shared across autoregressive approaches, as they generate tokens sequentially without iterative refinement or local consistency checks, causing mistakes to compound and degrade image quality.

## Key Insight

Token-level backtracking can be achieved by factorizing the image into locally-consistent patches and using a lightweight energy model trained to detect patch-level inconsistencies, enabling a single resampling step without global search.

## Method

**FECAL: Factorized Energy-based Correction for Autoregressive Image Generation**

(A) **What it is**: FECAL (Factorized Energy-based Correction for Autoregressive Image generation) is a module attached to an autoregressive transformer generator. Its input is the current hidden state and the generated token sequence so far; its output is a scalar energy indicating whether the last generated token should be resampled.

(B) **How it works** (pseudocode):
```python
# t: step index, h: hidden state, x: token sequence
# pretrained_patch_encoder: DINOv2 ViT-S/8 (frozen), output dim 384
# mlp: 2-layer MLP, hidden=256, GeLU activation
# energy_head: linear projection to scalar
# tau: threshold calibrated on validation set (1000 images) to 90th percentile of energy on consistent patches
# max_retries: 2

for t in range(max_length):
    logits_t = lm_head(h_t)  # autoregressive head
    x_t = sample(softmax(logits_t))
    # extract patch embedding using frozen pretrained encoder
    p_t = pretrained_patch_encoder(generated_image_partial)  # average patch embedding for current step's region
    c_t = mlp(concat(h_t, p_t))
    e_t = energy_head(c_t)  # scalar energy
    if e_t > tau:
        for _ in range(max_retries):
            x_t = sample(softmax(logits_t))
            p_t = pretrained_patch_encoder(generated_image_partial)
            c_t = mlp(concat(h_t, p_t))
            e_t = energy_head(c_t)
            if e_t <= tau:
                break
    sequence.append(x_t)
    h_{t+1} = transformer(h_t, x_t)
# If after retries e_t still > tau, accept the token anyway (no forced correction)
```

(C) **Why this design**: We chose a lightweight energy head over a full search because we want to preserve the sequential nature and avoid global loops. The energy head is trained on patch-level consistency targets (using a pretrained image model to judge realism). This trade-off accepts that the energy model may be imperfect, but it is fast and local. We factorize the image into patches rather than using a global energy because global energy would require full image rendering at each step, negating the speed advantage. We use a separate energy head that receives the hidden state and a patch embedding from a pretrained encoder (e.g., DINOv2) rather than the decoder's own activations to avoid overfitting to the generator's biases. This adds computational cost but provides a normative standard for consistency. The threshold tau is not tuned per token but fixed globally to allow occasional corrections; this may cause unnecessary corrections on borderline cases but is simple to implement. We limit retries to 2 to bound runtime; this may fail to correct severe errors but prevents infinite loops. **The load-bearing assumption of FECAL is that token-level errors in autoregressive image generation are detectable by evaluating local patch consistency via a lightweight energy model.**

(D) **Why it measures what we claim**: e_t measures local patch consistency because it is trained to assign low energy to patches from real images and high energy to patches with artificially injected inconsistencies (e.g., patch swaps, Gaussian noise). The assumption A is that local patch inconsistency is a necessary early indicator of error propagation. The failure mode F: errors that are globally consistent but locally inconsistent (e.g., a miscolored object within a uniform region) or errors that are locally consistent but globally inconsistent (e.g., two dissimilar objects swapped). In these cases, e_t may incorrectly signal an error or miss it entirely. The energy head is trained with a margin-based contrastive loss (margin=1.0) using a dataset of 50k real patches and 50k corrupted patches from ImageNet.

## Contribution

(1) FECAL, a local energy-based correction mechanism that enables token-level backtracking in autoregressive image generation without iterative refinement, using a factorized consistency criterion. (2) A design principle showing that lightweight patch-consistency energy is sufficient to mitigate error accumulation in autoregressive image models, as demonstrated by improving over GEAR baseline. (3) An empirical analysis of error propagation patterns in autoregressive vision transformers, highlighting the effectiveness of local corrections.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | ImageNet 256×256 | Standard benchmark for image generation |
| Primary metric | FID | Measures distributional fidelity |
| Baseline 1 | Vanilla VQ-AR | No correction, tests error propagation |
| Baseline 2 | Diffusion (DiT) | Strong generative baseline, different paradigm |
| Baseline 3 | VQ-AR with likelihood-based rejection (resample if probability < 0.05, max 2 retries) | Mimics text-domain rejection sampling |
| Ablation of ours | FECAL w/o pretrained encoder | Tests importance of external consistency (replaces DINOv2 patch encoder with a learnable linear projection of tokens) |
| Resources | 8 NVIDIA A100 GPUs, ~500 GPU-hours | Training FECAL head and fine-tuning threshold |

### Why this setup validates the claim
This experimental design forms a falsifiable test of FECAL's central claim: that local patch-consistency correction reduces error propagation in autoregressive image generation. Comparing to Vanilla VQ-AR directly isolates the effect of the correction module—if FECAL improves FID, it must be due to correcting local inconsistencies. Comparing to Diffusion (DiT) tests whether FECAL achieves comparable or better quality than a non-autoregressive paradigm, validating that the correction mechanism bridges the gap. The likelihood-based rejection baseline (Baseline 3) contextualizes the novelty: standard rejection sampling in text uses only token likelihood, whereas FECAL uses an external patch-consistency signal. If FECAL outperforms Baseline 3, it demonstrates the value of using visual coherence rather than proxy probabilities. The ablation (removing the pretrained encoder) tests whether the external consistency signal is crucial; if it performs similarly, the benefit may come from resampling alone rather than grounded patch evaluation. FID is chosen because it captures overall image realism and distributional match, and we expect improvements in reducing local artifacts to translate into better FID scores. Furthermore, we will analyze per-step energy values and compute FID at intermediate generation steps to directly observe the correlation between energy spikes and degradation in FID, thereby testing the assumption that local inconsistencies precede global errors.

### Expected outcome and causal chain

**vs. Vanilla VQ-AR** — On a case where an early token is sampled incorrectly (e.g., a fur texture patch with wrong orientation), the baseline autoregressive model propagates this error, causing a visible streak or global distortion. Our FECAL module detects the local patch inconsistency via high energy (because the generated patch mismatches its neighbors according to the pretrained encoder), triggers resampling, and likely replaces the token with a correct one. This prevents error accumulation. Therefore, we expect a noticeable FID improvement (on the order of 2-5 points) on images containing complex textures or long-range dependencies, but parity on simple, uniform backgrounds.

**vs. Diffusion (DiT)** — On a case requiring sharp local coherence (e.g., generating a geometric pattern with precise edges), diffusion models often produce oversmoothed or slightly blurry outputs due to the iterative denoising process. FECAL, by operating on discrete tokens and enforcing local patch consistency via the energy head, can preserve crisp details and sharp transitions. However, diffusion models may handle global structure better (e.g., overall object shape). We expect FECAL to achieve a lower FID on subsets rich in high-frequency details (e.g., textures, edges) but potentially slightly higher FID on smooth regions (e.g., skies, backgrounds), resulting in comparable overall FID but with a different distribution of errors.

**vs. Likelihood-based rejection (Baseline 3)** — On a case where a token has high probability but is locally inconsistent (e.g., a plausible but incorrect texture continuation), likelihood-based rejection will not resample, while FECAL will. Conversely, if a low-probability token is actually consistent (e.g., a rare but correct detail), likelihood-based rejection may incorrectly resample, while FECAL will accept. We expect FECAL to outperform Baseline 3 on images requiring careful local consistency, yielding FID improvements of 1-3 points.

Additionally, we will analyze failure cases by measuring per-step energy and comparing to per-step contributions to FID (using partial image generations). This will quantify how often local consistency fails to detect or falsely correct errors. We anticipate that in <5% of steps, local consistency may be misleading (e.g., globally coherent but locally ambiguous patches), causing unnecessary corrections or missed errors.

### What would falsify this idea
If FECAL's FID improvement over Vanilla VQ-AR is uniform across all image subsets (e.g., both simple and complex textures improve equally), it would suggest the gain is not from correcting local error propagation but from a general regularization effect, contradicting the central claim. Similarly, if the ablation without pretrained encoder matches FECAL's performance, the external consistency signal is unnecessary. If per-step analysis shows that energy does not correlate with downstream FID degradation, the assumption linking local consistency to error propagation is false.

## References

1. GEAR: Guided End-to-End AutoRegression for Image Synthesis
2. X-Omni: Reinforcement Learning Makes Discrete Autoregressive Image Generative Models Great Again
3. VQRAE: Representation Quantization Autoencoders for Multimodal Understanding, Generation and Reconstruction
4. Orthus: Autoregressive Interleaved Image-Text Generation with Modality-Specific Heads
5. TokenFlow: Unified Image Tokenizer for Multimodal Understanding and Generation
6. mPLUG-OwI2: Revolutionizing Multi-modal Large Language Model with Modality Collaboration
7. Generative Multimodal Models are In-Context Learners
8. NExT-GPT: Any-to-Any Multimodal LLM
