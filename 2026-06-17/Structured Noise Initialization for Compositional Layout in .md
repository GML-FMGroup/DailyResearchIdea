# Structured Noise Initialization for Compositional Layout in Diffusion Models

## Motivation

DesignDiffusion achieves end-to-end generation of text and images but fails on complex layouts with multiple text blocks because its isotropic noise initialization provides no spatial structure, forcing the model to learn layout implicitly from text prompts — a task that scales poorly with block count. The field's trajectory toward compositional layout decomposition requires a method that injects layout structure into the diffusion process without an explicit planner, preserving the end-to-end paradigm.

## Key Insight

The denoising process jointly refines visual content and spatial arrangement because the initial noise already encodes a layout hypothesis consistent with the text hierarchy, making layout a continuous latent variable rather than a separately planned discrete step.

## Method

Structured Layout Noise Initialization (SLNI) modifies the standard isotropic noise input of a diffusion model by replacing it with a tensor that encodes spatial layout hypotheses derived from the input text hierarchy. The method takes a text prompt with hierarchical structure (e.g., multi-line titles, captions) and outputs an image with proper composition.

**(B) How it works:**
```pseudocode
# Input: text prompt T with hierarchy H (e.g., line breaks, indentation)
# Output: image I

# Step 1: Parse hierarchy into layout blocks
blocks = parse_hierarchy(H)  # list of (text_content, relative_position)

# Step 2: Encode each block into a layout token via learned embedding
layout_tokens = []
for text, pos in blocks:
    text_emb = text_encoder(text)  # CLIP text encoder (ViT-L/14, dim=768)
    pos_emb = positional_encoding(pos)  # sinusoidal pos encoding, frequency=10000, dim=768
    layout_tokens.append(text_emb + pos_emb)  # additive combination, dim=768

# Step 3: Construct structured noise tensor
# noise_grid: 64x64 latent pixels (for 512x512 image)
# Assign each layout token to a spatial region based on pos
structured_noise = random_normal(1, 4, 64, 64)  # base noise, std=1.0
for i, token in enumerate(layout_tokens):
    spatial_mask = get_mask(blocks[i].pos, grid_size=64)  # binary mask, 64x64
    projected_token = linear_proj(token)  # 768 -> 4 channels, init with Xavier uniform
    # Calibration: scale projected_token to match base noise std (1.0) on average
    # Compute empirical std of projected_token over calibration set of 512 samples; rescale if >1.1
    projected_token = projected_token / (1.0 + max(0, empirical_std - 1.0))  # clamp to [1.0, inf)
    structured_noise += spatial_mask * projected_token  # add token to region

# Step 4: Standard diffusion denoising (DDPM/DDIM) with layout conditioning
# Model: U-Net (from DesignDiffusion) with cross-attention to layout tokens (in addition to prompt)
# Denoising steps: 50 steps with DDIM sampling, eta=0.0
# Loss: same as DesignDiffusion (character localization loss + noise prediction)
# Character localization loss uses CTC on character positions
output_latent = diffusion_sampler(structured_noise, condition=[prompt_emb, layout_tokens])
I = decoder(output_latent)
```
Hyperparameters: noise grid size 64x64, layout token dim 768, linear projection 768->4, DDIM steps 50, calibration set size 512, empirical std threshold 1.1.

**(C) Why this design:**
We chose structured noise over masked conditioning (e.g., layout mask as extra input) because structured noise allows the diffusion model to treat layout as a latent variable that is refined jointly with content, rather than as a hard constraint that may conflict with denoising dynamics — accepting the cost that early denoising steps must recover from any layout token placement errors. We chose additive token injection over concatenation to avoid increasing input channel count, which would require architectural changes to the U-Net; the trade-off is that tokens may dilute in early noise but are preserved via cross-attention conditioning throughout denoising. We chose a learned linear projection over a fixed mapping to allow the model to adapt token embeddings to the noise distribution; the downside is an extra 4*768 parameters. Compared to prior work like LayoutDiffusion that uses layout conditions as separate cross-attention, our method embeds layout directly into noise, enabling joint refinement without explicit layout decoder. A load-bearing assumption is that additively injecting projected layout tokens into the initial noise tensor preserves the denoising process's fidelity. We calibrate the projection to ensure the structured noise's distribution remains close to isotropic Gaussian (empirical std clamped to ≤1.1). To verify retention, we measure the Wasserstein distance between the noise distribution and standard normal at steps 5,10,20,50; if distance >0.2, we adjust scaling.

**(D) Why it measures what we claim:**
`structured_noise` encodes spatial arrangement (the motivation-level concept of compositional layout) because each layout token is placed at a spatial position corresponding to its text block, assuming that the text hierarchy can be reliably parsed into non-overlapping rectangular regions; this assumption fails when the input contains free-form text without clear hierarchy (e.g., poetic lines), in which case the noise may encode ambiguous layout, and the method degrades to a weaker prior. The `projected_token` vector measures the visual-content bias for that region because it is derived from the text embedding via a learnable mapping; the implicit assumption is that text semantics determine visual appearance in the region, which fails for abstract concepts like “transparent box” where appearance is not directly text-driven. The `diffusion_sampler` with layout token cross-attention measures joint refinement because the denoising updates both the latent and the cross-attention activations; this measures compositional consistency only if the cross-attention heads attend to correct regions, and fails when attention is diffuse, leading to content bleeding between blocks.

## Contribution

(1) A structured noise initialization framework that encodes latent layout hypotheses from text hierarchy into the diffusion latent space, eliminating the need for an explicit layout planner. (2) The design principle that layout can be injected as a continuous latent prior in the noise initialization, enabling joint end-to-end refinement of content and spatial arrangement.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | MARIO-Eval (challenging multi-block layout prompts with arbitrary text blocks, 1000 prompts) | Tests compositional layout with clear hierarchy and free-form text; includes 200 ambiguous prompts (poetic lines) |
| Primary metric | Local CLIP score per layout block (average over blocks) | Measures per-region semantic consistency |
| Secondary metric | Intersection over Union (IoU) between predicted and ground-truth block positions (extracted via OCR) | Directly measures layout accuracy |
| Baseline 1 | SDXL (standard diffusion) | Baseline without layout conditioning |
| Baseline 2 | DesignDiffusion | Explicit layout conditioning via cross-attention |
| Baseline 3 | Masked conditioning (layout mask as extra input to U-Net) | Tests alternative to structured noise |
| Baseline 4 | Dynamic conditioning (layout token added at each denoising step via cross-attention, but initial noise isotropic) | Isolates effect of initial injection vs. dynamic conditioning |
| Ablation | SLNI without layout tokens (isotropic noise), but keep cross-attention conditioning from prompt | Isolates effect of structured noise |

Additional analysis: Measure Wasserstein distance between structured noise and standard Gaussian at steps 5,10,20,50 to verify noise structure retention. Report user preference via Amazon Mechanical Turk (100 users, 50 prompts, forced-choice vs. DesignDiffusion).

### Why this setup validates the claim

This combination forms a falsifiable test of the claim that structured noise enables better compositional layout through joint refinement. The MARIO-Eval dataset explicitly requires arranging multiple text blocks into coherent spatial regions, making layout adherence critical. SDXL tests whether standard diffusion can infer layout from text alone—a failure here confirms the need for explicit layout guidance. DesignDiffusion tests a competing approach (separate cross-attention for layout) that does not embed layout into noise, isolating the benefit of joint refinement. Masked conditioning tests whether adding layout information as a separate input (rather than noise embedding) improves performance, which would challenge the claim that noise-level integration is key. Dynamic conditioning tests whether initial injection matters beyond repeated conditioning. The ablation removes the structured noise but keeps all other components, directly measuring the contribution of the layout token injection. The local CLIP score captures whether each region generates content matching its intended semantics, a prerequisite for correct composition. The IoU metric directly measures whether generated layout matches intended positions. If our mechanism works, we expect that only SLNI achieves high local CLIP on all blocks simultaneously and high IoU, while baselines suffer on some blocks due to misplacement or content bleeding. Additionally, the Wasserstein distance analysis will provide empirical evidence that the structured noise distribution remains near-Gaussian, supporting the load-bearing assumption.

### Expected outcome and causal chain

**vs. SDXL** — On a prompt like "Title: The Great Gatsby\nCaption: A roaring twenties party", SDXL often places both elements randomly or merges them, because it lacks any spatial prior. The result is a single disorganized scene with both concepts blended. Our method instead allocates distinct noise regions for title and caption, refined jointly via cross-attention, producing a clear title top and caption bottom. We expect a large gap (>0.15) in local CLIP for the caption block, and >0.20 IoU gap, but parity on global image quality (FID).

**vs. DesignDiffusion** — On a prompt with narrow text blocks (e.g., "Left: Dog; Right: Cat"), DesignDiffusion often generates partially overlapping content because its layout cross-attention operates on separate embeddings that can bleed into neighboring regions. Our method embeds layout into the noise itself, forcing spatial separation from the start, and the joint refinement corrects any misalignment. We expect a moderate gap (0.05-0.10) in local CLIP and 0.10-0.15 IoU, especially on boundary accuracy.

**vs. Masked conditioning** — On a prompt where the text hierarchy is ambiguous (e.g., poetic lines), masked conditioning enforces hard boundaries that conflict with diffusion dynamics, leading to unnatural cutoffs or empty regions. Our method's noise-level integration adapts to uncertainty, treating layout as soft prior. We expect a gap of 0.1-0.15 on local CLIP and 0.15-0.20 IoU on such ambiguous prompts, but similar performance on well-separated blocks.

**vs. Dynamic conditioning** — Dynamic conditioning adds layout tokens at each step but starts from isotropic noise. On well-separated layouts, initial placement is less critical, so both methods perform similarly (<0.03 gap). However, on complex layouts with many blocks, the initial structured noise provides a strong prior that accelerates convergence; we expect a gap of 0.05-0.08 in local CLIP and IoU in favor of SLNI.

### What would falsify this idea

If the local CLIP gain of our method is uniform across all types of layout (e.g., both well-separated and ambiguous), rather than concentrated on prompts where hierarchical structure is critical (e.g., multi-block vs. single-block), then the central claim that structured noise specifically improves compositional layout would be unsupported. Additionally, if the Wasserstein distance at step 5 is >0.3, our calibration may be insufficient and the assumption of noise distribution preservation is violated, warranting method redesign.

## References

1. DesignDiffusion: High-Quality Text-to-Design Image Generation with Diffusion Models
2. Diffusion Model Alignment Using Direct Preference Optimization
3. Improving alignment of dialogue agents via targeted human judgements
