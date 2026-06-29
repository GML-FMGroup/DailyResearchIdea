# Adaptive Context-Aware Denoising: Embedding Context Retrieval into Diffusion via Attention Entropy

## Motivation

Current agentic image generation methods (e.g., Qwen-Image-Agent) rely on separate planning and context acquisition loops, incurring multiple model calls and error accumulation. The root cause is that context acquisition is decoupled from the generation process, leading to overhead and lack of direct alignment with generation quality. We propose to embed context retrieval directly into the denoising steps, using cross-attention entropy as an intrinsic signal to detect when the generation lacks sufficient context, thereby eliminating the separate planning module.

## Key Insight

Cross-attention entropy in diffusion models reveals alignment uncertainty between text tokens and image latents, which directly corresponds to missing context, enabling self-decided retrieval without an external planner.

## Method

We present Adaptive Context-Aware Denoising (ACAD).

(A) **What it is**: ACAD augments each denoising step of a diffusion model with a lightweight context retrieval trigger. The input is the noisy latent and text embedding; the output is the denoised latent and optionally an updated text embedding. The trigger uses the entropy of cross-attention maps between text tokens and the latent to decide whether to query an external knowledge source (e.g., an LLM) for additional context. To reduce false positives from high entropy on abstract prompts, we add a lightweight secondary classifier (a 2-layer MLP, hidden=256, ReLU) that takes the cross-attention entropy and the mean latent variance as input and outputs a binary decision: whether to proceed with retrieval. This verifier is trained on a calibration set of 512 examples with ground-truth context necessity labels. The trigger fires only if both entropy > threshold(t) and the verifier predicts "retrieve".

(B) **How it works** (pseudocode):
```python
def denoising_step(latent, text_embed, t, K):
    # standard diffusion step
    noise_pred = model(latent, text_embed, t)
    latent_new = (latent - beta_t * noise_pred) / sqrt(alpha_t)
    # compute cross-attention entropy
    attn_maps = model.get_cross_attention(latent, text_embed)  # shape: [num_heads, T, S]
    entropy = - sum(attn_maps * log(attn_maps + 1e-8), dim=-1).mean()  # scalar
    # secondary verifier
    latent_var = latent_new.var()
    verifier_input = torch.tensor([entropy, latent_var])  # 2D input
    verifier_output = verifier_model(verifier_input)  # sigmoid output, threshold at 0.5
    if entropy > threshold(t) and verifier_output > 0.5:
        # retrieve missing context
        missing_info = query_llm(latent, text_embed, context_prompt)  # e.g., 'Describe missing objects'
        text_embed = update_text_embed(text_embed, missing_info)
        # re-predict noise with updated embedding
        noise_pred2 = model(latent, text_embed, t)
        latent_new = (latent - beta_t * noise_pred2) / sqrt(alpha_t)
    return latent_new, text_embed
```
Hyperparameters: `threshold(t) = base_thresh * (1 - t/T)` (more retrieval early, less later), `base_thresh` tuned on validation set. The verifier is trained offline using Adam, learning rate 1e-4, for 50 epochs, with early stopping on a held-out validation set.

(C) **Why this design**: We chose entropy over other uncertainty measures (e.g., variance of latent) because cross-attention entropy directly reflects text-to-image alignment: when the model cannot attend to specific image regions for a given word, entropy increases. We chose a linear decreasing threshold over a fixed one because early steps require more context gathering while later steps need stability; accepting the cost that early retrieval might be triggered unnecessarily. We chose to update the text embedding and recompute the noise prediction rather than directly modifying the latent because that keeps the diffusion trajectory consistent with the updated context. We chose to re-use the same model for the second noise prediction instead of a separate network to minimize added parameters. The secondary verifier was added to avoid false positives on abstract prompts; it uses both entropy and latent variance to better distinguish true context insufficiency from inherent ambiguity. This design avoids the overhead of a separate planning loop while introducing minimal computational cost per retrieval.

(D) **Why it measures what we claim**: The cross-attention entropy measures the alignment uncertainty between a text token and image regions: high entropy means the token's information is not well localized in the latent, indicating missing or ambiguous context. This assumption fails when the prompt is deliberately abstract or when the image is already correctly generated but has multiple plausible interpretations; in that case, entropy may be high even though context is sufficient. To mitigate this, we use a decreasing threshold by step: early steps tolerate less entropy (more retrieval), later steps tolerate more entropy (since generation is nearly complete). Additionally, the secondary verifier filters out cases where entropy is high but context is sufficient (e.g., abstract prompts). Thus, the entropy signal, combined with the verifier and adaptive threshold, operationalizes the concept of 'context insufficiency' in a way that directly aligns with generation quality. We explicitly assume that high cross-attention entropy paired with verifier positive indicates missing context, and we validate this assumption through the calibration dataset and controlled experiments.

## Contribution

(1) A novel method that integrates context retrieval into the diffusion denoising process using cross-attention entropy as a self-decided trigger, eliminating the need for a separate planning loop. (2) An adaptive thresholding mechanism that balances retrieval frequency across denoising steps, reducing unnecessary queries while ensuring sufficient context early. (3) A design principle that cross-attention entropy serves as an intrinsic signal for context insufficiency in diffusion models, validated through ablation studies.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|-----------------------|
| Dataset | ContextInsuff-1K | 1000 prompts with missing context. Add 500 unambiguous prompts as control. |
| Primary metric | CLIP Score | Measures text-image alignment. |
| Additional metric | User Study Preference | Subjective alignment via 100 participants, 20 comparisons each. |
| Baseline 1 | Stable Diffusion 3 | Default generation without retrieval. |
| Baseline 2 | ACAD w/o adaptive | Always retrieves context every step. |
| Ablation 1 | ACAD with fixed threshold | Same threshold for all steps. |
| Ablation 2 | ACAD with entropy-only trigger (no verifier) | Isolates verifier impact. |
| Ablation 3 | ACAD with alternative uncertainty measure | Replace entropy with latent variance (LV) or predictive entropy (PE) at input to verifier. |
| Cost metric | Average LLM queries per generation | Total queries across all steps. |

### Why this setup validates the claim

The central claim is that adaptive context retrieval based on cross-attention entropy improves text-image alignment. ContextInsuff-1K contains prompts where the text leaves out plausible details (e.g., "a bird on a branch" without specifying season), creating a need for external context. The additional unambiguous prompts test that retrieval does not harm simple cases. Comparing against SDS tests whether any retrieval helps; ACAD w/o adaptive tests whether adaptive triggering is superior to constant retrieval (which may introduce noise on simple prompts). Ablation 1 (fixed threshold) isolates the benefit of the time-dependent threshold. Ablation 2 (no verifier) tests the benefit of the secondary classifier in reducing false retrievals. Ablation 3 (alternative measures) tests whether entropy is the best signal for context insufficiency by replacing it with latent variance or predictive entropy in the verifier input. CLIP Score directly quantifies alignment, while User Study Preference captures perceptual quality. The cost metric ensures the method is practical; we expect to maintain low query counts on unambiguous prompts. By including these baselines and ablations, we can trace any performance gains to the specific design choices and validate the entropy-context link through controlled comparisons.

### Expected outcome and causal chain

**vs. Stable Diffusion 3** — On a case where the prompt lacks detail, e.g., "a person holding a smartphone" with no mention of environment, SD3 generates a generic person in a plain background because it has no mechanism to infer missing context. Our method instead triggers retrieval on high cross-attention entropy (verified by classifier), obtains additional context (e.g., "in a coffee shop") from the LLM, updates the text embedding, and re-predicts noise, resulting in a more aligned image. We expect a noticeable CLIP score gap (>0.05) on ambiguous prompts, but near parity on simple prompts where entropy is low and no retrieval occurs.

**vs. ACAD w/o adaptive** — On a prompt that is already fully specified, e.g., "a red apple on a white table", ACAD w/o adaptive always retrieves context even when unnecessary, potentially adding irrelevant details (e.g., "with a green leaf") that degrade alignment or slow generation. Our method avoids retrieval because entropy is low (and verifier negative), preserving the original prompt integrity. On underspecified prompts, both retrieve and improve, but our method retrieves only when needed, reducing noise and computational cost. We expect our method to achieve higher average CLIP score, especially on a mix of simple and complex prompts, with a >0.02 advantage.

**vs. ACAD with entropy-only trigger (no verifier)** — On abstract prompts like "an abstract painting of emotions", entropy may be high due to inherent ambiguity rather than missing context. Our full method (with verifier) suppresses retrieval because the verifier learns to recognize such cases from calibration data. The entropy-only method would falsely retrieve and likely degrade quality. We expect our method to significantly outperform on abstract prompts (+0.03 CLIP) while matching on concrete prompts.

**vs. ACAD with alternative uncertainty measures (LV or PE)** — Entropy better captures text-image misalignment than variance (which captures global image variance) or predictive entropy (which is not directly text-specific). We expect entropy-based verifier to yield highest CLIP on ambiguous prompts, with LV and PE performing worse by >0.01 due to higher false trigger rates.

### What would falsify this idea

If our method shows no correlation between retrieval decisions and prompt ambiguity (e.g., uniform gain across all prompts, or worse performance on ambiguous subsets), then the central claim that adaptive entropy-based retrieval improves alignment is invalid. Specifically, if the gain is equally large on simple prompts, it suggests retrieval is always beneficial, contradicting the adaptive trigger's necessity. Additionally, if the verifier does not reduce false retrievals on abstract prompts (comparing to entropy-only), then the load-bearing assumption that entropy indicates context insufficiency is not salvageable even with calibration.

## References

1. Qwen-Image-Agent: Bridging the Context Gap in Real-World Image Generation
2. DraCo: Draft as CoT for Text-to-Image Preview and Rare Concept Generation
3. FLUX.1 Kontext: Flow Matching for In-Context Image Generation and Editing in Latent Space
4. ELLA: Equip Diffusion Models with LLM for Enhanced Semantic Alignment
5. TokenFlow: Unified Image Tokenizer for Multimodal Understanding and Generation
6. Scaling Rectified Flow Transformers for High-Resolution Image Synthesis
7. Ranni: Taming Text-to-Image Diffusion for Accurate Instruction Following
8. PixArt-α: Fast Training of Diffusion Transformer for Photorealistic Text-to-Image Synthesis
