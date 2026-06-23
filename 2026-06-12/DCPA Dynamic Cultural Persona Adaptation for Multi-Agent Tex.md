# DCPA: Dynamic Cultural Persona Adaptation for Multi-Agent Text-to-Image Generation

## Motivation

Existing multi-agent systems for multicultural text-to-image generation, such as MosAIG (When Cultures Meet...), rely on static, predefined cultural personas that cannot adapt to dynamic cultural contexts or intersectional identities. This structural limitation arises because cultural identity is encoded as fixed attributes (e.g., nationality, landmark) without any mechanism to refine these representations based on generated visual outcomes. Our approach overcomes this by introducing a closed-loop adaptation mechanism that updates agent personas using contrastive feedback from generated images versus cultural prototypes.

## Key Insight

Cultural identity is a latent variable that can be inferred from the input prompt and iteratively refined by minimizing the discrepancy between generated visual features and prototypical cultural features, enabling adaptive, context-sensitive personas.

## Method

### (B) How it works
```pseudocode
# Hyperparameters: K=3 agents, prototype bank P (50 prototypes/culture), α=0.1 (update step), λ=0.5 (contrastive temperature), CMA-ES generations=20, initial sigma=0.01
# Assumption: Persona embeddings must be modified to influence the generated image; we use CMA-ES to avoid reliance on LLM differentiability.
1. Initialize persona embeddings e_c for each culture c (from pretrained CLIP embeddings of cultural keywords).
2. For each agent i (1..K), assign a primary culture c_i (sampled from set of cultures in prompt).
3. Generate prompt: Each agent produces a sub-prompt using its persona e_{c_i} (e.g., via a frozen LLM conditioned on e). Concatenate sub-prompts into final prompt P_final.
4. Generate image I using a frozen text-to-image model (e.g., Stable Diffusion).
5. Extract visual features v = CLIP_image(I).
6. For each culture c in prompt:
   a. Retrieve positive prototype p_pos = nearest prototype from bank P_c to v (cosine similarity).
   b. Sample negative prototypes p_neg from other cultures' banks (5 random).
   c. Compute contrastive loss L_c = -log( exp(sim(v, p_pos)/λ) / ( exp(sim(v, p_pos)/λ) + Σ exp(sim(v, p_neg)/λ) ) ).
   d. Update persona embedding: Use CMA-ES to optimize e_c to minimize L_c (objective). Run 20 generations per iteration, initial sigma=0.01.
7. Repeat steps 3-6 for T iterations (T=5).
8. Return final image I and updated personas e_c.
```

### (C) Why this design
We chose a contrastive update rule over a regression-based loss (e.g., L2 to prototype) because contrastive learning naturally handles the multi-cultural alignment: it pushes each persona toward prototypical features specific to its culture while repelling from others, avoiding mode collapse. One design decision is the use of a frozen LLM for sub-prompt generation conditioned on persona embeddings rather than fine-tuning the LLM; this avoids catastrophic forgetting but limits the expressiveness of prompt composition, accepting the cost that persona updates must be reflected through changes in the embedding rather than in the LLM parameters. Another decision is to update personas via CMA-ES (a derivative-free optimizer) rather than via gradient descent; this eliminates the need for LLM differentiability but adds computational overhead (20 evaluations per iteration). A third decision is to use a precomputed prototype bank per culture (50 prototypes from 1000 images each, scraped from Getty Images using cultural keyword queries and CLIP clustering) instead of online clustering; this ensures stable targets but limits adaptation to unseen visual styles not in the bank. Finally, we set T=5 iterations as a trade-off: more iterations improve alignment but add latency; we found 5 sufficient in preliminary experiments without overfitting.

### (D) Why it measures what we claim
Contrastive loss L_c measures cultural alignment because it operationalizes the concept of "persona adapting to generated visual output" under the assumption that the closest prototype in the bank represents the correct cultural visual features for that culture; when the generated image contains spurious or mixed features (e.g., a landmark from a non-target culture), the positive prototype may be mismatched, causing L_c to instead reflect similarity to a different culture's prototype, which would drive the persona toward the wrong visual norm. The prototype bank measures cultural prototypicality under the assumption that the bank exhaustively captures the visual diversity of each culture; this assumption fails for highly intersectional or novel cultural expressions not present in the bank, in which case the prototypes become noisy and persona updates may lead to unnatural combinations. The persona embedding update step measures adaptability because the CMA-ES optimization minimizes L_c, adjusting the persona to produce prompts that yield images more similar to the target culture's prototypes; but this relies on the assumption that the frozen LLM's sub-prompt generation is sensitive to changes in the persona embedding—if the LLM ignores small perturbations, updates may have no effect, and the measure reflects only the optimization progress without actual behavioral change.

## Contribution

(1) A novel framework for dynamic cultural persona adaptation in multi-agent text-to-image generation, where agent personas are updated based on visual feedback from generated images. (2) A contrastive update rule that aligns generated images with cultural prototypes without requiring explicit labels or fine-tuning the generative model. (3) Empirical demonstration that dynamic adaptation improves cultural alignment and fairness over static baselines.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|----------------------|
| Dataset | Multicultural benchmark (e.g., "When Cultures Meet") | Tests cross-cultural alignment diversity |
| Primary metric | CLIP alignment score | Measures culture-specific visual faithfulness |
| Baseline 1 | Single-prompt generation | Lacks cultural composition; naive baseline |
| Baseline 2 | MosAIG (multi-agent without dynamic update) | Tests need for persona adaptation |
| Baseline 3 | Single-agent generic persona | Tests benefit of multi-agent structure |
| Ablation 1 | DCPA w/o contrastive loss (fixed personas) | Isolates effect of contrastive learning |
| Ablation 2 | DCPA w/ regressive update (maximize positive similarity only) | Tests necessity of negative repulsion |

### Why this setup validates the claim

The experimental design tests the central claim that dynamic persona adaptation improves cultural alignment in multi-agent text-to-image generation. The multicultural benchmark with prompts containing multiple distinct cultures directly challenges alignment. Single-prompt generation tests the simplest composition, while MosAIG (static multi-agent) isolates the need for dynamic updates. A single-agent generic persona tests the necessity of multi-agent specialization. The first ablation removes contrastive learning to quantify its contribution; the second replaces contrastive with regressive update to isolate the effect of negative repulsion. CLIP score is chosen as the primary metric because it directly evaluates how well generated visual features match cultural prototypes. Additionally, we perform a sensitivity analysis: vary prototype bank size (10, 50, 100) and measure the cosine similarity between sub-prompts generated by the LLM with perturbed persona embeddings from CMA-ES updates, to verify that our optimization effectively changes LLM output. This combination yields a falsifiable test: if DCPA outperforms baselines specifically on multi-culture prompts (where adaptation is critical), the claim is supported; if gains are uniform, the mechanism is not as hypothesized.

### Expected outcome and causal chain

**vs. Single-prompt generation** — On a case like "A wedding in India and a temple in Japan", the baseline concatenates cultures superficially, producing an image with mixed or inaccurate cultural details (e.g., a Japanese-style wedding or Indian temple borrowed from wrong culture) because it fails to compose separate cultural norms. Our method generates sub-prompts per culture after dynamic persona updates, yielding distinct, accurate cultural elements (e.g., wedding with Indian rituals, temple with Japanese architecture) because contrastive learning refines each persona toward visual prototypes. We expect a noticeable gap on prompts with two or more cultures, with DCPA achieving 5-10% higher CLIP score on each culture's alignment, but parity on single-culture prompts.

**vs. MosAIG** — On a case with visually similar cultures (e.g., Japanese and Korean temples), MosAIG uses static personas that may produce overlapping features (e.g., shared roof styles) because it lacks feedback from the generated image. Our method iteratively updates personas via contrastive loss against culture-specific prototypes, repelling from negative prototypes of other cultures, so the sub-prompts become more discriminative over iterations. We expect DCPA to show increasing CLIP alignment over iterations (e.g., from 0.65 to 0.75) while MosAIG stagnates, and a larger gap on culture pairs with high visual similarity.

**vs. Single-agent generic persona** — On a case with three distinct cultures (e.g., Mexican, Indian, and Italian food in one scene), a single agent with a generic persona struggles to produce culturally accurate details for all three simultaneously because it averages across cultures. Our method assigns each culture a specialized agent with its own dynamic persona, enabling each to focus on its cultural prototypes. The generated image for DCPA will contain recognizable cultural artifacts for each (e.g., tacos, curry, pasta) while the baseline mixes features (e.g., curry-sauce pasta). We expect DCPA to achieve 10-15% higher CLIP score on each culture's subset and stronger human-judged cultural accuracy.

**vs. Ablation 2 (regressive update)** — On a case with overlapping visual cultures, the regressive ablation (maximizing similarity to positive prototype only) may converge to a shared visual norm (e.g., both Japanese and Korean temples resembling one another) because it lacks repulsion from negative prototypes. DCPA with contrastive update will maintain distinct prototypes per culture, yielding higher CLIP alignment for each individual culture. We expect DCPA to outperform the regressive ablation by 3-5% on CLIP score on multi-culture prompts with visually similar cultures, while being comparable on single-culture prompts.

### What would falsify this idea
If DCPA's improvement over baselines is uniform across all prompt types (e.g., similar gain on single-culture prompts) rather than concentrated on multi-culture prompts where the dynamic adaptation should provide a clear advantage, then the central claim that persona adaptation drives the improvement would be invalidated.

## References

1. When Cultures Meet: Multicultural Text-to-Image Generation
2. GenArtist: Multimodal LLM as an Agent for Unified Image Generation and Editing
3. The Power of Many: Multi-Agent Multimodal Models for Cultural Image Captioning
4. TextDiffuser-2: Unleashing the Power of Language Models for Text Rendering
5. Guiding Instruction-based Image Editing via Multimodal Large Language Models
6. AppAgent: Multimodal Agents as Smartphone Users
7. Self-Correcting LLM-Controlled Diffusion Models
8. Towards Language Models That Can See: Computer Vision Through the LENS of Natural Language
