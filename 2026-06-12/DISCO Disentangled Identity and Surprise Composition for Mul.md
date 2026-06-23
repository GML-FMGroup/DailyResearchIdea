# DISCO: Disentangled Identity and Surprise Composition for Multi-Agent Text-to-Image Generation

## Motivation

Existing multi-agent text-to-image methods like MosAIG (When Cultures Meet) assume static, pre-defined agent personas and lack mechanisms to preserve agent identity when composing with others, leading to identity distortion and an inability to introduce new agents for novel contexts. The root cause is the absence of context-invariant identity embeddings and a principled novelty detector, which this method addresses.

## Key Insight

Identity embeddings that are contrastively trained to be invariant across compositional contexts ensure each agent's visual signature remains consistent regardless of which other agents appear, while generative surprise over the resulting image distribution provides a principled detector for when a novel identity should be introduced.

## Method

### (A) What it is
DISCO (Disentangled Identity and Surprise Composition) is a framework that maintains a set of identity embeddings (1024-dimensional), each representing an agent, and uses them to condition a text-to-image model. Its inputs are a set of agent identities (possibly empty) and a global prompt; its output is a composed prompt with identity-specific tokens and a generated image.

### (B) How it works

```python
# DISCO Core Operation
# Hyperparameters: lambda_cont = 0.1, tau = 0.07, threshold_surprise = 0.5, embedding_dim = 1024, num_samples_surprise = 50

identity_embeddings = [...]  # learned 1024-d embedding per agent
composition_module = 2-layer MLP(input=1024, hidden=256, output=256, activation='GeLU')  # projects to joint space

for each training batch with ground-truth agents and prompts:
    # Step 1: Composed prompt generation with attention regularization
    composed_prompt = compose(global_prompt, identity_embeddings)  # includes [IDENTITY_i] tokens
    latents = diffusion_model(composed_prompt)
    # Compute attention distortion penalty: L_attn = sum_i Var(attention_maps_i) where variance is over spatial positions and heads
    L_attn = attention_variance_penalty(latents, identity_tokens)
    
    # Step 2: Contrastive identity invariance (InfoNCE with temperature tau)
    # For each agent i, create alternative compositions with different random subsets of other agents
    for i in agents:
        pos_compositions = sample_compositions_with(i)  # include i with various others
        neg_compositions = sample_compositions_without(i)
        # Extract identity embedding from composed prompt representations (after composition module)
        z_i = composition_module(identity_embeddings[i])  # project to joint space (256-d)
        z_pos = [composition_module(embed) for embed in pos_compositions]
        z_neg = [composition_module(embed) for embed in neg_compositions]
        # Contrastive loss: L_cont = -log( exp(sim(z_i, z_pos)/tau) / (exp(sim(z_i, z_pos)/tau) + sum exp(sim(z_i, z_neg)/tau)) )
        L_cont += contrastive_loss(z_i, z_pos, z_neg, tau)  # sim is cosine similarity
    
    # Step 3: Novelty detection via generative surprise
    # During inference, for a candidate new agent with embedding e_new (initialized randomly, 1024-d)
    composed_prompt_new = compose(global_prompt, identity_embeddings + [e_new])
    image_new = generate(composed_prompt_new)
    # Generative surprise: KL divergence between image feature distributions with and without e_new
    # Approximated via CLIP image features: compute mean and covariance from 50 samples per condition
    surprise = KL(p(image | existing), p(image | existing+new))
    if surprise > threshold_surprise:
        identity_embeddings.append(e_new)  # add as new identity
        # Optionally fine-tune e_new with few steps (e.g., 1 step using L_cont on a small set of compositions)
    else:
        reject e_new
    
    # Total loss: L = L_cont + lambda_attn * L_attn

# Load-bearing assumption: The contrastive loss can learn identity embeddings that are invariant across diverse compositional contexts and that this invariance transfers to generated images.
# Calibration: After training, collect human ratings on identity preservation for a held-out set of compositions. Compute Pearson correlation between contrastive similarity (cosine between z_i and z_pos) and human rating. If correlation < 0.5, increase number of identity tokens per agent to 3 or increase embedding dimension to 2048.
```

### (C) Why this design
We chose contrastive learning over an autoencoder reconstruction loss because contrastive learning explicitly enforces invariance across compositional contexts without requiring a reconstruction decoder, which would be computationally expensive and could couple identity with appearance details. The attention variance penalty was chosen over a simple identity token loss because it directly prevents the attention mechanism from shifting the identity token's focus to other agents, preserving the agent's contribution to the scene. We use a fixed threshold for generative surprise rather than a learned one because the threshold is tied to the diffusion model's intrinsic variability, making it dataset-agnostic and avoiding overfitting. The trade-off is that contrastive learning requires diverse compositional pairs, which may be costly to collect; attention penalty adds hyperparameter tuning; and a fixed threshold may miss subtle novelties. We also chose to initialize new identity embeddings randomly and accept them based on surprise, rather than clustering or using a prior, because random initialization allows the embedding to be shaped by subsequent fine-tuning, avoiding bias from cluster centroids. This design prioritizes flexibility and invariance at the cost of increased training data requirements and sensitivity to threshold selection.

### (D) Why it measures what we claim
The contrastive loss directly measures identity invariance because it forces the embedding of an agent to be closer to its own embeddings in different compositional contexts than to embeddings of other agents; this equivalence relies on the assumption that the composition module linearly projects identity embeddings into a joint space where similarity corresponds to identity consistency. This assumption fails if the composition module is highly nonlinear or if identity is not linearly separable, in which case contrastive loss may enforce invariance to irrelevant variations. The attention variance penalty measures identity preservation because it penalizes the spread of attention from an agent's identity token across the image, assuming that a focused attention map indicates the agent's influence on its own region; this fails if the token necessarily attends to multiple regions (e.g., for diffuse concepts like atmosphere), where the penalty would incorrectly suppress attention. Generative surprise, computed as KL divergence between image feature distributions with and without a new agent, measures novelty because a high divergence indicates the new agent induces a distribution shift beyond the model's existing knowledge. This assumes that the image feature distribution is sensitive to identity-relevant changes, which fails for very small agents or subtle attributes, where surprise might be low even when a genuinely new identity is introduced. Together, these components operationalize the motivation concepts of identity preservation and dynamic creation with explicit assumptions and failure modes.

## Contribution

(1) A contrastive learning framework for identity embeddings that are invariant to compositional context in multi-agent text-to-image generation. (2) A novelty detection mechanism using generative surprise to dynamically create new agent identities when needed, without retraining. (3) An integrated system that preserves agent identity across compositions and adapts to novel cultural contexts, demonstrated on the multicultural benchmark.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | Multi-Identity Composition (MIC) benchmark | Controlled identities & compositional variations |
| Primary metric | Identity-aligned CLIP Score | Measures identity preservation in compositions |
| Baseline 1 | Single-prompt naive concatenation | No multi-agent identity handling |
| Baseline 2 | MosAIG (multi-agent baseline) | Uses separate prompts per agent |
| Ablation-of-ours | DISCO without attention penalty | Tests necessity of attention regularization |

### Why this setup validates the claim

This combination forms a falsifiable test of DISCO's central claim—maintaining identity embeddings across compositions and dynamically adding new ones. The MIC benchmark provides controlled compositions with known identities, isolating the effect of multi-agent handling. The identity-aligned CLIP Score directly measures whether generated images preserve each identity's visual characteristics. Single-prompt naive concatenation tests the need for explicit identity separation; MosAIG tests whether contrastive invariance helps when identities are similar; the ablation of the attention penalty tests whether attention regularization prevents identity token drift. If DISCO truly enhances identity preservation and compositionality, it should outperform both baselines on composite prompts, especially where identities share features or require spatial separation, while the ablation should degrade on spatial tasks. Any deviation from this pattern would challenge the proposed mechanisms.

### Expected outcome and causal chain

**vs. Single-prompt naive concatenation** — On a case with two contrasting identities (e.g., "a robot and a firefighter"), the baseline produces a single hybrid entity because it naively concatenates descriptors without separating identities. Our method generates both distinct entities because contrastive loss keeps identity embeddings distinct and attention penalty prevents token overlap. We expect a large gap in identity-aligned CLIP Score on such complex compositions, but parity on single-identity prompts.

**vs. MosAIG** — On a case where identities share visual features (e.g., two similar cat breeds), MosAIG conflates them because it processes prompts independently and then combines, lacking identity invariance across contexts. Our method enforces embedding invariance via contrastive loss, so identities remain separable despite feature similarity. We expect a noticeable gap on subsets with high identity similarity, while performance on dissimilar identities remains comparable.

**vs. DISCO w/o attention penalty** — On a case requiring spatial separation (e.g., "a dog on the left, a cat on the right"), the ablation allows the dog's identity token to attend to the cat region, causing visual overlap. Our method penalizes attention variance, focusing each token on its own region. We expect a drop in metric for scenes with distinct spatial roles, but similar performance for spatially uniform scenes.

### What would falsify this idea

If DISCO's gain over baselines is uniform across all identity compositions rather than concentrated on challenging subsets (e.g., similar identities, spatial separation), or if the ablation does not degrade on spatial tasks, then the proposed identity preservation mechanisms are not driving the improvement.

## References

1. When Cultures Meet: Multicultural Text-to-Image Generation
2. GenArtist: Multimodal LLM as an Agent for Unified Image Generation and Editing
3. The Power of Many: Multi-Agent Multimodal Models for Cultural Image Captioning
4. TextDiffuser-2: Unleashing the Power of Language Models for Text Rendering
5. Guiding Instruction-based Image Editing via Multimodal Large Language Models
6. AppAgent: Multimodal Agents as Smartphone Users
7. Self-Correcting LLM-Controlled Diffusion Models
8. Towards Language Models That Can See: Computer Vision Through the LENS of Natural Language
