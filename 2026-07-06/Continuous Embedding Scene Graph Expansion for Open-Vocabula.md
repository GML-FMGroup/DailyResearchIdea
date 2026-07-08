# Continuous Embedding Scene Graph Expansion for Open-Vocabulary Conditional Image Generation

## Motivation

Existing scene graph enrichment methods, such as Generated Contents Enrichment and NeuSyRE, rely on a closed set of object and relation categories, preventing the generation of images containing novel visual concepts beyond the predefined vocabulary. This limitation arises because these methods represent categories as discrete indices or one-hot vectors, which cannot generalize to unseen categories. We need to overcome this closed-set assumption to enable truly open-vocabulary image generation.

## Key Insight

The continuous semantic embedding space of large vision-language models enables novel visual concepts to be represented as semantically coherent points, allowing open-vocabulary scene graph expansion by predicting embeddings in this space.

## Method

(A) **What it is**: Continuous Embedding Scene Graph Expansion (CESGE) is a framework that takes an initial scene graph with nodes and edges labeled as continuous embeddings from a pretrained vision-language model (e.g., CLIP), and uses a conditional diffusion model to predict additional nodes and edges as continuous embeddings in the same space. The enriched scene graph can then be decoded into visual features for image generation.

(B) **How it works** (pseudocode):

```python
# Training
for each training sample (initial_graph, target_graph):
    # Map discrete categories to CLIP text embeddings
    initial_emb = [CLIP_text(cat) for cat in initial_graph.nodes]
    initial_rel_emb = [CLIP_text(rel) for rel in initial_graph.edges]
    target_emb = [CLIP_text(cat) for cat in target_graph.nodes]
    target_rel_emb = [CLIP_text(rel) for rel in target_graph.edges]
    
    # Condition: initial embeddings + graph structure (adjacency matrix)
    cond = (initial_emb, initial_rel_emb, adjacency)
    
    # Diffusion forward: add noise to target embeddings
    noise = random_noise_like(target_emb)
    noised = sqrt(alpha_bar)*target_emb + sqrt(1-alpha_bar)*noise  # cosine schedule
    
    # Predict noise via denoising network conditioned on cond
    pred_noise = Denoiser(noised, timestep, cond)
    loss = MSE(pred_noise, noise)
    # Additional losses
    # - Semantic consistency: cosine distance between predicted clean embeddings and CLIP image region features
    # - Relation plausibility: pretrained relation classifier on predicted relation embeddings
    total_loss = diffusion_loss + lambda1 * semantic_loss + lambda2 * relation_loss
    optimize

# Inference
def enrich_graph(initial_graph, num_additional_nodes=5):
    initial_emb = [CLIP_text(cat) for cat in initial_graph.nodes]
    initial_rel_emb = [CLIP_text(rel) for rel in initial_graph.edges]
    cond = (initial_emb, initial_rel_emb, adjacency)
    
    # Start from random noise for additional nodes/edges
    noise = torch.randn(num_additional_nodes, 512)  # 512-dim CLIP space
    
    # Reverse diffusion (DDIM for speed)
    for t in reversed(range(1000)):
        noise = Denoiser(noise, t, cond)
    
    enriched_nodes = noise  # continuous embeddings
    # Also predict relations using a separate decoder or edge prediction network
    enriched_rels = EdgePredictor(enriched_nodes, cond)
    return enriched_nodes, enriched_rels
```

Hyperparameters: T=1000 diffusion steps, cosine noise schedule, DDIM sampling with 50 steps during inference. Additional nodes k=5 for experiments, but during training we sample k ~ Poisson(λ=3).

(C) **Why this design** (trade-off reasoning paragraph):
We chose a diffusion model over a deterministic GCN predictor because diffusion can model the multimodality of plausible enrichments (e.g., an empty room could contain either a chair or a table) by sampling from the posterior, at the cost of slower inference (multiple denoising steps) and more complex training (denoising objective). We use CLIP embeddings as the continuous representation because they are semantically structured and grounded in both vision and language, enabling interpolation to form novel concepts; however, CLIP embeddings may not capture fine-grained visual distinctions (e.g., between 'armchair' and 'recliner'), which we mitigate by adding a region-level similarity loss that encourages visual consistency. We condition the diffusion on the entire initial graph using a graph transformer encoder (with multi-head attention over nodes and edges) to capture global relational context, rather than treating each node independently; the trade-off is O(n^2) computational cost due to pairwise attention, but it ensures coherent enrichment. We generate a fixed number of additional nodes k during inference for simplicity, but during training we sample k from a Poisson distribution (λ=3) to encourage variable-length outputs; we accept that this hyperparameter may need tuning per dataset.

(D) **Why it measures what we claim** (causal-bridge paragraph):
The predicted node embedding e_v (continuous vector) measures the semantic category of the novel object because we assume that the CLIP embedding space is locality-preserving under semantic similarity, i.e., objects with similar meaning have close embeddings. This assumption fails when two visually distinct objects have overlapping textual descriptions (e.g., 'smartphone' vs. 'mobile phone'); in that case, e_v reflects textual similarity rather than visual distinctness. The diffusion reconstruction loss (MSE between predicted noise and true noise) measures the model's ability to capture the conditional distribution of enrichments given the initial graph, assuming the denoising objective proxies likelihood. This assumption fails if the noise schedule is poorly calibrated or the conditioning graph is out-of-distribution, in which case the loss may be low but samples are unrealistic. The semantic consistency loss (cosine similarity between predicted embedding and CLIP image region features) measures visual grounding, assuming the CLIP image encoder provides a reliable region representation; this fails for small or occluded objects where region features are noisy. The relation plausibility loss (from a pretrained classifier) measures interaction correctness, assuming the classifier generalizes to novel object types; it fails when the classifier was trained on fixed categories and sees an unseen pair, leading to arbitrary outputs.

## Contribution

(1) A novel framework for open-vocabulary scene graph expansion that represents categories as continuous embeddings from a pretrained vision-language model, enabling generation of novel visual concepts without predefined category sets. (2) A conditional diffusion model that operates on continuous graph embeddings, allowing stochastic sampling of plausible enrichments while maintaining structural coherence. (3) Empirical demonstration that the method can produce semantically diverse and visually plausible enriched scene graphs for categories unseen during training.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|----------------------|
| Dataset | Visual Genome | Large, diverse scene graphs for conditional generation |
| Primary metric | FID (Fréchet Inception Distance) | Measures image realism and diversity |
| Baseline 1 | Generated Contents Enrichment | Tests need for explicit scene graph |
| Baseline 2 | NeuSyRE | Tests symbolic vs continuous embeddings |
| Baseline 3 | Standard SGBI | Tests benefit of graph expansion |
| Ablation-of-ours | CESGE w/o semantic loss | Tests importance of visual grounding |

### Why this setup validates the claim

This experimental design creates a falsifiable test of our central claim that continuous embedding scene graph expansion via diffusion improves conditional image generation. The dataset Visual Genome provides complex scenes with rich object relations, where enrichment is crucial. FID as primary metric directly assesses generated image quality and diversity, capturing improvements in realism. The baselines isolate specific sub-claims: the implicit baseline (Generated Contents Enrichment) tests whether explicit scene graph representation is necessary; NeuSyRE tests whether continuous embeddings (ours) outperform symbolic ones; standard SGBI tests whether expansion adds value. The ablation removes semantic consistency loss, testing the contribution of visual grounding to embedding quality. Together, they form a ladder: if our method beats all baselines and the ablation underperforms, we confirm that continuous expansion with visual grounding drives gains. If only some comparisons hold, we pinpoint the mechanism.

### Expected outcome and causal chain

**vs. Generated Contents Enrichment** — On a scene with an empty room, the implicit baseline generates a plausible but generic image (e.g., a random chair) because it lacks an explicit graph to enforce coherent object relations. Our method instead enriches the graph with a table and lamp, producing a coherent scene because the diffusion models plausible multi-object configurations conditioned on the initial graph. We expect a noticeable FID improvement (e.g., ~2 points) on complex scenes with >3 objects, but parity on simple single-object scenes.

**vs. NeuSyRE** — On a scene containing a novel object combination (e.g., "dog riding bicycle"), NeuSyRE fails because its symbolic categories cannot generalize to unseen relations, outputting implausible interactions. Our method uses continuous CLIP embeddings that interpolate to plausible unseen concepts, enabling correct relation prediction via the relation classifier. We expect a larger gap (e.g., ~5 FID points) on scenes with rare or novel object pairs, but similar performance on common configurations.

**vs. Standard SGBI** — On a scene graph with sparse nodes (e.g., only "person"), SGBI generates an image of a person alone, missing context like furniture. Our expansion adds predicted objects (e.g., sofa, table) from the diffusion model, producing a richer scene that better matches real-world distributions. We expect a consistent FID improvement (e.g., ~3 points) across all scenes, with larger gains when initial graph under-specifies the scene.

### What would falsify this idea
If our method shows uniform FID improvement over all baselines without a larger gain on complex or novel scenes, or if the ablation matches our full method, the central claim of enrichment via continuous diffusion would be invalidated.

## References

1. Generated Contents Enrichment
2. NeuSyRE: Neuro-symbolic visual understanding and reasoning framework based on scene graph enrichment
3. Boosting Scene Graph Generation with Contextual Information
