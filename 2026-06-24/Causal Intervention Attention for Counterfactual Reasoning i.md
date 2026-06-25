# Causal Intervention Attention for Counterfactual Reasoning in Text-to-Image Generation

## Motivation

Current text-to-image models, as shown by the CF-World benchmark (Are Text-to-Image Models Inductivist Turkeys?), fail on implicit counterfactual scenarios because they lack explicit causal representations. These models treat objects and actions as independent features, ignoring the cause-effect structure that governs how changing one variable alters the scene. This structural gap prevents generalization to novel interventions.

## Key Insight

Embedding a differentiable causal graph into the attention mechanism forces the model to attribute visual features to their causal parents, making counterfactual reasoning a byproduct of the learned structure.

## Method

### Causal Intervention Attention (CIA) Module

**What it is**: The Causal Intervention Attention (CIA) module is a plug-in architectural extension for transformer-based T2I models (e.g., Janus-Pro). It accepts hidden states from the previous transformer layer and outputs causally-attended representations that respect a learned causal graph among object and action tokens.

**How it works** (pseudocode):
```python
def causal_attention(X, text_emb, causal_graph_logits):
    # X: hidden states [batch, seq_len, dim=768]
    # text_emb: token embeddings with object/action annotation (from CF-World parse)
    # causal_graph_logits: learnable parameter [num_obj+act, num_obj+act] (init=0, sigmoid to [0,1])
    
    # Step 1: Standard self-attention
    Q = X @ W_Q  # W_Q: dim×dim
    K = X @ W_K  # W_K: dim×dim
    V = X @ W_V  # W_V: dim×dim
    d = 768
    attn_scores = Q @ K.T / sqrt(d)  # [batch, heads, seq_len, seq_len]
    
    # Step 2: Convert causal graph to attention mask
    # For each token pair (i,j), mask = 0 if causal_graph_logits[i,j] > 0 or i==j, else -inf
    # Only apply to tokens corresponding to objects/actions; padding and special tokens attend to all
    causal_graph = torch.sigmoid(causal_graph_logits)  # [num_entities, num_entities]
    causal_mask = torch.where(causal_graph > 0.5, 0.0, float('-inf'))
    # Expand to full seq_len (including special tokens, connectors)
    causal_mask = expand_to_seq_len(causal_mask, entity_indices)  # [seq_len, seq_len]
    # Allow self-attention (i==j) always
    causal_mask = causal_mask + torch.eye(seq_len) * (0.0 - causal_mask)
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # broadcast over batch, heads
    
    # Step 3: Apply causal mask and softmax
    causal_attn = softmax(attn_scores + causal_mask)
    
    # Step 4: Weighted sum
    output = causal_attn @ V
    
    # Step 5: Supervised graph loss (from ground-truth causal edges)
    # During training, ground_truth_edges is provided from CF-World annotations
    graph_loss = F.binary_cross_entropy(causal_graph.flatten(), ground_truth_edges.flatten())
    
    # Optional sparsity loss (L1 on causal_graph)
    sparsity_loss = L1(causal_graph)
    
    return output, graph_loss, sparsity_loss
```
Hyperparameters: causal_graph_logits dimension [num_entities=50, num_entities=50] (maximum 50 object/action types in CF-World); learning rate for graph parameters 1e-4; graph supervised loss weight λ_graph=1.0; sparsity weight λ_sparse=0.1; attention heads=16.

Training uses paired data from CF-World where each sample consists of original prompt, original image, counterfactual prompt, counterfactual image, and ground-truth causal edges (binary matrix indicating which variable changes cause changes in others). The model is trained with a multi-task loss: total loss = L_generation (MSE on image pixels) + λ_graph * graph_loss + λ_sparse * sparsity_loss. Generation loss ensures the output matches the images; graph loss aligns causal graph with true causal structure; sparsity encourages minimality.

(A) **Load-bearing assumption**: The ground-truth causal edges from CF-World are accurate and complete for the tasks, and the entity-to-token mapping (via text parsing) is exact. If this fails, the supervised graph loss misleads the model.

(B) **Calibration/verification**: After training, we freeze the causal graph and evaluate its edge accuracy on a held-out subset of CF-World (512 samples with ground-truth edges). This measures how well the learned graph approximates the true causal structure. Additionally, we perform a stress test: on synthetic samples with known causal relations (e.g., two objects, one action that affects a single attribute), we compare learned edges to ground truth.

**Why this design**: We chose a learnable causal graph over a fixed commonsense knowledge base because the graph can adapt to the dataset's causal structure, avoiding the brittleness of hand-crafted rules. The graph is integrated as an attention mask rather than a separate module to keep inference end-to-end and avoid additional latency. We use a soft mask (softmax over logits) instead of hard masking to allow gradient flow through the graph during training, accepting the cost that spurious edges may initially persist. Counterfactual contrastive loss was chosen over direct causal effect estimation because it provides a training signal without requiring ground-truth intervention targets, but this relies on the assumption that the intervention on the token embedding simulates a real-world change—a failure mode if the token does not correspond to a causal variable. The sparsity loss on graph edges (L1) encourages a minimal causal structure, trading off flexibility for interpretability.

**Why it measures what we claim**: The causal attention mask (Step 2) operationalizes causal dependency: a non-zero edge between token i and j means the model must attend to i when generating j because the causal graph asserts that i is a cause of j. This measures causal necessity—the degree to which features of j depend on i—under the assumption that the learned graph correctly captures true causal relations; this assumption fails when the graph misses edges or includes spurious ones, in which case the mask either blocks necessary information or allows noise, degrading counterfactual consistency. The supervised graph loss directly enforces this assumption by using ground-truth edges from CF-World, making the graph accuracy measurable. The counterfactual contrastive loss (though replaced by generation loss) previously measured causal sensitivity; now generation loss ensures output matches true counterfactual images, which provides stronger supervision.

## Contribution

(1) The Causal Intervention Attention (CIA) module, an architectural extension for T2I transformers that incorporates a learnable causal graph into attention computation, enforcing explicit cause-effect reasoning during generation. (2) The counterfactual contrastive training objective that supervises the model to respect interventional consistency, operating on factual/counterfactual pairs generated from the causal graph. (3) Empirical demonstration that CIA improves performance on CF-World implicit counterfactual scenarios while maintaining generation quality.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|-----------------------|
| Dataset | CF-World | Tests counterfactual reasoning explicitly. |
| Primary metric | CF-Eval correctness | Measures causal consistency of generated images. |
| Baseline 1 | Janus-Pro (no CIA) | Isolates effect of CIA module. |
| Baseline 2 | Fixed commonsense graph | Tests advantage of learnable graph. |
| Baseline 3 | DALL-E 3 | Strong external SOTA for comparison. |
| Ablation | CIA w/o counterfactual loss | Checks necessity of contrastive training. |

### Why this setup validates the claim

The CF-World dataset explicitly targets counterfactual scenarios, and CF-Eval (a VLM) evaluates whether the generated image respects the intervention. Janus-Pro without CIA isolates the module's contribution, as both share the same base architecture. The fixed commonsense graph baseline tests the key claim that a learnable causal graph adapts to dataset-specific causality better than static rules. DALL-E 3 provides a strong external reference point for practical effectiveness. The ablation of the counterfactual loss tests whether the contrastive training is essential for aligning the graph with true causal structure. CF-Eval is the right metric because it directly scores the accuracy of causal changes, focusing on the reasoning ability we aim to improve. Together, these choices create a falsifiable test: if our method fails to outperform on subsets where causal reasoning is required, the central claim is undermined.

### Expected outcome and causal chain

**vs. Janus-Pro (no CIA)** — On a counterfactual prompt like "a red car that becomes blue," baseline attention mixes all tokens, often keeping red features because of strong correlation. Our CIA module imposes a causal mask forcing attention to the color token, and the counterfactual loss reinforces that change. We expect a noticeable gap on explicit counterfactuals (e.g., >20% accuracy improvement) but parity on factual prompts.

**vs. Fixed commonsense graph** — On an unusual causal relation like "a spinning top that causes the background to blur," the fixed graph lacks this edge so the model ignores the action-spin token. Our learned graph can discover this correlation from training data, enabling correct attending. We expect significant advantage on rare but plausible causal links, with the gap concentrated on prompts outside common knowledge.

**vs. DALL-E 3** — On an implicit counterfactual requiring a causal chain (e.g., "if the vase is knocked over, the water spills"), DALL-E 3 may rely on visual similarity to unrelated spills. Our method enforces causal dependency via the graph and loss, leading to more consistent outputs. We expect comparable or better performance on complex causal scenarios, but possibly lower on simple stylistic prompts.

**Ablation (CIA w/o counterfactual loss)** — Without the contrastive loss, the learned graph may converge to an all-ones mask (trivial attend-all). Thus, counterfactual consistency drops while factual remains high. We expect a noticeable decline in CF-Eval accuracy on counterfactual subsets, confirming the loss's role in driving causal structure.

### What would falsify this idea
If our method's accuracy gain is uniform across factual and counterfactual subsets, or if the fixed commonsense graph matches our performance on all subsets, then the central claim about learnable causal graphs enabling causal reasoning is false—improvements would stem from general factors like regularization rather than captured causality.

## References

1. Are Text-to-Image Models Inductivist Turkeys? A Counterfactual Benchmark for Causal Reasoning
2. Commonsense-T2I Challenge: Can Text-to-Image Generation Models Understand Commonsense?
3. ImagenHub: Standardizing the evaluation of conditional image generation models
4. X-IQE: eXplainable Image Quality Evaluation for Text-to-Image Generation with Visual Large Language Models
5. T2I-CompBench++: An Enhanced and Comprehensive Benchmark for Compositional Text-to-Image Generation
6. Scaling Instruction-Finetuned Language Models
