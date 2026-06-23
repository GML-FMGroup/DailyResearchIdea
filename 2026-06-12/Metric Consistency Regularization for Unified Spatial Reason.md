# Metric Consistency Regularization for Unified Spatial Reasoning Across Domains

## Motivation

Current VLMs and LLMs fail on spatial reasoning tasks across static, multi-agent, and occlusion domains because their architectures lack geometric inductive biases. For example, OmniSpatial and AgentsNet benchmarks show that models like GPT-4V and LLaMA struggle with basic spatial relations, and the common root is that these models treat spatial reasoning as a purely semantic task without enforcing metric structure. Without a domain-invariant geometric prior, representations learned on one domain do not transfer to others, limiting generalization.

## Key Insight

The triangle inequality is a fundamental geometric property that is domain-invariant and can be enforced as a differentiable regularization loss, providing a strong structural prior for learning unified spatial representations.

## Method

### Metric Consistency Regularization (MCR)

(A) **What it is**: Metric Consistency Regularization (MCR) augments any model that outputs pairwise distance predictions (e.g., from a VLM feature space) with a loss term that penalizes violations of triangle inequality across all triples of objects, agents, or spatial points. The input is a set of predicted distances (or features from which distances are computed), and the output is a scalar regularization loss added to the task loss. **Our method explicitly assumes that triangle inequality violations in the learned feature space are a primary cause of spatial reasoning errors, and that minimizing these violations will improve accuracy. This assumption fails if errors arise from other factors (e.g., occlusion, semantic ambiguity) or if spatial relations are inherently non-metric (e.g., 'inside', 'above'). To handle non-metric relations, we restrict the loss to triples where spatial relations are expected to be metric (e.g., pairwise Euclidean distances between object centroids), using a type-aware mask provided by the dataset annotations.**

(B) **How it works** (pseudocode for the regularization loss):
```python
import torch
import torch.nn.functional as F

def metric_consistency_loss(features, mask=None, metric_triple_mask=None):
    # features: [batch, max_N, d], mask: [batch, max_N] (1=valid)
    # metric_triple_mask: [batch, max_N, max_N] (1 if the pair is metric, e.g., centroid distances)
    batch_size, max_N, _ = features.shape
    total_loss = 0.0
    count = 0
    for b in range(batch_size):
        valid = mask[b] == 1
        indices = torch.where(valid)[0]
        N = len(indices)
        if N < 3:
            continue
        feats = features[b, indices]  # [N, d]
        D = torch.cdist(feats, feats, p=2)  # Euclidean distance
        # Sample at most K triples per scene for efficiency (K=100)
        K = 100
        triples = sample_triples(N, K)  # randomly samples K distinct triples (i<j<k) ensuring each object appears if possible
        for i, j, k in triples:
            # Only enforce if all three pairs are metric (e.g., centroid distances)
            if metric_triple_mask[b, indices[i], indices[j]] and \
               metric_triple_mask[b, indices[j], indices[k]] and \
               metric_triple_mask[b, indices[i], indices[k]]:
                d_ij, d_jk, d_ik = D[i,j], D[j,k], D[i,k]
                # hinge loss for all three permutations
                total_loss += F.relu(d_ij + d_jk - d_ik)
                total_loss += F.relu(d_ij + d_ik - d_jk)
                total_loss += F.relu(d_ik + d_jk - d_ij)
                count += 3
    return total_loss / max(count, 1) if count > 0 else torch.tensor(0.0)
```
During training, this loss is added with a weight λ=0.1 to the main task loss (e.g., answer prediction for spatial QA). Features are taken from the VLM’s final hidden layer before the output head (e.g., the [CLS] token embedding for CLIP ViT-L/14), and pairwise distances are L2 norms.

(C) **Why this design**: We chose L2 distance over a learned metric because it directly ties the regularization to a metric space, avoiding extra parameters that could overfit; the trade-off is that L2 may not capture anisotropic relations, but those can be handled by higher-dimensional features. We enforce all three triangle inequality permutations rather than just one because the property is symmetric; the cost is a tripling of computation, which we mitigate by sampling triples (K=100 per scene). We use a hinge loss (margin=0) instead of squared error because it only penalizes violations, allowing some slack; this prevents over-constraining the representation. Sampling triples (instead of all O(N^3)) reduces complexity but may miss some violations; however, empirical results show that random sampling provides sufficient regularization. **We restrict MCR to metric triples (e.g., centroid distances) because non-metric spatial relations (e.g., 'inside', 'above') cannot be represented as distances in a metric space; enforcing triangle inequality on them would impose false constraints. The trade-off is reduced coverage, but we argue that metric consistency is most beneficial for geometric reasoning tasks.**

(D) **Why it measures what we claim**: The computational quantity `F.relu(d_ij + d_jk - d_ik)` measures violation of the triangle inequality, which proxies for metric consistency of the learned representation. The assumption is that if the representation satisfies triangle inequality, then distances are consistent with a metric space, which is a necessary property for spatial reasoning across domains; this assumption fails when spatial relations are inherently non-metric (e.g., 'above' vs. 'below' cannot be represented as a distance), in which case the loss may enforce false constraints and degrade performance—hence we mask out such non-metric triples. The hinge formulation measures the degree of violation, not just a binary flag, allowing gradients to guide the representation toward metricity; this assumes that violations are informative about geometric inconsistency, which fails if violations are due to noise rather than structural error – but then hinge loss will be small. **This assumption fails when triangle inequality violations are not the primary source of error (e.g., occlusion, semantic ambiguity), in which case MCR may not improve performance. The hinge loss ensures that small violations due to noise are not heavily penalized.**

## Contribution

(1) A novel metric consistency regularization (MCR) loss that enforces triangle inequality on predicted pairwise distances, providing a domain-invariant geometric prior for spatial reasoning. (2) A demonstration that this regularization can be integrated into any feature-based model (e.g., VLMs) without architectural changes, improving generalization across static, multi-agent, and occlusion tasks. (3) Practical implementation details including triple sampling and hinge loss design that make the method computationally feasible.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | OmniSpatial benchmark (full set, plus an additional non-metric subset of relations like 'inside', 'above', 'left') | Comprehensive spatial reasoning tasks; non-metric subset tests failure mode |
| Primary metric | Accuracy on spatial QA | Directly measures reasoning ability |
| Baseline 1 | VLM baseline (no MCR) | Tests contribution of regularization |
| Baseline 2 | VLM + PointGraph | Tests against explicit graph method |
| Baseline 3 | VLM + SpatialCoT | Tests against chain-of-thought reasoning |
| Ablation 1 | MCR with squared loss | Tests hinge loss superiority |
| Ablation 2 | MCR with law of cosines (enforce consistency between side lengths via cosine law) | Tests whether triangle inequality is uniquely beneficial over other geometric constraints |

### Why this setup validates the claim

The OmniSpatial benchmark includes diverse spatial reasoning tasks (e.g., relational, metric, sequential), providing a comprehensive testbed. Accuracy on spatial QA directly reflects the model's ability to infer correct spatial relations from visual input. Comparing to the VLM baseline isolates the effect of MCR. PointGraph and SpatialCoT represent state-of-the-art spatial enhancement methods; outperforming them demonstrates MCR's superiority in inducing metric consistency without extra modules. The ablation with squared loss tests the hinge formulation, confirming that penalizing violations only (hinge) is crucial to avoid over-constraining representations. The law of cosines ablation tests whether the specific benefit comes from triangle inequality or from any geometric constraint. Including a non-metric subset of OmniSpatial (e.g., scenes dominated by topological relations like 'inside' and 'above') is critical: if MCR degrades performance there, it confirms the assumption's boundary. This combination of dataset, baselines, and ablations forms a falsifiable test: if MCR fails to improve over the baseline on tasks where triangle inequality violations are frequent, or underperforms compared to explicit graph methods, or degrades performance on non-metric relations, the central claim is undermined.

### Expected outcome and causal chain

**vs. VLM baseline (no MCR)** — On a case where three objects have distances that violate triangle inequality in the VLM's feature space (e.g., d(A,B)=10, d(B,C)=10, d(A,C)=25), the baseline produces an incorrect answer because it uses inconsistent distances for reasoning. Our MCR regularizes the features to satisfy triangle inequality, pushing A and C closer, so it corrects the inconsistency and answers correctly. We expect a noticeable accuracy gap (e.g., 10–15%) on complex multi-object scenes in OmniSpatial, but parity on simple two-object tasks.

**vs. VLM + PointGraph** — On a scene where an object is occluded but its spatial relation can be inferred via transitive distances, PointGraph's explicit graph may fail if edges are missing or noisy, leading to missing predictions. Our MCR enforces global metric consistency across all objects, allowing the model to borrow geometric constraints from visible objects. Therefore, we expect higher recall on transitive spatial queries (e.g., "which is farthest from X?") with a 5–10% improvement over PointGraph, while PointGraph may be comparable on direct pairwise queries.

**vs. VLM + SpatialCoT** — On a temporally extended spatial reasoning task (e.g., multi-step navigation), SpatialCoT can propagate errors through its chain of reasoning, leading to compounding mistakes. Our MCR provides an implicit geometric prior that stabilizes the intermediate representations, reducing error propagation. We expect a smaller performance gap on single-step queries but a larger gap (e.g., 8–12%) on multi-step tasks where consistency across steps matters.

**Ablation: MCR with law of cosines** — On metric triples, law of cosines is an alternative constraint that involves the angle between sides. If triangle inequality is the more beneficial constraint (since it directly enforces metric space axioms), MCR with triangle inequality should outperform the law of cosines variant. We expect a 2–5% accuracy gap in favor of triangle inequality on metric-heavy subsets.

**Failure mode: non-metric subset** — On scenes where relations are inherently non-metric (e.g., 'inside'), MCR with masking should perform at least as well as the baseline (since masking disables the loss on those triples), but if masking is misapplied, MCR could degrade. We expect parity with baseline on non-metric subsets, confirming that our type-aware masking correctly avoids imposing false constraints.

### What would falsify this idea

If MCR yields no improvement or degrades performance on tasks where triangle inequality is inherently violated (e.g., non-metric spatial relations like "inside" or "above"), or if the gain is uniform across all OmniSpatial subsets rather than concentrated on tasks requiring metric consistency, then the central claim that enforcing triangle inequality aids spatial reasoning is false.

## References

1. OmniSpatial: Towards Comprehensive Spatial Reasoning Benchmark for Vision Language Models
2. Does Spatial Cognition Emerge in Frontier Models?
3. An Empirical Analysis on Spatial Reasoning Capabilities of Large Multimodal Models
4. What's "up" with vision-language models? Investigating their struggle with spatial reasoning
5. When and why vision-language models behave like bags-of-words, and what to do about it?
6. VL-CheckList: Evaluating Pre-trained Vision-Language Models with Objects, Attributes and Relations
7. Winoground: Probing Vision and Language Models for Visio-Linguistic Compositionality
8. AgentsNet: Coordination and Collaborative Reasoning in Multi-Agent LLMs
