# CPA-MoE: Contrastive Prototype Adaptive Mixture of Experts for Unsupervised Skill Discovery

## Motivation

Existing mixture-of-experts (MoE) methods, such as Skill-MoE, rely on frozen expert models and supervised skill labels for routing, which prevents generalization to unseen task distributions and requires labeled data. The root cause is that routing signals and expert specialization are decoupled: experts are pre-trained and fixed, while routers depend on supervised mapping from inputs to skills. This structural property blocks adaptation to evolving tasks and zero-shot transfer. We need a mechanism where routing and expert fine-tuning co-adapt from unlabeled data alone, enabling emergent skill discovery.

## Key Insight

The coupling of routing and expert adaptation through a shared, dynamically updated prototype space forces the emergence of specialized skills because the contrastive objective aligns inputs with prototypes while the experts must produce representations consistent with their assigned prototype, creating a self-consistent specialization loop.

## Method

(A) **What it is**: CPA-MoE is an unsupervised framework that jointly learns routing and expert specialization without skill labels. It maintains K prototype vectors (skill centroids) and K lightweight expert adapters attached to a shared base encoder. Inputs are routed to the expert whose prototype has highest cosine similarity with the input embedding. Prototypes and adapters are updated via a contrastive loss that maximizes agreement between the input's adapted embedding and its selected prototype while penalizing agreement with others.

(B) **How it works** (pseudocode per batch):
```python
# Hyperparameters: temperature τ, momentum m, number of experts K, adapter learning rate η
for x in batch:
    h = base_encoder(x)                  # shared representation
    # Router: nearest prototype
    sim = cosine_similarity(h, prototypes)  # prototypes: shape [K, d]
    k = argmax(sim)                      # selected expert index
    p_sel = prototypes[k]
    # Expert forward: residual adapter
    z = e_k(h) + h                       # e_k is a small MLP; output same dim
    # Contrastive loss (InfoNCE)
    pos_logit = dot(z, p_sel) / τ
    neg_logits = [dot(z, prototypes[j]) / τ for j != k]
    loss = -log(exp(pos_logit) / (exp(pos_logit) + sum(exp(neg_logits))))
    # Momentum update for selected prototype
    prototypes[k] = normalize(m * prototypes[k] + (1 - m) * normalize(z))
    # Gradient update on e_k (and optionally base_encoder with lower lr)
    e_k.parameters() -= η * grad(loss, e_k.parameters())
```

(C) **Why this design**: We chose momentum-updated prototypes over k-means because momentum provides temporal consistency, preventing prototype oscillation at the cost of slight adaptation lag. We chose residual adapters (e_k(h)+h) rather than full expert models because they are parameter-efficient and allow the base model to retain general knowledge while forcing adapters to capture residual specialization; this limits capacity but prevents catastrophic forgetting. We chose InfoNCE contrastive loss over triplet loss because it normalizes across all prototypes, automatically handling hard negatives, though it requires careful temperature tuning. We used cosine similarity for routing rather than dot product to eliminate magnitude differences as a confounding factor, accepting that magnitude information (which might indicate uncertainty) is discarded. These design choices collectively enable unsupervised co-adaptation. Unlike static retrieval methods (e.g., Toolformer), our prototypes are not fixed but co-evolve with routing decisions, making the method a dynamic self-organization rather than repackaged retrieval.

(D) **Why it measures what we claim**: The prototype assignment probability (via cosine similarity) measures skill similarity because it assumes that embedding direction in the base encoder's feature space is sufficient to capture functional specialization; this assumption fails when two skills are distinguished only by magnitude or when the base encoder is not well-calibrated, in which case assignment may reflect spurious correlations. The contrastive loss between adapted embedding z and selected prototype p_sel measures expert specialization because it forces the adapter to produce embeddings consistent with the prototype while repelling others, under the assumption that the prototype represents a coherent skill cluster; this assumption fails if the prototype is noisy or the adapter memorizes instead of generalizes. The momentum update of prototypes measures the evolving skill distribution because it aggregates recent embeddings assigned to that prototype, assuming the running average is a sufficient centroid statistic; this fails when the skill distribution is non-convex or multimodal, causing the centroid to lie in a low-density region.

## Contribution

(1) Introduces CPA-MoE, a novel framework that jointly learns routing and expert adaptation in an unsupervised manner via contrastive prototype alignment, eliminating the need for skill labels or frozen experts. (2) Demonstrates that coupling prototype-based routing with contrastive learning enables emergent expert specialization without any supervised signal, providing a principled approach to self-organizing MoE. (3) Provides a design principle: momentum-updated prototypes act as stable attractors that guide both routing and adapter learning, enabling online adaptation to new task distributions.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | MMLU-Pro | Multidisciplinary benchmark with diverse reasoning skills |
| Primary metric | Accuracy | Direct measure of task correctness |
| Baseline | Single LLM | Tests necessity of dynamic routing |
| Baseline | Static ensemble | Tests benefit over simple averaging |
| Baseline | Skill-Based MoE | Tests against prior routing framework |
| Ablation | CPA-MoE (no contrastive) | Isolates effect of contrastive co-adaptation |

### Why this setup validates the claim

This setup forms a falsifiable test because it isolates the core claim—that unsupervised co-adaptation of routing and expert adapters improves skill specialization. The single LLM baseline establishes whether any routing is beneficial; if our method outperforms it, dynamic assignment adds value. The static ensemble baseline tests whether the benefit comes from simply having multiple experts rather than from adaptive routing—if our method beats averaging, routing is indeed superior. The Skill-Based MoE baseline represents a prior approach with different assumptions (often supervised or fixed prototypes); outperforming it demonstrates that our unsupervised co-adaptation is more effective or efficient. The ablation removes the contrastive loss, leaving prototype updates driven purely by assignment; if it underperforms the full method, the contrastive signal is essential for specialization. Accuracy on MMLU-Pro directly reflects reasoning quality across diverse domains, so any advantage must come from better skill-expert alignment. A uniform gain across all topics would not support our claim—we expect larger gains on questions requiring specialized knowledge, which this benchmark's diversity can reveal.

### Expected outcome and causal chain

**vs. Single LLM** — On a question requiring specialized medical knowledge (e.g., a rare disease diagnosis), a single LLM may rely on general patterns and produce a plausible but incorrect answer because it lacks a dedicated expert. Our method routes this input to a medical-specialized adapter via prototype similarity, which has been trained to produce embeddings consistent with that skill, leading to a more accurate prediction. We expect a noticeable gap on domain-specific subsets (e.g., MedMCQA within MMLU-Pro) but parity on common-sense topics.

**vs. Static ensemble** — On a question that combines multiple reasoning types (e.g., a physics problem requiring both calculation and conceptual understanding), a static ensemble averages outputs from all experts, diluting the contribution of the most relevant one. Our method routes to only the most similar prototype, then uses the adapter to produce a focused embedding. Therefore, we expect our method to outperform on questions that are clearly dominated by one skill, but may underperform on highly blended questions where averaging helps. The overall gain should be concentrated on single-skill subsets.

**vs. Skill-Based MoE** — On a question where prior method's skill labels are noisy or incomplete (e.g., an interdisciplinary problem not fitting predefined categories), Skill-Based MoE may misroute due to rigid classification. Our method learns prototypes dynamically from data, allowing emergent specialization for novel skill combinations. Expect larger gains on ambiguous or cross-domain questions, while performance on clearly labeled skills may be similar.

### What would falsify this idea
If our method's performance improvement over the single LLM baseline is uniform across all MMLU-Pro subsets rather than concentrated on those requiring specialized knowledge (e.g., the gain on general trivia is as large as on professional medicine), then the claimed specialization is not occurring and the routing is not learning meaningful skill clusters.

## References

1. Skill-Based Mixture-of-Experts: Adaptive Routing for Heterogeneous Reasoning via Inferred Skills
2. Mixture-of-Agents Enhances Large Language Model Capabilities
3. Efficient Dynamic Ensembling for Multiple LLM Experts
4. Exploring Collaboration Mechanisms for LLM Agents: A Social Psychology View
5. Fusing Models with Complementary Expertise
6. ChatEval: Towards Better LLM-based Evaluators through Multi-Agent Debate
7. Large Language Models Are Reasoning Teachers
8. Teaching Small Language Models to Reason
