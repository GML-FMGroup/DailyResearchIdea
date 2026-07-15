# Meta-Adaptive Knowledge Boundary Detection for Agentic Visual Generation

## Motivation

Existing knowledge boundary detection in agentic visual generation (e.g., 'Search Beyond What Can Be Taught') relies on benchmark-specific thresholds that fail to generalize to new domains because the threshold is optimized for a static distribution of prompts and generators, ignoring domain-dependent variation in internal representations. This structural limitation causes frontier generators to score only 21-28/100 on diverse benchmarks like SearchGen-Bench, and naive retrieval exacerbates noise. A meta-learning approach that adapts the boundary using domain-invariant features is needed to overcome this reliance on static benchmarks.

## Key Insight

The generator's internal consistency across multiple stochastic outputs provides a domain-invariant measure of whether a prompt lies inside its knowledge boundary, enabling meta-learning of a retrieval gating function that generalizes across domains.

## Method

**(A) What it is:** Meta-KBA is a lightweight gating network that takes the generator's latent representations from multiple stochastic forward passes and outputs a binary retrieval decision. It is meta-trained across a diverse set of domains to learn a domain-invariant mapping from consistency to optimal retrieval.

**(B) How it works (pseudocode):**
```
Input: generator G, prompt p, retrieval corpus C
Hyperparameters: K=5, temperature τ=1.0, MLP hidden=64, meta-learning rate=1e-4, inner steps=5

1. Sample K outputs from G(p) with temperature τ: {o_1, ..., o_K}
2. Compute consistency metric C_agg = 1 - avg_pairwise_clip_similarity(o_i, o_j)  (using CLIP embeddings)
3. Compute domain embedding d = mean of CLIP image embeddings from 5 exemplar prompts from the same domain (if available), else zero-vector.
4. Gating network f: MLP with 2 layers (hidden=64, ReLU) and sigmoid output, input=[C_agg, d]
5. Meta-training (MAML): for each domain in meta-training set:
   - Sample support set S (10 prompts) and query set Q (10 prompts)
   - For each prompt, compute C_agg and ground-truth retrieval decision y (from oracle: does retrieval from C improve generation?)
   - Inner loop: train f on S for 5 steps with cross-entropy loss, inner lr=0.01
   - Outer loop: compute loss on Q, update meta-parameters via gradient descent with outer lr=1e-4
6. At test time: given C_agg and d, if f([C_agg,d]) > 0.5, retrieve top-1 from C using CLIP similarity and condition generation; else generate unconditionally.
```

**(C) Why this design:** We chose consistency-based uncertainty over log-probability (anti-pattern 3) because generator log-probs are poorly calibrated for task correctness, whereas output consistency directly measures the model's confidence in its own outputs, which is a known proxy for knowledge boundary (as shown in prior work on ensemble uncertainty). We use a meta-learning approach (MAML) rather than a fixed threshold (as in SearchGen) to adapt the gating function across domains with few examples, accepting the additional computational cost of meta-training but gaining generalization. We adopt a simple MLP gating network instead of a more complex transformer to keep the method lightweight and avoid overfitting to the small meta-training dataset. The trade-off is that the MLP may not capture complex nonlinear interactions between consistency and domain features, but the low-dimensional input (2 scalars) makes it feasible. We avoid a controller (anti-pattern 4) by collapsing the decision into a single module that jointly processes consistency and domain signal, rather than having separate modules.

**(D) Why it measures what we claim:** The consistency metric C_agg measures the generator's output uncertainty because high consistency (low C_agg) indicates that the generator produces similar outputs across samples, implying the prompt is within its knowledge boundary (well-learned concept); this equivalence relies on the assumption that stochasticity in generation is due to epistemic uncertainty about the correct output, and that when the generator is confident, repeated samples converge. This assumption fails when the generator is confidently wrong (e.g., systematic bias) where consistency is high but output is incorrect; in that case, C_agg reflects calibration error rather than knowledge boundary. The domain embedding d operationalizes the concept of domain-specific shift because it captures visual features of the domain from exemplars; this relies on the assumption that CLIP embeddings encode domain-relevant variation, which may fail for domains with high intra-class variance (e.g., generic 'animals' vs. specific species). The meta-learning procedure operationalizes the notion of domain-invariant boundary identification because it trains the gating network to perform well across domains by updating its parameters via gradient steps on domain-specific losses; this relies on the assumption that the mapping from consistency to optimal retrieval is similar across domains modulo a linear transformation, which fails if the relationship is fundamentally different (e.g., some domains require retrieval even with high consistency due to factual correctness constraints).

## Contribution

(1) A meta-learning framework (Meta-KBA) that learns a domain-invariant gating function for retrieval decisions in agentic visual generation, using output consistency as a proxy for knowledge boundary. (2) An empirical finding that meta-training across diverse domains with few-shot adaptation yields retrieval accuracy comparable to oracle thresholds, while generalizing to unseen domains. (3) A set of domain-invariant features (consistency + domain embedding) that are minimal and computationally efficient.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | SearchGen-Bench | Diverse domains with known optimal decisions. |
| Primary metric | Mean generation score | Measures alignment with optimal retrieval. |
| Baseline 1 | No retrieval (vanilla) | Tests when retrieval is necessary. |
| Baseline 2 | Always retrieve (naive RAG) | Tests harm of unnecessary retrieval. |
| Baseline 3 | Fixed threshold (consistency) | Tests meta-learning advantage. |
| Ablation of ours | Ours w/o domain embedding | Tests domain adaptation value. |

### Why this setup validates the claim
This combination forms a falsifiable test because the central claim is that meta-KBA’s gating network learns a domain-invariant mapping from consistency to optimal retrieval. Using a benchmark with diverse domains and known optimal decisions (oracle) allows direct measurement of decision accuracy. The baselines isolate each component: no retrieval shows the baseline gap, always retrieve shows the cost of blind retrieval, and fixed threshold shows the limit of a static rule. The ablation removes domain embedding to test whether the domain signal is essential for adaptation. The primary metric—mean generation score conditioned on correct retrieval decisions—directly reflects the quality of the gating policy. If our method outperforms all baselines, especially on domains where the optimal decision varies, the claim is supported. Conversely, if the fixed threshold matches our method, meta-learning adds no benefit.

### Expected outcome and causal chain

**vs. No retrieval** — On a case where the generator lacks knowledge (e.g., a recent event), the baseline produces an incorrect or generic image because it has no way to incorporate external information. Our method detects low consistency (high C_agg) and triggers retrieval, augmenting generation with relevant context. We expect a noticeable gap on out-of-distribution prompts but parity on well-known concepts.

**vs. Always retrieve** — On a case where the generator is already competent (e.g., common object generation), naive retrieval may introduce irrelevant or conflicting visual features, degrading output. Our method detects high consistency and skips retrieval, preserving the generator’s natural output. We expect a comparable performance on hard prompts but a clear advantage on easy prompts where unnecessary retrieval hurts.

**vs. Fixed threshold** — On a case where the optimal threshold varies across domains (e.g., realistic vs. abstract art), a static threshold misclassifies many prompts. Our meta-trained gating adapts via domain embedding, correctly deciding retrieval per domain. We expect our method to show higher overall accuracy, especially on domains where the consistency-to-optimal mapping differs from the average.

### What would falsify this idea
If our method’s gain over the fixed-threshold baseline is uniform across all domains rather than concentrated on domains where the consistency distribution shifts, then the meta-learning is not adapting to domain-specific patterns, and the central claim of learning a domain-invariant mapping fails.

## References

1. Search Beyond What Can Be Taught: Evolving the Knowledge Boundary in Agentic Visual Generation
2. FLUX.1 Kontext: Flow Matching for In-Context Image Generation and Editing in Latent Space
3. In-Context LoRA for Diffusion Transformers
4. FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision
5. OmniGen: Unified Image Generation
6. FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning
7. QuIP: 2-Bit Quantization of Large Language Models With Guarantees
8. Unified-IO 2: Scaling Autoregressive Multimodal Models with Vision, Language, Audio, and Action
