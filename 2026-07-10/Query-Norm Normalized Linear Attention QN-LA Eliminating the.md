# Query-Norm Normalized Linear Attention (QN-LA): Eliminating the Hybrid Necessity via Adaptive Focusing

## Motivation

Existing pure linear attention mechanisms (e.g., DeltaNet, Gated DeltaNet) fail to produce sharp attention distributions, leading to a persistent performance gap compared to hybrid models that interleave softmax attention. This gap is attributed to the inability of linear attentions to dynamically focus on relevant tokens, as shown by Zoology's associative recall analysis. The shared assumption across Kimi Linear and Gated Delta Networks is that hybrid designs are necessary to compensate for this lack of focus. We challenge this assumption by introducing a query-dependent normalization that adaptively sharpens attention without softmax.

## Key Insight

The effective temperature of attention can be controlled by a learned monotonic function of the query's L2 norm, because the norm correlates with the model's confidence in the query's discriminative power, enabling adaptive focusing while preserving linearity.

## Method

**Query-Norm Normalized Linear Attention (QN-LA)**

(A) **What it is**: QN-LA is a linear-complexity attention mechanism that replaces softmax normalization with a query-dependent gain function \(g(\mathbf{q})\) learned from the Euclidean norm of the query. It takes as input queries \(\mathbf{Q}\), keys \(\mathbf{K}\), values \(\mathbf{V}\) (all matrices of shape \(L \times d\)) and outputs attended values \(\mathbf{O} \in \mathbb{R}^{L \times d}\). The key property is that the attention distribution is implicitly controlled by the query norm without explicit exponential normalization. The core assumption is that the L2 norm of a query is a monotonic proxy for the model's confidence in its discriminative power, enabling adaptive focusing via gain.

(B) **How it works** (pseudocode-style steps):  
\[
\begin{aligned}
&\text{1. For each query } \mathbf{q}_i \text{ compute norm } n_i = \|\mathbf{q}_i\|_2. \\
&\text{2. Compute gain } g_i = 1 + \tanh\left(\text{MLP}_{\theta}(n_i)\right) \in (0,2). \quad \text{MLP: 2-layer with hidden dim } d_{\text{hidden}} = d/4. \\
&\text{3. Apply feature map: } \phi(\mathbf{x}) = \text{ReLU}(\mathbf{W}_\phi \mathbf{x}) \in \mathbb{R}^{d_k}, \quad \mathbf{W}_\phi \in \mathbb{R}^{d_k \times d}. \quad (\text{Nonneg. feature map}) \\
&\text{4. Normalized query feature: } \tilde{\mathbf{q}}_i = g_i \cdot \phi(\mathbf{q}_i). \\
&\text{5. Key feature: } \mathbf{k}_j = \phi(\mathbf{k}_j). \\
&\text{6. Compute linear attention: } \mathbf{O} = \mathbf{D}^{-1} (\tilde{\mathbf{Q}} \mathbf{K}^\top) \mathbf{V}, \text{ where } \mathbf{D} = \tilde{\mathbf{Q}} \mathbf{K}^\top \mathbf{1}_L \text{ (row-wise sum for normalization).}
\end{aligned}
\]

(C) **Why this design**:  
We chose a tanh-gated MLP over a simple linear scaling because a bounded non-linearity allows the gain to saturate, preventing extreme amplification that could destabilize training (trade-off: expressivity vs stability). We used ReLU as the base feature map rather than exponential linear units because ReLU preserves linearity for positive activations, keeping the overall attention computation linear-time (trade-off: possible loss of negative information vs computational cheapness). We applied gain only to the query features and not to keys to avoid double-counting normalization; applying gain to queries alone is sufficient to control the effective temperature because the dot product scale is proportionate to query norm (trade-off: asymmetry vs simpler formulation). Compared to prior hybrid approaches (e.g., Kimi Linear's layerwise interleaving of linear and softmax layers), our method avoids any softmax component entirely, reducing computational overhead and enabling fully linear inference. A domain expert would not describe our method as a variant of Gated Delta Networks because Delta Networks use a state-space update rule with gating and delta, whereas our method is a direct linear attention with a learned query-adaptive scaling, retaining the original transformer structure.

(D) **Why it measures what we claim**:  
The gain \(g_i\) measures the effective focusing strength (temperature control) for query \(i\) because it multiplies the query feature before the dot product, thereby controlling the variance of the attention distribution; this assumes that the L2 norm of a query is a monotonic proxy for the model's confidence in the query's discriminative power. This assumption fails when input noise or token frequency causes the norm to be uncorrelated with confidence, leading to erroneous amplification. The ReLU feature map \(\phi\) ensures nonnegative attention scores, which together with the row normalization (step 6) yields a valid probability distribution; this assumption fails when negative dot products would be informative (e.g., in some tasks requiring negative correlations), causing information loss. The MLP parameterizes the monotonic relationship between norm and gain; this assumption fails if the optimal gain is non-monotonic in norm, requiring more complex functions. The computational quantities \(\tilde{\mathbf{Q}}\) and \(\mathbf{K}\) operationalize the concept of adaptive focusing because they directly determine the sharpness of attention without softmax; this assumption fails when the linear normalization step (sum) produces peakedness that does not correspond to task relevance (e.g., all queries have similar norms and gains), leading to uniform attention.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|----------------------|
| Dataset (primary) | The Pile | Large-scale diverse text corpus |
| Primary metric | Validation perplexity | Direct measure of language modeling |
| Baseline 1 | Softmax attention | Standard attention with exponential normalization |
| Baseline 2 | Standard linear attention | ReLU feature map, no gain |
| Baseline 3 | Learned temperature softmax | Softmax with per-layer learnable scalar temperature |
| Ablation-of-ours | QN-LA without gain (g=1) | Isolates effect of adaptive gain |
| Downstream task | NarrativeQA (long-document QA) | Tests practical impact on long-context reasoning |
| Downstream metric | F1 and Exact Match | Standard QA metrics |
| Overhead analysis | FLOPs per token, runtime (4096 tokens) | Measures efficiency against baselines |

### Why this setup validates the claim

The combination of The Pile and validation perplexity directly tests the core language modeling capability of our method. Softmax attention serves as the ultimate baseline to measure any performance degradation from linearization. The standard linear attention (with fixed gain) isolates the impact of the ReLU feature map alone, while our full method adds the learned gain. The ablation without gain disentangles the contribution of adaptive scaling from the base linear mechanism. If our method matches softmax perplexity while exceeding linear baseline, it confirms that query-norm gain compensates for softmax removal. Conversely, if gain fails to improve over fixed gain, the central claim is invalidated. Additionally, we evaluate on NarrativeQA to demonstrate that the adaptive focusing benefit transfers to long-document QA, where softmax attention's quadratic cost is prohibitive and linear attention's poor sharpness hurts performance; outperforming standard linear attention on F1/EM would show practical advantage. The overhead analysis verifies that our method retains linear complexity and is faster than hybrid models.

### Expected outcome and causal chain

**vs. Softmax attention** — On a sequence where some tokens require sharp focusing (e.g., rare word prediction), softmax uses exponential scaling to produce peaked attention, but standard linear attention with ReLU features cannot achieve such sharpness due to linear dot products. Our method instead applies a learned gain to the query feature before the dot product; for queries with large norm (indicating high confidence), the gain amplifies the attention distribution, mimicking softmax's temperature effect. We expect our method to achieve perplexity close to softmax (within 0.5-1.0) on such contexts, while standard linear attention lags significantly (1.5-2.0 higher perplexity).

**vs. Standard linear attention** — On a case where queries have varying uncertainty (e.g., predictable vs. surprising tokens), standard linear attention treats all queries equally, leading to uniform attention spread and poor uncertainty calibration. Our method adapts via gain: for confident queries (high norm), gain >1 sharpens attention; for uncertain queries (low norm), gain close to 1 or less broadens attention. This should yield better perplexity on rare tokens (by 0.3-0.5) while maintaining similar performance on common tokens. Empirically, we expect the gain values to correlate with predictive confidence, visible by plotting gain vs. next-token probability. To verify the assumption that query norm correlates with confidence, we will compute calibration plots on a held-out calibration set of 1024 sequences, binning queries by norm and plotting expected vs. observed accuracy; we expect a monotonic relationship (higher norm → higher accuracy).

**vs. Learned temperature softmax** — Learned temperature softmax also adapts sharpness but retains softmax, incurring quadratic cost. Our method should achieve comparable perplexity (within 0.2-0.5) while being linear in sequence length. On long sequences (e.g., 8k tokens), our method will be significantly faster (2-5x speedup) due to linear complexity.

**Downstream task (NarrativeQA)** — We expect our method to outperform standard linear attention by 1-2 F1 points on long-document QA, where context length is large and sharp attention is needed to locate answer evidence. Softmax attention is infeasible for very long documents, so our method provides a practical alternative.

**Overhead** — Our method adds negligible overhead (a small MLP per head) and maintains O(L) FLOPs. We expect runtime similar to standard linear attention on short sequences (e.g., 512 tokens) and increasingly faster than softmax on long sequences.

### What would falsify this idea

If the learned gain values are approximately uniform across all queries (e.g., all near 1.0) and the full method performs no better than the ablation, then the central claim that query-norm gain enables adaptive focusing is false. Additionally, if calibration plots show no correlation between query norm and accuracy, the assumption underlying the gain function is invalidated.

## References

1. Linear Attention Architectures: Mechanisms, Trade-offs, and Cross-Layer Routing
2. Kimi Linear: An Expressive, Efficient Attention Architecture
3. Gated Delta Networks: Improving Mamba2 with Delta Rule
4. Zoology: Measuring and Improving Recall in Efficient Language Models
5. Gated Linear Attention Transformers with Hardware-Efficient Training
