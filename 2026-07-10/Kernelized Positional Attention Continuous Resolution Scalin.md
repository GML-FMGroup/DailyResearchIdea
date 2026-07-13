# Kernelized Positional Attention: Continuous Resolution Scaling for Long-Context Transformers

## Motivation

Existing methods like SELF and LaMPE partition context into discrete resolution zones, which introduces hard boundaries and suboptimal granularity, forcing tokens at zone edges to receive the same resolution as distant tokens. Jet-Long's dual-window merge still retains discretization without a smooth transition between local and long-range attention. This structural limitation prevents the model from adapting granularity to the actual importance of each relative position. We need a method that allows attention weights to vary continuously with position, eliminating zone boundaries and enabling the model to learn position-dependent resolution patterns.

## Key Insight

Because the kernel is a continuous function of relative position, it can smoothly interpolate between fine and coarse attention without the boundary artifacts inherent in discrete zone methods.

## Method

(A) **What it is**: Kernelized Positional Attention (KPA) replaces discrete resolution zones with a learned kernel that scales attention logits via a continuous, head-specific function of normalized relative distance and absolute positions. Its input is a concatenated feature vector [Δ, p_i, p_j] where Δ = |i-j|/L is normalized distance, and p_i, p_j are learnable absolute position embeddings of dimension d=16 for positions i and j. Output is a scalar weight w(Δ, p_i, p_j) that adjusts the attention score before softmax.

(B) **How it works**: For each attention head, a small MLP with two hidden layers (32 units each, ReLU) maps the input vector to w:
    Δ = |i - j| / L   (L = sequence length)
    p_i, p_j ∈ ℝ^16 (learned per-head, shared across layers)
    x = [Δ; p_i; p_j]   (concatenated, length 33)
    h1 = ReLU(W1 x + b1)
    h2 = ReLU(W2 h1 + b2)
    w(Δ, p_i, p_j) = σ(W3 h2 + b3)   (sigmoid to [0,1])
Then the attention logit becomes: score = (q_i · k_j) + log(w(Δ, p_i, p_j)). Softmax is applied as usual. The MLP parameters are shared across positions but differ per head. Training is end-to-end with cross-entropy loss. Total additional parameters per head: (33*32 + 32*32 + 32*1) ≈ 2,144, for 12 heads ≈ 25,728 parameters—negligible compared to a 7B model.

(C) **Why this design**: We chose an MLP over fixed basis functions (e.g., radial basis functions) because it can adapt to arbitrary patterns, trading off interpretability for flexibility; we apply the kernel as an additive log-bias rather than a multiplicative weight to preserve the softmax normalization property (multiplication would break the log-sum-exp structure); we use one MLP per head rather than per layer because heads specialize in different positional behaviors, and sharing across heads would reduce expressiveness. This design accepts increased parameter count (≈2K per head) for better adaptation, but still negligible compared to total model size. We use sigmoid output to bound w(Δ) in (0,1), preventing unbounded amplification of attention weights. We include absolute position embeddings to account for non-stationary positional patterns, as empirical analysis shows attention varies with absolute position (What Position Do Language Models Even Need?, 2024); the trade-off is a slightly larger input dimension but still negligible parameters.

(D) **Why it measures what we claim**: The kernel weight w(Δ, p_i, p_j) measures the desired resolution at distance Δ and absolute positions because it directly modulates the attention logit, and the softmax converts it into a probability that controls how much a token is attended to based on distance and position. This relies on the assumption that the optimal resolution pattern is a stationary function of normalized distance and absolute positions; this assumption fails when content similarity dominates attention, in which case w might reflect training distribution biases rather than true resolution needs. The normalized distance Δ measures coarse relative position, assuming that the sequence length L is the only relevant scale; this assumption fails for tasks where semantic boundaries (e.g., sentence ends) matter more than distance, in which case Δ ignores local structure. The log-additive form measures the log-odds of attention, assuming that the base query-key dot product already captures content similarity; this assumption fails when content and position are strongly coupled, making the additive decomposition inaccurate. Assumption A: The kernel weight directly and independently controls attention allocation proportionally to its value. Failure mode F: If content similarity overwhelmingly determines attention, the kernel weight may not reflect resolution preferences but rather noise or spurious correlations.

## Contribution

(1) Introduction of kernelized positional attention, a continuous and learnable alternative to discrete resolution zones for long-context Transformers. (2) Demonstration that a simple per-head MLP can learn position-dependent attention granularity without hard boundaries, enabling smooth transition from fine local to coarse global attention. (3) A training scheme that integrates with FlashAttention by pre-computing the kernel weights as a matrix, requiring minimal overhead during forward pass.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset 1 | LongBench (long-context QA) | Standard long-context benchmark with diverse tasks |
| Dataset 2 | Synthetic: Every-k-th token retrieval (sequence length 16K, need to attend to tokens at positions that are multiples of k, with k randomly varied between 2 and 8) | Directly tests whether KPA recovers known optimal resolution pattern; ground truth is known |
| Primary metric | Accuracy on long-context QA; retrieval accuracy on synthetic task | QA measures comprehension; synthetic task measures resolution adaptation |
| Baseline 1 | Standard RoPE | No context extension baseline |
| Baseline 2 | Position Interpolation (PI) | Popular interpolation baseline |
| Baseline 3 | NTK-aware scaling | Another common scaling method |
| Baseline 4 | Learned scalar bias per head (a single learnable scalar per head added to all attention logits, no MLP) | Isolates benefit of continuous function over a simple learnable weight |
| Baseline 5 | Fixed learned kernel (learned per-head relative bias without input-dependent scaling; a learnable vector of length L of relative biases per head, added to logits) | Tests whether input-dependent scaling is necessary |
| Ablation 1 | KPA with shared MLP across all heads | Tests per-head specialization |
| Ablation 2 | KPA without absolute position embeddings (only Δ as input) | Tests contribution of absolute position dependence |

### Why this setup validates the claim
This combination directly tests the core claim that KPA's learned, head-specific positional kernel with absolute position awareness improves long-context reasoning over fixed schemes. Standard RoPE isolates the benefit of any extension; PI and NTK test fixed interpolation/scaling approaches that lack the flexibility of a learned kernel. The learned scalar bias baseline controls for the mere presence of a learnable parameter, showing whether the continuous function form is beneficial. The fixed learned kernel baseline controls for input dependence, testing whether the kernel's output varying with position is necessary. The synthetic task with known ground-truth resolution pattern provides a direct causal test: if KPA recovers the pattern, it confirms that the kernel adapts resolution as claimed. The ablation with shared MLP tests whether per-head specialization matters, and the ablation without absolute positions tests the contribution of the adversarial repair. Accuracy on long-context QA is the right metric because it requires aggregating information across distant positions—the exact scenario where adaptive resolution should matter. If KPA outperforms on long sequences specifically, the causal role of the learned kernel is confirmed. We will release the learned kernel weights for analysis, enabling qualitative validation of the learned resolution patterns.

### Expected outcome and causal chain

**vs. Standard RoPE** — On a case where the query must attend to a token 8K positions away in a 16K sequence, Standard RoPE collapses attention to nearby positions because its sinusoidal encoding decays or lacks the flexibility to emphasize far tokens. Our method instead learns to upweight far distances via the MLP, allowing the model to retrieve the needed token. We expect KPA to show a noticeable gap on tasks requiring reasoning across >8K tokens, while performing comparably on shorter tasks.

**vs. Position Interpolation (PI)** — On a sequence where optimal resolution varies non-uniformly (e.g., dense local context but sparse long-range cues), PI applies a uniform rescaling that either over-squashes local patterns or under-emphasizes distant ones. Our method learns a continuous weight per head, allowing it to tune a sharp local peak and a gentle long-range tail. We expect KPA to outperform PI on tasks with irregular dependency distributions, like multi-document summarization where only specific distant paragraphs are relevant.

**vs. NTK-aware scaling** — On a task where the query needs fine-grained positional discrimination at certain distances (e.g., distinguishing two far entities with similar content), NTK scaling’s high-frequency amplification may introduce aliasing or imbalance. Our method avoids this by learning a bounded sigmoid weight, preserving the relative ordering of scores. We expect KPA to achieve higher accuracy on long-context QA that requires precise distance discrimination, while NTK may suffer on such subsets.

**vs. Learned scalar bias** — The learned scalar bias baseline adds a constant offset to all attention logits within a head, which cannot adapt to distance or position. On tasks where optimal attention weight varies with distance (e.g., need to downweight nearby tokens for retrieval), KPA's distance-aware kernel should outperform. We expect a noticeable gap on the synthetic task where the ground truth pattern is distance-dependent.

**vs. Fixed learned kernel** — The fixed learned kernel learns a per-head bias vector of length L, which is position-sensitive but not input-dependent (same bias for all queries). On the synthetic task where the query identity matters (e.g., different queries need different resolution patterns), KPA's input-dependent kernel should outperform. We expect KPA to show higher accuracy on the synthetic task.

### What would falsify this idea
If KPA’s accuracy gain is uniform across all sequence lengths and tasks (rather than concentrated on long-range or nuanced-position tasks), the learned kernel is likely exploiting a trivial advantage (e.g., better initialization) rather than modeling resolution adaptively. Additionally, if on the synthetic task KPA fails to recover the known optimal pattern (e.g., does not learn to attend to every k-th token), the claim that the kernel adapts resolution is falsified.

## References

1. Jet-Long: Efficient Long-Context Extension with Dynamic Bifocal RoPE
2. SELF: Self-Extend the Context Length With Logistic Growth Function
3. LaMPE: Length-aware Multi-grained Positional Encoding for Adaptive Long-context Scaling Without Training
4. LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens
5. LLM Maybe LongLM: Self-Extend LLM Context Window Without Tuning
6. Scaling Laws of RoPE-based Extrapolation
7. Challenging BIG-Bench Tasks and Whether Chain-of-Thought Can Solve Them
8. GPT-NeoX-20B: An Open-Source Autoregressive Language Model
