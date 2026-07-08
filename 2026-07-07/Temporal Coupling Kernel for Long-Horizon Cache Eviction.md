# Temporal Coupling Kernel for Long-Horizon Cache Eviction

## Motivation

Existing eviction methods like KVpop (Key-Value Cache Compression with Predictive Online Pruning) score each cache key independently using only a fixed window of near-future attention, ignoring structural regularities across keys. This fails to predict far-future importance for keys with limited history because temporal attention dynamics are strongly coupled across similar keys—a property that prior work does not exploit. The core limitation is the lack of a mechanism to extrapolate beyond the observed horizon by leveraging inter-key similarity.

## Key Insight

The future attention pattern of any key is determined by its embedding through a learned, shared mapping that is consistent across keys with similar embeddings, enabling prediction of long-horizon importance without explicit future tokens.

## Method

### (A) What it is
**Temporal Coupling Kernel (TCK)** is a learned function $K(\mathbf{e}, d)$ that maps a key embedding $\mathbf{e} \in \mathbb{R}^p$ and a future time offset $d \in \mathbb{N}$ to a scalar importance score, trained to approximate observed future attention on a small window and then used to predict far-future importance for any key.

### (B) How it works
```python
# Training (offline, before inference)
# Input: set of N keys with observed future attention scores A_i(d) for d=1..W
#   where W is a small window (e.g., W=16).
# Embedding function f: key -> vector (e.g., mean of query-key dot products over recent steps, using a 2-layer MLP with hidden=256, GeLU activation to produce a 128-dimensional embedding)
# Kernel: K(e, d) = MLP([e; one_hot(d % T)])  # T is a periodicity hyperparameter, e.g., 64; MLP is 2 layers, hidden=256, GeLU activation, output scalar
# Loss: L = sum_{i=1..N} sum_{d=1..W} (K(e_i, d) - A_i(d))^2 + lambda * ||theta||^2
# Optimize theta (MLP weights) via SGD (learning rate 0.001, momentum 0.9, batch size 64).
# Load-bearing assumption: Keys with identical embeddings have identical future attention sequences. This assumption is explicitly verified after training (see calibration step).

# Calibration (after training, on held-out keys)
# Compute embedding similarity (cosine) and future attention correlation (Pearson) for held-out key pairs.
# If the median correlation across pairs with similarity > 0.9 is below 0.7, flag a warning.

# Inference (online, for cache scoring)
# For each key k in cache at current step t:
#   e_k = f(k)
#   importance_k = sum_{d=1}^{H} gamma^{d-1} * K(e_k, d)   # H is horizon (e.g., 512), gamma discount
# Evict keys with lowest importance under budget.

# Online updating (optional, to handle distribution shift)
# When new keys with full attention observations become available (e.g., after decoding a few more steps),
# perform one step of SGD on the same loss using those keys with learning rate 0.0001 (reduced to avoid catastrophic forgetting).
# Maintain a small buffer of recent training examples (size 2048) to replay.
```
Hyperparameters: $W=16$, $T=64$, $\lambda=0.01$, $\gamma=0.99$, $H=512$, embedding dimension 128, MLP hidden 256.

### (C) Why this design
We chose a parametric MLP kernel instead of a non-parametric Gaussian process because (a) GP inference scales cubically with the number of keys, unacceptable for large caches, while MLP inference is O(1) per key; (b) the MLP can be updated online via streaming SGD if new keys arrive, whereas a GP would require full retraining. We encode time offset $d$ using a modulo positional encoding (period $T$) to capture periodic repetition in attention patterns (e.g., recurring phrases), accepting the cost that non-periodic patterns may be modeled less accurately. We use a discounted sum over $H$ steps (rather than a fixed window) to give higher weight to near-future importance while still considering long-horizon contributions; this requires tuning the discount factor $\gamma$, and if the true attention does not decay exponentially, the approximation may be biased. We chose to train only on a short window ($W=16$) because collecting future attention beyond that is expensive; the resulting kernel extrapolates to $H\gg W$ via the continuous representation of $d$, assuming smoothness in time—a standard but strong assumption that fails when attention abruptly shifts due to topic changes. Additionally, we add a calibration step to verify the load-bearing assumption that embeddings are sufficient statistics for future attention.

### (D) Why it measures what we claim
The computational quantity $K(\mathbf{e}_k, d)$ measures the *expected future attention* for key $k$ at offset $d$ because it is trained to minimize squared error against ground-truth future attention on observed keys; the underlying assumption is that the embedding $\mathbf{e}_k$ is a sufficient statistic for the temporal attention profile—that is, keys with identical embeddings have identical future attention sequences. This equivalence fails when two keys share the same embedding but their attention diverges (e.g., due to different syntactic roles that are not captured by the embedding), in which case $K$ reflects an average over training keys with that embedding rather than the true value. The discounted sum $\sum_{d=1}^H \gamma^{d-1} K(\mathbf{e}_k, d)$ measures the *long-horizon cache importance* because it aggregates predictions across many future steps; the assumption is that the kernel's extrapolation beyond $W$ is accurate—if the true attention pattern has a discontinuity after $W$, the metric will be misled. The hyperparameter $\gamma$ encodes an assumed exponential decay of importance over time; if decay is slower or faster, the metric over- or under-weights late tokens. To guard against the load-bearing assumption, we include a calibration step that checks correlation between embedding similarity and future attention similarity; if correlation is low, the method may need a context-sensitive embedding.

## Contribution

(1) A temporal coupling kernel that jointly models inter-key similarity and temporal dynamics for cache eviction, enabling far-future importance prediction without requiring explicit future tokens. (2) A demonstration that a parametric kernel trained on a short window (W=16) can extrapolate to horizons orders of magnitude longer (H=512) by leveraging shared structure across keys. (3) A training procedure that reuses KVpop's future-attention target but operates on embeddings, making it architecture-agnostic.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | AIME mathematical reasoning, Code generation (HumanEval), Document QA (SQuAD) | Tests long-range dependencies across domains |
| Primary metric | Accuracy at 75% cache compression | Directly measures eviction quality |
| Baseline1 | Random eviction | Lower bound with no intelligence |
| Baseline2 | LRU | Tests recency bias |
| Baseline3 | Accumulated past attention | Simple learned baseline |
| Baseline4 | Linear regression on embedding (2-layer MLP with same embedding but linear output) | Isolates benefit of non-linear kernel |
| Ablation | TCK without periodic encoding | Tests temporal kernel contribution |

### Why this setup validates the claim
AIME problems require retaining key facts across many steps, making far-future importance critical. Code generation tasks have long-range dependencies between variable definitions and usage. Document QA often involves cross-referencing distant passages. Accuracy at 75% compression directly reflects whether the eviction policy preserves tokens needed later. Random and LRU test trivial baselines; accumulated attention tests if naive past-sum suffices; linear regression tests if a simple model on the same embedding can predict future attention, isolating the MLP's benefit. The ablation isolates the periodic encoding's role. Additionally, we compute the Spearman correlation between embedding cosine similarity and future attention profile similarity on a held-out set of keys, to directly verify the load-bearing assumption. If TCK improves accuracy over all baselines, especially on tasks with delayed references, it validates that short-window training generalizes to far-future prediction. The metric is sensitive to both correct retention and incorrect eviction, providing a clear signal for the predicted effect.

### Expected outcome and causal chain

**vs. Random eviction** — On a long AIME problem where a key definition is used only once early but critically invoked near the end, random eviction frequently discards it, causing a wrong answer. Our method predicts high future importance for that key via the kernel (trained on similar embedding profiles), so it retains it. We expect a >10% accuracy gap on problems with such delayed references.

**vs. LRU** — On a problem where a key lemma is accessed early, then unused for many steps, then needed again, LRU evicts it as "least recently used." Our method's temporal kernel, learning periodicity, recognizes the future re-access and keeps it. We expect a >5% gap on problems with long gaps between uses of the same token.

**vs. Accumulated past attention** — On a problem where attention abruptly shifts due to a topic change (e.g., from algebra to geometry), this baseline overweights past algebra tokens, evicting geometry tokens that will be needed soon. Our kernel, trained on short windows, can capture the shift via embedding changes, retaining geometry keys. We expect a >3% gap on problems with sharp topic transitions.

**vs. Linear regression on embedding** — On a problem where the relationship between embedding and future attention is non-linear (e.g., attention periodic with position), linear regression extrapolates poorly, whereas the MLP captures the non-linearity. We expect a >2% gap on tasks with periodic or complex attention patterns.

### What would falsify this idea
If TCK's accuracy gain over accumulated past attention and linear regression is small and uniform across all problem types (not concentrated on cases with delayed, shifted, or non-linear attention), then the claim that far-future prediction from short windows via a non-linear kernel is beneficial would be invalid. Additionally, if the embedding correlation analysis reveals that similar embeddings do not correspond to similar future attention profiles (median correlation <0.5), the load-bearing assumption is broken.

## References

1. KVpop -- Key-Value Cache Compression with Predictive Online Pruning
2. xLSTM 7B: A Recurrent LLM for Fast and Efficient Inference
3. Inference-Time Hyper-Scaling with KV Cache Compression
4. DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models
5. Eigen Attention: Attention in Low-Rank Space for KV Cache Compression
6. LoRC: Low-Rank Compression for LLMs KV Cache with a Progressive Compression Strategy
7. Bio-xLSTM: Generative modeling, representation and in-context learning of biological and chemical sequences
8. A Large Recurrent Action Model: xLSTM enables Fast Inference for Robotics Tasks
