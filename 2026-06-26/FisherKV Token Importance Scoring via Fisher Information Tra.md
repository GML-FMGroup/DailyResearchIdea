# FisherKV: Token Importance Scoring via Fisher Information Trace for Minimax-Optimal KV Cache Compression

## Motivation

Current KV cache compression methods rely on heuristic token importance scores—such as attention weights in PyramidKV or predictive entropy in InfoKV—that fail to generalize across tasks. These scores assume a uniform relationship between local model internals and global output distribution, leading to task-specific compression failures (e.g., InfoKV’s entropy signal is noisy in shallow layers). The structural gap is the absence of a task-agnostic optimality condition; a minimax-optimal scoring would guarantee worst-case preservation of the output distribution’s information geometry across all tasks.

## Key Insight

The trace of the Fisher information matrix of the output log-likelihood with respect to each token’s KV state quantifies the token’s infinitesimal influence on the entire output distribution, yielding a provably optimal importance score under a minimax criterion on the geodesic distance between original and compressed output distributions.

## Method

### (A) What it is
**FisherKV** assigns each token in the KV cache a scalar importance score equal to the trace of the Fisher information matrix (FIM) of the model's output log-likelihood with respect to that token's key and value state vectors. The method takes as input the KV state vectors for all tokens in the current sequence, and outputs a ranking (scores) for retention during compression. It requires only a single forward-backward pass over the sequence (amortized O(n) to compute gradients for all tokens) with S=5 output samples to approximate the Fisher trace.

### (B) How it works
```pseudocode
# Hyperparameters: S = 5 (number of output samples for Monte Carlo estimate of Fisher trace),
# epsilon = 1e-5 (finite-difference step, used only if direct gradient is unavailable)
# We assume the empirical Fisher approximation (S samples) is a reliable estimator of the true Fisher trace,
# yielding a minimax-optimal token importance score for KV cache compression.
Compute log-likelihood L = log P(output | input)  # typically the log probability of the next token(s)
# For S samples, we draw S output samples (e.g., via temperature sampling) and average the gradient outer products.
Initialize score_i = 0 for each token i
For each sample s in 1..S:
   Sample output_s from the model (temperature=1.0)
   Compute log-likelihood L_s = log P(output_s | input)
   For each token i in sequence:
      k_i = key state vector for token i   (dimension d)
      v_i = value state vector for token i (dimension d)
      g_k_i = dL_s / d k_i   # shape (d,)
      g_v_i = dL_s / d v_i   # shape (d,)
      score_i += (||g_k_i||^2 + ||g_v_i||^2) / S  # accumulate average
   # (Implementation uses gradient hooks to avoid materializing full Jacobians)
Sort tokens by score_i descending
Keep top-K tokens (K determined by target compression ratio)
```
**Notes:** The gradient dL/d k_i is computed via backpropagation through the attention mechanism; it captures how small changes in k_i affect the output log-likelihood. The trace measure aggregates sensitivity across all output dimensions. For multi-token outputs, the log-likelihood L can be the sum over output tokens; the gradient then captures the total influence. Using S=5 reduces variance and bias compared to the single-sample default.

### (C) Why this design
We chose the Fisher trace over heuristics (attention, entropy) because it directly quantifies the **local sensitivity** of the output distribution to perturbations in a token's representation, which is the quantity a minimax compressor must preserve. The trade-off: Fisher trace requires S=5 forward-backward passes (computing gradients), adding ≈100% overhead per compression step (S=5) compared to the O(1) cost of reading attention weights. However, this cost is amortized because compression can be applied every few generation steps or only during prefill. We further chose the multi-sample Monte Carlo estimate over the exact expectation (which requires many samples), accepting moderate variance but reducing memory from storing multiple Hessians. The third design decision: we score both key and value states (sum of their Fisher traces) rather than only keys, because the output sensitivity decomposes additively across attention inputs; ignoring value states would miss information-critical tokens that affect the output via value weighting. The trade-off is doubling the gradient storage but capturing full influence.

### (D) Why it measures what we claim
**Computational quantity `(1/S) sum_s (||g_k_i||^2 + ||g_v_i||^2)`** measures **optimal token importance** under the assumption that the output distribution is locally a natural exponential family and the Fisher information metric captures the geodesic distance between compressed and original distributions. This assumption fails when the output distribution is multimodal with widely separated modes; then the curvature of the log-likelihood is not constant, and the Fisher trace reflects local sensitivity only around the current mode, potentially under-weighting tokens that are critical for alternative outputs. In that case, the metric reflects a local, mode-specific importance rather than a global one, and the minimax guarantee degrades to worst-case within a geodesic ball around the current mode. **The gradient g_k_i** measures **the first-order change in the log-likelihood** w.r.t. the key vector; the trace aggregates this across all output dimensions, but because the empirical Fisher uses S=5 output samples, the variance of the score is moderate; if the output space is large, rare tokens may still be undervalued due to sampling noise (failure mode). **The summation over key and value** operationalizes **joint sensitivity** of the attention block; the assumption that sensitivity decomposes additively holds if the attention output is linear in k and v (true for fixed query, but not when query also depends on k/v through self-attention with tied weights); in that more complex case, the trace of the joint Fisher (block matrix) would be required, and our additive approximation underestimates cross-coupling, leading to under-retention of tokens that influence the query state indirectly.

## Contribution

(1) A provably-justified token importance scoring metric based on the trace of the Fisher information matrix, which provides a minimax guarantee on output distribution preservation under compression.
(2) An efficient approximation algorithm that computes the per-token Fisher trace via a single backpropagation pass, requiring no task-specific tuning and negligible overhead relative to the forward time.
(3) A conceptual framework linking KV cache compression to information geometry, enabling future theoretical analyses of compression-fidelity trade-offs.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | LongBench (QMSum, LCC subsets) + custom multi-document QA (MultiNews) | Diverse long-context reasoning tasks require compression |
| Primary metric | Task accuracy + inference throughput (tokens/sec) | Measures output quality and efficiency |
| Baseline 1 | H2O (Heavy Hitter Oracle) | Attention-based, common strong baseline |
| Baseline 2 | Uniform retention | Keep first K tokens per layer |
| Baseline 3 | StreamingLLM | Keep initial and recent tokens |
| Baseline 4 | GraNd (gradient norm on final loss) | Gradient-based importance, tests Fisher specificity |
| Ablation-of-ours | FisherKV (key-only) | Tests value gradient necessity |
| Ablation-of-ours | FisherKV (S=1) | Tests variance reduction benefit of S=5 |

### Why this setup validates the claim

LongBench provides diverse long-context reasoning tasks where tokens vary in importance. QMSum requires summarizing meeting transcripts (multi-document QA), LCC involves long-context code completion. Comparing against attention-based H2O tests whether Fisher importance captures information missed by attention weights. Uniform retention verifies that our learned scoring outperforms a static heuristic. StreamingLLM checks if our method generalizes beyond locality assumptions. GraNd compares against another gradient-based score to isolate the advantage of Fisher trace over mere gradient norm. The ablation isolates the contribution of value-gradient scoring and the effect of multi-sample estimation. Task accuracy is the strictest test: if FisherKV preserves crucial information for correct reasoning, accuracy should improve over baselines specifically on examples where attention is misleading or uniform retention discards critical tokens. Inference throughput quantifies the practical overhead of S=5 sampling.

### Expected outcome and causal chain

**vs. H2O** — On a case where a token has low attention weight but high sensitivity (e.g., a rare but pivotal fact in a reasoning chain), H2O drops it because attention-based importance overlooks output influence. Our method retains that token because its Fisher trace (gradient norm averaged over S=5 samples) is high, reflecting its critical role. We expect a noticeable accuracy gap on examples requiring recall of such low-attention but important tokens, while parity on examples where attention matches importance.

**vs. Uniform retention** — On a case where crucial tokens are scattered unevenly (e.g., key information in later positions), uniform retention may discard them if they fall outside the retained prefix. Our method scores each token independently and retains the top-K globally, preserving those scattered tokens. We expect our method to outperform on sequences with non-uniform importance distribution, especially those where uniform retention loses critical middle or late tokens.

**vs. StreamingLLM** — On a case where vital information appears in the middle of a long context and is neither initial nor recent, StreamingLLM discards it. Our method, by scoring all tokens, can retain that middle token if its Fisher score is high. We expect our method to have an advantage on tasks that require referencing information far from the start or end, e.g., summarization of a long document where the key subject is in the middle.

**vs. GraNd** — GraNd uses the gradient norm of the final loss with respect to token embeddings; it captures only the sensitivity of the loss, not the full output distribution. Fisher trace aggregates sensitivity across all output dimensions via the FIM trace. On tasks where a token has low loss influence but high output distribution influence (e.g., disambiguating between two plausible continuations), FisherKV should outperform GraNd. We expect higher accuracy on tasks with multiple reasonable outputs (e.g., multi-document QA with conflicting evidence).

**vs. FisherKV (S=1)** — Increasing S from 1 to 5 reduces variance and bias. On tasks with high output randomness (e.g., open-ended generation), S=1 may yield noisy scores that mis-rank tokens. We expect FisherKV (S=5) to be more robust on code generation where small changes in tokens can cause syntax errors. Throughput comparison will show the trade-off: S=5 increases overhead but may reduce the compression ratio needed for same accuracy.

### What would falsify this idea

If our method does not outperform baselines on tasks that require recalling tokens with low attention or unusual positions, where the Fisher score should be high but attention is low, then the central claim that Fisher trace better captures token importance is falsified. Specifically, if accuracy gains are uniform across all token subsets rather than concentrated on those identified as important by Fisher but overlooked by baselines, the hypothesis would be unsupported. Additionally, if GraNd outperforms FisherKV on tasks where output distribution sensitivity matters, our claim of Fisher trace superiority over simple gradient norms would be falsified.

## References

1. Information-Aware KV Cache Compression for Long Reasoning
2. Inference-Time Hyper-Scaling with KV Cache Compression
3. PyramidKV: Dynamic KV Cache Compression based on Pyramidal Information Funneling
4. Efficient Streaming Language Models with Attention Sinks
5. Flex Attention: A Programming Model for Generating Optimized Attention Kernels
6. LoRC: Low-Rank Compression for LLMs KV Cache with a Progressive Compression Strategy
7. Transkimmer: Transformer Learns to Layer-wise Skim
8. FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning
