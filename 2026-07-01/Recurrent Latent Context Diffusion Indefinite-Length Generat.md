# Recurrent Latent Context Diffusion: Indefinite-Length Generation with Parallel Block Denoising

## Motivation

Existing multi-block diffusion language models (e.g., MBD-LMs with Block Buffer) are restricted to a fixed-size running set of blocks, limiting generation length to the window size. The root cause is the lack of memory across windows: each window's denoising is conditioned only on its immediate prefix, not on the full history. This structural constraint prevents indefinite extension while maintaining the parallelism of block-wise denoising. The RLC module addresses this by compressing denoised block representations into a fixed-size recurrent latent state, enabling context to propagate across windows without sequential token-level dependencies.

## Key Insight

By aggregating denoised block representations into a fixed-size latent state via a learned recurrent function, we condition future block denoising on the entire history while keeping per-block compute constant and preserving parallel denoising within each sliding window.

## Method

### Recurrent Latent Context Diffusion: Indefinite-Length Generation with Parallel Block Denoising

(A) **What it is** – RLC-Diff adds a Recurrent Latent Context (RLC) module to a multi-block diffusion language model. The RLC takes as input the denoised representations of a completed window of blocks and outputs a fixed-size latent vector that conditions the next window's diffusion process. This enables unlimited generation length by sliding the window forward.

(B) **How it works** – The procedure for extended generation:

```
RLC-Diff Decoding Procedure:
1. Initialize latent state h_0 = 0 (size 256).
2. For window index w = 0, 1, 2, ...:
   a. Define current window W_w = {B_{w*K}, ..., B_{(w+1)*K-1}} (K = 4 blocks per window).
   b. For each block B_i in W_w, denoise it in parallel using the multi-block diffusion model conditioned on h_w and the clean prefix (for i>0, the clean prefix includes tokens from previous blocks in the window, via the Block Buffer attention mask).
   c. After all blocks in W_w are denoised, compute block representations: for each block B_i, extract its hidden states from the diffusion model's final layer (dimension 768). Let r_i = MeanPool(B_i) → 768-dim vector.
   d. Update recurrent state: h_{w+1} = GRU([r_{wK}, ..., r_{(w+1)K-1}], h_w) with hidden size 256.
3. Generated text is the concatenation of all blocks from all windows.
```
**Hyperparameters:** K = 4, GRU hidden size = 256, block representation dimension = 768 (from denoiser). The GRU is trained jointly with the base diffusion model using the same denoising loss (L2 loss on noise prediction).

(C) **Why this design** – We chose a recurrent neural network (GRU) to aggregate block representations because it can handle variable-length sequences of blocks and compress them into a fixed-size state, enabling indefinite extension. A simpler alternative like average pooling would discard temporal order, but we accept the additional computational cost of a GRU for preserving context ordering. We chose to update the latent state only after each window (not per block) to avoid sequential dependencies between blocks within a window, maintaining the parallel denoising property. This trades off some granularity in context for parallelism. We use mean pooling to extract each block's representation from the denoiser's final hidden states; we considered using the [CLS] token or the last token, but mean pooling is more robust to variable block lengths and reduces noise from outlier positions. The trade-off is that mean pooling may dilute strong signals from specific tokens, but we accept this since the diffusion model's hidden states already encode local semantics. Finally, we train the RLC module jointly with the base diffusion model using the same denoising objective, which ensures that the latent state learns to capture information useful for future block denoising, rather than using separate pre-training.

(D) **Why it measures what we claim** – The latent state h_w, computed as the output of the GRU over block representations, measures 'global context' because we assume that the mean-pooled block representation r_i captures the semantic content of block i that is relevant for future generation; this assumption fails when a block's meaning is dominated by a single rare token that mean pooling dilutes, in which case h_w would underrepresent that token's influence. More precisely, r_i measures the average semantic relevance of block i, under the assumption that critical information is uniformly distributed across tokens—this assumption is false for rare tokens with high importance, leading to diluted signal. The GRU's hidden state size (256) measures 'compression capacity' because we assume that 256 dimensions are sufficient to encode the relevant information from K=4 blocks; this assumption fails when the context is highly complex or K is large, causing information loss. The parallel denoising within a window (step 2b) measures 'parallel efficiency' because we assume that blocks within a window are independent given the latent state h_w and the fixed prefix; this assumption fails when there are strong cross-block dependencies that the latent state cannot capture, leading to quality degradation. The RNN update across windows introduces a sequential dependency at the window level, but this is acceptable because window-level sequentiality is much coarser than token-level sequentiality, preserving the parallel block denoising benefit within windows.

## Contribution

(1) The Recurrent Latent Context (RLC) module that aggregates denoised block representations into a fixed-size latent state, enabling indefinite-length generation with multi-block diffusion models. (2) A sliding-window decoding algorithm that maintains parallel block denoising within each window while propagating context across windows via the recurrent state. (3) A design principle that context compression via a recurrent network can be seamlessly integrated into diffusion language models without disrupting the parallel decoding paradigm.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|-----------|
| Dataset | PG-19 (long documents) | Tests long-range coherence |
| Primary metric | Perplexity on 1024-token spans | Measures model's next-token prediction |
| Baseline | SingleBD | No context extension limit |
| Baseline | FixedWindow (K=4) | Same window size, no recurrent state |
| Baseline | MeanPool (avg over blocks) | Recurrent replaced by average pooling |
| Baseline | TransformerXLRecurrence | Uses Transformer-XL-style recurrence for context |
| Ablation-of-ours | Ours w/o GRU (mean pool) | Tests necessity of recurrent mechanism |
| Resource metrics | Training time and peak memory (GPU-hours, GB) | Measures scalability (reported for 1M steps) |

Additionally, we evaluate on BookSum for long-document summarization using ROUGE-L as a downstream metric to demonstrate practical utility.

### Why this setup validates the claim

This setup directly tests the central claim that RLC-Diff enables unlimited generation with efficient parallelism and context compression. PG-19 contains documents far longer than typical diffusion model windows, requiring cross-window context to maintain low perplexity. The baselines isolate key mechanisms: SingleBD fails beyond fixed length; FixedWindow has no cross-window memory; MeanPool loses temporal order; TransformerXLRecurrence tests whether the specific GRU-over-block-representations design is superior to using attention-based recurrence at the token level. The metric (perplexity on long spans) captures how well each model aggregates far-away context. The ablation (ours w/o GRU) pinpoints the recurrent state's role. Resource metrics (training time, memory) verify feasibility and scalability. If our method outperforms all baselines, especially on later parts of documents, it validates that the RLC module effectively compresses and propagates context without sacrificing parallelism.

### Expected outcome and causal chain

**vs. SingleBD** — On a 2000-token document, SingleBD cannot generate beyond its maximum block count, so it either truncates or produces incoherent completions after that point because it has no mechanism to condition on earlier windows. Our method instead maintains a recurrent state that summarizes each completed window, allowing generation to continue indefinitely with coherent global context. We expect SingleBD perplexity to rise sharply beyond its fixed length (e.g., >15 points higher for the second half), while ours stays stable within a few points of its initial perplexity.

**vs. FixedWindow** — In a narrative where a character's name introduced in window 0 is referenced in window 3, FixedWindow lacks memory of window 0, so it may generate incorrect pronouns or contradictory facts because its context is limited to the current window. Our method's GRU compresses and carries forward block representations from all past windows, so it retains necessary information. We expect FixedWindow's perplexity on windows 3+ to be >10% higher than ours when coreference is required; on windows with no long-distance dependencies, both perform similarly.

**vs. MeanPool** — On a procedure like "First add salt, then stir" where order is critical, MeanPool treats all blocks as unordered, potentially predicting "stir" before "add" because it loses temporal sequence. Our GRU preserves order within the recurrent update, so it models the correct temporal dependencies. We expect MeanPool's perplexity to be ~10% higher than ours on sequences with strong temporal cues, but nearly identical on order-invariant content (e.g., lists of items).

**vs. TransformerXLRecurrence** — TransformerXLRecurrence uses self-attention across token representations from previous windows, which captures finer-grained context but introduces sequential token-level dependencies, reducing parallelism. Our method compresses block-level representations with a GRU, maintaining parallel denoising within windows. We expect our method to achieve lower perplexity than TransformerXLRecurrence on long documents due to better compression of block-level semantics, and faster decoding throughput (measured in tokens per second) due to parallel block denoising.

### What would falsify this idea

If our method does not significantly outperform MeanPool on long documents with clear temporal dependencies, or if the perplexity gap between FixedWindow and our method is uniform across all positions (not concentrated on later windows where context loss should matter), then the recurrent state is not capturing meaningful cross-window context as intended. Additionally, if TransformerXLRecurrence achieves lower perplexity with comparable or better speed, it would challenge the design choice of compressing into a fixed-size latent state.

## References

1. Multi-Block Diffusion Language Models
2. LoPA: Scaling dLLM Inference via Lookahead Parallel Decoding
3. AdaBlock-dLLM: Semantic-Aware Diffusion LLM Inference via Adaptive Block Size
4. Scalable Diffusion Models with Transformers
5. Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution
6. Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters
7. Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads
8. EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty
