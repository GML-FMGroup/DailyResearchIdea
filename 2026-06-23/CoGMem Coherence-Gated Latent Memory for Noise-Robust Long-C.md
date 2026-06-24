# CoGMem: Coherence-Gated Latent Memory for Noise-Robust Long-Context Retrieval

## Motivation

Existing latent memory models such as EvoEmbedding update memory with every input segment, leading to noise accumulation and degraded retrieval over long contexts. This occurs because they lack a mechanism to filter inputs that are irrelevant or inconsistent with the ongoing context. A structural convergence gap across multiple prior works (EvoEmbedding, LightMem) is that they assume raw input can be directly used to update memory without explicit noise modeling, violating noise-robustness.

## Key Insight

Gating memory updates on the semantic alignment between the current input and a slowly evolving context representation naturally suppresses noise without requiring a separate noise model.

## Method

We propose CoGMem, a coherence-gated latent memory update mechanism. **What it is:** CoGMem maintains a latent memory vector M and a temporal context representation C (exponential moving average of past coherent inputs). It updates M only when the current input embedding x_t exhibits sufficient cosine similarity to C, thereby filtering noise implicitly. **How it works:**

```pseudocode
Initialize:
  M ← 0 vector (latent memory)
  C ← 0 vector (temporal context representation)
  γ ← 0.9 (memory decay factor)
  β ← 0.9 (context decay factor)
  τ ← 0.5 (coherence threshold, default; calibrated per domain)

# Calibration step: select τ to maximize F1 on a calibration set of 512 examples
# with known noise labels (synthetic or annotated). Verify cosine similarity
# achieves AUC > 0.8 in distinguishing clean vs. noisy inputs.
τ ← calibrate_threshold({(x_i, noise_label_i) for i=1..512})

For each input segment embedding x_t:
  s_t ← cosine_similarity(x_t, C)
  if s_t > τ:
    M ← γ * M + (1 - γ) * x_t   # update memory with decay
    C ← β * C + (1 - β) * x_t    # update context representation
  else:
    # skip update; M and C unchanged

Use M as query for retrieval (e.g., dot product with external memory).
```

**Why this design:** We chose additive update with decay over a learned neural network because it is parameter-efficient, interpretable, and avoids overfitting to training distribution, accepting the cost that simple averaging cannot model complex interactions. We used cosine similarity as the gating signal rather than a learned classifier because it is zero-shot and requires no additional training data, though it may miss semantic nuances that a learned model would capture. We set the context representation as an exponential moving average (EMA) of past coherent inputs rather than a learned state because EMA provides a stable baseline that is robust to short-term fluctuations, but it introduces two hyperparameters (β and τ) that need tuning per domain. Finally, we apply gating to both memory and context update to prevent drift of the context from irrelevant inputs, but this coupling means that if a noisy input accidentally passes the gate, it corrupts both memory and context temporarily. A calibration step on a small domain-specific dataset (512 examples) is used to set τ (the coherence threshold) by maximizing F1 score for discriminating clean vs. noisy inputs, ensuring the assumption that cosine similarity reliably distinguishes coherence holds for the target domain.

**Why it measures what we claim:** The cosine similarity s_t measures coherence (consistency with ongoing context) because we assume that in a typical long-context task, topic shifts are gradual and inputs are semantically aligned with the current focus; this assumption fails when there is an abrupt topic change (e.g., a new unrelated question), causing s_t to be low for a relevant input, resulting in a false rejection. The gating condition s_t > τ operationalizes noise filtering: inputs with low similarity are treated as noise and excluded from memory under the assumption that noise is semantically dissimilar from context; this assumption fails when the noise is on-topic but redundant (e.g., repetition), allowing irrelevant content to enter memory. The EMA context C measures the temporally evolving context invariant because it aggregates past coherent inputs with exponentially decaying weights, approximating the central theme; this assumption fails when the topic oscillates rapidly (e.g., alternating subjects), causing C to be a poor representation of the current local context, leading to incorrect gating decisions.

## Contribution

(1) A novel coherence-gated latent memory update mechanism that uses a temporal context representation and cosine similarity to selectively incorporate inputs, implicitly filtering noise. (2) The design principle that simple non-parametric gating can improve noise robustness in latent memory without additional training or noise models. (3) A concrete algorithm (CoGMem) with interpretable hyperparameters that can be integrated into existing long-context retrieval architectures.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|-----------------------|
| Dataset | LongMemEval (with up to 1M tokens per document) | Long-context with topic shifts; supports 10× longer contexts than prior benchmarks. |
| Primary metric | Retrieval accuracy (top-1) | Directly measures retrieval success. |
| Baseline | LightMem | Recent memory-augmented generation system with learned update. |
| Baseline | EvoEmbedding | Evolvable representations for long-context, no gating. |
| Ablation-of-ours | CoGMem w/o gating | Tests necessity of coherence gating (τ=0 always update). |
| Diagnostic | Synthetic noise with ground-truth labels | Correlates cosine similarity to noise labels (AUC, Spearman ρ). |

### Why this setup validates the claim

LongMemEval contains documents with interleaved irrelevant queries and topic drifts, directly testing coherence-based noise filtering. LightMem uses a learned memory update, so comparing against it isolates whether coherence gating outperforms learned gating on noisy segments. EvoEmbedding updates memory incrementally without gating, testing if selective update beats full accumulation. The ablation (no gating) quantifies the net benefit of gating. Retrieval accuracy is the direct measure: correct retrieval after noisy context. If coherence gating works, we expect larger gains on high-noise subsets (e.g., 30% injected noise), and if gains are uniform, the core claim is not supported. A diagnostic experiment on a synthetic dataset with known noise labels computes the AUC of cosine similarity (vs. EMA context) for discriminating clean vs. noisy inputs; and reports the Spearman rank correlation between similarity and noise labels. This validates the load-bearing assumption that cosine similarity reliably distinguishes coherence from noise in high-dimensional spaces.

### Expected outcome and causal chain

**vs. LightMem** — On a case where a noisy, semantically unrelated sentence interleaves with relevant context, LightMem's learned memory update may mistakenly incorporate noise because its gate is trained to maximize overall performance, not specifically detect coherence. Our method instead uses cosine similarity with an exponential moving average context, which naturally filters out inputs that deviate from the ongoing theme. We expect a noticeable gap on the subset of long-doc queries with high noise injection (e.g., accuracy >80% vs. <70%), but parity on clean, well-structured documents.

**vs. EvoEmbedding** — On a case where the context gradually shifts topics, EvoEmbedding's incremental update accumulates all input representations, causing the representation to blur relevant and irrelevant content. Our method's gating only updates memory when coherence is high, preserving sharp topic boundaries. We expect our method to perform better on tasks requiring precise retrieval after a topic shift (e.g., +15% accuracy on shift-prone subsets), while EvoEmbedding may have similar accuracy on stationary topics.

### What would falsify this idea
If our method performs similarly to or worse than the ablation on the noisy subset, then the gating mechanism is not effectively filtering noise, contradicting the central claim. Alternatively, if our gains are uniform across all subsets, then the gating is not specifically addressing coherence but rather a general performance improvement. Additionally, if the diagnostic AUC is below 0.75 on the synthetic set, the assumption that cosine similarity reliably discriminates noise is violated, undermining the core mechanism.

## References

1. EvoEmbedding: Evolvable Representations for Long-Context Retrieval and Agentic Memory
2. LightMem: Lightweight and Efficient Memory-Augmented Generation
3. Memory in the Age of AI Agents
4. Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory
5. From Isolated Conversations to Hierarchical Schemas: Dynamic Tree Memory Representation for LLMs
6. Towards LifeSpan Cognitive Systems
7. Titans: Learning to Memorize at Test Time
8. Needle in the Haystack for Memory Based Large Language Models
