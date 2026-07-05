# AELOCOS: Attention Entropy-Guided Contrastive Scoring for Position-Agnostic Head Attribution

## Motivation

LOCOS (Logit-Contribution Scoring) effectively identifies non-literal retrieval heads but requires pre-specified needle positions, limiting its use when needle positions are unknown. This dependency exists because the contrastive logit score contrasts contributions from known needle vs. off-needle tokens. To enable head attribution without positional priors, we need a method that automatically infers needle tokens from the model's own attention patterns, grounded in the observation that non-literal retrieval heads exhibit high attention entropy due to distributed semantic associations.

## Key Insight

Attention entropy, which measures the dispersion of a head's attention distribution, reliably distinguishes non-literal retrieval heads (high entropy) from literal ones (low entropy), allowing automatic inference of needle tokens from the head's own attention weights.

## Method

(A) **What it is** — AELOCOS is a position-agnostic head attribution method that uses attention entropy to automatically select candidate needle tokens for a head, then computes a contrastive logit score between those tokens and the remaining context to measure the head's importance. We use a 12-layer, 12-head GPT-2 Medium (355M parameters) as the model, with hyperparameters entropy_threshold=2.0 and alpha=2.0 unless otherwise noted.

(B) **How it works** — Pseudocode:
```python
def aelocos(model, answer_token_position, entropy_threshold=2.0, alpha=2.0):
    # For each head h
    scores = {}
    for h in all_heads:
        # Get attention distribution from answer token to all source positions
        attn = get_attention(model, h, answer_token_position)  # shape (seq_len,)
        # Compute entropy
        entropy = -sum(attn * log(attn + 1e-8))
        if entropy < entropy_threshold:
            # Assume literal head, skip
            scores[h] = None
            continue
        # Set number of candidate needles proportional to entropy
        k = int(alpha * exp(entropy))
        k = min(k, seq_len-1)  # ensure valid
        # Select top-k attended positions as candidate needles
        candidate_indices = argmax_k(attn, k)
        # Compute OV-circuit output projection for each source position
        ov_proj = compute_ov_projection(model, h, answer_token_position)  # shape (seq_len,)
        # Contrastive score: mean over candidates - mean over rest
        needle_mean = mean(ov_proj[candidate_indices])
        off_mean = mean(ov_proj[~candidate_indices])
        scores[h] = needle_mean - off_mean
    # Validation: For a subset of heads (e.g., 10% with highest and lowest entropy), use causal tracing
    # (e.g., activation patching) to verify that the top-k candidate indices are causally important
    # for the answer token. This serves as a calibration step to confirm the load-bearing assumption.
    # If validation fails (e.g., low correlation), we adjust the threshold or k scaling.
    # Rank heads by score and ablate top-k (e.g., k=50)
    # Measure ROUGE-L drop on NoLiMa
```

(C) **Why this design** — We chose entropy as the criterion for needle inference because it captures the degree of dispersion in attention, which correlates with non-literal retrieval (where heads attend to multiple semantically related tokens) better than metrics like max attention weight or variance. A trade-off: using entropy directly to set k adaptively (k ∝ exp(entropy)) avoids a fixed hyperparameter but can be sensitive to outlier heads with extremely high entropy; we cap k to prevent selecting too many tokens. We threshold entropy at 2.0 to filter out heads with near-deterministic attention (likely literal) and focus on heads with spreading attention, accepting that some non-literal heads may be missed if entropy is slightly below threshold. The contrastive score itself is borrowed from LOCOS because it effectively isolates head contributions to non-literal retrieval; an alternative (e.g., logit difference from all tokens) would not contrast the inferred needle set. We used OV-projection onto the unembedding of the answer token because it measures direct logit contribution, consistent with prior work.

(D) **Why it measures what we claim** — The attention entropy measures the dispersion of a head's attention distribution; we assume that high dispersion (entropy > threshold) indicates non-literal retrieval because the head attends to multiple semantically related tokens that collectively form the needle; this assumption fails when a literal retrieval head accidentally spreads attention due to noise (e.g., repeated tokens), in which case entropy may be high but the contrastive score will be low because the candidate set includes many irrelevant tokens. The number of candidate needles k = α·exp(entropy) measures the size of the needle set, under the assumption that attention weights are uniformly distributed over all needle tokens and that entropy grows exponentially with set size. A failure mode occurs when non-uniform weights or irrelevant tokens inflate entropy without increasing the actual needle count, causing k to overestimate the needle set and dilute the contrastive score. The contrastive score (mean over candidates minus mean over rest) measures the head's differential logit contribution to the answer token from inferred needle tokens versus non-needle tokens, which under the assumption that the inferred set correctly captures all needle tokens extracts the head's role in non-literal retrieval; if the assumption fails (e.g., some needle tokens are not in the top-k), the score underestimates the head's importance.

An explicit load-bearing assumption is that attention entropy reliably distinguishes non-literal retrieval heads (high entropy) from literal ones (low entropy), and that the top-k attended positions correspond to actual needle tokens. To verify this, we conduct a calibration step: for a random subset of 100 heads, we use causal tracing (activation patching) to measure the causal effect of each candidate token on the answer token. If the average causal effect of candidate tokens is significantly higher than that of non-candidate tokens, the assumption is validated; otherwise, we adjust the entropy threshold or alpha.

## Contribution

(1) AELOCOS, a position-agnostic head attribution method that automatically infers needle tokens via attention entropy and contrastive logit scoring, eliminating the need for pre-specified needle positions. (2) An empirical finding that attention entropy reliably distinguishes non-literal retrieval heads from literal ones, enabling automatic needle inference without supervision. (3) A framework for adaptive needle set selection that adjusts the candidate set size proportionally to attention entropy, improving robustness across varying head behaviors.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | NoLiMa | Tests non-literal retrieval in long context |
| Primary metric | ROUGE-L | Measures generation quality concisely |
| Baseline 1 | LOCOS | Prior contrastive scoring without entropy |
| Baseline 2 | NoLiMa short-length | Detects context-specific degradation |
| Baseline 3 | Random head ablation | Sanity check for attribution quality |
| Ablation of ours | Fixed k (no entropy adaptation) | Isolates effect of adaptive needle selection |

### Why this setup validates the claim

This combination creates a falsifiable test of the central claim—that entropy-based adaptive needle selection improves attribution for non-literal heads. NoLiMa’s long-context, ambiguous retrieval tasks demand such adaptive selection; ROUGE-L captures generation fidelity. LOCOS serves as the direct prior art, isolating our improvement. The short-length baseline tests whether gains are specific to long context (where dispersion is needed). Random ablation verifies our scores are meaningful. The fixed-k ablation tests whether the entropy adaptation itself—not just the contrastive structure—drives performance. If our method only matches LOCOS on short contexts but surpasses it on long, the claim is supported.

### Expected outcome and causal chain

**vs. LOCOS** — On a case where a head’s attention is widely dispersed over many needle tokens, LOCOS uses a fixed top-k (e.g., 10) and may miss several true needles, diluting the contrastive score and underestimating head importance. Our method adapts k via entropy, capturing the full needle set, so the contrastive score better isolates the head’s true contribution. We expect a noticeable ROUGE-L improvement on NoLiMa subsets with high entropy distributions, but parity on low-entropy (literal) ones.

**vs. NoLiMa short-length** — On a long-context instance where a non-literal retrieval head attends to many relevant tokens spread across the input, models using our method maintain performance near the short-length baseline; other methods degrade sharply because their fixed-k or literal attention fails. Our method’s gain should concentrate on long-context tasks, with short-context performance similar across methods. We expect a clear gap (e.g., 5-10 ROUGE-L points) on long contexts but no gap on short.

**vs. Random head ablation** — On an instance requiring non-literal retrieval, random ablation of a head often leaves performance unchanged (since irrelevant heads are ablated). Our method ablates heads with high contrastive scores, which should be precisely those crucial for non-literal retrieval. Thus, our ablation causes a larger performance drop on non-literal subsets than random ablation. We expect a significant interaction: our ablation’s drop is 2-3× larger on non-literal than literal tasks.

### What would falsify this idea

If our method’s gains over LOCOS are uniform across all heads (i.e., no correlation with entropy) or if the fixed-k ablation performs similarly to our method, then the central claim—that entropy-driven adaptive needle selection is the key—is false.

## References

1. Logit-Contribution Scoring Identifies Non-Literal Retrieval Heads
2. Gemma 3 Technical Report
3. NoLiMa: Long-Context Evaluation Beyond Literal Matching
4. HELMET: How to Evaluate Long-Context Language Models Effectively and Thoroughly
5. Gemma 2: Improving Open Language Models at a Practical Size
6. BAMBOO: A Comprehensive Benchmark for Evaluating Long Text Modeling Capacities of Large Language Models
7. BooookScore: A systematic exploration of book-length summarization in the era of LLMs
8. GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints
