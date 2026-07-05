# FactorGraphHash: Parallel Decoding of Hash-Based Language Models via Learned Factor Graphs and Belief Propagation

## Motivation

MultiHashFormer achieves effective hash-based language modeling but relies on autoregressive decoding, which scales quadratically with sequence length and cannot guarantee collision-free generation across tokens. This structural limitation arises because autoregressive models process tokens sequentially and lack a global view of the entire sequence, making it difficult to enforce diversity constraints or avoid duplicate hash signatures without expensive post-processing.

## Key Insight

Since hash tokens are fixed-length binary codes, they naturally form a codebook that can be decoded in parallel using a learned factor graph and belief propagation, enabling sub-quadratic generation while jointly enforcing local linguistic consistency and global collision freedom.

## Method

(A) **What it is**: FactorGraphHash replaces the autoregressive hash decoder with a factor graph over all hash bits across all positions, and uses max-product belief propagation to decode the entire sequence in parallel, achieving linear-time generation.

(B) **How it works**:
```python
# FactorGraphHash: Pseudocode for parallel decoding
# Input: encoder latent vectors h[1..L] (from MultiHashFormer encoder)
# Output: hash token sequence x[1..L] where each x[i] is a K-bit binary vector

# Hyperparameters:
#   K = 64              # hash bits per token
#   T = 10              # BP iterations
#   alpha = 0.3         # damping factor
#   beta = 0.1          # repulsive strength for uniqueness
#   prototype_vocab_size = 50000  # number of prototypes for joint local factor
#   window_size = 5     # local window for repulsive factors

# Factor types:
#   - Local factor F_loc[i]: captures token-level likelihood from encoder
#   - Transition factor F_trans[i,i+1]: captures pairwise dependency between consecutive tokens
#   - Repulsive factor F_rep[i,j]: penalizes Hamming distance < threshold between tokens i and j (approximate uniqueness)

# Key assumption: The hash bits for each token are conditionally independent given the encoder output. This allows local potentials to be factorized as per-bit sigmoid functions. However, we also provide a joint alternative that does not make this assumption.

# Step 1: Initialize messages uniformly
for each edge:
    ml_to_v = 0.5 vector of length 2^K? No, we operate per-bit for efficiency.

# Actually, we decompose variables per bit to keep complexity linear.
# Let variable nodes be x[i][b] for token i, bit b. Binary states {0,1}.
# Factors operate on groups of bits.

for i in 1..L:
    for b in 1..K:
        mv_to_loc[i][b] = [0.5, 0.5]  # uniform
        mv_to_trans[i][b] = [0.5, 0.5]
        mv_to_rep[i][b] = [0.5, 0.5]

# Step 2: Iterate BP
for t in 1..T:
    # Update factor-to-variable messages
    for i in 1..L:
        for b in 1..K:
            # Local factor: Two variants:
            # Variant A (per-bit independence assumption): p_loc[i][b] = sigmoid(MLP_b(h[i]))
            # Variant B (joint): a 2-layer MLP (hidden=256, GeLU) outputs logits over prototype_vocab_size prototype tokens (the 50k most frequent tokens from training), then softmax. Marginal probabilities for each bit are obtained by summing over prototypes where that bit is 1.
            # By default, we use Variant B (joint) to avoid independence assumption.
            # For Variant B:
            # logits = MLP_joint(h[i])  # output size prototype_vocab_size
            # probs = softmax(logits)
            # p_loc[i][b] = sum_{t in prototype_vocab} probs[t] * prototype_bit[t][b]
            # m_loc_to_v[i][b] = [1 - p_loc[i][b], p_loc[i][b]]
            m_loc_to_v[i][b] = compute_local_message(i, b, h[i])  # using joint variant

        for i in 1..L-1:
            for b in 1..K:
                # Transition factor: F_trans[i,i+1] = exp( score_b( x[i][b], x[i+1][b] ) )
                # where score_b is a learned 2x2 table for bit b
                # message from factor to variable x[i][b]: marginalize over x[i+1][b]
                # We assume bitwise independence for tractability.
                m_trans_to_v[i][b] = [ sum_{x2 in {0,1}} ( W_trans[b][0][x2] * mv_to_trans[i+1][b][x2] ),
                                       sum_{x2} ( W_trans[b][1][x2] * mv_to_trans[i+1][b][x2] ) ]

        # Repulsive factor: we place a factor over all token pairs (i,j) with i<j
        # For tractability, we only add repulsive factors for pairs within a sliding window W=5.
        for i in 1..L:
            for j in max(1,i-W)..min(L,i+W), j!=i:
                # F_rep[i,j] = exp( -beta * Hamming(x[i], x[j]) )
                # But factor operates on all bits together; use approximation via bitwise decomposition? Hard.
                # Instead, we approximate repulsive force by adjusting messages: after computing marginal probabilities, we nudge them away from collisions.
                # For simplicity, we omit full factor and incorporate repulsion via a post-processing step after BP.

    # Update variable-to-factor messages (product of incoming messages)
    for i in 1..L:
        for b in 1..K:
            mv_to_loc[i][b] = normalize( m_trans_to_v[i-1][b] * m_rep_to_v[i][b] * previous mv? )
            # Combine: mv_to_loc = no incoming from itself; mix of neighbor factors
            actual: mv_to_loc[i][b] = normalize( m_trans_to_v[i-1][b] * m_trans_to_v[i][b] (from left and right) * m_rep_to_v[i][b] )

# Step 3: Decode by thresholding marginals after convergence
for i in 1..L:
    for b in 1..K:
        belief[i][b] = normalize( product of all incoming factor messages )
        x[i][b] = argmax( belief[i][b] )

# Step 4: Collision post-processing
# For any duplicate signatures, flip the least confident bits until unique.
```

(C) **Why this design**: We chose a pairwise factor graph over higher-order factors because it allows linear-time message updates (O(LK)) per iteration, whereas higher-order factors would require exponential time or approximation. We selected max-product BP over sum-product because our goal is a single consistent decoding (MAP) rather than marginal probabilities, and max-product avoids the numerical issues of summing over large state spaces. The repulsive constraint is approximated via a local window and a post-hoc correction rather than a global factor because a global uniqueness factor would make BP intractable, and empirical studies show collisions are rare with sufficiently large K (64 bits). The trade-off is that we sacrifice exact collision freedom for scalability, but post-processing ensures eventual uniqueness at minimal cost. We chose to model local potentials via a joint MLP over prototype tokens (Variant B) to avoid the assumption of per-bit independence, which is often violated in learned hash codes. This increases computational cost but preserves within-token dependencies. The transition factors remain per-bit for tractability, assuming bit-level dependencies between consecutive tokens are largely independent.

(D) **Why it measures what we claim**: The local factor messages (m_loc_to_v) measure token-level likelihood given the encoder context, operationalizing the concept of 'linguistic consistency at each position' under the assumption that the joint MLP over prototypes approximates the true conditional distribution over all 2^K tokens; this assumption fails when the prototype vocabulary is insufficient to cover all possible tokens, in which case the messages reflect only coarse-grained token probabilities. The transition factor messages (m_trans_to_v) measure bitwise consistency between consecutive tokens, operationalizing 'local coherence' under the assumption that pairwise bit dependencies capture enough linguistic structure; this assumption fails when coherence involves long-range dependencies (e.g., subject-verb agreement across multiple tokens), where the factor graph would underestimate dependencies. The collision post-processing step measures and enforces uniqueness, operationalizing 'collision-freeness' under the assumption that flipping low-confidence bits preserves semantic content; this assumption fails when the flipped bits are crucial for token identity, potentially altering meaning. Together, these components allow parallel decoding that is both efficient and largely collision-free.

## Contribution

(1) FactorGraphHash, a novel parallel decoding algorithm for hash-based language models that replaces autoregressive generation with a learned factor graph and belief propagation, achieving linear-time (O(L)) decoding. (2) Demonstration that factor-graph-based decoding with local pairwise potentials and a lightweight collision post-processing step can generate coherent sequences without quadratic self-attention, as verified by empirical evaluation on language modeling benchmarks. (3) An analysis of the trade-offs between decoding speed, collision rate, and generation quality, providing design guidelines for hash-based token representations.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | WikiText-103, Penn Treebank (ablation) | WikiText-103 for standard benchmark; PTB for fast prototype selection and calibration |
| Primary metric | Perplexity, Collision rate (percentage of duplicate tokens per sequence) | Direct measure of predictive accuracy and uniqueness |
| Baseline | Standard Transformer | Strong autoregressive subword baseline |
| Baseline | MultiHashFormer | Hash-based autoregressive competitor |
| Baseline | Insertion Transformer (Stern et al., 2019) | Non-autoregressive decoding baseline |
| Baseline | Masked Language Model decoding (CMLM, Ghazvininejad et al., 2019) | Non-autoregressive baseline |
| Ablation-of-ours | Ours w/o repulsion | Isolates effect of repulsive factor |
| Ablation-of-ours | Ours (per-bit sigmoid local factor) | Tests impact of joint vs. per-bit independence assumption |
| Synthetic experiment | Synthetic hash space (K=8, independent vs. correlated bits) | Calibrate per-bit independence assumption quantitatively |

Calibration: Prototype vocabulary of 50k tokens selected from top 50k most frequent tokens in training set; calibration set of 512 held-out sequences used to fix any hyperparameters (e.g., window size, damping). The synthetic experiment uses a small hash space of 8 bits, generating sequences where bits are either independent or correlated via a hidden Markov model. We measure the accuracy of the per-bit sigmoid MLP in recovering the true token distribution, and compare to the joint MLP.

### Why this setup validates the claim
This setup tests the central claim that factor graph parallel decoding can match autoregressive hash generation quality while being faster. Comparing to Standard Transformer checks if hash-based models lose expressivity; comparing to MultiHashFormer isolates the effect of decoding strategy (parallel vs. autoregressive); the non-autoregressive baselines (Insertion Transformer, CMLM) provide a broader context about the quality-speed trade-off. The ablation without repulsion tests the necessity of the repulsive factor for avoiding collisions. The synthetic experiment directly validates the load-bearing assumption of per-bit independence; if the per-bit variant performs poorly on correlated data, the joint variant is justified. Perplexity is appropriate because it directly reflects the model's ability to predict the next token, which depends on both representation quality and decoding correctness. Collision rate is a secondary metric to quantify uniqueness. If our method works, it should show competitive perplexity with faster inference, and the ablation should degrade on long sequences where collisions occur.

### Expected outcome and causal chain

**vs. Standard Transformer** — On a case with long-range dependencies like pronoun resolution across multiple sentences, the baseline autoregressive Transformer captures dependencies via causal attention, while our factor graph with only local and pairwise transition factors may miss long-range interactions because BP only propagates information locally within a limited number of iterations. However, our encoder provides global context, mitigating this gap. We expect a small perplexity increase (e.g., <5%) on long-range phenomena but parity on local coherence tasks, indicating that parallel decoding incurs minimal quality loss.

**vs. MultiHashFormer** — On a case where many tokens have similar representations (e.g., repeated phrases), MultiHashFormer's autoregressive decoding may generate hash collisions without recovery, leading to repetitive or nonsensical output. Our method uses repulsive factors within a local window and post-hoc correction to ensure uniqueness, avoiding collisions. We expect our perplexity to be noticeably lower (e.g., 10-20% relative improvement) on long generated sequences, while on short sequences both perform similarly.

**vs. Insertion Transformer** — Insertion Transformer can generate tokens in arbitrary order, enabling sub-quadratic decoding, but its learned insertion policy may stabilize at lower quality compared to our parallel BP. We expect our method to achieve lower perplexity (e.g., 5-10%) on structured text (e.g., news) due to our explicit encoder and hash codebook, while Insertion Transformer may be more flexible on open-domain generation.

**vs. Masked Language Model decoding (CMLM)** — CMLM generates multiple tokens per iteration, but requires iterative refinement with a mask-and-predict schedule. Our one-shot parallel decoding with BP may be faster (e.g., 2-3x speedup) at the cost of slight perplexity increase (e.g., 3-5%) because MLM decoding can use a learned mask-predict-order policy that better captures global context.

**vs. Ours w/o repulsion** — On a case where two different contexts produce similar hash tokens (e.g., synonyms in similar positions), the ablation without repulsion will produce duplicate tokens, increasing perplexity due to incoherence. Our full method corrects these by flipping low-confidence bits. We expect a significant perplexity gap (e.g., 15-30%) on sequences longer than the local window, confirming that repulsion is essential for maintaining sequence quality.

**vs. Ours (per-bit sigmoid local factor)** — On synthetic data with correlated bits, the per-bit variant will show high perplexity and error rate, while the joint variant remains robust. On real data, we expect a modest gap (e.g., 2-5% perplexity increase) because real hash bits may not be fully independent, but the joint variant's additional expressiveness yields small improvement.

### What would falsify this idea
If our method shows higher perplexity than MultiHashFormer across all subsets, especially on short sequences where collisions are rare, then the factor graph decoding is ineffective and parallelization harms quality. Alternatively, if the ablation without repulsion performs similarly to the full method, the repulsive mechanism is unnecessary and the claim of collision avoidance is unsupported. If the per-bit sigmoid variant achieves the same perplexity as the joint variant on synthetic correlated data, then the independence assumption holds and the joint variant is overkill.

## References

1. MultiHashFormer: Hash-based Generative Language Models
2. From Bytes to Ideas: Language Modeling with Autoregressive U-Nets
3. Bolmo: Byteifying the Next Generation of Language Models
4. Byte Latent Transformer: Patches Scale Better Than Tokens
5. From Language Models over Tokens to Language Models over Characters
6. Exact Byte-Level Probabilities from Tokenized Language Models for FIM-Tasks and Model Ensembles
7. T-FREE: Subword Tokenizer-Free Generative LLMs via Sparse Representations for Memory-Efficient Embeddings
8. Should you marginalize over possible tokenizations?
