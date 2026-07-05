# Hierarchical Perfect Hash Encoding for Collision-Free Token Representations in Language Models

## Motivation

Existing hash-based language models like MultiHashFormer treat tokens as flat signatures, leading to probabilistic collisions and hyperparameter sensitivity in hash function selection. The root cause is the lack of structural constraints that guarantee injectivity, forcing reliance on combinatorial signatures that require careful tuning to avoid ambiguity.

## Key Insight

Decomposing each token's unique minimal perfect hash into a binary tree path imposes a structural invariant that guarantees collision-free representations and enables hierarchical attention without hyperparameter tuning.

## Method

(A) **What it is**: Hierarchical Perfect Hash (HPH) encoding maps each token to a unique depth-limited binary tree path via a minimal perfect hash function, producing a structured representation that directly supports hierarchical attention in a Transformer. Input: sequence of tokens (vocabulary V, size |V|). Output: token representations and multi-level node embeddings for attention. **Assumption**: The minimal perfect hash function H is precomputed for a static vocabulary; dynamic vocabulary expansion is not supported in the current implementation but could be addressed by an incremental perfect hash or learned hash (future work).

(B) **How it works** (pseudocode):
```python
# Preprocessing: build minimal perfect hash H: V -> {0,1}^D (D=16, 2^D >= |V|).
# Tree has 2^{D+1}-1 nodes; each node has a learned embedding of dimension d_model=512.
# For token t, path P(t) = [root, node at level1, ..., leaf] selected by bits of H(t).
#
def forward(input_ids):
    # Step 1: retrieve node embeddings for each token's path
    path_embs = [[node_emb[node_id(bit_prefix)] for bit_prefix in token_path] for token in input_ids]  # list of lists
    # Step 2: aggregate path into token embedding (weighted sum with learned coefficients alpha_l, scalar per level)
    token_embs = [sum(alpha[l] * path_emb[l] for l in range(D)) for path_emb in path_embs]  # shape (seq_len, d_model)
    # Step 3: standard Transformer forward pass with token_embs as input (e.g., 12-layer transformer, 12 heads)
    hidden_states = transformer(token_embs)
    # Step 4: hierarchical attention — for each level l in {4,8,12}, compute cross-attention
    # between hidden_states and node embeddings at that level across the sequence
    level_embs = []
    for l in levels:
        l_nodes = [node_emb[node_id( token_path[l] )] for token_path in path_embs]  # shape (seq_len, d_model)
        # multi-head cross-attention: query from hidden_states, key/value from l_nodes (4 heads, d_k=64)
        level_context = cross_attn(l_nodes, hidden_states)
        level_embs.append(level_context)
    # Step 5: fuse level contexts into hidden_states (e.g., gated sum with learned scalar gates)
    fused = hidden_states + sum(gate * lc for gate, lc in zip(gates, level_embs))
    return lm_head(fused)
```
Hyperparameters: D=16, levels=[4,8,12], alpha_l and gates are learned scalars, d_model=512, d_k=64, 4 cross-attention heads, 12 transformer layers.

(C) **Why this design**: We chose a minimal perfect hash over a random hash to guarantee collision-free mapping, eliminating the sensitivity to signature length and number of hash functions present in MultiHashFormer. We chose a binary tree structure instead of a flat lookup to enable hierarchical decomposition, allowing the model to attend to common prefixes (e.g., subword units) without extra computation. We chose to sum path embeddings (with learned per-level weights) rather than concatenate or use a recurrent aggregator to keep representation size constant and computationally cheap, accepting the loss of positional information within the path. Finally, we incorporated hierarchical cross-attention at multiple fixed depths to capture both fine- and coarse-grained context, trading increased memory cost for richer structural understanding. Notably, unlike learned disentanglement heads (Anti-pattern 1), our method's encoding is deterministic and does not require statistical independence; unlike retrieval-based schemes (Anti-pattern 2), it does not rely on a nearest-neighbor lookup; and unlike confidence-based gating (Anti-pattern 3), it uses structural attention rather than log-probability thresholds. **Memory estimate**: The tree structure stores 2^{D+1}-1 = 131071 node embeddings, each of dimension 512, totalling ~268M parameters (half for embeddings, half for cross-attention and fusion). This is comparable to embedding a flat vocabulary of 131k tokens.

(D) **Why it measures what we claim**: The uniqueness of the minimal perfect hash directly measures collision-free token representation because H is a bijection between the vocabulary and leaf nodes; this assumption fails if the vocabulary changes (e.g., dynamic expansion), in which case collision-free status is lost and the model must use approximate hash. The hierarchical cross-attention over node embeddings at level l measures the degree of shared prefix among tokens because attention weights reflect similarity of node-level embeddings; this assumption fails if node embeddings are not properly trained (e.g., random initialization), in which case attention reflects noise rather than structural similarity. The learned per-level weights alpha_l measure the importance of each depth for token representation because they are optimized to minimize language modeling loss; this assumption fails if the model collapses to using only one depth (e.g., alpha_l=0 for most l), in which case the representation collapses to flat encoding. The gated fusion of level contexts measures the model's ability to use hierarchical information because the gate values are trained to adjust the contribution of each level; this assumption fails if gates saturate at zero, indicating the hierarchical attention is unused, and the model reverts to standard Transformer token representations. **Verification**: To verify that the perfect hash is collision-free, we check that H is a bijection; for dynamic vocabularies, an incremental perfect hash would need to maintain this property. To verify that hierarchical attention captures prefix similarity, we perform a cluster analysis of node embeddings (see experiment analysis).

## Contribution

(1) A novel hierarchical perfect hash encoding scheme that guarantees collision-free token embeddings and eliminates hyperparameter sensitivity in hash-based language models. (2) A hierarchical attention mechanism that operates over the binary tree structure, enabling multi-scale context aggregation from shared prefixes without additional computational overhead. (3) A demonstration that the proposed method addresses the flat encoding limitation identified as a convergent gap across multiple hash-based and byte-level model families.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|------------------------|
| Dataset | WikiText-103 | Standard language modeling benchmark. |
| Primary metric | Token-level perplexity | Measures language modeling quality. |
| Baseline 1 | Standard Transformer (BPE) | Strong subword baseline. |
| Baseline 2 | MultiHashFormer | Hash-based baseline with collisions. |
| Baseline 3 | Byte-level Transformer | Tests hierarchical representation. |
| Baseline 4 | HPH with flat perfect hash (no tree) | Isolates contribution of hierarchical structure. |
| Ablation of ours | HPH with random hash (same tree) | Tests need for collision-free encoding. |

### Why this setup validates the claim
This setup forms a falsifiable test of HPH's central claim—that deterministic collision-free hierarchical token encoding improves language modeling by enabling structured attention. Comparing to Standard Transformer (BPE) isolates the effect of the hash-based representation; outperforming it on perplexity would show that the hierarchical structure encodes subword patterns more effectively than fixed BPE segmentation. Comparing to MultiHashFormer tests whether collision-free mapping (via minimal perfect hash) is beneficial over random hash with collisions; if HPH outperforms, it validates the collision-avoidance mechanism. Comparing to Byte-level Transformer tests whether the learned tree path representation captures byte-level regularities better than a flat byte model; an advantage would highlight the benefit of hierarchical decomposition. The ablation with random hash isolates the contribution of the perfect hash itself—if it performs worse, the minimal perfect hash is critical. The flat perfect hash baseline (no tree) isolates the hierarchical structure—if it performs worse, the tree contributes beyond a perfect hash. Perplexity is ideal because it directly reflects the model's predictive probability, and gains should concentrate on tokens with common prefixes or rare spellings, aligning with the hierarchical attention mechanism. Additionally, we plan a cluster analysis of learned node embeddings: we compute k-means (k=10) on node embeddings at each level and measure the proportion of nodes in each cluster that share a common prefix (e.g., same first two bits). High purity (average >80%) would indicate that the tree structure captures linguistic regularities. We also plan a morphological inflection prediction experiment on CFM (Common Features Morphology) to assess generalization to morphological understanding.

### Expected outcome and causal chain

**vs. Standard Transformer (BPE)** — On a case where two tokens share a subword prefix (e.g., "unable" and "unlike"), BPE tokenizes them as separate units ("un", "able") and ("un", "like"), losing the shared root in the embedding lookup. HPH maps each token to a tree path with a common node at level representing the prefix "un-"; hierarchical attention can attend to that shared node across tokens, improving context modeling. Thus, we expect HPH to show a noticeable perplexity improvement on tokens with common morphological prefixes but parity on unrelated tokens.

**vs. MultiHashFormer** — On a case where the vocabulary contains two similar tokens (e.g., "cat" and "cats"), MultiHashFormer may hash them to distinct leaf signatures but with possible collisions due to random hash; collisions cause interference because the same node embedding is used for dissimilar tokens. HPH's minimal perfect hash guarantees unique leaf assignments, eliminating such interference. Therefore, we expect HPH to have lower perplexity, especially on rare or morphologically variable tokens where collisions are more detrimental.

**vs. Byte-level Transformer** — On a case where a long Unicode string (e.g., emoji sequence) has no clear subword boundaries, a byte-level transformer processes each byte independently, requiring long-range dependencies to capture patterns. HPH's tree structure compresses common byte sequences into shared nodes (e.g., fixed byte patterns map to same prefixes), enabling hierarchical attention to capture local patterns efficiently. Thus, HPH should achieve lower perplexity on byte-like tokens (e.g., URLs, non-ASCII characters) while performing comparably on common words.

**vs. HPH with flat perfect hash** — On a case where two tokens share a long common prefix (e.g., "unbelievable" and "unbalanced"), the flat perfect hash assigns independent leaf embeddings that ignore the prefix overlap. HPH's hierarchical representation allows sharing at multiple levels (e.g., up to level 4 for "un-", level 8 for "unbe-"), leading to better generalization. Thus, we expect HPH to outperform the flat version, especially on tokens with long common prefixes.

### What would falsify this idea
If HPH's perplexity improvement over MultiHashFormer is uniform across all tokens rather than concentrated on tokens with potential collisions, or if the ablation with random hash performs equally well, the central claim that collision-free perfect hashing is beneficial would be falsified. Additionally, if HPH does not outperform the byte-level baseline on morphologically rich or byte-oriented tokens, the hierarchical structure may not capture subword patterns as hypothesized. If the flat perfect hash baseline matches HPH, then the tree structure is unnecessary.

## References

1. MultiHashFormer: Hash-based Generative Language Models
2. From Bytes to Ideas: Language Modeling with Autoregressive U-Nets
3. Bolmo: Byteifying the Next Generation of Language Models
4. Byte Latent Transformer: Patches Scale Better Than Tokens
5. From Language Models over Tokens to Language Models over Characters
6. Exact Byte-Level Probabilities from Tokenized Language Models for FIM-Tasks and Model Ensembles
7. T-FREE: Subword Tokenizer-Free Generative LLMs via Sparse Representations for Memory-Efficient Embeddings
8. Should you marginalize over possible tokenizations?
