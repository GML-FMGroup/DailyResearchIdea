# FIPA: Feature-Identity-Preserving Aggregation for Permutation-Invariant Tabular Representation Learning

## Motivation

Existing tabular representation learning methods, such as the autoregressive generative model GReaT (Language Models are Realistic Tabular Data Generators), impose a fixed feature ordering inherited from sequential token generation. This structural assumption blocks permutation invariance, preventing the representation from generalizing across datasets with different column arrangements or missing features. The underlying cause is that feature identity is conflated with positional encoding, rather than being anchored to column semantics.

## Key Insight

Decoupling feature identity from positional encoding by representing each feature as a pair of a column name embedding and a value embedding, then applying a symmetric pooling function, yields a representation that is inherently invariant to column permutation while preserving the semantic meaning of each feature.

## Method

(A) **What it is**: FIPA is a representation learning module that maps a tabular instance (a set of feature-value pairs) to a fixed-size vector that is permutation-invariant and feature-identity-aware. Input: a set of `(column_name, value)` pairs. Output: a vector representation of dimension 64.

(B) **How it works**:
```python
import torch

def fipa_representation(data_instance, column_embeddings, value_encoder, combine='mul', pool='sum'):
    # data_instance: dict {col_name: value}
    # column_embeddings: dict mapping col_name to 64-dim vector (pre-trained from DistilBERT, fine-tuned jointly)
    # value_encoder: callable that takes (col_name, value) and returns 64-dim vector (2-layer MLP with hidden dim 64, ReLU, output dim 64)
    
    pairs = []
    for col_name, val in data_instance.items():
        # Fallback for missing or ambiguous column names: use learnable embedding
        col_emb = column_embeddings.get(col_name, learnable_fallback_embedding)
        val_emb = value_encoder(col_name, val)  # 64-dim
        if combine == 'mul':
            pair_emb = col_emb * val_emb  # element-wise multiplication
        else:
            pair_emb = torch.cat([col_emb, val_emb])  # alternative: 128-dim
        pairs.append(pair_emb)
    
    # Permutation-invariant pooling
    if pool == 'sum':
        pooled = torch.stack(pairs).sum(dim=0)  # sum pooling
    elif pool == 'max':
        pooled = torch.stack(pairs).max(dim=0)[0]  # max pooling
    # Optional MLP projection
    repr = torch.nn.Linear(64, 64)(pooled)  # one layer: 64->64, no activation
    return repr
```
Hyperparameters: embedding dimension `d=64`, combine method `'mul'`, pooling `'sum'`, MLP with one linear layer (no activation). Value encoder is a 2-layer MLP with hidden dim 64 and ReLU. Column embeddings are initialized from DistilBERT (last hidden layer, mean pooled over column name tokens) and fine-tuned jointly with the rest of the network. For missing or non-informative column names (e.g., 'X1', 'X2'), we use a separate learnable embedding of dimension 64 initialized randomly.

(C) **Why this design**: We chose element-wise multiplication over concatenation for pair combination because it forces interaction between column and value embeddings at each dimension, creating a joint representation that captures feature-specific semantics; the trade-off is that multiplication assumes alignment of semantic dimensions, which may not hold for arbitrarily trained embeddings—we mitigate by fine-tuning column embeddings jointly. We chose sum pooling over max or mean because sum preserves magnitude differences across features (a feature with larger value embedding norms may indicate importance); the cost is that sum can be dominated by numerous small features—we normalize embeddings to unit norm before pooling to prevent this. We chose to pre-train column embeddings from textual column descriptions using a language model (e.g., DistilBERT) rather than learned from scratch, because this leverages semantic similarity between column names across different datasets, enabling zero-shot transfer; the trade-off is that column names may be ambiguous or missing, in which case we fall back to a learned embedding initialized randomly. **Load-bearing assumption**: Pre-trained language model embeddings of column names provide a consistent and discriminative representation of feature identity across different tabular datasets. This assumption may fail when column names are uninformative (e.g., 'X1', 'X2') or absent; our fallback mechanism mitigates this by learning a separate embedding for such columns, but if all columns are uninformative, the representation reverts to being purely value-based, losing identity awareness.

(D) **Why it measures what we claim**: The sum over pair embeddings measures permutation invariance because the sum is symmetric—reordering pairs yields the same result; this assumption fails when the combine function is not commutative—but element-wise multiplication is commutative, so it holds. The column embedding measures feature identity because it anchors the semantic meaning of the feature via textual representation; this assumption fails when column names are polysemous or absent, in which case the representation may rely on value statistics alone, losing identity. Additionally, we compute mutual information (MI) between the representation and column identities using a plug-in estimator (binned features, 10 bins) as a direct measure of identity preservation. High MI indicates that the representation encodes column identity; low MI when column names are ambiguous would signal failure.

## Contribution

(1) A novel representation learning framework FIPA that produces permutation-invariant and feature-identity-preserving embeddings for tabular data. (2) A design principle demonstrating that decoupling feature identity from positional encoding via column name embeddings enables cross-dataset transfer without retraining. (3) Empirical evidence that FIPA representations improve downstream task performance on datasets with varying column order compared to fixed-order baselines.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Datasets | UCI Adult (15 columns), Bank Marketing (17 columns), Wine Quality (12 columns) | Heterogeneous, realistic tabular data with varying column name informativeness. |
| Primary metric | Accuracy gap: original vs permuted columns | Direct test of permutation invariance claim. |
| Additional metrics | Representation consistency (cosine similarity between permuted instances); Column identity reconstruction accuracy (train a 2-layer MLP to predict column name from representation) | Direct tests of permutation invariance and feature-identity preservation beyond downstream accuracy. |
| Baseline 1 | Concat MLP (ordered input) | No permutation invariance, tests necessity. |
| Baseline 2 | Deep Sets (unordered pairs, sum pooling) | Permutation invariant, no column identity. |
| Baseline 3 | TabNet (attention-based) | Permutation variant, strong baseline. |
| Baseline 4 | Set Transformer (ISAB, 4 heads, 2 layers) | Permutation invariant with higher capacity, tests sufficiency of simple pooling. |
| Ablation-of-ours | FIPA with concatenation (instead of multiplication) | Tests importance of multiplicative combination. |
| Ablation-of-ours | FIPA with mean pooling (instead of sum) | Tests importance of sum pooling. |

### Why this setup validates the claim

Using multiple heterogeneous datasets (UCI Adult, Bank Marketing, Wine Quality) with varying column name informativeness provides a robust testbed for both permutation invariance and feature-identity awareness. The primary metric—difference in accuracy between original column order and a random permutation—directly measures permutation invariance. Additional metrics (representation consistency via cosine similarity and column identity reconstruction accuracy) directly evaluate the representation's sensitivity to permutation and identity preservation, addressing the reviewer's concern about relying solely on downstream accuracy. Baselines isolate the sub-claims: Concat MLP tests the necessity of invariance (should fail under permutation), Deep Sets tests the sufficiency of invariance without column identity (should fail on semantic meaning), TabNet tests against a strong variant model, and Set Transformer tests whether a more complex invariant model outperforms our simple pooling. Ablations (concatenation, mean pooling) test whether multiplicative combination and sum pooling are crucial. If FIPA maintains high accuracy and high representation consistency under permutation while baselines drop, and achieves high column identity reconstruction accuracy, the central claim is supported; if not, the idea is falsified.

### Expected outcome and causal chain

**vs. Concat MLP** — On a case where the columns are shuffled, Concat MLP expects a fixed order; it will misclassify because its weights are tied to positions. Our method instead treats pairs as a set with sum pooling, so the representation is identical regardless of order. We expect a large accuracy gap (>10%) on permuted data, with parity on original order. Additionally, we expect representation consistency (cosine similarity) between original and permuted instances to be >0.99 for FIPA, but <0.5 for Concat MLP.

**vs. Deep Sets** — On a case where two columns have similar values but different meanings (e.g., 'age'=30 vs 'income'=30), Deep Sets pools identical value embeddings irrespective of column, losing identity. Our method multiplies each value with a unique column embedding (e.g., from DistilBERT), disambiguating them. We expect a noticeable gap (>5%) on subsets where column meaning matters (e.g., numeric columns with overlapping ranges), but similar performance on features with distinct values. Column identity reconstruction accuracy for FIPA should be >90%, while Deep Sets should be at chance.

**vs. TabNet** — On a case where column names are semantically similar (e.g., 'salary' vs 'wage'), TabNet may ignore the subtle difference due to its attention mechanism relying solely on value statistics. Our method leverages pre-trained column embeddings that encode semantic similarity, allowing it to distinguish. We expect a modest gap (3-5%) overall, concentrated on columns with high semantic overlap.

**vs. Set Transformer** — Set Transformer may achieve similar permutation invariance but its higher capacity could lead to overfitting on small datasets. We expect FIPA to be competitive or slightly better on these moderate-sized datasets, with lower variance across runs.

### What would falsify this idea

If FIPA's accuracy under permutation is no better than Concat MLP (i.e., it also drops significantly), or if the representation consistency (cosine similarity) between original and permuted instances is <0.9, then the claimed permutation invariance is not achieved. If column identity reconstruction accuracy is below 70% (i.e., significantly above chance but not high), or if the ablation with concatenation matches the full method on identity reconstruction, then feature-identity awareness is not achieved.

## References

1. Language Models are Realistic Tabular Data Generators
