# Consistency-Regularized Sparse Autoencoders for Human-Aligned Concept Discovery

## Motivation

Current sparse autoencoders (SAEs) assume that a linear decoder with sparsity yields interpretable features, but this assumption remains unvalidated—automated interpretability metrics do not guarantee alignment with human concepts. For instance, Bricken et al. (2023) evaluate features via automated scores without verifying human agreement, while Marks et al. (2024) improve optimization via whitening but do not address concept alignment. The structural gap is that no training constraint explicitly ties SAE features to human-interpretable semantic invariants, leaving a disconnect between computational efficiency and human intuition.

## Key Insight

If sparse feature activations are invariant under semantic-preserving transformations and decoder weights are topologically aligned with a pre-trained concept manifold, the resulting features necessarily correspond to human-interpretable concepts without requiring concept labels during training.

## Method

### Method: Consistency-Regularized Sparse Autoencoder (CR-SAE)

**(A) What it is:** CR-SAE extends standard SAE training with two novel regularizers: a consistency loss that enforces feature invariance under semantic-preserving transformations, and a topological alignment loss that matches decoder columns to human-concept vectors via optimal transport. Input: model activations x; Output: sparse features z and decoder weights aligned with human concepts.

**(B) How it works:**

```python
import torch
import torch.nn as nn
from sinkhorn import SinkhornDistance  # differentiable OT

def cr_sae_loss(x, encoder, decoder, T, concept_vectors, alpha, beta, lambda_sparse):
    # Standard SAE forward
    z = encoder(x)
    x_hat = decoder(z)
    recon_loss = torch.mean((x - x_hat)**2)
    sparsity_loss = lambda_sparse * torch.mean(torch.norm(z, p=1, dim=1))

    # Consistency loss: apply random semantic transformation
    t = random_choice(T)
    x_prime = t(x)
    z_prime = encoder(x_prime)
    cons_loss = torch.mean((z - z_prime)**2)  # L2 invariance

    # Topological alignment loss: Sinkhorn distance between decoder weights and concept vectors
    # decoder.W shape: (input_dim, num_features), concept_vectors shape: (num_concepts, embed_dim)
    sinkhorn = SinkhornDistance(eps=0.1, max_iter=100)
    cost_matrix = torch.cdist(decoder.W.T, concept_vectors, p=2)  # distances
    align_loss = sinkhorn(cost_matrix)  # Sinkhorn divergence

    total_loss = recon_loss + sparsity_loss + alpha * cons_loss + beta * align_loss
    return total_loss, {'recon': recon_loss, 'sparsity': sparsity_loss, 'cons': cons_loss, 'align': align_loss}
```

**Hyperparameters:**
- alpha: weight for consistency loss (default 0.1)
- beta: weight for alignment loss (default 0.05)
- lambda_sparse: sparsity penalty (default 1e-5)
- T: set of semantic-preserving transformations (e.g., synonym replacement for text, color jitter for images)
  - We pre-select T by filtering transformations that alter sentiment more than 0.1 on a held-out set of 200 examples, ensuring they preserve human-interpretable concepts.
- concept_vectors: 1000 concept embeddings from a pre-trained model (e.g., sentence-BERT)

**(C) Why this design:**
We chose an L2 consistency loss over contrastive alternatives because it directly penalizes activation drift without requiring negative pairs, accepting the cost that it may be less robust to low-frequency transformations. We selected Sinkhorn alignment over cosine similarity maximization because Sinkhorn enforces a one-to-one matching between decoder columns and concept vectors, preventing all features from collapsing to a single concept; the trade-off is increased computational cost (O(n^3) per batch) and a dependency on the temperature parameter. We used a pre-trained embedding model to define concept vectors rather than manual annotation because it scales to hundreds of concepts and ensures semantic richness, at the cost of potential biases inherited from the embedding space. Finally, we set the concept set size to 1000 as a balance between coverage and optimization difficulty, knowing that a larger set may overconstrain the decoder.

**(D) Why it measures what we claim:**
`cons_loss` measures invariance under semantic-preserving transformations because it quantifies feature activation change when the input's semantic content is held constant; the assumption is that T preserves human-meaningful concepts, and this fails when a transformation inadvertently alters the concept (e.g., synonym changing sentiment), in which case invariance would be penalized despite concept change. To mitigate this, we pre-validate T on a held-out set (sentiment change <0.1). `align_loss` measures alignment with human concepts because it computes the optimal transport cost between decoder directions and pre-trained concept vectors, directly minimizing the distance to known human-interpretable directions; the assumption is the concept vectors are a representative sample of human concepts, and this fails when the embedding space omits or distorts certain concepts, in which case align_loss biases features toward the available vectors. We additionally validate feature-concept correspondence via human raters: we sample 100 concept-feature pairs and have 3 annotators rate the alignment on a 1-5 Likert scale; acceptable agreement is Cohen's κ ≥0.6. Together, these losses close the chain: invariance ensures features track consistent semantics, and alignment ensures those semantics match human-defined categories, achieving human alignment without labels.

## Contribution

(1) A consistency regularization loss for sparse autoencoders that enforces feature invariance under semantic-preserving transformations, improving feature robustness and interpretability. (2) A topological alignment method using optimal transport (Sinkhorn distance) to force decoder columns to match human-concept directions without requiring labeled concept data during training. (3) Empirical validation on language model activations showing that CR-SAE features achieve higher human agreement scores than standard SAEs and whitened SAEs on interpretability benchmarks.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|----------------------|
| Dataset | GPT-2 activations on Wikipedia sentences; also ResNet-18 on ImageNet and Whisper on LibriSpeech | Standard SAE testbed; enables semantic transformations; demonstrate domain generality |
| Primary metric | Feature disentanglement (DCI score) | Measures feature independence and concept alignment |
| Baseline 1 | Standard SAE (L1 sparsity) | Baseline without any regularizers |
| Baseline 2 | Whitened SAE (input whitening) | Tests preprocessing vs. our regularizers |
| Baseline 3 | Cosine-aligned SAE (cosine loss) | Ablates optimal transport for alignment |
| Baseline 4 | Random-aligned SAE (Sinkhorn with random concept vectors) | Tests if OT provides unique benefit beyond any alignment |
| Ablation 1 | CR-SAE without alignment loss | Isolates contribution of topological alignment |
| Ablation 2 | CR-SAE with varying concept embedding model (SBERT, CLIP, GloVe) | Sensitivity analysis on concept vectors |
| Ablation 3 | CR-SAE with varying T size (10, 50, 200 transformations) | Sensitivity analysis on transformation set |

### Why this setup validates the claim
This experimental design provides a falsifiable test of whether consistency and alignment regularizers improve feature interpretability over existing SAE methods. The standard SAE baseline isolates the effect of adding L2 invariance and Sinkhorn alignment. The whitened SAE baseline tests whether our method outperforms a simple preprocessing fix for feature quality. The cosine-aligned baseline tests whether optimal transport is superior to naive cosine alignment. The random-aligned baseline verifies that the specific OT constraint provides unique benefit beyond any alignment. The ablations of alignment loss and varying concept vectors/T measure the contribution and robustness of each component. The DCI disentanglement metric directly quantifies feature independence and human-concept alignment, which are the explicit goals of the regularizers. If our method shows a clear improvement over all baselines and the ablation, the claim is supported.

### Expected outcome and causal chain

**vs. Standard SAE** — On a sentence where a synonym replacement preserves meaning (e.g., "car" → "automobile"), the standard SAE may produce different feature activations because its features are not trained to be invariant under semantic-preserving changes. Our method enforces consistency loss, forcing the encoder to output similar sparse codes for both versions. Therefore, we expect our method to show higher DCI stability scores (e.g., 10% lower variance in features under transformations) and better overall disentanglement, with the gap most pronounced on examples involving frequent synonyms or augmentations.

**vs. Whitened SAE** — On a concept like "animal" that is not linearly separable by whitened inputs, whitened SAE's decoder columns may not align with human-interpretable directions. Our topological alignment loss directly matches decoder weights to concept vectors (e.g., from sentence-BERT), so the features become aligned to those concepts. We expect our method to achieve higher interpretability scores (e.g., DCI > 0.7) on held-out concept-related probes, while whitened SAE may saturate at a lower level due to missing alignment.

**vs. Cosine-aligned SAE** — When there are many overlapping concepts (e.g., "car" and "vehicle"), cosine alignment can cause multiple columns to collapse to a single concept vector, reducing feature diversity. Our Sinkhorn alignment enforces a soft one-to-one matching between decoder columns and concept vectors, encouraging each column to specialize on a distinct concept. We expect our method to produce more unique and separable features, reflected in a higher number of activated features per input (e.g., 30% more unique features) and better clustering of features by concept.

**vs. Random-aligned SAE** — Random concept vectors provide no meaningful alignment; the Sinkhorn loss with random targets should not improve DCI over standard SAE. Our method with proper concept vectors should significantly outperform, isolating the benefit of correct alignment.

### What would falsify this idea
If the improvements over baselines are uniform across all test subsets (e.g., equally large gains on both transformed and untransformed inputs), rather than being concentrated in cases where invariance or alignment should matter (e.g., under transformations for consistency, or on rare concepts for alignment), then the regularizers are not causally responsible and the central claim is falsified.

## References

1. Data Whitening Improves Sparse Autoencoder Learning
2. Sparse Autoencoders Find Highly Interpretable Features in Language Models
3. Scaling and evaluating sparse autoencoders
4. Training Compute-Optimal Large Language Models
