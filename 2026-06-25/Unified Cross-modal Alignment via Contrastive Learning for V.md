# Unified Cross-modal Alignment via Contrastive Learning for Visual Diagrams and Code

## Motivation

Existing methods for multimodal code intelligence, as cataloged in the survey 'Beyond NL2Code: A Structured Survey of Multimodal Code Intelligence', treat each code role (e.g., NL2Code, code2GUI) independently, lacking a structural integration of cross-modal representations. This per-task design prevents generalization across diverse modalities and code roles because no shared embedding space exists. The root cause is that no prior work establishes a unified alignment principle that exploits the underlying semantic isomorphism between visual diagrams and code.

## Key Insight

Visual diagrams and code share a fundamental structural isomorphism in control flow and data dependencies, which contrastive learning can exploit by pulling corresponding pairs together while pushing apart non-corresponding ones, creating a modality-invariant embedding space.

## Method

### (A) What it is – We propose **UCAL** (Unified Cross-modal Alignment via Contrastive Learning), a framework that learns a shared embedding space for visual diagrams and code using a dual-encoder architecture trained with a supervised contrastive loss. Inputs: a batch of paired (diagram, code) examples from diverse domains (GUI, scientific visualization, structured graphics, frontiertasks). Outputs: aligned embeddings such that cosine similarity reflects semantic correspondence.

### (B) How it works – Pseudocode:
```python
import torch
import torch.nn.functional as F

# Encoders: DiagramEncoder (ViT-B/16), CodeEncoder (CodeBERT-base)
# Projection: 2-layer MLP (512->512->128) with ReLU

def train_step(batch):
    diagrams, codes = batch  # each of size N (batch size >= 256)
    v = diagram_encoder(diagrams)           # [N, d_model]
    t = code_encoder(codes)                 # [N, d_model]
    p_v = projection(v)                     # [N, 128]
    p_t = projection(t)                     # [N, 128]
    # L2-normalize
    p_v = F.normalize(p_v, dim=-1)
    p_t = F.normalize(p_t, dim=-1)
    # compute similarity matrix
    sim = torch.mm(p_v, p_t.T) / tau        # tau=0.07
    # contrastive loss (InfoNCE)
    labels = torch.arange(N).to(device)
    loss_v = F.cross_entropy(sim, labels)
    loss_t = F.cross_entropy(sim.T, labels)
    loss = (loss_v + loss_t) / 2
    return loss
```

### (C) Why this design – We chose a dual-encoder architecture over a joint encoder because it allows independent inference from diagrams or code, enabling retrieval and zero-shot tasks without paired input at test time; the cost is losing explicit cross-modal interactions during encoding. We adopt InfoNCE contrastive loss rather than triplet loss because it naturally handles many negatives per batch and yields more discriminative embeddings, though it requires large batch sizes (256+). The projection network with L2-normalization is used to map into a hypersphere, which improves training stability and tolerance to modality-specific scales; the downside is potential information loss from the original feature space. Temperature τ=0.07 balances uniformity and tolerance: lower τ pushes harder on hard negatives but can cause collapse if set too low. Unlike prior works that treat each code role separately (e.g., NL2Code-specific encoders), UCAL unifies all roles by training on diverse paired data, accepting that in-domain bias may arise if some roles dominate the batch.

### (D) Why it measures what we claim – **Load-bearing assumption:** We assume that every paired (diagram, code) example in the training data is fully semantically equivalent – the diagram and code describe the same computation without missing or extra details. Cosine similarity between p_v and p_t measures **cross-modal alignment** because, under the contrastive training objective, the embeddings of corresponding pairs are pulled together while non-corresponding pairs are pushed apart. This operationalizes the concept of semantic equivalence: the assumption is that paired diagrams and code are ground-truth semantically equivalent, so high similarity indicates successful alignment. This assumption fails when the pair is imperfectly aligned (e.g., a high-level diagram missing details present in code, or code containing implementation artifacts not in the diagram); in such cases, similarity may reflect partial or surface-level correspondence rather than full semantic equivalence. The temperature τ controls the sharpness of the similarity distribution and thus the tolerance to imperfect alignment: lower τ enforces harder alignment, risking false negatives for partially aligned pairs. The dual projection ensures both modalities are mapped to the same metric space, so that any deviation in similarity from the diagonal (labels) indicates misalignment; the loss quantifies this deviation as a single scalar, optimization of which drives the embeddings toward the claimed property.

## Contribution

(1) A novel unified framework (UCAL) that learns a shared embedding space for visual diagrams and code across multiple code roles using contrastive learning, overcoming the per-task fragmentation identified in prior surveys. (2) An empirical demonstration that contrastive alignment generalizes across diverse modalities (GUI, scientific visualization, structured graphics) and code roles (NL2Code, code2GUI, code2scientific, etc.), enabling zero-shot transfer and retrieval. (3) A curated benchmark suite of paired diagram-code data from multiple domains, facilitating reproducible evaluation of cross-modal alignment.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | Multimodal CodePair | Diverse diagram-code pairs |
| Primary metric | Recall@1 (diag→code) | Direct alignment retrieval |
| Baseline 1 | Unimodal code-only | Tests cross-modal necessity |
| Baseline 2 | Joint encoder (dual-attn) | Tests dual-encoder efficiency |
| Baseline 3 | CLIP (general-domain) | Tests domain adaptation |
| Ablation-of-ours | Ours w/o projection head | Tests projection importance |
| Robustness to imperfect alignment | Our method on noisy pairs | Tests load-bearing assumption: we create synthetic noise by masking 20% of diagram regions or adding irrelevant code lines to test robustness to partial semantic equivalence. |

### Why this setup validates the claim

This combination forms a falsifiable test of the central claim that UCAL learns a shared embedding space for cross-modal alignment. The unimodal baseline quantifies the benefit of incorporating visual information: if UCAL outperforms it, cross-modal alignment is necessary. The joint encoder baseline isolates the impact of architectural choice: if UCAL matches or exceeds it while being faster, the dual-encoder is sufficient. The general-domain CLIP baseline tests the value of domain-specific fine-tuning: if UCAL outperforms, the contrastive objective on diverse code-diagram pairs yields better alignment. The ablation of the projection head tests the design choice of normalized embeddings. The robustness test directly probes the load-bearing assumption that paired data is fully semantically equivalent: if performance drops significantly under noise, the assumption is critical; if not, the method may be more resilient than expected, which would require further analysis to understand why. The primary metric, Recall@1, directly measures whether the correct (diagram, code) pair has the highest similarity in the learned space, operationalizing alignment. Any observed improvement must be attributable to the proposed mechanisms, and the specific failure modes of each baseline provide diagnostic signals.

### Expected outcome and causal chain

**vs. Unimodal code-only retrieval** — On a case where two different diagrams (e.g., a login page vs a settings page) correspond to similar code structures (e.g., both use form elements), the unimodal baseline incorrectly retrieves the wrong code for both diagrams because it only relies on code text similarity, ignoring visual layout differences. Our method instead uses the diagram embedding to disambiguate, because the contrastive loss has learned to map distinct visual patterns to distinct code regions. Thus we expect Recall@1 to be at least 10 percentage points higher for our method, particularly on examples with high visual variability but low textual variability.

**vs. Joint encoder (dual-attention)** — On a case where we need to retrieve code from a diagram at test time without access to the code (e.g., zero-shot retrieval), the joint encoder cannot be used because it requires both modalities during encoding. Our method performs independent encoding of the diagram, enabling retrieval through nearest-neighbor search in the shared space. Thus we expect our method to achieve comparable Recall@1 (within 2%) while being 2x faster at inference, and to enable applications where the joint encoder is infeasible.

**vs. CLIP (general-domain)** — On a case involving a domain-specific diagram (e.g., a circuit schematic with specialized symbols), CLIP's general image embeddings do not align with code syntax because it was trained on natural images and captions. Our method, fine-tuned on paired code-diagram data, learns to associate those symbols with corresponding code tokens. Thus we expect Recall@1 to be at least 15 percentage points higher for our method on such domain-specific subsets.

**vs. Ours w/o projection head** — On a case where the original encoder outputs are at very different scales (e.g., deep vs shallow layers), the raw embeddings yield noisy similarities that hinder retrieval. The projection head with L2-normalization stabilizes the metric space. Thus we expect Recall@1 to drop by about 5 percent when the projection is removed.

**vs. Perfect alignment assumption** — On a case where the diagram omits details present in code (e.g., a high-level GUI diagram without event handlers), our method expects the pairs to be fully equivalent, so the loss will force an incorrect mapping. We simulate this by randomly masking 20% of the diagram regions or adding irrelevant code lines to the test set. We expect Recall@1 to drop from, say, 80% to 60%, indicating sensitivity to partial alignment. If the drop is less than 10% (e.g., 80% to 75%), this would suggest the model is more robust to imperfect alignment than assumed, which would be surprising and require further investigation.

### What would falsify this idea

If our method fails to significantly outperform the unimodal baseline (less than 5% improvement in Recall@1), or if the improvement is uniform across all data subsets rather than concentrated on cases with high visual variation, then the central claim that cross-modal alignment via contrastive learning is the driving factor would be falsified. Similarly, if removing the projection head yields no degradation, the design rationale for the projection would be undermined. Additionally, if the robustness test shows a drop of less than 5% under synthetic noise, the load-bearing assumption would be less critical, and the method's reliance on perfect alignment may not be as central as claimed.

## References

1. Beyond NL2Code: A Structured Survey of Multimodal Code Intelligence
