# Streaming Orthonormal Anchor Induction via Online Randomized SVD for Domain-Agnostic Multimodal Data Curation

## Motivation

DataClaw0's reliance on five predefined Factual Anchors structurally limits its applicability to novel, unseen domains because it assumes that the essential categories for data curation must be manually specified a priori. This assumption recurs across multiple curation pipelines (e.g., those using fixed domain ontologies or task-specific taxonomies), creating a bottleneck for streaming multimodal data where the domain distribution evolves over time. The root cause is the lack of a mechanism to discover emergent anchor structures directly from the data stream without human ontology engineering.

## Key Insight

The cross-modal covariance matrix between two modalities, computed over a sliding window of the data stream, reveals latent orthogonal signal directions via its principal eigenvectors, and these directions automatically form a set of anchors that maximally preserve shared information across modalities, with the number of anchors determined by the rank of the matrix estimated through eigenvalue-gap thresholding.

## Method

**(A) What it is:** We propose *Streaming Orthonormal Anchor Induction* (SOAI), an online algorithm that takes a multimodal data stream (e.g., image-text pairs) and outputs a dynamically growing set of orthonormal anchor vectors (one per anchor) and a curated subset of stream samples that are most relevant to the current anchors. The number of anchors is automatically determined by an adaptive eigenvalue-gap threshold on the cumulative cross-modal covariance matrix, estimated via streaming randomized SVD (rSVD).

**(B) How it works:**
```pseudocode
Input: Stream of multimodal pairs (x_t, y_t), t=1,2,... (x_t ∈ R^d_x, y_t ∈ R^d_y)
Parameters: window_size W, eigenvalue_gap_ratio θ, update_frequency F
Initialize: cumulative covariance matrix C = zeros(d_x, d_y), eigendecomposition not computed yet, anchors = []
For each new pair (x_t, y_t):
    # 1. Update running cross-modal covariance (online estimation)
    If t <= W: C = C + x_t * y_t^T   (outer product)
    Else: C = C - (x_{t-W} * y_{t-W}^T) + (x_t * y_t^T)   (sliding window)
    
    # 2. Every F steps, run streaming rSVD on C to get its truncated SVD U, S, V
    If t % F == 0:
        # Use an online randomized SVD algorithm (e.g., block Krylov method) to update
        # the top-k singular values and left/right singular vectors of C
        U_k, S_k, V_k = online_rSVD(C, k=adaptive_rank(C, θ))
        # Determine number of anchors via eigenvalue-gap thresholding:
        # Compute consecutive gaps in singular values: gap_i = s_i - s_{i+1}
        # Set k* = min{i | gap_i > θ * s_1}   (largest gap index)
        k_star = find_first_gap(S_k, θ)
        # Update anchors to top k_star left singular vectors
        anchors = columns of U_k[:, 1:k_star]
    
    # 3. Curate current sample: assign to nearest anchor if cosine similarity > threshold τ
    If len(anchors) > 0:
        sim = max_{a in anchors} (a^T x_t) / (||a|| * ||x_t||)   # since orthonormal, ||a||=1
        If sim > τ: keep sample for further processing
        Else: discard or store for future anchor growth
Output: growing set of anchors, curated stream
```

**(C) Why this design:** We chose an online sliding-window covariance over an infinite cumulative sum because it adapts to distribution shifts by forgetting outdated data, accepting the cost of a fixed memory budget and potential loss of long-term correlations. We use truncated randomized SVD instead of exact eigen decomposition because it reduces per-update complexity from O(d^3) to O(d^2 * k) for d = max(d_x, d_y), enabling streaming high-dimensional modalities; the trade-off is an approximation error that can be bounded by standard rSVD guarantees. We select anchors as left singular vectors (modality-agnostic, can be from either modality) rather than a fixed set of centroids because orthogonality ensures anchor diversity and prevents redundancy; the cost is that anchors may not correspond to semantically salient clusters if the cross-modal covariance is dominated by noise—mitigated by the eigenvalue-gap threshold which retains only high-signal directions. The eigenvalue-gap threshold θ replaces a fixed rank, allowing automatic adaptation to the effective dimensionality of shared signal; however, this threshold is sensitive to the noise floor and must be tuned per dataset or stream. We chose cosine similarity for sample assignment because it is computationally light and directly measures direction alignment with the anchor basis; a more complex distance would burden the streaming loop. τ is a hyperparameter controlling curation selectivity; if set too high, many samples are discarded and anchors may learn slowly; if too low, noise is kept. These design decisions together enable a fully data-driven, streaming anchor induction mechanism that avoids any predefined ontology.

**(D) Why it measures what we claim:** The singular values of the cross-modal covariance matrix C measure the strength of shared variance between modalities; each singular vector pair (left and right) defines a direction in each modality that co-varies maximally. Thus, the top k* singular vectors (anchors) measure the *most informative shared signal directions* because, under the assumption that the data is generated by a latent-factor model where each factor contributes equally across modalities, these directions are the maximal-information-projections that preserve the joint distribution. This assumption fails when one modality is dominated by independent noise (e.g., random text captions without visual grounding); in that case, the singular vectors may capture dominant but irrelevant correlations—then the metric reflects *dominant covariance* rather than *semantic information*. The rank k* from eigenvalue-gap thresholding measures the *intrinsic dimensionality of the shared signal* because it isolates the subspace where signal exceeds the noise floor; the threshold θ operationalizes the definition of “signal” as a relative gap above a noise baseline. This gap assumption fails when the eigenvalue spectrum decays smoothly without a clear gap (e.g., in high-noise regimes); then k* becomes sensitive to θ and may under- or over-estimate the true rank—the metric then reflects the *arbitrary cutoff* rather than ground truth dimensionality. The cosine similarity of a sample to the nearest anchor measures the *sample's relevance to the current shared signal directions* because it quantifies how much of the sample's variation aligns with the high-covariance subspace; this assumes the sample representation is isotropic in the orthogonal complement, which fails when useful task-specific information lives in low-covariance dimensions—then the metric reflects *alignment with dominant shared variation*, potentially discarding valuable out-of-distribution samples. Thus, each computational component in (B) directly operationalizes a motivation-level concept, with clearly named assumptions and failure modes.

## Contribution

(1) A streaming, modality-agnostic anchor induction algorithm (SOAI) that automatically determines the number of anchors via online randomized SVD with adaptive eigenvalue-gap thresholding, eliminating the need for predefined category structures. (2) The principle that orthonormal basis vectors of the cumulative cross-modal covariance matrix serve as effective, dynamically evolving anchors for multimodal data curation, enabling domain-agnostic operation. (3) A formal characterization of the trade-off between anchor plasticity and stability through the sliding-window size and eigenvalue-gap threshold, providing design guidelines for streaming curation systems.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|----------------------|
| Dataset | Conceptual Captions streaming | Large-scale multimodal stream with noise. |
| Primary metric | Downstream retrieval Recall@1 | Measures quality of curated pairs. |
| Baseline1 | Random sampling | Simple baseline, no curation. |
| Baseline2 | CLIP-score thresholding | Common heuristic for relevance filtering. |
| Baseline3 | Fixed-frequency subsampling | Ignores content, uniform rate. |
| Ablation | SOAI w/o sliding window | Cumulative covariance, no forgetting. |

### Why this setup validates the claim

Conceptual Captions streaming provides a real-world multimodal stream with distribution shifts, mimicking the target scenario. The primary metric, Recall@1 on a downstream retrieval task, directly measures the informativeness of curated pairs. Baselines span no curation (random), heuristic (CLIP threshold), and naive subsampling (fixed frequency), each testing a different sub-claim: random tests necessity of any curation, CLIP tests superiority over static similarity, fixed frequency tests adaptation to varying density. The ablation (cumulative covariance) isolates the benefit of the sliding window for forgetting outdated correlations. Together, they form a falsifiable test: if SOIA outperforms baselines particularly on shifting segments, the adaptive mechanism is validated.

### Expected outcome and causal chain

**vs. Random sampling** — On a stream with periodic domain shifts (e.g., sudden topic change), random sampling retains many irrelevant pairs because it ignores content. Our method anchors onto dominant covariance directions that track the current signal, so it discards out-of-distribution samples. We expect a noticeable gap in retrieval accuracy on subsets after shift, but parity on stationary segments.

**vs. CLIP-score thresholding** — On a stream where captions are mismatched to images (e.g., noisy web data), CLIP threshold may keep visually simple but semantically misaligned pairs because it relies on static image-text similarity. Our method uses cross-modal covariance from the stream itself, adapting to statistical regularities. So we expect higher precision on hard examples where CLIP fails due to spurious correlations, leading to a larger gap on noisy subsets.

**vs. Fixed-frequency subsampling** — On a stream with varying information density (bursty events), fixed-frequency subsampling misses clusters of important samples because it is uniform. Our method adaptively selects samples aligned with current anchors, so it captures denser clusters. We expect our curated set to yield higher retrieval performance per sample, with the gap most pronounced during bursts.

**Ablation (w/o sliding window)** — On a stream with a permanent distribution shift, the cumulative covariance retains old signal directions even after they become irrelevant. Our full method forgets and adapts. We expect the ablation to perform worse after the shift, with its performance degrading over time as outdated anchors persist.

### What would falsify this idea

If SOAI's advantage is uniform across all data subsets rather than concentrated on segments where distribution shifts occur (e.g., similar gains on stationary regions), then the central claim of dynamic adaptation is unsupported. Specifically, if the ablation without sliding window matches the full method's performance, that would falsify the importance of forgetting.

## References

1. DataClaw0: Agentic Tailoring Multimodal Data from Raw Streams
2. Wan: Open and Advanced Large-Scale Video Generative Models
3. Visual Instruction Tuning
4. Scaling Instruction-Finetuned Language Models
5. OPT-IML: Scaling Language Model Instruction Meta Learning through the Lens of Generalization
6. Self-Instruct: Aligning Language Models with Self-Generated Instructions
