# Domain-Invariant Sparse Phonological Features for Cross-Domain Phone Segmentation and Recognition

## Motivation

Self-supervised speech models like wav2vec 2.0 provide strong phonetic representations, but their frozen features are not domain-invariant: SPAM's phonological activation mapping (Phone Segmentation and Recognition through Phonological Activation Mapping) and MixGoP (Leveraging Allophony in Self-Supervised Speech Models) both rely on these frozen features, leading to performance degradation on accented or atypical speech. The root cause is that the SSL features entangle phonetic content with domain-specific acoustic properties, and no mechanism exists to suppress the domain variation without fine-tuning the backbone. This structural limitation across these methods calls for a representation that is both phonologically grounded and explicitly regularized for domain invariance.

## Key Insight

Sparse coding over a learned dictionary of phonological features, combined with adversarial domain invariance, forces the model to discard domain-specific acoustic variation while retaining linguistically universal phonetic content.

## Method

### (A) What it is
We propose **DISPOFE** (Domain-Invariant Sparse Phonological Feature Extractor), which extracts a sparse coefficient vector from frozen SSL features via a learned dictionary and enforces cross-domain consistency of those coefficients via adversarial training and a cross-domain dictionary consistency loss. The sparse coefficients are then used by lightweight linear heads for phone segmentation and recognition.

### (B) How it works (Pseudocode)

```python
# Input: SSL feature vector x ∈ R^d (d=768, wav2vec 2.0)
# Output: sparse coefficient c ∈ R^K (K=50), domain-invariant

# Learned dictionary D ∈ R^{d x K} (initialized via sparse autoencoder on source)
# Domain critic C: R^K → [0,1] (binary classifier, source vs target)
# Segmentation head S: R^K → R (phone boundary probability per frame)
# Recognition head R: R^K → R^{P} (P=phone classes)

# Training loop:
for batch in mixed source + target:
    x_s, x_t = batch
    # Sparse coding with L1 regularization (differentiable via ISTA)
    c_s = iterative_shrinkage_thresholding(D, x_s, lambd=0.1)  # 10 iterations
    c_t = iterative_shrinkage_thresholding(D, x_t, lambd=0.1)

    # Domain adversarial loss (gradient reversal layer)
    domain_loss = BCE(C(GRL(c_s)), 0) + BCE(C(GRL(c_t)), 1)

    # Cross-domain dictionary consistency loss (cycle-consistency)
    # For each target frame, find nearest source neighbor in coefficient space, enforce similarity
    nearest_s_indices = nearest_neighbor(c_t, c_s)  # find index in source batch
    c_s_nearest = c_s[nearest_s_indices]
    consistency_loss = MSE(c_t, c_s_nearest)

    # Reconstruction loss to keep dictionary faithful to SSL features
    recon_loss = MSE(x_s, D @ c_s) + MSE(x_t, D @ c_t)

    # Segmentation and recognition on source only (supervised)
    if source has labels:
        seg_loss = BCE(S(c_s), boundary_labels)
        rec_loss = CrossEntropy(R(c_s), phone_labels)
        total_loss = seg_loss + rec_loss + alpha*domain_loss + beta*recon_loss + gamma*consistency_loss
    else:
        total_loss = alpha*domain_loss + beta*recon_loss + gamma*consistency_loss

    update D, C, S, R
```

Hyperparameters: lambd=0.1 (sparsity), alpha=1.0 (domain loss weight), beta=0.5 (reconstruction weight), gamma=0.2 (consistency weight), ISTA iterations=10.

### (C) Why this design
We made three key design decisions. First, we chose a learned dictionary with L1-sparse coding over predefined phonological feature vectors (e.g., IPA binary features). The learned dictionary adapts to the SSL embedding space, capturing complex acoustic-to-phonological mappings, at the cost of losing direct interpretability of each atom as a single phonological feature. However, the sparsity constraint (lambd=0.1) ensures each frame uses only a few atoms, encouraging a compact phonological code. Second, we used adversarial domain training with a gradient reversal layer rather than moment matching (e.g., MMD). Adversarial training directly forces the distribution of sparse coefficients to be indistinguishable across domains, which is a stronger condition for invariance than matching first two moments; the trade-off is increased training instability (mitigated by using a small learning rate for the domain critic). Third, we retained a reconstruction loss (MSE between input SSL features and D @ c) to prevent the sparse coefficients from discarding phonetic information that is not captured by the dictionary. Without it, the adversarial loss might drive the coefficients to a trivial constant that is domain-invariant but contains no phonetic content. This design adds a supervised reconstruction burden but stabilizes the learned representation. Additionally, we introduced a cross-domain dictionary consistency loss (cycle-consistency) to explicitly encourage that the same phone activates similar dictionary atoms across domains, addressing the concern that dictionary atoms may be domain-specific.

### (D) Why it measures what we claim
In the method, the sparse coefficient vector `c` measures **phonological feature activations** under the assumption that each dictionary atom corresponds to a distinct phonetic/phonological property (e.g., [+voice], [+high]); this assumption fails if the dictionary learns correlated or non-phonological patterns (e.g., energy band), in which case `c` measures acoustic artifacts. The domain classifier's loss measures **domain invariance** of the sparse coefficients under the assumption that all domain-specific information is captured in the coefficients (not in dictionary or residual); this assumption fails if domain information leaks through the dictionary (e.g., different atoms activated for same phone across domains) or through the residual reconstruction error, in which case the loss only measures invariance of the coefficients themselves, not the full representation. The reconstruction loss ensures that the dictionary remains faithful to the SSL features, reducing the risk of collapse. The cross-domain consistency loss measures **domain-consistent atom usage** under the assumption that nearest-neighbor matching in coefficient space aligns phones across domains; this assumption fails if the coefficient space is not well-aligned and nearest neighbors do not correspond to the same phone, in which case the loss enforces similarity between different phones. Thus, the overall training minimizes a bound on the domain-specific information in the sparse coefficients, under the joint assumption that the dictionary spans the phonetic space, that residual information is not domain-discriminative, and that nearest-neighbor pairs are phonetically matched.

## Contribution

(1) A novel framework (DISPOFE) that combines sparse coding with adversarial domain invariance to extract phonological features from frozen SSL representations, enabling cross-domain phone segmentation and recognition without fine-tuning the backbone. (2) Empirical evidence that enforcing cross-domain consistency of sparse coefficients improves phone segmentation and recognition performance on unseen domains (e.g., accented English) compared to using raw SSL features or SPAM's phonological activations. (3) Analysis showing that learned sparse phonological features exhibit lower domain-specific variability (measured via domain classifier accuracy) than original SSL features, confirming the effectiveness of the domain-invariance regularization.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|----------|
| Dataset | TIMIT (source) + L2-ARCTIC (target) | Cross-domain phone segmentation |
| Primary metric | Phone boundary F1 score | Directly measures segmentation accuracy |
| Baseline 1 | wav2vec2-linear | Tests need for sparse coding |
| Baseline 2 | DISPOFE w/o domain adv | Tests domain invariance |
| Baseline 3 | IPA binary features | Tests learned dictionary superiority |
| Baseline 4 | DISPOFE w/o consistency loss | Tests consistency loss role |
| Ablation | DISPOFE w/o recon loss | Tests reconstruction role |
| Interpretability metric | Atom-phone correlation (APC) | Measures correspondence between dictionary atoms and known phonological features (e.g., [+voice], [+high]) |

### Why this setup validates the claim
This design evaluates whether sparse, domain-invariant coefficients from a learned dictionary improve phone segmentation across domains. TIMIT provides clean source supervision, L2-ARCTIC tests realistic cross-domain generalization. Comparing against wav2vec2-linear isolates the benefit of sparse coding; removing domain adversarial training measures invariance contribution; replacing with IPA features tests dictionary learning; ablating reconstruction loss checks if it prevents degenerate representations; adding a consistency loss ablation tests its added value. F1 score is standard for segmentation and sensitive to false positives/negatives, thus suitable for detecting improvements from better phonetic boundaries. The atom-phone correlation metric directly validates the assumption that dictionary atoms encode phonological features—if APC is high, the representation is interpretable and phonologically grounded.

### Expected outcome and causal chain

**vs. wav2vec2-linear** — On a cross-domain case (e.g., L2-ARCTIC speaker producing unusual formant transitions), wav2vec2-linear produces spurious boundary peaks because its dense features encode domain-specific acoustic patterns. Our DISPOFE instead activates only a few dictionary atoms (due to L1 sparsity) that correspond to coarse phonetic events, filtering out domain noise. Thus we expect a noticeable F1 gain on target domain (e.g., +5–10 points) but parity on source where both learn well.

**vs. DISPOFE w/o domain adv** — On a case where source and target differ in vowel space (e.g., /i/ formants shifted), the non-adversarial version learns source-biased dictionary atoms, leading to missed boundaries for target /i/ due to mismatched activation patterns. DISPOFE with domain adv forces atom activations to be indistinguishable across domains, so the same atoms fire for /i/ regardless of formant variation. We expect minimal degradation on source but substantial improvement on target (e.g., +8–12 F1 points).

**vs. IPA binary features** — On a case with coarticulated /nt/ where binary IPA features (e.g., [+nasal, +alveolar]) fail to capture the temporal blurring, the fixed IPA features cause boundary deletions. DISPOFE's learned dictionary adapts to SSL representation, allowing atoms to encode context-dependent spectral transitions, thus detecting the boundary between /n/ and /t/. We expect higher recall on targets (e.g., +6–8 F1 points) with similar precision.

**vs. DISPOFE w/o consistency loss** — On a case where the same phone (e.g., /s/) has different spectral realizations across domains, DISPOFE without consistency loss may activate different atoms for source and target /s/, causing recognition errors despite domain-invariant distribution. The consistency loss explicitly aligns atom usage for matched phones, reducing within-class variance. We expect an additional improvement of +2–4 F1 points on target.

**Atom-phone correlation** — We expect average absolute Pearson correlation of at least 0.4 between top-activated atoms and known binary phonological features (e.g., [+voice]), confirming interpretability.

### What would falsify this idea
If DISPOFE's domain-adversarial version shows no larger gain on target over the non-adversarial ablation, or if the gain is uniform across all phonetic contexts rather than concentrated on cross-domain mismatched ones, then the central claim of domain invariance is unsupported. Additionally, if atom-phone correlations are below 0.2, the assumption that dictionary atoms correspond to phonological features would be invalidated, undermining the claimed phonological grounding.

## References

1. Phone Segmentation and Recognition through Phonological Activation Mapping
2. Leveraging Allophony in Self-Supervised Speech Models for Atypical Pronunciation Assessment
3. A Simple HMM with Self-Supervised Representations for Phone Segmentation
4. Learning Dependencies of Discrete Speech Representations with Neural Hidden Markov Models
5. Homophone Disambiguation Reveals Patterns of Context Mixing in Speech Transformers
6. A Reality Check and a Practical Baseline for Semantic Speech Embedding
