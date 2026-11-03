# Omni-modal Information Bottleneck with Mixture-of-Gaussians Prior and Compositional Identifiability

## Motivation

Existing information-bottleneck alignment methods assume a fixed two-modality pair and a unimodal Gaussian posterior, so they cannot guarantee identifiability of a shared latent space when a third or fourth modality is added. The survey 'From Fragmentation to Fusion' explicitly flags compositionality across arbitrary modality sets as an open problem, yet no current bottleneck formulation provides a modality-agnostic latent variable whose posterior remains identifiable under schema growth. This structural limitation—pairwise alignment plus unimodal Gaussian—means that adding modalities either requires ad-hoc retraining or breaks identifiability altogether.

## Key Insight

A shared latent variable with a mixture-of-Gaussians posterior, trained under an n-way information bottleneck, yields a compositional identifiability condition: each modality's encoder recovers the same latent up to a linear transformation, and this property is preserved when new modalities are added because the mixture components act as modality-agnostic anchors.

## Method

**What it is** — We introduce the Omni-modal Information Bottleneck (OMIB), a training objective and architecture that learns a shared latent variable Z from n modalities. Inputs are n modality-specific encoders producing representations; output is a single latent Z with a mixture-of-Gaussians posterior and n decoders that reconstruct each modality from Z. The objective is an n-way information bottleneck: maximize I(Z; {X_i}) while minimizing I(Z; X_i | Z_{-i}) for all i, implemented via a variational bound.

**How it works** — The core operation is:
```python
# Given n modalities x_1, ..., x_n
# Each encoder e_i(x_i) outputs parameters for a mixture of K Gaussians:
#   mu_{i,k}(x_i), Sigma_{i,k}(x_i), pi_{i,k}(x_i) for k=1..K
# Shared latent Z ~ sum_k pi_k N(mu_k, Sigma_k) where pi_k, mu_k, Sigma_k are
# obtained by a product-of-experts aggregation across modalities:
#   pi_k ∝ prod_i pi_{i,k}^{w_i},  mu_k = sum_i w_i * mu_{i,k},  1/Sigma_k = sum_i w_i / Sigma_{i,k}
# with learnable modality weights w_i (softmax-normalized).
#
# LOAD-BEARING ASSUMPTION (stated explicitly): The index-wise PoE above is exact
# only if component k in modality i corresponds to component k in modality j for
# all i,j — i.e., the mixture components are already aligned across modalities.
# Mixture models do not guarantee this (label switching), so we add an explicit
# component-alignment step BEFORE aggregation: compute a cross-modality assignment
# matrix A (K x K, doubly stochastic) via optimal transport (Sinkhorn, entropic
# regularization eps=0.05, 20 iterations) on the component means {mu_{i,k}} and
# covariances {Sigma_{i,k}}, then permute each modality's components to a canonical
# order (Hungarian algorithm on A) so that index k is shared by construction.
# The identifiability theorem is restated for this aligned mixture: if the true
# generative process is Z -> X_i with invertible linear maps and K >= true number
# of modes, then the aligned OMIB latent is identifiable up to a linear transform.

for each batch:
    # 1. Encode each modality
    for i in 1..n:
        pi_i, mu_i, Sigma_i = encoder_i(x_i)   # each is K-dimensional, K=8
    # 2. Align mixture components across modalities (Sinkhorn + Hungarian)
    A = sinkhorn({mu_i, Sigma_i}, eps=0.05, iters=20)   # K x K doubly stochastic
    for i in 1..n:
        pi_i, mu_i, Sigma_i = permute(pi_i, mu_i, Sigma_i, hungarian(A_i))
    # 3. Aggregate to shared posterior q(Z|x_1..x_n)
    pi, mu, Sigma = PoE_aggregate({pi_i, mu_i, Sigma_i}, weights=w)
    # 4. Sample Z ~ q(Z|x_1..x_n) via reparameterization
    Z = reparameterize(pi, mu, Sigma)
    # 5. Decode each modality from Z
    for i in 1..n:
        x_hat_i = decoder_i(Z)
    # 6. Compute n-way IB loss
    L_rec = sum_i E_q[ -log p(x_i|Z) ]          # reconstruction
    L_kl  = KL( q(Z|x_1..x_n) || p(Z) )         # prior matching, p(Z)=N(0,I)
    L_align = sum_i KL( q(Z|x_i) || q(Z|x_1..x_n) )  # pairwise alignment to shared
    L = L_rec + beta * L_kl + gamma * L_align
    # beta, gamma are annealing hyperparameters (beta: 0→1 over 10k steps, gamma: 0→0.1 over 5k steps); K=8 mixture components
```
The compositional identifiability theorem (proved via the linear identifiability of mixture-of-Gaussians posteriors under the IB objective, with the explicit component-alignment step above) states that if the true generative process is Z -> X_i for all i with invertible linear maps and K >= true number of modes, then the learned Z is identifiable up to a linear transformation, and this holds for any n >= 2, including when modalities are added incrementally.

**Why this design** — We chose a mixture-of-Gaussians posterior over a unimodal Gaussian because a single Gaussian cannot capture multimodal latent structure that arises when different modality subsets activate different latent factors, accepting the cost of K-times more parameters and a product-of-experts aggregation that can be numerically unstable for large K. We chose product-of-experts aggregation over a learned attention-based fusion because PoE yields a closed-form posterior that preserves the mixture structure and enables the compositional identifiability proof, accepting the cost that modality weights must be learned via a separate softmax and may underweight informative but noisy modalities. We added an explicit component-alignment step (Sinkhorn + Hungarian) because the index-wise PoE is only exact when components are aligned across modalities, and mixture models do not guarantee this (label switching); the alignment makes the shared-component assumption explicit and testable rather than assumed. We chose the n-way IB objective with an explicit alignment term over a simple contrastive loss because the IB bound provides a principled trade-off between sufficiency and minimality, accepting the cost of three hyperparameters (beta, gamma, K) that require annealing schedules. A domain expert would not describe this as a variant of the pairwise IB of 'Aligning multimodal representations through an information bottleneck' (2025) because that work fixes n=2 and uses a unimodal Gaussian, whereas our mixture-of-Gaussians and PoE aggregation produce a compositional identifiability property that the pairwise method cannot have.

**Why it measures what we claim** — The reconstruction loss L_rec measures sufficiency of Z for each modality because it is the negative log-likelihood under the decoder, assuming the decoder is expressive enough to invert the encoder; this assumption fails when a modality has high-frequency details that the latent bottleneck discards, in which case L_rec reflects only coarse-grained sufficiency. The KL term L_kl measures minimality of Z because it penalizes deviation from the prior, assuming the prior is matched to the true latent distribution; this assumption fails when the true latent is heavy-tailed, in which case L_kl reflects a biased minimality. The alignment term L_align measures consistency across modalities because it forces each modality's posterior to match the shared posterior, assuming that all modalities are generated from the same latent Z; this assumption fails when a modality is corrupted or missing, in which case L_align reflects robustness to noise rather than true consistency. The mixture components K measure the number of latent factors because each component corresponds to a mode of the latent distribution, assuming the true latent is a mixture of K modes; this assumption fails when the true latent has more than K modes, in which case the mixture underfits and identifiability is only approximate. The component-alignment step (Sinkhorn + Hungarian) measures cross-modality component correspondence because it matches mixture components by their means and covariances, assuming that aligned components have similar sufficient statistics; this assumption fails when two distinct latent modes have nearly identical means and covariances, in which case the alignment may permute them and the shared posterior mixes factors.

## Contribution

['A novel n-way information bottleneck objective (OMIB) that uses a mixture-of-Gaussians posterior with product-of-experts aggregation to learn a shared latent variable from an arbitrary number of modalities, replacing the fixed two-modality pair and unimodal Gaussian of prior IB alignment.', 'A compositional identifiability condition for the shared latent space under OMIB, showing that the latent is identifiable up to a linear transformation for any n >= 2 and that this property is preserved when modalities are added incrementally.', 'An empirical demonstration on multimodal datasets (e.g., audio-visual-text) that OMIB achieves comparable or better cross-modal retrieval than pairwise IB while enabling zero-shot addition of a new modality without retraining the encoders.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | Synthetic multimodal benchmark with known latent factors (pre-registered: n=2,3,4 modalities; true mode count M ∈ {1,2,4,8}; latent dim d=16; 10k train / 2k test samples per configuration) | Ground truth Z enables identifiability check |
| Dataset | Semi-real disentanglement benchmark: dSprites-derived multimodal set (image + coordinate + shape attributes) with controlled latent-factor annotation | Tests transfer beyond synthetic where generative process is known |
| Primary metric | Linear identifiability score (R² of Z vs. true Z via linear regression on held-out test set) | Directly tests the compositional identifiability claim |
| Secondary metric | Mode-recovery accuracy (Hungarian-matched accuracy of recovered mixture components vs. true modes) | Tests whether K components recover true latent modes |
| Baseline | Pairwise IB (n=2, unimodal Gaussian) | Tests whether mixture+PoE adds compositional identifiability |
| Baseline | Contrastive fusion (CLIP-style, InfoNCE τ=0.07) | Tests whether IB objective beats alignment-only fusion |
| Baseline | Concatenation autoencoder | Tests whether explicit IB structure matters at all |
| Ablation-of-ours | OMIB with K=1 unimodal Gaussian | Isolates contribution of mixture posterior |
| Ablation-of-ours | OMIB without component alignment (index-wise PoE only) | Tests whether the alignment step is necessary for the claimed identifiability |
| Ablation-of-ours | OMIB with K varied relative to true mode count (K ∈ {M/2, M, 2M}) | Empirically tests the assumption boundary K >= true number of modes |

### Why this setup validates the claim

The central claim is that a mixture-of-Gaussians posterior with product-of-experts aggregation and explicit component alignment yields compositional identifiability of a shared latent Z across n modalities, including incremental modality addition. The only way to test identifiability directly is against a ground-truth latent, which is why a synthetic benchmark with known generating factors is the right dataset — real data cannot supply the target for the metric. The primary metric, linear identifiability score (how well a linear map recovers true Z from learned Z), is exactly the quantity the theorem predicts should be high for our method and undefined or degraded for baselines lacking the mixture structure. The pairwise IB baseline tests the sub-claim that n>2 compositional structure is necessary; the contrastive baseline tests whether the IB bound (sufficiency + minimality) is doing work beyond mere alignment; the concatenation autoencoder tests whether any explicit latent-sharing objective is needed. The K=1 ablation is the sharpest test: it removes only the mixture, keeping PoE and the n-way IB, so any identifiability gap between it and full OMIB is attributable to the mixture posterior — the mechanism the theorem hinges on. The no-alignment ablation tests whether the explicit component-alignment step (Sinkhorn + Hungarian) is necessary for the claimed identifiability, isolating the load-bearing assumption. The K-vs-M sweep tests the assumption boundary K >= true number of modes. The dSprites-derived semi-real dataset tests whether the identifiability gain transfers beyond the synthetic setting where the generative process is assumed known. If identifiability gains appear uniformly across all baselines and ablations rather than concentrated where mixture structure should matter, the design fails to isolate the claimed cause.

### Expected outcome and causal chain

**vs. Pairwise IB (n=2, unimodal Gaussian)** — On a case where the true latent has multiple modes activated by different modality subsets, the pairwise baseline collapses them into a single Gaussian mode because its unimodal posterior cannot represent subset-specific latent factors, so its recovered Z mixes factors and the linear identifiability score drops. Our method instead assigns separate mixture components to each mode via aligned PoE aggregation, so the linear map recovers each factor cleanly. We expect a large identifiability gap that widens as the number of true latent modes grows, with near-parity when the true latent is effectively unimodal (M=1).

**vs. Contrastive fusion (CLIP-style)** — On a case where one modality is noisy but informative, the contrastive baseline aligns representations by pulling positives together, which lets the noisy modality drag the shared embedding toward its own idiosyncratic directions because alignment alone imposes no minimality constraint. Our method instead penalizes I(Z; X_i | Z_{-i}) through the KL and alignment terms, so the noisy modality's unique information is suppressed in Z. We expect our identifiability score to stay stable as modality noise increases while the contrastive baseline degrades, with the gap concentrated on the noisy-modality subset.

**vs. Concatenation autoencoder** — On a case where modalities are added incrementally (train on n=2, freeze, add modality 3 with only new encoder/decoder trained), the concatenation baseline must relearn a joint latent from scratch each time because its latent dimension and structure are tied to the current modality set, so earlier modality factors get entangled with new ones and identifiability of the original factors collapses. Our method instead preserves the shared Z structure through aligned PoE aggregation, so adding a modality refines rather than overwrites existing factors. We expect our identifiability score on the original factors to remain roughly flat across incremental additions while the concatenation baseline drops sharply.

**vs. OMIB without component alignment (index-wise PoE only)** — On a case where mixture components are permuted across modalities (label switching), the no-alignment variant aggregates mismatched components because its index-wise PoE assumes component k is shared a priori, so the shared posterior mixes factors and the linear identifiability score drops. Our method instead aligns components via Sinkhorn + Hungarian before aggregation, so the shared posterior preserves the true mode structure. We expect a large identifiability gap that widens as the number of modalities grows, with near-parity when components happen to be aligned by initialization.

**vs. OMIB with K varied relative to true mode count** — On a case where K < M (under-specified mixture), the model cannot represent all true modes, so identifiability is only approximate and the linear identifiability score drops. When K >= M, the model can represent all modes and identifiability should be high. We expect the score to increase with K up to M and then plateau, empirically testing the assumption boundary K >= true number of modes.

### What would falsify this idea

If the K=1 unimodal ablation matches full OMIB on the linear identifiability score across all latent-mode counts, then the mixture posterior is not the cause of compositional identifiability and the central claim is wrong. Likewise, if identifiability gains over the pairwise IB baseline are uniform rather than concentrated on multi-mode latent cases, or if the no-alignment ablation matches full OMIB, the predicted causal mechanism is not what is producing the effect.

## References

1. From Fragmentation to Fusion: A Survey and Roadmap for Omnimodal Embeddings
