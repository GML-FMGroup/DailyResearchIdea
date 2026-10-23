# Unpaired Factor-Span Identifiability via Invariant Latent-Delta Prediction

## Motivation

Existing factor-aware video generation methods such as FAPL (Making MLLM Latents Matter) treat a frozen MLLM/V-JEPA latent space as already factor-disentangled and rely on paired videos to supply factor-span deltas, so identifiability of which latent directions encode which factors is imported from annotations rather than learned. When pairs are absent, the predictor is never forced to discover the factor-to-direction mapping, and the structural property—factor-span replacement—is undefined. This meta-gap recurs across feature-prediction and masked-Siamese branches (V-JEPA, MSN, data2vec 2.0): all assume the frozen encoder's latent space is separable and never establish identifiability from unpaired data.

## Key Insight

A factor direction is identifiable from unpaired clips exactly when the Jacobian of a learned latent-delta predictor has rank equal to the number of factors and its column space is invariant to nuisance perturbations, a structural constraint that replaces paired annotations with an orthogonality/rank condition on the predictor itself.

## Method

## FARA: Factor-Agnostic Rank-constrained Alignment

**What it is.** FARA learns a factor-identifiable latent subspace from unpaired video clips by training a predictor $P_\theta$ that, given two clips $x_a, x_b$ (unpaired, no factor labels), outputs a latent delta $\Delta z = P_\theta(z_a, z_b)$ in the frozen encoder's latent space, subject to a rank/orthogonality constraint on its Jacobian $J_P = \partial \Delta z / \partial z_a$. The output is a set of $K$ factor directions $\{u_1,\dots,u_K\}$ spanning the identifiable subspace, plus a per-factor magnitude predictor.

**Load-bearing assumption (stated explicitly).** FARA assumes that the column space of the predictor Jacobian $J_P$ coincides with the true factor directions in the frozen encoder's latent space, so that rank/orthogonality of $J_P$ implies factor identifiability rather than merely an arbitrary $K$-dimensional subspace of the predictor's own parameterization. This assumption is not enforced by the rank/orthogonality losses alone; it is anchored by a small calibration set of factor-labeled pairs (see step 7) that breaks the rotation symmetry and selects which of the $K$ directions corresponds to which named factor. Identifiability is therefore reported up to this anchor.

**How it works.**

```pseudocode
Input: frozen encoder E (MLLM/V-JEPA), unpaired clips {x_i}, K factors, margin m, temperature tau
       calibration set C = {(x_i, x_j, factor_label k)} of size N_cal = 512 (factor-labeled pairs)
Initialize predictor P_theta (2-layer MLP on concatenated latents, hidden=256, GeLU)
for each training step:
    sample x_a, x_b, x_c unpaired
    z_a, z_b, z_c = E(x_a), E(x_b), E(x_c)
    # 1. latent-delta prediction
    d_ab = P_theta(z_a, z_b)                      # predicted delta
    d_ac = P_theta(z_a, z_c)
    # 2. invariance to nuisance: augment clip b by nuisance-only
    #    perturbations (color jitter, temporal crop, background swap)
    x_b' = nuisance_augment(x_b)
    z_b' = E(x_b')
    d_ab' = P_theta(z_a, z_b')
    # 3. rank/orthogonality constraint on Jacobian
    J = jacobian(P_theta, z_a)                    # K x D, computed via Hutchinson estimator with 1 JVP per step
    L_rank = -logdet(J J^T + eps I)               # encourages rank K
    L_orth = ||J J^T - diag(J J^T)||_F^2          # orthogonality of factor directions
    # 4. invariance loss: delta must not change under nuisance
    L_inv = ||d_ab - d_ab'||^2
    # 5. factor-span consistency: composition of deltas
    L_comp = ||P_theta(z_a, z_b) + P_theta(z_b, z_c) - P_theta(z_a, z_c)||^2
    # 6. contrastive factor-direction separation
    L_con = -log( exp(sim(d_ab, d_ab')/tau) / sum_{c} exp(sim(d_ab, d_ac)/tau) )
    # 7. calibration anchor: select which direction corresponds to which factor
    sample (x_i, x_j, k) from C
    z_i, z_j = E(x_i), E(x_j)
    d_ij = P_theta(z_i, z_j)
    L_anchor = -log( exp(sim(d_ij, u_k)/tau) / sum_{l=1}^K exp(sim(d_ij, u_l)/tau) )
    loss = L_rank + lambda1*L_orth + lambda2*L_inv + lambda3*L_comp + lambda4*L_con + lambda5*L_anchor
    update theta
return factor directions U = top-K eigenvectors of J J^T, predictor P_theta
```

Hyperparameters: $K$ set by factor count (e.g., 4 for character/scene/action/camera), margin $m=0.1$, temperature $\tau=0.07$, $\lambda_1=0.1$, $\lambda_2=1.0$, $\lambda_3=0.5$, $\lambda_4=1.0$, $\lambda_5=1.0$, calibration set size $N_{cal}=512$ factor-labeled pairs (0.5% of training clips).

**Why this design.** We chose a rank/orthogonality constraint on the Jacobian $J_P$ over paired factor-delta supervision because paired annotations are the bottleneck FAPL explicitly acknowledges; the rank condition forces the predictor to discover $K$ independent directions without labels, accepting the cost that the constraint is only necessary, not sufficient, for true factor separation. We chose nuisance-invariance via augmentation over a learned disentanglement head because nuisance factors (color, background) are operationally separable from content factors in video, so a contrastive invariance loss is well-posed, accepting the cost that augmentation design becomes a hyperparameter. We chose composition consistency $P(a,b)+P(b,c)=P(a,c)$ over a cycle-consistency loss because it directly enforces the additive group structure of factor spans, accepting the cost that it assumes factors compose linearly in latent space. We added a small calibration anchor $L_{anchor}$ over a purely label-free objective because the rank/orthogonality constraints alone are rotation-invariant and cannot select which direction corresponds to which factor; the anchor breaks this symmetry with a handful of factor-labeled pairs, accepting the cost that identifiability is reported up to this anchor. A domain expert would not describe FARA as a variant of FAPL: FAPL supervises the generator with paired factor spans, whereas FARA learns the factor directions themselves from unpaired data via a structural constraint on the predictor's Jacobian, which FAPL never computes.

**Why this measures what we claim.** The Jacobian rank $\mathrm{rank}(J_P)$ measures factor identifiability because the assumption that factor directions are linearly independent in latent space makes full rank equivalent to a factor-spanning subspace; this assumption fails when the frozen encoder entangles two factors into one direction, in which case rank reflects encoder capacity rather than factor count. The invariance loss $\|d_{ab}-d_{ab'}\|^2$ measures nuisance invariance because the assumption that the augmentation set covers all nuisance factors makes the delta unchanged iff the predictor ignores nuisance; this assumption fails when an unmodeled nuisance factor (e.g., lighting) is present, in which case the loss reflects augmentation coverage rather than true invariance. The composition loss measures factor-span consistency because the assumption that factor deltas are additive in latent space makes composition exact; this assumption fails for non-linear factor interactions, in which case the loss reflects curvature rather than inconsistency. The calibration anchor $L_{anchor}$ measures factor-direction correspondence because the assumption that the 512 labeled pairs are factor-pure makes the anchor select the correct direction; this assumption fails when the labeled pairs contain mixed factors, in which case the anchor reflects label noise rather than true correspondence.

## Contribution

["A factor-identifiable latent subspace learning framework (FARA) that replaces paired factor-span annotations with a rank/orthogonality constraint on a latent-delta predictor's Jacobian, enabling factor discovery from unpaired video.", 'A structural identifiability condition for frozen-encoder latent spaces: full-rank Jacobian plus nuisance invariance is necessary for factor-span replacement to be well-defined without paired supervision.', "An unpaired factor-span prediction objective combining rank, orthogonality, invariance, composition, and contrastive losses, with ablation showing each term's role in recovering factor directions."]

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Cinematic video clips with character/scene/action/camera variation | Matches the four-factor structure the method claims to identify |
| Primary metric | Factor-direction retrieval accuracy under held-out factor swaps | Directly probes whether discovered directions track true factors |
| Baseline | Native MLLM conditioning (no factor learning) | Tests whether learned directions beat implicit conditioning |
| Baseline | Paired-supervision factor-span model (FAPL-style) | Tests unpaired identifiability against labeled upper bound |
| Baseline | No-rank / no-orthogonality variant | Tests whether the Jacobian constraint is what yields identifiability |
| Ablation-of-ours | FARA without composition loss | Isolates the additive-span assumption from rank constraint |
| Ablation-of-ours | FARA without calibration anchor | Tests whether the anchor is necessary to break rotation symmetry |

### Why this setup validates the claim

The central claim is that a rank/orthogonality constraint on the predictor Jacobian yields factor-identifiable directions from unpaired clips, up to a small calibration anchor that breaks rotation symmetry. The setup tests this by measuring whether the recovered directions $\{u_1,\dots,u_K\}$ retrieve the correct factor when a clip is edited along that direction and by another along a different factor. The native-conditioning baseline tests whether explicit direction discovery beats implicit MLLM conditioning; the paired-supervision baseline tests whether FARA approaches the labeled upper bound despite being unpaired, which is the core value proposition; the no-rank variant tests whether the Jacobian constraint itself, not merely training, produces identifiability; the no-anchor ablation tests whether the calibration anchor is necessary to select the correct factor directions. Retrieval accuracy is the right metric because it is sensitive to direction entanglement: if two factors share a direction, retrieval collapses even when reconstruction looks fine, so the metric detects exactly the predicted failure mode. The calibration set of 512 factor-labeled pairs (0.5% of training clips) is used only to anchor direction-to-factor correspondence, not to train the predictor's delta, so the unpaired nature of the main training is preserved.

### Expected outcome and causal chain

**vs. Native MLLM conditioning** — On a case where the prompt specifies a character change but the clip also varies camera, native conditioning produces entangled edits because it has no explicit factor basis and must infer the edit axis from text alone. Our method instead exposes orthogonal directions $u_k$ and applies the delta along the character direction only, because the rank/orthogonality constraint separates the span and the calibration anchor selects the character direction. We expect a clear retrieval gap on multi-factor clips (edits that change one factor while holding others) but near-parity on single-factor clips, where entanglement does not bite.

**vs. Paired-supervision factor-span model** — On a case with scarce paired factor annotations, the paired model overfits to the annotated factor combinations it saw and mis-retrieves unlabeled combinations, because its supervision does not constrain the full direction set. Our method instead learns directions from unpaired clips via composition and invariance, and uses only 512 labeled pairs to anchor direction-to-factor correspondence, so it covers unseen combinations. We expect FARA to trail the paired model slightly on annotated combinations but to close or reverse the gap on held-out combinations, with the advantage concentrated where annotation coverage is sparse.

**vs. No-rank / no-orthogonality variant** — On a case where character and action co-vary in the training clips, the unconstrained variant collapses both into a single dominant direction, because nothing forces independent Jacobian rows, so editing character also shifts action. Our method instead spreads mass across $K$ orthogonal directions, keeping the two separable. We expect the variant to show high retrieval for the dominant factor but near-chance retrieval for the collapsed one, while FARA shows balanced retrieval across all $K$ factors.

**vs. FARA without calibration anchor** — On a case where the frozen encoder's latent axes are rotated relative to the named factors, the no-anchor variant satisfies rank and orthogonality but assigns arbitrary factor labels to directions, because the rank/orthogonality losses are rotation-invariant. Our method uses the 512 labeled pairs to select the correct direction-to-factor mapping, so retrieval accuracy is measured against the true factor names. We expect the no-anchor variant to show near-chance retrieval for all factors despite full rank, while FARA shows high retrieval, confirming that the anchor is necessary to break rotation symmetry.

### What would falsify this idea

If FARA's retrieval gain over native conditioning were uniform across single-factor and multi-factor clips rather than concentrated on multi-factor clips where entanglement is the predicted failure mode, the rank/orthogonality mechanism would not be doing the claimed work. Likewise, if the no-rank variant matched FARA on balanced per-factor retrieval, or if the no-anchor variant matched FARA, the Jacobian constraint or the calibration anchor would be unnecessary and the central claim false.

## References

1. Making MLLM Latents Matter: Factor-Aware Predictive Learning for Cinematic Video Generation
2. Revisiting Feature Prediction for Learning Visual Representations from Video
3. HunyuanVideo: A Systematic Framework For Large Video Generative Models
4. Masked Siamese Networks for Label-Efficient Learning
5. Efficient Self-supervised Learning with Contextualized Target Representations for Vision, Speech and Language
