# Unpaired Factor Identification in Frozen MLLM Latents via Temporal-Adjacency Interventions

## Motivation

The convergent trajectory of factor-aware video generation, exemplified by FAPL (Making MLLM Latents Matter), has made factor-span replacement uniform but still pair-dependent: it requires paired videos that explicitly specify target-factor changes and non-target preservation. This requirement is structurally costly and limits scale, and—more importantly—it leaves the frozen-MLLM latent space's factor directions unverified, because the paired supervision only teaches the DiT to realize relationships rather than identifying the latent directions themselves. The field's next bottleneck is therefore to make those directions identifiable from unpaired data, breaking the circularity of assuming disentanglement without a verifiable intervention signal.

## Key Insight

Temporally adjacent video spans provide a natural, unpaired intervention: the generator's own controllable transformations (e.g., synthetic camera or object-attribute edits) supply a known intervention signal that is independent of the latent space, so a lightweight factor-direction estimator can learn which latent subspace changes under that intervention while remaining invariant to passively varying factors—yielding identifiable directions without paired factor-delta supervision.

## Method

**What it is** — We introduce **TAFI** (Temporal-Adjacency Factor Identification), a self-supervised objective that trains a lightweight factor-direction estimator to map a known controllable transformation (applied to one of two temporally adjacent spans) to a direction in the frozen-MLLM latent space, using only unpaired video spans. Inputs: frozen MLLM latents for two temporally adjacent spans and a known transformation label. Output: a set of factor-direction vectors in latent space that are identifiable up to sign and scale.

**How it works** —
```pseudocode
# Given: frozen MLLM encoder E, video V, temporal adjacency window W (W = 8 frames, i.e., spans are 8 frames apart)
# Known controllable transformation T (e.g., camera pan, object attribute edit)
# Lightweight estimator f_theta: (latent, T) -> direction vector d; f_theta is a 2-layer MLP with hidden size 256, GeLU activation, output dim = latent dim (1024)
# Nuisance basis: top-k PCA directions P of unpaired latents, k = 8, computed once per epoch on a random subset of 10,000 latents
# Intervention-to-nuisance ratio gate: r = ||alpha * d_T||^2 / ||z2 - z1||^2; apply L_int only when r >= 0.5 (calibrated on a held-out set of 512 spans)

for each training step:
    sample video V; split into two temporally adjacent spans S1, S2 (gap <= W)
    z1 = E(S1); z2 = E(S2)                     # frozen latents
    # Implicit intervention: assume S2 = T(S1) for a known T
    # (T is applied synthetically to S1 to create S2 during data construction)
    d_T = f_theta(z1, T)                        # predicted factor direction
    # Intervention loss: the latent change along d_T should explain z2 - z1
    alpha = (d_T^T (z2 - z1)) / (d_T^T d_T)     # scalar projection
    L_int = || (z2 - z1) - alpha * d_T ||^2     # alpha: scalar projection
    # Invariance loss: d_T should be orthogonal to passively varying factors
    # Estimate passive factors via PCA on a batch of unpaired latents
    P = top-k PCA directions of {z_i}          # k = 8
    L_inv = || P^T d_T ||^2
    # Total loss
    L = L_int + lambda * L_inv                  # lambda = 0.1
    update theta
```

**Why this design** — We chose temporal adjacency as an implicit intervention over paired factor-delta supervision because it exploits the causal structure of video generation: adjacent spans are naturally related by a small, known transformation, and the transformation label is independent of the latent space, breaking the circularity that FAPL's paired supervision inherits. We chose a lightweight estimator f_theta over a full generative model because the goal is to identify directions, not to generate; this trades off expressivity for identifiability, accepting that f_theta may underfit complex transformations. We chose an orthogonality penalty against PCA-estimated passive factors over an explicit disentanglement head because PCA provides a data-driven, assumption-light estimate of passively varying directions, accepting that PCA may conflate factors if the data is highly correlated. An expert would not describe this as a variant of FAPL because FAPL uses paired videos to specify target-factor changes, whereas TAFI uses unpaired temporal adjacency and a known transformation label to identify latent directions without any paired supervision.

**Why it measures what we claim** — The intervention loss `L_int` measures the alignment between the predicted factor direction and the actual latent change because we assume that the known transformation T is the only systematic cause of change between adjacent spans; this assumption fails when other factors (e.g., object motion) also change between spans, in which case `L_int` reflects a mixture of factors rather than the isolated target. The invariance loss `L_inv` measures the orthogonality of the estimated direction to passively varying factors because we assume that the top-k PCA directions of unpaired latents capture the dominant passive factors; this assumption fails when passive factors are not the top principal components (e.g., rare but strong factors), in which case `L_inv` may not penalize the correct directions. Together, these quantities operationalize factor identifiability: a direction is identifiable if it explains the intervention-induced change and is orthogonal to passive variation.

## Contribution

['A self-supervised factor-identification objective (TAFI) that uses temporally adjacent video spans as implicit interventions to learn identifiable factor directions in frozen-MLLM latents without paired factor-delta supervision.', 'A lightweight factor-direction estimator trained with an intervention loss and an orthogonality penalty against PCA-estimated passive factors, yielding directions that are both explanatory and invariant.', 'An empirical demonstration that unpaired temporal adjacency can substitute for paired supervision in factor-aware video generation, reducing data requirements while maintaining controllability.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Cinematic multi-shot video with synthetic camera/attribute edits | Provides known transformation labels and adjacent spans |
| Primary metric | Factor-direction alignment: projection of held-out latent delta onto estimated direction | Directly tests whether direction explains intervention change |
| Baseline | Native MLLM conditioning without factor-direction estimation | Tests whether explicit direction estimation adds control |
| Baseline | FAPL paired factor-delta supervision | Tests whether unpaired adjacency suffices vs paired supervision |
| Baseline | PCA-only passive-factor directions as control axes | Tests whether intervention-specific directions beat generic variation |
| Ablation-of-ours | TAFI minus invariance loss (L_int only) | Isolates whether orthogonality penalty is load-bearing |

### Why this setup validates the claim

The central claim is that a known transformation label applied to temporally adjacent spans, combined with an orthogonality penalty against passively varying directions, yields identifiable factor directions in frozen-MLLM latent space without paired supervision. The dataset must therefore contain transformations whose ground-truth factor is known (synthetic camera pan, object attribute edits) so that held-out latent deltas can be attributed to a single factor; this is what makes the alignment metric a genuine test rather than a proxy. The native-conditioning baseline tests the sub-claim that explicit direction estimation is necessary — if native conditioning already achieves comparable alignment, the estimator is redundant. The FAPL baseline tests the sub-claim that unpaired temporal adjacency can substitute for paired factor-delta supervision; if FAPL dominates, the circularity-breaking argument fails. The PCA-only baseline tests whether intervention-specific directions are distinguishable from generic principal variation, which is the core of the invariance argument. The ablation (L_int only) tests whether the orthogonality penalty actually prevents direction conflation. The alignment metric is the right detector because it measures exactly the quantity the loss optimizes on held-out data, so a real effect must transfer beyond the training spans; a method that merely memorizes training deltas will show high train alignment but collapse on held-out spans.

### Expected outcome and causal chain

**vs. Native MLLM conditioning** — On a case where the target factor is a subtle camera pan between adjacent spans, the native-conditioning baseline produces latents whose delta is dominated by content-preserving nuisance variation, because it has no mechanism to isolate the pan direction from passive scene drift. Our method instead projects the delta onto an estimated pan direction that was trained to explain intervention-induced change and to be orthogonal to PCA passive factors, so the target component is separated. We expect a clear alignment gap on subtle-transformation subsets and near-parity on large, unambiguous transformations where nuisance variation is small relative to the signal.

**vs. FAPL paired factor-delta supervision** — On a case where paired videos are unavailable and only temporal adjacency is given, FAPL cannot form a supervision target, so it either falls back to native conditioning or overfits to the few paired examples, producing directions that track dataset-specific correlations rather than the transformation. Our method instead uses the known transformation label as an intervention signal independent of the latent space, breaking the circularity, so its directions should generalize to held-out transformations. We expect TAFI to match FAPL on paired-rich subsets but show a marked advantage on unpaired or sparsely paired subsets, with the gap widening as paired supervision decreases.

**vs. PCA-only passive-factor directions** — On a case where the target transformation (e.g., object attribute edit) is correlated with a dominant passive factor such as global illumination, the PCA-only baseline selects a direction that conflates the two, so its alignment on held-out deltas is inflated by the nuisance component and its control of the target factor is weak. Our method instead penalizes overlap with the top-k PCA directions during training, forcing the estimated direction to be orthogonal to dominant passive variation. We expect PCA-only to show high raw alignment but poor target-specific control (measured by factor-isolation error), while TAFI shows slightly lower raw alignment but substantially better isolation, especially on correlated-factor subsets.

### What would falsify this idea

If TAFI's alignment gain over native conditioning is uniform across all subsets rather than concentrated on subtle or correlated-factor subsets where the predicted failure modes occur, the causal story about intervention-based isolation is wrong. Likewise, if the L_int-only ablation matches the full model on correlated-factor subsets, the invariance penalty is not doing the work claimed and the identifiability argument collapses.

## References

1. Making MLLM Latents Matter: Factor-Aware Predictive Learning for Cinematic Video Generation
2. Revisiting Feature Prediction for Learning Visual Representations from Video
3. HunyuanVideo: A Systematic Framework For Large Video Generative Models
4. Masked Siamese Networks for Label-Efficient Learning
5. Efficient Self-supervised Learning with Contextualized Target Representations for Vision, Speech and Language
