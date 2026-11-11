# NormAgnostic-VC: A Normalization-Agnostic Variance-Closure Certificate for Cross-Layer KV Source Topologies

## Motivation

Cross-layer KV sharing methods — DepthGraft's stride-aligned shallow-middle source pairs, FusedKV's bottom/middle bottom-value/middle-key split, and KVSharer's dissimilar-pair selection — all assert that a predetermined source subset carries sufficient KV information for deep-layer reconstruction, but this adequacy is never derived; it is only checked empirically and only under Pre-LN, where the residual-stream variance recursion is benign. Because layer-norm placement (Pre-LN, Post-LN, RMSNorm, sandwich) changes the variance recursion that actually propagates KV information across depth, the same source assignment can be adequate under one normalization and silently fail under another, so the field lacks a derivation-level condition tying source choice to the normalization-induced variance recursion that transfers across schemes.

## Key Insight

The reconstruction error of a deep-layer KV from a fixed source set is controlled by a per-layer variance recursion whose closure radius is a computable function of the normalization operator alone, so source adequacy is a decidable property of the topology-plus-normalization pair rather than an empirical outcome of Pre-LN sweeps.

## Method

**What it is.** NormAgnostic-VC takes as input a fixed cross-layer KV source topology (a stride-aligned shallow/middle source assignment as in DepthGraft) together with a declared normalization placement (Pre-LN, Post-LN, RMSNorm, or sandwich) and returns a per-layer closure radius and a binary certificate: ACCEPT if the scale-normalized radius stays below a dimensionless reconstruction-error threshold at every deep layer, REJECT otherwise. It introduces no learned module; it is an offline analysis pass run before training.

**Load-bearing assumption (stated explicitly).** The certificate assumes that the per-layer variance gain g_l and a tail-sensitive normalization Jacobian norm J_l, measured on a calibration batch of the declared normalization placement, are sufficient statistics that transfer across normalization schemes when the closure radius is made dimensionless by dividing by the deep-layer variance v_d, so that a single fixed dimensionless tolerance tau yields correct ACCEPT/REJECT verdicts across Pre-LN, Post-LN, RMSNorm, and sandwich placements. This assumption is load-bearing: if it fails, the verdict flip attributed to the closure coefficient is confounded with an arbitrary tau shift.

**How it works.**

```pseudocode
INPUT: model depth L, source assignment S (list of source layers per deep layer),
       normalization placement n in {PreLN, PostLN, RMSNorm, Sandwich},
       dimensionless tolerance tau (fixed, calibrated once on a held-out Pre-LN model
       as the 90th percentile of r_d/v_d over its deep layers),
       calibration batch B of N=512 sequences
OUTPUT: per-layer closure radius r_l, scale-normalized radius rho_l = r_l / v_l,
        certificate ACCEPT/REJECT

# Phase 1: measure the normalization-induced variance transfer operator
for each layer l in 1..L:
    collect residual-stream activations h_l(x) over B
    compute empirical variance v_l = Var(h_l)          # full covariance trace
    compute gain g_l = v_l / v_{l-1}                   # per-layer variance gain
    # tail-sensitive Jacobian norm: 95th-percentile empirical Lipschitz estimate
    # over the calibration batch, not the bulk operator norm
    for each x in B:
        J_l(x) = ||d Norm_l(h_l(x)) / d h_l||_2
    J_l = Quantile_{0.95}({J_l(x) : x in B})
    # normalization-specific closure coefficient
    if n == PreLN:      c_l = g_l * J_l
    if n == PostLN:     c_l = J_l * g_l^0.5
    if n == RMSNorm:    c_l = g_l * (1 + J_l) / 2
    if n == Sandwich:   c_l = J_l^2 * g_l

# Phase 2: propagate the variance recursion along each source edge
for each deep layer d in 1..L:
    r_d = 0
    for each source s in S[d]:
        # product of closure coefficients along the path s -> d
        path = s..d
        P = prod_{l in path} c_l
        # source-to-target variance mismatch term
        M = |v_s - v_d| / max(v_s, v_d)
        r_d += P * M
    r_d = r_d / len(S[d])          # average over sources
    rho_d = r_d / v_d              # scale-normalized radius (dimensionless)

# Phase 3: certificate
cert = ACCEPT if max_d rho_d < tau else REJECT
return {r_d}, {rho_d}, cert
```

The closure radius r_d is the weighted sum, over source edges, of the variance mismatch between source and deep layer, damped by the product of normalization closure coefficients along the path. The scale-normalized radius rho_d = r_d / v_d makes the certificate dimensionless and invariant to the calibration model's variance scale. A large normalized radius means the source set cannot close the variance gap at layer d under the declared normalization.

**Why this design.** We chose to compute the closure coefficients c_l from measured per-layer variance gains and tail-sensitive normalization Jacobian norms rather than from a single global Lipschitz constant because the recursion is layer-local and the gain varies by orders of magnitude across depth, accepting the cost that the certificate requires a calibration batch and is therefore model-specific rather than architecture-generic. We chose to average the per-source radius contributions within a deep layer rather than take the max because DepthGraft's topology distributes gradient flow across multiple source layers and the average reflects the redundancy of the source set, accepting the cost that a single pathological source can be masked by well-behaved siblings. We chose a fixed dimensionless tolerance tau calibrated once on a held-out Pre-LN model (as the 90th percentile of rho_d over its deep layers) rather than a per-model tuned threshold because the certificate must transfer across normalization schemes without re-tuning, accepting the cost that models whose variance scale differs substantially from the calibration model may need a scale-normalized tau — which is why rho_d is normalized by v_d. We chose the 95th-percentile empirical Jacobian norm rather than the bulk operator norm because Post-LN heavy tails dominate reconstruction error and the bulk norm averages them away, accepting the cost that the tail estimate is noisier on small calibration batches. A domain expert would not describe this as a variant of DepthGraft or KVSharer: those methods select or assume a topology and validate it by training loss, whereas this method computes a normalization-dependent closure quantity that decides topology adequacy before any training run, and it is the only one of the three that produces a REJECT verdict for a topology that DepthGraft would accept under Pre-LN but that fails under Post-LN.

**Why it measures what we claim.** The per-layer variance gain g_l measures the residual-stream variance recursion because it is the exact multiplicative factor by which layer l rescales the residual stream, under the assumption that the residual stream is approximately isotropic within a layer; this assumption fails when attention output and MLP output are strongly anisotropic and correlated, in which case g_l reflects the dominant direction's gain rather than the full covariance trace — we therefore compute v_l as the full covariance trace and report the anisotropy ratio (top-1 eigenvalue / trace) alongside the certificate to bound this deviation. The tail-sensitive Jacobian norm J_l measures the normalization operator's contraction/expansion of perturbations because the 95th-percentile empirical Lipschitz estimate bounds how a source-layer variance perturbation is scaled at layer l in the tail of the activation distribution; this assumption fails when the calibration batch is too small to populate the tail, in which case J_l reflects the bulk of the distribution rather than the tail that dominates reconstruction error. The path product P measures the cumulative damping of a source-to-deep variance mismatch because the recursion composes multiplicatively across layers, under the assumption that successive layers' gains are approximately independent; this assumption fails when consecutive layers are strongly correlated (e.g., tied or repeated blocks), in which case P overestimates damping. The mismatch term M measures the source set's coverage of the deep layer's variance because a source whose variance differs greatly from the target cannot reconstruct it regardless of path damping, under the assumption that variance magnitude is a sufficient proxy for KV information content; this assumption fails when a source has matching variance but mismatched key-direction geometry, in which case M understates the true reconstruction error — which is why the certificate is a necessary rather than sufficient condition for adequacy.

## Contribution

(1) A normalization-agnostic variance-closure certificate that maps any (source topology, normalization placement) pair to a per-layer closure radius and an ACCEPT/REJECT verdict without training, using measured per-layer variance gains and normalization Jacobian norms. (2) A design principle: source-set adequacy is a property of the normalization-induced variance recursion, so a topology certified under Pre-LN can be shown to fail under Post-LN or sandwich normalization, giving a pre-training filter for cross-layer KV sharing topologies. (3) An open analysis tool that reports per-layer closure radii for DepthGraft-style stride-aligned source sets across the four normalization placements, enabling topology comparison before empirical sweeps.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | Calibration batch of N=512 sequences from a held-out Pre-LN LM (6–12 layers) | Provides model-specific variance statistics for certificate |
| Primary metric | Certificate verdict accuracy vs. trained reconstruction error (per-layer KV reconstruction MSE) | Directly tests whether ACCEPT/REJECT predicts adequacy |
| Secondary metric | Correlation (Pearson and Spearman) between scale-normalized radius rho_d and measured per-layer KV reconstruction error | Tests the proxy equivalence that rho_d bounds reconstruction error |
| Baseline | DepthGraft topology under Pre-LN | Tests sub-claim that Pre-LN topologies pass |
| Baseline | Same topology under Post-LN | Tests normalization-dependence of verdict |
| Baseline | KVSharer layer-wise sharing | Tests closure radius against a sharing heuristic |
| Ablation-of-ours | Fixed global tau, no per-layer gain (c_l = 1) | Tests value of layer-local closure coefficients |
| Ablation-of-ours | Bulk operator norm J_l instead of 95th-percentile tail estimate | Tests value of tail-sensitive Jacobian |
| Ablation-of-ours | Unnormalized radius r_d with Pre-LN-calibrated tau | Tests value of scale normalization |

### Why this setup validates the claim

The central claim is that a normalization-dependent, scale-normalized closure radius computed offline predicts whether a cross-layer KV source topology will reconstruct deep-layer variance before any training run. The dataset of calibration activations from a held-out Pre-LN model supplies exactly the statistics the recursion consumes, so the certificate is tested on the same object it reasons about rather than on a proxy. The primary metric — agreement between the binary verdict and the actual post-training reconstruction error — is the right one because the method's only output is a verdict; if ACCEPT does not track low reconstruction error and REJECT does not track high error, the closure quantity is measuring nothing useful. The secondary correlation metric directly tests the proxy equivalence that rho_d bounds per-layer KV reconstruction error, which is the assumption that closes the certificate. The Pre-LN DepthGraft baseline tests the sub-claim that Pre-LN topologies should pass; the Post-LN baseline on the identical topology tests the sub-claim that normalization placement flips the verdict; KVSharer tests whether a variance-agnostic sharing heuristic produces the same verdict pattern, which would undercut the claimed novelty. The ablations remove per-layer gain measurement, tail sensitivity, and scale normalization in turn, so if verdict accuracy is unchanged under any ablation, the corresponding component is doing no work and the simpler variant suffices.

### Expected outcome and causal chain

**vs. DepthGraft under Pre-LN** — On a deep model with stride-aligned shallow/middle sources, the Pre-LN baseline produces a topology that trains to low reconstruction error because Pre-LN's variance gain stays near unity across depth, so source-to-target variance mismatch is small and the scale-normalized closure radius stays below tau. Our method computes the same path products and, because Pre-LN coefficients c_l = g_l * J_l remain modest with g_l near one, returns ACCEPT for the same topology. We expect the verdict to agree with the trained reconstruction error on nearly all layers, with the small residual disagreement concentrated at the deepest layers where variance drifts upward.

**vs. Same topology under Post-LN** — On the identical source assignment, the Post-LN baseline produces a topology that fails to reconstruct deep-layer variance because Post-LN's normalization Jacobian grows with the heavy-tailed pre-norm activation, so the effective variance gain diverges from the Pre-LN case and the source layers no longer cover the deep layer's scale. Our method instead uses c_l = J_l * g_l^0.5 with J_l the 95th-percentile tail estimate, which captures the Post-LN Jacobian amplification, so the path product P shrinks and the mismatch term M rises, pushing rho_d above tau and yielding REJECT. We expect a clear verdict flip relative to Pre-LN on the deepest third of layers, with the radius gap concentrated where Post-LN variance gain deviates most from unity.

**vs. KVSharer layer-wise sharing** — On a model where a pathological source sits alongside well-behaved siblings, the KVSharer baseline produces a sharing decision driven by layer-wise dissimilarity that ignores variance scale, so it accepts the sharing even though the pathological source cannot carry the deep layer's variance. Our method instead averages per-source radius contributions, so the pathological source inflates r_d only partially; but because the mismatch term M for that source is large, the average still crosses tau in the regime where the sibling sources cannot compensate. We expect our verdict to diverge from KVSharer precisely on layers whose source set contains a high-mismatch source, showing REJECT where KVSharer shares.

### What would falsify this idea

If the verdict accuracy is uniform across Pre-LN and Post-LN models rather than concentrated on the Post-LN layers where the closure coefficient is predicted to flip, then the normalization dependence is not the causal driver and the certificate is just a depth-generic threshold. Likewise, if removing per-layer gain measurement (the ablation) or replacing the tail-sensitive Jacobian with the bulk operator norm leaves verdict accuracy unchanged, the layer-local recursion and tail sensitivity are not doing the work we claim; and if the scale-normalized radius rho_d does not correlate with measured per-layer KV reconstruction error, the proxy equivalence fails.

## References

1. DepthGraft: Structural Regularization through Hierarchical Cross-Layer KV Reconstruction
2. Reconstructing KV Caches with Cross-layer Fusion For Enhanced Transformers
3. KVSharer: Efficient Inference via Layer-Wise Dissimilar KV Cache Sharing
