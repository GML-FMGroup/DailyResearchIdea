# Reconstruction-Coupled Channel Selection for Budget-Aware Diffusion Feature Transfer

## Motivation

Budget-aware feature selection methods such as Policy-Mask pick a subset of channels from a frozen diffusion backbone but offer no pathway to recover information carried by discarded channels, so the selection is a one-way lossy bottleneck whose damage grows as the channel budget shrinks. The root cause is structural: the policy is trained only on the downstream discriminative loss over selected features, so it has no incentive to preserve, and no mechanism to reconstruct, the complement it discards. This same one-way compression recurs across parameter-efficient transfer approaches that treat compression as fixed and irreversible (e.g., Policy-Mask inherits PixArt-Σ-style lossy compression), leaving a persistent information-recovery gap under tight budgets.

## Key Insight

If the discarded channel complement is made recoverable from the selected subset through a decoder trained with an information-recovery loss, then selection is no longer a pure lossy projection but a sufficient-statistic bottleneck, and the policy is forced to select channels that jointly span the discarded subspace rather than channels that are merely individually salient.

## Method

**What it is.** ReCo (Reconstruction-Coupled selection) augments Policy-Mask's budget-conditioned policy with a lightweight reconstruction decoder that maps the selected key/value channel subset back to the full compressed representation, and trains the decoder and policy jointly under a downstream loss plus an information-recovery loss. Inputs: frozen diffusion intermediate features at a given (timestep, block); output: a selected channel subset used by the downstream head, plus a reconstruction of the discarded complement used only as a training signal.

**How it works.**

```pseudocode
# Frozen backbone, per (timestep t, block b) feature map F in R^{C x H x W}
# Policy pi_theta(t, b, c | x, budget B) outputs channel scores s_c, selects top-B channels S
# Decoder g_phi: 2-layer conv (hidden=64, kernel=3, ReLU), input z_S (B x H x W) + learned
#   embeddings for t and b (dim=16, concatenated as channel-wise bias), output F_hat in R^{C x H x W}
# Task head h_psi: linear classifier on z_S (for segmentation, 1x1 conv to num_classes)
# Straight-through top-k: s_st = s + (onehot(topk(s,B)) - s).detach()
for each training step:
    F = backbone(t, b)                      # frozen, C channels
    s = pi_theta(x, budget=B)               # channel scores, cardinality-normalized
    S = topk(s, B);  M = complement(S)      # selected / discarded channel index sets
    z_S = F[S]                              # selected features -> downstream head
    # Reconstruction decoder g_phi maps selected -> full C channels
    F_hat = g_phi(z_S, t, b)                # reconstruct all C channels, including M
    L_task = CE_or_mIoU_head(z_S, y)        # downstream discriminative loss
    # Information-recovery loss on discarded complement, weighted by task sensitivity
    # w_c = || d L_task / d F[M]_c ||_2 (per-channel gradient norm, detached)
    # L_rec = sum_{c in M} w_c * || F_hat[c] - F[c] ||_2^2 / (sum_{c in M} w_c + eps)
    L_rec  = weighted_MSE(F_hat[M], F[M], w)  # w from task sensitivity
    # Policy is trained through a straight-through top-k so gradients flow to s
    L = L_task + lambda * L_rec             # lambda = 0.1, fixed
    update theta, phi by Adam
```

The decoder is discarded at inference; only the policy and selected features are kept, so the parameter budget at test time is unchanged. The reconstruction loss is computed only on the discarded complement $M$ (not the selected channels, which are trivially available), so it directly penalizes unrecoverable information. **Load-bearing assumption (stated explicitly):** the discarded channel complement $M$ is approximately a deterministic function of the selected channels $S$ (i.e., $M$ lies in the span of $S$ up to noise), so minimizing $\|F_hat[M]-F[M]\|$ forces the policy to select channels that span the task-relevant subspace rather than merely the dominant variance subspace. This assumption is not guaranteed by the reconstruction objective alone; it is the equivalence we operationalize and test via the task-sensitivity weighting and the diagnostic probes in the experiment.

**Why this design.** We chose to reconstruct only the discarded complement rather than the full feature map because reconstructing selected channels is trivially satisfied by an identity path and would dilute the gradient signal toward the channels that actually matter for the bottleneck; the cost is that the decoder cannot correct errors in selected channels, which we accept because selected channels are passed through unchanged. We chose a lightweight 2-layer conv decoder over a transformer decoder because the reconstruction target is a local feature map and a conv decoder adds negligible parameters (≈0.3M) and no inference cost, accepting lower reconstruction capacity than an attention decoder. We chose to train the policy through the reconstruction loss via a straight-through top-k rather than a separate selection objective because this couples selection and recovery into a single differentiable objective, accepting the variance of straight-through gradients. We weight the reconstruction loss by per-channel task sensitivity $w_c = \|\partial L_task / \partial F[M]_c\|_2$ so that low-variance but label-relevant directions in $M$ are not dominated by high-variance reconstruction error; this is a specification of the existing information-recovery loss, not a new module. A domain expert would not describe this as Policy-Mask plus an autoencoder: Policy-Mask's selection is trained only on the downstream loss and treats the discarded set as irrelevant, whereas ReCo's selection is explicitly optimized to make the discarded set reconstructible, which changes which channels are selected (spanning the discarded subspace) rather than merely adding a decoder.

**Why it measures what we claim.** The reconstruction loss $L_{rec}$ on the discarded complement measures the *information recoverability* of the selection because it assumes the discarded channels are a deterministic function of the selected channels plus noise; this assumption fails when the discarded channels carry task-relevant variation that is statistically independent of the selected channels (e.g., a rare class whose signature lives only in discarded channels), in which case $L_{rec}$ reflects reconstruction of the dominant subspace rather than task-sufficient information. The task-sensitivity weighting $w_c$ is intended to mitigate this failure by up-weighting channels with large downstream gradient, but it does not guarantee coverage of independent low-variance directions. The straight-through top-k gradient measures the *marginal recoverability cost* of each channel's exclusion because it assumes the decoder's reconstruction error is locally linear in the selected set; this assumption fails when the decoder is saturated, in which case the gradient reflects decoder capacity rather than selection quality. The cardinality-normalized channel scores measure *budget-relative salience* because they assume the score magnitude is comparable across budgets; this assumption fails at very small $B$ where normalization inflates noise, in which case the score reflects rank instability rather than salience.

## Contribution

['A reconstruction-coupled channel selection mechanism (ReCo) that adds a lightweight decoder trained with an information-recovery loss on the discarded channel complement, making budget-aware selection a sufficient-statistic bottleneck rather than a one-way lossy projection.', 'A design principle: penalizing reconstruction of the discarded complement (not the full feature map) changes which channels the policy selects, forcing selection to span the discarded subspace; this is an empirical finding about what the recovery signal must target to affect selection.', 'A training procedure that couples the selection policy and reconstruction decoder through a straight-through top-k with a single joint loss, requiring no extra inference-time parameters.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | ADE20K semantic segmentation | Dense per-pixel labels stress channel sufficiency |
| Primary metric | mIoU at matched channel budget | Directly measures task-sufficient information retained |
| Baseline | Policy-Mask (downstream-only selection) | Isolates value of reconstruction coupling |
| Baseline | Fixed hand-picked channel selection | Tests whether learned coupling beats static heuristics |
| Baseline | Random channel subset at same budget | Lower bound on selection informativeness |
| Ablation-of-ours | ReCo without L_rec (λ=0) | Removes recovery signal, keeps decoder architecture |
| Ablation-of-ours | ReCo with unweighted L_rec (w_c=1) | Tests whether task-sensitivity weighting is necessary |
| Diagnostic probe | Linear probe on discarded channels M | Directly measures task-relevant information in M |
| Diagnostic probe | Rank/condition number of selected-vs-discarded covariance | Tests spanning property of S over M |
| Diagnostic probe | Reconstruction error of M vs. budget | Correlates recoverability with mIoU |
| Sensitivity sweep | λ ∈ {0.01, 0.1, 1.0} | Confirms fixed λ=0.1 is not fragile |
| Compute accounting | Decoder params, training FLOPs overhead vs Policy-Mask | Reports resource cost |

### Why this setup validates the claim

The central claim is that optimizing selection for reconstructibility of the discarded complement changes *which* channels are chosen in a way that preserves more task-sufficient information at a fixed budget. ADE20K segmentation is the right testbed because its dense, spatially varying labels make channel-level information loss observable: if a selection drops channels carrying a rare-class signature, mIoU on that class drops even when global mIoU looks stable. The Policy-Mask baseline tests the sub-claim that downstream-only selection is insufficient — it shares architecture and budget but lacks the recovery signal, so any gap isolates the coupling itself. The fixed hand-picked baseline tests whether learned, reconstruction-coupled selection beats static heuristics that a practitioner would otherwise use, and the random baseline establishes the informativeness floor. The λ=0 ablation is the critical control: it keeps the decoder and straight-through path but removes L_rec, so it distinguishes "the decoder helps" from "the reconstruction objective reshapes selection," which is exactly our mechanism. The unweighted-L_rec ablation tests whether the task-sensitivity weighting is necessary to avoid the dominant-subspace failure mode. The linear probe on discarded channels directly measures task-relevant information in M, testing the load-bearing assumption that M is recoverable from S. The covariance rank/condition-number diagnostic tests the spanning property: if S spans M, the selected-vs-discarded covariance should have high rank and low condition number. The reconstruction-error-vs-budget diagnostic correlates recoverability with mIoU, showing whether the mechanism holds across budgets. mIoU at matched budget is the right metric because it is sensitive to the predicted effect (rare-class retention) while remaining comparable across all arms at identical channel counts.

### Expected outcome and causal chain

**vs. Policy-Mask** — On images containing rare classes whose discriminative signal concentrates in a few mid-level channels, Policy-Mask produces a selection that optimizes average downstream loss and discards those channels because they contribute little to the dominant-class gradient. Our method instead retains channels that make the discarded complement reconstructible, which forces the selection to span the discriminative subspace rather than collapse onto dominant modes because L_rec penalizes unrecoverable rare-class variation. We expect a noticeable rare-class mIoU gap (several points) with near-parity on dominant classes, and a larger overall gap at small budgets. The linear probe on discarded channels should show lower task-relevant information in M for ReCo than for Policy-Mask, and the covariance diagnostic should show higher rank/lower condition number for ReCo's selected set.

**vs. Fixed hand-picked selection** — On scenes where the optimal channel set shifts with timestep and block (e.g., texture-heavy vs. object-centric regions), the fixed baseline produces a static selection that is mismatched to the input, causing systematic errors on the region type it under-represents because it cannot adapt per (t, b). Our method instead adapts the selection via the budget-conditioned policy and is guided by reconstruction to keep complementary channels, because the straight-through gradient couples selection to recoverability. We expect the fixed baseline to trail uniformly but with a pronounced gap on the under-represented region type, and our gain to be concentrated there.

**vs. Random subset at same budget** — On any input, the random baseline produces an incoherent channel set that misses correlated feature groups, yielding high variance and poor mIoU because it ignores feature statistics entirely. Our method instead selects a coherent, reconstruction-coupled subset because the policy learns channel salience and L_rec enforces coverage of the discarded subspace. We expect a large, consistent margin over random that grows as budget shrinks, confirming the selection is informative rather than merely budget-matched.

**vs. ReCo without L_rec (λ=0)** — This ablation keeps the decoder and straight-through path but removes the recovery signal, so it isolates the effect of the reconstruction objective from the presence of the decoder. If the decoder alone were responsible for any gain, λ=0 would match full ReCo; we expect λ=0 to perform similarly to Policy-Mask, confirming that the reconstruction loss is the causal driver of selection reshaping.

**vs. ReCo with unweighted L_rec (w_c=1)** — This ablation tests whether task-sensitivity weighting is necessary to prevent L_rec from being dominated by high-variance directions. We expect unweighted L_rec to underperform full ReCo on rare-class mIoU, especially at small budgets, because it reconstructs dominant modes rather than task-relevant low-variance directions.

### What would falsify this idea

If ReCo's gain over Policy-Mask is uniform across all classes and budgets rather than concentrated on rare-class and small-budget subsets where the predicted information-loss failure occurs, then the reconstruction coupling is not reshaping selection as claimed. Equally falsifying: if the λ=0 ablation matches full ReCo, the recovery loss is not the causal driver; if the linear probe on discarded channels shows no reduction in task-relevant information for ReCo, the load-bearing assumption that M is recoverable from S is false.

## References

1. Policy-Mask: Budget-Aware Adaptation of Diffusion Features to Discriminative Tasks
2. Repurposing Diffusion-Based Image Generators for Monocular Depth Estimation
