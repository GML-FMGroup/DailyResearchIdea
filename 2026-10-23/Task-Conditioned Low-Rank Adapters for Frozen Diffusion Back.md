# Task-Conditioned Low-Rank Adapters for Frozen Diffusion Backbones: Verifying Backbone-Adaptation Sufficiency via Budget-Matched Residual Probing

## Motivation

Policy-Mask selects which frozen diffusion features to read at a fixed channel budget, but because the backbone is never updated, its selection can only re-weight features that were never calibrated to the discriminative task — the shared residual gap that recurs across frozen-backbone transfer methods (Policy-Mask) and full-fine-tuning methods that entangle the generative prior with the task (Marigold). The structural root cause is that selection and adaptation are decoupled: a budget-aware selection policy cannot detect whether the remaining frozen features are sufficient, because there is no mechanism that injects task signal into the backbone's internal representations and then measures the residual that selection must absorb. This meta-level bottleneck — no operational test of backbone-adaptation sufficiency under a fixed parameter budget — is what the proposed adapter is designed to close.

## Key Insight

If a task-conditioned low-rank update is applied only to the residual stream of frozen diffusion blocks, the residual reconstruction error after the update is a measurable proxy for backbone-adaptation sufficiency, because a low-rank null-space projection makes that error invariant to which features Policy-Mask selects and sensitive only to the task-directional component the frozen features lack.

## Method

**What it is.** TALA (Task-conditioned Adapter with Low-rank Adaptation) takes as input a frozen diffusion backbone, a downstream task head, and the budget-conditioned selection mask produced by Policy-Mask; it outputs a set of low-rank residual updates applied to the frozen block residual streams, plus a scalar sufficiency residual $\rho$ reported per block. The adapter is trained jointly with the selection policy but is the only component with trainable parameters inside the backbone.

**How it works.**
```pseudocode
Input: frozen diffusion backbone B with blocks b=1..L, task head H,
       Policy-Mask selector S (frozen during adapter warmup),
       channel budget C, rank r=8, warmup steps W=500,
       held-out probe set D_probe of 512 examples

# Phase 1: adapter warmup (selector frozen)
for step in 1..W:
    x = batch; t = sampled timestep
    h_b = B.block_b(x, t)                     # frozen residual stream, b in 1..L
    mask = S(x, C)                            # budget-conditioned selection (b, t, channel)
    for each selected block b:
        delta_b = U_b @ (V_b @ h_b)           # U_b: d x r, V_b: r x d, zero-init V_b
        h_b' = h_b + mask_b * delta_b         # mask_b is the selected channel subset
    loss = H(h_b') + lambda * ||V_b||_2^2     # lambda = 1e-4 weight decay on low-rank factors
    update U_b, V_b only

# Phase 2: joint refinement (selector unfrozen)
for step in W+1..T:
    same forward; update U_b, V_b and S jointly

# Phase 3: sufficiency residual report (gradient-free, held-out probe)
# Freeze all adapter parameters and selector S after training.
# For each block b, compute the task loss on D_probe with the adapter enabled
# and with the adapter disabled (delta_b = 0), all other blocks unchanged.
for each block b:
    L_enabled_b  = mean_{(x,y) in D_probe} H( forward_with_delta_b(x) )
    L_disabled_b = mean_{(x,y) in D_probe} H( forward_without_delta_b(x) )
    L_no_adapter = mean_{(x,y) in D_probe} H( forward_without_all_adapters(x) )
    r_b = (L_disabled_b - L_enabled_b) / L_no_adapter   # recoverable error reduction per block
    rho = mean_b r_b                          # scalar sufficiency residual
    if rho < tau (tau=0.05): flag block b as insufficient
# Additionally, report rho_full on the full channel set (not only masked channels)
# by evaluating the same leave-one-block-out probe with delta_b applied to all channels.
```
The low-rank factors are initialized at zero so the backbone is exactly recovered at step 0; the mask multiplication ensures the adapter only writes to channels the selection policy can read, so the residual is measured on the same surface selection operates on. The sufficiency residual $\rho$ is defined as the held-out, gradient-free recoverable error reduction per block, not as the norm of the learned update, so it is invariant to rank, weight decay, and mask truncation by construction.

**Why this design.** We chose a zero-initialized low-rank residual update over full fine-tuning (Marigold) because full fine-tuning entangles the generative prior with the task and destroys the frozen-feature invariance that Policy-Mask relies on, accepting the cost that the adapter's capacity is bounded by rank $r$ and cannot recover task directions outside the low-rank subspace. We chose to multiply the adapter output by the Policy-Mask selection mask rather than letting the adapter write to all channels because an unmasked adapter would make the sufficiency residual sensitive to channels the selector cannot access, accepting the cost that the residual under-reports insufficiency in unselected channels. We chose to freeze the selector during a 500-step warmup rather than train both from scratch because joint training from random low-rank factors lets the selector compensate for adapter noise, accepting the cost of a two-phase schedule and a hyperparameter (warmup length) that must be tuned per task. We chose a held-out recoverable-error residual $\rho$ over a learned sufficiency classifier because a learned classifier would reintroduce the very entanglement we are trying to avoid, accepting the cost that $\rho$ is a coarse scalar rather than a per-channel diagnostic. A domain expert would not describe this as Policy-Mask plus LoRA: Policy-Mask's selection head is a read-side operator that never changes the backbone, whereas TALA's residual is a write-side operator whose magnitude is the sufficiency signal — the two compose but neither subsumes the other.

**Why it measures what we claim.** The held-out recoverable-error residual $\rho = \frac{1}{L}\sum_b (L_{\text{disabled},b} - L_{\text{enabled},b}) / L_{\text{no-adapter}}$ measures backbone-adaptation sufficiency because it directly quantifies the task error that the adapter can recover on held-out data when block $b$ is updated, so any task signal the frozen features already contain yields a small recoverable error reduction (near zero $\rho$), while task signal absent from the frozen features forces a large recoverable reduction and inflates $\rho$; this assumption fails when the downstream task direction is orthogonal to the rank-$r$ subspace, in which case $\rho$ reflects the rank deficiency of the adapter rather than the insufficiency of the frozen backbone. The mask-multiplied adapter output measures selection-adaptation coupling because the mask is the exact channel subset Policy-Mask can read, so the residual is computed on the same surface the selector operates on; this assumption fails when the selector's budget $C$ is smaller than the adapter's effective rank, in which case the residual reflects budget truncation rather than backbone insufficiency. The per-block residual $r_b$ measures block-level adaptation need because each block's residual stream is updated independently and the leave-one-block-out probe isolates its contribution, so a block that receives no task-relevant gradient retains $r_b \approx 0$; this assumption fails when gradients from the task head are dominated by a single block, in which case downstream blocks show artificially low $r_b$ regardless of their true insufficiency.

## Contribution

(1) A task-conditioned low-rank adapter (TALA) that writes budget-masked residual updates into frozen diffusion blocks, making backbone adaptation and feature selection operate on the same channel surface. (2) A sufficiency residual $\rho$ computed from the adapter's relative update magnitude, providing an operational per-block test of whether frozen diffusion features are sufficient for a discriminative task without full fine-tuning. (3) A two-phase warmup schedule that decouples adapter fitting from selector training, preventing the selection policy from compensating for adapter noise during joint optimization.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|------------------------|
| Dataset | NYUv2 depth + ADE20K segmentation | Two tasks test sufficiency across dense prediction types |
| Primary metric | Task accuracy at matched channel budget | Directly tests whether adapter recovers task signal |
| Baseline | Policy-Mask (frozen backbone, no adapter) | Isolates whether write-side residual adds value |
| Baseline | Marigold full fine-tuning | Tests whether frozen-feature invariance is preserved |
| Baseline | LoRA on all channels (no mask) | Tests whether mask multiplication is essential |
| Ablation-of-ours | TALA without sufficiency residual (ρ removed) | Tests whether ρ flags true insufficiency |
| Ablation-of-ours | TALA with raw update-norm ρ (original confounded proxy) | Tests whether held-out recoverable-error ρ is necessary |
| Diagnostic | ρ vs. task-gradient projection onto frozen subspace | Separates rank/budget deficiency from backbone insufficiency |

### Why this setup validates the claim

This design creates a falsifiable test of the central claim: that a masked low-rank residual update on a frozen diffusion backbone recovers task signal that Policy-Mask's read-only selection misses, while preserving the frozen-feature invariance that full fine-tuning destroys. The Policy-Mask baseline tests the sub-claim that read-side selection alone is insufficient: if TALA's gain over Policy-Mask is near zero across both tasks, the write-side residual adds nothing. The Marigold baseline tests the sub-claim that frozen-feature invariance matters: if Marigold matches or beats TALA on task accuracy while showing much larger ρ, then entanglement is not a real cost. The unmasked-LoRA baseline tests the sub-claim that mask multiplication is essential: if it matches TALA, the mask is decorative. The no-ρ ablation tests whether the sufficiency signal is diagnostic: if removing ρ does not change which blocks are flagged as insufficient, ρ is noise. The raw-update-norm ablation tests whether the held-out recoverable-error definition of ρ is necessary: if the raw update-norm ρ flags the same blocks and correlates equally with true insufficiency, the confound is not material. The diagnostic ρ vs. task-gradient projection tests the load-bearing assumption directly: if ρ tracks the projection of the task gradient onto the orthogonal complement of the frozen feature subspace while remaining flat under rank and budget changes, then ρ separates backbone insufficiency from rank/budget deficiency. Task accuracy at matched channel budget is the right metric because it directly measures whether the adapter recovers the task signal that the frozen backbone lacks, and because budget matching prevents the trivial explanation that TALA simply uses more channels.

### Expected outcome and causal chain

**vs. Policy-Mask (frozen backbone, no adapter)** — On a case where the downstream task requires a feature direction orthogonal to the frozen backbone's principal subspace (e.g., fine depth boundaries in NYUv2), Policy-Mask produces blurred boundaries because its read-side selection can only reweight existing channels, not synthesize missing directions. Our method instead writes a low-rank residual into the selected channels because the zero-initialized adapter can inject task-specific directions within the mask surface, so we expect a noticeable accuracy gap on boundary-heavy subsets but parity on texture-dominated subsets where frozen features already suffice.

**vs. Marigold full fine-tuning** — On a case where the synthetic-to-real domain shift is large (e.g., real NYUv2 images after synthetic training), Marigold produces overconfident depth errors because full fine-tuning entangles the generative prior with task-specific corrections, destroying the frozen-feature invariance that supports generalization. Our method instead keeps the backbone frozen and applies only a rank-8 residual because the low-rank constraint limits entanglement, so we expect Marigold to win on in-domain synthetic accuracy but TALA to close or reverse the gap on real-domain subsets, with TALA showing much lower ρ on blocks that Marigold over-adapts.

**vs. LoRA on all channels (no mask)** — On a case where the selector's budget C is much smaller than the adapter's effective rank (e.g., low-budget segmentation), unmasked LoRA produces large residuals in channels the selector cannot read because it writes freely across all channels, inflating ρ without improving task accuracy. Our method instead multiplies the adapter output by the selection mask because the residual is constrained to the selector's read surface, so we expect unmasked LoRA to show higher ρ but no accuracy gain over TALA at low budgets, and parity only at high budgets where the mask becomes non-binding.

**vs. TALA with raw update-norm ρ** — On a case where the task direction is orthogonal to the rank-8 subspace (e.g., a held-out synthetic rotation experiment), the raw update-norm ρ is confounded by rank deficiency and weight decay, so it inflates even when the frozen backbone is sufficient. Our method instead uses the held-out recoverable-error ρ because it directly measures task error reduction, so we expect the raw-norm ρ to show high correlation with adapter rank and weight decay but low correlation with true insufficiency, while the recoverable-error ρ shows high correlation with true insufficiency and remains flat under rank and budget changes.

### What would falsify this idea

If TALA's accuracy gain over Policy-Mask is uniform across boundary-heavy and texture-dominated subsets rather than concentrated where frozen features lack task directions, or if the no-ρ ablation flags the same insufficient blocks as full TALA, or if the held-out recoverable-error ρ does not track the projection of the task gradient onto the orthogonal complement of the frozen feature subspace while remaining flat under rank and budget changes, then the central claim that the masked residual measures backbone-adaptation sufficiency is wrong.

## References

1. Policy-Mask: Budget-Aware Adaptation of Diffusion Features to Discriminative Tasks
2. Repurposing Diffusion-Based Image Generators for Monocular Depth Estimation
