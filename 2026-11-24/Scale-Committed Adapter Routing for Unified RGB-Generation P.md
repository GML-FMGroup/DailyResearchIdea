# Scale-Committed Adapter Routing for Unified RGB-Generation Perception

## Motivation

Unified RGB-generation perception frameworks such as Open Vision Banana (OVB) represent every perception output as a painted image, so absolute metric scale has no dedicated parameterization and never becomes an invariant of the shared representation — median-aligned NYUv2 δ_1 sits at 0.89–0.93 while unaligned δ_1 wanders between 0.11 and 0.50, and scale only improves via a ten-seed ensemble (0.348→0.516) rather than from the model itself. The same shared low-rank adapter is simultaneously contested by depth and 2D segmentation, so the one module that could carry a scale invariant is structurally overloaded. This is the meta-level bottleneck of unified visual perception: tasks whose outputs live in a common RGB surface inherit a representation in which task-specific invariants (here, metric scale) are emergent and sample-dependent rather than constrained outputs.

## Key Insight

Absolute scale is a one-dimensional gauge that can be factored out of the RGB-painted depth image and carried by a dedicated, explicitly metric-supervised scalar field, so committing it as a constrained output of the unified module removes the need for seed ensembling and frees the shared adapter from the depth/segmentation contention that currently erodes it.

## Method

**What it is.** SCAR (Scale-Committed Adapter Routing) augments the OVB diffusion generator with (i) a depth-branch low-rank adapter, (ii) a segmentation-branch low-rank adapter, and (iii) a metric-scale head that regresses a per-pixel log-scale field `s(u,v)`; the final metric depth is `exp(s) * D_paint`, where `D_paint` is the RGB-painted depth decoded by the task contract. The head is spatially conditioned and bias-aware: `s(u,v)` is predicted from depth-branch features and supervised with a per-pixel log-scale target derived from `D_gt / D_paint`, with degenerate regions (sky, textureless surfaces) masked out, and an explicit residual correction term `r(u,v)` is added so the head can absorb per-region bias rather than forcing it into one scene mean. The load-bearing assumption is stated explicitly: **A1: the RGB-painted depth `D_paint` is approximately scale-consistent within a scene up to a single multiplicative gauge, so that a per-pixel log-scale field supervised by a per-pixel target can recover the missing absolute scale and yield a single-sample unaligned δ_1 that beats OVB's ten-seed ensemble.** A1 fails when `D_paint` has systematic per-region bias (e.g., sky painted at near-zero depth), in which case the head reflects a bias-corrected gauge rather than the true scene scale; the residual term and masking are the concrete calibration story for this failure mode.

**How it works.**

```pseudocode
# Backbone: frozen Qwen-Image-2.1 (7B); all adapters low-rank (r=16, alpha=32)
# Task contract c in {depth, semantic, instance, referring}

function forward(x_rgb, instruction, c):
    h = frozen_diffusion_features(x_rgb, instruction)
    if c == depth:
        h_d = h + LoRA_depth(h)              # depth-only adapter branch
        D_paint = depth_decoder(h_d)          # contract RGB decoder -> relative depth image
        s = ScaleHead(h_d)                    # 2-layer MLP, 256 hidden, outputs 1 channel per pixel
        r = ResidualHead(h_d)                 # 2-layer MLP, 256 hidden, outputs 1 channel per pixel
        D_metric = exp(s + r) * D_paint
        return D_metric, s, r
    else:
        h_s = h + LoRA_seg(h)                 # segmentation-only adapter branch
        return seg_decoder(h_s)               # s is NOT produced for segmentation contracts

# Training (single stage, joint over all contracts)
for batch in loader:                          # mixed depth + segmentation batches
    if batch.task == depth:
        D_metric, s, r = forward(...)
        # per-pixel log-scale target from GT/painted ratio, masked for degenerate regions
        target = log(D_gt / D_paint)
        mask = valid_depth_mask(D_gt) & ~sky_mask(D_gt) & ~textureless_mask(D_gt)
        L_metric = |D_metric - D_gt|_1                       # absolute metric depth (meters)
        L_si     = SI_log(D_metric, D_gt)                     # scale-invariant log loss
        L_scale  = |(s + r - target) * mask|_1 / sum(mask)    # per-pixel gauge supervision
        L = L_metric + 0.5*L_si + 0.1*L_scale
    else:
        L = task_loss(forward(...))                           # contract loss (CE / mask BCE)
    update(LoRA_depth, LoRA_seg, ScaleHead, ResidualHead, decoders)  # backbone frozen
```

Hyperparameters: LoRA rank 16, α=32; ScaleHead and ResidualHead each 2-layer MLP (256 hidden, GELU, linear output); loss weights (1.0 metric, 0.5 SI-log, 0.1 scale-gauge); AdamW lr 1e-4 for adapters, 5e-4 for ScaleHead and ResidualHead; 100k updates, batch 64. Sky and textureless masks are computed from GT depth (sky: depth > 50 m or invalid; textureless: local depth gradient < 0.01 m/pixel over a 5×5 window).

**Why this design.** We route depth and segmentation through *separate* low-rank adapters rather than the single shared adapter of OVB because OVB's own weight-merge experiments show depth and 2D segmentation competing for adapter capacity, accepting the cost of ~2× adapter parameters and a slightly larger checkpoint. We factor scale multiplicatively as `exp(s + r) * D_paint` rather than regressing metric depth directly because the RGB-generation contract already learns relative geometry well (median-aligned δ_1 ≈ 0.9) while failing only on the gauge, accepting the cost that the head cannot correct structural depth errors. We supervise the per-pixel log-scale field against a per-pixel target derived from `D_gt / D_paint` with degenerate regions masked, rather than relying on the metric loss alone or on a single scene-mean gauge, because the metric L1 gradient on `s` is diluted by per-pixel noise and because the painted depth carries region-dependent bias that a single scalar cannot absorb; the residual term `r` is the explicit bias-correction channel. We deliberately do *not* follow the seed-ensembling route of OVB (which lifts unaligned δ_1 from 0.348 to 0.516 without retraining) because that route leaves scale as a post-hoc statistical property; a domain expert would not describe SCAR as a variant of ensemble averaging or of the shared-adapter recipe, since the scale gauge is a trained, contract-attached parameter rather than a sample statistic.

**Why it measures what we claim.** The quantity `mean(s + r)` measures *committed absolute scale* because the assumption A1 that the painted depth `D_paint` is approximately scale-consistent within a scene makes the scene-level mean of the log-scale field equal to the missing gauge; this assumption fails when the generator's painted depth has systematic per-region bias (e.g., sky regions painted at near-zero depth), in which case `mean(s + r)` reflects a bias-corrected gauge rather than the true scene scale. The quantity `L_scale` measures *gauge supervision* because the assumption that ground-truth metric depth yields a well-defined per-pixel log-scale target makes the target identifiable; this assumption fails for scenes with mixed-scale objects (a close-up object at 0.3 m next to a 50 m background), in which case the target reflects a scene-average compromise rather than a single physical scale. The quantity unaligned δ_1 measures *scale commitment* because the assumption that δ_1 is monotone in the multiplicative gauge error makes unaligned δ_1 a direct readout of scale accuracy; this assumption fails when depth errors are structural rather than multiplicative, in which case unaligned δ_1 under-reports the benefit of a correct gauge — which is why we also report `mean(s + r)` error separately.

## Contribution

(1) SCAR, a scale-committed adapter-routing framework for RGB-generation perception that factors absolute metric scale into an explicitly supervised per-pixel log-scale head attached to a depth-only low-rank adapter, with segmentation contracts routed to a disjoint adapter branch. (2) The design principle that task-specific invariants in a unified RGB-output perception model should be parameterized as explicit, separately supervised factors rather than left to emerge from the shared generation contract — demonstrated concretely for the metric-scale gauge. (3) A diagnostic decomposition of unaligned depth error into a gauge component (`mean(s)` error) and a structural component, enabling attribution of OVB's aligned/unaligned gap.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | NYUv2 + segmentation contracts (OVB public mix) | Same public data as OVB; isolates adapter/scale change |
| Primary metric | Unaligned δ_1 (plus mean(s+r) gauge error) | Direct readout of committed absolute scale |
| Baseline | OVB shared-adapter | Tests whether routing beats capacity sharing |
| Baseline | OVB ten-seed ensemble | Tests whether trained gauge beats post-hoc statistics |
| Baseline | OVB single shared adapter + direct metric-depth regression | Tests multiplicative factorisation vs direct regression |
| Ablation-of-ours | SCAR without L_scale gauge loss | Isolates the trained gauge from metric L1 alone |
| Ablation-of-ours | SCAR without ResidualHead (r=0) | Isolates the bias-correction term from the scale field |
| Diagnostic | Within-scene scale-consistency of D_paint | Grounds assumption A1 empirically |

### Why this setup validates the claim

The central claim is that scale commitment is a *learnable, contract-attached* quantity, and that separating depth and segmentation capacity plus factoring scale multiplicatively is what lets the model commit it. The OVB shared-adapter baseline tests the routing sub-claim: if depth and segmentation genuinely compete for one adapter (as OVB's own merge experiments suggest), then giving each its own low-rank branch should free depth capacity and show up as better structural depth, not just better scale. The ten-seed ensemble baseline tests the gauge sub-claim directly: it achieves higher unaligned δ_1 *without training*, so if SCAR's trained gauge is real it must beat the ensemble at matched training budget, and if it only matches the ensemble then scale is still a sample statistic and the design rationale is wrong. The direct-regression baseline tests the factorisation sub-claim: it can in principle fix structural errors too, so if SCAR's multiplicative form is the right inductive bias, direct regression should show *worse* median-aligned δ_1 (it throws away the well-learned relative geometry) while SCAR should show better unaligned δ_1 at comparable median-aligned δ_1. The no-gauge-loss ablation isolates whether L_scale does anything: if metric L1 alone already commits scale, the ablation matches SCAR and the dedicated gauge loss is unjustified. The no-residual ablation isolates whether the bias-correction term is necessary: if SCAR without r matches SCAR, then the per-region bias is negligible and the simpler scale field suffices. The within-scene scale-consistency diagnostic directly tests assumption A1: we regress a single per-scene scale against GT depth and report residual variance per region (including sky), so the claim that a per-pixel log-scale field can recover the gauge is empirically grounded rather than hand-waved. Unaligned δ_1 is the right primary metric because it is the only one that reads out the gauge; median-aligned δ_1 would mask exactly the failure SCAR targets, which is why it is reported alongside rather than as the primary.

### Expected outcome and causal chain

**vs. OVB shared-adapter** — On a scene with a large sky region and a nearby foreground object, the shared adapter must split its low-rank capacity between depth structure and segmentation boundaries, so its painted depth retains per-region bias and its unaligned δ_1 stays in the low-to-mid range despite median-aligned δ_1 near 0.9. Our method routes depth through LoRA_depth only, so the depth branch spends all its capacity on geometry and the ScaleHead sees cleaner features; we expect a modest but consistent unaligned δ_1 gain concentrated on scenes with large homogeneous regions, with near-parity on cluttered indoor scenes where the shared adapter was not capacity-starved.

**vs. OVB ten-seed ensemble** — On a single-sample evaluation of a scene whose painted depth is scale-consistent but whose gauge is arbitrary, the ensemble reduces gauge variance only by averaging independent samples, so it lifts unaligned δ_1 from roughly 0.35 toward 0.5 but cannot exceed the per-sample bias floor. Our method trains the per-pixel log-scale field against a per-pixel target derived from `D_gt / D_paint` with degenerate regions masked, so a single forward pass commits a gauge rather than averaging one; we expect SCAR at single-sample to match or exceed the ten-seed ensemble's unaligned δ_1, with the gap largest on scenes where the ensemble's seed spread is widest.

**vs. OVB shared adapter + direct metric-depth regression** — On a scene with fine structural detail (thin furniture legs, door frames), direct regression must relearn relative geometry from scratch through a metric L1 objective, so its median-aligned δ_1 degrades relative to the RGB-painting contract while its unaligned δ_1 may improve. Our method keeps D_paint from the well-learned contract and only corrects the gauge, so we expect SCAR to preserve median-aligned δ_1 near the contract's 0.9 while improving unaligned δ_1 — the diagnostic signature is *high median-aligned, high unaligned*, which direct regression should not achieve simultaneously.

**vs. SCAR without L_scale gauge loss** — On a scene where the painted depth is scale-consistent but the gauge is arbitrary, the ablation must rely on the metric L1 gradient alone, which is diluted by per-pixel noise, so its unaligned δ_1 should remain near the shared-adapter baseline. Our method adds the per-pixel gauge supervision, so we expect a clear unaligned δ_1 separation between SCAR and the ablation, with the gap largest on scenes where the metric L1 gradient is weakest (low-texture, large homogeneous regions).

**vs. SCAR without ResidualHead (r=0)** — On a scene with systematic per-region bias (e.g., sky painted near zero), the no-residual variant must force the bias into the scale field `s`, so its unaligned δ_1 should degrade relative to full SCAR. Our method adds the residual term `r` to absorb per-region bias, so we expect full SCAR to outperform the no-residual ablation on scenes with large sky or textureless regions, with near-parity on cluttered scenes where the bias is small.

### What would falsify this idea

If SCAR's unaligned δ_1 gain over the shared-adapter baseline is uniform across scene types rather than concentrated on scenes with large homogeneous regions and wide seed spread — or if the no-gauge-loss ablation matches SCAR, or if the within-scene scale-consistency diagnostic shows residual variance dominated by per-region bias that the residual term cannot absorb — then scale commitment is not being learned by the gauge head and the central claim is wrong.

## References

1. Open Vision Banana: An Open Framework for Unified Visual Perception through Generation in RGB Space
