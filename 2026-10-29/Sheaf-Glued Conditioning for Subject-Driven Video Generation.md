# Sheaf-Glued Conditioning for Subject-Driven Video Generation

## Motivation

Existing subject-driven video generators such as OmniShowcase and MAGREF condition the diffusion transformer through an open-ended set of injected maps (surface-normal turntables, masked reference features), with no defined composition operator, no conflict-resolution semantics, and no guarantee that adding a new 3D constraint type preserves the interface's closure. This structural gap means that when constraints disagree (e.g., a depth map implying a different camera pose than the reference identity), the model must implicitly learn ad-hoc arbitration from data rather than being given a principled gluing rule.

## Key Insight

The conditioning interface becomes closed under composition by construction when each 3D constraint is represented as a section of a constraint sheaf over the video token grid, because sheaf restriction and gluing provide a canonical composition operator and the non-vanishing first cohomology cocycle is a measurable obstruction that quantifies constraint conflict without retraining the backbone.

## Method

**What it is** — We introduce SheafCon, a closed conditioning algebra for subject-driven video diffusion transformers. SheafCon takes as input a set of 3D constraints (surface-normal turntable, depth maps, camera pose, reference identity features) and outputs a single glued conditioning tensor plus an obstruction residual, both consumed through one conditioning port of a pretrained video DiT (e.g., Wan). The load-bearing assumption is stated explicitly: the obstruction cocycle computed from pairwise restriction-map disagreements between overlapping spatiotemporal patches is a faithful, backbone-usable measure of semantic 3D constraint conflict, i.e., that ||R_{p->q}(s_k(p)) - R_{q->p}(s_k(q))||^2 on geometrically grounded restriction maps tracks real geometric incompatibility (e.g., depth-implied pose vs. reference identity) rather than arbitrary feature-space mismatch. We make this assumption testable by grounding the restriction maps in the known camera pose and depth (reprojection), so the cocycle is a reprojection residual with physical units, and by validating it against known injected conflict in a synthetic calibration experiment (see experiment).

**How it works** — The video token grid is partitioned into spatiotemporal patches: 16×16 pixel patches across 8-frame windows (stride 8 pixels, 4 frames). Each constraint type defines a sheaf over this grid: a presheaf assigns to each patch a local feature vector (e.g., normal direction (3D), depth value (1D), reference-identity embedding (512D)) and to each overlapping patch pair a restriction map. Restriction maps are geometrically grounded: for a pair of patches p, q with known camera pose T and depth D, R_{k,p->q} warps the local feature from p to q using the camera-pose-induced homography H_{p->q} = K (R + t n^T / d) K^{-1} derived from T and the mean depth d of patch p, followed by bilinear sampling; for identity features, which are not geometric, R_{k,p->q} is the identity on the overlapping region (no warp). This makes the cocycle a measurable reprojection residual with physical units (pixels or depth units) rather than an uncalibrated learned distance.

```pseudocode
# Input: list of constraints C = [c1, ..., cK], each with local features F_k(p) for patch p
# Hyperparameters: patch 16x16 px, window 8 frames, stride 8 px / 4 frames
# Step 1: Build the constraint sheaf
for each constraint k:
    assign local section s_k(p) = F_k(p) for each patch p
    define restriction map R_{k, p->q} for overlapping patches p, q:
        if k is geometric (normal, depth, pose):
            R_{k, p->q}(x) = bilinear_sample(warp(x, H_{p->q}))
            where H_{p->q} = K (R + t n^T / d) K^{-1} from known camera pose T and mean depth d of patch p
        else (identity features):
            R_{k, p->q}(x) = x on overlap, 0 elsewhere

# Step 2: Glue sections via sheaf gluing (true gluing where cocycle is small)
initialize glued section G(p) = 0, weight W(p) = 0
for each constraint k:
    for each patch p:
        G(p) += w_k * s_k(p)   # w_k is a learned per-constraint scalar (softmax over constraints)
        W(p) += w_k
G(p) = G(p) / W(p)
# True gluing: where obstruction(p,q) < tau (tau = 0.05 in normalized units), G(p) is the unique section;
# where obstruction >= tau, fall back to learned arbitration: G(p) = sum_k a_k(p) s_k(p) with a_k(p) = softmax_k(-||delta_k(p,q)||^2 / sigma)

# Step 3: Compute obstruction cocycle (conflict residual)
for each overlapping patch pair (p, q):
    for each constraint k:
        delta_k(p,q) = R_{k, p->q}(s_k(p)) - R_{k, q->p}(s_k(q))
    obstruction(p,q) = sum_k w_k * ||delta_k(p,q)||^2
aggregate obstruction residual O = mean over pairs

# Step 4: Conditioning port
conditioning_tensor = concat(G, O)   # channel-wise; O is broadcast to all spatial positions
feed conditioning_tensor into the video DiT via a single cross-attention port (or channel concatenation)
```

**Why this design** — We chose a sheaf-theoretic gluing over a fixed set of injected maps because sheaf restriction and gluing give a canonical composition operator that is closed under adding new constraint types (each new constraint is a new sheaf morphism), accepting the cost that the gluing weights w_k must be learned and may not be optimal for all constraint combinations. We chose to compute the obstruction cocycle explicitly rather than letting the network implicitly resolve conflicts because the cocycle norm provides a measurable, interpretable residual that can be used to modulate the conditioning strength, accepting the cost of additional computation at each forward pass. We chose to feed the glued section and obstruction residual through a single conditioning port rather than separate injection branches because this preserves the pretrained backbone's interface and avoids retraining, accepting the cost that the port must handle a higher-dimensional input (concatenated features). This design is not a variant of MAGREF's masked guidance or OmniShowcase's surface-normal turntable injection: those methods treat each constraint as an independent map with no composition operator, whereas SheafCon defines composition as sheaf gluing and conflict as a cohomological obstruction, which is a structural property that prior work does not produce.

**Why it measures what we claim** — The obstruction residual O measures constraint conflict because the assumption that overlapping patch restrictions should agree (the sheaf condition) is equivalent to geometric consistency when restriction maps are grounded in known camera pose and depth (reprojection); this assumption fails when constraints are fundamentally incompatible (e.g., a depth map implying a different camera pose than the reference identity), in which case O reflects the magnitude of that incompatibility in physical units rather than a learned artifact. The glued section G measures the consensus conditioning because the weighted average over constraints with learned weights approximates a maximum-likelihood fusion under the assumption that each constraint provides an unbiased estimate of the true local feature; this assumption fails when one constraint is systematically biased (e.g., noisy depth), in which case G reflects a biased consensus unless w_k downweights it. We test the unbiasedness assumption by injecting known bias and measuring how learned w_k responds (see experiment).

## Contribution

['A sheaf-theoretic conditioning algebra (SheafCon) that represents each 3D constraint as a section over the video token grid and composes them via gluing, with the first cohomology cocycle providing a measurable obstruction residual for constraint conflict.', 'A single conditioning port design that consumes the glued section and obstruction residual, enabling the addition of new 3D constraint types as new sheaf morphisms without retraining the backbone.', 'An empirical demonstration that SheafCon improves subject consistency and geometric fidelity in omnidirectional subject-driven video generation compared to fixed-map injection baselines, and that the obstruction residual correlates with visible artifacts.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Multi-view subject video benchmark with held-out camera orbits; conflict-heavy subset synthesized by injecting depth maps with known pose offsets (0°, 5°, 10°, 15°, 20°) | Tests novel-view subject consistency where conflicts arise; known conflict magnitude enables calibration |
| Primary metric | Subject-consistency score on conflict-heavy orbit subsets; plus calibration plot of obstruction residual O vs. known pose offset | Directly sensitive to constraint-conflict resolution; validates O tracks geometric conflict |
| Baseline | OmniShowcase | Independent per-constraint injection, no composition operator |
| Baseline | MAGREF-style masked guidance | Masked guidance without explicit conflict residual |
| Baseline | Wan (unmodified backbone) | Pretrained port capacity ceiling, no 3D constraints |
| Baseline | Plain weighted-average fusion (no sheaf, no obstruction) | Tests whether algebra adds beyond simple fusion |
| Ablation-of-ours | SheafCon w/o obstruction residual (glued section only) | Isolates contribution of explicit cocycle signal |
| Ablation-of-ours | SheafCon with learned linear restriction maps (no geometric grounding) | Tests whether geometric grounding of R is necessary |

### Why this setup validates the claim

The central claim is that sheaf gluing plus an explicit obstruction cocycle yields a closed composition operator that resolves inter-constraint conflicts better than independent injection. The dataset must contain instances where constraints genuinely disagree (e.g., a depth map whose implied pose contradicts the reference identity's viewpoint), because only there can the obstruction term matter. We synthesize these by injecting depth maps with known pose offsets (0° to 20°), which provides ground-truth conflict magnitude for calibration. OmniShowcase tests the sub-claim that independent per-constraint maps lack a composition operator; MAGREF tests whether masked guidance without an explicit residual can implicitly resolve the same conflicts; unmodified Wan establishes the backbone's port-capacity ceiling so any gain is attributable to SheafCon's conditioning algebra rather than the base model. The plain weighted-average baseline tests whether the sheaf structure (geometric restriction maps and true gluing) adds anything beyond simple fusion. The ablation without obstruction isolates the cocycle, and the ablation with learned linear restriction maps tests whether geometric grounding is necessary for O to track conflict. Subject-consistency on conflict-heavy orbits is the right metric because it is exactly the observable that should degrade when constraints are incompatibly fused, and it is measured on the subset where the predicted failure mode is triggered rather than averaged away. The calibration plot directly tests the load-bearing assumption that O is monotonic in known conflict magnitude.

### Expected outcome and causal chain

**vs. OmniShowcase** — On a case where a surface-normal turntable implies a different subject orientation than the reference identity image, OmniShowcase produces identity drift and normal-map smearing because each constraint is injected as an independent map with no composition operator to arbitrate. Our method instead computes a glued section weighted by learned per-constraint scalars and a cocycle norm that flags the orientation conflict, so we expect a clear consistency gap on conflict-heavy orbit segments but near-parity on single-constraint segments.

**vs. MAGREF-style masked guidance** — On a case where a noisy depth map systematically contradicts the identity features, masked guidance produces a biased consensus because masking suppresses conflicting regions without measuring the conflict magnitude. Our method instead exposes the incompatibility through the obstruction residual and modulates conditioning strength, so we expect a moderate gain concentrated on noisy-depth instances and little difference where constraints already agree.

**vs. Wan (unmodified backbone)** — On any constrained orbit, unmodified Wan produces plausible but geometrically unanchored video because its single conditioning port receives no 3D constraint signal. Our method instead feeds the glued tensor and residual through that same port, so we expect a large gap on all constrained orbits, with the gap narrowing only if the port saturates — a ceiling we can observe directly.

**vs. Plain weighted-average fusion** — On conflict-heavy instances, plain weighted-average fusion produces a blurred consensus because it always averages regardless of conflict, whereas our method uses true gluing where the cocycle is below threshold and learned arbitration only on obstructed patches. We expect our method to outperform plain fusion specifically on high-conflict subsets, demonstrating that the sheaf algebra adds beyond simple fusion.

**vs. SheafCon w/o obstruction residual** — Without the obstruction residual, the model cannot modulate conditioning strength on conflicting patches, so we expect degraded subject-consistency on conflict-heavy subsets, confirming the cocycle's contribution.

**vs. SheafCon with learned linear restriction maps** — With learned linear restriction maps, the cocycle is an uncalibrated feature discrepancy and may not track geometric conflict; we expect the calibration plot to show non-monotonicity and worse consistency on conflict-heavy subsets compared to geometrically grounded maps, validating the load-bearing assumption.

### What would falsify this idea

If the full SheafCon matches the no-obstruction ablation across all conflict-heavy subsets, the explicit cocycle is not doing the claimed work and the central mechanism is wrong. If the obstruction residual O is not monotonic in the known injected pose offset (calibration plot flat or non-monotonic), the load-bearing assumption that O tracks geometric conflict is false. If the gain over OmniShowcase is uniform across conflict-heavy and conflict-free subsets rather than concentrated where constraints disagree, the improvement is attributable to extra capacity, not to sheaf composition.

## References

1. OmniShowcase: Omnidirectional Subject-Consistent Video Generation from Multi-View Images
2. Wan: Open and Advanced Large-Scale Video Generative Models
3. VGGT: Visual Geometry Grounded Transformer
4. MAGREF: Masked Guidance for Any-Reference Video Generation with Subject Disentanglement
