# Cycle-Consistent Self-Distillation for Layered Image Editing from Unpaired Data

## Motivation

Existing layered instruction-based editing methods, such as the Compositional Edit-Prior Distillation framework of the reference paper, rely on paired RGBA layer annotations and a strong teacher model that emits layer supervision at every token position. This dependence is structurally limiting: the multi-modal distillation loss is defined only over observed token positions and requires paired layer targets, so the editing objective is not identifiable from unpaired images or weak instructions. As a result, the method cannot close over unpaired edit distributions, which is exactly the regime where most real-world images and instructions live.

## Key Insight

The composed RGBA layer output is invariant to the order and stochasticity of the decomposition when the layers are truly consistent with the edited canvas, so agreement between multiple stochastic decompositions of the same edit provides a supervision signal that replaces the external teacher oracle without paired RGBA annotations.

## Method

```python
# Inputs: unpaired image x, instruction t; student editing model S_theta
# Output: edited canvas c and ordered RGBA layers {L_k}
#
# Load-bearing assumption (stated explicitly): agreement across the K stochastic
# forward passes is a valid proxy for layer-decomposition correctness only if
# S_theta's sampling distribution concentrates on the true decomposition rather
# than on a self-consistent but wrong one. Because the cycle/agreement objective
# has trivial fixed points (identity edit, single opaque layer, collapsed
# deterministic decomposition), we ground the signal with two non-degenerate
# anchors that are part of the existing objective: (i) a deterministic ordering
# cue from the input image (occlusion-boundary ordering) that fixes layer order,
# and (ii) a variance-preserving repulsion term across the K passes that forbids
# collapse to a single deterministic decomposition. The composed canvas is NOT
# assumed invariant to ordering for semi-transparent layers; the ordering cue
# supplies the missing order information, and the agreement term is evaluated
# only on the composed canvas after ordering is fixed by the cue.

for each unpaired (x, t):
    # Phase 0: deterministic ordering cue from input image
    # depth-from-monocular (MiDaS small, frozen) gives per-pixel depth d(x);
    # occlusion boundaries from d(x) give a reference ordering pi_ref over
    # connected regions. This is a cheap deterministic cue, not a learned module.
    pi_ref = ordering_from_depth(d(x))          # permutation over regions

    # Phase 1: multiple stochastic forward passes
    for s in 1..K:                       # K = 4 stochastic samples
        (c_s, {L_k^s}) = S_theta(x, t; noise=eps_s)

    # Phase 2: recomposition and re-decomposition consistency
    recomposed_s = compose({L_k^s})       # alpha-composite in order pi_ref
    (c_s', {L_k^s'}) = S_theta(recomposed_s, t; noise=eps_s')  # re-edit recomposed

    # Phase 3: self-distillation losses
    L_cycle_canvas = || c_s - c_s' ||_1                 # cycle-consistency of canvas
    L_cycle_layers = sum_k || L_k^s - L_k^s' ||_1       # layer cycle-consistency
    L_agreement    = sum_{s<s'} || compose({L_k^s}; pi_ref) - compose({L_k^{s'}}; pi_ref) ||_1  # cross-pass agreement on ordered composition
    L_order        = sum_k || pi(L_k^s) - pi_ref ||_1   # ordering anchor from depth cue
    L_repel        = - sum_{s<s'} || c_s - c_{s'} ||_2^2  # variance-preserving repulsion across passes
    L_edit         = || c_s - x ||_1  * mask(edit region)  # preserve non-edited region
    L_total        = L_cycle_canvas + L_cycle_layers + lambda_agr * L_agreement + lambda_order * L_order + lambda_repel * L_repel + lambda_edit * L_edit
    update theta by gradient descent on L_total

# Hyperparameters: K=4, lambda_agr=1.0, lambda_order=0.5, lambda_repel=0.01, lambda_edit=1.0
# Edit mask: explicit mask-prediction head (2-layer MLP, hidden=256, GeLU) on S_theta's
# bottleneck features, trained with binary cross-entropy against the difference
# |c_s - x| > tau (tau=0.05) as pseudo-label; mask accuracy is ablated.
```

**Why this design**
We chose stochastic multi-pass sampling over a single deterministic forward pass because the consistency signal requires multiple independent draws of the layer decomposition to expose the invariance that defines a correct edit; this accepts the cost of K times more forward compute per training step. We chose cycle-consistency through recomposition and re-decomposition rather than direct layer-space regularization because recomposition is the only operation that makes the layer order and alpha values observable from the edited canvas, accepting that the cycle must be approximate and may accumulate composition error for very thin layers. We chose cross-pass agreement on the composed image rather than on individual layers because individual layers are not identifiable from unpaired data (layer permutation symmetry), while the composed image is invariant to that symmetry; the trade-off is that agreement alone does not pin down layer ordering, which the cycle term partially resolves. We chose an edit-region preservation loss rather than a global reconstruction loss because unpaired instructions may change only part of the image, accepting that the edit mask must be estimated from the difference between c_s and x, which can be noisy for subtle edits. We added the ordering anchor L_order and the repulsion term L_repel because the agreement objective alone has trivial fixed points (identity edit, single opaque layer, collapsed deterministic decomposition); the depth cue supplies the missing order information and the repulsion term forbids collapse, so the objective has a unique non-trivial minimum independent of the student's own mode. This design is not a variant of on-policy distillation (GKD) because there is no teacher model: the supervision comes from the student's own stochastic decompositions, and the consistency is enforced on the composed RGBA output rather than on token-level distributions.

**Why this measures what we claim**
The cycle-consistency loss `L_cycle_canvas` measures layer-canvas consistency because recomposing the predicted layers and re-editing should reproduce the same canvas under the assumption that the edit is a deterministic function of the instruction and the input image; this assumption fails when the instruction is ambiguous or the edit is stochastic, in which case the cycle term reflects instruction noise rather than layer error. The cross-pass agreement `L_agreement` measures decomposition stability because different stochastic passes should converge to the same composed image under the assumption that the true layer decomposition is unique up to composition; this assumption fails when multiple layer orderings compose to the same image (e.g., fully opaque layers), in which case agreement reflects only the composed image and not the layer structure. The ordering anchor `L_order` measures ordering correctness because the depth cue from the input image provides a reference permutation under the assumption that occlusion boundaries in the input predict layer order; this assumption fails for edits that introduce new occluders not present in the input, in which case the anchor reflects input-depth error rather than edit-order error. The repulsion term `L_repel` measures non-collapse because it penalizes identical canvases across passes under the assumption that the true decomposition has non-zero variance across stochastic draws; this assumption fails for genuinely deterministic edits, in which case the term reflects edit determinism rather than decomposition collapse. The edit-region preservation loss `L_edit` measures edit locality because the non-edited region should be unchanged under the assumption that the instruction specifies a localized edit; this assumption fails for global edits (e.g., style transfer), in which case the loss penalizes legitimate changes. Together these quantities operationalize the motivation-level concept of identifiability from unpaired data: the composed output is the sufficient statistic that is invariant to the unidentifiable layer permutation, and the cycle term ties it back to the canvas.

## Contribution

['A self-distillation objective for compositional image editing that replaces the external teacher-feedback oracle with a cycle-consistency and cross-pass agreement signal over composed RGBA layers, enabling training on unpaired images without any RGBA annotations.', 'A stochastic multi-pass decomposition scheme that exploits the invariance of the composed image to layer permutation and decomposition noise, providing a supervision signal that is identifiable from unpaired data.', 'An empirical demonstration that the proposed self-supervised objective preserves layered editing capability and layer reuse on standard benchmarks while removing the need for paired RGBA layer annotations.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | LayerEditor-100K with layered-editing benchmark | Matches task: instruction-driven layered edits with ground-truth layers. |
| Dataset (unpaired) | LayerEditor-100K images with instructions but RGBA annotations withheld | Tests the central claim: training without paired RGBA annotations. |
| Primary metric | Layer-wise alpha-IoU on edited regions | Directly measures recovered layer structure, not just canvas. |
| Secondary metric | Canvas L1 and LPIPS on edited regions | Checks that layer gains do not come at canvas cost. |
| Diagnostic | Alpha-IoU stratified by layer opacity (opaque vs. semi-transparent) and overlap (overlapping vs. non-overlapping) | Tests the load-bearing assumption that agreement resolves permutation symmetry only where the composed canvas is order-sensitive. |
| Baseline | Flattened instruction-editing model | Tests whether layers add value beyond flat canvas. |
| Baseline | Decompose-then-edit pipeline | Tests whether joint editing+decomposition beats separated stages. |
| Baseline | Single-pass student (K=1, no cycle) | Isolates contribution of stochastic consistency. |
| Baseline | Cycle-consistency on flat canvas only (no layer decomposition) | Isolates the composed-canvas invariance from generic cycle-consistency self-training. |
| Baseline | Reference teacher-distillation method trained with full paired RGBA supervision | Quantifies the practical value of dropping the oracle. |
| Ablation-of-ours | Ours without L_agreement term | Tests whether cross-pass agreement is load-bearing. |
| Ablation-of-ours | Ours without L_order term | Tests whether the ordering anchor is load-bearing. |
| Ablation-of-ours | Ours without L_repel term | Tests whether the repulsion term prevents collapse. |
| Ablation-of-ours | Ours with difference-based edit mask instead of mask-prediction head | Tests mask-head contribution. |
| Ablation-of-ours | K-sweep: K=1,2,4,8 | Reports wall-clock and alpha-IoU to justify compute cost. |

### Why this setup validates the claim

The central claim is that self-distillation over stochastic layer decompositions yields identifiable, reusable layers from unpaired instructions. This is falsifiable only if we compare against baselines that each remove one ingredient of that claim. The flattened editing model tests the sub-claim that layers matter at all: if it matches our layer-wise alpha-IoU, the layered output is decorative. The decompose-then-edit pipeline tests the sub-claim that joint editing and decomposition are necessary: it can recover layers, but only after the edit is committed to a flat canvas, so it should inherit decomposition error on edited regions. The single-pass student (K=1, no cycle) tests the sub-claim that stochastic multi-pass consistency is what exposes the invariant decomposition; without it, the model has no signal distinguishing layer orderings. The flat-canvas cycle-consistency baseline tests whether the composed-canvas invariance is reducible to generic cycle-consistency self-training. The reference teacher-distillation method with full paired supervision quantifies the cost of dropping the oracle. The L_agreement, L_order, and L_repel ablations each remove one anchor that prevents the trivial fixed points of the self-distillation objective. The opacity/overlap-stratified diagnostic directly tests the load-bearing assumption: if agreement resolves permutation symmetry, gains over the single-pass student must concentrate on overlapping semi-transparent layers and vanish on fully opaque non-overlapping layers where ordering is unidentifiable in principle. Layer-wise alpha-IoU on edited regions is the right metric because it is sensitive exactly to the failure modes above: a flattened model has no layers to score, a decompose-then-edit pipeline loses alpha accuracy where the edit changed content, and a single-pass student is unstable across draws. Canvas-level metrics would hide all three failures.

### Expected outcome and causal chain

**vs. Flattened instruction-editing model** — On a case where the instruction adds an object in front of an existing one, the flattened baseline produces a single canvas with the new object fused into the background because it has no layer representation to place the object at a distinct depth. Our method instead emits an ordered RGBA layer for the new object and preserves the occluded background in a lower layer because the cycle and agreement terms force the decomposition to be consistent across stochastic passes and the ordering anchor fixes the depth order. We expect a large gap in layer-wise alpha-IoU on occlusion-heavy edits, with near-parity on single-layer edits where no decomposition is needed.

**vs. Decompose-then-edit pipeline** — On a case where the instruction recolors a textured region, the decompose-then-edit baseline first commits the edit to a flat canvas and then optimizes layers to match it, so its alpha boundaries drift wherever the recolor changed local texture and the decomposition objective has no edit signal to correct it. Our method instead re-decomposes the recomposed canvas under the same instruction, so the cycle term ties layer boundaries back to the edited content and the agreement term suppresses boundary jitter across passes. We expect a moderate but consistent alpha-IoU gain concentrated on textured or high-frequency edit regions, with parity on flat-color regions.

**vs. Single-pass student (K=1, no cycle)** — On a case where two layer orderings compose to nearly the same canvas, the single-pass baseline collapses to one arbitrary ordering because a single forward pass provides no signal that distinguishes orderings; its output is unstable if retrained with a different seed. Our method instead samples K decompositions, fixes ordering with the depth cue, and penalizes disagreement in the composed image, which breaks the permutation symmetry and stabilizes ordering across passes. We expect lower variance in layer-wise alpha-IoU across draws and a visible gain on cases with overlapping semi-transparent layers, with parity on fully opaque non-overlapping cases where ordering is unidentifiable in principle.

**vs. Cycle-consistency on flat canvas only (no layer decomposition)** — On a case where the instruction moves an object behind another, the flat-canvas cycle baseline can match the edited canvas but has no layer representation to encode occlusion, so its alpha-IoU is undefined or zero on the moved object. Our method instead produces ordered layers whose composition matches the canvas and whose alpha boundaries are tied to the edit by the cycle term. We expect a large gap in layer-wise alpha-IoU and near-parity in canvas L1, showing that the composed-canvas invariance is not reducible to generic cycle-consistency.

**vs. Reference teacher-distillation method trained with full paired RGBA supervision** — On all edit types, the teacher-distillation baseline has access to paired layer targets and should achieve the highest alpha-IoU. Our method trains without any RGBA annotations and should approach but not match the teacher, with the gap largest on thin or semi-transparent layers where the depth cue is weakest. We expect our method to close most of the gap on opaque, non-overlapping edits and to retain a measurable gap on overlapping semi-transparent edits, quantifying the practical value of dropping the oracle.

**vs. Ours without L_agreement term** — On overlapping semi-transparent edits, removing L_agreement leaves only the cycle and ordering terms, so the decomposition can drift across passes while still satisfying the cycle. We expect a drop in alpha-IoU concentrated on overlapping semi-transparent layers, with parity on opaque non-overlapping layers.

**vs. Ours without L_order term** — On edits that introduce a new occluder, removing L_order leaves ordering to the cycle term alone, which is ambiguous when two orderings compose similarly. We expect a drop in alpha-IoU on occlusion-heavy edits and higher variance across seeds.

**vs. Ours without L_repel term** — On any edit, removing L_repel allows the K passes to collapse to a single deterministic decomposition, making agreement trivially zero. We expect the training loss to decrease while alpha-IoU degrades toward the single-pass student, confirming that the repulsion term is load-bearing against collapse.

**vs. Ours with difference-based edit mask instead of mask-prediction head** — On subtle edits, the difference-based mask is noisy and mislabels unchanged regions as edited. We expect a small but consistent alpha-IoU gain from the mask-prediction head, concentrated on low-contrast edits.

**K-sweep (K=1,2,4,8)** — We expect alpha-IoU to increase from K=1 to K=4 and to saturate at K=8, with wall-clock scaling linearly in K. K=4 is the chosen operating point balancing alpha-IoU and compute.

### What would falsify this idea

If our gain over the single-pass student is uniform across overlap and non-overlap subsets rather than concentrated on overlapping semi-transparent layers, then the agreement term is not resolving permutation symmetry and the central identifiability claim is wrong. Likewise, if removing L_agreement leaves layer-wise alpha-IoU unchanged while canvas metrics stay fixed, the cross-pass mechanism is inert and the design reduces to cycle-consistency alone.

## References

1. Editing Images into Reusable Layers via Compositional Edit-Prior Distillation
2. On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes
3. Large Language Models Are Reasoning Teachers
