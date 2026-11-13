# Cycle-Consistent Decode-and-Rerender for Generative Visual Perception

## Motivation

Generative visual perception methods such as Open Vision Banana frame every task as conditional image editing and read the answer off the generated pixels, but they never verify that the pixels actually encode the intended structured answer: the editor is trained only on image-space targets, so the mapping from task answer to pixel output is learned implicitly and has no enforced inverse. Because no ancestor introduces an explicit decode-and-verify step, the residual gap is structural — scaling data and backbones cannot force the pixel output to be an invertible codec of the instance-level answer, and cross-task fields (depth, normals, segmentation) can drift into mutually inconsistent renderings. This is the shared 'invertible task-output codec with cross-task output consistency' gap that recurs across the inpainting, DINO-grounding, and surface-normal branches.

## Key Insight

Making the task-specific decoder differentiable and shared across tasks turns the generated image into a latent code whose re-rendering must reproduce itself, which forces the pixel output to be an invertible, cross-task-consistent encoding of the structured answer rather than a plausible-looking picture.

## Method

**What it is.** CRD (Cycle-consistent Render-Decode) wraps a conditional image-editing backbone with a differentiable, task-specific decoder D and a fixed differentiable renderer R. Input: an image plus a task prompt. Output: a pixel-space edit whose decoded structured answer is guaranteed (up to cycle residual) to re-render to the same pixels.

**Load-bearing assumption (stated explicitly).** CRD assumes that the fixed analytic renderer R_h is a faithful forward model of how the structured answer appears in the editor's pixel output, so that the cycle residual ||R_h(D_h(y_hat)) - y_hat|| measures lost answer information rather than renderer-style mismatch. Because G's output style (palette, anti-aliasing, texture, learned colormap conventions) is not controlled by us, we calibrate R_h to G's output appearance before computing the cycle residual and report the residual decomposition (answer-information vs. style) so the cycle term is demonstrably measuring invertibility rather than palette/anti-aliasing mismatch.

**How it works.**

```pseudocode
# Given: editor G (open-weight image generator, LoRA-adapted), image x, prompt p
# Task head h in {mask, box, depth} selects decoder D_h and renderer R_h
# Renderer calibration: per-task appearance transform T_h fitted to G's output style

1. y_hat = G(x, p)                      # pixel-space task output (RGB-encoded)
2. a_hat = D_h(y_hat)                   # structured answer: mask logits / box coords / depth+scale
3. y_rec = T_h(R_h(a_hat, x))           # re-render answer into pixel space, then apply calibrated appearance transform
4. L_task = task_loss(a_hat, a_gt)      # supervised decode loss (CE / L1 / scale-invariant)
5. L_cycle = || y_rec - y_hat ||_1 + LPIPS(y_rec, y_hat)   # cycle consistency in calibrated space
6. L_cross = sum over task pairs (d,n): || R_d(D_d(y_hat)) - R_n(D_n(y_hat)) ||_consistency
              where consistency = gradient-of-depth vs normal agreement + mask boundary vs depth discontinuity
7. L = L_task + lambda_c * L_cycle + lambda_x * L_cross + lambda_r * L_task(D_h(y_rec), a_gt)
   # lambda_c=0.5, lambda_x=0.1, lambda_r=0.3; decoder lr = 10x editor lr
8. Update D_h and G jointly; D_h is shared-trunk with task-specific heads
```

**Renderer calibration (T_h).** For each task head h, T_h is a per-task appearance transform fitted to G's output style: a 3x3 color matrix plus per-channel bias (12 parameters) for mask and box, and a monotonic piecewise-linear tone curve (16 knots) for depth. T_h is fitted by minimizing ||T_h(R_h(a_gt, x)) - y_hat_gt||_1 on a calibration set of 512 examples where y_hat_gt = G(x, p_gt) is the editor's rendering of the ground-truth answer, with T_h frozen during main training. The residual floor is reported as the mean ||T_h(R_h(a_gt, x)) - y_hat_gt||_1 on a held-out calibration split of 128 examples; all cycle residuals are reported relative to this floor.

**Residual decomposition.** We report the cycle residual decomposed into (i) answer-information component: ||T_h(R_h(a_hat, x)) - T_h(R_h(a_gt, x))||_1, and (ii) style component: ||T_h(R_h(a_gt, x)) - y_hat||_1, so the cycle term's sensitivity to answer error versus style mismatch is explicit.

**Why this design.** We chose an explicit differentiable decoder D_h over an implicit read-out (e.g., an auxiliary classification head on G's features) because the cycle residual only constrains the answer if D_h is the *same* function used at inference, accepting the cost that D_h must be trained jointly and can dominate gradients early. We chose a fixed analytic renderer R_h over a learned renderer because a learned R_h can co-adapt with D_h to make the cycle trivially satisfiable (the decoder-renderer pair collapses to an autoencoder), accepting the cost that R_h must be hand-specified per task (soft-argmax mask splat, differentiable box rasterizer, depth-to-points projection) and calibrated to G's output style via T_h. We chose a shared-trunk decoder with task-specific heads over fully separate decoders because cross-task consistency terms require a common feature space to compare depth discontinuities against mask boundaries, accepting the cost that a single task's error propagates to others. We chose to up-weight the decoder learning rate (10x) over the editor because the editor is pretrained and the decoder is the invertibility bottleneck, accepting the cost that early training is decoder-dominated and the editor's image-editing ability must be protected by a low LoRA rank. This is not a variant of a standard autoencoder or of OVB's legend ensembling: OVB ensembles multiple rendered legends at inference and never decodes them back, whereas CRD makes the decode step a training-time constraint that reshapes the editor's output distribution.

**Why this design measures what we claim.** The cycle residual `||T_h(R_h(D_h(y_hat))) - y_hat||` measures *invertibility of the task-output codec* because the assumption is that a fixed analytic renderer R_h, after calibration T_h to the editor's output style, is a faithful forward model of how the structured answer appears in pixels — i.e., if a_hat is the true answer, T_h(R_h(a_hat,x)) equals the ideal pixel output; this assumption fails when the renderer's appearance model (e.g., color palette, anti-aliasing) diverges from the editor's rendering style even after calibration, in which case the residual reflects a renderer-style mismatch rather than a genuine loss of answer information. The cross-task term `||R_d(D_d(y_hat)) - R_n(D_n(y_hat))||` measures *cross-task output consistency* because the assumption is that depth discontinuities and segmentation boundaries coincide on real surfaces within a fixed pixel tolerance (3 pixels), so a consistent set of decoded fields must agree at those loci; this assumption fails on transparent or thin structures, in which case the term reflects physical ambiguity rather than decoder inconsistency. The supervised decode loss `L_task(a_hat, a_gt)` measures *answer fidelity* because the assumption is that the ground-truth annotation is the correct structured answer for the rendered pixels; this fails under annotation noise or occlusion, in which case it reflects label error. The re-decode term `L_task(D_h(y_rec), a_gt)` measures *cycle stability* because the assumption is that applying the decoder to its own re-rendering should recover the same answer; this fails when the renderer is non-injective (two answers render to the same pixels), in which case it reflects renderer ambiguity.

## Contribution

["A differentiable render-and-decode cycle-consistency module (CRD) that turns a conditional image-editing perception model's pixel output into an explicitly invertible codec of the structured task answer, with a fixed analytic renderer per task to prevent decoder-renderer co-adaptation.", 'A cross-task consistency objective that couples depth, surface-normal, and segmentation fields through shared decoder features and renderer-space agreement terms, yielding a single training signal that enforces mutual agreement among task outputs.', "A joint training recipe (shared-trunk task-specific decoders, 10x decoder learning rate, low-rank editor adaptation) that preserves the base generator's image-editing ability while imposing the cycle constraint, and an analysis of which renderer choices make the cycle constraint non-trivial."]

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Cityscapes zero-shot + referring/reasoning segmentation | Tests cross-task consistency on real surfaces |
| Dataset | NYUv2 (indoor) | Tests generalization beyond driving scenes |
| Primary metric | Boundary-aligned depth/seg agreement (F-score at discontinuities, tolerance sweep 1/3/5 px) | Directly measures the cross-task consistency claim |
| Secondary metric | Cycle residual relative to calibrated renderer floor | Measures invertibility net of style mismatch |
| Baseline | Vision Banana (reported) | Tests whether cycle constraint beats legend ensembling |
| Baseline | SOTA affine-invariant depth estimator | Tests whether CRD's decode beats implicit read-out |
| Baseline | Standard autoencoder wrapping same editor | Tests whether fixed renderer prevents codec collapse |
| Baseline | Test-time optimization of a_hat through R_h (no training-time cycle) | Tests whether training-time constraint is the novel ingredient |
| Baseline | Learned-renderer cycle (R_h replaced by small CNN) | Tests whether fixed renderer prevents co-adaptation |
| Ablation-of-ours | CRD without L_cross | Isolates the cross-task consistency contribution |
| Ablation-of-ours | CRD without renderer calibration T_h | Isolates the effect of style mismatch |
| Ablation-of-ours | CRD with tolerance sweep for L_cross (1/3/5 px) | Exposes physical ambiguity failure mode |

### Why this setup validates the claim

The central claim is that forcing the editor's pixel output through a fixed analytic renderer and back reshapes the output distribution so that decoded answers are invertible and mutually consistent across tasks. Cityscapes zero-shot is the right testbed because it exposes depth discontinuities and segmentation boundaries on the same real surfaces, letting the cross-task term be evaluated where its assumption holds; NYUv2 adds indoor surfaces to test that the shared-trunk cross-task claim is not dataset-specific. Vision Banana tests the sub-claim that a training-time cycle beats inference-time legend ensembling: if CRD only matches OVB, the cycle adds nothing beyond ensembling. The SOTA depth estimator tests whether an explicit differentiable decoder beats an implicit read-out on the same editor features. The autoencoder baseline tests the specific design choice of a fixed renderer: if a learned renderer co-adapts, the cycle becomes trivially satisfiable and gains vanish. The test-time optimization baseline tests whether the training-time cycle constraint is the novel ingredient rather than the loop itself. The learned-renderer baseline directly tests the fixed-renderer choice. The boundary-aligned agreement metric is chosen because it is exactly the quantity L_cross is supposed to improve; a uniform accuracy gain would not distinguish CRD from simply a better editor. The tolerance sweep exposes the failure mode where depth discontinuities and segmentation boundaries do not coincide within a fixed pixel tolerance. The ablation without L_cross isolates whether any gain comes from cross-task coupling or merely from the cycle term. The ablation without renderer calibration T_h isolates whether the cycle term is measuring style mismatch.

### Expected outcome and causal chain

**vs. Vision Banana** — On a case where a thin pole or sign boundary sits near a depth discontinuity, Vision Banana ensembles multiple rendered legends at inference but never decodes them back, so its depth and mask outputs are produced independently and can disagree at that locus. Our method instead decodes both fields from the same y_hat and penalizes disagreement via L_cross, so the shared trunk is pulled toward agreement at discontinuities. We expect a noticeable gap on boundary-adjacent pixels but parity on interior regions, since interior pixels are easy for both.

**vs. SOTA affine-invariant depth estimator** — On a case where a reflective surface produces a depth value that is locally plausible but inconsistent with the segmentation boundary (e.g., a window), the implicit read-out baseline produces a depth map that ignores the mask edge because its features were never constrained to be invertible through a renderer. Our method instead passes depth through R_h and back, so an answer that cannot re-render to the observed pixels incurs a cycle residual, forcing the decoder toward answers that are pixel-consistent. We expect CRD to close most of the gap on boundary-aligned depth but to be roughly at parity on smooth planar regions.

**vs. Standard autoencoder wrapping same editor** — On a case where the editor's output style (palette, anti-aliasing) diverges from the analytic renderer, a learned renderer would co-adapt with the decoder and make the cycle trivially satisfiable, collapsing to an autoencoder that reconstructs pixels without preserving answer information. Our fixed renderer cannot co-adapt, so the residual genuinely reflects lost answer information. We expect the autoencoder to show high pixel reconstruction but poor decode fidelity, while CRD shows slightly higher residual but markedly better boundary-aligned agreement.

**vs. Test-time optimization of a_hat through R_h** — On a case where the editor's output is ambiguous, test-time optimization can fit a_hat to the pixels without any training-time pressure on the editor's output distribution. Our method instead reshapes the editor during training so that its outputs are inherently decodable. We expect CRD to outperform test-time optimization on boundary-aligned agreement, especially on thin structures where test-time optimization overfits to style.

**vs. Learned-renderer cycle** — On a case where the learned renderer co-adapts with the decoder, the cycle becomes trivially satisfiable and the decoder-renderer pair collapses to an autoencoder. Our fixed renderer with calibration T_h prevents co-adaptation. We expect the learned-renderer baseline to show lower cycle residual but worse decode fidelity and boundary-aligned agreement, confirming that the fixed renderer is the novel ingredient.

### What would falsify this idea

If CRD's gain over Vision Banana and the autoencoder is uniform across boundary-adjacent and interior pixels rather than concentrated at discontinuities where L_cross is designed to act, the cross-task consistency mechanism is not what produces the improvement and the central claim is wrong. Likewise, if the autoencoder baseline matches CRD on boundary-aligned agreement, the fixed-renderer choice is not doing the work we claim. If the calibrated renderer residual floor is large and CRD's cycle residual relative to that floor does not correlate with answer error, the cycle term is measuring style mismatch rather than invertibility.

## References

1. Open Vision Banana: A Fully Open Recipe for Generative Visual Perception
