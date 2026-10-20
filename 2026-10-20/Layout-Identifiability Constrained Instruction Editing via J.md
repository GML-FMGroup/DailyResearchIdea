# Layout-Identifiability Constrained Instruction Editing via Joint Latent Layout Resynthesis

## Motivation

Existing instruction-based editors such as Prompt-to-Prompt and SmartEdit (and their descendants like InstructPix2Pix) condition generation on the source image's spatial structure, treating it as a fixed layout prior; consequently, any edit whose target layout is not already present in the source (e.g., 'move the cat to the left of the sofa') cannot be realized because no mechanism synthesizes a new layout independent of the source. The root cause is structural: these methods consume a region mask or source latent once and never re-synthesize the layout as a variable, so the source layout is always a sufficient statistic even when it should be replaced.

## Key Insight

A joint latent layout variable inferred from both instruction and source, together with a cycle-consistency identity that forces the inferred layout to reconstruct the edit's output, makes layout identifiability a decidable property of the latent space rather than a hand-designed router, so the editor can provably re-synthesize structure exactly when the source layout is not identifiable and remain invariant otherwise.

## Method

# LIC-Edit: Layout-Identifiability Constrained Editing
Input: source image x_s, instruction t
Output: edited image x_e

1. Encode source: z_s = E(x_s)                      # VAE latent, 4x64x64
2. Infer joint layout latent:
     l = L_theta(z_s, CLIP(t))                       # L_theta: 6-layer transformer, 8 heads, dim 512
     # l in R^{H_l x W_l x C_l}, H_l=W_l=16, C_l=32
3. Compute identifiability score:
     a = sigma( MLP_id(l) )                          # MLP_id: 3-layer, hidden 256, scalar output
     # a in [0,1]: probability that source layout is sufficient
4. Layout resynthesis (differentiable, always executed):
     l' = a * l + (1 - a) * L_resyn(l, CLIP(t))      # L_resyn: 4-layer transformer, dim 512
     # when a=1, l'=l (invariant); when a=0, l' is new layout
5. Structure-conditioned generation:
     x_e = D( z_s, l', CLIP(t) )                     # D: diffusion U-Net with cross-attn to l' and CLIP(t)
6. Cycle-consistency identity:
     l_hat = L_theta( E(x_e), CLIP(t) )              # re-infer layout from output
     L_cyc = || l_hat - l' ||_2^2                    # identity ties inferred layout to output

Training losses:
  L_diff = MSE( x_e, x_gt )                         # diffusion denoising loss
  L_id   = BCE( a, y_id )                           # y_id=1 if target layout matches source (from data)
  L_cyc  = || l_hat - l' ||_2^2
  L_total = L_diff + lambda_id * L_id + lambda_cyc * L_cyc
  # lambda_id=0.1, lambda_cyc=1.0

**Load-bearing assumption (explicit):** The cycle-consistency identity `||L_theta(E(x_e), t) - l'||^2` is claimed to force the diffusion generator to render the resynthesized layout `l'` in pixels. This holds only if `L_theta` is injective on the layout manifold (distinct layouts map to distinct `l_hat`) and if the generator's cross-attention actually uses `l'` (i.e., `l'` is not ignored). If `L_theta` collapses (e.g., becomes near-constant) or the generator bypasses `l'`, then `L_cyc` is minimized trivially and the layout variable is decorative. To prevent this, we (a) pretrain `L_theta` as a layout extractor with auxiliary pixel-space layout supervision (see training schedule), (b) add an information bottleneck on `l'` via a variance term `L_var = -\sum_j \log(\mathrm{Var}(l'_j) + \epsilon)` with weight `lambda_var=0.01` to discourage constant latents, and (c) include a decoder-ablation check at evaluation: zeroing or shuffling `l'` must change `x_e` (measured by LPIPS > 0.1).

**Training schedule (staged):**
- Stage 1 (pretrain `L_theta` and `L_resyn`): 50k steps on layout reconstruction. For `L_theta`, use auxiliary pixel-space layout supervision: from each training image, extract bounding boxes via an off-the-shelf detector (e.g., GroundingDINO) and render a 16x16 binary object-presence map `m`; train `L_theta` with `L_layout = BCE(MLP_layout(l), m)` where `MLP_layout` is a 2-layer MLP (hidden 256). For `L_resyn`, train with `L_resyn_rec = ||L_resyn(l, CLIP(t)) - l_gt||^2` where `l_gt` is the layout latent of the target image (from `L_theta` on `x_gt`).
- Stage 2 (joint fine-tune): initialize from Stage 1, train all modules end-to-end with `L_total` for 100k steps. Warmup `lambda_cyc` from 0 to 1.0 over the first 10k steps to avoid early instability.

**y_id label generation:** For each training pair (x_s, x_gt, t), compute `y_id = 1` if the bounding-box layout of `x_gt` (from GroundingDINO) matches that of `x_s` within IoU > 0.5 for all objects mentioned in `t`; else `y_id = 0`. Annotate a 1k subset with 3 human raters to measure inter-annotator agreement (Cohen's kappa); if kappa < 0.7, refine the IoU threshold or exclude ambiguous cases.

**Calibration check:** On a held-out calibration set of 512 examples, plot a reliability diagram of `a` vs. actual layout-match accuracy (binned by `a` in 10 bins) and report Expected Calibration Error (ECE).

**Injectivity diagnostic:** On 1k pairs of images with known layout distance (e.g., IoU of object boxes), compute pairwise latent distances `||l_hat_i - l_hat_j||` and report Spearman correlation with layout distance; a high correlation (>0.5) supports injectivity.

**Why this design**
We chose a single joint latent layout variable `l` inferred from both instruction and source over a separate planner/router because the identifiability decision must be a function of the same representation that generates structure; separating them would reintroduce the controller anti-pattern that SmartEdit's BIM already struggles with. We chose the differentiable resynthesis `l' = a*l + (1-a)*L_resyn(l, t)` over a hard binary switch because a hard switch is non-differentiable and would require REINFORCE; the soft blend keeps gradients flowing to both `a` and `L_resyn`, accepting the cost that `a` may not saturate to exactly 0 or 1 during training. We chose the cycle-consistency identity `||L_theta(E(x_e), t) - l'||^2` over a perceptual loss on `x_e` because the identity directly ties the inferred layout to the output's structure, making layout identifiability a property of the latent space rather than a pixel-space heuristic; the cost is that `L_theta` must be trained jointly, which can be unstable early. We chose to train `L_theta` and `L_resyn` end-to-end with the diffusion loss rather than pretraining them separately because the layout variable must be aligned with the generator's structure conditioning; the cost is higher compute. A domain expert would not describe this as a variant of Prompt-to-Prompt or InstructPix2Pix because those methods never instantiate a layout variable at all — they condition on the source latent directly, whereas LIC-Edit introduces a latent layout that is re-synthesized before pixel generation and tied to the output by a cycle identity.

**Why this measures what we claim**
The identifiability score `a` measures the probability that the source layout is a sufficient statistic for the edit because the BCE loss `L_id` trains it against ground-truth labels `y_id` that mark whether the target layout matches the source layout; this assumption fails when the training data's `y_id` labels are noisy (e.g., an edit that is ambiguous about layout), in which case `a` reflects label noise rather than true identifiability. The resynthesized layout `l'` measures the required target layout because the cycle-consistency loss `||L_theta(E(x_e), t) - l'||^2` forces the output's inferred layout to equal `l'`; this assumption fails when the generator ignores `l'` (e.g., when the diffusion loss dominates), in which case `l'` reflects the generator's bias rather than the intended layout. The cycle-consistency identity itself measures the causal link between the inferred layout and the output because it is an equality constraint on the latent space; this assumption fails when `L_theta` is not injective (e.g., two different layouts map to the same `l_hat`), in which case the identity reflects a collapsed representation rather than a true correspondence.

## Contribution

['A novel framework, LIC-Edit, that introduces a joint latent layout variable with a differentiable identifiability score and a cycle-consistency identity, enabling instruction-based editing to re-synthesize spatial layout when the source layout is not identifiable while remaining invariant otherwise.', 'A design principle: layout identifiability can be operationalized as a soft blend between the source layout latent and a resynthesized layout latent, trained end-to-end with a diffusion generator, avoiding a separate planner/router while closing the loop between layout inference and pixel generation.', 'An empirical finding that cycle-consistency on the layout latent is necessary for the generator to respect the resynthesized layout, and that the identifiability score correlates with the need for layout change, validated on a constructed benchmark of layout-changing edits.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | MagicBrush | Diverse instruction edits with layout-sensitive subset |
| Dataset | Synthetic relocation set (1k pairs) | Controlled layout changes for causal test |
| Primary metric | Layout-consistency accuracy | Directly tests whether output layout matches intended layout |
| Secondary metric | Expected Calibration Error (ECE) of a | Tests calibration of identifiability score |
| Secondary metric | Spearman correlation (latent vs. layout distance) | Tests injectivity of L_theta |
| Baseline | InstructPix2Pix | No explicit layout variable; conditions on source latent |
| Baseline | SmartEdit | Uses BIM controller; separate planner/router architecture |
| Baseline | Prompt-to-Prompt | Source-latent conditioning without layout resynthesis |
| Baseline | Direct layout map (off-the-shelf detector) | Tests if learned joint latent is necessary |
| Ablation-of-ours | LIC-Edit w/o L_cyc | Tests necessity of cycle-consistency identity |
| Ablation-of-ours | LIC-Edit w/o L_var | Tests necessity of information bottleneck |

### Why this setup validates the claim

This design forms a falsifiable test of the central claim that a single joint latent layout variable, resynthesized with a differentiable identifiability score and tied to the output by a cycle-consistency identity, yields better layout-identifiability decisions than either source-latent conditioning (InstructPix2Pix, Prompt-to-Prompt) or a separate planner/router (SmartEdit). The layout-consistency accuracy metric is the right detector because it directly measures whether the output's inferred layout matches the intended layout, which is exactly the property our mechanism is designed to control. InstructPix2Pix and Prompt-to-Prompt test the sub-claim that explicit layout resynthesis is necessary: if they match our accuracy, the layout variable is redundant. SmartEdit tests the sub-claim that a joint variable beats a separated controller: if SmartEdit matches us on layout-sensitive cases, the anti-pattern critique is wrong. The direct layout map baseline tests whether the learned joint latent with identifiability gating is necessary rather than a reparameterization of existing layout-conditioning: if a fixed detector-based layout map matches our accuracy, the learned latent is not needed. The ablation LIC-Edit w/o L_cyc isolates the cycle-consistency loss: if removing it does not degrade layout-consistency accuracy, the identity constraint is not doing the claimed work. The ablation LIC-Edit w/o L_var tests the information bottleneck: if removing it does not cause latent collapse (measured by injectivity diagnostic), the variance term is unnecessary. The calibration check (ECE) and injectivity diagnostic (Spearman correlation) directly test the two load-bearing assumptions: that a is calibrated and that L_theta is injective. Together, these comparisons ensure that any observed gain is attributable to the specific mechanism rather than to generic capacity or training data.

### Expected outcome and causal chain

**vs. InstructPix2Pix** — On a case where the instruction implies a layout change (e.g., "move the dog to the left"), the baseline produces an output that preserves the original dog position because it conditions on the source latent directly and has no explicit layout variable to modify. Our method instead infers a layout latent, sets the identifiability score low, resynthesizes a new layout, and conditions generation on that layout, so we expect a noticeable gap on layout-change subsets but parity on appearance-only edits.

**vs. SmartEdit** — On a case where the instruction is ambiguous about layout (e.g., "make it more dramatic"), the baseline's separate planner/router produces an inconsistent layout decision because the router and generator operate on different representations, reintroducing the controller anti-pattern. Our method instead makes the identifiability decision a function of the same joint layout latent that generates structure, so we expect higher layout-consistency accuracy specifically on ambiguous instructions, with a smaller gap on unambiguous ones.

**vs. Prompt-to-Prompt** — On a case where the instruction requires a new layout (e.g., "add a cat on the right"), the baseline produces an output that blends the new object into the existing layout without restructuring because it edits attention maps but never instantiates a layout variable. Our method instead resynthesizes the layout latent before pixel generation and ties the output to it via cycle consistency, so we expect a clear advantage on object-addition and relocation subsets, with parity on style-transfer edits.

**vs. Direct layout map (off-the-shelf detector)** — On a case where the instruction requires a layout change, the baseline uses a fixed detector to produce a layout map, which may be inaccurate for novel objects or ambiguous instructions. Our method learns a joint latent that adapts to the instruction and source, so we expect higher layout-consistency accuracy on ambiguous and novel-object cases, with parity on clear-cut cases.

### What would falsify this idea

If our gain over baselines is uniform across layout-change and appearance-only subsets, rather than concentrated on layout-change and ambiguous-instruction cases where the predicted failure modes occur, then the central claim that layout identifiability is the causal driver is wrong. Additionally, if the ablation without cycle consistency matches our full method on layout-consistency accuracy, the identity constraint is not doing the claimed work; if the injectivity diagnostic shows low correlation (<0.3) or the calibration check shows high ECE (>0.1), the load-bearing assumptions are violated.

## References

1. Instruction-Based Image Editing with Planning, Reasoning, and Generation
2. SmartEdit: Exploring Complex Instruction-Based Image Editing with Multimodal Large Language Models
3. InstructPix2Pix: Learning to Follow Image Editing Instructions
