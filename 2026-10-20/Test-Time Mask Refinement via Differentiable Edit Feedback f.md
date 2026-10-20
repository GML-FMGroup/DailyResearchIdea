# Test-Time Mask Refinement via Differentiable Edit Feedback for Instruction-Based Image Editing

## Motivation

Existing instruction-based editing pipelines such as the one in 'Instruction-Based Image Editing with Planning, Reasoning, and Generation' treat the region mask as a fixed, one-shot output of a frozen MLLM, consumed once by the editor with no mechanism for the editing loss to revise it. This structural separation means mask and edit are optimized as independent stages, so errors in the mask (e.g., including irrelevant regions or missing target areas) propagate uncorrected into the final image. The same limitation recurs across the SmartEdit lineage, where the intermediate spatial representation is a fixed product of a single forward pass, precluding any feedback coupling between representation and generation.

## Key Insight

The editor's differentiability with respect to its spatial hint makes the editing loss a well-defined function of a continuous mask, so the mask can be re-optimized at test time by gradient descent through the frozen editor—a coupling that fixed-mask pipelines structurally cannot achieve.

## Method

```python
# Phase 1: Initialization (frozen CoT planner, MLLM region reasoner, editor)
sub_prompts = CoT_Planner(instruction)          # frozen LLM, no gradients
mask_logits = MLLM_Region_Reasoner(image, instruction, sub_prompts)  # frozen MLLM, produces logits
mask_prob = sigmoid(mask_logits / tau_init)     # tau_init = 0.1, continuous spatial probability field

# Phase 2: Test-time bilevel refinement (only mask parameters updated)
# Load-bearing assumption (explicit): The frozen editor's output is differentiable and
# sufficiently sensitive to its spatial hint mask that the gradient of the edit loss
# w.r.t. mask logits points toward a genuinely better mask rather than toward whatever
# the editor's internal prior prefers. This requires the editor's Jacobian w.r.t. the
# hint to be well-conditioned and faithful to the instruction. We verify this assumption
# empirically via a Jacobian-norm diagnostic (see experiment) and mitigate potential
# self-referential bias by anchoring the refined mask to the initial MLLM mask with a
# fixed anchor loss (lambda_anchor = 0.1).
for step in range(N_refine):                    # N_refine = 20
    mask = sigmoid(mask_logits / tau)           # tau annealed from 0.1 to 0.01
    edited = Editor(image, instruction, hint=mask)  # frozen editor, differentiable w.r.t. hint
    loss = L_edit(edited, image, instruction, sub_prompts) + lambda_tv * TV(mask) + lambda_anchor * MSE(mask, mask_prob.detach())  # lambda_tv = 0.01, lambda_anchor = 0.1
    mask_logits = mask_logits - lr * grad(loss, mask_logits)  # lr = 0.05, Adam optimizer

# Phase 3: Final output
final_mask = sigmoid(mask_logits / tau_final)   # tau_final = 0.01
final_edited = Editor(image, instruction, hint=final_mask)
return final_edited, final_mask
```

**Why this design**
We chose to refine only the mask parameters (via logits) rather than fine-tuning the editor or the MLLM because the editor is already differentiable w.r.t. its spatial hint, so the editing loss provides a direct gradient signal to the mask without retraining any large model, accepting the cost that the refinement is local to the test example and does not improve the MLLM's mask prediction for future inputs. We chose a continuous sigmoid-parameterized mask over a hard binary mask because hard thresholding breaks the gradient chain, accepting the cost that the final mask must be annealed to near-binary for clean edits. We chose to include a total-variation regularizer on the mask to prevent the refinement from overfitting to high-frequency noise in the edit residual, accepting the cost that sharp mask boundaries may be slightly smoothed. We chose to anneal the temperature tau from 0.1 to 0.01 across refinement steps rather than using a fixed temperature because early steps need a soft mask to explore spatial adjustments while later steps need a near-binary mask to produce clean edits, accepting the cost of an additional hyperparameter schedule. We added an anchor loss (MSE between the current mask and the initial MLLM mask, weight 0.1) to prevent the refinement from drifting toward regions the editor already edits well, accepting the cost that the refinement cannot deviate too far from the initial mask. A domain expert would not describe this as a variant of SmartEdit because SmartEdit's bidirectional interaction module is a learned feature-fusion component trained end-to-end, whereas our method introduces a test-time optimization loop that treats the mask as a free variable and uses the frozen editor's gradients to revise it—a mechanism absent from any fixed-mask pipeline.

**Why it measures what we claim**
The edit reconstruction loss `L_edit` measures the alignment between the edited image and the instruction-conditioned target because we assume the editor's output is a differentiable function of the mask hint and that lower loss corresponds to better mask-edit consistency; this assumption fails when the editor has a strong prior that ignores the hint (e.g., for out-of-distribution instructions), in which case the loss reflects the editor's internal bias rather than mask quality. The total-variation term `TV(mask)` measures mask smoothness because we assume spatial coherence is a desirable property of editing regions; this assumption fails when the true edit region is legitimately fragmented (e.g., editing multiple disjoint objects), in which case TV penalizes correct masks. The gradient of `L_edit` w.r.t. `mask_logits` measures the sensitivity of the edit to the mask because we assume the editor's Jacobian is well-conditioned; this assumption fails when the editor saturates (e.g., near-binary hint inputs), in which case gradients vanish and refinement stalls. The anchor loss `MSE(mask, mask_prob.detach())` measures the deviation from the initial MLLM mask because we assume the initial mask provides a reasonable starting point; this assumption fails when the initial mask is severely misaligned, in which case the anchor prevents necessary corrections.

## Contribution

['A test-time mask refinement framework that closes the loop between mask and edit by backpropagating the editing loss through a frozen editor to update a continuous mask parameterization, requiring no retraining of the MLLM or editor.', 'A differentiable mask-to-edit operator with temperature annealing and total-variation regularization that enables stable gradient-based mask optimization while producing near-binary final masks.', 'An empirical demonstration that test-time mask refinement improves editing fidelity on complex instructions compared to fixed-mask pipelines, with analysis of when gradient-based refinement succeeds or fails.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Reason-Edit benchmark | Complex instructions needing spatial reasoning, matches our claim |
| Primary metric | Instruction-following edit accuracy | Directly reflects mask-edit consistency we optimize |
| Baseline | InstructPix2Pix | Fixed-mask-free end-to-end editor, no test-time refinement |
| Baseline | SmartEdit | Learned fusion, no test-time mask optimization loop |
| Baseline | Frozen-MLLM mask, no refinement | Isolates contribution of our test-time loop |
| Baseline | Test-time latent optimization | Optimizes editor's input latent under same L_edit, tests mask-specificity |
| Ablation-of-ours | Fixed tau (no annealing) | Tests whether temperature schedule matters |
| Ablation-of-ours | No anchor loss (lambda_anchor=0) | Tests whether anchor prevents self-referential drift |
| Diagnostic | Jacobian norm vs tau and instruction type | Verifies gradient signal non-vanishing and informative |
| Diagnostic | L_edit vs ground-truth mask IoU correlation | Validates L_edit as proxy for mask quality |
| Distillation | Distill refined masks into MLLM adapter, held-out gain | Tests transferable improvement beyond per-instance |

### Why this setup validates the claim

The central claim is that test-time bilevel refinement of a continuous mask, guided by the frozen editor's gradients, recovers mask-edit consistency that a one-shot MLLM mask prediction misses. Reason-Edit is the right dataset because its complex, multi-object instructions are exactly where a frozen MLLM's initial mask is most likely to be spatially imprecise, giving the refinement loop room to help. InstructPix2Pix and SmartEdit are the right baselines because both produce edits without a test-time mask optimization loop: if our gain over them is real, it must come from the refinement mechanism, not from a better editor backbone. The frozen-MLLM-no-refinement baseline is the crucial control—it shares our mask source and editor but removes the loop, so any improvement over it isolates the loop's contribution. The test-time latent optimization baseline optimizes the editor's input latent under the same L_edit, testing whether the mask-as-free-variable choice is interchangeable with generic input-space test-time optimization. The annealing ablation tests whether the tau schedule, rather than refinement per se, drives the effect. The no-anchor ablation tests whether the anchor loss is necessary to prevent self-referential drift. The Jacobian-norm diagnostic empirically verifies that the editor's gradient w.r.t. the mask is non-vanishing and informative across tau values and instruction types, directly addressing the load-bearing assumption. The L_edit-vs-IoU correlation diagnostic validates that L_edit is a faithful proxy for mask quality. The distillation experiment tests whether the refined masks yield transferable gains when distilled into a lightweight adapter on the MLLM region reasoner, moving beyond per-instance improvement. Instruction-following edit accuracy is the right metric because it directly measures the mask-edit consistency that L_edit optimizes, so the predicted effect should be visible there rather than in generic image fidelity.

### Expected outcome and causal chain

**vs. InstructPix2Pix** — On a case where the instruction requires editing a small, precisely localized region (e.g., "make the third button red"), InstructPix2Pix produces a diffuse or misplaced edit because it has no explicit spatial mask and relies on learned attention priors that blur region boundaries. Our method instead refines a continuous mask against the frozen editor's gradient, sharpening the region until the edit residual aligns with the instruction, so we expect a marked accuracy gap concentrated on small-region and multi-object instructions, with near-parity on global edits.

**vs. SmartEdit** — On a case where the instruction combines a spatial relation with an attribute change (e.g., "move the cup left of the plate and make it blue"), SmartEdit's learned fusion module produces a plausible but spatially off edit because its interaction is trained end-to-end and cannot be corrected at inference. Our method instead treats the mask as a free variable and uses test-time gradients to revise its spatial extent, so we expect a noticeable gap on relational and compositional instruction subsets, with parity on simple single-object attribute edits.

**vs. Frozen-MLLM mask, no refinement** — On a case where the MLLM's initial mask logits are slightly misaligned (e.g., covering 80% of the target object), the unrefined pipeline produces an edit that clips or bleeds because the mask is fixed after one forward pass. Our method instead anneals tau and descends the edit loss to grow or shrink the mask toward the true region, so we expect a consistent but modest accuracy gain over this control, largest on instances where initial mask IoU is intermediate, and near-zero on instances where the initial mask is already near-perfect.

**vs. Test-time latent optimization** — On a case where the instruction requires a spatially precise edit, optimizing the editor's input latent under the same L_edit may adjust global appearance but cannot directly control the spatial extent of the edit, because the latent lacks an explicit spatial correspondence to the mask. Our method instead optimizes the mask logits, which have a direct spatial interpretation, so we expect a larger accuracy gain on spatially localized edits, with parity on global edits.

**vs. Fixed tau (no annealing)** — With a fixed tau (e.g., 0.1), the mask remains soft throughout refinement, leading to blurry or imprecise edits because the editor receives a diffuse hint. Our method anneals tau to 0.01, producing a near-binary mask for clean edits, so we expect higher accuracy, especially on small-region edits.

**vs. No anchor loss** — Without the anchor loss, the refinement may drift toward regions the editor already edits well, ignoring the instruction's true region, because L_edit is self-referential. Our method adds an anchor to the initial MLLM mask, so we expect more stable and instruction-faithful edits, particularly on out-of-distribution instructions where the editor's prior is strong.

**Jacobian-norm diagnostic** — We expect the Jacobian norm of the editor output w.r.t. the mask hint to be non-zero and to decrease as tau decreases (near-binary hints), but remain above a threshold (e.g., 1e-3) for most instructions, confirming that gradients are informative. For OOD instructions, we expect lower Jacobian norms, indicating the editor ignores the hint, which would explain any failure cases.

**L_edit vs IoU correlation** — We expect a negative correlation (Spearman ρ < -0.5) between L_edit and ground-truth mask IoU, validating that lower L_edit corresponds to better mask quality. A weak correlation would falsify the proxy assumption.

**Distillation** — We expect that distilling the refined masks into a lightweight adapter (e.g., a 2-layer MLP on top of the MLLM's mask logits) yields a held-out accuracy gain over the frozen MLLM mask baseline, demonstrating that the refinement captures transferable knowledge beyond per-instance optimization.

### What would falsify this idea

If our gain over the frozen-MLLM-no-refinement control is uniform across all initial-mask-quality buckets rather than concentrated on intermediate-IoU instances, the refinement loop is not doing what we claim. Likewise, if the fixed-tau ablation matches the annealed schedule, the temperature-schedule rationale is wrong and the effect comes from something other than the soft-to-hard transition we posit. Additionally, if the Jacobian-norm diagnostic shows vanishing gradients for most instructions, or if L_edit does not correlate with mask IoU, the load-bearing assumption fails and the method is not viable.

## References

1. Instruction-Based Image Editing with Planning, Reasoning, and Generation
2. SmartEdit: Exploring Complex Instruction-Based Image Editing with Multimodal Large Language Models
3. InstructPix2Pix: Learning to Follow Image Editing Instructions
