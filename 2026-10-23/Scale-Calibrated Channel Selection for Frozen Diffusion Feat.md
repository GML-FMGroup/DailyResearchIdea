# Scale-Calibrated Channel Selection for Frozen Diffusion Features

## Motivation

Policy-Mask learns which diffusion feature channels to keep for a downstream task, but its channel-selection policy is trained only on the discriminative objective and never receives a signal about absolute depth scale. Marigold shows that full fine-tuning of a diffusion backbone implicitly learns the affine-invariant depth prior, yet Policy-Mask's frozen backbone discards that prior the moment channels are selected, so absolute-scale recovery remains structurally unestablished. The root cause is that channel selection and scale calibration are optimized on disjoint objectives: the policy sees only task loss, while the scale prior lives in channels the policy may drop.

## Key Insight

A differentiable affine-invariant normalization loss applied to a verification head's output creates a gradient path back into the channel-selection logits, so the policy is forced to retain channels whose features carry scale information rather than merely task-discriminative information.

## Method

**What it is.** SCALE-MASK extends Policy-Mask with a scale-consistency verification head (SCVH) that consumes only the selected diffusion feature channels and predicts a per-pixel affine-invariant depth map. A differentiable scale-normalization loss on the SCVH output is backpropagated through the channel-selection logits, so the channel budget is allocated to features that jointly support the downstream task and preserve depth scale. The backbone stays frozen; only the policy heads, SCVH, and task head are trained.

**Load-bearing assumption (stated explicitly).** The mechanism assumes that the gradient of the affine-invariant L1 loss on the SCVH output, backpropagated through the straight-through channel mask into the selection logits, carries a channel-specific signal about which frozen diffusion channels encode relative depth structure — i.e., that scale-relevant information is channel-separable and that dL_scale/da_c ranks channels by that property rather than by generic feature magnitude. This assumption is operationalized and tested as follows: (i) the per-channel gradient dL_scale/da_c is logged at every policy update; (ii) on a held-out calibration batch of 512 images, an oracle channel-importance ranking is computed by exhaustive leave-one-channel-out affine-invariant RMSE (drop channel c, recompute SCVH output, measure ΔRMSE); (iii) the Spearman rank correlation between dL_scale/da_c and the oracle ΔRMSE is reported as a diagnostic. If the correlation is near zero or dominated by high-variance channels, the assumption fails and the gradient reflects generic feature magnitude rather than scale relevance.

**How it works.**
```pseudocode
# Inputs: image x, frozen diffusion backbone F, budget B, task label y
# Policy heads: timestep pi_t, block pi_b, channel pi_c (logits)
# SCVH: 2-layer MLP (hidden=256, GeLU) + 1x1 conv, task head: existing Policy-Mask head

1. z_t, z_b, z_c = F(x)                      # frozen multi-timestep, multi-block features
2. a_t ~ pi_t(x, B); a_b ~ pi_b(x, B)        # Gumbel-softmax, straight-through
   a_c ~ pi_c(x, B)                          # channel mask, cardinality-normalized scores
3. f_sel = select(z_t, z_b, z_c; a_t, a_b, a_c)   # [N, C_sel, H, W]
4. d_hat = SCVH(f_sel)                       # affine-invariant depth prediction
5. L_task = task_loss(task_head(f_sel), y)
6. L_scale = affine_invariant_loss(d_hat, d_gt)   # scale+shift normalized L1
7. L_total = L_task + lambda * L_scale       # lambda = 0.5, fixed
8. Backprop L_total through SCVH into a_c logits (straight-through)
9. Diagnostic (every 500 steps, held-out batch of 512):
   - log dL_scale/da_c per channel
   - compute oracle ΔRMSE_c by leave-one-channel-out on the same batch
   - report Spearman rho(dL_scale/da_c, ΔRMSE_c)
```
The affine-invariant loss normalizes both prediction and ground truth by their per-image mean and standard deviation before computing L1, so the gradient with respect to a_c is nonzero whenever dropping a channel increases the residual after normalization.

**Why this design.** We chose a separate verification head over feeding selected features directly into the task head because the task head is optimized for segmentation/classification and its gradients do not encode scale; a dedicated head isolates the scale signal, accepting the cost of one extra lightweight module. We chose a differentiable affine-invariant loss over a hard scale-consistency constraint because the latter would require solving a per-image alignment at every policy update, which is incompatible with end-to-end Gumbel-softmax training; the normalization loss is a soft surrogate that still produces usable gradients. We chose to backpropagate through the straight-through channel mask rather than re-sampling channels per step because re-sampling would make the scale gradient stochastic and high-variance, accepting the known bias of straight-through estimators in exchange for stable policy updates. A domain expert would not describe this as Marigold-plus-Policy-Mask: Marigold fine-tunes the U-Net weights, whereas here the U-Net is frozen and the scale signal is used only to reshape a discrete channel-selection distribution — the mechanism is a gradient path through a selection mask, not a weight update.

**Why this measures what we claim.** The affine-invariant loss L_scale measures scale-preservation capacity of the selected channel subset because the per-image mean/std normalization removes exactly the affine ambiguity that Marigold's loss removes, so any residual L1 is attributable to channels that fail to encode relative depth structure; this assumption fails when the ground-truth depth distribution is itself degenerate (e.g., near-planar scenes), in which case L_scale reflects noise rather than scale information. The straight-through channel mask a_c measures the policy's scale-awareness because the gradient dL_scale/da_c is nonzero only for channels whose removal changes the normalized depth residual; this assumption fails when two channels are redundant for depth, in which case the gradient is split arbitrarily between them and a_c reflects an arbitrary tie-break rather than a unique scale-relevant ranking — the diagnostic in step 9 makes this failure mode observable via low Spearman rho against the leave-one-channel-out oracle. The SCVH output d_hat measures affine-invariant depth because it is trained only on L_scale with no absolute-scale supervision, so it cannot encode metric scale; this assumption fails when the synthetic depth data contains a consistent bias (e.g., all scenes share a similar depth range), in which case d_hat may implicitly learn that bias and the loss no longer isolates scale-invariant structure.

## Contribution

['A scale-consistency verification head (SCVH) that operates on the selected diffusion feature channels and is trained with a differentiable affine-invariant normalization loss, creating the first gradient path from a scale objective back into a channel-selection policy.', 'A design principle for budget-aware adaptation: channel-selection policies should be trained with auxiliary signals that reflect the invariances required by the downstream task, not only the task loss, because frozen-backbone selection otherwise discards priors that full fine-tuning would retain.', 'An empirical demonstration that scale-calibrated channel selection changes which channels the policy retains at matched budgets, with the effect isolated by comparing channel masks learned with and without the SCVH gradient path.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | NYUv2 + ScanNet depth subsets | Real indoor depth with non-degenerate scene structure |
| Primary metric | Affine-invariant RMSE (scale/shift-normalized) | Directly matches the claimed scale-preservation target |
| Baseline | Policy-Mask (task-only selection) | Tests whether scale signal adds beyond task gradients |
| Baseline | Marigold fine-tuned U-Net | Tests frozen-backbone channel selection against weight tuning |
| Baseline | Random channel subset at matched budget | Tests whether selection beats budget alone |
| Ablation-of-ours | SCALE-MASK without SCVH (λ=0) | Isolates the scale-consistency gradient path |
| Ablation-of-ours | SCALE-MASK with scale gradient routed only through shared feature extractor (no mask gradient) | Isolates the mask-gradient path from multi-task training |
| Diagnostic | Spearman rho(dL_scale/da_c, leave-one-channel-out ΔRMSE) on 512 held-out images | Grounds the claim that the gradient ranks scale-relevant channels |

### Why this setup validates the claim

The central claim is that a differentiable scale-consistency gradient routed through the channel-selection mask reallocates the budget toward channels that preserve relative depth structure, and that this reallocation is what improves affine-invariant depth under a frozen backbone. The metric is chosen to match the mechanism: affine-invariant RMSE removes exactly the mean/std ambiguity that the SCVH loss removes, so any improvement is attributable to the channel subset rather than to metric-scale calibration. Policy-Mask (task-only) tests the sub-claim that task gradients alone do not encode scale; if the scale gradient matters, SCALE-MASK must beat it at equal budget. Marigold tests the sub-claim that frozen-backbone selection can approach weight fine-tuning; parity on well-structured scenes and a gap where fine-tuning overfits or data is scarce is the predicted signature. Random-subset tests that selection, not budget, drives the effect. The λ=0 ablation is the sharpest test: it holds architecture, budget, and task head fixed and removes only the scale gradient path, so any gain of SCALE-MASK over it is causally attributable to the claimed mechanism rather than to the extra module's capacity. The no-mask-gradient ablation isolates the mask-gradient path itself: it routes L_scale through the shared feature extractor but not through a_c, so any gain of SCALE-MASK over it is attributable to the mask gradient rather than to multi-task training. The Spearman diagnostic directly tests the load-bearing assumption: if dL_scale/da_c does not correlate with the leave-one-channel-out oracle ΔRMSE, the gradient is not ranking channels by scale relevance and the causal story fails.

### Expected outcome and causal chain

**vs. Policy-Mask (task-only selection)** — On a case where the task head is a segmentation objective whose gradients are dominated by class boundaries, Policy-Mask allocates channels to texture and edge features and produces depth with correct object outlines but flattened within-object gradients, because its selection signal never sees relative depth. Our method instead keeps channels whose removal raises the normalized depth residual, because L_scale backpropagates through a_c into the policy logits, so we expect a noticeable affine-invariant RMSE gap concentrated on smooth, low-texture regions and near-parity on cluttered boundary regions.

**vs. Marigold fine-tuned U-Net** — On a case with limited real depth supervision, Marigold adapts U-Net weights and overfits the synthetic depth statistics, producing biased predictions on scenes whose depth range differs from training, because its weight update encodes the training distribution. Our method instead leaves the U-Net frozen and only reshapes a discrete channel distribution, because the scale signal is used to rank channels rather than to move weights, so we expect parity on in-distribution scenes and a smaller degradation on out-of-distribution depth ranges, with the gap widening as supervision shrinks.

**vs. Random channel subset at matched budget** — On a case where the budget is tight, a random subset retains many channels that are redundant for depth and drops a few that carry relative structure, producing high-variance depth errors across images because the selection is uncorrelated with the loss. Our method instead concentrates the budget on channels with nonzero scale gradient, because the straight-through mask lets L_scale rank channels by their contribution to the normalized residual, so we expect lower mean error and markedly lower across-image variance, with the advantage largest at the smallest budgets.

**vs. SCALE-MASK without SCVH (λ=0)** — On a case where the task head is fixed and the only difference is the presence of the scale gradient path, the λ=0 ablation selects channels purely by task loss and therefore matches Policy-Mask on affine-invariant RMSE, because it never receives a scale signal. Our method adds the SCVH and L_scale, so we expect a measurable RMSE improvement over λ=0 at equal budget, concentrated on smooth regions, confirming that the scale gradient path—not the extra module's capacity—drives the gain.

**vs. SCALE-MASK with scale gradient routed only through shared feature extractor (no mask gradient)** — On a case where L_scale is backpropagated through the shared feature extractor but not through a_c, the channel selection receives no scale-specific gradient and remains task-driven, because the mask gradient is zero. Our method routes L_scale through a_c, so we expect a larger RMSE improvement over this ablation than over λ=0, isolating the mask-gradient path as the causal factor.

**Diagnostic: Spearman rho(dL_scale/da_c, leave-one-channel-out ΔRMSE)** — On a held-out batch of 512 images, we compute the per-channel gradient dL_scale/da_c and the oracle ΔRMSE by exhaustive leave-one-channel-out evaluation. If the load-bearing assumption holds, we expect a positive Spearman rho (target > 0.3) between the two rankings, indicating that the gradient identifies scale-relevant channels. If rho is near zero or negative, the gradient reflects generic feature magnitude or redundancy-driven tie-breaking rather than scale relevance, and the central mechanism is not supported.

### What would falsify this idea

If SCALE-MASK matches the λ=0 ablation and Policy-Mask at equal budget on affine-invariant RMSE, or if its gain over Policy-Mask is uniform across smooth and boundary regions rather than concentrated where the predicted task-gradient failure occurs, then the scale-gradient path is not reallocating the budget as claimed and the central mechanism is wrong. Additionally, if the Spearman rho between dL_scale/da_c and the leave-one-channel-out oracle ΔRMSE is near zero or negative, the load-bearing assumption that the gradient ranks channels by scale relevance is falsified.

## References

1. Policy-Mask: Budget-Aware Adaptation of Diffusion Features to Discriminative Tasks
2. Repurposing Diffusion-Based Image Generators for Monocular Depth Estimation
