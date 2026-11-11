# Part-Compositional Spatial Foresight for Dexterous Policies

## Motivation

DexForesight freezes its action-conditioned spatial world model after offline training, so its predictive latents are only as compositional as the object geometries seen in demonstration data; the module has no mechanism to recombine known object parts into unseen configurations, and thus fails structurally on novel geometries rather than merely lacking data. This is the same meta-level gap that recurs across spatial VLA representations, which assume dataset-level coverage of object geometry suffices for generalization. The root cause is that the world model predicts geometry holistically, so part-level structure is never factored out and cannot be recombined at test time.

## Key Insight

If the spatial foresight module predicts hand-point motion and visual change as a composition of per-part geometry tokens whose cross-part interactions are learned independently of any specific whole-object arrangement, then novel geometries formed from familiar parts fall inside the model's support because the part-interaction function is shared across all training objects.

## Method

**What it is.** We introduce PC-Dex (Part-Compositional DexForesight), which replaces DexForesight's holistic spatial foresight target with a part-decomposed one. Inputs: depth observation, kinematic hand points, robot state, language instruction, and a part segmentation of the observed object. Outputs: predicted future 3D hand-point motion and visual feature change, factorized as a composition of per-part predictions, which are then aligned to policy queries exactly as in DexForesight.

**Load-bearing assumption (stated explicitly).** The method assumes that a permutation-equivariant cross-part attention function, trained so that the hand-motion head predicts correctly when part slots are permuted, learns a geometry-conditioned hand-motion prediction that is invariant to part identity, so that a novel whole-object arrangement of familiar parts is genuinely in-support for the frozen world model. This assumption is operationalized and tested in Phase 2 via the identity-preserving compositional loss and the identity-conditioned control described below; it fails when the correct hand motion depends on which part is which (e.g., a mug handle must be grasped at the handle, not its mirror), in which case the shuffled loss reflects an identity-agnostic motion prior rather than compositional generalization.

**How it works.**

```pseudocode
# Phase 1: part decomposition (offline, per training frame)
# Segmentation front-end is pre-registered and fixed: an off-the-shelf class-agnostic
# segmenter (e.g., a frozen SAM-style promptable segmenter) with k=4 slots, run once
# offline; its part-consistency error on DexJoCo objects is reported up front (see
# experiment) so segmentation quality is a controlled variable, not a confound.
for each frame in offline_data:
    parts = segment_object(depth, k=4)        # k part slots, fixed budget
    part_geo[p] = pointnet_encoder(parts[p])   # geometry token per part (PointNet, 3-layer MLP, hidden=256)
    part_pose[p] = centroid + PCA_frame(parts[p])
    part_id[p]   = part_type_embed(parts[p])   # learned part-type/affordance token, dim=64,
                                               # contrastively trained across objects so the same
                                               # semantic part maps to the same token

# Phase 2: compositional world model (trained, then frozen)
for each (frame, action, future) in offline_data:
    z_part[p] = part_geo[p] + pose_embed(part_pose[p]) + part_id[p]
    z_global = cross_part_attention(z_part[0..k])   # shared interaction fn, permutation-equivariant,
                                                    # 4 heads, 2 layers, hidden=256
    pred_hand_motion = hand_head(z_global, hand_points, action)   # 2-layer MLP, hidden=256
    pred_vis_change  = vis_head(z_global, action)                 # 2-layer MLP, hidden=256
    L_pred = L2(pred_hand_motion, future_hand_motion) + L2(pred_vis_change, future_vis_change)
    # identity-preserving compositional loss: shuffle ONLY the spatial arrangement
    # (pose embeddings) across objects in the batch, while keeping geometry tokens and
    # part-identity tokens bound to their semantic part. This trains the hand head to
    # predict motion conditioned on 'which part is where' rather than 'some parts somewhere'.
    z_pose_shuf = z_part with pose_embed(part_pose) replaced by pose_embed(part_pose_shuffled)
    z_shuf = cross_part_attention(z_pose_shuf)
    L_comp = L2(hand_head(z_shuf, hand_points, action), future_hand_motion)
    # identity-conditioned control: a diagnostic forward pass that permutes part-identity
    # tokens only (keeping geometry and pose fixed) and measures the change in hand-motion
    # output; this quantifies whether the head is identity-conditioned or degenerate.
    L = L_pred + lambda_comp * L_comp   # lambda_comp = 0.5

# Phase 3: freeze world model, train policy
for each (obs, lang, state, action) in offline_data:
    q_world = world_model.predictive_queries(obs, action)   # frozen
    q_policy = policy.encode(obs, lang, state)
    L_align = L2(q_policy, q_world)
    L_policy = L_action(action, policy.decode(q_policy)) + lambda_align * L_align  # lambda_align = 0.1
```

**Why this design.** We chose a fixed part budget with permutation-equivariant cross-part attention over a monolithic whole-object encoder because the interaction function is then shared across all part arrangements, so a novel whole-object geometry is represented by the same interaction function applied to a new slot configuration, accepting the cost that segmentation errors propagate directly into foresight. We chose an identity-preserving compositional loss (shuffling only spatial arrangement while keeping geometry and part-identity tokens bound to their semantic part) over a reconstruction-only loss because it forces the hand-motion head to remain correct under part rearrangements while preserving the identity signal that novel-arrangement grasping requires, accepting the cost that the shuffled configurations are physically implausible and may inject noise into the visual-change head, which is why we apply the shuffle loss only to the hand-motion head. We chose to freeze the world model and align policy queries as in DexForesight rather than co-train, because co-training would let the policy drift the foresight latents and destroy the part-interaction invariance we just built, accepting the cost that the foresight module cannot adapt online. A domain expert would not describe this as DexForesight plus a segmentation front-end, because the load-bearing object is the permutation-equivariant interaction function trained under arrangement shuffling with identity-preserving part tokens, which is absent from DexForesight and from retrieval-style part-matching approaches.

**Why it measures what we claim.** The identity-preserving compositional loss on the hand-motion head measures geometry-compositional extrapolation because permutation-equivariance of the cross-part attention makes the hand-motion prediction invariant to slot ordering while the part-identity token keeps 'which part is where' explicit, so a shuffled spatial arrangement is a valid in-support input; this assumption fails when the true hand motion depends on part identity rather than geometry (e.g., a mug handle must be grasped by the handle, not its mirror), in which case the shuffled loss reflects an identity-agnostic motion prior rather than compositional generalization. The part-geometry token measures object-part identity because the pointnet encoder is trained only on segmented parts with consistent semantics across objects; this assumption fails when segmentation is class-agnostic and splits the same physical part inconsistently, in which case the token reflects local shape statistics rather than a reusable part. The alignment loss between policy queries and frozen world-model queries measures whether the policy's latent state carries the same part-compositional foresight; this assumption fails when the policy can minimize alignment without using the part structure (e.g., by matching only the global mean), in which case the loss reflects latent-space proximity rather than shared compositional structure.

## Contribution

(1) A part-compositional spatial world model that factorizes dexterous foresight into permutation-equivariant per-part geometry tokens with a shared cross-part interaction function, trained with a slot-shuffle compositional loss on hand-motion prediction. (2) A design principle showing that freezing the compositional world model and aligning policy queries preserves part-interaction invariance, whereas co-training would destroy it — an empirical finding about where the compositional prior must be placed in the DexForesight pipeline. (3) A part-decomposed evaluation protocol on DexJoCo that separates seen-part/seen-arrangement, seen-part/novel-arrangement, and novel-part conditions to isolate geometry-compositional extrapolation from dataset coverage.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | DexJoCo 11-task suite, with a held-out arrangement-shifted split (objects whose part layout is mirrored or re-arranged relative to training) | Same benchmark as DexForesight; enables direct comparison and isolates arrangement shift. |
| Primary metric | Success rate, reported overall and on the arrangement-shifted subset | Directly measures task completion under novel part arrangements. |
| Secondary metric | Part-consistency error of the fixed segmentation front-end on DexJoCo objects (reported up front) | Controls segmentation quality as a variable, not a confound. |
| Diagnostic metric | Hand-motion output change under semantically meaningful slot permutations (handle-left vs handle-right) vs under identity-only permutations | Quantifies whether the head is identity-conditioned or a degenerate permutation-invariant prior. |
| Baseline | DexForesight | Tests whether part-compositional foresight beats holistic foresight. |
| Baseline | Whole-object world-action model | Tests whether part decomposition itself matters. |
| Baseline | Retrieval-style part-matching policy | Tests whether learned equivariant interaction beats matching. |
| Baseline | Slot-attention / object-centric dynamics model (Locatello et al., 2020) | Tests whether the shared interaction function under arrangement shuffling adds beyond object-centric factorization. |
| Ablation-of-ours | PC-Dex w/o compositional loss | Tests whether the identity-preserving arrangement-shuffle loss drives the gain. |
| Ablation-of-ours | PC-Dex w/o part-identity token (identity-agnostic control) | Tests whether the load-bearing assumption (identity-conditioned motion) holds. |

### Why this setup validates the claim

The central claim is that a permutation-equivariant, part-decomposed foresight module trained under identity-preserving arrangement shuffling yields geometry-compositional extrapolation that holistic foresight (DexForesight), retrieval-style part-matching, and object-centric slot-attention dynamics cannot achieve. The DexJoCo suite is the right dataset because it contains tasks whose objects vary in part arrangement while sharing part semantics, so a method that genuinely composes parts should transfer across arrangements while a holistic encoder must relearn each configuration; the held-out arrangement-shifted split makes this test explicit rather than relying on incidental variation. DexForesight is the correct baseline for the core sub-claim (compositional vs. holistic foresight) since it is identical except for the foresight target. The whole-object world-action model isolates whether the gain comes from part decomposition at all rather than from the frozen-alignment machinery. The retrieval-style part-matching baseline tests the alternative hypothesis that explicit part correspondence, not a learned equivariant interaction function, is what matters. The slot-attention / object-centric dynamics baseline tests whether the shared interaction function under arrangement shuffling adds anything beyond object-centric factorization. The ablation removing the compositional loss is the decisive test of the load-bearing ingredient: if the gain persists without arrangement shuffling, the claimed mechanism is wrong. The identity-agnostic control (removing the part-identity token) directly tests the load-bearing assumption that hand motion is identity-conditioned; if it matches the full method, the assumption fails. Success rate is the right primary metric because the claim is about task completion under novel part arrangements, and per-task and per-subset success rates let us localize the predicted gain to arrangement-shifted tasks rather than uniform improvement. The segmentation part-consistency error is reported up front so that segmentation quality is a controlled variable rather than a confound in the success-rate comparison.

### Expected outcome and causal chain

**vs. DexForesight** — On a task where the same object appears with a mirrored or re-arranged part layout (e.g., a handle on the opposite side), DexForesight's holistic spatial foresight target must interpolate a novel whole-object geometry it never encoded, producing hand-motion predictions biased toward the training arrangement and grasp attempts aimed at the wrong side. Our method instead represents the object as a new slot configuration passed through the same permutation-equivariant interaction function, so the hand-motion head sees the handle slot in its actual position and predicts a correct approach. We expect a noticeable success-rate gap concentrated on arrangement-shifted task variants, with near-parity on tasks whose objects appear in a single canonical arrangement.

**vs. Whole-object world-action model** — On a task requiring fine-grained contact with a small sub-part (e.g., a knob on a cluttered object), the whole-object encoder pools that sub-part into a global feature, so its foresight prediction blurs the knob's location and the policy's approach is coarse, leading to missed contacts. Our method keeps a dedicated part-geometry token for the knob, so the cross-part attention can route hand motion to that slot specifically, and we expect higher success on contact-precision tasks while remaining comparable on coarse whole-object grasping.

**vs. Retrieval-style part-matching policy** — On a task with a novel object whose parts are semantically familiar but arranged in an unseen configuration, retrieval-style matching fails to find a stored configuration and falls back to a poor match, producing a hand trajectory copied from a mismatched template. Our method instead applies the learned equivariant interaction function to the new slot arrangement, generating a foresight prediction conditioned on the actual geometry rather than a retrieved neighbor. We expect a success gap that grows with the novelty of the arrangement, and near-parity on objects with close retrieval neighbors.

**vs. Slot-attention / object-centric dynamics model** — On arrangement-shifted tasks, the object-centric baseline factorizes the scene into slots but aggregates them with a permutation-equivariant function that discards the identity information downstream heads need (as documented for slot attention), so its hand-motion prediction is identity-agnostic and degrades when the handle moves to the opposite side. Our method keeps part-identity tokens bound to their semantic part while shuffling only spatial arrangement, so the hand head is trained to predict motion conditioned on 'which part is where'. We expect a success gap concentrated on arrangement-shifted tasks, with near-parity on canonical arrangements.

**vs. PC-Dex w/o compositional loss** — On arrangement-shifted tasks, the ablation still decomposes parts but its hand-motion head is trained only on in-distribution arrangements, so it learns to associate specific slot orderings with motions and degrades when the ordering changes. Our full method, trained under identity-preserving arrangement shuffling, must produce correct hand motion regardless of slot order, so we expect the ablation to close most of the gap on canonical tasks but lose ground specifically on the arrangement-shifted subset, confirming the compositional loss as the load-bearing ingredient.

**vs. PC-Dex w/o part-identity token (identity-agnostic control)** — This control tests the load-bearing assumption directly: if hand motion is identity-conditioned, removing the part-identity token should degrade performance on tasks where the correct grasp depends on which part is which (e.g., mug handle), and the diagnostic metric should show near-zero hand-motion change under identity-only permutations. If the control matches the full method, the assumption fails and the shuffled loss reflects an identity-agnostic motion prior rather than compositional generalization.

### What would falsify this idea

If our gain over DexForesight is uniform across all tasks rather than concentrated on arrangement-shifted and contact-precision subsets, the part-compositional mechanism is not what is driving performance. Equally, if the no-compositional-loss ablation matches the full method on arrangement-shifted tasks, or if the identity-agnostic control matches the full method on identity-dependent grasps, the claimed role of the identity-preserving arrangement-shuffle loss and the load-bearing assumption are wrong.

## References

1. DexForesight: Grounding Dexterous Policies with Implicit Spatial Foresight
