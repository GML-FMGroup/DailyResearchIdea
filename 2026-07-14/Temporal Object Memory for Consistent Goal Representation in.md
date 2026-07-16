# Temporal Object Memory for Consistent Goal Representation in Dynamic Visual Language Navigation

## Motivation

Existing VLN methods like ABot-N1 rely on pixel goals that can be lost under occlusion because they lack explicit cross-frame identity tracking. The VLM must implicitly maintain object identity through noisy visual features, leading to goal drift when objects appear, disappear, or are occluded. This structural gap—the absence of an explicit memory module for tracking objects under occlusion—recurs across several branches including SocialNav and NavForesight, which ignore memory entirely.

## Key Insight

By coupling a differentiable memory state with an egomotion-conditioned predictive update, the identity of navigational goals is maintained as a latent variable that evolves predictively, making the system robust to occlusion without requiring explicit re-identification.

## Method

### (A) What it is
Temporal Object Memory (TOM) is a differentiable module that maintains a set of 8 object identity slots, each storing a 128-dimensional latent feature vector and a 3D position in the robot's reference frame. Input: current RGB observation (224x224) and egomotion (3D translation, 3D rotation as quaternion). Output: updated memory slots with predicted features and positions, and a refined pixel goal for navigation.

### (B) How it works
```python
def temporal_object_memory(rgb, depth, egomotion, object_detections, target_instruction_embedding):
    # object_detections: list of (bbox, visual_feat) from DINOv2 on rgb (patch size 14, output dim 768, then projected to 128)
    # Step 1: Predict slot positions using egomotion (assumes static world)
    ego_translation, ego_rotation = decompose_egomotion(egomotion)  # 3D translation vector, 3x3 rotation matrix
    for slot in slots:
        # Apply rotation then translation: new_pos = R * old_pos + t
        slot.pos = ego_rotation @ slot.pos + ego_translation
    
    # Step 2: Compute 3D positions of detections from depth (aligned depth map, intrinsic matrix K)
    detection_positions = [pixel_to_3d(det.bbox.center, depth, K) for det in object_detections]
    
    # Step 3: Soft assignment based on both position and appearance affinity (learned)
    # cost matrix: weighted combination of position L2 distance and feature cosine distance
    pos_cost = pairwise_l2_distance(slots.pos, detection_positions)  # shape [8, N]
    feat_cost = pairwise_cosine_distance(slot.feat, [det.visual_feat for det in object_detections])  # shape [8, N]
    cost_matrix = 0.5 * pos_cost + 0.5 * feat_cost  # weight alpha=0.5, learnable
    assignment = gumbel_sinkhorn(cost_matrix, tau=0.1, num_iterations=5)  # soft assignment, differentiable
    
    # Step 4: Update slots via cross-attention transformer (2 layers, 8 heads, 128 dim)
    slot_feats = [slot.feat for slot in slots]  # [8, 128]
    detection_feats = [det.visual_feat for det in object_detections]  # [N, 128]
    # Cross-attention: queries=slot_feats, keys=detection_feats, values=detection_feats; attention weighted by assignment
    # Implementation: use assignment as attention mask (soft), then multi-head attention
    updated_feats = cross_attention(queries=slot_feats, keys=detection_feats, values=detection_feats, assignment=assignment)
    
    # Step 5: Update slot features and positions with a learnable gate (gating parameter g=0.1, sigmoid range)
    for i, slot in enumerate(slots):
        gate = torch.sigmoid(slot.gate_param)  # slot-specific gate, initialized to -2.2 (gives ~0.1)
        slot.feat = slot.feat + gate * (updated_feats[i] - slot.feat)  # residual update
        # position update: weighted average of predicted and observation-aligned
        slot.pos = (1 - gate) * slot.pos + gate * (assignment[i] @ detection_positions)
    
    # Step 6: Select target slot using instruction embedding (from frozen CLIP text encoder, 512 dim, projected to 128)
    target_slot_idx = argmax(cosine_similarity(slot_feats, target_instruction_embedding))
    target_pos = slots[target_slot_idx].pos
    pixel_goal = project_3d_to_pixel(target_pos, camera_matrix)  # using K and no distortion
    return pixel_goal

# Hyperparameters: num_slots=8, slot_dim=128, update_lr=0.1 (gate learning rate), cross-attention: 2 layers, 8 heads, hidden dim 512, feedforward 1024, GELU activation
```

### (C) Why this design
We chose a fixed number of slots (8) over dynamic allocation to keep the matching tractable; this may limit simultaneous tracking but is sufficient for a single goal object. The egomotion prediction uses a simple rigid transformation (rotation + translation) because egomotion is known and the static-world assumption is reasonable for navigation, but moving objects cause errors. The assignment uses a weighted combination of position and feature affinity (with learnable alpha=0.5) to handle both positional and appearance changes, enhancing robustness over pure position matching. The cross-attention transformer with 2 layers and 8 heads provides a moderate capacity for feature integration without overfitting. We update features with a learnable gated residual (initial gate ~0.1) to avoid overwriting long-term memory, but this trade-off may slow adaptation to appearance changes. Finally, we select the target slot by cosine similarity to an instruction embedding from a frozen CLIP text encoder, which is efficient but may misalign if the embedding is coarse.

### (D) Why it measures what we claim
The slot position update from egomotion measures temporal consistency because it predicts the object's location under the assumption of a static world; this assumption fails when objects move, and the position then reflects the old location, causing drift. The soft assignment (combining position and feature affinity) measures identity preservation by associating detections to slots based on both proximity and feature similarity; this assumes features are discriminative, and when two objects cross and have similar features, assignment can still be ambiguous. The cross-attention transformer measures feature integration by combining new observations with stored memory through a gated residual; this assumes the gate balances old and new appropriately, and if appearance changes drastically, the update may still blend features if gate is too high. The target slot selection measures goal relevance by cosine similarity to the instruction; this assumes the slot feature represents the goal object, and if the instruction is ambiguous, the wrong slot may be chosen.

## Contribution

(1) Temporal Object Memory (TOM), an explicit differentiable memory module that maintains object identity across frames in VLN, enabling robust navigation under occlusion. (2) Design principle that decoupling object identity maintenance from VLM reasoning via egomotion-conditioned predictive updates reduces goal drift without increasing model size. (3) A practical integration with DINOv2 features and Hungarian matching for object-slot association, providing a reusable component for future VLN systems.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | REVERSE-val (unseen) | Tests object-goal navigation with language; 2019 episodes with occlusions labeled. |
| Primary metric | Success Rate (SR) | Directly measures goal completion. |
| Secondary metric | Position Error (PE) over time | Tracks the L2 distance between predicted goal position and ground-truth object centroid at each step; measures consistency. |
| Baseline 1 | HAMT | Strong transformer baseline without temporal memory. |
| Baseline 2 | VLN-BERT | Vision-language baseline without object slots. |
| Baseline 3 | Learned-Object-Goal | Uses detections but no persistent memory (per-frame DINOv2 detection, select closest to instruction embedding). |
| Ablation-of-ours | TOM w/o temporal update (static slot reuse) | Only current frame detections; slots initialized from first frame and never updated after occlusion. |
| Compute | 4 NVIDIA A100 40GB GPUs, training 3 days, evaluation 6 hours. | |

### Why this setup validates the claim

The core claim is that Temporal Object Memory (TOM) improves navigation by maintaining persistent object identity across frames. REVERIE evaluates object-goal navigation from language instructions, requiring recognition and tracking of a target object. Success Rate directly captures whether the robot reaches the correct object. The secondary metric Position Error (PE) measures temporal consistency by comparing the predicted goal position to ground truth at each step; lower PE indicates better tracking. Baselines span architectures with and without object awareness: HAMT and VLN-BERT rely on visual-textual cross-attention but lack explicit object persistence; Learned-Object-Goal uses per-frame detections without memory. Our ablation isolates the temporal update mechanism. If TOM's superiority stems from retaining object identity over time, it should outperform all baselines on episodes with occlusions (e.g., target hidden for >3 steps), and the ablation gap should be largest in those episodes. The metrics SR and PE directly reflect navigation success and tracking consistency.

### Expected outcome and causal chain

**vs. HAMT** — On a case where the target object appears, disappears behind an obstacle for 5 steps, and reappears later, HAMT treats each view independently; when the object is occluded, its cross-attention may drift to a distractor, causing a wrong turn. Our method retains the object's feature and predicted position in a slot during occlusion, then re-associates upon reappearance via assignment, maintaining goal consistency. We expect a noticeable gap in SR on episodes with occlusions (subset where occlusion duration >3 steps) but parity on straight-line paths. PE for TOM should remain low during occlusion (<0.5m drift) while HAMT's PE spikes (>2m).

**vs. VLN-BERT** — On a case with multiple similar objects (e.g., two red chairs 3m apart), VLN-BERT's global self-attention can confuse their appearances across time, leading to a wrong chair after a turn. Our slot mechanism assigns each detection to a specific identity based on position and appearance features, preserving the target chair's identity even if features are similar. We expect a larger gap on scenes with multiple instances of the same category (e.g., multiple chairs or tables) compared to distinct objects. PE for TOM should stay below 0.8m, while VLN-BERT may exceed 2m after the wrong turn.

**vs. Learned-Object-Goal** — On a case where the target object is partially visible initially (only edge at doorway), then fully visible later, the baseline uses only current detections; if the initial partial detection is missed, it may never commit to a goal. Our method updates slots gradually via gated residual, so even a weak detection (low confidence) can seed the slot and refine over time. We expect the baseline to fail more frequently when the target is first glimpsed briefly (e.g., through a doorway for 2 steps), while our method succeeds. PE for TOM should decrease from 1.5m to 0.3m as more observations are integrated, while baseline PE fluctuates.

**vs. TOM w/o temporal update** — The ablation uses static slots (no update) after first detection. On a case where target is occluded for 5 steps, the slot position drifts due to egomotion only, and on reappearance the old feature may not match, causing assignment failure. This ablation quantifies the contribution of temporal updates. We expect SR drop of >15% on occlusion episodes.

### What would falsify this idea

If TOM's gain over the non-temporal baseline (Learned-Object-Goal) is uniform across all navigation episodes rather than concentrated in scenarios with occlusions, re-acquisitions, or identity confusion, then the central claim that temporal memory drives improvement would be falsified. Specifically, if the gap in SR on a held-out occlusion-rich subset is less than 5% relative to the overall gap, the idea is falsified.

## References

1. ABot-N1: Toward a General Visual Language Navigation Foundation Model
2. SocialNav: Training Human-Inspired Foundation Model for Socially-Aware Embodied Navigation
3. NavForesee: A Unified Vision-Language World Model for Hierarchical Planning and Dual-Horizon Navigation Prediction
4. CityWalker: Learning Embodied Urban Navigation from Web-Scale Videos
5. NavCoT: Boosting LLM-Based Vision-and-Language Navigation via Learning Disentangled Reasoning
6. ImagineNav: Prompting Vision-Language Models as Embodied Navigator through Scene Imagination
7. DINO-Foresight: Looking into the Future with DINO
8. DINOv2: Learning Robust Visual Features without Supervision
