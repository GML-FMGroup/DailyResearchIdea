# Flow-Factorized World Models: Temporal Consistency via Optical Flow Supervision and Scene Decomposition

## Motivation

Existing world models like MindJourney and EgoWM generate frames independently or with weak action conditioning, leading to temporally inconsistent object motion in dynamic scenes. This structural limitation arises because they treat the entire scene as a single latent representation without distinguishing between persistent scene structure and transient object motion, and they lack a dense motion signal that ties actions to pixel-level changes, relying instead on weak action conditioning or static scene assumptions.

## Key Insight

By factorizing the scene representation into persistent and transient components and enforcing cross-time consistency via a loss that uses optical flow as a differentiable bridge between actions and pixel motion, the model can learn temporally consistent dynamics without requiring explicit object tracking.

## Method

(A) **What it is**: Flow-Factorized World Model (FFWM) is a world model that takes an initial observation and action sequence as input and outputs a temporally consistent video rollout. It factorizes the scene into a persistent canonical background and transient object representations, and uses optical flow as a supervisory signal for cross-time consistency.

(B) **How it works**:
```python
# Pseudocode for FFWM training
# Input: initial observation x0, action sequence a_1:T
# Encoder: E_phi maps x0 to persistent representation z_pers and transient z_trans_0
z_pers, z_trans_0 = Encoder_phi(x0)

# Dynamics model: D_theta predicts future transient states conditioned on actions
for t in 1..T:
    z_trans_t = D_theta(z_trans_{t-1}, a_t)
    # Decoder: G_psi generates frame from combined representations
    x_t_pred = G_psi(z_pers, z_trans_t)

# Optical flow estimation: FlowNet (pretrained, frozen) computes flow between consecutive predicted frames
flow_pred_t = FlowNet(x_{t-1}_pred, x_t_pred)

# Ground-truth flow: We warp x_{t-1}_pred to align with x_t_pred using flow_pred_t
warped_t = warp(x_{t-1}_pred, flow_pred_t)

# Loss: L_total = L_recon + lambda * L_flow_consistency
L_recon = MSE(x_t_pred, x_gt_t)  # pixel reconstruction loss
L_flow_consistency = MSE(warped_t, x_t_pred)  # enforces that predicted flow explains motion
# Also optionally: action-conditioned flow matching if action directly maps to flow (e.g., known camera intrinsics)
# Hyperparameters: lambda=0.1, learning rate=1e-4
```

(C) **Why this design**: We chose three key design decisions. First, we factorize the scene into persistent and transient components rather than using a monolithic latent representation (as in EgoWM) because it forces the model to separate static from dynamic content, enabling explicit handling of object motion. The trade-off is that the factorization may be imperfect if the scene has multiple moving objects or non-rigid deformations, requiring a more complex encoder. Second, we use optical flow as a supervisory signal via a flow consistency loss rather than relying solely on pixel reconstruction because flow provides a dense, motion-aware signal that directly ties actions to pixel displacements, unlike MindJourney's weak conditioning. The cost is that flow estimation is computationally expensive and may be inaccurate under large motions or occlusions, requiring a robust pretrained flow network (e.g., RAFT) and careful loss weighting. Third, we employ a simple L2 loss on warped frames rather than a more complex contrastive or adversarial loss because it is computationally efficient and directly optimizes for temporal consistency. The drawback is that L2 loss can be sensitive to image blur and may not capture high-frequency details, but we compensate by using pixel reconstruction loss in parallel.

(D) **Why it measures what we claim**: The flow consistency loss L_flow_consistency measures temporal consistency because it enforces that the optical flow computed between consecutive predicted frames accurately explains the pixel-level changes, assuming the flow network is a reliable proxy for true motion. This assumption fails when the flow network produces inaccurate flow (e.g., due to large motion, occlusions, or domain shift), in which case the loss may penalize the model for correct motion that the flow network cannot capture, or reward static predictions that accidentally align with incorrect flow. The persistent representation z_pers measures scene structure persistence because it remains unchanged across time, assuming the encoder successfully separates static background from moving objects; this assumption fails when the background itself moves (e.g., panning camera), in which case z_pers would need to incorporate camera motion or be updated. The transient representation z_trans measures action-contingent object motion because its dynamics are learned directly from action sequences; this assumption fails when object motion is not fully determined by actions (e.g., independent object motion), in which case the model may need additional latent variables to capture stochasticity. By combining these components, the method directly operationalizes the motivation-level concepts of temporal consistency and action-conditioned dynamics.

## Contribution

(1) A novel world model architecture that factorizes scene representation into persistent and transient components, enabling explicit modeling of dynamic objects while maintaining scene consistency. (2) A cross-time consistency loss using optical flow as a supervisory signal, which ties actions to pixel-level motion without requiring explicit object tracking or segmentation. (3) Empirical demonstration on egocentric manipulation tasks that the method produces temporally consistent rollouts that better align with action-conditioned dynamics compared to baselines like EgoWM and MindJourney.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | Ego4D (point-of-view video) | Largest egocentric dataset with actions |
| Primary metric | Structural Consistency Score (SCS) | Measures physical plausibility of motion |
| Baseline 1 | EgoWM (monolithic latent) | Tests factorization benefit |
| Baseline 2 | MindJourney (video diffusion) | Tests flow supervision vs. diffusion |
| Baseline 3 | Action-conditional LSTM video predictor | Tests naive temporal model |
| Ablation | FFWM w/o flow consistency loss | Isolates flow loss contribution |

### Why this setup validates the claim

Ego4D provides diverse egocentric videos with camera motion and object interactions, which stress temporal consistency. SCS directly quantifies structural coherence of predicted motion, aligning with our core claim. The baselines isolate key sub-claims: EgoWM tests the benefit of factorization, MindJourney tests flow supervision against generative diffusion, and the LSTM tests the impact of learned dynamics without structure. The ablation removes the flow consistency loss to verify its necessity. Together, these comparisons create a falsifiable test: if FFWM outperforms on dynamic scenes where factorization and flow matter, the claim is supported; if gains are uniform or reversed, the idea fails.

### Expected outcome and causal chain

**vs. EgoWM** — On a case where the camera pans while an object moves independently, EgoWM’s monolithic latent confuses background and object motion, producing warped artifacts. Our method factorizes persistent background and transient object, so background remains sharp and object motion is accurately conditioned on actions. We expect a noticeable SCS gap on videos with independent object motion, but parity on static scenes.

**vs. MindJourney** — On a case with small object displacement (e.g., shifting a cup), MindJourney’s diffusion model generates plausible frames but often misses precise pixel-level alignment due to lack of explicit motion supervision. Our flow consistency loss directly optimizes for accurate warping, leading to higher SCS on fine-grained motion tasks. Expect a larger gap on subsets with subtle object movements.

**vs. Action-conditional LSTM** — On a long-horizon sequence (50+ steps), the LSTM accumulates errors from recurrent dynamics, causing severe drift and blur. With our persistent representation, background remains fixed and transient dynamics are short-term, reducing drift. We expect a growing SCS advantage as horizon increases, with the LSTM degrading rapidly.

**Ablation: FFWM w/o flow consistency** — On a video with fast object motion, removing the flow loss causes the model to rely solely on pixel reconstruction, which often predicts static backgrounds or blurred objects. The resulting SCS drops significantly because temporal coherence is lost. This confirms that the flow loss is essential for our claim.

### What would falsify this idea

If FFWM’s SCS improvement over EgoWM is uniform across all subsets (instead of concentrated on dynamic scenes) or if removing the flow consistency loss does not degrade SCS on fast-motion cases, then the central claims about factorization and flow supervision are not supported.

## References

1. Walk through Paintings: Egocentric World Models from Internet Priors
2. MindJourney: Test-Time Scaling with World Models for Spatial Reasoning
3. Expanding Performance Boundaries of Open-Source Multimodal Models with Model, Data, and Test-Time Scaling
4. Unified World Models: Coupling Video and Action Diffusion for Pretraining on Large Robotic Datasets
5. Wan: Open and Advanced Large-Scale Video Generative Models
6. Diffusion Forcing: Next-token Prediction Meets Full-Sequence Diffusion
7. Prediction with Action: Visual Policy Learning via Joint Denoising Process
