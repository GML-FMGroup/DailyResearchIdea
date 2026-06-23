# Geometric Consistency Self-Supervision for Deterministic Parallel Decoders via Differentiable Physics

## Motivation

The Fine-Tuning Vision-Language-Action Models (OFT) method achieves high success rates by using parallel decoding and action chunking, but this deterministic prediction of multiple future actions without uncertainty requires dense task-specific demonstration data. This dependency arises because the decoder lacks a self-supervised signal to verify the geometric plausibility of its outputs, limiting scalability and zero-shot generalization.

## Key Insight

Differentiable physics simulation provides a physically grounded consistency signal by comparing the rendered depth image of a predicted action chunk to the observed depth after execution, enabling self-supervised training that reduces the need for action labels.

## Method

### (A) What it is
GeoConsistency is a self-supervised training framework that enforces geometric consistency between predicted action chunks and observed depth images using a differentiable physics simulator. It takes as input an RGB-D observation and a language instruction, and outputs a deterministic action chunk of length K. The training uses unlabeled interaction trajectories (sequences of observations and recorded actions) to compute a consistency loss.

### (B) How it works
```python
# GeoConsistency Training Loop
# Hyperparameters: K=8 (chunk size), lambda_g=1.0 (geometric loss weight), lambda_a=0.1 (action loss weight, if sparse demo available)
# Calibration: before training, evaluate simulator depth error on 100 random transitions; if mean L1 depth > 5 px, train a residual correction network (2-layer CNN, 64 filters, ReLU) on rendered depth.

for each training step:
    sample a transition (obs_t, action_t, obs_{t+1}) from unlabeled dataset
    # obs_t includes RGB-D, action_t is a single executed action
    # Predict action chunk from current policy (e.g., a 6-layer transformer with hidden dim 512, 8 heads)
    chunk_pred = policy(obs_t, instruction)  # shape (K, action_dim)
    
    # Use differentiable physics simulator to compute outcome of first action of chunk
    # Assume sim can roll out action chunk and render depth at each step
    # For simplicity, we simulate first action only
    current_state = extract_state(obs_t)  # e.g., object poses, robot joint angles from simulator ground truth
    next_state = physics_sim.step(current_state, chunk_pred[0])  # differentiable (e.g., IsaacGym with analytical gradients)
    depth_pred = physics_sim.render_depth(next_state)  # differentiable (rendering via soft rasterizer)
    
    # Optional: apply learned residual correction to depth_pred if calibration step was triggered
    if residual_corrector is trained:
        depth_pred = residual_corrector(depth_pred)
    
    # Geometric consistency loss: L1 between predicted depth and observed depth
    depth_obs = extract_depth(obs_{t+1})
    L_g = L1(depth_pred, depth_obs)
    
    # Optional: if sparse demo action labels exist for this state
    L_a = L1(chunk_pred, action_label) if action_label available else 0
    
    total_loss = lambda_g * L_g + lambda_a * L_a
    update policy via gradient descent (Adam, lr=1e-4)
```

### (C) Why this design
We chose to use differentiable physics simulation over a learned world model because the physics model provides physically grounded consistency that generalizes without task-specific training, accepting the computational cost of simulation at each gradient step (approximately 0.1 seconds per step on an RTX 4090). We chose to compare rendered depth images rather than other state representations (e.g., object poses) because depth is directly observable and avoids the need for state estimation; the trade-off is that depth consistency may be insensitive to certain fine-grained motions (e.g., precise object grasping). We chose to include actions from unlabeled interactions as conditioning for the consistency check rather than relying solely on random actions; the cost is that the dataset actions may be suboptimal, but the consistency loss still provides a signal for the policy to match those actions' outcomes. We chose to simulate only the first action of the chunk for computational efficiency, though simulating the full chunk could provide a stronger signal; we accept the reduced temporal horizon to keep training tractable.

### (D) Why it measures what we claim
The depth rendering difference between predicted and observed depth images measures geometric consistency between the predicted action's outcome and the real world, which is equivalent to the policy's ability to predict physically plausible actions that respect scene geometry; this assumption fails when the physics simulator inaccurately models real-world dynamics (e.g., friction, deformations), in which case the loss reflects simulation-reality mismatch rather than action quality. To mitigate this, we introduce a calibration step: before training, we evaluate the simulator's depth prediction error on 100 held-out random transitions; if the mean L1 depth error exceeds 5 pixels, we train a learned residual correction network (2-layer CNN, 64 filters, ReLU) to adjust the rendered depth before computing the loss. The action prediction loss from sparse demonstrations directly measures task-specific action correctness; this assumption fails when demonstrations are not representative of the full task distribution, in which case the loss biases the policy towards a narrow subset. The use of unlabeled interaction data ensures that the policy learns from diverse states without requiring human labels; this assumption fails if the interaction data does not cover relevant state-action regions, in which case the geometric loss may saturate prematurely.

## Contribution

(1) A self-supervised training framework, GeoConsistency, that uses differentiable physics simulation to enforce geometric consistency between predicted action chunks and observed depth images, reducing the need for dense demonstration action labels. (2) The design principle that deterministic parallel decoders can be trained with minimal action supervision by leveraging physical consistency as a dense reward signal.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|------------------------|
| Dataset | LIBERO simulation benchmark | Standard benchmark for spatial manipulation |
| Primary metric | Task success rate | Directly measures task completion performance |
| Baseline 1 | OpenVLA (fine-tuned on demos) | Strong prior VLA approach without geometric grounding |
| Baseline 2 | Behavioral cloning with learned world model | Isolates benefit of differentiable physics grounding |
| Baseline 3 | GeoConsistency without chunking (single-step action prediction with differentiable physics loss) | Isolates contribution of chunking |
| Ablation 1 | GeoConsistency without geometric loss | Tests necessity of geometric consistency loss |
| Ablation 2 | GeoConsistency without calibration (no residual correction) | Tests importance of sim-to-real calibration |
| Demonstration trajectories | Vary: 1, 10, 100 trajectories | Quantify reduction in need for action labels |
| Validation set size | 200 episodes for correlation analysis | Compute Spearman ρ between avg. geometric loss and success rate |

### Why this setup validates the claim
This experimental design tests the hypothesis that enforcing geometric consistency via differentiable physics improves spatial reasoning in VLA models. Using LIBERO, a simulation benchmark with diverse spatial tasks, success rate captures whether predicted actions lead to correct physical outcomes. Comparing against OpenVLA, which relies solely on demonstration action labels, isolates the benefit of geometric grounding. The learned world model baseline assesses the necessity of physics-based simulation versus a data-driven alternative. The ablation removes the geometric loss, verifying its contribution. The chunking-vs-single-step baseline measures whether the chunking mechanism provides additional benefit beyond geometric consistency. The calibration ablation tests the importance of addressing sim-to-real gap. Varying demonstration sizes directly tests the claim that geometric loss reduces label dependency. Finally, computing Spearman correlation between geometric loss and success rate on a validation set provides quantitative evidence that the loss is predictive of task performance. If our method outperforms both baselines, it supports that differentiable physics provides a stronger signal for spatial consistency. Moreover, the ablation should degrade, confirming the loss’s role. Together, this setup provides a falsifiable test: if our method fails to improve over OpenVLA, the geometric consistency assumption is insufficient.

### Expected outcome and causal chain
**vs. OpenVLA** — On a case where the demonstration dataset does not cover a specific geometric arrangement (e.g., objects in novel spatial configuration), OpenVLA may output an action that ignores depth constraints because it only matches action distributions from demos. Our method uses the geometric loss to enforce that the predicted action's outcome matches observed depth, enabling generalization to unseen arrangements. We expect a noticeable gap in success rate on tasks with novel object positions, with our method achieving higher success (e.g., 70% vs 45%) while performing similarly on standard arrangements.

**vs. Behavioral cloning with learned world model** — On a case where a learned world model mispredicts dynamics due to distribution shift (e.g., unseen object mass), the baseline's action prediction may be physically inconsistent because its world model generalizes poorly. Our method uses a differentiable physics simulator with fixed dynamics, ensuring physical plausibility. We expect our method to maintain performance under such distribution shifts, while the baseline's success rate drops significantly (e.g., 60% to 30% vs our 70% to 65%).

**vs. GeoConsistency without chunking** — On tasks requiring precise temporal sequencing (e.g., pushing an object then grasping it), the no-chunking baseline predicts actions independently, potentially leading to conflicting motions. Our method's chunking allows the policy to plan a coherent sequence that satisfies geometric consistency at each step. We expect our method to outperform on such temporal coordination tasks (e.g., 75% vs 60% success).

### What would falsify this idea
If our method shows improvement on standard tasks but fails to outperform OpenVLA specifically on tasks requiring novel spatial generalization, then geometric consistency is not the key factor.

## References

1. Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success
2. TinyVLA: Toward Fast, Data-Efficient Vision-Language-Action Models for Robotic Manipulation
3. π0: A Vision-Language-Action Flow Model for General Robot Control
4. Visual Instruction Tuning
5. MiniGPT-v2: large language model as a unified interface for vision-language multi-task learning
6. Scaling Instruction-Finetuned Language Models
7. Self-Instruct: Aligning Language Models with Self-Generated Instructions
8. Unnatural Instructions: Tuning Language Models with (Almost) No Human Labor
