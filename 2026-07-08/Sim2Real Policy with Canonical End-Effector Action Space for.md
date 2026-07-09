# Sim2Real Policy with Canonical End-Effector Action Space for Hardware-Agnostic Robot Control

## Motivation

Existing vision-language-action models like OpenVLA (7B parameters) and the approach in 'From Foundation to Application' rely on robot-specific action spaces (joint angles or task-specific poses) and computationally expensive video/depth prediction, forcing retraining for each new platform and high compute cost at deployment. The root cause is that action representations are coupled with robot dynamics, so the policy must be re-optimized for every kinematic structure. This structural coupling prevents lightweight, generalist deployment across diverse hardware.

## Key Insight

Canonical end-effector poses from simulation form an invariant action representation across embodiments because the policy only needs to predict relative pose changes in a shared reference frame, decoupling visual perception from robot-specific joint configurations.

## Method

### A) What it is  
We propose **CanoVLA**, a lightweight policy (6-layer Transformer, ~80M parameters) that takes frozen DINOv2 visual features from a single RGB image and outputs a sequence of delta end-effector poses (position + orientation) in a canonical simulation frame. The policy is trained on multi-embodiment data and, at test time, paired with a per-robot inverse kinematics solver to convert poses to joint commands.  

**Load-bearing assumption (explicit):** The predicted canonical end-effector delta pose can be realized by any target robot via inverse kinematics without modification. This requires that the delta pose lies within the robot's reachable workspace and that a feasible joint configuration exists. To mitigate infeasibility, we incorporate a feasibility filter that checks reachability of predicted delta poses before passing to IK and uses a fallback (task-space proximity adjustment) to adjust infeasible targets.  

**Calibration step:** At deployment, a fixed transformation T_canonical_to_base is computed (e.g., via a known marker placed at the robot's base in simulation and reality) and applied to delta_pose before adding to current pose. This transformation is assumed known and constant; if calibration drifts, estimation errors accumulate.  

### B) How it works (pseudocode)  
```python
# Training
sim_stack = MuJoCo + RLBench  # simulation environment
for each episode in multi-embodiment dataset:
    # Step 1: Record robot-agnostic canonical trajectory from simulation
    canonical_traj = get_ee_poses(sim, in_canonical_frame=True)  # shape (T, 7)
    # Filter trajectories: skip any episode where any delta exceeds the reachable workspace of any embodiment (e.g., max position delta > 0.1m?)
    if not all_feasible(canonical_traj): continue
    # Step 2: Extract visual features offline
    for t in range(T):
        img = get_rgb_frame(t)
        feat_t = DINOv2(img)  # frozen, ViT-B/8
    # Step 3: Train CanoVLA to autoregressively predict delta poses
    # Model: 6-layer Transformer with 8 heads, hidden dim 512, feed-forward 1024
    # Loss: L2 on delta position (3-dim), L2 on quaternion delta (4-dim)
    optimizer = AdamW(lr=1e-4, weight_decay=0.01)
    for t in range(1, T):
        context = [feat_t, pos_emb(t)]  # pos_emb: learned timestep embedding, 128 dim
        pred_delta = CanoVLA(context)
        loss += L2(pred_delta[:3], canonical_traj[t][:3] - canonical_traj[t-1][:3]) + L2(pred_delta[3:], quaternion_delta(canonical_traj[t-1], canonical_traj[t]))

# Deployment
calib_T = compute_calibration()  # fixed 4x4 transform from canonical to robot base
for step in robot_run:
    img = robot_camera()
    feat = DINOv2(img)
    delta_pose = CanoVLA([feat, pos_emb(step)])
    # Apply calibration
    delta_pose_transformed = calib_T * delta_pose
    target_pose = current_ee_pose + delta_pose_transformed  # in robot's base frame
    # Feasibility filter: if target_pose not in reachable workspace, adjust delta to nearest feasible pose via linear interpolation
    if not is_reachable(target_pose, robot):
        target_pose = adjust_to_feasible(target_pose, current_ee_pose, robot)
    joint_cmd = IK_solver.ik(target_pose, q_current)  # PyBullet damped least-squares, damping=0.05
    robot.execute(joint_cmd)
```

### C) Why this design  
We chose a delta-pose action space rather than absolute poses to avoid drift and allow the policy to focus on relative changes, which are more invariant across viewpoints. We chose a lightweight Transformer (6 layers, 512 hidden) instead of large language models (like Llama 7B in OpenVLA) because the action space is structured and low-dimensional, removing the need for large capacity—this trades expressiveness for speed and hardware-agnostic finetuning cost. We use frozen DINOv2 features instead of finetuning a visual encoder to preserve generalizability across diverse scenes; the cost is potential domain mismatch with simulated training data, which we mitigate by using a diverse multi-embodiment sim dataset. We opted for a simple autoregressive decoder rather than a diffusion or flow-matching head because direct regression on poses is sufficient for quasi-static manipulation and reduces runtime complexity. The feasibility filter ensures that predicted deltas remain within robot-specific constraints without retraining; if the filter is triggered frequently, performance degrades because the adjusted pose may not follow the intended task.  

### D) Why it measures what we claim  
The canonical frame action space is designed to remove embodiment-specific dynamics from the policy output. The computational quantity `canonical_traj[t] - canonical_traj[t-1]` measures the relative end-effector motion in a coordinate frame aligned to a standard robot base in simulation, which we claim is hardware-agnostic because any real robot's end-effector trajectory can be expressed as a sequence of such deltas after calibration (base frame alignment via T_canonical_to_base). This assumption holds when the predicted delta lies within the robot's reachable workspace; it fails when the real robot has severe joint limits or different reachability (e.g., a long-arm vs a short-arm), in which case the feasibility filter adjusts the pose but may deviate from the intended task. The L2 loss on deltas measures regression accuracy in that canonical space, which we claim corresponds to task success across embodiments; this is valid if the IK solver and feasibility filter together map the predicted poses to feasible joint angles—if not, accumulated error or singularity causes failure. The frozen DINOv2 feature extractor measures visual invariance to task-relevant objects and scenes, assuming that the training sim scenes cover sufficient variety; when test scenes are out-of-distribution (e.g., novel object shapes), features may mislead, and the policy's regression then reflects mismatch rather than generic skill.

## Contribution

(1) A hardware-agnostic action space defined by canonical end-effector poses in simulation, decoupling visual policy from robot dynamics. (2) A lightweight NVP (6-layer Transformer, 80M parameters) that maps visual features to this action space, demonstrating that a small model can generalize across embodiments without retraining. (3) A multi-embodiment dataset of canonical end-effector trajectories generated from simulation, enabling training of such decoupled policies.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Multi-Embodiment Manipulation (MEM) | Covers 5 robot arms, 20 tasks |
| Primary metric | Cross-embodiment success rate | Average task completion across all robots |
| Baseline 1 | OpenVLA (7B) | Large pretrained VLA, not embodiment-aware |
| Baseline 2 | Octo (base) | Modular VLA, uses tokenized actions |
| Baseline 3 | AbsPose-VLA (our arch.) | Same model but predicts absolute poses |
| Ablation | CanoVLA-finetuned-feat | Same model but finetune DINOv2 |

**Comparison to prior relative-action works** (novelty): Unlike GAPart (which outputs part-level poses only) and UniPi (which predicts video frames and extracts actions), our canonical frame enables zero-shot transfer across embodiments without retraining or video generation. We achieve this by decoupling action representation from robot dynamics via a fixed canonical frame and per-robot IK. 

### Why this setup validates the claim

The multi-embodiment dataset directly tests the claim of hardware-agnosticism by including diverse robot kinematics and tasks. OpenVLA and Octo serve as strong, widely-used VLA baselines that lack explicit embodiment abstraction; their failure patterns will highlight the benefit of canonical delta actions. AbsPose-VLA isolates the action space contribution—if delta outperforms absolute, the mechanism is validated. The ablation checks whether frozen visual features are crucial for generalization. Cross-embodiment success rate measures the central functional claim: the policy's ability to produce feasible commands for any robot, aggregated over all tasks and embodiments. The feasibility filter and calibration step are integral to the method; we evaluate their impact by measuring filter activation frequency and task success with vs. without filter. We also report per-embodiment success rates to identify which kinematics benefit most.

### Expected outcome and causal chain

**vs. OpenVLA** — On a case where a novel robot (e.g., a short-arm robot) must reach an object at the edge of its workspace, OpenVLA predicts joint angles or absolute poses that exceed joint limits or require a non-existent reach, because its 7B model overfits to its pretraining embodiment (e.g., Franka). Our method predicts delta poses in a canonical frame, which are relative and small, so the IK solver with feasibility filter finds a feasible joint solution within limits. We expect a large gap (e.g., >30% success) on novel embodiments, but parity on the training robot.

**vs. Octo** — On a multi-step task (e.g., pick-and-place) with calibration offsets between simulation and real robot base frames, Octo's absolute end-effector actions accumulate drift because they are grounded in a fixed robot frame. Our delta action space is invariant to base frame errors since it predicts changes relative to the current pose, and calibration transformation T_canonical_to_base is applied once. We expect our success rate to be higher on long-horizon tasks (e.g., 60% vs. 30%), with minimal drift.

**vs. AbsPose-VLA** — On a task with high viewpoint variation (e.g., camera angle shifts across instances), AbsPose-VLA predicts absolute poses that are sensitive to camera extrinsics, causing systematic errors in canonical frame alignment. Our delta predictions focus on relative motion, reducing sensitivity to viewpoint. We expect a noticeable gap (e.g., 15-20% higher success) on tasks with camera variation, but similar performance when viewpoint is fixed.

**Ablation (CanoVLA-finetuned-feat)** — On out-of-distribution scenes with novel textures or lighting, finetuned DINOv2 features overfit to the training sim appearance, leading to incorrect pose regression. Frozen DINOv2 retains broad visual invariance. We expect the frozen version to outperform finetuned by >10% on OOD scene subsets.

### What would falsify this idea

If our method does not outperform AbsPose-VLA on tasks with viewpoint variation, then the delta action space is not the source of robustness, or the calibration step is insufficient. Alternatively, if the feasibility filter is triggered in >50% of steps (indicating the canonical deltas are often infeasible), then the load-bearing assumption is false and the method cannot achieve task success reliably.

## References

1. From Foundation to Application: Improving VLA Models in Practice
2. Igniting VLMs toward the Embodied Space
3. FastUMI: A Scalable and Hardware-Independent Universal Manipulation Interface with Dataset
4. Robotic Control via Embodied Chain-of-Thought Reasoning
5. OpenVLA: An Open-Source Vision-Language-Action Model
6. GELLO: A General, Low-Cost, and Intuitive Teleoperation Framework for Robot Manipulators
7. Scalable. Intuitive Human to Robot Skill Transfer with Wearable Human Machine Interfaces: On Complex, Dexterous Tasks
8. VILA: On Pre-training for Visual Language Models
