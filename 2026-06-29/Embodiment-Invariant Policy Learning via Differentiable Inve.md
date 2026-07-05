# Embodiment-Invariant Policy Learning via Differentiable Inverse Kinematics and Domain Randomization

## Motivation

Current vision-language-action models (e.g., X-VLA) require per-embodiment adaptation via separate soft prompts or fine-tuning, because they condition action generation on robot identity. This structural assumption—that embodiment is a conditioning variable rather than a nuisance variable—prevents zero-shot transfer to unseen robots without any target-platform data. We argue that this bottleneck can be broken by decoupling task-level actions (end-effector poses) from embodiment-specific kinematics through a fixed differentiable inverse kinematics module.

## Key Insight

By outputting end-effector poses and using the robot's known kinematic model as a fixed differentiable transformation, the policy learns embodiment-invariant representations because the kinematic parameters become a nuisance variable that can be marginalized via domain randomization during training.

## Method

We propose **Embodiment-Agnostic Policy (EAP)**, a vision-language-action model that outputs end-effector poses in a canonical task space. A differentiable inverse kinematics (IK) module converts these poses to joint commands using the robot's kinematic model. The model is trained with domain randomization over robot kinematic parameters, forcing the policy to ignore embodiment-specific details.

```pseudocode
Input: Image observation o_t, language instruction l, robot kinematic parameters k (sampled from distribution K during training)
Output: Joint position command q_t

// 1. Encode observation and instruction
v_t = VisionEncoder(o_t)  // SigLIP-So400m
n_t = LanguageEncoder(l)   // Gemma-2B text encoder
h = CrossAttention(v_t, n_t)

// 2. Predict end-effector pose in canonical space
eef_pose = PosePredictionHead(h)  // 6-DOF: position (3) + orientation (6D rotation)

// 3. Differentiable IK: convert eef_pose to joint commands using robot kinematics
q_t = DifferentiableIK(eef_pose, k)  // Levenberg-Marquardt, 5 iterations, step 0.1, damping 1e-3

// 4. Execute q_t on robot
```

**Why this design**: We chose to output end-effector poses in a canonical space (6-DOF) rather than joint positions because end-effector poses are embodiment-independent by definition; the kinematic model then maps this to embodiment-specific commands. This decouples task planning from motor control. We fixed the IK module as a differentiable function rather than learning it because learning IK would require embodiment-specific data, contradicting our goal; the trade-off is that the policy must rely on the accuracy of the kinematic model, which we assume is known. We used domain randomization over kinematic parameters (sampled from a uniform distribution over link lengths [0.2, 0.5] m and joint limits [ -2.5, 2.5] rad) during training to force the policy to learn a robust mapping from visual-language input to end-effector poses that is invariant to physical variations; this adds training complexity but is necessary for zero-shot transfer. **Crucially, this design relies on the assumption that domain randomization over kinematic parameters forces the policy to learn embodiment-invariant representations, but this holds only if the test robot's kinematics fall within the randomization range.** To address this, during inference we estimate the test robot's kinematics from a small number of joint-level observations (50 gradient steps with learning rate 1e-3 using the differentiable IK to align predicted joint angles with observed ones) and center the randomization distribution accordingly. We selected a Levenberg-Marquardt IK solver over a closed-form analytical IK because it is differentiable and works for arbitrary kinematic chains; the cost is higher computational overhead (5 iterations per timestep), which is acceptable given modern hardware. We implement the environment in MuJoCo with parameterized robot models.

**Why it measures what we claim**: The predicted end-effector pose `eef_pose` measures the agent's desired spatial outcome independent of embodiment because it is defined in a canonical coordinate frame common across all robots; this assumption holds when kinematic parameters are known exactly, but fails when robots have unmodeled dynamics (e.g., compliance), in which case `eef_pose` reflects only the intended configuration without guaranteeing execution accuracy. The domain randomization over `k` measures the policy's robustness to kinematic variations by forcing the representation to be invariant under changes in `k`; this assumption holds when the distribution `K` covers the test robot's kinematics, but fails when test kinematics are outside the training distribution, in which case the policy may still output unreasonable poses. **However, domain randomization over kinematics alone does not guarantee full embodiment invariance, as other factors (e.g., dynamics, friction) also vary across embodiments; this is a limitation we test separately.** The differentiable IK module ensures that the gradient flows through the kinematic chain, making the pose prediction trainable via end-to-end backpropagation; this measures the coupling between pose and final joint angles, assuming the IK is invertible and differentiable, but fails when the Jacobian is singular (e.g., near joint limits), causing gradients to vanish or explode.

## Contribution

(1) We introduce Embodiment-Agnostic Policy (EAP), a vision-language-action model that predicts end-effector poses and uses a fixed differentiable IK module, enabling zero-shot cross-embodiment transfer without any target robot data. (2) We show that domain randomization over robot kinematic parameters forces the policy to learn embodiment-invariant representations, which is a necessary condition for zero-shot generalization to unseen robots. (3) We demonstrate that the differentiable IK module allows end-to-end training, and the model can be pre-trained on large-scale heterogeneous datasets without conditioning on robot identity.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Simulated multi-robot manipulation (Franka, UR5, Panda) in MuJoCo | Tests zero-shot transfer across different kinematics. |
| Primary metric | Task success rate (binary) | Direct measure of task completion; embodiment-agnostic. |
| Baseline 1 | Direct joint prediction (2-layer MLP, hidden=256, ReLU) | Tests necessity of canonical EEF space. |
| Baseline 2 | Canonical EEF, no domain randomization | Tests importance of kinematic randomization. |
| Baseline 3 | Learned IK network (4-layer MLP, hidden=512, ReLU) + domain randomization | Isolates benefit of differentiable IK. |
| Ablation of ours | EAP with learned IK (same architecture as Baseline 3) | Tests differentiable IK necessity for gradient flow. |

### Why this setup validates the claim

The central claim is that EAP achieves embodiment-agnostic policy by predicting canonical end-effector poses and using domain randomization over kinematics. To test this, we use a multi-robot benchmark with distinct kinematic chains. The direct joint prediction baseline cannot transfer because it learns embodiment-specific mappings. The canonical EEF without randomization overfits training kinematics and fails on new ones. The learned IK baseline tests whether the differentiability of the IK module is critical beyond simply having a learned mapping. Our ablation with learned IK (same as Baseline 3 but trained w/ gradient flow through IK) isolates the effect of gradient propagation. The task success rate directly captures whether the method solves tasks on unseen robots. If our method succeeds where baselines fail, it validates the claim that the design choices enable zero-shot transfer. We expect the performance gap to be largest on unseen robots, confirming that our method is truly embodiment-agnostic.

### Expected outcome and causal chain

**vs. Direct joint prediction** — On a case where the test robot has different joint limits and link lengths (e.g., Franka vs UR5), the baseline outputs joint angles that are out of range or cause collisions because it learned a mapping from visual input to joint positions specific to the training robot. Our method predicts end-effector pose, which is independent of joint geometry, then uses differentiable IK to convert, so it handles different kinematics robustly. We expect the baseline to have near 0% success on unseen robots while our method achieves >50% success, showing a large gap.

**vs. Canonical EEF, no domain randomization** — On a case where training robots have a forearm length of 0.3m but the test robot has 0.4m, the baseline learns output poses biased to the training kinematics (e.g., reaching targets with a specific arm orientation). Without domain randomization, it fails to adapt. Our method, trained with randomized kinematic parameters, learns to output poses invariant to physical variations. We expect the baseline to drop to <20% on out-of-distribution kinematics while ours remains >70%, demonstrating the critical role of domain randomization.

**vs. Learned IK network + domain randomization** — On a case where the robot enters a singular configuration (e.g., elbow locked), the learned IK baseline may produce incorrect joint angles because it maps pose to joints without analytic constraints; also gradients do not flow through IK, so pose prediction is not directly optimized for joint-space accuracy. Our differentiable IK (Levenberg-Marquardt) handles singularities with damping and maintains gradient flow, improving both training and inference. We expect the learned IK baseline to have success <30% on singular poses while our method achieves >60%, highlighting the importance of differentiable IK.

**vs. EAP with learned IK** — On a case where the robot enters a singular configuration, the ablation (same as Baseline 3 but trained with gradient flow through a learned IK) may still suffer from approximation errors; the differentiable analytical IK is exact. We expect similar performance gap as vs. Baseline 3.

### What would falsify this idea
If our method's success rate on unseen robots is not significantly higher than all baselines, or if the performance gap is uniform across seen and unseen robots rather than concentrated on novel kinematics, then the embodiment-agnostic claim is invalid. Additionally, if on robots with significantly different dynamics (e.g., torque limits, friction) our method fails while baselines succeed, it would indicate that embodiment invariance is not achieved.

## References

1. Qwen-RobotManip Technical Report: Alignment Unlocks Scale for Robotic Manipulation Foundation Models
2. Qwen3-VL Technical Report
3. Scalable Vision-Language-Action Model Pretraining for Robotic Manipulation with Real-Life Human Activity Videos
4. X-VLA: Soft-Prompted Transformer as Scalable Cross-Embodiment Vision-Language-Action Model
5. TimeMarker: A Versatile Video-LLM for Long and Short Video Understanding with Superior Temporal Localization Ability
6. MegaSaM: Accurate, Fast, and Robust Structure and Motion from Casual Dynamic Videos
7. PaliGemma 2: A Family of Versatile VLMs for Transfer
8. Predictive Inverse Dynamics Models are Scalable Learners for Robotic Manipulation
