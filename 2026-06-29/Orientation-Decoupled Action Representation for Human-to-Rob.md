# Orientation-Decoupled Action Representation for Human-to-Robot Skill Transfer

## Motivation

Existing methods for human-to-robot skill transfer either discard rotation entirely (e.g., 'Translation as a Bridging Action') or handle it via distribution alignment or weighted loss without learning a rotation-invariant latent representation. This structural failure arises because rotation and translation are treated as coupled signals in the action space, whereas in many manipulation tasks end-effector orientation is conditionally determined by translation given task context (e.g., object pose, grasp type). No prior work exploits this conditional structure to decouple rotation from translation.

## Key Insight

The orientation of the end-effector in many manipulation tasks is conditionally determined by the translation trajectory and task context, enabling a learned mapping that factors rotation out of the action space while preserving task-relevant orientation.

## Method

### Orientation-Decoupled Action Representation (ODAR)

**Assumption:** In manipulation tasks, end-effector orientation is conditionally determined by translation and task context. This assumption is central and is explicitly tested in experiments via tasks where it holds and where it fails (e.g., random wrist waving).

**(A) What it is:** We propose Orientation-Decoupled Action Representation (ODAR), a framework that factorizes human action into translation (relative wrist displacement) and a canonical orientation component predicted from translation and task context. ODAR takes as input a sequence of human demonstrations (3D hand poses) with task context (object pose, grasp type) and outputs a factorized action representation where the translation is the cross-embodiment bridging signal and the orientation is a learned function of translation and context.

**(B) How it works:**
```
Training Phase:
1. For each human demonstration, extract wrist translation t_i (3D) and rotation R_i (4D quaternion) at each timestep.
2. Encode task context c (e.g., object pose (7D), grasp type (one-hot of 5 classes)) using a 2-layer MLP (hidden=256, ReLU) to produce a 32D embedding.
3. Train a conditional VAE (cVAE) that takes translation t_i and context embedding c as input and outputs a canonical orientation R_hat_i.
   - Encoder: 2-layer MLP (hidden=256, ReLU) that encodes concatenated (t_i, c) (3+32=35D) into latent z (mean and logvar, d_z=8).
   - Decoder: 2-layer MLP (hidden=256, ReLU) that outputs 4D quaternion (normalized) from (z, t_i, c) (8+3+32=43D).
   - Loss: L = MSE(R_hat_i, R_i) + β * KL(N(μ,σ) || N(0,1)), with β=0.001.
   - Training: Adam optimizer, lr=1e-4, batch_size=256, 100 epochs.
4. The learned canonical orientation R_hat_i is the orientation that is task-relevant given translation and context.

Inference Phase (robot):
1. Given robot proprioception (current translation) and task context, compute canonical orientation using the trained cVAE decoder (feeding z sampled from N(0,1) at test time).
2. Form action representation as (translation, canonical orientation).
3. Execute with a low-level impedance controller (stiffness=1000 N/m, damping=50 Ns/m).
```

**(C) Why this design:** We chose a conditional VAE over a deterministic regressor because orientation can be multi-modal (e.g., multiple grasp orientations leading to same translation). The VAE captures this multimodality via the latent distribution. We chose translation as the bridging action (instead of full pose) because it is robust across embodiments (as shown in 'Translation as a Bridging Action'), but we augment it with orientation prediction to handle tasks requiring precise orientation. We include task context (object pose, grasp type) as input to disambiguate orientation when translation alone is insufficient. The trade-off is that the VAE requires more data and computational cost than a simple regressor, but it provides a principled way to model uncertainty. We avoid using a deterministic alignment because it would force a one-to-one mapping that may not exist across different embodiments.

**(D) Why it measures what we claim:** The computational quantity canonical orientation R_hat_i measures the task-relevant orientation because it is conditioned on translation and task context; the assumption is that for the target manipulation tasks, orientation is a deterministic (or stochastic) function of translation and context. This assumption fails when orientation is independent of translation (e.g., random wrist waving), in which case the VAE learns to ignore orientation and predicts a mean orientation, but the representation still remains invariant to irrelevant rotations. The translation signal measures the cross-embodiment bridging action because relative wrist translation is consistent across human and robot kinematics; this assumption fails for tasks relying on absolute position (e.g., pushing a button at a fixed location relative to the body), in which case the representation may need a reference frame alignment. 

**Resource estimates:** Training the cVAE on 10k demos (each 100 timesteps) takes ~2 GPU hours on an NVIDIA A100. Model size: ~1.2M parameters.

## Contribution

(1) A novel factorized action representation that decouples translation from rotation by learning a conditional canonical orientation from translation and task context. (2) A practical framework (ODAR) that leverages human demonstrations to learn this factorization without requiring paired robot data. (3) Demonstration that the learned representation enables zero-shot transfer of orientation skills to robots on manipulation tasks with varying orientation demands.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|-----------------------|
| Dataset | RH20T human demonstrations (10k demos, 20 tasks) | Diverse tasks with ground-truth context labels |
| Primary metric | Task success rate (50 rollouts per task) | Direct measure of transfer performance |
| Baseline 1 | Full 6DoF human pose (raw 7D action) | Tests translation-only vs. full orientation |
| Baseline 2 | BC from robot data (Diffusion Policy, 200M params) | Tests benefit of human data for transfer |
| Ablation-of-ours | Deterministic orientation regressor (2-layer MLP, L2 loss) | Tests need for VAE's multimodality |

### Why this setup validates the claim

This setup validates the central claim that ODAR enables robot skill transfer from humans by directly comparing against key alternatives. The dataset RH20T provides 10k realistic human demonstrations with task context across 20 manipulation tasks, allowing training of both our factorized representation and baselines. Baseline 1 (full 6DoF pose) tests whether translation alone suffices as a bridging signal, while Baseline 2 (BC from robot data) tests if human data adds value. The ablation of a deterministic regressor isolates the benefit of the conditional VAE for handling multimodal orientation. Task success rate over 50 rollups per task is a direct, unambiguous metric that reflects whether the robot can execute the skill after transfer. Together, these components form a falsifiable test: if our method outperforms both baselines and the ablation, it supports our reasoning; otherwise, specific sub-claims are refuted.

### Expected outcome and causal chain

**vs. Full 6DoF pose** — On a task where human orientation varies across demos due to embodiment differences (e.g., same translation but different wrist orientation), the baseline uses full noisy 6DoF as action, causing conflicting commands and execution failure because it treats orientation as a direct action. Our method instead uses translation as the bridging action and predicts canonical orientation from translation and context, reducing noise, so we expect higher success rate especially on tasks with orientation variability (e.g., 20-30% absolute gap on orientation-sensitive tasks like cup grasping). We also test on a random wrist-waving task (orientation independent of translation) to verify that ODAR degrades gracefully: both methods fail, but ODAR's representation remains invariant, achieving similar performance as ignoring orientation.

**vs. BC from robot data** — On a task requiring dexterous manipulation (e.g., peg insertion), the baseline lacks human demonstrations of the skill, so it must rely on robot data that may be scarce or different. Our method leverages human demos to learn the canonical orientation mapping, enabling better generalization. We expect a 15-20% higher success rate on tasks that benefit from human data, such as grasping diverse objects.

**vs. Deterministic orientation regressor** — On a task where multiple orientations can achieve the same translation (e.g., grasping a cup from different angles), the deterministic regressor outputs a single average orientation, which may be suboptimal for the specific instance. Our VAE captures multimodality, sampling appropriate orientation, leading to higher success on multi-modal tasks (expect 10-15% gap). On unimodal tasks (e.g., inserting a key into a slot), both perform similarly.

### What would falsify this idea

If ODAR fails to outperform the full 6DoF baseline on tasks known to require precise orientation (like assembly), meaning the translation-only bridging signal is insufficient, then the central claim that translation is a robust bridging signal is wrong. Similarly, if ODAR performs worse than the deterministic regressor on multimodal tasks, then the VAE's multimodality handling is unnecessary and our motivation is undermined.

## References

1. Translation as a Bridging Action: Transferring Manipulation Skills from Humans to Robots
2. Emergence of Human to Robot Transfer in Vision-Language-Action Models
3. XR-1: Towards Versatile Vision-Language-Action Models via Learning Unified Vision-Motion Representations
4. From Multimodal LLMs to Generalist Embodied Agents: Methods and Lessons
5. Latent Action Pretraining from Videos
6. TinyVLA: Toward Fast, Data-Efficient Vision-Language-Action Models for Robotic Manipulation
7. OpenVLA: An Open-Source Vision-Language-Action Model
8. EgoMimic: Scaling Imitation Learning via Egocentric Video
