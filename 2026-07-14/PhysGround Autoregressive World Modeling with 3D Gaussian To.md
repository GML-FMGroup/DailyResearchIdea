# PhysGround: Autoregressive World Modeling with 3D Gaussian Tokens and Differentiable Physics

## Motivation

Current world models, such as Xiaomi-Robotics-U0, rely on 2D image/video token prediction, inheriting the assumption that spatial-temporal dynamics can be captured implicitly. This structural limitation leads to hallucinations and implausible dynamics because 3D spatial consistency and physical laws are never explicitly enforced, causing the model to fail in long-horizon embodied reasoning.

## Key Insight

Representing the world state as a set of 3D Gaussians provides an explicit, differentiable geometric scaffold, and integrating a differentiable physics engine into the autoregressive prediction loop ensures that every predicted future state respects Newtonian dynamics by construction, eliminating hallucinated trajectories.

## Method

(A) **What it is** — PhysGround is an autoregressive world model that predicts future world states as sets of 3D Gaussians. Its inputs are a sequence of past Gaussian sets (each set encoding scene geometry and appearance) and optional actions; its outputs are predicted future Gaussian sets that are physically plausible. (B) **How it works** — 
```python
# Input: history of Gaussian sets G_1...G_T (each G_t has N Gaussians)
# Step 1: Encode each set into token embeddings via a shared MLP (256-dim per Gaussian)
# Step 2: Append action tokens if available (also 256-dim)
history_tokens = [encode_gaussians(G_t) for G_t in G_1..G_T]
# Step 3: Transformer autoregressive prediction (12 layers, 16 heads, 512 hidden)
for step in 1..K:
    # Transformer takes sequence of tokens, outputs next token embedding
    next_token = transformer(history_tokens)
    # Decode token to N Gaussians: mean (3), covariance (6), opacity (1), color (3)
    G_pred = decode_gaussians(next_token)  # MLP with tanh for means, softplus for covariance
    # Differentiable physics simulation (IsaacGym, dt=0.01s, gravity=-9.81, restitution=0.5)
    # Physics receives current state (G_prev) and predicted G_pred as action, outputs next physical state
    G_phys = physics_simulate(G_prev, G_pred)
    # Final predicted state: blend with learned alpha (α=0.7) to allow deviations
    G_next = α * G_phys + (1-α) * G_pred
    # Append G_next as new token and continue
    history_tokens.append(encode_gaussians(G_next))
return G_T+1 ... G_T+K
```
(C) **Why this design** — We chose 3D Gaussian tokens over voxel grids or meshes because they are continuously differentiable and compact, enabling efficient gradient flow while preserving explicit 3D structure. We integrated a differentiable physics engine (IsaacGym) directly into the autoregressive loop rather than using a post-hoc physicality filter, because this allows the transformer to be trained with gradients from the physics loss, end-to-end learning to predict physically consistent states. We contrasted this with a pure data-driven approach that lacks physics, accepting the trade-off that the physics engine may not cover all real-world effects (e.g., plastic deformation), but for rigid and articulated objects it provides strong inductive bias. Finally, we use a learned blending factor (α=0.7) to balance physics fidelity and data-driven flexibility, sacrificing some consistency in favor of handling dynamics not captured by the simulator.
(D) **Why it measures what we claim** — The quantity `G_phys` (physics simulation output) measures **physical plausibility** because it enforces Newtonian dynamics under the assumption that the environment is governed by classical mechanics; this assumption fails when objects undergo non-physical interactions (e.g., teleportation or spontaneous generation), in which case `G_phys` deviates from ground truth and the physics loss penalizes the transformer. The token encoding `encode_gaussians` measures **spatial consistency** because it captures continuous 3D positions and covariances; this assumption fails when a single Gaussian represents multiple disconnected regions, leading to blurry representations. The blending operation `α * G_phys + (1-α) * G_pred` measures **model uncertainty** via α, because a low α indicates low confidence in the physics model; this assumption fails when both physics and prediction are wrong but cancel out, yielding an incorrectly confident blend. The autoregressive loop over Gaussian sets measures **temporal coherence** because each step uses the previous physical state as input; this assumption fails when the dynamics are non-Markovian (e.g., memory effects), in which case the model may ignore long-range dependencies.

## Contribution

(1) A novel autoregressive world model architecture that represents the world state as explicit 3D Gaussian tokens and integrates a differentiable physics engine into the prediction loop. (2) The design principle that physics-grounded token prediction enforces spatial consistency and physical plausibility, reducing hallucinations in long-horizon embodied scenarios. (3) A training paradigm combining reconstruction loss and physics consistency loss, enabling the model to learn from both data and physical laws.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|---|---|---|
| Dataset | Isaac Gym tabletop manipulation | Simulated rigid object dynamics. |
| Primary metric | Physical Plausibility Accuracy (PPA) | Measures adherence to Newtonian physics. |
| Baseline 1 | LearnedOnly (no physics) | Tests necessity of physics integration. |
| Baseline 2 | PostHocPhys (post-hoc correction) | Tests advantage of in-loop physics. |
| Baseline 3 | VoxelGridWorldModel | Tests representation choice. |
| Ablation of ours | Ours w/o blending (α=1) | Tests learned blending effect. |

### Why this setup validates the claim

The combination of dataset, baselines, metric, and ablation forms a falsifiable test of the claim that integrating differentiable physics into the autoregressive loop yields physically plausible predictions. The Isaac Gym dataset provides ground truth physics, enabling direct measurement of PPA. LearnedOnly tests whether any physics integration is beneficial; PostHocPhys tests whether in-loop physics outperforms post-hoc correction; VoxelGridWorldModel tests whether the Gaussian representation is superior. The ablation w/o blending isolates the value of the learned blend. PPA is the right metric because it directly measures physical plausibility. If our method outperforms baselines on physics-heavy subsets (e.g., long-horizon, contact-rich tasks), the claim is supported; if gains are uniform, the physics integration may be unnecessary.

### Expected outcome and causal chain

**vs. LearnedOnly** — On a case where a block is pushed and friction should stop it, LearnedOnly predicts indefinite sliding because it lacks explicit friction modeling. Our method simulates physics, so the block stops realistically. We expect a notable gap on long-horizon tasks (e.g., 30+ steps) where LearnedOnly's error accumulates (PPA < 50%) while ours stays high (>90%).

**vs. PostHocPhys** — On a case where an object is knocked and should tumble, PostHocPhys first predicts a rough trajectory then snaps to physics, losing momentum. Our in-loop physics propagates momentum through time. We expect a gap on tasks requiring temporal consistency (e.g., stacking) where PostHocPhys yields >20% interpenetration, ours <5%.

**vs. VoxelGridWorldModel** — On a peg insertion task requiring sub-millimeter precision, VoxelGrid discretizes positions, causing misalignment. Our Gaussian representation is continuous. We expect VoxelGrid to fail on precision tasks (PPA < 30%) while ours succeeds (>95%).

### What would falsify this idea

If our method shows uniform PPA improvement across all subsets rather than concentrated gains on physics-heavy subsets (e.g., long-horizon or contact-rich tasks), the central claim that physics integration is the cause would be falsified.

## References

1. Xiaomi-Robotics-U0: Unified Embodied Synthesis with World Foundation Model
2. Galaxea Open-World Dataset and G0 Dual-System VLA Model
3. Qwen3-VL Technical Report
4. Emu3.5: Native Multimodal Models are World Learners
5. TimeMarker: A Versatile Video-LLM for Long and Short Video Understanding with Superior Temporal Localization Ability
6. TimeChat: A Time-sensitive Multimodal Large Language Model for Long Video Understanding
7. VTimeLLM: Empower LLM to Grasp Video Moments
8. LLaMA-VID: An Image is Worth 2 Tokens in Large Language Models
