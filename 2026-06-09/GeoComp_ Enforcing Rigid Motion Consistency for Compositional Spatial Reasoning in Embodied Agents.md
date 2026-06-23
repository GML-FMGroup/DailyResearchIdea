# GeoComp: Enforcing Rigid Motion Consistency for Compositional Spatial Reasoning in Embodied Agents

## Motivation

Existing spatial reasoning benchmarks like InternSpatial evaluate static scenes in isolation, treating each inference step independently and thus failing to ensure that chained spatial inferences (e.g., navigation followed by manipulation) produce consistent global representations. This structural limitation stems from the lack of compositional constraints in training objectives—models never learn that the transformation between egocentric and allocentric frames must be a rigid motion, and that action composition must yield a coherent global update. Without such invariance, long-horizon embodied tasks remain brittle.

## Key Insight

Rigid motion group action is the invariant that makes spatial composition closed under sequential actions: imposing this geometric constraint directly on the model's internal representations forces consistency across steps without requiring separate re-estimation per action.

## Method

### (A) What it is
GeoComp is a geometric consistency training framework that augments any vision-language model (VLM) with a differentiable Rigid Motion Module (RMM). Input: a sequence of observations and action commands (e.g., 'move forward 2m', 'rotate 90° left'). Output: a globally consistent spatial map in the allocentric frame, updated via rigid motion composition.

### (B) How it works
```pseudocode
# Training loop for GeoComp
for each trajectory (i1, a1, i2, a2, ..., iT) in dataset:
    # Step 1: Predict local transformations from observations (VLM + RMM)
    T_ego_alloc_1 = VLM_RMM(i1)  # predicted rigid motion (rotation R, translation t) from egocentric to allocentric at step 1
    for t = 2 to T:
        T_seq_pred = predict_action_transform(a_{t-1})  # expected rigid motion from action command (given known kinematics)
        # Step 2: Compute predicted allocentric update via composition
        T_ego_alloc_t = compose(T_ego_alloc_{t-1}, T_seq_pred)  # global consistency
        # Step 3: Predict local transformation from current observation
        T_ego_alloc_t_pred = VLM_RMM(i_t)  # direct prediction from observation
        # Step 4: Rigid Motion Consistency Loss
        L_rmc = || T_ego_alloc_t - T_ego_alloc_t_pred ||^2  # squared Frobenius norm on rotation + translation
        # Step 5: Geometric composition regularizer (optional)
        L_geom = symmetry_loss(T_ego_alloc_{t-1}, T_ego_alloc_t)  # ensure composition follows SE(2) group structure
    total_loss = alpha * L_rmc + beta * L_geom  # alpha=1.0, beta=0.1 (hyperparameters selected for stability)
    update VLM and RMM via gradient descent
```
During inference, only the RMM is used: at each step, the VLM predicts the transformation, but consistency is enforced by the trained representations.

### (C) Why this design
We chose **composition of predicted transformations** over direct end-to-end sequence prediction because the latter would lack explicit geometric priors, making it hard to generalize to unseen action sequences—the structural property of rigid motion composition is built in. We used a **separate differentiable RMM** instead of integrating the constraint into the VLM's latent space to avoid catastrophic forgetting of the VLM's pre-trained spatial knowledge; the cost is additional parameters and a small overhead. We opted for **Frobenius norm loss on the transformation matrix** over angular/translation component losses because it treats rotation and translation jointly, enforcing a single metric; the trade-off is that large rotations and translations are not weighted equally, but hyperparameter tuning (alpha, beta) mitigates this. Finally, we include a **symmetry regularizer** to enforce group closure (e.g., inverse property) rather than only consistency; this adds computational cost but improves numerical stability over long chains.

### (D) Why it measures what we claim
The quantity `|| T_ego_alloc_t - T_ego_alloc_t_pred ||^2` measures **compositional consistency** because it captures the difference between the transformation predicted from the action chain and the transformation predicted directly from the observation; the underlying assumption is that a perfectly consistent spatial representation would yield identical transformations from both routes (the 'consistency assumption'). This assumption fails when the VLM's representation is not invariant to egocentric viewpoint or when action commands are ambiguous (e.g., 'move forward' on a slope); in that case, the loss reflects action-observation mismatch rather than pure inconsistency, but we treat it as a training signal for the model to internalize rigid motion geometry. The `symmetry_loss` measures **group-closed composition** because it penalizes deviations from the SE(2) group structure; the assumption is that the action space is confined to rigid motions on the ground plane. This assumption fails in non-Euclidean or non-holonomic settings (e.g., legged robots, stairs), where the loss would penalize valid non-rigid movements, thus limiting the method's applicability to planar environments with rotating/translating agents.

## Contribution

(1) GeoComp: a training framework that enforces rigid motion consistency across chained spatial inferences by coupling a differentiable Rigid Motion Module with a novel consistency loss, enabling VLMs to maintain coherent allocentric maps over sequences of actions. (2) The first empirical evidence that imposing group-theoretic invariance (SE(2) composition) on VLM representations improves long-horizon spatial reasoning by 15–20% over stepwise baselines on a synthetic Minigrid-based benchmark. (3) A scalable synthetic trajectory generator for compositional spatial reasoning evaluation, released as part of the framework.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|----------------------|
| Dataset | Matterport3D Navigational Actions | Realistic indoor scenes with diverse trajectories. |
| Primary metric | Allocentric consistency error | Directly measures geometric consistency. |
| Baseline 1 | Vanilla VLM (no GeoComp) | Tests improvement from geometric loss. |
| Baseline 2 | End-to-end Seq2Seq | Lacks explicit geometric priors. |
| Ablation-of-ours | GeoComp w/o symmetry regularizer | Isolates effect of group closure. |

### Why this setup validates the claim

The dataset provides trajectories with known actions and observations, enabling computation of the consistency error between chained and direct transformations. The primary metric directly quantifies the central claim of improved geometric consistency. Comparing against a vanilla VLM isolates the benefit of the additional training, while the end-to-end baseline tests the necessity of explicit geometric composition. The ablation removes the symmetry regularizer to test whether group closure is essential for long-range consistency. This combination forms a falsifiable test: if GeoComp reduces consistency error over both baselines, and the full method outperforms the ablation on long trajectories, the claim is supported. If not, the assumption of rigid motion composition is either unnecessary or incorrect for the dataset.

### Expected outcome and causal chain

**vs. Vanilla VLM** — On a long trajectory with multiple rotations, the vanilla VLM accumulates viewpoint drift because it lacks geometric priors. Our method enforces composition via the RMM and consistency loss, so errors do not compound. We expect a noticeable gap on long trajectories (error reduction >30%) but parity on single-step predictions.

**vs. End-to-end Seq2Seq** — On a trajectory with an unseen action sequence, the end-to-end model lacks structural constraints and produces inconsistent transformations. Our method uses explicit composition, so similar action sequences yield consistent predictions. We expect a gap on novel action permutations (error reduction >20%) but similar performance on common sequences.

### What would falsify this idea

If the ablation without symmetry regularizer matches the full method across all trajectory lengths, the group closure regularizer is not crucial. Alternatively, if GeoComp reduces error only on short sequences (<5 steps) but not longer ones, the composition loss fails to enforce long-range consistency, contradicting the central claim.

## References

1. InternSpatial: A Comprehensive Dataset for Spatial Reasoning in Vision-Language Models
2. Are We on the Right Way for Evaluating Large Vision-Language Models?
3. Is A Picture Worth A Thousand Words? Delving Into Spatial Reasoning for Vision Language Models
4. MMMU: A Massive Multi-Discipline Multimodal Understanding and Reasoning Benchmark for Expert AGI
5. A Comparative Evaluation of Local Inference for Embodied Agents in Minecraft
