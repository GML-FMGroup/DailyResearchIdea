# Conditional Neural Fields for Zero-Shot Robotic Manipulation

## Motivation

Current vision-language-action (VLA) models such as OpenVLA require fine-tuning on task-specific demonstrations, failing to generalize to new tasks or environments. As highlighted by LIBERO-PRO, these models collapse under perturbations due to memorization rather than understanding. The root cause is that existing architectures do not factorize environment geometry from task goals, forcing the model to implicitly memorize task-specific mappings from visual input to action sequences. A structured representation that separates these factors is needed to enable compositional generalization at test time.

## Key Insight

A conditional neural field (CNF) inherently factorizes environment geometry and task goals by conditioning action predictions on spatial coordinates and a learnable task code, allowing zero-shot composition of novel tasks via interpolation or combination of task codes.

## Method

### (A) What it is
**CNF-Zero** is a conditional neural field that takes as input a 3D spatial coordinate \( \mathbf{p} \in \mathbb{R}^3 \) and a task code \( \mathbf{z} \in \mathbb{R}^d \), and outputs an action \( \mathbf{a} \in \mathbb{R}^k \) (e.g., delta end-effector position). The field is defined by a neural network \( F_\theta(\mathbf{p}, \mathbf{z}) \). The environment geometry is encoded into a latent grid \( \mathbf{E} \) from a single RGB-D image via a pre-trained vision encoder (e.g., EVA). The task code \( \mathbf{z} \) is learned from a few demonstrations or inferred from a language instruction via a lightweight mapping network.

### (B) How it works
```python
# Training: optimize theta, phi (task code encoder) on a dataset of demonstrations
for each episode:
    # Encode environment from single view
    img, depth = get_observation()
    E = vision_encoder(img, depth)  # (H, W, C) latent feature grid
    
    # Sample a set of query points along the demonstration trajectory
    points = sample_3d_points_from_trajectory()  # (N, 3)
    actions = get_actions_at_points()  # (N, k)
    
    # Learn task code via a hypernetwork or attention
    # Option 1: learn a separate embedding for each task (task_id -> z)
    # Option 2: infer z from a single demonstration via a learned encoder H_phi
    # We use Option 2 for zero-shot: z = H_phi(visual_demo)
    z = task_encoder(demo_images)  # (d,)
    
    # For each point, retrieve spatial feature from E at projected 2D location
    # (using bilinear sampling) and concatenate with positional encoding of p
    features = bilinear_sample(E, project(points))  # (N, C)
    pos_enc = positional_encoding(points)  # (N, P)
    
    # Predict action via field
    a_pred = F_theta(torch.cat([features, pos_enc, z.expand(N, -1)], dim=-1))
    
    loss = L2_loss(a_pred, actions)
    update theta, phi

# Inference on a new task:
# 1. Get environment image, encode to E.
# 2. Infer task code z from a single language instruction using a frozen LLM encoder (e.g., MiniGPT-v2 style task ID) or from one demonstration.
# 3. Define waypoints as 3D points (e.g., via a simple planner or direct querying).
# 4. Query F_theta at each waypoint to produce actions.
```

### (C) Why this design
We chose a continuous 3D query over discrete voxel grids to enable arbitrary resolution without memory explosion, at the cost of needing a coordinate encoding that may lose fine-grained geometry. We use a bilinear sampling from a 2D latent grid (image-based) rather than a full 3D field to leverage pre-trained vision encoders—trading exact 3D reasoning for computational efficiency. The task code \( \mathbf{z} \) is learned via a hypernetwork from demonstration images rather than a fixed lookup, allowing zero-shot inference by encoding novel task visuals, though it assumes the hypernetwork generalizes well. Finally, we adopt L2 regression over diffusion to keep training simple and fast, accepting that multi-modal action distributions may be less accurately captured; for tasks requiring high precision, a diffusion decoder (like in TinyVLA) could be integrated but would increase sampling cost. This design prioritizes compositionality and zero-shot transfer over state-of-the-art performance on seen tasks, aligning with our goal of eliminating fine-tuning.

### (D) Why it measures what we claim
The computational quantity \( \mathbf{z} \) (task code) measures the *task goals* because it is trained to encode the visual demonstration of a task through the hypernetwork; this relies on the assumption that the hypernetwork maps task-specific visual patterns to a low-dimensional embedding that captures the task's spatial constraints. This assumption fails when the novel task's visual appearance is outside the distribution of training demonstrations, in which case \( \mathbf{z} \) may reflect irrelevant visual features instead of task intent. The quantity \( \mathbf{E} \) (environment latent from vision encoder) measures the *environment geometry* because it is extracted from a single view via a robust encoder (EVA), assuming the view captures sufficient spatial structure for manipulation; this fails when occlusions or depth noise corrupt the latent, causing \( \mathbf{E} \) to encode missing or incorrect geometry. The query operation \( F_\theta(\mathbf{p}, \mathbf{z}) \) measures the *action at task-specified locations* because the field is conditioned on both \( \mathbf{p} \) (spatial coordinate carrying position information) and \( \mathbf{z} \) (task code carrying constraint information); this equivalence holds when the field's learned mapping is sufficiently smooth and the query points are well-chosen, but fails if waypoints are inadequately sampled, leading to actions that ignore task constraints.

### (E) Training details
- Vision encoder: EVA-g (ViT-g/14) fine-tuned on robotic data.
- Task encoder: 4-layer MLP with 256 hidden units, input = average pooled features from 5 demo frames.
- Field network: 8-layer MLP with 512 units, SiLU activations, layer norm. Input dimension: C (features) + 64 (pos. enc) + d (task code).
- Learning rate 1e-4, AdamW, batch 64.
- Action representation: continuous delta position (3D) with L1 loss.

## Contribution

(1) CNF-Zero, a conditional neural field framework that factorizes environment geometry and task goals for VLA models, enabling zero-shot generalization to novel tasks without fine-tuning. (2) Empirical demonstration that CNF-Zero achieves non-zero success rates on unseen tasks in LIBERO-PRO perturbations, whereas existing models collapse to 0%. (3) Analysis of task code interpolation showing that task constraints (e.g., grasping position) can be composed by linear combinations of learned codes, providing a mechanism for compositional generalization.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|-------|---------|------------------------|
| Dataset | LIBERO benchmark | Standard for VLA evaluation with diverse tasks. |
| Primary metric | Success rate | Direct measure of task completion. |
| Baseline 1 | OpenVLA (fine-tuned) | State-of-the-art VLA requiring full fine-tuning. |
| Baseline 2 | TinyVLA (fine-tuned) | Efficient VLA with data-efficient fine-tuning. |
| Ablation-of-ours | CNF-Zero (no task code) | Removes task code to test its necessity. |

### Why this setup validates the claim

The central claim is that CNF-Zero achieves zero-shot transfer to new tasks without fine-tuning by leveraging a learned task code from a single demonstration or language instruction. To validate this, we evaluate on the LIBERO benchmark, which includes multiple distinct manipulation tasks with varying goals and object configurations. The primary metric, success rate, directly quantifies task completion. Comparing against OpenVLA and TinyVLA, which require task-specific fine-tuning, provides upper bounds on performance when much more data is used. The ablation (CNF-Zero without task code) isolates the contribution of the task code; if the full method outperforms the ablation, it confirms that the task code encodes task-specific goals. This design forms a falsifiable test: if our method does not approach the fine-tuned baselines on novel tasks, or if the ablation performs similarly, the claim that the task code enables zero-shot transfer is invalid.

### Expected outcome and causal chain

**vs. OpenVLA (fine-tuned)** — On a novel LIBERO task like "put the bowl on the plate," OpenVLA without fine-tuning would produce random actions because its large vision-language model was pre-trained on different instructions and does not generalize to new goal-conditioned manipulation without adaptation. Our method instead infers a task code from a single demonstration or language instruction, conditioning the neural field on both the environment geometry and the task intent, so it can produce appropriate actions without any fine-tuning. We expect our success rate to be roughly 60–70% on tasks with similar visual patterns to training, but drop to ~30% on highly out-of-distribution tasks, while fine-tuned OpenVLA achieves >90% after full fine-tuning.

**vs. TinyVLA (fine-tuned)** — On a task like "open the drawer," TinyVLA fine-tunes quickly with 5 demonstrations to adapt its policy, achieving high success. Our method operates zero-shot: it encodes a single demonstration into a task code and queries the field at waypoints. Since TinyVLA uses a diffusion decoder that captures multi-modal action distributions, it may more accurately handle ambiguous actions (e.g., different grasp orientations), while our L2 regression may produce averaged actions. However, our method incurs no fine-tuning cost. We expect our success rate to be about 10–20% lower than TinyVLA on tasks where fine-tuning data is available, but comparable on tasks where the goal is unambiguous (e.g., pushing a button).

**vs. CNF-Zero (no task code)** — On a task requiring specific goal direction (e.g., "move the mug to the right"), the ablation without task code ignores the goal and produces an action that only depends on the environment features and the waypoint position, effectively performing a generic reaching behavior. This leads to success only if the correct action coincides with a learned bias (e.g., if all training tasks involved moving objects to the same location). Our full method uses the task code to alter the field’s output, correctly directing the mug rightward. We expect a large gap (e.g., 70% vs 20%) on goal-sensitive tasks, and parity on tasks where geometry alone determines the action (e.g., pushing a button at a fixed location).

### What would falsify this idea

If the success rate of our full method is not significantly higher than the ablation (CNF-Zero without task code) across all task types, or if the gap is no larger on goal-sensitive tasks than on geometry-dominated tasks, then the task code is not meaningfully capturing task intent, and the claim of zero-shot transfer via learned task codes is false.

## References

1. Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success
2. TinyVLA: Toward Fast, Data-Efficient Vision-Language-Action Models for Robotic Manipulation
3. MiniGPT-v2: large language model as a unified interface for vision-language multi-task learning
4. EVA: Exploring the Limits of Masked Visual Representation Learning at Scale
5. LIBERO-PRO: Towards Robust and Fair Evaluation of Vision-Language-Action Models Beyond Memorization
6. CogACT: A Foundational Vision-Language-Action Model for Synergizing Cognition and Action in Robotic Manipulation
7. Benchmarking Vision, Language, & Action Models on Robotic Learning Tasks
8. Vision-Language Foundation Models as Effective Robot Imitators
