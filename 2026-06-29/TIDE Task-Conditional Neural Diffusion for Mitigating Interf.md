# TIDE: Task-Conditional Neural Diffusion for Mitigating Interference in Unified Embodied Reasoning Models

## Motivation

Vesta (Zhang et al., 2024) unifies all embodied tasks under a single architecture but suffers from negative interference: shared representations must satisfy conflicting objectives across tasks, degrading performance on each. Existing generalist models lack a mechanism to dynamically specialize without breaking the unified forward pass. The root cause is the absence of task-specific attractors in representation space, forcing a single fixed point that underperforms for all tasks. This is the convergent_action bottleneck: the field's trajectory toward unified models has created a need for specialization that does not compromise the shared backbone.

## Key Insight

By learning a conditional score function that directs the shared representation toward task-specific high-probability regions (attractors) via few-step Langevin dynamics, the model can specialize while preserving a single forward path, because the diffusion process is invertible and leaves base parameters untouched.

## Method

### (A) What it is
TIDE (Task-Implicit Diffusion Extender) is a neural diffusion module that takes the frozen Vesta backbone's intermediate representation h (a vector of dimensionality D=2048) and a task embedding t (a learned 128-D vector per task, or inferred from instruction) and outputs a task-adapted representation h' through a conditional reverse diffusion process. The module is a lightweight score network s_θ (2-layer MLP with hidden size 512, GeLU activation) trained via denoising score matching.

### (B) How it works
```pseudocode
# Training Phase
# For each task T, collect a large set of backbone representations h^+ from task-specific forward passes (1,000 examples per task, augmented via small translations/rotations if applicable).
# Define noise levels: σ_1, ..., σ_N (geometric progression from 0.01 to 0.2, N=10).
# Validation set: hold out 200 examples per task for early stopping.

for each task T and each representation h^+:
    for step i in 1..N:
        σ = σ_i
        ε ~ N(0, I)
        h_noisy = h^+ + σ * ε
        target = -ε / σ   # score of perturbed sample
        loss = MSE(s_θ(h_noisy, t_T), target)
    s_θ ← optimize loss until validation score matching loss plateaus

# Inference Phase
# Given a task t and backbone representation h (frozen), refine via reverse diffusion:

h_cur = h + σ_1 * z   # initial noise, z ~ N(0, I)
for step i in 1..N:
    σ = σ_i
    z ~ N(0, I)
    h_cur = h_cur + (σ_i^2 - σ_{i+1}^2) * s_θ(h_cur, t) + (σ_i - σ_{i+1}) * z   # approximate reverse step (DDPM-style)
h' = h_cur
```
Load‐bearing assumption: Denoising score matching with N=10, geometric σ schedule, and 1,000 examples per task yields an accurate score function of the task-specific attractor distribution, and the reverse diffusion converges to that attractor. We verify this by computing attractor reconstruction error (Euclidean distance from h' to the centroid of 100 held-out task-specific examples); if average error >0.1, we double the training set.

### (C) Why this design
We made three key design decisions with explicit trade-offs. First, we chose score-based diffusion over explicit task-specific modules (e.g., per-task adapter layers) because it preserves the unified forward pass—every task shapes the same backbone representation without routing—and allows continuous interpolation between tasks via learned embeddings. The cost is added inference time (10 iterative steps) versus a single forward pass, and increased memory due to the score network. Second, we condition on learned task embeddings rather than one-hot vectors to enable generalization to unseen tasks whose embeddings can be inferred via similarity to known tasks (e.g., from instruction encoding). The trade-off is that joint training of embeddings and score network introduces non-convexity and requires careful initialization. Third, we used denoising score matching (DSM) with a fixed noise schedule over contrastive divergence (CD) because DSM provides stable gradients and empirically faster convergence for score estimation; CD often requires longer Markov chains and careful tempering. The cost is that DSM assumes the perturbed data distribution is a Gaussian mixture, which may oversmooth sharp attractor boundaries. Notably, unlike retrieval-based methods (e.g., k-NN on exemplars), TIDE does not store exemplars—the score network generalizes across the attractor region, so it does not collapse to nearest-neighbor retrieval.

### (D) Why it measures what we claim
The computational quantity s_θ(h, t) measures the gradient of the log-probability of the task-specific attractor distribution (i.e., ∇_h log p(h|t)). This relies on the assumption that denoising score matching recovers the true score of the data distribution when the noise schedule covers all frequency components and the network capacity is sufficient; this assumption fails when the attractor distribution has disconnected high-density regions (e.g., two separate modes for the same task), in which case s_θ averages across modes, producing a blurred gradient. The iterative update h' = h + (σ² - σ'²) s_θ(h, t) + (σ - σ')ε measures the process of pulling the representation toward the task-specific attractor because it is a discretized simulation of the reverse-time SDE that generates samples from p(h|t); this assumes the learned score approximates the true gradient well. If the approximation is poor (e.g., due to insufficient training data), the update may fail to converge to the correct attractor, yielding a representation that is not task-specialized. The contrast between initial and refined representation quantifies the degree of task-specific adjustment, and the Euclidean distance to the attractor centroid (if known) would measure residual specialization error.

## Contribution

(1) We introduce TIDE, a task-conditional neural diffusion module that adapts a unified backbone representation to task-specific attractors without modifying the base model, enabling specialization without explicit routing. (2) We demonstrate that denoising score matching can effectively learn attractor subspaces for distinct embodied tasks from a limited number of examples per task (e.g., 100 backbone representations), providing a practical path to add specialization to generalist models. (3) We provide a design principle: preserving the unified forward pass while enabling dynamic specialization is achievable through a reversible diffusion-based refinement head, avoiding the need for task-specific modules or controllers.

## Experiment

### Evaluation Setup
| Role | Choice | Rationale (≤12 words) |
|------|--------|-----------------------|
| Dataset | Vesta Evaluation Suite (5 tasks) | Diverse embodied reasoning tasks |
| Primary metric | Task-Averaged Success Rate | Direct measure of task performance |
| Baseline 1 | Frozen Vesta backbone | Tests necessity of task adaptation |
| Baseline 2 | Per-task linear adapters | Compares with module-based adaptation |
| Baseline 3 | Task-specific hypernetworks (Ha et al., 2017) | Compares with another dynamic specialization method |
| Ablation of ours | TIDE with one-hot task embeddings | Tests importance of learned embeddings |

### Why this setup validates the claim
This combination of multi-task dataset, three baselines (including a hypernetwork baseline), and a controlled ablation forms a falsifiable test. The frozen backbone checks whether any adaptation is needed at all; if TIDE fails to improve on all tasks, the method is unnecessary. The per-task adapters test whether diffusion-based adaptation offers advantages over explicit per-task modules; if adapters match or exceed TIDE, the diffusion design is not justified. The hypernetwork baseline tests whether TIDE's diffusion mechanism outperforms another dynamic specialization method that also generates task-specific parameters (via weight generation) but does not use iterative refinement. The ablation (one-hot embeddings) specifically tests the claim that learned embeddings enable generalization to unseen tasks; if one-hot embeddings perform equally, the learned embeddings are superfluous. The task-averaged success rate is the right metric because it aggregates performance across tasks, revealing both overall improvement and trade-offs. A uniform gain across tasks would contradict our predicted mechanism (that gains concentrate on tasks requiring specific adaptation), while a gain only on tasks with shared structure would support it.

### Expected outcome and causal chain
**vs. Frozen Vesta backbone** — On a case where the task requires a representation distinct from the default (e.g., fine-grained localization vs. coarse navigation), the frozen backbone produces a one-size-fits-all embedding that is suboptimal for both, causing lower success on the specialized task because it lacks task-specific cues. Our method instead refines the representation via reverse diffusion conditioned on the task embedding, pulling it toward the task-specific attractor, so we expect a noticeable gap (≥5% absolute improvement) on tasks requiring distinct representations, but parity on tasks where the default is already adequate.

**vs. Per-task linear adapters** — On a case where two tasks share latent structure (e.g., spatial reasoning for both navigation and manipulation), the per-task adapters learn independent parameters and may interfere or fail to transfer, leading to lower performance on the shared component because they do not exploit commonality. Our method instead conditions on continuous task embeddings, allowing the score network to interpolate between attractors and capture shared structure, so we expect a smaller gap (<3%) on fully independent tasks but a larger advantage (≥8%) on tasks with overlapping structure.

**vs. Task-specific hypernetworks** — On a case where the task requires fine-grained specialization (e.g., object manipulation with precise gripper control), hypernetworks may overfit to training tasks and generate unstable weights for rare tasks, leading to lower success rates because weight generation depends on limited training data. Our method instead uses diffusion to smoothly adapt the representation, so we expect a modest advantage (≥3%) on tasks with limited training data and on novel tasks, while performance is similar on tasks with abundant data.

**vs. TIDE with one-hot task embeddings** — On a case where a novel task is encountered (e.g., a new object type), one-hot embeddings cannot generalize, leading to poor adaptation because the score network sees an unseen embedding. Our learned embeddings can be inferred via similarity to known tasks (from instruction features), so we expect a large gap (≥10%) on novel tasks, while on seen tasks performance should be similar.

### What would falsify this idea
If TIDE shows uniform improvement (or no improvement) across all tasks relative to the frozen backbone, or if the per-task adapters or hypernetworks consistently outperform TIDE, the central claim that diffusion-based adaptation is beneficial would be falsified. Specifically, if the gain on tasks with shared structure is not larger than on independent tasks, our causal mechanism of interpolation would be unsupported. Additionally, if the attractor reconstruction error remains high (>0.1) despite doubling training data, the score learning assumption fails.

## References

1. Vesta: A Generalist Embodied Reasoning Model
2. Qwen3-VL Technical Report
3. Vision-Language Memory for Spatial Reasoning
4. RoboSpatial: Teaching Spatial Understanding to 2D and 3D Vision-Language Models for Robotics
5. Intern VL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks
6. VILA: On Pre-training for Visual Language Models
7. EmbodiedScan: A Holistic Multi-Modal 3D Perception Suite Towards Embodied AI
8. Scaling Instruction-Finetuned Language Models
