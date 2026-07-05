# Attribute Basis Reward: Compositional Reward Functions for Zero-Shot RL Fine-Tuning of Diffusion Models

## Motivation

Existing RL fine-tuning for diffusion models (e.g., Qwen-Image-2.0-RL) requires training a separate reward model for each task, which is costly due to task-specific VLM fine-tuning. These reward models produce a single scalar score without decomposable components, preventing zero-shot adaptation to novel tasks. This limitation arises because the reward signal is not factorized into task-agnostic attribute dimensions that can be linearly composed. For instance, Qwen-Image-2.0-RL uses a fixed, task-specific combination of alignment, aesthetics, and portrait fidelity scores learned from fine-tuned VLMs, but does not learn reusable attribute basis functions that generalize across tasks.

## Key Insight

By training attribute-specific reward heads across a diverse set of tasks with an independence penalty, we learn a linear decomposition of reward where each head captures an orthogonal aspect of human preference, enabling zero-shot reward construction via a user-specified weight vector.

## Method

**Attribute Basis Reward (ABR)**

**(A) What it is:** A multi-head reward model with K=10 attribute heads, each producing a scalar score for an image-prompt pair. The total reward for a task is the inner product of a task-specific weight vector (K-dim) and the attribute scores. The heads are trained jointly across M=50 diverse tasks with known reward labels and weight vectors.

**(B) How it works:**
```python
# Input: task weight vector w (K-dim), image x, prompt p
backbone = Qwen2-VLEncoder(frozen=True)  # pretrained vision-language encoder
embedding = backbone(x, p)  # multimodal embedding

# Attribute heads (K=10 MLPs, each with 2 layers, hidden size 256, GeLU activation, output linear to 1)
scores = []
for i in range(K):
    s_i = MLP_i(embedding)  # scalar output
    scores.append(s_i)

# Composition
reward = sum(w_i * s_i for i in range(K))

# Training loop over tasks T = {task_1, ..., task_M}
# Each task has known weight vector w_t and reward label r_t for (x,p)
batch = sample_batch(task, data)  # batch size 64
for task, (x, p, r_true, w_task) in batch:
    s = [head(backbone(x,p)) for head in heads]
    r_pred = dot(w_task, s)
    mse_loss = (r_pred - r_true)**2
    # Independence penalty: off-diagonal of empirical covariance of s across batch
    s_tensor = torch.stack(s, dim=1)  # (bsz, K)
    cov = (s_tensor.T @ s_tensor) / (bsz - 1) - (s_tensor.mean(0).outer(s_tensor.mean(0)))  # empirical covariance
    ind_penalty = sum_{i!=j} cov[i,j]**2
    # Contrastive loss using MPS-style attribute labels (InfoNCE with temperature τ=0.07)
    # l_contrast = ... (details omitted for brevity)
    total_loss = mse_loss + 0.1 * ind_penalty + 0.05 * l_contrast
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()  # AdamW, lr=1e-4, update only MLP head parameters
```

**(C) Why this design:** We chose a linear combination rather than a nonlinear function (e.g., deep network on concatenated scores) because linearity enables interpretable zero-shot composition without retraining, accepting the trade-off that some tasks may require nonlinear interactions we cannot capture. We froze the visual backbone to leverage strong pretrained representations efficiently; the cost is that attribute heads must adapt to fixed embeddings, limiting expressivity. We added an independence penalty during training to encourage attribute heads to capture orthogonal reward dimensions, reducing redundancy; this may harm accuracy if attributes are inherently correlated, but ensures that weight vector adjustments have clear effects. We optionally incorporate contrastive loss from multi-dimensional preference datasets (e.g., MPS) to bootstrap attribute learning, which risks biasing attribute definitions toward pre-defined dimensions rather than discovering truly task-agnostic ones. **This design assumes that the total reward for any unseen task can be accurately predicted as a linear combination of a fixed set of learned attribute scores that are decorrelated and task-agnostic.**

**(D) Why it measures what we claim:** The attribute score s_i measures the degree to which the image satisfies attribute i (e.g., aesthetic quality) because the linear weight w_i multiplies s_i to contribute to total reward; this operationalization assumes reward is a linear function of independent attribute strengths, which holds if training tasks span diverse weight combinations and the independence penalty decorrelates the heads. This assumption fails when two attributes are inherently correlated (e.g., aesthetic quality and text alignment are often coupled), in which case scores may both capture the same variance, making weight vector effects ambiguous. To mitigate, we explicitly penalize covariance, forcing heads to be as decorrelated as possible, but at the cost of potentially discarding shared variance that is genuinely part of reward. The covariance penalty measures the degree of orthogonality between attribute score distributions, assuming linear decorrelation suffices (assumption A). Failure mode: nonlinear dependencies (e.g., third-order correlations) may still exist, in which case the penalty may not capture true disentanglement.

## Contribution

(1) A novel method for learning composable reward functions for diffusion model RL fine-tuning, using task-agnostic attribute heads trained with a linear factorization objective and independence regularization. (2) A training procedure that combines supervised learning from diverse tasks with a covariance-based independence penalty to discover interpretable, orthogonal reward dimensions. (3) Zero-shot reward composition for novel tasks via user-specified weight vectors, eliminating the need for task-specific VLM fine-tuning.

## Experiment

### Evaluation Setup
| Role | Choice | Rationale (≤12 words) |
|------|--------|----------------------|
| Dataset | Multi-task with diverse weight vectors (synthetic + real, synthetic: varying correlation strengths 0.0, 0.3, 0.6, 0.9) | Tests compositional generalization & robustness to correlation |
| Primary metric | Held-out task reward accuracy (Pearson correlation between predicted and true reward) | Directly measures composition |
| Baseline 1 | GRPO with monolithic reward head | Lacks attribute decomposition |
| Baseline 2 | GRPO with non-linear attribute combiner (2-layer MLP, hidden 256) | Tests linearity assumption |
| Ablation-of-ours | ABR without independence penalty | Isolates decorrelation effect |

### Why this setup validates the claim
This experimental design directly tests whether linear composition of decorrelated attribute heads enables zero-shot generalization to unseen reward tasks. The multi-task dataset with diverse weight vectors for each task ensures that the model must learn to recombine attributes rather than memorize task-specific mappings. The inclusion of synthetic data with controlled attribute correlations (0.0 to 0.9) allows us to systematically test the robustness of the independence penalty when attributes are inherently correlated. The primary metric (Pearson correlation between predicted and true reward on held-out weight combinations) measures performance on weight combinations not seen during training, which is the core claim. Baseline 1 (monolithic reward head) cannot decompose attributes, so it must interpolate between tasks; if our method succeeds, it should clearly outperform on held-out combinations where interpolation fails. Baseline 2 (non-linear combiner) tests whether the linearity assumption is beneficial; if attributes are indeed linearly separable, our method should match or beat it with better interpretability. The ablation (no independence penalty) isolates the effect of decorrelation: if correlated attributes hurt composition, this variant should underperform on tasks requiring independent adjustment of individual attributes.

### Expected outcome and causal chain
**vs. GRPO with monolithic reward head** — On a held-out task requiring a new combination of aesthetics and safety (e.g., high aesthetics but low safety), the monolithic head, trained on tasks with different covariances, may produce a reward that inaccurately reflects the desired trade-off because it cannot isolate the two dimensions. Our method instead learns separate attribute heads for aesthetics and safety, so the linear combination with the new weight vector directly yields the correct reward. We expect a noticeable gap in reward accuracy on held-out tasks (e.g., >10% improvement) while performance on training tasks may be similar.

**vs. GRPO with non-linear attribute combiner** — On a task where training weight vectors are linearly separable in attribute space, the non-linear combiner may overfit to the training distribution and extrapolate poorly to unseen weight vectors, while our linear combiner generalizes by construction. We expect our method to achieve higher held-out reward accuracy, especially when the number of training tasks is limited, but may be similar on training tasks. The non-linear combiner might also show higher variance across seeds.

**vs. ABR without independence penalty** — On a task where two attributes (e.g., aesthetic quality and text alignment) are inherently correlated in the training data, the variant without penalty may learn entangled representations, causing the weight vector adjustments to affect both attributes simultaneously. Our method with penalty forces decorrelation, so changing one weight cleanly changes the corresponding attribute. We expect the full ABR to show better interpretability and monotonicity when sweeping a single weight, and also better accuracy on tasks that require independent control, e.g., increasing weight on text alignment while decreasing aesthetics.

### What would falsify this idea
If ABR's held-out task accuracy does not significantly outperform the monolithic baseline, or if its gains are uniform across all subsets rather than concentrated on tasks requiring attribute recomposition, then the central claim of compositional generalization is falsified. Additionally, if on synthetic data with high attribute correlation (e.g., 0.9) the ABR underperforms the no-penalty variant, then the independence penalty harms generalization in correlated settings.

## References

1. Qwen-Image-2.0-RL Technical Report
2. GRPO-Guard: Mitigating Implicit Over-Optimization in Flow Matching via Regulated Clipping
3. HPSv3: Towards Wide-Spectrum Human Preference Score
4. Scaling Rectified Flow Transformers for High-Resolution Image Synthesis
5. Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution
6. Learning Multi-Dimensional Human Preference for Text-to-Image Generation
7. Boosting Latent Diffusion with Flow Matching
8. AGIQA-3K: An Open Database for AI-Generated Image Quality Assessment
