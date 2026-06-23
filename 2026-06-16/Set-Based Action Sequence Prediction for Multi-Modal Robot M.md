# Set-Based Action Sequence Prediction for Multi-Modal Robot Manipulation

## Motivation

Existing vision-language-action models (VLAs) fine-tuned with the OFT recipe (e.g., OpenVLA-OFT) rely on deterministic L1 regression, forcing the model to average over multiple feasible action trajectories, leading to poor performance on tasks with ambiguous solutions. Diffusion-based methods (e.g., RDT-1B, π0) model multi-modality but require iterative denoising steps at inference, increasing latency and computational cost. There is a need for a deterministic, single-pass approach that can output multiple distinct action sequences.

## Key Insight

By treating action sequences as unordered set elements and optimizing a bipartite matching between predictions and ground-truth trajectories, the model is forced to spread its predictions across the support of the data distribution without requiring iterative sampling.

## Method

**A) What it is**: Set-based Action Sequence Prediction (SASP) is a deterministic VLA that outputs a fixed-size set of N action sequences (each of length T) and is trained with a set prediction loss using bipartite matching. Input: visual observation and language instruction; Output: N action sequences.

**B) How it works**:
```python
# Training phase:
1. Encode observation o and language l into latent representation h via a Transformer encoder.
2. Decode N action sequences X_pred = {x_i in R^{T×D}} from h using a set of N learned query embeddings and a Transformer decoder with cross-attention.
3. Let Y = {y_j} be the set of ground-truth action sequences for the current task (may be multiple).
4. Compute cost matrix C where C_{i,j} = ||x_i - y_j||_1 / T (averaged over time steps).
5. Use Hungarian algorithm to find matching π: {1..N} -> {1..M} ∪ {∅} (M = |Y|).
6. Loss = (1/|matched|) * Σ_{(i,j) matched} L1(x_i, y_j) + λ * Σ_{unmatched i} max(0, ||x_i||_2 - τ)
   Hyperparameters: N=5, τ=0.1, λ=0.01.
7. Backpropagate through the entire graph (matching is fixed during loss computation).
```

**C) Why this design**: We chose a deterministic set-to-set matching over a probabilistic mixture because it avoids iterative sampling seen in diffusion methods, reducing inference to a single feedforward pass. The trade-off is that the number of modes N must be fixed a priori; tasks with more than N modes will be under-covered. We use a push-away loss for unmatched trajectories rather than a binary no-trajectory loss (as in DETR) because in robotics there is no concept of 'no object'; instead, we want unmatched predictions to be diverse and not collapse to average. The cost is that the push-away loss may encourage unrealistic trajectories far from the data manifold. We use L1 distance for matching because it aligns with the regression objective and is differentiable (via subgradients), though it assumes temporal alignment; we accept the cost that the model cannot handle varying-length trajectories without padding. Finally, we employ a Transformer decoder with learned queries to generate the set, as it naturally handles set prediction and can be trained end-to-end; this adds computational overhead compared to a single-output decoder.

**D) Why it measures what we claim**: The bipartite matching loss ensures that each predicted trajectory is assigned to a unique ground-truth trajectory, thereby enforcing that the predicted set covers the support of the ground-truth distribution — `C_{i,j}` measures the `pairwise dissimilarity` between predicted and ground-truth trajectories; the matching minimizes total dissimilarity, which operationalizes `coverage of feasible modes` under the assumption that ground-truth trajectories are independent draws from the true multi-modal action distribution. This assumption fails when the dataset contains only one trajectory per task (unimodal), in which case the matching still forces one predicted trajectory to match the single ground-truth and penalizes the others, encouraging them to be diverse but potentially unrealistic. The push-away loss `||x_i||_2` measures `diversity of unmatched trajectories` on the assumption that large-norm sequences are further from the origin and thus diverse; this assumption fails when most feasible trajectories have small norm, in which case the push-away loss may push predictions outside the feasible region.

## Contribution

(1) Introduces SASP, a deterministic set-prediction framework for vision-language-action models that outputs multiple feasible action sequences in a single forward pass without iterative sampling. (2) Reveals that bipartite matching over action sequences with a push-away loss effectively encourages multi-modal coverage, providing a design principle for building non-probabilistic multi-modal robot policies.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | Multi-modal manipulation benchmark (e.g., LIBERO with multiple demos) | Contains tasks with multiple feasible action sequences |
| Primary metric | Task success rate (%) | Measures end-to-end task completion |
| Baseline 1 | OpenVLA | Standard deterministic VLA baseline |
| Baseline 2 | RDT-1B | Diffusion-based multi-modal VLA |
| Baseline 3 | Multimodal Diffusion Transformer | Transformer-based diffusion baseline |
| Ablation-of-ours | SASP w/o push-away loss | Tests importance of diversity regularization |

### Why this setup validates the claim

This experimental design directly tests the central claim that set-based deterministic prediction with bipartite matching and push-away loss improves coverage of multi-modal action distributions, leading to higher success on tasks with multiple feasible trajectories. Using a multi-modal benchmark with multiple demonstrations per task ensures that ground-truth multimodality exists, so the metric can detect whether SASP captures modes. Comparison against OpenVLA (single-output) isolates the benefit of set prediction; against diffusion baselines (RDT-1B, MDT) tests whether deterministic set prediction can match or exceed stochastic multi-modal generation. The ablation removes the push-away loss to measure its contribution to diversity. Success rate is chosen because it is the ultimate goal of action sequence prediction and directly reflects whether the predicted actions enable task completion. If SASP outperforms OpenVLA on multi-modal tasks but not single-modal tasks, the claim that diversity drives improvement is supported, while parity with diffusion baselines would confirm competitiveness.

### Expected outcome and causal chain

**vs. OpenVLA** — On a case where the task has two distinct feasible trajectories (e.g., picking up a block from left or right), OpenVLA produces a single averaged trajectory that fails because its deterministic decoder collapses modes. Our method instead predicts a set of N trajectories, with matching assigning one to each ground-truth mode and push-away keeping others diverse, so we expect SASP success rate to be noticeably higher on such tasks (e.g., >20% relative improvement) while parity holds on tasks with only one feasible trajectory.

**vs. RDT-1B** — On a case where three modes exist but diffusion sampling is stochastic, RDT-1B may miss one mode due to uneven probability mass because its iterative denoising can suffer from mode collapse. Our method deterministically covers all modes (as long as N ≥ number of modes) because each learned query embedding specializes in a different mode via the set loss. Thus we expect SASP to achieve higher consistency across seeds and a higher average success rate on tasks with many modes (e.g., 5-10% absolute improvement), with similar performance on single-mode tasks.

**vs. Multimodal Diffusion Transformer (MDT)** — On a long-horizon task (e.g., 100-step sequence) with complex dependencies, MDT's iterative sampling can accumulate errors over steps because each denoising step is conditioned on noisy inputs. Our method predicts entire sequences in a single feedforward pass with no iterative refinement, so errors do not compound. We expect SASP to show a larger success rate gap on long-horizon tasks (e.g., >15% relative) than on short tasks.

### What would falsify this idea

If SASP does not significantly outperform OpenVLA on clearly multi-modal tasks (e.g., tasks where human demonstrations exhibit disjoint trajectories), or if the performance gain is uniform across all tasks including single-modal ones, then the central claim that set prediction specifically improves multi-modal coverage would be falsified. Additionally, if the ablation without push-away loss matches the full method, the diversity loss is unnecessary.

## References

1. Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success
2. RDT-1B: a Diffusion Foundation Model for Bimanual Manipulation
3. Multimodal Diffusion Transformer: Learning Versatile Behavior from Multimodal Goals
4. ALOHA Unleashed: A Simple Recipe for Robot Dexterity
5. π0: A Vision-Language-Action Flow Model for General Robot Control
6. TinyVLA: Toward Fast, Data-Efficient Vision-Language-Action Models for Robotic Manipulation
7. Manipulate-Anything: Automating Real-World Robots using Vision-Language Models
8. Predictive Inverse Dynamics Models are Scalable Learners for Robotic Manipulation
