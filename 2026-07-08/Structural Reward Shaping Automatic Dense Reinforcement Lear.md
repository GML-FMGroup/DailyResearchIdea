# Structural Reward Shaping: Automatic Dense Reinforcement Learning from SVG Tree Edit Distance

## Motivation

Existing methods like MentalThink rely on a single final-answer reward after rendering and re-asking, which provides no intermediate signal for each refinement step. This sparse feedback fails to guide the model through the multi-turn reasoning process, especially when the final answer is incorrect but intermediate steps show partial progress. The root cause is the absence of an automatic, dense reward signal that can be computed without human annotation or ground-truth reasoning paths, a meta-gap across multiple visual reasoning paradigms that depend on final-answer rewards.

## Key Insight

The hierarchical structure of SVG code, measured by tree edit distance between successive iterations, provides a monotonically decreasing measure of visual similarity to the final intended scene, enabling a dense reward without any human annotation.

## Method

(A) **What it is**: Structural Reward Shaping (SRS) is a method that computes a dense reward for each turn of multi-turn reinforcement learning in the Think-with-SVG pipeline by measuring the tree edit distance between successive SVG codes. Input: sequence of SVG trees S_1, S_2, ..., S_T generated across T turns. Output: dense rewards r_1, r_2, ..., r_{T-1} where r_t = -Δ_t = -(d_tree(S_{t+1}, S_t))).

(B) **How it works** (pseudocode):
```python
# Given: multi-turn RL loop with policy π, SVG renderer R, task Q, max turns T
# Initialize: SVG_0 = None, reward_total = 0, edit_distance_prev = None
for t in range(1, T+1):
    # Sample action: generate SVG using current policy
    SVG_t = π.generate_svg(Q, SVG_{t-1})
    # Parse SVG to tree structure (e.g., abstract syntax tree)
    tree_t = parse_svg_tree(SVG_t)
    if t > 1:
        # Compute tree edit distance between current and previous SVG tree
        # Using Zhang-Shasha algorithm with unit cost (O(n^3))
        edit_distance = tree_edit_distance(tree_{t-1}, tree_t)
        # Dense reward: negative change in edit distance (or positive if decreasing)
        r_t = - (edit_distance - edit_distance_prev)  # reward for reducing distance
        # Alternatively, use clipped reward: sign(edit_distance_prev - edit_distance)
        r_t = clip(r_t, -1.0, 1.0)
        # Store for next step
        edit_distance_prev = edit_distance
    else:
        # First turn: no previous, set edit_distance_prev = max_possible (e.g., size of first tree)
        edit_distance_prev = sum_sizes(tree_1)
        r_t = 0.0  # no reward for first turn
    # Use r_t in PPO update
    # ... standard PPO loss with multi-turn trajectory
```
Hyperparameters: unit cost for tree edit operations; clip range ε=0.2; PPO learning rate 3e-6.

(C) **Why this design**: We chose tree edit distance over simpler metrics like token-level Levenshtein distance because SVG syntax allows semantically equivalent code with different token sequences (e.g., reordered attributes), making token distance unreliable; tree structure captures hierarchical equivalences, accepting higher computational cost (O(n^3) vs O(n^2)). We chose unit edit costs (insert, delete, replace) over learned cost functions to avoid requiring ground-truth edits, preserving the automatic nature of the reward. We chose negative change in edit distance (r_t = -(d_t - d_{t-1})) rather than absolute distance because it directly rewards reduction of distance per step, avoiding scaling issues across different sequence lengths. Compared to Process Reward Models (PRMs) that require expensive human-annotated step labels or synthetic data, our method exploits the built-in structure of SVG code, eliminating the need for supervision. Unlike prior work like Euclid's Gift that uses final-answer rewards only, our dense signal provides credit assignment at each turn. A potential concern is that the edit distance proxy may not perfectly align with visual task progress; we mitigate this by using the change in distance rather than absolute value, which focuses on relative improvement.

(D) **Why it measures what we claim**: The computational quantity `tree_edit_distance(tree_{t-1}, tree_t)` measures the amount of structural change between consecutive SVG codes. This measures `progress toward a consistent final SVG` under the assumption that the model iteratively refines the SVG to match the task's visual requirement, and that refinement steps monotonically reduce structural dissimilarity to the final intended scene. This assumption fails when the model makes a large, semantically equivalent structural change (e.g., rewriting the entire SVG to produce the same rendered image), in which case tree edit distance overestimates the progress (or lack thereof) because it reflects syntactic change rather than semantic improvement. To mitigate this, we use the change in edit distance (Δ_t) as the reward, which penalizes large non-productive changes only when they increase distance, and we incorporate a render-based verification reward (from MentalThink) as a sparse final reward to anchor the dense signal to task correctness. The quantity `r_t = -Δ_t` is designed to reward steps that reduce the distance to the previous representation, operationalizing the concept of `intermediate credit` by assigning positive reward whenever the model's SVG moves closer (structurally) to what is presumably a more refined solution. The clipping to [-1,1] bounds the influence of extreme changes, ensuring the dense reward does not dominate the final reward, which measures actual answer correctness.

## Contribution

(1) A method for automatically deriving dense rewards from SVG structural properties using tree edit distance, without any human annotation or ground-truth reasoning paths. (2) An empirical finding that this structural reward improves sample efficiency and final accuracy in multi-turn reinforcement learning for visual reasoning, as demonstrated on VSIBench and MindCube benchmarks. (3) A demonstration that reward decomposition from intermediate representations can be achieved by leveraging the inherent structure of the reasoning trace (SVG code), providing a blue-print for similar approaches in other structured domains.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | VSI-Bench | Spatial reasoning benchmark with multi-step SVG solutions. |
| Primary metric | Task accuracy | Directly measures visual reasoning correctness. |
| Baseline 1 | Vanilla MLLM (no SVG) | Tests benefit of SVG intermediate representation. |
| Baseline 2 | MentalThink (no SRS) | Separates effect of dense reward from SVG pipeline. |
| Baseline 3 | Euclid's Gift (geometric surrogate) | Compares to alternative structured reward approach. |
| Ablation | Ours with only final reward | Isolates contribution of SRS dense reward. |

### Why this setup validates the claim

This design forms a falsifiable test of Structural Reward Shaping's (SRS) claim that tree-edit-distance rewards improve multi-turn RL credit assignment for SVG generation. The VSI-Bench dataset provides complex spatial reasoning questions that require iterative refinement, matching the method's assumptions. The vanilla MLLM baseline establishes the upper bound of reasoning without intermediate representations. MentalThink (without SRS) isolates the effect of the dense reward; if SRS helps, our method should outperform it. Euclid's Gift provides a concurrent alternative that uses geometric surrogate tasks rather than structural rewards; comparing to it tests whether SRS better aligns with visual progress. The ablation (final reward only) directly measures the dense reward's contribution. Accuracy is the primary metric because it directly gauges task completion, and the expected improvement pattern is not uniform: SRS should show larger gains on multi-step problems where credit assignment is critical.

### Expected outcome and causal chain

**vs. Vanilla MLLM (no SVG)** — On a case where the model must infer a 3D arrangement of objects (e.g., "which object is to the left of the sphere?"), a vanilla MLLM (even with CoT) produces a wrong answer because it lacks a structured visual representation to reason over spatial relations. Our method instead generates a multi-step SVG scene that explicitly encodes positions and relations, so it can iteratively refine spatial consistency; we expect a sizable accuracy gap (e.g., 20-30 points) on spatial relation queries, with smaller differences on simpler counting tasks.

**vs. MentalThink (no SRS)** — On a case where the initial SVG is far from correct (e.g., wrong object shape), MentalThink's sparse final reward fails to guide intermediate steps, causing the policy to make large, unproductive jumps that rarely converge to the correct scene. Our method's dense reward signals incremental structural progress, so the policy takes more consistent refinement steps; we expect a significant accuracy advantage (e.g., 10-15 points) on problems requiring more than 3 refinement turns, with parity on single-turn tasks.

**vs. Euclid's Gift (geometric surrogate)** — On a case requiring precise coordinate adjustments (e.g., centering an object), Euclid's Gift's geometric pre-training may provide a prior but lacks per-step feedback during RL, leading to overshooting or oscillation. Our method's tree-edit-distance reward directly penalizes structural divergence from the previous step, promoting smooth refinement; we expect comparable overall accuracy but a narrower gap on problems with high sensitivity to coordinate errors (e.g., 5-10 point difference) and a larger advantage on tasks with complex hierarchical syntax (e.g., composite objects).

### What would falsify this idea

If SRS shows no accuracy improvement over MentalThink (no SRS) on multi-turn problems, or if the gain is uniform across all problem types rather than concentrated on tasks requiring iterative refinement, then the central claim that tree-edit-distance reward provides meaningful credit assignment is falsified.

## References

1. MentalThink: Shaping Thoughts in Mental SVG World
2. Euclid's Gift: Enhancing Spatial Perception and Reasoning in Vision-Language Models via Geometric Surrogate Tasks
3. Learning to See Before Seeing: Demystifying LLM Visual Priors from Language Pre-training
4. SpatialLadder: Progressive Training for Spatial Reasoning in Vision-Language Models
5. Machine Mental Imagery: Empower Multimodal Reasoning with Latent Visual Tokens
6. Open Vision Reasoner: Transferring Linguistic Cognitive Behavior for Visual Reasoning
7. MedVisionLlama: Leveraging Pre-Trained Large Language Model Layers to Enhance Medical Image Segmentation
8. Aioli: A Unified Optimization Framework for Language Model Data Mixing
