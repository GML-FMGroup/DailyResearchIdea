# Task-Aware Active Viewpoint Selection for Task-Sufficient Digital Twins

## Motivation

Existing sim-to-real pipelines like SimFoundry reconstruct digital twins from a single video, but occlusion and incomplete geometry inevitably occur. Prior reconstruction methods treat geometric completeness as the sole objective, ignoring that for a downstream policy, missing geometry in task-irrelevant regions is harmless while missing task-critical geometry can cause policy failure. This limitation is structural: without task-driven guidance, viewpoint selection defaults to coverage heuristics, and volumetric fusion weights all voxels equally. We need a method that actively selects viewpoints to fill only the task-relevant gaps, minimizing acquisition cost while ensuring policy success.

## Key Insight

The policy's sensitivity to local geometry can be approximated by a learned importance field that maps partial reconstructions to per-voxel task relevance, enabling utility-driven viewpoint selection without exhaustive rollouts.

## Method

## Task-Aware Active Viewpoint Selection for Task-Sufficient Digital Twins

### (A) What it is
We propose **TAVST** (Task-Aware Viewpoint Selection for Twins), a framework that takes a single video input and a frozen downstream policy, and outputs a task-sufficient digital twin by actively selecting additional views based on a learned task-relevance field. Input: initial video frames + policy parameters; Output: a volumetric reconstruction where voxels are weighted by their expected impact on policy success. **Load-bearing assumption**: The utility network U, trained on synthetic data, generalizes to real-world scenes without fine-tuning; we mitigate this with domain randomization during training.

### (B) How it works (pseudocode)
```python
# Phase 1: Initial reconstruction and task-relevance field training
# Pre-train a lightweight utility network U (MLP with 3 layers, 256 units, GeLU activation) using synthetic data
# with domain randomization (varying lighting, textures, camera noise). 
# For each object in a simulator, sample random partial voxel grids X_partial and compute
# policy success rate delta when that voxel is completed vs. missing. U maps a local feature
# (voxel occupancy + distance to policy trajectory) to a scalar relevance score.
# Phase 2: Active viewpoint selection
# Input: initial video frames V0, camera pose P0, task policy pi
# 1. Reconstruct initial volumetric grid G0 (128^3 voxels) via task-weighted fusion (Eq. 1)
# 2. For each candidate viewpoint p in a discrete set (e.g., 8 octahedron vertices):
#    Compute the set of voxels V_p that would become visible from p.
#    Utility(p) = sum_{v in V_p} U(features(v, G0))
# 3. Select p* = argmax utility(p)
# 4. Capture new view at p*, update G using weighted integration:
#    new_weight(v) = old_weight(v) + (1 if v visible else 0) * (1 + U(features(v, G0)))
# 5. Repeat steps 2-4 for K iterations (K=3) or until utility gain < threshold
# Eq. 1: task-weighted volumetric fusion:
#   G(v) = sum_{i} w_i(v) * color_i(v) / sum w_i(v), where w_i(v) = confidence_i(v) * (1 + alpha * task_relevance(v))
#   alpha = 0.5 (hyperparameter)
```

### (C) Why this design
We chose a lightweight MLP for the utility network over an end-to-end reinforcement learning controller because: (1) The utility network is trained offline on synthetic data, avoiding expensive online policy rollouts during viewpoint selection; the trade-off is a potential domain gap between synthetic and real task-relevance distributions. (2) We use a discrete candidate viewpoint set (octahedron vertices) instead of continuous optimization because it simplifies the search and suffices for tabletop manipulation scenes; the cost is suboptimal coverage for highly concave objects. (3) Task-weighted fusion with alpha=0.5 balances between raw geometric confidence and task relevance; we chose a fixed alpha over learned weights to avoid overfitting to a specific policy, accepting that a fixed weighting may not be optimal across all tasks. (4) We iterate only 3 times to bound acquisition cost; the trade-off is possible incomplete coverage if the initial view is extremely poor, but in practice three extra views capture most task-critical occlusions.

### (D) Why it measures what we claim
The utility network score U(features(v, G0)) measures **task sufficiency** of completing voxel v because we trained U on the policy success delta when v is completed, assuming that the delta correlates with the policy's local geometric sensitivity; this assumption fails when the policy is robust to geometry changes (e.g., via visual invariance), in which case U may overestimate relevance. The task-weighted fusion weight w_i(v) = confidence_i(v) * (1 + alpha * U(v)) measures **informativeness** of observation i for voxel v because it emphasizes voxels that are both reliably observed and task-critical; this assumption fails when U is poorly calibrated, leading to overemphasis on noisy task-relevance estimates. The viewpoint selection utility(p) = sum_{v in V_p} U(features(v, G0)) measures **expected policy benefit** of viewing from p because it aggregates task-relevance over newly visible voxels, assuming that the policy benefit is additive across voxels (i.e., the policy treats each voxel independently); this assumption fails when interactions between multiple voxels matter (e.g., a policy requires two occluded parts simultaneously visible), in which case the sum underestimates the true benefit.

## Contribution

(1) A lightweight utility network that predicts per-voxel task relevance from partial reconstructions, enabling task-driven viewpoint selection. (2) An active viewpoint selection algorithm that maximizes expected policy benefit using a discrete candidate set and a stopping criterion based on utility gain. (3) A task-weighted volumetric fusion method that integrates observations with weights reflecting both confidence and task relevance, producing a digital twin sufficient for task success.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|-----------------------|
| Dataset | SimFoundry: 7 real-world manipulation tasks | Represents diverse sim-to-real challenges |
| Primary metric | Task success rate | Directly measures policy performance |
| Baseline 1 | Passive multi-view (fixed rig) | Tests necessity of active selection |
| Baseline 2 | Random viewpoint selection | Controls for exploration vs. task-aware |
| Baseline 3 | Full coverage (dense views) | Upper bound for reconstruction quality |
| Ablation-of-ours | Uniform-weighted fusion (no task weight) | Isolates benefit of task-relevance weighting |

### Why this setup validates the claim
This combination forms a falsifiable test of TAVST's central claim: that task-aware active view selection yields task-sufficient digital twins with minimal views. The dataset's variety in object geometry and task demands challenges the utility network's generalization from synthetic training. Including a passive baseline tests whether active selection is necessary at all; a random baseline tests whether the task-relevance field provides meaningful guidance; a full-coverage baseline establishes an upper bound. Task success rate is the correct metric because it directly measures the policy's ability to act on the reconstructed twin, aligning with the goal of task sufficiency. The ablation isolates the contribution of task-weighted fusion, showing whether the gain comes from active selection alone or the weighting mechanism.

### Expected outcome and causal chain

**vs. Passive multi-view (fixed rig)** — On a case where the initial view misses task-critical geometry (e.g., a thin handle occluded by the robot arm), the passive baseline obtains only a partial reconstruction, causing the policy to fail because it cannot perceive the handle. Our method selects an additional view that reveals the handle, because the utility network assigns high relevance to voxels near the handle's grasp location. We therefore expect a noticeable gap: our method succeeds on >80% of such occlusion cases, while the passive baseline fails on most (success rate <30%).

**vs. Random viewpoint selection** — On an object with multiple candidate grasping points but only one that is task-relevant (e.g., a cup's rim versus its side), random selection may occasionally choose a view that reveals an irrelevant part, wasting the acquisition budget. Our method consistently picks the view that exposes the rim because the utility network scores rim voxels higher due to the policy's dependence on rim geometry for lifting. We expect our success rate to be relatively stable (e.g., 90% across trials), whereas random selection shows high variance (range 40–70% depending on luck).

**vs. Full coverage (dense views)** — On a simple convex object (e.g., a cube), the full coverage baseline uses many redundant views and achieves near-perfect reconstruction, leading to high success. Our method, using only 3 additional views, also achieves high success because the utility network correctly identifies that most voxels are non-critical, so early stopping does not harm performance. We expect both methods to achieve >95% success on such cases, with full coverage marginally higher (~98%). The key signal is parity on simple objects but a larger gap on complex ones where task relevance is uneven.

### What would falsify this idea
If the ablation of uniform weighting performs as well as the full method across all object types, then task relevance is not contributing. Alternatively, if our method's gain over random is uniform across all tasks rather than concentrated on occlusion-heavy or geometrically diverse subsets, then our utility network is not capturing true task sensitivity.

## References

1. SimFoundry: Modular and Automated Scene Generation for Policy Learning and Evaluation
2. ART: Articulated Reconstruction Transformer
3. SAM 3D: 3Dfy Anything in Images
4. IGen: Scalable Data Generation for Robot Learning from Open-World Images
5. TabletopGen: Instance-Level Interactive 3D Tabletop Scene Generation from Text or Single Image
6. Articulate your NeRF: Unsupervised articulated object modeling via conditional view synthesis
7. MeshArt: Generating Articulated Meshes with Structure-Guided Transformers
8. LEIA: Latent View-invariant Embeddings for Implicit 3D Articulation
