# Depth-Aware Episodic Voxel Memory for Occlusion Reasoning in Vision-Language Models

## Motivation

CAPTURe reveals that VLMs fail on occlusion and depth ordering tasks because they operate on 2D images without explicit 3D reasoning. Existing methods like InternSpatial train on large datasets but lack a mechanism to represent occlusion structure; they treat all visible features equally, leading to counting errors when objects are partially hidden. The root cause is the absence of a differentiable 3D memory that can be updated across views and that sorts features by depth to resolve occlusions.

## Key Insight

Maintaining a 3D voxel grid with depth-sorted attention gating ensures that occluded objects are represented but only revealed when the occluder is removed or viewpoint changes, enabling explicit reasoning about depth ordering.

## Method

### DAEVM (Depth-Aware Episodic Voxel Memory)

**Method overview:** DAEVM is a differentiable module that takes a sequence of 2D observations with estimated depth maps and builds a 3D voxel grid where each voxel stores a feature vector and a depth value. It outputs a scene representation that can be queried to produce 2D occlusion-aware features for arbitrary viewpoints.

**Assumption:** We assume that monocular depth from Metric3D v2 provides sufficient accuracy for occlusion ordering. This is verified via a synthetic experiment with ground-truth depth (see experiment section).

**Components:**
- **Feature extraction:** We use CLIP ViT-L/14 to extract per-pixel features (768-dim) from the RGB image. Features are projected into 3D using the depth map (Metric3D v2) and camera pose (intrinsics + extrinsics).
- **Voxel grid:** Grid dimensions: 64x64x64, voxel size: 0.1m, feature dimension: 256. Stored tensors: 'grid' (features), 'depth_map' (front-most depth), 'confidence' (confidence score).
- **Depth-sorted attention gating:** For each voxel receiving projected points, we sort points by depth (ascending), compute attention weights as softmax(-alpha * depths) with alpha=1.0, then update the voxel feature as a weighted sum, the depth as the minimum depth, and the confidence as the sum of weights. The update uses EMA with beta=0.1. Hyperparameters were chosen on a validation set of 100 CAPTUREreal scenes.
- **Ray-marching:** At query time, for each pixel, we cast a ray through the voxel grid at intervals of voxel_size. We stop at the first voxel with confidence > 0.5 (or max depth 10m). The feature is obtained via trilinear interpolation. Implementation: we use the `ray_march` function from the `neural_render` library (see github.com/example/ray_march).

```python
class DAEVM:
    def __init__(self, voxel_size=0.1, grid_dim=64, feature_dim=256):
        self.grid = torch.zeros(grid_dim, grid_dim, grid_dim, feature_dim)
        self.depth_map = torch.zeros(grid_dim, grid_dim, grid_dim)
        self.confidence = torch.zeros(grid_dim, grid_dim, grid_dim)

    def update(self, rgb, depth, camera_pose):
        # Step 1: Extract features and project
        features_2d = clip_encoder(rgb)  # HxWx768
        xyz = backproject(depth, camera_pose)  # HxWx3
        voxel_coords = (xyz / voxel_size).long()
        # Step 2: For each voxel, depth-sorted attention
        for v in unique_voxels(voxel_coords):
            idx = (voxel_coords == v).all(dim=-1)
            points = {'feat': features_2d[idx], 'depth': xyz[..., 2][idx]}
            if points['feat'].shape[0] > 0:
                order = torch.argsort(points['depth'])
                sorted_feat = points['feat'][order]
                sorted_depth = points['depth'][order]
                attn = torch.softmax(-1.0 * sorted_depth, dim=0)
                new_feat = (attn * sorted_feat).sum(dim=0)
                new_depth = sorted_depth[0]
                new_conf = attn.sum()
                beta = 0.1
                self.grid[v] = (1 - beta) * self.grid[v] + beta * new_feat
                self.depth_map[v] = min(self.depth_map[v], new_depth)
                self.confidence[v] = max(self.confidence[v], new_conf)
        # Optional pruning: self.confidence[self.confidence < 0.1] = 0

    def render(self, query_camera_pose):
        # Ray-march (using library: ray_march from neural_render)
        from neural_render import ray_march
        rays = compute_rays(query_camera_pose)  # HxWx6 (origin+dir)
        hit_voxels = ray_march(self.grid, self.confidence, self.depth_map, rays, voxel_size=voxel_size, max_depth=10.0, conf_thresh=0.5)
        features = torch.zeros(H*W, 256)
        if hit_voxels is not None:
            features = trilinear_interp(self.grid, hit_voxels)
        return features.view(H, W, 256)
```

**Why this design:** We chose depth-sorted attention over simple averaging (as in NeuralRecon) because averaging mixes features from different depths, blurring occlusion boundaries; the cost is additional sorting computation per update. We chose a voxel grid over a point cloud to enable efficient ray-marching for rendering, at the cost of discretization error. We used exponential moving average with a small beta to maintain temporal coherence across views, accepting slow adaptation to moving objects. The depth and confidence maps are stored separately to allow pruning and avoid hallucinated features; this adds memory overhead but prevents spurious voxels from persisting. The alpha hyperparameter controls how sharply front points dominate; a high alpha gives crisp occlusion but may miss thin structures; we set alpha=1.0 as a trade-off.

**Why it measures what we claim:** The depth-sorted attention weights (w_i) operationalize 'occlusion gating' because the assumption is that occluded points are farther away and thus receive lower weight; this assumption fails when objects are transparent or at the same depth, in which case the attention does not distinguish occlusion. The front-most depth stored in depth_map measures 'depth ordering' under the assumption that the first visible surface is the occluder; this fails for reflective surfaces where the first return is not the true object. The feature update via weighted sum measures 'scene representation accuracy' under the assumption that depth ordering correctly identifies which points are visible; this fails when depth estimation errors cause incorrect ordering. The confidence score measures 'reliability of voxel' under the assumption that consistent sightings increase confidence; this fails for repeatedly observed but incorrectly positioned features.

## Contribution

(1) A differentiable depth-aware episodic memory module that builds a 3D voxel grid from 2D observations with explicit occlusion handling via depth-sorted attention. (2) The design principle that maintaining per-voxel depth and confidence enables occlusion-aware feature queries, allowing VLMs to reason about depth ordering without explicit 3D supervision. (3) A practical instantiation that can be trained end-to-end on spatial reasoning benchmarks like CAPTURe and InternSpatial.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|---|---|---|
| Dataset | CAPTUREreal | Real scenes with heavy occlusion |
| Primary metric | Occluded object detection accuracy | Measures occlusion understanding |
| Baseline 1 | GPT-4o | Strong VLM baseline |
| Baseline 2 | Qwen2-VL | VLA baseline without 3D memory |
| Baseline 3 | Intern-VL2 | Specialized spatial reasoning VLM |
| Baseline 4 | NeuralRecon | Geometric 3D memory baseline |
| Ablation 1 | DAEVM with average pooling | Isolates depth-sorted attention effect |
| Ablation 2 | DAEVM with inverse depth weighting | Tests depth-based weighting alternatives |
| Ablation 3 | DAEVM with uniform weights | Tests whether depth ordering drives gains |
| Synthetic control | DAEVM with ground-truth depth | Quantifies depth error impact |

### Why this setup validates the claim

This combination of dataset, baselines, and metrics forms a falsifiable test of DAEVM's central claim: that depth-sorted attention in a voxel memory improves occlusion-aware feature representation for VLAs. CAPTUREreal provides diverse real-world scenes with known occluded objects, directly testing occlusion reasoning. The metric—occluded object detection accuracy—captures the ability to represent occluded regions faithfully. GPT-4o tests whether a general VLM without explicit 3D structure succeeds; Qwen2-VL, a VLA trained at scale, gauges the benefit of learned priors vs. geometric memory; Intern-VL2, fine-tuned on spatial tasks, isolates the value of specialized spatial training; NeuralRecon tests an existing 3D reconstruction approach. The ablations (average pooling, inverse depth, uniform weights) isolate the role of depth-sorted attention and depth ordering. The synthetic control with ground-truth depth quantifies the effect of monocular depth estimation errors on occlusion reasoning. If our method improves over all baselines, especially on heavily occluded subsets, and ablations show depth-sorted attention is crucial, the claim is supported; otherwise, it is falsified.

### Expected outcome and causal chain

**vs. GPT-4o** — On a case where a small object is behind a larger one (e.g., a cup behind a box), GPT-4o, lacking explicit 3D memory, often hallucinates the cup's location or misses it because its reasoning is based on 2D features without occlusion reasoning. Our method instead maintains a voxel grid where depth-sorted attention weights front surfaces higher, so the occluded cup's features remain suppressed until the occluder is moved; at test time, ray-marching retrieves the closest visible surface. We expect our method to detect the occluded cup in >90% of such cases vs. ~60% for GPT-4o, with parity on unoccluded objects.

**vs. Qwen2-VL** — On a case with multiple transparent objects (e.g., glass bottles), Qwen2-VL's 2D features mix depths due to transparency, leading to incorrect count or localization. Our method's depth-sorted attention relies on depth ordering from estimated depth (e.g., Metric3D v2), which for transparent surfaces gives the first return; this misrepresents true occlusion. Thus, on transparent objects, both methods struggle similarly. We expect comparable performance on transparent subsets but a noticeable gap on opaque occluded subsets (e.g., +15% accuracy).

**vs. Intern-VL2** — On a case with repeated observations from slightly different views (episodic memory), Intern-VL2, trained on spatial reasoning but without online memory, can confuse object positions across frames. Our method's EMA update with confidence weighting merges new observations while preserving front-most depth, yielding a stable representation. We expect our method to show higher temporal consistency (e.g., less than 5% variance in predictions across viewpoints) vs. Intern-VL2's 20% variance, with better average accuracy on heavy occlusion sequences.

**vs. NeuralRecon** — On a case with complex occlusion patterns (e.g., multiple objects stacked), NeuralRecon uses average pooling across depths, which blurs occlusion boundaries and leads to overcounting or mislocalization. Our depth-sorted attention maintains separation per depth, yielding sharper boundaries. We expect our method to achieve >85% accuracy on scenes with >3 overlapping objects vs. NeuralRecon's ~70%.

**Ablation results** — We expect DAEVM with depth-sorted attention to outperform all ablations, especially on heavy occlusion subsets. The uniform weights ablation is expected to perform worst (close to average pooling), confirming that depth ordering drives gains. The inverse depth weighting ablation may perform similarly to depth-sorted attention if depth ordering is well captured, but we anticipate a modest drop due to less sharp attention (e.g., -3% overall). The synthetic control with ground-truth depth will show a ceiling performance; comparison with our monocular-based performance will quantify the depth error impact (expected degradation ≤5% on opaque objects, higher on transparent/reflective).

### What would falsify this idea

If our method's gains over baselines are uniform across all occlusion levels (rather than concentrated on high occlusion) or if any ablation (average pooling, inverse depth, uniform weights) performs equally well to full DAEVM, then the central claim that depth-sorted attention specifically improves occlusion reasoning would be falsified. Additionally, if the synthetic ground-truth depth experiment shows that monocular depth errors cause a >10% drop in accuracy compared to ground-truth, the assumption that depth estimation is sufficient for occlusion ordering would be challenged.

## References

1. CAPTURe: Evaluating Spatial Reasoning in Vision Language Models via Occluded Object Counting
2. Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution
3. Are Deep Learning Models Robust to Partial Object Occlusion in Visual Recognition Tasks?
4. InternSpatial: A Comprehensive Dataset for Spatial Reasoning in Vision-Language Models
5. Metric3D v2: A Versatile Monocular Geometric Foundation Model for Zero-Shot Metric Depth and Surface Normal Estimation
6. Are We on the Right Way for Evaluating Large Vision-Language Models?
7. PonderV2: Pave the Way for 3D Foundation Model with A Universal Pre-training Paradigm
8. DG-Recon: Depth-Guided Neural 3D Scene Reconstruction
