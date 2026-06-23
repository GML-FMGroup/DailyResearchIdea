# Topological Affordance Embedding for Deformable Object Manipulation

## Motivation

Existing affordance-driven manipulation methods (e.g., BridgeACT) assume quasi-static objects, failing when objects deform during interaction. This structural limitation arises because they rely on static keypoint detection that breaks under deformation. A representation invariant to deformation is needed to ground affordances on deformable objects.

## Key Insight

The topological structure of functional parts is preserved under continuous deformation, making topological embeddings learned via persistent homology naturally invariant to non-rigid transformations.

## Method

### (A) What it is
TAEDO takes a sequence of RGB-D frames and outputs a deformation-invariant topological embedding per pixel, which is then used to predict affordance maps. The embedding is learned via a contrastive loss that aligns temporal correspondences across deformation states.

### (B) How it works (pseudocode)
```python
# Phase 1: Feature and topology extraction per frame
for frame in video_sequence:
    cnn_features = ResNet50(frame)  # shape H x W x 512, frozen backbone
    # Compute persistence diagram on local patches (32x32) using 0-dim homology (Gudhi library, cubical complex)
    # For each pixel, extract 20-dim persistence image from top-10 birth-death pairs
    persistence_vectors = persistent_homology(cnn_features, patch_size=32, max_persistence_pairs=10)  # H x W x 20
    embedding = concat(cnn_features, persistence_vectors)  # H x W x 532

# Phase 2: Temporal contrastive alignment (InfoNCE loss, temperature τ=0.1)
for each pair (t, t+1):
    # Use RAFT optical flow to get correspondence map C_flow
    positives = sample_pixels_from_flow(frame_t, frame_t+1, C_flow)
    negatives = random_pixels_from_different_objects (32 per positive)
    loss = InfoNCE(embedding_t[positives], embedding_t+1[positives], negatives, temperature=0.1)

# Phase 3: Affordance decoder (2-layer MLP, hidden=256, GeLU, trained with human demonstration labels)
affordance_map = MLP(embedding, hidden_dim=256, num_layers=2, activation='GeLU')  # per-pixel affordance heatmap

# Training: AdamW optimizer, lr=1e-4, batch size 16, 100 epochs
# Verification: Compute average embedding distance on held-out deformation pairs; if >0.5, retrain with deformation augmentation
```
Hyperparameters: patch size 32, top-10 persistence features, temperature τ=0.1, MLP hidden 256, 2 layers, AdamW lr=1e-4, batch size 16, 100 epochs.

### (C) Why this design
We chose topological embedding via persistent homology over learned deformable keypoints (e.g., SuperPoint) because topology is geometrically invariant to deformation by definition, whereas learned keypoints may collapse under non-rigid motion; the cost is that topological features are less discriminative for fine-grained motion. We used temporal contrastive alignment over a reconstruction loss because it directly enforces invariance without modeling the deformation process, avoiding generative artifacts; the trade-off is reliance on accurate optical flow for positive pairs. We integrated topological features with CNN features rather than pure topology because CNNs capture local appearance critical for affordance grounding; the downside is higher computational cost. We selected persistent homology over simpler Betti numbers because persistence diagrams are differentiable and richer, enabling end-to-end training. **Load-bearing assumption:** Persistent homology computed on local patches of CNN features yields embeddings that are invariant to non-rigid deformations of the object. This assumption may fail under large deformations that change the topology of the feature space; we verify its validity by measuring embedding stability on synthetic deformation pairs (see (D)).

### (D) Why it measures what we claim
The persistence diagram's birth-death pairs measure the topological structure (connected components) of the object's surface; this measures deformation-invariant affordance because under continuous deformation the topology of functional parts (e.g., handle) remains unchanged; this assumption fails when deformation changes connectivity (e.g., cutting a rope), in which case the embedding captures new topology but may not align with original affordance. The contrastive loss aligns embeddings across frames by minimizing distance between same-physical-point embeddings; this measures temporal consistency of topological features because it forces invariance to non-rigid motion; this assumption fails when optical flow is inaccurate (e.g., occlusions), in which case the loss aligns incorrect points, degrading invariance. The affordance decoder maps aligned embeddings to task heatmaps; this measures affordance grounding because the decoder learns associations between topological structure and successful interaction; this assumption fails when training data lacks deformation diversity, leading to overfitting to seen states. **Equivalence chain:** X=persistence diagram, Y=deformation-invariant affordance, assumption A=topology of functional parts preserved under deformation, failure F=when deformation changes connectivity (cutting), in which case the embedding captures new topology and may not align with original affordance.

## Contribution

(1) A framework integrating persistent homology with contrastive learning to learn deformation-invariant affordance embeddings for robotic manipulation. (2) The design principle that topological structure preserved under continuous deformation enables zero-shot generalization to unseen deformations. (3) A method for temporal alignment of topological embeddings using optical flow and contrastive loss, enabling training from human demonstration videos without explicit deformation modeling.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|-----------------------|
| Dataset | Human demo videos of tool tasks (scoop, hammer, sweep) | Captures diverse deformation states |
| Primary metric | Task success rate over 10 trials per task | Direct measure of affordance utility |
| Baseline 1 | AffordanceNet (supervised affordance) | Tests need for topological invariance |
| Baseline 2 | SuperPoint + affordance decoder (keypoint-based) | Tests learned vs. topological keypoints |
| Baseline 3 | ResNet-only affordance (no topology) | Tests contribution of topology over CNN |
| Baseline 4 | GNN on keypoints (graph neural network on detected keypoints, then affordance decoder) | Tests learned topology vs. persistent homology |
| Ablation | TAEDO w/o temporal contrastive (static embedding) | Tests role of temporal alignment |

### Why this setup validates the claim

The chosen dataset includes tasks with varying degrees of deformation (e.g., flexible spatula for scooping, rigid hammer) to stress-test invariance claims. AffordanceNet and ResNet-only baselines isolate the impact of topological features: if topology is key, these should fail on deformable objects where appearance changes but affordance topology remains. SuperPoint tests whether learned keypoints can match topology's deformation robustness; if topology is superior, SuperPoint will struggle on non-rigid motion. The GNN keypoint baseline tests whether learning topological relations from keypoints can approximate persistence homology; if persistent homology is more effective, our method will outperform on tasks where keypoints are unreliable. The ablation (w/o contrastive) determines whether temporal alignment enforces invariance necessary for video-based affordance. Success rate as metric directly reflects task performance, grounding the claim in practical robot deployment. Additionally, we analyze part-level correspondence by computing correlation between persistence diagram features (birth-death pairs) and functional part labels (e.g., handle, scoop) across the dataset; high correlation (>0.7) would support the claim that topological features capture functional parts, providing groundedness evidence. Falsification would occur if our method shows uniform improvement across all conditions rather than concentrating on deformable or non-rigid subsets, indicating the improvement stems from unrelated factors (e.g., better architecture) rather than topological invariance.

### Expected outcome and causal chain

**vs. AffordanceNet** — On a flexible spatula scooping task where the tool bends during use, AffordanceNet produces scattered affordance predictions because its CNN learns brittle appearance patterns that change with deformation. Our method instead captures the persistent topological structure (e.g., the concavity of the scoop remains topologically connected despite shape change) due to homology-based embedding, so we expect a large success-rate gap (~30% higher) on deformable object trials, but parity on rigid objects where appearance suffices.

**vs. SuperPoint** — On a cloth folding task where keypoints drift and collapse under non-rigid stretching, SuperPoint-based affordance loses correspondence, leading to erroneous grasp points. Our topological embedding measures connected components (e.g., corners of cloth as separate components) which remain stable under in-plane stretch, yielding consistent affordance. We expect our method to outperform SuperPoint by ~25% on cloth tasks, but similar on rigid body tasks where keypoints are reliable.

**vs. ResNet-only** — On a hammering task with different hammer shapes (varying head size, handle length), ResNet-only affordance overfits to the training shape's visual features, failing on unseen hammers. Our method integrates topology (e.g., handle as one connected component) with CNN appearance, so it generalizes to shape variations by focusing on topological invariants. We expect our method to show ~20% higher success on unseen hammer variants, with comparable performance on seen shapes.

**vs. GNN on keypoints** — On a deformable scooping task where keypoints are sparse and noisy, the GNN baseline may learn topology but is limited by keypoint detection quality; our persistence-based embedding captures dense topology from the whole image patch, leading to more robust affordance. We expect our method to outperform GNN by ~15% on deformable tasks, with similar performance on rigid tasks where keypoints are accurate.

**vs. Ablation (w/o contrastive)** — On a scooping video with occluded tool part, the static embedding (no temporal alignment) misidentifies affordance due to viewpoint change, while our full method uses optical flow to align embeddings across frames, maintaining invariance. We expect a ~15% drop in success rate for the ablation on occluded sequences.

### What would falsify this idea

If our method shows nearly uniform improvement across all task subsets (rigid, deformable, non-rigid) rather than a pronounced advantage specifically on conditions where topological invariance is critical (deformable objects, non-rigid motion, unseen shape variants), then the central claim that topological embedding drives the gain is falsified.

## References

1. BridgeACT: Bridging Human Demonstrations to Robot Actions via Unified Tool-Target Affordances
2. TraceGen: World Modeling in 3D Trace Space Enables Learning from Cross-Embodiment Videos
3. UAD: Unsupervised Affordance Distillation for Generalization in Robotic Manipulation
4. MegaSaM: Accurate, Fast, and Robust Structure and Motion from Casual Dynamic Videos
5. Find any Part in 3D
6. RoboPoint: A Vision-Language Model for Spatial Affordance Prediction for Robotics
7. Metric3D: Towards Zero-shot Metric 3D Prediction from A Single Image
8. Deep geometry-aware camera self-calibration from video
