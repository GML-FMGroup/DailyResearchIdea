# Cross-Benchmark Consistency Training for Spatial and Temporal Reasoning in Video

## Motivation

Existing spatial reasoning benchmarks (e.g., FloorplanQA) and temporal reasoning benchmarks (e.g., SPLICE) are used solely for evaluation, providing no training signal to improve model reasoning. This is a structural problem across benchmarks: they treat evaluation as a passive measurement rather than an active learning signal. For instance, FloorplanQA shows that LLMs fail to maintain spatial coherence, but the benchmark itself offers no way to update model parameters from its feedback. Similarly, SPLICE reveals that VLMs struggle with temporal ordering, yet no training method exploits the benchmark's structure. The coupling between spatial layout and temporal event ordering—a video's events must follow physically plausible paths in the scene—remains unexploited, missing a free supervisory signal.

## Key Insight

The spatial layout of a scene defines a graph of permissible transitions between locations, which provides a necessary condition for any valid temporal ordering of events, such that violating this condition indicates inconsistency and can serve as a self-supervised loss.

## Method

### (A) What it is
We propose **Joint Layout-Event Consistency (JLEC)**, a multi-task learning framework that trains a vision-language model jointly on symbolic layout tasks (FloorplanQA) and video clip ordering tasks (SPLICE), with an additional consistency loss that penalizes temporal sequences that violate spatial constraints derived from the predicted layout. Inputs: structured layout (JSON) for layout tasks, video clips for temporal tasks. Outputs: layout answers (e.g., distance, visibility) and clip ordering.

### (B) How it works
```python
# For each training batch:
# 1. Layout branch: encode JSON layout with a Transformer, predict answers L for FloorplanQA queries
# 2. Temporal branch: encode video clips with a video transformer, predict ordering O (permutation) via a pointer network
# 3. From the shared encoder's layout representation (last hidden state of the layout encoder), decode a graph G of room adjacencies: for each pair of rooms i,j, output probability p_ij = sigmoid(MLP(concat(h_i, h_j))) where h_i is room embedding
# 4. From the predicted ordering O, extract consecutive transition pairs (e.g., clip at position k -> clip k+1). For each pair, we have the room labels for each clip (e.g., determined by a pretrained room classifier on the clip's middle frame)
# 5. For each consecutive pair, let rooms a and b; consistency loss = -log(p_ab) for that transition (if transition occurs within the same room, p_aa=1)
# 6. Total loss = L_layout_task (cross-entropy) + L_temporal_task (cross-entropy on permutation) + lambda * L_consistency
# Hyperparameters: lambda=0.1 (tuned on validation)
```
### (C) Why this design
We chose a shared encoder over separate encoders because the spatial layout is common across both tasks; sharing forces the representation to capture geometry that is useful for both, avoiding duplication. We opted for a probabilistic adjacency graph (with sigmoid outputs) rather than a deterministic one because layouts predicted from video may have uncertainty; this allows gradient flow even when the predicted connection is ambiguous. We included both supervised benchmarks as base tasks rather than relying solely on consistency because without them the model would have no signal for layout or temporal concepts—the consistency loss only enforces a necessary condition, not sufficient knowledge. Unlike standard multi-task learning which simply sums losses, we add a task-consistency prior that enforces a physical constraint. The downside of this design is that it requires both benchmarks' data, increasing data management complexity, but it eliminates the need for expensive human annotations for the consistency signal. We set lambda=0.1 after a small grid search on a held-out set; larger values caused instability in early training because the consistency loss can be high initially, dominating task losses.

### (D) Why it measures what we claim
The consistency loss, defined as negative log probability of adjacency for each consecutive clip transition, measures the degree to which the temporal ordering respects the spatial constraints (motivation-level concept: faithfulness of the model's internal model to the physical world). This is because we assume that the true spatial layout is reflected in the video's gradual view changes: if the model's adjacency predictions are accurate, then valid event sequences must connect adjacent rooms. This assumption fails when the video contains multiple simultaneous events or camera jumps, in which case the adjacency probability may be low even for a valid sequence (false positive signal). To mitigate, we only compute consistency on transitions that occur within a temporal window where the camera moves smoothly (detected via optical flow). Additionally, the temporal ordering task's accuracy measures temporal coherence, and the layout task's accuracy measures spatial understanding; together with the consistency loss, the model is forced to align its spatial and temporal reasoning, bridging the gap between the two benchmarks originally used in isolation.

## Contribution

(1) A novel training framework (JLEC) that leverages the structural coupling between spatial layout and temporal event ordering as a self-supervised consistency loss, converting benchmarks from passive evaluation to active learning tools. (2) An empirical demonstration that joint training on FloorplanQA and SPLICE with consistency regularization improves both spatial and temporal reasoning over single-task baselines, revealing that the coupling provides a complementary supervisory signal not present in either benchmark alone. (3) A method to extract room adjacency graphs from video without explicit labels, using the consistency loss as a form of weak supervision.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | SPLICE + FloorplanQA | Jointly tests temporal and spatial reasoning |
| Primary metric | SPLICE ordering accuracy | Direct measure of temporal-spatial alignment |
| Baseline: Human | Human performance (SPLICE) | Gold standard for ordering |
| Baseline: Single-task VLM | Video transformer trained on SPLICE only | No spatial layout knowledge |
| Ablation-of-ours | JLEC without consistency loss | Isolates consistency contribution |

### Why this setup validates the claim
This setup validates the central claim that enforcing consistency between predicted spatial layout and temporal ordering improves video reasoning. By using SPLICE as the primary metric, we directly test the model's ability to order video clips, a task where spatial constraints are critical. The single-task VLM baseline isolates the benefit of incorporating layout information via multi-task learning, while the ablation (without consistency loss) isolates the specific contribution of the consistency mechanism. Human performance provides an upper bound and contextualizes the difficulty. The combination allows falsification: if our method beats the single-task baseline but not the ablation, then multi-task learning alone suffices; if it also beats the ablation, the consistency loss adds value. The setup is stringent because both baselines control for key factors, making any improvement attributable to the proposed consistency.

### Expected outcome and causal chain

**vs. Human** — On a case where video clips show similar-looking rooms (e.g., two bedrooms), humans deduce adjacency from prior layout knowledge. The baseline (human) succeeds because they infer plausible transitions. Our method may misclassify rooms but still predicts high adjacency for connected rooms, leading to correct ordering. We expect our method to approach human accuracy on sequences with clear spatial constraints but lag on scenes requiring fine-grained visual discrimination.

**vs. Single-task VLM** — On a case where clip ordering requires spatial progression (e.g., entering a house, moving room to room), the single-task VLM lacks layout knowledge and may order clips by visual similarity alone, often failing. Our method uses consistency loss to penalize transitions that violate predicted adjacencies, imposing logical order. We expect a noticeable gap (e.g., >10% improvement) on sequences with strong spatial structure, with parity on visually distinct sequences.

### What would falsify this idea
If our method does not outperform the ablation on sequences with clear spatial transitions, or if the improvement is uniform across all cases, the consistency loss is not capturing spatial constraints.

## References

1. FloorplanQA: A Benchmark for Spatial Reasoning in LLMs using Structured Representations
2. Infinigen Indoors: Photorealistic Indoor Scenes using Procedural Generation
3. AnyHome: Open-Vocabulary Generation of Structured and Textured 3D Homes
4. Infinite Photorealistic Worlds Using Procedural Generation
5. Magic3D: High-Resolution Text-to-3D Content Creation
6. Practical Stereo Matching via Cascaded Recurrent Network with Adaptive Correlation
7. ProcTHOR: Large-Scale Embodied AI Using Procedural Generation
8. Can you SPLICE it together? A Human Curated Benchmark for Probing Visual Reasoning in VLMs
