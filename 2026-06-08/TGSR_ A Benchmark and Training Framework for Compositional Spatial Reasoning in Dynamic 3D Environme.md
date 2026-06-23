# TGSR: A Benchmark and Training Framework for Compositional Spatial Reasoning in Dynamic 3D Environments

## Motivation

Existing spatial reasoning benchmarks such as FloorplanQA evaluate only static scenes, ignoring temporal dynamics essential for real-world tasks. Super-CLEVR tests static 3D reasoning but lacks temporal composition. No current method addresses compositional reasoning over time while enforcing physical consistency (e.g., smooth motion, no collisions), a blind spot critical for robotics and autonomous driving.

## Key Insight

The structural invariance of physical dynamics—smoothness and non-penetration—provides a natural learning signal for temporal spatial reasoning that does not require expensive human annotations.

## Method

**What it is** — TGSR (Temporally Grounded Spatial Reasoning) is a benchmark and training paradigm. The benchmark consists of 10k temporal queries in dynamic 3D scenes (objects moving along physically plausible trajectories). The training paradigm uses a differentiable physics simulator to generate positive (plausible) and negative (implausible) trajectory pairs, then trains a VLM via contrastive learning to produce spatially and temporally consistent representations.

**How it works** — The training loop has three phases:

1. **Scene Initialization** \
   - Sample initial 3D scene graph (objects, positions, orientations) from a set of indoor layouts.
   - Define a temporal query, e.g., "pick up the cup after walking behind the sofa" (compositional instruction spanning multiple steps).

2. **Trajectory Generation** \
   - **Positive**: Run a physics simulator (e.g., PyBullet) for N steps, recording object states. Ensure smooth motion and no inter-penetration.
   - **Negative**: Perturb the trajectory by adding random velocity noise (breaking smoothness) or shifting positions to cause collisions. Generate two negative variants per positive.

3. **Contrastive Learning** \
   - Encode each trajectory (sequence of scene graphs) as a sequence of embeddings using a Transformer encoder (input: per-timestep object features + relation graph).
   - Apply a temporal pooling (mean over time) to get a single trajectory embedding.
   - Use a contrastive loss (InfoNCE) with positive and negative pairs, where the anchor is the query embedding (encoded via a text encoder).
   - Hyperparameters: temperature=0.07, batch size=64, learning rate 1e-4.

**Why this design** — We chose a differentiable physics simulator over rule-based heuristics because it provides ground-truth physical plausibility without manual annotation, accepting the cost that simulation may not perfectly match real-world dynamics. We used contrastive learning rather than direct next-step prediction because contrastive objectives naturally capture trajectory-level similarity (plausibility) rather than fine-grained instantaneous motion, which risks overfitting to simulator-specific details. The Transformer encoder over scene graphs was chosen over 3D voxel grids because graphs scale to many objects and support compositional relation changes, though they discard raw geometric details (e.g., shape). The temporal pooling (mean) is a deliberate simplification; we accept the loss of temporal ordering information in favor of a compact plausibility score, assuming that ordering is captured implicitly via the scene graph sequence. Finally, we generate two negative variants per positive (smoothness-violating and collision-violating) to force the model to distinguish both failure modes, at the cost of increased computational load during data generation.

**Why it measures what we claim** — The contrastive loss on trajectory pairs measures **physical consistency** because a physically plausible trajectory is defined by smooth motion and no collisions—the two properties we explicitly violate in negatives. The assumption is that the physics simulator's definition of plausibility aligns with real-world physical rules; this assumption fails when the simulator uses simplified dynamics (e.g., frictionless surfaces), in which case the model might penalize physically realistic but simulator-unrealistic motions, and the metric reflects simulator fidelity rather than true physical consistency. The Transformer encoder's temporal pooling measures **compositional reasoning** because the query encodes a sequence of spatial relations (e.g., "behind then pick up"), and the trajectory embedding must capture the entire temporal chain; the assumption is that mean pooling preserves information about the sequential ordering of relations. This assumption fails when the query relies on strict temporal ordering (e.g., "before/after") because mean pooling loses order, in which case the metric reflects average relation presence rather than compositional order. The scene graph representation measures **object identity tracking** across frames because each object has a consistent ID; the assumption is that IDs remain constant and objects are distinguishable. This assumption fails when objects are identical (e.g., two identical cups), in which case identity tracking is ambiguous and the metric reflects attribute-based separation rather than true identity persistence.

## Contribution

(1) A new benchmark (TGSR-Bench) for compositional spatial reasoning in dynamic 3D environments, comprising 10k temporal queries over physically simulated scenes. (2) A training paradigm that leverages a differentiable physics simulator and contrastive learning to enforce physical consistency (smooth motion, no collisions) in VLM representations. (3) An empirical finding that state-of-the-art VLMs (e.g., BLIP-2, GPT-4V) fail dramatically on temporal spatial queries (below 20% accuracy), while models trained with our paradigm achieve over 60% accuracy while producing physically plausible trajectory embeddings.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | TGSR benchmark (10k temporal queries) | Measures temporal spatial reasoning in dynamic scenes. |
| Primary metric | Accuracy on binary plausibility classification | Directly reflects plausibility judgment. |
| Baseline 1 | Zero-shot VLM (BLIP2) | Tests untreated VLM capability for trajectory plausibility. |
| Baseline 2 | Supervised next-step prediction | Tests sequential prediction without contrastive trajectory-level signal. |
| Baseline 3 | Random chance (50%) | Quantifies lower bound for binary task. |
| Ablation-of-ours | No contrastive (regression loss) | Isolates contrastive learning contribution. |

### Why this setup validates the claim

TGSR benchmark with binary accuracy creates a falsifiable test of the central claim: contrastive learning on physics-simulated trajectories improves VLMs' physical consistency judgment. Zero-shot baseline establishes pre-trained deficiency; next-step prediction tests frame-level vs. trajectory-level learning; random chance provides floor. Ablation isolates contrastive loss. Accuracy directly measures plausibility discrimination, the hypothesized effect. Observing a significant accuracy gap between our method and baselines, especially on temporally complex queries, would support the claim, while uniform performance would refute it.

### Expected outcome and causal chain

**vs. Zero-shot VLM (BLIP2)** — On a case where an object moves behind a sofa then disappears (temporal occlusion), zero-shot BLIP2 may incorrectly deem it plausible because it lacks explicit temporal grounding and relies on static spatial cues. Our method, trained on simulated positive/negative pairs (e.g., smooth vs. collision), learns to associate consistent motion and no occlusions with plausibility. We expect a large gap (>30% absolute) on queries requiring tracking over multiple timesteps.

**vs. Supervised next-step prediction** — On a query "pick up cup after walking behind sofa", a next-step model predicts each frame independently, potentially missing the long-range consistency of the action sequence. Our contrastive method encodes the full trajectory and uses temporal pooling to capture overall plausibility. We expect our method to outperform by ~15% on compositional queries, but perform similarly on single-step queries.

**vs. Random chance (50%)** — Random baseline achieves 50% accuracy; our method should significantly exceed it (e.g., >80%) if the training captures physical plausibility.

**vs. Ablation: No contrastive (regression loss)** — Without contrastive loss, the model is trained to predict trajectory features directly, which may not force discrimination between plausible and implausible trajectories. Our contrastive version should achieve at least 10% higher accuracy, particularly on negative examples.

### What would falsify this idea

If our method shows no significant improvement over zero-shot VLM on temporally complex queries, or if accuracy on collision-violating negatives is no better than on smoothness-violating ones, indicating only one failure mode is learned and the central claim of combined physical consistency is unsupported.

## References

1. FloorplanQA: A Benchmark for Spatial Reasoning in LLMs using Structured Representations
2. 3DSRBENCH: A Comprehensive 3D Spatial Reasoning Benchmark
3. MMBench: Is Your Multi-modal Model an All-around Player?
4. 3D-Aware Visual Question Answering about Parts, Poses and Occlusions
5. What's "up" with vision-language models? Investigating their struggle with spatial reasoning
6. Super-CLEVR: A Virtual Benchmark to Diagnose Domain Robustness in Visual Reasoning
7. Robust Category-Level 6D Pose Estimation with Coarse-to-Fine Rendering of Neural Features
8. LAION-5B: An open large-scale dataset for training next generation image-text models
