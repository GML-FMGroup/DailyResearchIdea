# CurriculaCompose: Curriculum Learning over Spatial Compositions for Photography Guidance MLLMs

## Motivation

ShutterMuse's training data inherits a narrow distribution of spatial compositions from Qwen-VL, limiting its generalization to novel photographic scenes. Existing methods use static datasets or random augmentations, but they do not systematically explore the space of possible crop positions, subject sizes, and overlaps. This results in a persistent spatial coverage gap that causes the model to fail on compositions outside its training distribution.

## Key Insight

Spatial compositions can be parametrically generated and their complexity quantified by a closed-form function of geometric salience and clutter, enabling a curriculum that provably expands coverage beyond the original data distribution.

## Method

### (A) What it is
We propose CurriculaCompose, a curriculum learning framework that dynamically samples training examples from a parametric generative model of spatial compositions, ordered by a closed-form complexity score, to improve the spatial coverage of an MLLM for capture-time photography guidance.

### (B) How it works
```pseudocode
Input: Base MLLM (e.g., ShutterMuse), original dataset D_orig, generative model G (sampled from parameter space of spatial compositions), complexity function C(example), curriculum schedule S(step) -> threshold.
Hyperparameters: initial threshold T0=0.2, step size delta=0.01, sampling ratio alpha=0.5.

1. Initialize threshold T = T0.
2. For each training step:
   a. Sample a batch from D_orig with probability alpha, from G with probability 1-alpha.
   b. For each example from G, compute complexity c = C(example) (see below).
   c. Accept the example if c <= T; otherwise reject and resample.
   d. Update threshold T = T + delta * (number of accepted examples / batch size) to gradually increase complexity.
3. Train the MLLM using the mixed batch (standard supervised fine-tuning loss).
```
Complexity function C(example) = w1*d_subject_center + w2*v_subject_area + w3*o_overlap + w4*a_crop_type, with weights w1=0.4, w2=0.3, w3=0.2, w4=0.1.
- d_subject_center: distance from subject center to nearest rule-of-thirds intersection (normalized by image diagonal).
- v_subject_area: variance of subject bounding box area relative to image area (smaller area → higher complexity).
- o_overlap: proportion of subject area overlapped by other objects (0 to 1).
- a_crop_type: discrete value (tight=2, medium=1, loose=0).

Generative model G samples compositions: subject position from a Gaussian centered at image center (σ increasing with training step), subject size from uniform distribution over a range that expands as curriculum progresses, background from a pool of outdoor scene images, and other objects (e.g., trees, buildings) placed with controlled overlap distribution.

### (C) Why this design
We chose a closed-form complexity function over a learned difficulty predictor because it is fully interpretable and requires no additional training data, avoiding the risk of learning spurious correlations from the limited original dataset; the trade-off is that the function may not perfectly align with all human notions of difficulty. We used a parametric generative model rather than random augmentations of existing images because it allows explicit control over the diversity of spatial features (position, size, overlap) and enables sampling in regions of the composition space that are underrepresented in D_orig; the cost is that generated compositions may appear less photorealistic, but we mitigate this by blending with real backgrounds. We adopted a linear schedule for complexity threshold instead of an adaptive one based on model performance because performance-based thresholds require evaluation on a held-out set and complicate training dynamics; the linear schedule is simple and ensures monotonic increase. Finally, we mix original and synthetic data with a fixed ratio alpha=0.5 to retain the real distribution, trading off reduced reinforcement from real data for broader coverage.

### (D) Why it measures what we claim
The computational quantity d_subject_center measures geometric salience because it quantifies how far the subject is from standard aesthetically pleasing positions; this assumes that compositions following the rule-of-thirds are simpler for the model to judge, but it fails when a subject placed at a rule-of-thirds intersection is part of a complex scene with many conflicting elements, in which case the metric underestimates complexity. v_subject_area measures subject prominence difficulty because smaller subjects are harder to localize and compose; this assumes that subject size is independent of other factors, but it fails when a small subject has high contrast against the background, making it easier than expected. o_overlap directly measures clutter, assuming that higher overlap requires the model to disentangle boundaries; it fails when overlapping objects are semantically related (e.g., person holding a cup) where overlap is expected, so the metric overestimates difficulty. a_crop_type encodes crop tightness, assuming tighter crops require more precise framing; this ignores that tight crops can simplify composition by removing background distractions. By weighting these four components, the aggregate complexity score covers the identified axes of spatial difficulty that the original dataset lacks coverage in, and the curriculum's monotonic increase ensures that the model encounters progressively harder examples, building generalization incrementally.

## Contribution

(1) A parametric generative model for spatial compositions in photography that explicitly controls factors such as subject position, size, overlap, and crop type. (2) A closed-form complexity metric based on geometric salience and clutter, enabling principled curriculum ordering without additional training. (3) A curriculum learning framework integrated with the ShutterMuse training pipeline that dynamically mixes generated and original examples to improve spatial coverage.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | CaptureGuide-Bench | Standard benchmark for photography guidance |
| Primary metric | Guidance accuracy | Measures correctness of composition suggestions |
| Baseline: GPT-4V | General-purpose MLLM | Tests necessity of domain-specific training |
| Baseline: GAIC | Specialized cropping model | Tests need for generative coverage of composition space |
| Baseline: ShutterMuse | Original model without curriculum | Baseline to beat (same architecture, no curriculum) |
| Ablation: No curriculum | Random order from generative model | Isolates effect of complexity ordering |

### Why this setup validates the claim

This design tests the central claim—that a curriculum over spatially diverse generative examples improves MLLM guidance—via controlled comparisons. CaptureGuide-Bench provides a realistic, varied set of capture-time scenarios. Comparing against GPT-4V isolates the benefit of domain-specific training; GAIC tests whether generative coverage adds value over a specialized cropping model; baseline ShutterMuse quantifies improvement from the curriculum alone; and the ablation disentangles the effect of complexity ordering from mere exposure to synthetic data. The primary metric, guidance accuracy, directly captures the quality of spatial composition advice. If the claimed mechanism works, our method should outperform all baselines on cases demanding tricky spatial reasoning, with the ablation confirming that ordering matters. Conversely, failure to outperform on those cases would falsify the idea.

### Expected outcome and causal chain

**vs. GPT-4V** — On a case where a small subject is placed near the edge with heavy background clutter, GPT-4V may produce generic advice like "move subject to center" because its training lacks nuanced composition constraints. Our CurriculaCompose, having been trained on progressively complex compositions (e.g., tiny subjects, high overlap), will suggest precise adjustments (e.g., "move subject to upper-right third and decrease background brightness") thanks to the ordered exposure to such patterns. We expect a noticeable gap on difficult composition subsets (e.g., small subjects, high overlap) but near-parity on simple, central compositions.

**vs. GAIC** — Consider a scene where the subject is a small bird on a branch with flowers. GAIC, as a cropping model, will output a crop framing the bird tightly, removing the context. Our method, designed for capture-time guidance, can suggest repositioning the camera to include both subject and context, producing a more aesthetically pleasing composition. Because GAIC is limited to cropping, it cannot provide such holistic advice. We expect our method to score significantly higher on open-ended guidance tasks, while GAIC may match on tasks reducible to cropping.

**vs. ShutterMuse (original)** — Imagine an unusual composition where the subject is placed at a non-standard intersection (e.g., one third from bottom and one third from left). The original ShutterMuse, trained on a dataset lacking such examples, may give inconsistent advice or fail to recognize the composition. Our CurriculaCompose explicitly exposes the model to such cases via generative sampling with increasing complexity, leading to accurate and consistent guidance. We expect a clear improvement on the long-tail of composition types underrepresented in the original data, while performance on common compositions remains similar.

### What would falsify this idea

If the performance gain over ShutterMuse is uniform across all composition subsets (i.e., no concentration on rare or complex spatial layouts), then the curriculum is not addressing the claimed coverage deficiency, and the central hypothesis is wrong.

## References

1. ShutterMuse: Capture-Time Photography Guidance with MLLMs
2. Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond
3. Image as a Foreign Language: BEiT Pretraining for All Vision and Vision-Language Tasks
4. OFASys: A Multi-Modal Multi-Task Learning System for Building Generalist Models
5. PaLI: A Jointly-Scaled Multilingual Language-Image Model
6. LAION-5B: An open large-scale dataset for training next generation image-text models
