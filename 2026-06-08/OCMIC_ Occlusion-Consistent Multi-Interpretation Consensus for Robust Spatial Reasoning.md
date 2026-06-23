# OCMIC: Occlusion-Consistent Multi-Interpretation Consensus for Robust Spatial Reasoning

## Motivation

Current VLMs exhibit brittle spatial reasoning under perceptual noise, as shown by the low accuracy on the SpatiaLQA benchmark even with recursive scene graph assistance. The root cause is that single-pass reasoning cannot detect when a noisy scene graph (e.g., from object detector errors or occlusion) leads to an unreliable prediction. Existing methods lack a verifiability signal to reject such inputs, leaving a residual gap in robustness to perceptual noise and uncertainty handling.

## Key Insight

Occlusion introduces ambiguity that is inherently irresolvable from a single view; requiring unanimous predictions across multiple occlusion-consistent interpretations exploits this ambiguity to flag cases where the visual input is insufficient for a confident spatial relation judgment.

## Method

We propose **OCMIC** (Occlusion-Consistent Multi-Interpretation Consensus), a method that takes an image, a spatial relation query, and a VLM, and outputs either a unanimous prediction or an *uncertain* signal.

### (A) What it is
OCMIC generates multiple text-based scene descriptions that are plausible under occlusion by perturbing an initial scene graph in occlusion-consistent ways, then queries the VLM on each description and checks for consensus.

### (B) How it works
```python
import random
import numpy as np

def OCMIC(image, query, VLM, num_interpretations=5):
    # Step 1: Extract initial scene graph and depth map
    scene_graph = extract_scene_graph(image)  # object detection + relation prediction (e.g., using GLIP + RELTR)
    depth_map = estimate_depth(image)         # monocular depth estimator (e.g., MiDaS v2)
    
    # Step 2: Generate occlusion-consistent perturbations
    interpretations = []
    for _ in range(num_interpretations):
        sg = scene_graph.copy()
        # Perturbation A: Remove objects likely occluded (depth-based removal probability)
        for obj in sg.objects:
            # Depth-normalized distance: objects farther have higher removal probability
            depth = depth_map[obj.bbox.center]
            removal_prob = sigmoid(depth - threshold)  # hyperpar: thresh=0.5 (tuned on 100 OSQA valid images)
            if random.random() < removal_prob:
                sg.remove_object(obj)
        # Perturbation B: Perturb object positions (spatial noise)
        for obj in sg.objects:
            obj.position += np.random.normal(0, 0.05 * image_size_bounds)
        # Perturbation C: Regenerate relations based on new positions (using a simple geometry-based relation predictor)
        sg.update_relations()
        # Generate text description
        desc = sg.to_text()  # e.g., "A cup on a table. A book left of the cup."
        interpretations.append(desc)
    
    # Step 3: Query VLM on each interpretation
    answers = []
    for desc in interpretations:
        prompt = f"Scene: {desc}\nQuestion: {query}\nAnswer yes or no."
        answer = VLM(prompt, temperature=0)  # greedy
        answers.append(answer)
    
    # Step 4: Consensus check
    if len(set(answers)) == 1:
        return answers[0]
    else:
        return "uncertain"
```

**Explicit assumption (load-bearing):** Depth-based object removal probability accurately reflects occlusion likelihood. We calibrate the threshold on a validation set of 100 images from OSQA with ground-truth occlusion annotations (or human-annotated plausible occlusions) by maximizing the agreement between perturbed scenes and human judgments of plausible alternatives.

### (C) Why this design
We chose **depth-based removal** over random removal because depth is a direct cue for occlusion; this ensures perturbations are physically plausible (occluded objects are more likely to be invisible). The cost is that we depend on a depth estimator, which may itself be noisy. We chose **position perturbation within a small Gaussian** rather than large shifts because exaggerated shifts create scenes that are not occlusion-consistent (objects floating far away), which would break the assumption of plausible alternatives. This limits the coverage to depth-based uncertainty only. We chose **text-based scene descriptions** instead of modifying the actual image (e.g., inpainting) because it is computationally cheaper and avoids artifacts from image generation; however, it tests the VLM's language grounding rather than its visual grounding directly, which may underestimate its visual robustness. The **consensus check** is a simple all-or-nothing rule; we rejected weighted voting because it conflates confidence with frequency, whereas unanimity strictly enforces invariance.

### (D) Why it measures what we claim
The **depth-based removal probability** measures the *plausibility of occlusion* because an object's depth relative to a threshold is a proxy for how likely it is occluded by other objects; this assumption fails when the depth estimator is inaccurate (e.g., for transparent objects), in which case removal probabilities become arbitrary. The **position perturbation** measures *spatial uncertainty* because small noise captures typical errors in detection bounding boxes; this assumption fails when the VLM is highly sensitive to minor positional changes (e.g., for tight spatial relations like "touching"), in which case the method may over-flag uncertainty. The **unanimity of answers across interpretations** measures *prediction reliability* because invariance under plausible scene variations implies the prediction is grounded in features robust to occlusion; this assumption fails when the perturbations do not cover all relevant uncertainties (e.g., lighting or viewpoint), in which case a unanimous answer may still be unreliable. We explicitly assume that perturbations span all occlusion-induced ambiguities; failure occurs if the perturbation set is incomplete (e.g., missing lighting variations).

## Contribution

(1) A novel method, OCMIC, that generates occlusion-consistent scene graph perturbations and uses consensus to detect unreliable spatial relation predictions, providing a verifiability signal without requiring additional training. (2) A design principle: depth-based removal and small position perturbations are sufficient to capture the dominant sources of perceptual uncertainty in spatial reasoning, as opposed to random or adversarial perturbations. (3) An analysis of the assumptions underlying the operationalization of 'reliability' via consensus, including failure modes.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | OccludedSpatialQA (OSQA) | Focuses on occlusion-induced spatial ambiguity. |
| Primary metric | Unanimous Accuracy (UA) | Correctness when model is certain. |
| Baseline 1 | Direct VLM (e.g., GPT-4V) | Tests baseline without scene decomposition. |
| Baseline 2 | SG Single Description | Uses initial scene graph; no perturbation. |
| Baseline 3 | Random Guessing | Lower bound for chance performance. |
| Baseline 4 | MC Dropout (5 stochastic forward passes) | Tests uncertainty via stochastic inference. |
| Baseline 5 | Deep Ensemble (5 models) | Tests uncertainty via model disagreement. |
| Ablation-of-ours | OCMIC without depth removal | Tests importance of depth-based occlusion modeling. |
| Ablation-of-ours | OCMIC with varied depth threshold (0.3, 0.4, 0.5, 0.6, 0.7) | Tests sensitivity to threshold. |

Additionally, we evaluate on CLEVR (compositional reasoning) and GQA (realistic scenes) to test generality to non-occlusion challenges.

### Why this setup validates the claim

OSQA contains images where objects are partially occluded, requiring reasoning about plausible alternatives. Unanimous Accuracy (UA) captures the method's reliability when it commits to an answer—high UA on OSQA would indicate that the consensus mechanism successfully filters out noise. Direct VLM baseline tests the claim that explicit scene decomposition is necessary; if OCMIC outperforms it on occluded queries, it confirms that occlusion-consistent perturbations add value. SG Single baseline tests the perturbation component; if OCMIC outperforms it, depth-based removal and position noise are effective. Random guessing provides a sanity check. Ablation isolates depth removal, showing its crucial role. MC Dropout and Deep Ensemble baselines test whether OCMIC's uncertainty mechanism is distinct from standard uncertainty methods. Together, these comparisons form a falsifiable test: if OCMIC's advantage is largest on occluded subsets where depth cues matter, the central claim is supported.

### Expected outcome and causal chain

**vs. Direct VLM** — On a case where a cup is partially occluded by a mug, the VLM might answer "left" because it sees the mug's visible portion and ignores the cup's hidden part. Direct VLM lacks explicit occlusion handling, so it treats visible evidence as complete. Our method generates multiple scene descriptions where the cup may be removed (depth-based) or shifted, leading to varied answers; if all variations still indicate "right", the prediction is robust. We expect a noticeable gap on OSQA (e.g., OCMIC UA ~80% vs. Direct VLM accuracy ~55%) especially on high-occlusion queries.

**vs. SG Single Description** — On an image where a book is behind a box, the initial scene graph might incorrectly place the book left of the box due to bounding box errors. Without perturbation, the single description yields a wrong answer. Our method perturbs positions, causing the relation to flip in some interpretations, breaking consensus and triggering uncertainty. Thus, OCMIC abstains on this query, avoiding error, while SG Single commits and fails. We expect OCMIC's UA to be higher (e.g., 80% vs. 65%) but with an abstention rate of ~20% on difficult cases.

**vs. Random Guessing** — Random guessing yields 50% accuracy on binary spatial questions. OCMIC's UA should significantly exceed this (e.g., >70%), otherwise the method adds no value. If OCMIC's UA is near 50%, the consensus mechanism is effectively random — contradicting the claim.

**vs. MC Dropout** — MC Dropout samples stochastic forward passes with dropout enabled; it measures epistemic uncertainty but does not generate occlusion-consistent perturbations. On OSQA, OCMIC should achieve higher UA because its perturbations are tailored to occlusion scenarios. We expect OCMIC UA ~80% vs. MC Dropout UA ~70%.

**vs. Deep Ensemble** — Deep Ensemble averages predictions from 5 independently trained models; it captures model uncertainty but ignores structural occlusion cues. OCMIC is expected to outperform on highly occluded subsets (UA gap ~10%).

### What would falsify this idea

If OCMIC's advantage over Direct VLM is uniform across all OSQA queries (including non-occluded ones) rather than concentrated on high-occlusion instances, then the depth-based perturbation is not targeting the intended failure mode, suggesting the method's success stems from a confounding factor (e.g., text preprocessing) rather than occlusion reasoning. Additionally, if OCMIC fails to outperform MC Dropout or Deep Ensemble on OSQA, then the occlusion-consistent perturbation mechanism is not uniquely beneficial relative to general uncertainty methods.

## References

1. SpatiaLQA: A Benchmark for Evaluating Spatial Logical Reasoning in Vision-Language Models
2. FloorplanQA: A Benchmark for Spatial Reasoning in LLMs using Structured Representations
3. 3DSRBENCH: A Comprehensive 3D Spatial Reasoning Benchmark
4. MMBench: Is Your Multi-modal Model an All-around Player?
5. 3D-Aware Visual Question Answering about Parts, Poses and Occlusions
6. What's "up" with vision-language models? Investigating their struggle with spatial reasoning
7. Super-CLEVR: A Virtual Benchmark to Diagnose Domain Robustness in Visual Reasoning
8. Robust Category-Level 6D Pose Estimation with Coarse-to-Fine Rendering of Neural Features
