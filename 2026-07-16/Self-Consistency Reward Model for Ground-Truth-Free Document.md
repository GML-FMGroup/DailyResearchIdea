# Self-Consistency Reward Model for Ground-Truth-Free Document Parsing

## Motivation

Existing document parsing models, such as OvisOCR2, rely on ground-truth Markdown annotations for reinforcement learning, creating an annotation bottleneck that limits scalability. The frontier still uses supervised rewards that penalize deviation from ground truth. Verification actions (e.g., self-consistency checks) are underused but successful. A new verification mechanism—a learned reward without ground truth—could close this gap by leveraging the structural property that correct parses should be invariant to benign image perturbations.

## Key Insight

Parse consistency across input perturbations provides a reliable proxy for parsing quality because a correct parse is invariant to transformations that preserve document content.

## Method

## (A) What it is
We propose **SCORE** (Self-Consistency Oriented Reward Estimator), a learned reward model that estimates document parsing quality from an image and a predicted Markdown sequence, without requiring ground-truth Markdown. Inputs: image `I` and predicted Markdown `M`. Output: scalar quality score `r`.

## (B) How it works
```pseudocode
# Training SCORE on batches of real (no ground truth) images
for each image I in training set:
  # Step 1: Generate augmented views
  augmentations = [gaussian_blur(I, kernel=3), 
                   random_crop(I, scale=(0.9,1.0)), 
                   brightness_jitter(I, delta=0.1)]
  views = [I] + augmentations

  # Step 2: Parse each view with fixed parser P (e.g., OvisOCR2 checkpoint)
  parses = [P(view) for view in views]   # each parse is a Markdown string

  # Step 3: Compute pairwise consistency scores
  consistency_matrix = [[levenshtein(parses[i], parses[j]) for j in range(len(parses))] for i in range(len(parses))]
  # Normalize to [0,1]: higher means more consistent
  avg_consistency = 1.0 - (sum(consistency_matrix) / (len(parses)*(len(parses)-1)) * 10)  # arbitrary scaling

  # Step 4: Train SCORE (a small Transformer) to predict avg_consistency from (I, M_original)
  # M_original is parse of original (non-augmented) image
  loss = MSE(SCORE(I, M_original), avg_consistency)
  update SCORE parameters
```

## (C) Why this design
We chose **self-consistency across augmentations** over reconstruction-based rewards (e.g., rendering Markdown back to an image) because reconstruction requires a renderer that matches the original document style—an additional domain-specific engineering challenge. The trade-off: augmentations must be content-preserving; overly aggressive augmentations (e.g., heavy cropping) break the invariance assumption and produce noisy targets. We selected three mild augmentations—blur, slight crop, brightness—that are known to preserve text readability, trading off diversity for signal quality. We used **Levenshtein distance** rather than a learned semantic metric because it is interpretable and requires no extra training, though it ignores structural similarity (e.g., Table order errors are penalized equally with trivial typos). We train SCORE on the **original image and its parse** rather than on per-view pairs, because the reward will be used at inference on a single (image, parse) pair; this design choice makes SCORE directly applicable as a reward model for RL training of the parser. The cost is that SCORE must generalize from only the original view during RL, though consistency targets from augmented views provide a richer training signal.

## (D) Why it measures what we claim
The computed **avg_consistency** measures **parsing quality** because we assume that a correct parse is invariant to perturbations that do not alter the document's underlying content. Formally, *Levenshtein distance between parses of augmented views* measures *content fidelity* under the assumption that benign transformations do not change the ground-truth Markdown string; this assumption fails when an augmentation (e.g., crop) removes text, in which case consistency reflects missing-content mismatch, not parsing quality. The **SCORE** model operationalizes the concept of **parsing quality** by mapping (image, parse) to the expected consistency score. This mapping is trained via regression to the consistency target, and its validity depends on the assumption that consistency distribution across views is monotonic in parsing correctness; that assumption fails when multiple low-quality parses are coincidentally consistent (e.g., all miss the same symbol), in which case SCORE overestimates quality.

## Contribution

(1) A self-consistency-based reward learning framework (SCORE) that eliminates the need for ground-truth Markdown annotations in training document parsers via reinforcement learning. (2) A training protocol that bootstraps the reward model using only real document images and a fixed parser, then uses the learned reward to fine-tune on real data without any human annotation. (3) The insight that mild, content-preserving augmentations (blur, slight crop, brightness) provide a sufficient signal for consistency-based reward learning.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|----------|
| Dataset | OmniDocBench (overall score) | Standard benchmark for document parsing quality |
| Primary metric | OmniDocBench overall score | Directly measures end-to-end parsing accuracy |
| Baseline 1 | OvisOCR2 (no RL) | Baseline parser without reinforcement learning |
| Baseline 2 | Pipeline (OCR+layout analysis) | Multi-stage baseline with error propagation |
| Baseline 3 | Reconstruction-based reward RL | Alternative reward using rendered image reconstruction |
| Ablation-of-ours | SCORE w/o augmentations (single view) | Tests necessity of multi-view self-consistency |

### Why this setup validates the claim
This experimental design tests whether SCORE's self-consistency reward improves parser quality beyond strong baselines and plausible alternatives. By comparing against OvisOCR2 (base parser) and a pipeline method, we isolate the gain from RL fine-tuning and end-to-end learning. The reconstruction-based reward baseline tests whether a different reward design that requires a renderer yields similar gains; if SCORE outperforms it, the self-consistency approach is validated as simpler and more effective. The ablation (single view) tests whether the multi-view augmentation is necessary; if it fails, the core assumption that consistency across augmentations provides a useful signal is falsified. The chosen metric (OmniDocBench overall score) is holistic and measures final parsing quality, making it a direct test of the claim that SCORE leads to better parses.

### Expected outcome and causal chain

**vs. OvisOCR2** — On a document with mild blur or lighting variation, OvisOCR2 may misrecognize characters because it was trained on clean data and lacks robustness. Our method uses RL with SCORE, which rewards consistency across augmentations (including blur), so the parser learns to ignore such artifacts. We expect a noticeable gap (e.g., 2-3 points) on perturbed subsets but parity on clean documents.

**vs. Pipeline** — On a multi-column layout with nested images, the pipeline fails because OCR errors propagate to layout analysis. Our end-to-end parser with SCORE reward learns to maintain consistent structure across views, avoiding error cascades. We expect a significant gap (e.g., 5-10 points) on complex layouts, especially those with tables or multi-columns.

**vs. Reconstruction-based reward** — On a document with a non-standard font or color scheme, the reconstruction reward may be misaligned because the renderer cannot reproduce the exact style. SCORE's self-consistency is renderer-free and invariant to stylistic variation. We expect stable gains across diverse document styles, while reconstruction reward may degrade on stylistically unusual samples, leading to a smaller or negative gain.

### What would falsify this idea
If SCORE+RL shows uniform improvement across all subsets (clean and perturbed, simple and complex) instead of concentrated gains on subsets where augmentations test robustness or where pipeline error propagation is severe, then the central claim that SCORE learns to identify parsing failures via self-consistency is false.

## References

1. OvisOCR2 Technical Report
2. Logics-Parsing Technical Report
3. dots.ocr: Multilingual Document Layout Parsing in a Single Vision-Language Model
4. Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution
5. Qwen Technical Report
6. Nougat: Neural Optical Understanding for Academic Documents
7. DocXChain: A Powerful Open-Source Toolchain for Document Parsing and Beyond
8. Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond
