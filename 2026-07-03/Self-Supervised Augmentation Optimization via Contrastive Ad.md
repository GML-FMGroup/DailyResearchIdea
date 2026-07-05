# Self-Supervised Augmentation Optimization via Contrastive Adversarial Learning for Biometric Recognition

## Motivation

TeachAugment requires a pre-trained teacher network and labeled data, limiting its applicability in biometric scenarios where labels are scarce. Additionally, AGVBench reveals that mixing augmentations like MixUp improve accuracy but suffer from poor calibration and adversarial robustness. The root cause is that augmentation optimization guided by a supervised teacher does not enforce reliability properties, and the teacher itself needs labels. Therefore, a label-free augmentation optimization method that inherently promotes both discriminability and calibration is needed.

## Key Insight

The contrastive learning objective naturally incentivizes augmentations that maintain instance-level discriminability without supervision, and its induced feature uniformity on the hypersphere inherently improves calibration.

## Method

(A) **What it is**: Self-Supervised Augmentation Optimization (SSAO) is a bi-level adversarial framework that learns a neural augmentation generator (G) to produce augmentations that maximize the contrastive loss of a student encoder (E), while E is trained to minimize that same loss. Inputs are unlabeled biometric images; outputs are augmented images. The success of this approach relies on the assumption that the generator's maximization of the contrastive loss produces augmentations that improve downstream recognition accuracy and calibration; this assumption is explicitly tested in the experiment section.

(B) **How it works** (pseudocode):
```python
Initialize generator G (outputs mixing coefficients λ for StarMixup) and encoder E.
Hyperparameters: τ=0.5 temperature, k=3 mixing images, λ drawn from Dirichlet(α=0.2); E: ResNet-18.
for each epoch:
    Sample batch of N=256 unlabeled images x.
    For each image, G outputs λ_i (vector of mixing weights for k=3 random images).
    Generate augmented samples x_aug = StarMixup(x, λ)  # mixing with multiple images
    Compute contrastive loss L_cont = -log( exp(sim(z_i, z_i')/τ) / sum_{j!=i} exp(sim(z_i, z_j)/τ) ) where z_i = E(x_i), z_i' = E(x_aug_i), negatives are other samples in batch.
    Update E to minimize L_cont (gradient descent, lr=0.001, momentum=0.9).
    Update G to maximize L_cont (gradient ascent, lr=0.0001, with stop-gradient on E).
    # Regularization: clip mixing coefficients to [0,1] and apply entropy penalty to G outputs (coefficient 0.01) to prevent degenerate solutions.
```

(C) **Why this design**: We chose to augment using StarMixup rather than simple MixUp because AGVBench shows mixing methods benefit accuracy but StarMixup is tailored for vein patterns; however, this introduces more hyperparameters (k, α) that must be tuned. We opted for a single contrastive loss without auxiliary regularization because adding a supervised signal would require labels, defeating the self-supervised purpose; the cost is that G may generate overly difficult augmentations that collapse representation learning if not carefully regularized. We used gradient ascent on G instead of reinforcement learning for simplicity, but this requires careful learning rate scheduling to avoid instability. We chose to share the encoder E between the augmentation optimization and the final task (after training) to avoid separate networks, accepting that E may overfit to the adversarial augmentations. The load-bearing assumption is that the adversarial generator's maximization of the contrastive loss yields augmentations that improve downstream biometric recognition accuracy and calibration. This assumption is explicitly tested via the experiment's expected outcomes and falsification criteria.

(D) **Why it measures what we claim**: The contrastive loss L_cont measures instance-level discriminability because it forces the encoder to produce similar representations for augmented views of the same image and dissimilar for different images; this assumption fails when augmentations are so strong that views of the same image become too different (e.g., extreme mixing), in which case L_cont becomes dominated by false negatives and instead measures the inconsistency of generative augmentations. The generator's objective (maximize L_cont) measures the difficulty of preserving identity under augmentation because a higher L_cont indicates that augmentations are challenging the encoder's ability to recognize the same instance; this assumption fails if the generator learns to produce trivial augmentations that collapse variance (e.g., always output identical images), in which case the objective measures nothing. The calibration improvement claim is grounded in the uniformity property of contrastive learning (Wang & Isola, 2020) and is directly measured using Expected Calibration Error (ECE) in the experiment.

## Contribution

(1) A self-supervised augmentation optimization framework that eliminates the need for labeled data and pre-trained teachers by leveraging contrastive learning as the guiding objective. (2) Empirical evidence that the proposed method improves calibration and adversarial robustness compared to accuracy-only augmentation methods, addressing the reliability gap identified by AGVBench. (3) An analysis of the trade-off between augmentation difficulty and representation quality in self-supervised settings.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset 1 | PolyU Palm Vein | Standard benchmark for palm vein. |
| Dataset 2 | CASIA Iris (Iris Recognition) | Evaluates generalization to iris modality. |
| Dataset 3 | NIST Fingerprint (Fingerprint Verification) | Evaluates generalization to fingerprint modality. |
| Dataset 4 | LFW (Face Recognition) | Evaluates generalization to face modality. |
| Primary metric | Rank-1 accuracy | Measures identity discriminability directly. |
| Secondary metric | Expected Calibration Error (ECE) | Measures calibration quality (lower is better). |
| Baseline 1 | No augmentation | Measures impact of any augmentation. |
| Baseline 2 | Standard augmentations (random crop, rotation, flip) | Represents typical data augmentation. |
| Baseline 3 | StarMixup (fixed mixing ratio α=0.2) | Isolates benefit of learned mixing. |
| Ablation-of-ours | SSAO w/o adversarial (fixed mixing, no generator) | Shows contribution of adversarial optimization. |
| Hyperparameter sensitivity | Vary k in {1,2,3,5}, α in {0.1,0.2,0.5}, τ in {0.2,0.5,1.0} | Ensures robustness of results. |

### Why this setup validates the claim

The central claim is that SSAO improves discriminability and calibration by learning augmentations that challenge the encoder. Using Rank-1 accuracy on a clean test set directly measures discriminability across four diverse biometric modalities, testing generalizability. Including Expected Calibration Error (ECE) tests the claim that contrastive learning's uniformity property translates to better calibration—a key assumption. Comparing against no augmentation and standard augmentations shows baseline improvement. StarMixup (fixed) tests if learned mixing alone is sufficient. The ablation w/o adversarial isolates the effect of adversarial optimization. Hyperparameter sensitivity analysis ensures the method is not brittle. This combination ensures any improvement is attributable to learned augmentation difficulty and the calibration claim is directly testable: if ECE does not improve relative to baselines, the calibration claim is rejected.

### Expected outcome and causal chain

**vs. No augmentation** — On a case where intra-class variation is high (e.g., finger rotation), the baseline produces poor discrimination because no augmentation increases invariance. Our method instead generates mixed views that force the encoder to learn invariant features, so we expect a noticeable gap on high-variability subsets but parity on low-variability subsets. For calibration, we expect lower ECE on all subsets due to uniformity induced by contrastive learning.

**vs. Standard augmentations** — On a case where standard augmentations are suboptimal (e.g., they distort vein patterns), baseline produces inconsistent representations because they are not tailored to vein structure. Our method learns augmentations that preserve identity via contrastive loss, so we expect higher accuracy on subsets with vein-specific patterns but similar on generic ones. Calibration is expected to improve because our augmentations are identity-preserving, reducing overconfidence.

**vs. StarMixup (fixed)** — On a case where fixed mixing ratios are too aggressive for some samples, baseline causes collapse because same augmentation difficulty for all. Our method adapts mixing per sample, so we expect better performance on hard samples (low quality) but comparable on easy ones. ECE is expected to be lower due to adaptive difficulty.

### What would falsify this idea

If the improvement of SSAO over fixed StarMixup is uniform across all subsets rather than concentrated on hard samples, then the adversarial optimization is not targeting challenging instances as intended, falsifying the central claim. Additionally, if ECE does not improve over baselines, the calibration claim is falsified.

## References

1. AGVBench: A Reliability-Oriented Benchmark of Data Augmentation for Vein Recognition
2. StarMixup: A More Suitable Mixup Method for Palm-Vein Identification
