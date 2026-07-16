# Noise-Augmented Consistency Training for Robust Document Parsing

## Motivation

Existing end-to-end document parsing models like OvisOCR2 are trained on synthetic HTML-generated images that lack realistic noise (e.g., blur, skew, occlusions), creating a domain gap that causes performance degradation on real-world documents. The root cause is that training data is perfectly clean while test data is noisy, and no mechanism forces the model's predictions to be invariant to these task-irrelevant perturbations.

## Key Insight

Enforcing output-level consistency between clean synthetic samples and their noise-augmented counterparts directly aligns the model's predictions to be invariant to task-irrelevant noise, leveraging the fact that the correct Markdown output is invariant under realistic document corruptions that preserve semantic content.

## Method

**Noise-Augmented Consistency Training (NACT)**

(A) **What it is:** NACT is a training framework for end-to-end document parsing models. It trains a base vision-language transformer (e.g., OvisOCR2) on a mix of clean synthetic images and their noisy counterparts produced by a learned noise model. The key additional loss is a consistency regularization term that penalizes divergence between the model's output on clean and noisy versions of the same page.

(B) **How it works:**

```python
# Training loop for NACT
# Hyperparameters: lambda_cons=0.5, momentum=0.999, noise_prob=0.8
# Pre-trained noise model N (conditional GAN) on a small set of real-world noisy documents
# Clean synthetic dataset D_clean (images + ground truth Markdown)

model = OvisOCR2Backbone()  # base model
ema_model = EMA(model, decay=momentum)  # teacher for consistency targets

for batch in dataloader(D_clean):
    x_clean, y_gt = batch

    # Step 1: Generate noisy version (with probability noise_prob)
    x_noisy = apply_noise(x_clean, noise_model=N) if random() < noise_prob else x_clean

    # Step 2: Forward passes (no gradient for ema_model)
    logits_clean = model(x_clean)
    logits_noisy = model(x_noisy)
    with torch.no_grad():
        logits_teacher = ema_model(x_clean)

    # Step 3: Supervised loss on clean image (ground truth is perfect from synthetic)
    L_sup = cross_entropy(logits_clean, y_gt)

    # Step 4: Consistency loss (KL divergence between teacher on clean and model on noisy)
    probs_teacher = softmax(logits_teacher, dim=-1)
    for each token position t:
        p_t = probs_teacher[t]
        q_t = softmax(logits_noisy[t])
    L_cons = KL_divergence(p_t, q_t)  # averaged over positions

    # Step 5: Total loss
    L = L_sup + lambda_cons * L_cons

    # Backward and update
    optimizer.zero_grad()
    L.backward()
    optimizer.step()
    update_ema(ema_model, model, momentum)
```

(C) **Why this design:** We chose a learned noise model over hand-crafted augmentations because real-world noise distributions (e.g., camera blur, skewed scans) are complex and unlikely to be fully covered by manual parameter ranges; the cost is requiring a small dataset of real-world noisy documents to train the generator. We used an exponential moving average (EMA) teacher to provide consistency targets rather than using the clean output from the same model because the EMA teacher yields more stable and accurate targets, preventing the model from collapsing to a trivial solution (e.g., always predicting the same output); the trade-off is increased memory but not compute. We applied the consistency loss at the sequence level (KL on logits per token) rather than at the feature level because sequence-level invariance directly enforces that the final parse remains unchanged under noise, while feature-level invariance might suppress discriminative features needed for parsing; the cost is that KL divergence may be sensitive to calibration, but we mitigate this via the EMA teacher. Finally, we used a probability of 0.8 for applying noise rather than always applying it, because occasional clean samples prevent overfitting to the noise model and maintain performance on clean data; the cost is slightly slower robustness acquisition.

(D) **Why it measures what we claim:** The consistency loss `L_cons` measures the KL divergence between teacher predictions on clean input and model predictions on noisy input. This operationalizes *noise invariance* because the assumption is that the correct Markdown output is identical for both clean and noisy versions; minimizing `L_cons` forces the model to produce the same output distribution under noise, thereby learning to ignore task-irrelevant perturbations. This assumption fails when noise alters the visual information needed to generate the correct parse (e.g., an occluded character), in which case invariance would be harmful; however, our noise model is trained to generate corruptions that preserve semantic content (e.g., blur, skew, lighting), so the assumption holds for the targeted noise types. The supervised loss `L_sup` measures prediction accuracy on clean images, and it ensures that the consistency targets are correct (since the teacher is also trained with this loss); without `L_sup`, the teacher might provide incorrect targets, leading to a degenerate invariance.

## Contribution

(1) A noise-augmented consistency training framework (NACT) for end-to-end document parsing that combines a learned real-world noise model with output-level consistency regularization. (2) A design principle: using a noise model trained on a small set of real-world noisy documents to generate plausible corruptions that preserve semantic content, enabling effective robustness training. (3) A systematic analysis showing that output-level consistency is more effective than feature-level or input-level regularization for document parsing tasks.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | OmniDocBench | Diverse real & synthetic noisy docs |
| Primary metric | Markdown F1 score | Direct parsing accuracy measure |
| Baseline: OvisOCR2 | Base model without NACT | Quantifies improvement over backbone |
| Baseline: Pipeline | Tesseract + LayoutLM | Standard multi-stage competitor |
| Baseline: Donut | Prior end-to-end LVLM | Tests against noise-agnostic method |
| Ablation: NACT w/o consistency | Supervised on clean+noisy only | Isolates consistency term effect |

### Why this setup validates the claim
This combination forms a falsifiable test of noise invariance: OmniDocBench contains both clean and naturally noisy pages (blur, skew, lighting), letting us isolate the consistency mechanism's effect. The primary metric (Markdown F1) directly captures the model's ability to produce accurate, robust parses. Baselines cover two key failure modes: OvisOCR2 shows the base model's vulnerability to noise; Pipeline reveals weaknesses of multi-stage systems on complex layouts; Donut represents prior end-to-end methods lacking noise-specific training. The ablation (removing consistency loss) tests whether the claimed invariant is due to the consistency term or simply joint training on noisy data. If the method's gain concentrates on noisy subsets while clean performance stays high, the central causal claim is supported; uniform gain or clean degradation would refute it.

### Expected outcome and causal chain

**vs. OvisOCR2** — On a heavily blurred receipt, the base model misreads digits because it relies on sharp edges for character recognition. NACT learns to ignore blur via the consistency loss, which forces the output distribution to match the teacher's clean prediction. We expect at least a 5% F1 gap on the blurred subset of OmniDocBench, but parity on clean pages.

**vs. Pipeline** — On a multi-column page with extreme skew, Tesseract's layout analysis misidentifies columns, causing text from different columns to mix. NACT processes the entire page in one pass without explicit layout analysis, and its consistency training makes it robust to skew. We expect NACT to outperform pipeline by >10% on skewed and complex-layout examples.

**vs. Donut** — On a low-light scanned document, Donut mistakes dark regions as blank or mispredicts characters, as its training on mostly clean images ignores illumination variations. NACT's consistency against a learned noise model that includes lighting changes forces invariance. We expect NACT to achieve higher character-level recall on the low-light subset, with a gap of over 3% F1.

### What would falsify this idea
If NACT shows uniform improvement across all test subsets (clean, blurred, skewed, low-light) rather than a concentrated gain on noise-specific splits, or if the ablation without consistency matches NACT's performance, then the central invariance claim is invalid.

## References

1. OvisOCR2 Technical Report
2. Logics-Parsing Technical Report
3. dots.ocr: Multilingual Document Layout Parsing in a Single Vision-Language Model
4. Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution
5. Qwen Technical Report
6. Nougat: Neural Optical Understanding for Academic Documents
7. DocXChain: A Powerful Open-Source Toolchain for Document Parsing and Beyond
8. Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond
