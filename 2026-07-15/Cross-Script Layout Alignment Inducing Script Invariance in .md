# Cross-Script Layout Alignment: Inducing Script Invariance in Visual Document Encoders

## Motivation

Current visual document encoders like MonkeyOCRv2 achieve strong performance on a limited set of languages (17) but rely solely on data scaling without any explicit inductive bias for script invariance. Consequently, their representations can be inconsistent across different scripts of the same document, degrading performance when encountering unseen scripts. This is a structural limitation: the pretraining objective (generation and reconstruction) does not force the encoder to discard script-specific visual cues, making generalization fragile.

## Key Insight

Document layout (headings, paragraphs, figures) is invariant to script, so enforcing consistency of layout predictions across script variants provides a natural supervisory signal that forces the encoder to learn script-invariant features.

## Method

### (A) What it is
We propose **Cross-Script Layout Alignment (CSLA)**, a training constraint for visual document encoders that enforces consistency of per-pixel layout predictions across synthetic script variants of the same document. The encoder outputs a dense feature map; a lightweight layout head (two 1×1 conv layers + softmax) predicts layout class probabilities per pixel. The input is a document image; output is a layout probability map.

### (B) How it works
```python
# Training loop for encoder E and layout head L
for each document D in batch:
    # Step 1: Generate K script variants (K=3: English, Arabic, Chinese)
    variants = generate_script_variants(D)  # translate text, render with script font, keep layout
    
    # Step 1a: Geometric validation to ensure layout preservation
    validated_variants = []
    for i, img in enumerate(variants):
        # Compute bounding box IoU between variant and original layout (text regions)
        # Use precomputed bounding boxes from synthetic generation
        iou = compute_mean_iou(D.bounding_boxes, img.bounding_boxes)  # threshold=0.95
        if iou >= 0.95:
            validated_variants.append(img)
        else:
            # Skip this variant; regenerate with adjusted font size to preserve layout
            img = regenerate_with_fit(D, script=i)  # scale font to fit original boxes
            validated_variants.append(img)  # after regeneration, iou meets threshold
    # Use validated_variants for loss computation
    
    # Step 2: Extract layout predictions for each validated variant
    layout_maps = []
    for img in validated_variants:
        f = E(img)               # feature map, spatial dims H/32 x W/32
        layout = L(f)            # shape: (H/32, W/32, C) where C=number of layout classes (e.g., 5: text, title, figure, table, background)
        layout_maps.append(layout)
    
    # Step 3: Compute pairwise KL divergence (soft targets)
    L_cons = 0.0
    count = 0
    for i in range(K):
        for j in range(i+1, K):
            # KL(layout_i || layout_j) = sum_c layout_i[c] * log(layout_i[c] / layout_j[c])
            L_cons += kl_div(layout_maps[i], layout_maps[j])  # per-pixel average
            count += 1
    L_cons /= count
    
    # Step 4: Combine with standard task loss (e.g., generation loss from MonkeyOCRv2)
    # task_loss is computed on one variant (e.g., English) as usual
    L_total = task_loss + lambda * L_cons   # lambda=0.1 by default
    update E, L via backprop
```
### (C) Why this design
We chose **KL divergence** over alternatives like cosine similarity or contrastive loss because KL divergence directly measures the difference in predicted distributions, which aligns with the goal of making layout predictions identical across scripts (zero KL when invariant). We generate **synthetic script variants** via machine translation and rendering rather than using real multilingual documents because synthetic data ensures identical layout structure (same text positions, same figures), isolating script as the only variable. Real data would confound layout differences with script differences. We use a **per-pixel layout head** rather than a global classifier because dense predictions provide spatially fine-grained alignment; a global classifier would lose localization information. The trade-off for KL divergence: it requires paired data and soft labels, increasing memory, but avoids the mode collapse of contrastive methods. The synthetic generation introduces translation noise, but we mitigate this by using high-quality translation APIs and only generating for a few major scripts (English, Arabic, Chinese) to cover diverse scripts. The per-pixel head adds computational cost (≈5% more parameters), but it is lightweight and can be discarded at inference time. **Load-bearing assumption**: Synthetic script variants preserve exactly identical layout structure (text positions, sizes, figures) with only the visual script changing. We enforce this via geometric validation (Step 1a) requiring mean bounding box IoU ≥ 0.95 between variant and original; any variant below threshold is regenerated with adjusted font size to fit original boxes. This ensures layout invariance across scripts. Failure modes (e.g., text overflow due to longer translation) are explicitly detected and corrected, with a failure rate quantified during training (expected <5% of variants require regeneration).

### (D) Why it measures what we claim
The **pairwise KL divergence** between layout predictions from different script variants measures **cross-script consistency** because it quantifies how much the layout distribution changes when only the script varies. If the encoder is perfectly invariant, all layout maps are identical and KL=0. The assumption that script variants have identical layout structure is enforced by the synthetic pipeline and geometric validation (same text bounding boxes, same image size). This assumption is verified by the bounding box IoU check; if a variant fails, it is corrected. In the rare case that regeneration cannot achieve IoU ≥ 0.95 (e.g., extreme text length disparity), that variant is discarded from the batch (failure rate reported in experiment). The **layout head** operationalizes the concept of layout prediction as a distribution over classes; it is trained jointly with the encoder, so its outputs directly reflect the encoder's representation. The **consistency loss** is a direct proxy for script invariance: reducing it forces the encoder to produce similar features regardless of script.

## Contribution

(1) A novel training constraint (CSLA) that enforces per-pixel layout consistency across synthetic script variants to induce cross-script invariance in visual document encoders. (2) A synthetic multilingual document generation pipeline that produces paired documents with identical layout but different scripts, enabling controlled supervision for script invariance. (3) The principle that layout consistency serves as a structurally grounded inductive bias for script invariance, which can be integrated into existing pretraining frameworks like MonkeyOCRv2.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|-----------------------|
| Dataset | SyntheticDocBench (script-aligned) | Identical layout across scripts |
| Primary metric | Cross-script text detection F1 | Measures layout accuracy across scripts |
| Layout preservation metric | Mean bounding box IoU across variants | Validates assumption; target ≥0.95 |
| Baseline 1 | Vanilla encoder (no CSLA) | Absence of script-invariance constraint |
| Baseline 2 | Contrastive layout alignment (InfoNCE) | Alternative distribution matching method |
| Baseline 3 | Cyclic script translation (round-trip) | Noise injection instead of consistency |
| Ablation of ours | CSLA w/o per-pixel head (global avg) | Tests dense vs. global layout prediction |

### Why this setup validates the claim
SyntheticDocBench guarantees that layout structure is identical across English, Arabic, and Chinese script variants, so any performance variation in text detection directly measures script-induced representation shift. The primary metric (cross-script F1) averages detection accuracy across all scripts, penalizing models that fail on certain scripts. We additionally compute mean bounding box IoU across variants to verify the load-bearing assumption; we report the percentage of variants that fail the 0.95 IoU threshold and require regeneration (expected <5%). Vanilla encoder tests the premise that standard training lacks invariance. Contrastive alignment probes whether pairwise distribution matching (ours) outperforms alternative forms of consistency. Cyclic translation tests whether simple data augmentation (translating back and forth) suffices. The ablation of the per-pixel head isolates whether dense spatial alignment is essential. Together, this design forces the model to exhibit cross-script consistency to succeed, and failure patterns across baselines will reveal which mechanism breaks invariance.

### Expected outcome and causal chain
**vs. Vanilla encoder (no CSLA)** — On an Arabic document with the same layout as English, the vanilla encoder produces different feature maps due to script-specific visual patterns (e.g., cursive shapes causing layout head confusions). This yields lower detection F1 on Arabic. Our method forces the encoder to ignore script variation via KL divergence, so layout features remain stable. We expect vanilla to show a significant drop on non-English scripts (e.g., 10-15% F1 gap), while our method maintains near-constant F1 across scripts (gap <3%).

**vs. Contrastive layout alignment (InfoNCE)** — On a case where English and Chinese variants have similar background patterns but different script, contrastive loss may push their layout features apart because it maximizes mutual information between positive pairs (same script vs. different script) and can confuse script invariance with instance discrimination. Our KL divergence directly minimizes distribution difference, avoiding such repulsion. We expect contrastive to show moderate improvement over vanilla but still exhibit script-specific biases (e.g., 5% F1 gap), while ours achieves consistently high F1 across all scripts.

**vs. Cyclic script translation (round-trip)** — On a document with long Arabic text that overflows its bounding box after translation, cyclic translation introduces layout distortions (resized elements) that degrade detection. Our synthetic pipeline uses geometric validation and regeneration to preserve exact bounding boxes, so layout is identical across variants. Thus cyclic translation harms downstream performance (uniform drop ~5% F1), whereas our method retains full layout fidelity and yields higher absolute F1.

### What would falsify this idea
If our CSLA model shows uniform F1 improvement across all scripts (including English) compared to vanilla, and the improvement is not concentrated on cross-script consistency (i.e., the per-script variance remains similar), then the claim of script invariance is unsupported. More specifically, if the pairwise KL divergence between script variants does not decrease significantly relative to vanilla, the central mechanism is failing. Furthermore, if the mean bounding box IoU across variants drops below 0.95 for more than 10% of documents (indicating layout preservation failure), the assumption is violated and the results may not reflect pure script invariance.

## References

1. MonkeyOCRv2: A Visual-Text Foundation Model for Document AI
2. olmOCR 2: Unit Test Rewards for Document OCR
3. DeepSeek-OCR: Contexts Optical Compression
4. MinerU2.5: A Decoupled Vision-Language Model for Efficient High-Resolution Document Parsing
5. General OCR Theory: Towards OCR-2.0 via a Unified End-to-end Model
6. TÜLU 3: Pushing Frontiers in Open Language Model Post-Training
7. DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model
8. UReader: Universal OCR-free Visually-situated Language Understanding with Multimodal Large Language Model
