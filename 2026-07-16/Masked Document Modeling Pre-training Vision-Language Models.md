# Masked Document Modeling: Pre-training Vision-Language Models for Document Parsing without Paired Data

## Motivation

Existing end-to-end document parsers like OvisOCR2 rely on supervised fine-tuning with paired image-Markdown data, which requires expensive synthetic data generation from HTML or manual annotations. This dependence on paired data limits scalability to diverse document types where HTML sources are unavailable, and fails to leverage the abundant unlabeled document images available in practice. The root cause is the supervised-only assumption that paired image-text data is necessary for learning effective representations.

## Key Insight

By jointly training a model to reconstruct masked image patches and predict the text content within those patches from context, the model learns layout-aware and reading-order-sensitive representations using only the document image itself as supervision.

## Method

(A) **What it is**: MDM is a self-supervised pre-training objective that takes a document image, randomly masks a high proportion (75%) of image patches of size 16×16, and trains the model to predict both the visual pixels of each masked patch and the OCR text appearing in that patch region, using a dual-decoder architecture.

(B) **How it works**:
```pseudocode
Input: Document image I, mask ratio r=0.75
1. Divide I into N non-overlapping patches of size 16x16.
2. Sample a random binary mask m ∈ {0,1}^N where P(m_i=1)=r.
3. Encode visible patches (masked patches removed) using a ViT-B/16 encoder E (12 layers, hidden dim 768, 12 heads).
4. For each masked patch i:
   a. Visual decoder: predict pixel values p_i via a single linear layer → MSE loss L_vis = Σ_i ||p_i - ground truth||^2.
   b. Text decoder: predict text token sequence t_i (max length 16 chars) using a 2-layer transformer (hidden dim 256, 4 heads) that takes encoder output and visual decoder features as cross-attention input → cross-entropy loss L_text = Σ_i CE(t_i, ground truth text). Text tokens are character-level (vocab size 128: ASCII printable + [NO_TEXT]). For patches with no text, target is [NO_TEXT].
5. Total loss: L = L_vis + 0.1 * L_text.
Hyperparameters: mask ratio r=0.75, λ=0.1, text decoder: 2-layer transformer, hidden dim 256, 4 heads, max text length 16, character-level tokenizer.
```

(C) **Why this design**: We chose a dual-decoder architecture over a single reconstruction decoder because recovering text requires discrete token prediction, which is fundamentally different from pixel reconstruction; using a shared encoder forces the model to learn representations that encode both visual and textual content, beneficial for downstream parsing. We set mask ratio to 75% following MAE findings that higher ratios force stronger understanding of global structure. We used a small text decoder (2 layers, 256 dim) to keep computation manageable, accepting that it may not capture long-range text dependencies but focusing on local patch-level text—this trade-off suits document parsing where local text recognition is critical. We opted for patch-level text prediction rather than full-page OCR because it aligns with the masked input structure and avoids needing global text order information, which is unavailable in self-supervision. This design means the pre-trained model is best suited for tasks requiring local text recognition, but downstream fine-tuning can learn global ordering and more complex layout understanding. **Load-bearing assumption**: The text decoder's ability to predict patch-level text (max 16 chars) from surrounding visible context is sufficient to drive the encoder to learn layout-aware representations. We verify this by measuring top-1 character accuracy on a validation set of masked patches; if accuracy falls below 40% (chance is ~1/128), the assumption fails and pre-training is ineffective for downstream parsing.

(D) **Why it measures what we claim**: The visual reconstruction loss L_vis measures low-level pixel fidelity, not layout understanding; it serves as a proxy for layout awareness under the assumption that high pixel reconstruction requires understanding spatial arrangement of objects. This assumption fails when patches are blank or contain low-frequency textures (e.g., uniform backgrounds), in which case L_vis reflects only low-level statistics rather than layout understanding. To validate the proxy, we compute the correlation between L_vis and layout accuracy (e.g., reading order correctness on a synthetic dataset with known layout perturbations); a Pearson correlation r>0.7 supports the proxy. The text prediction loss L_text measures the model's ability to read text in masked regions using surrounding context, approximating OCR capability driven by layout cues; this assumption fails when text is ambiguous from context alone (e.g., overlapping text boxes or handwritten text), in which case L_text reflects guessing based on visual similarity rather than genuine reading. We verify the text proxy by measuring character accuracy on patches where OCR can be reliably inferred from context (e.g., structured forms) vs. ambiguous cases; we expect accuracy>70% on structured forms. Together, the combined loss forces the model to learn representations that are both visually coherent and text-aware, directly reducing dependence on paired data by leveraging the document image's own structure as supervision. The quantity L_vis + λL_text operationalizes the concept of self-supervised document understanding because it requires the model to jointly reason about visual appearance and textual content from partial observations.

## Contribution

(1) A novel self-supervised pre-training objective, Masked Document Modeling, that combines masked image reconstruction and local OCR prediction from a single document image without needing paired text. (2) Demonstration that pre-training with this objective improves downstream document parsing performance over training from scratch on limited labeled data, as evidenced by experiments on OmniDocBench and PureDocBench. (3) Release of pre-trained models and code for reproducible research.

## Experiment

### Evaluation Setup
| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | OmniDocBench | Tests layout and text jointly. |
| Primary metric | Overall score | Captures both visual and textual accuracy. |
| Baseline 1 | Pipeline (OCR+Layout) | Strong traditional approach. |
| Baseline 2 | End-to-end LVLM (Donut) | Current SOTA end-to-end. |
| Baseline 3 | Self-supervised MAE (pixel-only) | Tests need for text decoder. |
| Ablation 1 | Ours (visual decoder only) | Isolates text loss contribution. |
| Ablation 2 | Ours (shared decoder) | Tests dual-decoder architecture. |
| Compute budget | Pre-train on 100K RVL-CDIP images, 200 epochs, 4×A100 GPUs, ~72 hours | Ensures feasibility. |
| Data preprocessing | Resize to 224×224, apply random horizontal flips and color jitter; extract OCR text via Tesseract (ground truth for text decoder) | Reproducible pipeline. |

### Why this setup validates the claim
OmniDocBench evaluates both layout understanding and text recognition jointly, making it ideal for testing whether MDM's joint visual-text reconstruction yields superior document representations. The pipeline baseline tests whether separate OCR+layout stages can match an end-to-end self-supervised method; if MDM outperforms it, that shows the benefit of joint representation. The LVLM baseline tests the value of self-supervision over supervised pre-training; if MDM matches or exceeds it without paired data, the claim gains support. The MAE baseline isolates the effect of adding a text decoder—improvement over MAE demonstrates that text prediction drives better understanding. The shared decoder ablation (Ablation 2) tests whether the dual-decoder architecture itself contributes nontrivially; if the shared decoder performs comparably, the dual-decoder choice is not crucial. The compute budget and preprocessing details reduce implementation uncertainty and help assess reproducibility. The overall score is the right metric because it aggregates both aspects the method is designed to improve.

### Expected outcome and causal chain

**vs. Pipeline (OCR+Layout)** — On a document with overlapping text boxes and complex layouts, the pipeline fails to correctly assign reading order and merges separate text blocks, because its OCR and layout analyses are independent and rigid. Our method, by jointly reconstructing visual and textual content from masked patches, learns to associate text with its surrounding visual context, so it produces coherent text blocks even when layout is complex. We expect a noticeable gap (e.g., 5-10 points) on multi-column or overlapping layout subsets, but parity on simple single-column documents.

**vs. End-to-end LVLM (Donut)** — On a rare document type (e.g., historical manuscript or handwritten form), the LVLM underperforms because it was pre-trained on large-scale diverse but not domain-specific data. Our method uses self-supervision on the target domain's own images, learning domain-specific visual and textual patterns from the training data itself. Thus we expect a smaller gap on standard documents (within 2-3 points) but a clear advantage (5+ points) on uncommon document categories in the benchmark.

**vs. Self-supervised MAE (pixel-only)** — On a document image where text is dense and crucial for layout understanding (e.g., a newspaper page), MAE fails to read text because it only reconstructs pixels, missing semantic content. Our method explicitly predicts text tokens in masked patches, forcing the encoder to learn text-aware features. Therefore we expect MDM to significantly outperform MAE on text-heavy subsets (e.g., 10-15 point difference), while on purely visual or text-sparse documents the gap shrinks.

**vs. Ours (shared decoder)** — On a document with complex layout, the shared decoder may struggle to simultaneously optimize pixel and text objectives, leading to poorer representation. We expect the dual-decoder variant to outperform the shared decoder by 3-5 points on the overall score, confirming that separate decoders are beneficial.

### What would falsify this idea
If our method shows no significant improvement over MAE on text-heavy document subsets, or if the pipeline baseline outperforms MDM on complex layouts, or if the shared decoder variant matches the dual-decoder variant, then the core claim that joint visual-text self-supervision is beneficial for document parsing would be undermined.

## References

1. OvisOCR2 Technical Report
2. Logics-Parsing Technical Report
3. dots.ocr: Multilingual Document Layout Parsing in a Single Vision-Language Model
4. Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution
5. Qwen Technical Report
6. Nougat: Neural Optical Understanding for Academic Documents
7. DocXChain: A Powerful Open-Source Toolchain for Document Parsing and Beyond
8. Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond
