# PatchText: Multi-Scale Cross-Attention Consistency for Layout-Aware Visual Encoder Pretraining

## Motivation

Existing visual encoder pretraining methods for documents, such as MonkeyOCRv2, rely on generation and reconstruction losses that align visual features with text at a global sequence level but do not enforce explicit correspondence between individual visual patches and the text tokens they contain. This root-cause inconsistency leads to poor layout-text mapping in encoder-only architectures, because without the joint cross-attention mechanism of full VLMs (e.g., dots.ocr), the encoder lacks a training signal that ties spatial patch arrangements to text semantics. As a result, downstream tasks like document VQA suffer from misalignment between visual regions and textual content.

## Key Insight

The spatial arrangement of overlapping visual patches in a document image forms a graph whose adjacency structure is isomorphic to the co-occurrence graph of text tokens in the OCR transcription; enforcing feature similarity between pattices to be proportional to text token similarity embeds the layout-text invariant into the encoder without requiring a decoder.

## Method

(A) **What it is** — PatchText is a self-supervised pretraining method for a vision transformer (ViT) visual encoder. Input: document images with OCR text annotations. Output: a trained encoder whose features at multiple scales encode both visual content and spatial layout-text correspondences. (B) **How it works** — The core operation is a multi-scale contrastive loss computed over pairs of visual patches. For a given document image, we extract patches at two scales (e.g., 16x16 and 32x32) and obtain their visual features from the ViT encoder. For each patch at the finer scale, we identify its overlapping coarser patch and compute a text similarity score based on the Jaccard overlap of the token sequences contained in each patch (as determined by OCR bounding boxes). We then enforce that the cosine similarity of the visual features matches the text similarity using a InfoNCE loss with temperature τ. The same procedure is applied across different scales and across spatial neighbors within the same scale (to encourage local consistency). The overall loss is a weighted sum: L = λ_scale * L_cross_scale + λ_neighbor * L_neighbor + λ_global * L_global (where L_global is a standard global contrastive loss between the [CLS] token and the document-level text embedding to maintain task alignment). (C) **Why this design** — We chose a contrastive formulation over a direct regression of feature similarity because contrastive learning naturally handles the high-dimensional feature space and is less sensitive to absolute scale matching. We used multi-scale patches to capture both fine-grained character-level and coarse block-level layout relations; the trade-off is increased computational cost (two forward passes for scales) versus a single-scale approach that would miss hierarchical structure. We opted for a simpler token overlap (Jaccard) rather than a learned text encoder similarity to avoid introducing another trainable module that could collapse; the trade-off is that Jaccard may be a coarse proxy for semantic relatedness but ensures the signal is grounded in actual document content. We included both cross-scale and within-scale neighbor terms to enforce consistency both hierarchically and locally, which prevents the encoder from learning scale-specific shortcuts. (D) **Why it measures what we claim** — The cosine similarity between patch features (in the contrastive space) measures the degree of layout-text consistency because the loss explicitly pulls patches with high token overlap to have similar features and pushes patches with low overlap apart. This works under the assumption that overlapping patches indeed contain overlapping tokens; this assumption fails when OCR bounding boxes are inaccurate or when patches are large and contain multiple unrelated text segments, in which case the similarity target becomes noisy. The multi-scale design mitigates this by capturing alignment at different granularities. The InfoNCE loss computes the probability that a positive pair is correctly identified among negatives, which quantifies the encoder's ability to match visual and textual structure; this assumption fails when negative pairs are not sufficiently discriminative (e.g., all patches look similar), in which case the loss saturates and does not provide gradient.

## Contribution

(1) A novel multi-scale cross-attention consistency pretraining objective that enforces alignment between visual patch features and text token similarity without requiring a decoder or joint attention. (2) Empirical demonstration that this pretraining consistently improves encoder-only document VQA performance over global objectives like generation and reconstruction, establishing that patch-level layout-text correspondence is a critical missing signal. (3) A scalable training pipeline that uses only OCR transcriptions and patch overlapping statistics, avoiding expensive synthetic data generation.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|-----------------------|
| Dataset | DocVQA | Standard document VQA evaluation. |
| Primary metric | ANLS | Robust to answer variations. |
| Baseline 1 | ImageNet-pretrained ViT | No document-specific pretraining. |
| Baseline 2 | CLIP-pretrained ViT | Lacks fine-grained layout-text alignment. |
| Baseline 3 | Single-scale PatchText | Ablates multi-scale contribution. |
| Ablation-of-ours | Ours w/o neighbor loss | Tests importance of local consistency. |

### Why this setup validates the claim

DocVQA tests visual-text reasoning on diverse layouts, making it ideal to evaluate layout-text alignment. ANLS captures partial correctness, critical for OCR-dependent answers. Comparing against ImageNet-ViT isolates the benefit of document pretraining; CLIP-ViT tests whether global image-text alignment suffices. Single-scale PatchText reveals whether multi-scale contrast is necessary. The ablation of neighbor loss identifies if local consistency drives performance. This combination directly tests whether multi-scale patch-text contrast yields better alignment, as improvement on DocVQA ANLS must stem from the proposed mechanism rather than generic pretraining.

### Expected outcome and causal chain

**vs. ImageNet-pretrained ViT** — On a case where a receipt has prices aligned in columns, the baseline produces wrong answers because it lacks any layout-text correspondence — it treats the image as natural scene and fails to associate text with spatial positions. Our method instead aligns visual patches with overlapping text tokens via the contrastive loss, so for such a receipt the encoder learns to group column-wise patches. We expect a clear ANLS gap (e.g., +5–10%) on all DocVQA questions, especially numeric and list-type queries.

**vs. CLIP-pretrained ViT** — On a case where a form has a label "Date:" far from its input field, CLIP's global image-text matching cannot resolve the fine-grained association; it produces random guesses because it only captures coarse scene-level semantics. Our method enforces patch-level correspondence, linking the label patch to the nearby input field patch even when separated. Hence we expect improvement concentrated on relational questions (e.g., "What is the date?") with a +3–5% ANLS gain over CLIP.

**vs. Single-scale PatchText** — On a case where a multi-column article has a header spanning columns, single-scale PatchText (16x16) treats each column separately, missing cross-column layout; it confuses which text belongs to the header. Our multi-scale approach (16x16 + 32x32) captures both fine details and coarse blocks, correctly associating the header with both columns. We expect a modest gain (+1–3% ANLS) on multi-column layouts, with parity on simple layouts.

### What would falsify this idea

If our method's ANLS gain over single-scale PatchText is uniform across all layout types rather than concentrated on multi-column or spatially complex pages, then the multi-scale mechanism is not the cause — the gain would stem from other factors like increased capacity or data augmentation.

## References

1. MonkeyOCRv2: A Visual-Text Foundation Model for Document AI
2. olmOCR 2: Unit Test Rewards for Document OCR
3. DeepSeek-OCR: Contexts Optical Compression
4. MinerU2.5: A Decoupled Vision-Language Model for Efficient High-Resolution Document Parsing
5. General OCR Theory: Towards OCR-2.0 via a Unified End-to-end Model
6. TÜLU 3: Pushing Frontiers in Open Language Model Post-Training
7. DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model
8. UReader: Universal OCR-free Visually-situated Language Understanding with Multimodal Large Language Model
