# Spectral Co-Occurrence Alignment for Visual-Semantic Embedding

## Motivation

ViQ's text-aligned pre-training requires large-scale paired image-text data and a large pretrained LM, which are expensive and may introduce domain bias. Existing methods all assume the availability of abundant paired data and powerful pretrained components. This structural dependency on large-scale resources limits adoption in data-scarce or low-resource settings.

## Key Insight

The co-occurrence statistics of visual discrete codes and word tokens form structurally analogous graphs (e.g., power-law degree distributions, community structures), enabling alignment via spectral matching of their Laplacians without requiring paired data.

## Method

(A) **What it is:** The method, named Spectral Co-Occurrence Alignment (SCoVA), takes as input unpaired collections of images and texts, and outputs a shared embedding space for visual discrete codes and word tokens by aligning the spectral embeddings of two co-occurrence graphs.

(B) **How it works:**
1. **Graph Construction:** Extract visual codes from images using a semantically-aware visual tokenizer (e.g., SEED tokenizer or CLIP-based quantization) to obtain a discrete code sequence per image. Build visual co-occurrence graph G_v where nodes are visual tokens (codebook indices), edges weighted by normalized co-occurrence frequency within a sliding window (window_size=5). Similarly, build text co-occurrence graph G_t from unpaired text corpus using word tokens and co-occurrence within a sliding window (window_size=5).
2. **Spectral Embedding:** Compute normalized graph Laplacians L_v and L_t. Compute top-k eigenvectors (k=128) of each Laplacian using randomized SVD (Halko et al., 2011) to reduce computational cost from O(n^3) to O(n^2 log k), then scale by inverse square root of eigenvalues (spectral clustering embedding). Obtain embedding matrices U_v ∈ ℝ^{|V|×k} and U_t ∈ ℝ^{|T|×k}.
3. **Aligned Space via Seed Set:** Learn a linear transformation W ∈ ℝ^{k×k} by minimizing ||U_v W - U_t||^2 using a small paired seed set (e.g., 1000 image-text pairs) to anchor alignment. The shared embedding space is defined by transformed visual code embeddings U_v W and word embeddings U_t.

(C) **Why this design:** We chose spectral embedding over alternative graph alignment methods (e.g., graph neural networks) because spectral methods provide a closed-form alignment that is robust to small graph variations and does not require iterative training, accepting that the alignment quality depends on the structural similarity of the graphs. We used a small paired seed set for alignment rather than fully unsupervised alignment (e.g., by matching graph distances) because the seed set ensures the embedding axes correspond across modalities, at the cost of requiring a minimal amount of paired data. We selected the Laplacian eigenmaps over other spectral embeddings (e.g., node2vec) because it directly captures low-frequency community structure that reflects semantic similarity, though it may miss high-frequency details. The window-based co-occurrence weighting approximates semantic proximity, but may be noisy for rare tokens. A key load-bearing assumption is that the visual codes from the SEED tokenizer capture semantic concepts, so that co-occurrence graphs reflect semantic similarity analogous to word co-occurrence. This assumption is supported by design of SEED, which aligns visual tokens with textual semantics, but may fail if the tokenizer is not sufficiently semantic.

(D) **Why it measures what we claim:** The spectral embedding of the visual co-occurrence graph (U_v) encodes the structural roles of visual tokens in a low-dimensional space, where distances correspond to co-occurrence similarity. This operationalizes the concept of 'visual semantic similarity' because the assumption that nodes with similar co-occurrence patterns (high spectral affinity) have similar visual semantics holds in well-structured codebooks; this assumption fails when the codebook is poorly trained and codes are noisy, in which case the embedding reflects spurious co-occurrence patterns rather than true semantics. Similarly, the text embedding U_t operationalizes 'textual semantic similarity' under the same assumption for word co-occurrence. The linear alignment W, learned from a seed set, ensures that the axes of the two embeddings are mapped consistently, operationalizing 'cross-modal alignment' because the seed set provides a small set of known correspondences; this assumption fails if the seed set is biased or unrepresentative, leading to misalignment for out-of-sample tokens.

## Contribution

(1) A novel framework for learning shared visual-textual embeddings from co-occurrence graphs using spectral alignment, requiring minimal paired data. (2) The identification of structural isomorphism between visual code co-occurrence and word co-occurrence graphs as a viable bridge for cross-modal learning. (3) A method to avoid large pretrained language models and massive paired datasets, making visual-language representation learning accessible in low-resource settings.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|----------------------|
| Dataset | MS-COCO | Standard paired image-text dataset. |
| Primary metric | Recall@1 (mean) | Measures cross-modal alignment quality. |
| Baseline | VQGAN | No cross-modal alignment; shows baseline. |
| Baseline | CLIP | Strong paired alignment baseline. |
| Ablation-of-ours | SCoVA without seed set | Tests contribution of seed alignment. |
| Ablation-of-ours | Seed set size: 100, 500, 1000 | Tests sensitivity to paired data amount. |
| Additional analysis | Failure cases: structural dissimilarity between G_v and G_t (e.g., spectral gap mismatch) | Identifies when assumption fails. |

### Why this setup validates the claim
MS-COCO provides both paired data for seed/evaluation and abundant unpaired data for graph construction. VQGAN isolates the benefit of any alignment, CLIP sets a state-of-the-art paired baseline, and the ablation tests the necessity of the seed set. Recall@1 directly quantifies how well the shared embedding matches corresponding image-text pairs, making it the appropriate metric to detect improved alignment—if our method's spectral alignment captures semantic similarity, recall should improve over VQGAN and approach CLIP, especially on rare concepts where CLIP may falter due to data sparsity. The seed set size ablation evaluates the minimal supervision required, while failure case analysis checks the load-bearing assumption.

### Expected outcome and causal chain

**vs. VQGAN** — On a case where two semantically similar images (e.g., different breeds of dogs) have dissimilar visual code sequences, VQGAN's embeddings are purely visual and lack semantic grounding, so retrieval fails. Our method aligns visual codes to text embeddings via spectral graphs, mapping dog-related codes close together in the shared space. Thus we expect a large gap (e.g., 10–15 points) in Recall@1 on MS-COCO, especially for object categories with high visual variance.

**vs. CLIP** — On a case where a rare concept (e.g., "axolotl") appears infrequently in the paired training data, CLIP may misalign its visual and text embeddings due to weak supervision. Our method leverages unpaired data to build robust co-occurrence graphs, then uses a small seed set to anchor alignment. We expect our method to match CLIP on common concepts (within 1–2 points) but outperform by 5–10 points on long-tail concepts where CLIP lacks paired examples.

**vs. SCoVA without seed set** — On a case where the visual co-occurrence graph has a different spectral structure than the text graph (e.g., visual codes form tight clusters due to codebook artifacts), unsupervised alignment (e.g., Procrustes) may map unrelated nodes together. The seed set in our full method provides a few known correspondences to break symmetry. We expect the full method to show a 3–5 point improvement in Recall@1 over the ablation, concentrated on tokens present in the seed set.

**Seed set size ablation** — We expect Recall@1 to plateau after 500 seed pairs, indicating that only a small amount of paired data is needed. If no plateau is observed, the method may require more paired supervision than claimed.

**Failure case analysis** — If G_v and G_t have mismatched spectral properties (e.g., number of communities), alignment quality degrades. We will quantify this by measuring the spectral gap difference and correlating with per-concept recall drops. If the correlation is high, the assumption is critical; if low, other factors matter.

### What would falsify this idea
If the performance gain over CLIP is uniformly distributed across all concept frequencies rather than concentrated on rare concepts, then our claimed advantage from unpaired data is unsupported. Additionally, if the ablation (no seed set) achieves similar recall to the full method, the seed set is unnecessary and our reliance on paired data is unjustified. If the seed set size ablation shows no improvement beyond 100 pairs, the minimal paired data claim is weakened.

## References

1. ViQ: Text-Aligned Visual Quantized Representations at Any Resolution
2. Patch n' Pack: NaViT, a Vision Transformer for any Aspect Ratio and Resolution
3. Multimodal Autoregressive Pre-training of Large Vision Encoders
4. Unified-IO 2: Scaling Autoregressive Multimodal Models with Vision, Language, Audio, and Action
5. PaLI: A Jointly-Scaled Multilingual Language-Image Model
6. Pix2Struct: Screenshot Parsing as Pretraining for Visual Language Understanding
