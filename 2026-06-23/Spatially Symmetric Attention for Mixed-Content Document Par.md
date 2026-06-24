# Spatially Symmetric Attention for Mixed-Content Document Parsing

## Motivation

Existing OCR systems, such as Unlimited OCR Works, rely on sliding window attention that processes text linearly, ignoring spatial interdependencies between structured elements like tables, equations, and figures. This forces models to either use separate layout stages (as in Dolphin) or treat all content as flat text, leading to error cascades or loss of structural coherence. We need a single-pass method that jointly parses mixed-content documents by exploiting spatial relationships.

## Key Insight

Enforcing attention symmetry between spatially proximate tokens inherently groups contiguous structured regions into coherent units, enabling integrated layout understanding without explicit segmentation.

## Method

**(A) What it is:** Spatially Symmetric Attention (SSA) is a modified self-attention layer that replaces standard scaled dot-product attention. Inputs are token embeddings plus 2D positional encodings (row and column coordinates). Outputs are context-aware representations where attention between tokens is constrained to be symmetric for spatially close tokens. **Central assumption:** For tokens within spatial distance τ, enforcing symmetric attention is sufficient to group them into coherent layout regions without harming intra-region hierarchies.

**(B) How it works:**
```pseudocode
Algorithm: Spatially Symmetric Attention (SSA)
Input: Token embeddings E ∈ R^{n×d}, 2D positions P ∈ R^{n×2} (row, col)
Hyperparameters: spatial distance threshold τ=5 (pixels), symmetry strength α=0.5, top-k (optional, default: n)
Output: Updated embeddings E'

1. Compute Q, K, V from E: Q = E W_q, K = E W_k, V = E W_v
2. Compute raw attention scores A_raw = Q K^T / sqrt(d)
3. Compute pairwise spatial distances D[i,j] = ||P[i] - P[j]||_2
4. (Optional) For efficiency, retain only top-k smallest distances per token; set others to large value.
5. Compute symmetry mask M[i,j] = 1 if D[i,j] < τ else 0
6. For each (i,j): if M[i,j]==1: A_sym[i,j] = (A_raw[i,j] + A_raw[j,i])/2; else A_sym[i,j] = A_raw[i,j]
7. A_soft = softmax(A_sym, dim=-1)
8. Output: E' = A_soft V
```

**(C) Why this design:** We chose symmetry enforcement over attention bias from spatial distances because it directly encodes structural invariance: tokens within the same layout region should attend to each other equally, not just proportionally to distance. This trade-off sacrifices the ability to model directional flow (e.g., left-to-right reading order) inside a region, but that directionality is already captured by positional encodings; symmetry ensures region coherence. We used a hard threshold τ rather than learnable gating because it simplifies training and avoids gradient instability from binary decisions; this introduces a hyperparameter but ensures consistent grouping. We kept standard softmax after symmetry to preserve probabilistic interpretation, accepting that raw attention might be distorted by averaging; the α parameter (set to 0.5) balances symmetry and original signal allowing partial symmetry. We chose Euclidean distance over Manhattan or Chebyshev because it yields isotropic neighborhoods consistent with spatial contiguity.

**(D) Why it measures what we claim:** The quantity (A_raw[i,j] + A_raw[j,i])/2 for spatially close tokens measures the degree of mutual attention between two tokens, which operationalizes the concept of "coherent unit" because symmetry implies bidirectional influence; this assumption fails when tokens within a region have asymmetric importance (e.g., a table header attending more to body cells than vice versa), in which case the symmetric average underestimates the stronger direction and may blur intra-region hierarchies. The spatial threshold τ measures the boundary of contiguous regions; it operationalizes "layout structure" by assuming that all tokens within τ pixels belong to the same region; this assumption fails for overlapping or nested regions (e.g., a figure caption inside a figure), in which case the mask over-includes tokens from different regions, causing attention to blend distinct elements and potentially cross-contaminate representations. **To quantify this failure, we construct synthetic data with N=1000 random token pairs: half from same region (distance < τ) and half from different regions (distance > τ). Inside same-region pairs, we simulate asymmetric importance by scaling token importance (e.g., header token importance 2× body token). We measure precision of SSA (using M[i,j] as classifier) in predicting same-region membership. Expected precision drops from >95% (symmetric importance) to ~70% (asymmetric), confirming the assumption's fragility.**

## Contribution

(1) We introduce Spatially Symmetric Attention, a novel attention mechanism that enforces attention symmetry for spatially proximate tokens, enabling integrated parsing of mixed-content documents without explicit layout stages. (2) We demonstrate that this structure reduces error cascades compared to two-stage pipelines like Dolphin, as layout parsing and content extraction occur concurrently. (3) We provide an open-source implementation and pre-trained models trained on a synthetic dataset of mixed-content documents covering tables, equations, and figures.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset 1 | FUNSD (Form Understanding) | Evaluate layout-based entity extraction |
| Dataset 2 | PubTables-1M | Evaluates table structure parsing (coherent rows/cols) |
| Dataset 3 | ICDAR 2019 (cTDaR) | Competition dataset for table recognition, tests generalization |
| Primary metric | Entity-level F1 score (FUNSD), Grid-F1 (PubTables), IOU (ICDAR) | Directly measures correct region linking per dataset |
| Baseline 1 | LayoutLMv3 | State-of-the-art with spatial positions |
| Baseline 2 | Vanilla Transformer + 2D pos | No spatial prior beyond positions |
| Baseline 3 | Distance-biased attention | Spatial bias but not symmetric |
| Ablation of ours | SSA with learnable α | Tests fixed vs adaptive symmetry |
| Ablation of ours | SSA with top-k=50 nearest tokens | Tests efficiency-accuracy trade-off |

### Why this setup validates the claim
This combination forms a falsifiable test of the central claim that spatially symmetric attention improves region coherence for OCR layout understanding. FUNSD requires linking spatially close label-value pairs, directly probing the symmetry mechanism. PubTables-1M extends to dense tabular structures with many close tokens, where symmetry should enforce row/column grouping. ICDAR 2019 tests generalization to unseen table styles. LayoutLMv3 tests whether adding explicit symmetry to an already strong positional model yields gains. Vanilla Transformer shows the necessity of any spatial prior. Distance-biased attention isolates the symmetry contribution from mere distance weighting. The ablation with learnable α controls for hyperparameter flexibility, while top-k ablation tests robustness to reduced computation. Entity-level F1, Grid-F1, and IOU are chosen because they reflect correct grouping of tokens into coherent units—the precise effect symmetry is predicted to improve.

### Expected outcome and causal chain

**vs. LayoutLMv3** — On a case where a form has a multi-token address block (e.g., "123 Main St, Springfield"), LayoutLMv3 may attend asymmetrically (e.g., "Main" attending more to "St" than vice versa) due to directional biases from pretraining. Our SSA enforces symmetric attention among all tokens within τ, so each token equally attends to its neighbors, strengthening the block's internal coherence. We expect a noticeable F1 gap on multi-token field pairs but parity on single-token fields.

**vs. Vanilla Transformer + 2D pos** — On a case where two tokens are close but have dissimilar content (e.g., "Price" next to "$10"), vanilla attention may assign low mutual attention due to semantic mismatch, causing them to be treated as separate. Our symmetry forces high mutual attention for close tokens, linking them regardless of content. We expect a large F1 gain on close token pairs and no gain on distant pairs.

**vs. Distance-biased attention** — On a case with nested regions (e.g., a table cell inside a larger cell), distance-biased attention may downweight cross-cell interactions but still allow asymmetric attention within the cell (e.g., header attending more to data cells than vice versa). Our symmetry ensures equal bidirectional attention within τ, preventing such hierarchies from breaking region coherence. We expect a moderate F1 gain on complex layouts with nested structures.

**vs. SSA with learnable α** — On all cases, learnable α may adapt to different region types but fixed α=0.5 should already capture the core symmetry effect. We expect similar performance, with fixed α being slightly less flexible but avoiding overfitting.

**vs. SSA with top-k=50** — On dense layouts (e.g., PubTables), top-k approximation reduces computational cost without harming grouping for near neighbors. We expect negligible F1 drop (<0.5%) but 2× faster training.

**Across datasets** — Gains are largest on PubTables (dense tabular regions) and moderate on FUNSD, smallest on ICDAR (more layout variety). The causal chain is: symmetry → stronger intra-region attention → better grouping → higher region-level scores.

### What would falsify this idea
If the improvement over baselines is uniform across all token pairs (both close and distant) or concentrates on single-token entities, rather than being largest on close multi-token regions, then the symmetry claim is unsupported. Additionally, if the synthetic data precision drop (from >95% to <70%) is not observed, the assumption that symmetric attention implies coherent grouping is invalid.

## References

1. Unlimited OCR Works
2. PaddleOCR 3.0 Technical Report
3. Dolphin: Document Image Parsing via Heterogeneous Anchor Prompting
4. PP-StructureV2: A Stronger Document Analysis System
