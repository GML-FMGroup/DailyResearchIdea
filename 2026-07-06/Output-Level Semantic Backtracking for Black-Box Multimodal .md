# Output-Level Semantic Backtracking for Black-Box Multimodal Evidence Attribution

## Motivation

Current multimodal evidence localization methods, such as MultAttnAttrib, rely on internal attention heads or embeddings, which are inaccessible for proprietary black-box models. This structural dependence on white-box features limits their applicability to a narrow set of open models. We propose to overcome this by exploiting only the model's output text and token probabilities, enabling attribution for any multimodal QA model without internal access.

## Key Insight

Atomic claims in the generated answer can be independently verified via minimal counterfactual input perturbations, isolating evidence for each claim without requiring internal model features.

## Method

**SEMA (Semantic Evidence Mapping via Attribution)**

(A) **What it is:** SEMA is a training-free, black-box attribution method that takes a multimodal input (image and text), a model's answer, and outputs a spatial attribution map over input regions. It uses only the model's output probabilities and a semantic parser (DeBERTa-large fine-tuned on MNLI).

(B) **How it works:**
```
1. Atomic Claim Decomposition:
   - Parse answer A into a set of atomic claims C = {c1, c2, ..., ck} using DeBERTa-large fine-tuned on MNLI with a decomposition prompt.
2. For each claim ci:
   a. Baseline probability: Query model M with verification prompt "Is it true that <ci>?" and record p_i = P("yes" | M, input).
   b. Perturbation sampling: Generate N=1000 masked inputs by randomly masking 20% of image superpixels (SLIC, n_segments=50, compactness=10) and randomly deleting 20% of text sentences. For each masked input, compute perturbed probability p_i_m.
   c. Linear surrogate: Fit a ridge regression model (alpha=0.01) predicting p_i_m from binary indicators of mask presence. Coefficients yield importance weights for each region.
   d. Calibration: For each claim, compute deletion AUC on a held-out calibration set of 100 examples (random masks) for the top-10% regions ranked by coefficient. If AUC > 0.5, set all coefficients for that claim to zero.
3. Aggregation: Average the absolute coefficients across all claims to produce a final attribution map.
```
Hyperparameters: N=1000, ridge alpha=0.01, SLIC n_segments=50, compactness=10, masking ratio 20%, calibration set size=100, AUC threshold=0.5.

(C) **Why this design:** We chose per-claim attribution over whole-answer attribution (as in LIME) because claims are atomic and can be verified independently, reducing interference from unrelated information in the input. This design accepts the cost of multiple round-trips to the model (one per claim), increasing query complexity. We chose a linear surrogate rather than a more expressive model (e.g., random forest) because linear coefficients provide direct region importance, at the cost of possibly missing non-linear interactions. We used random masking over gradient-free optimization (e.g., genetic algorithms) because random sampling is simpler and does not require model-specific tuning, though it may require more samples for high-dimensional inputs. Finally, we chose image superpixels over pixel-level masks to maintain semantic coherence and reduce perturbation space, accepting a coarser granularity.

(D) **Why it measures what we claim:** 
- The probability drop when a region is masked, as captured by the linear coefficient, measures the **causal necessity** of that region for the claim because we assume masking is a valid intervention that removes information; this assumption fails when masking introduces artifacts (e.g., blank regions) that the model misinterprets, in which case the coefficient reflects the model's sensitivity to those artifacts rather than evidence necessity.
- The drop in P("yes" | verification prompt, masked input) relative to baseline measures the causal necessity of the region for the claim within the original context, under the assumption that the verification prompt is a valid query for the claim's truth in the original context. This assumption fails when the model exhibits mismatched behavior on the verification prompt (e.g., it may treat the prompt as a separate task), in which case the coefficient reflects the model's behavior on the prompt rather than the evidence necessity.
- The aggregation across claims measures **overall evidence importance** because we assume claims are independent and equally weighted; this assumption fails when claims are interdependent or partially overlapping, in which case the aggregate may double-count evidence shared across claims.
- The calibration step (discarding claims with AUC > 0.5) ensures that only attributions with measurable causal necessity are aggregated, mitigating the risk of spurious coefficients from non-causal correlations.

## Contribution

(1) The first black-box attribution method for multimodal QA that uses only output text and probabilities, enabling evidence localization for proprietary models. (2) A demonstration that semantic decomposition into atomic claims allows more targeted perturbation analysis, improving attribution granularity over whole-output methods. (3) An open-source implementation and a small-scale evaluation dataset for black-box multimodal attribution.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | VQA-X (existing) + Synthetic (custom) | VQA-X provides ground-truth evidence masks; Synthetic dataset (1000 examples) where each image contains two independent objects with independent claims (e.g., "There is a cat" and "There is a dog") allows validating claim independence assumption. |
| Primary metric | Deletion AUC | Measures causal necessity of regions. |
| Baseline 1 | LIME (whole-answer) | Standard black-box attribution baseline. |
| Baseline 2 | RISE | Whole-answer masking without decomposition. |
| Baseline 3 | Random attribution | Chance-level performance baseline. |
| Baseline 4 | SEMA w/ Random Forest | Tests linear surrogate necessity. |
| Baseline 5 | SEMA w/ Random Claim Boundaries | Decomposes answer into random chunks of equal length instead of semantic claims, isolating the effect of semantic decomposition. |
| Additional demonstration | GPT-4V (black-box) | Tests applicability to a truly black-box model; VQA-X test set (1000 examples) |

### Why this setup validates the claim
This experimental design tests SEMA's core claim that per-claim attribution with a linear surrogate over random masks yields causal necessity maps. The VQA-X dataset provides ground-truth evidence regions, enabling direct evaluation of attribution accuracy. Deletion AUC quantifies how well the method identifies causally necessary regions, aligning with SEMA's goal. Comparing against LIME and RISE isolates the benefit of atomic claim decomposition, as both baselines operate on whole answers. Random attribution provides a sanity check. The ablation with Random Forest surrogate tests whether the linearity assumption is critical. The ablation with Random Claim Boundaries tests whether semantic decomposition adds value beyond simple chunking. The synthetic dataset with known independent claims directly tests the claim independence assumption: if SEMA's aggregation over-weights overlapping evidence, performance will drop on synthetic data. Finally, evaluation on GPT-4V validates the black-box applicability (no internal access required). If SEMA outperforms these baselines specifically on instances involving multiple sub-claims, it supports the decomposition hypothesis; if performance is uniform across subsets, the claim is weakened.

### Expected outcome and causal chain

**vs. LIME** — On a multi-claim answer like "The dog is brown and sitting on a mat", LIME's whole-answer attribution conflates evidence for both claims. For the first claim it might highlight the dog's color, but for the second it might also highlight the mat inconsistently, resulting in a noisy map. Our method instead decomposes into atomic claims and attributes each separately, so the aggregated map cleanly highlights both dog color and mat regions. We expect a noticeable gap on multi-claim samples (e.g., Deletion AUC 0.15 lower for SEMA) but parity on single-claim samples.

**vs. RISE** — RISE uses random masking on the whole answer, averaging probability drops across all tokens. On a question like "What color is the car?" with answer "Red", both methods perform similarly. However, on a question requiring multiple reasoning steps (e.g., "Is the woman holding an umbrella?" with answer "No, she is holding a bag"), RISE's whole-answer attribution may capture spurious correlations with unrelated objects (e.g., background clouds). Our per-claim decomposition focuses on "She is holding a bag" and ignores the irrelevant part, yielding a cleaner map focused on the bag. We expect SEMA to achieve lower Deletion AUC (by ~0.1) on compositional questions, while matching RISE on simple ones.

**vs. Random attribution** — Random attribution assigns uniform importance to all regions. On any input, it will fail to correlate with ground-truth evidence. Our method should consistently outperform random, with Deletion AUC at least 0.2 lower across all samples. If SEMA's AUC is close to random, it indicates the attribution is no better than chance, falsifying the method.

**vs. SEMA w/ Random Forest** — If the linear surrogate is critical, SEMA with Random Forest will underperform due to lack of direct interpretation of coefficients. We expect SEMA (linear) to achieve 0.05 lower Deletion AUC than Random Forest version.

**vs. SEMA w/ Random Claim Boundaries** — If semantic decomposition is beneficial, SEMA with random chunks will show worse performance on datasets with multiple independent claims. On VQA-X multi-claim samples, we expect SEMA to have 0.1 lower Deletion AUC than random chunk version.

### What would falsify this idea
If SEMA's improvement over LIME/RISE is uniform across all sample types rather than concentrated on multi-claim questions, then the decomposition hypothesis is unsupported and the observed gain likely stems from other design choices (e.g., mask granularity) rather than the claimed mechanism. Additionally, if on synthetic data SEMA fails to correctly attribute regions to independent claims (i.e., both claims highlight both objects), then the claim independence assumption is invalidated.

## References

1. MultAttnAttrib: Training-Free Multimodal Attribution in Long Document Question Answering
2. VeriCite: Towards Reliable Citations in Retrieval-Augmented Generation via Rigorous Verification
3. VISA: Retrieval Augmented Generation with Visual Source Attribution
4. Towards Verifiable Text Generation with Evolving Memory and Self-Reflection
5. LongCite: Enabling LLMs to Generate Fine-grained Citations in Long-context QA
