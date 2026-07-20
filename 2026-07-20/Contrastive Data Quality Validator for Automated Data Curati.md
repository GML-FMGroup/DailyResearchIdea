# Contrastive Data Quality Validator for Automated Data Curation in Healthcare LLM Self-Evolution

## Motivation

The Cura 1T self-evolution loop relies on human gating for each evolution round, limiting scalability because expert human input is deemed necessary for data quality. This human-in-the-loop bottleneck prevents fully automated data curation. We need an automated method to detect errors and inconsistencies in synthetic training data, thereby removing the human oversight requirement.

## Key Insight

Valid medical data exhibits consistent statistical regularities (e.g., symptom-diagnosis co-occurrence, temporal consistency), while errors and inconsistencies are outliers in this learned manifold, enabling a contrastive validator to distinguish them without explicit medical knowledge.

## Method

## Contrastive Data Quality Validator (CDQV)

(A) **What it is**: CDQV is a fine-tuned transformer (125M parameters) that takes a data sample (e.g., a patient consultation transcript) and outputs a validation score ∈ [0,1] indicating the likelihood of being error-free. Its inputs are candidate data from the self-evolution loop; its output gates data acceptance.

(B) **How it works**:
1. **Positive set**: Collect 500 high-quality samples from human-validated sources (e.g., MedAgentBench or expert-curated records).
2. **Negative generation**: For each positive sample, apply 5 corruption operators to produce 5 negative samples. The operators are: (1) Entity swap: replace a random medical entity (e.g., drug name, diagnosis) with another from a knowledge base (e.g., RxNorm). (2) Temporal inconsistency: swap the order of a symptom and its corresponding treatment in the narrative. (3) Irrelevant insertion: add a sentence about an unrelated symptom or treatment. (4) Omission: remove a key attribute (e.g., dosage, duration). (5) Numerical contradiction: change a numerical value (e.g., dosage) to a conflicting value within plausible range.
3. **Contrastive training**: Train a small transformer (125M parameters, 12 layers, 768 hidden size) using InfoNCE loss with batch size 64, temperature τ=0.1. For each anchor (positive), we have 5 negatives (hard negatives from same positive) and 59 in-batch negatives from other positives. The model learns to assign higher scores to positives.
4. **Inference**: For each candidate data point, compute CDQV score; accept if score > threshold τ (tuned on a held-out validation set of 100 human-labeled samples to achieve 95% recall on positives).
5. **Integration**: Accepted data enters the training mixture; rejected data is logged and used for data mixture adjustment (as in Cura 1T's failure-based refinement).
6. **Calibration and verification**: After initial training, collect 100 human-annotated real errors from the self-evolution loop's initial rounds. Compare their feature distribution (e.g., using t-SNE on CDQV embeddings) with the distribution of generated negatives. If the Jensen-Shannon divergence between the two distributions exceeds 0.1, adjust the corruption operators (e.g., modify operation probabilities or add new operation types) until divergence falls below 0.1. This ensures that the negative distribution remains representative of real errors.

(C) **Why this design**: (1) We chose a small transformer over a larger LLM to avoid high inference cost during the self-evolution loop, accepting that subtle errors may be missed, but our corruption operators are designed to produce detectable inconsistencies. (2) We chose contrastive learning over binary classification because it naturally handles the open-ended set of possible errors without requiring exhaustive error-type labeling, though training is more sensitive to negative sampling quality. (3) We chose automated corruption operators over human annotation of negatives for scalability, though they may not cover all real-world error patterns, necessitating periodic re-validation of the validator against new failure modes.

(D) **Why it measures what we claim**: CDQV's score measures the model's confidence that the sample is drawn from the valid data distribution. This quantifies **data quality** because our training pairs ensure valid samples are concentrated in a low-dimensional manifold via contrastive separation; the score gap directly reflects deviation from that manifold. However, this equivalence relies on the **assumption** that our corruption operators generate negative samples that accurately represent the distribution of real errors in synthetic healthcare data. If real errors follow a different distribution (e.g., subtle contradictions not induced by operators), CDQV may measure similarity to corruption templates rather than true quality, in which case the score reflects artifact detection rather than data validity. We mitigate this by the calibration step in (B.6), which actively aligns the operator output with observed real errors.

## Contribution

(1) A contrastive data quality validator that automates the human gating step in healthcare LLM self-evolution, enabling fully automated data curation. (2) A method for generating negative training examples via domain-specific corruption operators that capture common medical data errors. (3) A practical integration of the validator into an existing self-evolution loop, replacing human oversight without reducing data quality.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | MedAgentBench virtual EHR tasks | Realistic healthcare agent benchmark with multi-round evolution |
| Primary metric | Task completion rate | Directly measures agent ability after training on filtered data |
| Baseline | Cura 1T (no validator) | Baseline without any data quality filtering |
| Baseline | GPT-4 as online validator | Strong LLM-based quality filter, high cost |
| Ablation-of-ours | CDQV with binary classifier | Replaces contrastive loss with cross-entropy on same labels |

### Why this setup validates the claim

This design isolates the effect of CDQV by comparing against an unfiltered baseline (Cura 1T) and a powerful but expensive validator (GPT-4). If CDQV’s contrastive learning better captures realistic data errors than binary classification (ablation), then its outputs should improve task completion more than the ablation. The primary metric directly reflects downstream agent performance, which is the ultimate claim: higher-quality training data leads to better agents. The falsifiable prediction is that CDQV’s advantage is largest on tasks requiring factual consistency (where corruption operators matter most) and minimal on simple tasks. Using a single metric avoids cherry-picking and forces the method to demonstrate overall utility.

### Expected outcome and causal chain

**vs. Cura 1T (no validator)** — On a case where a synthetic consultation has a temporal inconsistency (e.g., symptoms appearing after treatment), Cura 1T’s training includes that sample, causing the agent to learn flawed causality and fail in follow-up diagnosis. Our method rejects such samples via CDQV’s low score because the corruption operator (temporal swap) creates a contrastive negative that the model learns to downweight. Thus we expect a noticeable gap (e.g., ~15–20% higher task completion) on temporally-sensitive tasks like medication scheduling, but parity on static information retrieval tasks.

**vs. GPT-4 as online validator** — On a straightforward but volume-heavy case (e.g., hundreds of routine checkups), GPT-4’s per-sample cost and latency become prohibitive, forcing a trade-off between coverage and accuracy. Our compact CDQV processes samples cheaply, allowing full-coverage filtering. Therefore, while GPT-4 may excel on rare complex errors (higher per-sample precision), its overall training data quality suffers from reduced coverage, leading to worse average agent performance. We expect CDQV to match or exceed GPT-4 on aggregate task completion, especially under budget constraints (cost per filtered sample < 1/100 of GPT-4).

### What would falsify this idea

If CDQV’s gain over Cura 1T is uniform across all task types (rather than concentrated on tasks where factual consistency errors matter), or if its performance is consistently below that of the binary classifier ablation, then the contrastive design does not capture error modes better than simpler methods, invalidating the central claim. Additionally, if the calibration step fails to reduce distribution mismatch (Jensen-Shannon divergence >0.1) after two operator adjustments, the assumption linking CDQV score to true data quality is falsified.

## References

1. Cura 1T: Specialized Model for Agentic Healthcare
2. MedAgentBench: A Realistic Virtual EHR Environment to Benchmark Medical LLM Agents
