# PARIS: Paraphrase-Augmented Response Invariance Self-Distillation for Language Models

## Motivation

Existing knowledge distillation (KD) and uncertainty estimation methods for language models depend on a teacher model to provide reference distributions or knowledge, as seen in Semantic Self-Distillation (SSD) which distills a teacher LLM's semantic clusters into a student. This reliance on a teacher is costly, may be unavailable, and introduces a static reference that cannot adapt to the student's evolving representation. The structural root cause is that both tasks require a fixed target distribution, which prevents truly self-supervised learning of robust representations and calibrated uncertainty without external supervision.

## Key Insight

Training a language model to produce consistent predictions across semantically equivalent input paraphrases and independent sampling seeds forces the model to internalize a robust semantic representation and makes disagreement across samples a natural, teacher-free uncertainty signal.

## Method

PARIS (Paraphrase-Augmented Response Invariance Self-Distillation) is a training procedure for a single language model that optimizes for consistent output distributions across semantically equivalent paraphrases and independent sampling seeds. No teacher model is used.

**Key assumption (explicit):** The model's own generated paraphrases are assumed to be semantically equivalent to the original input and sufficiently diverse to provide a meaningful invariance signal. To verify this during training, we filter self-generated paraphrases using a BERTScore threshold (≥0.85 with respect to the original input). Only paraphrases passing the filter are kept. If fewer than 2 paraphrases remain, the example is skipped.

**Training procedure (specified):**
- Hyperparameters: K=5 paraphrases, S=10 samples per paraphrase, temperature=1.0, α=0.1 (entropy regularization coefficient).
- Clustering: Use Sentence-BERT (all-MiniLM-L6-v2) to embed all outputs (K*S samples), then agglomerative clustering with a distance threshold of 0.5 to form semantic clusters.
- Loss: For each paraphrase i, compute the empirical distribution over clusters from its S samples. Then average KL divergence over all K(K-1)/2 paraphrase pairs.
- Entropy regularization: Subtract α times the average entropy of the per-paraphrase distributions.
- Optimization: AdamW with learning rate 5e-5, batch size 16, train for 3 epochs.
- Precomputation: Paraphrases and their embeddings can be computed offline and stored, reducing memory overhead.

## Contribution

(1) PARIS, a self-distillation framework that eliminates the need for a teacher model by enforcing consistency across self-generated paraphrases and sampling seeds, enabling both knowledge transfer and uncertainty estimation from a single model. (2) A novel uncertainty quantification approach that leverages cross-sample disagreement from multiple seeds and paraphrases, producing calibrated uncertainty scores without external reference. (3) The insight that invariance to input paraphrases and sampling stochasticity serves as a proxy for semantic robustness, providing a principled objective for self-supervised distillation.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| ---- | ------ | --------------------- |
| Dataset | MMLU | Standard multi-subject MCQA benchmark. |
| Primary metric | Accuracy | Measures final prediction correctness. |
| Baseline 1 | Vanilla pretrained model | No self-distillation, direct generation. |
| Baseline 2 | Sampling-based semantic clustering | Post-hoc consistency without training. |
| Baseline 3 | Direct finetuning on few-shot examples | Standard supervised baseline. |
| Baseline 4 | PARIS with external paraphrase model (GPT-3.5) | Isolates effect of self-generated paraphrases. |
| Ablation-of-ours | PARIS without entropy regularization | Tests entropy term importance. |

### Why this setup validates the claim
This combination of dataset, baselines, and metric provides a falsifiable test of PARIS's central claim: that enforcing output distribution invariance across paraphrases via self-distillation improves model robustness and accuracy. Using MMLU, a diverse MCQA benchmark, ensures the method must generalize across subjects. The vanilla baseline tests whether training itself adds value over an untrained model. Sampling-based clustering tests whether post-hoc aggregation suffices, isolating the benefit of training. Direct finetuning tests whether self-distillation can match or exceed supervised few-shot learning. The external paraphrase baseline controls for the quality of self-generated paraphrases. The ablation isolates the entropy regularization term. Accuracy directly captures the end-task goal, and improvements should stem from better calibrated, more consistent probability distributions.

### Expected outcome and causal chain

**vs. Vanilla pretrained model** — On an ambiguous question (e.g., "What is the capital of ") where different paraphrases yield different answers due to surface form sensitivity, the vanilla model picks a random answer based on the specific prompt, leading to low accuracy because it lacks invariance. Our method instead enforces consistency across paraphrases during training, pushing the model towards the semantically correct answer cluster. We expect PARIS to achieve higher accuracy on such ambiguous questions, with a noticeable gap (5-10% absolute) on the subset of questions with high paraphrase variance, while matching vanilla on unambiguous ones.

**vs. Sampling-based semantic clustering** — On a question where the model's initial outputs are all wrong but consistently wrong (e.g., a common misconception), post-hoc clustering aggregates errors and still selects a wrong cluster, failing because it cannot correct representational biases. Our method trains the model to produce distributions that align across paraphrases, which indirectly penalizes overconfident but erroneous clusters and encourages exploration of correct alternatives. We expect PARIS to outperform clustering on questions where the vanilla model's answers are confidently wrong, with a gap of 3-8% accuracy, while similar on already correct questions.

**vs. Direct finetuning on few-shot examples** — On a niche subject with few training examples (e.g., "astronomy"), direct finetuning overfits to those examples and performs poorly on held-out test questions due to limited data and lack of domain coverage. Our method generates diverse paraphrases and answer candidates from the model itself, effectively augmenting the training distribution without requiring external data. We expect PARIS to achieve comparable or better accuracy (2-5% higher) on subjects with few few-shot examples, while matching on subjects with ample examples.

**vs. PARIS with external paraphrase model** — If self-generated paraphrases are of lower quality, the external model baseline would yield higher accuracy. We expect PARIS (self-generated) to achieve accuracy within 1-2% of the external model baseline, validating that self-generation does not significantly degrade the invariance signal. If the gap is larger, it suggests the need for the verification filter we introduced.

### What would falsify this idea
If PARIS's accuracy gain is uniform across all questions regardless of paraphrase variance, rather than concentrated on high-variance or low-data subsets, then the central claim of improved invariance is unsupported and the method likely only adds regularization. Similarly, if PARIS without entropy regularization performs identically to full PARIS, the entropy term is unnecessary and the design rationale collapses.

## References

1. Semantic Self-Distillation for Language Model Uncertainty
2. Knowledge distillation and dataset distillation of large language models: emerging trends, challenges, and future directions
3. LLM Distillation for Efficient Few-Shot Multiple Choice Question Answering
4. DiffLM: Controllable Synthetic Data Generation via Diffusion Language Models
5. TAGCOS: Task-agnostic Gradient Clustered Coreset Selection for Instruction Tuning Data
6. FIRST: Teach A Reliable Large Language Model Through Efficient Trustworthy Distillation
7. SEEKR: Selective Attention-Guided Knowledge Retention for Continual Learning of Large Language Models
8. Controlled Text Generation via Language Model Arithmetic
