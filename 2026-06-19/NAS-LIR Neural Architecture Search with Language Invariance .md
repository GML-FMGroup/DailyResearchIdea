# NAS-LIR: Neural Architecture Search with Language Invariance Regularization for Cross-Lingual Code Generation

## Motivation

Existing methods for low-resource code generation, such as MultiPL-T (Knowledge Transfer from High-Resource to Low-Resource Programming Languages), focus on data augmentation or fine-tuning but assume that generic LLM architectures are sufficiently flexible to handle structural variation across languages. This assumption remains untested, and no mechanism ensures that learned representations capture language-agnostic program semantics. The forest-level meta-gap is that all current approaches inherit the belief that a one-size-fits-all architecture works for any language, ignoring the need for explicit invariance.

## Key Insight

By incorporating a language invariance regularization term into neural architecture search, we can directly optimize for architectures whose internal representations are insensitive to surface-level syntactic differences across languages, thereby discovering structures that inherently generalize to low-resource languages.

## Method

### NAS-LIR: Neural Architecture Search with Language Invariance Regularization for Cross-Lingual Code Generation

(A) **What it is**: NAS-LIR is a differentiable architecture search framework that augments the standard validation loss with a Language Invariance Regularization (LIR) term. Input: a set of parallel program pairs (semantically equivalent programs in a high-resource language L_H and a low-resource language L_L). Output: a neural architecture (from a defined search space) that minimizes both task loss on L_H and representation distance between L_H and L_L programs.

(B) **How it works** (pseudocode):
```python
# Define search space S (e.g., shared/separate attention heads, language embeddings)
# Initialize architecture parameters alpha (e.g., via DARTS)
# Initialize network weights w (shared)
# Preprocess: filter synthetic parallel pairs by execution equivalence on 10 test inputs
#   Keep only pairs where both programs produce identical outputs.
#   Retain 80% of pairs; discard 20% to reduce semantic noise.
# Additionally, maintain a small calibration set of 200 manually verified parallel pairs.
For each search epoch:
  1. Sample a batch of filtered parallel program pairs (x_H, x_L) from synthetic data
  2. Compute representations:
       r_H = Forward(x_H; w, alpha, language=L_H)
       r_L = Forward(x_L; w, alpha, language=L_L)
  3. Compute task loss L_task = CrossEntropy(r_H, y_H)  # train on L_H only
  4. Compute LIR = CosineDistance(r_H, r_L)  # measure invariance
  5. Validation loss L_val = L_task + lambda * LIR  # lambda=0.1 (hyperparameter)
  6. Update alpha by gradient descent on L_val (second-order approx)
  7. Update w on training loss (first-order) using bilevel optimization
# After search, select final architecture by argmax alpha
```
Hyperparameters: lambda (0.1) balances task performance and invariance; search epochs=50; batch_size=64; learning rate alpha=0.001; execution filter uses 10 test inputs per pair; calibration set size=200.

**Load-bearing assumption**: The parallel program pairs (L_H and L_L) are semantically equivalent, so that minimizing the cosine distance between their representations encourages language invariance without penalizing correct semantic differences. To mitigate potential translation errors, we filter synthetic pairs via execution equivalence on 10 test inputs and manually verify a calibration set of 200 pairs. This ensures the LIR term is computed only on pairs that are verified to be semantically identical.

(C) **Why this design**: We chose differentiable NAS over evolutionary methods because it is faster and differentiable, allowing gradient-based optimization of the invariance term directly. However, differentiable NAS relies on continuous relaxation, which may underestimate discrete architecture trade-offs; we accept potential discretization error. We opted for cosine distance over MMD for LIR because it is computationally cheaper and bounded, though MMD might capture higher-order moments but adds complexity. We use synthetic parallel programs from MultiPL-T style translation; this risks biased representations if translation quality is low, but it provides the necessary paired data that real-world collections lack. The bilevel optimization (updating w on train, alpha on val) separates learning of weights from architecture, preventing overfitting to the invariance regularizer during weight updates. The trade-off is increased computational cost from two separate passes. To reduce cost, we use weight-sharing DARTS, which requires approximately 200 GPU hours on a single NVIDIA A100.

(D) **Why it measures what we claim**: The LIR term, computed as cosine distance between representations of parallel programs, measures cross-lingual invariance because it quantifies how similar the model's internal encoding is for the same program logic expressed in different languages. This operationalizes the concept of language-agnostic representation under the assumption that the parallel programs are indeed semantically equivalent; this assumption fails when translation errors change semantics, in which case the LIR penalizes irrelevant differences or encourages over-invariance to semantic changes. Our filtering step mitigates this failure by retaining only execution-equivalent pairs. The task loss ensures that the architecture retains functional correctness, while LIR pushes it to abstract away language-specific surface forms. The validation loss (L_val) as a whole captures the trade-off between task performance and invariance; optimizing alpha with respect to L_val discovers architectures where invariance is achieved without sacrificing accuracy.

## Contribution

(1) A novel neural architecture search framework (NAS-LIR) that incorporates a Language Invariance Regularization term to explicitly optimize for cross-lingual representation invariance in code generation models. (2) Empirical demonstration that architectures discovered by NAS-LIR achieve significantly higher pass@k scores on low-resource language benchmarks (e.g., Bambara, Ojibwe) compared to generic LLMs and fine-tuned baselines, revealing that architectural modifications can compensate for data scarcity. (3) Analysis identifying which architectural components (e.g., shared early attention heads, language-specific embedding layers) are most important for cross-lingual generalization, providing design principles for multilingual code models.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|-----------------------|
| Dataset | MultiPL-T synthetic pairs + low-resource benchmark (including Kekchi) | Parallel data needed; benchmark for evaluation. Calibration set of 200 manually verified pairs for Kekchi. |
| Primary metric | Functional correctness (pass@1) and bug detection accuracy (F1) | Direct success measure plus practical impact. |
| Baseline | Fine-tuned LLM (supervised on pairs) | Represents standard transfer learning baseline. |
| Baseline | Zero-shot LLM (no adaptation) | Tests initial capability on low-resource. |
| Baseline | NAS without LIR (search only on L_H) | Isolates effect of invariance regularization. |
| Ablation-of-ours | NAS-LIR with MMD instead of cosine | Tests sensitivity of regularization choice. |
| Resource | 200 GPU hours (A100) for NAS | Feasibility estimate for reproducibility. |

### Why this setup validates the claim
This experimental design forms a falsifiable test of the central claim—that augmenting differentiable architecture search with a Language Invariance Regularization (LIR) term yields architectures that produce invariant cross-lingual representations, thereby improving low-resource code generation without sacrificing high-resource performance. The dataset combines synthetic parallel programs (needed for LIR) with a real low-resource benchmark (including Kekchi) and a calibration set of 200 manually verified pairs to mitigate translation errors. Baselines isolate sub-claims: fine-tuned LLM tests if standard supervised transfer achieves invariance; zero-shot tests the inherent gap; NAS without LIR tests whether search alone suffices. The ablation (MMD variant) tests if cosine distance is critical. The primary metric (pass@1) directly measures functional correctness, and the additional bug detection metric (F1) demonstrates practical impact. If LIR is effective, our method must outperform all baselines on low-resource languages, especially those with high syntactic divergence from the high-resource language, and the ablation must show cosine distance is superior. This setup allows unambiguous interpretation: failure to beat a baseline disproves a corresponding sub-claim.

### Expected outcome and causal chain

**vs. Fine-tuned LLM** — On a case where the low-resource language has very different syntax from the high-resource language (e.g., Python vs. Kekchi), the fine-tuned LLM may overfit to surface patterns of the low-resource data because it only sees a small training set. Our method leverages LIR to align representations of equivalent programs, forcing the architecture to abstract away syntactic differences. Thus we expect our method to outperform the fine-tuned baseline by a noticeable margin (e.g., +10-15% pass@1) on languages with high syntactic divergence, and also show higher bug detection F1 (e.g., +8-12%).

**vs. Zero-shot LLM** — On a case where the low-resource language is completely unseen (no examples), zero-shot LLM fails because it cannot generate syntactically valid code due to lack of training data. Our method uses parallel pairs to learn language-invariant features, enabling generalization to the target language. We expect our method to show large gains (e.g., >20% improvement) over zero-shot, which may score near zero on such languages, and similarly for bug detection.

**vs. NAS without LIR** — On a case where the optimal architecture for high-resource language overfits to language-specific patterns (e.g., heavy use of word embeddings for English), NAS without LIR picks an architecture that performs poorly on low-resource because representations are not invariant. Our method's LIR term directly optimizes for invariance, so we expect a significant gap (e.g., 5-10% better) on low-resource, while high-resource performance remains similar.

### What would falsify this idea
If our method shows no improvement over NAS without LIR on low-resource tasks, or if the improvement is uniform across all languages (including high-resource) rather than concentrated on low-resource with high syntactic divergence, then the central claim that LIR induces invariance is wrong. Specifically, if the gain is also seen on high-resource tasks, LIR may merely act as a regularizer rather than an invariance mechanism. Additionally, if the execution filter or calibration set does not prevent degradation from translation errors, and the method underperforms baselines, the load-bearing assumption is violated.

## References

1. No Resource, No Benchmarks, No Problem? Evaluating and Improving LLMs for Code Generation in No-Resource Languages
2. Efficient Model Development through Fine-tuning Transfer
3. Enhancing Code Generation for Low-Resource Languages: No Silver Bullet
4. Knowledge Transfer from High-Resource to Low-Resource Programming Languages for Code LLMs
5. Investigating the Performance of Language Models for Completing Code in Functional Programming Languages: A Haskell Case Study
6. Code Llama: Open Foundation Models for Code
7. Layer Swapping for Zero-Shot Cross-Lingual Transfer in Large Language Models
8. Examining Modularity in Multilingual LMs via Language-Specialized Subnetworks
