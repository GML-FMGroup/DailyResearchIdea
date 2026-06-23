# FADE: Unsupervised Detection of Hallucinations in Legal LLMs via Sparse Autoencoder Factuality Directions

## Motivation

Supervised linear probes for hallucination detection (e.g., Real-Time Detection of Hallucinated Entities) require costly token-level labels, while purely unsupervised contrast-consistent search fails because it recovers the most prominent feature (e.g., sentiment) rather than factual content (Challenges with unsupervised LLM knowledge discovery). We need a method that exploits the known linear separability of factual content in hidden states (evidenced in Do I Know This Entity?) without requiring labeled examples.

## Key Insight

Factual content in legal LLM hidden states is encoded as a sparse combination of latent directions that can be isolated via sparse autoencoding trained with contrastive pairs, because the statistical structure of factual vs. hallucinated tokens induces a low-dimensional subspace orthogonal to confounding features.

## Method

(A) **What it is** — FADE (Factuality-Aware Direction Extraction) trains a sparse autoencoder on hidden states from a legal LLM, using contrastive pairs to identify 'factuality' directions. At inference, the projection of each token's hidden state onto these directions yields a real-time hallucination score.

(B) **How it works** — Pseudocode:
```python
# Training
1. Generate responses from legal LLM for a set of prompts.
2. For each prompt, use a verifier (e.g., web search) to label tokens as factual or hallucinated (only for training autoencoder, not for probe training).
3. Extract hidden states from a specific layer (e.g., last layer) for each token.
4. Normalize hidden states to zero mean and unit variance.
5. Train a sparse autoencoder (encoder: linear + ReLU, decoder: linear) with L1 sparsity penalty λ on hidden states from both factual and hallucinated tokens.
   - Objective: minimize reconstruction loss + λ * ||latent activations||_1.
6. After training, compute for each latent neuron its average activation on factual tokens and on hallucinated tokens from the training set.
7. Select as 'factuality directions' the latent neurons where the difference in average activation is largest (top k by absolute difference) and where the direction is consistent across multiple prompts.
8. Define the projection score for a new token's hidden state h as: score = sum_{i in selected neurons} |a_i| * sign(mean_factual_i - mean_hallucinated_i), where a_i is the activation of neuron i for h.

# Inference
9. For each generated token, compute its hidden state, encode with autoencoder, compute projection score.
10. Issue warning if score exceeds threshold τ (set based on desired precision/recall).
```

(C) **Why this design** — We chose a sparse autoencoder over a dense autoencoder because sparsity enforces that each latent neuron captures a distinct feature, reducing interference from confounding variables (trade-off: higher sparsity may discard some factual signal, but we compensate by selecting top k). We used contrastive pairs (factual vs. hallucinated) to guide direction selection, rather than purely unsupervised clustering, because previous work (Challenges with unsupervised LLM knowledge discovery) shows that unsupervised methods default to the most prominent non-knowledge features; by using the contrastive signal, we bias selection toward factual content (trade-off: requires a small amount of labeled data for direction selection, but this is far less than training a full probe). We chose linear encoder/decoder to maintain interpretability and ensure that the directions correspond to the original hidden space geometry; a non-linear autoencoder would create a black-box transformation that obscures the linear separability property (trade-off: linear autoencoder may have higher reconstruction error, but the goal is not reconstruction but direction discovery).

(D) **Why it measures what we claim** — The projection score measures the degree to which a token's hidden state aligns with the 'factuality' direction identified via contrastive pairs. The sparse autoencoder activations measure the presence of underlying features that are commonly activated on factual tokens; the selection of neurons with largest activation difference ensures that we capture features that are discriminative between factual and hallucinated content under the assumption that factual content is linearly separable in the latent space (supported by Do I Know This Entity? and Future Lens). This assumption fails when factual and hallucinated content share similar features (e.g., both are syntactically correct but factually wrong), in which case the projection score reflects surface-level similarity rather than factual correctness. The sign of the projection additionally measures the direction of deviation: positive indicates factual-like, negative indicates hallucinated-like. The threshold τ operationalizes the detection decision; its value is chosen based on a validation set, assuming that training and inference distributions are similar; if the legal domain shifts (e.g., new types of hallucinations), the threshold may need recalibration.

## Contribution

(1) Introduction of FADE, a method that combines sparse autoencoding with contrastive direction selection to identify latent factuality directions in LLM hidden states without supervised probe training. (2) Demonstration that the linear separability of factual content (from Do I Know This Entity?) can be exploited for unsupervised hallucination detection by leveraging contrastive pairs, reducing labeling effort. (3) A practical inference-time pipeline that uses projection onto these directions as a real-time hallucination score, suitable for deployment in legal LLMs.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|----------------------|
| Dataset | Legal Factuality Benchmark | Real legal documents with factuality labels |
| Primary metric | AUC-ROC | Threshold-independent detection ranking performance |
| Baseline 1 | Semantic Entropy | Output-level uncertainty for hallucination |
| Baseline 2 | Logistic Probe | Direct hidden state classification probe |
| Baseline 3 | Direct Contrastive Probe | Mean activation difference without SAE |
| Ablation | Dense Autoencoder | Non-sparse autoencoder for direction discovery |

### Why this setup validates the claim

This setup tests FADE's central claim—that a sparse autoencoder trained with contrastive pairs identifies factuality directions in hidden states, enabling real-time hallucination detection. The legal benchmark provides a realistic domain with known factuality labels. AUC-ROC is chosen because it evaluates ranking quality independent of decision thresholds, directly measuring discriminative power. Semantic Entropy tests whether output-level uncertainty is sufficient, isolating the need for internal state analysis. Logistic Probe tests if a simple supervised classifier on hidden states matches performance, challenging the advantage of unsupervised direction discovery. Direct Contrastive Probe tests the necessity of the autoencoder itself. The dense autoencoder ablation isolates the role of sparsity. Superior performance of FADE over all baselines and the ablation would support the claim that sparse contrastive direction extraction captures distinct factuality features. Conversely, if FADE fails to outperform any baseline, the method's mechanisms are not advantageous.

### Expected outcome and causal chain

**vs. Semantic Entropy** — On a case where the LLM confidently hallucinates a legal entity (e.g., a fake case citation), Semantic Entropy yields low uncertainty because the model assigns high probability to the hallucinated token, failing to detect the error. Our method instead extracts hidden state features learned from contrastive pairs, activating factuality directions even when the output is confident, so we expect FADE to assign high hallucination scores to such tokens while Semantic Entropy misses them. Observable signal: a large AUC-ROC gap on high-confidence hallucination subsets, with FADE near-perfect and SE near-chance.

**vs. Logistic Probe** — On a case where factual and hallucinated tokens share similar surface forms (e.g., both legal terms like "plaintiff" vs. "defendant" in wrong context), a logistic probe trained on hidden states may overfit to correlated but non-factual features, generalizing poorly to unseen examples. Our method uses sparse autoencoder activations that isolate distinct features via contrastive pairs, reducing interference from irrelevant correlations, so we expect better out-of-distribution generalization. Observable signal: FADE maintains high AUC-ROC on held-out legal document types, while logistic probe degrades significantly.

**vs. Direct Contrastive Probe** — On a case where the mean activation difference between factual and hallucinated tokens is noisy due to irrelevant features (e.g., formatting or length differences), direct contrastive probe (using raw mean difference without SAE) produces unreliable direction estimates, leading to high false positive rates. Our method's sparsity selects only the most discriminative latent neurons, filtering noise, so we expect more precise detection. Observable signal: FADE achieves higher precision at low recall thresholds, with fewer false alarms on non-hallucinated tokens.

### What would falsify this idea
If FADE's AUC-ROC gain over the dense autoencoder ablation is negligible or if the improvement is uniform across all hallucination types (e.g., simple misspellings and subtle factual errors alike) rather than concentrated on confident or subtle hallucinations, then the sparsity and contrastive direction selection are not capturing the intended factuality features, and the central claim is wrong.

## References

1. Real-Time Detection of Hallucinated Entities in Long-Form Generation
2. Semantic Entropy Probes: Robust and Cheap Hallucination Detection in LLMs
3. LLM Internal States Reveal Hallucination Risk Faced With a Query
4. FactCheckmate: Preemptively Detecting and Mitigating Hallucinations in LMs
5. Challenges with unsupervised LLM knowledge discovery
6. Future Lens: Anticipating Subsequent Tokens from a Single Hidden State
7. Cognitive Dissonance: Why Do Language Model Outputs Disagree with Internal Representations of Truthfulness?
8. On Early Detection of Hallucinations in Factual Question Answering
