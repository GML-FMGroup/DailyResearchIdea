# Unsupervised Detection of Logical Hallucinations via Deductive Consistency Probing

## Motivation

Existing hallucination detection methods focus on entity factual errors and rely on supervised data or external knowledge, as seen in Real-Time Detection of Hallucinated Entities and Semantic Entropy Probes. However, legal hallucinations often involve logical inconsistencies—such as contradicting premises or invalid deductions—that escape token-level or uncertainty-based detectors. This structural gap arises because these methods assume the primary hallucination type is factual entity errors, ignoring the logical structure central to legal reasoning.

## Key Insight

The hidden state representation of logically related statements deviates systematically when the model's reasoning is inconsistent, enabling unsupervised detection via a probe trained on automatically generated logical constraints.

## Method

**(A) What it is:** Deductive Consistency Probe (DCP) is a lightweight linear classifier that takes as input the difference of LLM hidden-state vectors for two logically related statements and outputs a binary label indicating whether they are consistent (i.e., the LLM's reasoning is deductively closed).

**(B) How it works:**
```python
# Pseudocode for training DCP
Input: Frozen LLM, set of legal documents D
Hyperparameters: target layer L (last layer), rule set R = {contrapositive, De Morgan, modus ponens}

Training:
1. For each document d in D:
   a. Generate factual statement s = LLM.answer(query about d)
   b. For each rule r in R:
        - Apply r to s to produce logically related statement s' (e.g., contrapositive)
        - Ensure s' is grammatical via LLM paraphrase if needed
        - Extract hidden states h = LLM.hidden(s, layer=L) and h' = LLM.hidden(s', layer=L)
        - Compute delta = h - h'
        - Label y = 1 if (s, s') are consistent (e.g., both true given r), else 0
   c. Add (delta, y) to training set
2. Train linear classifier f(delta) via logistic regression with L2 regularization

Inference for new query:
1. Generate answer a from LLM
2. Sample K=5 logical variants v_k of a using R (same process as step 1b)
3. For each variant, compute delta_k = h_a - h_vk
4. Compute average consistency score = (1/K) sum_k f(delta_k)
5. If score < threshold (selected via held-out validation set), flag a as hallucinated
```

**(C) Why this design:** We chose a linear probe over a neural network because it enforces interpretability of the decision boundary—necessary for understanding which logical relations drive detection—and avoids overfitting to spurious correlations in the small automatic training set. The trade-off is limited capacity to capture non-linear interactions between representation differences, which may miss subtle inconsistencies. We opted for predefined logical rules (contrapositive, De Morgan, modus ponens) over free-form LLM-generated logical variants because rule-based generation provides ground-truth labels without ambiguity, ensuring the training signal is tied to logical structure rather than surface statistics. This comes at the cost of coverage: rules may not capture all forms of legal reasoning (e.g., case-based analogies). Finally, we train on pairs from the same document to control for topic drift, accepting domain specificity that limits zero-shot transfer to new legal domains without retraining.

**(D) Why it measures what we claim:** The computational quantity delta = h - h' measures the distance between the LLM's internal representations of two statements that should be logically equivalent or contradictory. This operationalizes the concept of deductive closure because we assume that a deductively closed reasoner assigns similar hidden states to logically equivalent statements and distinct states to contradictory ones; this assumption fails when the LLM relies on lexical overlap (e.g., synonyms) rather than logical inference, in which case delta reflects surface similarity instead. The label y = consistency (based on rule r) ensures the probe learns the mapping from representation differences to logical validity, assuming that the chosen rule set accurately reflects the relevant deduction rules for legal reasoning; this assumption fails when the rule set is incomplete, causing the probe to misclassify valid but unmodeled inferences as inconsistent (or vice versa). The use of multiple rules during inference averages over individual rule failures, reducing the impact of any single assumption's violation.

## Contribution

(1) A novel unsupervised method for detecting logical hallucinations by probing the consistency of LLM hidden states across automatically generated logical transformations. (2) A self-supervised data generation pipeline that creates training pairs using deterministic logical rules, eliminating the need for human-labeled hallucination data. (3) An empirical finding that the difference in hidden-state representations between logically related statements correlates with logical consistency, enabling detection of non-entity factual errors in legal text.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | LegalBench hallucination subset | Real legal queries with gold labels |
| Primary metric | AUC-ROC | Threshold-free ranking metric |
| Baseline 1 | Semantic Entropy | Uncertainty via output consistency |
| Baseline 2 | Semantic Entropy Probes | Learned probe on entropy stats |
| Baseline 3 | Direct Hidden State Probe | Simple probe on layer output |
| Ablation-of-ours | DCP-single-rule | Uses only modus ponens rule |

### Why this setup validates the claim

This experimental design tests whether DCP's logical consistency signal outperforms existing hallucination detectors on legal reasoning. The LegalBench dataset provides realistic legal queries with known truths, ensuring the task is non-trivial and domain-specific. AUC-ROC is chosen because it measures ranking ability across thresholds, critical for deployment. The three baselines represent distinct families: Semantic Entropy (output-level uncertainty), Semantic Entropy Probes (learned uncertainty features), and Direct Hidden State Probe (raw internal states). The ablation (single-rule DCP) isolates the contribution of multiple logical rules. Together, these allow us to isolate whether DCP's success stems from the logical delta representation, the multi-rule averaging, or both. If DCP outperforms all baselines, it confirms that deductive consistency is a better predictor of hallucination than confidence or raw states. If the ablation matches full DCP, multiple rules are unnecessary; if it falls short, diversity matters.

### Expected outcome and causal chain

**vs. Semantic Entropy** — On a case where the LLM confidently produces a plausible but logically inconsistent statement (e.g., "The contract is valid because it was signed" but the contract requires a witness signature), Semantic Entropy sees low uncertainty because the LLM repeats similar phrasing across samples, so it fails to flag the hallucination. DCP encodes the logical implication "if valid then signed" and its contrapositive; the hidden-state difference between the answer and its contrapositive reveals inconsistency because the LLM's internal representation does not obey logical equivalence. We expect AUC-ROC gap of ≥0.10 favoring DCP on such logically anomalous instances.

**vs. Semantic Entropy Probes** — On a case where the LLM is uncertain but correct (e.g., it hedges with "possibly, but..." for a true statement), Semantic Entropy Probes (trained on entropy features) may incorrectly flag the statement as hallucinated due to high uncertainty, causing false positives. DCP does not rely on uncertainty; it checks logical consistency with variants. The correct answer will have consistent delta vectors (since the rule holds), so DCP scores it as non-hallucinated. We expect DCP to have lower false positive rate (e.g., 5% vs. 15% at fixed recall) on high-uncertainty correct answers.

**vs. Direct Hidden State Probe** — On a case where the LLM internally encodes a misleading correlation (e.g., topic words like "statute" appear in both correct and hallucinated answers), a Direct Probe trained on raw hidden states might pick up spurious patterns. DCP instead uses the difference of hidden states between logically related sentences, cancelling out topic-specific features and isolating logical structure. So a Direct Probe may achieve high accuracy on in-distribution but lose to DCP on out-of-distribution examples where surface correlations change. We expect DCP's OOD generalization (e.g., on unseen legal subdomain) to be ≥0.07 AUC-ROC higher than Direct Probe.

### What would falsify this idea

If the full DCP does not outperform the single-rule ablation (i.e., adding more rules does not improve detection), then the causal role of logical diversity is unsupported. Alternatively, if DCP's AUC-ROC is not significantly above Direct Hidden State Probe on logical inconsistency cases, then the delta representation adds nothing beyond raw hidden states.

## References

1. Real-Time Detection of Hallucinated Entities in Long-Form Generation
2. Semantic Entropy Probes: Robust and Cheap Hallucination Detection in LLMs
3. LLM Internal States Reveal Hallucination Risk Faced With a Query
4. FactCheckmate: Preemptively Detecting and Mitigating Hallucinations in LMs
5. Challenges with unsupervised LLM knowledge discovery
6. Future Lens: Anticipating Subsequent Tokens from a Single Hidden State
7. Cognitive Dissonance: Why Do Language Model Outputs Disagree with Internal Representations of Truthfulness?
8. On Early Detection of Hallucinations in Factual Question Answering
