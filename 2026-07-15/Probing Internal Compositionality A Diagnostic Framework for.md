# Probing Internal Compositionality: A Diagnostic Framework for Multimodal Model Evaluation

## Motivation

Existing benchmarks like Blind-Spots-Bench (Gu et al., 2024) assess multimodal models solely through output-level accuracy, leaving internal representations unexamined. This fails to reveal whether models truly compose visual and textual components or rely on spurious correlations. For instance, a model may correctly answer a visual question by attending only to the text or only to the image, a failure mode that output metrics alone cannot detect.

## Key Insight

The discrepancy between per-component probe accuracy (how linearly decodable each task-relevant factor is from hidden states) and output accuracy directly reveals the compositionality gap: if each factor is individually decodable but the final answer is wrong, the model fails to integrate them.

## Method

We propose the **Compositional Probe Analysis (CPA)** framework.

(A) **What it is**: CPA trains linear classifiers (logistic regression probes) on intermediate hidden representations of a multimodal model to predict task-relevant visual and textual latent factors. The probe accuracies are compared to the model's output accuracy to derive a compositionality score. Input: model's hidden states from a chosen layer; output: diagnostic report of per-component encoding strength and compositionality metric.

(B) **How it works** (pseudocode):
```python
def CPA(model, dataset, layer_index, pool_mode='cls', C=1.0):
    # dataset has (image, text) inputs, ground-truth answer, and annotated latent factors
    factor_names = ['visual_obj', 'spatial_rel', ...]  # e.g., object presence (binary), relation type (categorical)
    # Initialize probes: linear classifiers with L2 regularization (C=1.0)
    probes = {f: LogisticRegression(C=C, max_iter=1000) for f in factor_names}
    
    # Collect hidden states and factor labels
    all_hiddens = []
    all_factor_labels = []
    for (img, txt), ans in dataset:
        hidden = model.get_layer_output(layer_index, img, txt)  # shape (seq_len, dim)
        if pool_mode == 'cls':
            pooled = hidden[0, :]  # assume CLS token at index 0
        elif pool_mode == 'mean':
            pooled = hidden.mean(axis=0)  # mean over token dimension
        else:
            pooled = hidden.mean(axis=0)  # default mean
        factor_labels = dataset.get_factors_for_sample(img, txt)  # dict of factor->label
        all_hiddens.append(pooled)
        all_factor_labels.append(factor_labels)
    X = np.stack(all_hiddens)
    
    # Train each probe
    for f in factor_names:
        y = np.array([labels[f] for labels in all_factor_labels])
        probes[f].fit(X, y)
    
    # Evaluate probe accuracy per factor
    probe_acc = {}
    for f in factor_names:
        y_pred = probes[f].predict(X)
        y_true = np.array([labels[f] for labels in all_factor_labels])
        probe_acc[f] = accuracy_score(y_true, y_pred)
    
    # Compute output accuracy
    output_acc = compute_output_accuracy(model, dataset)  # standard answer accuracy
    
    # Compute compositionality score: gap between the worst-encoded factor and output
    # Hyperparameter: aggregation function 'min' emphasizes the weakest component
    comp_score = min(probe_acc.values()) - output_acc
    
    return {'probe_accuracies': probe_acc, 'output_accuracy': output_acc, 'compositionality_score': comp_score}
```

(C) **Why this design**: We chose linear probes over nonlinear probes (e.g., MLP) because linearity forces interpretation of representation directions, directly indicating whether factor information is linearly separable; the trade-off is that some factors may be encoded nonlinearly, causing probe accuracy to underestimate encoding — but this is diagnostically useful as it reveals the model's reliance on linear mechanisms. We selected logistic regression over SVM for its probabilistic outputs and efficiency, though SVM might be more robust to outliers. We fixed pooling to CLS token (or average over text tokens) to get a single vector; this loses spatial information but simplifies analysis, and we rely on the model's cross-attention to bring relevant information to the pooled representation. The compositionality score uses the minimum probe accuracy across factors, based on the assumption that the weakest component limits composition; alternative aggregations (e.g., mean) would dilute diagnosis of a single failed factor. We chose L2 regularization (C=1.0) as a standard default to prevent overfitting on small datasets, accepting that too much regularization might suppress genuine linear structure.

**Load-bearing assumption:** Linear probe accuracy on hidden representations directly measures whether a factor is encoded and usable for composition, such that a gap with output accuracy implies composition failure. This assumption is explicit and will be verified by including a nonlinear probe baseline.

(D) **Why it measures what we claim**: The probe accuracy for a factor (e.g., 'object_present') measures the extent to which the model's hidden representation linearly encodes that factor, under the assumption that linear separability implies the model explicitly represents that information; this assumption fails when the factor is encoded nonlinearly but still used by the model — in that case, low probe accuracy would misclassify the representation as not encoding the factor, yielding a false positive on compositionality failure. The compositionality score (min(probe_acc) - output_acc) measures the gap between the worst-encoded factor's linear decodability and the final answer correctness; this gap is interpreted as the cost of composition: if the model can linearly decode each factor well but still gets the answer wrong, it likely fails to compose them. This interpretation relies on the assumption that correct composition requires both factors to be present in the output; if the answer can be determined from one factor alone (e.g., always 'yes'), the score may be misleading. To mitigate, the dataset must require joint reasoning (e.g., spatial relations). Line 4 in (C) explicitly states the load-bearing assumption. The nonlinear probe baseline in the experiment serves as a check: if the compositionality gap persists when using nonlinear probes, it confirms the gap is due to composition failure rather than linear probe blindness.

## Contribution

(1) A diagnostic framework (CPA) that integrates linear probing of internal representations with output-level evaluation to assess compositionality in multimodal models. (2) A concrete compositionality metric derived from the gap between per-factor probe accuracies and output accuracy, revealing where models fail to compose visual and textual components. (3) [Optional] An analysis protocol applicable to any multimodal model and task to generate fine-grained diagnostic reports.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | CLEVR | Compositional visual reasoning with known factors. |
| Primary metric | Compositionality gap (min(probe_acc) - output_acc) | Directly measures composition failure cost. |
| Baseline | Standard accuracy-only | No probing; cannot diagnose factor encoding. |
| Baseline | Nonlinear probe (MLP, 2 layers, hidden=256, ReLU) | Tests necessity of linear probes. |
| Ablation | Mean aggregation for compositionality score | Tests sensitivity to aggregation method. |
| Synthetic experiment | Artificially disrupt compositionality by swapping factor labels during training | Verify that gap increases only when composition fails. |
| Parallelization | Probe training parallelized across 4 GPUs using joblib | Feasible for models up to 7B parameters. |

### Why this setup validates the claim

This setup forms a falsifiable test of the claim that CPA can diagnose compositionality failures. CLEVR requires joint reasoning over objects and spatial relations, making the task compositional. The primary metric, compositionality gap, quantifies the cost of composition: if probe accuracies for individual factors are high but output accuracy is low, the model fails to compose. Standard accuracy-only baseline cannot reveal this internal failure. The nonlinear probe baseline tests whether linear probes are necessary to detect compositionality gaps; if nonlinear probes also show high accuracies on composition-failure cases, linear probes may miss nonlinearly encoded factors, but our method would still reveal a gap if those factors are not linearly decodable. The ablation (mean aggregation) tests the assumption that the weakest factor limits composition; if mean yields a gap similar to min, the choice of aggregation is less critical. The synthetic experiment with disrupted compositionality ensures that the gap increases only when actual composition fails, not due to artifacts. Parallelization ensures feasibility for large models. The combination ensures that any observed gap is attributable to composition failure rather than missing factors or aggregation artifacts.

### Expected outcome and causal chain

**vs. Standard accuracy-only** — On a case where CLEVR requires reasoning about an object's relative position (e.g., "the cube is left of the sphere") but the model gets the answer wrong, standard accuracy-only shows low accuracy without explanation. Our method measures high probe accuracies for object presence and relation type individually, but low output accuracy, producing a positive compositionality gap. We expect a noticeable gap on such failure instances (e.g., gap > 0.2) while accuracy-only remains uniformly low across all subsets.

**vs. Nonlinear probe (MLP)** — On the same composition-failure case, a nonlinear probe trained on hidden states to predict the same factors might achieve high accuracy (e.g., 0.95) because it can use nonlinear feature combinations, masking the compositionality issue. Our linear probe shows lower probe accuracy (e.g., 0.70) for the relation factor because the representation is not linearly separable, producing a compositionality gap. We expect linear probe to yield a larger gap than nonlinear on samples with nonlinearly encoded factors, while nonlinear probe shows minimal gap.

**vs. Ablation (mean aggregation)** — On a sample where one factor (e.g., object presence) is perfectly linearly decodable (probe_acc=0.99) but another (relation) is poorly decodable (0.60), the min aggregation gives compositionality gap = 0.60 - output_acc, while mean gives (0.99+0.60)/2 - output_acc = 0.795 - output_acc. If output_acc=0.80, min gap = -0.20 (negative, suggesting composition success) but actual output is low relative to factor decodability; mean gap = -0.005 (near zero). We expect min to produce a more pronounced negative gap, correctly flagging the weakest factor, while mean dilutes the signal.

**vs. Synthetic disruption** — In a CLEVR variant where during training we randomly swap 50% of object-color labels (disrupting composition), we expect the compositionality gap on vanilla CLEVR test set to be significantly larger than in the original training condition (e.g., mean gap increase > 0.15), while output accuracy decreases. This confirms the gap measures composition failure, not artifacts.

### What would falsify this idea
If the compositionality gap is zero or invariant across all samples (regardless of actual composition success or failure), or if it consistently shows a gap on samples where the model composes correctly (false positives), then the central claim that CPA diagnoses compositionality failures is wrong. Additionally, if the gap does not increase in the synthetic disrupted-composition experiment, the assumption linking linear probe accuracy to composition fails.

## References

1. Blind-Spots-Bench: Evaluating Blind Spots in Multimodal Models
2. SIV-Bench: A Video Benchmark for Social Interaction Understanding and Reasoning
