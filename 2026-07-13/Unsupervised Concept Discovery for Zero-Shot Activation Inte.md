# Unsupervised Concept Discovery for Zero-Shot Activation Intervention in Language Models

## Motivation

Existing activation intervention methods, such as the self-patching technique in 'Towards Mechanistically Understanding Why Memorized Knowledge Fails to Generalize', require matched correct/incorrect pairs to identify intervention targets. This reliance on curated reference activations limits deployment in open-ended settings where gold-standard correct cases are unavailable. The frontier thus lacks a method for zero-shot intervention that can operate on arbitrary inputs without paired data, a meta-gap that recurs across multiple intervention-based approaches.

## Key Insight

Language models encode concepts as linear directions in activation space (the linear representation hypothesis), and these directions can be extracted from unlabeled data via Independent Component Analysis, which yields statistically independent axes that correspond to causally separable concepts, enabling intervention without paired examples.

## Method

**Unsupervised Concept Activated Patching (UnCAP)**

(A) **What it is:** UnCAP is a zero-shot activation intervention method that learns causal concept directions from unlabeled model activations using ICA, then modifies test-time activations along these directions to steer model behavior without needing matched correct/incorrect pairs. Input: a pre-trained language model, an unlabeled corpus of prompts, a target layer L, a concept label (e.g., “factual knowledge”). Output: a modified activation vector that enhances or suppresses the concept at layer L for a given test input.

(B) **How it works:**
```python
# Phase 1: Unsupervised direction learning
corpus = [...]  # 10,000 diverse unlabeled prompts from OpenWebText
activations = []
for prompt in corpus:
    x = model(prompt, layer=L)  # get hidden state at layer L, L=12 for GPT-2 medium
    activations.append(x)
A = np.array(activations).T  # shape: (d_model, N), d_model=1024, N=10000
A = A - A.mean(axis=1, keepdims=True)  # center
# fastICA with n_components=50, algorithm='parallel', whiten=True, max_iter=500
W = fastICA(A, n_components=50, algorithm='parallel', whiten=True, max_iter=500, random_state=42)
# W: (50, 1024) matrix, each row is a concept direction

# Phase 2: Concept labeling and causal verification (small supervision, 512 labeled examples per concept)
label_prompts = {concept: [positive_examples, negative_examples]}
# e.g., for concept 'capital knowledge': 256 positive prompts containing known capitals, 256 negative not
concept_directions = {}
for concept, (pos, neg) in label_prompts.items():
    pos_acts = [model(p, layer=L) for p in pos]
    neg_acts = [model(p, layer=L) for p in neg]
    pos_mean = np.mean(pos_acts, axis=0)
    neg_mean = np.mean(neg_acts, axis=0)
    scores = abs(W @ (pos_mean - neg_mean))
    best_idx = np.argmax(scores)
    candidate_dir = W[best_idx, :] / np.linalg.norm(W[best_idx, :])
    # Causal verification: on a held-out validation set (128 prompts per concept), intervene with alpha=0.5 and measure change in concept-related logit
    val_logit_change = causal_verification(candidate_dir, concept, val_set)
    if val_logit_change > threshold (0.3):  # threshold trained on synthetic data
        concept_directions[concept] = candidate_dir
    else:
        # Fallback: use contrastive direction from Phase 2 directly (direction = pos_mean - neg_mean)
        concept_directions[concept] = (pos_mean - neg_mean) / np.linalg.norm(pos_mean - neg_mean)

# Phase 3: Zero-shot intervention
# Hyperparameters: alpha=0.5, layer L=12
def intervene(concept, strength=alpha):
    test_act = model(test_prompt, layer=L)  # unmodified
    proj = (concept_directions[concept] @ test_act) * concept_directions[concept]
    modified_act = test_act + alpha * proj
    model.override_activation(layer=L, modified_act)
    final_logits = model.forward_rest()
    return final_logits
```

(C) **Why this design:** We chose ICA over PCA for direction learning because ICA maximizes statistical independence, aligning with the linear representation hypothesis that concepts are independent factors; PCA would yield orthogonal but not necessarily interpretable directions, potentially mixing concepts. The trade-off is that ICA is computationally more expensive and requires careful whitening, but the resulting directions are more causally separable. We chose to label ICA components using a small set of contrastive examples (not paired correct/incorrect for the same input) to bridge the gap between unsupervised components and semantic meaning; this requires a small labeled set, but it is far less restrictive than needing matched pairs for every test input. The alternative of using random component selection would not guarantee targeting the desired concept, so the minimal supervision is a necessary cost. We set α as a hyperparameter based on validation set performance; a fixed α avoids per-input tuning but risks over- or under-shifting, which we accept for simplicity and zero-shot applicability. Finally, we intervene only at a single layer L (chosen via prior analysis of where the concept is encoded) to avoid cascade effects; intervening on multiple layers could amplify noise but might be needed for stronger effects—we leave multi-layer extension for future work.

(D) **Why it measures what we claim:** The projection of test-time activation onto the ICA component (computational quantity: `proj` = (direction · activation)) measures **concept presence** because we assume that the ICA component's direction is the linear embedding axis along which the model encodes that concept—an assumption supported by the linear representation hypothesis; this assumption fails when the concept is nonlinearly entangled with other features (e.g., a factual fact that depends on contextual interactions), in which case the projection reflects a mixture of correlated factors rather than a pure concept. The shift magnitude α controls **intervention strength**, which under the assumption of linear additivity directly maps to the change in concept evidence; failure of additivity (e.g., when activation space is curved) would cause the shift to produce unintended side effects rather than a monotonic concept change. The use of contrastive examples to label components (via `pos_mean - neg_mean` dot product) ensures that the chosen ICA component corresponds to the intended concept, under the assumption that the concept direction is the direction along which positive and negative examples maximally differ; this fails if the difference is contaminated by spurious correlations, in which case the labeled component may encode confounders instead of the target concept. The causal verification step (checking intervention effect on a held-out validation set) measures **causal relevance** because it directly tests whether shifting along the candidate direction changes the concept-related output; this fails if the validation set is not representative of test-time distribution, leading to overconfident selection.

**Load-bearing assumption:** ICA of unlabeled activations yields independent components that correspond to causally separable concepts, enabling zero-shot intervention via linear projection. This assumption is empirically verified via causal verification on a held-out validation set (128 prompts per concept) – only components that cause a significant change in concept-related logit (threshold 0.3) are retained; otherwise, a fallback contrastive direction is used. This verification mitigates the risk of components being statistically independent but not causally relevant.

**Resource estimates:** Phase 1 requires ~10 minutes on a single A100 GPU for 10,000 prompts (GPT-2 medium). Phase 2 labeling and verification requires ~2 minutes per concept (512+128 prompts). Phase 3 inference time is identical to standard forward pass plus a negligible vector operation. Total compute: ~15 GPU-hours for all experiments.

## Contribution

(1) A novel method (UnCAP) for zero-shot activation intervention that learns causal concept directions from unlabeled activations via ICA, eliminating the need for paired correct/incorrect cases. (2) A design principle that unsupervised independence-seeking decomposition (ICA) can identify linear concept axes that generalize across inputs, validated by the linear representation hypothesis of language models. (3) An algorithm for labeling ICA components with minimal supervision that bridges unsupervised directions to semantic concepts while still avoiding the heavy paired-data requirement of self-patching.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|-----------------------|
| Dataset | MMLU (subset of 1,000 factual knowledge questions: capitals, presidents, geography) | Assess concept steering on known facts. |
| Primary metric | Accuracy on factual question answering | Direct measure of concept manipulation. |
| Secondary metric | Average change in correct answer logit (Δlogit) | Quantifies intervention effect size. |
| Baseline 1 | Standard forward pass (no intervention) | Null hypothesis: method should outperform no steering. |
| Baseline 2 | Random direction intervention (random unit vector at layer L) | Tests necessity of data-driven direction. |
| Baseline 3 | PCA-based intervention (same labeling with PCA components instead of ICA) | Tests ICA superiority over orthogonal directions. |
| Ablation (ours) | UnCAP with PCA instead of ICA (same causal verification step) | Isolates benefit of independence assumption. |

### Why this setup validates the claim
This setup provides a falsifiable test of UnCAP's central claim: that unsupervised ICA directions, labeled with a small contrastive set and causally verified, enable effective zero-shot causal concept steering. MMLU's factual questions are ideal because they rely on a single clear concept (e.g., capital knowledge) that can be isolated. Accuracy directly reflects whether intervention enhances correct concept evidence. The secondary metric Δlogit (logit of correct token minus logit of incorrect token, averaged over questions) provides a continuous measure of intervention effect. Baselines test sub-claims: No intervention checks null effect; random direction checks whether any direction works (if it does, the method's data-driven learning is unnecessary); PCA checks whether orthogonal (but not necessarily independent) directions suffice. The ablation using PCA isolates whether ICA's independence property yields better separation. The causal verification step ensures that only causally relevant directions are used, directly addressing the load-bearing assumption.

### Expected outcome and causal chain

**vs. No intervention** — On a factual question like "What is the capital of France?", the baseline model may produce incorrect answer (e.g., "Paris" is correct, but suppose it outputs "Lyon") due to context noise. UnCAP enhances the "capital knowledge" concept activation at layer L, shifting the hidden state toward the correct cluster, so it outputs "Paris". We expect a noticeable accuracy gap: at least 10 percentage points improvement on factual questions where the model initially errs (initial accuracy < 70%). Δlogit should increase by at least 0.5 on average.

**vs. Random direction intervention** — On the same question, random direction adds a random vector to the activation, likely amplifying unrelated features and causing erratic outputs (e.g., "France" or gibberish). UnCAP's learned direction aligns with the concept axis, so the intervention is targeted and effective. We expect UnCAP accuracy to be significantly higher (e.g., >20 points) on the held-out test set, while random direction may even hurt performance (accuracy drop of 5-10 points). Δlogit for random direction should be near zero or negative.

**vs. PCA-based intervention** — On a question involving a concept that is nonlinearly entangled, such as distinguishing between capital of Germany and a city population, PCA directions (orthogonal but not independent) mix these features. UnCAP with ICA separates them, so intervention suppresses the wrong concept. We expect UnCAP to outperform PCA by 5-10 points on subsets where concept entanglement is high (e.g., multiple factual relations per prompt). Δlogit for PCA should be lower than UnCAP by at least 0.2.

### What would falsify this idea
If UnCAP achieves no better accuracy than random direction intervention on any subset, or if its advantage over PCA is uniform across all questions (not concentrated on entangled subsets), then the central claim that ICA-based direction learning is beneficial would be falsified. Additionally, if the causal verification step fails to identify useful components (i.e., all candidates fail threshold), the method would collapse to the fallback, indicating that the assumption does not hold for the given model and concept.

## References

1. Towards Mechanistically Understanding Why Memorized Knowledge Fails to Generalize in Large Language Model Finetuning
2. Hopping Too Late: Exploring the Limitations of Large Language Models on Multi-Hop Queries
3. Knowledge Circuits in Pretrained Transformers
4. How much do language models memorize?
5. Just How Flexible are Neural Networks in Practice?
6. Physics of Language Models: Part 3.3, Knowledge Capacity Scaling Laws
7. Scaling Laws for Fact Memorization of Large Language Models
8. Memory Injections: Correcting Multi-Hop Reasoning Failures During Inference in Transformer-Based Language Models
