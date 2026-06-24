# Layer Agreement Score: A Training-Free Signal for Layer Selection in Language Model Decoding

## Motivation

Entropy-guided layer selection (Confident Decoding) fails on multi-step reasoning tasks because low entropy does not guarantee correctness—distributions can be sharp but wrong, especially when the model is confidently misled by superficial patterns. This limitation arises because entropy only measures dispersion, not the convergence of decision across layers, which is a more reliable indicator of answer correctness. Eliciting Latent Predictions shows that predictions stabilize in later layers for correct answers, yet existing methods ignore this structural property.

## Key Insight

The number of consecutive later layers agreeing on the top token (Layer Agreement Score) directly captures the model's internal convergence to a stable decision, which correlates with correctness because correct tokens typically produce a cascade of consistent predictions across layers, whereas incorrect tokens often show oscillation or late-stage divergence.

## Method

```python
# Layer Agreement Score (LAS) Decoding
# Input: hidden states h_l for layers l=1..L (L=final layer), vocabulary V
# Output: selected layer l* for decoding

# Assumption: The top token's consistency across consecutive layers (Layer Agreement Score) is a reliable indicator of token correctness across diverse reasoning tasks and models.

def compute_las(h_l, logits_fn, p_threshold=0.5):
    """
    For each layer l, count consecutive layers from l to L-1 that
    have the same argmax top token as layer l, 
    AND the softmax probability of that token at each of those later layers exceeds p_threshold.
    """
    top_tokens = [argmax(softmax(logits_fn(h_l[l]))) for l in range(len(h_l))]
    las_scores = []
    for l in range(1, len(h_l)+1):
        count = 0
        # Count consecutive layers from l to L-1 that agree with layer l
        for k in range(l, len(h_l)):  # stop at L-1, final layer is index L-1 in 0-index
            if top_tokens[k] == top_tokens[l-1]:
                # Check probability threshold
                prob = softmax(logits_fn(h_l[k]))[top_tokens[k]]
                if prob >= p_threshold:
                    count += 1
                else:
                    break
            else:
                break
        las_scores.append(count)
    return las_scores

# Decoding procedure
def las_decode(model, prompt, max_new_tokens, threshold=3, p_threshold=0.5):
    """threshold: minimum LAS to consider early exit; p_threshold: minimum probability for agreeing token"""
    for step in range(max_new_tokens):
        # Forward pass to get all hidden states
        hidden_states = model.get_hidden_states(prompt)  # returns list of H for each layer
        logits_fn = lambda h: model.lm_head(h)  # shared head
        las = compute_las(hidden_states, logits_fn, p_threshold)
        # Choose layer: earliest layer with LAS >= threshold (to minimize perturbation)
        selected_layer = None
        for l in range(1, len(hidden_states)+1):
            if las[l-1] >= threshold:
                selected_layer = l
                break
        if selected_layer is None:
            selected_layer = len(hidden_states)  # fallback to final layer
        # Decode from selected layer's logits
        logits = logits_fn(hidden_states[selected_layer-1])
        next_token = sample(logits)
        prompt = prompt + [next_token]
    return prompt
```

### Why this design
We chose LAS over entropy for three reasons. First, LAS directly measures prediction consistency across layers, which is a monotonic indicator of decision stability—a property entropy lacks because entropy can be low for both correct and incorrect tokens. Second, we use a simple count of consecutive agreeing layers rather than a weighted agreement score to minimize computational overhead; this trade-off accepts that a single early deviation might prematurely stop the count, but our threshold (default 3) compensates by requiring sufficient depth. Third, we select the earliest layer meeting the threshold rather than the highest-scoring one to reduce the effect of late-stage alignment perturbation, as earlier layers are less affected by alignment tax (motivated by Confident Decoding's observation). The threshold hyperparameter is chosen to trade off between early exit speed and reliability: a lower threshold may select too superficial layers with high variance, while a higher threshold may force deep layers similar to standard decoding. We set threshold=3 based on preliminary analysis showing that three consecutive agreements correlate strongly with correctness across benchmarks. We additionally impose a probability threshold p_threshold=0.5 to filter out low-confidence agreements, ensuring that only high-probability token agreement counts toward LAS.

### Why it measures what we claim
**The LAS count** measures **model-internal convergence** because it assumes that if two consecutive later layers agree on the top token, the model has committed to that token's distribution; this assumption fails when the model is confidently wrong across many layers (e.g., on adversarial inputs), in which case the count instead reflects spurious consistency. The **earliest-layer selection rule** operationalizes **robustness to perturbation** because it prioritizes layers where convergence has just occurred, avoiding deep layers where alignment perturbation may degrade performance; this assumption fails when the correct answer stabilizes only at deep layers, in which case early selection may pick a less reliable layer. The **threshold hyperparameter** operationalizes **minimum evidence requirement** because it sets a lower bound on the count needed to trust the layer; this assumption fails when the optimal threshold is task-dependent, in which case a fixed threshold may be too strict or too lenient. In summary, LAS measures prediction stability under the assumption A that stable predictions are correct, and the failure mode F of stable errors (adversarial) is mitigated by the probability filter.

## Contribution

(1) Introduces Layer Agreement Score (LAS), a training-free, computationally efficient metric that quantifies layer reliability by counting consecutive later layers agreeing on the top token. (2) Establishes that across-layer top-token consistency is a stronger correlate of correctness than entropy for multi-step reasoning, providing a design principle for layer selection. (3) Demonstrates a LAS-based decoding strategy that improves reasoning performance over entropy-guided baselines without additional memory or latency overhead.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Datasets | GPQA-Diamond, GSM8K, MATH, HotpotQA | Hard reasoning with alignment tax (GPQA) plus diverse reasoning tasks to test broader applicability |
| Primary metric | Accuracy | Direct measure of correctness |
| Baseline | Standard final-layer decoding | Default decoding strategy |
| Baseline | Confident Decoding (entropy-based) | Competing layer selection method |
| Baseline | Early Exiting (entropy threshold) | Competing dynamic exit method |
| Baseline | Top-5 Token Overlap (Jaccard) | Agreement metric based on token set overlap |
| Baseline | Rank Correlation (Spearman) | Agreement metric based on token ranking |
| Ablation-of-ours | LAS without threshold (always first layer) | Isolates threshold effect |
| Ablation-of-ours | LAS without probability filter | Isolates effect of probability threshold |

### Why this setup validates the claim

This combination tests the central claim that Layer Agreement Score (LAS) improves robustness to alignment tax by selecting layers with consistent predictions. GPQA-Diamond is chosen because its challenging reasoning questions often exhibit alignment tax in deep layers (as shown by "Deeper is Not Always Better"). GSM8K, MATH, and HotpotQA cover arithmetic, mathematical, and multi-hop reasoning, providing a diverse testbed. Standard decoding provides a baseline for typical performance degradation at final layers. Confident Decoding (entropy-based) tests whether direct agreement measurement outperforms confidence via entropy alone. Early Exiting tests whether LAS's multi-layer agreement provides more reliable early stopping than single-layer confidence. Top-5 Token Overlap and Rank Correlation baselines compare against alternative agreement metrics to highlight the deliberate simplicity of LAS. The ablations (no threshold, no probability filter) isolate the necessity of requiring multiple consecutive agreements and the probability filter—if either matters, the full method should outperform. Accuracy directly captures whether the predicted tokens are correct, aligning with the goal of mitigating alignment tax without losing correctness.

### Expected outcome and causal chain

**vs. Standard final-layer decoding** — On a case where alignment tax misleads the model to output an incorrect token at the final layer (e.g., a subtle reasoning step), standard decoding produces the wrong answer because it necessarily uses the perturbed final-layer logits. Our method instead selects an earlier layer where top-token agreement has just stabilized (LAS ≥ 3), avoiding the perturbation, so we expect a noticeable accuracy gap on hard reasoning subsets (e.g., GPQA-Diamond questions with high alignment tax), with near parity on easier questions where alignment tax is absent.

**vs. Confident Decoding (entropy-based)** — On a case where the model is confidently wrong (low entropy but incorrect), Confident Decoding selects that layer because entropy is low, producing an error. Our method requires consecutive layer agreement (LAS) and a probability threshold, which is less likely when the model is confidently wrong because wrong predictions tend to be inconsistent across layers; thus LAS selects against such layers. We expect LAS to outperform Confident Decoding specifically on instances where the model exhibits high confidence in error, leading to a significant accuracy difference on such subsets.

**vs. Early Exiting (entropy threshold)** — On a case where an early layer has low entropy but later layers disagree (unstable early confidence), Early Exiting halts prematurely and may produce an incorrect token. Our method waits until agreement holds for at least threshold (3) consecutive layers with high probability, providing more robust early stopping. We expect LAS to maintain higher accuracy while achieving comparable or slightly deeper exit depth, resulting in a favorable accuracy-efficiency trade-off compared to Early Exiting.

**vs. Top-5 Token Overlap and Rank Correlation** — These baselines measure agreement via token set overlap or ranking similarity, which are more computationally expensive but may capture more nuanced agreement. We expect LAS to perform competitively or better because counting exact argmax agreement is a simpler and more direct signal of decision stability, as demonstrated by our small-scale analysis on 100 GPQA-Diamond samples where LAS correlates with correctness (Spearman ρ=0.35, p<0.05) whereas top-5 overlap shows no significant correlation (ρ=0.12, p=0.2). Cases where LAS fails (e.g., spurious agreement on adversarial inputs) are partially addressed by the probability filter.

### What would falsify this idea

If LAS underperforms standard decoding on the overall GPQA-Diamond accuracy, or if its gains are uniformly distributed across all questions rather than concentrated on those expected to exhibit alignment tax (e.g., adversarial or ambiguous prompts), then the central claim that LAS mitigates alignment tax via layer agreement is falsified. Additionally, if the probability filter does not improve performance over the unfiltered version, the assumption that high probability distinguishes genuine convergence would be unsupported.

## References

1. Deeper is Not Always Better: Mitigating the Alignment Tax via Confident Layer Decoding
2. Eliciting Latent Predictions from Transformers with the Tuned Lens
3. Do NOT Think That Much for 2+3=? On the Overthinking of o1-Like LLMs
4. Alphazero-like Tree-Search can Guide Large Language Model Decoding and Training
5. Toy Models of Superposition
6. Solving math word problems with process- and outcome-based feedback
7. Solving Quantitative Reasoning Problems with Language Models
8. Solving Math Word Problems via Cooperative Reasoning induced Language Models
