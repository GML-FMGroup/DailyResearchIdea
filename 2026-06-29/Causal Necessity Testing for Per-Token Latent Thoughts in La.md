# Causal Necessity Testing for Per-Token Latent Thoughts in Language Models

## Motivation

The axiomatic framework of Formalizing Latent Thoughts evaluates representations globally using causality and minimality, but fails to determine which individual token-level latent thoughts are causally necessary for reasoning. This limitation stems from the lack of a per-token intervention test, leaving the assumption that all latent thoughts contribute equally unverified. Without such a test, we cannot identify redundant or non-causal representations, hindering the development of efficient reasoning models.

## Key Insight

The drop in answer probability after replacing a token's latent representation with a neutral baseline directly quantifies the causal necessity of that token's thought, under the assumption that the intervention removes its unique contribution.

## Method

(A) **What it is**: Causal Necessity Test (CNT) — a method that computes a necessity score for each token's latent representation at a chosen layer by measuring the change in answer probability when that token's hidden state is replaced with a baseline.

(B) **How it works** (pseudocode):
```python
def causal_necessity_test(model, input_ids, answer_token_id, layer_idx, baseline='zero'):
    # Original forward pass
    orig_out = model(input_ids, output_hidden_states=True)
    orig_hidden = orig_out.hidden_states[layer_idx]  # (1, seq_len, d)
    orig_logits = orig_out.logits[:, -1, :]
    orig_prob = softmax(orig_logits)[0, answer_token_id]

    # Baseline: zero vector or mean across tokens
    if baseline == 'zero':
        baseline_hidden = torch.zeros_like(orig_hidden)
    else:  # mean
        baseline_hidden = orig_hidden.mean(dim=1, keepdim=True).expand_as(orig_hidden)

    necessity_scores = []
    for token_idx in range(input_ids.shape[1]):
        # Register a forward hook to replace the token's hidden at layer_idx
        def replacement_hook(module, input, output):
            output[0, token_idx, :] = baseline_hidden[0, token_idx, :]
            return output
        handle = model.layers[layer_idx].register_forward_hook(replacement_hook)
        with torch.no_grad():
            inter_logits = model(input_ids).logits[:, -1, :]
        handle.remove()
        inter_prob = softmax(inter_logits)[0, answer_token_id]
        necessity = orig_prob - inter_prob
        necessity_scores.append(necessity)
    return necessity_scores
```
Hyperparameters: layer_idx (e.g., 20), baseline choice ('zero' or 'mean').

(C) **Why this design**: Three design decisions: (1) We chose zero baseline over mean baseline because zero isolates the token's unique contribution by removing its signal entirely; mean baseline retains some information from the distribution, potentially diluting necessity. The trade-off is that zero baseline may produce artificially high drops due to out-of-distribution hidden states. (2) We intervene on each token individually rather than all tokens jointly because we seek per-token necessity; joint intervention measures combined necessity, losing granularity. The cost is that sequential interventions ignore interactions, so necessity scores may not be additive. (3) We intervene at a single layer to localize the representation of a thought; this assumes thoughts are layer-localized, which may not hold given the structural gap identified by Formalizing Latent Thoughts. The trade-off is computational simplicity versus potential mislocalization.

(D) **Why it measures what we claim**: The computational quantity (orig_prob - inter_prob) measures causal necessity of the token representation because we assume that replacing the token's hidden state with a baseline removes its causal effect on the final answer; this assumption holds if the model's computation is approximately linear in the neighborhood of the intervention, such that the baseline represents a neutral input. This assumption fails when later layers compensate for the removal, in which case the metric underestimates necessity. Alternatively, (orig_prob - inter_prob) measures the contribution of that token relative to the baseline; we claim it measures necessity because if the token is unnecessary, the drop should be small (close to zero); if the token is necessary, the drop should be large. The failure mode: the baseline may be out-of-distribution, causing a drop unrelated to necessity, in which case the metric reflects distribution shift. To mitigate, we recommend comparing against a random token intervention as a control.

## Contribution

(1) A quantitative method (CNT) for assessing per-token causal necessity of latent thoughts in autoregressive LMs, enabling identification of causally redundant tokens. (2) Empirical finding that per-token necessity varies widely across reasoning tasks, and many intermediate tokens have near-zero necessity, challenging the assumption that all latent thoughts are necessary. (3) A new evaluation protocol that complements global axiom metrics with granular necessity analysis.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | BIG-bench Logic Puzzles | Requires multi-step reasoning with clear token roles. |
| Primary metric | Accuracy drop after removing top-k necessary tokens | Directly measures causal impact of token removal. |
| Baseline: Random Intervention | Replace tokens at random | Tests if necessity scores beat random chance. |
| Baseline: Attention Weight | Use attention weights to select tokens | Compares against saliency-based importance. |
| Ablation: Mean baseline | Use mean hidden state instead of zero | Tests the zero baseline design choice. |

### Why this setup validates the claim

Logic puzzles demand precise token-level reasoning; a correct necessity score should identify tokens whose removal collapses accuracy. By measuring the accuracy drop when removing top-k scored tokens, we directly test whether our scores reflect true causal necessity. Random baseline checks if any specific token removal drives accuracy loss; attention baseline tests whether simpler saliency measures suffice. The ablation (mean vs. zero baseline) isolates whether zero baseline better isolates token contribution. If our method truly captures necessity, the accuracy drop should be significantly larger than both baselines and the ablation on subsets requiring multi-step inference.

### Expected outcome and causal chain

**vs. Random Intervention** — On a puzzle where a single connecting phrase is essential (e.g., "therefore"), random intervention likely removes irrelevant tokens, causing negligible accuracy drop. Our method specifically targets the critical token, so removal of that token causes a sharp drop, while removal of others does not. We expect a large gap in accuracy drop between our top-k and random top-k (e.g., >30% difference on hard puzzles).

**vs. Attention Weight** — On a case where a token is attended to but not causally necessary (e.g., a distractor in a logic puzzle), attention weights may highlight it, but its removal causes little accuracy change. Our necessity score would correctly assign low importance because the probability drop is small. Thus we expect our top-k removal to yield a larger accuracy drop than attention-based removal on such puzzles (e.g., 20% larger drop).

**Ablation: Mean baseline** — On a token with unique positive contribution (e.g., the correct mathematical operator), zero baseline removes all signal, causing a large probability drop. Mean baseline retains some distribution information, diluting the drop and underestimating necessity. Thus we expect the mean baseline version to produce smaller accuracy drops than our zero baseline (e.g., 15% lower drop on operator tokens).

### What would falsify this idea

If the accuracy drop from removing our top-k tokens is not significantly larger than that from random removal across all puzzle types, or if the mean baseline performs equally well, then our central claim that zero baseline necessity scores capture token-level causal importance is falsified.

## References

1. Formalizing Latent Thoughts: Four Axioms of Thought Representation in LLMs
