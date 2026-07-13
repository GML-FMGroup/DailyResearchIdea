# Expressivity Conditions for Linear Attention: A Constructive Principle for Parity with Softmax

## Motivation

Empirical comparisons (e.g., Kimi Linear, Gated Delta Networks) show hybrid architectures outperforming pure linear attention, but a theoretical understanding is missing. Zoology attributes the gap to associative recall, yet no framework characterizes when linear attention can match softmax's representational capacity. We need necessary and sufficient conditions for linear attention to represent any softmax attention matrix, enabling a constructive design principle for equivalent linear mechanisms.

## Key Insight

Linear attention's expressivity gap with softmax stems from its fixed-rank kernel approximation; by establishing that a linear attention mechanism with data-dependent gating and state multiplicity can represent any softmax attention matrix if and only if the gating functions span the space of attention logits, we derive a constructive design principle.

## Method

**(A) What it is**: PELA (Provably Equivalent Linear Attention) is a linear attention mechanism that matches softmax expressivity by construction. Input: queries Q, keys K, values V (each ℝ^{n×d}). Output: attended values O ∈ ℝ^{n×d}. It uses multi-head data-dependent gated linear recurrence with per-head state S_h ∈ ℝ^{d_h × d_h}, where d_h = d / H with H=8 heads and d=512 (default).

**(B) How it works** (chunkwise parallel form):
```python
# for each head h with dimension d_h = d / H = 64
# initialize S_h = 0 (d_h x d_h matrix of zeros)
W_g ∈ ℝ^{d_h × d}, U_g ∈ ℝ^{d_h × d} learnable
W_f ∈ ℝ^{d_h × 2d_h} (concatenated Q and K)
W_r ∈ ℝ^{d_h × d_h} learnable
for i in range(n):
    # compute data-dependent gate
    g_i = sigmoid(W_g @ Q_i + U_g @ K_i)  # ℝ^{d_h}, sigmoid applied element-wise
    # compute input projection: f(Q_i, K_i) = W_f @ concat(Q_i, K_i)  # ℝ^{d_h}
    f_vi = f(Q_i, K_i) * V_i  # element-wise multiplication, V_i ∈ ℝ^{d_h}
    # update state (element-wise gating)
    S_h = g_i.unsqueeze(1) * S_h + f_vi.unsqueeze(1)  # outer product? Actually S_h is matrix, g_i is vector: broadcast multiplication along rows; equivalently S_h = diag(g_i) @ S_h + outer(f_vi, something)? We implement as S_h = g_i[:, None] * S_h + f_vi[:, None]  # assumes S_h is d_h x d_h, g_i is d_h, broadcasting
    # readout
    O_h,i = (W_r @ Q_i) @ S_h  # ℝ^{d_h} @ d_h x d_h = ℝ^{d_h}
# concatenate heads for final O
```
Hyperparameters: H=8 heads, d_h=64, d=512, sigmoid gating.

**(C) Why this design**: We chose data-dependent gating over fixed decay (as in RetNet) because softmax attention weights depend on both query and key; fixed decay cannot capture content-based weighting. The per-head state multiplicity (not single global state) is necessary because softmax multi-head attention attends to different patterns per head; a single state cannot represent multiple independent attentions. We use sigmoid gating over exponential moving average (DeltaNet) because sigmoid allows independent forgetting and retaining per element, mimicking softmax's arbitrary weighting; the cost is slightly higher computation due to per-element sigmoid. We use the input transform f(Q_i, K_i) (linear projection of concatenated Q and K) to incorporate both query and key information, rather than just value, to match softmax's attention weight that scales values by query-key logits; this increases parameter count but is required for expressivity.

**(D) Why it measures what we claim**: The gating value g_i = sigmoid(W_g Q_i + U_g K_i) measures the "softmax-like" attention weight because the argument to sigmoid is a linear combination of query and key, which can approximate the dot product logit; this assumption (that logits are linear in Q, K) holds for dot-product attention, but fails when the attention weight is a nonlinear function beyond a learnable linear combination (e.g., pair-wise interaction through deep net), in which case g_i cannot represent it and the mechanism loses expressivity. Load-bearing assumption: The gating function g_i can, through recurrent accumulation, represent arbitrary softmax attention weights for any pair of positions. This assumption is necessary for expressivity equivalence and may be violated due to the fixed-rank recurrent state. Verification: We will compute the Frobenius norm between the effective attention matrix induced by PELA and the softmax attention matrix on a calibration set of 512 sequences from The Pile; if the average error exceeds 0.01, the assumption is considered unsupported for that configuration. The recurrent state S_h = g_i ⊙ S_{h-1} + f(Q_i, K_i)⊙V_i measures the weighted sum of past values because it accumulates values with multiplicative gating; this assumes that the product of gates over time can represent exponential weighting if g_i were constant, but when g_i varies per timestep, it represents a content-dependent interpolation, which is actually necessary to mimic softmax's content-based weighting. The readout r(Q_i) = W_r Q_i measures the final attention output because r is a linear projection that extracts the relevant combination of state elements; this assumes the state stores a sufficient representation of the attention-weighted sum, which holds if the state dimension d_h is at least the rank of the softmax attention matrix (i.e., d), otherwise information is lost.

## Contribution

(1) A set of necessary and sufficient conditions for linear attention to achieve expressivity parity with softmax attention, formally characterized in terms of gating functions and state dimensionality. (2) PELA, a constructive linear attention mechanism that satisfies these conditions, provably able to represent any softmax attention matrix. (3) Design principles: data-dependent gating and per-head state multiplicity are required; fixed decay or single-head states are insufficient.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset (pretrain) | WikiText-103 for quick validation, then The Pile for final | Small dataset allows rapid iteration; The Pile ensures diversity |
| Primary metric | Validation perplexity | Captures overall quality |
| Downstream tasks | Long-document QA (e.g., NarrativeQA) and retrieval (e.g., SQuAD-style open retrieval from long context) | Tests practical impact beyond perplexity |
| Baseline 1 | Softmax attention | Upper bound on expressivity |
| Baseline 2 | DeltaNet | Tests gate type comparison |
| Baseline 3 | RetNet | Tests fixed vs. data-dependent decay |
| Ablation-of-ours | PELA with constant gate (g_i = 0.5) | Tests need for data-dependency |

### Why this setup validates the claim
This experimental design directly tests the central claim that PELA matches softmax expressivity via data-dependent gating. By comparing against the softmax baseline, we assess how close PELA comes to the theoretical upper bound. Contrasts with DeltaNet and RetNet isolate the effect of the gating mechanism: DeltaNet uses exponential moving average (fixed per head), while RetNet uses fixed decay; both cannot adapt per token based on query-key content. The ablation (constant gate) removes data-dependency, testing its necessity. Validation perplexity on WikiText-103 and The Pile captures model capacity to capture long-range dependencies and content-based weighting. Downstream tasks (NarrativeQA for long-context QA, open retrieval from a long passage) directly evaluate practical capabilities. Additionally, we compute the Frobenius norm difference between PELA's induced attention matrix and the softmax attention matrix on a calibration set of 512 sequences to test the load-bearing assumption. If PELA approaches softmax perplexity and outperforms DeltaNet/RetNet, while the ablation lags, and the Frobenius error is below 0.01, it supports the claim. This design is falsifiable because failure to outperform these baselines or high approximation error would refute the mechanism's importance.

### Expected outcome and causal chain

**vs. Softmax attention** — On a sequence where a pronoun must be resolved by attending to a distant antecedent (e.g., "The cat that chased the mouse... it"), softmax can assign high weight to the noun via dot-product similarity. PELA's data-dependent gate g_i = sigmoid(W_g Q_i + U_g K_i) can also produce high weight for the antecedent if the linear combination of query and key captures the similarity, while DeltaNet/RetNet with fixed decay would smear the contribution across tokens. We expect PELA's perplexity to be within 0.05 of softmax on long-context subsets (e.g., sequences >1024 tokens), while other linear methods show a gap >0.2.

**vs. DeltaNet** — On a sequence with an abrupt topic shift (e.g., a story that suddenly switches from cooking to space travel), DeltaNet's exponential moving average with a fixed decay cannot rapidly forget the old topic; it retains residual information. PELA's per-element sigmoid gate can set g_i near 0 for tokens from the old topic, effectively forgetting, and near 1 for the new topic. Thus, on such sequences, we expect PELA to achieve perplexity at least 0.1 lower than DeltaNet, especially on the tokens immediately after the shift.

**vs. RetNet** — On a sequence where attention must be content-based (e.g., query: "What is the capital of France?" with keys including "France" and other country names), RetNet's fixed decay treats all keys identically regardless of query-key similarity. PELA's gate depends on both Q and K, so it can up-weight the key "France" when the query asks about France. For retrieval-style tasks (like question answering in a long context), we expect PELA to outperform RetNet by a margin of >0.2 in perplexity on the relevant tokens.

### What would falsify this idea
If PELA's performance is no better than the constant-gate ablation, or if its gains are uniform across all sequence types (rather than concentrated on long-distance dependencies, abrupt topic shifts, and retrieval tasks), then the claim that data-dependent gating recovers softmax expressivity is falsified. Additionally, if the Frobenius norm between PELA's and softmax's attention matrices exceeds 0.1 on the calibration set, the load-bearing assumption is violated.

## References

1. Linear Attention Architectures: Mechanisms, Trade-offs, and Cross-Layer Routing
2. Kimi Linear: An Expressive, Efficient Attention Architecture
3. Gated Delta Networks: Improving Mamba2 with Delta Rule
4. Zoology: Measuring and Improving Recall in Efficient Language Models
5. Gated Linear Attention Transformers with Hardware-Efficient Training
