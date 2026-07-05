# DynaGate: Input-Dependent Dynamic Gating for Layer-Wise Hybrid Attention via Attention Density Monotonicity

## Motivation

Existing hybrid attention models, such as FlashMorph, learn layerwise gates that are fixed after training, assuming a static assignment per input is sufficient. This fails because real-world inputs vary in attention complexity—some require dense full attention while others are sparse. The structural limitation is that no current method adapts the hybridization decision per input, leading to suboptimal efficiency-accuracy trade-offs for diverse sequences.

## Key Insight

Attention density (the concentration of attention weights) is a monotone indicator of the need for full attention, enabling a closed-form gating rule that dynamically allocates full attention to inputs with high density and linear attention to those with low density.

## Method

### DynaGate: Input-Dependent Dynamic Gating for Layer-Wise Hybrid Attention

(A) **DynaGate** is a dynamic gating mechanism that, for each layer, computes a gate value from an estimated attention density measure, then routes each input through either the full-attention or linear-attention branch based on a monotone decision rule. Input: hidden states from the previous layer; Output: gated attention output.

(B) **How it works** (pseudocode):
```python
# Per layer: for a batch of sequences
# Input: H (batch, seq_len, d_model)
# Compute density estimate (lightweight predictor)
# Density predictor: 2-layer MLP, hidden_dim=256, GeLU activation, output_dim=1
# Trained on held-out attention maps to predict true attention density (fraction of probability mass in top-10% positions)
density_pred = MLP(H.mean(dim=1))  # (batch, 1) 
# Load-bearing assumption: The necessity of full attention is a monotone non-decreasing function of attention density.
# Monotone gating: gate = sigmoid((density_pred - tau) / temperature)  # tau learned per layer, initialized 0.5; temperature fixed 0.1
gate = torch.sigmoid((density_pred - tau) / 0.1)  # shape (batch, 1)
# Hard binary decision during inference: if gate > 0.5 use full attention, else linear
# During training, use soft gating with straight-through estimator:
# forward pass: gate_hard = (gate > 0.5).float(); backward pass: gradients from gate_soft
full_out = full_attention(H)  # (batch, seq_len, d_model)
linear_out = linear_attention(H)  # (batch, seq_len, d_model)
output = gate_hard.detach() * full_out + (1 - gate_hard.detach()) * linear_out + \
         (gate - gate_hard.detach()) * (full_out - linear_out)  # straight-through estimator
# Density predictor training (separate regression loss on held-out attention maps):
# Collect true density (fraction of total probability mass in top-10% positions) from a pretrained full-attention model on training set.
# Train MLP with MSE loss on 10% of training set (cost: ~100 GPU hours on LRA).
# Calibration: On a held-out calibration set of 512 examples, compute Pearson correlation between predicted density and actual accuracy gain (full - linear). If correlation < 0.5, method is not applied (falls back to fixed first-2-layers-full hybrid).
```

(C) **Why this design**: We chose a learned density predictor over a hand-crafted formula (e.g., entropy of attention weights) because a predictor can adapt to model-specific attention patterns, but it requires training data (full attention maps). We chose a monotone sigmoid gating over a learned softmax router (like FlashMorph) to ensure that gate value respects the monotonicity between density and full-attention necessity—this avoids the risk of the router learning input-invariant policies. We chose straight-through estimation over REINFORCE for the binary decision to reduce variance and enable end-to-end gradient flow. The cost is that the density predictor may overfit to the training distribution, and the straight-through estimator can introduce bias. We also chose to use a global density estimate per sequence (mean across tokens) rather than token-level to keep overhead low, accepting that token-level variation is smoothed out.

(D) **Why it measures what we claim**: The density predictor is trained to estimate the true attention density (fraction of total probability mass in the top-10% positions) from the hidden states; density measures the concentration of dependencies in the attention map—higher density implies more tokens are crucial, thus full attention is needed. This holds under the assumption that the predictor generalizes across inputs and that the true density is a valid proxy for attention complexity; this assumption fails when the predictor misses long-range dependencies that are dispersed (low density) but still require full attention (e.g., for retrieval), in which case the gate underestimates full-attention need, leading to degraded recall. The monotone sigmoid ensures that gate is strictly increasing in predicted density, so the decision rule aligns with the hypothesized causal relationship. The calibration set correlation check verifies this assumption per experiment.

## Contribution

(1) A novel input-dependent dynamic gating mechanism for layer-wise hybrid attention that adapts routing per sequence based on attention density monotonicity. (2) A lightweight density predictor that can be trained with minimal overhead using precomputed attention maps from the target model. (3) Empirical demonstration that dynamic gating consistently outperforms static gating baselines (e.g., FlashMorph) across varying input complexity, improving efficiency by up to 30% without accuracy loss.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|----------------------|
| Dataset | Long-Range Arena (LRA): ListOps, Text, Retrieval, Image, Pathfinder | Tests long-context retrieval and reasoning across diverse patterns. |
| Primary metric | Accuracy on each subtask (average over 3 seeds) | Directly measures attention effectiveness. |
| Baseline 1 | Fixed hybrid: first 2 layers full, remaining linear | Heuristic placement lacks adaptive routing. |
| Baseline 2 | Entropy-based gating: compute softmax attention entropy, threshold tuned on validation set | Static rule may miss density patterns. |
| Baseline 3 | Random gating: each layer randomly selects full or linear with prob 0.5 | Upper bound for non-adaptive methods. |
| Ablation-of-ours | DynaGate with random gate (density predictor replaced by uniform random gate) | Isolates benefit of learned density. |

### Why this setup validates the claim

LRA's retrieval tasks (e.g., Pathfinder, Retrieval) require distinguishing dense vs sparse attention patterns, so a method that adaptively routes per layer should outperform fixed heuristics. The primary metric (accuracy) directly reflects whether the gating selects the correct branch. The baselines test distinct subclaims: Fixed hybrid shows need for adaptivity, entropy-based shows need for learned density over hand-crafted, random shows baseline non-adaptive performance. Ablation (random gate) isolates the density predictor's contribution. The metric is sensitive to the gate's correctness, as wrong routing hurts retrieval. Additionally, we compute the correlation between predicted density and accuracy gain on a calibration set to directly validate the causal link.

### Expected outcome and causal chain

**vs. Fixed hybrid (first 2 layers full)** — On a case where critical tokens appear mid-sequence (e.g., Pathfinder where path connects distant positions), fixed hybrid using full attention only in early layers may miss dependencies for later layers because they use linear attention. Our method predicts high density for that layer and routes to full attention, thus capturing the dependency. We expect a noticeable gap (5-10% absolute) on Pathfinder and Retrieval, but parity on ListOps where structure is tree-based and early layers suffice.

**vs. Entropy-based gating** — On a case where attention entropy is low but density is high (e.g., Image retrieval where many patches have uniform attention), entropy-based gate incorrectly routes to linear (thinking sparse) while our density predictor correctly identifies many tokens as important. Our method then uses full attention, preserving accuracy. We expect a gap of 3-7% absolute on Image and Retrieval tasks.

**vs. Random gating** — Random gating occasionally routes correctly but often wrong. Our method consistently gates based on true density. We expect a large gap (10-15% absolute) across all LRA tasks, as the random baseline is a lower bound. The causal chain: random gate has 50% chance of wrong branch per layer, compounding error; our method adjusts per instance.

### What would falsify this idea

If our method does not outperform entropy-based gating on tasks where entropy is a misleading indicator (e.g., Image retrieval), or if the improvement is uniform across all subsets (suggesting the gate is not adapting but just always choosing one branch), then the claim that learned density helps would be false. Specifically, if on the "low entropy but high density" subset our method is not better than entropy-based, or if the calibration correlation is below 0.5, the central hypothesis fails.

## References

1. Morphing into Hybrid Attention Models
2. Jet-Nemotron: Efficient Language Model with Post Neural Architecture Search
3. Kimi Linear: An Expressive, Efficient Attention Architecture
4. The Zamba2 Suite: Technical Report
5. Hymba: A Hybrid-head Architecture for Small Language Models
6. Puzzle: Distillation-Based NAS for Inference-Optimized LLMs
7. Mamba: Linear-Time Sequence Modeling with Selective State Spaces
8. Mistral 7B
