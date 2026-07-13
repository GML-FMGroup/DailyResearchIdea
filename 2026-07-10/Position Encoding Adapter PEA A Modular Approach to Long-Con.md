# Position Encoding Adapter (PEA): A Modular Approach to Long-Context Extension for Any Positional Encoding

## Motivation

Jet-Long's dynamic RoPE rescaling inherits RoPE-dependence from all its predecessors, meaning that models using ALiBi or NoPE cannot benefit from its efficient long-context extension. This reliance on a specific positional encoding limits the applicability of context extension methods and forces practitioners to adopt RoPE even when other encodings may be preferable for their task. A solution that works across arbitrary positional encodings would unify extension techniques and reduce engineering overhead.

## Key Insight

By enforcing a consistency loss that aligns attention patterns across different base positional encodings for the same relative positions, the adapter learns a canonical representation that makes long-context extension techniques agnostic to the underlying position encoding.

## Method

**A) What it is:** PEA (Position Encoding Adapter) is a lightweight module inserted before each attention layer that maps a position index and a one-hot indicator of the base PE type to additive biases on attention logits, enabling position-encoding-agnostic long-context extension.

**B) How it works:**
```python
# Hyperparameters: d_model=4096, num_heads=32, PE_TYPES=3 (RoPE, ALiBi, NoPE), hidden_dim=64
class PEA(nn.Module):
    def __init__(self, num_heads, pe_types, hidden_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1 + pe_types, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_heads)
        )

    def forward(self, pos, pe_type):
        # pos: float tensor of shape (batch, seq_len) normalized to [0,1]
        # pe_type: one-hot tensor of shape (batch, seq_len, PE_TYPES)
        x = torch.cat([pos.unsqueeze(-1), pe_type], dim=-1)  # (batch, seq_len, 1+PE_TYPES)
        bias = self.mlp(x)  # (batch, seq_len, num_heads)
        return bias

# Training loop (for one step across all PE types):
for (positions, pe_type_onehot) in dataloader:
    # positions: relative position indices (e.g., 0..L-1) for a batch of sequences
    bias = PEA(positions, pe_type_onehot)  # adapter outputs per-head biases
    # For each PE type, compute adapted attention scores: score_orig + bias (broadcast)
    # Compute attention pattern A_adapted via softmax
    # Compute A_orig from base model's attention without adapter
    loss = KL_div(A_orig || A_adapted)  # per-head, averaged over heads
    # Additionally, for pairs of PE types (e.g., RoPE and ALiBi) on the same positions:
    # Enforce consistency loss: MSE between their adapted attention patterns
    consistency_loss = MSE(A_adapted_rope, A_adapted_alibi)
    total_loss = loss + lambda * consistency_loss  # lambda=0.1
    optimizer.zero_grad(); total_loss.backward(); optimizer.step()
```

**C) Why this design:** We chose a simple MLP over a transformer layer because the mapping from position and PE type to bias is expected to be relatively simple and a transformer would overfit; the cost is that the adapter may not capture complex interactions between position and PE type. We opted for additive biases on attention scores rather than modifying the embeddings because it is more parameter-efficient and directly influences attention without altering the representation space; however, this assumes that the base model's representations are already well-formed and only the relative position encoding needs adjustment. We used KL divergence as the consistency loss because it penalizes overconfidence in the adapted attention, preventing degenerate solutions; but KL divergence is asymmetric and sensitive to the target distribution, meaning that if the original attention pattern is noisy or suboptimal, the adapter inherits those flaws. Finally, we enforced cross-PE consistency via MSE loss because it is symmetric and simple, but it assumes that different PEs should produce identical attention patterns for the same positions, which may not hold for short-range vs long-range patterns; we mitigate this by only enforcing consistency on a held-out set of relative positions that are within the training context length.

**D) Why it measures what we claim:** The KL divergence between original and adapted attention patterns measures the extent to which the adapter preserves the relative position information encoded by the base PE; this operationalizes the concept of "position encoding invariance" because if two different base PEs produce different attention patterns for the same positions, the adapter must learn to map both to a common pattern. The assumption is that attention patterns are sufficient statistics for position encoding behavior; this assumption fails when attention patterns are dominated by content rather than position, in which case the KL divergence reflects content-related discrepancies rather than positional ones. To mitigate this, we average over multiple random masks that isolate positional dependencies. The cross-PE MSE loss measures the degree to which the adapter produces consistent attention across different PE types; this operationalizes "position-encoding-agnostic" because identical inputs (positions and content) should yield the same output regardless of PE type. The assumption is that the adapter can perfectly align attention patterns from different PEs; this fails when the PEs have fundamentally different frequency regimes (e.g., extrapolation vs interpolation), in which case MSE reflects irreducible differences rather than adapter error.

## Contribution

(1) Introduces Position Encoding Adapter (PEA), a lightweight module that retrofits any positional encoding to support long-context extension without retraining the base model, by learning to map position indices and PE type to attention biases. (2) Demonstrates that attention pattern consistency across different base PEs can be enforced via a joint KL and MSE loss, enabling position-encoding-agnostic context extension. (3) Provides a design principle that additive biases on attention scores are sufficient to adapt positional encodings, supported by an analysis of parameter efficiency and trade-offs.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | LongBench | Tests long-context understanding across diverse tasks |
| Primary metric | Average accuracy | Direct measure of task performance |
| Baseline 1 | RoPE (original) | Standard position encoding without extension |
| Baseline 2 | NTK-aware scaling | Popular RoPE-based long-context method |
| Baseline 3 | Position Interpolation (PI) | Simple rescaling of position indices |
| Ablation of ours | PEA without consistency loss | Isolates effect of cross-PE alignment |

### Why this setup validates the claim

This design tests whether PEA can match or surpass state-of-the-art position encoding extension methods while being agnostic to the base PE type. RoPE (original) shows the baseline failure without extension; NTK-aware and PI represent two distinct strategies that exploit specific properties of RoPE. If PEA (trained only on in-context positions) achieves comparable or better accuracy on LongBench, it demonstrates that a lightweight adapter can generalize to long contexts regardless of the underlying PE. The ablation reveals whether the cross-PE consistency loss is necessary for this generalization or if the KL divergence alone suffices. Average accuracy across diverse tasks (e.g., summarization, QA, few-shot learning) ensures the metric captures genuine positional understanding rather than task-specific artifacts.

### Expected outcome and causal chain

**vs. RoPE (original)** — On a case where the input length exceeds the training maximum (e.g., 8K tokens), RoPE produces degraded attention because its sinusoidal encoding cannot extrapolate to unseen positions. Our method, by normalizing positions to [0,1] and learning additive biases via MLP, can smoothly extend to any length, preserving attention patterns. We expect a noticeable gap (e.g., 5-10% accuracy drop for RoPE vs. PEA) on long sequences (>4K), with parity on short sequences.

**vs. NTK-aware scaling** — On a case with extremely long contexts (e.g., 32K tokens), NTK-aware scaling forces high-frequency dimensions to cover dense positions, but low-frequency dimensions remain sparse, causing distortion in attention for close tokens. Our method avoids frequency splitting by directly predicting biases from normalized positions, maintaining uniform resolution. We expect PEA to outperform NTK-aware by a moderate margin (e.g., 2-4% on the longest tasks) while being simpler.

**vs. Position Interpolation (PI)** — On a case with fine-grained relative positions (e.g., question-answering where answer is near the end of a long document), PI uniformly compresses all position gaps, blurring short-range dependencies. Our method, by learning a nonlinear mapping from normalized positions to biases, can preserve local patterns. We expect PEA to exceed PI on tasks requiring precise relative position (e.g., 3-5% on passage retrieval subsets).

### What would falsify this idea

If PEA's performance is no better than the worst baseline (e.g., RoPE original) across all lengths, or if the ablation without consistency loss performs equally, then the cross-PE alignment claim is unsupported. Also, if the gain appears uniform across short and long contexts rather than concentrated where baselines fail (e.g., >4K tokens), the method likely memorizes content rather than learns positional adaptation.

## References

1. Jet-Long: Efficient Long-Context Extension with Dynamic Bifocal RoPE
2. SELF: Self-Extend the Context Length With Logistic Growth Function
3. LaMPE: Length-aware Multi-grained Positional Encoding for Adaptive Long-context Scaling Without Training
4. LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens
5. LLM Maybe LongLM: Self-Extend LLM Context Window Without Tuning
6. Scaling Laws of RoPE-based Extrapolation
7. Challenging BIG-Bench Tasks and Whether Chain-of-Thought Can Solve Them
8. GPT-NeoX-20B: An Open-Source Autoregressive Language Model
