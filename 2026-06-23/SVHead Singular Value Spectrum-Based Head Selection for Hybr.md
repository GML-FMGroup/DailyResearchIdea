# SVHead: Singular Value Spectrum-Based Head Selection for Hybrid Attention

## Motivation

HydraHead's reliance on interpretability scores to decide which heads use full versus linear attention introduces brittleness because interpretability may not correlate with the feasibility of low-rank approximation. This creates a bottleneck as the field moves toward structured hybridization, as the selection mechanism lacks a principled guarantee. We need a data-driven, provably correct criterion that directly measures whether a head can be accurately approximated by linear attention.

## Key Insight

The singular value spectrum of an attention matrix directly measures its intrinsic dimensionality, so heads with rapidly decaying singular values are exactly representable by linear attention, with approximation error bounded by the tail energy.

## Method

### (A) What it is
SVHead is a calibration-based method that assigns each attention head in a Transformer to either full attention (softmax) or linear attention (e.g., linearized via random features) based on the singular value spectrum of its attention matrix, computed over a calibration dataset. **Load-bearing assumption:** The singular value spectrum of the attention matrix averaged over a calibration set provides a stable and generalizable measure of how well a head can be approximated by linear attention across all test inputs.

### (B) How it works — pseudocode
```python
def assign_head_type(model, calibration_data, threshold_energy=0.95, max_rank_ratio=0.5, perplexity_threshold=0.5):
    # model: pretrained Transformer with M heads per layer
    # calibration_data: 512 examples from PG19 validation set, sequences clipped to max length 4096
    head_types = {}  # (layer_idx, head_idx) -> 'full' or 'linear'
    # Step 1: SVD-based candidate selection
    candidate_linear = []
    for batch in calibration_data:
        hidden_states = model.get_all_hidden_states(batch)  # shape (batch, seq_len, d_model)
        for layer_idx, layer in enumerate(model.layers):
            for head_idx in range(layer.num_heads):
                attn_matrix = layer.self_attention.get_attention_matrix(hidden_states[layer_idx])[:, head_idx, :, :]
                avg_attn = attn_matrix.mean(dim=0)  # shape (T, T)
                U, S, Vh = torch.linalg.svd(avg_attn, full_matrices=False)
                total_energy = S.sum()
                cumulative_energy = S.cumsum(0)
                effective_rank = (cumulative_energy <= threshold_energy * total_energy).sum().item()
                if effective_rank <= max_rank_ratio * avg_attn.size(0):
                    candidate_linear.append((layer_idx, head_idx, batch))
    # Step 2: Perplexity-based verification
    for (layer_idx, head_idx, batch) in candidate_linear:
        # Compute original perplexity on this batch
        original_loss = model(batch).loss
        # Create a modified model where this head is replaced by linear attention
        modified_model = replace_head_with_linear(model, layer_idx, head_idx)
        modified_loss = modified_model(batch).loss
        perplexity_diff = (modified_loss - original_loss).exp().item()
        if perplexity_diff <= perplexity_threshold:
            head_types[(layer_idx, head_idx)] = 'linear'
        else:
            head_types[(layer_idx, head_idx)] = 'full'
    # Aggregate across batches via majority vote
    final_types = {}
    # ... aggregation logic ...
    return final_types
```
Hyperparameters: `threshold_energy` (default 0.95) controls how much of the total singular value energy must be captured; `max_rank_ratio` (default 0.5) limits the effective rank relative to sequence length; `perplexity_threshold` (default 0.5) is the maximum allowable perplexity increase per head when replaced by linear attention.

### (C) Why this design
We chose SVD over heuristic interpretability scores because the singular value spectrum provides a direct measure of low-rank approximation error (the tail energy), bypassing the circular reasoning that interpretability correlates with retrieval-critical heads. We average attention matrices across calibration batches to reduce per-sample noise, accepting that some head-specific patterns may be washed out but gaining a stable estimate of typical behavior. We combine an energy threshold with a rank ratio threshold to avoid selecting heads that are low-rank but still large (e.g., effective rank 50 in a length-100 sequence is borderline; larger sequences need stricter cutoffs). We perform calibration offline rather than online to avoid overhead during inference, sacrificing adaptivity to input distribution shifts but gaining simplicity. The majority vote across batches provides robustness to outlier batches without introducing learned parameters. The additional perplexity verification step directly measures the task-level impact of linear approximation, mitigating the risk that SVD-based selection is not aligned with downstream loss.

### (D) Why it measures what we claim
The effective rank (number of singular values that explain a fraction `threshold_energy` of total energy) measures the intrinsic dimensionality of the attention matrix because a low effective rank implies that the matrix lies in a low-dimensional subspace and can be approximated by a rank-`effective_rank` matrix with error bounded by `(1 - threshold_energy) * total_energy`; this holds under the assumption that the Frobenius norm is the correct error metric for approximation quality, which fails when attention matrices are sparse but high-rank, in which case the metric may overestimate the need for full attention because sparse matrices can still be approximated by structured linear attention like kernel methods. The `max_rank_ratio` term operationalizes the notion of 'feasibility': even if a head is low-rank, if the absolute rank is large relative to sequence length, linear attention may still be computationally expensive; this assumption relies on the cost of linear attention growing with effective rank, which is true for most implementations, but fails when linear attention implementations use fixed-rank approximations that do not scale with data. The additional perplexity verification ensures that the SVD-based decision correlates with actual task performance, addressing the load-bearing assumption that SVD generalizes to test inputs. **Why averaging is valid:** Prior work (e.g., Kovaleva et al., 2019) shows that attention patterns are often stable across samples for the same head in a pretrained model; we quantify this by computing the variance of effective rank across calibration samples. If variance is high, we fall back to full attention for that head.

## Contribution

(1) A novel head selection algorithm for hybrid attention based on singular value decomposition of attention matrices, replacing heuristic interpretability with a structural criterion. (2) A design principle that uses effective rank as a data-driven indicator for low-rank approximation feasibility. (3) A calibration-based framework that enables principled, offline assignment without additional training or interpretability analysis.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|-----------------------|
| Dataset | PG19 long-context language modeling | Tests attention patterns in long sequences |
| Primary metric | Perplexity at equal FLOPs | Measures quality-efficiency tradeoff |
| Baseline 1 | Pure full attention (softmax) | Upper bound on quality |
| Baseline 2 | Pure linear attention (random features) | Lower bound on quality |
| Baseline 3 | Layer-wise hybrid (alternating full/linear) | Compares head-level vs layer-level hybridization |
| Ablation-of-ours | SVHead with random head assignment | Controls for allocation ratio effect |
| Additional baseline | SVHead without perplexity verification | Isolates benefit of verification step |

### Why this setup validates the claim

PG19 (long sequences) and perplexity at equal FLOPs directly test whether SVHead's head-level selection improves the tradeoff between quality and efficiency. Pure full attention provides the quality ceiling but high cost; pure linear attention provides the efficiency floor but quality loss; layer-wise hybrid evaluates whether coarse-grained allocation suffices. The ablation with random assignment isolates the benefit of SVD-based selection from merely having a mixture. The added baseline without perplexity verification teases apart the contribution of the verification step. Sensitivity analysis on threshold_energy (0.9, 0.95, 0.99) and max_rank_ratio (0.3, 0.5, 0.7) will show how robust the method is. The metric of perplexity at equal FLOPs ensures that any improvement is not due to uneven compute budgets. We also compute the variance of effective rank across calibration samples for each head; heads with high variance (coefficient of variation > 0.5) are flagged as unstable and excluded from linear assignment. This design creates a falsifiable test: if SVHead's gains are solely due to head mixing (not the spectral rule), the random ablation will match performance; if the spectral rule is effective, SVHead should outperform both pure linear and layer-wise hybrid on long sequences where low-rank heads occur frequently.

### Expected outcome and causal chain

**vs. pure full attention** — On a long sequence (e.g., 4K tokens) where many heads have low effective rank (e.g., attending to local tokens only), pure full attention wastes compute by computing full softmax for all heads. Our method detects via SVD that those heads are linear-eligible and uses linear attention, reducing FLOPs while preserving quality because low-rank matrices are well-approximated by linear mechanisms. Thus we expect SVHead to achieve similar perplexity (within 0.1) with ~40% fewer FLOPs, with the gap widening as sequence length increases.

**vs. pure linear attention** — On a long sequence containing global retrieval dependencies (e.g., a head that attends to a distant subject), pure linear attention fails because its rank-constrained approximation cannot capture the full attention distribution. Our method retains full attention for high-rank heads (effective rank >0.5×length), so it handles such cases correctly. We expect SVHead to significantly outperform pure linear attention on perplexity (e.g., 0.5–1.0 nats lower) while still using fewer FLOPs than full attention.

**vs. layer-wise hybrid** — On a sequence where low-rank heads and high-rank heads coexist within the same layer (e.g., in middle layers), a layer-wise hybrid either assigns full attention to the entire layer (wasting compute) or linear attention to the entire layer (sacrificing quality). Our method assigns different heads within the same layer appropriately. We expect SVHead to achieve better perplexity than a linear layer (which loses high-rank heads) and similar FLOPs to a full layer, but with lower perplexity than a layer-wise hybrid when averaged across layers, because it tailors per head.

**vs. SVHead without verification** — On heads where SVD indicates low rank but perplexity verification shows high loss, the verification step reverts to full attention, avoiding quality degradation. Thus we expect the verified version to have slightly higher FLOPs but lower perplexity than the unverified version.

### What would falsify this idea

If SVHead's perplexity at equal FLOPs is no better than the random assignment ablation, or if its gains are uniform across all sequence lengths rather than concentrated on long sequences where low-rank heads are prevalent, then the spectral rule is not driving improvement and the core claim is invalid. Additionally, if the perplexity verification step does not improve results over SVD alone, then the load-bearing assumption may be correct but the verification is redundant.

## References

1. HydraHead: From Head-Level Functional Heterogeneity to Specialized Attention Hybridization
2. Gated Delta Networks: Improving Mamba2 with Delta Rule
3. Gated Linear Attention Transformers with Hardware-Efficient Training
