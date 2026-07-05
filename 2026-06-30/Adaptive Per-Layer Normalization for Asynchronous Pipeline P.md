# Adaptive Per-Layer Normalization for Asynchronous Pipeline Parallelism

## Motivation

Existing work (NorMuon) assumes row-wise normalization of optimizer updates is universally beneficial, but this assumption remains untested under gradient delay in asynchronous pipeline parallelism. The one-step delay analysis (One-Step Gradient Delay is Not a Barrier) shows that while Muon is robust, the impact of normalization schemes under delay is unexplored. The root cause is that normalization parameters are static, ignoring layer-specific gradient statistics that shift under asynchrony, leading to suboptimal updates in non-transformer architectures.

## Key Insight

The second-order momentum statistics of each neuron across layers evolve at different rates under delay, so a layer-wise adaptive normalization coefficient derived from the ratio of delayed-to-synchronous gradient variance can dynamically balance update scales without global tuning.

## Method

**(A) What it is:**
We propose **Adaptive Layer-wise Normalization (ALN)**, a drop-in modification to NorMuon that adjusts the per-neuron normalization strength for each layer separately based on the running variance of delayed gradients. ALN takes as input the delayed gradients and the current orthogonalized update from Muon, and outputs a rescaled update where the normalization factor per neuron is modulated by a layer-specific correction coefficient.

**(B) How it works:**
```pseudocode
Input: delayed gradient g_d (per layer), orthogonalized update u (from Muon), 
       running variance v (exponential moving average of g_d^2, beta=0.9)
       hyperparameter alpha (default 0.5)
# Assumption: variance differences are due to delay, not architecture. If violated, c_l may overcorrect.
For each layer l:
    avg_var_l = mean(v[l])  # scalar
    global_avg_var = mean([avg_var_l for all layers])
    c_l = 1.0 + alpha * (global_avg_var - avg_var_l) / (global_avg_var + 1e-8)
    u_norm = row_normalize(u[l])  # same as NorMuon
    u_adapted[l] = c_l * u_norm
Output: u_adapted per layer
```
Hyperparameters: alpha controls how strongly the normalization adjusts to variance mismatch; default 0.5 based on grid search.

**(C) Why this design:**
We chose a simple linear scaling over a learned gating mechanism (e.g., small neural network) to avoid introducing additional parameters that would require synchronous training, accepting the cost that the adjustment may be coarse if variance dynamics are highly non-linear. We used running variance (exponential moving average) instead of raw gradient norm because variance captures stability; the trade-off is that variance lags behind sudden changes, but this matches the slow drift typical under one-step delay. We opted for a layer-wise rather than per-neuron correction to keep the overhead minimal (one scalar per layer vs. per neuron), acknowledging that this may under-correct for outlier neurons. We set the reference as global average variance instead of a fixed constant to make it architecture-agnostic; the cost is that if all layers have similar variance, the correction vanishes, which is acceptable as it reduces to standard NorMuon. We assume that variation in avg_var_l across layers primarily reflects delay-induced mismatch rather than inherent architectural differences (e.g., deeper layers naturally having larger gradients). If this assumption fails (e.g., in architectures with inherent variance differences), ALN may overcorrect, potentially harming performance.

**(D) Why it measures what we claim:**
`v[l]` measures the per-neuron gradient variance under delay because variance captures the spread of gradient magnitudes, which correlates with optimization stability; this assumption fails when gradients are noisy (e.g., small batch sizes) in which case variance reflects noise rather than delay-induced mismatch. `c_l` measures the deviation of layer l from the global variance distribution; it operationalizes the concept of 'layer-specific delay sensitivity' under the assumption that variance differences are caused by delay rather than architecture. This assumption fails when variance differences are due to architecture properties (e.g., deeper layers naturally have larger gradients), in which case c_l incorrectly over-corrects, leading to suboptimal updates. The `alpha` hyperparameter measures the extent to which we trust the variance signal: it assumes the relationship between variance difference and optimal normalization is linear; this assumption fails when the relationship is non-linear, causing suboptimal adaptation.

## Contribution

(1) An adaptive per-layer normalization scheme (ALN) that adjusts the strength of row-wise normalization based on delayed gradient variance, requiring no architectural changes. (2) A design principle: under asynchronous delay, the uniformity assumption of normalization breaks down, and layer-specific statistics provide a simple corrective signal without extra training.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|----------------------|
| Dataset | C4 | Standard LLM pretraining corpus |
| Primary metric | Perplexity on held-out set | Direct measure of language model quality |
| Baseline 1 | Synchronous Muon | Upper bound, no delay artifacts |
| Baseline 2 | Asynchronous Muon (no ALN) | Tests effect of delay without correction |
| Baseline 3 | Asynchronous AdamW | Common optimizer, baseline for comparison |
| Ablation of ours | ALN with global-only variance | Tests contribution of layer-wise adaptation |
| Ablation of ours | ALN with learned scaling (2-layer MLP, hidden=256, trained synchronously) | Tests necessity of variance heuristic |

### Why this setup validates the claim

This combination forms a falsifiable test of the claim that ALN improves asynchronous pipeline parallelism by adaptively per-layer normalizing updates based on delayed gradient variance. Synchronous Muon provides the ideal performance without delay, bounding the maximum possible improvement. Asynchronous Muon (no ALN) isolates the effect of the delay; if ALN is effective, it should reduce the gap between synchronous and asynchronous. Asynchronous AdamW represents a strong and widely used optimizer that does not leverage variance information from delay, thus testing whether ALN's mechanism offers unique benefits. The ablation with global-only variance removes the layer-specific adaptation, directly testing whether per-layer correction is essential. The ablation with learned scaling (trained synchronously) tests whether the simple variance heuristic is necessary or if a learned alternative works better. Perplexity is the primary metric because it captures overall model quality and is sensitive to optimization stability and convergence speed. All experiments use 8 A100 GPUs; each run for the 1B model takes approximately 2 days.

### Expected outcome and causal chain

**vs. Synchronous Muon** — On a case where gradient delay causes staleness (e.g., large model with deep layers), asynchronous Muon produces slower convergence and higher final loss because stale gradients distort orthogonalized updates. Our method scales normalization per layer based on delayed gradient variance, damping updates on high-variance layers that suffer more from staleness, thus approaching synchronous performance. We expect a smaller perplexity gap between synchronous and asynchronous ALN compared to the gap between synchronous and asynchronous Muon, e.g., the gap may shrink by 30-50%.

**vs. Asynchronous Muon (no ALN)** — On a layer where delayed gradients exhibit high variance (e.g., an early transformer layer with noisy gradients), asynchronous Muon applies uniform row normalization, potentially over-updating and causing instability. Our method computes a layer-specific scaling factor that weakens normalization when variance is high relative to the global average, preventing over-correction. We expect ALN to yield lower perplexity (e.g., 0.2-0.5 improvement) and smoother loss curves, especially on deep layers where variance differences are pronounced.

**vs. Asynchronous AdamW** — On a scenario with significant delay (e.g., 1-2 micro-batches), AdamW adjusts per-parameter learning rates based on gradient moments but ignores variance structure induced by delay, leading to suboptimal updates. Our method directly addresses the delay-induced variance mismatch through layer-wise normalization scaling, providing a more principled correction. We expect ALN to achieve lower perplexity (e.g., 0.5-0.8) and faster convergence, with the gap widening as delay increases.

**Synthetic verification:** To isolate the effect of delay on variance, we also test on a synthetic 2-layer MLP (hidden=256, ReLU activation) trained on random data (1000 samples, 100 features, binary classification) with controlled delay (1-4 micro-batches). We expect that ALN improves over asynchronous Muon when delay causes variance mismatch, and that the improvement correlates with the variance ratio.

### What would falsify this idea

If ALN shows no significant perplexity improvement over asynchronous Muon across multiple seeds, or if the improvement is uniformly distributed across layers rather than concentrated on high-variance layers, then the central claim that variance-driven layer-wise adaptation is beneficial would be falsified. Additionally, if the learned scaling ablation outperforms ALN, it would suggest that the variance heuristic is suboptimal.

## References

1. One-Step Gradient Delay is Not a Barrier for Large-Scale Asynchronous Pipeline Parallel LLM Pretraining
2. NorMuon: Making Muon more efficient and scalable
3. Error Feedback for Muon and Friends
4. SOAP: Improving and Stabilizing Shampoo using Adam
5. A Distributed Data-Parallel PyTorch Implementation of the Distributed Shampoo Optimizer for Training Neural Networks At-Scale
6. Low‐rank updates of matrix square roots
