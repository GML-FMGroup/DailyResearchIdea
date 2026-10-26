# Orthogonal Subspace Fusion with Evidence-Prior Identifiability for Omni LLM Hallucination Mitigation

## Motivation

Existing inference-time hallucination correction methods like Omni-Woodpecker rely on a post-hoc orthogonal decomposition of fused hidden states into evidence and prior components, but this decomposition is not identifiable because the fusion pipeline is trained end-to-end without any architectural constraint enforcing that evidence and prior subspaces remain separable. As a result, priors and evidence become deeply entangled in the fused representation, and the training-free correction cannot reliably isolate them, especially when conflicts are subtle. This identifiability gap is structural: neither frozen-encoder alignment nor standard end-to-end training enforces modality-view separability or linear separability of evidence and prior, so any downstream detector or decomposer operates on representations where the target components are not identifiable.

## Key Insight

Enforcing an orthogonal structural bias between evidence and prior subspaces during fusion training makes the evidence-prior decomposition identifiable from fused hidden states, because the training objective guarantees that the two components lie in mutually orthogonal subspaces that are linearly separable by construction.

## Method

```pseudocode
# Orthogonal Subspace Fusion (OSF) training and inference
# Backbone: Qwen2.5-Omni-7B (frozen encoders, trainable fusion layers)

Input: multimodal sequence X = [x_text, x_audio, x_visual]
Output: fused hidden states H_fused = H_evidence + H_prior

# Phase 1: Architectural split
1. For each modality m in {text, audio, visual}:
   a. Compute modality-specific view V_m = Encoder_m(x_m)  # frozen
   b. Project V_m to evidence subspace: E_m = W_E_m @ V_m  # dim d_E=256, W_E_m ∈ R^{256×d_m}
   c. Project V_m to prior subspace: P_m = W_P_m @ V_m    # dim d_P=256, W_P_m ∈ R^{256×d_m}
   d. Concatenate across modalities: E = [E_text; E_audio; E_visual] ∈ R^{768}, P = [P_text; P_audio; P_visual] ∈ R^{768}

# Phase 2: Orthogonality constraint
2. Compute orthogonality loss: L_orth = ||E^T P||_F^2 / (d_E * d_P)
   # Frobenius norm of cross-covariance, normalized; d_E=d_P=256 per modality, 768 total
   # Also compute nonlinear cross-covariance diagnostic: HSIC(E, P) with Gaussian kernel (σ=1.0) on a held-out probe set of 512 examples, reported but not optimized

# Phase 3: Fusion and rebalancing
3. Fused representation: H_fused = E + P  # additive composition
4. Conflict detector: C = sigmoid(MLP([V_text; V_audio; V_visual]))  # scalar per token; MLP: 2-layer, hidden=512, GeLU, dropout=0.1
5. Rebalanced output: H_out = E + (1 - C) * P  # suppress prior when conflict high

# Phase 4: Training objective
6. L_total = L_task + lambda_orth * L_orth + lambda_conf * L_conf
   # L_task: standard next-token prediction loss (cross-entropy)
   # L_conf: binary cross-entropy for conflict detection (labels from synthetic conflicts; see experiment)
   # lambda_orth = 0.1, lambda_conf = 0.05
   # Training: AdamW, lr=1e-4 for fusion layers, batch size=32, 3 epochs, linear warmup 500 steps

# Phase 5: Inference-time correction (training-free, but now identifiable)
7. At inference, given H_fused, solve for E and P via orthogonal projection:
   E_hat = Proj_E(H_fused), P_hat = Proj_P(H_fused)
   # Because E and P are orthogonal by training, Proj_E and Proj_P are exact for the linear component;
   # we measure projection error ||E_hat - E|| on held-out data (see experiment).
8. Apply conflict-conditioned rebalancing as in Omni-Woodpecker:
   H_corrected = E_hat + (1 - C) * P_hat
9. Decode from H_corrected
```

**Why this design**
We chose an architectural split with separate evidence and prior subspaces over a learned disentanglement head because the underlying task couples evidence and prior through the same fusion layers, and no learned head can manufacture independence that the architecture does not enforce; the orthogonal projection constraint guarantees linear separability by construction, accepting the cost that the subspaces are fixed-dimension and may underfit if the true evidence-prior manifold is higher-dimensional. We chose an additive fusion (H_fused = E + P) over concatenation or gated fusion because additivity makes the orthogonal decomposition exact and invertible at inference, accepting the cost that the model cannot learn multiplicative interactions between evidence and prior. We chose a conflict detector trained with synthetic conflict labels over a self-supervised contrastive objective because synthetic labels provide direct supervision for the rebalancing gate, accepting the cost that the detector may not generalize to real conflicts not represented in the synthetic distribution. Unlike Omni-Woodpecker, which applies a post-hoc orthogonal decomposition to a fused representation that was never trained to be separable, our method trains the fusion to be orthogonal by design, so the decomposition is exact rather than approximate.

**Why it measures what we claim**
The orthogonality loss L_orth = ||E^T P||_F^2 measures evidence-prior identifiability because the assumption that orthogonal subspaces imply linear separability holds when the subspaces are truly orthogonal; this assumption fails when the projection matrices W_E and W_P are rank-deficient, in which case L_orth reflects numerical noise rather than true separation. The conflict detector C measures cross-modal conflict because the assumption that synthetic conflict labels are representative of real conflicts holds when the synthetic distribution covers the conflict modes; this assumption fails when real conflicts arise from subtle temporal misalignment not present in synthetic data, in which case C reflects the detector's bias toward synthetic patterns. The rebalanced output H_out = E + (1-C)*P measures evidence-prior rebalancing because the assumption that suppressing P when C is high reduces hallucination holds when P indeed contains misleading priors; this assumption fails when P contains legitimate contextual information, in which case H_out may suppress useful signals. The inference-time projection E_hat = Proj_E(H_fused) measures the evidence component because the assumption that H_fused = E + P with E and P orthogonal makes the projection exact; this assumption fails when the fusion introduces nonlinearities that break additivity, in which case E_hat reflects an approximation error. To ground this, we report the projection error ||E_hat - E|| on held-out conflict data and the condition number of the projection matrices.

## Contribution

['A novel architectural constraint for multimodal fusion that enforces orthogonal evidence and prior subspaces during end-to-end training, making the evidence-prior decomposition identifiable from fused hidden states by construction.', 'A conflict-conditioned rebalancing mechanism that operates on the identifiable decomposition, enabling reliable suppression of misleading priors and amplification of grounding evidence at inference time without additional training.', 'An empirical demonstration that training with the orthogonality constraint improves hallucination mitigation on CMM and AVHBench benchmarks compared to post-hoc decomposition methods, with consistent gains across general reasoning tasks.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | CMM + AVHBench hallucination benchmarks | Directly probe omni-modal evidence-prior conflicts |
| Primary metric | Hallucination rate (lower is better) | Directly measures the target failure mode |
| Baseline | Unmitigated Qwen2.5-Omni-7B | Isolates gains from any mitigation at all |
| Baseline | Omni-Woodpecker (post-hoc projection) | Tests exact vs approximate decomposition claim |
| Baseline | Output-level hallucination mitigation | Tests architectural vs output-level correction |
| Ablation-of-ours | OSF w/o orthogonality loss (λ_orth=0) | Isolates contribution of enforced separability |
| Diagnostic | Projection error ||E_hat - E|| and HSIC(E,P) | Measures identifiability directly |

### Why this setup validates the claim

The central claim is that training fusion to be orthogonal by construction makes evidence-prior decomposition exact, and that this exactness—not merely the presence of a rebalancing gate—is what reduces hallucination. CMM and AVHBench are the right datasets because they contain omni-modal instances where audio or visual evidence contradicts a strong linguistic prior, exactly the regime where identifiability matters. The unmitigated backbone establishes the ceiling of the raw hallucination problem; the output-level baseline tests whether architectural separation beats post-hoc correction; Omni-Woodpecker is the critical comparison because it shares our rebalancing mechanism but applies projection to a representation never trained to be separable, so any gap between it and OSF isolates the value of enforced orthogonality. The ablation with λ_orth=0 removes the constraint while keeping the architecture and gate, so if the ablation matches full OSF, the orthogonality loss is doing no causal work and the claim collapses. Hallucination rate is the right metric because it directly counts the predicted failure event rather than a proxy; a decomposition-quality proxy could improve without any behavioral change. To further ground the exactness claim, we report projection error ||E_hat - E|| on a held-out set of 512 conflict examples (where E is known from the forward pass) and the nonlinear cross-covariance HSIC(E,P); these diagnostics directly test whether the orthogonal decomposition is exact in practice, not just in the linear projection.

### Expected outcome and causal chain

**vs. Unmitigated Qwen2.5-Omni-7B** — On a case where the audio says "no" but the text prompt strongly implies affirmation, the unmitigated backbone produces the prior-consistent answer because its fusion layers have no mechanism to separate evidence from prior, so the prior dominates the fused state. Our method instead routes the conflicting evidence through E and suppresses P via the conflict gate C, because the orthogonal split lets the gate act on an identifiable prior component rather than a tangled mixture. We expect a large hallucination-rate drop concentrated on conflict-heavy subsets, with near-parity on conflict-free subsets where no rebalancing is needed. We also expect projection error ||E_hat - E|| to be low (< 0.1 relative) on conflict examples, confirming exactness.

**vs. Omni-Woodpecker** — On a case where evidence and prior are entangled in the fused representation, Omni-Woodpecker's post-hoc projection yields an approximate E_hat with leakage from P because the representation was never trained to be orthogonal, so its rebalancing partially suppresses genuine evidence and leaves residual prior. Our method instead projects onto subspaces that are orthogonal by training, so Proj_E and Proj_P are exact and the gate cleanly attenuates only P. We expect a moderate but consistent gap over Omni-Woodpecker, largest on instances with strong prior-evidence conflict and small on easy instances. Additionally, we expect OSF to have lower HSIC(E,P) than Omni-Woodpecker, indicating better nonlinear separation.

**vs. OSF w/o orthogonality loss (λ_orth=0)** — On a case where the conflict detector fires correctly, the ablation still cannot cleanly separate E from P because without the orthogonality constraint the two projections overlap, so rebalancing suppresses evidence along with prior and the corrected output drifts. Our method instead keeps L_orth low, so the additive decomposition remains invertible and the gate's attenuation is targeted. We expect the ablation to show partial gains over the backbone but a clear deficit relative to full OSF on conflict subsets, confirming that the constraint—not the architecture alone—drives the effect. We also expect the ablation to have higher projection error and HSIC, linking the constraint to measurable identifiability.

### What would falsify this idea

If OSF's hallucination-rate gain over Omni-Woodpecker is uniform across conflict-heavy and conflict-free subsets rather than concentrated where evidence-prior conflict occurs, or if the λ_orth=0 ablation matches full OSF, then enforced orthogonality is not the causal mechanism and the central claim is wrong. Additionally, if projection error ||E_hat - E|| is high or HSIC(E,P) remains high despite low L_orth, the exactness claim is falsified.

## References

1. Omni-Woodpecker: Hallucination Correction via Evidence-Prior Rebalancing for Omni LLMs
2. Qwen2.5-Omni Technical Report
3. CosyVoice 2: Scalable Streaming Speech Synthesis with Large Language Models
4. MIO: A Foundation Model on Multimodal Tokens
5. F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching
