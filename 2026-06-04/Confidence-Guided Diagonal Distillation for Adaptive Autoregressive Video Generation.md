# Confidence-Guided Diagonal Distillation for Adaptive Autoregressive Video Generation

## Motivation

Streaming autoregressive video generation via diagonal distillation (Streaming Autoregressive Video Generation via Diagonal Distillation) uses a globally fixed asymmetry schedule that allocates more denoising steps to early chunks and fewer to later chunks, regardless of actual content complexity. This wastes computation on simple chunks and risks underprocessing complex ones. DSA (Dynamic Step Allocation) demonstrated that a confidence head trained under distribution-matching distillation can dynamically allocate steps per frame, but it does not exploit the diagonal temporal structure. We bridge these approaches by integrating a confidence head into the diagonal distillation framework, replacing the fixed asymmetry with a content-adaptive step allocation that emerges from a distribution-matching objective.

## Key Insight

The distribution-matching distillation objective guarantees that the confidence head's estimate of remaining denoising steps is monotonic in the reconstruction error, a property that holds irrespective of the diagonal temporal arrangement, allowing the generator to allocate steps adaptively across diagonal elements.

## Method

Confidence-Guided Diagonal Distillation (CGDD) is a training and inference framework where a lightweight confidence head, trained jointly with an autoregressive diagonal distillation generator, predicts the number of denoising steps needed for each diagonal element. Input: a video chunk with diagonal noise schedule; Output: a denoised chunk with variable steps per diagonal element.

**Architecture details:** The generator is a 3D U-Net with 200M parameters, identical to the diagonal distillation baseline (prior work). The confidence head is a 2-layer MLP with hidden size 256, GELU activation, and a sigmoid output in [0,1]. It takes as input the latent representation of a diagonal element after applying a denoising step (the current latent of that element).

**Hyperparameters:** Maximum denoising steps K=4, confidence threshold τ=0.8, loss weight λ=0.1, binary threshold θ=0.05 (based on empirical reconstruction error distribution). Training: batch size 16, learning rate 1e-4 for both generator and confidence head (AdamW). The generator is pre-trained with the fixed asymmetry schedule for 500k steps, then fine-tuned with CGDD for 100k steps.

**Load-bearing assumption:** The method assumes that reconstruction error L_dist (MSE in latent space) is a reliable proxy for perceptual and motion quality. This assumption is unverified; if it fails (e.g., low-frequency errors with high temporal incoherence), the confidence head may be miscalibrated.

### Pseudocode for training and inference

```python
# Training
for each batch of video chunk X with teacher target T (from full denoising):
    for each diagonal element d in X:
        sample k_d uniformly from {1,...,K}  # number of steps to apply
        G_rep = generator.apply_denoising(X, k_d)  # apply k_d steps to d
        c_d = confidence_head(current_latent_of_d)  # scalar confidence
        # Distillation loss: MSE between G_rep and T (teacher output after full steps)
        L_dist = MSE(G_rep[d], T[d])
        # Confidence target: 1 if L_dist < threshold θ (0.05) else 0
        target_c = 1 if L_dist < θ else 0
        L_conf = BCE(c_d, target_c)
    total_loss = mean(L_dist) + λ * mean(L_conf)
    update generator and confidence head

# Inference
for each diagonal element d in new chunk:
    for step in reversed(range(1, K+1)):
        latent = apply_denoising_step(d, step)
        if step > 1:
            c = confidence_head(latent)
            if c > τ:
                break  # stop early
    output = final latent
```

### Why this design

We chose to train the confidence head with a binary target (whether current output is already close to teacher) over a continuous target because it makes the head predict a probability of being 'done', which directly enables threshold-based early stopping; the cost is losing fine-grained step count information, but for discrete denoising steps a binary decision suffices. We sample k uniformly during training rather than using the confidence head's own predictions to avoid confirmation bias and ensure coverage of all step counts; the trade-off is that the head may see mismatched training examples, but we compensate with the distribution-matching loss that encourages generalization. We set the threshold θ for target computation to a small constant (0.05) based on empirical reconstruction error distributions; this choice balances sensitivity – too low misses acceptable outputs, too high allows early stopping on poor quality. The loss weight λ=0.1 prioritizes generation quality over confidence calibration, accepting that confidence may be slightly miscalibrated but retaining high generation fidelity. **A load-bearing assumption is that reconstruction error L_dist (MSE in latent space) is a reliable proxy for perceptual and motion quality; this assumption is unverified and may not hold for temporally incoherent outputs.**

### Why it measures what we claim

The confidence head output c measures the probability that the current latent after k steps is sufficiently close to the teacher output, because the BCE loss trains it to predict the binary indicator derived from L_dist; this indicator directly reflects whether additional steps significantly change the output. The assumption is that reconstruction error L_dist is a reliable proxy for perceptual or motion quality – an assumption that fails when the generator produces low-frequency errors that are optically small but temporally incoherent, in which case c may overestimate quality and cause premature stopping. The distillation loss L_dist on the final output measures how well the generator matches the teacher under variable step counts, because it directly compares latent space distances; this assumes the teacher provides a high-quality target, which is standard in distillation and justified by the diagonal distillation baseline. The uniform sampling of k during training ensures that the confidence head sees all possible step counts, preventing distribution mismatch at inference – a design choice that assumes the difficulty distribution is stationary, which may not hold for very long video streams where content complexity drifts, leading to slight calibration drift that we mitigate with periodic fine-tuning.

## Contribution

(1) A confidence-guided diagonal distillation framework that replaces the fixed asymmetry schedule with a learned, content-adaptive step allocation, combining the strengths of DSA and diagonal distillation. (2) A training procedure that jointly optimizes the generator and confidence head under a distribution-matching objective, extending the DSA mechanism to the diagonal temporal structure. (3) Empirical demonstration that adaptive step allocation reduces computational cost while maintaining or improving video quality compared to fixed asymmetry, tested on standard autoregressive video generation benchmarks.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|-----------------------|
| Dataset | UCF-101 | Standard benchmark with varied motion dynamics. |
| Primary metric | FVD | Captures perceptual quality and temporal coherence. |
| Baseline: Fixed-step | Uniform 4 denoising steps per diagonal | No adaptive allocation; all elements same steps. |
| Baseline: Diagonal Distillation | Generator without confidence head | Identical to ours but no early stopping. |
| Baseline: DSA extended to diagonal | DSA step allocation per element, no diagonal structure | Tests if diagonal structure helps adaptive allocation. |
| Ablation: Continuous confidence target | Continuous MSE target for confidence (regression) | Isolates binary thresholding effect. |
| Ablation: Perceptual confidence target | Replace MSE with LPIPS for confidence target | Tests reconstruction error proxy assumption. |
| Compute budget | 1000 GPU-hours on A100 | Enables reproduction and scaling estimates. |

### Why this setup validates the claim

The central claim is that confidence-guided early stopping improves efficiency without sacrificing quality. UCF-101 offers diverse motion dynamics, testing adaptability across static and dynamic content. The fixed-step baseline tests the null hypothesis that uniform steps are sufficient. Diagonal distillation baseline isolates the benefit of the confidence head over the same generator. The DSA baseline tests whether our diagonal-aware approach outperforms a naive per-element adaptive method. The continuous target ablation tests whether binary thresholding is crucial. The perceptual ablation empirically evaluates the load-bearing assumption that reconstruction error correlates with quality. FVD measures both per-frame quality and temporal consistency, sensitive to early stopping artifacts. If our method achieves lower FVD at comparable or fewer steps than fixed-step, and outperforms the uniform-step baseline and DSA, the claim is supported. The ablations help attribute gains to the binary decision mechanism and validate the core assumption.

### Expected outcome and causal chain

**vs. Fixed-step (K=4)** — On a static background diagonal element, the fixed-step baseline applies all 4 denoising steps, risking over-denoising and wasted compute. Our method's confidence head recognizes early (after 1-2 steps) that the latent is close to teacher, triggering early stopping. We expect similar FVD on dynamic regions but better on static regions, and average steps per element ~2.5 vs 4.

**vs. Diagonal Distillation (teacher-agnostic)** — On a fast-motion diagonal element where the teacher target is blurry, the uniform-step baseline cannot adapt, possibly under-denoising. Our method's confidence head predicts low confidence (since L_dist is high) and applies all steps, improving quality. We expect lower FVD on challenging clips for our method, with net FVD improvement of a few points (e.g., 2-3 FVD) at same average steps.

**vs. Ablation (Continuous target)** — On a diagonal element where optimal steps is 3, the continuous head might predict 2.7, leading to early stopping after 2 steps if threshold is 0.8, causing quality drop. The binary head produces a sharp decision, reducing such intermediate halting. We expect binary variant to show lower FVD, especially on clips with mixed difficulty levels, by about 1-2 FVD.

**vs. DSA extended to diagonal** — DSA allocates steps per element independently, ignoring the diagonal structure. Our method leverages the diagonal arrangement to share confidence predictions across temporally related elements. On clips with correlated motion across time, our method should achieve comparable FVD with fewer total steps (e.g., 10% fewer steps on average). On clips with independent motion, both methods perform similarly. We expect our method to match DSA FVD with lower compute.

**vs. Ablation (Perceptual confidence target)** — If the perceptual target ablation (using LPIPS instead of MSE) yields similar FVD to the default MSE-based method, then the assumption holds. If the perceptual variant achieves lower FVD (especially on temporally incoherent samples), then the MSE-based method is suboptimal. We expect the perceptual variant to slightly outperform MSE-based on clips with high motion but at higher computational cost for computing LPIPS.

### What would falsify this idea

If the binary confidence head yields similar or worse FVD compared to uniform steps across all subsets, or if the latency gains are uniform across all diagonal elements rather than concentrated on simpler ones, then the confidence head is not exploiting difficulty variation as intended.

## References

1. DSA: Dynamic Step Allocation for Fast Autoregressive Video Generation
2. Streaming Autoregressive Video Generation via Diagonal Distillation
3. MAGI-1: Autoregressive Video Generation at Scale
4. Long-Context Autoregressive Video Modeling with Next-Frame Prediction
5. HunyuanVideo: A Systematic Framework For Large Video Generative Models
6. ACDiT: Interpolating Autoregressive Conditional Modeling and Diffusion Transformer
7. LTX-Video: Realtime Video Latent Diffusion
