# Continuous Latency Adaptation for Real-Time Speech Enhancement via Differentiable Time Warping

## Motivation

Prior work, such as 'One Model, Many Latencies', achieves flexible latency by using parallel convolutional layers for discrete look-ahead settings. However, it cannot adapt to continuous latency requirements or varying deployment constraints because the look-ahead is quantized to pre-defined budgets. This forces a trade-off between model capacity and latency granularity, and prevents smooth trade-offs between quality and delay. The root cause is the discrete representation of look-ahead, which inherently limits adaptation to unseen latency values.

## Key Insight

A differentiable warping of the input time axis, conditioned on a scalar latency parameter, effectively simulates any desired look-ahead by non-linearly scaling the temporal context, enabling a single enhancement backbone to cover the entire latency continuum without retraining.

## Method

### Latency-Warped Enhancement Network (LWEN)

**(A) What it is:** The Latency-Warped Enhancement Network (LWEN) takes a noisy waveform x and a continuous latency parameter λ ∈ [0,1] as input, and outputs an enhanced waveform ŷ with an effective look-ahead proportional to λ. Internally, a differentiable time-warping module non-linearly remaps the input time axis via a learned monotonic warping function, and a shared causal convolutional recurrent backbone processes the warped signal.

**(B) How it works (pseudocode):**
```python
def lwen(x, λ, network):
    # x: (batch, T) waveform, λ: scalar in [0,1]
    # Step 1: Generate warping grid
    t_orig = linspace(0, 1, T)  # normalized time
    t_warped = warp_fn(t_orig, λ)  # learned monotonic function; outputs values in [0,1]
    # warp_fn: 2-layer MLP (hidden=128, GeLU) with positive weights and cumsum for monotonicity, derivative bounded to [0.5,2] via sigmoid scaling
    # Step 2: Apply differentiable resampling via cubic spline interpolation (order=3, grid oversampling factor=20)
    x_warped = interp1d(t_orig, x, t_warped)  # shape (batch, T)
    # Step 3: Feed to causal enhancement backbone (conv + LSTM)
    # Backbone expects a conditioning vector for λ: concat λ to frame features
    z = encoder(x_warped)  # per-frame features (DCCRN encoder: 5 conv layers, channels=32,64,128,256,256, kernel=(5,2), stride=(2,1))
    z = concat(z, repeat(λ, num_frames))  # condition on λ
    y_hat = decoder(z)  # (batch, T) (DCCRN decoder: 5 deconv layers mirroring encoder, with skip connections)
    return y_hat
```
**Calibration:** A synthetic calibration set of 100 impulse responses with known delay (0–20ms) is used to build a lookup table mapping λ to effective look-ahead (ms). For each λ, a Dirac impulse at time t0 is fed, and the first nonzero output before t0 in the warped backbone is measured; this yields the effective look-ahead. During inference, given a desired latency, λ is selected from this table.

**(C) Why this design:** We chose differentiable spline-based warping over discrete parallel branches (as in 'One Model, Many Latencies') because it allows a single backbone to handle a continuum of latencies, avoiding the overhead of multiple decoders and enabling smooth transitions between budgets. We accept the cost that warping may introduce interpolation artifacts, mitigated by oversampling and a smoothness regularizer on the warping function. We condition the backbone by concatenating λ to frame features rather than using FiLM, because FiLM requires additional parameters and may not generalize across extreme scalings; concatenation is simpler and the backbone can learn to ignore it for extreme λ. We use a learned monotonic warping function (MLP with positive output weights + cumsum) rather than a fixed parametric form (e.g., power-law) because the optimal warping may be data-dependent; the trade-off is increased training complexity. Finally, we use a causal backbone to ensure no future information beyond the warped receptive field is used; the warping itself controls effective look-ahead.

**(D) Why it measures what we claim:** The warping function's derivative at each time point, conditioned on λ, determines the effective receptive field size (look-ahead): a larger derivative means more input samples are compressed into a fixed output window, effectively increasing look-ahead. The scalar λ directly controls this derivative via the learned warping MLP; thus, varying λ from 0 to 1 smoothly adjusts the look-ahead from 0 ms to a maximum set by the interpolation grid. The critical assumption is that the effect of warping on the backbone's causal convolutions is equivalent to observing more future frames; this holds because the backbone sees only the warped signal and lacks access to original timing. This assumption fails when the warping is too aggressive (e.g., heavy compression) causing aliasing or loss of detail, in which case λ no longer faithfully represents a physical look-ahead but rather a distortion level. To quantify when equivalence holds, we define a 'warping fidelity' metric: the peak-to-side-lobe ratio (PSLR) of the cross-correlation between the center sample's impulse response from the warped model and that of a model with true look-ahead (e.g., non-causal model with future window). A PSLR ≥ 20 dB indicates good equivalence. We enforce a soft regularizer during training to keep PSLR above 15 dB on a held-out set of synthetic impulses.

## Contribution

(1) A differentiable time-warping module that enables continuous latency control in speech enhancement with a single scalar parameter, replacing discrete parallel branches. (2) A training framework that jointly optimizes the warping and backbone across the entire latency continuum without retraining for each budget. (3) Demonstration that the continuous model can achieve performance on par with discrete multi-latency models on the DNS Challenge benchmark while supporting any arbitrary latency between 0 and max look-ahead.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | DNS Challenge (noisy speech) | Standard real-world noise, diverse conditions |
| Synthetic dataset | 1000 clean utterances convolved with RIRs of known delay (0–20ms) | Measure effective look-ahead and warping fidelity |
| Primary metric | PESQ (perceptual quality) | Sensitive to artifacts and latency trade-off |
| Warping fidelity metric | Peak-to-side-lobe ratio (PSLR) on impulse responses | Quantifies equivalence of warping to true look-ahead |
| Baseline 1 | One Model, Many Latencies | Discrete multi-branch, tests smoothness |
| Baseline 2 | DCCRN (fixed 5ms) | Strong causal model, fixed look-ahead |
| Baseline 3 | FullSubNet (fixed 10ms) | Another fixed-latency predictive model |
| Baseline 4 | LWEN with fixed power-law warp (λ^α, α learned) | Isolate benefit of learned monotonic warping |
| Ablation | LWEN with fixed λ=0.5 | Tests necessity of λ conditioning |

### Why this setup validates the claim

The combination of DNS Challenge, PESQ, and these baselines provides a direct falsifiable test of whether the differentiable warping enables continuous latency-quality trade-off via a single backbone. DNS is the de facto benchmark for real-time enhancement, ensuring ecological validity. PESQ captures both quality and artifacts from warping, which is the primary risk. The synthetic dataset with known RIRs allows precise measurement of effective look-ahead and warping fidelity (PSLR), directly testing the core assumption. Each baseline targets a distinct sub-claim: "One Model" tests whether a single backbone can match multiple discrete branches; DCCRN (5ms) tests whether our method at low λ can compete with dedicated low-latency models; FullSubNet (10ms) tests whether higher λ yields quality comparable to a high-latency model; fixed power-law warp tests whether the learned warp provides additional benefit. The ablation (fixed λ) tests whether the conditioning on λ is actually responsible for any observed flexibility. If the method's performance is not better than all baselines on their respective latency regimes, or if the ablation performs similarly, the central claim fails.

### Expected outcome and causal chain

**vs. One Model, Many Latencies** — On a test utterance requiring λ=0.3 (an intermediate look-ahead not covered by discrete branches), the discrete model must choose either λ=0 or λ=0.5 branch, causing either insufficient look-ahead (missing future context) or excessive delay (violating latency budget). Our method's warping function exactly realizes λ=0.3, producing a better quality-latency trade-off. We expect PESQ of our method to be at least 0.15 higher than the closest discrete branch on intermediate λ values, with parity on extreme λ (0 and 1).

**vs. DCCRN (fixed 5ms)** — On an utterance with a sudden noise burst (e.g., door slam), DCCRN's fixed 5ms look-ahead may miss the onset, causing a distorted transient. Our method with λ near 1 (e.g., 0.9) can allocate larger look-ahead via stronger time-warping, effectively capturing the burst envelope. We expect our PESQ on transient-rich segments to be 0.3 higher than DCCRN, while on steady noise (e.g., fan hum) performance should be similar.

**vs. FullSubNet (fixed 10ms)** — In a low-latency hearing aid scenario requiring <5ms, FullSubNet's 10ms delay is unacceptable. Our method with λ=0.1 achieves comparable look-ahead while preserving quality close to FullSubNet's native level. We expect our PESQ at λ=0.1 to be within 0.05 of FullSubNet's best PESQ, but with half the algorithmic latency, demonstrating a better latency-quality Pareto front.

**vs. Fixed power-law warp** — On a test set with diverse noise types, the learned warp should adapt better than a fixed parametric form. We expect PESQ of LWEN (learned warp) to exceed fixed power-law warp by 0.1 on average, and PSLR to be 5 dB higher, confirming the benefit of data-dependent warping.

**Real-time adaptation scenario:** We demonstrate a hearing aid use case where CPU load triggers dynamic λ adjustment: when CPU usage exceeds 80%, λ is reduced by 0.2 (reducing warping grid oversampling from 20× to 10×) and we measure PESQ drop less than 0.05 while processing latency halves.

### What would falsify this idea

If the PESQ of LWEN does not increase monotonically with λ, or if at any λ the performance is strictly worse than both fixed-latency baselines operating at similar effective look-ahead, then the warping fails to provide a meaningful trade-off or introduces excessive distortion. Additionally, if the measured effective look-ahead from the synthetic experiment has Pearson correlation <0.9 with λ, or if PSLR < 15 dB on average, the core assumption is violated and the method's design is invalid.

## References

1. One Model, Many Latencies: Universal Speech Enhancement for Diverse Real-Time Applications
2. Diffusion Buffer for Online Generative Speech Enhancement
3. Towards a flexible and unified architecture for speech enhancement
4. Scalable Speech Enhancement With Dynamic Channel Pruning
5. Early-Exit Deep Neural Network - A Comprehensive Survey
6. Rolling Diffusion Models
7. FIFO-Diffusion: Generating Infinite Videos from Text without Training
8. Slim-Tasnet: A Slimmable Neural Network for Speech Separation
