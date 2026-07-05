# Signal-Adaptive Early Exit via Confidence-Weighted Branch Aggregation for Multi-Latency Speech Enhancement

## Motivation

Fixed-depth early exits in multi-latency architectures (e.g., One Model, Many Latencies) fail to adapt to varying signal difficulty, wasting compute on easy samples or underperforming on hard ones. The root cause is that exit decisions are not conditioned on the input signal's quality—the architecture lacks a mechanism to dynamically weight branches based on per-sample confidence.

## Key Insight

The confidence of a branch's output, estimated from the input signal's SNR, provides a principled, signal-dependent weight that can be used to combine branches continuously, achieving adaptive quality-latency trade-off without a separate router.

## Method

## Signal-Adaptive Weighted Aggregation (SAWA)

(A) **What it is**: We propose Signal-Adaptive Weighted Aggregation (SAWA), a mechanism that replaces hard early-exit decisions in a multi-latency convolutional architecture with a soft, confidence-weighted combination of branch outputs, where the confidence weights are predicted by a lightweight SNR estimator integrated into the shared encoder. The SNR estimator consists of 2 convolutional layers with kernel size 3 and 16 channels, adding approximately 0.12 GFLOPS per frame (assuming a 257×128 frequency-time input grid). Inputs: multi-branch outputs and noisy input spectrum; output: final enhanced frame.

(B) **How it works** (pseudocode):
```python
def forward(branch_outputs, input_spec):
    # branch_outputs: list of tensors from N=3 branches (shallow, medium, deep)
    # input_spec: magnitude spectrum of noisy input (shape [batch, freq, time])
    
    # Step 1: Estimate per-frame SNR
    snr_est = snr_estimator(input_spec)  # 2 conv layers, kernel 3, 16 channels; output scalar per frame
    
    # Step 2: Compute confidence weights for each branch
    # Learned mapping: for branch i, confidence = sigmoid(alpha_i * snr_est + beta_i)
    alpha = nn.Parameter(torch.randn(N))  # trainable per-branch slope, initialized N(0,1)
    beta = nn.Parameter(torch.randn(N))   # trainable per-branch bias, initialized N(0,1)
    confidences = []
    for i in range(N):
        conf_i = torch.sigmoid(alpha[i] * snr_est + beta[i])
        confidences.append(conf_i)
    confidences = torch.stack(confidences, dim=0)  # [N, batch]
    
    # Step 3: Normalize confidences to sum to 1 across branches
    normalized_weights = torch.softmax(confidences, dim=0)  # [N, batch]
    
    # Step 4: Weighted sum of branch outputs
    final_output = sum(normalized_weights[i] * branch_outputs[i] for i in range(N))
    return final_output
```
Hyperparameters: `N=3` (low/medium/high depth); the SNR estimator uses 2 convolutional layers with kernel size 3 and 16 channels; training uses the SI-SNR loss with no additional regularization. The network is trained end-to-end using Adam optimizer with learning rate 1e-3 and batch size 16 for 100 epochs.

(C) **Why this design**: We chose a per-branch sigmoid function of SNR to generate confidences, rather than a full neural network, because the linear-in-sigmoid form provides interpretability and avoids overfitting to limited SNR conditions, at the cost of potentially missing complex interactions between branches. We normalize weights via softmax over branches to ensure a convex combination, preserving the total energy, rather than using a winner-take-all selection that would be brittle. We integrate the SNR estimator into the shared encoder rather than requiring a separate model, leveraging shared representations to reduce overhead; the trade-off is that the encoder must be jointly trained for both enhancement and SNR estimation, which may degrade primary task performance if not carefully balanced. The weights are input-dependent but do not require per-branch confidence classifiers, simplifying training and inference.

(D) **Why it measures what we claim**: The SNR estimate measures signal confidence because, under the assumption that higher SNR corresponds to easier enhancement with less needed depth (Assumption A), a branch's weight should increase with SNR if it has lower latency (shallower). This assumption fails when noise is non-stationary or when distortion is not captured by SNR (e.g., reverberation) (Failure mode F), in which case the weight may misallocate compute. The sigmoid mapping from SNR to confidence operationalizes the 'signal-adaptivity' concept by providing a monotonic function; the sigmoid's saturation ensures weights stay bounded, preventing extreme values. The softmax normalization ensures the combined output is a convex combination, maintaining the interpretation that each branch contributes proportionally to its estimated reliability; this assumption fails when branches are not calibrated to share the same output distribution, leading to potential artifacts from averaging conflicting outputs. The joint training of the SNR estimator with the main network ensures that the confidence weights are optimized for the final enhancement objective, but this assumption that the gradient through the softmax weight is informative may be violated if the SNR estimator is poorly conditioned at start.

**Load-bearing assumption**: The SNR estimate is a sufficient statistic for determining the optimal per-branch confidence weights for speech enhancement. **Calibration**: We will verify this assumption by performing a synthetic experiment: we generate clean utterances, add white Gaussian noise at SNRs ranging from 0 to 20 dB in 2 dB steps, and compute the oracle optimal branch weights (the convex combination that maximizes SI-SNR on a validation set). We then check whether the learned alpha/beta parameters produce a monotonic mapping consistent with the oracle trend (i.e., shallower branches get higher weight as SNR increases). A Spearman rank correlation >0.9 between predicted and oracle weights across SNR levels is expected.

## Contribution

(1) A signal-adaptive early exit mechanism, SAWA, that replaces fixed-depth exits with confidence-weighted branch aggregation in multi-latency architectures. (2) The design principle that a simple SNR-based confidence mapping, trained jointly, achieves dynamic compute allocation without a separate router. (3) An extension of the multi-latency paradigm (One Model, Many Latencies) to support continuous latency-quality trade-off.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|------------------------|
| Dataset | DNS Challenge (synthetic validation) | Standard real-time enhancement benchmark with diverse SNRs |
| Supplementary datasets | Deep Noise Suppression (DNS) real test, AEC-Challenge | Broader real-world applicability |
| Primary metric | SI-SNR | Sensitive to signal quality and distortion |
| Baseline 1 | Hard early-exit (deepest branch) | Non-adaptive deep baseline |
| Baseline 2 | Uniform branch averaging | No adaptation baseline |
| Baseline 3 | Learned fixed weights (softmax of trainable scalars) | Non-adaptive but learnable |
| Baseline 4 | Learned router (RL-based; e.g., policy gradient) | Highlights simplicity of our deterministic approach |
| Ablation | SAWA w/o SNR estimator (replace with constant) | Tests need for SNR input |
| Synthetic verification | Oracle weight analysis on controlled SNR sweep | Empirically confirm monotonic confidence-SNR trend |

### Why this setup validates the claim
This experimental design isolates whether SNR-adaptive weighting improves speech quality over non-adaptive alternatives. DNS Challenge provides diverse SNRs (0–20 dB), covering the regime where adaptivity should matter. Supplementary datasets (DNS real test, AEC-Challenge) test generalization to non-stationary noise and echo scenarios. SI-SNR directly measures enhancement quality and penalizes both residual noise and distortion. Hard early-exit tests the necessity of shallower branches; uniform averaging tests the benefit of any weighting; learned fixed weights tests the value of input-dependent weights; the RL router baseline highlights the simplicity of our deterministic approach. The ablation removes the SNR estimator to verify its role as the source of adaptivity. The synthetic oracle analysis directly tests whether SNR is a sufficient proxy for optimal weights. If our method outperforms all baselines especially on high-SNR (where deep branches hurt) and low-SNR (where shallow branches are weak) subsets, the claim is supported; uniform gains would falsify the adaptivity hypothesis.

### Expected outcome and causal chain

**vs. Hard early-exit (deepest branch)** — On a high-SNR utterance (e.g., 20 dB), the deep branch may introduce over-suppression artifacts because it is trained to handle noisy inputs. The baseline always uses this deepest output, causing unnecessary distortion. Our method instead assigns higher confidence to shallower branches when SNR is high, as the SNR estimator produces a high value, and the learned sigmoid mapping for shallow branches (likely positive slope) yields larger weights. Thus, we expect a noticeable SI-SNR improvement of ~0.5–1 dB on high-SNR subsets, with parity on low-SNR subsets.

**vs. Uniform branch averaging** — On a low-SNR utterance (e.g., 0 dB), shallow branches fail to suppress noise adequately, but uniform averaging gives them equal weight, leading to residual noise. Our method detects low SNR and increases confidence on deeper branches (which have stronger suppression) via the SNR estimator and sigmoid mapping. Consequently, we expect better noise suppression, reflected by ~0.5–1 dB higher SI-SNR on low-SNR subsets, while performance on moderate SNRs is similar.

**vs. Learned fixed weights** — On a non-stationary noise clip (e.g., babble noise varying in level), fixed weights cannot adapt per frame, so the combination may be suboptimal when SNR fluctuates. Our method adjusts weights frame-by-frame based on instantaneous SNR, better tracking the optimal branch blend. We expect a moderate gain (~0.3–0.5 dB) on non-stationary noise examples, with parity on stationary noise.

**vs. Learned router (RL-based)** — The RL router requires a separate training stage and may converge to a suboptimal policy. Our deterministic SNR-based method is simpler and should achieve comparable or better performance without reinforcement learning. We expect SI-SNR within ±0.2 dB of the RL router, but with lower variance and faster inference (no policy evaluation).

**Synthetic oracle analysis**: We will generate clean speech from the DNS dataset, add stationary white Gaussian noise at SNRs 0,2,…,20 dB, and compute for each SNR the optimal convex combination of branch outputs that maximizes SI-SNR. We then compare the learned weights (from our SNR estimator) to the oracle weights. We expect the learned weights to be monotonic in SNR (higher SNR → higher weight on shallower branch) and to have a Spearman rank correlation >0.9 with the oracle weights across the SNR range.

### What would falsify this idea
If SAWA does not outperform uniform averaging on low-SNR subsets, or if its SI-SNR gain is uniformly distributed across all SNR ranges (instead of concentrated where adaptivity is expected to matter), then the central claim of signal-adaptive benefit is falsified. Additionally, if the learned SNR-to-weight mapping is not monotonic in the synthetic experiment (i.e., correlation <0.5), the assumption that SNR is a sufficient statistic would be violated, undermining the mechanism's interpretability.

## References

1. One Model, Many Latencies: Universal Speech Enhancement for Diverse Real-Time Applications
2. Diffusion Buffer for Online Generative Speech Enhancement
3. Towards a flexible and unified architecture for speech enhancement
4. Scalable Speech Enhancement With Dynamic Channel Pruning
5. Early-Exit Deep Neural Network - A Comprehensive Survey
6. Rolling Diffusion Models
7. FIFO-Diffusion: Generating Infinite Videos from Text without Training
8. Slim-Tasnet: A Slimmable Neural Network for Speech Separation
