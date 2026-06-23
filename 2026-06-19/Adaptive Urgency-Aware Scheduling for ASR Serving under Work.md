# Adaptive Urgency-Aware Scheduling for ASR Serving under Workload Drift

## Motivation

Prior duration-aware schedulers (e.g., HRRN in 'Duration Aware Scheduling for ASR Serving Under Workload Drift') implicitly assume that audio duration is a stable proxy for processing time, but under workload drift this proxy becomes biased and breaks the implicit tail-latency bound. The root cause is that static urgency calculations fail to adapt to changes in the relationship between duration and actual processing time, causing long requests to be further delayed when drift increases their processing time disproportionately.

## Key Insight

Workload drift manifests as a low-dimensional multiplicative bias between audio duration and actual processing time, which can be tracked online with a single exponential moving average, enabling a corrected urgency metric that preserves the tail-latency stability of HRRN without requiring complex learned models.

## Method

AUAS (Adaptive Urgency-Aware Scheduler) is an online scheduling algorithm for ASR inference that estimates a multiplicative drift factor from completed requests and uses it to adjust the predicted processing time in an HRRN-style urgency score. Input: a stream of requests each with audio duration \(d_i\); output: a scheduling order determined by the highest urgency \(U_i\).

**Load-bearing assumption (explicit):** Actual processing time \(t_j\) equals \(\alpha \cdot d_j \cdot \gamma\) plus negligible noise, where \(\gamma\) is a global multiplicative drift factor independent of duration. This assumption is verified by checking that the ratio \(t_j / (\alpha d_j)\) is approximately constant across durations in a held-out calibration set.

**How it works**
```pseudocode
Initialize: gamma = 1.0, alpha_ema = 0.9, alpha (calibration constant from offline profiling)

For each completed request j:
    actual_time = t_j
    predicted_baseline = alpha * d_j
    ratio_j = t_j / predicted_baseline   // raw deviation
    gamma = alpha_ema * gamma + (1 - alpha_ema) * ratio_j   // EMA update

For each waiting request i:
    predicted_time_i = gamma * alpha * d_i
    waiting_time_i = current_time - arrival_time_i
    urgency_i = (waiting_time_i + predicted_time_i) / predicted_time_i   // response ratio

When a scheduling decision is needed:
    select request with highest urgency_i
```
Hyperparameters: \(\alpha\) is calibrated once on a held-out dataset (e.g., 1.05 for Whisper models); \(\alpha_{\text{ema}} = 0.9\) balances responsiveness and noise.

**Why this design**
We chose an exponential moving average (EMA) over a sliding window or batch update because EMA smoothly tracks gradual drift without storing history, at the cost of slower reaction to sudden bursts. We calibrate \(\alpha\) offline to ensure unbiased prediction under static conditions; this decouples baseline calibration from drift adaptation, but requires re-calibration for new model architectures. We retain HRRN's response ratio formula (rather than switching to SJF or a learned rank) because it naturally prevents starvation by incorporating waiting time, which is critical for tail latency under drift. The drift factor \(\gamma\) is updated per completed request rather than in mini-batches to capture drift quickly, accepting minimal per-request overhead (a single division and moving average). We deliberately avoid a learned proxy model (as in SSJF) because drift is a low-dimensional, multiplicative phenomenon amenable to a single parameter, sidestepping the complexity and retraining costs of neural predictors. These choices collectively ensure the scheduler remains simple, interpretable, and robust to slow drift while maintaining the latency-fairness properties of HRRN.

**Why it measures what we claim**
The computational quantity \(\texttt{gamma}\) measures the multiplicative drift factor relative to the static duration proxy, because we assume that actual processing time \(t_j\) equals \(\alpha \cdot d_j \cdot \text{drift}\) plus negligible noise; this assumption fails when drift is non-multiplicative (e.g., additive offset or model changes that affect short and long requests differently), in which case \(\texttt{gamma}\) reflects an average bias rather than a per-request correction. The urgency \(U_i = (w_i + p_i)/p_i\) measures the inverse of slowdown (the ratio of total latency to predicted processing time), which under accurate \(p_i\) yields priority equivalent to Shortest Remaining Processing Time with fairness guarantees (HRRN); this equivalence relies on the assumption that \(p_i\) is a consistent estimator of true processing time, which is violated when \(\texttt{gamma}\) fails to capture non-multiplicative drift, causing \(U_i\) to misrank requests and potentially worsening tail latency. We verify the multiplicative assumption by computing the Pearson correlation between \(d_i\) and \(t_i/(\alpha d_i)\) on a held-out set; a low correlation (<0.3) indicates non-multiplicative drift, triggering a fallback to static HRRN.

## Contribution

(1) An adaptive urgency-aware scheduling algorithm that uses online drift estimation via an exponential moving average to correct the duration proxy in ASR serving. (2) The design principle that tail latency stability under workload drift can be maintained with a single-parameter online tracker rather than complex learned predictors, offering a practical trade-off between adaptability and simplicity. (3) Integration guidelines for incorporating drift estimation into existing vLLM-based ASR serving systems.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | LibriSpeech test-clean | Varied duration, realistic ASR workload. |
| Primary metric | 90th percentile latency | Captures tail latency under drift. |
| Baseline 1 | FCFS | Default scheduler, no prioritization. |
| Baseline 2 | SJF (oracle duration) | Ideal efficiency, no adaptation. |
| Baseline 3 | Static HRRN (γ=1) | Fairness without drift tracking. |
| Baseline 4 | Sliding-window HRRN (window=1000) | Contrast with EMA adaptation. |
| Baseline 5 | Learned predictor (SSJF) | Highlights simplicity advantage. |
| Ablation of ours | AUAS without drift (γ fixed) | Isolates drift adaptation benefit. |
| Synthetic workload 1 | Additive drift: t = α*d + ε (ε ~ N(10,2)) | Tests non-multiplicative failure mode. |
| Synthetic workload 2 | Multiplicative drift: t = α*d*γ(t) | Validates core assumption. |
| Real-world trace | Production ASR trace (anonymized) | Practical significance. |

### Why this setup validates the claim
This experimental design tests the central claim that adaptive drift-aware scheduling (AUAS) improves tail latency under workload drift while preserving fairness. Using LibriSpeech test-clean provides a realistic distribution of audio durations. Comparing against FCFS, SJF (oracle), static HRRN, sliding-window HRRN, and SSJF isolates the effect of drift adaptation: FCFS shows no prioritization, SJF oracle provides an upper bound on efficiency, static HRRN demonstrates fairness without adaptation, sliding-window HRRN tests a contrast adaptation method, and SSJF tests a learned alternative. The ablation (AUAS with fixed γ=1) directly tests whether the EMA drift factor drives improvements. Synthetic workloads test the load-bearing multiplicative assumption: if AUAS fails under additive drift, we identify the failure mode. The real-world trace ensures practical relevance. The 90th percentile latency is chosen because the method specifically targets tail latency, which is most affected by starvation under drift. If AUAS reduces the 90th percentile under multiplicative drift while maintaining median latency, the claim is supported; if AUAS performs poorly under additive drift, we document the limitation.

### Expected outcome and causal chain

**vs. FCFS** — On a case where short requests arrive at low load but drift makes them slow, FCFS processes in arrival order, causing a long request ahead to block many short ones, inflating tail latency. Our method sees that short requests have high urgency (waiting time plus predicted time), so they get priority, reducing their wait. We expect AUAS to show 20-40% lower 90th percentile latency under drift, with median similar or better.

**vs. SJF (oracle)** — On a case where drift suddenly increases processing times, SJF with oracle durations still ranks by true duration, so under heavy load it may starve long requests, worsening fairness. Our method uses dynamic urgency that increases with waiting time, preventing starvation. Thus, we expect AUAS to have slightly higher median latency (≤10%) but significantly lower 90th percentile (15-30%) because it avoids extreme waits.

**vs. Static HRRN (γ=1)** — On a case where drift gradually increases (e.g., 1.2x over time), static HRRN underestimates processing times, giving overly high urgency to long requests and causing short requests to wait longer. Our method adapts γ upward, correcting the urgency and restoring fairness. We expect AUAS to match static HRRN under no drift but outperform by 15-25% on 90th percentile under drift.

**vs. Sliding-window HRRN (window=1000)** — Under slow drift, both EMA and sliding window track drift; however, sliding window requires storing history and reacts abruptly to outliers. We expect AUAS to have lower variance in tail latency and 5-10% better 90th percentile due to smoother adaptation.

**vs. SSJF (learned predictor)** — SSJF uses a neural network to predict processing time, requiring retraining on drift. Under gradual drift, AUAS matches or slightly exceeds SSJF's tail latency (within 5%) while being simpler and not requiring retraining. Under sudden drift, SSJF may initially perform worse due to stale model, giving AUAS a 10-20% advantage.

**Synthetic additive drift:** On synthetic workload where t = α*d + 10 (constant additive), multiplicative assumption fails. AUAS's γ captures average bias but not per-request correction, leading to misranking. We expect AUAS to degrade to static HRRN performance (no improvement), with 90th percentile similar to static HRRN. This documents the failure mode.

### What would falsify this idea
If AUAS shows no improvement in 90th percentile latency compared to static HRRN under multiplicative drift, or if the improvement is uniform across all request durations (rather than concentrated on short requests during drift), then the central claim that drift adaptation is beneficial would be falsified. Additionally, if AUAS performs worse than static HRRN under additive drift (e.g., increases 90th percentile by >10%), the multiplicative assumption is violated, and the method needs redesign (e.g., additive bias term).

## References

1. Duration Aware Scheduling for ASR Serving Under Workload Drift
2. Efficient LLM Scheduling by Learning to Rank
3. Efficient Interactive LLM Serving with Proxy Model-based Sequence Length Prediction
4. Efficient Memory Management for Large Language Model Serving with PagedAttention
5. S3: Increasing GPU Utilization during Generative Inference for Higher Throughput
6. Fast Distributed Inference Serving for Large Language Models
