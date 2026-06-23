# Online Proxy Calibration for Duration-Aware ASR Scheduling Under Workload Drift

## Motivation

Static audio-duration proxies, as used in 'Duration Aware Scheduling for ASR Serving Under Workload Drift', degrade when workload characteristics shift (e.g., longer utterances, different model versions), causing tail latency violations in HRRN scheduling. Existing approaches lack a lightweight mechanism to correct the proxy without retraining a learned model, forcing a trade-off between adaptation and complexity. This limitation arises because the proxy error is structured by request class and drift patterns, which can be corrected via simple feedback signals rather than full model retraining.

## Key Insight

The tail-latency violation rate for a request class is a monotonic indicator of systematic proxy underestimation in that class, enabling corrective adjustments without observing true processing times.

## Method

### (A) What it is
**CALIBRA** (Calibration via Latency-Informed Bias Rectification for ASR) is an online control loop that maintains per-class multiplicative correction factors $c_k$ for a static duration proxy $d_{i,k}$ of request $i$ in class $k$ (e.g., utterance length bucket). Its input is the observed end-to-end latency of each request and a tail-latency target (e.g., 90th percentile bound). Its output is a corrected proxy $\hat{p}_{i,k} = c_k \cdot d_{i,k}$, used for scheduling via HRRN in the vLLM engine.

### (B) How it works
```python
# Hyperparameters:
#   T = target tail latency (e.g., 90th percentile bound)
#   W = feedback window (e.g., 100 requests per class)
#   alpha = step size (e.g., 0.05)
#   waiting_time_factor = multiplier for waiting time threshold (e.g., 2.0)
#   c_k initialized to 1.0 for each class k

while serving:
    # Phase 1: Collect feedback per class
    for each completed request i in class k:
        waiting_time_i = start_time_i - arrival_time_i
        if latency_i > T and waiting_time_i < waiting_time_factor * d_{i,k}:
            violations[k] += 1
        total[k] += 1

    # Phase 2: Compute violation rate
    for each class k with total[k] >= W:
        v_rate = violations[k] / total[k]
        # Adjust correction factor
        if v_rate > v_threshold (e.g., 0.1):
            c_k *= (1 + alpha)  # increase proxy for class k
        else:
            c_k *= (1 - alpha)  # decrease proxy towards static value
        # Reset counters
        violations[k] = 0
        total[k] = 0

    # Phase 3: Schedule requests with corrected proxy
    # Use HRRN on corrected proxy values
```

### (C) Why this design
We chose a **tail-latency violation feedback** over continuous error measurement because it avoids needing ground-truth processing times and is directly tied to the scheduling objective (avoiding high latencies). The cost is that violations provide a coarse signal, potentially leading to slower adaptation when the proxy error is moderate. We opted for **per-class correction factors** rather than per-request to exploit the structure of workload drift (e.g., all long utterances become slower due to a model update) and to maintain statistical robustness; the trade-off is that within-class variability is ignored, which may under-correct for individual outliers. We use a **multiplicative correction with bounded step size** instead of additive correction because processing times scale with duration (e.g., a 10-second utterance has larger absolute error than a 1-second one), and the step size prevents overshooting. The cost is that multiplicative updates can become unstable if the factor diverges, but we cap the factor to [0.5, 2.0]. These decisions collectively keep the feedback loop lightweight, avoiding model retraining or complex statistics, while aligning with the simplicity goal of the original trend. **Core assumption**: The violation rate for a class is monotonic with systematic proxy underestimation, which holds only when interference from other classes is bounded.

### (D) Why it measures what we claim
The **tail-latency violation rate for a class** measures **systematic proxy underestimation for that class** because we assume that if the proxy is accurate on average, the violation rate should match the target tail quantile (e.g., 10% for P90); a higher violation rate implies the proxy is systematically too low. This assumption fails when violations are caused by random noise or interference from other classes (e.g., a request from a different class waiting for a long time due to SJF preemption), in which case the violation rate reflects a mix of the class-specific error and global scheduling artifacts. To mitigate this, we augment the violation condition with a per-class waiting-time threshold (waiting_time < waiting_time_factor * proxy), so violations due to queuing delay are filtered out. The **correction factor update** operationalizes the adaptation to drift by increasing the proxy for a class when its violation rate exceeds a threshold, assuming that drift is class-consistent (e.g., all long utterances become slower). This assumption fails when drift affects only a subset of the class (e.g., only certain audio lengths), in which case the correction factor may be too coarse or overly aggressive. The **feedback window W** measures the statistical confidence in the violation rate estimate; it assumes that within that window, the workload distribution does not change sharply, failing if the window is too long and drift occurs mid-window, leading to outdated adjustments.

**Equivalence and failure modes**: The violation rate measures proxy underestimation under the assumption that interference from other classes is bounded (Assumption A). When interference is high (e.g., bursty arrivals causing head-of-line blocking), violation rate may be inflated even with accurate proxies, breaking monotonicity (Failure mode F). The waiting-time filter helps, but cannot eliminate the risk entirely. Additionally, if drift is not class-consistent, per-class correction may be too coarse.

## Contribution

(1) An online feedback-based calibration mechanism (CALIBRA) for duration-aware scheduling that adapts per-class correction factors using only tail-latency violation signals, avoiding model retraining. (2) A design principle that leverages the monotonic relationship between proxy underestimation and tail violations to enable lightweight drift adaptation in ASR serving.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | LibriSpeech test-clean with injected workload drift | Matches ASR serving drift scenario |
| Secondary dataset | Video decoding workload with synthetic length drift (e.g., variable video duration) | Tests broader applicability |
| Primary metric | P90 latency | Directly measures tail violation target |
| Baseline | FCFS | Default scheduler, naive |
| Baseline | SJF (static proxy) | Standard length-aware, no correction |
| Baseline | HRRN (static proxy) | Combines length and waiting, static |
| Ablation | CALIBRA with global correction (single c across all classes) | Isolates per-class adaptation benefit |

### Why this setup validates the claim
This setup tests whether CALIBRA's per-class correction can reduce tail latency under workload drift compared to static scheduling policies. The dataset includes drift (e.g., sudden increase in long utterances) to trigger proxy errors. FCFS is a naive baseline that ignores duration, SJF uses static proxies without adaptation, and HRRN also relies on static proxies. The ablation (global correction) isolates the benefit of per-class over global adaptation. P90 latency is the metric because the method explicitly targets tail violations. If CALIBRA shows improvement specifically on drift-affected classes, the claim is supported; if not, it fails. Additionally, we include a simulation study where synthetic workload drift and controlled interference are used to validate the monotonicity assumption of the violation signal.

### Expected outcome and causal chain

**vs. FCFS** — On a case where a burst of long utterances arrives, FCFS will process them first-come-first-served, causing short requests to wait behind long ones, inflating tail latency for short requests. Our method instead boosts the proxy for long utterances (via correction factor) so they are deprioritized when violation rate rises, thus protecting short requests. We expect a noticeable improvement in P90 latency under drift for classes that are systematically underestimated.

**vs. SJF (static proxy)** — On a case where a model update makes all long utterances slower than predicted, SJF uses the static proxy and schedules them as short, causing them to run long and block others, leading to high tail latency. Our method detects increased violations for the long class and increases its correction factor, making those requests appear longer and thus scheduled later, reducing interference. We expect SJF to have high tail latency on long utterances while CALIBRA maintains control.

**vs. HRRN (static proxy)** — On a case where a specific class's duration shifts gradually, HRRN's static proxy leads to suboptimal prioritization for that class, causing occasional tail violations. Our method continuously adjusts the proxy per class, so the scheduling remains balanced. We expect HRRN to show increasing tail latency as drift progresses, while CALIBRA's correction factor adapts and keeps tail latency stable.

### What would falsify this idea
If CALIBRA's tail latency improvement over HRRN with static proxy is uniform across all classes and drift conditions, rather than concentrated on classes that experience systematic drift, then the per-class adaptation is not the cause and the claim is false.

## References

1. Duration Aware Scheduling for ASR Serving Under Workload Drift
2. Efficient LLM Scheduling by Learning to Rank
3. Efficient Interactive LLM Serving with Proxy Model-based Sequence Length Prediction
4. Efficient Memory Management for Large Language Model Serving with PagedAttention
5. S3: Increasing GPU Utilization during Generative Inference for Higher Throughput
6. Fast Distributed Inference Serving for Large Language Models
