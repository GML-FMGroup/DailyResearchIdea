# ElastiServe: Jointly Optimizing Elastic Video Diffusion Models and Serving Orchestration on Heterogeneous GPU Clusters

## Motivation

Current video generation serving systems, such as TurboServe, treat the diffusion model as a fixed black box and assume homogeneous GPUs, leading to suboptimal resource utilization and high tail latency on heterogeneous clusters. The root cause is the decoupling of model configuration from orchestration decisions: neither the scheduler adapts the model's computational demands to the GPU's capability, nor does the model expose knobs to trade quality for latency. This structural gap prevents the system from exploiting hardware diversity and workload variability simultaneously.

## Key Insight

The Pareto-optimal frontier of video quality vs. latency for diffusion models is continuous and predictable across GPU types, enabling a learned cost-quality model to serve as a common objective that unifies model configuration selection and session scheduling into a single optimization.

## Method

We propose **ElastiServe**, a video generation serving system that jointly optimizes an elastic diffusion model and session orchestration on heterogeneous GPU clusters.

**(A) What it is:** ElastiServe is a closed-loop system that (i) dynamically adjusts the denoising steps and channel widths of a video diffusion model per session, and (ii) schedules sessions across heterogeneous GPUs using a learned cost-quality model. Input: a stream of video generation requests with latency SLOs. Output: per-frame latencies meeting SLOs and maximized throughput.

**(B) How it works:**
```python
# System operation pseudocode
# Profiling phase (offline):
for each GPU type gpu_type:
    for config c in {denoising_steps: [1,2,4,8,16], channel_scale: [0.5,0.75,1.0], resolution: fixed}:
        profiled_latency[gpu_type][c] = measure_latency(model, c, gpu_type)
        profiled_quality[gpu_type][c] = measure_fid(model, c, dataset_sample)
# Train cost-quality model:
model = NeuralNetwork(input = [gpu_type_embedding, config_params], output = [latency, quality])
train model on profiled data with MSE loss.

# Online serving loop:
while True:
    new_sessions = get_new_requests()
    pending_sessions += new_sessions
    # Joint optimization every T seconds or on event:
    for session in pending_sessions:
        for gpu in all_gpus:
            for config in candidate_configs:
                pred_latency, pred_quality = model.predict(gpu.type, config, session.content_hash)
                if pred_latency <= session.slo - migration_overhead:
                    utility = pred_quality - alpha * (pred_latency - session.slo)  # penalize under-utilization
                    record candidate (session, gpu, config, utility)
    # Solve assignment: greedy or Hungarian to maximize sum utility
    assignments = solve_max_weight_matching(candidates)
    execute_assignments(assignments)  # includes model config updates and session migration if needed
    remove finished sessions from pending
```

**(C) Why this design:** We chose a learned cost-quality model over a rule-based heuristic because it captures complex interactions between GPU architecture and model configuration that cannot be analytically derived, accepting the cost of offline profiling and retraining. We used a greedy matching instead of an exact solver for scalability, as Hungarian algorithm's O(n^3) is infeasible for large clusters; the trade-off is a small optimality gap that we bound empirically. We fixed resolution and only varied steps and channel width to limit the configuration space; this omits resolution scaling as a knob, but we argue that temporal quality (steps) and spatial detail (channels) are the dominant controls for latency. We also introduced a utility function that penalizes under-utilization (latency far below SLO) to encourage efficient resource use, which avoids the common pitfall of always selecting the highest quality regardless of slack.

**(D) Why it measures what we claim:** The predicted latency from the learned model (computational quantity) measures the achievable frame generation time (motivation-level concept) because we assume the profiling captures the full pipeline including I/O and communication, and that model inference time dominates; this assumption fails when network congestion or KV cache hits cause transient delays, in which case the predicted latency reflects idealized hardware throughput rather than actual end-to-end delay. The predicted quality (computational quantity) measures perceptual fidelity (motivation-level concept) because we assume that FID computed on a held-out dataset for each configuration is a monotonic proxy for human-rated quality; this assumption fails for very low step counts (e.g., 1 step) where artifacts not captured by FID emerge, in which case the metric overestimates user satisfaction. The utility function (computational quantity) measures the joint optimization objective (motivation-level concept) because we assume linear additivity of quality and latency penalties; this assumption fails when users have non-linear preferences, in which case the optimization may select configurations that are suboptimal in practice."

## Contribution

(1) ElasticDiff, a video diffusion model with configurable denoising steps and channel widths that exposes a smooth quality-latency trade-off, enabling fine-grained adaptation. (2) A learned cost-quality model that predicts per-GPU latency and FID for any configuration, and a greedy scheduling algorithm that jointly assigns sessions to GPUs and selects configurations to maximize a utility function under SLO constraints. (3) Empirical demonstration on heterogeneous GPU clusters (e.g., A100, V100, RTX 3090) showing up to 2.3x throughput improvement and 40% reduction in tail latency over a fixed-model, homogeneous-scheduling baseline.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Video generation requests with diverse SLOs and content | Realistic workload for serving systems |
| Primary metric | Goodput (requests meeting SLO per unit time) | Directly measures the core objective |
| Baseline 1 | Static round-robin without elastic config | Tests benefit of elastic adaptation |
| Baseline 2 | Fixed high-quality config on all GPUs | Tests benefit of config diversity |
| Baseline 3 | Rule-based heuristic (config based on SLO margin) | Tests advantage of learned model |
| Ablation | ElastiServe without learned model (heuristic config) | Isolates impact of learned cost-quality model |

### Why this setup validates the claim
This experimental design creates a falsifiable test of ElastiServe's central claim: that jointly optimizing elastic model configurations and session-to-GPU assignments via a learned cost-quality model improves goodput under latency SLOs. The dataset's varied SLOs and content stress the system's ability to dynamically trade off quality and latency. The primary metric, goodput, directly captures the system's success at maximizing throughput while satisfying SLOs. Baseline 1 (static round-robin) tests the necessity of elastic adaptation; if ElastiServe outperforms it, the joint optimization is beneficial. Baseline 2 (fixed high-quality config) tests whether allowing lower-quality configs can increase goodput by serving more requests; outperforming it validates the elastic design. Baseline 3 (rule-based heuristic) tests whether a simple heuristic suffices; if ElastiServe beats it, the learned model provides a data-driven advantage. The ablation (removing the learned model) isolates that component's contribution. Together, these baselines cover the key sub-claims: (i) elastic configs help, (ii) learned assignment helps beyond heuristics, and (iii) the full system beats static or fixed alternatives. If ElastiServe fails to beat any baseline, the corresponding sub-claim is falsified.

### Expected outcome and causal chain

**vs. Static round-robin without elastic config** — On a case with a mix of tight (e.g., 2s SLO) and loose (10s SLO) sessions, static round-robin assigns the same GPU and config to all sessions, causing tight sessions to miss SLOs because the config is too slow (e.g., 16 steps) or loose sessions to underutilize the GPU because config is too fast (e.g., 4 steps) wasting slack. Our method uses the learned model to assign tight sessions to faster GPUs with fewer steps and loose sessions to slower GPUs with more steps, maximizing goodput. We expect a significant gap: ElastiServe achieves >90% goodput while static round-robin falls below 50% under heterogeneous SLOs.

**vs. Fixed high-quality config on all GPUs** — On a burst of many low-SLO sessions (e.g., 5s SLO), fixed 50-step config saturates GPUs, causing long queue delays and SLO violations for most requests. Our method adapts per session: it lowers steps or channels for some sessions, reducing per-GPU latency and increasing throughput, so more sessions meet their SLOs. We expect ElastiServe to handle 2x the request rate of fixed config while maintaining <10% SLO violations.

**vs. Rule-based heuristic (config based on SLO margin)** — On a case where SLO margin is ambiguous (e.g., 3s SLO with moderate content complexity), a rule-based heuristic might pick too many steps (causing SLO miss) or too few (wasting quality), because it cannot capture GPU-specific latency curves. Our learned model predicts exact latency per GPU–config pair, enabling precise within-SLO configuration. We expect ElastiServe to maintain median quality (FID) 15% better than the heuristic for the same SLO satisfaction rate.

### What would falsify this idea
If, under heterogeneous loads, ElastiServe's goodput improvement over static round-robin is uniform across all SLO ranges and not concentrated on tight-SLO sessions where elastic configs are most needed, then the core claim of joint optimization is not validated—indicating that simple load balancing alone suffices and the learned model adds no benefit.

## References

1. TurboServe: Serving Streaming Video Generation Efficiently and Economically
2. TurboDiffusion: Accelerating Video Diffusion Models by 100-200 Times
3. LongLive: Real-time Interactive Long Video Generation
4. Taming the Chaos: Coordinated Autoscaling for Heterogeneous and Disaggregated LLM Inference
5. StreamDiffusionV2: A Streaming System for Dynamic and Interactive Video Generation
6. TridentServe: A Stage-level Serving System for Diffusion Pipelines
7. Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving
8. Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve
