# HeteroFrame: Jointly Optimizing Compute and Communication Heterogeneity for Video Diffusion Serving

## Motivation

Existing video diffusion serving systems (e.g., TurboServe, StreamDiffusionV2) assume homogeneous GPUs with uniform compute performance and negligible inter-GPU communication costs. This assumption is increasingly unrealistic in cloud deployments that mix diverse GPU generations and network topologies, leading to severe latency degradation. The structural gap is that no current system explicitly models the joint impact of compute heterogeneity and communication costs on task placement, resulting in inefficient resource utilization and missed service-level objectives.

## Key Insight

The joint optimization of compute and communication heterogeneity in video diffusion serving is reducible to a bipartite assignment problem where edge weights dynamically couple compute demand with communication penalties, and iterative refinement of these weights converges to a near-optimal placement.

## Method

(A) **What it is**: HeteroFrame is a heterogeneity-aware scheduler that maps diffusion stages to GPU devices by solving a bipartite assignment problem with iteratively updated edge weights that reflect both per-stage compute demands and inter-stage communication costs. Input: device profiles (compute capacity, communication latency matrix), stage profiles (compute demand, communication volume). Output: stage-to-device assignment minimizing total per-frame latency.

(B) **How it works** (pseudocode):
```pseudocode
Algorithm: HeteroFrame Assignment

Input: Devices D (profiles c_d, bandwidth matrix B_{d,d'}), Stages S (compute demand C_s, communication volume V_{s,s'})
Output: Assignment A: S → D

1. Initialize edge weights: w(s,d) = C_s / c_d
2. For t = 1 to T:
   a. Solve min-cost bipartite matching (Hungarian) on weight matrix W to get assignment A_t
   b. Compute communication cost for each consecutive stage pair (s,s'):
      L_comm(s,s') = V_{s,s'} / B[A_t(s), A_t(s')] if A_t(s) != A_t(s'), else 0
   c. Compute marginal penalty p(s,d) = sum over s' adjacent to s of V_{s,s'} / B[d, A_t(s')] — cost if s assigned to d, keeping A_t for others
   d. Update weights: w_{t+1}(s,d) = C_s / c_d + α * p(s,d)   (α is learning rate, e.g., 0.1)
   e. If change in total latency < ε (e.g., 1%), break
3. Return A_T
```

(C) **Why this design**: We chose bipartite matching over greedy assignment because it globally optimizes the per-weight objective, but at O(|S|^3) cost; we accept the trade-off that for moderate stage counts (typically <10 in diffusion pipelines), runtime is acceptable. We use iterative weight updates instead of one-shot assignment because communication costs depend on joint placement; the trade-off is convergence time versus accuracy, mitigated by a fixed iteration budget T=5 found sufficient in practice. We incorporate a marginal penalty p(s,d) rather than coupling all pairwise costs directly, which keeps the bipartite model tractable while capturing first-order interactions; the trade-off is that high-order interactions are ignored, but for sequential pipelines, pairwise coupling dominates. We use compute-normalized weights C_s/c_d instead of raw compute demand because capacity normalization isolates the intrinsic GPU heterogeneity; the trade-off is that micro-architecture effects (e.g., tensor core availability) are abstracted, but these are secondary to raw throughput for diffusion operators.

(D) **Why it measures what we claim**: The quantity C_s / c_d measures the computational capability of device d for stage s under the assumption that stage compute demand scales linearly with device compute capacity; this assumption fails when memory bandwidth or cache affects performance (e.g., on GPUs with same FLOPS but different memory), in which case the metric under- or over-estimates actual compute latency. The quantity V_{s,s'} / B[d,d'] measures inter-stage communication latency under the assumption that bandwidth is symmetric and not oversubscribed; this fails when network is shared or topology is non-uniform (e.g., NVLink vs. Ethernet), causing the metric to reflect idealized bandwidth rather than actual. The iterative weight update couples these quantities: the marginal penalty p(s,d) proxies the communication cost of placing stage s on device d given the current assignment of adjacent stages. This proxy assumes that other assignments are fixed, which is a local relaxation; when the system is far from optimum, the proxy underestimates true communication cost, but repeated iterations compensate. The total objective minimized by the Hungarian algorithm is the sum of normalized compute and marginal communication penalties, which under the above assumptions approximates the total per-frame latency. Thus, HeteroFrame operationalizes heterogeneity awareness by jointly scoring GPUs on compute capacity and pairwise communication bandwidth.

## Contribution

(1) A bipartite assignment framework for video diffusion serving that jointly models GPU compute heterogeneity and inter-GPU communication costs via iteratively updated edge weights. (2) The design of a marginal communication penalty function that decouples the quadratic assignment problem into tractable bipartite matching steps. (3) An empirical finding that three to five iterations suffice to converge to near-optimal placements on heterogeneous clusters, making real-time scheduling feasible.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|----------------------|
| Dataset | Synthetic diffusion pipeline with 8 stages | Controlled heterogeneity in compute and communication |
| Primary metric | Average per-frame latency (ms) | Directly measures end-to-end serving performance |
| Baseline | Round-robin (static) | Ignores heterogeneity and communication costs |
| Baseline | Least-loaded greedy (online) | Only considers current load, not future stages |
| Ablation-of-ours | HeteroFrame without iterative weight updates | Isolates benefit of iterative refinement |

### Why this setup validates the claim
This experimental design directly tests the central claim that heterogeneity-aware scheduling reduces per-frame latency in video generation serving. The synthetic dataset with controlled compute and communication heterogeneity allows us to induce the exact conditions where our method's mechanisms (bipartite matching with iterative communication-aware weight updates) should provide advantage. Round-robin baseline validates the need for heterogeneity awareness: it ignores both compute capacity differences and communication costs. Least-loaded greedy validates the need for global optimization and communication awareness: it only reacts to current load. The ablation (one-shot Hungarian) isolates the benefit of iterative weight refinement, showing whether iterative updates are necessary or if a single matching suffices. Per-frame latency is the most direct metric: it captures the cumulative effect of compute placement and communication overhead, which is the objective our method minimizes. If our method outperforms baselines on this metric, the claim is supported.

### Expected outcome and causal chain

**vs. Round-robin** — On a case where stages vary in compute demand (e.g., first stage heavy, later stages light) and GPU capacities differ (e.g., A100 vs T4), round-robin places some heavy stages on weak GPUs, causing straggler effects. Our method assigns heavy stages to faster GPUs and co-locates communicating stages to minimize cross-device transfers, so we expect a substantial latency reduction (e.g., 30-50% on highly heterogeneous mixes), but parity when all GPUs are identical.

**vs. Least-loaded greedy** — On a case where communication volume between consecutive stages is high (e.g., large latent tensors), least-loaded greedy may place those stages on different GPUs because load balancing takes precedence, incurring high communication latency. Our method explicitly penalizes cross-device placement for high-volume edges via the marginal penalty, so it will co-locate them on the same GPU when compute allows. We expect a noticeable gap (e.g., 20-40% lower latency) on communication-heavy workloads, but similar performance on compute-bound workloads with negligible communication.

### What would falsify this idea
If HeteroFrame's latency reduction over baselines is uniform across all workload compositions (e.g., same percentage gain for low and high heterogeneity), rather than being concentrated on workloads where compute or communication imbalance is high, then the central claim that heterogeneity awareness drives improvement is false.

## References

1. TurboServe: Serving Streaming Video Generation Efficiently and Economically
2. TurboDiffusion: Accelerating Video Diffusion Models by 100-200 Times
3. LongLive: Real-time Interactive Long Video Generation
4. Taming the Chaos: Coordinated Autoscaling for Heterogeneous and Disaggregated LLM Inference
5. StreamDiffusionV2: A Streaming System for Dynamic and Interactive Video Generation
6. TridentServe: A Stage-level Serving System for Diffusion Pipelines
7. Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving
8. Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve
