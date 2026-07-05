# AdaExpert: Online Adaptive Expert Placement via Drift-Driven Cache Miss Monitoring for MoE Serving

## Motivation

Static offline expert partitioning, as in ELDR, assumes workload patterns are stationary, but real-world traffic exhibits topic shifts that alter expert activation distributions, causing gradual locality degradation and increased latency. The root cause is the lack of a runtime signal that directly reflects locality health and a mechanism to adjust placement online without manual retraining. Existing methods cannot adapt to dynamic shifts, forcing operators to periodically retrain the partition, which is costly and reactive.

## Key Insight

Per-expert cache miss rates provide a direct, low-cost runtime measure of locality degradation, and their deviation from stationarity under stable workload enables a principled drift detection trigger for expert placement adaptation.

## Method

**AdaExpert**

(A) **What it is:** AdaExpert is an online adaptation module that monitors per-expert cache miss rates as a locality health signal, detects workload shifts using a CUSUM drift test on cluster-aggregated miss rates, and triggers cluster splitting/merging to restore locality.

(B) **How it works (pseudocode):**
```pseudocode
Input: Workers W (size N), expert-to-worker map M, per-expert cache miss rate stream R_e over sliding window L, drift threshold τ, min cluster size min_cluster.
Hyperparameters: L=1000 requests, τ=2.0, min_cluster=2, K=10 batches, δ=0.001 (expected drift per batch).

Initialization:
  For each worker w in W:
    cluster C_w = {e | M[e]=w}
  For each cluster, compute average miss rate CR_w = mean_{e in C_w} R_e over window L.
  Initialize CUSUM statistic S_w = 0.

Online loop (per batch of requests):
  1. Update R_e(t) for each expert e with new cache hits/misses.
  2. Every K batches:
      For each cluster C_w:
        Update CR_w(t) = mean_{e in C_w} R_e(t).
        Update S_w(t) = max(0, S_w(t-1) + CR_w(t) - CR_w(t-1) - δ).
        If S_w(t) > τ:
          // Split cluster C_w
          Features: For each expert e in C_w, compute activation frequency vector f_e (counts per request over window).
          Run balanced K-means (2 clusters) on {f_e}.
          Assign one subcluster to worker w, another to an idle worker (or reassign to least-loaded worker).
          Update M and reset S_w(t) = 0.
      For each pair of workers (w1, w2):
        Compute Jaccard similarity of their expert activation frequency sets.
        If similarity > 0.8 and CR_w1 < 0.1 and CR_w2 < 0.1:
          // Merge clusters
          Move all experts from w2 to w1.
          Remove w2 from active set, update M.
  3. Return updated M.
```

(C) **Why this design:** We chose CUSUM over simple thresholding because CUSUM accumulates small persistent shifts that a threshold would miss, reducing false negatives, while introducing sensitivity to noise that we mitigate by updating only every K batches. We chose per-expert activation frequency vectors as features for splitting rather than raw cache miss rates because activation frequency captures co-activation patterns directly, whereas miss rates are influenced by total load and cache capacity; this introduces a risk that split clusters may not align with actual locality improvement, but we cross-check post-split by monitoring miss rates. We chose balanced K-means with a fixed split size (2) rather than dynamic split count to avoid over-fragmentation; the cost is occasional suboptimal splits that require later merging. We chose a merge trigger based on expert set similarity and low miss rates rather than a single threshold, to avoid merging clusters that are similar but still benefit from separation. The trade-off: increased computational overhead from periodic similarity checks (O(N^2) per merge check) versus reduced manual retuning. Unlike ELDR, which performs a single offline K-means and never adapts, AdaExpert re-clusters online based on a drift signal, making it robust to workload shifts.

(D) **Why it measures what we claim:** The per-expert cache miss rate R_e(t) measures locality degradation because it directly quantifies the frequency of loading expert weights from remote memory; the assumption is that miss rate is monotonic with expert co-activation distance under the current placement. This assumption fails when the workload contains many cold experts that are rarely accessed, in which case miss rates reflect sparsity rather than poor placement; we filter by only considering experts with non-zero activation in the window. The CUSUM statistic S_w(t) measures workload shift because it tracks deviations from a zero-drift null hypothesis; the assumption is that miss rate increments are i.i.d. under stationary workload. This assumption fails when miss rates have seasonality (e.g., periodic bursts), causing false alarms; we mitigate by resetting S_w after each split and using a window that averages out short-term fluctuations. The cluster similarity metric based on activation frequency vectors measures suitability for merge because it captures functional overlap; the assumption is that experts with similar activation patterns produce similar locality benefit when co-located. This assumption fails when two similar expert sets have high cache contention, causing increased misses post-merge; we guard by only merging when both clusters have low current miss rates.

## Contribution

(1) AdaExpert, the first online adaptive expert placement framework for MoE serving that uses cache miss rate drift to trigger repartitioning, replacing static offline partitioning. (2) A demonstration that per-expert cache miss rates serve as a valid online signal for workload shift detection in MoE decode, enabling proactive adaptation. (3) A drift detection and clustering adjustment protocol that maintains locality without offline retraining or manual intervention.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|----------------------|
| Dataset | Open-source MoE workload traces | Realistic and reproducible evaluation. |
| Primary metric | Cache miss rate (per-expert) | Directly measures locality health. |
| Baseline 1 | Round-robin expert placement | Standard baseline with no locality awareness. |
| Baseline 2 | ELDR (offline K-means) | Static clustering baseline from prior work. |
| Baseline 3 | Opportunistic Expert Activation | Batch-aware routing but no locality adaptation. |
| Ablation | AdaExpert w/o CUSUM (threshold only) | Isolates drift detection effect. |

### Why this setup validates the claim

The experimental design directly tests the central claim that online drift detection with CUSUM improves locality adaptation under workload shifts. The dataset includes both static and shifting access patterns, with periods of gradual drift and abrupt changes. Round-robin and ELDR represent static placement strategies that lack adaptation, while Opportunistic Expert Activation reduces activation counts but does not reorganize expert placement. The ablation (threshold-only) isolates the benefit of CUSUM accumulation. Cache miss rate is the metric AdaExpert directly optimizes, making it a sensitive and appropriate measure. If AdaExpert consistently achieves lower miss rates than baselines during shift periods, and outperforms its ablation, the claim is validated. Conversely, if gains appear uniformly across all time, the adaptation may be unnecessary.

### Expected outcome and causal chain

**vs. Round-robin expert placement** — On a case where workload shifts to favor a new set of co-activated experts, round-robin distributes experts evenly across workers, so each worker must often fetch remote expert weights, leading to high cache miss rates. Our method detects the resulting miss rate increase via CUSUM, triggers a split or merge to co-locate frequently co-activated experts, reducing remote fetches. We expect a noticeable gap in miss rate for AdaExpert, especially during and after shifts, while round-robin remains high.

**vs. ELDR (offline K-means)** — On a case where the workload gradually changes from one pattern to another, ELDR's static clustering becomes outdated, causing experts that are now frequently co-activated to be placed on different workers, raising miss rates. AdaExpert continuously monitors cluster miss rates and detects the drift, then re-clusters using activation frequency vectors to restore locality. We expect AdaExpert's miss rate to remain low and stable, while ELDR's miss rate increases after the shift, creating a growing gap over time.

**vs. Opportunistic Expert Activation** — On a case where the batch size is small but the expert set per batch varies, Opportunistic reduces the number of activated experts but does not change placement, so the remaining active experts may still be placed far apart, leading to cache misses. AdaExpert adapts the placement of all experts to align with current co-activation patterns, directly reducing remote weight access. We expect AdaExpert to achieve lower miss rates than Opportunistic, especially on batches where frequent remote accesses would otherwise occur.

### What would falsify this idea

If AdaExpert's cache miss rate does not substantially drop below that of ELDR during workload shifts, or if the threshold-only ablation performs equally well, then the claim that CUSUM-based drift detection improves adaptation is falsified. Additionally, if gains are uniform across all time segments rather than concentrated in shift periods, the adaptation mechanism may not be targeting the true failure mode.

## References

1. ELDR: Expert-Locality-Aware Decode Routing for PD-Disaggregated MoE Serving
2. Semantic Parallelism: Redefining Efficient MoE Inference via Model-Data Co-Scheduling
3. Preble: Efficient Distributed Prompt Scheduling for LLM Serving
4. Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving
5. Opportunistic Expert Activation: Batch-Aware Expert Routing for Faster Decode Without Retraining
6. MoETuner: Optimized Mixture of Expert Serving with Balanced Expert Placement and Token Routing
7. Lynx: Enabling Efficient MoE Inference through Dynamic Batch-Aware Expert Selection
8. Efficient MoE Serving in the Memory-Bound Regime: Balance Activated Experts, Not Tokens
