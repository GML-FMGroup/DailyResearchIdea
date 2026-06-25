# Gradient-Guided Adaptive Data Scheduling for Efficient LLM Pre-training

## Motivation

Existing RL-based data schedulers like HDS require pre-defined domain labels, limiting their ability to adapt to emerging or merging data clusters during training. This reliance on static labels is a structural limitation inherited from prior methods such as ODM and DoReMi, which treat data groups as fixed inputs. Without dynamic discovery, the scheduler cannot adjust mixing proportions when the data distribution shifts, leading to suboptimal convergence.

## Key Insight

Value function gradients from the RL policy naturally encode the differential impact of data characteristics on scheduling performance, enabling unsupervised clustering that is directly aligned with the RL objective.

## Method

### (A) What it is
GADS (Gradient-Guided Adaptive Data Scheduling) is an RL-based data scheduler that jointly learns a clustering of data points and mixing ratios. It takes a stream of data points, outputs mixing weights for each identified cluster, and updates the clustering online using value gradients.

### (B) How it works
```pseudocode
Initialize actor π_θ, critic V_ψ, clustering net f_ϕ, replay buffer D
Set number of clusters K, entropy coefficient α, gradient norm coefficient β
For each training step:
  1. Sample batch {(x_i, l_i)} from D (l_i is loss or reward signal)
  2. Compute soft assignments a_i = softmax(f_ϕ(x_i))   # ∈ R^K
  3. Compute cluster centroids c_k = (∑_i a_i_k * x_i) / (∑_i a_i_k + ε)
  4. Policy input s = [c_1, ..., c_K, importance_metrics]
  5. Sample mixing ratios m ~ π_θ(·|s)   # Dirichlet or Gumbel-Softmax
  6. Draw training batch according to m, compute reward r (multi-objective as in HDS)
  7. Update critic: minimize (r + γ V_ψ(s') - V_ψ(s))^2
  8. Update actor: maximize Q_ψ(s, m) via SAC objective
  9. Update clustering net: maximize β * ||∇_{a_i} V_ψ(s)||^2 + α * H(a_i)
     where H(a_i) is entropy of assignments
```

### (C) Why this design
We made three design decisions. First, clustering is driven by maximizing the gradient norm of the critic with respect to soft assignments, rather than using standard unsupervised clustering (e.g., k-means). This ties cluster discovery directly to scheduling utility, preventing spurious clusters that do not affect value; the trade-off is increased computational cost from backpropagating through the critic. Second, we use soft assignments with entropy regularization instead of hard assignments to enable smooth, differentiable boundary adjustments; this blurs cluster definitions but avoids discrete jumps that could destabilize RL training. Third, the policy is conditioned on cluster centroids and summary statistics rather than the full set of assignments; this compresses the state space for scalability, but loses per-point detail—we mitigate by updating centroids at each step using the current batch. The two-stage update (clustering then scheduling) ensures that clustering adapts to the critic's sensitivity, while the actor uses the resulting structure to improve mixing.

### (D) Why it measures what we claim
The gradient norm of the value function with respect to cluster assignments measures the sensitivity of expected return to grouping decisions, which in turn measures the causal importance of a cluster for scheduling optimality. This rests on the assumption that the critic is a consistent estimator of true expected return; when the critic is poorly approximated (e.g., early in training or under high variance rewards), gradients may reflect approximation error rather than true value, yielding noisy clusters. The cluster centroid measures the central tendency of data assigned to that cluster, relying on the assumption that data within a cluster share reward-relevant characteristics; this fails when clusters are highly heterogeneous (e.g., containing both simple and complex examples), causing centroids to misrepresent the group's effect on rewards. The entropy regularizer ensures the assignments remain informative, but introduces a bias towards uniform assignments when α is too high.

## Contribution

(1) A value-gradient-based clustering mechanism that eliminates the need for pre-defined domain labels in RL data schedulers, enabling online discovery and adaptation of data groups. (2) The finding that RL value function gradients provide an unsupervised signal for grouping data that is directly aligned with scheduling performance, as opposed to prior work using fixed labels or heuristics. (3) An integrated algorithm that jointly optimizes clustering and RL scheduling within a multi-objective framework.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|------------------------|
| Dataset | RedPajama (6 domains) | Standard multi-domain pre-training corpus |
| Primary metric | Validation perplexity | Direct measure of language model quality |
| Baseline | Uniform mixing | Simple default with no adaptation |
| Baseline | DoReMi | Prior ODM using proxy-model weights |
| Baseline | HDS | Prior RL-based data scheduler |
| Ablation | GADS w/o gradient clustering | Isolates effect of gradient-guided updates |

### Why this setup validates the claim

This setup directly tests whether gradient-guided clustering improves data scheduling over static or proxy-model methods. Using RedPajama with distinct domains, clusters should capture content-based utility for scheduling. Perplexity measures overall language modeling performance. Uniform mixing tests if adaptation helps at all. DoReMi tests if our RL approach outperforms proxy-model weighting. HDS tests if adding online clustering to RL yields further gains. The ablation isolates the gradient-guidance component, verifying whether critic sensitivity ties clustering to scheduling utility. Together, these comparisons form a falsifiable test: if our method outperforms HDS especially on heterogeneous domains, the central claim is supported; otherwise, gradient guidance adds no value.

### Expected outcome and causal chain

**vs. Uniform mixing** — On a case where domain difficulty varies, uniform mixing over-feeds easy domains, wasting capacity because it ignores model's learning needs. Our method instead clusters domains by gradient-sensitivity and adjusts weights accordingly, so we expect a noticeable perplexity gap (e.g., 2-5% improvement) overall and especially on hard domains (e.g., math vs. web).

**vs. DoReMi** — On a case where proxy model's domain weights diverge from target model's actual needs, DoReMi's fixed assignment misallocates because it doesn't adapt to target model's evolving state. Our method uses online value-gradients to reassign clusters, so we expect better final perplexity (e.g., 1-3% gain) and faster convergence, particularly on domains where proxy and target disagree (e.g., code vs. natural language).

**vs. HDS** — On a case where manual domain definitions are coarse (e.g., a single 'web' domain containing diverse quality), HDS's fixed domains miss intra-domain heterogeneity because it treats all web data identically. Our method discovers sub-clusters within domains via gradient-guided clustering, so we expect performance gains on heterogeneous domains (e.g., 3-5% perplexity reduction) while performing similarly on homogeneous ones (e.g., legal text).

### What would falsify this idea

If our method's improvement over HDS is uniform across all domains (i.e., no concentration in heterogeneous subsets), then the gradient-guided clustering is not exploiting intra-domain variation, invalidating the core claim. Alternatively, if overall perplexity is worse than HDS, the central assertion fails.

## References

1. Holistic Data Scheduler for LLM Pre-training via Multi-Objective Reinforcement Learning
2. Efficient Online Data Mixing For Language Model Pre-Training
3. Aioli: A Unified Optimization Framework for Language Model Data Mixing
4. DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining
5. Automatic Document Selection for Efficient Encoder Pretraining
