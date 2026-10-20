# Conformal Tool-Error Calibration for Decision-Theoretic Tool Selection in Multimodal Agents

## Motivation

OpenVLHarness renders heterogeneous tool outputs as descriptive evidence but provides no calibrated, decision-theoretic account of how wrong each backend's spatial output is, so the tool-selection policy cannot compare a depth estimator against an object detector on a common scale. This is the meta-level failure recurring across the spatial-evaluation, spatial-tool-RL, and hierarchical-ReAct branches: evidence rendering is descriptive, not decision-theoretic, so every downstream policy optimizes over uncalibrated, backend-specific error signals. The root cause is structural — each backend reports outputs on its own scale with no shared uncertainty currency, and no branch produced a per-tool calibrated error model.

## Key Insight

Conformal prediction over logged invocations yields backend-specific spatial-error intervals with a finite-sample marginal coverage guarantee that is exchangeability-only and thus directly comparable across heterogeneous tools, giving the selection policy a decision-theoretic currency that retrieval-style tool routing (Gorilla/Toolformer) cannot produce because it never estimates the error of a tool it does not call.

## Method

## CSEE: Calibrated Spatial Error Estimator

**(A) What it is.** CSEE is a layer inside OpenVLHarness that, for each tool backend b, maintains a conditional conformal predictor mapping the current multimodal memory state m (query image embedding + task-type token + backend id) to a prediction interval on spatial error, and exposes `query(b, m, τ) -> (q̂_b(τ), L_b)` where q̂_b(τ) is the τ-quantile of spatial error and L_b is the expected task loss of selecting b. The tool-selection policy picks argmin_b L_b.

**Load-bearing assumption (stated explicitly).** CSEE assumes that the logged invocations used for per-backend conformal calibration are exchangeable with decision-time test invocations, so the empirical nonconformity quantile q_b(alpha) yields valid marginal coverage for the backend errors the selector actually compares at test time. This assumption is marginal (not conditional) and can fail under scene-type shift; we therefore add a shift-robust calibration gate (see B, step 3) and report per-subgroup coverage diagnostics (see experiment) to verify rather than assume cross-backend comparability.

**(B) How it works.**

```pseudocode
# Offline: per-backend conformal calibration
for each backend b in {detector, depth, segmentation, VLM-grounding}:
    D_b = logged invocations {(m_i, e_i)}   # e_i = spatial error vs. human-verified GT
    fit nonconformity score s_b(m, e) = |e - mu_b(m)| / sigma_b(m)
        # mu_b, sigma_b: small MLP (2 hidden layers, 64 units, ReLU) on m
        # trained with MSE on log-error, 200 epochs, Adam lr=1e-3, batch size 64
    for alpha in {0.05, 0.10, 0.20}:
        q_b(alpha) = ceil((n+1)(1-alpha))/n empirical quantile of {s_b(m_i,e_i)}
    # Shift-robust gate (repair for exchangeability failure):
    #   maintain per-scene-type calibration pools (scene-type token from m)
    #   compute OOD score o(m) = density-ratio estimate via logistic regression
    #       on {logged m_i} vs. {test m_j} embeddings; threshold o(m) > o_thresh
    #   if o(m) > o_thresh: fall back to conservative quantile q_b(alpha/2) (widened)
    #   else: use per-scene-type pool if n_pool >= 50, else global q_b(alpha)
    store (mu_b, sigma_b, q_b, per-scene-type pools, OOD threshold) per backend

# Online: decision-time query
def query(b, m, tau):
    alpha_tau = tau  # tau ∈ {0.05, 0.10, 0.20}
    if o(m) > o_thresh:
        q_hat = mu_b(m) + q_b(alpha_tau/2) * sigma_b(m)   # conservative fallback
    else:
        q_hat = mu_b(m) + q_b(alpha_tau) * sigma_b(m)     # interval upper edge
    L_b  = task_loss_weight * q_hat + latency_cost_b * lambda
    return (q_hat, L_b)

def select_tool(m, tau):
    return argmin_b query(b, m, tau)[1]
```

Hyperparameters: alpha_tau ∈ {0.05, 0.10, 0.20} selected per task-type by held-out coverage; lambda = 0.1 (latency weight); MLP trained with MSE on log-error, 200 epochs, Adam lr=1e-3, batch size 64; OOD threshold o_thresh set to 95th percentile of logged OOD scores; per-scene-type pool minimum n_pool = 50.

**(C) Why this design.** We chose split-conformal calibration over a learned probabilistic error head (e.g., a Gaussian NLL regressor) because conformal gives a distribution-free marginal coverage guarantee under exchangeability, accepting the cost that coverage is marginal rather than fully conditional on m and can degrade under distribution shift. We chose a per-backend nonconformity score normalized by a learned sigma_b(m) over a raw residual quantile because normalization makes intervals comparable across backends with different error scales (a depth tool's meters vs. a detector's IoU), accepting the cost that a poorly fit sigma_b inflates interval width. We chose a decision-time argmin over expected loss rather than a fixed tool priority or a learned router because the loss is a closed-form function of the calibrated quantile and a latency term, giving a transparent decision rule, accepting the cost that the loss weighting must be specified per task. A domain expert would not describe this as Gorilla-style retrieval-based tool selection: retrieval ranks tools by embedding similarity to the query and never estimates the error of an uncalled tool, whereas CSEE computes a counterfactual error interval for every backend from logged invocations and selects by expected loss.

**(D) Why it measures what we claim.** The conformal quantile q̂_b(τ) measures backend b's calibrated spatial error at confidence τ because the exchangeability of logged and test invocations makes the empirical nonconformity quantile a valid marginal coverage threshold; this assumption fails when the test distribution shifts (e.g., a new scene type unseen in logs), in which case q̂_b reflects the logged distribution's error rather than the true test error. The normalized score s_b(m,e) measures cross-backend comparability because dividing by sigma_b(m) maps each backend's error onto a common standardized scale; this assumption fails when sigma_b is misspecified for a subpopulation, in which case the interval width reflects model fit rather than genuine uncertainty. The expected loss L_b measures the decision-theoretic cost of selecting backend b because the task loss is assumed monotone and risk-neutral in spatial error at the chosen confidence; this assumption fails when task loss is non-monotone (e.g., a small error in a critical region), in which case L_b ranks by average error rather than task-critical error.

## Contribution

(1) CSEE, a per-backend conditional conformal spatial-error estimator that produces cross-tool-comparable error intervals from logged OpenVLHarness invocations. (2) A decision-theoretic tool-selection rule that selects backends by expected calibrated loss, converting descriptive evidence rendering into a calibrated currency for tool choice. (3) An analysis of when marginal conformal coverage transfers across heterogeneous backends and where distribution shift breaks comparability.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | OpenVLHarness 22-benchmark suite with logged per-backend invocations, restricted to 6 benchmarks with existing spatial GT (e.g., VQA-spatial, depth-estimation, object-detection, segmentation, grounding, and a held-out tool-use suite) | Provides counterfactual error labels across heterogeneous backends; scoped to tractable subset with pre-registered annotation budget of 500 human-verified spatial GT per benchmark |
| Primary metric | Task success rate under a fixed latency budget | Captures the accuracy-cost tradeoff the selector is built to optimize |
| Secondary metrics | Empirical vs. nominal coverage per backend and per scene type; rank correlation (Spearman) between q̂_b and realized task loss per backend | Tests coverage validity and the monotone-loss assumption directly |
| Baseline | Base MLLM with enlarged inference budget | Tests whether calibration beats raw model scaling |
| Baseline | Gorilla-style embedding-retrieval router | Tests whether counterfactual error estimation beats similarity ranking |
| Baseline | Gaussian NLL learned error head | Tests whether conformal calibration beats a parametric uncertainty head |
| Ablation-of-ours | CSEE with raw residual quantile (no sigma normalization) | Tests whether cross-backend normalization is load-bearing |
| Ablation-of-ours | CSEE without shift-robust gate (global quantile only) | Tests whether the exchangeability repair is load-bearing under scene-type shift |

### Why this setup validates the claim

The central claim is that a per-backend conformal error estimator, queried counterfactually for every tool at decision time, yields better tool selection than either scaling the base model, ranking tools by embedding similarity, or a parametric uncertainty head. The base-MLLM baseline tests the sub-claim that the gain comes from calibrated tool routing rather than raw inference budget; if CSEE does not beat it, the harness premise collapses. The retrieval router tests the sub-claim that estimating an uncalled tool's error matters more than matching the query to a tool description. The NLL head tests the sub-claim that distribution-free calibration is what buys the gain, not merely having any uncertainty estimate. The raw-residual ablation isolates normalization as the mechanism that makes intervals comparable across backends with different error scales. The no-gate ablation isolates the shift-robust calibration as the mechanism that maintains coverage under scene-type shift. Task success under a fixed latency budget is the right metric because the selector's loss function is exactly a spatial-error term plus a latency term; a metric that ignored latency would fail to detect whether the argmin rule is doing the work it claims to do. Coverage diagnostics (empirical vs. nominal per backend and per scene type) verify the load-bearing exchangeability assumption rather than assume it, and the rank-correlation diagnostic tests the monotone-loss assumption directly.

### Expected outcome and causal chain

**vs. Base MLLM with enlarged inference budget** — On a case where the query requires a precise spatial readout (e.g., a small object in a cluttered scene), the base model produces a plausible but spatially imprecise answer because more inference budget does not add a calibrated error signal for any backend. Our method instead routes to the backend whose conformal interval upper edge is lowest at the chosen confidence, because it has a counterfactual error estimate for every backend. We expect a noticeable success-rate gap concentrated on spatially demanding benchmarks, with near-parity on coarse semantic ones.

**vs. Gorilla-style embedding-retrieval router** — On a case where the query embedding is closest to a detector's description but depth is the backend with genuinely lower spatial error for that scene, the retrieval router picks the detector because similarity ranking never estimates the error of an uncalled tool. Our method instead computes a calibrated error interval for depth from logged invocations and selects it, because its decision rule is argmin over expected loss rather than embedding distance. We expect a gap that grows with the mismatch between tool-description similarity and true backend error, and parity when similarity happens to align.

**vs. Gaussian NLL learned error head** — On a case where the test scene type is underrepresented in logs, the NLL head produces overconfident narrow intervals because its parametric Gaussian assumption is fit by likelihood rather than calibrated to empirical residuals. Our method instead widens the interval via the empirical nonconformity quantile and the shift-robust gate, because split-conformal calibration sets the threshold from held-out residuals and the gate falls back to a conservative quantile when the test embedding is OOD. We expect the NLL baseline to over-select a single backend and our method to show a smaller gap on in-distribution subsets but a clearer advantage on shifted subsets.

**Coverage and rank-correlation diagnostics** — We expect empirical coverage to be close to nominal (within ±0.03) per backend and per scene type when the shift-robust gate is active, and to degrade (e.g., coverage below nominal by >0.05) without the gate on shifted subsets. We expect Spearman rank correlation between q̂_b and realized task loss to be positive (e.g., >0.3) per backend, supporting the monotone-loss assumption; a near-zero correlation would indicate the assumption fails and L_b ranks by average error rather than task-critical error.

### What would falsify this idea

If CSEE's gain over the retrieval router and NLL head is uniform across all benchmark subsets rather than concentrated on the spatially demanding and distribution-shifted subsets where the predicted failure modes occur, then the calibration mechanism is not what produces the effect and the central claim is wrong. Likewise, if the raw-residual ablation matches normalized CSEE everywhere, cross-backend normalization is not load-bearing and the design rationale collapses; if the no-gate ablation matches gated CSEE on shifted subsets, the shift-robust repair is not load-bearing and the exchangeability assumption is not the operative constraint.

## References

1. OpenVLHarness: Real World Visual Agentic Tool-Use with Multimodal Memory
