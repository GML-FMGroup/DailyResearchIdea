# Error-Correlated Agreement: A Calibrated Correctness Oracle for Cross-VLM Spatial Answer Filtering

## Motivation

Across the current frontier of spatial-reasoning data pipelines, cross-VLM agreement is treated as a hard binary filter for retaining reference answers — EvidenceConnect in Zooming without Zooming v2 keeps a candidate answer only when independent VLMs agree, and discards region-ablation survivors otherwise. This implicitly assumes conditional independence of VLM errors, so when two VLMs share a systematic spatial bias (e.g., both flip left/right under occlusion, both anchor to the wrong reference object in a chained comparison), agreement is not evidence of correctness but of a shared failure mode; the filter cannot distinguish consensus-plausible from correct, and no branch in this paper-tree models or estimates inter-VLM error correlation.

## Key Insight

Cross-VLM agreement is only informative about correctness when the estimated off-diagonal error correlation for that query's spatial operation is near zero, so a per-query conditional-independence-violation statistic — not the raw vote count — is the sufficient statistic that separates genuine corroboration from shared-bias consensus.

## Method

**What it is.** We introduce ECO (Error-Correlated Oracle), a calibrated correctness oracle that ingests the multi-VLM answer set for a spatial query plus a small anchor set of items with known ground truth, estimates the full inter-VLM error-correlation matrix conditioned on the query's spatial operation, and outputs a correctness probability that is monotone in the estimated shared-bias component rather than in vote count.

**How it works.**

```pseudocode
INPUT: query q with spatial operation op(q) in {left/right, above/below, containment, chained-comparison, occlusion-localization}
       answer set A = {a_1,...,a_K} from K=4 frontier VLMs
       anchor set D_op = {(x_i, y_i*)} with ground truth, |D_op| = 60 per operation
       region-ablation signal r(q) in {0,1} from EvidenceConnect-style ablation

# Phase 1: per-operation error model from anchors
for each op in operations:
    for each pair (j,k):
        e_jk[op] = empirical P(VLM_j wrong AND VLM_k wrong | op)   # from D_op
        rho_jk[op] = (e_jk[op] - p_j[op]*p_k[op]) / sqrt(p_j[op](1-p_j[op]) p_k[op](1-p_k[op]))
    Sigma[op] = [rho_jk[op]]  # K x K error-correlation matrix, shrinkage lambda=0.1 toward identity
    # NEW: posterior over rho_jk[op] via bootstrap (B=1000 resamples of D_op) or Beta posterior on e_jk[op]
    #      store 95% CI [rho_lo_jk, rho_hi_jk] for each pair

# Phase 2: per-query conditional-independence-violation statistic
for query q:
    votes = [1{a_k == mode(A)} for k in 1..K]          # agreement indicator
    agree_set = {k : votes[k]=1}
    # NEW: query-conditional null using difficulty proxy d(q) = normalized negative mean token log-prob of the modal answer across agreeing VLMs (or self-consistency spread)
    #      p_j(q) = sigmoid(alpha_j[op] + gamma_j[op] * d(q))  # per-VLM logistic recalibration on anchors
    #      fit alpha_j, gamma_j on D_op via logistic regression (L2 penalty 0.01)
    # observed joint error mass among agreeing VLMs, proxied by anchor-calibrated per-VLM error rates
    obs = mean_{j<k in agree_set} p_j(q) * p_k(q) + rho_jk[op] * sqrt(p_j(q)(1-p_j(q)) p_k(q)(1-p_k(q)))  # model-implied joint error
    exp_CI = mean_{j<k in agree_set} p_j(q) * p_k(q)
    CIV(q) = (obs - exp_CI) / sqrt(var_CI + eps)        # conditional-independence-violation z-score
    # NEW: informative flag gated on posterior: informative = (CIV(q) < tau_op) AND (rho_lo_jk[op] > 0 for some j<k in agree_set)

# Phase 3: calibrated correctness probability (monotone in shared-bias, NOT vote count)
shared_bias(q) = max_{j<k in agree_set} rho_jk[op(q)]    # dominant shared-bias component
logit P_correct(q) = beta0[op] + beta1 * (-CIV(q)) + beta2 * (-shared_bias(q)) + beta3 * r(q)
# beta fit by isotonic-then-logistic calibration on held-out anchors; beta2 < 0 enforced
OUTPUT: P_correct(q) in [0,1], plus flag informative = (CIV(q) < tau_op) with tau_op = 1.5
```

**Why this design.** We chose to estimate the full K×K error-correlation matrix per spatial operation rather than a single scalar agreement rate because shared spatial biases are operation-specific — occlusion-induced left/right flips do not co-occur with chained-comparison anchoring errors — so pooling across operations would average a strong correlation with a zero one and destroy the signal; the cost is that each operation needs its own anchor budget (60 items), which is feasible given frontier pipelines already emit region-ablation-labeled answer sets. We chose a conditional-independence-violation z-score over the raw vote count because vote count is exactly the statistic that EvidenceConnect (Zooming without Zooming v2) uses as a hard filter, and any monotone function of vote count cannot distinguish two agreeing VLMs with rho=0.6 from two with rho=0.05; the cost is that CIV requires per-VLM error-rate estimates, which we proxy from anchors rather than from the query itself. We chose an isotonic-then-logistic calibration of P_correct over a learned neural head because the oracle has only ~300 anchor items per operation and a 4-parameter logistic with sign constraints (beta2 < 0) is identifiable at that scale, whereas a neural head would overfit and, more importantly, could learn a non-monotone map that violates the motivation's requirement that P_correct be monotone in shared bias; the cost is that the calibration is only as good as the anchor coverage of each operation. We chose to keep region-ablation r(q) as a separate additive term rather than folding it into the agreement statistic because ablation measures a different property (evidence dependence) and conflating it with error correlation would reintroduce the very identifiability problem we are solving.

**Why it measures what we claim.** The conditional-independence-violation statistic CIV(q) measures the excess joint-error mass among agreeing VLMs over what conditional independence predicts, because under the named assumption that anchor-estimated per-VLM error rates p_j[op] transfer to the query's operation class, the CI-predicted joint error p_j·p_k is the correct null and any excess is attributable to shared bias; this assumption fails when a query's difficulty is far outside the anchor distribution (e.g., a novel occlusion pattern), in which case CIV reflects anchor-distribution mismatch rather than shared bias. The shared-bias component max rho_jk[op(q)] measures the dominant pairwise error correlation, because under the assumption that the largest off-diagonal correlation dominates the joint failure probability (a union-bound argument over pairs), the max is a sufficient summary; this assumption fails when many small correlations jointly outweigh one large one, in which case the max underestimates total shared bias. The calibrated P_correct measures the probability that the modal answer is correct, because under the assumption that the logistic calibration on held-out anchors is well-specified and beta2 < 0 is enforced, the map is monotone decreasing in shared bias as the motivation requires; this assumption fails when the anchor set is too small to identify beta2, in which case P_correct collapses toward the base rate and the oracle becomes uninformative rather than wrong. The region-ablation term r(q) measures evidence dependence, because under the assumption that ablating the selected regions removes the evidence needed for the answer, a surviving answer is evidence-independent; this assumption fails for partially redundant evidence, in which case r(q) flags a valid answer as evidence-independent.

**(D) Explicit assumptions and failure modes.**
- **Transfer of anchor error rates:** p_j[op] and rho_jk[op] estimated from 60 anchor items per operation are assumed to transfer to arbitrary queries within that operation class. This fails when query difficulty is outside the anchor distribution (e.g., novel occlusion pattern), in which case CIV reflects difficulty shift rather than shared bias. We mitigate by query-conditional recalibration p_j(q) = sigmoid(alpha_j[op] + gamma_j[op] * d(q)) using a difficulty proxy d(q) (normalized negative mean token log-prob of the modal answer across agreeing VLMs), fit on anchors.
- **Max-rho dominance:** max_{j<k} rho_jk is assumed to capture total shared-bias mass via a union bound. This fails when many small correlations jointly outweigh one large one; we test this with a synthetic correlated-error generator where true shared-bias mass is known.
- **Anchor sampling uncertainty:** rho_jk[op] from 60 binary items has sampling error ~0.13; we report bootstrap 95% CIs and gate the informative flag on rho_lo_jk > 0, so the oracle abstains when the anchor budget cannot distinguish rho=0.6 from rho=0.05.

## Contribution

(1) ECO, a calibrated correctness oracle that replaces the binary cross-VLM agreement filter with a per-query conditional-independence-violation statistic computed from an anchor-estimated inter-VLM error-correlation matrix, producing a correctness probability monotone in shared bias rather than in vote count. (2) The design principle that agreement is only evidence of correctness when the estimated off-diagonal error correlation for the query's spatial operation is low, operationalized as an operation-conditioned correlation matrix with a shrinkage estimator and an isotonic-logistic calibration with a sign constraint enforcing monotonicity. (3) An anchor-supervision protocol that reuses region-ablation-labeled multi-VLM answer sets already emitted by frontier spatial pipelines, requiring only 60 ground-truth anchors per spatial operation to identify the K×K correlation structure.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Spatial-reasoning benchmark with per-operation strata (left/right, above/below, containment, chained-comparison, occlusion-localization) | Operation strata let us test operation-specific correlation claims |
| Primary metric | Selective accuracy vs. coverage (risk–coverage curve) on oracle-flagged answers | Directly measures whether oracle confidence is calibrated |
| Baseline | EvidenceConnect agreement filter (vote-count threshold) | Tests sub-claim that vote count conflates rho=0.6 and rho=0.05 |
| Baseline | Per-VLM confidence average (no correlation modeling) | Tests sub-claim that marginal error rates alone are insufficient |
| Baseline | Random / base-rate oracle | Establishes the uninformative floor for risk–coverage |
| Baseline | Dawid-Skene with learned KxK error-covariance (correlated-error ensemble) | Tests that ECO's CIV is not a reparameterization of existing correlated-error aggregation |
| Ablation-of-ours | ECO with shared_bias term removed (CIV + r only) | Isolates whether the correlation term carries the signal |
| Ablation-of-ours | ECO with query-conditional null replaced by operation-mean null | Tests whether difficulty-proxy recalibration matters |
| Ablation-of-ours | ECO with informative flag disabled (always output) | Tests value of posterior-gated abstention |

### Why this setup validates the claim

The central claim is that correctness is predictable from the *structure* of inter-VLM error correlation, not from how many VLMs agree. A risk–coverage curve is the right metric because it asks the exact question the oracle is built to answer: when ECO says "trust this answer," is it right more often than when it says "don't"? The vote-count baseline (EvidenceConnect) is the decisive comparison, because if ECO's gain over it is real, it must appear precisely where vote count is blind — queries where agreeing VLMs share high rho. The per-VLM-confidence baseline tests whether correlation modeling adds anything beyond better-calibrated marginals; if ECO ties it, the K×K matrix is doing no work. The Dawid-Skene baseline with learned error covariance tests whether ECO's per-operation CIV statistic is merely a reparameterization of existing correlated-error aggregation; if ECO does not outperform it, the novelty of the CIV statistic is questionable. The ablation dropping shared_bias is the internal falsifier: if removing the correlation term does not degrade the curve, the mechanism we claim is causal is not. The operation-stratified dataset is essential because our theory predicts operation-specific correlation, so pooling would mask the effect we are trying to demonstrate. Additionally, we hold out a difficulty stratum (e.g., high-occlusion-severity items) to validate the transfer assumption: if ECO's query-conditional null does not improve calibration on this held-out stratum relative to the operation-mean null, the transfer assumption fails and the oracle's gains are suspect.

### Expected outcome and causal chain

**vs. EvidenceConnect agreement filter** — On an occlusion-localization query where all four VLMs answer "left" but share the same occlusion-induced left/right flip bias (high rho), the baseline accepts the answer because vote count is maximal, producing a confidently wrong output. Our method instead reads rho_jk[occlusion] from anchors, sees CIV strongly positive (excess joint-error mass among the agreeing set), and lowers P_correct because beta2<0 penalizes shared bias. We expect a pronounced risk–coverage gap concentrated on high-rho occlusion and chained-comparison subsets, with near-parity on low-correlation left/right items where vote count is already sound.

**vs. Per-VLM confidence average** — On a chained-comparison query where each VLM is individually well-calibrated (p_j≈0.15) but their errors co-occur, the baseline averages marginals to a high confidence because it assumes errors are independent, so it over-trusts the agreeing set. Our method instead uses the off-diagonal rho to correct the joint-error prediction, so CIV and shared_bias both flag the correlated failure and P_correct drops. We expect the baseline to look competitive on independent-error operations (above/below) but to lose coverage-adjusted accuracy specifically on operations with the largest estimated off-diagonal correlations.

**vs. Random / base-rate oracle** — On any query, the base-rate oracle assigns a constant P_correct, so its risk–coverage curve is flat and it cannot trade coverage for accuracy. Our method instead ranks queries by monotone-decreasing shared bias, so selecting the top-coverage slice should yield accuracy noticeably above base rate while the bottom slice falls below it. We expect a monotone, non-flat curve across all operations — the minimal signal that the oracle is informative at all — and a flat curve would already indicate the calibration collapsed.

**vs. Dawid-Skene with learned KxK error-covariance** — On queries where the error correlation is operation-specific and the Dawid-Skene model pools across operations, the baseline will average a strong correlation with a zero one, diluting the shared-bias signal. Our method estimates rho per operation and uses a query-conditional null, so it should maintain calibration on high-rho subsets where Dawid-Skene's pooled covariance underfits. We expect ECO to outperform Dawid-Skene on occlusion-localization and chained-comparison, with parity on low-correlation operations.

### What would falsify this idea

If ECO's risk–coverage gain over the vote-count baseline is uniform across operations rather than concentrated on the high-rho subsets our causal story predicts, the shared-bias mechanism is not what drives the improvement. Likewise, if the ablation dropping shared_bias matches the full oracle, the correlation term is doing no work and the central claim is wrong. Additionally, if the query-conditional null does not improve calibration on the held-out difficulty stratum, the transfer assumption fails and the oracle's gains are not attributable to shared-bias estimation.

## References

1. Zooming without Zooming v2: From Seeing Details to Connecting Evidence for Generalizable Multimodal Perception
2. HiDe: Rethinking The Zoom-IN method in High Resolution MLLMs via Hierarchical Decoupling
3. InSight-o3: Empowering Multimodal Foundation Models with Generalized Visual Search
4. Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution
5. ZoomEye: Enhancing Multimodal LLMs with Human-Like Zooming Capabilities through Tree-Based Image Exploration
6. Divide, Conquer and Combine: A Training-Free Framework for High-Resolution Image Perception in Multimodal Large Language Models
