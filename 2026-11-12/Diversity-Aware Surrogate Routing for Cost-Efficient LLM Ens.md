# Diversity-Aware Surrogate Routing for Cost-Efficient LLM Ensembles

## Motivation

Existing cost-aware ensemble routers such as PISA-E (Ensemble Routing for Prompt-Adaptive Cost-Aware LLM Inference) select ensembles using a surrogate that predicts ensemble success from individual model capabilities alone, structurally ignoring pairwise error correlation and prediction disagreement. Because ensemble gain is governed by complementarity rather than per-model accuracy, this surrogate systematically over-values redundant strong models and under-values weaker but decorrelated ones, so cost-aware routing cannot exploit the very redundancy structure that makes heterogeneous LLM ensembles worthwhile.

## Key Insight

Ensemble success decomposes into a capability term plus a diversity term, and the diversity term is estimable at routing time from cheap pairwise disagreement probes without running the ensemble, so a router that scores both terms selects complementary ensembles that a capability-only surrogate provably cannot distinguish.

## Method

**What it is.** DART (Diversity-Aware Routing with Tandem surrogates) is a cost-aware ensemble router that scores each candidate ensemble with two additive surrogate heads — a capability head inherited from PISA-E and a diversity head trained on pairwise disagreement statistics — then solves a cardinality-and-composition selection under an explicit cost-quality preference λ.

**How it works.**

```pseudocode
Input: prompt x, candidate pool M, cost c(m), preference λ, probe set P (K=8 short prompts)
# Phase 1: capability term (reused from PISA-E)
for m in M:
    z(x)      = soft_cluster_membership(x)            # frozen clusters (C=20 clusters, k-means on prompt embeddings)
    cap(m,x)  = f_m(z(x))                             # model-specific predictor (2-layer MLP, hidden=128, GeLU)
# Phase 2: diversity term via disagreement probes
for m in M:
    for p in P:
        y_hat(m,p) = cached_prediction(m,p)           # one-time offline cache
for (i,j) in pairs(M):
    d_ij(x) = sum_p 1[y_hat(i,p) != y_hat(j,p)] / K    # disagreement rate
    rho_ij  = disagreement_correlation(d_ij over clusters)  # Pearson correlation of per-cluster d_ij values
# Phase 3: ensemble score
for S in candidate_ensembles(M, max_size=4):
    cap_S  = g( {cap(m,x)}_{m in S} )                 # learned aggregation g (linear layer, weights initialized to uniform)
    div_S  = (1/|S|^2) * sum_{i,j in S} rho_ij * (1 - d_ij(x))
    score(S) = cap_S + beta * div_S - λ * sum_{m in S} c(m)   # beta=0.5
pick S* = argmax_S score(S)
Output: S*
```

**Why this design.** We chose a *cached offline disagreement probe set* over on-the-fly ensemble execution because probing every candidate ensemble at routing time would erase the cost advantage that motivates routing, accepting that the probe set must be representative of the prompt distribution. We chose an *additive two-head surrogate* (capability + diversity) over a single monolithic predictor because the two terms have different data requirements — capability needs per-model labels, diversity needs only pairwise outputs — so separating them lets us train the diversity head on unlabeled model outputs, accepting a small loss of expressivity from the additive form. We chose *pairwise disagreement correlation* over embedding-space similarity because disagreement is the quantity that actually enters ensemble error, whereas similarity is a proxy that conflates stylistic with error-level agreement. A domain expert would not describe this as PISA-E plus a diversity feature: PISA-E's surrogate is trained end-to-end on ensemble outcomes with no mechanism that can *distinguish* two ensembles with identical capability profiles but different correlation structure, whereas DART's diversity head is a separately identified quantity whose training signal is pairwise disagreement, not ensemble success.

**Why it measures what we claim.** The disagreement rate `d_ij(x)` measures *error decorrelation between models i and j* because under the assumption that probe-set errors are exchangeable with target-prompt errors, the probe disagreement rate is an unbiased estimator of target disagreement; this assumption fails when the probe set is drawn from a different difficulty regime, in which case `d_ij` reflects stylistic divergence rather than error independence. The correlation term `rho_ij` measures *redundancy* because under the assumption that per-cluster disagreement rates share a common factor structure, their correlation is proportional to shared error variance; this fails when clusters are too few, in which case `rho_ij` reflects sampling noise. The additive score `cap_S + beta*div_S` measures *expected ensemble accuracy minus cost* because under the assumption that ensemble error decomposes into a bias term (capability) and a variance term (diversity), the two heads estimate the two terms separately; this fails when models are strongly correlated in capability, in which case the additive form double-counts shared bias.

**(D) Equivalence, assumption, and failure mode.** The diversity term `div_S` is intended to measure *expected ensemble error reduction from decorrelation* because under assumption A — probe-set errors are exchangeable with target-prompt errors (same difficulty regime and label distribution) — the probe disagreement rate `d_ij` is an unbiased estimator of target-prompt disagreement; this assumption fails when the probe set is drawn from a different difficulty regime, in which case `d_ij` reflects stylistic divergence rather than error independence. To operationalize and bound this assumption, we add a *regime-conditional calibration*: for each frozen cluster c (C=20), we compute a cluster-specific disagreement rate `d_ij^c` from the cached probe predictions, and we report the Pearson correlation between `d_ij^c` and held-out target-prompt ensemble error across difficulty regimes (defined by prompt length and cluster membership). We flag prompts where the probe regime diverges (e.g., when the prompt's cluster has fewer than 5 probe examples or when the cluster-conditional disagreement rate deviates by more than 2 standard deviations from the global rate) and report the regime boundaries where the exchangeability assumption breaks, so the estimator's validity is empirically bounded rather than assumed.

## Contribution

(1) DART, a two-head surrogate router that adds an explicitly trained pairwise-disagreement diversity head to cost-aware ensemble selection, making redundancy a first-class routing quantity. (2) A design principle: diversity must be estimated from pairwise model outputs rather than from capability embeddings, because capability-identical ensembles can have different correlation structure that capability-only surrogates cannot resolve. (3) An offline disagreement-probe cache that makes diversity-aware routing cost-neutral at inference time.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Four routing benchmarks, in- and out-of-distribution splits | Matches PISA-E protocol; tests probe-transfer assumption |
| Primary metric | Macro-average cost-performance hypervolume | Jointly rewards quality and cost savings |
| Baseline | Single-model router (best per-prompt model) | Tests whether ensembles beat best-single selection |
| Baseline | PISA-E end-to-end ensemble surrogate | Tests whether explicit diversity head adds signal |
| Baseline | Fixed top-k ensemble | Tests whether composition selection matters |
| Ablation-of-ours | DART with diversity head disabled (β=0) | Isolates contribution of disagreement term |
| Calibration diagnostic | Regime-conditional correlation between probe d_ij and target ensemble error | Bounds exchangeability assumption A |

### Why this setup validates the claim

The central claim is that a separately identified diversity head, trained on cached pairwise disagreement rather than ensemble outcomes, lets DART select ensembles whose correlation structure improves accuracy per unit cost beyond what a capability-only surrogate can find. The single-model router tests the baseline sub-claim that ensembling is worth its cost at all: if DART cannot beat best-single selection on hypervolume, the ensemble premise collapses. PISA-E is the decisive baseline because it shares DART's capability head but lacks any mechanism to distinguish ensembles with identical capability profiles but different redundancy — so any gap between DART and PISA-E is attributable specifically to the diversity head. The fixed top-k ensemble isolates composition selection from diversity scoring. The β=0 ablation is the internal control: it removes only the diversity term, so the predicted effect should appear as a DART-minus-ablation gap concentrated on prompts where candidate models are individually strong but correlated. The calibration diagnostic directly measures the correlation between probe-set d_ij and held-out target-prompt ensemble error across difficulty regimes, reporting the regime boundaries where the exchangeability assumption breaks, so the estimator's validity is empirically bounded rather than assumed. Hypervolume is the right metric because it directly measures the quality-cost tradeoff the router is supposed to optimize, and it penalizes both over-spending on redundant models and under-spending on weak ones.

### Expected outcome and causal chain

**vs. Single-model router** — On prompts where two mid-capability models make different errors, the single-model router commits to whichever model has the higher capability score and inherits that model's full error mass, because it has no mechanism to exploit complementary failure modes. Our method instead selects a pair whose disagreement rate is high on the probe set, so their errors partially cancel, because the diversity head explicitly rewards decorrelated members. We expect a visible hypervolume gap concentrated on prompts whose best-single capability sits in the mid range, with near-parity on prompts where one model dominates the pool.

**vs. PISA-E end-to-end surrogate** — On a case where two candidate ensembles have nearly identical capability profiles but one pairs correlated models and the other pairs decorrelated ones, PISA-E's surrogate assigns them near-identical scores because its training signal is ensemble success, which is dominated by capability variance and cannot resolve correlation structure at this granularity. Our method instead separates the two scores because the diversity head is trained directly on pairwise disagreement and rewards the decorrelated pair. We expect DART to win specifically on the subset of prompts where top-capability ensembles are internally redundant, with parity elsewhere.

**vs. Fixed top-k ensemble** — On prompts drawn from a difficulty regime where the top-k models happen to share a stylistic prior, the fixed ensemble produces confidently wrong agreement because it cannot drop a redundant member or swap in a decorrelated lower-ranked model. Our method instead re-selects composition per prompt under the λ cost preference, substituting a decorrelated member when the diversity term outweighs the capability loss. We expect a gap that grows with the correlation among top-ranked models and shrinks toward parity on prompts where top models are already diverse.

### What would falsify this idea

If DART's gain over PISA-E is uniform across prompts rather than concentrated on the subset where candidate models are individually strong but highly correlated, then the diversity head is not doing the work we claim and the improvement is attributable to something other than disagreement-aware composition. Equivalently, if the β=0 ablation matches full DART everywhere, the central claim is wrong.

## References

1. Ensemble Routing for Prompt-Adaptive Cost-Aware LLM Inference
