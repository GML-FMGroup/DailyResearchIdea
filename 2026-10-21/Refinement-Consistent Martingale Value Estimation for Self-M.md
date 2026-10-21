# Refinement-Consistent Martingale Value Estimation for Self-Modifying Agent Search

## Motivation

Across self-improving-agent search (HGM's CMP) and constraint-following evaluation (AgentIF, CFBench), the guidance signal consumed by downstream decisions is a raw finite-sample point estimate used as a sufficient statistic for ranking, with no calibrated uncertainty and no guarantee that refining an evaluation preserves or monotonically improves the ranking. HGM itself concedes its simulation equivalence to the Gödel Machine holds only under assumptions about estimation accuracy, and its CMP estimate is a descendant-performance aggregate whose reliability silently varies with how many descendants happen to be evaluated — so the expansion decision cannot distinguish a genuinely promising lineage from one whose apparent promise is an artifact of sparse or noisy sampling. The structural gap is that no prior lineage makes the estimator's own refinement path the object of calibration, coupling uncertainty to the expansion decision.

## Key Insight

If the value functional is defined as a posterior-predictive over a hypothesis class of self-modification effects and every refinement of a clade's evaluation is required to satisfy a coarse-to-refined consistency identity, then the estimator is a martingale by construction, so uncertainty is calibrated without any held-out recalibration and the expansion threshold becomes a decision-theoretic quantity on a well-defined posterior rather than on a noisy point estimate.

## Method

**What it is.** RC-MVE (Refinement-Consistent Martingale Value Estimator) is a drop-in replacement for HGM's CMP guidance signal. Inputs: a clade tree where each node is a self-modified agent, a hypothesis class H of self-modification effect models, and per-clade evaluation budgets. Outputs: for each clade c, a posterior-predictive value functional V(c), a calibrated uncertainty band, and a binary expand/commit decision.

**Load-bearing assumption (stated explicitly).** The method assumes that the refinement path is a martingale: E[V_fine | V_coarse] = V_coarse. This holds only if (i) the filtration is nested — the fine evaluation's observation set contains the coarse one and the likelihood is unchanged — and (ii) the hypothesis class H is correctly specified. The method acknowledges that increasing n can change the descendant pool composition (deeper verification surfaces different descendants), which breaks (i), and that a misspecified H breaks (ii). The design therefore does not assert the martingale; it treats the refinement path as a supermartingale with an explicit drift term and uses a sequential test to detect drift, inflating the decision band by the estimated drift so the LCB remains valid under the acknowledged misspecification.

**How it works.**

```pseudocode
# Phase 0: hypothesis class over self-modification effects
H = { h_theta : theta in Theta }  # each h maps (parent_capability, edit) -> child_capability
# theta is a vector of per-edit-type effect coefficients; prior p(theta) = N(0, sigma0^2 I)
# sigma0 is set once from the variance of benchmark deltas across sibling edits in a pilot clade
# dimensionality of theta: one coefficient per edit type in the tree (typically 20-50 for SWE-bench Verified)

# Phase 1: clade value as posterior-predictive functional
def V(c, n_rollouts):
    # n_rollouts = evaluation budget for clade c; deeper verification = larger n
    obs = evaluate_descendants(c, n=n_rollouts)   # observed benchmark scores of descendants
    # bayes_update uses a conjugate Normal-Normal update: posterior precision = prior precision + n/sigma_obs^2
    posterior = bayes_update(prior=p(theta), likelihood=Normal(obs | h_theta, sigma_obs^2))
    return E_{theta ~ posterior}[ h_theta(c) ]   # posterior-predictive mean

# Phase 2: refinement-consistency identity (the martingale constraint)
# We do NOT assert the martingale. We define the refinement path on a fixed nested filtration:
#   - freeze the descendant pool at coarse n (do not add new descendants)
#   - only add rollouts to existing descendants
#   - model pool-shift as an explicit nuisance term in the likelihood: obs = h_theta(c) + beta * pool_shift + noise
# Then the coarse estimate is the conditional expectation of the fine estimate under the same prior.
# We VERIFY the supermartingale property empirically per clade via a sequential drift test:
#   - maintain a CUSUM statistic over repeated refinements: S_t = max(0, S_{t-1} + (V_fine_t - V_coarse_t) - delta_0)
#   - flag a violation when S_t > h (threshold h set to control false-positive rate at alpha=0.05)
#   - if flagged, inflate the decision band by the estimated drift term: band_c <- band_c + |mean(V_fine - V_coarse)|
def refine(c, n_coarse, n_fine):
    V_coarse = V(c, n_coarse)
    V_fine   = V(c, n_fine)
    # consistency identity: E[V_fine | V_coarse] == V_coarse under nested filtration and correct H
    # we VERIFY it empirically per clade and flag violations (see Phase 3).
    return V_coarse, V_fine, |V_fine - V_coarse|

# Phase 3: decision-theoretic expand/commit threshold
def decide(c):
    V_c, band_c = V(c, n_current), posterior_sd(c)   # calibrated band from posterior
    # expansion gains: expected value of one more refinement step
    gain = E[V(c, n_current + delta)] - V_c
    # cost of refinement (rollouts, verification depth)
    cost = rollout_cost(delta)
    # decision rule: expand iff lower confidence bound on gain exceeds cost
    # band_c is inflated by the estimated drift term from Phase 2 if a violation was flagged
    return (V_c - kappa * band_c) > cost_threshold(c)   # kappa = 1.0, cost_threshold from budget

# Phase 4: search loop (replaces HGM's CMP ranking)
while budget_remaining:
    for c in frontier_clades:
        if decide(c):
            expand(c); refine(c, n_current, n_current + delta)
        else:
            commit(c)   # freeze clade, allocate no more rollouts
```

**Why this design.** We chose a posterior-predictive functional over a plug-in descendant mean (HGM's CMP) because the plug-in mean has no mechanism that ties its value at budget n to its value at budget n+delta, so refinement can reorder clades arbitrarily; the posterior-predictive makes the coarse estimate the conditional expectation of the refined one, accepting the cost that we must commit to a parametric hypothesis class H whose misspecification will bias V. We chose to verify the consistency identity empirically per clade rather than merely assert it, because a misspecified H breaks the martingale property silently, accepting the cost of extra rollouts on a held-out refinement check. We chose a lower-confidence-bound decision rule with kappa=1.0 over a raw point-estimate threshold because the point estimate cannot separate a high-variance clade from a genuinely high-value one, accepting the cost that conservative clades with wide bands are under-expanded early. A domain expert would not describe this as a variant of HGM's CMP aggregation: HGM aggregates descendant scores into a scalar and ranks; RC-MVE defines a posterior over edit-effect coefficients and makes the refinement path itself the calibrated object, which is a different computational quantity, not a smoothed version of the same one.

**Why it measures what we claim.** The posterior-predictive mean V(c) measures clade metaproductivity because the assumption that each edit's effect is an additive coefficient in H makes the posterior mean equal the expected capability gain under the true edit-effect distribution; this assumption fails when edits interact non-additively (e.g., two edits that are individually neutral but jointly enabling), in which case V reflects the additive projection of a non-additive effect and understates synergistic clades. The posterior standard deviation measures estimator uncertainty because the assumption that the likelihood is correctly specified makes the posterior variance equal the frequentist variance of the refinement estimator; this assumption fails under model misspecification, in which case the band reflects within-H uncertainty rather than true sampling uncertainty and can be too narrow. The refinement-consistency residual |V_fine - V_coarse| measures calibration of the refinement path because the martingale identity makes this residual mean-zero under correct specification; this assumption fails when the evaluation budget changes the descendant pool composition (deeper verification surfaces different descendants), in which case the residual reflects pool shift rather than miscalibration. The lower-confidence-bound decision measures reliable expansion because the assumption that the band is calibrated makes the LCB a valid one-sided confidence bound on true gain; this assumption fails when the pilot-derived sigma0 is stale, in which case the LCB over-expands high-variance clades.

## Contribution

(1) RC-MVE, a posterior-predictive clade-value estimator for self-modifying agent search whose refinement path is constrained by a coarse-to-refined consistency identity, replacing HGM's raw CMP aggregate with a calibrated functional. (2) A decision-theoretic expand/commit rule derived from a lower confidence bound on the posterior-predictive gain, which separates high-variance from high-value clades that a point-estimate threshold conflates. (3) An empirical consistency-residual diagnostic that flags hypothesis-class misspecification per clade, giving a concrete signal for when the martingale property is violated in practice.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | SWE-bench Verified self-modification tree | Matches HGM's coding-agent setting and edit-effect structure |
| Primary metric | Metaproductivity per CPU-hour at fixed budget | Directly measures clade value under refinement, not raw score |
| Baseline 1 | HGM CMP ranking | Tests whether posterior-predictive beats plug-in mean ranking |
| Baseline 2 | Point-estimate LCB with fixed kappa | Tests whether calibrated band beats raw threshold |
| Baseline 3 | RC-MVE with asserted (unverified) martingale | Tests whether empirical consistency verification matters |
| Ablation-of-ours | RC-MVE with additive-only H (no interaction terms) | Isolates the cost of hypothesis-class misspecification |
| Control | RC-MVE with fixed descendant pool across refinement levels | Separates true miscalibration from pool shift |

### Why this setup validates the claim

The central claim is that making the refinement path itself a calibrated object — a posterior-predictive value functional with a verified martingale constraint and a banded decision rule — yields better metaproductivity per unit budget than HGM's plug-in descendant mean. SWE-bench Verified is the right substrate because HGM already operates there, so the edit-effect distribution, clade structure, and rollout cost are comparable and the comparison is not confounded by task shift. The primary metric, metaproductivity per CPU-hour at fixed budget, is chosen deliberately: a raw benchmark score would reward clades that happen to sit on high-scoring descendants, whereas per-hour metaproductivity rewards the allocation decision that RC-MVE actually changes. HGM-CMP tests the sub-claim that the posterior-predictive functional reorders clades more stably than the plug-in mean under refinement. The point-estimate-LCB baseline tests the sub-claim that the calibrated band, not the threshold shape, is what separates high-variance from high-value clades. The asserted-martingale baseline tests the sub-claim that empirically verifying the consistency identity per clade catches misspecification that assertion would silently absorb. The additive-only ablation isolates the acknowledged cost: if removing interaction terms does not hurt, the misspecification penalty we accept is small, and if it does, we learn where the gain actually comes from. The fixed-pool control holds the descendant pool constant across refinement levels, so any residual |V_fine - V_coarse| in that condition is genuine miscalibration rather than pool shift; comparing the fixed-pool residual to the free-pool residual isolates the pool-shift component. Together these form a falsifiable test: if RC-MVE's advantage is real, it must be concentrated in the clades where the predicted failure modes of each baseline occur, not spread uniformly.

### Expected outcome and causal chain

**vs. HGM CMP ranking** — On a case where a clade has a single high-scoring descendant that dominates its plug-in mean, HGM-CMP ranks that clade top and expands it, because its scalar aggregation has no mechanism tying the value at budget n to the value at n+delta, so refinement can reorder clades arbitrarily and the early ranking is driven by one lucky rollout. RC-MVE instead conditions on the edit-effect posterior, so the same clade's posterior-predictive mean is pulled toward the additive projection of its edits and its band stays wide, and the LCB rule declines to expand it. We expect a noticeable metaproductivity gap concentrated on clades with high within-clade score variance and near-parity on low-variance clades.

**vs. Point-estimate LCB with fixed kappa** — On a case where a genuinely high-value clade has a wide posterior band because its edit type is under-observed, the point-estimate rule expands it on the raw mean, because a point estimate cannot separate a high-variance clade from a genuinely high-value one, so early budget flows to noisy clades. RC-MVE's calibrated band makes the LCB conservative on exactly those clades and redirects expansion to clades whose bands have narrowed, so we expect RC-MVE to spend fewer rollouts on high-variance clades and show a per-hour advantage that grows with the fraction of under-observed edit types in the tree.

**vs. RC-MVE with asserted martingale** — On a case where the additive hypothesis class is misspecified because two edits are individually neutral but jointly enabling, the asserted-martingale variant never flags the violation, so its consistency residual is silently nonzero and its band is too narrow, and it over-expands synergistic clades while under-expanding the clades its misspecified V understates. RC-MVE's per-clade sequential drift test (CUSUM over repeated refinements) flags the violation, widens the band by the estimated drift, and defers the commit decision. We expect the asserted variant to show a calibration residual that grows with tree depth while RC-MVE's stays near zero, and a metaproductivity gap concentrated on clades containing interaction-prone edit pairs.

**vs. Fixed-pool control** — On a case where deeper verification surfaces a different descendant set, the free-pool RC-MVE's residual |V_fine - V_coarse| includes a pool-shift component that the fixed-pool control removes. We expect the fixed-pool residual to be near zero under correct specification, and the difference between free-pool and fixed-pool residuals to account for the pool-shift contribution, confirming that the sequential drift test is detecting genuine miscalibration rather than pool shift.

### What would falsify this idea

If RC-MVE's metaproductivity advantage over HGM-CMP is uniform across clades rather than concentrated on high-within-clade-variance and interaction-prone clades, the causal story that the posterior-predictive functional and verified martingale are doing the work is wrong. Likewise, if the additive-only ablation matches RC-MVE, the misspecification penalty we claim to accept is not the binding constraint and the design rationale collapses.

## References

1. Huxley-Gödel Machine: Human-Level Coding Agent Development by an Approximation of the Optimal Self-Improving Machine
2. AGENTIF: Benchmarking Instruction Following of Large Language Models in Agentic Scenarios
3. CFBench: A Comprehensive Constraints-Following Benchmark for LLMs
4. Judging LLM-as-a-judge with MT-Bench and Chatbot Arena
5. COLLIE: Systematic Construction of Constrained Text Generation Tasks
