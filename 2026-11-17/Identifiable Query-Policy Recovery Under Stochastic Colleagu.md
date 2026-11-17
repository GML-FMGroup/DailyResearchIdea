# Identifiable Query-Policy Recovery Under Stochastic Colleague Responses

## Motivation

Benchmarks for software-engineering agents that simulate organizational collaboration, such as SWE-CoWork, fix colleague responsiveness through a controlled-collaboration protocol, so the authored evidence distribution and the agent's query-selection policy are confounded: an agent that scores well may simply be following a favorable response schedule rather than exhibiting rational information-seeking. Because the mapping from the agent's partial-observation state to its chosen queries is unobservable under a single fixed response distribution, outcome-only metrics like fail-to-pass rates cannot separate a genuinely belief-conditioned query policy from a lucky one; SWE-CoWork's own finding that supplying complete requirements upfront lifts F2P by ~16 points yet leaves complete-task success unchanged is exactly the kind of confound this design cannot resolve. The meta-level problem is that adaptive information-seeking under partial observability — the core competence of a digital employee — is never identified as a policy, only measured as an outcome.

## Key Insight

If colleague response distributions are varied exogenously while the agent's belief state is held fixed by a matched handoff, then the agent's query policy becomes an identifiable function of belief rather than of the response schedule, because the only quantity that must stay invariant across response regimes is the belief-conditioned query map itself.

## Method

**What it is.** QUIVER (QUery-policy Identifiability under Varied Evidence Responses) is a diagnostic protocol plus estimator that takes as input a SWE-CoWork-style assignment, a family of colleague response distributions, and an agent, and outputs a recovered query-policy map together with an identifiability score and an information-gain-per-query curve.

**How it works.**

```pseudocode
Input: assignment A (brief handoff H, 26 colleagues C, records R),
       agent M, response-distribution family {P_theta}, K=8 regimes,
       N=5 seeds per regime, probe budget B=40 queries

Phase 1 — Freeze the belief surface
  For each regime theta_k in {P_theta}:
    Re-instantiate A with colleague answers drawn i.i.d. from P_theta
    Hold H and R fixed byte-identical across all k
    Verify belief-equivalence: run M's first-turn belief probe on H only;
      require KL(belief_k || belief_0) < eps_belief=0.02, else discard k

Phase 2 — Elicit query policy
  For each seed n in 1..N, regime k:
    Roll out M; log each query q_t, the observation o_t returned,
      and M's pre-query belief state b_t (via forced elicitation prompt)
    Build dataset D = {(b_t, q_t, o_t, theta_k)}

Phase 3 — Estimate the belief-conditioned query map
  Fit a categorical query-selection model
    pi(q | b, o_{1:t}) = softmax( (f(b) . e_q + g(o_{1:t}) . e_q + beta_k) / tau ),  tau=1.0
  where f is a fixed 768-d sentence embedding of the elicited belief
  text, g is a fixed 768-d sentence embedding of the concatenated
  observation history o_{1:t} (truncated to the last 2048 tokens),
  and e_q is an embedding of the query's colleague+attribute.
  Fit by maximum likelihood over D with regime fixed effects beta_k
  to absorb any residual response-schedule influence.

Phase 4 — Identifiability test
  For each held-out regime k*:
    Predict pi(q | b, o_{1:t}) using the model fitted WITHOUT k*
    Compare to empirical query frequencies in k*
    Report identifiability score ID = 1 - TV(pi_pred, pi_emp)
    (total variation distance, averaged over belief bins)
    Also report ID_b = 1 - TV(pi(q|b) marginal, pi_emp) where pi(q|b)
    is obtained by averaging pi(q|b, o_{1:t}) over the empirical
    observation-history distribution in k*.

Phase 5 — Information-gain-per-query
  For each query q at belief b, compute expected reduction in entropy
  of the requirement-attribute posterior under P_theta:
    IG(q | b) = H[attr | b] - E_{o~P_theta(.|q)} H[attr | b, o]
  Report rational-consistency R = corr(pi(q|b), IG(q|b)) per regime,
  and the slope of cumulative IG vs. query count.

Output: recovered pi(q|b, o_{1:t}), ID per regime, ID_b per regime,
        R per regime, IG curves
```

**Why this design.** We chose exogenous variation of colleague response distributions over SWE-CoWork's single fixed protocol because the fixed protocol makes the query policy unidentifiable — the agent's choices and the response schedule are collinear — accepting the cost that we must run K=8 regimes and N=5 seeds per assignment, an 8x compute multiplier. We chose a matched belief surface (byte-identical handoff and records, with a KL gate at 0.02) over merely averaging across regimes because averaging conflates belief drift with policy variation, accepting the cost that some regimes will be discarded when the belief gate fails. We chose a low-capacity categorical query-selection model with regime fixed effects over a neural policy head because the fixed effects absorb any residual schedule influence that the belief gate misses, accepting the cost that we cannot capture highly nonlinear belief-query interactions. We chose total-variation identifiability over KL divergence because TV is bounded and directly interpretable as a probability mass of mispredicted queries. A domain expert would not describe this as a variant of retrieval-based tool selection (Toolformer/Gorilla/RAG) because the target quantity is not a nearest-neighbor match between a query descriptor and a library entry; it is the invariance of a belief-to-query map across exogenously varied response distributions, a property those systems neither compute nor test.

**Why it measures what we claim.** The identifiability score ID measures the recoverability of the belief-conditioned query policy because the assumption that belief-equivalence (KL < 0.02 on the first-turn probe) plus regime fixed effects removes response-schedule confounding makes cross-regime prediction error attributable to policy rather than schedule; this assumption fails when the agent's belief is itself shaped by early observations that differ across regimes, in which case ID reflects residual belief drift rather than policy instability. The rational-consistency correlation R measures the rationality of query selection because the assumption that the requirement-attribute posterior is correctly specified under P_theta makes expected information gain the correct utility for a belief-conditioned query; this assumption fails when the agent's elicited belief text omits attributes the posterior tracks, in which case R reflects elicitation fidelity rather than query rationality. The information-gain slope measures adaptive information-seeking because the assumption that queries are the only channel through which belief updates makes cumulative IG monotone in policy quality; this assumption fails when records R provide information independent of queries, in which case the slope reflects record-reading behavior rather than query policy.

**(D) Assumptions and failure modes.** The KL belief gate measures true belief-state equivalence only under assumption A: that the forced-elicitation text is a faithful and sufficient report of the agent's belief. This assumption fails under failure mode F: the agent's belief is shaped by early regime-specific observations not captured in the first-turn probe, so the elicited text omits or distorts the regime-induced drift. To bound this, we add a per-timestep belief-matching instrument: at each query step t, we counterfactually replay the same elicited belief b_t under every regime by conditioning the query model on the realized observation history o_{1:t} as an explicit nuisance covariate with a regime-interaction term, and estimate pi(q | b, o_{1:t}) so that schedule-induced belief drift is modeled rather than assumed away. We report ID as the residual cross-regime variation after conditioning on the observation history, and add a diagnostic that measures belief drift magnitude per step (the KL between b_t and the belief obtained by replaying the same observation history under a reference regime) to bound how much of ID is drift versus policy.

## Contribution

(1) QUIVER, an identifiability protocol and estimator that recovers a belief-conditioned query-selection map from agent trajectories by exogenously varying colleague response distributions while holding the belief surface fixed. (2) A design principle: cross-regime prediction of the query map, not outcome-only F2P, is the quantity that separates rational information-seeking from response-schedule-following, operationalized as an identifiability score and a rational-consistency correlation. (3) A reusable diagnostic harness on top of SWE-CoWork-style assignments that reports per-regime information-gain curves and exposes where outcome metrics are confounded.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | SWE-CoWork assignments (101 repos) | Real multi-colleague requirement recovery with private tests |
| Primary metric | Identifiability score ID (cross-regime TV) | Bounded, interpretable mass of mispredicted queries |
| Secondary metric | ID_b (belief-marginal TV) | Isolates policy from observation-history nuisance |
| Baseline | Unmodified SWE-CoWork fixed protocol | Tests whether exogenous variation is needed at all |
| Baseline | Regime-pooled policy (no fixed effects) | Tests whether belief gate alone removes schedule confounding |
| Baseline | Neural policy head (high capacity) | Tests whether low-capacity map is sufficient for recovery |
| Ablation-of-ours | QUIVER without KL belief gate | Isolates matched-belief-surface contribution to ID |
| Ablation-of-ours | QUIVER without observation-history covariate | Isolates per-timestep belief-matching contribution |

### Why this setup validates the claim

The central claim is that a belief-conditioned query policy becomes recoverable only when colleague response distributions are varied exogenously while the belief surface is held fixed. The fixed-protocol baseline tests the first half: if query policy were identifiable under SWE-CoWork's single schedule, ID would already be high there and our 8x regime multiplier would buy nothing. The regime-pooled baseline tests the second half: it shares our exogenous variation but drops regime fixed effects, so any ID advantage we hold over it is attributable specifically to absorbing residual schedule influence rather than to variation itself. The neural policy head tests the low-capacity assumption — if a flexible head recovers the map equally well, our categorical model is unnecessary. The no-gate ablation isolates the belief-matching mechanism: without it, cross-regime prediction error should inflate because belief drift masquerades as policy variation. The no-observation-history ablation isolates the per-timestep instrument: without it, ID should drop on regimes with high belief drift, and the drift diagnostic should correlate with that drop. Total-variation ID is the right metric because it directly quantifies the probability mass of queries the held-out-regime model mispredicts, which is exactly the recoverability claim; a KL-based or accuracy-based metric would not distinguish a policy that is stable but diffuse from one that is genuinely unstable.

### Expected outcome and causal chain

**vs. Unmodified SWE-CoWork fixed protocol** — On a case where an agent must reconcile a conflicting headcount constraint split across two colleagues, the baseline produces a single observed query trace whose colleague choices are perfectly collinear with the one response schedule it ever sees, so its ID is near the floor and indistinguishable from a policy that simply mirrors the schedule. Our method instead re-instantiates the same assignment under eight response distributions while holding the handoff byte-identical, so the belief-to-query map is observed against eight different schedules and can be predicted out-of-regime. We expect a large ID gap on assignments with distributed conflicting requirements, with near-parity on single-source assignments where the schedule never mattered.

**vs. Regime-pooled policy (no fixed effects)** — On a case where two regimes differ only in one colleague's verbosity, the pooled baseline produces a query map that averages the verbose and terse schedules, so held-out prediction drifts toward whichever regime dominated the fit and ID drops even though the underlying policy is stable. Our method instead includes regime fixed effects that absorb the residual schedule influence the belief gate misses, leaving the belief-conditioned slope intact. We expect our ID to exceed the pooled baseline specifically on regimes whose response distributions are most skewed relative to the pool mean, with convergence on near-identical regimes.

**vs. Neural policy head (high capacity)** — On a case with sparse belief-bin coverage, the high-capacity head produces a query map that fits idiosyncratic seed-level noise and generalizes poorly to the held-out regime, so its ID is unstable across belief bins. Our method instead constrains the map to a low-capacity categorical form over fixed sentence embeddings, which cannot absorb seed noise and therefore predicts held-out query frequencies more faithfully where data are thin. We expect a modest ID advantage concentrated in sparsely populated belief bins, with rough parity in dense bins where both models have enough signal.

**vs. QUIVER without KL belief gate** — On a case where the first-turn belief probe is noisy, the no-gate ablation admits regimes whose initial beliefs differ, so cross-regime prediction error inflates because belief drift masquerades as policy variation. Our method instead discards regimes failing the KL < 0.02 gate, keeping the belief surface matched. We expect the full method to achieve higher ID than the no-gate ablation, with the gap largest on assignments where the handoff is ambiguous and the first-turn probe is most sensitive.

**vs. QUIVER without observation-history covariate** — On a case where early observations under one regime shift the agent's belief mid-trajectory, the no-covariate ablation attributes the resulting query change to policy instability, lowering ID. Our method instead conditions on o_{1:t} and reports ID as residual variation after modeling schedule-induced drift. We expect the full method to achieve higher ID than the no-covariate ablation, with the gap correlating positively with the per-step belief-drift diagnostic.

### What would falsify this idea

If the no-gate ablation matches full QUIVER's ID across regimes, the matched-belief-surface mechanism is doing no work and the central claim about confounding is wrong. Equally, if the unmodified fixed-protocol baseline already achieves high ID, exogenous variation is unnecessary and the 8x multiplier is unjustified. If the no-observation-history ablation matches full QUIVER's ID, the per-timestep belief-matching instrument is unnecessary and the load-bearing assumption about belief drift is not violated in practice.

## References

1. SWE-CoWork: Benchmarking Coding Agents as Digital Employees in Organizations
