# Compositional Success-Estimator Algebra for Routed Embodied Agents

## Motivation

REA's route memory stores per-policy success rates measured in isolation, and the router selects the argmax policy at each boundary; however, this estimator is not proven compositional across sub-task boundaries or distribution shifts. As a result, the router's 'highest expected success' rule can silently accumulate failure probability across a multi-step task, a structural gap shared by PaLM-E's monolithic planner and LIBERO-PRO's per-task policy selection, both of which rely on empirical lookup rather than an algebra that composes sub-task success estimates.

## Key Insight

Per-policy success is a monoid homomorphism from the free monoid of sub-task traces into a success semiring whose multiplication is the composition operator; this algebraic structure makes the router's decision invariant exactly the semiring product of boundary-local estimates, so route selection is compositional by construction rather than by empirical lookup.

## Method

## Compositional Success-Estimator Algebra (CSEA)

**What it is.** CSEA replaces REA's per-policy success table with a semiring-valued estimator: each fast policy $p$ is assigned a success element $s(p) \in \mathcal{S}$ in a success semiring $(\mathcal{S}, \oplus, \otimes, 0, 1)$, and a sub-task trace $\tau = (p_1, \ldots, p_k)$ receives the product $S(\tau) = s(p_1) \otimes \cdots \otimes s(p_k)$. The router selects the boundary policy whose composed estimate maximizes $S$ under the semiring order.

### (B) How it works

```pseudocode
# Success semiring S = (R_{>=0} ∪ {⊥}, ⊕, ⊗, 0, 1)
#   ⊕ = probability-of-success union:  a ⊕ b = a + b - a*b
#   ⊗ = independent-failure product:   a ⊗ b = a * b   (a,b in [0,1])
#   ⊥ absorbs:  a ⊗ ⊥ = ⊥  (unmeasurable boundary)
#   0 = 0.0,  1 = 1.0
#
# Load-bearing assumption A (stated explicitly):
#   Failures at distinct boundaries are conditionally independent given the trace prefix.
#   Under A, s(p_i) is a marginal per-policy success rate and ⊗ is the independent-failure product.
#   When A fails (a single silent failure corrupts multiple downstream boundaries, e.g., a dropped object),
#   the product under-estimates true failure probability and the trace should collapse to ⊥.
#
# Correlation-aware fallback (operationalizes A):
#   For each ordered pair (p_i, p_{i+1}) in the trace, route_memory stores a pairwise failure-correlation
#   estimate ρ(p_i, p_{i+1}) ∈ [-1, 1] computed from co-observed boundary outcomes in the same route-memory cell.
#   If any |ρ| > τ_ρ (τ_ρ = 0.3), the trace is treated as a single ⊥ element (silent-failure signature).

function CSEA_route(trace_prefix, candidate_policies, route_memory, boundary_ctx):
    # 1. Retrieve boundary-local success element per candidate
    for p in candidate_policies:
        s_local[p] = route_memory.lookup(p, boundary_ctx)   # ∈ S, may be ⊥
    # 2. Compose with prefix via monoid homomorphism h
    #    h(trace) = s(p_1) ⊗ ... ⊗ s(p_k);  h(empty) = 1
    s_prefix = fold(⊗, [route_memory.element(q) for q in trace_prefix], init=1)
    # 2b. Correlation-aware collapse: if any adjacent pair in trace_prefix has |ρ| > τ_ρ, set s_prefix = ⊥
    if any(|route_memory.corr(q_i, q_{i+1})| > τ_ρ for adjacent q_i, q_{i+1} in trace_prefix):
        s_prefix = ⊥
    # 3. Router decision invariant: argmax over composed estimate
    for p in candidate_policies:
        s_composed[p] = s_prefix ⊗ s_local[p]
    # 4. Boundary-triggered judge gate: consult brain iff any s_composed = ⊥
    #    or if max_s - second_max_s < τ_margin (τ_margin = 0.05)
    best = argmax_p s_composed[p]
    if s_composed[best] == ⊥ or (max - second_max) < τ_margin:
        best = brain_consult(boundary_ctx, trace_prefix)   # fallback to REA judge
    # 5. After execution, update route memory via semiring homomorphism
    observed = boundary_judge.success(boundary_ctx)         # ∈ {0,1}
    route_memory.update(best, boundary_ctx, observed, α=0.1) # exponential smoothing in S
    # 5b. Update pairwise correlation estimate for adjacent boundaries in the executed trace
    route_memory.update_corr(prev_policy, best, observed_prev, observed, α=0.1)
    return best
```

**Why this design.** We chose a semiring homomorphism over an empirical lookup table because the lookup table has no composition operator — REA's router can only compare policies at a single boundary, whereas the homomorphism lets the router compare *traces* by folding per-boundary estimates through $\otimes$, accepting the cost that the semiring's independence assumption (multiplication as independent-failure product) is an approximation that must be audited. We chose the probability-union $\oplus$ over a max operator because union preserves the monotone accumulation of failure risk across sub-tasks, accepting that $\oplus$ is not idempotent and therefore route memory must be updated with exponential smoothing rather than overwrite. We chose to keep the boundary-triggered judge gate from REA rather than replacing it with a learned confidence head because the gate's trigger condition is now algebraically defined ($s = \bot$ or margin $< \tau_{\text{margin}}$), accepting that the gate fires more often in unfamiliar boundary contexts where $\bot$ is returned. A domain expert would not describe CSEA as a variant of Toolformer-style retrieval because the router's decision is a semiring product over the trace, not a nearest-neighbor lookup; nor is it a variant of a Process Reward Model because the per-policy success elements are measured boundary statistics composed by a fixed algebra, not a learned step-level value function.

**Why it measures what we claim.** The computational quantity $s(p) \in \mathcal{S}$ measures per-policy boundary-local success because the route-memory update is an exponential-smoothing estimator of the Bernoulli success rate conditioned on boundary context, under the assumption that boundary contexts are exchangeable within a route-memory cell; this assumption fails when the distribution shifts within a cell (e.g., lighting change mid-episode), in which case $s(p)$ reflects a stale mixture rather than current success. The composition $s(p_1) \otimes \cdots \otimes s(p_k)$ measures trace-level success because $\otimes$ is the independent-failure product, under the assumption A that failures at distinct boundaries are conditionally independent given the trace prefix; this assumption fails when a single silent failure corrupts multiple downstream boundaries (e.g., a dropped object), in which case the product under-estimates the true failure probability and the router should treat the trace as a single $\bot$ element — which the correlation-aware fallback (step 2b) enforces when measured $|\rho| > \tau_\rho$. The margin threshold $\tau_{\text{margin}}$ measures decision confidence because it is the gap between the top two composed semiring estimates, under the assumption that near-ties in composed success correspond to genuine ambiguity requiring brain consultation; this assumption fails when two policies have correlated success (e.g., both fail on the same object class), in which case the margin reflects a shared blind spot rather than ambiguity and the gate should fire on correlation, not margin.

## Contribution

["A success-semiring algebra for routed embodied agents (CSEA) that replaces REA's per-policy success table with a monoid homomorphism from sub-task traces to a semiring, giving the router an explicit composition operator across boundaries and distribution shifts.", 'A boundary-triggered judge gate whose trigger condition is algebraically defined ($s = \\bot$ or margin $< \\tau_{\\text{margin}}$), collapsing the empirical lookup and the brain fallback onto the same semiring surface.', 'An audit procedure for the independence assumption underlying the semiring product, identifying when a trace should be collapsed to a single $\\bot$ element rather than composed.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | LIBERO-PRO long-horizon suite: 500 multi-boundary episodes (≥3 boundaries each), 100 single-boundary episodes, boundary contexts discretized into 64 cells by (object-class × receptacle × lighting-bin) | Exercises trace composition across several routers |
| Primary metric | Trace success rate on multi-boundary traces | Directly measures composed-estimate claim |
| Baseline | REA per-boundary lookup router | Tests whether composition beats single-boundary comparison |
| Baseline | Flat max-success router | Tests whether ⊗ product beats naive max aggregation |
| Baseline | Learned step-value router | Tests whether fixed algebra beats learned value function |
| Baseline | Probabilistic chain-rule (HMM) router: composes boundary success via conditional probabilities $s(p_i \mid p_{i-1})$ estimated from the same route memory | Isolates what the semiring framing (⊥ absorption, margin gate) buys beyond conditional composition |
| Ablation-of-ours | CSEA with ⊗ replaced by max | Isolates the independent-failure product contribution |
| Ablation-of-ours | CSEA with correlation-aware fallback disabled (τ_ρ = ∞) | Isolates the correlation-aware collapse contribution |

Minimum per-cell sample count for a non-⊥ estimate: 20 observed boundary outcomes per route-memory cell; cells with fewer than 20 samples return ⊥.

### Why this setup validates the claim

The central claim is that folding per-boundary success elements through a semiring homomorphism lets the router compare whole traces, which a per-boundary lookup table cannot do. The REA lookup baseline tests exactly this: it can only rank policies at the current boundary, so any gain we show on multi-boundary traces isolates the value of composition. The flat max-success baseline tests the second sub-claim, that the independent-failure product captures accumulated risk better than a max aggregation that ignores how many boundaries remain. The learned step-value baseline tests whether a fixed, auditable algebra is competitive with a learned value function, which is the claim that boundary statistics composed by a fixed operator suffice. The probabilistic chain-rule (HMM) baseline tests whether the semiring framing adds anything beyond conditional-probability composition: it composes boundary success via $s(p_i \mid p_{i-1})$ estimated from the same route memory, so any gain of CSEA over it isolates the ⊥ absorption and the algebraically defined margin gate rather than the act of composing at all. The ablation replacing ⊗ with max isolates the product operator itself, and the ablation disabling the correlation-aware fallback isolates the contribution of the explicit correlation check that operationalizes assumption A. Trace success rate on multi-boundary episodes is the right metric because the predicted effect is specifically about composed estimates over traces; a per-boundary metric would be blind to the composition claim and could not distinguish CSEA from REA.

### Expected outcome and causal chain

**vs. REA per-boundary lookup router** — On a case where a trace has three boundaries and the locally best policy at boundary one has a poor downstream success profile, the REA router picks it because its lookup table has no operator to fold downstream estimates into the current comparison. Our method instead composes the prefix element with each candidate's boundary-local element and selects the policy whose composed product is highest, because the homomorphism carries downstream risk into the current decision. We expect a noticeable gap on traces with three or more boundaries but parity on single-boundary traces, where composition degenerates to the lookup comparison.

**vs. Flat max-success router** — On a case where each of four boundaries has a policy with 0.8 local success but the policies fail on correlated downstream conditions, the flat max router reports 0.8 and proceeds confidently because max ignores accumulation across boundaries. Our method instead multiplies the local elements into a composed estimate near 0.41, because the independent-failure product monotonically decreases with trace length, and the reduced estimate triggers the margin gate. We expect fewer silent trace failures on long traces and a higher rate of brain consultation exactly where trace length exceeds three boundaries.

**vs. Learned step-value router** — On a case where the boundary context is novel and the learned value function has sparse coverage, the learned router produces an overconfident step value because its generalization is unconstrained by measured boundary statistics. Our method instead returns ⊥ for the unmeasured cell (fewer than 20 samples) and routes to the brain judge, because unmeasurable boundaries absorb through ⊗ and force the gate. We expect the learned baseline to show higher variance in trace success on novel boundary contexts and our method to show a higher judge-consultation rate there, with comparable success on familiar contexts.

**vs. Probabilistic chain-rule (HMM) router** — On a case where the boundary context is unmeasured (fewer than 20 samples), the chain-rule router must still emit a conditional probability from its prior, so it proceeds without flagging unmeasurability. Our method instead returns ⊥ and forces the brain gate, because ⊥ is an absorbing element in the semiring. We expect CSEA to match the chain-rule router on familiar contexts but to show a higher judge-consultation rate and fewer silent failures on unmeasured cells, isolating the value of ⊥ absorption and the algebraically defined margin gate.

**vs. CSEA with ⊗ replaced by max (ablation)** — On a case where two boundaries each have moderate local success, the max-ablation reports the larger element and underweights the second boundary's failure contribution, because max is idempotent and discards accumulation. Our method instead multiplies the elements and yields a strictly lower composed estimate, because ⊗ is not idempotent and preserves monotone failure risk. We expect the ablation to match full CSEA on single-boundary traces but trail it on multi-boundary traces, with the gap concentrated on traces whose boundaries have similar local success.

**vs. CSEA with correlation-aware fallback disabled (τ_ρ = ∞)** — On a case where a silent failure (e.g., dropped object) corrupts two downstream boundaries, the disabled-fallback variant multiplies through and reports an optimistic composed estimate, because it treats the correlated failures as independent. Our method instead detects |ρ| > 0.3 for the adjacent pair and collapses the trace to ⊥, forcing the brain gate. We expect the disabled variant to show more silent trace failures on episodes containing a silent-failure signature, with the gap concentrated on traces where measured cross-boundary correlation exceeds 0.3.

### What would falsify this idea

If CSEA's gain over the REA lookup router were uniform across trace lengths rather than concentrated on multi-boundary traces, the composition claim would be wrong and the gain would be attributable to something other than the semiring homomorphism. Likewise, if the max-ablation matched full CSEA on multi-boundary traces, the independent-failure product would not be doing the work we claim; and if the correlation-aware fallback ablation matched full CSEA on traces with measured |ρ| > 0.3, the explicit correlation check would not be doing the work we claim.

## References

1. REA: A Routed Embodied Agent Runtime for Asynchronous Brain-Cerebellum Control
