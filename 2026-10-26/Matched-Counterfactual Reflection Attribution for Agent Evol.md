# Matched-Counterfactual Reflection Attribution for Agent Evolution Lineages

## Motivation

Across agent-evolution frameworks, reflection is treated as a pluggable operator whose output is accepted whenever an external scalar score improves — RSIHub instantiates AHE, Hyperagents, A-Evolve, and GEPA this way, and ACE likewise curates reflection outputs by downstream performance. The structural problem is that the score delta is confounded with the trajectory sample that conditioned the reflection, so no framework can establish whether an admitted update is a validated diagnosis channel or a lucky sample; this recurs wherever reflection is pluggable but attribution is absent. Because RSIHub already records lineage and evaluation context, the missing piece is an identifiability condition, not more logging.

## Key Insight

Conditioning on a sufficient statistic of the pre-reflection state makes the reflection-applied and reflection-withheld trajectories exchangeable within a matched pair, so the within-pair score contrast isolates the reflection's causal effect from the trajectory-sampling component without any learned disentanglement head.

## Method

**What it is.** The Reflection Identifiability Operator (RIO) is a first-class RSIHub operator that sits between a reflection operator and lineage admission. Its inputs are a reflection-derived update $u$, the pre-reflection state $s$, and the recorded lineage/evaluation context; its outputs are an attributable effect estimate $\hat{\tau}$, a sampling-component estimate $\hat{\sigma}$, and an admission decision.

**How it works.**

```pseudocode
RIO(u, s, context, k=4, alpha=0.05):
  # 1. Sufficient-statistic key for the pre-reflection state
  T   = (task_family(s), tool_call_histogram(s), step_index_bucket(s, width=3),
         terminal_status(s), evaluator_config(s))
  # 2. Matched counterfactual trajectory: exact replay of the pre-reflection state s
  #    (no resampling; the counterfactual branch starts from the identical state and RNG stream)
  s_cf = exact_replay(s)   # returns s itself; residual state distance = 0 by construction
  # 3. Paired rollouts: apply u vs withhold u on the SAME matched state s_cf
  score_on,  trace_on  = evaluate(apply(u, s_cf),  context)
  score_off, trace_off = evaluate(withhold(u, s_cf), context)
  # 4. Matched-pair contrast
  delta   = score_on - score_off
  # 5. Decompose observed score delta on the original trajectory
  delta_obs = score_after(u, s) - score_before(s)
  sigma_hat = delta_obs - delta          # trajectory-sampling component
  tau_hat   = delta                       # reflection-attributable component
  # 6. Gate: admit iff the reflection effect is positive and stable across k pairs
  #    Each pair i uses an independent seed drawn from a per-pair RNG family
  #    (seed_i = hash(T, u.id, i) with i=1..k, no seed reuse across pairs).
  se    = std([delta_i for i in 1..k]) / sqrt(k)
  admit = (tau_hat > 0) and (tau_hat - 1.96*se > 0) and (abs(sigma_hat) < |delta_obs|)
  return {tau_hat, sigma_hat, admit, T}
```

Hyperparameters: $k=4$ matched pairs per update; significance threshold $\alpha=0.05$; step-index bucket width 3. The operator writes $\hat{\tau}$, $\hat{\sigma}$, and the statistic key $T$ into the lineage record so downstream operators can audit admission.

**Why this design.** We chose a *matched-pair counterfactual on an exact replay of the pre-reflection state* over a learned disentanglement head because reflection quality and trajectory quality are operationally coupled in the execution substrate, and no head can manufacture independence the data does not contain; the cost is that we must pay $2k$ extra evaluations per reflection update. We chose a *discrete sufficient-statistic key* (task family, tool histogram, step bucket, terminal status, evaluator config) over a continuous embedding match because the key must be auditable and reproducible from lineage alone, accepting that coarse keys may leave residual trajectory variation inside a pair. We chose a *hard admission gate with a one-sided significance test* over a soft score-weighted admission because lineage integrity requires a binary validated/unvalidated label, accepting that borderline updates are rejected rather than down-weighted. A domain expert would not describe this as a variant of AHE or GEPA: those methods accept reflection outputs on an external scalar score, whereas RIO intervenes on the trajectory to construct the counterfactual that those methods never construct.

**Why it measures what we claim.** The quantity $\hat{\tau} = \text{score}_{\text{on}} - \text{score}_{\text{off}}$ measures the *reflection-attributable causal effect* because the matched-pair design assumes that conditional on the sufficient statistic $T$, the resampled trajectory is exchangeable with the original trajectory; this assumption fails when $T$ omits a state variable that both determines the reflection's applicability and correlates with score (e.g., a latent user intent not captured by task family), in which case $\hat{\tau}$ reflects a mixture of reflection effect and residual trajectory-selection bias. The quantity $\hat{\sigma} = \delta_{\text{obs}} - \hat{\tau}$ measures the *trajectory-sampling component* because it assumes the observed delta decomposes additively into reflection and sampling terms; this assumption fails when reflection and trajectory interact non-additively (a reflection that only helps on easy trajectories), in which case $\hat{\sigma}$ absorbs the interaction term. The admission gate's significance test measures *validated diagnosis* because it assumes the $k$ matched pairs are independent draws from the same conditional distribution; this assumption fails when the resampler reuses the same seed family, in which case the standard error is underestimated and the gate over-admits.

## Contribution

(1) The Reflection Identifiability Operator (RIO), a first-class RSIHub operator that constructs matched counterfactual trajectory pairs conditioned on a sufficient statistic of the pre-reflection state and decomposes observed score deltas into reflection-attributable and trajectory-sampling components. (2) A lineage-level admission protocol that replaces external-score acceptance with a significance-gated identifiability condition, yielding a validated diagnosis channel whose admitted updates carry an attributable effect estimate. (3) An audit statistic (the sufficient-statistic key and its matched-pair contrast) recorded per update, enabling post-hoc inspection of which admitted reflections were sampling artifacts.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Terminal Bench 2 | Multi-step tool trajectories expose reflection/trajectory coupling |
| Primary metric | Reflection-attributable gain (τ̂) vs. observed delta | Directly separates the two components the method claims to disentangle |
| Baseline | AHE | Accepts reflection on external scalar score, no counterfactual |
| Baseline | GEPA | Score-weighted reflection acceptance, no matched-pair control |
| Baseline | A-Evolve | Lineage admission without counterfactual attribution |
| Baseline | AHE with 8 extra rollouts (cost-matched) | Isolates attribution from extra compute |
| Ablation-of-ours | RIO with coarse key (task family only) | Tests whether the sufficient-statistic key does the work |
| Ablation-of-ours | RIO with resampled-trajectory counterfactual (original design) | Tests whether exact replay is necessary |

### Why this setup validates the claim

The central claim is that reflection-attributable effect and trajectory-sampling effect are operationally coupled, and that only an interventional matched-pair construction can separate them before lineage admission. Terminal Bench 2 is the right substrate because its multi-step tool trajectories make the coupling severe: the same reflection applied on a resampled trajectory can look beneficial or harmful depending on which tool-call path was realized, so any method that reads only the external scalar delta cannot tell the two apart. AHE and GEPA are the correct baselines because they represent the two dominant acceptance logics — external-score acceptance and score-weighted acceptance — and each should fail in a distinguishable way if the coupling is real. The cost-matched AHE baseline (8 extra rollouts) controls for the extra compute RIO spends on paired evaluations, so any gain over it isolates attribution rather than re-evaluation. The ablation with a coarse key isolates the mechanism: if the full key matters, the coarse-key variant should lose most of the gain specifically on the subsets where the omitted statistics vary. The ablation with the resampled-trajectory counterfactual (original design) tests whether exact replay of the pre-reflection state is necessary for the exchangeability assumption to hold. The primary metric, τ̂ versus observed delta, is the right detector because it is exactly the quantity the method constructs; a method that merely improves average score could do so by trajectory luck, whereas a method that genuinely separates the components should show τ̂ concentrated where the coupling is predicted to be largest.

### Expected outcome and causal chain

**vs. AHE** — On a case where a reflection fires after a lucky tool-call path (e.g., a retry that happened to hit a cached endpoint), AHE sees a positive external scalar delta and admits the reflection, crediting the reflection for what was trajectory sampling. Our method instead replays the exact pre-reflection state, applies and withholds the reflection on the same matched state, and attributes only the paired contrast to the reflection. So we expect AHE to admit a cluster of updates whose τ̂ is near zero or negative, visible as a gap between admitted-update τ̂ and AHE's reported score gain, concentrated on the high-variance tool-histogram buckets.

**vs. GEPA** — On a case where a reflection helps only on easy trajectories but is neutral on hard ones, GEPA's soft score-weighted admission down-weights rather than rejects, so the reflection leaks into lineage with a diluted but nonzero weight. Our method's hard gate with a one-sided significance test rejects the update outright because the matched-pair τ̂ is not significantly positive. We expect GEPA to show a small but persistent score drag on the hard-trajectory subset, while our method shows parity on easy trajectories and a visible gap on hard ones where the interaction term is largest.

**vs. A-Evolve** — On a case where lineage admission is driven by post-hoc evaluation of the evolved agent rather than the reflection itself, A-Evolve admits updates whose apparent benefit is partly due to the resampled trajectory being easier. Our method writes τ̂, σ̂, and the statistic key into the lineage record, so the same update is tagged as sampling-dominated when |σ̂| is comparable to |δ_obs|. We expect A-Evolve to accumulate lineage entries with high σ̂ that later regress, whereas our admitted set should show a noticeably tighter τ̂ distribution and fewer post-admission reversals.

**vs. AHE with 8 extra rollouts (cost-matched)** — This baseline spends the same extra compute as RIO (2k=8 evaluations) on re-evaluating the original trajectory rather than constructing a matched counterfactual. We expect it to reduce variance in the observed delta but still confound reflection and trajectory-sampling components, so its admitted-update τ̂ distribution will be similar to AHE's and show the same gap on high-variance buckets. If RIO's gain over this baseline is negligible, the improvement comes from extra compute, not attribution.

**vs. RIO with resampled-trajectory counterfactual (original design)** — This ablation replaces exact replay with a resampled trajectory conditioned on T. We expect it to show higher variance in τ̂ and more false admissions on cases where the omitted trigger context (error message) drives reflection applicability, because the matched pair is not truly matched on the pre-reflection state. If this ablation matches full RIO, exact replay is not necessary and the original design's assumption holds.

### What would falsify this idea

If the gain over AHE and GEPA is uniform across tool-histogram buckets and step-index buckets rather than concentrated where the predicted reflection/trajectory coupling is strongest, then the matched-pair counterfactual is not doing the claimed work and the improvement is coming from something else. Equally falsifying: if the coarse-key ablation matches the full-key RIO everywhere, the sufficient-statistic key is not the mechanism, and the method reduces to an expensive re-evaluation wrapper.

## References

1. RSIHub: A Composable Framework for Agent-Evolution Methods
2. The OpenHands Software Agent SDK: A Composable and Extensible Foundation for Production Agents
3. Darwin Godel Machine: Open-Ended Evolution of Self-Improving Agents
4. Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models
