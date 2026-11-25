# Compositional Error-Bounded Safety Gating for Modular Medical Language Models

## Motivation

Modular medical AI systems such as KMed-31B and VILA-M3 assemble clinical capability by stacking specialist teachers or domain expert models, but neither establishes a compositional safety invariant: KMed-31B distills four category specialists into one student via MOPD^2 with no declared per-specialist error contract, and VILA-M3 injects domain expert knowledge additively, so a composed answer's error is treated as if it were the sum of independently valid module outputs rather than a propagated quantity. Structurally, this fails because the composition operator is unconstrained addition — there is no rule that forces a module's declared error bound to enter the downstream confidence of the composed prediction, so an overconfident specialist silently dominates the fused output and no mechanism can reject or abstain at the point where propagated error crosses a clinical tolerance.

## Key Insight

If each specialist module carries a declared, empirically calibrated error bound on its own output distribution and the fusion operator propagates those bounds through an interval-arithmetic composition rule, then the composed confidence interval is a conservative envelope of the true composed error, making a threshold-based abstention rule sound rather than heuristic.

## Method

```pseudocode
# CEB-Gate: Compositional Error-Bounded Gate
# Inputs: query q, module set M = {m_1..m_k} (KMed-31B specialists /
#         VILA-M3 expert models), clinical tolerance tau (e.g., 0.05),
#         calibration split D_cal, held-out probe set D_probe
#
# LOAD-BEARING ASSUMPTION (stated explicitly):
#   Each module's empirically calibrated 95% residual quantile alpha_i is a
#   valid, transferable per-module error bound, and the RSS/interval-
#   intersection composition rule yields a conservative envelope of the
#   composed error on deployment queries. This holds only if module residuals
#   are independent (or their pairwise correlation is bounded by rho_max);
#   it fails when modules share a base model or training data (as KMed-31B's
#   MOPD^2 specialists and VILA-M3's expert models do), because then h_fused
#   reflects only the diagonal of the residual covariance and understates
#   true composed error. We therefore (i) estimate the residual-correlation
#   matrix on D_probe as a first-class diagnostic, (ii) inflate h_fused by
#   the measured correlation, and (iii) calibrate the fused interval at the
#   composition level on held-out composed predictions so the composed bound
#   is valid under arbitrary residual correlation.

PHASE 1 — Per-module error contracts (offline, per module m_i)
  for each m_i in M:
      split D_cal into K=5 folds
      for fold: fit m_i on train, predict on held fold -> residual scores
      contract_i = (alpha_i, beta_i) where
          alpha_i = empirical 95% quantile of |residual| on D_probe
          beta_i  = coverage rate of [pred - alpha_i, pred + alpha_i]
      store contract_i with m_i

PHASE 2 — Composition rule (interval arithmetic over module outputs)
  given q, obtain outputs {y_i, s_i} from each m_i, where s_i = contract_i
  # each module emits an interval I_i = [y_i - alpha_i, y_i + alpha_i]
  # fusion is a declared composition operator C (choose: weighted interval
  #   intersection with weights w_i = 1/alpha_i, normalized)
  I_fused = weighted_interval_intersection({I_i}, {w_i})
  # propagate: composed half-width = sqrt( sum_i (w_i * alpha_i)^2 )  (RSS rule)
  h_fused = sqrt( sum_i (w_i * alpha_i)^2 )
  # correlation inflation: estimate pairwise residual correlation rho_ij on
  #   D_probe; inflate h_fused by the measured correlation (rho_max parameter)
  rho_max = max_{i,j} |rho_ij| on D_probe
  h_fused = h_fused * sqrt(1 + (k-1) * rho_max)   # conservative inflation
  I_fused = [ y_fused - h_fused, y_fused + h_fused ]

PHASE 3 — Safety gate (online, per query)
  if h_fused > tau:
      ABSTAIN (route to human clinician)
  elif I_fused straddles a decision boundary for the task:
      ABSTAIN (ambiguous)
  else:
      RETURN y_fused with I_fused and coverage record

PHASE 4 — Contract audit (periodic)
  on D_probe stream: recompute beta_i; if beta_i < 0.90,
      flag module for re-contracting or removal from M
  # also recompute rho_max; if rho_max > 0.5, flag shared-backbone correlation
```

**Why this design**
We chose per-module empirical error contracts over a single learned confidence head because a learned head on top of a distilled student (as in KMed-31B's MOPD^2 output) collapses all specialist uncertainty into one scalar and cannot localize which module's error drove the composition — accepting the cost that contracts must be re-estimated whenever a module is retrained. We chose weighted interval intersection with inverse-bound weights over uniform averaging because modules with tighter declared bounds should dominate the fused interval, accepting the cost that a single very tight but miscalibrated module can shrink the interval incorrectly (mitigated by Phase 4 audit). We chose the root-sum-square propagation rule over naive addition because independent module errors should not be summed linearly (which would make h_fused grow with |M| and force near-constant abstention), accepting the cost that the RSS rule assumes module errors are approximately uncorrelated, which we verify empirically via residual correlation on D_probe and correct by the rho_max inflation factor. This is not a variant of standard ensemble uncertainty (deep ensembles, MC dropout) or of Toolformer/Gorilla-style routing: those methods either average predictions or select a module by retrieval, whereas CEB-Gate propagates declared per-module error bounds through a fixed composition operator and uses the resulting interval width — not a log-prob or a retrieval score — as the gate signal.

**Why it measures what we claim**
The composed half-width h_fused measures the propagated clinical error envelope because the RSS composition rule is equivalent to the true standard deviation of a weighted sum under the named assumption that module residuals are independent and each alpha_i is a correct 95% quantile of that module's residual distribution; this assumption fails when two specialists share training data or a common base model (as KMed-31B's specialists do, all distilled from one student), in which case h_fused understates the true composed error and reflects only the diagonal of the residual covariance — which is why we estimate rho_max on D_probe and inflate h_fused by sqrt(1 + (k-1)*rho_max). The coverage rate beta_i measures a module's contract validity because it is the empirical frequency with which the true label falls inside the declared interval under the assumption that D_probe is exchangeable with deployment queries; this fails under distribution shift (a new clinical population), in which case beta_i reflects in-distribution coverage and the gate can be overconfident on shifted inputs. The abstention threshold tau measures clinical safety tolerance because it is the maximum interval half-width a clinician accepts before deferring under the assumption that interval width is monotone in decision risk; this fails for tasks where a narrow interval still surrounds a high-stakes boundary, which is why Phase 3 adds an explicit boundary-straddle check rather than relying on width alone.

## Contribution

(1) A compositional error-bounded gating framework (CEB-Gate) that attaches declared per-module error contracts to specialist/expert modules and propagates them through a fixed interval-arithmetic composition rule, converting additive expert stacking into constrained composition. (2) A design principle: the abstention signal is the propagated interval width under a stated independence assumption, with a periodic contract-audit loop that detects when that assumption is violated. (3) A reproducible contract-calibration protocol (5-fold residual quantiles + coverage audit) applicable to both distilled-specialist and expert-injection architectures.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | HealthBench Hard + Professional, KMed-31B specialist tasks | Matches medical LMs and rubric-scored clinical inquiry |
| Primary metric | Selective accuracy at fixed abstention rate (risk-coverage) | Directly tests error-bounded gating signal |
| Baseline 1 | KMed-31B single distilled student | Tests whether per-module contracts beat one scalar |
| Baseline 2 | Uniform interval averaging over modules | Tests inverse-bound weighting sub-claim |
| Baseline 3 | Learned confidence head / MC-dropout gate | Tests declared contracts vs learned uncertainty |
| Ablation-of-ours | CEB-Gate with naive linear error addition | Tests RSS propagation sub-claim |
| Ablation-of-ours | CEB-Gate without rho_max inflation (rho_max=0) | Tests correlation-inflation sub-claim |

### Why this setup validates the claim

The central claim is that propagating declared per-module error contracts through a fixed composition operator yields a gate signal that localizes and bounds composed clinical error better than a single collapsed uncertainty scalar. The KMed-31B distilled student baseline tests the "localization" sub-claim: if a single student head already matches our selective accuracy, then per-module contracts add nothing. Uniform interval averaging tests the weighting sub-claim: if inverse-bound weighting does not beat uniform fusion, tighter modules are not dominating as claimed. The learned confidence head / MC-dropout gate tests whether declared empirical contracts outperform a learned scalar — the exact alternative our design rejects. The naive-linear-addition ablation tests the RSS propagation rule: if linear addition matches RSS, the independence assumption and its RSS consequence are not doing work. The rho_max=0 ablation tests the correlation-inflation sub-claim: if removing the inflation factor does not degrade selective accuracy, then the measured residual correlation is not driving the gate's behavior. Selective accuracy at a matched abstention rate is the right metric because the gate's entire function is to trade coverage for correctness; a metric that ignores abstention (raw accuracy) cannot detect whether the interval width is a valid risk signal, and a metric that ignores correctness (coverage alone) is trivially maximized by never abstaining. Together these form a falsifiable test: the method must win specifically on the risk-coverage frontier, and specifically where the predicted failure modes (collapsed scalar, mis-weighted fusion, correlated residuals) bite.

### Expected outcome and causal chain

**vs. KMed-31B single distilled student** — On a case where two specialists disagree (e.g., a diagnosis-prediction specialist and a clinical-inquiry specialist give conflicting reads on the same presentation), the distilled student produces one confident answer because MOPD^2 collapses all specialist signals into a single scalar with no attribution. Our method instead keeps each module's interval separate and the fused width widens when they conflict, triggering abstention. We expect a clear selective-accuracy gap concentrated on disagreement cases, with near-parity on cases where all specialists agree.

**vs. Uniform interval averaging** — On a case where one module has a tight, well-calibrated contract and another has a wide, noisy one, uniform averaging lets the noisy module dominate the fused interval, producing a width that is either too wide (spurious abstention) or mis-centered. Our method instead weights by inverse bound so the tight module dominates, yielding a sharper interval. We expect higher coverage at matched abstention, with the gain concentrated on queries where module bound widths are heterogeneous.

**vs. Learned confidence head / MC-dropout gate** — On a case where the model is confidently wrong (a known MLLM failure mode, e.g., visually similar but clinically distinct inputs), a learned head or dropout variance stays low because it reflects training-time variance, not declared residual quantiles. Our method instead gates on the empirical 95% residual quantile, which is high for that module regardless of its internal confidence. We expect the baseline's risk-coverage curve to degrade sharply at high-coverage regimes while ours degrades gracefully, with the gap largest on confidently-wrong subsets.

**vs. CEB-Gate with linear error addition (ablation)** — On a case with many modules, linear addition makes h_fused grow with |M| and forces near-constant abstention, so the gate cannot distinguish easy from hard queries. Our RSS rule keeps h_fused bounded and query-sensitive. We expect the ablation to show near-flat coverage (over-abstention) and our method to recover a usable risk-coverage frontier, with the gap widening as |M| increases.

**vs. CEB-Gate without rho_max inflation (ablation)** — On a case where specialists share a base model (as KMed-31B's MOPD^2 specialists do), residuals are positively correlated and the un-inflated h_fused understates true composed error, causing the gate to accept queries whose true error exceeds tau. Our method inflates h_fused by sqrt(1 + (k-1)*rho_max), widening the interval and triggering abstention on those correlated-error cases. We expect the rho_max=0 ablation to show higher selective accuracy at low coverage but worse selective accuracy at high coverage, with the gap concentrated on queries where the measured rho_max is large.

### What would falsify this idea

If CEB-Gate's selective-accuracy gain over the distilled-student baseline is uniform across all query subsets rather than concentrated on specialist-disagreement and confidently-wrong cases, then the claimed mechanism (localizing per-module error) is not what drives performance and the idea is wrong. Likewise, if the RSS ablation matches the full method, the propagation rule and its independence assumption are doing no work; and if the rho_max=0 ablation matches the full method, the correlation-inflation sub-claim is not doing work.

## References

1. KMed-31B Technical Report
