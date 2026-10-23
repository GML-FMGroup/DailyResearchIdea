# Reliability-Weighted Dynamic Bradley-Terry Aggregation for Shift-Invariant Robot Preference Signals

## Motivation

Distributed real-world evaluation of generalist robots, as instantiated by RoboArena, aggregates pairwise preferences into a single static Bradley-Terry ranking that treats the evaluator population and the policy set as fixed. This is structurally inadequate for recursive self-improvement: as the policy set evolves across RSI rounds, evaluator reliability drifts and site heterogeneity induces distribution shift in the preference-generating process, so the ranking signal fed back into the improvement loop is neither identifiable nor invariant. The synthesis in 'Toward Recursive Self-Improvement in Physical AI' notes that gains carry forward only when feedback is relevant and robust across task or hardware changes, a condition that a static, shift-unaware aggregation operator cannot certify.

## Key Insight

Modeling per-evaluator reliability and per-site bias as latent variables that are jointly re-estimated with a time-indexed skill trajectory turns the aggregation problem into a dynamic latent-variable estimation whose invariance to evaluator-population shift follows from a site-centering sufficient statistic, so the self-improvement signal is recovered as the evaluator- and site-invariant component of the latent trajectory rather than as a raw ranking.

## Method

**What it is.** We introduce **DRIFT-BT** (Dynamic Reliability-Invariant Filtered Tournament Bradley-Terry), an aggregation operator that consumes a stream of distributed pairwise preferences over evolving policies and outputs a per-policy skill trajectory whose round-over-round increments are the self-improvement signal. Inputs: preference tuples (evaluator e, site s, policy pair (a,b), round t, outcome y). Outputs: latent skill θ_{i,t} for each policy i at round t, evaluator reliability ρ_e, site bias β_s, a shared round effect γ_t, and the invariant increment Δθ_{i,t} used as the RSI feedback signal.

**How it works.**

```pseudocode
DRIFT-BT(preferences, rounds T, hyperparams λ_ρ=1.0, λ_β=1.0, η=0.05, κ=2.0, σ=0.1, anchor policy a0):
  initialize θ_{i,0}=0 for all policies, ρ_e=1, β_s=0, γ_0=0
  for t in 1..T:
    # (1) dynamic prior: random-walk skill transition
    θ_{i,t} ← θ_{i,t-1} + ε_i,   ε_i ~ N(0, σ²),  σ fixed at 0.1
    γ_t ← γ_{t-1} + ε_γ,   ε_γ ~ N(0, σ_γ²),  σ_γ fixed at 0.1
    # (2) site-centering sufficient statistic (with ρ_e fixed from previous round)
    for each site s:
      m_s ← mean over pairs at s of (y - 0.5)          # site-level outcome offset
      β_s ← shrink(m_s, κ)                              # empirical-Bayes shrink toward 0
    # (3) reliability re-estimation via residual consistency (with β_s fixed from step 2)
    for each evaluator e:
      r_e ← mean over e's pairs of |y - σ(θ_{a,t} - θ_{b,t} - β_s - γ_t)|
      ρ_e ← 1 / (1 + λ_ρ * r_e)                         # low residual ⇒ high reliability
    # (4) weighted dynamic BT likelihood, MAP update
    for each observed pair (e,s,a,b,t):
      p ← σ(ρ_e * (θ_{a,t} - θ_{b,t} - β_s - γ_t))
      θ_{a,t} += η * ρ_e * (y - p)
      θ_{b,t} -= η * ρ_e * (y - p)
      γ_t += η * (y - p) * (1 - ρ_e)   # shared round effect updated with residual
    # (5) invariant increment extraction relative to anchor policy a0
    for each policy i:
      Δθ_{i,t} ← (θ_{i,t} - θ_{a0,t}) - (θ_{i,t-1} - θ_{a0,t-1})
  return {θ, ρ, β, γ, Δθ}
```

**Why this design.** We chose a random-walk dynamic prior over a static Bradley-Terry fit because the policy set genuinely evolves across RSI rounds, accepting that the model must re-estimate skill every round rather than reuse a cached ranking. We chose site-centering with empirical-Bayes shrinkage (κ=2.0) over free per-site intercepts because unshrunk site biases absorb genuine skill differences when a site evaluates few policies, accepting a small bias toward the global mean in exchange for identifiability when site coverage is sparse. We chose a residual-consistency reliability score (inverse residual) over a learned reliability head because the residual is directly tied to the BT likelihood and requires no extra supervision, accepting that a systematically biased but consistent evaluator will be scored reliable — a limitation we address by coupling ρ_e to the site-centered residual rather than the raw one. We chose to model the shared round effect γ_t explicitly and to anchor the skill trajectory to a reference policy a0 because the Bradley-Terry likelihood identifies only skill differences; without an anchor, a common additive shift to all θ_{i,t} is unidentifiable and would be silently removed by cross-policy centering. By exposing γ_t and reporting Δθ relative to a0, a genuine global capability shift is reported as a shift in γ_t rather than being subtracted away. This is not a variant of static Bradley-Terry (RoboArena) nor of standard Dawid-Skene aggregation: those treat reliability and site as fixed nuisance parameters, whereas DRIFT-BT re-estimates them jointly with a time-indexed trajectory and extracts the invariant increment as the signal.

**Why it measures what we claim.** The site-centered residual (step 2) measures site heterogeneity because, under the assumption that site bias is additive and constant within a round, subtracting the site mean removes exactly the site offset; this assumption fails when site bias interacts with policy identity (e.g., a site systematically favors legged robots), in which case the residual reflects an unmodeled interaction rather than pure site bias. The inverse-residual reliability ρ_e (step 3) measures evaluator reliability because, under the assumption that reliable evaluators produce outcomes consistent with the latent BT probabilities, low residual implies high reliability; this assumption fails when an evaluator is consistently wrong in the same direction, in which case ρ_e reflects consistency rather than correctness. The invariant increment Δθ_{i,t} (step 5) measures the self-improvement signal because, under the assumption that the only round-over-round change common to all policies is evaluator-population drift, subtracting the anchor policy's trajectory removes that drift; this assumption fails when a genuine global capability shift affects all policies equally, in which case Δθ reflects a real shared gain as if it were drift — a failure mode we make explicit by modeling the shared round effect γ_t and reporting it separately, so that a genuine global capability shift appears in γ_t rather than being silently removed.

## Contribution

(1) DRIFT-BT, a dynamic latent-variable preference aggregation operator that jointly estimates per-evaluator reliability, per-site bias, and a time-indexed policy skill trajectory, with a site-centering sufficient statistic that yields an evaluator- and site-invariant increment as the RSI feedback signal. (2) A design principle: reliability should be scored on site-centered residuals rather than raw residuals, because raw-residual reliability conflates evaluator consistency with site heterogeneity and inflates the apparent reliability of site-biased evaluators. (3) An open aggregation benchmark constructed by replaying RoboArena-style distributed preference streams under controlled evaluator-population shift and site heterogeneity, with per-round invariant-increment recovery as the target metric.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Multi-site pairwise robot-policy preference logs over RSI rounds | Provides evaluator, site, round, and outcome structure the method consumes. |
| Primary metric | Rank correlation of Δθ with held-out true skill gain | Directly tests whether increment tracks real improvement. |
| Baseline | Static Bradley-Terry (RoboArena-style) | Tests whether time-indexing is necessary. |
| Baseline | Dawid-Skene fixed-reliability aggregation | Tests whether joint dynamic re-estimation matters. |
| Baseline | Unshrunk per-site intercept BT | Tests whether empirical-Bayes shrinkage is needed. |
| Baseline | DRIFT-BT without invariant increment (raw θ diff) | Isolates whether anchor-relative centering removes drift. |
| Baseline | DRIFT-BT without shared round effect γ_t (γ_t=0) | Tests whether modeling shared round effect is necessary. |

### Why this setup validates the claim

The central claim is that DRIFT-BT yields a round-over-round increment Δθ that tracks genuine per-policy self-improvement rather than evaluator-population drift or site artifacts. Each baseline targets one sub-claim. Static Bradley-Terry tests the necessity of the random-walk dynamic prior: if a frozen ranking already correlates with held-out skill gain as well as DRIFT-BT, the time-indexed trajectory is doing no work. Dawid-Skene with fixed reliability tests whether joint re-estimation of ρ_e and β_s per round matters, or whether static nuisance parameters suffice. The unshrunk site-intercept baseline tests whether the empirical-Bayes shrinkage is load-bearing when site coverage is sparse. The ablation without invariant increment isolates the anchor-relative centering by using raw θ differences, so any gain attributable to anchor-relative centering is directly measurable. The ablation without shared round effect γ_t tests whether modeling the shared round effect is necessary to separate genuine global capability shifts from evaluator-population drift. The primary metric — rank correlation of Δθ against held-out true skill gain, computed per round and averaged — is the right detector because it asks exactly whether the increment orders policies by genuine improvement, which is the signal the RSI loop consumes. A metric like raw win-rate would conflate drift with skill and cannot falsify the invariance claim.

### Expected outcome and causal chain

**vs. Static Bradley-Terry (RoboArena-style)** — On a case where the evaluator population shifts between rounds (new evaluators enter with stricter thresholds), the static baseline produces a spurious downward shift in every policy's apparent skill because it pools all rounds into one fixed ranking with no round index. Our method instead lets θ_{i,t} follow a random walk and re-estimates ρ_e and β_s each round, so evaluator drift is absorbed into nuisance parameters rather than skill, so we expect a clear rank-correlation gap on rounds with high evaluator turnover and near-parity on rounds with stable evaluator populations.

**vs. Dawid-Skene fixed-reliability aggregation** — On a case where one evaluator is reliable early but becomes noisy in later rounds, the fixed-reliability baseline produces a skill trajectory contaminated by that evaluator's later noise because its ρ_e is estimated once and never updated. Our method re-estimates ρ_e each round from the site-centered residual, down-weighting the now-noisy evaluator, so we expect DRIFT-BT to track true skill gain more closely specifically on the subset of policies most often judged by the drifting evaluator, with parity elsewhere.

**vs. Unshrunk per-site intercept BT** — On a case where a site evaluates only two or three policies, the unshrunk baseline produces a site intercept that absorbs those policies' genuine skill differences because the free β_s is unidentified under sparse coverage. Our method applies empirical-Bayes shrinkage (κ=2.0) toward zero, so β_s stays near the global mean and skill differences remain in θ, so we expect a noticeable rank-correlation advantage on sparse-site subsets and parity on dense-site subsets.

**vs. DRIFT-BT without invariant increment (raw θ diff)** — On a case where a genuine capability shift raises all policies together, the ablation produces raw θ differences that report that shared gain as per-policy improvement because it never subtracts the anchor policy's trajectory. Our method extracts Δθ_{i,t} relative to the anchor policy, so shared drift is suppressed and only differential improvement survives, so we expect the ablation to over-report improvement uniformly across policies while DRIFT-BT concentrates its signal on the policies that actually improved differentially.

**vs. DRIFT-BT without shared round effect γ_t (γ_t=0)** — On a case where a genuine global capability shift raises all policies together, the ablation without γ_t attributes the shared gain to the anchor policy's trajectory and thus reports zero differential improvement for all policies, because the anchor absorbs the common shift. Our method models γ_t explicitly, so the shared gain is captured in γ_t and Δθ relative to the anchor still reflects differential improvement, so we expect the ablation to show near-zero rank correlation on rounds with shared capability shifts while DRIFT-BT maintains positive correlation.

### What would falsify this idea

If the invariant-increment ablation matches DRIFT-BT's rank correlation with held-out skill gain uniformly across rounds — rather than DRIFT-BT winning specifically where evaluator turnover, sparse site coverage, or shared capability shifts are present — then the anchor-relative centering and dynamic nuisance re-estimation are not measuring what we claim, and the central invariance argument is wrong. Additionally, if the ablation without shared round effect γ_t matches DRIFT-BT on rounds with genuine global capability shifts, then modeling γ_t is not necessary to separate shared gain from drift.

## References

1. Toward Recursive Self-Improvement in Physical AI
