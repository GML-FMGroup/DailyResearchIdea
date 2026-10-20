# Counterfactual Harness Editing with Randomized Ablation on Replayed Traces

## Motivation

Self-HPO and similar self-evolving coding agents rely on the policy's own failure introspection to propose harness edits, treating the failure-to-harness mapping as a trusted introspection channel rather than a calibrated estimator. This shared residual gap recurs across SWE-Bench Pro, Promptbreeder, and Self-Play Fine-Tuning: the self-improvement signal is never grounded in an intervention or factorization, so an accepted edit may reduce the targeted failure mode while silently degrading unrelated behavior. Without randomized ablation, there is no identifiability of which harness component caused which performance change.

## Key Insight

A harness edit is accepted only if its causal effect on the targeted failure mode, estimated by randomized edit-vs-no-edit ablation on replayed traces with matched control cases, is statistically distinguishable from zero and non-degrading on controls — making the self-improvement signal an intervention estimate rather than an unverified introspection claim.

## Method

**A. What it is.** CHE-RA (Counterfactual Harness Editing with Randomized Ablation) takes as input the current policy π, its harness H, a failure set F surfaced during the preceding P-step, and a set of control traces C sampled from tasks the policy already solves. It outputs an accepted harness edit H' or a rejection, where acceptance is gated by a calibrated causal effect estimate τ̂ computed from randomized edit/no-edit replays. The load-bearing assumption, stated explicitly: **replaying a task under the harness with and without a candidate edit yields a well-defined, stable per-task outcome y_t(e) whose only source of variation is the edit itself, so that a paired difference-in-means over a single replay per task per arm is an unbiased estimate of the edit's causal effect on the targeted failure mode.** This assumption is false for stochastic coding agents at nonzero temperature; CHE-RA therefore replaces the single-replay estimator with a repeated-measures design (Phase 2 below) that estimates and accounts for trajectory-level variance.

**B. How it works.**

```pseudocode
Input: policy π, harness H, failure tasks F (|F|=n_f), control tasks C (|C|=n_c),
       candidate edits E = {e_1..e_k} from the H-step proposer,
       significance level α=0.05, minimum effect δ=0.02, control-tolerance ε=0.01,
       replays per (task, arm) R=5, shared seed set S={s_1..s_R},
       minimum independent replays for gate R_min=5

# Phase 1: trace replay and stratification
for each task t in F ∪ C:
    replay π under H with R draws from S, record traces τ_t^(1..R)
    label failure mode m(τ_t^(r)) ∈ {targeted, other, none} for each draw
F* = tasks in F whose majority trace fails with the targeted mode m*
C* = tasks in C whose majority trace succeeds under H  # matched controls

# Phase 2: randomized ablation per candidate edit (repeated measures)
for each edit e in E:
    for each task t in F* ∪ C*:
        flip coin b_t ~ Bernoulli(0.5)   # edit applied iff b_t=1
        for r in 1..R:
            run π under H_e = apply(H, e) if b_t=1 else H, with seed s_r
            record outcome y_t^(r)(e) ∈ {0,1} (targeted failure resolved)
    # within-task paired mean over R draws
    for each task t in F*:
        ȳ_t(e) = (1/R) Σ_r y_t^(r)(e)
    τ̂_e = mean_{t∈F*, b_t=1} ȳ_t(e) - mean_{t∈F*, b_t=0} ȳ_t(e)
    # cluster-robust standard error by task (between-replay variance)
    σ̂_e = sqrt( Var_{t∈F*}(ȳ_t(e) | b_t=1)/n_{F*,1} + Var_{t∈F*}(ȳ_t(e) | b_t=0)/n_{F*,0} )
    deg_e = mean_{t∈C*, b_t=1} ȳ_t(e) - mean_{t∈C*, b_t=0} ȳ_t(e)

# Phase 3: gate and select
accept e iff  τ̂_e / σ̂_e > z_{1-α}  AND  τ̂_e ≥ δ  AND  deg_e ≥ -ε
              AND  R ≥ R_min  AND  n_{F*,1} ≥ 2  AND  n_{F*,0} ≥ 2
if multiple accepted: pick argmax_e τ̂_e - λ·max(0, -deg_e), λ=1.0
return H' = apply(H, e*) or H unchanged
```

**C. Why this design.** We chose randomized ablation over the H-step's introspection-based mapping because introspection provides no counterfactual: it cannot distinguish an edit that fixes the targeted mode from one that merely correlates with easier tasks, accepting the cost that each candidate edit requires 2·R·(|F*|+|C*|) replays rather than a single forward pass. We chose matched control tasks over a global held-out set because controls must share the failure task's surface features (repository, language, tool sequence) to isolate the targeted mode, accepting the cost that control matching reduces the effective sample size. We chose a repeated-measures paired difference-in-means estimator with a fixed significance threshold over a learned acceptance head because the ablation design with shared seeds makes the estimator unbiased by construction and the cluster-robust variance accounts for trajectory-level noise, accepting the cost that low-frequency failure modes with fewer than ~30 traces yield wide confidence intervals and are deferred. This is not a variant of Self-HPO: Self-HPO's H-step is a generative proposer whose output is trusted; CHE-RA treats the proposer as a hypothesis generator and subjects each hypothesis to a randomized experiment, so the two systems differ in the load-bearing operation (proposal vs. intervention), not in the proposer architecture.

**D. Why it measures what we claim.** τ̂_e measures the causal effect of edit e on the targeted failure mode because randomization of b_t makes the edit independent of all task-level confounders, and the repeated-measures design with shared seeds S across arms makes the within-task paired mean ȳ_t(e) stable against trajectory-level sampling noise; this assumption fails when the policy's sampling temperature is nonzero and trace outcomes are stochastic, in which case the cluster-robust σ̂_e captures the between-replay variance and the gate τ̂_e/σ̂_e > z_{1-α} requires R ≥ R_min independent replays, so acceptance is driven by the edit rather than by trajectory noise. deg_e measures non-degradation on controls because C* contains only tasks the policy solves under H, which is the assumption that success on C* is a sufficient statistic for the behaviors the edit must not break; this assumption fails when an edit breaks a capability not exercised by C*, in which case deg_e=0 but the edit is harmful on out-of-distribution tasks. The significance gate τ̂_e/σ̂_e > z_{1-α} measures statistical identifiability of the causal effect because the paired randomized design with cluster-robust variance yields a consistent variance estimator, which is the assumption that F* tasks are exchangeable under the null; this assumption fails when the H-step proposer selects F* non-randomly (e.g., only the hardest failures), in which case the estimator reflects a biased subpopulation effect rather than the population effect.

## Contribution

["A counterfactual harness-editing protocol (CHE-RA) that replaces the H-step's failure-to-harness introspection with a randomized ablation estimator over replayed traces, producing a calibrated causal effect estimate with a significance gate and a matched-control non-degradation constraint.", "A design principle for self-evolving coding agents: the self-improvement signal should be an intervention estimate on a matched failure/control set rather than an unverified introspection claim, with the gate's power determined by the size of the targeted failure stratum.", 'An open protocol specification and replay harness that makes harness edits auditable — each accepted edit carries a τ̂, σ̂, and deg_e record that can be re-checked against new traces.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | SWE-bench Verified + held-out OOD coding suite | Real harness failures plus unseen-capability controls |
| Primary metric | Targeted-failure resolution rate on F* | Directly measures the causal effect τ̂ claims to estimate |
| Secondary metric | Independent OOD resolution rate on a held-out probe set not used in the gate | Breaks the metric-estimator circularity; measures silent degradation |
| Baseline | Self-HPO (harness-policy alternation) | Tests whether intervention beats trusted proposal |
| Baseline | Introspection-only harness editing | Tests whether counterfactual replay is load-bearing |
| Baseline | Global held-out acceptance (no matched controls) | Tests whether control matching prevents silent degradation |
| Ablation-of-ours | CHE-RA minus randomized ablation (deterministic replay) | Tests whether randomization is what buys unbiasedness |
| Ablation-of-ours | CHE-RA with R=1 (single replay per task per arm) | Tests whether repeated measures are necessary for stable τ̂ |
| Ablation-of-ours | CHE-RA with control-set size |C*| ∈ {5, 10, 20, 50} | Measures deg_e sensitivity to control-set size (control-sufficiency diagnostic) |

### Why this setup validates the claim

The central claim is that a randomized intervention on the harness, gated by a paired difference-in-means estimate with repeated measures, accepts edits that causally fix the targeted failure mode while rejecting edits that merely correlate with easier tasks. Each baseline isolates one sub-claim. Self-HPO tests whether treating the proposer as a hypothesis generator rather than an oracle yields better accepted edits — if our gain over it is absent, the load-bearing-operation argument collapses. Introspection-only editing tests whether the counterfactual replay is necessary at all: if introspection matches us, randomization adds nothing. Global held-out acceptance tests whether matched controls are required to catch silent degradation. The ablation removing randomization while keeping everything else isolates the intervention itself; the R=1 ablation isolates the repeated-measures variance reduction. Targeted-failure resolution rate on F* is the right primary metric because it is exactly the quantity τ̂ estimates; the independent OOD resolution rate on a held-out probe set not used in the gate breaks the circularity by measuring degradation on capabilities absent from C*. The control-set-size ablation measures whether C* is a sufficient statistic for the capability set the edit must preserve, turning the silent-degradation failure mode from a conceded assumption into a measured diagnostic.

### Expected outcome and causal chain

**vs. Self-HPO** — On a case where the proposer suggests an edit that co-occurs with easier task selection, Self-HPO accepts it because its H-step trusts the proposal and has no counterfactual to distinguish correlation from causation, so the harness drifts toward edits that help easy tasks while the targeted failure mode persists. Our method instead replays the edit under randomized ablation on the stratified F* with R=5 shared-seed replays, so the correlated-but-causally-inert edit yields τ̂ near zero and is rejected. We expect a noticeable gap on the hardest failure-mode subset (where proposal correlation is strongest) but rough parity on easy subsets where both methods accept the same obvious fixes.

**vs. Introspection-only harness editing** — On a case where the failure mode is a subtle tool-call ordering bug, introspection produces a plausible narrative ("the agent forgot to check the file") and proposes an edit that addresses the narrative rather than the mechanism, so the targeted failure persists at nearly the same rate. Our method instead randomizes the edit across F* replays and measures whether the targeted outcome flips, so a narrative-only edit shows τ̂ indistinguishable from zero and is rejected. We expect introspection to accept roughly as many edits as we do but with a markedly lower post-acceptance resolution rate on F*, i.e., a precision gap rather than a recall gap.

**vs. Global held-out acceptance (no matched controls)** — On a case where an edit fixes the targeted mode but breaks a capability exercised only by tasks sharing the failure task's repository and tool sequence, the global held-out set never exercises that capability, so deg_e reads zero and the harmful edit is accepted. Our method instead matches controls on surface features, so the broken capability appears in C* and deg_e goes negative past −ε, triggering rejection. We expect our method to show a visible advantage concentrated on OOD tasks that share surface features with F* but not with the global held-out set, with parity elsewhere, and the independent OOD resolution rate to confirm the advantage without circularity.

**vs. CHE-RA minus randomized ablation (deterministic replay)** — The deterministic-replay ablation removes the coin flip b_t and runs a single replay per task under H_e, so any edit that correlates with easier task selection is accepted without a counterfactual. We expect the full CHE-RA to show higher targeted-failure resolution on F* and lower silent degradation on the independent OOD probe set, with the gap concentrated on edits whose proposal is correlated with task difficulty.

**vs. CHE-RA with R=1 (single replay per task per arm)** — With R=1, the cluster-robust σ̂_e is estimated from a single draw per task, so the gate τ̂_e/σ̂_e > z_{1-α} becomes a ratio of noisy quantities and acceptance is driven by trajectory noise. We expect the full CHE-RA with R=5 to show higher precision (fewer accepted edits that fail to resolve the targeted mode on independent replays) and a narrower confidence interval on τ̂_e, with the gap largest on failure modes with high trajectory variance.

**vs. CHE-RA with control-set size |C*| ∈ {5, 10, 20, 50}** — We expect deg_e to become more negative (more sensitive) as |C*| increases, and the silent-degradation rate on the independent OOD probe set to decrease monotonically with |C*|, providing a measured diagnostic of control sufficiency. At |C*|=5 we expect deg_e to miss some harmful edits that the OOD probe set catches, quantifying the coverage gap.

### What would falsify this idea

If CHE-RA's gain over introspection-only editing is uniform across failure-mode strata rather than concentrated on the targeted-mode subset where τ̂ is estimated, the claim that randomized ablation identifies causal fixes is wrong. Equally, if the ablation without randomization matches full CHE-RA on targeted-failure resolution, or if the R=1 ablation matches full CHE-RA on precision, then randomization or repeated measures are not what produce the effect and the central mechanism is unsupported.

## References

1. Self-Evolving Coding Agents via Harness-Policy Optimization
