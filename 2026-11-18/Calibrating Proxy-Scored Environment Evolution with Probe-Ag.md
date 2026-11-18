# Calibrating Proxy-Scored Environment Evolution with Probe-Agent Rank Consistency

## Motivation

Automated environment factories for coding agents — RepoFactory's Environment Evolution Loop and SetUpAgent's dependency-setup pipeline — both optimize or select environments using hand-designed proxies (RepoFactory's multiplicative objective of validity, structural coupling, and implementation scope; SetUpAgent's historically-accurate execution setup) without ever establishing that these proxies are monotone in the quantity that actually matters: downstream agent success. The structural root cause is that the proxy is a construction heuristic evaluated only for environment *usability*, so the loop can freely accept environments that are valid and complex yet pedagogically useless, and this unvalidated-proxy assumption recurs independently across both branches — evidence that it is a field-level gap rather than one pipeline's local bug.

## Key Insight

A proxy-scored environment generator is only trustworthy if its acceptance decisions are rank-monotone in measured probe-agent success, so injecting a periodically re-estimated rank-correlation residual as an acceptance gate converts the multiplicative proxy from an unverified heuristic into a signal whose ordering is empirically tied to training value.

## Method

**What it is.** CALIB-REPO (Calibrated Environment Acceptance) wraps RepoFactory's Environment Evolution Loop. Inputs: a stream of candidate environments produced by the loop's mutation operators, each carrying its multiplicative proxy score `p = validity × structural_coupling × implementation_scope`. Output: an accepted/rejected decision per candidate plus a maintained calibration model `g` mapping proxy score to predicted probe success, and a Spearman-ρ acceptance threshold τ.

**Load-bearing assumption (stated explicitly).** CALIB-REPO assumes that the frozen probe agent's success ranking over candidate environments is rank-monotone with the target model's (Qwen3.5-35B-A3B) marginal training-value ranking, so that Spearman ρ between proxy score and probe success is a valid surrogate for proxy-to-target-success ordering. This assumption is not assumed away: it is measured directly as a pre-registered diagnostic (Phase 4) and the method's claims are gated on it.

**How it works.**

```pseudocode
# Phase 1: probe bank (fixed, small ensemble spanning capability range)
probe_bank = [Qwen3.5-1.7B, Qwen3.5-7B]   # frozen weights, LoRA-free, single seed each

# Phase 2: periodic calibration sweep, every K=200 accepted environments
for round r in 1..R:
    S_r = sample M=40 candidates uniformly from current pool
    for env in S_r:
        for probe in probe_bank:
            s[probe, env] = probe_success_rate(probe, env, n_rollouts=8)   # 0..1
        s_min[env] = min_over_probes(s[probe, env])   # conservative probe signal
    rho_r = spearman(p[S_r], s_min[S_r])
    # fit calibration model: isotonic regression of s_min on p
    g_r = isotonic_fit(p[S_r], s_min[S_r])
    # residual gate: accept only candidates whose independently measured
    # probe residual is consistent with the calibrated prediction
    tau_r = max(tau_min, rho_r - delta)      # delta=0.05 slack

# Phase 3: per-candidate acceptance inside the loop
for candidate c from mutation operator:
    p_c = validity(c) * coupling(c) * scope(c)
    s_hat = g_r(p_c)                          # predicted probe success
    # independent probe rollout on c (not used to fit g_r)
    s_meas = min_over_probes(probe_success_rate(probe, c, n_rollouts=8))
    resid_c = s_meas - s_hat
    if abs(resid_c) <= eps_resid and rho_r >= rho_min:
        accept(c)
    else:
        reject(c); log resid_c for loop feedback

# Phase 4: pre-registered probe-target rank-preservation diagnostic
# (run once per domain on a held-out sample of accepted environments)
for domain d in 1..17:
    H_d = sample 20 accepted environments from domain d
    s_probe_d = rank of s_min over H_d
    s_target_d = rank of post-training target-model success over H_d
    rho_pt[d] = spearman(s_probe_d, s_target_d)
# Claims are gated on rho_pt[d] >= rho_pt_floor (0.5) for the domains reported
```

The residual `resid_c = s_meas - g_r(p_c)` is fed back to the loop's mutation operator as a per-feature adjustment signal (environments with large positive residuals are up-weighted as mutation seeds). Hyperparameters: `rho_min = 0.6`, `tau_min = 0.55`, `K = 200`, `M = 40`, `n_rollouts = 8`, `eps_resid = 0.10`, `rho_pt_floor = 0.5`.

**Why this design.** We chose a *frozen* small probe bank (Qwen3.5-1.7B and Qwen3.5-7B) over the target training model because running the full Qwen3.5-35B-A3B for probe rollouts would cost as much as the downstream evaluation the loop was designed to avoid, accepting the cost that the probes' success ranking may diverge from the 35B model's — a divergence we treat as a measurable calibration error (Phase 4) rather than assume away. We chose the *minimum* probe success over the bank as the gating signal because it is conservative with respect to capability divergence: an environment that any probe solves trivially is not credited as high training value. We chose isotonic regression over a parametric linear fit because the proxy-to-success relationship is monotone-but-nonlinear (validity saturates, scope has diminishing returns), accepting the cost that isotonic regression needs enough calibration points (M=40 per round) and cannot extrapolate beyond the observed proxy range. We chose a *residual gate on independently measured probe success* over a rank-agreement check between `p_c` and `g_r(p_c)` because `g_r` is monotone in `p` by construction, so a rank-agreement check would be near-tautological; gating on `s_meas - g_r(p_c)` uses a signal not used to fit `g_r` and can therefore disagree. We chose to feed residuals back as mutation-seed weights over discarding rejected environments because rejected-but-high-residual environments carry information about where the proxy is miscalibrated, accepting the cost of a second bookkeeping path in the loop. A domain expert would not describe this as a variant of standard active learning or of RepoFactory itself: active learning selects *labels* to reduce model uncertainty, whereas CALIB-REPO gates *environment acceptance* on a rank-consistency invariant between a construction proxy and an independently measured probe signal — the object being calibrated is the environment generator, not a predictor.

**Why it measures what we claim.** `spearman(p[S_r], s_min[S_r])` measures *proxy validity* because it is equivalent to the assumption that the multiplicative objective's ordering is monotone in probe-agent success; this assumption fails when the probe bank's capability profile differs from the target model's (e.g., the probes solve small-scope environments that the 35B model finds trivial), in which case ρ reflects probe-specific rather than target-specific training value. `g_r` (isotonic fit) measures *predicted training value* because it assumes the proxy-to-success map is monotone and stable within a round; this assumption fails when the environment distribution shifts between mutation rounds, in which case `g_r` reflects the previous round's value landscape. `rho_r >= rho_min` measures *calibration sufficiency* because it assumes a rank correlation above 0.6 is enough for ordering decisions to be reliable; this assumption fails for near-tied candidates, in which case the gate reflects noise in probe rollouts rather than genuine ordering. The residual `resid_c = s_meas - g_r(p_c)` measures *proxy miscalibration at a specific environment* because it assumes independently measured probe success is a sufficient statistic for training value; this assumption fails when probe success is high but the environment teaches a skill the target model already has, in which case the residual reflects probe capability rather than training value. `rho_pt[d]` (Phase 4) measures *probe-target rank preservation* because it assumes the probe bank's success ordering over held-out accepted environments matches the target model's post-training success ordering; this assumption fails when the probe and target have different capability profiles (e.g., the probe solves small-scope environments the target finds trivial), in which case `rho_pt[d]` is low and the method's claims are reported as domain-specific rather than field-level.

## Contribution

["A calibration-gated environment acceptance mechanism (CALIB-REPO) that replaces RepoFactory's unverified multiplicative proxy with a periodically re-estimated rank-consistency invariant between proxy score and frozen-probe-agent success, including a residual-feedback path that re-weights mutation seeds by calibration error.", 'An empirical design principle: proxy-scored environment generators should be validated by rank monotonicity against an independent probe signal rather than by construction validity alone, with a concrete protocol (K=200, M=40, ρ_min=0.6) for doing so at scale.', "A reusable calibration artifact — the isotonic proxy-to-success map and per-round Spearman ρ trace — that exposes where the multiplicative objective's components (validity, coupling, scope) are miscalibrated, enabling targeted repair of the proxy rather than wholesale replacement."]

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | RepoFactory environment stream (10,159 envs, 17 domains) | Native distribution for the evolution loop being wrapped |
| Primary metric | Spearman ρ between proxy score and held-out probe success | Directly tests the ranking invariant the gate assumes |
| Secondary metric | Probe-target rank agreement ρ_pt per domain (Phase 4) | Gates claims on the load-bearing assumption |
| Baseline | RepoFactory unmodified acceptance | Isolates the effect of the calibration gate alone |
| Baseline | Proxy-threshold acceptance (top-quantile p) | Tests whether raw multiplicative score already orders well |
| Baseline | Random acceptance from candidate pool | Lower bound on ordering signal |
| Baseline | Offline isotonic recalibration of proxy without gating | Tests whether gating adds anything beyond recalibration |
| Baseline | Uncertainty-sampled active-learning probe labeling | Standard active-learning contrast on same stream |
| Ablation-of-ours | CALIB-REPO without residual feedback to mutation seeds | Isolates the residual-feedback bookkeeping contribution |
| Ablation-of-ours | CALIB-REPO with single 1.7B probe (no ensemble min) | Isolates the probe-bank ensemble contribution |

### Why this setup validates the claim

The central claim is that a rank-consistency gate between the multiplicative construction proxy and an independently measured probe signal improves the *ordering* of environments by training value, not just their average quality. The primary metric — Spearman ρ between proxy score and held-out probe success on a fresh sample of accepted environments — is the right detector because the method's mechanism is explicitly a rank-correlation invariant; if the gate works, accepted environments should exhibit a higher proxy-to-success rank correlation than the unfiltered stream, and this is exactly what ρ measures. The unmodified RepoFactory baseline tests whether the gate adds anything beyond the loop's own validation; the proxy-threshold baseline tests whether the raw multiplicative score already orders candidates well enough that calibration is unnecessary (a strong competitor, since the proxy is monotone by construction); the random baseline establishes the lower bound and guards against the metric being trivially high. The offline isotonic recalibration baseline tests whether the *gating* step is necessary or whether recalibrating the proxy alone suffices; the active-learning baseline tests whether the gate is distinguishable from standard uncertainty sampling. The ablation without residual feedback isolates the second mechanism — using miscalibration residuals as mutation seeds — so that any gain can be attributed to the gate versus the bookkeeping path; the single-probe ablation isolates the ensemble-min contribution. The Phase 4 diagnostic directly measures the load-bearing probe-target rank-preservation assumption per domain and gates the method's claims on `rho_pt[d] >= 0.5`. Together these form a falsifiable test: the idea predicts a *specific ordering* of outcomes (ours > proxy-threshold ≈ unmodified > random on ρ, with the residual ablation landing between ours and unmodified), and a different ordering would refute the causal story.

### Expected outcome and causal chain

**vs. RepoFactory unmodified acceptance** — On a candidate whose proxy score is inflated by high `validity` but low `implementation_scope` (a small, easy environment that the proxy rewards), the unmodified loop accepts it because the multiplicative score passes the loop's own validity check, and the resulting environment stream is dominated by easy environments that the probe solves trivially. Our method instead computes `s_hat = g_r(p_c)` and `s_meas` from an independent probe rollout, and rejects the candidate when `abs(resid_c) > eps_resid`, because the isotonic fit has already learned that high-validity/low-scope candidates sit in a saturated region where proxy rank diverges from measured success. We expect a noticeable ρ gap on the low-scope subset and near-parity on high-scope environments, with the overall ρ difference driven by the low-scope tail.

**vs. Proxy-threshold acceptance (top-quantile p)** — On a candidate whose proxy score is high because `structural_coupling` is large but whose probe success is mediocre (the coupling is real but the environment teaches skills the probe already has), the proxy-threshold baseline accepts it since `p_c` sits in the top quantile, and the accepted pool accumulates environments that are structurally rich but pedagogically redundant. Our method instead routes the candidate through `g_r`, which maps the high proxy score to a predicted success that is *lower* than the proxy rank implies, and the residual check `abs(s_meas - s_hat) > eps_resid` fails, so the candidate is rejected and its positive residual is logged as a mutation seed for environments with similar coupling but different scope. We expect the ρ gap to concentrate on the high-coupling subset and to widen as rounds progress, since the residual feedback progressively steers mutation away from redundant coupling.

**vs. Random acceptance from candidate pool** — On a candidate drawn uniformly, random acceptance ignores the proxy entirely, so the accepted stream inherits the pool's natural proxy-success correlation, which is diluted by the loop's mutation operators that occasionally produce high-proxy/low-success outliers. Our method instead gates on `rho_r >= rho_min`, so any round in which the probe bank's ranking of the sampled candidates falls below 0.6 triggers rejection of the entire round's candidates, and the maintained `g_r` re-fits on the next sweep. We expect a large ρ gap versus random on the full stream, with the gap largest in domains where the mutation operators are most aggressive (where the proxy-success relationship is most nonlinear) and smallest in domains where the proxy is already near-monotone.

**vs. Offline isotonic recalibration of proxy without gating** — On a candidate whose proxy score is high but whose independently measured probe success is low, the offline recalibration baseline maps `p_c` through a fitted isotonic curve and accepts based on the recalibrated score alone, so it cannot reject a candidate whose recalibrated score is high but whose measured residual is large. Our method instead requires `abs(resid_c) <= eps_resid`, so it rejects candidates whose recalibrated prediction disagrees with an independent probe rollout. We expect our method to show higher ρ than offline recalibration on the subset of candidates where the recalibration curve is flat (near-tied proxy scores), because the residual check breaks ties using measured success rather than the monotone fit.

**vs. Uncertainty-sampled active-learning probe labeling** — On a candidate whose proxy score is near the decision boundary, the active-learning baseline selects it for probe labeling to reduce uncertainty, but it does not gate acceptance on the probe result; it merely updates the calibration model. Our method instead uses the probe result as an acceptance gate, so candidates with large residuals are rejected regardless of their uncertainty. We expect our method to show higher ρ on the accepted stream than active learning, because active learning optimizes label efficiency rather than the ordering of accepted environments, and the two objectives diverge when the proxy is miscalibrated in a region the active learner has already labeled.

**vs. CALIB-REPO without residual feedback** — On a candidate that is rejected with a large positive residual (probe success far above what `g_r` predicts), the no-feedback ablation discards it and continues mutating from the same seed distribution, so the loop never learns that its proxy systematically under-ranks a particular environment family. Our method instead up-weights that candidate as a mutation seed, so subsequent rounds generate more environments in the miscalibrated region and `g_r` re-fits with denser coverage there. We expect the residual ablation to match ours early (rounds 1–3) and diverge in later rounds, with our method showing a higher ρ on the environment families that were initially under-ranked and near-parity elsewhere.

**vs. CALIB-REPO with single 1.7B probe (no ensemble min)** — On a candidate that the 1.7B probe solves trivially but the 7B probe finds non-trivial, the single-probe ablation records high `s_meas` and accepts the candidate, whereas the ensemble-min records the 7B probe's lower success and rejects it. We expect the single-probe ablation to show lower ρ on the low-scope subset, where the 1.7B probe's capability profile diverges most from the target model's, and near-parity on high-scope environments where both probes agree.

### What would falsify this idea

If the calibration gate's gain in ρ is uniform across all environment subsets — including the low-scope, high-coupling, and initially-under-ranked families where the predicted failure modes live — rather than concentrated where the proxy-success divergence is predicted, then the gate is not doing what we claim and the central mechanism is wrong. A second falsifier: if the residual-feedback ablation matches the full method on the initially-under-ranked families, the residual bookkeeping path contributes nothing and the design's second mechanism is unjustified. A third falsifier: if the Phase 4 diagnostic yields `rho_pt[d] < 0.5` for the majority of domains, the probe-target rank-preservation assumption fails and the method's ρ gains reflect probe-specific rather than target-specific training value.

## References

1. From Repositories to Training Environments: Scaling Long-Horizon Coding Agents with Environment Evolution Loop
