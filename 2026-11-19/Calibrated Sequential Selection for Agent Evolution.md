# Calibrated Sequential Selection for Agent Evolution

## Motivation

Agent evolution loops currently accept candidate artifacts by comparing raw evaluation scores to a threshold, but The Resolution Hypothesis (E1, E7) shows selection is no better than random at distinguishing helpful from harmful candidates, and method-level margins fall below run-to-run variance. The root cause is structural: a scalar score is treated as a sufficient statistic for latent utility, so evaluator noise is never modeled and the accept/reject boundary is never calibrated to a false-accept rate. This makes the selection operator a noisy threshold on an uncalibrated signal, so errors accumulate across evolution rounds regardless of proposal quality.

## Key Insight

Repeated evaluations of the same artifact yield a directly estimable evaluator-variance term, and once that variance is known a SPRT on the score difference controls the false-accept probability at a nominal level, so the accept/reject decision tracks latent utility rather than a single noisy score comparison.

## Method

**What it is.** Calibrated Sequential Selection (CSS) is a drop-in replacement for the scalar score-threshold acceptance kernel in an agent evolution loop. Inputs: a current artifact A_cur with its cached evaluation scores, a candidate artifact A_new, and a black-box evaluator E that returns a scalar score per evaluation call. Outputs: an ACCEPT/REJECT decision for A_new plus the posterior odds and the realized false-accept control level.

**Load-bearing assumption (stated explicitly).** CSS assumes that repeated evaluations of a fixed artifact are i.i.d. draws around a stable latent mean within a single decision round, so that the pooled sample variance sigma2_pool estimated from n_min=3 warm-up calls is an unbiased, artifact-transferable estimate of the noise governing the score-difference process the SPRT is testing. This is assumption A: evaluator variance is stationary within a single artifact's evaluation sequence. The failure mode F is evaluator non-stationarity (temperature drift, context-window effects, position/verbosity bias), under which the warm-up variance underestimates the true noise and realized false accepts inflate above alpha. CSS includes a stationarity diagnostic (below) that tests A before trusting the SPRT boundary; if the diagnostic rejects stationarity, CSS falls back to the posterior-odds decision at n_max rather than the SPRT boundary, and flags the round as uncalibrated.

**How it works.**

```pseudocode
# Hyperparameters: alpha=0.05 (nominal false-accept), beta=0.10 (false-reject),
#   delta=0.5*sigma_pool (indifference margin), n_min=3, n_max=20, B=200 bootstrap reps,
#   stationarity_alpha=0.05 (Levene/runs test level), audit_K=50 rounds, audit_SE=2,
#   audit_inflation=1.5

function CSS(A_cur, A_new, E, alpha, beta, delta, n_min, n_max):
    # Phase 1: estimate per-artifact evaluator variance
    S_cur = [E(A_cur) for i in 1..n_min]          # repeated evals of current
    S_new = [E(A_new) for i in 1..n_min]          # repeated evals of candidate
    sigma2_cur = sample_var(S_cur)
    sigma2_new = sample_var(S_new)
    sigma2_pool = (sigma2_cur + sigma2_new) / 2
    sigma_pool  = sqrt(sigma2_pool)
    delta       = 0.5 * sigma_pool                 # indifference margin from noise

    # Phase 1b: stationarity diagnostic (tests assumption A)
    # Runs test on the concatenated sequence S_cur ++ S_new (length 2*n_min)
    # and Levene test comparing variance of first half vs second half of each sequence.
    # If either test rejects at stationarity_alpha, set stationary = False.
    stationary = (runs_test(S_cur ++ S_new) p > stationarity_alpha)
                 and (levene_test(S_cur[1..n_min/2], S_cur[n_min/2+1..n_min]) p > stationarity_alpha)
                 and (levene_test(S_new[1..n_min/2], S_new[n_min/2+1..n_min]) p > stationarity_alpha)

    # Phase 2: SPRT on the mean score difference (only if stationary)
    log_LR = 0
    n = n_min
    while n <= n_max and stationary:
        d_n = mean(S_new[1..n]) - mean(S_cur[1..n])
        # Gaussian SPRT log-likelihood ratio with known variance sigma2_pool
        log_LR = (n * delta / sigma2_pool) * d_n - (n * delta^2) / (2 * sigma2_pool)
        A_bound = log((1 - beta) / alpha)          # accept boundary
        R_bound = log(beta / (1 - alpha))          # reject boundary
        if log_LR >= A_bound:  return ACCEPT, log_LR, alpha
        if log_LR <= R_bound:  return REJECT, log_LR, alpha
        # else continue: draw one more eval for each artifact
        S_new.append(E(A_new)); S_cur.append(E(A_cur))
        n += 1
        # re-check stationarity every 5 draws
        if n % 5 == 0 and not stationarity_check(S_cur, S_new, stationarity_alpha):
            stationary = False
    # n_max reached without crossing, or stationarity rejected: fall back to posterior-odds decision
    # Posterior odds under a N(0, tau^2) prior on the true difference, tau = sigma_pool
    post_odds = posterior_odds(d_n, n, sigma2_pool, tau=sigma_pool)
    return (ACCEPT if post_odds > 1 else REJECT), log_LR, alpha

function posterior_odds(d_n, n, sigma2, tau):
    # conjugate Gaussian: prior N(0, tau^2), likelihood N(d_n | mu, sigma2/n)
    prec_post = 1/tau^2 + n/sigma2
    mu_post   = (n/sigma2) * d_n / prec_post
    var_post  = 1 / prec_post
    # odds that mu > 0 under posterior
    return (1 - Phi(-mu_post / sqrt(var_post))) / Phi(-mu_post / sqrt(var_post))
```

A per-round calibration audit runs every K=50 evolution rounds on a fixed held-out set of 200 artifact pairs with independent ground-truth ordering (human-annotated or held-out stronger-evaluator orderings, not the evaluator's own n_max mean). CSS's realized false-accept rate on this set is compared to alpha; if it deviates by more than 2 standard errors (audit_SE=2), alpha is inflated by a factor of 1.5 (audit_inflation=1.5) and the audit repeats. This keeps the nominal level honest under evaluator drift and breaks the circularity of using the evaluator's own mean as pseudo-ground-truth.

**Why this design.** We chose a SPRT over a fixed-n t-test because evolution loops have a natural asymmetry: most candidates are clearly worse, so early rejection saves evaluation budget, accepting the cost that borderline candidates consume up to n_max=20 evaluations. We chose to estimate sigma2_pool from repeated evaluations of both artifacts rather than from a global evaluator-variance prior because evaluator noise is artifact-dependent (some prompts are more variable than others), accepting the cost that n_min=3 warm-up evaluations are spent before any decision. We chose the indifference margin delta = 0.5*sigma_pool rather than a fixed delta because a fixed margin is either too permissive when noise is high (defeating the false-accept control) or too strict when noise is low (rejecting real improvements), accepting the cost that the effective effect size being tested shifts with evaluator noise. We chose a Gaussian posterior-odds fallback at n_max rather than a forced reject because a forced reject at the budget boundary systematically penalizes candidates that are genuinely better but noisy, accepting the cost that the fallback is less calibrated than the SPRT boundary. We added the stationarity diagnostic because the SPRT's alpha/beta guarantees are void if assumption A fails; the diagnostic costs 2*n_min extra evaluations and a runs/Levene test, accepting the cost that a false stationarity rejection triggers the less-calibrated fallback. A domain expert would not describe CSS as a variant of LLM-as-a-Judge (E10) or Agent-as-a-Judge: those replace the evaluator with a stronger agent but still compare raw scores; CSS keeps the evaluator black-box and changes the decision rule, so the two are orthogonal and composable.

**Why it measures what we claim.** The computational quantity sigma2_pool (pooled sample variance of repeated evaluations) measures evaluator noise because the assumption that repeated evaluations of a fixed artifact are i.i.d. draws around a latent mean makes sample variance an unbiased estimator of that noise; this assumption fails when the evaluator is non-stationary across calls (e.g., temperature drift or context-window effects), in which case sigma2_pool reflects drift-plus-noise and the SPRT boundaries become conservative rather than nominal, which the stationarity diagnostic detects and the calibration audit corrects. The computational quantity log_LR (Gaussian SPRT log-likelihood ratio) measures evidence that the candidate's latent utility exceeds the current artifact's by at least delta because the assumption that score differences are Gaussian with variance sigma2_pool makes the log-likelihood ratio a sufficient statistic for the two hypotheses; this assumption fails when score differences are heavy-tailed or bounded (e.g., binary pass/fail evaluators), in which case log_LR reflects evidence under a misspecified likelihood and the realized false-accept rate diverges from alpha, which the per-round calibration audit detects. The computational quantity post_odds (posterior odds under a N(0, tau^2) prior) measures the probability that the candidate is better because the assumption that the prior on the true difference is centered at zero with scale sigma_pool makes posterior mass above zero equal to that probability; this assumption fails when the evolution loop has a strong directional prior (e.g., later rounds systematically propose worse artifacts), in which case post_odds reflects a mis-centered posterior and the fallback decision inherits that bias.

## Contribution

['A variance-aware calibrated selection operator (CSS) that replaces the scalar score-threshold acceptance kernel in agent evolution loops with a SPRT on repeated evaluations, controlling the false-accept rate at a nominal level and outputting a posterior-odds decision when the sequential budget is exhausted.', 'A per-round calibration audit that detects evaluator drift and inflates the nominal level, making the false-accept control robust to non-stationary evaluators without requiring a stationary-evaluator assumption.', 'An empirical design principle: the indifference margin must scale with estimated evaluator noise (delta = 0.5*sigma_pool) rather than being fixed, because a fixed margin either defeats false-accept control under high noise or over-rejects genuine improvements under low noise.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Agent evolution loop over seven evolution methods, six LLMs | Directly matches the resolution-hypothesis setting |
| Primary metric | Realized false-accept rate on ground-truth-ordered artifact pairs (200 pairs, human-annotated or held-out stronger-evaluator orderings) | Directly measures whether selection beats random |
| Secondary metric | End-task performance after K=10 evolution rounds | Tests loop-level significance, not just pair-level |
| Baseline | Random selection at candidate level | Reproduces the reported selection-is-random failure |
| Baseline | Fixed-n t-test acceptance kernel | Tests whether sequential stopping alone explains gains |
| Baseline | Scalar score-threshold acceptance kernel | The exact kernel CSS replaces in the loop |
| Baseline | Bandit-based best-arm identification (successive halving over artifacts) | Tests whether CSS is subsumed by noise-aware selection |
| Ablation-of-ours | CSS with fixed delta instead of 0.5*sigma_pool | Isolates the noise-adaptive indifference margin |
| Ablation-of-ours | CSS without stationarity diagnostic | Isolates the contribution of assumption-A testing |

### Why this setup validates the claim

The central claim is that agent evolution stalls because noisy evaluation cannot reliably identify improvements, so a decision rule that explicitly controls false accepts against evaluator noise should recover real improvements that raw score-thresholding misses. The dataset choice is the strongest possible test because the resolution-hypothesis work already established that candidate-level selection is no better than random across seven methods and six models; if CSS is genuinely fixing the bottleneck, that is exactly where the gap must appear. Random selection establishes the floor and confirms the reported pathology reproduces. The fixed-n t-test baseline tests the sub-claim that sequential stopping and variance estimation matter, not just repeated evaluation. The scalar score-threshold baseline is the drop-in comparison: it isolates the decision rule as the causal lever, since the evaluator is identical. The bandit-based best-arm identification baseline tests whether CSS is subsumed by existing noise-aware selection methods. The fixed-delta ablation tests the sub-claim that the noise-adaptive margin is load-bearing rather than incidental. The no-stationarity-diagnostic ablation tests whether assumption A is load-bearing. The primary metric is the right one because it directly operationalizes the failure mode — if selection cannot distinguish helpful from harmful candidates, realized false-accept rate on known-order pairs will exceed alpha, and that is the quantity CSS claims to hold. The secondary metric (end-task performance after K=10 rounds) demonstrates significance at the loop level where the community cares.

### Expected outcome and causal chain

**vs. Random selection at candidate level** — On a case where a candidate artifact is genuinely better but only by a margin near the evaluator's noise floor, random selection accepts at chance because it has no evidence accumulation at all. Our method instead draws repeated evaluations and accumulates a Gaussian SPRT log-likelihood ratio, so evidence that the mean difference exceeds delta builds monotonically with each draw and crosses the accept boundary. We expect CSS to show a clear gap over random on the subset of pairs whose true difference is nonzero but small relative to sigma_pool, with parity on pairs where the difference is far above noise (both accept) and far below (both reject).

**vs. Fixed-n t-test acceptance kernel** — On a case where most candidates are clearly worse, the fixed-n t-test spends n_max evaluations on every candidate because it cannot stop early, and its threshold is calibrated to a single n, so it over-accepts borderline candidates when the effective sample size is small. Our method instead rejects early via the SPRT lower boundary when the log-likelihood ratio falls below beta/(1-alpha), saving budget on obvious losers, and its boundaries are calibrated to the pooled variance so borderline candidates are held to the nominal alpha. We expect CSS to match the t-test on realized false-accept rate while consuming noticeably fewer evaluations per rejected candidate, with the gap concentrated on the early-rejection subset.

**vs. Scalar score-threshold acceptance kernel** — On a case where evaluator noise is artifact-dependent (a high-variance prompt versus a low-variance one), the scalar threshold accepts a noisy candidate whose single score happens to exceed the current artifact's cached score, because it treats one draw as the latent mean and has no false-accept control. Our method instead estimates sigma2_pool from repeated evaluations of both artifacts, sets delta from that noise, and requires the SPRT to accumulate evidence before accepting, so a lucky high draw on a noisy artifact does not cross the boundary. We expect CSS to show a marked reduction in realized false-accept rate on the high-variance artifact subset, with near-parity on the low-variance subset where the threshold's error is small.

**vs. Bandit-based best-arm identification (successive halving over artifacts)** — On a case where many candidates are clearly worse and a few are borderline, successive halving allocates evaluations to promising arms but its stopping rule is calibrated to regret, not to a false-accept rate, so it may accept a borderline candidate whose early draws were lucky. Our method instead controls false accepts at nominal alpha via the SPRT boundaries and the calibration audit, so we expect CSS to match or exceed successive halving on realized false-accept rate while consuming comparable evaluation budget, with the gap concentrated on the borderline-difference subset.

**vs. CSS with fixed delta ablation** — On a case where evaluator noise is high, a fixed delta is too permissive relative to sigma_pool, so the SPRT accepts candidates whose true gain is within noise, inflating false accepts exactly where control matters. Our method instead scales delta to 0.5*sigma_pool, keeping the tested effect size commensurate with the noise. We expect the ablation to show elevated false-accept rate concentrated on the high-noise subset, confirming the adaptive margin is the load-bearing component.

**vs. CSS without stationarity diagnostic** — On a case where evaluator drift is present (temperature drift or context-window effects), the no-diagnostic variant trusts the warm-up variance and the SPRT boundary, so realized false accepts inflate above alpha. Our method instead runs the runs/Levene diagnostic and falls back to posterior-odds when stationarity is rejected, so we expect the full CSS to hold realized false-accept rate closer to alpha on the drifting subset, with parity on the stationary subset.

### What would falsify this idea

If CSS's realized false-accept rate on ground-truth-ordered pairs is statistically indistinguishable from the scalar score-threshold kernel's across all noise subsets — and its gain over random is uniform rather than concentrated on the high-variance, borderline-difference subset where the predicted failure mode lives — then the claim that the decision rule, not the evaluator, is the bottleneck would be wrong. Equivalently, if the fixed-delta ablation matches full CSS everywhere, the noise-adaptive margin is not doing the causal work we claim.

## References

1. The Resolution Hypothesis: Agent Evolution's Key Is Identifying Improvements
