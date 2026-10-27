# Certified Counterfactual Replay for Harness Failure Attribution

## Motivation

Failure attribution in agent harnesses is currently performed on a single execution trace, as in TraceTrial-Bench's hand-designed mechanism chains and the Who&When dataset's LLM-based log matching, which cannot distinguish a harness component that caused a failure from one that merely co-occurred with it. The structural root cause is that a single static run is assumed to contain enough information to assign causal responsibility, even though non-determinism and cross-component coupling make co-occurrence the only observable signal. This meta-level gap recurs across expert-curated trace benchmarks and multi-agent failure logs alike: none provides an interventionist protocol that certifies when a component-level perturbation is admissible before blame is assigned.

## Key Insight

A harness component's do-perturbation is causally identifiable exactly when it passes a per-component replay-invariance certificate — the perturbed trace distribution is statistically indistinguishable from the unperturbed one on all non-target components under a fixed seed — so the certificate, not the raw replay, is what licenses counterfactual blame.

## Method

**What it is.** CARA (Certified Attribution via Replay Admissibility) takes as input a harness decomposed into the seven TraceTrial-Bench components (context, execution, governance, lifecycle, observability, tools, verification), a task suite with a downstream success statistic, and a seed-fixed replay engine. It outputs, for each failed run, a certified set of components whose surgical do-perturbation changes the success statistic, plus an admissibility certificate per component stating whether its intervention is a valid structural counterfactual.

**Load-bearing assumption (stated explicitly).** The certificate is sound only under assumption A: every cross-component coupling that could confound a do-perturbation routes through an observable marginal trace statistic of some non-target component, so that joint-only coupling channels (two components coupled purely through their joint distribution, or through a common cause such as the seed or context length that leaves each marginal unchanged) do not exist. Under A, marginal invariance on all non-target components implies joint invariance and hence do-admissibility. When A fails — the failure mode F — a perturbation passes the marginal test while leaking through a joint channel, and the certificate is inert on exactly the coupled cases it exists to flag. CARA therefore (i) tests joint invariance directly rather than only marginals, (ii) fails closed to UNIDENTIFIED whenever the joint test lacks power at the replay budget B, and (iii) reports the achieved power and a calibration curve so that 'admissible' means 'jointly invariant at the achieved power', not 'marginally invariant'.

**How it works.**

```pseudocode
Input: harness H = (C1..C7), tasks T, success stat S, seed s, replay budget B=200 per task,
       invariance alpha=0.05, effect threshold delta (grid: 0.02, 0.05, 0.10),
       power floor pi_min=0.80, permutation count P=1000
Output: certified blame set per failure, admissibility certificates A[1..7],
        achieved power pow[1..7], calibration report

# Phase 1: Reference distribution under fixed seed
R0 = { replay(H, t, s) for t in T }            # B=200 replays per task, same seed s
S0 = S(R0)                                      # downstream success statistic

# Phase 2: Per-component surgical do-perturbation
for k in 1..7:
    Rk = { replay(H with do(Ck := Ck'), t, s) for t in T }   # Ck' = component-level intervention (e.g. swap tool router, disable verifier)
    Sk = S(Rk)

    # Phase 3: Joint replay-invariance certificate (replaces per-component marginal KS)
    # Concatenate the non-target trace statistics into one vector per replay:
    #   X_i = [ stats(C_j) for j != k ]  (tool-call sequence hash, context token count,
    #          governance decision code, lifecycle transition code, observability event code,
    #          verification outcome code)
    # Joint two-sample test: permutation test on the maximum mean discrepancy (MMD)
    # between {X_i : replay in R0} and {X_i : replay in Rk}, RBF kernel bandwidth set
    # to the median pairwise distance of the pooled sample, P=1000 permutations.
    p_joint[k] = permutation_mmd_test(R0[C_{-k}], Rk[C_{-k}], P=1000)

    # Power calibration: estimate the test's power at B by splitting R0 into two
    # disjoint halves of size B/2 and running the same joint test between them;
    # the fraction of splits with p < alpha is the null-rejection rate (should be ~alpha).
    # Power is estimated by injecting a known shift of magnitude epsilon (the smallest
    # effect the experiment is designed to detect, epsilon = 0.5 SD of the pooled
    # non-target statistic) into one half and measuring the rejection rate.
    pow[k] = power_estimate(R0[C_{-k}], B, epsilon=0.5*SD, alpha=0.05, P=1000)

    # Fail-closed admissibility: admissible only if the joint test does not reject
    # AND the test has adequate power to have detected a leak of size epsilon.
    A[k] = (p_joint[k] > alpha) and (pow[k] >= pi_min)

    # Phase 4: Certified effect
    if A[k]:
        effect[k] = Sk - S0                        # counterfactual change in success stat
    else:
        effect[k] = UNIDENTIFIED                   # confounded by joint coupling or insufficient power

# Phase 5: Blame assignment
blame = { k : A[k] and |effect[k]| > delta }
return blame, A, effects, pow, calibration_report
```

The joint invariance test is the load-bearing step: it compares the empirical distribution of the concatenated non-target trace vector between the reference and perturbed replay sets using a permutation MMD test at alpha=0.05 with P=1000 permutations, and it fails closed to UNIDENTIFIED whenever the estimated power at B=200 falls below pi_min=0.80. A component whose perturbation leaks into the joint non-target trace distribution — including through a joint-only channel invisible to marginals — is marked UNIDENTIFIED rather than blamed, which is exactly the co-occurrence-vs-causation separation that single-trace attribution cannot perform. The calibration report records, for each k, the null-rejection rate (should be ~alpha) and the power curve as a function of B in {50, 100, 200} and coupling strength epsilon in {0.25, 0.5, 1.0} SD, so the operating characteristics are empirically grounded rather than assumed.

**Why this design.** We chose a joint permutation MMD test over per-component marginal KS tests because marginal invariance is strictly weaker than joint invariance: two components can be coupled purely through their joint distribution or through a common cause such as the seed while each marginal remains unchanged, so a marginal test passes exactly the coupled cases the certificate exists to flag. We chose a permutation test over a parametric test because the joint trace vector mixes discrete codes (governance decisions, lifecycle transitions) and continuous statistics (token counts), for which no single parametric two-sample test is valid, accepting the cost of P=1000 permutations per component per task (7 x 200 x 1000 = 1.4M kernel evaluations per task, feasible on CPU). We chose to fail closed to UNIDENTIFIED when power < pi_min rather than to admit on an underpowered test because an underpowered non-rejection is not evidence of invariance, accepting the cost of lower coverage on tightly coupled harnesses and at small B. We chose a fixed-seed replay engine over stochastic replay because seed-fixing is the only operational way to make the reference and perturbed distributions comparable without assuming determinism of the underlying LLM, accepting the cost that seed-fixed replay under-represents the true variance of the harness and that seed-routed coupling is invisible to any test that conditions on the same seed realization — which is why the calibration report includes a seed-variation check (re-running Phase 1 with 5 different seeds and reporting the between-seed variance of S0) so that seed-routed coupling is at least measured, even though it is not certified. We chose to mark non-admissible perturbations as UNIDENTIFIED rather than to fall back on single-trace mechanism chains (the TraceTrial-Bench approach) because the fallback reintroduces exactly the co-occurrence confound the method exists to remove, accepting the cost of lower coverage on tightly coupled harnesses. A domain expert would not describe CARA as a variant of TraceTrial-Bench's mechanism-chain attribution: TraceTrial-Bench matches one trace to a hand-designed chain, whereas CARA certifies admissibility of an intervention before any blame is computed, and the certificate — now a joint test with an explicit power floor — is the contribution, not the replay.

**Why this measures what we claim.** The joint permutation MMD p-value on the concatenated non-target trace vector C_{-k} measures replay invariance because the assumption is that under a valid do(Ck := Ck'), the joint trace distribution of every non-target component is invariant; this assumption fails when the harness has a cross-component coupling that routes through a joint-only channel (e.g., a tool router that consumes the context token count, or two components coupled through the seed), in which case the p-value reflects the coupling strength rather than the admissibility of the intervention — which is why the test is joint rather than marginal, and why the power floor pi_min=0.80 fails closed when the test cannot detect a leak of size epsilon. The certified effect Sk - S0 measures causal necessity of component Ck because the assumption is that the seed-fixed replay removes non-deterministic confounds and the joint invariance certificate removes cross-component confounds; this assumption fails when the downstream success statistic is itself seed-sensitive (e.g., a verifier with a random tie-break), in which case the effect reflects seed variance rather than component causation, and the between-seed variance check in the calibration report bounds this failure mode. The admissibility flag A[k] measures the identifiability of the do-perturbation because the assumption is that joint invariance on all non-target components at power >= pi_min is sufficient for the surgical do-operation; this assumption fails when two components are coupled only through a channel not captured by the concatenated trace statistics (an unobserved joint channel), in which case A[k] reflects invariance of the observed statistics rather than full admissibility — the calibration report's null-rejection rate (~alpha) and power curve are the empirical evidence that this residual failure mode is bounded at the stated B, alpha, and epsilon.

## Contribution

['A counterfactual replay protocol (CARA) that replaces single-trace failure attribution with per-component surgical do-perturbations under a fixed seed, producing a certified blame set rather than a mechanism-chain match.', 'A replay-invariance certificate that operationalizes the identifiability condition for harness-component interventions, marking confounded perturbations as UNIDENTIFIED instead of blaming co-occurring components.', 'An admissibility-aware attribution analysis showing which of the seven TraceTrial-Bench components are surgically perturbable in practice and which are structurally coupled, providing a design principle for modular harness construction.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | TraceTrial-Bench 95-task harness-failure suite | Has per-component traces and known failure labels |
| Dataset | Synthetic harness suite with documented coupling graph (30 harnesses: 10 loosely coupled, 10 coupling-heavy, 10 joint-only coupling) | Ground-truth causal graph lets us measure certificate false-admit rate directly |
| Dataset | Who&When multi-agent failure log (subset with per-component traces) | Independent construction tests transfer of the certified-blame advantage |
| Primary metric | Certified-blame precision at fixed coverage (coverage fixed at 60% of failures) | Penalizes blaming confounded components |
| Secondary metric | Certificate false-admit rate on synthetic joint-only coupling harnesses | Directly measures whether the joint test catches the failure mode F |
| Secondary metric | Achieved power pow[k] and null-rejection rate of the joint test at B in {50, 100, 200} | Grounds the operating characteristics empirically |
| Baseline | TraceTrial-Bench mechanism-chain attribution | Single-trace co-occurrence attribution |
| Baseline | Random component attribution | Lower bound on blame precision |
| Baseline | Who&When decisive-step attribution | Agent/step blame without admissibility |
| Ablation-of-ours | CARA with per-component marginal KS certificate (original Phase 3) | Tests whether the joint test drives precision over the marginal test |
| Ablation-of-ours | CARA without invariance certificate | Tests whether the certificate drives precision at all |
| Ablation-of-ours | CARA with joint test but no power floor (pi_min=0) | Tests whether fail-closed behavior drives precision |

Hyperparameter grid (pre-registered): B in {50, 100, 200}, alpha=0.05, delta in {0.02, 0.05, 0.10}, pi_min=0.80, P=1000 permutations, epsilon in {0.25, 0.5, 1.0} SD for power calibration. All results reported at B=200, delta=0.05 unless stated.

### Why this setup validates the claim

The central claim is that certifying intervention admissibility before computing blame separates causation from co-occurrence, and the load-bearing assumption is that joint invariance on the non-target trace vector at adequate power is a sound certificate. The synthetic harness suite with documented coupling graphs is the direct test of that assumption: on the 10 joint-only coupling harnesses, a marginal KS certificate should pass the coupled perturbation (false admit) while the joint permutation MMD test should reject it, so the false-admit rate on that subset measures exactly whether the joint test closes the failure mode F the marginal test leaves open. The power calibration at B in {50, 100, 200} grounds the operating characteristics: if the joint test's null-rejection rate is ~alpha and its power at epsilon=0.5 SD exceeds pi_min=0.80 at B=200, then 'admissible' means 'jointly invariant at the achieved power' rather than 'marginally invariant'. TraceTrial-Bench mechanism-chain attribution tests the first sub-claim on real traces: on traces where the failing component co-occurs with a downstream symptom but is not causally necessary, chain matching should blame the co-occurring component, while CARA should mark the coupled perturbation UNIDENTIFIED and blame the true driver. Random attribution calibrates the floor. Who&When tests transfer to an independently constructed multi-agent log. The marginal-KS ablation is the load-bearing control for the repair: it isolates the joint test as the source of any precision gain on coupling-heavy and joint-only subsets, while the no-certificate ablation isolates the certificate itself and the no-power-floor ablation isolates the fail-closed behavior. Certified-blame precision at fixed coverage (60%) is the right metric because CARA deliberately trades coverage for correctness on coupled harnesses; a recall-oriented metric would reward the very fallback behavior the design rejects, whereas precision at matched coverage directly measures whether certified blame is trustworthy.

### Expected outcome and causal chain

**vs. TraceTrial-Bench mechanism-chain attribution** — On a case where a tool-router failure co-occurs with a context-token spike but the router merely consumes context length, the baseline matches the trace to a hand-designed chain and blames the router because co-occurrence is its only signal. Our method instead perturbs the router and observes that the joint non-target trace vector shifts (the context token count and the tool-call sequence co-vary), marking it UNIDENTIFIED because the joint test detects the coupling. We expect a noticeable precision gap concentrated on coupling-heavy and joint-only subsets, with near-parity on loosely coupled subsets.

**vs. Random component attribution** — On a case where a verifier tie-break introduces seed-sensitive success variance, random attribution sometimes lands on the true component by chance and sometimes does not, with no mechanism to distinguish seed variance from causation. Our method's joint invariance certificate does not catch this, so on seed-sensitive cases CARA should behave close to random, but on deterministic-failure cases its certified effect should isolate the culprit cleanly. We expect CARA to sit far above random overall, with the gap collapsing specifically on seed-sensitive tasks, which is the honest signature of the stated assumption boundary.

**vs. Who&When decisive-step attribution** — On a case where a governance decision inside one agent step is correct in isolation but fails only because an earlier lifecycle transition changed the available tools, the baseline blames the decisive step inside the governance agent because it localizes by step position. Our method instead perturbs the lifecycle component, certifies joint admissibility across non-target components at power >= 0.80, and attributes to the lifecycle transition because the do-operation changes the success statistic. We expect a clear precision advantage on multi-component coupling cases and parity on single-component failures, and we expect the advantage to replicate on the Who&When subset if the certified-blame mechanism transfers beyond TraceTrial-Bench.

**vs. CARA with per-component marginal KS certificate (ablation)** — On the 10 joint-only coupling synthetic harnesses, the marginal KS certificate should pass the coupled perturbation because each marginal is unchanged, yielding a false admit and a wrong blame, while the joint permutation MMD test should reject and mark UNIDENTIFIED. We expect the marginal-KS ablation's false-admit rate on the joint-only subset to be substantially higher than full CARA's, and its certified-blame precision on that subset to be correspondingly lower, with near-parity on loosely coupled subsets where marginals and joint agree.

**vs. CARA without invariance certificate (ablation)** — Stripping the certificate leaves the same replay engine and effect threshold, so any precision gain of full CARA over this ablation must come from the certificate itself rather than from replay or seed-fixing. We expect the no-certificate ablation to blame co-occurring components on coupling-heavy and joint-only subsets, with a precision gap that scales with coupling strength.

**vs. CARA with joint test but no power floor (ablation)** — Removing the pi_min=0.80 floor admits underpowered non-rejections as admissible, so at B=50 the no-power-floor variant should false-admit coupled perturbations that full CARA marks UNIDENTIFIED. We expect the gap to be largest at B=50 and to shrink at B=200, which is the empirical signature that the power floor is doing the fail-closed work.

### What would falsify this idea

If the joint permutation MMD certificate's false-admit rate on the joint-only coupling synthetic harnesses is not substantially lower than the marginal KS certificate's, the joint test is not closing the failure mode F and the central claim fails. Equally, if full CARA's precision gain over the marginal-KS ablation is uniform across coupling-heavy and loosely coupled subsets rather than concentrated where joint coupling occurs, or if the no-certificate ablation matches full CARA's precision, the certificate is inert and the central claim fails.

## References

1. Trial by Trace: Benchmarking AI Agent Harnesses for Reliability and Failure Attribution
2. OpenHands: An Open Platform for AI Software Developers as Generalist Agents
3. SWE-Bench Pro: Can AI Agents Solve Long-Horizon Software Engineering Tasks?
4. Which Agent Causes Task Failures and When? On Automated Failure Attribution of LLM Multi-Agent Systems
5. OpenAgents: An Open Platform for Language Agents in the Wild
6. SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering
7. BigCodeBench: Benchmarking Code Generation with Diverse Function Calls and Complex Instructions
8. SWE-Bench+: Enhanced Coding Benchmark for LLMs
