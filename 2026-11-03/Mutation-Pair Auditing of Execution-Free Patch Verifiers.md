# Mutation-Pair Auditing of Execution-Free Patch Verifiers

## Motivation

Execution-free repair loops such as PegasusAgent (Execution-free Agentic Program Repair) and self-evolved harnesses (Self-Evolving Coding Agents via Harness-Policy Optimization) select patches and harnesses using surrogate verifiers—static analysis, LLM critique, self-introspection—whose correctness ordering is never checked against the executable oracle that was deliberately removed. The structural failure is that the field replaced the oracle but kept treating the surrogate's ranking as if it inherited the oracle's ordering, so a confidently wrong top-1 patch is indistinguishable from a correct one and no detection mechanism exists. This is the new bottleneck implied by the convergent move to execution-free loops: not a better surrogate, but knowing when the surrogate's ordering is trustworthy without reintroducing execution.

## Key Insight

A surrogate's correctness ordering can be audited without execution because counterfactual mutation pairs manufacture a ground-truth ordering by construction—one member provably semantics-preserving, one provably semantics-breaking—so ranking agreement on these pairs is a directly measured quantity rather than an inferred one.

## Method

**What it is.** MUTAUDIT takes an execution-free surrogate verifier V (any scorer mapping a patch to a scalar plausibility, e.g., an LLM-critique-plus-static-analysis scorer as in PegasusAgent) and a corpus of source functions, and outputs a rank-preserving audit certificate: an estimate of the probability that V's ordering of two candidate patches diverges from the true correctness ordering, plus a per-pair divergence decomposition. It never executes the mutated code.

**Load-bearing assumption (stated explicitly).** The audit's ground truth is manufactured, not executed: for each pair we *assume* the operator's side conditions hold on the concrete function so that the 'preserving' member is observationally equivalent and the 'breaking' member is incorrect. This assumption is not free — it is checked per instance (step 1b) and its residual mislabel rate is measured on a small executed calibration subset (step 1c) and propagated as a bias term. If the side-condition checker is unsound, D conflates surrogate error with mislabeled ground truth; the certificate therefore reports the drop rate and the calibration mislabel rate alongside d_hat so the reader can see how much of the estimate rests on unverified labels.

**How it works.**
```pseudocode
INPUT: surrogate V, function corpus F, mutation operators M (fixed set of 10),
       pair budget N=2000, bootstrap draws B=1000, threshold tau=0.10,
       calibration subset C (|C|=100 functions, executed once)
OUTPUT: divergence estimate d_hat, confidence band, per-operator breakdown,
        drop rate, calibration mislabel rate

1. PAIR CONSTRUCTION (per function f in F):
   for each mutation operator m in M:
     p_break  = apply(f, m)            # intended semantics-breaking by operator contract
     p_pres   = apply(f, m_preserving) # intended semantics-preserving by operator contract
     # 1b. PER-INSTANCE SIDE-CONDITION CHECK (no execution of f):
     #     discharge m's side conditions on the concrete body with a symbolic checker
     #     (aliasing, volatile access, exception ordering, concurrency, loop-carried deps).
     #     If any side condition is unprovable, DROP the pair and increment drop_count[m].
     if not side_conditions_proven(f, m, p_pres, p_break):
        drop_count[m] += 1; continue
     record (p_pres, p_break) with assumed ground-truth ordering p_pres > p_break

1c. CALIBRATION OF MISLABEL RATE (executed once on C, not on F):
   for each certified pair on f in C:
     run f, p_pres, p_break on a fixed input battery (100 inputs from f's test pool)
     label pair as mislabeled if p_pres and p_break differ on any input
   mislabel_rate[m] = mislabeled[m] / certified[m]   # per operator

2. SURROGATE SCORING (no execution):
   s_pres = V(p_pres, f); s_break = V(p_break, f)
   divergence indicator D = 1[ s_pres <= s_break ]   # surrogate inverted the known order
   margin = s_pres - s_break

3. RANK-PRESERVING ESTIMATE:
   d_hat = mean(D) over N certified pairs; bootstrap B=1000 resamples -> 95% band
   bias-corrected d_hat_corr = (d_hat - mislabel_rate_pooled) / (1 - 2*mislabel_rate_pooled)
   stratify d_hat by mutation operator (10 strata) and by function size decile

4. TRUST CERTIFICATE:
   emit (d_hat, d_hat_corr, band, operator table, drop_count per operator,
         mislabel_rate per operator)
   flag V as untrustworthy if upper band of d_hat_corr > tau (tau=0.10 default)
```
The mutation operators have construction-fixed ground truth *conditional on the per-instance side-condition check passing*: a *semantics-preserving* operator is one whose algebraic contract guarantees observational equivalence (e.g., renaming a local, reordering independent statements, adding a dead store to an unused variable), while a *semantics-breaking* operator violates a contract (e.g., off-by-one on a loop bound, swapping `<=` for `<`, dropping a null guard). Contract validity is established by symbolic reasoning over the operator template *and* discharged per instance by the side-condition checker; pairs whose side conditions cannot be proven are dropped and the drop rate is reported. The fixed operator set (10 operators) is: rename-local, reorder-independent-statements, add-dead-store, off-by-one-loop-bound, swap-le-to-lt, drop-null-guard, invert-boolean-condition, remove-early-return, change-arithmetic-operator, swap-argument-order.

**Why this design.** We chose construction-certified mutation operators over LLM-generated "plausible bug" pairs because LLM-generated breaks have no fixed ground truth and would make the audit circular (the surrogate and the pair generator could share failure modes), accepting the cost that our operator set covers only local semantic faults and not deep multi-file behavioral bugs. We chose a rank-divergence rate with bootstrap bands over a point correlation because the downstream use is *selection* (top-1 patch or harness choice), so what matters is ordering inversion, not absolute score calibration, accepting the cost that we do not diagnose magnitude errors when ordering is preserved. We chose per-operator stratification over a single pooled number because a surrogate can be reliable on renaming-style mutations and systematically inverted on guard-removal mutations, and a pooled rate would hide exactly the failure mode that matters for real patches, accepting the cost that the certificate is higher-dimensional and needs a threshold per stratum. We added the per-instance side-condition check and the executed calibration subset because the 'ground truth by construction' claim is only as good as the operator template's side conditions, and the equivalent-mutant literature shows those routinely fail under aliasing, volatility, and exception ordering; the drop rate and mislabel rate make that residual uncertainty a measured quantity rather than an assumption. A domain expert would not describe this as Promptbreeder-style self-referential prompt evolution or as a PegasusAgent static-analysis wrapper: those methods *produce* a score, whereas MUTAUDIT *certifies the ordering* of an existing score against a manufactured ground truth, which no prior execution-free repair system (PegasusAgent, Self-HPO) does at all.

**Why it measures what we claim.** The divergence indicator `D = 1[s_pres <= s_break]` measures *surrogate ordering error against the true correctness order* because the operator contract plus the per-instance side-condition check guarantee the true order is `p_pres > p_break`; this assumption fails when an operator's 'preserving' contract is itself wrong (e.g., reordering statements that are not actually independent due to aliasing), in which case D reflects a mislabeled pair rather than surrogate error — which is why we report the drop rate and the executed calibration mislabel rate and emit the bias-corrected `d_hat_corr`. The bootstrap band measures *sampling stability of the divergence estimate* because pairs are drawn i.i.d. from the operator-function cross product; this assumption fails under function-family clustering (many near-duplicate functions), in which case the band understates true uncertainty. The per-operator stratification measures *which semantic fault classes the surrogate is blind to* because each operator isolates one fault class by construction; this assumption fails when an operator's effect is confounded with code length, in which case the stratum reflects length sensitivity rather than fault-class blindness. The trust threshold `tau` measures *deployment readiness of the surrogate* because it operationalizes the tolerable selection-inversion rate for top-1 patch choice; this assumption fails when the downstream patch distribution differs sharply from the mutation distribution, in which case the certificate is a lower bound on real-world divergence.

## Contribution

(1) MUTAUDIT, an execution-free audit protocol that certifies a surrogate verifier's correctness ordering using construction-certified counterfactual mutation pairs, producing a bootstrap-banded rank-divergence estimate with per-operator decomposition. (2) A design principle: surrogate trustworthiness should be reported as an ordering-divergence certificate stratified by semantic fault class rather than as a scalar accuracy, because surrogates are reliable on some fault classes and systematically inverted on others. (3) A reusable mutation-pair corpus with algebraic operator contracts (semantics-preserving vs. semantics-breaking) that supplies ground-truth patch orderings without executing any code.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | Real-world repair patches from SWE-Bench Pro (held-out split, 200 patches) plus a function corpus of 500 source functions for pair construction | Long-horizon defects stress surrogate ordering; the function corpus supplies the mutation-pair base |
| Primary metric | Rank-divergence rate d_hat (and bias-corrected d_hat_corr) on certified pairs | Directly measures ordering inversion, not calibration |
| Secondary metric | Spearman correlation across the 10 operator strata between per-stratum d_hat and observed top-1 selection error on held-out real patches; predictive AUC of the certificate for flagging a surrogate as unsafe | Tests the claim that the certificate predicts deployment readiness |
| Baseline | PegasusAgent static-analysis scorer | Execution-free scorer our audit targets |
| Baseline | LLM-critique scorer | Common surrogate with no certified ground truth |
| Baseline | Random-ordering scorer | Lower bound on divergence |
| Baseline | Classical mutation-kill-rate adequacy score on the same pairs | Tests whether rank-divergence yields information kill-rate does not |
| Baseline | Score-correlation (Pearson between surrogate score and a binary correctness label on the same pairs) | Tests whether ordering inversion is a relabeling of correlation |
| Ablation-of-ours | Pooled d_hat without per-operator stratification | Tests whether stratification is load-bearing |
| Ablation-of-ours | d_hat without the per-instance side-condition check (template-only contracts) | Tests whether the side-condition check is load-bearing for the ground-truth claim |

### Why this setup validates the claim

The central claim is that MUTAUDIT certifies whether an execution-free surrogate's ordering of patches matches true correctness order, and that this certification predicts where the surrogate will fail on real patches. The two surrogate baselines test the core sub-claim: if MUTAUDIT's divergence estimate is meaningful, the PegasusAgent-style and LLM-critique scorers should show elevated divergence specifically on the operator strata where their mechanisms are blind, not uniformly. The random-ordering baseline anchors the metric — d_hat near 0.5 for random ordering confirms the estimator is not degenerate. The mutation-kill-rate and score-correlation baselines test the novelty sub-claim: if rank-divergence is a relabeling of existing metrics, then kill-rate and correlation should order the surrogates identically to d_hat; we report a case where kill-rate is high but top-1 ordering is inverted to establish the mechanism is distinct. The stratification ablation tests a distinct sub-claim: that pooling across operators hides the failure mode, so removing stratification should produce a reassuring pooled number that masks a high-divergence stratum. The side-condition ablation tests the load-bearing assumption: if template-only contracts inflate d_hat relative to the per-instance-checked version, the ground-truth claim depends on the check. The secondary metric — Spearman across strata between d_hat and observed top-1 selection error, plus predictive AUC — directly tests the significance claim that the certificate predicts deployment readiness; without it the claim is untested. The primary metric is rank-divergence rate rather than correlation because the downstream use is top-1 selection; a surrogate can have decent correlation yet invert the critical pair. If the predicted stratum-concentrated divergence pattern appears, tracks real-patch selection errors (Spearman > 0.5, AUC > 0.7), and the side-condition ablation shows the check matters, the claim holds; if divergence is flat, unrelated to selection behavior, or unchanged by the side-condition check, it fails.

### Expected outcome and causal chain

**vs. PegasusAgent static-analysis scorer** — On a case where a patch removes a null guard, the baseline produces a high plausibility score because static analysis sees fewer branches and no type error, missing the runtime null dereference. Our method instead flags this operator stratum as high-divergence because the certified breaking mutation isolates the dropped-guard class and the surrogate scores it above the preserving counterpart. We expect a noticeable divergence gap on guard-removal and loop-bound strata but parity on renaming strata, mirroring the surrogate's mechanism blind spots. We expect the per-stratum d_hat to correlate with observed top-1 selection error on held-out SWE-Bench Pro patches (Spearman > 0.5 across the 10 strata) and the certificate to flag this surrogate as unsafe with AUC > 0.7.

**vs. LLM-critique scorer** — On a case where an off-by-one loop bound is introduced, the baseline produces a plausible-sounding critique that rates the patch highly because the critique shares the generator's priors and never reasons about the boundary. Our method instead detects inversion because the certified breaking pair fixes the true order by construction, so the surrogate's preference for the broken patch registers as divergence. We expect elevated divergence concentrated on boundary-condition operators and low divergence on dead-store and renaming operators. We expect the bias-corrected d_hat_corr to remain elevated after subtracting the executed calibration mislabel rate, confirming the signal is not labeling noise.

**vs. Random-ordering scorer** — On any pair, the baseline produces an ordering with no relation to correctness, yielding divergence near chance. Our method instead measures a well-separated divergence estimate for real surrogates, so the gap between real-surrogate divergence and random divergence confirms the estimator discriminates rather than sitting at floor. We expect real surrogates to sit clearly below random on preserving-heavy strata while approaching random on their blind strata.

**vs. Classical mutation-kill-rate adequacy score** — On the same certified pairs, the kill-rate baseline aggregates to a single number that can be high even when the surrogate inverts the critical top-1 pair. Our method instead reports the ordering-inversion rate per stratum, so we expect to find at least one stratum where kill-rate is high (surrogate kills the breaking mutant) yet d_hat is also high (surrogate ranks the breaking mutant above the preserving one), establishing that rank-divergence carries information kill-rate does not.

**vs. Score-correlation baseline** — On the same pairs, Pearson correlation between surrogate score and the binary correctness label can be moderate while the surrogate still inverts the top-1 pair. Our method instead measures inversion directly, so we expect a stratum where correlation is positive but d_hat exceeds tau, showing the two metrics diverge on the selection-relevant quantity.

### What would falsify this idea

If the divergence estimate is uniform across all operator strata rather than concentrated where the surrogate's mechanism is predicted to be blind, the certified-pair construction is not isolating fault classes and the central claim of operator-specific diagnosis is wrong. Equally, if the per-operator divergence pattern shows no relationship to observed top-1 selection errors on real patches (Spearman near 0, AUC near 0.5), or if the side-condition ablation shows d_hat is unchanged when template-only contracts are used, the certificate does not predict deployment readiness and the ground-truth claim is unverified.

## References

1. Execution-free Agentic Program Repair for Enterprise-Scale Development
2. Self-Evolving Coding Agents via Harness-Policy Optimization
3. SWE-Bench Pro: Can AI Agents Solve Long-Horizon Software Engineering Tasks?
4. Promptbreeder: Self-Referential Self-Improvement Via Prompt Evolution
5. OpenHands: An Open Platform for AI Software Developers as Generalist Agents
6. SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering
7. BigCodeBench: Benchmarking Code Generation with Diverse Function Calls and Complex Instructions
8. Large Language Models Can Self-Improve
