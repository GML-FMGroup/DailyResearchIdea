# Monotone Regression-Filtered Harness Self-Improvement for Code Agents

## Motivation

Self-referential self-improvement systems such as Promptbreeder and the harness-search loop of Self-HPO accept a self-modification purely because it raises aggregate population or search-set fitness, with no guarantee that competence already acquired is retained; because the acceptance signal is computed on the same distribution that drove the search, a harness edit that overfits to current hard tasks can silently degrade previously solved cases and the loop has no mechanism to notice. This is a structural gap, not a tuning issue: performance selection is a point estimate over a moving task distribution, so the frontier of accepted harnesses is not required to be monotone in competence, and the field's trajectory toward recursive harness-policy optimization (Self-HPO) makes this omission increasingly costly because each round's RL update is conditioned on a harness that may have quietly discarded prior skills.

## Key Insight

Constraining harness acceptance to a one-sided significance test on a held-out prior-task regression suite makes the accepted-harness sequence a monotone non-decreasing chain in retained competence, a property that fitness-based selection cannot provide because it never evaluates the candidate against the policy's own past successes.

## Method

**What it is.** We introduce CRASH (Constrained Regression Acceptance for Self-improving Harnesses), a drop-in acceptance filter wrapped around the harness-proposal loop of Self-HPO. Inputs: the current policy π, the current accepted harness h_t, a candidate harness h' proposed by π's failure introspection, a held-out prior-task regression suite D_reg, and a fresh exploration suite D_expl. Output: a binary accept/reject decision and the updated accepted harness h_{t+1}.

**How it works.**

```pseudocode
# One round of CRASH; called wherever Self-HPO would select a harness
# Hyperparameters: alpha=0.05 (per-case significance), delta=0.10 (non-inferiority margin),
#                  n_boot=2000, m_cases=number of solved prior cases, k=3 seeds,
#                  max_S=200 (cap on |S_t| via stratified subsample by task family),
#                  n_paraphrase=3 (paraphrase ensemble size for invariant probe)

1. FREEZE the competence set:
     S_t = { c in D_reg : pi solves c under h_t on >= k-1 of k seeds }
     (only cases the CURRENT accepted harness actually solves are protected)
     If |S_t| > max_S, subsample max_S cases stratified by task family.

2. EVALUATE candidate h' on S_t and on D_expl:
     for each c in S_t:  y'_c = pass-rate of pi under h' over k seeds
     for each c in S_t:  y_t_c = stored pass-rate of pi under h_t over k seeds
     fit_expl(h') = mean pass-rate on D_expl; fit_expl(h_t) = stored value

3. REGRESSION TEST (one-sided, paired bootstrap over cases):
     d_c = y'_c - y_t_c                      # per-case competence delta
     d_bar = mean_c d_c ;  CI_lo = 5th percentile of {mean of m_cases resampled d_c} over n_boot draws
     regress = (CI_lo < -delta)              # statistically significant drop beyond margin

4. GAIN TEST (exploration must not be sacrificed):
     gain = (fit_expl(h') - fit_expl(h_t)) > 0

5. ACCEPT if (not regress) and gain:
     h_{t+1} = h' ;  S_{t+1} = S_t union {newly solved cases from D_expl}
   else:
     h_{t+1} = h_t ;  return h' to the proposal pool with the regression report attached

6. HAND OFF h_{t+1} to the P-step (RL policy update) exactly as in Self-HPO.

7. HARNESS-INVARIANCE PROBE (applied to S_t before step 3):
     For each c in S_t, evaluate pi under h_t and h' using a fixed canonical harness
     (a neutral, task-agnostic prompt template with no harness-specific strings)
     and a paraphrase ensemble of n_paraphrase=3 semantically equivalent harness variants.
     Compute invariant pass-rates y_t_c^inv and y'_c^inv as the mean over the canonical
     harness and the paraphrase ensemble. Replace y_t_c and y'_c in step 3 with these
     invariant estimates. Flag cases where |y_t_c - y_t_c^inv| > 0.2 as harness-instance-affine;
     report the fraction of S_t flagged.
```

A diagram of the data flow: `π failures → candidate h' → [FREEZE S_t from h_t] → [HARNESS-INVARIANCE PROBE on S_t] → paired eval on S_t ∪ D_expl → bootstrap CI → accept/reject → P-step`. The regression report from rejected candidates is fed back into the next H-step proposal prompt so the policy sees which prior cases its edit would have broken.

**Why this design.** We chose a *paired per-case* bootstrap over a pooled aggregate comparison because aggregate fitness lets a large gain on a few exploration tasks mask a small but systematic loss across many prior cases, accepting the cost that we must store per-case pass-rates and re-evaluate every protected case each round. We chose a *frozen competence set S_t recomputed from the current accepted harness* over a fixed global regression suite because a fixed suite drifts out of distribution as the policy improves and would eventually protect cases the policy has outgrown, accepting the cost that S_t can shrink if the policy's own competence shifts. We chose a *non-inferiority margin delta* over a strict zero-drop rule because pass-rate estimates on k seeds are noisy and a strict rule would reject nearly all edits, accepting the cost that a genuinely tiny regression below delta can slip through. We chose to *conjoin the regression test with an exploration-gain test* over regression-only acceptance because a monotone filter alone would freeze the harness at round one, accepting the cost that some safe-but-stagnant edits are rejected. We chose to *apply a harness-invariance probe* (canonical harness + paraphrase ensemble) before the regression test because harness edits are prompt-string mutations and per-case pass-rate deltas under a single harness are confounded with harness-instance affinity; the probe decouples the measurement instrument from the treatment, accepting the cost of n_paraphrase+1 extra evaluations per protected case per round. A domain expert comparing this to Promptbreeder's population-fitness selection or Self-HPO's harness search would not call CRASH a variant of either: those methods rank candidates by a single scalar fitness, whereas CRASH's decision is a two-sided constraint that requires the candidate to be simultaneously non-inferior on the protected set and superior on the exploration set, a feasible region that fitness ranking does not define.

**Why it measures what we claim.** The per-case paired delta d_c measures *retained competence on case c* because we assume the k-seed pass-rate under a fixed harness is an unbiased estimate of that case's solve probability and that the same seeds' task instances are exchangeable across harnesses; this assumption fails when a case is solved by memorized harness-specific prompt strings rather than general capability, in which case d_c reflects harness-instance affinity rather than competence. The bootstrap lower confidence bound CI_lo measures *whether the observed drop is systematic rather than sampling noise* because we assume the m_cases protected cases are drawn i.i.d. from the policy's competence distribution and that the bootstrap resampling preserves the paired structure; this assumption fails when S_t is dominated by a few task families, in which case CI_lo reflects the variance of the dominant family rather than the competence distribution. The frozen set S_t measures *the policy's existing competence set* because we assume a case solved by the current accepted harness is a case the policy is expected to retain; this assumption fails when the accepted harness itself was overfit to that case, in which case S_t overstates retained competence and the filter protects a spurious skill. The exploration gain on D_expl measures *forward progress* because we assume D_expl is drawn from the same task distribution as the next round's hard-task seed set; this assumption fails when D_expl is easier than future seeds, in which case gain reflects easy-task improvement rather than frontier advance. The harness-invariance probe measures *harness-independent competence* because we assume the canonical harness and paraphrase ensemble remove surface-form sensitivity; this assumption fails when the canonical harness itself is out of distribution for the task, in which case the probe reflects canonical-harness affinity rather than invariant competence.

## Contribution

(1) CRASH, a monotone acceptance filter for harness self-improvement that replaces scalar fitness selection with a paired per-case non-inferiority test on a frozen competence set, yielding an accepted-harness sequence that is non-decreasing in retained competence by construction. (2) The design principle that harness search and policy RL should be coupled through a *protected-set* constraint rather than a shared fitness scalar, together with the finding that rejected candidates' regression reports are themselves useful proposal signal for the next H-step. (3) An open regression-suite construction procedure that derives the protected set from the current accepted harness's own solved cases rather than a static benchmark, applicable to any harness-policy alternation loop.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Multi-round coding task stream with held-out prior-task regression suite and fresh exploration suite | Separates protected competence from forward frontier |
| Primary metric | Retained-solve rate on frozen competence set at matched exploration gain | Directly measures the two-sided constraint |
| Baseline | Self-HPO harness-policy loop (scalar fitness selection) | Tests whether two-sided constraint beats fitness ranking |
| Baseline | Harness-only optimization | Tests whether policy updates matter absent protected competence |
| Baseline | Promptbreeder-style population fitness selection | Tests whether population ranking masks regressions |
| Ablation-of-ours | CRASH with regression test removed (gain-only acceptance) | Isolates the protection mechanism's contribution |
| Ablation-of-ours | CRASH with fixed global regression suite (no frozen S_t) | Tests whether frozen set is necessary |
| Ablation-of-ours | CRASH with strict zero-drop rule (delta=0) | Tests whether margin is necessary |
| Ablation-of-ours | CRASH with pooled aggregate comparison (no paired bootstrap) | Tests whether paired per-case test is necessary |
| Diagnostic | Harness-invariance probe: fraction of S_t flagged as harness-instance-affine | Detects memorized-harness-string failure mode |
| Cost metric | Per-round wall-clock and token overhead of CRASH vs Self-HPO | Bounds feasibility |

### Why this setup validates the claim

The central claim is that a two-sided constraint — non-inferiority on a frozen competence set conjoined with exploration gain — yields harnesses that accumulate capability instead of trading prior competence for frontier gains. The Self-HPO baseline tests the sub-claim that scalar fitness ranking permits aggregate gains to mask systematic per-case regressions; if CRASH beats it specifically on retained-solve rate at matched exploration gain, the constraint is doing work that ranking cannot. The harness-only baseline tests whether the policy-update step is necessary for the protected set to remain meaningful; if harness-only also retains competence but stagnates on exploration, the gain test is shown to be the binding constraint. The Promptbreeder-style baseline tests whether population-level fitness selection, which averages over candidates, hides regressions that a paired per-case test would catch. The gain-only ablation is the critical control: it removes the regression test while keeping everything else, so any difference between it and full CRASH is attributable to the protection mechanism rather than to the harness proposal distribution or the RL step. The fixed-global-suite ablation tests whether the frozen S_t is necessary by replacing it with a fixed suite that may drift out of distribution. The strict-zero-drop ablation tests whether the non-inferiority margin is necessary by rejecting any negative delta. The pooled-aggregate ablation tests whether the paired per-case bootstrap is necessary by comparing mean pass-rates without pairing. The harness-invariance probe diagnostic quantifies how often the load-bearing assumption fails: if a large fraction of S_t is flagged as harness-instance-affine, the regression test is measuring prompt-string affinity rather than competence. Retained-solve rate at matched exploration gain is the right metric because it is the only quantity that simultaneously reflects both sides of the constraint; a method that wins on exploration alone or retention alone has not demonstrated the conjunction the claim asserts.

### Expected outcome and causal chain

**vs. Self-HPO (scalar fitness selection)** — On a round where a candidate harness unlocks several fresh exploration cases but silently breaks a handful of previously solved edge-case tasks, Self-HPO accepts it because the pooled fitness delta is positive and the per-case losses are invisible to a scalar ranker. Our method instead freezes the competence set from the current accepted harness, computes paired per-case deltas under the harness-invariance probe, and rejects the candidate when the bootstrap lower bound falls below the margin, so we expect a noticeable retained-solve advantage concentrated on the previously-solved edge-case subset with rough parity on the exploration subset.

**vs. Harness-only optimization** — On a round where the harness has been tuned to the current policy's weaknesses but the policy itself has not improved, harness-only optimization cannot exploit the newly solved exploration cases because it never updates the policy, so its exploration gain saturates while its competence set stays flat. Our method hands the accepted harness to the P-step exactly as Self-HPO does, so the policy update converts exploration gains into new competence, and we expect our retained-solve rate to climb across rounds while harness-only plateaus after the first few rounds.

**vs. Promptbreeder-style population fitness selection** — On a round where a candidate is a slight regression on many prior cases but a large win on a few exploration tasks, population fitness selection ranks it highly because the aggregate is dominated by the few large wins, and the systematic small losses are averaged away. Our method's paired per-case bootstrap under the harness-invariance probe detects the systematic drop because the many small negative deltas produce a lower confidence bound below the margin even when the mean is positive, so we expect a visible gap on the many-prior-cases subset and near-parity on the few-exploration-tasks subset.

**vs. CRASH with regression test removed (gain-only acceptance)** — On a round where a candidate harness overfits to a memorized prompt string that solves one exploration task, gain-only acceptance admits it and the next round's competence set silently loses cases the overfit string does not cover. Our method's frozen-set regression test under the harness-invariance probe rejects the candidate because the paired deltas on the protected cases are systematically negative, so we expect gain-only to show higher exploration peaks but a steadily eroding retained-solve rate, while full CRASH shows a flatter exploration curve with a preserved competence floor.

**vs. CRASH with fixed global regression suite** — On a round where the policy has improved and the fixed suite contains cases the policy has outgrown, the fixed-suite variant protects obsolete cases and rejects candidates that would advance the frontier. Our method's frozen S_t recomputed from the current accepted harness drops outgrown cases, so we expect the fixed-suite variant to show lower exploration gain at matched retained-solve rate, while full CRASH maintains both.

**vs. CRASH with strict zero-drop rule (delta=0)** — On a round where a candidate is genuinely non-inferior but pass-rate estimates on k=3 seeds are noisy, the strict rule rejects nearly all edits because any negative delta triggers rejection. Our method's non-inferiority margin delta=0.10 tolerates noise, so we expect the strict rule to show near-zero acceptance rate and stagnant exploration, while full CRASH accepts safe edits and accumulates competence.

**vs. CRASH with pooled aggregate comparison** — On a round where a candidate has a large gain on a few exploration tasks and a small systematic loss across many prior cases, the pooled aggregate comparison sees a positive mean delta and accepts, masking the regression. Our method's paired per-case bootstrap detects the systematic drop because the many small negative deltas produce a lower confidence bound below the margin, so we expect the pooled variant to show higher exploration peaks but lower retained-solve rate, while full CRASH preserves the competence floor.

### What would falsify this idea

If full CRASH's retained-solve advantage over Self-HPO is uniform across all case subsets rather than concentrated on the previously-solved edge-case subset where the predicted masking failure occurs, the two-sided constraint is not the causal mechanism and the central claim is wrong. Likewise, if the gain-only ablation matches full CRASH on retained-solve rate, the regression test is not doing the work we attribute to it. Additionally, if the harness-invariance probe flags a large fraction of S_t as harness-instance-affine and the regression test's decisions are unchanged when using invariant estimates, the load-bearing assumption is violated and the filter certifies spurious monotonicity.

## References

1. Self-Evolving Coding Agents via Harness-Policy Optimization
