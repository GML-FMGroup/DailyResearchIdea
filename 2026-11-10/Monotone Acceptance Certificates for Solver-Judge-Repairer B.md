# Monotone Acceptance Certificates for Solver-Judge-Repairer Benchmark Validity

## Motivation

Benchmark construction pipelines such as the solver-judge-repairer loop in Longtail-SWE: High-Quality SWE Tasks for Underrepresented Languages treat the judge's acceptance decision as a static gate fixed at construction time, so task validity is entangled with the specific solver and judge model versions used during construction rather than with the latent issue the task is supposed to test. This is a meta-level structural failure that recurs across every model-assisted benchmark-building pipeline: because acceptance is recorded as a single binary outcome, there is no way to distinguish a genuinely well-specified task (which any competent judge would accept for a correct solution and reject for an incorrect one) from a task that happened to pass one particular judge-solver pair, leaving validity unidentifiable under model substitution. The root cause is that acceptance is never modeled as a function of model configuration, so monotone consistency — the property that refining the solver or judge should not flip a valid task from accepted to rejected — is neither measured nor enforced.

## Key Insight

If judge acceptance is recorded as a monotone function of an explicit model-capability ordering rather than as a single threshold outcome, then a task's validity becomes a certificate over that ordering — valid tasks are exactly those whose acceptance curve is non-decreasing — which is a property of the latent issue specification and is invariant to which particular model versions instantiate the ordering.

## Method

**What it is.** MAC (Monotone Acceptance Certificate) replaces the fixed judge acceptance threshold in a solver-judge-repairer pipeline with a per-task acceptance curve evaluated over a ladder of model configurations ordered by capability. Inputs: a candidate task (issue description, executable environment, tests), a capability-ordered ladder of solver versions S = (s_1 ≺ s_2 ≺ ... ≺ s_m) and judge versions J = (j_1 ≺ j_2 ≺ ... ≺ j_n), and repeated rollouts per cell. Outputs: a per-task acceptance matrix A ∈ {0,1}^{m×n} (averaged over rollouts), a monotonicity certificate (pass/fail plus a violation set), and a repair signal that localizes which cells violate monotonicity.

**How it works.**

```pseudocode
INPUT: task t, solver ladder S=(s_1≺...≺s_m), judge ladder J=(j_1≺...≺j_n), rollouts R=5
OUTPUT: certificate c(t) ∈ {valid, invalid, underdetermined}, violation set V

# Phase 0: ladder calibration (held-out calibration set of 512 unambiguous tasks)
# Each calibration task has a trivially correct and a trivially incorrect solution.
# For each model version v, compute calibration accuracy acc(v) = fraction of calibration
# tasks where v accepts the correct solution and rejects the incorrect one.
# Order solvers by acc(s_i) ascending; ties broken by parameter count (smaller first).
# Order judges by acc(j_i) ascending; ties broken by parameter count (smaller first).
# Record rank-inconsistency diagnostic: for each pair of calibration tasks (a,b) where
# a is strictly easier than b (by construction), check that acc(v) on a ≥ acc(v) on b.
# If any model version violates this rank consistency on >5% of calibration pairs,
# flag the ladder as 'ordering-unfaithful' and report the violating pairs.

# Phase 1: acceptance matrix
for i in 1..m, j in 1..n:
    A[i,j] = mean over r in 1..R of judge(j).accept(t, solver(s_i).solve(t))

# Phase 2: monotone consistency check
V = {}
for each adjacent pair (i,j),(i',j') with i'≥i and j'≥j and (i',j')≠(i,j):
    if A[i',j'] < A[i,j] - tau:      # tau = 0.1 tolerance for rollout noise
        V.add(((i,j),(i',j'), A[i,j]-A[i',j']))

# Phase 3: certificate assignment
if |V| == 0 and A[1,1] < 1 - tau and A[m,n] > tau:
    c(t) = valid                      # non-decreasing and non-degenerate
elif |V| == 0 and (A[1,1] ≥ 1-tau or A[m,n] ≤ tau):
    c(t) = underdetermined            # task too easy or too hard to certify
else:
    c(t) = invalid

# Phase 4: repair localization
for each violation ((i,j),(i',j'), delta) in V sorted by delta desc:
    signal = judge(j').rationale(t, solver(s_i').solve(t)) vs judge(j).rationale(...)
    propose edit to t.tests or t.description targeting the disagreement
    # re-run Phases 1-3 on edited task; accept edit iff |V| strictly decreases

RETURN c(t), V
```

The ladder ordering is established by a held-out calibration set of 512 unambiguous tasks (trivially correct and trivially incorrect solutions) on which each model version is scored; versions are ordered by calibration accuracy, and ties are broken by parameter count. Rollouts R = 5 per cell; tolerance tau = 0.1. The calibration set size of 512 is chosen to give a standard error of at most 0.022 on accuracy estimates (binomial proportion), sufficient to separate adjacent ladder rungs that differ by at least 0.05 in calibration accuracy with 80% power at alpha=0.05.

**Why this design.** We chose an explicit capability-ordered ladder over a random sample of model configurations because monotonicity is only well-defined relative to an ordering, accepting the cost that the ladder must be recalibrated when new model versions appear. We chose a tolerance tau on the acceptance matrix over exact binary monotonicity because rollout noise on stochastic solvers would otherwise flag valid tasks as invalid, accepting the cost that small genuine violations below tau are missed. We chose to localize violations to specific (solver, judge) cells and use the judge rationale contrast as the repair signal, rather than re-running the full repair loop blindly, because the violation cell identifies exactly which capability transition the task specification fails to handle, accepting the cost that rationale-based repair may still require multiple iterations. We chose a three-way certificate (valid/invalid/underdetermined) over a binary gate because tasks that no configuration solves or that all configurations solve carry no discriminative signal and should be excluded rather than silently accepted. A domain expert would not describe MAC as a variant of threshold tuning or of Longtail-SWE's judge-guided repair: those methods optimize a single acceptance outcome at one operating point, whereas MAC's object is the shape of the acceptance surface across the capability ordering, and the repair target is the violation set rather than the judge's verdict.

**Why it measures what we claim.** The acceptance matrix entry A[i,j] measures the latent task validity concept because it operationalizes the assumption that a well-specified issue specification induces a monotone response to solver and judge capability — if the specification truly encodes the intended requirement, a more capable solver should never produce a solution that a more capable judge rejects while a less capable pair accepts; this assumption fails when the capability ordering is miscalibrated (e.g., a model that is better at the target language but worse at test execution is ranked too high), in which case A reflects ordering error rather than specification quality. The violation set V measures identifiability of latent task intent because each violation localizes a capability transition at which the judge's acceptance criterion diverges from the intended requirement; this assumption fails when the divergence is caused by judge stochasticity rather than by a specification ambiguity, in which case V reflects sampling noise. The certificate c(t) measures model-invariant validity because it requires monotone consistency across the entire ladder rather than at one configuration; this assumption fails when the ladder is too short to expose a non-monotone region, in which case c(t) reflects validity relative to the sampled capability range rather than universal validity. The rank-inconsistency diagnostic in Phase 0 measures ordering faithfulness because it checks that the calibration-derived scalar ordering is consistent with the partial order induced by task difficulty on the calibration set; this assumption fails when a model version's accuracy on easy calibration tasks is lower than on hard ones (rank inversion), in which case the ladder ordering is flagged as unfaithful and the certificate is reported as conditional on that ordering.

## Contribution

(1) A monotone acceptance certificate (MAC) that replaces the fixed judge threshold in solver-judge-repairer pipelines with a per-task acceptance curve over a capability-ordered model ladder, yielding a model-invariant validity predicate and a localized violation set. (2) A repair procedure that uses the violation set as a targeted signal, editing task specifications only when the edit strictly reduces the number of monotonicity violations, which converts judge disagreement from a binary verdict into a directional gradient. (3) A released evaluation protocol and violation-localization tooling that can be applied to existing benchmarks such as Longtail-SWE to audit how many of their 202 tasks carry configuration-dependent validity.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | Longtail-SWE candidate tasks (13 languages) | Tasks span solver-judge-repairer pipeline and underrepresented languages |
| Primary metric | Monotonicity violation rate | Directly measures the acceptance-surface property MAC certifies |
| Baseline | Fixed-threshold judge gate | Tests whether per-task curves beat one operating point |
| Baseline | Judge-guided repair (Longtail-SWE) | Tests whether violation localization beats blind repair |
| Baseline | Random-configuration acceptance sampling | Tests whether capability ordering is essential |
| Baseline | IRT model fit to the same acceptance matrix | Tests whether MAC's certificate is reducible to a standard difficulty-discrimination model |
| Ablation-of-ours | MAC without repair localization (certificate only) | Isolates contribution of violation-targeted repair |
| Ablation-of-ours | Reduced ladder (m=n=3 vs. 5) | Tests whether certificate discrimination is preserved at lower cost |
| Ground-truth injection | Controlled test mutation (known-ambiguous vs. known-well-specified) | Quantifies certificate false-positive/negative rates |

### Why this setup validates the claim

The central claim is that a task's validity is identifiable from the *shape* of the acceptance surface across a capability-ordered ladder, not from a single judge verdict at one operating point. The fixed-threshold baseline tests the sub-claim that one operating point cannot distinguish specification ambiguity from solver weakness: if MAC's gain over it is concentrated on tasks where the threshold gate's verdict flips across the ladder, the ordering carries the signal. The Longtail-SWE judge-guided repair baseline tests whether localization matters: if MAC's repair converges in fewer edits specifically on tasks with localized violations, the violation set is doing causal work. The random-configuration baseline tests whether capability ordering is essential rather than incidental — if random sampling achieves similar monotonicity certification, the ordering premise is wrong. The IRT baseline tests whether MAC's certificate is reducible to a standard difficulty-discrimination model fit to the same matrix: if IRT assigns high discrimination to tasks that MAC flags as violations, the two methods disagree and MAC's certificate is not merely a reparameterization. The certificate-only ablation isolates whether repair localization, not merely detection, drives retained-task quality. The reduced-ladder ablation tests whether the certificate's discrimination survives at m=n=3, bounding the cost of the method. The ground-truth injection study (controlled test mutation) provides known-ambiguous and known-well-specified tasks, allowing measurement of the certificate's false-positive rate (well-specified tasks flagged invalid) and false-negative rate (ambiguous tasks certified valid), and separates violations attributable to ordering error (detected by Phase 0 rank-inconsistency) from genuine ambiguity. Monotonicity violation rate is the right primary metric because it is exactly the quantity MAC claims to reduce and it is measurable on every retained task without ground-truth labels, making the comparison falsifiable rather than circular.

### Expected outcome and causal chain

**vs. Fixed-threshold judge gate** — On a case where a mid-capability solver produces a plausible patch that a strict judge rejects on a stylistic rationale while a weaker solver's patch passes, the fixed-threshold baseline produces a single accept/reject verdict that silently discards the task or admits it, because it has no notion of how acceptance varies with capability. Our method instead reads the acceptance matrix and flags the (solver, judge) cell transition as a violation, because a more capable pair rejecting what a less capable pair accepted is exactly the non-monotone signature. We expect a noticeable gap in violation rate concentrated on tasks whose verdict flips across the ladder, with parity on tasks that are uniformly accepted or rejected.

**vs. Judge-guided repair (Longtail-SWE)** — On a case where the task's tests under-specify an edge case, the baseline produces a repaired task by iterating the judge's rationale against the *same* solver output, because its repair signal is the judge's verdict at one operating point and cannot tell which capability transition exposed the gap. Our method instead targets the specific violating cell pair and contrasts judge rationales across the two capability levels, because the violation localizes the transition the specification fails to handle. We expect MAC to reach a monotone certificate in fewer repair iterations on tasks with a single localized violation, and a smaller residual violation rate after repair.

**vs. Random-configuration acceptance sampling** — On a case where the task is genuinely ambiguous, random sampling of model configurations produces an acceptance matrix whose rows and columns are not ordered, so the baseline's monotonicity check either never fires or fires spuriously, because monotonicity is undefined without a capability ordering. Our method instead evaluates the same cells under a calibration-derived ordering, because the ordering is what makes a violation interpretable as a specification gap rather than sampling noise. We expect the random baseline to show near-chance discrimination between ambiguous and unambiguous tasks, while MAC separates them by a clear margin on the calibration-held-out subset.

**vs. IRT model fit** — On a task family where IRT assigns high discrimination (because acceptance varies sharply across the matrix) but MAC flags a violation (because the variation is non-monotone along the capability ordering), the IRT baseline produces a high-discrimination score that treats the task as informative, because IRT models item discrimination without enforcing monotonicity along a capability ordering. Our method instead reports the task as invalid and localizes the violating cell pair, because the non-monotone signature is exactly what MAC certifies against. We expect a disagreement rate of at least 15% on the injected ambiguous task set, with MAC's certificate aligning with ground-truth ambiguity more closely than IRT's discrimination score.

**vs. Reduced ladder (m=n=3)** — On the full task set, the reduced ladder (m=n=3) uses 9 cells instead of 25, reducing API cost by 64%. We expect the certificate's discrimination (measured by AUC of violation rate against ground-truth ambiguity) to drop by no more than 0.05 compared to the full ladder, because the monotonicity signal is carried by the ordering rather than by the number of rungs. We will report wall-clock time and API budget per task for both ladder sizes.

**vs. Ground-truth injection** — On the injected set of 200 tasks (100 known-ambiguous via controlled test mutation, 100 known-well-specified), we expect MAC's false-positive rate (well-specified flagged invalid) to be below 10% and false-negative rate (ambiguous certified valid) to be below 15%, with the Phase 0 rank-inconsistency diagnostic attributing at least 30% of false positives to ordering error rather than genuine ambiguity.

### What would falsify this idea

If MAC's reduction in violation rate and repair iterations is uniform across tasks rather than concentrated on tasks whose verdicts flip across the ladder, then the acceptance-surface shape is not the causal signal and the method is merely a noisier threshold tuner. Equally, if the random-configuration baseline matches MAC's certificate discrimination, the capability-ordering premise is wrong and the central claim fails. If the ground-truth injection study shows false-positive and false-negative rates above 25%, the certificate is not a reliable measure of latent task validity.

## References

1. Longtail-SWE: High-Quality SWE Tasks for Underrepresented Languages
