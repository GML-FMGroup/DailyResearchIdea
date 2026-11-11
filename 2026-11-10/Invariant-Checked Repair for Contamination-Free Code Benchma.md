# Invariant-Checked Repair for Contamination-Free Code Benchmark Construction

## Motivation

The Longtail-SWE solver-judge-repairer pipeline treats decontamination as a one-shot pre-filter applied before repair, so judge-guided revisions can silently reintroduce overlap with training data while nominally preserving task requirements. This is a structural failure of closure: because the decontamination predicate is never re-evaluated after a repair step, the pipeline cannot certify that the repaired task set satisfies the same contamination constraint as the pre-filtered candidate set, leaving the benchmark's validity dependent on an unchecked assumption that repair preserves the predicate.

## Key Insight

Decontamination is a repair-invariant property only if every repair operator is constrained to act within the decontamination predicate's feasible set, which can be enforced by treating each repair as a candidate transition that must pass a re-evaluation of the predicate before acceptance, making the predicate a maintained invariant of the construction loop rather than a pre-filter.

## Method

**What it is.** We introduce CIRCE (Closure-Insured Repair with Contamination Enforcement), a construction protocol that wraps any solver-judge-repairer loop with an invariant monitor. Its inputs are a candidate task set, a decontamination predicate D (n-gram overlap against a training corpus plus embedding-similarity threshold), and a repairer R; its output is a certified task set where every retained task satisfies D at the moment of admission and after every repair step. **Load-bearing assumption (stated explicitly):** CIRCE certifies D-invariance, not contamination-freedom. The certificate is conditional on D being a sound and complete certificate of training-set contamination; D is a surface-overlap proxy (n-gram + embedding similarity) and is known to be evaded by paraphrase and semantic-preserving rewrites (Sainz et al. 2023; Oren et al. 2024). We therefore (i) report the certified set's residual contamination as a bound measured against an independent paraphrase-leakage probe (see experiment), and (ii) treat every certified task as 'D-clean' rather than 'contamination-free' in all claims.

**How it works.**

```pseudocode
CIRCE(candidates, D, solver S, judge J, repairer R, max_iters=5):
  certified = []
  for task in candidates:
    if not D(task): continue          # pre-filter (unchanged)
    state = task
    history = [state]
    for i in 1..max_iters:
      traj = S(state)
      verdict, revision_hint = J(state, traj)
      if verdict == ACCEPT: break
      candidate = R(state, revision_hint)
      # invariant check: candidate must satisfy D AND be a bounded edit of state
      if not D(candidate):
        candidate = project_to_D(candidate, state, D)   # see below
      if edit_distance(state, candidate) > tau_edit:
        candidate = state          # reject over-large semantic drift
      state = candidate
      history.append(state)
    if D(state) and verdict == ACCEPT:
      certified.append(state)
  return certified

project_to_D(candidate, anchor, D):
  # minimal edit that restores D: revert the smallest contiguous diff
  # between candidate and anchor until D(candidate) holds or anchor is reached
  for span in sorted_diffs(candidate, anchor, by=size_ascending):
    candidate = revert(candidate, span)
    if D(candidate): return candidate
  return anchor   # fallback: no contamination-preserving repair exists
```

Hyperparameters: `max_iters=5` (matches Longtail-SWE's repair budget), `tau_edit` set to the 90th percentile of accepted-repair edit distances observed on a stratified 100-task pilot (≥7 tasks per language across all 13 Longtail-SWE languages; pilot repairs run with the same S, J, R), and D's embedding threshold calibrated to a 1% false-positive rate on held-out clean tasks (n=2,000 held-out tasks disjoint from the candidate pool). We additionally report sensitivity of certified yield and contamination-preservation rate to `tau_edit` ∈ {50th, 75th, 90th, 95th percentile} and to embedding threshold at FPR ∈ {0.5%, 1%, 2%}.

**Why this design.** We chose to re-check D after every repair rather than only at admission because the judge's revision hints can introduce new code snippets or problem statements drawn from the training corpus, and a one-shot filter cannot detect this; the cost is extra overlap computations per iteration, which we amortize with n-gram hashing. We chose `project_to_D` as a minimal-revert projection over rejecting the candidate outright because outright rejection would discard judge-validated repairs and stall the loop, accepting the cost that the projected repair may be a weaker fix than the judge intended. We chose the `tau_edit` cap over an unbounded repair because the shared residual gap concerns semantic drift under judge substitution, and a bounded edit distance keeps the repaired task anchored to the original issue, accepting the cost that some legitimate large repairs are rejected. A domain expert would not describe this as a variant of standard data-filtering pipelines (e.g., the pre-filter in Longtail-SWE) because those pipelines have no notion of a repair transition; CIRCE's contribution is the invariant-maintenance loop and projection operator, not the filter itself.

**Why it measures what we claim.** The quantity `D(candidate)` after each repair measures contamination-preservation of the repair operator because the assumption is that D's n-gram and embedding thresholds are a sufficient certificate for training-set overlap; this assumption fails when the training corpus is incomplete or when paraphrased leakage evades both signals, in which case `D(candidate)` reflects only surface overlap rather than true contamination. We make this assumption explicit and bound it empirically: on a held-out paraphrase-leakage set (n=500 tasks constructed by LLM paraphrase of known-contaminated tasks, verified to evade D), we measure D's false-negative rate and report the certified set's residual contamination as `FN_rate × |certified|` with a 95% bootstrap CI. The quantity `edit_distance(state, candidate) <= tau_edit` measures semantic anchoring of the repair because the assumption is that bounded edit distance preserves the latent task intent; this assumption fails when a small edit (e.g., changing one assertion) flips the intended behavior, in which case the cap reflects textual proximity rather than intent preservation. We bound this by measuring intent-preservation on a 200-task sample: for each certified task, we run the original test suite and record pass rate; we report the fraction of certified tasks whose original tests fail (intent-flip rate). The fallback `return anchor` measures closure of the invariant because the assumption is that the original pre-filtered task is a feasible point of D; this assumption fails if D itself is misspecified, in which case the fallback certifies a task that was never truly clean. We report the frequency of fallback invocation and, for those tasks, their residual contamination under the paraphrase probe.

## Contribution

(1) CIRCE, a repair-invariant construction protocol that replaces one-shot decontamination with a maintained predicate and a minimal-revert projection operator for restoring the invariant after judge-guided repairs. (2) The design principle that repair operators in benchmark construction must be constrained to the feasible set of the validity predicate, with a bounded-edit cap that couples contamination closure to semantic anchoring. (3) An open-source invariant monitor and projection tool that can wrap existing solver-judge-repairer pipelines without modifying the solver or judge.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Longtail-SWE 13-language candidate pool (stratified 100-task pilot + full pool) | Matches repair-loop setting where leakage arises |
| Primary metric | Contamination-preservation rate post-repair (D-satisfaction) | Directly measures invariant maintenance claim |
| Secondary metric | Residual contamination bound via paraphrase-leakage probe (FN rate × certified) | Bounds D's false negatives |
| Secondary metric | Intent-preservation rate (original test pass rate on certified tasks) | Bounds edit-distance proxy for intent |
| Baseline | One-shot pre-filter (Longtail-SWE) | Tests need for per-iteration re-checking |
| Baseline | Reject-on-contamination repairer | Tests value of projection vs. outright rejection |
| Baseline | Unbounded-edit repairer | Tests tau_edit anchoring cap |
| Ablation-of-ours | CIRCE without project_to_D | Isolates projection operator contribution |
| Calibration set | 500 paraphrase-injected tasks (LLM paraphrase of known-contaminated tasks, verified to evade D) | Measures D false-negative rate |

### Why this setup validates the claim

The central claim is that contamination can be reintroduced during repair, and that CIRCE's invariant-maintenance loop keeps certified tasks D-clean while preserving judge-validated repairs. The one-shot pre-filter baseline tests the sub-claim that admission-time filtering is insufficient: if contamination re-enters only through repairs, this baseline should show post-repair D-violations that CIRCE eliminates. The reject-on-contamination baseline tests the sub-claim that projection beats discarding: it should retain fewer certified tasks because valid repairs are thrown away, revealing the yield cost CIRCE avoids. The unbounded-edit baseline tests the anchoring sub-claim: without tau_edit, repaired tasks drift and either fail D or diverge from original intent, which CIRCE's cap prevents. The ablation (no project_to_D) isolates the projection operator specifically, separating its contribution from the re-checking loop. Contamination-preservation rate is the right primary metric because it directly instantiates the invariant the method claims to maintain; a method that preserves D at admission and after every repair is exactly what the certificate asserts, so this metric is not a proxy but the claim itself. Because D is a surface-overlap proxy, we additionally report the residual contamination bound (D's false-negative rate on the paraphrase-leakage calibration set × certified set size) and the intent-preservation rate, so the certificate is explicitly conditional on D's measured recall and edit-distance's measured intent fidelity.

### Expected outcome and causal chain

**vs. One-shot pre-filter (Longtail-SWE)** — On a case where the judge's revision hint pulls a code snippet or problem phrasing from the training corpus into the repaired task, the baseline produces a certified task whose post-repair n-gram overlap exceeds D's threshold because it only checked D at admission and never re-queried after the repairer ran. Our method instead re-evaluates D after each repair and projects the candidate back into the clean region, so we expect a concentrated gap on the subset of tasks whose repairs introduce new corpus-adjacent text, with near-parity on tasks whose repairs are localized to non-leaking regions. We additionally expect the paraphrase-leakage probe to show non-zero residual contamination in both methods, with CIRCE's residual bounded by D's measured false-negative rate.

**vs. Reject-on-contamination repairer** — On a case where the judge validates a repair that happens to include a small contaminated span, the baseline produces a rejected candidate and stalls the loop, discarding an otherwise correct fix because it treats any D violation as fatal. Our method instead reverts only the smallest offending span via project_to_D, retaining the judge-validated fix with the contaminated fragment excised, so we expect a higher certified-task yield on tasks requiring multi-iteration repair, with the gap widening as max_iters increases. We expect the intent-preservation rate to be comparable between the two, since both preserve the judge-validated core.

**vs. Unbounded-edit repairer** — On a case where iterative revision hints progressively rewrite the problem statement or assertions, the baseline produces a repaired task that either drifts far from the original issue intent or accumulates contamination, because nothing constrains edit magnitude between iterations. Our method instead caps edit distance at tau_edit and reverts over-large drifts to the prior anchored state, so we expect fewer certified tasks that later fail intent-preservation checks, with the effect concentrated on tasks whose repair trajectories span multiple iterations. We report the intent-flip rate (original tests failing) for both methods to quantify this.

### What would falsify this idea

If CIRCE's contamination-preservation gain over the one-shot pre-filter is uniform across all tasks rather than concentrated on the subset whose repairs introduce corpus-adjacent text, the claimed causal mechanism (repair-introduced contamination) is wrong and the gain is an artifact. Likewise, if the no-projection ablation matches full CIRCE on certified yield, the projection operator is not doing the work we claim. Additionally, if D's false-negative rate on the paraphrase-leakage calibration set is high (e.g., >20%), the certificate is weak and the residual contamination bound will be large, falsifying the practical value of D-invariance.

## References

1. Longtail-SWE: High-Quality SWE Tasks for Underrepresented Languages
