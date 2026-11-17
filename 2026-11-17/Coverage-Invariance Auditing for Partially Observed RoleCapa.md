# Coverage-Invariance Auditing for Partially Observed Role–Capability Benchmark Atlases

## Motivation

The InclusionMed Index aggregates model scores over a 3×3 role–capability Atlas while only 3 of 9 cells are populated, yet the aggregation step treats the partially observed matrix as if it were the declared full matrix — so a model's reported rank is structurally confounded with which cells happen to be instantiated rather than with the capability the Index claims to summarize. InclusionMed (E1, E6) established rigorous provenance via structured audits and independent reproduction of 388 candidates down to four benchmarks, but its selection pipeline never generalized aggregation to unpopulated cells, so the Index is not closed under its own declared cell set. This coverage-invariance gap recurs across medical benchmark trees (MedXpertQA's specialty/body-system grid, OpenMIBOOD's dataset/covariate-shift grid, MediConfusion's failure-mode grid), where aggregate rankings are reported over unevenly populated strata with no identifiability condition attached.

## Key Insight

Treating the Atlas as a partially observed matrix and requiring the aggregate Index to be invariant (within a declared tolerance) to which cells are populated reduces aggregation to an identifiability question whose answer is determined by whether the cell-population pattern is a balanced design — a property that can be audited from the coverage pattern alone, independent of any model's scores.

## Method

**(A) What it is.** Coverage-Invariance Audit (CIA) is an audit layer that wraps InclusionMed's existing audit-and-reproduction selection pipeline. Inputs: the Atlas cell-population mask M ∈ {0,1}^{3×3}, the per-model per-cell scores S, and a declared tolerance τ (default 2.0 Index points). Outputs: a coverage-invariance certificate (invariant / non-identifiable with named unpopulated cells), a cell-weighted Index with calibrated prior contribution for missing cells, and a rank-stability interval per model.

**(B) How it works.**

```pseudocode
INPUT: cell mask M (3x3), per-model scores S[m,c] for populated cells,
       tolerance tau (default 2.0 Index points), prior family P (cell-mean shrinkage),
       n_boot = 1000, interaction_bound = tau

PHASE 1 — Coverage pattern characterization
  for each cell c in 9 cells:
      populated[c] = M[c] == 1
  design_balance = (populated cells form a complete row OR complete column
                    in every role and every capability margin)
      # operationalized as: every role has >=1 populated capability AND
      # every capability has >=1 populated role

PHASE 2 — Identifiability test (conditional on observed interaction)
  # Load-bearing assumption stated explicitly: design balance alone does NOT
  # guarantee that the cell-weighted Index's expectation is independent of
  # which cells are populated, because the shrinkage prior in Phase 3 uses
  # populated-cell scores. The certificate is therefore conditional on the
  # populated cells' role×capability interaction being bounded.
  if design_balance is FALSE:
      flag NON_IDENTIFIABLE; report the specific empty role/capability margins
      # no aggregate emitted; only per-cell breakdowns
  else:
      # Fit additive model S[m,c] = mu + a[role(c)] + b[capability(c)] + e[m,c]
      # on populated cells only (two-way ANOVA without replication).
      interaction_hat = max over populated cells of |e[m,c]|, averaged over m
      if interaction_hat > interaction_bound:
          flag NON_IDENTIFIABLE_CONDITIONAL; report interaction_hat and
          the specific role×capability pairs driving it
          # no aggregate emitted; only per-cell breakdowns
      else:
          proceed to Phase 3

PHASE 3 — Cell-weighted estimator with calibrated prior
  for each model m:
      for each unpopulated cell c:
          S_hat[m,c] = shrinkage_prior(m, c; P)
              # P = empirical Bayes cell mean from populated cells sharing
              #     the same role or same capability, weighted by n_benchmarks
      Index[m] = sum_c w[c] * (populated[c] ? S[m,c] : S_hat[m,c])
          # w[c] = 1/9 uniform over declared cells (default)

PHASE 4 — Coverage-invariance certificate
  for b in 1..n_boot:
      resample the populated-cell subset M_b (stratified by role, capability)
      recompute Index_b[m] under M_b
  invariance_gap[m] = max_b |Index_b[m] - Index[m]|
  n_distinct_masks = number of distinct resampled masks M_b
  certificate[m] = INVARIANT if (invariance_gap[m] <= tau AND
                                 n_distinct_masks >= 10)
                   else NON_IDENTIFIABLE with the gap value and n_distinct_masks

PHASE 5 — Rank-stability interval
  rank_lo[m], rank_hi[m] = min/max rank of m across all bootstrap subsets M_b
  emit (Index[m], invariance_gap[m], n_distinct_masks, rank_lo[m], rank_hi[m], certificate[m])
```

**(C) Why this design.** We chose a *design-balance* identifiability condition over a model-fit-based imputation because balance is a property of the coverage pattern alone and cannot be gamed by a strong model on the populated cells, accepting the cost that some Atlases with rich but unbalanced coverage will be flagged non-identifiable rather than scored. We chose an empirical-Bayes cell-mean shrinkage prior over a zero-fill or random-imputation prior because zero-fill silently conflates 'unmeasured' with 'failed', accepting the cost that the prior injects a small bias toward the populated-cell mean whose magnitude is reported in the certificate. We chose a bootstrap-over-populated-subsets invariance gap over a single-point Index because a point estimate cannot distinguish a stable rank from a lucky coverage pattern, accepting the cost of n_boot=1000 recomputations per model. We chose a uniform 1/9 cell weight over data-driven weights because data-driven weights re-introduce the coverage dependence the audit is meant to remove. This is not a variant of standard benchmark aggregation or of retrieval-style benchmark selection (the closest prior being InclusionMed's own Index, which reports breakdowns but no coverage-invariance condition): a domain expert would not describe CIA as 'InclusionMed plus bootstrap', because the load-bearing object is the identifiability test on the coverage mask, which InclusionMed's pipeline never computes.

**(D) Why it measures what we claim.** The design-balance flag measures *identifiability of the aggregate Index* because the assumption that every role and capability is observed at least once makes the cell-weighted estimator's expectation independent of the coverage pattern; this assumption fails when a role or capability margin is entirely empty, in which case the flag correctly reports non-identifiability rather than a score. The shrinkage prior S_hat measures *the calibrated contribution of an unpopulated cell* because the assumption that cells sharing a role or capability have exchangeable score distributions makes the empirical-Bayes mean an unbiased estimate of the missing cell's expected score; this assumption fails when a capability interacts strongly with role (e.g., patient communication scoring higher than clinical decision-making, as InclusionMed reports), in which case the prior reflects the shared-margin mean rather than the true cell value. The bootstrap invariance gap measures *rank stability under coverage variation* because the assumption that resampling populated cells approximates the space of admissible Atlases makes the gap an upper estimate of rank sensitivity to coverage; this assumption fails when the populated cells are few enough that the bootstrap distribution is degenerate, in which case the gap under-reports true sensitivity and the certificate should be read as optimistic. The interaction bound in Phase 2 measures *the conditional validity of the design-balance certificate* because the assumption that the populated cells' role×capability interaction is within τ of zero makes the shrinkage prior's bias bounded by τ; this assumption fails when the interaction exceeds τ, in which case the certificate is withheld and the interaction magnitude is reported instead.

## Contribution

['(1) A coverage-invariance audit layer (CIA) that formalizes the InclusionMed Index as a cell-weighted estimator over a partially observed 3×3 role–capability matrix and attaches an explicit identifiability condition (design balance) plus a bootstrap invariance certificate to every reported aggregate.', "(2) A cell-weighted estimator with an empirical-Bayes shrinkage prior for unpopulated cells that separates 'unmeasured' from 'failed', together with a rank-stability interval that reports the range of model ranks admissible under coverage variation.", "(3) An audit protocol that wraps InclusionMed's existing structured-audit-and-reproduction selection pipeline without altering benchmark provenance, demonstrated on the first-release Atlas (3 of 9 cells) to show which reported rankings survive the invariance test."]

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | InclusionMed Atlas (5 benchmarks, partial 3×3 mask) | Real uneven coverage is the target phenomenon |
| Primary metric | Certificate agreement with held-out-cell reconstruction error | Directly tests identifiability claim, not score level |
| Baseline | InclusionMed Index (point estimate, no coverage test) | Tests whether coverage-invariance condition adds signal |
| Baseline | Zero-fill cell-weighted Index | Tests shrinkage prior against unmeasured-as-failed conflation |
| Baseline | Complete-case Index (populated cells only) | Tests whether bootstrap gap detects rank instability |
| Ablation-of-ours | CIA without Phase 2 (identifiability test disabled) | Isolates the load-bearing design-balance condition |
| Ablation-of-ours | CIA with interaction_bound = ∞ (design-balance only, no interaction check) | Isolates the conditional interaction bound |

### Why this setup validates the claim

The central claim is that the coverage mask, not the score values, determines whether an aggregate Index is identifiable and rank-stable. The InclusionMed Atlas is the right dataset because its mask is genuinely partial (three of nine cells populated), so a coverage audit has something real to bite on; a complete Atlas would make every method trivially agree. The primary metric — agreement between the certificate's predicted reconstruction error on a held-out cell and the actual error once that cell is populated — is the only metric that can distinguish a real identifiability test from a decorative one: a method that merely reports breakdowns will have no prediction to check. InclusionMed's Index tests whether the invariance condition adds anything beyond a point estimate; zero-fill tests the shrinkage prior's role; complete-case tests whether the bootstrap gap actually tracks rank instability rather than noise. The ablation removes Phase 2, so if the identifiability flag is load-bearing, disabling it should cause certificates to be issued on masks where held-out reconstruction error is large — a diagnostic, not just a score drop. The second ablation sets interaction_bound = ∞, so if the interaction bound is load-bearing, certificates should fire INVARIANT on masks where the populated cells' role×capability interaction exceeds τ and held-out reconstruction error is large.

### Expected outcome and causal chain

**vs. InclusionMed Index** — On an Atlas where one role margin is entirely empty (e.g., no benchmark populates a care-setting role), the baseline produces a single Index number and a rank ordering because it aggregates populated cells without asking whether the aggregate's expectation depends on which cells are populated. Our method instead flags NON_IDENTIFIABLE and emits only per-cell breakdowns because Phase 2 tests the mask's margins before any aggregate is formed. We expect the baseline's rank order to disagree with the bootstrap rank interval precisely on the models whose scores are highest in the populated cells, so the observable signal is a rank inversion concentrated in the top-ranked models rather than a uniform score shift.

**vs. Zero-fill cell-weighted Index** — On a case where an unpopulated cell belongs to a capability (e.g., clinical decision-making) that InclusionMed reports as systematically harder than patient communication, the baseline produces a depressed Index for every model because zero-fill silently treats 'unmeasured' as 'scored zero'. Our method instead substitutes an empirical-Bayes cell mean from the shared capability margin because the shrinkage prior assumes exchangeability within a margin, so we expect the zero-fill Index to under-rank all models by a roughly constant offset while our Index tracks the held-out cell's true value with a small, reported bias toward the populated-cell mean.

**vs. Complete-case Index** — On an Atlas where the populated cells are few enough that resampling them yields a degenerate bootstrap distribution, the baseline produces a stable-looking point Index because it never varies the coverage pattern. Our method instead reports an invariance gap and a rank interval, and — critically — when the populated set is small, the gap under-reports true sensitivity. We expect the observable signal to be that our rank interval is wide exactly where the complete-case Index looks most confident, and that the certificate correctly reads as optimistic in that regime rather than as invariant.

**vs. CIA with interaction_bound = ∞** — On a mask that satisfies design balance but whose populated cells lie on the easy side of the role×capability interaction (e.g., all populated cells are patient-communication cells), the baseline (interaction_bound = ∞) fires INVARIANT because it tests only the mask margins. Our method instead flags NON_IDENTIFIABLE_CONDITIONAL and reports the interaction magnitude because Phase 2 checks the fitted interaction term against τ. We expect the observable signal to be that the baseline's certificate agrees with held-out reconstruction error only on masks where the interaction is small, and disagrees precisely on the interaction-dominated masks our method flags.

### What would falsify this idea

If the certificate's predicted reconstruction error on held-out cells is uncorrelated with the actual error once those cells are populated — i.e., the identifiability flag fires on masks where reconstruction is accurate and stays silent where it is poor — then the design-balance condition is not measuring identifiability. A uniform gain of our Index over the zero-fill baseline across all cells, rather than a gain concentrated in the shared-margin cells the prior is designed for, would likewise indicate the shrinkage prior is not doing the work we claim. If the interaction bound never changes a certificate relative to the design-balance-only ablation, the conditional interaction check is not load-bearing.

## References

1. InclusionMed: Benchmarking AI for Inclusive Healthcare
