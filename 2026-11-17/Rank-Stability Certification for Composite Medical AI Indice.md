# Rank-Stability Certification for Composite Medical AI Indices

## Motivation

Composite medical AI benchmarks such as the InclusionMed Index collapse heterogeneous role- and capability-tagged scores into a single scalar, yet InclusionMed's own within-benchmark analysis shows that a model's rank varies with the role or capability considered and that the ten highest-ranked models all score higher on patient communication than on clinical decision-making (E1, E6). The structural root cause is that the Index is defined as a fixed weighted aggregation over a currently-selected benchmark set, so its induced model ordering is an artifact of that particular set and weighting rather than a property that survives benchmark-set refinement; no published Index reports whether its ordering is invariant to admissible perturbations such as dropping a benchmark or reweighting a role-capability cell. MedXpertQA (E7) and OpenMIBOOD (E8) similarly report leaderboards as point estimates without certifying order stability under perturbation of their benchmark composition.

## Key Insight

If a scalar composite Index is to be published as a ranking, its induced total order must be a fixed point of the admissible refinement group (leave-one-benchmark-out deletions and role-capability cell reweightings), and this invariance is directly testable by computing rank-correlation between the Index and each refinement variant, so unstable orderings can be detected before publication rather than discovered post hoc.

## Method

**A. What it is.** Rank-Stability Certification (RSC) takes as input a model-by-benchmark score matrix S (models m, benchmarks b in the Atlas, with role-capability cell labels c(b)) and a declared aggregation rule A (fixed cell weights w_c and within-cell averaging), and outputs either a certified Index with a stability certificate (Kendall tau and rank-biased overlap against every admissible refinement) or a rejection when the ordering fails the declared tolerance. It is a wrapper around A, not a new aggregator.

**B. How it works.**

```pseudocode
INPUT: score matrix S[m,b], cell labels c(b), weights w_c, tolerance tau_min=0.85, RBO p=0.9, B=2000 bootstrap reps, coverage threshold k=2
OUTPUT: certified Index I, certificate Cert (two independent branches), or REJECT

# --- Precondition: coverage sufficiency ---
1. For each cell c, let n_b(c) = number of benchmarks b with c(b)=c.
   If any n_b(c) < k, mark cell c as SPARSE and exclude its benchmarks from the structural branch (leave-one-out); report SPARSE cells in the certificate as coverage-limited, not stability-limited.

# --- Baseline Index ---
2. I = aggregate(S, w_c)                    # baseline Index per cell-weighted mean
3. rank_I = argsort_desc(I)

# --- Structural branch: admissible refinements ---
4. REFINEMENTS = []
   for each benchmark b in Atlas where c(b) is not SPARSE:
       S_b = S with column b deleted
       REFINEMENTS.append(aggregate(S_b, w_c))      # leave-one-benchmark-out
   for each cell c in {roles x capabilities}:
       for each w' in {0.5*w_c, 2.0*w_c} over c:    # cell-reweighting perturbation
           REFINEMENTS.append(aggregate(S, w'))
5. STABILITY = []
   for each R in REFINEMENTS:
       STABILITY.append( (kendall_tau(rank_I, argsort_desc(R)),
                          rbo(rank_I, argsort_desc(R), p)) )

# --- Sampling branch: bootstrap ---
6. BOOT = []
   for i in 1..B:
       S_i = bootstrap_resample_models_and_items(S)
       BOOT.append(kendall_tau(rank_I, argsort_desc(aggregate(S_i, w_c))))

# --- Two independent certificates ---
7. Cert_struct = {min_tau_struct, min_rbo_struct, per-refinement table}
   Cert_samp   = {bootstrap_CI_of_tau, min_boot_tau}

# --- Rejection rule (separate branches) ---
8. if min_tau_struct < tau_min:
       return REJECT_STRUCTURAL with Cert_struct and per-refinement diagnostic table
   if min_boot_tau < tau_min:
       return REJECT_SAMPLING with Cert_samp
   else:
       return I, Cert = {Cert_struct, Cert_samp, SPARSE_cells}
```

The certificate is published alongside the Index and reports, for each role slice and capability slice, the minimum Kendall tau and RBO across all refinements (structural branch) and the bootstrap confidence interval of tau (sampling branch), so a reader can see exactly which refinement would overturn which pairwise ordering and which branch drove any rejection. Cells with fewer than k=2 benchmarks are explicitly flagged as coverage-limited and excluded from the structural branch.

**C. Why this design.** We chose leave-one-benchmark-out and bounded cell-reweighting as the admissible refinement group over arbitrary random perturbations because InclusionMed's Atlas is a curated set whose selection is itself an audited decision, so the perturbations that matter are exactly the ones an auditor could plausibly make; accepting the cost that refinement families outside this group are not certified. We chose Kendall tau as the primary statistic over Spearman or Pearson because the Index is ordinal in use (it is used to rank models), and Kendall tau is defined on concordant/discordant pairs, which maps directly onto the pairwise orderings the Index claims; accepting the cost that tau is less sensitive to magnitude changes than Pearson. We chose RBO as a secondary statistic over a single scalar because RBO is top-weighted, and the top of a leaderboard is where deployment decisions are made; accepting the cost of an extra hyperparameter p=0.9. We chose a bootstrap over a closed-form variance because the aggregation is nonlinear in the score matrix once cell weights are perturbed, and a closed-form would require distributional assumptions the benchmark scores do not satisfy. A domain expert would not describe RSC as a variant of standard leaderboard reporting (as in MedXpertQA's 18-model evaluation or OpenMIBOOD's 24-method comparison) because those report a single point-estimate ordering, whereas RSC reports the ordering's invariance class and rejects publication when invariance fails — the deliverable is a certificate, not a score.

**D. Why it measures what we claim.** The minimum Kendall tau across leave-one-benchmark-out refinements measures the order-stability of the Index because the assumption is that the benchmark set is exchangeable with respect to the ordering claim, i.e., no single benchmark is load-bearing for the ranking; this assumption fails when one benchmark is the only source of signal for a capability cell (as in InclusionMed's three-of-nine coverage), in which case the tau drop reflects coverage sparsity rather than Index instability. To separate these causes, the structural branch excludes cells with fewer than k=2 benchmarks (SPARSE cells) and reports them as coverage-limited; the remaining tau drop is attributed to order instability. The cell-reweighting tau measures robustness of the ordering to the arbitrary choice of cell weights because the assumption is that admissible weights lie within a factor of two of the published weights; this assumption fails when the published weights are themselves unprincipled, in which case the tau reflects weight arbitrariness rather than Index quality. The bootstrap confidence interval measures sampling stability of the ordering because the assumption is that benchmark items are exchangeable within a benchmark; this assumption fails under item-level contamination or leakage, in which case the CI is too narrow and the certificate overstates stability. The two branches are reported independently so that a reader can see whether rejection is driven by structural refinement or sampling noise.

## Contribution

(1) A rank-stability certification procedure that treats the composite Index as a function on a refinement group and publishes a certificate (minimum Kendall tau, top-weighted RBO, bootstrap CI) or rejects the Index when the ordering is not invariant within a declared tolerance. (2) A design principle that a scalar medical AI Index is only publishable as a ranking if its induced order is a fixed point of admissible benchmark-set refinements, operationalized as a per-refinement diagnostic table that localizes which benchmark or cell is load-bearing for which pairwise ordering. (3) An audit artifact, the refinement-diagnostic table, that reports for each role and capability slice which leave-one-out deletion or cell reweighting would overturn the Index ordering, usable as a standard appendix for any future InclusionMed release.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | InclusionMed Atlas (5 benchmarks, 3/9 cells) | Curated, audited, sparsely covered — stresses stability |
| Dataset | Synthetic score matrix with known ground-truth ordering and injected load-bearing structure (e.g., one benchmark per cell for a subset of cells) | Allows attribution of tau drops to structural cause rather than coverage sparsity |
| Primary metric | Min Kendall tau across admissible refinements (structural branch) | Directly encodes the pairwise ordering claim |
| Secondary metric | Min Kendall tau across bootstrap resamples (sampling branch) | Separates sampling from structural instability |
| Baseline | Point-estimate InclusionMed Index | The status quo RSC wraps and certifies |
| Baseline | Bootstrap-only rank CI | Isolates refinement-stability from sampling-stability |
| Baseline | Frequency-of-top-k across perturbations | Naive perturbation reporting without a rejection rule |
| Ablation-of-ours | RSC without cell-reweighting refinements | Tests whether reweighting branch carries the signal |
| Ablation-of-ours | RSC without coverage-sufficiency precondition (k=1) | Tests whether sparsity-driven tau drops are absorbed into stability verdict |

### Why this setup validates the claim

The central claim is that the InclusionMed Index ordering is not invariant across admissible auditor refinements, and that a certificate can detect this before publication. The point-estimate Index baseline tests the sub-claim that a single ordering hides refinement sensitivity: if RSC's certificate rejects or flags orderings that the point estimate silently publishes as stable, the sub-claim holds. The bootstrap-only baseline tests the sub-claim that sampling variance is not the dominant instability source: if refinement tau drops well below bootstrap tau, the instability is structural, not noise. The frequency-of-top-k baseline tests whether a rejection rule adds anything beyond descriptive perturbation counts — RSC should reject cases that frequency reporting would pass. The ablation isolates whether the cell-reweighting branch, versus leave-one-benchmark-out alone, drives rejection decisions. The synthetic ground-truth experiment tests whether the coverage-sufficiency precondition (k=2) correctly separates sparsity-driven tau drops from genuine order instability: when a cell has a single load-bearing benchmark, the structural branch should flag it as SPARSE rather than reject for instability, and when a cell has redundant benchmarks but the ordering is unstable, the structural branch should reject. Min Kendall tau is the right primary metric because the Index is used ordinally; it maps onto concordant/discordant pairs the Index actually asserts, and a drop below tolerance is a direct, falsifiable signature of order non-invariance.

### Expected outcome and causal chain

**vs. Point-estimate InclusionMed Index** — On a case where the Atlas covers only three of nine cells, the point-estimate Index publishes a full model ranking as if all cells were equally supported, because its aggregation treats missing cells as weight-neutral and reports no invariance check. Our method instead recomputes the ordering under leave-one-benchmark-out over the non-sparse cells and under bounded cell reweighting, so a benchmark that is the sole signal source for a covered cell is flagged as SPARSE and excluded from the structural branch, while benchmarks in cells with k≥2 are perturbed. We expect the certificate to show a marked tau drop concentrated on the sparsely covered role-capability slices (reported as coverage-limited), with near-parity on densely covered slices.

**vs. Bootstrap-only rank CI** — On a case where one benchmark dominates a capability cell, the bootstrap-only baseline resamples items within that benchmark and returns a narrow tau interval, because item resampling cannot detect the absence of alternative benchmarks for that cell. Our method instead perturbs the benchmark set itself via leave-one-benchmark-out on non-sparse cells, so removing a load-bearing benchmark in a cell with k≥2 produces a large tau shift the bootstrap never sees. We expect refinement tau to fall noticeably below the bootstrap tau interval specifically on the non-sparse cells, indicating structural rather than sampling instability.

**vs. Frequency-of-top-k across perturbations** — On a case where a mid-ranked pair swaps under one cell-reweighting perturbation, the frequency baseline reports a low swap count and reads as stable, because it counts top-k membership without a declared tolerance or rejection rule. Our method instead applies a declared tau_min to the full ordering and rejects when any admissible refinement crosses it, so a single pairwise inversion below the top-k is enough to trigger rejection. We expect RSC to reject a non-trivial fraction of orderings that the frequency baseline would pass, concentrated where cell weights are perturbed by the factor-of-two bound.

**vs. Synthetic ground-truth** — On a synthetic score matrix with a known ground-truth ordering and a known load-bearing structure (e.g., one benchmark per cell for a subset of cells), the coverage-sufficiency precondition should flag the load-bearing cells as SPARSE and exclude them from the structural branch, while the remaining cells with redundant benchmarks should yield high tau if the ordering is stable. We expect the structural branch to reject when the injected instability is in a non-sparse cell, and to report SPARSE (not reject) when the instability is due to a single-benchmark cell, thereby separating the two causes.

### What would falsify this idea

If the certificate's tau values are uniformly high across all refinements and slices — so that RSC never rejects and the refinement tau distribution matches the bootstrap tau distribution — then the claim that admissible refinements threaten Index orderings is wrong, and RSC reduces to redundant reporting. Equally, if rejection is driven entirely by sampling noise rather than by refinement structure, or if the coverage-sufficiency precondition fails to separate sparsity-driven tau drops from genuine instability on the synthetic ground-truth experiment, the central distinction collapses.

## References

1. InclusionMed: Benchmarking AI for Inclusive Healthcare
