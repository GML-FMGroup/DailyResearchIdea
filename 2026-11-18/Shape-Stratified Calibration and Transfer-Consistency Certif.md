# Shape-Stratified Calibration and Transfer-Consistency Certification for NPU Kernel Optimization Benchmarks

## Motivation

AscendGYM's cost-aware kernel optimization benchmark is built on a fixed initial suite of 48 kernels in three difficulty tiers, with test shapes sampled from real workloads; as the abstract itself frames it (E1), this is an 'initial suite' whose tier structure and proposed cost metric are asserted to map to workloads but are never shown to be calibration-invariant under kernel-shape or workload-distribution shift. The structural root cause is that the suite is a hand-assembled convenience sample over an undeclared design space, so the tier-to-difficulty mapping and the induced agent ranking carry no stated coverage or stability guarantee outside the sampled shapes. A benchmark cost signal that silently changes its ranking when shapes are perturbed cannot support the cross-agent comparisons that AscendGYM, and any downstream kernel-optimization agent evaluator, depends on.

## Key Insight

If the kernel suite is drawn as a stratified sample over an explicitly declared (operator family × shape-regime × arithmetic-intensity) space, then tier-monotonicity and cost-ranking stability become auditable properties of the sampling design rather than incidental facts about 48 kernels, because held-out shape perturbations drawn from the same strata are exchangeable with the suite and therefore expose any tier or ranking instability as a measurable, attributable quantity.

## Method

**What it is.** STRAT-CERT (Stratified Tier Certification) takes an operator-family × shape-regime × arithmetic-intensity design space and an existing kernel corpus (AscendGYM's 48 kernels plus a candidate pool), and outputs (i) a calibrated tier assignment with per-tier coverage counts, (ii) a transfer-consistency certificate consisting of tier-monotonicity and cost-ranking stability statistics, and (iii) an invariance audit report flagging any tier or ranking that fails certification.

**Load-bearing assumption (stated explicitly).** The certificate treats shape perturbations drawn from within a declared (operator family × shape-regime × arithmetic-intensity) stratum as exchangeable with the native test shapes, so that bootstrap rank-spread over these perturbations yields a valid variance estimate for the cost-ranking and tier-monotonicity certificates. This assumption is load-bearing and is verified, not presumed, by the variance-components decomposition in Phase 2b: if the within-kernel shape-variance component dominates the between-kernel cost component in a cell, that cell's certificate is computed on the between-kernel component only (shape main effect regressed out) and the cell is flagged as shape-sensitive in the audit report.

**How it works.**

```text
INPUT: design space D = {operator family f} x {shape regime s} x {arith. intensity band a}
       kernel pool K (AscendGYM 48 + candidates), cost metric m (AscendGYM cost-aware score)
       perturbation operator P (shape jitter within a stratum), n_pert = 8, n_boot = 1000
       sensitivity sweep: n_pert in {2,4,8,16}, n_boot in {200,500,1000}

PHASE 1  DECLARE AND FILL THE STRATUM GRID
  for each cell (f,s,a) in D: record target weight w(f,s,a) from workload trace histogram
  assign each kernel k in K to exactly one cell via its declared (op, shape, intensity)
  coverage(k) = |{k in cell}| ; report uncovered cells (w>0, count=0)

PHASE 2  COST-CALIBRATED TIER ASSIGNMENT
  for each kernel k: measure cost c(k) on its native test case; and c_P(k,j) for j=1..n_pert
     perturbations drawn from the SAME cell (shape jitter, dtype fixed)
  tier(k) = quantile-bin of median_j c_P(k,j) within cell, then pooled across cells
            using w(f,s,a) so tiers are workload-weighted, not count-weighted

PHASE 2b  SHAPE-SENSITIVITY DECOMPOSITION (per cell)
  for each cell, fit a one-way variance-components model on the n_pert x |cell| cost matrix:
     c_P(k,j) = mu + alpha_k + epsilon_{k,j},  alpha_k ~ N(0, sigma_between^2), epsilon ~ N(0, sigma_within^2)
  report sigma_within^2 (within-kernel shape variance) and sigma_between^2 (between-kernel cost variance)
  if sigma_within^2 / (sigma_within^2 + sigma_between^2) > 0.5: mark cell SHAPE-SENSITIVE
     and compute tier(k) and all downstream certificates on the shape-adjusted cost
     c_adj(k) = median_j c_P(k,j) - (mean_j c_P(k,j) - mean_k median_j c_P(k,j))
     (i.e. regress out the shape main effect before ranking)

PHASE 3  TIER-MONOTONICITY CERTIFICATE
  for each adjacent tier pair (t, t+1): compute delta_t = median cost gap, and
     mono_t = fraction of cross-tier kernel pairs where cost(t+1) > cost(t)
  certify tier t if mono_t >= tau_mono (tau_mono = 0.95)

PHASE 4  COST-RANKING STABILITY CERTIFICATE
  for each agent A in a fixed reference panel (>=3 agents, incl. one trivial baseline):
     score_A on suite; and score_A^(b) on bootstrap resamples of held-out perturbations
     (for SHAPE-SENSITIVE cells, resample the shape-adjusted cost c_adj)
  stability_A = 1 - (rank spread of A across n_boot resamples)/(|panel|-1)
  certify ranking if min_A stability_A >= tau_rank (tau_rank = 0.90)

PHASE 5  INVARIANCE AUDIT / REPORT
  emit per-cell coverage, per-cell sigma_within^2/sigma_between^2 and SHAPE-SENSITIVE flag,
  per-tier mono_t, per-agent stability_A, and a list of (cell, tier) pairs failing certification;
  mark uncertified cells as OUT-OF-SCOPE
```

**Pilot measurement (grounds the premise).** Before any certification run, measure on the actual 48 AscendGYM kernels: (i) the empirical within-cell cost variance sigma_within^2 under shape jitter at n_pert=8, and (ii) the observed mono_t for the existing count-based three-tier grouping. These two numbers are reported as the baseline premise; if the existing tiers already satisfy mono_t >= 0.95 on all adjacent pairs, the premise that current tiers are uncertified is falsified and the audit is reported as such.

**Compute budget.** Phase 2 costs n_pert × |K| = 8 × |K| NPU kernel invocations per certification run (|K| ≈ 48 + candidate pool). Phase 4 costs n_boot × |panel| × |K| = 1000 × 3 × |K| scored evaluations, of which the per-kernel cost measurements are cached from Phase 2 and only the agent-scoring pass is repeated. The sensitivity sweep over n_pert ∈ {2,4,8,16} and n_boot ∈ {200,500,1000} is run once on a fixed reference panel to confirm that certification verdicts (the pass/fail set of (cell, tier) pairs) are identical at n_pert=8, n_boot=1000 and at the smaller budgets; if verdicts differ, the smallest budget at which they agree is reported as the operating point.

**Why this design.** We chose a workload-weighted stratified tier assignment over AscendGYM's count-based three-tier grouping because the latter lets a heavily-weighted but under-represented cell be outvoted by many low-weight kernels, accepting the cost that tier boundaries now shift when the workload trace changes and must be re-declared per deployment. We chose a perturbation-based exchangeability certificate over a single held-out split because a single split gives one number with no variance estimate, accepting the cost of n_pert × |K| extra measurements per certification run. We chose bootstrap rank-spread over parametric confidence intervals on the cost metric because the cost distribution over kernels is heavy-tailed and non-normal, accepting the cost of n_boot resamples and a coarser (rank-level) rather than absolute-scale guarantee. Crucially, this is not a retrieval or nearest-neighbor scheme: the certificate is a property of the sampling design and its perturbation distribution, not a lookup of similar kernels, and it is not a 'combine X+Y' composition — the tier assignment and stability audit are coupled through the same stratum weights, so neither can be certified independently of the other.

**Why it measures what we claim.** The per-cell coverage count measures *design-space coverage* because it assumes the declared stratum grid is a sufficient statistic for workload variation, i.e., that no workload-relevant shape variation lives outside (f,s,a); this assumption fails when a real workload uses a fused operator not in the family list, in which case coverage reports a false-complete grid. The tier-monotonicity fraction mono_t measures *difficulty-tier validity* because it assumes cost on perturbed shapes within a cell is exchangeable with cost on native shapes, so cross-tier cost separation reflects genuine optimization difficulty rather than shape memorization; this assumption fails when perturbations leave the cell's intensity band, in which case mono_t reflects shape-sensitivity rather than difficulty — the Phase 2b SHAPE-SENSITIVE flag and shape-adjusted cost c_adj are the concrete detector for this failure. The bootstrap rank-spread stability_A measures *cost-ranking transfer-consistency* because it assumes the reference agent panel spans the optimization-depth range of interest, so rank stability on the panel generalizes to unseen agents; this assumption fails for an agent far outside the panel's capability range, in which case stability_A reflects only within-panel ordering and the audit marks that agent's ranking as uncertified rather than certifying it. The certification-pass fraction measures *benchmark validity* because it assumes that passing mono_t and stability_A thresholds on held-out perturbations implies correct tier and ranking statements on held-out real workloads; this assumption fails when a suite passes certification yet misranks agents on a genuinely new workload distribution, in which case the pass fraction reflects perturbation-distribution fit rather than workload validity — the held-out real-workload validation split in the experiment is the concrete test of this equivalence.

## Contribution

(1) A stratified calibration and certification procedure (STRAT-CERT) that turns AscendGYM's fixed 48-kernel tier suite into a workload-weighted, coverage-audited suite with explicit tier-monotonicity and cost-ranking stability certificates computed on held-out shape perturbations. (2) A design principle: benchmark cost signals should be accompanied by an invariance audit that marks uncertified strata as out-of-scope, so that agent rankings are reported with a stated domain of validity rather than as unconditional. (3) An open audit report format and reference perturbation protocol for NPU kernel benchmarks that other kernel-optimization suites can adopt without re-deriving the certification statistics.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | AscendGYM 48-kernel suite + candidate pool | Real workload shapes, cost-aware metric already defined |
| Dataset (cross-benchmark) | KernelBench and TritonBench kernel suites | Tests whether the sparse-high-weight failure mode generalizes beyond a single 48-kernel suite |
| Dataset (held-out real-workload validation) | Held-out real-workload split: workload traces disjoint from those used to set w(f,s,a), with their native shapes and dtypes | Directly tests the equivalence between certification-pass and correct tier/ranking statements on unseen workloads |
| Primary metric | Fraction of (cell, tier) pairs passing certification | Directly measures the audit's verdict, not raw speed |
| Secondary metric | Held-out real-workload rank agreement: Spearman rho between certified agent ranking and ranking measured on the held-out real-workload split | Tests the predictive validity of the certificate |
| Secondary metric | Per-cell sigma_within^2 / (sigma_within^2 + sigma_between^2) from Phase 2b | Reports the empirical shape-sensitivity that motivates the decomposition |
| Baseline | AscendGYM count-based 3-tier grouping | Tests whether workload weighting fixes under-represented cells |
| Baseline | Single held-out split certification | Tests whether perturbation exchangeability beats one-shot variance |
| Baseline | Parametric CI on cost ranking | Tests whether bootstrap rank-spread handles heavy tails |
| Ablation-of-ours | Count-weighted tier assignment (drop w) | Isolates the contribution of stratum weights |
| Ablation-of-ours | No Phase 2b shape-sensitivity decomposition (certify on raw median cost) | Isolates the contribution of the variance-components repair |
| Sensitivity sweep | n_pert in {2,4,8,16}, n_boot in {200,500,1000} | Confirms certification verdicts stabilize at a tractable budget |

### Why this setup validates the claim

The central claim is that a workload-weighted, perturbation-certified stratification yields tier and ranking statements that survive audit, whereas existing groupings silently accept uncertified cells. Each baseline targets one sub-claim: the count-based AscendGYM grouping tests whether workload weighting rescues heavily-weighted but sparse cells; the single-split certification tests whether perturbation exchangeability produces a variance estimate that a one-shot split cannot; the parametric-CI baseline tests whether bootstrap rank-spread tolerates heavy-tailed cost distributions where normal-theory intervals misstate rank uncertainty. The ablation removes the stratum weights while keeping the certificate machinery, so any gain attributable to weighting is separated from gains from certification itself. The second ablation removes the Phase 2b shape-sensitivity decomposition while keeping weights and certificate machinery, so any gain attributable to the variance-components repair is separated from gains from weighting and certification. The primary metric — the fraction of (cell, tier) pairs passing certification — is the right detector because the claim is about audit verdicts, not about raw cost: if weighting and perturbation actually couple, certified coverage should rise specifically in sparse high-weight cells, which a raw-cost metric would blur. The held-out real-workload rank agreement metric closes the equivalence between certification-pass and benchmark validity: a suite that passes certification but yields low Spearman rho against the held-out real-workload ranking is a concrete instance of the failure mode the audit is meant to catch.

### Expected outcome and causal chain

**vs. AscendGYM count-based 3-tier grouping** — On a case where a high-weight cell (e.g. a dominant shape regime for a common operator family) holds only two kernels while many low-weight cells hold dozens, the count-based grouping produces tier boundaries dominated by the numerous low-weight kernels, so the high-weight cell's kernels land in tiers that do not reflect their workload importance and mono_t at the boundary adjacent to that cell falls below tau_mono. Our method instead pools quantile bins using w(f,s,a), so the sparse high-weight cell pulls its own tier boundaries, and we expect certification passes concentrated in exactly those sparse high-weight cells, with near-parity on dense low-weight cells. The pilot measurement reports the observed mono_t of the existing count-based tiers on the 48 AscendGYM kernels; we expect at least one adjacent tier pair to fall below tau_mono=0.95, grounding the premise in a measured number.

**vs. Single held-out split certification** — On a case where the held-out split happens to draw perturbations from one corner of a cell's shape jitter distribution, the single split yields one rank ordering with no spread, so it certifies a ranking that a different split would have reversed; the audit cannot flag this because it has no variance estimate. Our method instead resamples held-out perturbations n_boot times, so rank spread across resamples exposes the reversal, and we expect the single-split baseline to certify cells that our audit marks OUT-OF-SCOPE, producing a visible set of disagreements concentrated in cells whose perturbation distribution is widest (i.e. cells flagged SHAPE-SENSITIVE in Phase 2b).

**vs. Parametric CI on cost ranking** — On a case where cost is heavy-tailed within a cell (a few kernels far above the median), the parametric interval assumes near-normality and reports a narrow rank interval, certifying an ordering whose true rank spread is large. Our method instead uses bootstrap rank-spread, which reflects the tail, so stability_A drops below tau_rank for agents whose ordering depends on tail kernels; we expect the parametric baseline to over-certify by a noticeable margin on tail-heavy cells while agreeing with us on light-tailed cells.

**vs. Count-weighted tier assignment (ablation)** — On a case where the certificate machinery is intact but weights are dropped, the tier boundaries revert toward count-driven placement, so the same sparse high-weight cells that our full method rescues now fail mono_t. We expect the ablation to recover most of the ranking-stability gains (since the certificate is unchanged) but to lose the tier-monotonicity gains, isolating weighting as the driver of the coverage improvement.

**vs. No Phase 2b shape-sensitivity decomposition (ablation)** — On a case where a cell is flagged SHAPE-SENSITIVE (sigma_within^2 dominates sigma_between^2), certifying on raw median cost conflates shape-sensitivity with ranking uncertainty, so stability_A is depressed for agents whose ordering is stable in kernel quality but sensitive to shape; the decomposition regresses out the shape main effect and restores stability_A above tau_rank. We expect the ablation to under-certify exactly the SHAPE-SENSITIVE cells while agreeing with the full method on cells where sigma_between^2 dominates.

**vs. Held-out real-workload validation** — On a case where a suite passes certification on held-out perturbations but misranks agents on a genuinely new workload distribution, the certified ranking and the held-out real-workload ranking disagree; we expect Spearman rho between the two to be high (>=0.8) for cells that pass certification and low for cells marked OUT-OF-SCOPE, which is the concrete test of the equivalence between certification-pass and benchmark validity.

**Cross-benchmark check** — On KernelBench and TritonBench, we expect the same sparse-high-weight failure mode to appear: cells with few kernels but high workload weight fail mono_t under count-based grouping and pass under workload-weighted grouping, with the same direction of effect as on AscendGYM.

### What would falsify this idea

If certification-pass rates were uniform across sparse high-weight and dense low-weight cells rather than concentrated in the sparse high-weight cells where the count-based baseline fails, the claim that workload weighting is the causal driver would be wrong. Likewise, if the ablation matched the full method on mono_t, the coupling between stratum weights and tier validity would not exist; and if the no-decomposition ablation matched the full method on stability_A in SHAPE-SENSITIVE cells, the variance-components repair would not be doing the work claimed for it.

## References

1. AscendGYM: A Cost-Aware Kernel Optimization Benchmark for Real-World Workloads on Ascend NPUs
