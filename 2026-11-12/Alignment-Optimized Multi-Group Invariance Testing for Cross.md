# Alignment-Optimized Multi-Group Invariance Testing for Cross-Species Cognitive Task Batteries

## Motivation

CogBattery compares 54 language models against condition-level human references, but its comparisons are non-identifiable across species: matching effect directions (e.g., false-alarm rates, confidence separation) does not establish that a task's difficulty parameter or error signature means the same thing for a text-adapted model as for a human participant. The root cause is structural — the battery treats human condition-level data as an external benchmark rather than as anchor indicators within a single measurement model, so no formal link constrains model and human parameters to a shared metric. Without such a link, any observed difference could be a difference in the construct, a difference in task interpretation, or an artifact of the text adaptation, and CogBattery cannot tell these apart.

## Key Insight

Condition-level human references can be re-cast as anchor indicators in a single confirmatory factor model, and under alignment optimization the difficulty parameters are identified up to a shared metric only if a small set of anchor tasks is invariant — so invariance becomes a testable, localizable property of specific tasks rather than an all-or-nothing assumption.

## Method

**What it is.** We introduce CIMBA (Confirmatory Invariance Model for Battery Alignment), a multi-group confirmatory factor-analytic measurement model fit jointly to human and model condition-level data from CogBattery's 13 tasks. Inputs: per-condition accuracy/error-type/confidence summaries for each of 54 models and the matched human source-study conditions. Outputs: task-difficulty parameters, a latent cognitive-ability factor score per group, and per-task invariance diagnostics (which tasks are anchor-invariant vs. non-invariant across groups and modalities).

**Load-bearing assumption (stated explicitly).** CIMBA assumes that a model's condition-level summary on a text-adapted task is a measurement of the same latent cognitive-ability factor eta that the human condition-level data measure, up to a task-specific loading, and that a subset of tasks (the anchor set) has group-invariant loadings on eta. If a model's condition-level error signature is instead driven by prompt-format or adaptation artifacts, then its nu and lambda are not functions of a shared eta, and the common-factor structure is misspecified for that model group. CIMBA therefore does not assume this away: it includes an explicit adaptation/format method factor (see step 2b) and admits a task to the anchor set only if its loading on that method factor is negligible and its eta loading is invariant. The shared-metric claim is thus conditional on a measured, falsifiable artifact factor.

**How it works.**
```pseudocode
# Data: G groups (1 human + 54 models), T=13 tasks, C conditions per task
# x[g,t,c] = observed condition-level error/accuracy summary
# Model (bifactor with method factor):
#   x[g,t,c] = nu[t,c] + lambda[t] * eta[g] + gamma[t] * m[g,t,c] + eps[g,t,c]
#   eta[g] ~ N(0,1) latent cognitive ability for group g
#   m[g,t,c] = adaptation/format method factor score for group g on task t condition c
#   nu[t,c] = difficulty intercept; lambda[t] = task loading on eta
#   gamma[t] = task loading on method factor m
#   eps ~ N(0, psi[t,c]) residual
# Method-factor indicators: for each model group, m[g,t,c] is computed from
#   the standardized difference between the original condition summary and
#   the summary under a prompt-format/paraphrase perturbation of the same
#   condition (available for the 54 models; human m is fixed to 0).

1. INITIALIZE: fix human group eta=0 for identification;
   set lambda[t]=1 for all t (marker-indicator start);
   nu[t,c] estimated freely per group initially;
   gamma[t] initialized at 0.1 for all t.

2. FIT baseline configural model (all parameters free across groups)
   via maximum likelihood with L-BFGS; 500 random restarts; retain best.
   2b. METHOD-FACTOR GATE: after configural fit, for each task t compute
       the standardized method-factor loading gamma[t]/sqrt(psi[t,c]).
       Flag task t as ADAPTATION-CONTAMINATED if |gamma[t]| > 0.30
       (standardized loading threshold) for any model group.
       Contaminated tasks are excluded from anchor candidacy.

3. ALIGNMENT STEP (multi-group, Asparouhov-Muthen):
   for each task t and condition c:
       compute pairwise group-to-group parameter similarity
       (normalized dot product of (nu, lambda) vectors)
   build cost matrix; solve assignment of groups to a common
       parameter ordering via Hungarian algorithm
   (cost = 1 - similarity; ensures groups share metric).

4. TEST INVARIANCE LEVELS (nested likelihood-ratio):
   M0 configural  -> M1 metric (lambda[t] equal across groups)
   -> M2 scalar (nu[t,c] equal across groups)
   For each task t, run a PARTIAL-invariance test:
       release task t from M2, refit, record Delta chi-square and
       Delta CFI; flag task t as NON-INVARIANT if Delta CFI > 0.01.
   4b. RESIDUAL-HETEROSCEDASTICITY GATE: before accepting a Delta-CFI
       flag as parameter non-invariance, test equality of residual
       variances psi[t,c] across groups using a robust scaled chi-square
       (Satorra-Bentler). If the residual-variance equality test rejects
       at alpha=0.05, recompute Delta CFI using the robust scaled
       chi-square difference; a task is flagged NON-INVARIANT only if
       the robust Delta CFI > 0.01. This prevents heteroscedasticity
       from inflating Delta-CFI and shrinking the anchor set.

5. ANCHOR SELECTION: keep tasks with Delta CFI <= 0.01 (robust where
   applicable) AND not flagged ADAPTATION-CONTAMINATED in step 2b;
   re-fit M2 with only anchor tasks constrained equal;
   report per-group eta[g] on the anchor-defined metric.

6. OUTPUT: anchor set A, per-group eta, per-task invariance flags,
   per-task adaptation-contamination flags, and difficulty-ordering
   correlation (Kendall tau) between human and each model on A.
```

**Why this design.** We chose a confirmatory factor model with alignment optimization over a simple rank-correlation comparison because CogBattery's condition-level references do not by themselves define a common metric across species, and alignment provides a principled way to establish one without assuming full invariance a priori; the cost is that alignment can mis-assign groups when the number of groups is small or tasks are few, so we retain 500 random restarts and report assignment stability. We chose partial-invariance testing per task over a single global invariance test because a global test conflates which tasks drive non-invariance and would reject the whole battery on the basis of one or two badly adapted paradigms, accepting the cost of a larger multiple-comparison burden that we address with a CFI-delta threshold rather than raw p-values. We chose to anchor the latent metric on human data (eta_human = 0, human loadings as markers) over estimating a symmetric multi-group metric because the research question is whether models land on the human scale, not whether a pooled scale exists; the cost is that human data must be complete at the condition level, which limits us to the 13 source studies CogBattery already uses. We added the explicit method factor and the residual-heteroscedasticity gate because the load-bearing assumption — that model condition summaries measure the same eta as human data — is exactly what the LLM-evaluation literature puts in doubt; making the artifact factor measured and the Delta-CFI robust to unequal residual variances turns that assumption into a testable, localizable property of specific tasks rather than an all-or-nothing premise. A domain expert would not describe this as a variant of standard multi-group CFA applied to a new dataset, because the alignment step, the per-task partial-invariance release, the method-factor gate, and the robust Delta-CFI gate are what make the cross-species metric identifiable when the groups are 55 heterogeneous systems rather than demographic strata, and because the anchor set is itself an output rather than an assumption.

**Why it measures what we claim.** The difficulty intercept nu[t,c] measures task difficulty on a shared metric because the alignment assignment plus the anchor constraint makes nu comparable across groups only when the anchor tasks' loadings are equal; this assumption fails when a text adaptation changes what the task requires (e.g., a model treats a non-target as a target because of prompt format), in which case nu reflects adaptation artifact rather than cognitive difficulty. The loading lambda[t] measures the task's dependence on the latent cognitive-ability factor because the common-factor structure assumes all group differences in task performance flow through eta; this assumption fails when a model uses a task-specific shortcut (e.g., memorized benchmark items), in which case lambda reflects task-specific variance rather than cognitive loading. The method-factor loading gamma[t] measures the degree to which a task's condition-level summary is driven by prompt-format/adaptation artifacts because it is estimated from the difference between original and perturbed condition summaries; this assumption fails when the perturbation does not capture the relevant artifact (e.g., the artifact is semantic rather than surface-format), in which case gamma underestimates contamination and the gate is too permissive. The per-task Delta-CFI measures measurement invariance because a non-significant drop in fit when constraining parameters equal is equivalent to the parameters being group-invariant under the likelihood-ratio assumption; this assumption fails when groups differ in residual variance, in which case Delta-CFI reflects heteroscedasticity rather than parameter non-invariance — which is why we recompute Delta-CFI with a robust scaled chi-square when the residual-variance equality test rejects. The anchor-defined eta[g] measures model cognitive ability relative to humans because anchoring on invariant, non-contaminated tasks makes the metric identified by human data; this assumption fails when the anchor set is empty or trivially small, in which case eta is unidentified and the reported score reflects only the arbitrary identification constraint.

## Contribution

(1) CIMBA, a multi-group confirmatory factor-analytic model with alignment optimization and per-task partial-invariance release that jointly estimates task-difficulty parameters and a latent cognitive-ability factor across human and model groups. (2) A design principle: treating condition-level human references as anchor indicators rather than external benchmarks converts cross-species comparison from an assumption into a testable, task-localizable property. (3) An open invariance-diagnostic protocol and anchor set for CogBattery's 13 tasks, reporting which tasks are invariant across the text-adapted and human paradigms and which drive non-invariance.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | CogBattery 13-task condition-level human + 54-model data | Matched condition-level human references enable cross-group metric |
| Primary metric | Anchor-set size and per-group eta on invariant tasks | Directly tests whether a human-anchored metric is identifiable |
| Baseline | Global scalar-invariance CFA (no partial release) | Tests whether per-task release rescues battery-wide rejection |
| Baseline | Rank-correlation task-ordering comparison | Tests whether alignment adds beyond raw difficulty ordering |
| Baseline | Random task groupings (CogBattery baseline) | Tests whether anchor structure beats arbitrary grouping |
| Ablation-of-ours | CIMBA without alignment step (fixed group ordering) | Isolates whether alignment prevents mis-assignment |
| Ablation-of-ours | CIMBA without method-factor gate (gamma[t] fixed to 0) | Isolates whether adaptation-contamination gate changes anchor set |
| Ablation-of-ours | CIMBA without robust Delta-CFI gate (raw Delta-CFI only) | Isolates whether heteroscedasticity inflates non-invariance flags |
| Simulation | Synthetic groups with known ground-truth invariance, permuted condition labels, unequal residual variances | Quantifies false-positive/negative rates of alignment and Delta-CFI |

### Why this setup validates the claim

The central claim is that a human-anchored, alignment-optimized partial-invariance CFA can identify a shared cognitive-ability metric across 55 heterogeneous groups where naive approaches cannot. The global scalar-invariance baseline tests the sub-claim that a single global test destroys the battery: if one or two adapted paradigms drive non-invariance, the global model should reject wholesale while our per-task release retains a nonempty anchor set. The rank-correlation baseline tests whether alignment and latent-metric estimation add anything beyond raw difficulty ordering — if our eta and the rank ordering agree everywhere, the machinery is redundant. The random-groupings baseline tests whether the anchor-defined structure carries cognitive signal rather than arbitrary partition structure. The no-alignment ablation isolates the alignment step: if removing it does not change anchor selection or eta stability, alignment is doing no work. The no-method-factor ablation isolates the adaptation-contamination gate: if removing it does not change the anchor set, the gate is not doing work. The no-robust-Delta-CFI ablation isolates the heteroscedasticity correction: if removing it does not change non-invariance flags, the correction is not doing work. The simulation study with known ground truth quantifies when the alignment assignment and Delta-CFI threshold recover true invariance versus when they are driven by heteroscedasticity, reporting false-positive and false-negative rates for the non-invariant flag. The primary metric — anchor-set size plus eta on anchors — is the right one because the claim is specifically about identifiability of a human-referenced metric, not about raw accuracy; an empty or degenerate anchor set is a direct falsification of identifiability, and eta values on a stable anchor set are the only quantities the theory licenses as interpretable.

### Expected outcome and causal chain

**vs. Global scalar-invariance CFA** — On a case where two adapted paradigms (e.g., a text-format working-memory task) shift condition-level difficulty, the global scalar model forces all 13 tasks equal and rejects the whole battery because the misfit concentrates in those two tasks. Our method instead releases each task individually and retains the remaining tasks as anchors because partial invariance localizes the violation. We expect the global model to yield no usable anchor set while our method yields a substantial anchor set (roughly 8-11 of 13 tasks) with stable eta estimates.

**vs. Rank-correlation task-ordering comparison** — On a case where a model's difficulty ordering matches humans by coincidence of surface format rather than shared cognitive demand, rank correlation reports high agreement because it ignores loading structure and treats all tasks as equally informative. Our method instead estimates per-task loadings and flags tasks whose lambda differs across groups, so coincidental ordering on non-invariant tasks is excluded from the anchor metric. We expect rank correlation to show near-uniform agreement across models while our anchor-defined eta separates models that differ specifically on invariant tasks, with a noticeable gap on the non-invariant subset but parity on anchors.

**vs. Random task groupings** — On a case where an arbitrary subset of tasks happens to be easy for both humans and models, random groupings produce a plausible-looking but unstable grouping structure because they carry no measurement model. Our method instead selects anchors by invariance diagnostics, so the grouping is constrained by cross-group parameter equality rather than chance. We expect random groupings to show high variance in grouping quality across resamples while the anchor-defined structure is stable, with anchor tasks showing consistently higher cross-group parameter similarity than random subsets.

**vs. CIMBA without alignment** — On a case where two groups' parameter orderings are permuted (a model's condition labels map differently to difficulty), the no-alignment model mis-assigns groups to a common ordering because it assumes shared labeling, producing inflated non-invariance flags and a shrunken anchor set. Our method instead solves the assignment via Hungarian matching on parameter similarity, recovering the correct ordering. We expect the no-alignment ablation to show a markedly smaller anchor set and unstable eta across restarts, while full CIMBA shows a larger anchor set and stable eta, with the gap concentrated in groups whose condition orderings diverge most from humans.

**vs. CIMBA without method-factor gate** — On a case where a model's condition-level summaries are driven by prompt-format artifacts (e.g., a paraphrase changes accuracy on a task without changing its cognitive demand), the no-gate model admits the contaminated task to the anchor set because it only tests eta-loading invariance and ignores the method factor. Our method instead estimates gamma[t] from original-vs-perturbed condition summaries and excludes tasks with |gamma[t]| > 0.30 from anchor candidacy. We expect the no-gate ablation to include contaminated tasks in the anchor set, inflating anchor-set size but degrading eta stability, while full CIMBA yields a smaller but more stable anchor set with lower cross-restart eta variance.

**vs. CIMBA without robust Delta-CFI gate** — On a case where groups differ in residual variance (e.g., some models have much noisier condition-level summaries than humans), the raw Delta-CFI flags tasks as non-invariant because heteroscedasticity inflates the chi-square difference, shrinking the anchor set. Our method instead tests residual-variance equality and recomputes Delta-CFI with a robust scaled chi-square when equality is rejected. We expect the no-robust ablation to show a smaller anchor set and more false non-invariance flags concentrated in high-residual-variance groups, while full CIMBA retains a larger anchor set with flags matching the simulation ground truth.

**vs. Simulation ground truth** — On synthetic data with known invariant and non-invariant tasks, permuted condition labels, and unequal residual variances, we compute false-positive and false-negative rates for the non-invariant flag under raw Delta-CFI and robust Delta-CFI. We expect raw Delta-CFI to show elevated false positives under heteroscedasticity, while robust Delta-CFI reduces false positives to near the nominal 5% level without materially increasing false negatives, and alignment assignment stability to be high when condition labels are permuted but low when tasks are few.

### What would falsify this idea

If the anchor set is empty or trivially small (fewer than three tasks) for the majority of models, or if anchor-defined eta is indistinguishable from eta computed on randomly chosen task subsets, then the human-anchored metric is not identifiable and the central claim fails. Additionally, if the alignment step's assignment stability is near zero across restarts, or if the method-factor gate excludes most tasks (indicating pervasive adaptation contamination), the metric rests on arbitrary ordering or an unmeasured artifact rather than a recoverable shared structure.

## References

1. CogBattery: Human-Referenced Evaluation of Cognitive Task Behavior and Structure in Language Models
