# Coverage-Audited Environment Evolution for Repository-Level Coding Agents

## Motivation

Environment-generation pipelines for long-horizon coding agents — RepoFactory's Environment Evolution Loop (10,159 environments across 17 domains) and SetUpAgent-style automated benchmark construction (SWEE-Bench, SWA-Bench) — both control difficulty and diversity through construction heuristics (validity, structural coupling, implementation scope; historically accurate dependency setup) and implicitly treat scaling environment counts as a coverage guarantee. SetUpAgent explicitly warns that evaluating on few, popularity-selected repositories risks distributional mismatch (up to 60% lower agent success on broader corpora), yet neither branch quantifies how representative a generated environment set is of the real-world task distribution it is meant to proxy. The root cause is structural: the Environment Evolution Loop optimizes a multiplicative objective with no term that references a held-out target distribution, so redundant environments are as valuable as novel ones and under-covered regions of task space are invisible to the proxy signal.

## Key Insight

A generated environment set is only a valid training proxy insofar as its embedding support covers the held-out real-world task corpus, and this coverage deficit is computable from the environment's own specification/dependency/requirement signature before any model rollout, making it a cheap additive term the existing proxy loop can directly optimize.

## Method

**What it is.** COVEE (COVerage-constrained Environment Evolution) augments RepoFactory's Environment Evolution Loop with a distributional coverage audit. Inputs: a candidate environment pool produced by RepoFactory, a held-out real-world task corpus (issues + repository snapshots not used in generation), and the current Environment Evolution Loop objective. Outputs: a per-environment coverage score, a quantified representativeness report (coverage, redundancy, and per-region deficit), and a revised environment set selected under a coverage-constrained objective.

**Load-bearing assumption (stated explicitly).** The audit treats a 192-d PCA-concatenated signature of language spec, dependency graph, and issue/requirement text, partitioned by k-means on the held-out corpus, as a sufficient statistic for 'task region' — i.e., proximity in this signature space predicts which agent skills a task exercises, so per-cell pool-to-corpus count ratios measure semantic coverage rather than surface-feature coverage. This assumption is load-bearing and is not assumed away: it is calibrated and tested by the signature-validity procedure in Phase 2b, and the audit reports a measured 'skill residual' alongside each cell's deficit so that a cell is only treated as a coverage target when both its deficit and its probe-difficulty variance are high.

**How it works.**

```pseudocode
# Phase 1: signature extraction (per environment / per corpus task)
for env in pool + heldout_corpus:
    s_lang  = normalize(language_spec(env))            # API surface + type signatures
    s_dep   = dependency_graph_signature(env)          # node types, depth, fan-in/out histogram
    s_req   = requirement_signature(env)               # issue text embedding + touched-file set
    z(env)  = concat(PCA_64(s_lang), PCA_64(s_dep), PCA_64(s_req))   # 192-d signature

# Phase 2: coverage audit against held-out corpus H
fit_kmeans(H_signatures, K=200)                        # K chosen by silhouette on H
for each cell c in {1..K}:
    cover[c] = (# pool envs in c) / (# heldout tasks in c)   # per-cell coverage ratio
    deficit[c] = max(0, 1 - cover[c])
redundancy(env) = 1 - (min distance to any OTHER pool env in z-space)
representativeness = mean_c min(1, cover[c])           # reported alongside N envs

# Phase 2b: signature-validity calibration (rollout-free skill grounding)
# Fixed reference model M_ref (the same checkpoint used for all probes; no training).
# Stratified subsample H_probe ⊂ H: 20 tasks per cell for the 20 cells with the
# largest held-out mass (400 tasks total; if fewer than 20 tasks in a cell, take all).
for task in H_probe:
    d(task) = 1 - pass@1(M_ref, task)                  # probe difficulty in [0,1]
for each cell c with >= 5 probed tasks:
    skill_residual[c] = std({d(task) : task in H_probe ∩ c})   # difficulty heterogeneity
    # cells with skill_residual[c] < 0.05 are treated as difficulty-homogeneous
# Component ablation: refit k-means and recompute deficit using each of
# {s_lang only, s_dep only, s_req only, full 192-d} to report which axes carry
# the coverage signal (reported as deficit-profile correlation across variants).

# Phase 3: coverage-constrained Environment Evolution Loop
objective(env) = validity * structural_coupling * implementation_scope
                 * (1 + lambda * deficit[cell(env)])   # lambda = 0.5
                 / (1 + mu * redundancy(env))           # mu = 0.3
# Environment Evolution Loop accepts env iff objective improves AND
#   (a) it fills a cell with deficit[c] > 0 AND skill_residual[c] >= 0.05, OR
#   (b) it replaces an env whose redundancy exceeds 0.7
# Selection: greedy set-cover over deficit cells with skill_residual >= 0.05,
#            tie-broken by objective.
```

**Why this design.** We chose a fixed k-means partition of the held-out corpus over a learned density model because the audit must be computable without model rollouts (the whole point of the proxy loop) and because a partition gives an interpretable per-region deficit that can be reported as a number, accepting the cost that K must be chosen by silhouette and may mis-split rare regions. We chose to penalize redundancy by nearest-neighbor distance in signature space rather than by exact duplicate detection because construction heuristics can produce near-duplicates that differ in surface syntax but not in task content, accepting the cost that the distance threshold (0.7) is a hyperparameter. We chose a multiplicative coverage multiplier over an additive penalty because the existing objective is multiplicative and we want coverage to be unable to rescue an invalid environment, accepting the cost that lambda interacts nonlinearly with the other terms. We added the Phase 2b calibration because the load-bearing assumption — that signature proximity predicts agent skill — is exactly what SetUpAgent's distributional-mismatch finding and MaintainCoder's static-metric warning put in doubt; the calibration uses a fixed reference model on a 400-task stratified subsample (no training, no rollouts on pool environments) to measure per-cell difficulty heterogeneity, so a cell is only admitted as a coverage target when it is both under-covered and skill-heterogeneous. This is not a variant of SetUpAgent: SetUpAgent automates dependency setup and test execution to expand the number of repositories, whereas COVEE computes a representativeness statistic against a held-out corpus and feeds it back into the generation loop — a domain expert would describe SetUpAgent as a benchmark-construction tool and COVEE as a coverage audit of any such tool.

**Why it measures what we claim.** `deficit[c]` measures distributional under-coverage of task space because the k-means cells are fit on held-out real-world tasks and the ratio of pool-to-corpus counts per cell is a plug-in estimator of the density ratio between generated and target distributions; this assumption fails when the 192-d signature discards a task-relevant axis (e.g., a new library ecosystem), in which case deficit reflects surface-feature coverage rather than semantic coverage. `redundancy(env)` measures marginal information of an environment because nearest-neighbor distance in signature space is a proxy for the novelty of its specification/dependency/requirement combination; this assumption fails when two environments with distant signatures are nevertheless solved by the same agent skill, in which case redundancy overstates novelty. `representativeness` measures the fraction of held-out task regions with at least proportional environment support because the min(1, cover[c]) cap treats any cell at parity as covered; this assumption fails when a cell contains heterogeneous tasks, in which case representativeness overstates coverage of that cell's sub-regions. `skill_residual[c]` measures the difficulty heterogeneity of a cell because the standard deviation of reference-model pass@1 across probed tasks in the cell is a direct, rollout-cheap estimate of how much agent skill varies within the cell; this assumption fails when the 400-task probe subsample is too small or too skewed to estimate the cell's difficulty spread, in which case skill_residual reflects probe-sampling noise rather than true within-cell heterogeneity, and the admission rule (a) may reject a genuinely heterogeneous cell or admit a homogeneous one.

## Contribution

['A coverage-audit module (COVEE) that embeds each generated environment by language specification, dependency graph, and requirement signature, and reports a quantified representativeness metric against a held-out real-world task corpus alongside raw environment counts.', 'A coverage-constrained reformulation of the Environment Evolution Loop objective that penalizes redundant environments and rewards filling under-covered regions, with a greedy set-cover selection step that is computable without model rollouts.', 'An empirical finding on whether coverage-audited environment selection changes downstream long-horizon coding agent performance relative to count-scaled selection at matched environment budget.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | RepoFactory pool + held-out NL2Repo-Bench repos | Held-out repos give uncovered regions to audit |
| Dataset (secondary) | Held-out SWA-Bench application-style tasks | Tests audit on application-vs-library axis |
| Primary metric | Per-cell deficit reduction on held-out corpus | Directly measures the coverage claim |
| Secondary metric | Representativeness (mean_c min(1, cover[c])) | Reported alongside N envs as the audit statistic |
| Secondary metric | Skill-residual-weighted deficit reduction | Tests whether admitted cells are skill-heterogeneous |
| Baseline | RepoFactory (original EEL objective) | Tests whether coverage multiplier adds value |
| Baseline | SetUpAgent-derived SWEE-Bench environments | Tests coverage audit on a different generator |
| Baseline | Uniform random env selection from pool | Tests whether selection, not pool size, drives gains |
| Ablation-of-ours | COVEE minus deficit term (λ=0) | Isolates coverage term from redundancy term |
| Ablation-of-ours | COVEE minus skill-residual gate (admit all deficit>0 cells) | Isolates the Phase 2b calibration |
| Ablation-of-ours | Signature-component ablation (s_lang only / s_dep only / s_req only / full) | Tests which axes carry the coverage signal |
| Calibration | Fixed reference model M_ref, 400-task stratified probe (20 per top-20 cells) | Rollout-cheap skill grounding; no training |

### Why this setup validates the claim

The central claim is that RepoFactory's Environment Evolution Loop optimizes validity, structural coupling, and implementation scope without tracking which regions of real-world task space remain under-covered, so a coverage-constrained objective should shift the accepted environment set toward deficit cells and yield a more representative set. RepoFactory-original is the direct test: if COVEE's coverage multiplier is doing anything, the revised set must show lower per-cell deficit on held-out NL2Repo-Bench repos than the unmodified loop accepts, and downstream agent performance should improve most on the under-covered subsets where the original loop under-samples. SetUpAgent-derived environments test generality: if the audit is a genuine statistic over any generator, applying it to a differently-constructed pool should also expose deficit cells and improve selection, not just RepoFactory's specific heuristics. Random selection isolates whether the gain comes from the coverage-constrained objective or merely from re-sampling the pool. The λ=0 ablation separates the coverage term's contribution from redundancy penalization, so a gain that survives only with both terms is not attributable to coverage. The skill-residual gate ablation and the signature-component ablation directly test the load-bearing assumption: if the gate changes which cells are admitted and the component ablation shows that one axis (e.g., s_req) carries most of the deficit signal, then the audit is not merely a popularity reweighting of surface features. Per-cell deficit reduction is the right primary metric because it is exactly what the method claims to optimize; downstream agent gains are the consequence we predict but not the thing we are measuring.

### Expected outcome and causal chain

**vs. RepoFactory (original EEL objective)** — On a case where RepoFactory accepts environments concentrated in a few high-frequency software domains (its multiplicative objective rewards validity and scope, which are easier to satisfy in well-represented domains), the baseline produces a pool whose held-out cells for less common dependency graph shapes and issue types stay near-zero coverage, because its objective has no term that observes the target distribution. Our method instead flags those cells via the deficit ratio and admits environments that fill them, because the coverage multiplier scales objective by (1 + λ·deficit) and the greedy set-cover selection prioritizes deficit cells with skill_residual ≥ 0.05. We expect a noticeable per-cell deficit reduction concentrated in low-density cells, with near-parity on already-covered cells, and downstream agent gains largest on the under-covered subsets; the skill-residual-weighted deficit reduction should exceed the unweighted reduction if the calibration is selecting skill-heterogeneous cells.

**vs. SetUpAgent-derived SWEE-Bench environments** — On a case where SetUpAgent expands repository count but inherits the popularity bias of its source repositories, the baseline produces a large pool that is broad in repository identity yet narrow in dependency-graph and requirement signatures, so held-out application-style tasks (SWA-Bench-like) remain under-covered. Our method instead computes signatures on that pool and reports a deficit profile showing application regions under-supported, because the audit is generator-agnostic and does not assume RepoFactory's construction. We expect the audit to expose deficit cells in the SetUpAgent pool and selection to improve representativeness, with the gap between the two generators' pools narrowing after coverage-constrained selection; the signature-component ablation should show whether the application-vs-library gap is carried by s_dep or s_req.

**vs. Uniform random selection from the pool** — On a case where the candidate pool already contains a few environments in rare cells but they are drowned out by the majority, random selection produces a set whose per-cell coverage tracks pool frequency rather than held-out frequency, because selection is blind to the target distribution. Our method instead weights by deficit and greedily covers under-supported cells, because the objective and selection rule are explicitly coverage-aware. We expect the same pool size to yield lower deficit and higher representativeness than random selection, with the advantage growing as the pool's frequency skew increases.

### What would falsify this idea

If COVEE's per-cell deficit reduction is uniform across cells rather than concentrated in the low-coverage cells the audit flags, or if the λ=0 ablation matches the full method, then the coverage term is not the causal driver and the central claim fails. A second falsifier: if the skill-residual gate ablation shows no difference in admitted cells, or if the signature-component ablation shows that deficit profiles are dominated by a single surface axis (e.g., s_req) with no skill-residual correlation, then the 192-d signature is not a sufficient statistic for task-relevant coverage and the audit measures surface-feature coverage rather than semantic coverage.

## References

1. From Repositories to Training Environments: Scaling Long-Horizon Coding Agents with Environment Evolution Loop
