# Language-Stratified Compositional Difficulty for Code Benchmarks

## Motivation

Longtail-SWE (2025) evaluates issue-resolution tasks across 13 underrepresented languages but reports difficulty as a per-task scalar pass rate, so the benchmark cannot establish whether difficulty composes across language strata or whether a model's weakness in one language predicts difficulty in another. This is a structural gap: pass rates are marginal statistics that discard the language-stratified structure of the task population, and without a compositional decomposition there is no quantity that supports language-level difficulty comparison or transfer. The field's trajectory toward multi-language benchmarks makes this the recurring bottleneck—every new language stratum adds a column of independent scalars rather than a coordinate in a shared difficulty space.

## Key Insight

If each task's difficulty is modeled as a bilinear form over a language-specific skill vector and a language-agnostic task vector, then the matrix of expected pass rates across language×task pairs is low-rank, which makes per-language difficulty identifiable up to a shared scale and turns cross-language transfer into a matrix-completion problem that marginal pass rates cannot solve.

## Method

**What it is.** LSCD (Language-Stratified Compositional Difficulty) takes a model's binary pass/fail matrix over tasks×languages and returns, for every (task, language) cell, a compositional difficulty estimate `d_{t,l} = σ(μ + a_t + b_l + u_t^T v_l)` where `a_t` is the language-agnostic task difficulty, `b_l` is the language-specific offset, and `u_t^T v_l` is the interaction that captures whether task t is disproportionately hard in language l. Inputs: the pass matrix `Y ∈ {0,1}^{T×L}` (T=202 tasks, L=13 languages, with missing cells), a language-similarity graph `G` over the 13 strata (edges from shared syntax/ecosystem lineage), and a rank hyperparameter `k=2`. Output: a completed difficulty matrix, per-language difficulty profiles, and a transfer score between any pair of languages.

**Load-bearing assumption (stated explicitly).** The method assumes that the task×language interaction matrix is well-approximated by a rank-2 bilinear form whose latent language vectors `V` are identifiable from the 202×13 binary pass matrix with many missing cells, such that `cos(V[l,:], V[l',:])` is a valid measure of empirical cross-language difficulty co-variation rather than an artifact of the graph-Laplacian prior. This assumption is load-bearing because the transfer score is only meaningful if `V` is identified by the likelihood, but the method imposes a graph-Laplacian prior on `V` and then normalizes `||V[l,:]||_2 = 1`; with L=13 and k=2 the latent space has only ~26 free parameters constrained by a prior that already encodes the answer, so the posterior over `V` may be dominated by the Laplacian rather than the data. We make this assumption explicit and add a calibration/verification story (below) that separates the data-driven signal from graph-induced structure.

**How it works.**

```pseudocode
Input: pass matrix Y (T×L, entries in {0,1,?}), language graph G, rank k=2, λ=0.1
Initialize a_t=0, b_l=0, U∈R^{T×k}, V∈R^{L×k} ~ N(0,0.01)
Repeat until convergence (max 200 iters, tol 1e-4):
  # E-step: impute missing cells with current model expectation
  for each (t,l) with Y[t,l] = ?:
      Y_hat[t,l] = σ(μ + a_t + b_l + U[t,:]·V[l,:])
  # M-step: fit bilinear logistic model with graph-Laplacian penalty on V
  for each observed or imputed cell (t,l):
      p = σ(μ + a_t + b_l + U[t,:]·V[l,:])
      gradient step on a_t, b_l, U[t,:], V[l,:] with learning rate 0.05
  # graph regularization: pull V[l,:] toward neighbors in G
  V ← V - λ · L_G · V    where L_G is the Laplacian of G
  # identifiability: fix scale by normalizing ||V[l,:]||_2 = 1 for all l
Return d_{t,l} = σ(μ + a_t + b_l + U[t,:]·V[l,:]) for all cells
Return language transfer score S(l,l') = cos(V[l,:], V[l',:])
```

**Calibration / verification of the transfer score (inside the existing structure).** To test whether `S(l,l')` reflects empirical co-difficulty rather than graph geometry, we add a two-stage verification procedure that does not alter the core training above:

1. **Graph-free reference score.** Fit the same bilinear model with a flat ridge penalty only (λ=0, i.e., no Laplacian term) and compute `S_flat(l,l') = cos(V_flat[l,:], V_flat[l',:])`. Independently, compute an empirical co-difficulty residual correlation `R(l,l')` directly from held-out cells: for each pair (l,l'), take the set of tasks observed in both languages, compute the residual `r_{t,l} = Y_{t,l} - σ(μ + a_t + b_l)` using the additive fit only, and define `R(l,l') = corr(r_{·,l}, r_{·,l'})` over that task set. This `R` is computed without any graph.
2. **Graph as hypothesis, not prior.** Treat the language graph `G` as an external predictor to be tested against `R`: report the correlation between graph adjacency (or graph distance) and `R(l,l')` across all 91 language pairs. The graph is only supported if this correlation is positive and stable; otherwise the graph is not encoding empirical co-difficulty.
3. **λ-sweep.** Run the full LSCD fit for λ ∈ {0, 0.01, 0.05, 0.1, 0.5, 1.0} and for each λ report (a) held-out log-loss and (b) the correlation between `S_λ(l,l')` and the graph-free reference `R(l,l')`. If `S_λ` correlation with `R` is stable as λ→0 and degrades as λ grows, the transfer score is data-driven; if it tracks graph adjacency instead, the score is graph-driven. This sweep is the concrete test of the load-bearing assumption.

**Why this design.** We chose a bilinear low-rank model over a per-language independent logistic regression because independent fits cannot share statistical strength across the 13 strata, accepting the cost that the rank-k assumption may be violated for tasks whose difficulty is genuinely language-idiosyncratic (we diagnose this via residual analysis). We chose a graph-Laplacian penalty over a flat ridge penalty because the 13 languages are not exchangeable—they cluster by syntax and ecosystem—and we want `V[l,:]` to borrow strength from structurally similar languages, accepting the cost that a mis-specified language graph will bias transfer scores toward graph neighbors; the λ-sweep and graph-free reference above are how we detect and bound that bias. We chose EM-style imputation over listwise deletion because Longtail-SWE has only 202 tasks and many model×language cells are missing, accepting the cost that imputed cells contribute to the fit and can amplify model bias; we mitigate this by down-weighting imputed cells by 0.5. We chose a fixed rank k=2 over cross-validated rank because with T=202 and L=13 the interaction signal is small and higher rank overfits, accepting the cost that genuinely multi-dimensional language interactions are collapsed. This is not a variant of standard item-response theory (IRT): IRT assumes a single population of items and persons with no language stratum, and it has no mechanism for cross-language transfer; our graph-regularized bilinear term is what produces the transfer score, and a domain expert would not describe graph-regularized bilinear IRT as 'IRT with a language dummy' because the Laplacian couples the latent language vectors, which is the quantity we care about.

**Why this measures what we claim.** The quantity `a_t` measures language-agnostic task difficulty because it is the task effect shared across all 13 strata under the assumption that the bilinear interaction is mean-zero across languages; this assumption fails when a task is trivially easy in 12 languages and only hard in one (e.g., a language-specific build-system quirk), in which case `a_t` reflects an average that no language actually experiences. The quantity `b_l` measures language-specific offset because it is the language main effect after marginalizing over the benchmark's task distribution; this assumption fails when the 202 tasks are not representative of the language's ecosystem, in which case `b_l` reflects the benchmark's task sample rather than the language. The quantity `u_t^T v_l` measures compositional interaction because it is the residual after additive effects, under the assumption that the rank-k bilinear form captures the dominant interaction direction; this assumption fails when interactions are high-rank, in which case the residual is absorbed into noise and the interaction is underestimated. The transfer score `cos(v_l, v_l')` measures cross-language difficulty similarity because it is the cosine between latent language vectors under the assumption that the graph-Laplacian penalty identifies the latent space up to rotation; this assumption fails when the language graph is disconnected or misspecified, in which case the cosine reflects graph structure rather than empirical difficulty co-variation—the λ-sweep and graph-free reference `R(l,l')` above are the concrete checks for this failure mode.

## Contribution

(1) A language-stratified compositional difficulty model, LSCD, that replaces per-task scalar pass rates with a bilinear decomposition into language-agnostic task difficulty, language-specific offset, and a graph-regularized interaction term, yielding a completed difficulty matrix and a cross-language transfer score. (2) A design principle: graph-Laplacian regularization over a language-similarity graph is what makes the latent language vectors identifiable and transferable with only 13 strata and 202 tasks, whereas independent per-language fits cannot share strength. (3) An open diagnostic tool that reports per-cell difficulty residuals and flags tasks whose difficulty is language-idiosyncratic (rank-k violation), enabling benchmark maintainers to audit whether Longtail-SWE's 202 tasks actually span the language strata compositionally.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Longtail-SWE pass matrix (202 tasks × 13 languages) | Native multi-language strata with realistic missing cells |
| Primary metric | Held-out cell pass-probability log-loss on masked cells | Directly tests imputation and interaction recovery |
| Secondary metric | Correlation between transfer score S(l,l') and graph-free empirical co-difficulty R(l,l') | Tests whether transfer score is data-driven, not graph-driven |
| Baseline | Per-language independent logistic regression | No cross-strata sharing; tests statistical-strength claim |
| Baseline | Standard 2PL IRT with language dummy | Additive only; tests interaction/transfer claim |
| Baseline | Flat-ridge bilinear (no graph) | Tests whether language graph adds real signal |
| Ablation-of-ours | LSCD with shuffled language graph | Isolates graph structure from bilinear fit |
| Ablation-of-ours | LSCD λ-sweep (λ ∈ {0, 0.01, 0.05, 0.1, 0.5, 1.0}) | Tests stability of transfer score as λ→0 |

**Masking protocol (pre-registered).** We hold out 20% of observed cells (≈ 202×13×0.2 ≈ 525 cells), stratified by language coverage: for each language, we mask a fixed fraction (20%) of its observed cells, ensuring that sparse languages (few observed cells) still contribute held-out cells. We repeat this masking over 5 random seeds and report mean ± std of held-out log-loss and transfer-score correlation. This makes the small-sample log-loss gaps statistically distinguishable from noise.

### Why this setup validates the claim

The central claim is that language-stratified compositional difficulty is recoverable from a sparse pass matrix, and that the recovered interaction term and transfer scores reflect real cross-language structure rather than artifacts of the fit. Masking held-out cells and scoring log-loss is the right test because it forces the model to predict exactly the cells it cannot see, so any gain must come from sharing information across tasks and languages rather than from memorizing observed cells. The per-language independent logistic baseline tests the sub-claim that sharing statistical strength across the 13 strata matters: if independent fits match LSCD, the compositional sharing story is empty. The standard 2PL IRT with a language dummy tests the sub-claim that a rank-k bilinear interaction is needed beyond additive effects: if additive IRT matches LSCD, the interaction term is not earning its keep. The flat-ridge bilinear baseline tests whether the graph-Laplacian penalty contributes real signal or just smooths noise. The shuffled-graph ablation is the decisive control: it preserves the model's capacity and rank while destroying the language-similarity structure, so any gap between LSCD and the shuffled ablation isolates the graph as the source of the transfer-score validity. The λ-sweep and the graph-free reference score R(l,l') directly test the load-bearing assumption: if the transfer score S(l,l') correlates with R(l,l') even as λ→0, the score is data-driven; if it instead tracks graph adjacency, the score is graph-driven. Together these comparisons form a falsifiable chain: LSCD must beat independent fits (sharing), beat additive IRT (interaction), beat flat-ridge (graph structure), and beat the shuffled graph (graph correctness), and the metric is sensitive to all four because each failure mode produces systematically worse held-out probabilities on the cells where the missing mechanism bites.

### Expected outcome and causal chain

**vs. Per-language independent logistic regression** — On a case where a language has only a handful of observed tasks and many missing cells, the independent baseline produces near-chance or wildly overfit probabilities for that language because it has no way to borrow strength from the other twelve strata. Our method instead pools task difficulty `a_t` across all languages and lets the graph-coupled latent vectors `V[l,:]` regularize the sparse language's offset, so we expect a clear log-loss gap concentrated on the sparsest languages (large gap on low-coverage strata, near-parity on well-covered ones).

**vs. Standard 2PL IRT with language dummy** — On a case where a task is easy in twelve languages but disproportionately hard in one (e.g., a build-system quirk), additive IRT produces a single task difficulty plus a language offset and therefore mispredicts the interaction cell, because its additive structure cannot represent a task×language residual. Our method instead captures that residual in `u_t^T v_l`, so we expect a noticeable gap on the subset of cells with high estimated interaction magnitude and parity on purely additive tasks.

**vs. Flat-ridge bilinear (no graph)** — On a case where a language has few observations but shares syntax and ecosystem lineage with a well-covered neighbor, the flat-ridge model fits that language's latent vector from its own sparse data and drifts toward noise, because nothing couples it to its structural neighbor. Our method instead pulls `V[l,:]` toward graph neighbors via the Laplacian, so we expect the gap to concentrate on sparse languages that have dense graph neighbors, with parity on languages whose neighbors are equally sparse.

**vs. LSCD with shuffled language graph** — On a case where two languages are genuinely similar in empirical difficulty co-variation, the shuffled-graph ablation couples each language to arbitrary partners, so its transfer scores and sparse-language predictions reflect random graph structure rather than real similarity, because the Laplacian no longer encodes the true adjacency. Our method instead preserves the syntax/ecosystem graph, so we expect its transfer-score ranking to correlate with held-out co-difficulty while the shuffled variant's does not, and a log-loss gap on graph-adjacent sparse languages.

**vs. LSCD λ-sweep and graph-free reference** — On the load-bearing assumption, we expect the correlation between S_λ(l,l') and the graph-free empirical co-difficulty R(l,l') to be stable (within ±0.05) as λ→0 and to degrade as λ grows, indicating that the transfer score is data-driven at small λ and graph-driven at large λ. If instead S_λ tracks graph adjacency regardless of λ, the transfer score is an artifact of the prior.

### What would falsify this idea

If LSCD's held-out log-loss gain over independent per-language fits is uniform across sparse and dense languages rather than concentrated on low-coverage strata, the cross-strata sharing claim is wrong. If the shuffled-graph ablation matches LSCD on both held-out log-loss and transfer-score correlation with empirical co-difficulty, the graph is not doing the work we claim and the transfer score is not measuring cross-language difficulty similarity. If the λ-sweep shows that S_λ(l,l') correlation with R(l,l') does not improve as λ→0, the transfer score is graph-driven rather than data-driven, falsifying the load-bearing assumption.

## References

1. Longtail-SWE: High-Quality SWE Tasks for Underrepresented Languages
