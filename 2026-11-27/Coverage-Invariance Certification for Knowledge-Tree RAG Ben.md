# Coverage-Invariance Certification for Knowledge-Tree RAG Benchmarks

## Motivation

Knowledge-centric RAG benchmarks such as DRBench Pro score deep-research agents by rubric coverage on a fixed Wikipedia-derived knowledge tree, treating the resulting breadth×depth coverage number as an intrinsic property of the agent. RAGGED shows that the same RAG system's performance swings substantially with retrieval depth and reader noise-sensitivity, so a coverage score computed at one retrieval configuration is a point estimate with unmeasured configuration variance — yet DRBench Pro never sweeps depth or retriever/reranker choice, and RAGGED sweeps configurations but never sweeps knowledge representations. Neither work can answer whether a coverage score is a stable property of the agent or an artifact of the (tree sample, retrieval configuration) pair, and no existing metric separates the two variance sources.

## Key Insight

A coverage score is only a valid agent property if it is invariant under perturbations that should not change what the agent knows — the variance of coverage across a jointly swept (knowledge-subgraph, retrieval-depth, retriever, reranker) grid decomposes into a configuration-identifiable component and a representation-sampling component, and the ratio of these components certifies whether a reported coverage number is attributable to the agent rather than to its evaluation configuration.

## Method

**A. What it is**
We introduce CIC (Coverage-Invariance Certificate), a benchmark-auditing procedure. Its inputs are a DRBench Pro case (a knowledge tree T, a query q, and its evidence-grounded rubric set R) and a target deep-research agent A. Its output is a certificate quadruple (C_bar, V_conf, rho, sep) where C_bar is the configuration-averaged coverage, V_conf is the configuration-attributable variance component, rho is the representation-to-configuration variance-component ratio, and sep is the separability indicator (the fraction of total variance explained by the representation×configuration interaction term), plus a three-way label {PASS, FAIL, NON-SEPARABLE} at a stated tolerance.

**B. How it works**

```pseudocode
INPUT: tree T, query q, rubrics R, agent A
HYPERPARAMS: depths D = {2,4,8,16,32}; retrievers = {BM25, Contriever, E5-large};
             rerankers = {none, bge-reranker-large}; subgraph samples S = 20;
             tolerance tau = 0.05 (absolute coverage units);
             separability threshold gamma = 0.20 (interaction fraction of total variance);
             seeds per cell = 3

# Phase 1 — representation perturbation
for s in 1..S:
    G_s = sample_connected_subgraph(T, anchor=q, size_budget=B)   # B = |T|/4
    R_s = {r in R : relation(r) is an edge in G_s}                 # rubric subset induced by G_s

# Phase 2 — configuration perturbation (RAGGED-style sweep, applied per subgraph)
for each G_s, each d in D, each retriever m, each reranker k, each seed e in 1..3:
    docs = retrieve(m, q, depth=d, corpus=G_s)
    if k != none: docs = rerank(k, q, docs)
    report = A(q, docs)                                            # agent sees only q
    cov[s,d,m,k,e] = geo_mean(breadth(report,R_s), depth(report,R_s)) # DRBench Pro coverage

# Phase 3 — two-way variance-component model (representation x configuration)
# Fit a crossed random-effects model on the cell means cov_bar[s,d,m,k] = mean_e cov[s,d,m,k,e]:
#   cov_bar[s,c] = mu + a_s + b_c + g_{s,c} + eps,  a_s ~ N(0, V_repr), b_c ~ N(0, V_conf),
#   g_{s,c} ~ N(0, V_int), eps ~ N(0, V_resid),  c = (d,m,k) indexes the |D|*|retrievers|*|rerankers| configuration cells
# Estimate variance components (V_repr, V_conf, V_int, V_resid) by restricted maximum likelihood (REML)
# using the lme4-style formulation; report 95% CIs via parametric bootstrap (200 resamples).
V_total = V_repr + V_conf + V_int + V_resid
sep     = V_int / V_total                                          # interaction fraction
rho     = V_repr / (V_conf + eps)                                  # ratio of variance components
C_bar   = mean over all cells of cov_bar

# Phase 4 — certificate (three-way outcome)
if sep > gamma:
    label = NON-SEPARABLE        # representation and configuration effects are not separable; certificate uninterpretable
elif V_conf <= tau^2 AND rho <= 1:
    label = PASS
else:
    label = FAIL
return (C_bar, V_conf, rho, sep, label)
```

**C. Why this design**
We chose a full factorial sweep over depth, retriever, and reranker rather than a single 'best' configuration because RAGGED's central finding is that reader noise-sensitivity, not retriever quality, governs stability; a single configuration cannot reveal that sensitivity, accepting the cost of a |D|·|retrievers|·|rerankers|·S·3 evaluation budget. We chose connected-subgraph sampling with a fixed size budget B = |T|/4 over random rubric deletion because random deletion destroys the knowledge-chain structure that DRBench Pro's depth metric depends on, accepting the cost that sampled subgraphs may omit some rubric relations entirely (handled by restricting R_s). We chose a crossed random-effects variance-component model with an explicit interaction term over a raw variance ratio because a raw ratio conflates representation sampling noise with configuration sensitivity and absorbs the interaction into whichever factor is averaged last, accepting the cost of fitting a mixed model and reporting bootstrap CIs. We chose a fixed absolute tolerance tau over a relative one so certificates are comparable across domains with different coverage scales, and we chose to make non-separability a third certificate outcome rather than an uninterpreted residual so that the pass/fail rule is never evaluated on confounded marginals. A domain expert would not describe this as 'RAGGED + DRBench Pro': RAGGED's metrics are defined within a fixed corpus and never perturb the knowledge representation, and DRBench Pro's coverage is defined at a single configuration; CIC's object is the joint (representation × configuration) variance surface and the identifiable variance-component ratio rho with an explicit separability gate, which neither prior method produces.

**D. Why it measures what we claim**
`cov[s,d,m,k,e]` measures the agent's rubric-grounded knowledge coverage because the DRBench Pro rubric-to-relation mapping assumes each rubric item is satisfied iff the report expresses the corresponding tree relation; this assumption fails when an agent paraphrases a relation without naming its constituent entities, in which case coverage reflects lexical surface overlap rather than knowledge. `V_conf` measures configuration sensitivity of coverage because the crossed random-effects model with a fixed subgraph effect assumes the only varying factor within a subgraph is the retrieval pipeline; this assumption fails when an agent's output is stochastic across calls, in which case V_conf absorbs sampling noise and overstates configuration sensitivity — mitigated by fixing agent temperature and averaging over 3 seeds per cell. `rho` measures the dominance of representation-sampling variance over configuration variance because the variance-component model assumes representation and configuration effects are separable (interaction term small); this assumption fails when subgraph sampling interacts with retrieval depth (e.g., small subgraphs saturate at low depth), in which case the interaction fraction sep exceeds gamma and the certificate returns NON-SEPARABLE rather than a misleading ratio. `sep` measures the degree to which representation and configuration effects are separable because the crossed random-effects model partitions total variance into additive components plus an explicit interaction; this assumption fails when the interaction is large, in which case sep itself signals that the certificate is uninterpretable and the pass/fail rule is not applied.

## Contribution

(1) A coverage-invariance certificate (CIC), a benchmark-auditing procedure that jointly perturbs knowledge representation (connected subgraph sampling) and retrieval configuration (depth, retriever, reranker) and outputs a variance-ratio certificate for DRBench Pro-style coverage scores. (2) A design principle: coverage scores should be reported with a configuration-variance band and a representation-to-configuration variance ratio, since a bare coverage number is not attributable to the agent without it. (3) An open auditing harness over the DRBench Pro cases that produces per-case certificates and flags agents whose reported coverage is configuration-fragile.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | DRBench Pro cases (144 cases, 24 domains) | Knowledge trees with relation-grounded rubrics enable representation perturbation |
| Primary metric | CIC certificate: V_conf, rho, sep, three-way label at tau=0.05, gamma=0.20 | Directly tests configuration-attributable variance vs representation variance with an explicit separability gate |
| Baseline | Single-config DRBench Pro coverage | Tests whether one configuration hides instability |
| Baseline | RAGGED stability sweep (fixed corpus) | Tests whether corpus perturbation adds identifiable signal |
| Baseline | Naive random-rubric-deletion variance | Tests whether connected-subgraph sampling matters |
| Ablation-of-ours | CIC without rho (V_conf only) | Isolates whether variance-ratio decomposition adds value |
| Ablation-of-ours | CIC without separability gate (sep ignored) | Tests whether the interaction term changes pass/fail decisions |
| Calibration set | 6 agents with known reliability: 3 deliberately degraded (e.g., truncated context, no reranker, temperature=1.0) and 3 strong (e.g., GPT-4-class with full context) | Calibrates tau and the rho<=1 boundary via ROC/decision boundary rather than fixing arbitrary constants |
| Budget analysis | Full grid on 24 stratified cases (one per domain); reduced grid (depths {4,16}, 2 retrievers, 10 subgraphs, 1 seed) on remaining 120 cases | Justifies reduced protocol as a cheap surrogate by comparing rho and pass/fail decisions |

### Why this setup validates the claim

The central claim is that RAG evaluation must certify both representation-sampling stability and configuration sensitivity jointly, and that a single configuration or a fixed-corpus sweep cannot reveal the interaction. The single-config DRBench Pro baseline tests the sub-claim that one configuration is insufficient: if coverage at a single configuration is stable but CIC's V_conf is large, the certificate exposes hidden instability the baseline cannot see. The RAGGED-style fixed-corpus sweep tests the sub-claim that corpus perturbation is necessary: if its variance is small while CIC's rho is large, representation sampling dominates and the fixed-corpus view is misleading. The random-rubric-deletion baseline tests the sub-claim that connected-subgraph sampling preserves the depth metric's knowledge-chain structure, so its variance should be inflated by structural destruction rather than genuine representation sensitivity. The V_conf-only ablation tests whether the rho ratio is doing real work: if pass/fail decisions are unchanged when rho is removed, the decomposition is decorative. The separability-gate ablation tests whether the interaction term changes decisions: if labels are unchanged when sep is ignored, the gate is unnecessary. The calibration set of 6 agents with known reliability provides ground truth for tuning tau and the rho<=1 boundary, and the budget analysis tests whether a reduced grid yields the same certificate decisions as the full grid. Because the primary metric is the certificate itself, not raw coverage, the design directly detects the predicted effect rather than proxying it.

### Expected outcome and causal chain

**vs. Single-config DRBench Pro coverage** — On a case where the agent's report quality is sensitive to retrieval depth (e.g., a multi-hop case where shallow retrieval omits a chain link), the baseline produces one coverage number at a default configuration and labels the agent adequate because that single cell happens to be high. Our method instead sweeps depth, retriever, and reranker per subgraph and computes V_conf from the crossed random-effects model, because a single cell cannot reveal the variance surface. We expect a noticeable V_conf elevation on multi-hop cases with parity on single-hop cases, and a subset of agents that pass the single-config check but fail the CIC certificate.

**vs. RAGGED stability sweep (fixed corpus)** — On a case where the agent's coverage depends on which subgraph of the knowledge tree is sampled (e.g., a domain where some relation chains are only reachable through a few edges), the baseline holds the corpus fixed and reports low stability variance because it never perturbs representation. Our method instead samples connected subgraphs and computes rho from the variance-component model, because representation variance is invisible under a fixed corpus. We expect rho to be large on domains with sparse tree connectivity and small on dense domains, producing a domain-dependent certificate pattern rather than a uniform pass.

**vs. Naive random-rubric-deletion variance** — On a case where deleting rubrics at random severs a knowledge chain, the baseline's depth metric collapses because the chain structure is destroyed, inflating variance for reasons unrelated to representation sensitivity. Our method instead uses connected-subgraph sampling with a fixed size budget, preserving chain structure so that variance reflects genuine representation sampling. We expect the naive baseline to show uniformly higher variance across all domains, while our V_repr concentrates on domains where subgraph choice genuinely changes which relations are reachable.

**vs. CIC without rho (V_conf only)** — On a case where configuration variance is moderate but representation variance is large, the V_conf-only ablation passes the certificate because it ignores rho, mislabeling an unstable agent as certified. Our method instead requires rho <= 1 alongside V_conf <= tau^2, because representation dominance is a distinct failure mode. We expect the ablation to pass agents that the full certificate fails, concentrated on sparse-connectivity domains, demonstrating that the variance ratio changes pass/fail decisions.

**vs. CIC without separability gate (sep ignored)** — On a case where the interaction between subgraph sampling and retrieval depth is large (e.g., small subgraphs saturate at low depth), ignoring sep would evaluate the pass/fail rule on confounded marginals and could produce a spurious PASS or FAIL. Our method instead returns NON-SEPARABLE when sep > gamma, because the certificate is uninterpretable under strong interaction. We expect the gate to change labels on a subset of cases, concentrated on domains with small trees or high depth sensitivity, demonstrating that the interaction term is load-bearing.

**Calibration set (6 agents with known reliability)** — We expect the deliberately degraded agents to yield higher V_conf and rho than the strong agents, and the ROC/decision boundary to show that tau=0.05 and rho<=1 separate the two groups with high AUC (e.g., >0.8), justifying the thresholds rather than fixing them arbitrarily.

**Budget analysis (full vs. reduced grid)** — We expect the reduced grid (depths {4,16}, 2 retrievers, 10 subgraphs, 1 seed) to yield rho and pass/fail decisions that agree with the full grid on at least 80% of cases, justifying the reduced protocol as a cheap surrogate for large-scale auditing.

### What would falsify this idea

If V_conf and rho are uniformly small across all domains and agents, so that the CIC certificate passes everything the single-config baseline already passes, then the joint variance surface carries no identifiable signal and the central claim is wrong. If instead the certificate's pass/fail decisions are driven entirely by rho with V_conf contributing nothing, the configuration-sensitivity sub-claim is unsupported and the method reduces to representation-sampling noise auditing.

## References

1. DRBench Pro: A Knowledge-Centric Benchmark for Professional Deep Research Agents
