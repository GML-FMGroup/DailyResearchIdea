# Learning the Label Space: Identifiability-Calibrated Failure Attribution for Agent Harnesses

## Motivation

Every recent trace-attribution effort inherits an expert-specified decomposition of agent behavior — TraceTrial-Bench's fixed seven harness components, OpenHands' module boundaries, the τ2 reasoning/communication split — and then maps trace evidence onto that label space without ever estimating whether the labels are identifiable from the evidence or exhaustive over real failures. The meta-gap is that the label space itself is never an object of estimation: TraceTrial-Bench's Failure Attribution module uses hand-authored mechanism chains, and Who&When reports that even SOTA reasoning models reach only 14.2% step-level accuracy, evidence that the field has been fitting evidence to a decomposition that may not be identifiable rather than testing the decomposition. This structural defect recurs across all ancestor trees because none of them treat trace evidence as a noisy observation channel whose partition into distinguishable failure signatures must be estimated and certified.

## Key Insight

If the attribution label space is defined as the coarsest partition of trace-evidence distributions that is simultaneously mutually exclusive-and-exhaustive on held-out traces and identifiable up to a calibrated equivalence class, then any expert taxonomy is either recovered as a refinement of that partition or exposed as containing labels that no observable evidence can separate — making identifiability a measurable output rather than an untested assumption.

## Method

**What it is.** ICAP (Identifiability-Calibrated Attribution Partition) takes a corpus of harness failure traces with observable evidence features (event logs, tool-call sequences, state snapshots, error signatures), and outputs (a) a learned coarsest label partition over traces, (b) per-trace posterior over learned components, and (c) a per-component identifiability certificate stating which components are distinguishable on held-out evidence and which collapse into a calibrated equivalence class.

**How it works.**

```pseudocode
Input: traces T = {t_i}, evidence encoder f (frozen LLM embedding of
       structured trace evidence), expert taxonomy E (7 components)
Hyperparams: K_max=12, alpha=0.5 (concentration), tau=0.15 (identifiability
             threshold), n_boot=200, delta=0.05, n_perm=1000 (permutation null),
             overlap_threshold=0.20 (calibrated on synthetic pairs)

Phase 1 — Evidence channel estimation:
  z_i = f(t_i)                        # evidence embedding per trace
  Fit a Dirichlet-process mixture (DPMM, alpha=0.5, K_max=12) over {z_i}
  -> posterior q(c | z_i) over components c in partition P_hat

Phase 2 — Coarsest-partition search:
  For each candidate partition P (from DPMM truncation levels):
    score(P) = MEE(P) - lambda * |P|   # MEE = mutual-exclusivity/exhaustiveness
    MEE(P) = fraction of held-out traces whose max q(c|z) exceeds
             a calibrated margin AND whose residual mass < tau
  Select P* = argmax score(P) subject to MEE(P*) >= 1 - delta

Phase 3 — Identifiability certificate (calibrated separability test):
  For each pair (c_a, c_b) in P*:
    # Directly estimate overlap of evidence distributions on held-out traces
    Fit a two-sample classifier g_{ab} (logistic regression on z) to
    distinguish held-out traces assigned to c_a vs c_b by q(c|z).
    Compute AUC_{ab} on a held-out split (5-fold cross-validation).
    Compute permutation null: shuffle labels n_perm=1000 times, record
    AUC_{ab}^{(perm)}; set overlap statistic O_{ab} = 1 - (AUC_{ab} - mean(AUC^{(perm)})) / (1 - mean(AUC^{(perm)}))
    (O_{ab} in [0,1]; O=0 means perfectly separable, O=1 means chance).
    if O_{ab} > overlap_threshold: mark {c_a,c_b} as an EQUIVALENCE CLASS
    else: mark DISTINGUISHABLE
  For each expert label e in E: map e -> nearest P* component by
    evidence-distribution overlap (using the same O statistic between
    e's traces and P* components); flag e as UNIDENTIFIABLE if its
    mapped component lies in an equivalence class of size > 1

Output: P*, posterior q(c|z_i), certificate {(c_a,c_b,status)},
        expert-label coverage report
```

**Why this design.** We chose a Dirichlet-process mixture over the evidence embeddings rather than a fixed-K latent-class model because the number of distinguishable failure signatures is exactly what is under test, accepting the cost that DPMM truncation introduces a hyperparameter K_max that must be validated on held-out traces. We chose a coarsest-partition search with an explicit mutual-exclusivity/exhaustiveness constraint rather than maximizing likelihood alone, because a higher-likelihood finer partition can always be manufactured by splitting components that carry no separable evidence, accepting the cost that we must calibrate the margin tau on a held-out split. We chose a calibrated separability test (two-sample classifier AUC with permutation null) as the identifiability signal rather than bootstrap-refit collapse frequency, because bootstrap instability of a DPMM is dominated by label-switching and sampler variance rather than by overlap of the true generative distributions (Redner & Walker 1984; Stephens 2000; Miller & Harrison 2018), accepting the cost of fitting n_perm=1000 permutation nulls per pair and a held-out split for the two-sample classifier. A domain expert would not describe ICAP as a variant of TraceTrial-Bench's mechanism-chain attribution: TraceTrial-Bench fixes the seven-component label space and maps evidence onto it, whereas ICAP estimates the label space and certifies which of TraceTrial-Bench's seven labels are recoverable from evidence at all.

**Why it measures what we claim.** The DPMM posterior q(c|z_i) measures component membership because the assumption that traces generated by the same harness-failure mechanism share an evidence-embedding distribution makes the mixture component equivalent to the mechanism; this assumption fails when two distinct mechanisms produce near-identical evidence (e.g., a governance-policy violation and a lifecycle-timeout both manifest as a truncated tool call), in which case q(c|z_i) reflects evidence similarity rather than mechanism identity. The MEE constraint measures mutual exclusivity and exhaustiveness because the assumption that held-out traces are drawn from the same mechanism distribution as training traces makes held-out margin exceedance equivalent to partition validity; this assumption fails under distribution shift to unseen harnesses, in which case MEE reflects in-distribution coverage only. The overlap statistic O_{ab} measures identifiability because the assumption that a two-sample classifier's AUC on held-out evidence embeddings reflects the true separability of the underlying evidence distributions makes O_{ab} a proxy for an equivalence class; this assumption fails when the frozen LLM embedding collapses semantically distinct inputs into near-identical representations (Ethayarajh 2019), in which case O_{ab} reflects embedding-space geometry rather than mechanism overlap. The expert-label recovery rate measures label recoverability because the assumption that a label is recoverable iff its traces concentrate on one P* component under the overlap metric makes nearest-component mapping equivalent to recoverability; this assumption fails when a label is genuinely recoverable but split across two P* components by incidental surface features, in which case recovery rate undercounts identifiable labels.

## Contribution

(1) ICAP, a data-driven attribution framework that estimates the failure-label partition jointly with per-trace posteriors and emits an explicit identifiability certificate over component pairs and expert labels. (2) A design principle: expert taxonomies should be evaluated as refinements of an estimated coarsest identifiable partition, and labels that fall inside a calibrated equivalence class should be reported as unidentifiable rather than attributed. (3) A reusable certificate format and expert-label coverage report that makes the identifiability of any harness-failure taxonomy a measurable, reportable quantity on trace benchmarks.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Harness failure traces from Trial by Trace (95 tasks, multi-harness) | Provides observable evidence plus fixed seven-component expert taxonomy |
| Dataset (additional) | Harness failure traces from two additional harnesses (e.g., OpenHands and τ2) | Tests certificate on at least one widely-used taxonomy label |
| Synthetic dataset | Controlled failure mechanisms with known overlap structure (e.g., 10 mechanisms, 5 pairs with overlap coefficient 0.1–0.9) | Grounds certificate validity in controlled setting |
| Primary metric | Expert-label recovery rate under certified equivalence classes | Directly tests which taxonomy labels are identifiable from evidence |
| Secondary metric | Precision/recall of collapsed pairs vs. synthetic ground-truth equivalence classes | Validates certificate against known overlap |
| Baseline | Fixed-K latent-class attribution (TraceTrial-Bench style) | Tests whether estimating label space beats fixing it |
| Baseline | Posterior-entropy identifiability scoring | Tests whether calibrated separability beats entropy as identifiability signal |
| Baseline | Likelihood-maximizing fine partition (no MEE constraint) | Tests whether coarsest-partition constraint is necessary |
| Ablation-of-ours | ICAP with entropy signal replacing calibrated separability test | Isolates the identifiability-certificate mechanism |
| Ablation-of-ours | ICAP with bootstrap-collapse signal replacing calibrated separability test | Tests whether bootstrap collapse is a valid identifiability signal |

### Why this setup validates the claim

The central claim is that the number of distinguishable harness-failure components is itself under test, and that a coarsest-partition search with a calibrated separability certificate recovers exactly the expert labels that evidence can actually separate. The fixed-K baseline tests the first sub-claim: if forcing seven components is fine, then estimating the partition buys nothing, and ICAP's recovery rate should match it on every subset. The entropy baseline and the likelihood-maximizing partition test the second sub-claim: if either simpler signal matches ICAP's certificate on the traces where mechanisms genuinely overlap in evidence, then the DPMM-plus-separability machinery is unjustified. The ablation isolates the certificate: it keeps the DPMM and MEE constraint but swaps the identifiability signal, so any gain attributable to calibrated separability rather than to partition estimation shows up as a difference between ICAP and the ablation. The synthetic dataset provides ground-truth equivalence classes, allowing us to measure precision/recall of collapsed pairs and verify that the certificate recovers true overlap structure as a function of tau and K_max. Expert-label recovery rate is the right metric because the claim is about which labels are recoverable, not about raw attribution accuracy on a fixed label space — a method that confidently mislabels an unidentifiable component would score well on accuracy but badly here. The falsifiable structure is thus: gains must concentrate on the traces where evidence distributions overlap, and parity must hold where they do not.

### Expected outcome and causal chain

**vs. Fixed-K latent-class attribution (TraceTrial-Bench style)** — On a case where a governance-policy violation and a lifecycle-timeout both manifest as a truncated tool call, the fixed-K baseline produces confident but arbitrary splits between the two labels because it must assign every trace to one of seven fixed components regardless of whether evidence separates them. Our method instead collapses those two labels into a certified equivalence class because the calibrated separability test yields high overlap O_{ab} (AUC near chance), so we expect a noticeable gap in expert-label recovery concentrated on the overlapping-mechanism subset, with parity on traces whose evidence signatures are cleanly separated.

**vs. Posterior-entropy identifiability scoring** — On a case where the DPMM is miscalibrated (overconfident posteriors on a small component), the entropy baseline marks the component as identifiable because low entropy reads as certainty, even though the evidence distributions overlap. Our method instead flags the component as an equivalence-class member because the two-sample classifier AUC is near chance despite low entropy, so we expect our certificate to disagree with entropy precisely on the miscalibrated components, producing higher recovery on the subset where entropy falsely certifies separability and parity elsewhere.

**vs. Likelihood-maximizing fine partition (no MEE constraint)** — On a case where a single mechanism's traces split into two embedding clusters by incidental surface features (e.g., tool-call phrasing), the likelihood baseline manufactures a finer partition that scores higher on held-out likelihood but assigns no held-out trace a clean margin. Our method instead rejects the finer partition because MEE falls below the calibrated threshold, so we expect our selected partition to be coarser than the likelihood optimum on exactly those traces, with higher expert-label recovery and fewer spurious components.

**vs. ICAP with entropy signal (ablation)** — On a case where two mechanisms share evidence but the DPMM happens to be stable under resampling for optimization reasons, the entropy-signal ablation marks them distinguishable because posteriors are sharp, whereas full ICAP's calibrated separability test also yields high overlap O_{ab} and marks them as an equivalence class. Here the two disagree, and the ablation's value shows up on the complementary case where entropy is sharp but evidence distributions overlap. We expect full ICAP to outperform the ablation only on that complementary subset, confirming the certificate rather than the partition estimator carries the gain.

**vs. ICAP with bootstrap-collapse signal (ablation)** — On a case where two well-separated mechanisms collapse across bootstrap refits purely because the sampler lands in different modes, the bootstrap-collapse ablation marks them as an equivalence class (false positive), whereas full ICAP's calibrated separability test yields low overlap O_{ab} and marks them distinguishable. Conversely, on a case where two genuinely overlapping mechanisms appear stable because the sampler consistently finds the same merged mode, the bootstrap-collapse ablation marks them distinguishable (false negative), whereas full ICAP correctly marks them as an equivalence class. We expect full ICAP to have higher precision/recall of collapsed pairs on the synthetic dataset and higher expert-label recovery on the real datasets, confirming that calibrated separability is a more valid identifiability signal than bootstrap collapse.

### What would falsify this idea

If ICAP's expert-label recovery gain over the fixed-K baseline is uniform across overlapping-mechanism and cleanly-separated subsets rather than concentrated where evidence distributions overlap, then the coarsest-partition-plus-certificate mechanism is not doing the work we claim and the central argument fails. Likewise, if the calibrated separability test does not recover the true equivalence classes on the synthetic dataset (precision/recall below 0.8), then the certificate is not a valid identifiability signal.

## References

1. Trial by Trace: Benchmarking AI Agent Harnesses for Reliability and Failure Attribution
2. OpenHands: An Open Platform for AI Software Developers as Generalist Agents
3. SWE-Bench Pro: Can AI Agents Solve Long-Horizon Software Engineering Tasks?
4. Which Agent Causes Task Failures and When? On Automated Failure Attribution of LLM Multi-Agent Systems
5. OpenAgents: An Open Platform for Language Agents in the Wild
6. SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering
7. BigCodeBench: Benchmarking Code Generation with Diverse Function Calls and Complex Instructions
8. SWE-Bench+: Enhanced Coding Benchmark for LLMs
