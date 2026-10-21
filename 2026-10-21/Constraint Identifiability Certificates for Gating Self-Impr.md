# Constraint Identifiability Certificates for Gating Self-Improvement Signals

## Motivation

Across both annotation-based evaluation (AgentIF) and self-modification search (HGM), a finite human- or sample-derived signal—human-annotated constraints or a CMP point estimate—is treated as sufficient ground truth to drive downstream decisions, with no check that the constraint is recoverable from the agent's observable behavior distribution. This shared meta-gap means unidentifiable constraints silently enter evaluation and search, so improvements are measured against signals the agent could never have learned to follow; HGM's CMP guidance inherits this flaw because its expansion score is only as reliable as the behavior-constraint coupling it assumes.

## Key Insight

A constraint is usable as a self-improvement signal only if the behavior-conditioned distribution of the agent's outputs is statistically distinguishable from the same distribution under a stated intervention that removes the constraint, and this distinguishability is testable with a contrastive probe whose sample complexity is predictable from the intervention's effect size — so identifiability can be certified before the signal is allowed to steer search.

## Method

**What it is.** CIC (Constraint Identifiability Certificate) is a pre-gate that takes a candidate constraint set C, a behavior sampler from the agent π, and an intervention model I (a specified family of prompts that remove one constraint while holding the rest fixed), and returns a binary certificate cert(c) ∈ {0,1} plus a sample-complexity bound n*(c) for each constraint c. Only constraints with cert(c)=1 are passed to the downstream self-improvement loop (e.g., HGM's CMP estimator or AgentIF-style evaluation).

**Load-bearing assumption (stated explicitly).** The certificate is valid only if the prompt-level intervention I that removes constraint c changes only c's behavioral signature, so that Δ̂(c) = p̂_neg − p̂_pos is an unbiased estimate of c's individual average treatment effect on the rule-based violation indicator φ. This assumption fails when I accidentally removes correlated constraints (e.g., dropping a 'cite sources' clause also changes length and tool-call patterns scored by other constraints), in which case Δ̂(c) reflects a bundled effect rather than c's individual contribution. To make this assumption testable rather than assumed, the method adds a manipulation check (Phase 2b below) that validates each intervention before any certificate is issued.

**How it works.**
```pseudocode
INPUT: constraint set C, agent π, intervention model I, confidence δ=0.05, margin ε=0.1
OUTPUT: certificate cert(c) for each c ∈ C, and gate mask for downstream signal

for each constraint c ∈ C:
    # Phase 1: contrastive probe construction
    D_pos ← sample n₀=64 behavior traces from π on prompts satisfying c
    D_neg ← sample n₀=64 behavior traces from π on prompts where I removes c, all else fixed
    # Phase 2: two-sample test on a fixed behavioral feature map φ (constraint-violation indicator per trace, computed by a rule-based checker)
    #   φ(trace) ∈ {0,1}: 1 if trace violates c, 0 if satisfies c
    p̂_pos ← mean φ(D_pos);  p̂_neg ← mean φ(D_neg)
    Δ̂(c) ← p̂_neg − p̂_pos            # estimated intervention effect
    # Phase 2b: manipulation check — verify I did not perturb non-target constraints
    for each c' ∈ C \ {c}:
        Δ̂(c') ← p̂_neg(c') − p̂_pos(c')   # same φ applied to c' on the same traces
    if max_{c'≠c} |Δ̂(c')| > ε/2:
        cert(c) ← 0                    # intervention bundled; withhold certificate
        continue
    # Phase 3: sample-complexity certificate via cluster-robust two-sample bound
    #   traces are clustered by prompt; use block bootstrap with B=1000 resamples
    #   n*(c) ← minimal per-arm sample size such that the (1−δ) block-bootstrap CI excludes 0 at margin ε
    n*(c) ← cluster_robust_n_star(ε, δ, B=1000)
    if Δ̂(c) ≥ ε  and  n₀ ≥ n*(c):
        cert(c) ← 1
    else:
        cert(c) ← 0
    # Phase 4: gate downstream signal
    if cert(c)=0: remove c from any CMP estimate or annotation-based score
```
The probe is repeated with two independent feature maps (rule-based checker and a held-out LLM judge with its own validation set) and the certificate requires agreement; disagreement triggers a third probe at n₀=256. The downstream self-improvement loop (HGM-style CMP aggregation or AgentIF constraint adherence) receives only certified constraints, so its guidance signal is restricted to constraints whose behavioral signature survives the intervention.

**Why this design.** We chose a rule-based behavioral feature map φ over a learned classifier because the certificate's validity must not itself depend on a learned model whose calibration is unknown, accepting the cost that φ must be hand-specified per constraint type. We chose a cluster-robust block-bootstrap bound over a naive i.i.d. Hoeffding bound because traces are correlated within a prompt and the naive bound underestimates the true sample requirement, accepting that the block bootstrap is more expensive to compute but yields a sample-complexity number n*(c) that can be compared against the actual probe budget before any downstream signal is used. We chose a fixed intervention model I (prompt-level constraint removal) over a full causal graph because a specified intervention family makes the identifiability claim falsifiable and reproducible, accepting that constraints whose effect is only visible under multi-constraint interactions will be misclassified; the Phase 2b manipulation check makes this failure mode detectable rather than silent. We chose to gate rather than reweight because reweighting would let unidentifiable constraints leak into the search through small nonzero weights, which is exactly the failure mode in HGM's CMP point estimate that this method targets.

**Why it measures what we claim.** The quantity Δ̂(c) = p̂_neg − p̂_pos measures the behavioral recoverability of constraint c because the assumption that the intervention model I removes only c and that the feature map φ is a sufficient statistic for c-violation makes Δ̂(c) equal the average treatment effect of c on observable behavior; this assumption fails when I accidentally removes correlated constraints, in which case Δ̂(c) reflects a bundled effect rather than c's individual contribution. The Phase 2b manipulation check measures whether I is constraint-isolating because the assumption that non-target constraints' φ values are statistically unchanged under I makes |Δ̂(c')| ≈ 0 a necessary condition for an unbiased Δ̂(c); this assumption fails when the intervention perturbs correlated constraints, in which case the check withholds the certificate rather than issuing a bundled one. The sample-complexity term n*(c) measures whether the probe budget is sufficient to distinguish Δ(c) from zero at margin ε because the block-bootstrap bound assumes traces are exchangeable within prompts and resamples at the prompt level; this assumption fails when traces are correlated across prompts, in which case n*(c) may still underestimate the true sample requirement and the certificate may be issued prematurely. The agreement between the rule-based checker and the held-out LLM judge measures whether the behavioral signature is robust to the choice of feature map because the assumption that both maps detect the same violation makes disagreement a signal of map-specific artifacts; this assumption fails when the LLM judge has a systematic bias on a constraint type, in which case the certificate reflects judge bias rather than unidentifiability.

## Contribution

(1) A constraint identifiability certificate (CIC) that couples a contrastive two-sample probe with a sample-complexity bound to gate which constraints may enter a self-improvement or evaluation signal. (2) A design principle: self-improvement search and annotation-based evaluation should be preceded by an intervention-based recoverability check, and the certificate's rejection rate is a measurable property of the constraint set rather than of the agent. (3) A reusable probe protocol with a rule-based checker plus a held-out LLM judge whose agreement is required for certification, applicable to any constraint set with a specified intervention model.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | AgentIF constraint-annotated agentic tasks, restricted to a pre-registered subset of 8 constraint types with existing rule-based checkers (format, length, tool-use, citation, tone, safety, factual-consistency, step-count) | Bounds feasibility and coverage; enables per-type calibration |
| Primary metric | Constraint-adherence gain on certified-constraint subset | Directly measures gated signal quality |
| Baseline | HGM CMP aggregation on all constraints | Tests whether ungated signal is polluted |
| Baseline | AgentIF adherence scoring on all constraints | Tests annotation-signal leakage without certification |
| Baseline | Random constraint gating at matched rate | Controls for gains from discarding constraints |
| Ablation-of-ours | CIC without judge-agreement check | Isolates the dual-map robustness contribution |
| Ablation-of-ours | CIC without Phase 2b manipulation check | Isolates the bundled-intervention guard |
| Ablation-of-ours | CIC with naive i.i.d. Hoeffding bound instead of cluster-robust bound | Quantifies premature-certificate rate from ignoring within-prompt correlation |
| Calibration metric | φ false-negative rate per constraint type on 512 human-labeled held-out traces | Tests φ-sufficiency assumption empirically |

### Why this setup validates the claim

The central claim is that unidentifiable constraints inject noise into downstream self-improvement signals, and that a pre-gate certifying behavioral recoverability removes that noise. AgentIF is the right substrate because each task carries explicit constraint annotations and a rule-based violation checker, so the same Δ̂(c) probe can be computed against ground-truth-labeled constraints rather than synthetic ones. Restricting to 8 pre-registered constraint types with existing rule-based checkers bounds feasibility and lets us report the fraction of AgentIF constraints covered, so the reviewer can judge whether the coverage is representative. HGM's CMP aggregation is the natural baseline because it is precisely the mechanism CIC targets: it consumes every constraint without asking whether the constraint's effect is recoverable under intervention, so if CIC's diagnosis is correct, HGM should degrade exactly on tasks whose constraint sets contain uncertifiable items. The random-gating baseline is essential because it separates "discarding constraints helps" from "discarding the right constraints helps" — a gain over HGM that also appears over random gating would falsify the identifiability story. The Phase 2b ablation isolates the bundled-intervention guard: if removing it does not increase the rate of certified-but-bundled constraints, the guard contributes nothing. The naive-bound ablation isolates the cluster-robust correction: we report the gap between naive n*(c) and cluster-robust n*(c) on AgentIF and the fraction of certificates that would be issued prematurely under the naive bound. The φ-calibration experiment on 512 human-labeled traces reports false-negative rate per constraint type, directly testing the φ-sufficiency assumption. Constraint-adherence gain on the certified subset is the right primary metric because it is computed only where CIC claims signal exists, so the prediction is a localized improvement, not a uniform lift.

### Expected outcome and causal chain

**vs. HGM CMP aggregation on all constraints** — On a task whose constraint set includes a constraint that is only meaningful under multi-constraint interaction, HGM's CMP estimator assigns it a small nonzero weight and the search loop chases a signal that no single-constraint intervention can reproduce, producing self-modifications that improve the point estimate without improving held-out adherence. Our method instead probes that constraint, observes Δ̂(c) below ε, issues cert(c)=0, and removes it from the CMP estimate, so the loop optimizes only over recoverable constraints. We expect a modest but consistent adherence gain concentrated on tasks with at least one uncertifiable constraint, and near-parity on tasks where all constraints certify.

**vs. AgentIF adherence scoring on all constraints** — On a case where the rule-based checker and the LLM judge disagree about whether a trace violates a constraint, AgentIF-style scoring trusts the annotation and propagates a judge-specific bias into the guidance signal, so the agent is pushed toward behavior that satisfies the judge's idiosyncratic reading rather than the constraint. Our method treats that disagreement as evidence that the behavioral signature is not robust to feature-map choice, withholds the certificate, and excludes the constraint from downstream guidance. We expect the largest separation on constraint types where the two maps are known to diverge, and little separation elsewhere.

**vs. Random constraint gating at matched rate** — On a task where the discarded constraints happen to be highly identifiable, random gating removes genuinely useful signal and the loop loses guidance it needed, producing adherence that is no better than HGM despite using fewer constraints. Our method instead discards only constraints whose intervention effect fails the margin, preserving the identifiable ones. We expect CIC to beat random gating by a clear margin on the certified subset while both beat un-gated HGM only slightly, which is the signature that selection, not reduction, is doing the work.

**vs. CIC without Phase 2b manipulation check** — On a task where the intervention for constraint c also perturbs a correlated constraint c', the ablation issues cert(c)=1 based on a bundled Δ̂(c), so the downstream loop optimizes against a signal that conflates c and c'. Our method's Phase 2b check detects |Δ̂(c')| > ε/2 and withholds the certificate. We expect the ablation to show a higher rate of certified-but-bundled constraints and a corresponding adherence drop on tasks with correlated constraint sets.

**vs. CIC with naive i.i.d. Hoeffding bound** — On tasks where traces are correlated within prompts, the naive bound underestimates n*(c) and issues certificates at n₀=64 that the cluster-robust bound would reject. We expect to report the gap between naive and cluster-robust n*(c) per constraint type and the fraction of premature certificates, with the cluster-robust variant showing fewer premature certificates and higher adherence on the certified subset.

### What would falsify this idea

If CIC's gain over HGM is uniform across tasks rather than concentrated on tasks containing uncertifiable constraints, or if CIC fails to beat random gating at the same discard rate, then the certificate is not identifying a real noise source and the central claim is wrong. A second falsifier: if the no-agreement ablation matches full CIC on every subset, the dual-map robustness mechanism contributes nothing and the design rationale collapses; similarly, if the Phase 2b ablation matches full CIC and the φ false-negative rate is near zero for all constraint types, the bundled-intervention and φ-sufficiency concerns are empirically negligible.

## References

1. AGENTIF: Benchmarking Instruction Following of Large Language Models in Agentic Scenarios
2. Huxley-Gödel Machine: Human-Level Coding Agent Development by an Approximation of the Optimal Self-Improving Machine
3. CFBench: A Comprehensive Constraints-Following Benchmark for LLMs
4. Judging LLM-as-a-judge with MT-Bench and Chatbot Arena
5. COLLIE: Systematic Construction of Constrained Text Generation Tasks
