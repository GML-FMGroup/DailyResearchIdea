# Composition-Closure Scoring for Chained Scientific Discovery Workflows

## Motivation

Existing benchmarks and frameworks decompose scientific discovery into independently scored single-artifact stages: LLM-SRBench scores each generated equation against one reference equation, and ScienceAgentBench scores each agent-produced program against one expert-written target program. This per-stage ground-truth matching is structurally blind to compensating errors — an equation or program that is wrong at stage k but happens to yield the correct downstream artifact at stage k+1 receives full credit, and a legitimately equivalent-but-different intermediate artifact is penalized. Because neither lineage defines a correctness predicate over the composition of stages, the field cannot distinguish a workflow that is robust from one that is accidentally right along a single sampled trajectory.

## Key Insight

A workflow is correct precisely when its final artifact is invariant under re-derivation of every intermediate stage from that stage's declared interface alone, so compensating errors become detectable as interface-conditioned divergence rather than as per-stage score discrepancies.

## Method

**What it is.** Composition-Closure Scoring (CCS) is an evaluation-and-training signal for chained scientific-discovery workflows. Input: a workflow decomposed into stages s_1..s_K, each with a declared interface I_k (the typed signature of the artifact it produces and the inputs it consumes), plus a reference final artifact. Output: a scalar closure score C that is high only when the composed artifact is reproducible from interfaces rather than from any single sampled intermediate artifact.

**Load-bearing assumption (stated explicitly).** CCS assumes that the declared interface I_k is a sufficient statistic for stage k's legitimate behavior: that re-deriving stage k from interface_spec(interfaces[k]) alone produces a valid alternative artifact whose downstream divergence measures compensating error rather than the interface's under-specification. This assumption fails when the interface is under-specified (a typed signature such as `equation: R^d -> R` does not encode units, boundary conditions, conserved quantities, or domain priors) or when a downstream stage ignores its input; in those cases the divergence reflects interface sparsity or downstream noise, not composition. To make the assumption testable rather than asserted, CCS reports the closure score as a contrast against a matched control (defined in step 2b) and stratifies results by independently measured interface underdetermination and downstream input-sensitivity (experiment).

**How it works.**

```pseudocode
def CCS(workflow, stages, interfaces, reference_final, n_rederive=5, n_control=5):
    # 1. Sample one full forward trajectory
    arts = {}
    arts[0] = workflow.inputs
    for k in 1..K:
        arts[k] = LLM_stage(s_k, arts[k-1])          # sampled artifact

    # 2. Interface-conditioned re-derivation at every stage
    #    interface_spec(interfaces[k]) is a fixed prompt template:
    #      - the exact typed signature (e.g. "f: R^3 -> R"),
    #      - the declared input/output types and units,
    #      - the stage's declared invariants (conserved quantities, boundary conditions),
    #      - the reference final artifact's declared invariants,
    #      - NO upstream artifact content, NO intermediate reference artifact.
    #    "interface_only" conditioning is enforced by omitting arts[k-1] from the prompt
    #    and by a post-hoc check that the re-derived artifact type-checks against I_k.
    divergences = []
    for k in 1..K-1:
        for j in 1..n_rederive:
            arts_k_re = LLM_stage(s_k, interface_spec(interfaces[k]),
                                  upstream=None,
                                  condition="interface_only")
            final_re = run_downstream(s_{k+1}..s_K, arts_k_re)
            divergences.append(compare(final_re, reference_final))

    # 2b. Matched control: re-derive stage k from the FULL upstream artifact
    #     (same prompt, but arts[k-1] included). This is the null condition.
    control_divergences = []
    for k in 1..K-1:
        for j in 1..n_control:
            arts_k_ctl = LLM_stage(s_k, interface_spec(interfaces[k]),
                                   upstream=arts[k-1],
                                   condition="full_context")
            final_ctl = run_downstream(s_{k+1}..s_K, arts_k_ctl)
            control_divergences.append(compare(final_ctl, reference_final))

    # 3. Composition-closure predicate as a contrast against the control
    C = 1 - mean(divergences)          # in [0,1]
    C_control = 1 - mean(control_divergences)
    C_contrast = C - C_control         # positive only if interface-only diverges MORE
    C_final = compare(arts[K], reference_final)
    return C, C_control, C_contrast, C_final
```

The two signals are reported jointly: `C_final` is the ordinary per-stage score (what LLM-SRBench reports), and `C` is the closure score. A workflow with high `C_final` but low `C` is flagged as *compensating*: it reached the right answer by a path that does not survive re-derivation of its own intermediate artifacts. The contrast `C_contrast = C - C_control` is the load-bearing quantity: it is positive only when interface-only re-derivation diverges more than full-context re-derivation, i.e., when the upstream artifact's specific content is load-bearing. For training, `C_contrast` is used as the reward in a rejection-sampling loop over candidate stage artifacts, with `n_rederive=5`, `n_control=5`, and a divergence threshold of 0.05 relative L2 on numeric artifacts / exact string match on symbolic ones.

**Why this design.** We chose interface-conditioned re-derivation over per-stage ground-truth matching because the ground-truth artifact for an intermediate stage is not identifiable from the interface — many artifacts satisfy the same interface — so matching against one reference artifact penalizes legitimate alternatives; we accept that this makes the signal stochastic (re-sampling variance) and mitigate it with `n_rederive=5`. We chose to re-derive the *final* artifact under substitution rather than to compare intermediate artifacts directly, because the final artifact is the only quantity the workflow is actually accountable for; the cost is that `C` cannot localize which stage failed, only that the chain is not closed. We chose a fixed divergence threshold rather than a learned scorer because a learned scorer would need a labeled set of compensating-error workflows that does not exist; the cost is that the threshold must be tuned per artifact type. We added the matched full-context control (step 2b) because the interface-only divergence alone conflates compensating error with interface under-specification; the contrast `C - C_control` isolates the load-bearing content of the upstream artifact. This differs from the closest named prior, LLM-SRBench's per-equation ground-truth matching: a domain expert would not describe CCS as a variant of it, because CCS never compares an intermediate artifact to a reference — it compares a *re-derived final artifact* to a reference under an intervention on the upstream interface, which is an operation LLM-SRBench does not perform.

**Why it measures what we claim.** `mean(divergences)` measures composition-closure because the assumption is that a workflow whose final artifact is invariant to interface-conditioned re-derivation of its intermediate stages has no stage whose output is load-bearing only through its accidental specific content; this assumption fails when the downstream stages are themselves insensitive to the substituted artifact (e.g., a downstream stage that ignores its input), in which case the divergence reflects only downstream noise, not closure. `interface_spec(interfaces[k])` measures the declared interface because the assumption is that the interface is a sufficient statistic for what stage k is allowed to depend on; this assumption fails when the interface is under-specified and the re-derivation has no information to condition on, in which case the divergence reflects interface sparsity rather than compensating error. The contrast `C - C_control` measures load-bearing content because the assumption is that full-context re-derivation preserves the upstream artifact's specific content while interface-only re-derivation does not; this assumption fails when the full-context re-derivation also ignores the upstream artifact (e.g., a stage that copies its input verbatim), in which case the contrast is zero regardless of compensating error. The joint report of `C` and `C_final` measures the distinction between accidental and robust correctness because the assumption is that compensating errors manifest as high `C_final` with low `C`; this assumption fails when a compensating error is itself reproducible under re-derivation (a systematic bias shared across stages), in which case the pair reflects a consistent but wrong workflow rather than a masked error.

## Contribution

(1) A composition-closure predicate for chained scientific-discovery workflows that scores a workflow by invariance of its final artifact under interface-conditioned re-derivation of intermediate stages, operationalized as the CCS algorithm. (2) The finding that high per-stage ground-truth scores and high closure scores dissociate on LLM-generated workflows, establishing that per-stage matching is not a sufficient proxy for workflow correctness. (3) A joint C / C_final reporting protocol and a rejection-sampling training loop that uses closure as the reward, applicable to equation-discovery and program-synthesis workflows alike.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | LLM-SRBench 239-problem suite | Multi-stage equation-discovery workflows with reference finals |
| Primary metric | Compensating-error rate (high C_final, low C_contrast) | Directly detects masked-path correctness; C_contrast = C - C_control isolates load-bearing content |
| Baseline | LLM-SRBench per-equation ground-truth matching | Closest named prior; per-stage matching only |
| Baseline | Single-pass best-of-n sampling | Tests whether closure needs intervention at all |
| Baseline | Principia-style iterative refinement | Tests whether test-time revision closes chains |
| Ablation-of-ours | CCS with interface-only re-derivation disabled (C_final only) | Isolates the closure signal from ordinary scoring |
| Ablation-of-ours | CCS with full-context control disabled (C only, no C_control) | Tests whether the contrast is necessary to avoid interface-sparsity confound |
| Stratification | Interface underdetermination score (number of type-checking artifacts sampled from interface_spec, n=20) and downstream input-sensitivity score (relative L2 change in final artifact when upstream artifact is perturbed by 10%) | Tests whether compensating-error rate survives within-stratum, not just across strata |
| Calibration | Synthetic chains with known injected compensating error (magnitude 0.01, 0.05, 0.1, 0.2 relative L2) and no-error chains | Shows mean(divergences) rises monotonically with injected error and stays flat on no-error chains |
| Threshold selection | Calibrate on 50 held-out no-error chains to target false-positive rate 0.05; threshold = 95th percentile of C_contrast on no-error chains | Reproducible, automated tuning |
| Compute budget | Report wall-clock and API cost per problem; variance of C_contrast vs n_rederive in {1,3,5,10} on 20-problem pilot | Feasibility and variance study |

### Why this setup validates the claim

The central claim is that ordinary per-stage correctness (C_final) cannot distinguish a workflow that reached the right answer through load-bearing intermediate artifacts from one that reached it through accidental specific content, and that interface-conditioned re-derivation exposes this. LLM-SRBench's per-equation matching is the natural foil: if per-stage matching were sufficient, then workflows it scores highly should also survive re-derivation, so C and C_final should co-move. The compensating-error rate is the right primary metric because it is precisely the disagreement between the two signals, not a marginal accuracy gain; a method that merely improves discovery accuracy would not move it. The single-pass and Principia-style baselines test whether the effect is a property of the workflow family rather than of CCS: if all baselines show the same C/C_final dissociation, closure is a dataset property, not a discriminative signal. The ablation (C_final only) is the decisive control: it holds the workflow and dataset fixed and removes only the intervention, so any gain in detecting compensating errors must come from re-derivation itself. The full-context control ablation tests whether the contrast C - C_control is necessary: if C alone matches C_contrast on the compensating subset, the control adds nothing; if C alone flags many non-compensating workflows, the control is doing the work of separating interface sparsity from load-bearing content. The stratification by interface underdetermination and downstream input-sensitivity tests the load-bearing assumption directly: if the compensating-error rate is uniform across strata, the signal is not measuring composition; if it is concentrated where interfaces are underdetermined or downstream stages are insensitive, the signal is confounded. The calibration experiment with known injected error tests that mean(divergences) is monotonic in error magnitude and flat on no-error chains, which is the minimal requirement for the signal to be a faithful intervention.

### Expected outcome and causal chain

**vs. LLM-SRBench per-equation ground-truth matching** — On a case where an intermediate stage emits a symbolically valid but accidentally-parameterized artifact (e.g., a fitted constant that happens to match the reference at that stage), the baseline scores it correct because it matches one reference intermediate, so its C_final is high and it reports no error. Our method instead re-derives that stage from its interface alone and propagates the substitute to the final artifact, because the interface does not encode the accidental constant, so we expect a marked compensating-error rate on the subset where intermediate artifacts are underdetermined by their interfaces, and near-parity elsewhere. The contrast C - C_control is expected to be positive on this subset because full-context re-derivation preserves the accidental constant while interface-only re-derivation does not.

**vs. Single-pass best-of-n sampling** — On a case where a downstream stage is a near-identity transform that copies its input into the final artifact, best-of-n reaches the reference final by luck of the upstream sample, so it looks correct without any stage being load-bearing. Our method substitutes interface-conditioned re-derivations upstream and observes divergence in the final artifact, because the copy-through stage transmits the substitution, so we expect single-pass to show high C_final with high C (closure) on identity-heavy chains but a large gap on chains with genuine composition, revealing that best-of-n's apparent correctness is fragile. The full-context control on identity-heavy chains is expected to show C_control ≈ C because the copy-through stage ignores the difference between interface-only and full-context upstream artifacts, so C_contrast ≈ 0, correctly flagging these chains as not load-bearing.

**vs. Principia-style iterative refinement** — On a case where iterative revision converges on a self-consistent but systematically biased principle (a shared bias across stages), the baseline reports a stable, improving trajectory and treats convergence as success. Our method's re-derivation reproduces the same bias, so C is also high, and we expect the joint C/C_final pair to correctly flag this as a consistent-but-wrong workflow rather than a compensating one; the observable signal is that Principia reduces variance across revisions while CCS's compensating-error rate stays flat, isolating closure from convergence. The contrast C - C_control is expected to be near zero on this subset because the shared bias is reproduced under both interface-only and full-context re-derivation.

**vs. CCS with full-context control disabled (C only)** — On the subset where interfaces are underdetermined (high number of type-checking artifacts sampled from interface_spec), C alone is expected to be low even for non-compensating workflows because interface-only re-derivation samples from the LLM's prior over all artifacts of that type. The contrast C - C_control is expected to remain high only on the compensating subset, demonstrating that the control is necessary to separate interface sparsity from load-bearing content.

### What would falsify this idea

If the compensating-error rate (high C_final, low C_contrast) is uniform across all strata — including those where interfaces are fully specified (low underdetermination score) and downstream stages are demonstrably input-sensitive (high sensitivity score) — rather than concentrated where interfaces are underdetermined or downstream stages are insensitive, then the closure signal is measuring interface sparsity or downstream noise rather than composition, and the central claim is wrong. Equivalently, if the C_final-only ablation matches CCS on the compensating subset, or if the calibration experiment shows mean(divergences) is not monotonic in injected error magnitude, the intervention adds nothing beyond ordinary scoring.

## References

1. LLM-SRBench: A New Benchmark for Scientific Equation Discovery with Large Language Models
2. Principia: Test-Time Hypotheses to Principle Construction for Scientific Discovery
3. Accelerating scientific discovery with Co-Scientist
4. ScienceAgentBench: Toward Rigorous Assessment of Language Agents for Data-Driven Scientific Discovery
