# Monotone Evidence-Refinement Artifacts for Reproducibility Agents

## Motivation

ReproBridge constructs a standardized artifact (refined codebase, environment setup, launchers, guide) from whatever codebase evidence is available at schema-design time, then freezes that schema before all evidence is integrated. Because the artifact schema is fixed a priori, subsequent evidence discovered during reproduction cannot be incorporated without violating the schema, so recovered-procedure coverage is not monotone under belief refinement — an agent can actually lose coverage when a later, more informative observation contradicts an early schema choice. This is the reproducibility instance of the field-wide bottleneck: agent workflows that commit to a representation before evidence is complete cannot certify that their output is a fixed point of further evidence.

## Key Insight

If the artifact schema is itself a partially ordered object and each evidence-integration step is constrained to map to a schema element at or above the current one, then recovered-procedure coverage is non-decreasing by construction, and the only stable stopping point is a schema fixed point that no additional evidence can refine — making convergence a property of the evidence order rather than of a pre-chosen template.

## Method

**What it is.** MERA (Monotone Evidence-Refinement Artifacts) takes as input a raw codebase plus a stream of reproduction evidence (execution traces, environment probes, agent observations) and outputs a *sequence* of artifacts A_0 ⊑_V A_1 ⊑_V … ⊑_V A_T over a partially ordered artifact lattice, together with a fixed-point certificate at A_T. Each artifact is a tuple (codebase_refinement, env_spec, launcher_set, guide_claims). The order ⊑_V is the **verified coverage order**: A ⊑_V A' iff (i) cov(A', C) >= cov(A, C) as measured by the coverage oracle on each artifact, and (ii) no claim in A is contradicted by A'. The syntactic component-wise order ⊑_S (A ⊑_S A' iff every recovered procedure covered by A is covered by A' with equal or finer granularity, and no claim in A is contradicted by A') is used only to *enumerate* candidate refinements; it is not assumed to imply ⊑_V. Every edge in the output sequence is labeled either `monotone` (cov(A_{t+1},C) >= cov(A_t,C)) or `retraction` (cov(A_{t+1},C) < cov(A_t,C)); retraction edges are recorded as explicit revision nodes in the certificate DAG rather than silently accepted.

**Load-bearing assumption (stated explicitly).** MERA assumes that the coverage oracle cov(·,C) and the contradiction relation are *aligned* with the syntactic component-wise order: for every candidate A' with A_t ⊑_S A', cov(A',C) >= cov(A_t,C) and contradictions(A',e) = 0. This assumption is **not** asserted to hold by construction; it is **checked per step** by the oracle, and any violation is recorded as a retraction edge. The monotonicity claim of MERA is therefore conditional: the sequence is non-decreasing in cov along `monotone` edges, and the certificate reports the exact set and magnitude of `retraction` edges so that the reader can see where the syntactic order and the measured coverage order diverge.

**How it works.**

```pseudocode
Input: raw codebase C, evidence stream E = (e_1, e_2, ...), coverage oracle cov(·,·)
Hyperparams: min_gain = 0.01 (coverage increment to accept a refinement),
             max_rounds = 12, contradiction_penalty = 1.0,
             oracle_cache = {} (memoizes cov(A, C) keyed by artifact hash)

# Phase 1: seed lattice element from bare codebase
A_0 ← seed_artifact(C)                      # schema = {file: inferred_role}, no env/launcher/guide claims
cov_0 ← oracle_cov(A_0, C)                  # oracle_cache[A_0] = cov_0
t ← 0
edge_labels ← []                            # one label per accepted step

# Phase 2: monotone refinement loop (verified per step)
while t < max_rounds:
    e ← next_evidence(E)                    # execution trace, env probe, or agent observation
    Δ ← propose_refinements(A_t, e)         # candidates A' with A_t ⊑_S A' (syntactic enumeration)
    # each candidate is scored by the oracle, not by the syntactic order:
    for each A' in Δ:
        if A' not in oracle_cache: oracle_cache[A'] ← oracle_cov(A', C)
        cov(A') ← oracle_cache[A']
        contra(A') ← contradictions(A', e)
    # acceptance rule (unchanged from original):
    #   cov(A') >= cov(A_t) - contradiction_penalty * contra(A')
    A_{t+1} ← argmax_{A' in Δ} cov(A')       # ties broken by smaller |A' \ A_t|
    if cov(A_{t+1}) >= cov(A_t):
        edge_labels.append('monotone')
    else:
        edge_labels.append('retraction')    # explicit revision node, not silent
    if cov(A_{t+1}) - cov(A_t) < min_gain and contra(A_{t+1}) == 0:
        break                               # fixed point under this evidence
    A_t ← A_{t+1}; t ← t + 1

# Phase 3: fixed-point certification over the MEASURED coverage order
certificate ← {A_T, evidence_used: e_1..e_T,
               edge_labels: edge_labels,          # monotone vs retraction per step
               coverage_trace: [cov_0, ..., cov_T],
               closure_check: for all e in held-out probe set,
                              propose_refinements(A_T, e) = {A_T}
                              OR every A' in propose_refinements(A_T, e) has
                                 cov(A') < cov(A_T) (i.e., no oracle-verified improvement)}
return sequence (A_0, ..., A_T), certificate
```

`propose_refinements` enumerates, for each uncovered procedure mentioned in `e`, the minimal lattice element under ⊑_S that adds coverage for it while preserving all prior claims; `contradictions` counts prior guide claims or env entries that `e` falsifies. The `min_gain` threshold and `contradiction_penalty` are the only tunable quantities; both are set by a 10% held-out split of ReproBridge-Bench tasks before evaluation on the remainder. `oracle_cov` is the ReproBridge Auditor re-run on the candidate artifact; results are memoized by artifact hash so that repeated candidates across rounds and across probe draws are not re-executed.

**Why this design.** We chose a *lattice* over a flat schema because a flat schema forces every refinement to either fit an existing slot or be discarded, which is precisely the ReproBridge failure mode; the cost is that we must maintain an explicit order relation, which we accept because it is what makes monotonicity checkable rather than asserted. We chose *coverage increment* as the acceptance criterion over *artifact completeness* because completeness rewards adding boilerplate that does not recover any procedure, whereas coverage is tied to the Auditor's execution evidence; the cost is that coverage must be estimated by the coverage oracle, which we implement as the ReproBridge Auditor re-run on each candidate. We chose *fixed point under held-out evidence* as the convergence certificate over a fixed round budget because a budget certifies termination but not sufficiency, whereas a fixed point certifies that no further evidence in the probe set can refine A_T; the cost is that the certificate is only as strong as the probe set, which we make explicit in the limitation. We deliberately do *not* adopt the ReproBridge template-then-fill pipeline: an expert would not describe MERA as a variant of ReproBridge because ReproBridge fixes the schema before evidence arrives, whereas MERA's output is a monotone sequence whose last element is defined by the evidence order, not by a template. The verified-order revision (⊑_V checked by the oracle, retraction edges recorded) is the smallest change that makes the order relation honest: the syntactic order ⊑_S is used only for enumeration, and the certificate reports exactly where the measured coverage order diverges from it.

**Why this measures what we claim.** `cov(A, C)` measures recovered-procedure coverage because we assume the coverage oracle's procedure inventory is complete with respect to the procedures the Auditor can execute; this assumption fails when a procedure is tacit (never executed in any trace), in which case `cov` reflects only executable procedures and overstates coverage of tacit ones. The verified order `A ⊑_V A'` measures monotone evidence integration because we assume each refinement step consumes evidence that is *about* the codebase rather than about the agent's own prior artifact, and that the oracle's coverage measurement is stable across re-runs on the same artifact; this assumption fails when an agent's observation is contaminated by its own earlier guide claims, or when the Auditor's execution environment changes between runs, in which case the order reflects self-consistency or environment drift rather than evidence integration. The fixed-point certificate measures convergence because we assume the held-out probe set is drawn from the same evidence distribution as the stream; this assumption fails under distribution shift (e.g., a new dependency regime), in which case the certificate reflects convergence on the probe distribution only. The `contradiction_penalty` measures belief revision cost because we assume contradictions are equally costly regardless of which claim they falsify; this assumption fails when a contradiction targets a load-bearing environment pin, in which case the penalty underweights the revision.

## Contribution

['MERA: a monotone evidence-refinement layer in which the artifact schema is a partially ordered lattice and every evidence-integration step is constrained to move upward, yielding a sequence of artifacts with non-decreasing recovered-procedure coverage.', 'A fixed-point certification procedure that replaces schema-freezing with evidence closure: an artifact is certified when no held-out evidence can refine it, making convergence a property of the evidence order rather than of a pre-chosen template.', 'A design principle for reproducibility agents — that schema commitment should be deferred to the evidence order — together with an implementation on ReproBridge-Bench that exposes where tacit procedures break the coverage oracle.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | ReproBridge-Bench (1,029 tasks, 387 experiments) | Matches the failure mode MERA targets |
| Primary metric | Recovered-procedure coverage on held-out probes, measured by the ReproBridge Auditor | Tied to Auditor execution evidence, not boilerplate |
| Secondary metric | Fraction of accepted steps labeled `retraction` in the certificate DAG, and mean |Δcov| on those steps | Directly audits the load-bearing assumption: how often the syntactic order ⊑_S and the measured coverage order ⊑_V diverge |
| Baseline | ReproBridge template-then-fill | Tests whether fixed schema discards refinements |
| Baseline | Raw-code agent (no artifact) | Tests whether any structured artifact helps at all |
| Baseline | Flat-schema refinement (no lattice) | Isolates the lattice order's contribution |
| Ablation-of-ours | MERA without fixed-point certificate (round budget only) | Tests whether the certificate, not just iteration, drives convergence |
| Ablation-of-ours | MERA with syntactic order only (no oracle-verified ⊑_V check; original pseudocode) | Tests whether the verified-order repair changes measured coverage or only the honesty of the certificate |
| Probe | Tacit-procedure injection: on a 100-task sample, inject one procedure that is never executed in any trace and measure the gap between `cov` and ground-truth procedure inventory | Quantifies the coverage-oracle completeness assumption stated in block D |

### Why this setup validates the claim

The central claim is that a monotone artifact lattice, refined by coverage increment against execution evidence, recovers more procedures than a fixed-schema pipeline and does so with a checkable convergence guarantee. Each baseline isolates one sub-claim. The raw-code agent tests whether structured artifacts help at all; if MERA does not beat it, the lattice is doing no work. The ReproBridge template-then-fill baseline tests the specific mechanism gap we name — that fixing the schema before evidence arrives forces refinements to be discarded — so a gain over it directly supports the monotonicity argument. The flat-schema baseline holds the coverage criterion constant while removing the order relation, isolating whether monotonicity itself (not just evidence-driven refinement) produces the gain. The ablation removes the fixed-point certificate while keeping the refinement loop, so any difference between it and full MERA attributes the effect to certification rather than iteration. The syntactic-order-only ablation isolates the contribution of the verified-order repair: if it matches full MERA on mean coverage but shows more unlabeled coverage drops, the repair is doing certificate work rather than coverage work, which is exactly what the load-bearing-assumption repair claims. The retraction-fraction metric makes the load-bearing assumption auditable: it reports how often the oracle-measured coverage order diverges from the syntactic order, which is the quantity the original method assumed away. Coverage on held-out probes is the right metric because it is measured by the same Auditor that generates the evidence stream, so it cannot be inflated by boilerplate the way artifact-completeness would be; it detects exactly the predicted effect — recovery of procedures the baseline discards — rather than a proxy. The tacit-procedure probe tests the completeness assumption in block D directly rather than leaving it as a stated limitation.

### Expected outcome and causal chain

**vs. Raw-code agent** — On a case where a task requires reconstructing an undocumented launcher whose entry point is only visible in an execution trace, the raw-code agent produces a partial reproduction because it has no persistent artifact to accumulate evidence across sessions and re-derives each procedure from scratch. Our method instead folds each trace into a lattice element that adds coverage for the newly seen procedure while preserving prior claims, so we expect a large gap on tasks with multi-step launcher chains and near-parity on single-entry-point tasks where the raw agent already succeeds.

**vs. ReproBridge template-then-fill** — On a case where a procedure is mentioned in evidence only after the template schema is fixed (e.g., a conditional environment pin discovered in a later trace), the baseline produces an artifact that either forces the pin into a mismatched slot or discards it, because its schema is chosen before evidence arrives. Our method instead admits the pin as a refinement A_{t+1} ⊒_S A_t because the syntactic order permits finer granularity without schema violation, and the oracle-verified ⊑_V check confirms the coverage gain, so we expect a concentrated gap on the subset of tasks with late-arriving procedures and parity on tasks whose procedures are all visible in the first trace.

**vs. Flat-schema refinement** — On a case where two refinements are both plausible from a single trace but only one is consistent with a later contradiction, the flat-schema baseline produces an artifact that overwrites or drops the earlier refinement without recording the order, because it has no partial order to check monotonicity against. Our method instead rejects the contradicting candidate via the contradiction penalty and keeps the earlier claim, so we expect a modest gap on tasks with at least one falsifying probe and near-parity elsewhere — the signature that monotonicity, not refinement per se, is the active ingredient.

**vs. MERA without fixed-point certificate** — On a case where the round budget expires before the probe set is exhausted, the ablation produces an artifact that stops at A_T without evidence that no further probe can refine it, so its reported coverage is unstable across probe draws. Our method instead returns a certificate whose closure check confirms that every A' in propose_refinements(A_T, e) has cov(A') < cov(A_T) for held-out probes, so we expect the ablation to match MERA on mean coverage but show higher variance across probe resamples — the signal that certification, not iteration, stabilizes the outcome.

**vs. MERA with syntactic order only** — On a case where a later trace reveals that an earlier recovered procedure was actually a different procedure under a now-falsified env pin, the syntactic-order-only variant accepts the refinement under the contradiction penalty and reports a coverage drop without labeling it, because it assumes ⊑_S implies ⊑_V. Our method instead records the step as a `retraction` edge and reports the measured |Δcov|, so we expect the two to match on mean coverage but the syntactic-only variant to show unlabeled coverage drops on the retraction-fraction metric — the signal that the verified-order repair changes the honesty of the certificate, not the coverage trajectory.

### What would falsify this idea

If MERA's gain over the flat-schema baseline were uniform across all task subsets rather than concentrated on tasks with late-arriving or contradicting evidence, the monotonicity mechanism would not be the cause and the central claim would be wrong. Likewise, if the no-certificate ablation matched MERA's coverage variance across probe resamples, the fixed-point certificate would be doing no work and the convergence claim would be unsupported; and if the retraction fraction were near zero on all tasks, the load-bearing assumption would hold empirically and the verified-order repair would be unnecessary.

## References

1. ReproBridge: Building Reproducibility into ML Research Artifacts
