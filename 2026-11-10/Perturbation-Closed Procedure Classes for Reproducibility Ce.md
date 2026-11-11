# Perturbation-Closed Procedure Classes for Reproducibility Certification

## Motivation

ReproBridge (2026) certifies an artifact by recovering a single procedure that reproduces a human-verified reference score, so the schema can only represent procedures anchored to a fixed score under one environment. Structurally, this means any procedure whose observable effects legitimately drift across dependency versions, hardware backends, or seeds is silently excluded rather than recorded, and independent reproduction is judged by point-matching a score instead of by membership in the set of environment-equivalent procedures. The reproducibility gap is therefore an artifact-expressiveness gap: the schema has no first-class representation of the perturbation family under which a procedure is supposed to remain valid.

## Key Insight

A procedure is reproducible exactly when its observable effect is invariant under the declared perturbation family, so the correct certificate is not a score but a perturbation-closed equivalence class of procedures whose membership can be decided by a finite, exhaustive perturbation-closure computation.

## Method

**What it is.** Perturbation-Closure ReproBridge (PCR-Bridge) extends the ReproBridge artifact schema with a perturbation family P and a closure operator Cl_P that maps any recovered procedure to the maximal set of procedures whose observable effect is P-invariant. Its inputs are a recovered procedure, a declared perturbation family, and a set of observable-effect probes; its output is a closed procedure class stored in the artifact and a membership certificate that any independent reproduction attempt can be checked against. The certificate is explicitly **relative to (O, P)**: it asserts membership in the perturbation-closed class modulo the declared probe suite O and perturbation family P, and it is stored alongside a measured residual-drift bound and a held-out-probe coverage report (Phase 6) so that drift outside O is detected rather than silently admitted.

**How it works.**

```pseudocode
INPUT: recovered procedure pi_0, perturbation family P = {p_1..p_k},
       observable-effect probe suite O = O_fit U O_heldout (|O_fit|=12, |O_heldout|=4),
       tolerance tau=1e-3, budget B=256

# Phase 1: declare perturbation family as first-class artifact fields
P <- {dependency pins (k_dep=3 version tuples),
      hardware backends (k_hw=3 in {cpu, cuda-A, cuda-B}),
      seeds (k_seed = 8 sampled seeds)}

# Phase 2: probe the observable effect of pi_0 under each perturbation
for p in P:
    e_p <- EXECUTE(pi_0 under p); OBSERVE(e_p, O)   # records loss curve, metric vector, resource trace

# Phase 3: compute the invariance signature on the FIT probe subset only
sigma(pi_0) <- ( o in O_fit : max_p || e_p[o] - e_{p_0}[o] || <= tau )
# sigma is the set of FIT observables that are P-invariant; non-invariant FIT observables
# are recorded as drift channels with their measured drift magnitudes.

# Phase 4: enumerate the closed class by closure under P (membership gated on sigma over O_fit)
C <- { pi_0 }
frontier <- { pi_0 }
while frontier nonempty and |C| < B:
    pi <- POP(frontier)
    for p in P:
        pi' <- PERTURB_PROCEDURE(pi, p)   # version swap, backend swap, seed swap
        if sigma(pi') == sigma(pi_0) and pi' not in C:
            C <- C U {pi'}; PUSH(frontier, pi')

# Phase 5: certify and store (certificate is relative to O_fit, P)
ARTIFACT.class <- C
ARTIFACT.signature <- sigma(pi_0)
ARTIFACT.drift_channels <- {o in O_fit not in sigma(pi_0): measured drift}
ARTIFACT.certificate_scope <- (O_fit, P)   # explicit: certified modulo these
CERTIFY(pi_hat) := [ sigma(pi_hat) == ARTIFACT.signature ]   # sound modulo O_fit, P

# Phase 6: held-out-probe coverage / falsification step (detects drift outside O_fit)
for pi' in C:
    for o in O_heldout:
        delta[pi', o] <- max_p || e_p[pi'][o] - e_{p_0}[pi'][o] ||
residual_drift_bound <- max_{pi' in C, o in O_heldout} delta[pi', o]
ARTIFACT.residual_drift_bound <- residual_drift_bound
ARTIFACT.coverage_report <- { o in O_heldout : residual_drift_bound[o] > tau }
# if any held-out observable exceeds tau, the certificate is flagged as
# 'relative-with-residual' and the offending channel is recorded, not hidden.
```

**Why this design.** We chose to make the perturbation family an explicit, enumerated artifact field rather than an implicit assumption in the reproduction guide because the failure mode ReproBridge exhibits is exactly that unstated environment drift is invisible to the schema; the cost is that the artifact must now carry a combinatorial family that grows with dependency and hardware dimensions. We chose exhaustive bounded enumeration of the closed class (budget B) over a learned or sampled approximation because closure membership must be decidable by a deterministic check at reproduction time, accepting that very large perturbation families are truncated and the class is certified relative to the enumerated subset rather than the full family. We chose an observable-effect signature with a fixed tolerance tau over a scalar score match because the motivation is invariance of effects, not equality of numbers, accepting that the signature is only as good as the probe suite O and can miss drift in unprobed observables; to make that limitation measurable rather than hidden, we split O into a fit subset O_fit (12 observables used to compute sigma and gate enumeration) and a held-out subset O_heldout (4 observables never used in sigma), and Phase 6 reports a residual-drift bound on O_heldout alongside every certificate. This is not a variant of ReproBridge's single-procedure recovery plus Auditor: ReproBridge stores one procedure anchored to a reference score and an Auditor that checks execution evidence, whereas PCR-Bridge stores a set closed under a declared perturbation group and certifies membership by signature equality; a domain expert would not describe it as the same method because the object being stored and checked is structurally different (a closed set and an invariance signature, not a point and a score).

**Why it measures what we claim.** The invariance signature sigma(pi_0) measures perturbation-closure because we assume that the fit probe suite O_fit is a sufficient statistic for the observable effect of the procedure, so that equality of signatures under all p in P is equivalent to membership in the same perturbation-closed class; this assumption fails when an observable that matters for reproduction is not in O_fit, in which case sigma reflects invariance only over the probed subset and the certificate can admit a procedure that drifts on an unprobed channel — which is why the certificate is explicitly scoped to (O_fit, P) and Phase 6 reports a residual-drift bound on the held-out probes O_heldout. The closure enumeration size |C| measures the breadth of the equivalence class because we assume the perturbation operators PERTURB_PROCEDURE commute with execution so that reachability under P equals semantic equivalence; this assumption fails when two distinct procedures produce identical signatures by coincidence, in which case the class reflects signature-collision rather than true procedure equivalence, and the held-out-probe residual-drift bound is the diagnostic that surfaces such collisions. The drift channels record measures the residual non-invariance because we assume the measured max drift under the sampled seeds upper-represents the drift under unsampled seeds; this fails under high seed variance, in which case drift channels understate the true drift magnitude, and the reported residual_drift_bound is a lower bound on the true worst-case drift.

## Contribution

(1) A perturbation-closure operator and closed-class schema extension for reproducibility artifacts, which replaces single-procedure score anchoring with a declared perturbation family and an invariance signature that certifies reproduction by class membership. (2) A bounded enumeration procedure that computes the perturbation-closed equivalence class of a recovered procedure and records drift channels as first-class artifact fields, yielding a design principle that reproducibility certificates should be set-valued and perturbation-relative rather than point-valued. (3) An artifact schema and membership checker that any independent reproduction attempt can be validated against without access to a human-verified reference score.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | ReproBridge 1,029-task suite (387 experiments) | Matches baseline's task distribution and reference scores |
| Primary metric | Certified-reproduction rate on drift-heavy task subsets, with certificate false-accept rate on held-out probes | Directly measures closure certificates holding under independent re-execution |
| Baseline | Raw-code agents (ReproBridge baseline) | Tests whether explicit perturbation family beats no artifact schema |
| Baseline | ReproBridge refined-artifact agents + Auditor | Tests whether closure set beats single-procedure point storage |
| Baseline | PCR-Bridge minus closure (signature only, no enumeration) | Isolates contribution of the closed-class enumeration |
| Ablation-of-ours | PCR-Bridge with reduced probe suite O_fit (12→4 observables) | Tests dependence of certificate validity on probe coverage |
| Ablation-of-ours | PCR-Bridge with fixed P, enumeration depth varied (B=16, 64, 256) | Isolates enumeration depth from perturbation declaration |
| Ablation-of-ours | PCR-Bridge with held-out probes O_heldout removed (no Phase 6) | Measures the sufficiency gap of O_fit directly |

### Why this setup validates the claim

The central claim is that storing a perturbation-closed class plus an invariance signature makes reproduction certificates survive environment drift that a point-procedure artifact cannot. The raw-code baseline tests the coarse claim that any structured artifact helps; the ReproBridge baseline tests the sharper claim that a closed set beats a single anchored procedure when drift is present; the signature-only ablation isolates whether the enumeration of the closed class, rather than the mere act of declaring perturbations, drives the gain. The primary metric is certified-reproduction rate restricted to drift-heavy subsets, because that is precisely where the predicted mechanism should bite: on drift-free subsets all methods should be near parity, so a uniform gain would contradict the causal story. The probe-suite ablation checks the stated assumption that the signature is only as good as O_fit by shrinking O_fit from 12 to 4 observables and predicting a drop concentrated on tasks whose drift channels leave the probed subset. The held-out-probe ablation (removing Phase 6) directly estimates the sufficiency gap: we measure the certificate false-accept rate on O_heldout, i.e., the fraction of procedures admitted by sigma over O_fit that exceed tau on a held-out observable, and predict this rate rises when O_fit is shrunk. The enumeration-depth ablation fixes P and varies B to isolate the contribution of enumeration breadth from the act of declaring perturbations.

### Expected outcome and causal chain

**vs. Raw-code agents** — On a task whose reference score depends on a pinned dependency version that the paper never states, the raw-code baseline silently installs a newer version, produces a shifted loss curve, and the agent mis-attributes the shift to a procedural error, yielding a failed or wrongly-flagged reproduction. Our method instead has already recorded that dependency tuple inside P and observed its effect on the probe suite, so the drift is a named channel rather than a surprise. We expect a large gap on the dependency-drift subset and near-parity on tasks with no unstated environment sensitivity.

**vs. ReproBridge refined-artifact agents + Auditor** — On a task where the recovered procedure is correct but a hardware backend swap changes a metric vector, the ReproBridge artifact stores one procedure anchored to a reference score, so the Auditor sees execution evidence that does not match the anchor and rejects a genuinely valid reproduction. Our method instead certifies any procedure whose signature equals sigma(pi_0), and the backend-swapped procedure is inside the closed class by construction. We expect the gap to concentrate on hardware- and seed-sensitive tasks while remaining flat elsewhere.

**vs. PCR-Bridge minus closure (signature only)** — On a task where two distinct dependency pins produce identical signatures on the probed observables, the signature-only variant accepts a procedure that drifts on an unprobed channel, because it checks equality without enumerating reachable procedures. Our method's bounded closure enumeration surfaces the collision as a drift channel during Phase 3 and records it, so the certificate is narrower but correct; additionally, Phase 6's held-out-probe residual-drift bound flags the collision when the unprobed channel is in O_heldout. We expect the signature-only variant to show inflated certified-reproduction on collision-prone tasks and a corresponding drop in true reproduction, while our method trades a little coverage for accuracy and reports a nonzero residual_drift_bound on exactly those tasks.

**vs. PCR-Bridge with reduced probe suite O_fit (12→4)** — Shrinking O_fit removes observables that were carrying invariance information, so sigma becomes coarser and more procedures pass the membership test. We expect certified-reproduction to rise spuriously on tasks whose drift channels leave the reduced O_fit, and the held-out-probe false-accept rate to rise correspondingly, confirming that the certificate is only as good as O_fit and that Phase 6 detects the gap.

**vs. PCR-Bridge with fixed P, enumeration depth varied (B=16, 64, 256)** — With P fixed, increasing B can only add procedures to C that already match sigma over O_fit. We expect certified-reproduction to be flat or slightly increasing with B on collision-free tasks, and on collision-prone tasks to show that larger B surfaces more signature-colliding procedures whose held-out residual drift exceeds tau, so the coverage report grows with B while the certificate remains sound modulo (O_fit, P).

**vs. PCR-Bridge with held-out probes O_heldout removed (no Phase 6)** — Without Phase 6, the certificate has no diagnostic for drift outside O_fit. We expect the measured certificate false-accept rate on O_heldout to be strictly positive and to concentrate on tasks with known nondeterministic ops (GPU atomics, cuDNN autotuning, BLAS non-associativity), demonstrating that the sufficiency gap is real and that Phase 6 is what makes it visible.

### What would falsify this idea

If our gain over the ReproBridge baseline were uniform across drift-free and drift-heavy subsets rather than concentrated where unstated dependency, hardware, and seed drift occurs, the perturbation-closure mechanism would not be doing the causal work claimed. Likewise, if shrinking the probe suite O_fit left certified-reproduction unchanged and the held-out-probe false-accept rate did not rise, the signature would not be measuring the invariance we attribute to it.

## References

1. ReproBridge: Building Reproducibility into ML Research Artifacts
