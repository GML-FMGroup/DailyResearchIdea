# Behavioral Signature Anchoring for Procedure Identifiability in Reproduced ML Artifacts

## Motivation

ReproBridge (2026) anchors reproduction success to five human-verified reference scores per task, so a recovered procedure is judged identical to the original whenever its scalar outcome matches. This collapses procedure identity onto a single number: two structurally different procedures — e.g., one that hard-codes a data split versus one that derives it from a seed — can produce the same reference score on the canonical environment yet diverge the moment that environment shifts, and the artifact has no way to tell the faithful recovery from the score-equivalent substitute. The residual reproducibility gap is thus structural: scalar anchoring is not an identifiable measurement of procedure, and no amount of extra reference scores fixes it because scores are invariant to exactly the procedural differences that matter under perturbation.

## Key Insight

A procedure is identifiable from its observable behavior if and only if the vector of its outputs across a battery of environment perturbations separates it from every score-equivalent substitute, so procedure identity can be certified by a behavioral signature rather than inferred from a scalar outcome.

## Method

**What it is.** BSA (Behavioral Signature Anchoring) replaces ReproBridge's five reference scores with a *behavioral signature*: a vector recording the recovered procedure's observable outputs under a battery of environment perturbations, plus an active-testing loop that grows the battery until every score-equivalent substitute in an explicitly stated, auditable candidate pool is separated. Inputs: a recovered procedure P, an environment perturbation generator, the original artifact's recorded signature, and the artifact's recorded provenance/execution evidence (the procedural choices logged by ReproBridge: seed handling, split derivation, I/O ordering, constant definitions, RNG source). Output: a signature S(P) and an identifiability certificate stating that no substitute in the stated candidate pool shares S(P), reported together with the pool's coverage argument and a held-out-operator miss-rate estimate.

**Load-bearing assumption (stated explicitly).** The certificate is valid only relative to the candidate pool C. We assume C contains the substitute class the artifact is meant to rule out. This is the mutation-testing adequacy problem: a hand-enumerated pool is a convenience sample, and `forall c in C: S(c) != S(P)` certifies exclusion relative to C, not absolute identifiability. We therefore (i) derive C from the artifact's own recorded provenance rather than from generic syntactic operators alone, (ii) map each pool operator to the procedural degree of freedom it is claimed to cover, and (iii) estimate the pool miss rate on a held-out operator set never used to grow the battery.

**How it works.**

```pseudocode
# Phase 0: provenance-derived candidate pool
# Each operator op is mapped to a procedural degree of freedom (DoF) it covers:
#   drop_seed        -> seed handling
#   drop_split_logic -> split derivation
#   reorder_io       -> I/O ordering
#   hardcode_const   -> constant definitions
#   replace_rng      -> RNG source
# Provenance-derived operators: for each DoF logged in the artifact's provenance,
#   instantiate the operator that reverts that DoF to its default (e.g., replace the
#   logged seed-derived split with a hard-coded split of the same sizes).
C = { substitute(P, op) for op in provenance_operators(P) }
# Held-out operator set (never used to grow B; used only to estimate miss rate):
C_heldout = { substitute(P, op) for op in {drop_seed, drop_split_logic, reorder_io, hardcode_const, replace_rng} \ provenance_operators(P) }

# Phase 1: seed battery (k0 = 8 perturbations)
B = {perturb(env, seed=s) for s in 0..7}    # seeds, thread counts, dtype, batch order, clock, path, locale, hardware quota
S(P) = [ observable(P, b) for b in B ]       # observables: output tensor hash, RNG draw sequence, file-write order, log line multiset
# Flakiness filter: each perturbation b is run twice; if observable(P,b) differs across the two runs, b is discarded from B and re-drawn.

# Phase 2: candidate substitution pool (from Phase 0)
# C is the provenance-derived pool; C_heldout is reserved for miss-rate estimation.

# Phase 3: active perturbation testing (loop until closure or budget)
while exists c in C with score(c) == score(P) and S(c) == S(P):
    # Separation-gain objective: for each candidate perturbation b, compute the
    # partition of colliding substitutes induced by observable(c, b), and select b
    # maximizing the expected reduction in colliding-substitute count:
    #   gain(b) = |colliding| - sum_{v in values(observable(c,b))} |{c' in colliding : observable(c',b)=v}|^2 / |colliding|
    b* = argmax_{b in candidate_perturbations} gain(b)   # tie-break by lowest seed index
    B = B + {b*}
    S(P) = S(P) + [ observable(P, b*) ]
    prune C of substitutes now separated

# Phase 4: certificate
certify(P) iff for all c in C: S(c) != S(P)   # score-equivalent substitutes in C all separated
# Report alongside the certificate:
#   - coverage table: operator -> DoF covered
#   - miss rate: fraction of c in C_heldout that are score-equal to P but share S(P)
```

Hyperparameters inline: seed battery size k0 = 8; active-testing budget = 24 perturbations; separation-gain tie-break by lowest seed index; observable set fixed to {tensor hash, RNG sequence, write order, log multiset}; flakiness filter = 2 reruns per perturbation.

**Why this design.** We chose a *behavioral* signature over storing the procedure's source code or execution trace because source-level identity is brittle to legitimate refactors and trace identity is brittle to nondeterminism, accepting the cost that the signature only certifies behavior we actually probe. We chose *active* perturbation selection over a fixed large battery because a fixed battery wastes budget on perturbations that no substitute in the pool distinguishes, accepting the cost that the loop can terminate early on a pool that misses the true substitute class. We chose a *candidate substitution pool* as the identifiability target over an abstract 'all procedures' quantifier because full behavioral equivalence is undecidable in general, accepting the cost that the certificate is relative to the pool — and we make that cost auditable by deriving the pool from provenance, mapping operators to DoFs, and estimating miss rate on a held-out operator set. Crucially, a domain expert would not describe this as a variant of ReproBridge's Auditor or of score-anchored reproduction: ReproBridge checks that execution evidence exists and matches a scalar, whereas BSA computes a *separating* signature whose whole purpose is to distinguish score-equal procedures — the mechanism is a perturbation-response fingerprint, not an evidence check.

**Why it measures what we claim.** The signature component `S(P)[b] = observable(P, b)` measures *procedure identity under perturbation* because the assumption that a procedure's observable behavior under an environment perturbation is a function of its procedural choices (not just its output on the canonical environment) makes the perturbation-response vector a separating statistic; this assumption fails when two procedural choices are behaviorally inert on every probed perturbation, in which case the signature reflects only the equivalence class of procedures indistinguishable on B rather than the procedure itself. The active-testing selection `b* = argmax gain(b)` measures *identifiability pressure* because the assumption that the perturbation maximizing expected reduction in colliding-substitute count is the binding collision makes separation gain a proxy for separation efficiency; this assumption fails when a cheaper perturbation would have separated a lower-gain substitute first, in which case the loop spends budget on a substitute that a cheaper perturbation would have separated. The certificate `forall c in C: S(c) != S(P)` measures *score-equivalent-substitute exclusion* because the assumption that the provenance-derived candidate pool contains the substitute class the artifact is meant to rule out makes pool-relative separation equivalent to the identifiability claim; this assumption fails when the deployed substitute is outside the pool, in which case the certificate reflects exclusion relative to the enumerated pool rather than absolute identifiability — which is why the certificate is reported with a coverage table and a held-out-operator miss rate.

## Contribution

(1) A behavioral-signature anchoring framework that replaces scalar reference-score anchoring with a perturbation-response fingerprint and an active-testing loop that certifies separation from score-equivalent substitutes. (2) The design principle that procedure identifiability in reproduced artifacts is a *separating-statistic* property over environment perturbations, not an outcome-matching property, yielding pool-relative identifiability certificates. (3) A candidate substitution pool of five procedural mutation operators (drop_seed, drop_split_logic, reorder_io, hardcode_const, replace_rng) usable as a reproducibility stress-test suite.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | 1,029 ReproBridge reproduction tasks (387 experiments, 36 papers) | Same task pool as the baseline being challenged |
| Primary metric | Fraction of score-equal substitutes separated by signature | Directly measures the identifiability claim |
| Secondary metric | Held-out-operator miss rate: fraction of C_heldout substitutes that are score-equal to P but share S(P) | Estimates pool incompleteness, the load-bearing assumption |
| Baseline | ReproBridge Auditor (evidence check) | Tests whether evidence presence suffices for identifiability |
| Baseline | Fixed-battery signature (k=32, no active loop) | Tests whether active perturbation selection earns its budget |
| Baseline | Source/trace identity check | Tests whether behavioral signature beats brittle identity |
| Baseline | Random perturbation selection (same budget as BSA) | Tests whether separation-gain selection beats random |
| Baseline | Oracle separation-gain selection (same budget as BSA) | Upper bound on what active selection can achieve on the enumerated pool |
| Ablation-of-ours | BSA with substitution pool reduced to {drop_seed} only | Isolates the contribution of pool breadth to the certificate |
| Ablation-of-ours | BSA with entropy selection instead of separation-gain selection | Tests whether the separation-gain objective earns its complexity |
| Ablation-of-ours | BSA with provenance-derived pool replaced by the five fixed syntactic operators | Tests whether provenance derivation improves pool adequacy |

### Why this setup validates the claim

The central claim is that a perturbation-response signature separates score-equal procedures that an evidence check cannot, and that active selection reaches this separation with fewer perturbations than a fixed battery. The ReproBridge Auditor baseline tests the first sub-claim directly: on tasks where the Auditor passes a substitute that is score-equal but procedurally divergent, BSA should fail to certify that substitute, producing a separation gap. The fixed-battery baseline tests the second sub-claim: with the same total perturbation budget, active selection should separate more substitutes than a uniform battery, because the loop concentrates probes on binding collisions rather than spending budget on inert perturbations. The random and oracle selection baselines bracket the active loop: random shows the gain from any selection, oracle shows the headroom left by the separation-gain objective. The source/trace baseline tests the framing claim that behavioral signatures dominate identity checks under legitimate refactors and nondeterminism. The ablation narrowing the pool to a single substitution class tests whether BSA's separation advantage persists when pool breadth is removed; if it persists, the gain is from signature design, not pool breadth. The ablation replacing separation-gain with entropy tests whether the separation-gain objective earns its complexity. The ablation replacing the provenance-derived pool with the five fixed syntactic operators tests whether provenance derivation improves pool adequacy. The held-out-operator miss rate directly measures the load-bearing assumption: if the provenance-derived pool misses substitutes that the held-out operators realize, the certificate is only as strong as the stated pool, and the miss rate quantifies that gap. The primary metric is the right detector because it counts exactly the events the certificate asserts are impossible: score-equal substitutes that share the signature.

### Expected outcome and causal chain

**vs. ReproBridge Auditor** — On a case where a substitute drops the seed but happens to reproduce the canonical score, the Auditor produces a pass because execution evidence exists and the scalar matches, since its mechanism only checks evidence presence and score agreement. Our method instead probes the substitute under seed, thread-count, and batch-order perturbations, where dropping the seed changes the RNG draw sequence and output tensor hash, because the signature records perturbation responses rather than canonical outputs. We expect a concentrated separation gap on the score-equal-but-divergent substitute subset, with near-parity on tasks where substitutes are already score-distinct.

**vs. Fixed-battery signature** — On a case where the binding collision is a reordered-I/O substitute that only diverges under a specific write-order perturbation not in the seed battery, the fixed battery produces a certificate failure or wastes budget on inert probes, because its perturbation set was chosen without reference to the substitute pool. Our method instead selects the perturbation maximizing expected reduction in colliding-substitute count, which surfaces the write-order probe, because the active loop targets the binding collision. We expect active selection to separate noticeably more substitutes at equal budget, with the advantage largest on tasks whose collisions are perturbation-specific.

**vs. Source/trace identity check** — On a case where the recovered procedure is a legitimate refactor with identical behavior but different source or nondeterministic trace ordering, the identity baseline produces a false non-match because source and trace identity are brittle to refactoring and nondeterminism. Our method instead certifies the refactor because its perturbation-response vector is invariant across the refactor, since the signature records behavior under perturbation rather than code or trace. We expect identity checks to show spurious mismatches on refactored tasks while BSA shows parity, isolating the brittleness cost.

**vs. Random perturbation selection** — On a case where the binding collision is a hardcoded constant visible only under dtype or locale probes, random selection spends budget on inert perturbations and separates fewer substitutes at equal budget, because it does not target colliding substitutes. Our method instead selects the perturbation maximizing expected reduction in colliding-substitute count, which surfaces the dtype or locale probe, because the separation-gain objective concentrates probes on the binding collision. We expect separation-gain selection to separate more substitutes than random at equal budget, with the gap largest on tasks whose collisions are perturbation-specific.

**vs. Oracle separation-gain selection** — On a case where the oracle has access to the true separation gain of every perturbation, the oracle separates the maximum number of substitutes at equal budget, because it selects the globally best perturbation at each step. Our method instead uses the estimated gain over the enumerated pool, which may diverge from the oracle when the pool is incomplete. We expect the separation-gain objective to approach but not match the oracle, with the gap quantifying the cost of pool-relative estimation.

**vs. BSA with reduced pool** — On a case where the true divergent substitute is a hardcoded constant that is inert under seed perturbations but visible under dtype or locale probes, the reduced-pool ablation produces a certificate that misses the substitute, because its pool never contained that substitution class. Our method instead separates it because the full pool forces the active loop to grow the battery toward dtype and locale probes. We expect the ablation to under-certify relative to full BSA, with the gap concentrated on tasks whose divergences lie outside the seed substitution class.

**vs. BSA with entropy selection** — On a case where the highest-entropy colliding substitute is not the one whose separation is cheapest to certify, entropy selection spends budget on a substitute that a cheaper perturbation would have separated first, because entropy is a proxy for separation gain rather than the gain itself. Our method instead selects the perturbation maximizing expected reduction in colliding-substitute count, which directly targets the binding collision. We expect separation-gain selection to separate more substitutes than entropy selection at equal budget, with the gap largest on tasks whose collisions have heterogeneous separation costs.

**vs. BSA with fixed syntactic pool** — On a case where the artifact's provenance logs a seed-derived split but the five fixed syntactic operators do not instantiate the hard-coded-split substitute, the fixed-pool ablation misses the substitute, because its pool is a convenience sample rather than a provenance-derived enumeration. Our method instead derives the pool from the artifact's own recorded procedural choices, which includes the hard-coded-split substitute, because the provenance-derived pool covers the DoFs the artifact actually exercises. We expect the provenance-derived pool to separate more substitutes than the fixed syntactic pool, with the gap concentrated on tasks whose divergences lie outside the five generic operators.

### What would falsify this idea

If BSA's separation advantage over the Auditor is uniform across all task subsets rather than concentrated on the score-equal-but-procedurally-divergent subset, the signature is not doing the separating work the claim requires. Likewise, if active selection shows no separation advantage over the fixed battery at equal budget, or if the held-out-operator miss rate is high enough that the certificate's pool-relative exclusion does not cover the substitute class the artifact is meant to rule out, the identifiability claim is only as strong as the stated pool.

## References

1. ReproBridge: Building Reproducibility into ML Research Artifacts
