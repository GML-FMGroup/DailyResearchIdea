# Probe-Injective Intent Identification for Agent Task Synthesis

## Motivation

WorkFactory instantiates tasks from professional intents and uses probe executions only as support-failure detectors, but it never verifies that the instantiated task is the intended task: two distinct latent intents can produce the same observable probe outcomes, so an executable task may be misidentified. This is a structural gap, not a tuning issue — WorkFactory's information support graph and repair loop assume the professional intent is fully observable and correctly parsed at generation time, so no quantity in the pipeline measures intent identifiability under partial observability. The same residual recurs across R2E-Gym-style synthesis, meaning any pipeline that treats probes as pass/fail rather than as discriminative evidence inherits an unidentifiable intent-to-task map.

## Key Insight

Intent identifiability is achievable exactly when the probe battery is designed so that the map from latent intent to probe-outcome signature is injective, which converts each probe from a support-failure test into an intent-identification test and makes any residual collision a certified, repairable ambiguity rather than a silent misidentification.

## Method

**What it is.** PIIS (Probe-Injective Intent Synthesis) is a wrapper layer inserted into WorkFactory between information-level planning and task instantiation. Inputs: a professional intent `I` with its candidate latent-intent set `C(I)` (alternate readings produced by a paraphrase/underspecification generator) and the information support graph `G`. Outputs: either an instantiated task `T` whose probe-outcome signature is unique among `C(I)`, or a flagged ambiguity pair plus a repair request. The layer does not replace WorkFactory's planner; it gates instantiation on an injectivity certificate.

**Load-bearing assumption (stated explicitly).** The certificate is valid only relative to the enumerated hypothesis set `C(I)`: we assume the true latent intent lies in `C(I)`, so that row-uniqueness of the signature matrix over `C(I)` certifies correct intent identification rather than merely uniqueness within a generator-bounded set. This is a partial-identification assumption and we treat it as such: the certificate is accompanied by a measured coverage estimate (Phase 6) that reports the fraction of held-out readings colliding with the chosen `T`, so a true intent outside `C(I)` surfaces as a non-zero residual rather than a silent pass. We do not claim injectivity over all conceivable intents.

**How it works.**

```pseudocode
# Phase 1: candidate latent-intent enumeration
C = {I} ∪ ParaphraseUnderspec(I, n=8)          # alternate readings, generator G_para (Qwen 27B, T=0.7)
# NOTE: no DedupBySupportGraph pruning before probing — near-miss readings with
# identical information requirements are retained as hard negatives.

# Phase 1b: held-out adversarial candidate set (never used to design probes)
H = ParaphraseUnderspec_heldout(I, n=8)        # independent generator G_held (different model,
                                               # human-authored ambiguity seeds, red-team prompt)

# Phase 2: discriminative probe design (over C only)
for each pair (c_i, c_j) in C x C, i<j:
    d_ij = MinimalDiscriminatingArtifact(c_i, c_j, G)   # smallest input whose correct
                                                        # handling differs between readings
P = GreedySetCover({d_ij})                      # probe battery covering all pairs, |P| minimized

# Phase 3: execute probes, build signature matrix S (over C)
for p in P:
    for c in C:
        S[c, p] = ExecuteUnderReading(T_c, p)   # trace + output hash, deterministic seed

# Phase 4: injectivity test (over C)
if S has duplicate rows for any (c_i, c_j):
    collision = {(c_i, c_j)}
    P += RefineProbe(c_i, c_j, G)               # add one more discriminative artifact
    goto Phase 3                                 # bounded: max 3 refinement rounds
else:
    T = Instantiate(I, G, probes=P)             # attach probe battery to task
    cert = InjectivityWitness(S)                # row-uniqueness over C

# Phase 5: held-out coverage check (open-world residual)
for h in H:
    sig_h = ExecuteUnderReading(T_h, P)         # same probe battery P, held-out reading h
    collide_h = (sig_h == S[c] for some c in C)
coverage_residual = |{h in H : collide_h}| / |H|   # fraction of held-out readings colliding with T
cert = cert ∪ {coverage_residual}

# Phase 6: downstream use
# WorkFactory's support-failure repair loop runs ONLY on T;
# any repair that breaks row-uniqueness of S triggers re-entry to Phase 4.
# Tasks with coverage_residual > 0.25 are flagged for human adjudication rather than auto-instantiated.
```

Hyperparameters: `n=8` alternate readings per generator (covers typical intent ambiguity without combinatorial blowup); `max 3` refinement rounds (empirically sufficient; beyond this we flag rather than loop); probe traces hashed with fixed seed for reproducibility; held-out generator `G_held` is a different model family from `G_para` (e.g., Llama-3-70B vs Qwen 27B) plus 2 human-authored ambiguity seeds per intent; coverage_residual threshold 0.25 for flagging.

**Why this design.** We chose to enumerate alternate readings explicitly rather than rely on a single parsed intent (decision 1), because WorkFactory's parser is the failure point the motivation names — an explicit candidate set makes the ambiguity surface measurable, accepting the cost that we can only certify injectivity relative to `C`, not to all conceivable intents. We chose a minimal discriminating artifact per pair via set-cover (decision 2) over running every possible probe, because probe executions are expensive and most pairs are distinguished by one targeted input; the trade-off is that a non-minimal battery might catch unforeseen collisions, so we bound refinement to 3 rounds and flag residual ambiguity rather than silently pass. We chose a hard injectivity gate (decision 3) over a soft confidence score, because the motivation is a structural identifiability claim, not a calibration claim — a soft score would reintroduce the log-prob-as-gate anti-pattern; the cost is that some intents get flagged and require human or planner repair instead of being auto-instantiated. We chose to retain near-miss readings in `C` (no support-graph dedup before probing) and to add a held-out adversarial set `H` (decision 4), because the closed-world certificate is only as good as the hypothesis set it ranges over, and pruning by support graph would delete exactly the hard negatives most likely to collide with the true intent; the cost is a larger `|C|` and thus more probe executions, which we bound by the compute budget below. A domain expert would not describe this as WorkFactory-plus-probes: WorkFactory's probes are support-failure detectors chosen by the planner; our probes are chosen by pairwise discriminability and the gate is a row-uniqueness test on a signature matrix, which is an identifiability object WorkFactory never constructs.

**Why it measures what we claim.** The signature-matrix row-uniqueness test measures intent identifiability because we assume that two latent intents differing on any task-relevant dimension will produce different execution traces on at least one minimal discriminating artifact; this assumption fails when the artifact generator cannot construct an input that separates two readings (e.g., readings that differ only in tone, not in information requirements), in which case row-uniqueness reflects probe-battery coverage rather than true intent distinguishability. The set-cover minimality measures probe economy because we assume discriminating artifacts for different pairs are largely non-overlapping; this fails when one artifact separates many pairs, in which case set-cover underserves pairs it did not explicitly target. The refinement-round bound measures repair tractability because we assume each added artifact strictly increases the row-rank of `S`; this fails when an added artifact is redundant, in which case the round count reflects search effort rather than remaining ambiguity. The injectivity certificate measures the absence of silent misidentification because we assume the candidate set `C` contains the true intent; this fails when the true intent is not in `C`, in which case the certificate guarantees uniqueness within `C` but not correctness — which is why the certificate carries a measured `coverage_residual` against a held-out set `H` that was never used to design probes, so an intent outside `C` surfaces as a non-zero residual rather than a silent pass.

## Contribution

(1) A probe-injectivity layer for intent-to-task synthesis that replaces pass/fail probe execution with a signature-matrix row-uniqueness certificate, including a pairwise minimal-discriminating-artifact construction and a bounded refinement loop. (2) The design principle that intent identifiability should be certified relative to an explicitly enumerated candidate set before task instantiation, with residual collisions surfaced as flagged ambiguity rather than silent misidentification. (3) An ambiguity-flagging interface that lets WorkFactory's repair loop re-enter injectivity testing whenever a repair breaks row-uniqueness, turning repairs from support-failure fixes into identifiability-preserving operations.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | WorkFactory 612-occupation task pool with injected latent-intent paraphrases; stratified subsample of 150 tasks by ambiguity class (unambiguous / easy-ambiguous / hard-ambiguous) | Provides real professional ambiguity surface at tractable compute |
| Primary metric | Task success rate on GDPval-AA v2.1, split by ambiguity-collision subset (collisions defined by held-out set H, not by C) | Detects effect where predicted failure occurs, without self-referential collision definition |
| Secondary metric | `coverage_residual` = fraction of held-out readings in H colliding with chosen T; reported per ambiguity class | Measures open-world identifiability residual the certificate cannot close |
| Secondary metric | `artifact-constructibility rate` = fraction of candidate pairs (c_i, c_j) for which MinimalDiscriminatingArtifact returns a non-degenerate input; reported per ambiguity class | Operationalizes the trace-divergence assumption |
| Baseline | Base Qwen 27B (no WorkFactory) | Tests whether synthesis pipeline matters at all |
| Baseline | WorkFactory single-parsed-intent instantiation | Tests whether explicit candidate enumeration matters |
| Baseline | WorkFactory + soft confidence gate (log-prob threshold) | Tests hard injectivity gate vs calibration gate |
| Baseline | PIIS with closed-world certificate only (no held-out H, no coverage_residual) | Tests whether open-world coverage check carries the gain |
| Ablation-of-ours | PIIS with refinement rounds disabled (single probe pass) | Tests whether bounded refinement carries the gain |
| Ablation-of-ours | PIIS with DedupBySupportGraph applied to C before probing | Tests whether retaining hard negatives matters |

### Why this setup validates the claim

The central claim is that silent misidentification of intent, not raw capability, is the binding constraint on professional-work task synthesis, and that a structural injectivity gate removes it. The single-parsed-intent baseline isolates exactly this: it has WorkFactory's full pipeline and teacher trajectories, differing only in that it commits to one reading of the intent, so any gap between it and PIIS is attributable to ambiguity handling rather than scale or data. The soft-confidence-gate baseline tests the anti-pattern the motivation names — if a calibrated score sufficed, PIIS's hard gate should show no advantage, and the claim collapses to a calibration story. The closed-world-certificate baseline tests the load-bearing assumption directly: if PIIS's gain survives only when the held-out coverage check is present, then the open-world residual is doing causal work and the closed-world certificate alone is insufficient. The no-refinement ablation tests whether the gain comes from the injectivity mechanism itself or merely from attaching more probes, and the dedup ablation tests whether retaining hard negatives in C matters. Task success on GDPval-AA v2.1 is the right metric because a silently misidentified intent produces a task that is executable but wrong, which surfaces as failed downstream evaluation rather than as a planner error — precisely the failure the gate is built to intercept, and precisely what a capability-only baseline cannot fix by getting stronger. The `coverage_residual` and `artifact-constructibility rate` are reported alongside so the certificate's validity conditions are measured rather than asserted.

### Expected outcome and causal chain

**vs. Base Qwen 27B (no WorkFactory)** — On a case where a professional intent carries an unstated deliverable format, the base model produces a plausible but off-target answer because it has no task-instantiation scaffolding and no probe discipline. Our method instead instantiates only after the probe battery certifies that alternate readings diverge on execution traces, so we expect a large overall gap driven mostly by the pipeline, with PIIS's incremental contribution visible only on the ambiguity subset — parity on unambiguous tasks, a clear gap on collided ones.

**vs. WorkFactory single-parsed-intent instantiation** — On a case where the parser commits to the wrong reading of an underspecified intent, the baseline produces a fully executed task with the wrong information requirements because it never enumerates alternates and never tests discriminability. Our method instead enumerates candidate readings, designs minimal discriminating artifacts, and refuses to instantiate when rows collide, so we expect a concentrated gain on the ambiguity-collision subset (defined by held-out H) with near-parity elsewhere — a non-uniform pattern that is the signature of the mechanism firing where predicted.

**vs. WorkFactory + soft confidence gate** — On a case where two readings both receive high log-prob under the parser, the baseline passes the gate and instantiates the wrong one because a soft score cannot certify structural distinguishability. Our method instead requires row-uniqueness on the signature matrix, flagging or repairing the collision, so we expect the soft-gate baseline to sit between single-parsed and PIIS, with its residual errors concentrated on high-confidence-but-wrong cases that PIIS catches — a gap that widens as intent underspecification increases.

**vs. PIIS with closed-world certificate only (no held-out H, no coverage_residual)** — On a case where the true intent is outside C, the closed-world baseline produces a unique row, passes the gate, and is scored as a success of the mechanism even though the instantiated task is wrong. Our method instead executes the same probe battery against held-out readings H and reports `coverage_residual`; tasks with residual > 0.25 are flagged for human adjudication rather than auto-instantiated. We expect the closed-world baseline to match PIIS on tasks where C contains the true intent and to diverge on the subset where it does not, with the divergence appearing specifically in flagged-then-adjudicated cases and in the `coverage_residual` distribution rather than uniformly.

**vs. PIIS without refinement rounds** — On a case where the first probe battery leaves two readings indistinguishable, the ablation flags or mis-instantiates because it never adds a targeted artifact. Our method instead spends up to three refinement rounds to break the collision, so we expect the ablation to match PIIS on easy ambiguities and diverge on the hard tail, with the divergence appearing specifically in flagged-then-repaired cases rather than uniformly.

**vs. PIIS with DedupBySupportGraph applied to C before probing** — On a case where two readings differ only in a near-miss information requirement, the dedup ablation deletes the harder negative from C, so the signature matrix has fewer rows and the gate passes more easily — but the deleted reading is exactly the one most likely to collide with the true intent. Our method instead retains it, so we expect the dedup ablation to show higher apparent pass rates but lower task success on the hard-ambiguous subset, with the gap concentrated in cases where the deleted reading matches a held-out H reading.

### What would falsify this idea

The idea is falsified if PIIS's gain over single-parsed-intent WorkFactory is uniform across the ambiguity-collision and unambiguous subsets rather than concentrated on the collision subset, or if the soft-confidence-gate baseline matches PIIS on the collision subset — either pattern would mean the injectivity gate is not doing the causal work the motivation claims. It is also falsified if the closed-world-certificate baseline matches PIIS on the hard-ambiguous subset and on `coverage_residual`, which would mean the open-world coverage check is not load-bearing and the certificate is self-referential.

## References

1. WorkFactory: Turning Professional Intents into Executable Work for Scalable Agent Training
