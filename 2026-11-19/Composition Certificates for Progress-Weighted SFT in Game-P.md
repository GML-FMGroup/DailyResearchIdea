# Composition Certificates for Progress-Weighted SFT in Game-Playing Visual Agents

## Motivation

Across game-playing visual-agent work, task decomposition is asserted rather than verified: LVLM-Playground evaluates Perceiving, QA, Rule Following, and End-to-End Playing as four separate tasks with no composition rule (E1), and Game2Player assigns per-step progress weights from Builder-authored progress measures without any guarantee that those weights compose into the task-level success predicate (E4). The structural root cause is that the Builder emits a scalar progress measure and a boolean success check as two independently authored artifacts, so progress-weighted SFT can up-weight a trajectory that never satisfies the success check, teaching the VLM Player a reward that is not the task. A composition certificate is the missing structural property that turns step weights into a checkable refinement of the success predicate, and it recurs across both trees rather than being local to Game2Player.

## Key Insight

A hierarchical progress measure is a valid SFT weighting signal exactly when it is a monotone refinement of the task-level success predicate under a declared composition algebra, and this is a finite, per-task checkable property of the Builder's emitted state machine rather than something that must be learned or estimated.

## Method

**What it is.** We introduce **PC-Audit** (Progress-Composition Audit), a pre-training gate that sits between Game2Player's Builder and its progress-weighted SFT stage. Its input is a Builder-emitted task specification `T = (S, s0, g, {phi_k}, {w_k}, succ, alg)` — state space, initial state, sub-goal predicates, per-step weights, task-level success predicate, and a declared composition algebra `alg` — together with the collected trajectory set `D_T` and a calibration set `D_cal` of 512 held-out trajectories per task with ground-truth outcome labels `y_engine` (game-engine terminal win/lose signals or human-verified success). Its output is a per-task **composition certificate** `C_T ∈ {LICENSED, FLAGGED(reason)}` and, when licensed, a re-normalized weight vector `w*` used by SFT.

**Load-bearing assumption (explicit).** The certificate is sound only if the Builder's emitted specification is a faithful description of the underlying game. We state this assumption in plain language: *the Builder's state abstraction S represents every state transition the game permits, and the declared success predicate succ agrees with the game's true terminal outcome.* This assumption fails when the Builder collapses hidden game state or over-claims sufficiency, in which case a symbolic violation found in the spec reflects the Builder's abstraction rather than the game. To make the assumption checkable rather than declared, Phase 0 below tests succ against `D_cal` before any symbolic check runs.

**How it works.**

```pseudocode
INPUT: task T = (S, s0, g, {phi_k}_{k=1..K}, {w_k}, succ, alg), trajectories D_T,
       calibration set D_cal (512 trajectories with engine-level labels y_engine), horizon H
OUTPUT: certificate C_T, licensed weights w*

# Phase 0 - Ground-truth grounding check (tests the load-bearing assumption)
# D_cal is a held-out set of 512 trajectories per task, each labeled by the game engine
# (terminal win/lose) or by human verification; it is disjoint from D_T.
for each trajectory tau in D_cal:
    y_declared = succ(s_T)          # Builder's declared success predicate
    y_true     = y_engine(tau)       # engine-level ground truth
agreement = mean(y_declared == y_true over D_cal)
if agreement < 0.95:
    return FLAGGED("succ disagrees with engine on calibration set (agreement=%.3f)" % agreement)

# Phase 1 - Symbolic monotonicity check (no data needed)
# BFS over S is capped at 10^5 states; if the cap is hit, fall back to symbolic
# reachability with BDDs and report the cap in the audit log.
for each trajectory prefix s_0..s_t of length t <= H:
    W_t   = sum_{i<=t} w_{i}                    # cumulative weighted progress
    W*_t  = max over reachable prefixes of W_t   # upper envelope via BFS over S (cap 10^5 states)
    if succ(s_t) == False and W*_t > W*_{t-1} and no phi_k newly satisfied at t:
        return FLAGGED("progress-without-success at t")

# Phase 2 - Composition-algebra check
for each pair (phi_j, phi_k) with declared compose(phi_j, phi_k) in alg:
    if not implies(compose(phi_j, phi_k), succ) and alg.claims_sufficient(phi_j, phi_k):
        return FLAGGED("non-composing sub-goal pair (j,k)")

# Phase 3 - Empirical certificate on D_T (guards against Builder spec drift)
for each trajectory tau in D_T:
    r_tau = sum_t w_t * 1[step t advances phi]
    y_tau = 1[succ(s_T)]
    collect (r_tau, y_tau)
fit isotonic regression r -> p(success) on D_T (monotone by construction)
if Spearman(r, y) < 0.5 or isotonic residual > eps=0.15:
    return FLAGGED("empirical non-monotonicity")

# Phase 4 - License and reweight
w* = w / sum(w)   # normalize so SFT loss weights sum to 1 per task
return LICENSED, w*
```

Flagged tasks are routed back to the Builder with the failing prefix, sub-goal pair, or calibration disagreement attached, exactly as Game2Player's existing repair loop routes player-reported problems (E4).

**Why this design.** We chose a *symbolic* monotonicity check over a purely empirical correlation check because the Builder already exposes the internal state machine and success predicate (E4), so the check is decidable on the spec itself and does not require collecting new trajectories — accepting the cost that the check is only as sound as the Builder's declared state abstraction. We chose a *declared composition algebra* over an implicit one because LVLM-Playground's four task types (Perceiving, QA, Rule Following, End-to-End Playing) have no composition rule (E1), and forcing the Builder to name `compose` makes the missing rule explicit and auditable — accepting the cost that some legitimate tasks will be flagged until the Builder supplies the algebra. We chose *isotonic regression* over a parametric monotonicity test because isotonic regression is the minimal monotone fit and its residual directly quantifies the violation, avoiding a calibration assumption about any learned score — accepting the cost that with few trajectories per task the residual is high-variance, which is why Phase 3 uses `eps=0.15` as a conservative gate. We added *Phase 0* because the symbolic checks in Phases 1–2 compare one Builder artifact against another; Phase 0 anchors succ to engine-level outcomes on a 512-trajectory calibration set, so a LICENSED certificate requires both internal consistency and external agreement. A domain expert would not describe PC-Audit as a variant of process reward models (Lightman 2023) or VinePPO-style per-step Monte-Carlo value estimation, because those *estimate* step values from rollouts whereas PC-Audit *checks a declared* progress measure against a declared success predicate and never estimates a value it does not already have.

**Why it measures what we claim.** The cumulative weighted progress `W*_t` measures *monotone refinement of the success predicate* because we assume the Builder's state abstraction is faithful (every state transition the game permits is representable in `S`), which makes `W*_t > W*_{t-1}` with `succ(s_t)=False` a genuine progress-without-success violation; this assumption fails when the Builder collapses hidden game state, in which case `W*_t` reflects the Builder's abstraction rather than the game. The composition-algebra check measures *sub-goal composability* because we assume `alg.compose` is a sound sufficient condition for `succ`, so a non-composing pair is a real gap; this assumption fails when the Builder over-claims sufficiency, in which case the check reflects the Builder's claim rather than the game's structure. The isotonic residual measures *empirical weight-success consistency* because we assume `D_T` is drawn from the same Builder spec being checked, so a monotone fit should have low residual; this assumption fails under distribution shift between collection and audit, in which case the residual reflects the shift rather than a spec defect. The Phase-0 calibration agreement measures *external faithfulness of succ* because we assume `D_cal` is labeled by the game engine or human verification independent of the Builder, so agreement bounds the false-license rate; this assumption fails when the calibration labels are themselves Builder-generated, in which case agreement reflects Builder self-consistency rather than grounding.

## Contribution

['PC-Audit: a per-task composition-certificate framework that checks a Builder-authored hierarchical progress measure for monotone refinement of the task-level success predicate under a declared composition algebra, producing a LICENSED/FLAGGED certificate and a re-normalized weight vector for progress-weighted SFT.', 'A design principle for game-playing visual-agent data pipelines: progress-weighted SFT is only as sound as the composition certificate of its weighting signal, so the certificate should gate training rather than be assumed.', "A repair-routing protocol that attaches the failing trajectory prefix or sub-goal pair to the Builder's existing repair loop, making composition defects actionable rather than silent."]

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | Game2Player 568 games / 3,436 tasks | Builder-exposed state and success predicates make the audit decidable |
| Calibration set | 512 held-out trajectories per task with engine-level win/lose labels (or human-verified success) | Tests the load-bearing assumption that succ is faithful to the game |
| Primary metric | Task success rate on PlayerBench after SFT | Directly tests whether licensed weights improve learning |
| Secondary metric | False-license rate: fraction of LICENSED certificates whose succ disagrees with engine on a held-out 512-trajectory calibration set | Empirically bounds certificate soundness |
| Baseline | Unfiltered progress-weighted SFT (Game2Player default) | Tests whether auditing weights matters at all |
| Baseline | Empirical-only monotonicity filter (Phase 3 alone) | Isolates the contribution of symbolic checks |
| Baseline | Random task flagging at matched rate | Controls for the benefit of simply discarding tasks |
| Ablation-of-ours | PC-Audit without Phase 2 composition check | Tests whether the algebra check carries weight |
| Ablation-of-ours | PC-Audit without Phase 0 grounding check | Tests whether calibration against engine labels changes the licensed set |

### Why this setup validates the claim

The central claim is that Builder-emitted progress weights can be silently non-monotone with respect to the declared success predicate, and that gating on this defect before SFT improves the learned visual agent. PlayerBench success rate is the right metric because it is the verifiable outcome the weights are supposed to predict; if auditing the weights does not move this number, the claim is empty regardless of how many tasks get flagged. The unfiltered-SFT baseline tests the core sub-claim that the default weights contain harmful noise. The empirical-only baseline tests whether symbolic checks add anything beyond what trajectory correlation already detects — if symbolic and empirical flagging produce identical downstream gains, Phases 1–2 are decorative. The random-flagging baseline at matched rate controls for the trivial possibility that discarding any tasks helps by shrinking a noisy set. The Phase-2 ablation isolates whether the composition algebra, the most novel and most assumption-laden component, earns its place. The Phase-0 ablation and the false-license metric directly test the load-bearing assumption: if removing Phase 0 does not change the licensed set or the false-license rate, the grounding step is redundant; if it does, the assumption was doing real work and must be checked. Together these comparisons form a falsifiable ladder: each baseline removes one mechanism, so a gain that survives all removals is attributable to the full audit.

### Expected outcome and causal chain

**vs. Unfiltered progress-weighted SFT** — On a task where the Builder assigns weight to a sub-goal predicate that can be satisfied without advancing toward `succ` (e.g. a pickup predicate reachable in a dead-end state), unfiltered SFT upweights trajectories that accumulate progress but fail, teaching the VLM agent to imitate dead-end behavior. Our method flags that task in Phase 1 by finding a prefix with `W*_t > W*_{t-1}` and `succ=False`, routing it to the Builder before SFT ever sees it. We expect a noticeable PlayerBench gap concentrated on tasks whose specs contain progress-without-success prefixes, with near-parity on tasks that pass the audit untouched.

**vs. Empirical-only monotonicity filter** — On a task with few collected trajectories, the isotonic residual is high-variance, so the empirical-only filter either passes a genuinely broken spec by chance or flags a sound one; it cannot distinguish a spec defect from sampling noise. Our method catches the same defect symbolically in Phase 1 without needing trajectories, and uses Phase 3 only as a drift guard. We expect the empirical-only baseline to show gains on high-trajectory tasks but to lose most of its advantage on low-trajectory tasks, where our symbolic phases carry the improvement.

**vs. Random task flagging at matched rate** — On a task set where a fraction of specs are genuinely broken, random flagging discards sound tasks at the same rate it discards broken ones, so the surviving set has the same defect density as the original. Our method concentrates flags on the defective specs, lowering defect density in the training set. We expect random flagging to show little to no improvement over unfiltered SFT, while PC-Audit shows a gain that scales with the true defect rate — a pattern random flagging cannot produce.

**vs. PC-Audit without Phase 0 grounding check** — On a task where the Builder's succ over-claims success (e.g. declares win when the engine reports loss), Phases 1–3 can pass because they compare the Builder's weights to the Builder's own succ; the spec is internally consistent but externally wrong. Without Phase 0, such a task is LICENSED and its weights are used for SFT, teaching the agent a reward that disagrees with the game. With Phase 0, the 512-trajectory calibration set detects the disagreement (agreement < 0.95) and flags the task. We expect the no-Phase-0 ablation to show a higher false-license rate and a PlayerBench gap concentrated on tasks where succ disagrees with engine labels, while full PC-Audit shows near-zero false licenses on the calibration set.

### What would falsify this idea

If PC-Audit's gain over unfiltered SFT is uniform across tasks rather than concentrated on tasks the audit flags, the mechanism is not doing what we claim and the improvement must come from something else. Equally, if the Phase-2 ablation matches full PC-Audit, the composition algebra is not earning its cost and the central novelty claim collapses to Phase 1 alone; and if the false-license rate on the 512-trajectory calibration set is not lower with Phase 0 than without, the grounding step does not test the load-bearing assumption.

## References

1. Game2Player: Scaling Gameplay Experience for Visual Agents through Environment-Player Co-Development
