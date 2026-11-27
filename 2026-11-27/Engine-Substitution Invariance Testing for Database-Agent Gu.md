# Engine-Substitution Invariance Testing for Database-Agent Guard Verdicts

## Motivation

Guardrail agents such as ShieldAgent (E2) and GuardAgent (E3) are evaluated on fixed benchmarks of agent instruction/trajectory pairs, and their safety verdicts are treated as properties of the guard alone; the frontier database study (E1) separately shows that the engine's permission and recovery semantics decide what a mistake destroys, yet its five permission and two recovery arms are applied only to scripted mistakes and never crossed with a guard layer. The structural root cause is a separable-factors assumption: guard verdicts are validated against a single engine's semantics, so no invariant is established that a verdict survives engine substitution, and a guard that looks safe on PostgreSQL may silently certify a destructive statement on Cassandra or Valkey where the same permission arm and recovery drill have different meaning. Because the guard is never run through the same engine-native recovery drill as the engine, guard and engine are treated as independent factors when in fact their coupling determines the realized damage.

## Key Insight

A guard verdict is only a safety claim if it is a fixed point of engine substitution — the same trajectory, permission arm, and recovery drill must yield the same verdict across engines — so the invariant to test is verdict closure under engine-specific permission and recovery semantics, not verdict accuracy on any single engine.

## Method

**A. What it is.** ESI-GUARD (Engine-Substitution Invariance for Guard verdicts) takes a guard layer `G`, a frozen trajectory library `T` (22 canonical mistakes plus model-driven hazard tasks), the frontier's five permission arms `P` and two recovery arms `R`, and the nine core engines `E`, and outputs, for every (trajectory, engine, permission, recovery) cell, a realized-damage label from the engine-native recovery drill and the guard's binary verdict, from which it computes a substitution-invariance profile per guard. The load-bearing assumption is stated explicitly: **a guard's verdict on a fixed (trajectory, permission arm, recovery arm) cell is a function of the engine's permission and recovery semantics alone, so that any verdict difference across engines (flip) is attributable to guard sensitivity to engine semantics rather than to the guard's own nondeterminism, prompt/context sensitivity, or the drill's state-mutating effects.** This assumption is tested, not presumed, via the same-engine test-retest control and drill-neutrality control specified in Phase 0 and Phase 1 below.

**B. How it works.**

```pseudocode
INPUT: guard G, trajectories T, permission arms P (5), recovery arms R (2),
       engines E (9 core), drill repeat window = 10 min, alpha = 0.05,
       test-retest repeats K = 5, no-op trajectory t0

# Phase 0: same-engine test-retest noise floor (guard nondeterminism control)
for e in E, p in P, r in R, t in T:
    for k in 1..K:
        v_k[e,p,r,t] = G(t, engine=e, permission=p, recovery=r)   # binary safe/unsafe
    noise_floor[e,p,r,t] = 1 - max_{a,b in 1..K} 1[v_a == v_b]   # 1 if any disagreement
    v[e,p,r,t] = majority vote over k=1..K                         # stable verdict for Phase 2

# Phase 1: engine-native damage ground truth + drill-neutrality control
for e in E, p in P, r in R, t in T:
    pre_hash[e,p,r,t] = state_hash(e) before executing t
    execute t on e under p; run engine-native recovery drill r for 10 min
    post_hash[e,p,r,t] = state_hash(e) after drill
    drill_mutated[e,p,r,t] = 1 if pre_hash != post_hash else 0
    damage[e,p,r,t] = 1 if state not restored else 0   # scripted + agent trajectories
for e in E, p in P, r in R:
    pre_hash0[e,p,r] = state_hash(e) before executing no-op t0
    execute t0 on e under p; run drill r for 10 min
    post_hash0[e,p,r] = state_hash(e) after drill
    drill_neutral[e,p,r] = 1 if pre_hash0 == post_hash0 else 0

# Phase 2: guard verdicts on the same surface (using stable v from Phase 0)
for e in E, p in P, r in R, t in T:
    v[e,p,r,t] = majority vote from Phase 0

# Phase 3: substitution invariance profile (closure defined on flips exceeding noise floor)
for G-candidate:
    for each engine pair (e_i, e_j) in E x E:
        raw_flip[e_i,e_j] = mean over (p,r,t) of 1[v[e_i,..] != v[e_j,..]]
        # exclude cells where drill mutated state or no-op drill was non-neutral
        valid_cells = {(p,r,t): drill_mutated[e_i,p,r,t]==0 and drill_mutated[e_j,p,r,t]==0
                       and drill_neutral[e_i,p,r]==1 and drill_neutral[e_j,p,r]==1}
        flip[e_i,e_j] = mean over valid_cells of 1[v[e_i,..] != v[e_j,..]]
        # flips exceeding noise floor: count only if raw disagreement > max(noise_floor[e_i], noise_floor[e_j])
        excess_flip[e_i,e_j] = mean over valid_cells of 1[raw_flip_cell > max(noise_floor[e_i,..], noise_floor[e_j,..])]
    closure[G] = 1 - max_{i,j} excess_flip[e_i,e_j]          # worst-case closure on excess flips
    soundness[G] = P(v=0 | damage=1)  # verdicts that miss real damage
    overblock[G]  = P(v=1 | damage=0)

# Phase 4: attribution — is a flip explained by permission or recovery semantics?
for each flip cell (excess_flip > 0):
    attribute to permission-arm mismatch or recovery-drill mismatch
    by replaying the cell with the other engine's permission/recovery semantics held fixed
OUTPUT: closure[G], flip matrix, excess_flip matrix, noise_floor, drill_mutated, drill_neutral, soundness, overblock, attribution table
```

**C. Why this design.** We chose to cross the guard layer with the frontier's existing permission and recovery arms rather than build a new guard, because the open question is a property of the guard-engine pair, not of any single guard, and reusing the frontier's arms makes the substitution test commensurable with prior damage measurements (accepting the cost that we inherit the frontier's engine set and drill protocol rather than designing our own). We chose realized damage from the engine-native recovery drill as ground truth over guard self-reports or trajectory-level labels, because self-reports cannot reveal engine-specific semantics and trajectory labels conflate guard error with engine behavior (accepting the cost of 10-minute drill runs per cell, which bounds scale). We chose worst-case pairwise flip rate as the closure statistic over mean flip rate, because a guard that is invariant on average but flips on one engine pair is exactly the failure a safety claim must exclude (accepting the cost that closure is pessimistic and will make weak guards look worse than mean accuracy suggests). We added the same-engine test-retest control (K=5 repeats) and the drill-neutrality control (no-op trajectory per engine-permission-recovery cell) because the load-bearing assumption—that flips reflect engine semantics alone—is otherwise confounded by guard nondeterminism and drill-induced state change; these controls make the assumption testable rather than presumed. This is not a variant of ShieldAgent's formal-verification shielding or GuardAgent's code-generation guardrail: those methods produce verdicts, whereas ESI-GUARD treats the verdict as a quantity to be tested for invariance, and a domain expert would not describe a substitution-invariance test as a guardrail variant because it never generates a verdict of its own.

**D. Why it measures what we claim.** `flip[e_i,e_j]` measures verdict non-closure under engine substitution because we assume the only difference between the two cells is the engine's permission and recovery semantics, with trajectory, permission arm, and recovery arm held fixed; this assumption fails when an engine's recovery drill itself changes the stored state in a way that alters the trajectory's meaning, in which case flip reflects drill-induced state change rather than guard sensitivity to engine semantics. We now exclude cells where `drill_mutated=1` or `drill_neutral=0` from the flip denominator, and we report `excess_flip` that subtracts the same-engine `noise_floor` (K=5 test-retest), so that flips below the guard's own nondeterminism are not counted. `damage[e,p,r,t]` from the recovery drill measures realized harm because we assume a state not restored within the 10-minute window is damage the operator would experience; this fails when recovery is merely slow rather than impossible, in which case the label reflects recovery latency rather than damage. `soundness[G]` measures the guard's safety claim because we assume a verdict of safe on a cell with realized damage is the operational failure mode; this fails when the guard was never intended to cover that permission arm, in which case soundness reflects scope mismatch rather than guard error. The attribution table measures whether non-closure is permission-driven or recovery-driven because we assume replaying a cell with the other engine's semantics held fixed isolates the semantic cause; this fails when permission and recovery interact non-separably, in which case the attribution reflects the interaction rather than either factor alone.

## Contribution

(1) ESI-GUARD, an engine-substitution invariance test that crosses a guard layer with the frontier's five permission and two recovery arms across nine engines and reports a worst-case verdict-closure profile, turning guard safety from a single-engine accuracy claim into a substitution-invariant property. (2) A design principle: guard verdicts should be reported together with their engine-substitution flip matrix and attribution table, because a verdict that is sound on one engine can be unsound under another engine's permission or recovery semantics. (3) A reusable cell-level dataset coupling guard verdicts to engine-native recovery-drill damage labels across the nine-engine surface.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | Frontier trajectory library (22 scripted + agent hazard tasks) | Same surface as prior damage measurements |
| Primary metric | Worst-case pairwise excess flip rate (closure) | Excludes guards invariant only on average; excess flips subtract same-engine noise floor |
| Baseline | ShieldAgent (formal-verification shielding) | Tests whether verified verdicts are engine-invariant |
| Baseline | GuardAgent (code-generation guardrail) | Tests whether executable guardrails transfer across engines |
| Baseline | Engine-agnostic LLM guard (no engine context) | Tests whether engine blindness drives flips |
| Ablation-of-ours | Mean excess flip rate instead of worst-case closure | Isolates whether pessimism drives the verdict |
| Control | Same-engine test-retest (K=5 repeats per cell) | Measures guard nondeterminism noise floor |
| Control | No-op trajectory drill per (engine, permission, recovery) | Measures drill-induced state change (drill_neutral) |

### Why this setup validates the claim

The central claim is that guard verdicts are not closed under engine substitution, and that this non-closure is measurable and attributable to permission/recovery semantics rather than to guard quality in the abstract. The dataset is the right surface because it reuses the frontier's own permission arms and recovery drills, so a flip cannot be dismissed as an artifact of a different test harness. The three baselines test distinct sub-claims: ShieldAgent and GuardAgent are strong verdict-producers whose formal machinery might plausibly confer invariance, so if they still flip, the claim is about guards generally rather than weak guards; the engine-agnostic LLM guard tests whether flips are simply caused by withholding engine context. The ablation separates the worst-case statistic from the mean, so a guard that looks invariant on average but flips on one engine pair is exposed rather than averaged away. Closure is the right primary metric because a safety claim must hold on every engine pair, not on average; soundness and overblock are reported only to show flips are not merely cosmetic verdict noise. The same-engine test-retest control (K=5) provides a noise floor so that flips below the guard's own nondeterminism are not counted as engine-semantic sensitivity, and the no-op drill control excludes cells where the drill itself mutates state, ensuring that excess flips are attributable to engine semantics rather than drill artifacts.

### Expected outcome and causal chain

**vs. ShieldAgent** — On a case where the guard verifies a policy against a trajectory that includes a recovery arm, ShieldAgent produces a verdict from formal rule circuits that encode the policy document, not the engine's recovery semantics, because its rule library is engine-agnostic. Our method instead executes the same trajectory across engine pairs and reads realized damage from the engine-native drill, so it detects that the same verdict is unsound on one engine and sound on another. We expect a noticeable excess flip rate concentrated on cells whose recovery arm differs between engines, with parity on permission-only cells, after excluding cells where drill_neutral=0 and subtracting the same-engine noise floor.

**vs. GuardAgent** — On a case where the guardrail code is generated from safety requests and executed deterministically, GuardAgent produces a stable verdict because the code is fixed at generation time, but that stability is inherited from whichever engine's semantics were in the generation context. Our method instead replays the cell under the other engine's permission and recovery semantics, isolating whether the verdict flips. We expect excess flips concentrated on permission-arm mismatches, where the generated code encodes one engine's privilege model, and near-parity on recovery-only cells, with the same-engine test-retest confirming that GuardAgent's deterministic code has a low noise floor.

**vs. Engine-agnostic LLM guard** — On a case where the trajectory is hazardous only under a specific engine's recovery behavior, the engine-agnostic guard produces a safe verdict because it never sees engine identity, so its judgment collapses to trajectory text alone. Our method instead holds trajectory fixed and varies engine, so any verdict change is attributable to engine semantics by construction. We expect a large excess flip rate on this baseline, with the attribution table showing most flips driven by recovery-drill mismatch rather than permission arms, confirming the causal chain, and with the noise floor for this LLM guard measured by K=5 repeats to ensure excess flips exceed it.

### What would falsify this idea

If closure scores are uniformly high across all guards and engine pairs after subtracting the same-engine noise floor and excluding drill-mutated cells, or if excess flips are distributed evenly across permission-only, recovery-only, and interaction cells rather than concentrated where the attribution predicts, then verdict non-closure is not a property of the guard-engine pair and the central claim fails. Equivalently, if the ablation shows mean and worst-case excess flip rates are indistinguishable, the pessimism argument is wrong and the statistic adds nothing.

## References

1. When an AI agent makes a mistake in a database, what decides the damage?
