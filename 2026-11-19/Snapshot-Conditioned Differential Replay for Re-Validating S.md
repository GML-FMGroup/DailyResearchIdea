# Snapshot-Conditioned Differential Replay for Re-Validating Synthesized GUI Checkpoint Hooks

## Motivation

StatePortal treats the per-application restoration hook as a durable artifact synthesized once by an onboarding agent, and accepts a checkpoint only through two independent gates (a programmatic-state oracle and a visual oracle) that each judge the hook-restored state in isolation. Structurally, this leaves hook-to-checkpoint replay equivalence under source mutation unestablished: because the hook is never re-executed against the original interaction trajectory on the same source snapshot, a hook can silently drift when the source evolves while still passing both static oracles, so the oracle pair is never calibrated to detect hook-original divergence. The root cause is that the hook is treated as a one-shot artifact rather than as a function of the source snapshot, and the two oracles are never forced to agree on the same trajectory.

## Key Insight

If the hook is re-derived as a function of the source snapshot and its restored state is compared against the original interaction trajectory re-executed on that identical snapshot, then any hook-original divergence is a snapshot-conditioned residual that both static oracles are blind to, making trajectory equivalence a measurable quantity rather than an assumed property.

## Method

**What it is.** DRH (Differential Replay Hook) takes StatePortal's checkpoint C, its synthesized hook H, the original natural-language test case T, and the current source snapshot S, and outputs a divergence report Δ plus a re-synthesis trigger decision; when Δ exceeds a threshold, it re-invokes the onboarding agent to produce H' conditioned on S and re-runs the differential replay until Δ is within tolerance or a budget is exhausted.

**Load-bearing assumption (stated explicitly).** On an identical source snapshot S, the original interaction trajectory execute(T, S) is deterministic enough that any residual difference between the hook-restored terminal state and the trajectory-reached terminal state is attributable to hook drift rather than run-to-run stochasticity, so Δ is a valid measure of hook-original divergence. This assumption is not taken on faith: DRH estimates a per-task stochasticity floor σ by executing execute(T, S) N=3 times on the same snapshot with pinned seeds and mocked network/time sources, and conditions the re-synthesis trigger on hook-attributable divergence Δ_hook = d(hook, traj) − σ rather than on the raw residual.

**How it works.**

```pseudocode
INPUT: checkpoint C, hook H, test case T, source snapshot S, tolerance tau, budget K, repeats N=3, noise multiplier k=2
# Phase 0: estimate per-task stochasticity floor sigma on the SAME snapshot S
for i in 1..N:
    traj_i = execute(T, S, seed=i, mock_network=True, mock_time=True)   # pinned seeds, mocked nondeterminism
    state_i, render_i = final_state(traj_i)
sigma_state  = mean_{i<j} programmatic_oracle(state_i, state_j)          # run-to-run state variance
sigma_visual = mean_{i<j} visual_oracle(render_i, render_j)              # run-to-run render variance
sigma        = w_s * sigma_state + w_v * sigma_visual                    # w_s=0.6, w_v=0.4
# Phase 1: two independent replays on the SAME snapshot S
state_hook, render_hook = restore(C, H, S)            # hook-restored path
traj_orig               = execute(T, S, seed=0, mock_network=True, mock_time=True)  # original interaction trajectory
state_orig, render_orig = final_state(traj_orig)      # terminal state of original path
# Phase 2: joint trajectory-equivalence test with noise calibration
Delta_state  = programmatic_oracle(state_hook, state_orig)   # structural diff of state maps
Delta_visual = visual_oracle(render_hook, render_orig)       # calibrated cross-VLM diff
Delta_raw    = w_s * Delta_state + w_v * Delta_visual        # w_s=0.6, w_v=0.4
Delta_hook   = Delta_raw - sigma                             # hook-attributable divergence
# Phase 3: snapshot-conditioned re-synthesis loop
k_iter = 0
while Delta_hook > tau + k * sigma and k_iter < K:           # tau=0.05, K=3, k=2
    H = onboarding_agent.rebind_or_synthesize(C, T, S)       # hook is a function of S
    state_hook, render_hook = restore(C, H, S)
    Delta_raw  = w_s * programmatic_oracle(state_hook, state_orig) \
               + w_v * visual_oracle(render_hook, render_orig)
    Delta_hook = Delta_raw - sigma
    k_iter += 1
OUTPUT: H_final, Delta_raw, Delta_hook, sigma, accepted = (Delta_hook <= tau + k * sigma)
```

Each replay executes against the *same* S, so the only difference between the two paths is the mechanism by which the target state is reached (hook restoration vs. original interaction), which is exactly the quantity of interest. The N-repeat Phase 0 makes the stochasticity floor σ an explicit, logged quantity so that the trigger is conditioned on hook-attributable divergence rather than on raw residual.

**Why this design.** We chose to re-execute the original interaction trajectory on the same snapshot rather than compare against a stored golden state because a stored golden state cannot distinguish source-induced drift from hook-induced drift, accepting the cost that every validation pays the full interaction cost of T (N+1 executions per validation, N=3). We chose to fuse the two oracles into a single weighted divergence score rather than keep them as independent acceptance gates because independent gates can both pass while disagreeing on the same trajectory (the exact failure StatePortal's design leaves open), accepting the cost that a single scalar can mask which modality diverged and requires the per-modality report to be logged alongside it. We chose to treat the hook as a function of S and re-synthesize on divergence rather than cache and reuse H across snapshots because the structural relationship between source and hook is what makes equivalence meaningful, accepting the cost that re-synthesis invokes the onboarding agent up to K=3 times per validation. We chose to estimate σ via N=3 repeats with pinned seeds and mocked network/time sources rather than assume zero stochasticity because the load-bearing assumption is that execute(T, S) is deterministic enough; measuring σ makes that assumption explicit and bounds the spurious-trigger rate. A domain expert would not describe this as a variant of StatePortal's hybrid acceptance: StatePortal scores each checkpoint against two static oracles, whereas DRH scores the *hook-vs-trajectory gap* on a shared snapshot, which no static oracle can express.

**Why it measures what we claim.** The weighted divergence Δ_hook measures hook-original replay equivalence because we assume that, on an identical snapshot S with pinned seeds and mocked network/time sources, any difference between the hook-restored terminal state and the original-trajectory terminal state beyond the measured stochasticity floor σ is attributable to the hook rather than to the environment; this assumption fails when the original interaction trajectory is itself nondeterministic beyond what N=3 repeats can estimate (e.g., time-seeded rendering, async network races), in which case Δ_hook reflects residual run-to-run stochasticity rather than hook drift and σ under-estimates the floor. The programmatic-state component Δ_state measures structural state fidelity because we assume the state map exposes all fields the test case constrains; this assumption fails when the test case depends on state the programmatic oracle does not serialize, in which case Δ_state under-reports divergence. The visual component Δ_visual measures rendering fidelity because we assume the calibrated cross-VLM oracle is monotone in perceptual difference for the target application; this assumption fails on applications whose visual vocabulary was not represented in the calibration set, in which case Δ_visual reflects oracle unfamiliarity rather than render divergence. The re-synthesis trigger (Δ_hook > τ + k·σ) measures the necessity of rebinding because we assume τ + k·σ separates benign nondeterminism from genuine hook drift; this assumption fails when nondeterminism exceeds τ + k·σ, in which case the trigger fires spuriously and the budget K bounds the cost.

## Contribution

(1) A differential replay oracle that converts StatePortal's two independent acceptance gates into a joint trajectory-equivalence test executed on a shared source snapshot, yielding a snapshot-conditioned divergence score for each checkpoint-hook pair. (2) A snapshot-conditioned re-synthesis loop that treats the hook as a function of the source snapshot and uses divergence as the trigger, together with a per-modality divergence report that localizes whether state or render fidelity broke. (3) A design principle: hook validity under source mutation is a measurable residual against the original trajectory, not an assumption baked into one-shot onboarding.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | DeepStateBench (61 deep-state tasks, 10 apps) | Purpose-built for deep-state checkpoint restoration; spans five language families. |
| Primary metric | Hook-original replay equivalence rate at Δ_hook ≤ τ + k·σ | Directly measures the claimed hook-vs-trajectory gap, noise-calibrated. |
| Baseline | StatePortal static hybrid acceptance | Tests whether independent static gates miss hook drift. |
| Baseline | Cached-hook reuse across snapshots | Tests whether snapshot-conditioned re-synthesis is required. |
| Baseline | Single-oracle (state-only) acceptance | Tests whether fusing visual oracle is necessary. |
| Ablation-of-ours | DRH without re-synthesis loop (K=0) | Isolates contribution of the re-synthesis trigger. |
| Ablation-of-ours | DRH without noise calibration (σ=0, raw Δ) | Isolates contribution of the N=3 stochasticity floor. |

### Why this setup validates the claim

The central claim is that hook-original replay equivalence on a shared snapshot captures a failure mode no static oracle can express. DeepStateBench is the right substrate because its tasks require long interaction histories, so hook-restored and trajectory-reached terminal states are genuinely distinct paths to the same target — the exact condition under which the gap can appear. The primary metric, equivalence rate at Δ_hook ≤ τ + k·σ, is the direct operationalization of the claim: if the method works, hook-restored states should match trajectory-reached states on the same snapshot at a higher rate than static acceptance predicts, and the N=3 noise calibration ensures that the measured gap is hook-attributable rather than run-to-run stochasticity. Each baseline isolates one sub-claim. StatePortal's static hybrid acceptance tests whether independent gates actually let divergent hooks through; if DRH does not beat it, the "both gates can pass while disagreeing" premise is empty. Cached-hook reuse tests whether the hook must be a function of S; if caching matches DRH, re-synthesis is wasted budget. Single-oracle acceptance tests whether the visual component is load-bearing. The K=0 ablation isolates the re-synthesis trigger from the differential-replay scoring itself, so any gain can be attributed to the loop rather than to the divergence metric alone. The σ=0 ablation isolates the contribution of the stochasticity floor: if DRH with σ=0 matches full DRH, the N=3 calibration is unnecessary and the load-bearing determinism assumption is not load-bearing in practice. Together these form a falsifiable test: the predicted effect is a gain concentrated on tasks where static gates pass but replay diverges, not a uniform lift.

### Expected outcome and causal chain

**vs. StatePortal static hybrid acceptance** — On a case where a source commit changes a serialization field that the programmatic oracle reads but a layout constant that only rendering reflects, static acceptance checks each oracle against its own stored reference and can pass both while the hook-restored render is stale relative to the trajectory-reached render. Our method instead replays both paths on the same snapshot and fuses the two oracles into one Δ_hook, so the visual component surfaces the render gap that the state oracle alone cannot. We expect a noticeable gap on tasks whose commits touch rendering-relevant source but leave serialized state stable, with parity on tasks where state and render co-vary.

**vs. Cached-hook reuse across snapshots** — On a case where a source change renames or relocates the state-restoration entry point, a cached hook still resolves to the old binding and restores a state that no longer corresponds to the current snapshot, producing a hook-restored terminal state that diverges from the trajectory-reached one. Our method treats the hook as a function of S and re-synthesizes within K=3 when Δ_hook exceeds τ + k·σ, so it recovers the binding on the current snapshot. We expect cached reuse to fail disproportionately on commits that alter restoration-relevant structure, with DRH closing most of that gap while both are comparable on structurally untouched commits.

**vs. Single-oracle (state-only) acceptance** — On a case where the hook restores a correct programmatic state but the checkpoint renders incorrectly, a state-only oracle reports equivalence and accepts a hook that a human would reject. Our method fuses the calibrated visual oracle at weight 0.4, so the render divergence pushes Δ_hook above τ + k·σ and triggers re-synthesis. We expect state-only acceptance to over-accept on precisely the tasks StatePortal flags as requiring hybrid scoring, while DRH's acceptance rate tracks the hybrid reference more closely, with the gap concentrated on visually divergent tasks.

**vs. DRH without noise calibration (σ=0, raw Δ)** — On a task with high run-to-run rendering variance (e.g., time-seeded animation), the raw Δ exceeds τ even when the hook is correct, so the σ=0 variant fires the re-synthesis trigger spuriously and consumes the K=3 budget chasing noise. Our method subtracts the measured σ and requires Δ_hook > τ + k·σ, so spurious triggers are suppressed. We expect the σ=0 variant to show a higher false-trigger rate and lower equivalence rate on high-variance tasks, with parity on low-variance tasks, confirming that the N=3 stochasticity floor is load-bearing.

### What would falsify this idea

If DRH's gain over static hybrid acceptance is uniform across all DeepStateBench tasks rather than concentrated on tasks where static gates pass but replay diverges, the claimed failure mode is not the source of the improvement and the central claim is wrong. Likewise, if the K=0 ablation matches full DRH, the re-synthesis loop contributes nothing and the hook-as-function-of-S premise is unsupported; if the σ=0 ablation matches full DRH, the noise calibration is unnecessary and the determinism assumption is not load-bearing in practice.

## References

1. StatePortal: Agentic Checkpoint Synthesis for Testing Deep States in Interactive Software
