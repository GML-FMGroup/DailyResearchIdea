# Deadline-Bounded Controller Supervision with Fallback Substitution for Real-Time Game Agents

## Motivation

AutoPILOT expands the model's action interface to primitive actions, open-loop macros, and closed-loop controllers, and reports large real-time gains over VISTA (68.6% vs 38.3% on AFK-Bench), but its own framing treats action expressivity as action reliability: once a controller is emitted, the harness has no invariant that maps a failing controller outcome to a defined next action, so a controller that stalls, spams no-ops, or never hands back simply runs. BALROG's finding that several models perform worse when given visual representations (E7) shows this is a general property of vision-language game agents, not a single-tree quirk: perception-driven control degrades silently, and the field's trajectory toward longer-lived autonomous execution makes silent degradation the binding bottleneck rather than action coverage. The root cause is structural — AutoPILOT's action set is open under controller failure because the hand-back protocol is a convention the controller is assumed to honor, not a supervised state transition.

## Key Insight

A closed-loop controller is a program whose liveness and progress can be witnessed externally from the observation stream alone, so pairing every controller with an externally checked deadline-and-progress contract makes controller execution a total function from controller outcome to next action rather than a partial one the model is trusted to complete.

## Method

**What it is.** We introduce SCK (Supervisory Control Kernel), a wrapper around the AutoPILOT harness that compiles each emitted closed-loop controller into a *supervised controller* carrying an explicit contract, and that owns a single total transition function `next_action(controller_outcome) -> {continue, substitute, reissue}`. Inputs: the controller source emitted by the model, the per-frame observation stream, the per-frame action stream, and the harness budget clock. Outputs: a stream of primitive actions to the game, plus a supervision trace (contract events, substitutions, re-issues) used for evaluation.

**Load-bearing assumption (stated explicitly).** The method assumes that an *action-effect prediction error* — the divergence between the frame predicted from the emitted action and the observed next frame — is a valid external witness of controller liveness and progress in real-time games: a controller that is genuinely advancing game state produces low prediction error, while a failing controller (stuck, spamming no-ops, desynchronized) produces high prediction error. This is the assumption the entire substitute/re-issue transition function is keyed off. It is *not* the assumption that raw pixel change indicates progress; raw pixel change is confounded by non-agent motion (scrolling backgrounds, particle effects, NPC motion, camera drift), so the witness is the *residual* after an action-effect model accounts for the action's expected visual consequence.

**How it works.**

```pseudocode
# --- Phase 0: action-effect model (fixed, per-game-agnostic) ---
# A small convolutional predictor f_theta(frame_t, a_t) -> frame_{t+1}_hat
# trained offline on logged (frame, action, next_frame) triples from
# scripted/human play across the AFK-Bench training split (see calibration).
# Architecture: 4-layer CNN encoder (32/64/64/128 channels, 3x3 kernels,
# ReLU), action embedding (dim 32) concatenated at the bottleneck,
# 4-layer transposed-CNN decoder to full resolution. ~1.2M parameters.
# Trained with L1 loss on normalized frames, Adam lr=1e-3, batch=64,
# 50k steps. Frozen at evaluation time.

# --- Phase 1: contract compilation (once per model call) ---
def compile(controller_src, screen_0, budget):
    contract = {
      max_wall_ms   = 0.6 * budget.remaining_ms,   # deadline fraction
      max_frames    = 900,                          # hard frame cap
      progress_win  = 120,                          # frames per progress window
      min_progress  = 0.02,                         # action-effect residual floor
      handback_ok   = lambda s: s.handback_flag,     # explicit protocol
      sandbox_kill_ms = contract.max_wall_ms,       # hard kill on controller process
    }
    return SupervisedController(controller_src, contract, screen_0)

# --- Phase 2: per-frame supervision loop ---
def step(sc, frame, t):
    a = sc.controller.act(frame)          # the model-written controller acts
    sc.trace.append((t, a, frame))
    # --- progress witness: action-effect prediction residual over the window ---
    # residual_t = mean_abs_diff(f_theta(frame_{t-1}, a_{t-1}), frame_t) / 255.0
    # progress = mean residual over the last progress_win frames
    sc.win.append((frame, a))
    if len(sc.win) > contract.progress_win: sc.win.pop(0)
    progress = mean_action_effect_residual(sc.win, f_theta)
    # --- failure detectors (all external, none read model confidence) ---
    stalled   = progress < contract.min_progress
    noop      = (a == sc.prev_action) and stalled
    overtime  = (t - sc.start_t) > contract.max_wall_ms
    overframe = sc.n_frames > contract.max_frames
    no_handbk = overtime and not contract.handback_ok(frame)
    if stalled or noop or overtime or overframe or no_handbk:
        return substitute(sc, reason=first_true(...))
    sc.prev_action = a
    return a

# --- Phase 3: total transition on failure ---
def substitute(sc, reason):
    # 1) try a predefined safe fallback from a small per-game-agnostic library
    fb = SAFE_FALLBACKS[reason]      # e.g. NOOP-then-UNDO, or random legal primitive
    # 2) if the same reason fires twice for this controller, re-issue model control
    sc.fail_count[reason] += 1
    if sc.fail_count[reason] >= 2:
        return REISSUE_MODEL(reason, sc.trace)   # hand back to harness, budget debited
    return fb
```

Hyperparameters are inline: `0.6` deadline fraction of remaining budget, `900`-frame cap, `120`-frame progress window, `0.02` action-effect residual floor, and the `2`-strike re-issue threshold. `SAFE_FALLBACKS` is a fixed table keyed by failure reason (stall → short random legal primitive burst; noop → undo/back; overtime/no-handback → no-op then re-issue), not learned per game. The action-effect model `f_theta` is trained once offline and frozen; it is not a per-game policy and does not play the game.

**Why this design.** We chose an *external action-effect prediction residual* over raw pixel change as the progress witness because raw pixel change is confounded by non-agent motion (scrolling backgrounds, particle effects, NPC motion, camera drift) that produces large frame differences while the agent is stuck, so a raw-difference detector would not fire on the canonical stuck-loop case; the residual after subtracting the action's expected visual consequence is a more specific witness of whether the agent's action is advancing game state. We chose an *external observation-based* witness over any model-reported self-assessment because the failure we target is exactly the case where the controller's internal state is untrustworthy; the cost is that a game with a static-but-correct screen (e.g., a turn-based menu awaiting input) can be misread as stalled, which is why the `no_handbk` and `noop` detectors are gated on the same window rather than firing on a single frame. We chose a *fixed safe-fallback table* over a learned fallback policy because the kernel's job is to guarantee a defined next action, not to play well; a learned fallback would reintroduce the same silent-failure surface we are closing, accepting the cost that fallbacks are weak and only buy time for a re-issue. We chose a *two-strike re-issue* over immediate re-issue because a single transient detector trip should not burn an API call, accepting the cost that a genuinely broken controller consumes one extra fallback cycle. We chose an *interruptible execution sandbox* (hard wall-clock kill on the controller process at `sandbox_kill_ms`) over a post-hoc hand-back flag check so that a non-returning controller is caught by construction rather than by convention, accepting the cost that the sandbox must be able to preempt the controller's own inner loops. Unlike AutoPILOT, which extends the action *interface*, and unlike BALROG, which only *measures* vision-decision deficiency, SCK changes the action *execution semantics*: a domain expert would describe it as adding a supervisor/contract layer to a program-synthesis harness, not as a variant of AutoPILOT's macro/controller taxonomy, because the mechanism is a total transition function over controller outcomes rather than a new action type.

**Why it measures what we claim.** The action-effect residual `progress` measures *controller liveness* because we assume that a controller making progress on a real-time game produces low prediction error under the action-effect model within a 120-frame window; this assumption fails on static-await screens (menus, dialogue boxes) and on animation-heavy scenes where the action-effect model cannot account for ambient motion, in which case `progress` reflects *model residual* rather than controller failure, and the `no_handbk`/`noop` conjunction is what re-labels that case. The `max_wall_ms = 0.6 * remaining` deadline measures *hand-back compliance* because we assume the model-written controller's runtime is bounded by the harness budget it was compiled against; this fails when a controller spawns its own long inner loop that the harness cannot interrupt, in which case the deadline reflects *harness preemption latency* rather than controller intent — which is why the sandbox kill is set to the same deadline so the transition function is total by construction. The `fail_count >= 2` re-issue threshold measures *persistent controller inadequacy* because we assume two independent detector trips on the same reason are not both transient; this fails under a single long stall that trips the same detector twice, in which case the threshold reflects *stall duration* rather than a broken controller. Each of these is a named assumption, and the supervision trace records which one fired so the failure mode is auditable rather than hidden.

## Contribution

(1) SCK, a supervisory action-execution kernel that compiles every model-emitted closed-loop controller into a contract-bearing supervised controller and supplies a total transition function from controller outcome to next action (continue / safe fallback / model re-issue). (2) A design principle: controller reliability in real-time game agents is separable from controller expressivity, and can be enforced by external observation-stream witnesses (progress window, deadline, hand-back protocol) rather than by model self-report. (3) A supervision trace format and an AFK-Bench-derived failure-mode taxonomy (stall, no-op, overtime, hand-back timeout) that makes controller failure a first-class, measurable outcome.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | AFK-Bench real-time browser games, 40-game stratified subset (by inaction-response category: 10 games per category across the four AFK-Bench categories) | Real-time clock exposes silent controller failure; stratified subset controls variance and API cost |
| Primary metric | Real-time success rate | Directly measures liveness under running clock |
| Baseline | AutoPILOT harness | Tests whether contract layer adds beyond interface extension |
| Baseline | VISTA | Tests whether gains need supervision or just memory |
| Baseline | AutoPILOT minus supervision (no re-issue) | Isolates the total transition function's contribution |
| Baseline | Generic watchdog (fixed 5s timeout + no-op fallback, no contract compilation, no two-strike re-issue) | Tests whether SCK's specific transition semantics beat a generic timeout-and-fallback layer |
| Ablation-of-ours | SCK with model self-reported progress | Tests external action-effect witness vs internal self-assessment |
| Ablation-of-ours | SCK with raw pixel-change witness (no action-effect model) | Tests whether the action-effect residual is necessary over raw frame difference |
| Calibration set | 512 labeled (frame, action, next_frame) triples from scripted/human play on the AFK-Bench training split, labeled stall/non-stall by a scripted stuck-loop injector | Grounds the 0.02 threshold and reports AUC/ROC separation of the witness |

### Why this setup validates the claim

The central claim is that silent failure of model-written closed-loop controllers — where the controller keeps acting but stops making progress — is a distinct failure surface that interface-level harness design does not close, and that an external contract-plus-total-transition layer does. AFK-Bench in real-time mode is the right dataset because the running clock makes silence costly: a controller that stops progressing loses wall-clock budget that paused-clock evaluation would refund, so success rate there is sensitive precisely to the mechanism we add. The 40-game stratified subset is chosen to control variance and API cost while preserving coverage of the inaction-response categories that determine whether a stall is terminal or transient; a pilot on 10 games (2-3 per category) estimates per-category variance and required sample size before the full sweep. AutoPILOT is the correct primary baseline because it already fixes what the model sees and remembers and what action types it can emit; if SCK's gain survives against it, the gain cannot be attributed to interface richness. VISTA tests the weaker counter-hypothesis that memory alone suffices. The generic-watchdog baseline tests whether any timeout-and-fallback layer would suffice, isolating SCK's specific transition semantics (contract compilation, action-effect witness, two-strike re-issue) from a generic watchdog. The no-supervision ablation is the load-bearing one: it keeps AutoPILOT's controller interface but removes contract compilation and the continue/substitute/reissue transition, so any gap between it and full SCK is attributable to the supervisor rather than to the controller-writing prompt. The self-reported-progress ablation tests our specific design choice of an external action-effect witness over internal self-assessment, and the raw-pixel-change ablation tests whether the action-effect residual is necessary over raw frame difference. The calibration set grounds the 0.02 threshold: we report the AUC/ROC separation of the action-effect residual between known-good (scripted/human) and known-stalled (injected stuck-loop) traces, and the false-positive rate of each detector on static-await and animation-heavy game categories. Real-time success rate is the right primary metric rather than API cost or per-call accuracy because the claim is about outcomes over time, not about any single model call being good.

### Expected outcome and causal chain

**vs. AutoPILOT** — On a case where the model writes a controller that enters a stuck loop against a wall or a repeating menu, AutoPILOT produces actions indefinitely because its harness converts each model call into at most one controller and never inspects controller outcome; the controller runs until the model is next called. Our method instead trips the stalled/noop detector on the 120-frame action-effect residual window, substitutes a safe fallback, and re-issues after two strikes because the total transition function owns the outcome. We expect a modest but consistent gap concentrated on games with long-horizon locomotion and menu navigation, with near-parity on short arcade games where controllers rarely stall.

**vs. VISTA** — On a case where the model's controller has correct memory of the level layout but its emitted action stream desynchronizes from the screen after a transient animation, VISTA produces a controller that acts confidently on stale context because better memory does not detect that the controller has stopped changing the screen. Our method instead detects the same stall through action-effect residual evidence that is independent of memory quality, substitutes, and re-issues. We expect SCK to hold most of AutoPILOT's real-time advantage over VISTA and to widen it on the desynchronization-prone subset, with the gap largest where VISTA's paused-to-real-time drop is steepest.

**vs. AutoPILOT minus supervision** — On a case where a single transient detector trip would have been recoverable, the no-supervision ablation produces a controller that either never recovers or burns a model call on the first trip, because it lacks the two-strike fallback-then-reissue policy. Our method instead absorbs one trip with a cheap fallback and only spends budget on re-issue at the second. We expect a smaller but real gap concentrated in games where stalls are transient rather than terminal, confirming that the transition function's ordering — not merely the presence of a detector — carries the gain.

**vs. Generic watchdog** — On a case where a controller stalls against a wall while ambient animation continues, the generic watchdog's fixed 5s timeout fires only after the full timeout and its no-op fallback does not re-issue, so it either wastes budget or never recovers; our method's action-effect residual fires earlier (within the 120-frame window) and its two-strike re-issue recovers. We expect SCK to outperform the generic watchdog on the stall-prone subset, with near-parity on games where stalls are rare, confirming that the specific transition semantics — not merely the presence of a timeout-and-fallback layer — carry the gain.

**vs. SCK with model self-reported progress** — On a case where the controller's internal state is untrustworthy (the exact failure we target), the self-reported-progress ablation fails to detect the stall because the controller reports progress it is not making; our external action-effect residual detects it. We expect the self-reported ablation to match SCK on easy games and diverge sharply on the stall-prone subset, confirming the external witness is necessary.

**vs. SCK with raw pixel-change witness** — On a case where ambient motion (scrolling background, particle effects) produces large raw frame differences while the agent is stuck, the raw-pixel-change ablation fails to fire the stalled detector; our action-effect residual subtracts the action's expected visual consequence and fires. We expect the raw-pixel-change ablation to underperform SCK on animation-heavy games and match it on static-await games, confirming the action-effect model is necessary over raw frame difference.

### What would falsify this idea

If SCK's gain over AutoPILOT is uniform across games rather than concentrated on the stall-prone, long-horizon, desynchronization-prone subsets where the predicted failure mode lives, or if the self-reported-progress ablation matches the external action-effect witness, or if the raw-pixel-change ablation matches the action-effect residual on animation-heavy games, or if the generic watchdog matches SCK on the stall-prone subset, then the contract-and-supervisor mechanism is not what produces the improvement and the central claim is wrong.

## References

1. AutoPILOT: A Visual Agent Harness for Real-Time Gameplay
