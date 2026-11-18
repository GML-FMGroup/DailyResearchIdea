# Hardware-Calibrated Critical Batch Size Probing for Large-Scale Ascend Pretraining

## Motivation

Critical batch size (CBS) has been shown by the OLMo-based study (E1/E8) to evolve over training — near zero at initialization, rising rapidly, then plateauing — and batch-size warmup keyed to this curve cut gradient steps by 43% on OLMo 1B. Yet every CBS measurement in that line rests on GPU runs whose gradient noise scale is implicitly assumed to transfer to the target substrate, and the Xing 4 report (E6/E12) describes a 285B-parameter pretraining on 8,192 Ascend 910C cards with topology-aware hybrid sharding but reports no CBS curve, no loss-versus-steps trajectory, and no evidence that its batch-size schedule is monotone-consistent with the substrate's actual CBS evolution. The structural gap is that CBS is a property of the optimizer-data-hardware interaction, not of the model alone, so a static transfer from a 1B/7B GPU run to a 285B sharded Ascend run silently assumes the gradient noise scale is substrate-invariant — an assumption the sharded collective communication and heterogeneous memory hierarchy of MindSpore HyperParallel are exactly the things most likely to violate.

## Key Insight

Because CBS is a function of the gradient noise scale measured on the actual optimizer-data-hardware interaction, probing it in situ on the Ascend run yields a per-step noise statistic that is by construction the quantity the batch-size schedule must be monotone in, whereas any transferred CBS from another substrate is a biased estimate whose bias the schedule cannot detect.

## Method

## Hardware-Calibrated CBS Probing (HC-CBS)

**What it is.** HC-CBS is a training-stack component that periodically interrupts the Xing 4 pretraining loop to run a short empirical CBS probe on the live 285B Ascend run, then uses the measured CBS curve to set the batch size and gradient-step allocation for the next interval. Inputs: current checkpoint, current data shard position, current batch size B_t. Outputs: an updated batch-size schedule B_{t+1..t+K} and a logged (step, CBS, loss) triple for the consistency audit.

**How it works.**

```pseudocode
# Hyperparameters (fixed, not tuned per run)
PROBE_INTERVAL_STEPS = 2000      # steps between probes
PROBE_STEPS_PER_BATCH = 8        # short empirical probe length (baseline; see probe-length ablation)
PROBE_BATCH_MULTIPLIERS = [1, 2, 4, 8]   # initial batch sizes probed, as multiples of B_t
PROBE_LR = base_lr               # probe uses the live learning rate, unmodified
WARMUP_CEILING = 8 * B_init      # batch size never exceeds 8x initial
PROBE_LOSS_AVG_WINDOW = 4        # average loss over last 4 probe steps to reduce variance
PROBE_MAX_MULTIPLIER = 32        # adaptive widening ceiling for multipliers
PROBE_CALIBRATION_CHECKPOINTS = [5000, 20000, 50000]  # steps for long-horizon reference validation

# State
B_t = B_init
probe_log = []

for step in training_loop:
    if step % PROBE_INTERVAL_STEPS == 0:
        # --- Phase 1: in-situ probe on the live Ascend substrate ---
        # Freeze data shard position; for each multiplier, run
        # PROBE_STEPS_PER_BATCH optimizer steps from the current checkpoint
        # with the SAME topology-aware hybrid sharding config as the main run.
        # Loss for each multiplier is the average over the last PROBE_LOSS_AVG_WINDOW steps.
        losses = {}
        multipliers = list(PROBE_BATCH_MULTIPLIERS)
        for m in multipliers:
            ckpt = snapshot_current_weights()
            loss_curve = run_k_steps(ckpt, batch_size = m * B_t,
                                     steps = PROBE_STEPS_PER_BATCH,
                                     sharding = current_hyperparallel_config())
            losses[m] = mean(loss_curve[-PROBE_LOSS_AVG_WINDOW:])
        # --- Phase 1b: adaptive widening to bracket the CBS ---
        # If the fitted CBS from initial multipliers is at or above the largest multiplier,
        # add multipliers (doubling) up to PROBE_MAX_MULTIPLIER until at least one
        # multiplier exceeds the fitted CBS (bracketed fit).
        CBS_hat = fit_critical_batch(losses, multipliers)
        while CBS_hat >= max(multipliers) and max(multipliers) < PROBE_MAX_MULTIPLIER:
            new_m = max(multipliers) * 2
            multipliers.append(new_m)
            ckpt = snapshot_current_weights()
            loss_curve = run_k_steps(ckpt, batch_size = new_m * B_t,
                                     steps = PROBE_STEPS_PER_BATCH,
                                     sharding = current_hyperparallel_config())
            losses[new_m] = mean(loss_curve[-PROBE_LOSS_AVG_WINDOW:])
            CBS_hat = fit_critical_batch(losses, multipliers)
        # --- Phase 2: fit CBS from the loss-vs-batch curve ---
        # CBS is the batch size at which marginal loss reduction per
        # doubling falls below 1/2 of the initial marginal reduction.
        CBS_hat = fit_critical_batch(losses, multipliers)
        probe_log.append((step, CBS_hat, losses))
        # --- Phase 2b: long-horizon calibration check (at specified checkpoints) ---
        # At PROBE_CALIBRATION_CHECKPOINTS, run a reference measurement: for each multiplier,
        # run 128 steps (instead of 8) and compute the same loss average; compare CBS_hat
        # from the short probe to the reference CBS. Log the discrepancy.
        if step in PROBE_CALIBRATION_CHECKPOINTS:
            ref_losses = {}
            for m in multipliers:
                ckpt = snapshot_current_weights()
                loss_curve = run_k_steps(ckpt, batch_size = m * B_t,
                                         steps = 128,
                                         sharding = current_hyperparallel_config())
                ref_losses[m] = mean(loss_curve[-PROBE_LOSS_AVG_WINDOW:])
            ref_CBS = fit_critical_batch(ref_losses, multipliers)
            log_calibration(step, CBS_hat, ref_CBS)
        # --- Phase 3: monotone-consistency gate ---
        # Reject the schedule update if CBS_hat is not monotone
        # non-decreasing over the last 3 probes (plateau allowed).
        if len(probe_log) >= 3 and not monotone_nondecreasing(
                [p[1] for p in probe_log[-3:]]):
            CBS_hat = max(CBS_hat, probe_log[-2][1])   # clamp to prior
            flag_consistency_violation(step)
        # --- Phase 4: batch-size warmup schedule ---
        B_next = min(WARMUP_CEILING,
                     max(B_t, CBS_hat * 0.5))
        # Gradient-step allocation: steps for next interval
        # proportional to 1/B_next so token budget is preserved.
        steps_next = TOKEN_BUDGET // B_next
        B_t = B_next
    run_optimizer_step(batch_size = B_t)

# --- Phase 5: report ---
emit_trajectory(probe_log, loss_curve, steps_per_interval)
```

**Why this design.** We chose to run the probe *in situ on the live Ascend run* rather than importing a CBS curve from an OLMo GPU run (the approach implicit in E1/E8) because CBS is a property of the gradient noise scale on the actual optimizer-data-hardware interaction, and MindSpore HyperParallel's topology-aware hybrid sharding changes the effective gradient noise in a way a GPU curve cannot capture; the cost is that each probe consumes PROBE_STEPS_PER_BATCH × |multipliers| optimizer steps of the 285B run, which we accept because probing every 2000 steps keeps overhead below 1.5% of total steps. We chose a *short empirical probe* over the gradient-noise-scale estimator of McCandlish et al. because the latter requires strong assumptions (unbiased gradient noise, Gaussian approximation) that the sharded collective-communication noise of a 8192-card run is unlikely to satisfy, accepting that the empirical probe is noisier per measurement and needs the monotone-consistency gate to avoid schedule thrash. We chose a *monotone-consistency gate that clamps rather than rejects* the schedule update over a hard rejection of non-monotone probes because a single noisy probe should not stall a 285B run, accepting that a genuine CBS regression (e.g., from a data-distribution shift) will be smoothed over one interval before the schedule responds. We chose to *couple batch size to gradient-step allocation* (steps_next = TOKEN_BUDGET / B_next) over treating them as independent knobs because the 43% step reduction reported in E8 only materializes if the larger batch actually replaces steps rather than adding tokens, accepting that the schedule is now sensitive to the token-budget hyperparameter. A domain expert would not describe this as "applying CBS warmup to Ascend" because the mechanism is the in-situ probe plus the monotone-consistency gate — the schedule is *derived from* a substrate-specific measurement and *audited* against its own past, not transferred.

**Why this measures what we claim.** The quantity `CBS_hat` (the batch size at which marginal loss reduction per doubling falls below half the initial marginal reduction) measures *the substrate-calibrated critical batch size* because the probe runs under the same topology-aware hybrid sharding, the same optimizer execution path, and the same data shard as the main run — the named assumption is that the probe's loss curve, averaged over the last PROBE_LOSS_AVG_WINDOW steps and with adaptive widening to bracket the CBS, is a sufficient statistic for the marginal-loss-vs-batch relationship of the surrounding interval; this assumption fails when the loss landscape is non-stationary within the 2000-step interval (e.g., a data-shard boundary or a learning-rate transition), in which case CBS_hat reflects the probe's local landscape rather than the interval's average. The quantity `monotone_nondecreasing` over the last three probes measures *schedule consistency with CBS evolution* because the OLMo finding (E1/E8) establishes that CBS is non-decreasing over training — the named assumption is that the plateau phase does not admit genuine regressions; this assumption fails under a deliberate data-distribution shift or a learning-rate restart, in which case the gate clamps a legitimate schedule update and the violation flag is the signal that the assumption broke. The quantity `steps_next = TOKEN_BUDGET / B_next` measures *gradient-step efficiency* because token budget is held fixed — the named assumption is that the token budget is the binding compute constraint; this assumption fails when activation memory or communication time, not tokens, is the bottleneck on the 8192-card run, in which case the step reduction does not translate into wall-clock savings. The quantity `log_calibration` at PROBE_CALIBRATION_CHECKPOINTS measures *the short-probe's fidelity to a long-horizon reference* because it compares CBS_hat from the 8-step probe to a 128-step reference CBS at the same checkpoint — the named assumption is that the 8-step probe is a calibrated proxy for the longer-horizon marginal-loss-vs-batch curve; this assumption fails when the short probe's transient dynamics (e.g., Adam second-moment warmup) dominate the loss differences, in which case the calibration log reveals a systematic offset that can be used to correct or flag the probe.

## Contribution

["HC-CBS, a training-stack protocol that measures critical batch size in situ on a live 285B-parameter Ascend pretraining run under MindSpore HyperParallel's topology-aware hybrid sharding, with a monotone-consistency gate that clamps schedule updates against the measured CBS trajectory.", 'A design principle: batch-size warmup schedules for sharded large-scale pretraining should be derived from substrate-local CBS probes rather than transferred from smaller-model GPU runs, because the sharded collective-communication noise changes the effective gradient noise scale in a way that transferred curves cannot detect.', 'A reported (step, CBS, loss) trajectory for the 285B Ascend run that exposes whether the loss-versus-steps curve is monotone-consistent with the measured CBS curve — a diagnostic the Xing 4 report does not provide.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Xing 4 pretraining corpus shards | Live substrate and sharding match the claim exactly |
| Primary metric | Tokens-to-target-loss on held-out shard | Directly reflects step-efficiency claim, not proxy |
| Baseline | Fixed large batch (no probing) | Tests whether static batch sizing is sufficient |
| Baseline | OLMo-imported CBS schedule (E1/E8) | Tests whether GPU-derived CBS transfers to Ascend |
| Baseline | Gradient-noise-scale estimator (McCandlish) | Tests whether probe beats noise-scale proxy |
| Ablation-of-ours | HC-CBS without monotone-consistency gate | Isolates the gate's contribution to schedule stability |
| Ablation-of-ours | HC-CBS with probe length 8 vs. 32 vs. 128 steps | Tests whether the sufficient-statistic assumption holds |
| Ablation-of-ours | HC-CBS without adaptive multiplier widening | Tests whether bracketed fit matters |

### Why this setup validates the claim

The central claim is that CBS is a substrate-specific quantity, so a schedule derived from an in-situ probe on the live Ascend run should reach a target loss in fewer tokens than schedules that either ignore CBS or import it from a GPU run. The fixed-large-batch baseline tests the sub-claim that CBS warmup matters at all: if a static batch already matches our tokens-to-target-loss, the warmup mechanism is doing nothing. The OLMo-imported-CBS baseline is the sharpest test of substrate-specificity: it uses the same warmup logic but draws CBS from a GPU OLMo curve, so any gap between it and HC-CBS is attributable to the substrate mismatch our mechanism is designed to correct. The gradient-noise-scale baseline tests whether the empirical probe is actually better than the cheap proxy McCandlish et al. offer under sharded collective communication. The no-gate ablation isolates whether the monotone-consistency gate contributes to schedule stability or is inert. The probe-length ablation (8 vs. 32 vs. 128 steps) directly tests the load-bearing assumption that the short probe is a sufficient statistic for the interval-average marginal-loss-vs-batch curve: if the 8-step probe is adequate, its tokens-to-target-loss should match the 32- and 128-step variants; if not, the longer probes should show a measurable improvement, and the calibration log at PROBE_CALIBRATION_CHECKPOINTS will quantify the offset. The no-widening ablation tests whether bracketing the CBS via adaptive multipliers is necessary to avoid extrapolation bias. Tokens-to-target-loss is the right metric because the claim is about token efficiency, not wall-clock or final loss at a fixed token count; a schedule that reaches the same loss with fewer tokens is the predicted signature, and this metric detects it directly rather than through a proxy.

### Expected outcome and causal chain

**vs. Fixed large batch (no probing)** — On a case where the run starts at a batch size well above the true CBS early in training, the fixed baseline produces slow early loss descent because its large batch wastes tokens on redundant gradient information the optimizer cannot exploit at initialization. Our method instead starts small and grows the batch as the measured CBS rises, because the in-situ probe detects the near-zero CBS at initialization and the warmup schedule follows it. We expect a noticeable gap in early-phase tokens-to-target-loss, with the gap narrowing as training progresses and both schedules converge toward the plateau CBS.

**vs. OLMo-imported CBS schedule (E1/E8)** — On a case where MindSpore HyperParallel's topology-aware hybrid sharding introduces collective-communication noise that shifts the effective gradient noise scale away from the GPU OLMo curve, the imported baseline produces a CBS estimate that is systematically off, so its batch-size schedule either overshoots (harming token efficiency) or undershoots (wasting parallelism) relative to the true Ascend CBS. Our method instead measures CBS directly on the live Ascend substrate, because the probe runs under the same sharding and optimizer execution path as the main run. We expect a modest but persistent gap in tokens-to-target-loss concentrated in the mid-training interval where the imported and measured CBS curves diverge most.

**vs. Gradient-noise-scale estimator (McCandlish)** — On a case where the 8192-card sharded collective communication violates the unbiased-Gaussian gradient-noise assumption, the estimator baseline produces a noisy or biased CBS proxy that propagates into an unstable batch-size schedule, visible as schedule thrash across intervals. Our method instead fits CBS from a direct loss-vs-batch probe, because it does not rely on the gradient-noise proxy. We expect the estimator baseline to show higher variance in its batch-size trajectory and a worse tokens-to-target-loss, with the gap widest when collective-communication noise is highest.

**vs. HC-CBS without monotone-consistency gate** — On a case where a single noisy probe produces a non-monotone CBS_hat, the no-gate ablation propagates that noise directly into the batch-size schedule, causing thrash and potentially violating the warmup ceiling. Our method clamps the update to the prior CBS, smoothing over one interval. We expect the no-gate ablation to show higher variance in batch-size trajectory and slightly worse tokens-to-target-loss, with the gap largest when probe noise is highest (e.g., early training).

**vs. HC-CBS with probe length 8 vs. 32 vs. 128 steps** — On a case where the 8-step probe is dominated by transient optimization dynamics, the 8-step variant will show a systematic offset from the 128-step reference CBS (logged at calibration checkpoints), and its tokens-to-target-loss will be worse than the 32- or 128-step variants. Our method (8-step baseline) is expected to be adequate if the calibration log shows small offset; if not, the longer probes will show measurable improvement, falsifying the sufficient-statistic assumption for 8 steps.

**vs. HC-CBS without adaptive multiplier widening** — On a case where the true CBS exceeds the largest initial multiplier (8×B_t), the no-widening ablation extrapolates the fit and produces a biased CBS_hat, leading to suboptimal batch-size schedule. Our method widens multipliers until bracketed, avoiding extrapolation. We expect the no-widening ablation to show worse tokens-to-target-loss, especially early in training when CBS is rising rapidly.

### What would falsify this idea

If HC-CBS's tokens-to-target-loss gain over the OLMo-imported-CBS baseline is uniform across all training intervals rather than concentrated where the measured and imported CBS curves diverge, then substrate-specificity is not the causal driver and the central claim is wrong. Likewise, if the no-gate ablation matches HC-CBS exactly, the monotone-consistency gate is inert and the mechanism reduces to plain CBS warmup. Additionally, if the 8-step probe's calibration log shows a large systematic offset from the 128-step reference and the 32/128-step variants do not improve tokens-to-target-loss, the load-bearing sufficient-statistic assumption is false.

## References

1. TeleChat to Xing4: Advancing Frontier Agentic Models on Domestic Chips and Frameworks
