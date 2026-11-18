# Factorial Interaction Ablation of Training-Efficiency Mechanisms in a 285B Ascend Pretraining Run

## Motivation

Training-efficiency mechanisms—batch-size warmup (CBS), memory-efficient adapter training (LoRAM), and checkpoint merging—are each validated in isolation on separate model/hardware regimes, and the frontier report (TeleChat to Xing4) describes a 285B Ascend run only at the level of design principles with no ablations isolating joint effects. Because each branch's evidence comes from a different scale and stack (e.g., CBS measured on OLMo 1B/7B GPU runs, LoRAM measured on 70B inference-memory budgets, merging measured on dense/MoE checkpoints up to 100B), the field structurally cannot say whether these mechanisms compose additively, orthogonally, or antagonistically when co-deployed on one large-scale run—an interaction effect that no single-mechanism study can observe by construction.

## Key Insight

When two efficiency mechanisms act on the same resource axis—gradient-noise-limited step count, activation/optimizer memory, or checkpoint parameter geometry—their joint effect on token efficiency is not the product of their marginal effects, so the sign and magnitude of the interaction term is a measurable property of the shared axis that factorial ablation recovers and marginal studies cannot.

## Method

**What it is.** We introduce FACTOR-285, a factorial ablation protocol that decomposes the token-efficiency effect of three training-efficiency mechanisms on a single large-scale Ascend pretraining run into main effects and pairwise/triple interaction terms. Inputs: a fixed 285B-parameter architecture and MindSpore HyperParallel stack (from Xing 4), a fixed token budget and data order, and three binary factors. Outputs: a per-factor main effect, three pairwise interaction coefficients, one triple interaction coefficient, and a downstream-loss table, all measured against a common reference run.

**Load-bearing assumption (stated explicitly).** The 2^3 contrasts estimate genuine resource-axis coupling only if each factor toggle leaves the optimization regime invariant: effective learning rate (LR) and effective batch size must be matched across cells, or any shift must be measured and regressed out. This assumption is load-bearing because the factors are structurally entangled with schedule and parameterization: batch-size warmup (A) changes effective batch and thus effective LR/noise scale over training; LoRAM-style adapter training (B) changes the effective parameterization and gradient geometry; checkpoint merging (C) is only well-posed on constant-LR segments. We therefore add a per-cell calibration step (D) that measures effective LR and effective batch trajectories and equalizes them before contrasts are estimated, so that inter_XY reflects resource-axis coupling rather than hyperparameter mismatch.

**How it works.**

```pseudocode
FACTOR-285(config):
  # Factors (binary), each toggled on the SAME 285B architecture, data order, token budget T
  A = batch_size_warmup    # off: constant batch B0=512 sequences; on: B(t) ramps from B0 to Bmax=4096 as CBS(t) grows
  B = adapter_training     # off: full-parameter update; on: LoRAM-style structured-pruned low-rank update (rank r=64, prune ratio 0.5)
  C = checkpoint_merging   # off: single trajectory; on: merge checkpoints of constant-LR segments (every 10% of T, equal-weight averaging)

  # (D) Per-cell calibration to equalize effective LR and effective batch
  for each cell (a,b,c) in {0,1}^3:            # 8 cells
     # Calibration run: short run of 1% of T to measure effective LR and effective batch
     calib = pretrain(arch=285B, tokens=0.01*T, data_order=fixed_seed,
                      warmup=a, adapter=b, merging=c,
                      lr=constant, optimizer=distributed_adamw)
     measure(calib): effective_lr(t) = lr * (1 - momentum) / (1 - momentum^t) * (batch_size(t)/B_ref)
                      effective_batch(t) = batch_size(t) * grad_accum_steps
     # Adjust per-cell constant LR and gradient accumulation to match reference cell's effective trajectories
     lr_cell = lr_ref * (B_ref / mean_batch_cell)
     grad_accum_cell = round(B_ref / batch_size_cell)
     # Record calibration diagnostics for regression
     record_calib(cell): mean_effective_lr, mean_effective_batch, lr_shift, batch_shift

  # Main runs with calibrated schedules
  for each cell (a,b,c) in {0,1}^3:            # 8 cells
     run = pretrain(arch=285B, tokens=T, data_order=fixed_seed,
                    warmup=a, adapter=b, merging=c,
                    lr=lr_cell, optimizer=distributed_adamw, grad_accum=grad_accum_cell)
     record(run): token_efficiency = tokens_to_target_loss / T
                  downstream_loss = mean over held-out suite
                  effective_lr_traj, effective_batch_traj

  # Estimands (standard 2^3 factorial contrasts, per run seed)
  main_X    = mean(cells with X=1) - mean(cells with X=0)
  inter_XY  = mean(XY=11) - mean(XY=10) - mean(XY=01) + mean(XY=00)
  inter_ABC = (+++) - (+--) - (-+-) - (--+) + (+--) ... (full 8-term contrast)

  # Regress out residual effective-LR/batch shifts if calibration did not fully equalize
  if max(|lr_shift|, |batch_shift|) > 0.05:
     inter_XY_adj = inter_XY - beta_lr * delta_lr_XY - beta_batch * delta_batch_XY
     # beta estimated from calibration runs across cells

  classify(inter): additive if |inter| <= tau; orthogonal if inter~0 but main effects nonzero;
                   antagonistic if inter < -tau; synergistic if inter > +tau
  # tau = 2 * seed std of the reference cell, fixed before runs (seed std from 3 seeds of reference cell)
```

**Why this design.** We chose a full 2^3 factorial over one-factor-at-a-time sweeps because the compositionality question is precisely about interaction terms, which OFAT cannot estimate, accepting the cost of 8 full 285B runs instead of 4. We chose to hold architecture, data order, token budget, and learning-rate schedule constant across all cells (rather than letting each mechanism use its native tuned schedule) because otherwise the interaction term would absorb hyperparameter re-tuning confounds, accepting the cost that each cell may be slightly suboptimal for its own mechanism. We chose a constant-LR schedule as the common backbone because checkpoint merging is defined on constant-LR segments (Model Merging in Pre-training), so this keeps the merging factor well-posed while still allowing CBS warmup and LoRAM to be toggled independently. We chose a pre-registered threshold tau from seed variance rather than post-hoc significance testing because with 8 expensive runs we cannot afford many seeds, accepting reduced power in exchange for a falsifiable additive/antagonistic verdict. We added the (D) calibration sub-block because the factors are structurally entangled with effective LR and effective batch; without equalizing these trajectories, inter_XY would conflate resource-axis coupling with schedule mismatch. A domain expert would not describe this as "CBS + LoRAM + merging applied to Ascend," because the artifact is the interaction-coefficient estimator and its pre-registered classification rule, not any one mechanism; the mechanisms are held fixed as inherited components and the contribution is the decomposition.

**Why it measures what we claim.** The token-efficiency contrast `tokens_to_target_loss / T` measures a mechanism's contribution to data efficiency because we assume the target-loss threshold is reachable in every cell and that the same held-out loss surface is used across cells; this assumption fails when a cell (e.g., adapter-only) cannot reach the threshold within T, in which case the contrast reflects a reachability gap rather than an efficiency difference. The interaction coefficient `inter_XY` measures non-additivity of mechanisms X and Y because we assume each cell differs only in the factor toggles and shares data order and schedule; this assumption fails when a factor silently changes effective learning rate or effective batch, in which case `inter_XY` reflects a hyperparameter mismatch rather than a genuine resource-axis coupling. The (D) calibration step measures and equalizes effective LR and effective batch trajectories because we assume that after calibration the residual shifts are below 5% and can be regressed out; this assumption fails when a factor (e.g., LoRAM's pruned parameterization) induces an effective-LR shift that cannot be compensated by constant-LR scaling, in which case the adjusted interaction `inter_XY_adj` reflects residual mismatch rather than resource-axis coupling. The classification threshold `tau` measures whether an interaction is distinguishable from run-to-run noise because we assume seed variance from the reference cell is representative of all cells; this assumption fails when variance is heteroscedastic across cells (e.g., merging cells are lower-variance), in which case the verdict is biased toward "additive." The downstream-loss delta measures whether efficiency gains preserve capability because we assume the held-out suite is sensitive to the same capabilities the target-loss threshold proxies; this assumption fails when the suite saturates, in which case downstream loss reflects suite ceiling rather than capability preservation.

## Contribution

(1) FACTOR-285, a pre-registered 2^3 factorial ablation protocol with a seed-variance-derived classification rule that separates main effects from pairwise and triple interaction terms for training-efficiency mechanisms on a single large-scale run. (2) An interaction taxonomy (additive / orthogonal / antagonistic / synergistic) tied to shared resource axes—gradient-noise-limited steps, activation/optimizer memory, checkpoint geometry—that predicts which mechanism pairs should interact before running the ablation. (3) A reusable design principle: mechanisms sharing a resource axis must be co-ablated, not validated in isolation, to report token-efficiency claims at scale.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Fixed-seed 285B pretraining corpus, held-out suite | Same data order isolates factor toggles from data confounds |
| Primary metric | 2^3 factorial interaction coefficients on tokens-to-target-loss | Directly estimates non-additivity, the central claim |
| Baseline | OFAT single-mechanism runs (CBS warmup only, LoRAM only, merging only) | Tests whether interaction terms are truly nonzero vs additive |
| Baseline | Naive stacked all-on run (A+B+C jointly, no decomposition) | Tests whether joint gains equal sum of parts |
| Baseline | Reference cell (all-off, constant batch, full-param, single trajectory) | Anchors contrasts and defines pre-registered noise threshold tau |
| Ablation-of-ours | Post-hoc tau (significance test after runs) vs pre-registered tau | Tests whether verdicts depend on threshold discipline |
| Ablation-of-ours | No-calibration (D off) vs calibration (D on) | Tests whether effective-LR/batch equalization changes interaction signs |

### Why this setup validates the claim

The central claim is that the three mechanisms compose non-additively, so the only test that can falsify it is one that estimates interaction terms directly. The OFAT baselines test the sub-claim that single-mechanism effects are insufficient to predict joint behavior: if the all-on stacked run's gain equals the sum of OFAT main effects, the interaction coefficients should be near zero, and our classification would correctly return "additive." The naive stacked baseline tests the complementary sub-claim that joint deployment without decomposition hides antagonism: if stacking underperforms the sum-of-parts prediction, the triple interaction term should be negative. The ablation isolates the pre-registration rule: if verdicts flip when tau is chosen post-hoc, the classification is an artifact of threshold discipline rather than a property of the mechanisms. The calibration ablation isolates the load-bearing assumption: if interaction signs change when effective-LR/batch equalization is turned off, the contrasts were measuring schedule mismatch rather than resource-axis coupling. The interaction-coefficient metric is the right one because it is exactly the quantity the claim asserts is nonzero; token-efficiency ratios or downstream loss alone cannot distinguish additive composition from interaction. The reference cell anchors every contrast and defines tau from seed variance (3 seeds), making the additive/antagonistic verdict falsifiable rather than descriptive.

### Expected outcome and causal chain

**vs. OFAT single-mechanism runs** — On a case where CBS warmup alone reaches the target loss but LoRAM alone cannot within T, the OFAT baseline produces a main effect that looks like a clean efficiency gain because it never observes the two mechanisms co-resident. Our method instead runs the A=1,B=1 cell alongside A=1,B=0 and A=0,B=1, because the factorial contrast subtracts the marginal effects, so we expect a visible negative inter_AB coefficient: the adapter cell's reachability gap suppresses the warmup gain, showing antagonism concentrated in the cells where adapter training is on. With calibration (D) on, we expect the negative inter_AB to persist but shrink in magnitude relative to the no-calibration ablation, confirming that part of the raw interaction was effective-LR mismatch.

**vs. Naive stacked all-on run (A+B+C)** — On a case where checkpoint merging lowers loss variance within segments, the stacked baseline produces a joint gain that looks like the sum of parts because it never separates merging's variance reduction from warmup's batch-size effect. Our method instead estimates inter_AC and inter_BC from the eight cells, because the factorial isolates each pair while holding the third factor fixed, so we expect the triple interaction to be small but the pairwise merging interactions to be positive, revealing that merging is synergistic with warmup but orthogonal to adapter training. The calibration diagnostic is expected to show that merging cells have the smallest effective-LR shift, so their interaction estimates are least affected by the (D) adjustment.

**vs. Reference cell (all-off)** — On a case where the all-off cell has the highest seed variance, the reference baseline produces a tau that is too wide for the lower-variance merging cells, because seed variance is assumed homoscedastic across cells. Our method instead classifies with that same tau but reports per-cell variance alongside the coefficients, because the pre-registered rule is fixed while the diagnostic is transparent, so we expect the verdict to be conservative (biasing toward additive) and the per-cell variance table to show merging cells with visibly tighter spread, flagging where the tau assumption is strained.

**vs. No-calibration ablation (D off)** — On a case where batch-size warmup (A) changes effective batch and thus effective LR, the no-calibration run produces interaction coefficients that conflate resource-axis coupling with schedule mismatch, because the raw contrasts absorb the effective-LR shift. Our method instead applies per-cell calibration to equalize effective LR and effective batch before estimating contrasts, so we expect the calibrated inter_AB and inter_AC to differ in sign or magnitude from the uncalibrated ones, with the calibrated estimates showing smaller variance across seeds and the calibration diagnostic table reporting residual shifts below 5% for all cells except the adapter-on cells, where LoRAM's pruned parameterization may leave a residual shift that the regression adjustment (beta_lr) partially removes.

### What would falsify this idea

If all seven interaction coefficients (three pairwise, one triple, and the implied contrasts) fall within ±tau of zero while the main effects are nonzero, the compositionality claim is wrong and the mechanisms are additive. If the interaction pattern is uniform across cells rather than concentrated where the predicted mechanism coupling occurs (e.g., no negative inter_AB despite the adapter reachability gap), or if the calibrated and uncalibrated interaction signs are identical and within tau of each other, the decomposition is measuring noise or schedule mismatch rather than resource-axis coupling.

## References

1. TeleChat to Xing4: Advancing Frontier Agentic Models on Domestic Chips and Frameworks
