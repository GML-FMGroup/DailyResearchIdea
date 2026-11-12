# Self-Contained Correction for Quantized LLM Agents via Disagreement-Driven Ensembles

## Motivation

Existing quantization-aware distillation for LLM agents, such as SteerDistill, structurally depends on an external full-precision teacher to revise failed rollouts, so correction cannot close the loop when no such teacher exists. This reliance is not incidental: the teacher is the only source of corrective supervision, and without it the quantized agent has no mechanism to recover from execution failures. The shared residual gap across SWE-agent and SteerDistill is precisely this dependence on an external oracle for correction.

## Key Insight

Independent quantization noise across a small ensemble of quantized policy heads produces disagreement that is concentrated at decision points where the quantized policy is genuinely uncertain, so execution-feedback-triggered majority voting over these heads yields corrective actions without any full-precision reference.

## Method

**What it is.** We introduce SECQA (Self-Contained Ensemble Correction for Quantized Agents), a training and inference procedure that replaces SteerDistill's full-precision teacher with an internal ensemble of K=3 independently quantized policy heads over a shared quantized backbone. Inputs: a quantized agent policy π_q, a task distribution, and execution feedback. Outputs: a corrected quantized policy π_q* whose ensemble heads are collapsed into a single head for deployment.

**How it works.**

```pseudocode
# Phase 1: Build the ensemble
1. Quantize a pretrained agent policy with independent random seeds for
   weight-rounding and adapter initialization, producing K=3 heads {h_1..h_3}
   (low-rank adapters rank r=8) sharing one W4A16 backbone.
   # Load-bearing assumption (A): independent quantization seeds produce
   # per-head rounding noise that is uncorrelated across heads and uncorrelated
   # with the task's correct action, so ensemble disagreement at a decision
   # point reflects genuine policy uncertainty rather than shared systematic
   # quantization bias. This assumption fails when all heads share a systematic
   # quantization artifact (e.g., tokenizer-level bias, activation outliers,
   # salient-channel rounding error), in which case disagreement reflects shared
   # bias rather than genuine uncertainty.

# Phase 2: Execute and detect failure
2. For each task, run the agent with head h_1. Record the trajectory τ and
   execution feedback f (test pass/fail, tool error, syntax error).
3. If f indicates success, add τ to the success buffer D_ok.
4. If f indicates failure at turn t*, mark t* as the correction point.

# Phase 3: Disagreement-driven correction
5. At t*, query all K=3 heads with the same prefix. Each head proposes an
   action a_k. Compute pairwise action disagreement d = fraction of pairs
   with differing actions (d ∈ {0, 1/3, 2/3, 1}).
6. If d > 0 (any disagreement), run the top-2 voted actions through a
   cheap simulator/execution probe and select the action whose probe
   succeeds; if both succeed, select the majority action.
7. Append the corrected turn (prefix, a_corrected) to correction buffer D_corr.
   Resume execution from the corrected turn; repeat steps 2-7 until success
   or a turn budget of T_max=20 is exhausted.

# Phase 3b: Calibration gate (operationalizes assumption A)
   # Before trusting d as an uncertainty signal, measure its calibration on a
   # held-out set of 200 rollouts from the task distribution (disjoint from
   # D_ok and D_corr). For each turn in these rollouts, record d and whether
   # the majority-voted action is correct (verified by execution probe).
   # Compute AUROC of d vs. action correctness. If AUROC < 0.6, d is
   # uninformative (assumption A fails); fall back to probe-only correction:
   # at each failure point, probe all K=3 proposed actions and select the one
   # whose probe succeeds (majority vote only as tie-breaker). This gate is
   # run once after Phase 1 and re-checked every 50 corrected trajectories.

# Phase 4: Selective turn-level distillation
8. For each successful corrected trajectory, train all K=3 heads with loss
   L = Σ_{t∈D_corr} CE(h_k(prefix_t), a_corrected_t)
     + λ Σ_{t∈D_ok} CE(h_k(prefix_t), a_t)   with λ=0.5.
   Only corrected turns receive the ensemble-consensus target; uncorrected
   turns retain the original action target.

# Phase 5: Collapse for deployment
9. Average the K=3 adapter weights into a single head h* and deploy
   π_q* = (backbone, h*).
```

**Why this design.** We chose independent quantization seeds over a single quantized head because independent rounding noise is what makes ensemble disagreement informative about genuine policy uncertainty rather than about a shared systematic bias; the cost is that K=3 heads triple adapter memory during training. We chose execution-feedback-triggered correction over always-on ensembling because running all heads at every turn would multiply inference cost by K, accepting that correction only activates at detected failures. We chose top-2 probe verification over pure majority voting because majority voting among quantized heads can be confidently wrong when all heads share a quantization artifact, accepting the extra cost of two probe executions per correction point. We chose selective turn-level distillation over full-trajectory distillation because distilling uncorrected turns would overwrite the base policy's competent behavior with ensemble noise, accepting that the correction signal is sparse. Unlike SteerDistill, which requires a full-precision teacher to generate revised responses, SECQA's corrective signal is produced entirely by the quantized ensemble, so a domain expert would not describe it as a variant of teacher steering.

**Why this measures what we claim.** The ensemble disagreement rate d measures policy uncertainty at a decision point because independent quantization noise across heads is assumed to be uncorrelated with the task's correct action (assumption A); this assumption fails when all heads share a systematic quantization artifact (e.g., a tokenizer-level bias), in which case d reflects shared bias rather than genuine uncertainty. The Phase 3b calibration gate operationalizes this by measuring AUROC of d vs. action correctness on 200 held-out rollouts; if AUROC < 0.6, the method falls back to probe-only correction, so the correction signal rests on a measured calibration property rather than an assumed independence. The probe success of a candidate action measures corrective validity because the execution probe is assumed to be a faithful proxy for the downstream task outcome; this assumption fails when the probe is too shallow (e.g., syntax check only) to detect semantic errors, in which case probe success reflects superficial well-formedness rather than task correctness. The selective distillation loss on corrected turns measures internalization of the correction because corrected turns are assumed to be the only turns where the base quantized policy was wrong; this assumption fails when a failure at t* is caused by an earlier uncorrected turn, in which case the loss attributes credit to the wrong turn.

## Contribution

(1) A self-contained correction framework, SECQA, that replaces the external full-precision teacher in quantization-aware agent distillation with an internal ensemble of independently quantized policy heads and disagreement-driven voting. (2) A selective turn-level distillation procedure that internalizes ensemble-consensus corrections into a single collapsed quantized head while preserving base-policy behavior on uncorrected turns. (3) An empirical demonstration that independent quantization noise provides a usable correction signal for LLM agents without any full-precision reference model.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | SWE-bench-Lite (300 instances) + HumanEvalFix (164 tasks) | Agentic tasks with execution feedback; SWE-bench-Lite is the scoped pilot subset |
| Primary metric | pass@1 | Directly measures task success |
| Baseline | BF16 full-precision | Upper bound on achievable quality |
| Baseline | SteerDistill (W4A16) | Closest quantized-teacher-steering competitor |
| Baseline | Quantized baseline (W4A16, no correction) | Isolates gain from correction itself |
| Ablation-of-ours | SECQA w/o probe verification | Tests whether probe or voting drives gains |
| Ablation-of-ours | SECQA w/o calibration gate (always trust d) | Tests whether the calibration gate is necessary |
| Ablation-of-ours | SECQA with turn-attribution shifted to preceding turn | Tests whether credit assignment is correct |
| Calibration metric | AUROC of disagreement d vs. action correctness on 200 held-out rollouts | Operationalizes assumption A |
| Probe-faithfulness metric | Agreement rate between probe success and full downstream outcome on 100 correction points | Operationalizes probe-to-validity equivalence |

### Why this setup validates the claim

The central claim is that a self-contained quantized ensemble, triggered by execution failure and verified by a cheap probe, can recover the quality lost to W4A16 quantization without any full-precision teacher. SWE-bench-Lite and HumanEvalFix are the right testbed because they emit hard execution feedback (test pass/fail, syntax errors) at every turn, which is exactly the signal SECQA's correction loop consumes; without such feedback the method has no trigger and the claim is untestable. The BF16 baseline fixes the ceiling: if SECQA's pass@1 approaches BF16 while the no-correction quantized baseline lags, the gap closed is attributable to correction, not to the task being easy. SteerDistill is the decisive comparison for the "self-contained" sub-claim — it has a full-precision teacher, so if SECQA matches it, the teacher is shown to be unnecessary; if it falls short, the teacher still carries irreducible value. The probe-free ablation isolates whether gains come from disagreement-driven correction or merely from majority voting, testing the design's core causal bet. The calibration gate ablation and AUROC metric directly test assumption A: if d is uninformative (AUROC < 0.6), the gate falls back to probe-only correction, and the ablation shows whether trusting d without calibration hurts. The turn-attribution ablation tests whether the selective distillation loss assigns credit to the correct turn. pass@1 is the right metric because it is the only one that reflects end-to-end task completion rather than intermediate turn quality, and it is the metric on which quantization damage actually manifests.

### Expected outcome and causal chain

**vs. BF16 full-precision** — On a case where a multi-step edit requires an exact API signature, the BF16 model recalls the signature and passes; the W4A16 backbone's rounding noise corrupts the token distribution and it emits a near-miss call that fails the test. SECQA's heads disagree at that turn, the probe rejects the wrong candidate and accepts the correct one, and the corrected turn is distilled back. We expect SECQA to close most of the quantization gap on multi-turn tasks (near-parity on single-turn HumanEvalFix, a visible but smaller residual gap on long SWE-bench-Lite trajectories where credit assignment across turns is harder).

**vs. SteerDistill (W4A16)** — On a case where the quantized policy fails at a turn the full-precision teacher would easily fix, SteerDistill succeeds because it can query the BF16 teacher for a revised action; SECQA has no teacher and must rely on its own ensemble. Where the ensemble's independent rounding noise happens to span the correct action, SECQA recovers it; where all three heads share a systematic quantization artifact, SECQA cannot. We expect rough parity on the majority of failures but a measurable deficit concentrated on failures caused by shared systematic bias, which is the honest cost of removing the teacher.

**vs. Quantized baseline (W4A16, no correction)** — On a case where the quantized agent takes a wrong action at turn t* and never recovers, the no-correction baseline simply proceeds and fails the task. SECQA detects the failure via execution feedback, corrects the turn, and resumes. We expect the largest and most consistent gap here, concentrated on tasks that fail mid-trajectory rather than at the first turn, since those are precisely the cases the correction loop is designed to rescue.

**vs. SECQA w/o probe verification** — On a case where all three heads share a quantization artifact and confidently agree on a wrong action, majority voting alone selects the wrong action, whereas probe verification catches it because the action fails the execution probe. We expect the ablation to underperform specifically on the subset of correction points where head disagreement is zero or where the majority is wrong — a small but diagnostic subset that isolates the probe's contribution.

**vs. SECQA w/o calibration gate** — On a case where d is uninformative (AUROC < 0.6 on held-out rollouts), the uncalibrated variant trusts d and may select a wrong action when heads disagree due to shared bias; the calibrated variant falls back to probe-only correction and avoids this. We expect the uncalibrated variant to underperform on the subset of tasks where assumption A fails, and the calibration gate to recover most of that gap.

**vs. SECQA with turn-attribution shifted to preceding turn** — On a case where a failure at t* is caused by an earlier uncorrected turn, the shifted-attribution variant distills the wrong turn and may degrade performance; the original variant attributes to t* and may also mis-attribute. We expect the shifted variant to underperform on tasks with cascading errors, isolating the credit-assignment assumption.

### What would falsify this idea

If SECQA's pass@1 gain over the no-correction quantized baseline is uniform across all task subsets rather than concentrated on mid-trajectory failures with detectable execution feedback, the correction mechanism is not what is producing the gain and the central claim is wrong. Equally, if SECQA matches SteerDistill even on failures traceable to shared systematic quantization bias, then the ensemble disagreement is not measuring genuine uncertainty and the design's core assumption is falsified. Additionally, if the AUROC of d vs. action correctness is below 0.6 on held-out rollouts, assumption A fails and the disagreement-driven correction is not measuring genuine uncertainty.

## References

1. SteerDistill: Quantizing LLM Agents via Closed-Loop Teacher Steering
2. SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering
