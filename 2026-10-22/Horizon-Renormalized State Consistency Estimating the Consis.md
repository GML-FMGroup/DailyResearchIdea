# Horizon-Renormalized State Consistency: Estimating the Consistency Scaling Exponent of Streaming Video Models

## Motivation

Across streaming-video evaluation branches, state consistency is reported as a single aggregate over a fixed, short, asset-initialized rollout window (StreamEval scores ten task-specific metrics over 15–300s windows; TC-Bench scores transition completeness over a fixed initial-to-final interval; WorldScore aggregates controllability/quality/dynamics over a fixed next-scene sequence). Because the window is the evaluation unit, the metric's behavior under horizon growth is structurally undefined: a model that scores 80 at 15s and one that scores 80 at 60s are indistinguishable, even though one may be drifting and the other stationary, and no branch can certify that a model's state consistency is horizon-invariant or that drift is bounded. The root cause is that consistency is aggregated over a window rather than modeled as a function of horizon, so the fixed horizon silently absorbs the very drift the metric claims to measure.

## Key Insight

If per-step state consistency is treated as a renormalization-group flow over rollout horizon, then its behavior under horizon doubling collapses onto a single scaling exponent (or fixed point), which is a horizon-invariant property of the model rather than a window-dependent aggregate — so two models can be compared by their drift exponent even when they were evaluated at different horizons.

## Method

**What it is.** HRSC (Horizon-Renormalized State Consistency) takes a streaming video model M, a scripted test case with initial assets and an incremental request stream (StreamEval format), and a geometric ladder of rollout horizons H = {h, 2h, 4h, 8h, ...}. It outputs a per-case consistency scaling exponent β (and a fixed-point consistency s*, when the flow converges), replacing the single fixed-window aggregate with a horizon-invariant scalar.

**Load-bearing assumption (stated explicitly).** HRSC assumes that the per-step state-consistency score φ(state_t, request_t) is a horizon-independent, seed-dominated measurement of the model's state, so that the only thing changing across the geometric horizon ladder is H itself and the resulting s(H) curve reflects pure horizon renormalization rather than a change in the model's operating regime or the judge's detection behavior. This assumption is load-bearing: if it fails, s(H) conflates model drift with exposure bias (self-conditioned rollouts feed drifted frames back into conditioning) and with VLM-judge drift (detection rate varies with frame count and temporal context). We therefore make the assumption testable inside the method via two concrete controls: (i) a teacher-forced rollout mode that holds the conditioning distribution fixed, and (ii) a judge-noise calibration that subtracts the judge's horizon-dependent detection bias from φ.

**How it works.**

```pseudocode
INPUT: model M, case C (initial assets, request stream r_1..r_T), base horizon h0=15s, ratio r=2, levels L=4
OUTPUT: exponent beta, fixed point s_star, per-level scores s_0..s_L, teacher-forced beta_tf, judge-noise floor eps(H)

0. Judge-noise calibration (per case, before the ladder):
     For each level l in 0..L:
       H_l = h0 * r^l
       # score the SAME fixed reference states at each horizon length to measure judge drift
       eps_l = mean_{t=1..H_l} phi(ref_state_t, request_t) - phi(ref_state_t, request_t) evaluated at t=1
       # eps_l is the judge's horizon-dependent detection bias; subtract it from all s_l below
     eps(H_l) = eps_l   # stored as the judge-noise floor

1. For each level l in 0..L:
     H_l = h0 * r^l
     rollout = M.run(C, horizon=H_l, seed=SEED, mode=FREE)   # free-running, self-conditioned
     s_l^raw = mean_{t=1..H_l} phi(state_t, request_t)        # per-step state-consistency score phi in [0,100]
        # phi = fraction of active state requirements satisfied at step t
        #       (object identity, attribute, spatial relation, request execution)
     s_l = s_l^raw - eps(H_l)                                 # judge-bias-corrected score

1b. Teacher-forced control (per level, same seed):
     rollout_tf = M.run(C, horizon=H_l, seed=SEED, mode=TEACHER_FORCED)
        # at each step, condition on ground-truth (or fixed reference) prefix states
        # rather than the model's own drifted history
     s_l^tf = mean_{t=1..H_l} phi(state_t^tf, request_t) - eps(H_l)

2. Renormalization step (horizon doubling):
     For each l in 0..L-1:
       s_l^{(1)} = mean of s_l over the first H_l steps     # already have
       s_{l+1}   = mean of s_{l+1} over H_{l+1} steps        # already have
     # consistency flow: s_l -> s_{l+1} under horizon doubling

3. Fit scaling law (two-parameter, committed form):
     s(H) = s_star + A * H^{-beta}
     Fit (s_star, A, beta) by weighted least squares over the L+1 points (H_l, s_l),
     weights w_l = H_l (longer horizons get more weight, since they carry more
     information about drift).
     Also fit the same form to the teacher-forced points (H_l, s_l^tf) to obtain beta_tf.

4. Report:
     beta   = consistency scaling exponent (drift rate under horizon growth, free-running)
     beta_tf = teacher-forced consistency scaling exponent (exposure-bias-free drift rate)
     s_star = fixed-point consistency (limit as H -> infinity)
     residual = fit residual (diagnostic that the scaling law holds)
     drift_signal = beta - beta_tf   # divergence between free-running and teacher-forced
        # if drift_signal is large, the free-running beta reflects exposure bias, not pure horizon renormalization
```

**Why this design.** We chose a geometric horizon ladder with a shared seed over a single fixed window because a fixed window cannot separate the model's stationary consistency from its drift rate, accepting the cost that each case requires L+1 rollouts (4x compute for L=4) and that the shared-seed requirement constrains the model's stochasticity. We chose a two-parameter power-law fit s(H) = s* + A·H^{-β} over a nonparametric per-horizon table because a table has no fixed point and cannot be compared across models evaluated at different horizons, accepting the cost that the power-law form is an assumption that can fail (diagnosed by the fit residual). We chose horizon-weighted least squares (w_l = H_l) over uniform weighting because short-horizon points are dominated by initialization transients and carry little information about drift, accepting the cost that a single bad long-horizon rollout can dominate the fit. We added the teacher-forced control and the judge-noise calibration because the load-bearing assumption (that φ is horizon-independent and seed-dominated) is structurally at risk from exposure bias and VLM-judge drift; these controls make the assumption testable inside the method rather than assumed away, accepting the cost of a second rollout per level (2x compute) and a calibration pass. This is not a variant of VBench-style dimension aggregation or StreamEval-style fixed-window scoring: those methods report a single number per dimension per window, whereas HRSC reports the exponent of the consistency-vs-horizon flow, a quantity that does not exist in either prior method's output space.

**Why it measures what we claim.** The per-step score φ(state_t, request_t) measures instantaneous state consistency because it counts the fraction of active state requirements satisfied at step t, under the assumption that the scripted request stream's requirements are a complete operationalization of state (this assumption fails when a requirement is unobservable from pixels, in which case φ reflects the VLM judge's detection rate rather than true state; the judge-noise calibration eps(H) subtracts the horizon-dependent part of this detection bias, and the per-case observability audit in the experiment flags cases where unobservable requirements dominate). The horizon-doubling flow s_l → s_{l+1} measures renormalization-group flow over rollout horizon because the same seed and same request stream are held fixed while only H changes, under the assumption that the model's stochasticity is seed-dominated and horizon-independent (this assumption fails when the model's sampling temperature or KV-cache truncation changes with H, in which case the flow reflects a change in the model's operating regime rather than pure horizon renormalization; the teacher-forced control beta_tf and the drift_signal = beta - beta_tf isolate the exposure-bias contribution, and models whose sampling temperature or KV-cache policy cannot be held fixed across horizons are flagged as out-of-scope). The fitted exponent β measures the drift rate of state consistency under horizon growth because the power-law form s(H) = s* + A·H^{-β} is the unique two-parameter family in which β is invariant to the base horizon h0, under the assumption that consistency decays as a power law (this assumption fails when drift is exponential, in which case β is a local slope and the fit residual flags the misspecification; we additionally compare power-law vs. exponential vs. logarithmic decay by AIC/BIC on held-out horizons to diagnose the form). The fixed point s* measures horizon-invariant consistency because it is the H→∞ limit of the fitted law, under the assumption that the power law extrapolates beyond the largest observed horizon (this assumption fails when the model has a horizon-dependent regime change, in which case s* is an extrapolation artifact and the residual again flags it).

## Contribution

(1) A horizon-renormalized state-consistency metric (HRSC) that replaces fixed-window aggregation with a per-case consistency scaling exponent β and fixed point s*, making state consistency a measurable function of rollout horizon rather than a window-dependent scalar. (2) A design principle — shared-seed geometric horizon laddering with horizon-weighted power-law fitting — that makes cross-model comparison possible even when models are evaluated at different horizons, together with a fit-residual diagnostic that flags when the power-law assumption fails. (3) An analysis protocol that re-scores existing StreamEval-style test cases along the horizon axis, yielding per-model drift exponents that are not recoverable from any single-window aggregate.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | StreamEval 300-case scripted suite | Provides incremental request streams and initial assets |
| Primary metric | Consistency scaling exponent β (free-running) and β_tf (teacher-forced) | Horizon-invariant drift rate, comparable across models; drift_signal = β − β_tf isolates exposure bias |
| Baseline | StreamEval fixed-window score | Single-window aggregate, no drift separation |
| Baseline | VBench dimension aggregation | Per-dimension means, no horizon flow |
| Baseline | TC-Bench temporal compositionality | Ground-truth comparison, no fixed point |
| Baseline | Post-hoc power-law fit to StreamEval/VBench per-step scores | Tests whether HRSC's shared-seed horizon ladder yields a β that a post-hoc fit cannot recover |
| Ablation-of-ours | HRSC with uniform weights | Tests whether horizon weighting drives the fit |
| Ablation-of-ours | HRSC without judge-noise calibration (eps=0) | Tests whether judge drift drives the fit |
| Ablation-of-ours | Reduced ladder L=3, ratio=2 | Tests β stability under reduced compute |
| Predictive-validity target | Held-out long-horizon rollouts at 16h (beyond largest ladder level) | Tests whether β predicts long-horizon consistency better than fixed-window score |
| Calibration set | 512 examples with ground-truth state labels | Quantifies VLM-judge detection-rate confound for φ |

### Why this setup validates the claim

The central claim is that a fixed-window aggregate conflates a model's stationary consistency with its drift rate, whereas the exponent β of the consistency-vs-horizon flow separates them. StreamEval's scripted request streams are the right substrate because they give per-step, verifiable state requirements, so φ is well-defined at every horizon. The StreamEval fixed-window baseline tests the sub-claim that a single number cannot distinguish a high-baseline-but-fast-drifting model from a low-baseline-but-stable one: if β carries information the fixed window lacks, models matched on fixed-window score must still separate on β, and that separation must predict long-horizon behavior. The VBench and TC-Bench baselines test the sub-claim that aggregation-style and ground-truth-comparison metrics have no fixed point and cannot be compared across horizons, so their rankings should be horizon-sensitive while β is not. The post-hoc power-law baseline tests whether HRSC's shared-seed horizon ladder uniquely contributes information beyond a re-fit of existing per-step curves: if the post-hoc fit recovers the same β, the ladder design adds nothing; if it cannot (e.g., due to regime changes), the ladder's controlled renormalization is doing real work. The uniform-weight ablation tests whether the horizon-weighted fit is doing real work: if weighting is irrelevant, β should be unchanged and the ablation should match ours, which would undercut the design rationale. The no-calibration ablation tests whether the judge-noise correction is doing real work: if eps=0 leaves β unchanged, the judge-drift confound is negligible for this suite. The reduced-ladder ablation tests feasibility: if β is stable at L=3, the 4x compute cost can be reduced. The predictive-validity target tests external consequence: if β predicts held-out 16h consistency better than the fixed-window score (rank correlation and calibration), the exponent has consequence beyond the fitted ladder. The 512-example calibration set tests the φ-to-state-consistency equivalence: comparing φ against ground-truth state labels quantifies the VLM-judge detection-rate confound, and a per-case observability audit flags cases where unobservable requirements dominate. The fit residual is the built-in falsifier for the power-law assumption, and the AIC/BIC comparison of power-law vs. exponential vs. logarithmic decay diagnoses the functional form, so a low-residual, cross-model-separating β is exactly the pattern the claim predicts.

### Expected outcome and causal chain

**vs. StreamEval fixed-window score** — On a case where a model starts with strong object identity but slowly loses spatial relations after the window closes, the fixed-window baseline reports a high score because it only averages the first H steps, so drift past the window is invisible. Our method instead runs the geometric ladder and fits s(H) = s* + A·H^{-β}, so the late-horizon decline enters the fit as a nonzero β. We expect a noticeable gap on cases with late-onset drift while near-parity on cases whose drift is front-loaded, and β should rank-order models that the fixed window ties. On the held-out 16h rollouts, we expect β to predict long-horizon consistency with higher rank correlation than the fixed-window score.

**vs. VBench dimension aggregation** — On a case where a model is stable on appearance but degrades on request execution as the stream lengthens, VBench aggregation averages per-dimension scores over a fixed clip, so the degradation is diluted across dimensions and never expressed as a rate. Our method instead isolates the per-step state-consistency flow and recovers β as the decay exponent, giving a horizon-invariant number. We expect VBench rankings to shift when the evaluation horizon is changed while β rankings stay stable, the signature that aggregation lacks a fixed point.

**vs. TC-Bench temporal compositionality** — On a case where a model matches ground-truth frames early but accumulates state error over a long rollout, TC-Bench scores similarity to a reference video of fixed length, so it cannot express what happens beyond that length. Our method instead extrapolates s* as the H→∞ limit and flags regime changes via the fit residual. We expect TC-Bench and β to agree on short-horizon cases but diverge on long-horizon cases, with β exposing drift that TC-Bench's fixed reference cannot see.

**vs. Post-hoc power-law fit to StreamEval/VBench per-step scores** — On a case where the model exhibits a regime change (e.g., KV-cache truncation at a fixed context length), the post-hoc fit to existing per-step scores sees only the aggregate curve and cannot separate the regime change from the power-law decay, so its β is biased. Our method's shared-seed horizon ladder with teacher-forced control and judge-noise calibration yields a β that the post-hoc fit cannot recover, and the drift_signal = β − β_tf exposes the exposure-bias contribution. We expect the post-hoc fit to match HRSC on stationary cases but diverge on regime-change cases, isolating what the ladder design uniquely contributes.

### What would falsify this idea

If β fails to separate models that the fixed-window score ties, or if the uniform-weight ablation reproduces β within noise, or if the no-calibration ablation shows eps=0 leaves β unchanged (judge drift negligible), or if the post-hoc power-law fit recovers the same β as HRSC, then the exponent carries no information beyond the aggregate and the central claim is wrong. A uniform gain of β-prediction across all subsets, rather than concentration on late-onset-drift cases, would likewise indicate the mechanism is not what we claim; and if β does not predict held-out 16h consistency better than the fixed-window score, the exponent lacks external consequence.

## References

1. StreamEval: Evaluating Interactive Streaming Video Generation Models
2. TC-Bench: Benchmarking Temporal Compositionality in Text-to-Video and Image-to-Video Generation
3. WorldScore: A Unified Evaluation Benchmark for World Generation
4. Davidsonian Scene Graph: Improving Reliability in Fine-grained Evaluation for Text-to-Image Generation
5. FETV: A Benchmark for Fine-Grained Evaluation of Open-Domain Text-to-Video Generation
6. Towards World Simulator: Crafting Physical Commonsense-Based Benchmark for Video Generation
7. VBench++: Comprehensive and Versatile Benchmark Suite for Video Generative Models
8. All You May Need for VQA are Image Captions
