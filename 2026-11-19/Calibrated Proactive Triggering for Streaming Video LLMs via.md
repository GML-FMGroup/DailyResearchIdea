# Calibrated Proactive Triggering for Streaming Video LLMs via Proper-Scoring-Rule Policy Optimization

## Motivation

Proactive-activation streaming assistants — StreamChat (E1), StreamBridge (E2), LiveStar (E3), Dispider (E5), and ALIVE (E6) — all learn *when to speak* as an uncalibrated classification/RL policy: ALIVE's BACA optimizes trajectory-level and local advantages but never ties the firing probability to the probability that the downstream answer is correct under the same partial observation. Structurally, the trigger decision and the answer-correctness event live in different objective surfaces (policy advantage vs. answer likelihood), so a high-confidence trigger can fire on prefixes where the model would answer wrong, and a low-confidence trigger can suppress prefixes where it would answer right — the field has no formal adequacy criterion linking the two.

## Key Insight

Because the trigger policy and the answer model are trained on the same streaming prefixes, a proper scoring rule over the trigger probability evaluated against the *realized answer-correctness label of the model's own answer* is a strictly proper incentive: it is minimized uniquely when the firing probability equals the conditional probability of answer correctness given the prefix, which forces monotonicity of trigger confidence in correctness under fixed partial observation.

## Method

**What it is.** CaliTrigger is a post-hoc calibration and fine-tuning procedure that wraps an ALIVE-style proactive streaming model. Inputs: a frozen (or LoRA-adapted) streaming video LLM with a trigger head producing firing logit $z_t$ at prefix $t$, plus a held-out set of streaming prefixes with realized answer-correctness labels from the model's own answers. Outputs: a recalibrated trigger head whose firing probability $p_t = \sigma(z_t)$ is a proper-scoring-rule-optimal estimate of $\Pr[\text{answer correct} \mid \text{prefix}_t]$, and a deployment threshold $\tau^\star$ chosen on the calibration split.

**Load-bearing assumption (stated explicitly).** CaliTrigger assumes that the streaming LLM's last hidden state $h_t$ is a sufficient statistic for the correctness of its own answer at prefix $t$, so that a Brier-trained head $g_\phi(h_t)$ can recover $\Pr[\text{answer correct} \mid \text{prefix}_t]$ from the same representation the answer head already uses. Because the answer head is a deterministic argmax over $h_t$, the realized label $y_t$ is a deterministic function of $h_t$; the Brier minimizer therefore learns the local base rate of correct answers in each hidden-state neighborhood (a nearest-neighbor correctness classifier) rather than a smooth correctness probability tied to the answer event. To make the correctness signal an explicit measured quantity rather than an assumed property of $h_t$, CaliTrigger decomposes the trigger input into two signals: (i) the ALIVE actionability logit $z^{\text{act}}_t$ (frozen, from the original trigger head), and (ii) a separately-trained answer-verification signal $v_t$ produced by a held-out verifier head $g_\psi$ trained on the same prefixes to predict $y_t$ from $h_t$ with a logistic loss. The fused trigger logit is $z_t = \alpha \, z^{\text{act}}_t + (1-\alpha) \, v_t$ with $\alpha=0.5$ fixed, and the calibration head $g_\phi$ is trained on the fused logit. The proper-scoring-rule argument survives because the target is now a genuine non-degenerate correctness probability estimated from two independent signals, not a degenerate function of a single deterministic head.

**How it works.**

```pseudocode
# Phase 0: prefix harvesting (per held-out video stream)
for each held-out stream S with ground-truth answers:
    run frozen model on S; at every prefix t where a query is available:
        record prefix features h_t (last hidden state), firing logit z_t,
                model's answer a_t, correctness y_t = 1[a_t matches GT]
    keep only prefixes where the model was queried (|D| ~ 50k prefixes)

# Phase 0b: train held-out verifier head g_psi (2-layer MLP, hidden=256, GeLU)
#   input: h_t; output: scalar v_t; loss: binary cross-entropy against y_t
#   trained on a disjoint 20% split of D (10k prefixes), Adam lr=1e-4, 5 epochs
#   (this is the explicit correctness signal; it is NOT the ALIVE trigger head)

# Phase 1: fit a proper-scoring-rule calibration head
#   g_phi: [z_act_t, v_t] -> scalar logit; initialized from the ALIVE trigger head
#   fused logit z_t = 0.5 * z_act_t + 0.5 * v_t
for epoch in 1..E (E=5):
    for batch B subset D:
        p_t = sigmoid(g_phi([z_act_t, v_t]))
        # Brier loss against realized correctness (proper scoring rule)
        L_brier = mean_{(t) in B} (p_t - y_t)^2
        # monotonicity regularizer: within a stream, later prefixes that are
        # strictly more informative (more frames seen) must not have lower p
        L_mono = mean over consecutive (t, t+1) in same stream:
                     relu(p_t - p_{t+1})
        # temperature-scaled ECE penalty on the batch
        L_ece  = | mean_bin(p_t) - mean_bin(y_t) |   over 15 equal-mass bins
        L = L_brier + lambda_m * L_mono + lambda_e * L_ece
        # lambda_m = 0.1, lambda_e = 0.05 (fixed, not tuned on test)
        update phi by Adam (lr=1e-4)

# Phase 2: threshold selection on a disjoint calibration split
sort calibration prefixes by p_t; choose tau* = argmax_tau  F1(1[p_t>=tau], y_t)
# (F1 over firing-vs-correctness; ties broken by lower firing rate)

# Phase 3: deployment
at inference, fire iff p_t >= tau*; answer is produced by the unchanged answer head
```

**Why this design.** We chose a Brier-score regression head over ALIVE's BACA advantage objective because BACA optimizes a *policy advantage* that is invariant to monotone rescaling of the trigger logit, so it cannot by construction enforce that $p_t$ equals the conditional correctness probability; accepting the cost that CaliTrigger needs a held-out prefix set with ground-truth answers and cannot be trained purely online. We chose to initialize from the ALIVE trigger head rather than train from scratch because the ALIVE head already encodes *when the stream is semantically actionable*, and we only need to recalibrate its confidence surface; accepting the cost that CaliTrigger inherits ALIVE's failure modes on out-of-distribution scenes. We chose the monotonicity regularizer $L_{mono}$ over a post-hoc isotonic regression because isotonic regression would break the shared hidden-state representation across streams and destroy multi-turn context reuse; accepting the cost that $L_{mono}$ is a soft constraint that can be violated on noisy prefixes. We chose an explicit ECE penalty in addition to Brier because Brier alone can be minimized by a well-ranked but miscalibrated head on heavy-tailed correctness distributions; accepting the cost of an extra hyperparameter $\lambda_e$. Crucially, a domain expert would not describe CaliTrigger as a variant of BACA: BACA is an advantage estimator that never sees the correctness label, whereas CaliTrigger's loss is a proper scoring rule whose unique minimizer is the conditional correctness probability — a different objective surface, not a reweighting of BACA.

**Why it measures what we claim.** The firing probability $p_t = \sigma(g_\phi([z^{\text{act}}_t, v_t]))$ measures the probability that the model's own answer at prefix $t$ is correct because the Brier score is a strictly proper scoring rule whose population minimizer is exactly $\Pr[y_t = 1 \mid z^{\text{act}}_t, v_t]$ under the assumption that the held-out prefix distribution matches deployment and that the fused signal $[z^{\text{act}}_t, v_t]$ is a sufficient statistic for correctness; this assumption fails when the model's answer head is stochastic (sampling temperature > 0) or when the held-out videos differ in scene statistics from deployment, in which case $p_t$ reflects the held-out correctness rate rather than the deployment one. The monotonicity penalty $L_{mono}$ measures the informational monotonicity of the stream because a later prefix strictly contains more frames than an earlier one under the assumption that additional frames never reduce the information available about the current query; this assumption fails when the query is about an event that has already passed, in which case $L_{mono}$ reflects temporal decay rather than informativeness. The threshold $\tau^\star$ measures the deployment operating point because F1 on a disjoint calibration split estimates the F1 at deployment under the assumption of exchangeability between calibration and deployment streams; this assumption fails under distribution shift, in which case $\tau^\star$ reflects the calibration-set prevalence of correct answers rather than the deployment prevalence.

## Contribution

['CaliTrigger: a proper-scoring-rule calibration objective (Brier + ECE + monotonicity) for proactive trigger heads in streaming video LLMs, replacing uncalibrated policy-advantage training with a confidence surface whose unique population minimizer is the conditional answer-correctness probability.', "A design principle — the trigger probability must be trained against the realized correctness label of the model's own answer, not against a policy advantage — that decouples the trigger objective from the answer objective while keeping them on the same prefix distribution.", 'A diagnostic protocol that measures the gap between trigger confidence and answer correctness (trigger-correctness AUC, calibration ECE) on streaming-prefix held-out sets, exposing where existing proactive-activation methods (ALIVE, StreamBridge, LiveStar, Dispider) are miscalibrated.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | FSI-Bench + OmniStar streaming splits | Proactive firing and correctness labels co-occur. |
| Primary metric | Firing–correctness F1 (calibrated) | Directly scores the trigger's decision quality. |
| Baseline | ALIVE (uncalibrated trigger) | Tests whether calibration alone improves firing. |
| Baseline | StreamBridge proactive activation | Tests against an alternate proactive mechanism. |
| Baseline | LiveStar response-silence decoding | Tests against a timing-decision alternative. |
| Baseline | Plain correctness classifier on $h_t$ | Tests whether proper-scoring-rule framing beats a classifier. |
| Ablation-of-ours | CaliTrigger minus Brier (ECE+mono only) | Isolates the proper-scoring-rule contribution. |
| Ablation-of-ours | CaliTrigger minus $L_{mono}$ | Tests whether monotonicity regularizer is redundant. |
| Ablation-of-ours | CaliTrigger minus verifier head (single-head Brier) | Tests the load-bearing sufficiency assumption. |

### Why this setup validates the claim

The central claim is that ALIVE's trigger head, though semantically actionable, is miscalibrated as a correctness predictor, and that a proper-scoring-rule recalibration of that head yields a firing rule whose precision/recall trade-off is genuinely tied to answer correctness. The design makes this falsifiable by pitting CaliTrigger against the uncalibrated ALIVE trigger (does recalibration matter at all?), against StreamBridge's decoupled activation model (does a different proactive mechanism already capture this?), against LiveStar's response-silence decoding (does an alternate timing rule dominate?), and against a plain correctness classifier trained on the same $h_t$ (does the proper-scoring-rule framing buy anything beyond a standard classifier?). The ablation removing the Brier term isolates whether the proper-scoring-rule objective—not just the extra regularization—drives any gain; if ECE+mono alone matches full CaliTrigger, the theoretical motivation collapses. The ablation removing $L_{mono}$ tests whether the monotonicity regularizer is redundant with Brier. The ablation removing the verifier head tests the load-bearing sufficiency assumption: if single-head Brier matches the two-signal version, the assumption holds; if not, the explicit correctness signal is doing the work. Firing–correctness F1 is the right metric because it jointly penalizes firing when answers are wrong and staying silent when answers are right, which is exactly the operating-point question the calibration is supposed to answer; accuracy alone would reward a trigger that fires constantly.

### Expected outcome and causal chain

**vs. ALIVE (uncalibrated trigger)** — On a case where the stream becomes semantically salient early but the model's answer is still wrong (e.g. an ambiguous object appears before disambiguating context), ALIVE's BACA-trained head fires because salience is high, producing a confident-but-incorrect response because its logit encodes actionability, not correctness probability. CaliTrigger instead maps that same hidden state through a Brier-trained head whose target is realized correctness, so it suppresses firing until the prefix is informative enough for a correct answer. We expect a concentrated F1 gain on early-salience/ambiguous-prefix subsets with near-parity on unambiguous ones.

**vs. StreamBridge proactive activation** — On a case requiring multi-turn context where the activation model sees only recent frames, StreamBridge fires on locally novel content that is irrelevant to the standing query, producing off-topic interjections because its activation is decoupled from answer correctness. CaliTrigger's head consumes the streaming LLM's own hidden state, which carries the query-conditioned correctness signal, so it fires only when the model's answer would actually be right. We expect a gap on multi-turn/query-persistence subsets, with parity on single-turn salient-event subsets.

**vs. LiveStar response-silence decoding** — On a case where a single forward-pass verification is under-confident because the event is subtle and slow-building, LiveStar stays silent past the optimal moment, producing missed interactions because its silence decision is a one-shot verification rather than a calibrated probability. CaliTrigger's threshold τ* is chosen on a disjoint calibration split to maximize F1, so subtle-but-correct prefixes clear the bar. We expect fewer missed firings on slow-onset event subsets, with comparable firing rate on abrupt events.

**vs. Plain correctness classifier on $h_t$** — On a case where the class prior of correct answers shifts between calibration and deployment (e.g., a deployment stream with more ambiguous prefixes), a plain classifier trained with cross-entropy on $h_t$ will track the training prior and misrank prefixes, producing a firing rule that is not invariant to prior shift. CaliTrigger's Brier head is a proper scoring rule whose ranking is invariant to class-prior shift under the assumption that the fused signal is sufficient, so it should maintain F1 under prior shift while the classifier degrades. We expect a gap on prior-shifted subsets, with parity on in-distribution subsets.

### What would falsify this idea

If CaliTrigger's gain over uncalibrated ALIVE is uniform across all prefix subsets rather than concentrated on early-salience/ambiguous and multi-turn subsets where the miscalibration mechanism predicts failure, the claim that recalibration targets a specific correctness-probability gap is wrong. Likewise, if the ECE+mono ablation matches full CaliTrigger, the Brier term is not doing the work and the proper-scoring-rule argument is unsupported; if the single-head Brier ablation matches the two-signal version, the load-bearing sufficiency assumption holds and the verifier head is unnecessary.

## References

1. ALIVE: Making Streaming Video Models Interactive
