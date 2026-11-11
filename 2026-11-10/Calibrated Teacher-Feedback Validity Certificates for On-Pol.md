# Calibrated Teacher-Feedback Validity Certificates for On-Policy Distillation

## Motivation

On-policy distillation methods such as GKD (Agarwal et al.) train the student on its own rollouts with teacher token-level supervision, but they treat that supervision as an oracle whose reliability is only diagnosed post hoc, never bounded before it is applied. The recent mismatch-tolerance study (How Much Training-Inference Mismatch Can On-Policy Distillation Tolerate?) likewise measures where teacher feedback degrades but offers an empirical mismatch budget that cannot certify, per token, whether the teacher's feedback is calibrated to the student's current rollout distribution. The structural gap is that no existing work replaces the oracle assumption with a mechanism that bounds teacher token-level error as a function of rollout distribution shift and uses that bound to gate or reweight teacher feedback during training.

## Key Insight

Teacher token-level error on student rollouts is governed by how far the rollout state leaves the teacher's high-confidence support, so a support-calibrated certificate computed from teacher entropy and a held-out residual probe yields a monotone per-token validity score that can be used as a gating weight without any oracle labels at training time.

## Method

**What it is.** We introduce TVC (Teacher Validity Certificate), a per-token scalar certificate `c_t ∈ [0,1]` attached to every teacher feedback signal during on-policy distillation. Inputs: student rollout tokens, teacher next-token distribution, a small held-out calibration set of (state, teacher-error) pairs. Output: a gated distillation loss `Σ_t c_t · D(student_t || teacher_t)` where `c_t` downweights teacher feedback whose certified error exceeds a target budget.

**Load-bearing assumption (stated explicitly).** The certificate is valid only under the assumption that teacher token-level error on student rollouts is monotone in (and therefore certifiable from) teacher entropy `H(s)`, max-probability confidence `M(s)`, and Euclidean distance `d(s)` to the teacher's high-confidence support, so that an isotonic probe fit on held-out states yields a per-token upper bound on teacher error at training time. This assumption fails when the teacher is confidently wrong on off-support states (high `M`, low `H`, large `d` but small `e`), in which case the certificate reflects geometric novelty rather than actual teacher error. The method below is specified to make this assumption testable and to report the empirical coverage of the resulting bound.

**How it works.**

```pseudocode
# Phase 1: fit the residual probe (one-time, held-out set S_cal)
for (state s, teacher_dist p_T, true_next_token y) in S_cal:
    H(s) = entropy(p_T(·|s))                    # teacher uncertainty
    M(s) = max_v p_T(v|s)                        # teacher confidence
    d(s) = ||s - nearest_teacher_support(s)||    # distance to teacher high-confidence support
    e(s) = -log p_T(y|s)                         # realized teacher token error
fit isotonic regression g: (H, M, d) -> e        # monotone in each argument

# Phase 2: per-token certificate during OPD training
for each student rollout x_1..x_T sampled from pi_S:
    for t in 1..T:
        s_t = (x_<t)
        e_hat_t = g(H(s_t), M(s_t), d(s_t))      # predicted teacher error
        c_t = clip(1 - e_hat_t / epsilon, 0, 1)  # epsilon = target error budget
        loss += c_t * KL(pi_S(·|s_t) || p_T(·|s_t))
    # Phase 3: online refresh every K=500 steps
    recompute d(s) against updated teacher support; refit g on fresh S_cal
```

Hyperparameters: `epsilon` (per-token error budget, set to the 90th percentile of `e(s)` on `S_cal`), `K` (probe refresh interval, 500 steps), `S_cal` size (2000 states sampled from student rollouts, teacher labels only).

**Why this design.** We chose an isotonic-regression residual probe over a learned neural gate because isotonicity is exactly the monotonicity assumption we need for the certificate to be interpretable as an error bound, accepting the cost that the probe cannot capture non-monotone interactions between entropy and support distance. We chose support-distance `d(s)` over raw KL between student and teacher because KL conflates high-entropy agreement with low-entropy disagreement, accepting the cost that support distance is a coarser geometric signal. We chose per-token gating over a global scalar weight because the mismatch-tolerance study shows layer- and token-level sensitivity, accepting the cost that per-token certificates add variance to the gradient. We chose `epsilon` as a quantile of the calibration residual rather than a fixed constant because the teacher's error scale varies across model pairs, accepting the cost that `epsilon` must be re-estimated per pair. Crucially, a domain expert would not describe TVC as a variant of GKD's loss or of the mismatch-tolerance measurement: GKD uses teacher feedback unweighted, and the mismatch study measures aggregate mismatch post hoc; TVC produces a per-token, pre-hoc certificate that modulates the loss.

**Why it measures what we claim.** The certificate `c_t` measures *teacher-feedback validity* because the isotonic probe `g` is fit on realized teacher error `-log p_T(y|s)` and is monotone in entropy, confidence, and support distance, so `e_hat_t` is an upper envelope of expected teacher error under the assumption that teacher error is monotone in these three statistics; this assumption fails when the teacher is confidently wrong on off-support states (high `M`, low `H`, large `d` but small `e`), in which case `c_t` reflects geometric novelty rather than actual teacher error. The support distance `d(s)` measures *rollout distribution shift* because it is the Euclidean distance from the student state to the teacher's high-confidence support set, under the assumption that the teacher's high-confidence support is a sufficient statistic for where its feedback is reliable; this fails when the teacher's confidence is miscalibrated on in-support states, in which case `d(s)` reflects teacher overconfidence rather than genuine shift. The refresh interval `K` measures *non-stationarity of the student rollout distribution* because the probe is refit on fresh student states every `K` steps, under the assumption that `K` steps is shorter than the timescale over which the student's distribution drifts; this fails when the student drifts faster than `K`, in which case the certificate lags the true shift.

## Contribution

(1) A per-token teacher-feedback validity certificate (TVC) that replaces the oracle assumption in on-policy distillation with an isotonic, support-calibrated error bound and gates the distillation loss accordingly. (2) A design principle: teacher feedback should be weighted by certified error relative to a per-pair error budget `epsilon` rather than used unweighted, with the budget derived from a held-out calibration set. (3) A calibration protocol that refits the residual probe on fresh student rollouts every `K` steps, making the certificate track non-stationary rollout distributions.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | GSM8K + summarization with injected rollout mismatch | Reasoning and free-form tasks expose teacher error differently |
| Primary metric | Student accuracy on off-support rollout states | Directly targets where certificate should help |
| Baseline | Unweighted GKD (on-policy KD) | Tests whether gating beats uniform teacher feedback |
| Baseline | Global-scalar-weighted KD | Tests whether per-token gating beats a single weight |
| Baseline | Quantization/noise-mismatch OPD | Tests robustness under the mismatch regime of interest |
| Ablation-of-ours | TVC with shuffled certificates (random gate) | Isolates whether certificate content, not gating, drives gains |

### Why this setup validates the claim

The central claim is that per-token teacher-feedback validity certificates improve on-policy distillation precisely where teacher feedback is unreliable, i.e. on off-support, high-teacher-error rollout states. The design is falsifiable because each baseline removes one specific ingredient. Unweighted GKD removes gating entirely, so any gain of TVC over it must come from downweighting bad teacher tokens. Global-scalar weighting keeps gating but discards per-token resolution, so a gain of TVC over it isolates the value of token-level certification rather than mere loss reweighting. The quantization/noise-mismatch baseline reproduces the regime where teacher feedback degrades, testing whether TVC's gains concentrate there rather than being uniform. The shuffled-certificate ablation is the critical control: it preserves the gating machinery and loss scale but destroys the certificate's information, so if TVC's advantage survives shuffling, the certificate is not doing the causal work. The primary metric — accuracy on off-support rollout states — is chosen because the predicted effect is specifically localized to states where the teacher's high-confidence support is distant; a metric aggregated over all states would dilute exactly the signal the certificate is meant to produce.

### Expected outcome and causal chain

**vs. Unweighted GKD** — On a case where the student drifts to an off-support state (large support distance, teacher confidently wrong), unweighted GKD applies full KL to a bad teacher distribution and pulls the student toward an incorrect next-token. Our method assigns a low certificate there because the isotonic probe maps large distance and mismatched confidence to high predicted error, so the loss downweights that token. We expect a noticeable accuracy gap concentrated on the off-support subset, with near-parity on in-support states.

**vs. Global-scalar-weighted KD** — On a mixed rollout containing both reliable and unreliable teacher tokens, a global scalar must compromise: a weight high enough to learn from good tokens over-trusts bad ones, and a weight low enough to suppress bad tokens starves good ones. Our per-token certificate separates these cases, applying full weight in-support and near-zero weight off-support within the same sequence. We expect TVC to beat the global scalar specifically on sequences that interleave reliable and unreliable tokens, with the gap shrinking on homogeneous sequences.

**vs. Quantization/noise-mismatch OPD** — Under heavy injected mismatch, the baseline's teacher feedback becomes systematically noisier, and without a validity signal the student absorbs that noise, degrading faster than the mismatch magnitude alone would predict. TVC's online-refreshed support distance and probe detect the widened unreliable region and gate it out, so degradation should be flatter in mismatch magnitude. We expect the baseline to show a steep accuracy drop as mismatch grows while TVC stays near the low-mismatch reference over a wider mismatch range.

### What would falsify this idea

If TVC's gain over unweighted GKD is uniform across on-support and off-support subsets rather than concentrated where support distance and predicted teacher error are high, the certificate is not modulating the loss for the reason claimed. Equally, if the shuffled-certificate ablation matches TVC's accuracy, the certificate carries no causal information and the benefit is merely from downweighting the loss.

## References

1. How Much Training-Inference Mismatch Can On-Policy Distillation Tolerate?
2. On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes
3. Large Language Models Are Reasoning Teachers
