# Verifier-Calibrated Acceptance for Recursive Self-Improvement Loops

## Motivation

Recursive self-improvement (RSI) loops such as Piccolo commit every round's self-generated data and model update without any structural guarantee that the new distribution dominates the prior one; the only progress signal is a bare correctness filter, which is systematically miscalibrated under the distribution shift that RSI itself induces. This is a meta-level failure that recurs across the self-improvement literature: Piccolo's feedback-driven refinement, SCoRe's RL on self-generated correction traces, and the sharpening framework all assume that self-evaluation remains a valid ranking of response quality at each round, yet none provides an operational invariant that detects when that assumption breaks. The root cause is that acceptance is decided by an absolute correctness score rather than by a relative, calibrated comparison against the previously accepted distribution on a retained anchor set.

## Key Insight

A round-to-round monotonicity certificate is obtainable without ground-truth labels because a verifier ensemble calibrated on held-out self-generated data yields a ranking that, restricted to a fixed anchor set, must remain statistically consistent with the prior round's accepted ranking if and only if the new round's distribution has not regressed; inconsistency is a detectable certificate of regression.

## Method

**What it is.** VC-RSI (Verifier-Calibrated Recursive Self-Improvement) wraps an RSI loop with an acceptance gate: at each round it takes the candidate self-generated dataset and candidate model update, and outputs either COMMIT (advance the loop) or REJECT (retain the prior model and data recipe), together with a per-round certificate recording the calibrated ranking consistency on a retained anchor set.

**Load-bearing assumption (stated explicitly).** The gate treats the round-0 verifier ensemble c_0, frozen for the entire run, as a fixed measurement instrument: KS_t computed with c_0 on the anchor set is assumed to change only when the anchor samples' true quality distribution regresses, not when the verifier's calibration map would have changed. This holds only if (A) the anchor set is exchangeable with the previously-accepted distribution and (B) the isotonic-calibrated ranking under c_0 is monotone in true quality. The failure mode F is anchor drift or slice-level calibration/train shift making KS_t reflect staleness rather than regression; we measure detection power under injected regressions to stress-test F.

**How it works.**

```pseudocode
Input: base model M_0, research agent A, anchor set S_anchor (|S|=512, fixed at round 0),
       verifier ensemble V = {v_1,...,v_K} (K=5, heterogeneous: 3 judge LLMs + 2 reward models),
       rounds T, calibration fraction rho=0.2, consistency threshold tau=0.05 (KS stat), min anchor coverage gamma=0.9

Round 0 (initialization):
  - Generate D_0^cal (512 self-generated samples, stratified by semantic slice) from M_0.
  - Fit isotonic regression per verifier v_k on D_0^cal: map raw score -> empirical correctness.
  - Freeze the resulting calibrated ensemble c_0 (median of K isotonic maps) for ALL subsequent rounds.
  - Score S_anchor with c_0; store the round-0 anchor score vector s_0 = {c_0(x): x in S_anchor}.

Round t = 1..T:
  1. Candidate generation: A(M_{t-1}) produces candidate data D_t (self-generated samples + labels)
     and candidate model M_t^cand trained on D_t.
  2. Held-out calibration split: partition D_t into D_t^cal (fraction rho, stratified by semantic slice)
     and D_t^train (remainder). M_t^cand is trained ONLY on D_t^train.
  3. Diagnostic per-round refit (NOT used for acceptance): fit a fresh isotonic regression per verifier
     on D_t^cal to obtain c_t^refit; record per-verifier calibration error e_k = ECE(v_k on D_t^cal)
     and the refit KS_t^refit = sup_z |F_t^refit(z) - F_0(z)| on S_anchor. These are logged as diagnostics.
  4. Anchor scoring: score every x in S_anchor with the FROZEN ensemble c_0 (same map every round).
     Let s_t = {c_0(x): x in S_anchor}.
  5. Consistency test: compute the two-sample Kolmogorov-Smirnov statistic
     KS_t = sup_z |F_t(z) - F_0(z)| where F_t and F_0 are empirical CDFs of s_t and s_0 respectively.
     (Because c_0 is frozen, KS_t measures only change in the anchor score distribution.)
  6. Acceptance rule (closed form):
       COMMIT iff  KS_t <= tau  AND  max_k e_k <= 0.1  AND  anchor coverage >= gamma
       else REJECT: discard D_t, keep M_{t-1}, and log the certificate (KS_t, e_k, coverage, KS_t^refit).
  7. On COMMIT: append 64 fresh self-generated samples (stratified by semantic slice) to S_anchor
     to track distribution drift; M_t <- M_t^cand. On REJECT: anchor set unchanged.
Output: final model M_T, per-round certificates {(KS_t, e_k, coverage, KS_t^refit, decision)}.
```

**Why this design.** We chose a *relative* consistency test on a fixed anchor set over an *absolute* correctness threshold because absolute thresholds are exactly what breaks under RSI-induced distribution shift (the failure mode of Piccolo's bare correctness filter), accepting the cost that we can detect regression but not certify absolute improvement beyond the anchor. We chose to freeze the verifier ensemble c_0 at round 0 and score the anchor set with that single frozen ensemble every round, so KS_t measures only change in the anchor score distribution and not change in the calibration map; the per-round refit KS is reported as a separate diagnostic, not as the acceptance signal. We chose isotonic regression per verifier over a shared temperature-scaling head because isotonic regression is non-parametric and monotone by construction, which is required for the KS test to be meaningful, accepting the cost that it needs a minimum calibration split (rho=0.2) and cannot extrapolate beyond the observed score range. We chose a median ensemble over a mean or a learned aggregator because median is robust to a single miscalibrated verifier, accepting the cost that it discards information when verifiers are correlated. We chose to append fresh anchor samples each round over a fully frozen anchor because a frozen anchor becomes unrepresentative after several rounds, accepting the cost that the anchor set grows and the KS test must be recomputed on a growing set (we cap at 2048 by stratified reservoir sampling). A domain expert would not describe this as a variant of SCoRe or sharpening: SCoRe trains a self-correction policy with RL and sharpening analyzes SFT-vs-RLHF amplification, whereas VC-RSI introduces an *acceptance invariant* external to the training objective that can veto a round; the closest named prior method is Piccolo's correctness-filtered RSI, and the structural property VC-RSI adds — a calibrated relative ranking that must remain consistent on a retained anchor set — is absent from Piccolo, SCoRe, and sharpening.

**Why it measures what we claim.** The KS statistic KS_t on the anchor set measures *round-to-round distributional non-regression* because the anchor set is fixed (modulo controlled growth) and the calibrated ensemble c_0 is frozen and monotone in correctness by isotonic construction, so a consistent ranking implies the new round does not systematically demote previously accepted high-quality samples; this assumption fails when the anchor set drifts faster than the 64-sample-per-round replenishment can track, in which case KS_t reflects anchor staleness rather than regression. The per-verifier ECE e_k measures *calibration reliability of the acceptance signal* because isotonic regression on a held-out self-generated split yields an unbiased estimate of P(correct | score) under the assumption that D_t^cal is exchangeable with D_t^train, which fails when the research agent's data construction induces a slice-level shift between calibration and training splits, in which case e_k reflects split mismatch rather than verifier miscalibration. Anchor coverage measures *representativeness of the certificate* because coverage counts the fraction of anchor semantic slices that receive at least one calibrated score above the median, under the assumption that semantic slices partition the capability space; this fails when a new capability emerges outside the existing slice taxonomy, in which case coverage reflects taxonomy completeness rather than distributional coverage.

## Contribution

['A verifier-calibrated acceptance test (VC-RSI) that converts a bare correctness filter into a round-to-round monotonicity certificate using isotonic-calibrated verifier ensembles and a KS consistency statistic on a retained anchor set.', 'A design principle: RSI acceptance should be decided by relative ranking consistency against the prior accepted distribution rather than by absolute correctness, making capability regressions structurally detectable without ground-truth labels.', 'A per-round certificate record (KS statistic, per-verifier ECE, anchor coverage) that makes each RSI commit/reject decision auditable and attributable to a specific failure mode.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | Length-generalization arithmetic + maze suite (10→100 digit) | Directly stresses RSI-induced distribution shift |
| Primary metric | Out-of-distribution accuracy on hardest held-out tier | Tests capability retention across rounds |
| Baseline | Piccolo-style correctness-filtered RSI | Closest named prior; bare absolute filter |
| Baseline | SCoRe (RL self-correction) | Tests whether acceptance invariant beats RL self-correction |
| Baseline | Sharpening (SFT self-improvement) | Tests whether gate beats pure amplification |
| Baseline | Replay-buffer with fixed absolute threshold on anchor set | Isolates what calibrated KS test adds beyond 'keep old data and check it' |
| Ablation-of-ours | VC-RSI with absolute threshold instead of KS | Isolates the relative-consistency mechanism |
| Ablation-of-ours | VC-RSI with per-round refit ensemble (c_t^refit) as acceptance signal | Tests the load-bearing assumption that freezing c_0 is necessary |

### Why this setup validates the claim

The central claim is that an *acceptance invariant* on calibrated relative ranking prevents the silent regression that plagues absolute-threshold RSI. The dataset choice matters because length-generalization tasks produce exactly the drift regime where absolute correctness filters break: as the self-generated distribution creeps toward longer, harder instances, an absolute threshold either freezes progress or waves through degraded rounds. Piccolo-style filtering tests the sub-claim that bare correctness filtering accepts regressive rounds; SCoRe tests whether RL self-correction alone can substitute for an external veto; sharpening tests whether pure SFT amplification saturates. The replay-buffer baseline with a fixed absolute threshold on the anchor set is the critical control: it keeps old data and checks it, but without calibrated relative ranking, so any gap between full VC-RSI and this baseline isolates the contribution of the calibrated KS test over simple replay. The ablation swapping KS for an absolute threshold keeps everything else identical, so any gap between full VC-RSI and the ablation is attributable to the relative-consistency mechanism rather than to the ensemble or calibration machinery. The second ablation (per-round refit ensemble) directly tests the load-bearing assumption: if the refit ensemble's KS_t^refit drives COMMIT/REJECT decisions as well as the frozen c_0, then freezing is unnecessary; if it produces spurious REJECTs, the frozen-ensemble design is validated. The primary metric — hardest-tier OOD accuracy — is the right detector because regression under RSI manifests as loss of previously-solved hard instances, not as degradation on easy in-distribution items where all methods saturate.

### Expected outcome and causal chain

**vs. Piccolo-style correctness-filtered RSI** — On a case where round t's self-generated data shifts toward longer instances and the model's verifier scores inflate on those instances, the baseline commits because its absolute correctness filter sees high pass rates, yet the committed model has silently demoted previously-solved 80-digit items; its mechanism gap is that correctness on the *new* distribution says nothing about retention on the *old*. Our method instead compares the calibrated ranking distribution on the fixed anchor set against the round-0 frozen ensemble, so an inflated-but-shifted verifier produces a large KS statistic and triggers REJECT. We expect a noticeable gap on the hardest retained tier (where regression concentrates) but near-parity on easy tiers where both methods saturate.

**vs. SCoRe** — On a case where the RL self-correction policy learns to fit high-reward correction traces that are mode-collapsed onto one correction style, SCoRe produces corrections that look good under its own reward but fail on instances requiring a different correction mode; its mechanism gap is that the reward shapes the policy toward in-distribution high-reward responses without any cross-round retention check. Our method instead vetoes any round whose calibrated ensemble ranking on the anchor set drifts beyond tau, keeping the prior model when the new policy's ranking consistency breaks. We expect SCoRe to show higher variance across rounds (some rounds improving, some regressing) while VC-RSI shows monotone-or-flat hardest-tier accuracy with regressions converted to REJECTs.

**vs. Sharpening (SFT self-improvement)** — On a case where the SFT self-improvement loop has insufficient coverage of the hard tail, sharpening amplifies mass on high-quality sequences but cannot manufacture the tail capability, so successive rounds saturate; its mechanism gap is that SFT-based self-improvement is minimax-optimal only under coverage, and the anchor-consistency signal would detect the saturation as a *small* KS (no drift) — but the ablation distinguishes the two. Our method's gate does not prevent saturation, but the ablation (absolute threshold) is expected to accept saturated rounds that inflate verifier scores, whereas full VC-RSI's KS test stays quiet on genuine saturation, so the diagnostic signal is the ablation's divergence from full VC-RSI on the saturated subset.

**vs. Replay-buffer with fixed absolute threshold on anchor set** — On a case where the replay buffer retains old data but the absolute threshold on anchor scores is set from round 0, the baseline accepts rounds where the new model's anchor scores remain above the fixed threshold even if the *ranking* of anchor items has degraded (e.g., previously top-ranked hard items now score below easy items); its mechanism gap is that an absolute threshold on a fixed scale does not detect rank inversions that preserve the threshold crossing. Our method instead uses the KS statistic on the full anchor score distribution, which is sensitive to any change in the CDF shape, including rank inversions. We expect VC-RSI to outperform the replay-buffer baseline specifically on rounds where the model's anchor score distribution shifts without crossing the absolute threshold, leading to higher hardest-tier retention.

**vs. VC-RSI with per-round refit ensemble (ablation)** — On a case where the verifier's calibration map drifts due to shifted self-generated calibration data (as predicted by the sharpening framework), the refit ensemble produces a nonzero KS_t^refit even when anchor quality is unchanged, causing spurious REJECTs that stall progress; its mechanism gap is that it conflates verifier recalibration with quality regression. Our method freezes c_0, so KS_t is zero when quality is unchanged, and REJECTs occur only on genuine regression. We expect the refit ablation to show lower hardest-tier accuracy due to unnecessary REJECTs and higher variance in COMMIT/REJECT decisions across seeds.

### What would falsify this idea

If the full VC-RSI's gain over the absolute-threshold ablation is uniform across easy, medium, and hard subsets rather than concentrated on the hardest retained tier where the predicted regression mode lives, the relative-consistency mechanism is not doing what we claim. Equivalently, if REJECT decisions do not correlate with subsequent-round hardest-tier accuracy drops under the ablation, the KS statistic is measuring anchor staleness rather than regression.

## References

1. Piccolo: Co-Evolving Decision Models and Data through Recursive Self-Improvement
