# Calibrated Selective Camera Switching for Streaming Esports Direction

## Motivation

DirCast's director uses a deterministic argmax over per-camera relevance scores, providing no calibrated confidence and thus no ability to abstain when the relevance estimate is unreliable; this structural limitation recurs across streaming video-language agents that must act under partial observability, where point-estimate policies cannot defer. Because the offline Ego-Exo4D benchmark never required online decision confidence, the field's streaming replacements inherit an uncalibrated switch policy that silently commits to noisy cuts.

## Key Insight

A temperature-scaled per-camera relevance distribution, trained with a proper scoring rule and gated by its entropy, yields a selective switch policy whose abstention rate is monotone in the probability that the argmax cut disagrees with the human director, because entropy is a sufficient statistic for the calibration of the relevance head under the streaming context.

## Method

**What it is.** We introduce **CalDir** (Calibrated Director), an extension of DirCast's low-resolution all-camera scanner that outputs a per-camera relevance distribution over the K player cameras at each second, together with a scalar confidence, and gates the hold-or-switch action on a deferral score computed from that distribution. Inputs: low-resolution frames from all K cameras at time t plus the streaming context (aired video, spoken commentary, expiring non-selected views). Outputs: a switch decision (hold current camera or switch to camera j) or an abstain (hold) action, and a calibrated relevance distribution.

**Load-bearing assumption (stated explicitly).** The gating mechanism assumes that the deferral score is monotone in the probability that the argmax cut disagrees with the human director. The original design used entropy H_t as that score, which assumes entropy is a sufficient statistic for the 0-1 Bayes risk of the argmax. This assumption is not guaranteed: entropy is a function of the whole softmax vector, so a calibrated distribution can be high-entropy yet have a confident argmax (many low-probability cameras), or low-entropy yet be wrong (a sharp but miscalibrated peak). We therefore replace the entropy gate with a margin-based deferral score and validate monotonicity empirically on the calibration split before claiming it. Temperature scaling is retained for the distribution, but we do not assert that entropy is a sufficient statistic for Bayes risk.

**How it works.**
```pseudocode
# CalDir forward pass at second t
# K = number of player cameras (10 for Dota 2)
# tau = learned temperature scalar, initialized to 1.0
# H_thresh = entropy threshold, tuned on validation (range 0.5–2.0 nats)
# m_thresh = margin threshold, tuned on validation (range 0.05–0.5)

1. z_t = LowResScanner(frames_t[1..K], streaming_context)   # K logits, one per camera
2. p_t = softmax(z_t / tau)                                  # calibrated relevance distribution
3. H_t = -sum_k p_t[k] * log p_t[k]                          # entropy in nats
4. j* = argmax_k p_t[k]                                      # candidate switch target
5. m_t = p_t[j*] - max_{k != j*} p_t[k]                      # top-1 minus top-2 margin
6. if j* == current_camera:
       action = HOLD
   elif m_t < m_thresh:                                      # low margin -> uncertain
       action = ABSTAIN_HOLD                              # uncertain -> do not switch
   else:
       action = SWITCH(j*)
7. if action == SWITCH(j*):
       read high-res frame of camera j*; generate commentary
   else:
       continue reading current camera; generate commentary

# Training (two-stage):
# Stage 1: fit LowResScanner with cross-entropy to human director's camera label
# Stage 2: fit tau by minimizing temperature-scaled cross-entropy on a held-out
#          calibration split of matches (proper scoring rule)
# Stage 3: fit m_thresh by maximizing selective switching accuracy on validation
#          subject to abstention rate <= 0.3
```

**Why this design.** We chose a temperature-scaled softmax over a separate learned confidence head because temperature scaling is a proper scoring rule whose calibration guarantees hold for any logit distribution, whereas a separate head would need its own training signal and could decouple from the relevance ranking; the cost is that a single scalar tau cannot capture per-camera heteroscedasticity. We chose margin over entropy as the gating statistic because margin directly measures the confidence of the argmax decision and is monotone in the probability that the top-1 choice is correct under a calibrated distribution, whereas entropy aggregates the full distribution and is not a sufficient statistic for 0-1 Bayes risk; the cost is that margin ignores the mass on lower-ranked cameras, which matters when many cameras are plausible. We chose an abstain-hold action over forced switching because holding is the safe default in live directing (missing a cut is less costly than a bad cut), accepting that the policy may under-switch in fast-moving teamfights. Unlike DirCast, which commits to the argmax of a point-estimate relevance score, CalDir's switch policy is a function of a calibrated distribution and can defer; a domain expert would not describe this as a variant of DirCast because the decision surface is the margin of a properly scored distribution, not the argmax of a deterministic score.

**Why it measures what we claim.** The temperature-scaled cross-entropy on the held-out calibration split measures the calibration of the per-camera relevance distribution because temperature scaling preserves the argmax while making the softmax probabilities match empirical frequencies under the assumption that the logits are a sufficient statistic for the director's choice; this assumption fails when the streaming context drifts (e.g., a new match with different camera dynamics), in which case the calibrated probabilities reflect the training distribution's frequencies rather than the current match's. The margin m_t measures the unreliability of the switch decision because margin is monotone in the probability that the argmax is correct under a calibrated distribution; this assumption fails when the distribution is multimodal with two equally likely cameras, in which case margin is low but the argmax is still correct half the time, so the abstention rate overestimates the error rate. The threshold m_thresh measures the selective switching accuracy trade-off because it is tuned on validation to maximize accuracy subject to an abstention budget, under the assumption that the validation matches are exchangeable with the test matches; this assumption fails when the test matches have different camera-switching dynamics (e.g., different hero compositions), in which case the abstention rate is mis-calibrated for the test distribution.

## Contribution

["CalDir, a calibrated selective switching mechanism for streaming video-language directors that augments DirCast's low-resolution scanner with a temperature-scaled per-camera relevance distribution and an entropy-gated abstain-hold action.", 'A two-stage training procedure that fits the relevance head with a proper scoring rule (temperature-scaled cross-entropy) on a held-out calibration split and tunes the entropy threshold to maximize selective switching accuracy under an abstention budget.', "An empirical characterization of the trade-off between abstention rate and selective switching accuracy on held-out Dota 2 matches, showing that entropy-gated abstention reduces bad cuts relative to DirCast's deterministic argmax."]

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | 27-match Dota 2 director dataset (DirCast) | Real human director labels with per-hero cameras |
| Primary metric | Selective switching accuracy vs abstention rate | Directly measures calibrated deferral trade-off |
| Baseline | Rule-based director | Tests whether learned calibration beats heuristics |
| Baseline | Zero-shot GPT-5 director | Tests whether calibration beats strong point-estimate VLM |
| Baseline | DirCast argmax director | Tests whether margin gating beats deterministic argmax |
| Ablation-of-ours | CalDir with fixed tau=1, no abstain | Isolates calibration and deferral contributions |
| Diagnostic | Margin-vs-entropy comparison | Tests whether margin is monotone in director-disagreement while entropy is not |

### Why this setup validates the claim

The central claim is that a calibrated per-camera relevance distribution plus margin-gated abstention yields better switch decisions than deterministic argmax directing, especially where the director's choice is genuinely ambiguous. The dataset provides human director labels on real matches, so selective switching accuracy is measured against the ground-truth action, and the abstention rate is measured directly from the policy. The rule-based baseline tests whether any learned signal beats hand-coded heuristics; the zero-shot GPT-5 baseline tests whether CalDir's calibration beats a strong point-estimate VLM that commits to argmax without a deferral option; the DirCast baseline tests the specific sub-claim that margin-gated deferral on a properly scored distribution beats argmax of a deterministic score. The ablation (fixed temperature, no abstain) isolates whether gains come from calibration and deferral rather than from the scanner architecture. The margin-vs-entropy diagnostic directly tests the load-bearing assumption: we plot abstention rate against empirical director-disagreement rate for both scores on the calibration split and report where monotonicity breaks. Selective switching accuracy at a bounded abstention rate is the right metric because it captures both correctness and the cost of deferring, which is exactly the trade-off the method claims to improve; a plain accuracy metric would let a forced-switch policy hide its bad cuts.

### Expected outcome and causal chain

**vs. Rule-based director** — On a case where a teamfight breaks out across three lanes and multiple heroes are simultaneously relevant, the rule-based director produces a switch to a fixed priority camera because its hand-coded rules cannot weigh streaming commentary against visual salience. Our method instead reads the calibrated distribution over all cameras and holds when the margin is low, because temperature scaling exposes genuine multi-camera ambiguity. We expect a clear gap on multi-relevant-camera frames but rough parity on single-salient-hero frames.

**vs. Zero-shot GPT-5 director** — On a case where commentary mentions a hero off the current camera and the visual evidence is weak, the zero-shot director produces a confident wrong switch because it commits to the argmax of an uncalibrated score with no deferral path. Our method instead abstains and holds the current camera because its margin is below the threshold, avoiding the bad cut. We expect fewer bad cuts on commentary-visual-conflict frames, with the gain concentrated there rather than uniform across all frames.

**vs. DirCast argmax director** — On a case where two player cameras are near-equally relevant (a split push), the DirCast director produces a hard switch to one camera because it takes the argmax of a point-estimate score with no notion of ambiguity. Our method instead computes a calibrated distribution whose margin flags the near-tie and abstains, holding the safe default. We expect a noticeable reduction in switch-induced error on near-tie frames but parity on frames with a single dominant camera.

### What would falsify this idea

If CalDir's gain over the DirCast argmax baseline is uniform across all frame subsets rather than concentrated on the low-margin near-tie and commentary-conflict frames where the mechanism predicts its advantage, the central claim that margin-gated calibrated deferral is the causal driver would be wrong. A second falsifier: if the margin-vs-entropy diagnostic shows that entropy is equally monotone in director-disagreement on the calibration split, then the replacement of entropy by margin is unnecessary and the original entropy gate would suffice.

## References

1. DirCast: Live Esports Directing and Commentary with a Single Streaming Video-Language Model
