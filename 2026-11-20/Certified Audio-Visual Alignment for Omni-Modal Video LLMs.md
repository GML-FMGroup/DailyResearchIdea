# Certified Audio-Visual Alignment for Omni-Modal Video LLMs

## Motivation

Existing audio-visual video LLMs such as WeMM-Video and Dolphin achieve cross-modal alignment only as an implicit byproduct of next-token prediction or interleaved token merging, so nothing in the model certifies that an audio token stream and a visual token stream are actually describing the same moment. This is a structural gap, not a data-scale gap: scaling the corpus to 1.2T tokens (WeMM-Video) or adding multi-scale adapters (Dolphin) does not create an invariant that can be checked, rejected, or re-weighted at inference. Consequently, misaligned or hallucinated audio-visual segments pass through the model with no signal that alignment has failed.

## Key Insight

A cross-modal alignment certificate computed as the bimodal agreement of a held-out alignment probe on contrastively matched versus temporally shifted audio-visual token windows is a structural invariant that next-token prediction cannot produce, because it is defined on a statistic (probe margin) that is invariant to the language-modeling objective and therefore can be enforced as a regularizer and checked at inference.

## Method

**What it is.** We introduce CALM (Certified Audio-visual Language Model alignment), a training-time regularizer plus inference-time certificate for WeMM-Video-style omni-modal video LLMs. CALM takes as input the per-layer cross-modal attention maps between audio tokens and visual tokens at multiple temporal scales, and outputs (i) a scalar certificate c in [0,1] per temporal segment and (ii) a segment-level re-weighting mask applied to the audio-visual fusion before the LLM decoder.

**Load-bearing assumption (stated explicitly).** CALM's certificate is valid only if a frozen linear probe trained on held-out aligned/shifted pairs, fed pooled audio-visual features plus the contrastive margin r_{t,s}, yields a certificate whose value at inference tracks true cross-modal alignment rather than the probe's own in-distribution confidence. We make this assumption explicit and testable: the certificate is defined as the margin between the real probe and a random-label control probe (Hewitt & Liang, 2019), and is computed on features from a held-out temporal split disjoint from the LLM's training corpus, so that c_t measures alignment decodability beyond what the LLM's own training distribution supplies. Without this selectivity control and held-out split, the certificate would be a restatement of the shifted-negative construction rather than an independent invariant.

**How it works.**

```pseudocode
# Inputs: video V with T segments, audio A, visual tokens X_v (N_v x d),
#         audio tokens X_a (N_a x d) from WeMM-Video encoder at layers L = {4, 8, 12}
# Hyperparams: scales S = {1, 2, 4} seconds, probe margin m = 0.5, temperature tau = 0.07,
#              regularizer weight lambda = 0.1, rejection threshold theta = 0.35,
#              control-probe weight beta = 0.5, selectivity margin delta = 0.1

1. For each scale s in S and each temporal segment t:
     a. Pool X_v, X_a over window [t, t+s) -> v_{t,s}, a_{t,s} in R^d
     b. Cross-modal attention map M_{t,s} = softmax(v_{t,s} a_{t,s}^T / sqrt(d))  # (N_v x N_a)
     c. Compute contrastive correspondence score:
          pos = sim(v_{t,s}, a_{t,s}) / tau
          neg = mean_{t' != t} sim(v_{t,s}, a_{t',s}) / tau   # temporally shifted negatives
          r_{t,s} = pos - logsumexp(neg)

2. Alignment probe P (frozen linear head trained on a held-out alignment set):
     p_{t,s} = sigmoid(P([v_{t,s}; a_{t,s}; r_{t,s}]))
     certificate c_{t,s} = p_{t,s} * sigmoid(r_{t,s})   # in [0,1]

2b. Selectivity control (Hewitt & Liang):
     Train a random-label control probe P_ctrl on the same features with shuffled labels.
     Compute control output p_ctrl_{t,s} = sigmoid(P_ctrl([v_{t,s}; a_{t,s}; r_{t,s}])).
     Selectivity-controlled certificate:
          c_{t,s} = max(0, p_{t,s} - beta * p_ctrl_{t,s}) * sigmoid(r_{t,s})
     Require c_{t,s} > delta for the certificate to be considered valid; otherwise set c_{t,s} = 0.

3. Training regularizer (added to WeMM-Video loss):
     L_align = - mean_{t,s} [ y_{t,s} log c_{t,s} + (1-y_{t,s}) log(1-c_{t,s}) ]
     where y_{t,s} = 1 if segment (t,s) is a ground-truth aligned pair from AVU/TriSense-2M,
                   0 if it is a temporally shifted decoy.
     L_total = L_LM + lambda * L_align

4. Inference certificate and re-weighting:
     For each segment t: c_t = mean_s c_{t,s}
     if c_t < theta: mask audio tokens in segment t (set attention weight to 0)
     else: scale audio-visual fusion by c_t (soft re-weight)
     Emit certificate vector c = [c_1, ..., c_T] alongside the answer.
```

**Why this design.** We chose a held-out linear probe over a learned gating head because a learned head can co-adapt with the LLM and collapse to a constant, whereas a frozen probe trained on an independent alignment set gives a signal the LLM cannot trivially satisfy by memorization; the cost is that the probe must be refreshed when the encoder changes, which we accept as a periodic offline step. We chose temporally shifted negatives over random negatives because random negatives are trivially separable and would inflate the certificate without measuring alignment; the cost is that shifted negatives are harder and require segment-level ground-truth pairing. We chose multi-scale windows (1s, 2s, 4s) over a single scale because audio-visual alignment is not stationary across event durations, and a single scale would certify only one granularity; the cost is three forward passes of the probe per segment. We chose a hard rejection threshold plus soft re-weighting over a pure soft mask because a soft mask alone cannot reject a segment outright, and a hard mask alone loses graded confidence; the cost is one extra hyperparameter theta. We added a random-label control probe (Hewitt & Liang) and a held-out temporal split disjoint from the LLM's training corpus because a single probe trained on the same corpora as the LLM can achieve high accuracy on structurally meaningless encodings; the cost is training a second probe and partitioning the data, which we accept as a necessary selectivity control. A domain expert would not describe CALM as a variant of Dolphin's interleaved merging or WeMM-Video's progressive pre-training, because those mechanisms optimize the language-modeling loss and treat alignment as emergent, whereas CALM's certificate is defined on a probe margin that is explicitly trained against temporally shifted decoys and is therefore not a byproduct of next-token prediction.

**Why this measures what we claim.** The contrastive correspondence score r_{t,s} measures cross-modal alignment because it is the log-ratio of the matched-pair similarity to the temporally shifted-pair similarity, under the assumption that a temporally shifted audio window is a valid negative that preserves content statistics while destroying alignment; this assumption fails when the video contains repetitive events (e.g., a looping soundtrack), in which case r_{t,s} reflects content repetition rather than alignment. The probe output p_{t,s} measures the certificate's reliability because it is trained on a held-out set with known aligned and shifted pairs, under the assumption that the held-out distribution matches inference-time distribution; this assumption fails under domain shift (e.g., a new language or recording condition), in which case p_{t,s} reflects probe overconfidence rather than true alignment. The selectivity-controlled certificate c_{t,s} measures alignment decodability beyond the LLM's training distribution because it is the margin between the real probe and a random-label control probe on a held-out temporal split, under the assumption that the control probe captures the probe's in-distribution confidence; this assumption fails when the control probe is not matched in capacity or training data, in which case c_{t,s} reflects probe capacity differences rather than true alignment. The multi-scale average c_t measures segment-level alignment because alignment is assumed stationary within a segment window; this assumption fails at event boundaries, in which case c_t reflects an average of aligned and misaligned sub-segments rather than a clean segment label.

## Contribution

['A certified audio-visual alignment mechanism (CALM) that computes a per-segment certificate from multi-scale contrastive correspondence and a held-out alignment probe, enforced as a training regularizer and used at inference to reject or re-weight misaligned segments in a WeMM-Video-style omni-modal LLM.', 'A design principle: alignment certificates defined on probe margins against temporally shifted decoys are invariant to the language-modeling objective and therefore cannot be produced by scaling next-token prediction, establishing a checkable structural property for audio-visual video understanding.', 'An inference-time certificate vector emitted alongside model answers, enabling downstream consumers to audit which temporal segments were certified as aligned.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | TriSense-2M + AVU | Paired aligned/shifted segments with long-form audio-visual video |
| Primary metric | Alignment-conditioned accuracy gap (aligned vs. shifted subsets) | Isolates whether certificate tracks true alignment |
| Baseline | WeMM-Video-35B-A3B | Omni-modal SOTA without explicit alignment certificate |
| Baseline | Dolphin (AV-LLM) | Strong audio-visual LLM with emergent-only alignment |
| Baseline | TriSense | Query-reweighted fusion, no held-out alignment probe |
| Ablation-of-ours | CALM w/o shifted negatives (random negatives) | Tests whether decoy hardness drives certificate validity |
| Ablation-of-ours | CALM w/o selectivity control (single probe) | Tests whether control probe is necessary for validity |
| Ablation-of-ours | CALM w/ matched-control subset (propensity-matched) | Tests whether gap indexes alignment vs. subset difficulty |

### Why this setup validates the claim

The central claim is that CALM's certificate measures genuine audio-visual alignment rather than a byproduct of language-modeling loss. That claim is falsifiable only if we can separate segments where audio and vision are truly aligned from segments where they are temporally consistent but misaligned. TriSense-2M and AVU supply exactly this contrast because they contain ground-truth aligned tuples and temporally shifted decoys, letting the primary metric — the accuracy gap between aligned and shifted subsets — directly index whether the certificate discriminates alignment. WeMM-Video-35B-A3B tests the sub-claim that an explicit probe margin beats emergent alignment in a SOTA omni-modal model; Dolphin tests whether multi-scale certificate adds value over interleaved-merging alignment; TriSense tests whether a frozen held-out probe beats query-conditioned soft reweighting. The random-negative ablation is the key diagnostic: if the certificate's power comes from the hardness of shifted decoys, removing them should collapse the aligned-vs-shifted gap specifically, not uniformly degrade all subsets. The selectivity-control ablation tests whether the certificate's validity depends on the control probe margin, as required by the load-bearing assumption. The matched-control subset ablation tests whether the aligned-vs-shifted gap is confounded by subset difficulty: we stratify the gap by content statistics (event type, duration, audio energy) and use a propensity-matched subset to rule out the confound. A metric that merely averaged accuracy would confound alignment sensitivity with general capability, so the gap metric is the right detector.

### Expected outcome and causal chain

**vs. WeMM-Video-35B-A3B** — On a case where a long-form video contains a looping soundtrack over a scene change, the baseline produces confident answers that hallucinate the earlier event's audio because its alignment is emergent from next-token prediction and cannot reject a temporally stale audio window. Our method instead masks or down-weights the segment where the shifted-negative contrastive score r_{t,s} collapses, because the frozen probe was trained to fire on exactly this decoy pattern. We expect a noticeable accuracy gap on the repetitive-soundtrack subset but parity on clean single-event clips.

**vs. Dolphin (AV-LLM)** — On a case where an off-screen narrator describes an action not visible in frame, the baseline produces an answer that fuses the narration as if visually grounded because interleaved merging has no mechanism to certify cross-modal correspondence. Our method instead emits a low certificate c_t for that segment and soft-reweights the fusion, because the probe margin separates matched from temporally shifted audio-visual pairs. We expect a gap concentrated on off-screen-audio subsets with little change on tightly coupled speech-and-face clips.

**vs. TriSense** — On a case where the query is ambiguous and modality dropout is simulated, the baseline produces unstable reweighting that over-trusts audio because its Query-Based Connector adapts per query without an external alignment signal. Our method instead applies a segment certificate independent of the query, so audio is rejected when c_t < theta regardless of query phrasing. We expect a stable gain on modality-dropout and ambiguous-query subsets, with parity on queries that already name the audio event explicitly.

### What would falsify this idea

If the aligned-vs-shifted accuracy gap is uniform across all subsets rather than concentrated on the repetitive-soundtrack, off-screen-audio, and modality-dropout subsets where the predicted failure modes live, then the certificate is not measuring alignment but general capability. Equally, if the random-negative ablation matches the shifted-negative version on the gap metric, the claimed causal role of hard decoys is wrong. If the selectivity-control ablation (single probe) matches the full CALM on the gap metric, the control probe is unnecessary and the certificate is a restatement of the shifted-negative construction. If the matched-control subset shows the gap disappears after propensity matching, the gap indexes subset difficulty rather than alignment.

## References

1. WeMM-Video: Pushing the Frontiers of Open-Source Omni-Modal Video Understanding
