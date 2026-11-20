# Calibrated Modality-Reliability Gating for Omni-Modal Video Understanding

## Motivation

Existing audio-visual video LLMs such as WeMM-Video, Dolphin, and TriSense assume that all input streams (visual, audio, speech) are present and jointly informative at inference; their fusion layers are either fixed or query-conditioned but not calibrated to per-sample modality reliability. When a stream is corrupted, missing, or uninformative, these models cannot degrade gracefully because they lack an explicit, calibrated estimate of each modality's contribution, leading to hallucination and brittle fusion. This structural gap recurs across the WeMM-Video, Dolphin, and TriSense branches, and no single branch resolves it with a calibration-aware gating mechanism.

## Key Insight

The reliability of each modality is identifiable from the model's own per-modality predictive distributions under controlled dropout interventions, and a temperature-scaled reliability head trained on those interventions yields weights that are monotone in true modality informativeness, enabling calibrated gating without any external reliability labels.

## Method

**What it is.** We introduce CMRG (Calibrated Modality-Reliability Gating), a lightweight reliability estimator attached to a frozen WeMM-Video backbone. Given a video sample with visual, audio, and speech streams, CMRG outputs a per-modality reliability score $r_m \in [0,1]$ for $m \in \{v,a,s\}$ and a calibrated uncertainty $u_m = 1 - r_m$; these gate the fusion layer so that unreliable or missing streams are down-weighted before the LLM decoder.

**How it works.**

```pseudocode
# Inputs: frozen WeMM-Video backbone B, per-modality encoders E_v, E_a, E_s
# Training set D_train = {(video, audio, speech, QA)}; dropout validation D_val
# Hyperparams: temperature T (init 1.0), dropout rate p_drop=0.3, calibration bins K=15
# Reliability head: 2-layer MLP per modality, hidden=256, GeLU activation
# Corruption-augmentation distribution for D_val: {masking (p=0.3), Gaussian noise (SNR ∈ {5,10,20} dB), temporal desync (offset ∈ {0.5,1.0,2.0} s), cross-modal mismatch (swap with random sample)}

1. For each training sample x = (v, a, s, q):
     h_v, h_a, h_s = E_v(v), E_a(a), E_s(s)          # per-modality embeddings
     z_m = MLP_m(h_m)                                # reliability logit per modality
     r_m = sigmoid(z_m / T)                          # calibrated reliability score
2. Reliability head training (frozen backbone):
     For each modality m, sample dropout mask d ~ Bernoulli(p_drop) on the other modalities;
     compute fused representation F = sum_m r_m * h_m / sum_m r_m;
     compute per-modality predictive distribution p_m(y | h_m, q) via a shared linear probe;
     # Direct per-sample informativeness target:
     For each held-out sample in D_val, compute leave-one-modality-out accuracy gain g_m = Acc(all modalities) - Acc(all except m) using the frozen backbone's decoder on the same QA pair;
     loss_cal = ECE(p_m, y) on D_val + lambda * CE(p_m, y) on D_train + mu * MSE(r_m, g_m) on D_val  (lambda=1.0, mu=0.5)
     update MLP_m and T via gradient descent on loss_cal
3. At inference:
     compute r_m, u_m = 1 - r_m for each available modality;
     if a modality is missing, set r_m = 0;
     F = sum_m r_m * h_m / sum_m r_m;                  # gated fusion
     decoder(F, q) -> answer
```

**Why this design.** We chose a temperature-scaled sigmoid reliability head over a softmax query-conditioned connector (as in TriSense) because temperature scaling gives a single scalar per modality that is directly calibrated by ECE, whereas a query-conditioned connector conflates reliability with query relevance and cannot be calibrated per modality; the cost is that CMRG cannot express query-specific modality preferences, which we accept because the motivation targets dropout robustness rather than query adaptation. We chose to train the reliability head on controlled dropout interventions rather than on naturally occurring missing-modality data because interventions give a known ground-truth availability signal, whereas natural missingness is confounded with content; the cost is that the dropout distribution must approximate test-time corruption, which we mitigate by using a mixture of random masking, Gaussian noise, temporal desync, and cross-modal mismatch. We chose a frozen backbone with a lightweight head over end-to-end fine-tuning because freezing preserves WeMM-Video's omni-modal capabilities and isolates the calibration objective, whereas fine-tuning would entangle reliability learning with representation drift; the cost is that the head cannot reshape the backbone's features, which we accept because the reliability signal is a post-hoc gate. A domain expert would not describe CMRG as a variant of TriSense's Query-Based Connector because TriSense reweights modalities by query similarity without any calibration objective or dropout intervention, whereas CMRG's weights are trained to minimize ECE on a held-out dropout validation set and are monotone in modality informativeness by construction.

**Why it measures what we claim.** The per-modality reliability score $r_m = \sigma(z_m / T)$ measures modality informativeness because the temperature-scaled sigmoid is trained to minimize expected calibration error on a dropout validation set where the ground-truth availability of each modality is known by construction, and additionally trained to predict the per-sample leave-one-modality-out accuracy gain $g_m$; this assumption fails when the dropout distribution does not cover the test-time corruption type (e.g., temporal desynchronization rather than masking), in which case $r_m$ reflects the model's confidence under the training dropout distribution rather than true reliability. The ECE on $p_m(y|h_m,q)$ measures calibration of the per-modality predictive distribution because ECE compares predicted confidence to empirical accuracy across $K=15$ bins on held-out data; this assumption fails when the validation set is small or not representative, in which case ECE is a noisy estimate and the reliability head may overfit to bin boundaries. The gated fusion $F = \sum_m r_m h_m / \sum_m r_m$ measures graceful degradation because down-weighting a corrupted modality reduces its contribution to the fused representation proportionally to its estimated unreliability; this assumption fails when the corrupted modality's embedding $h_m$ is adversarially aligned with the correct answer, in which case down-weighting does not prevent hallucination and the gate must be combined with a consistency check.

## Contribution

['A calibrated modality-reliability estimator (CMRG) that produces per-modality reliability scores and uncertainty estimates via temperature-scaled sigmoid heads trained with an ECE objective on controlled dropout validation sets.', 'A gating mechanism that replaces fixed or query-conditioned fusion with reliability-weighted fusion, enabling graceful degradation under modality dropout without retraining the backbone.', 'An empirical design principle: reliability calibration on dropout interventions is a sufficient signal for gating in omni-modal video LLMs, and temperature scaling is necessary for monotonicity of the gate.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | WeMM-Video omni-modal benchmark suite with controlled modality dropout and corruption injection (masking, Gaussian noise, temporal desync, cross-modal mismatch) | Covers visual, audio, speech; supports corruption injection |
| Primary metric | Calibration-aware accuracy under modality corruption (ECE + accuracy) | Directly tests reliability gating and graceful degradation |
| Baseline | Frozen WeMM-Video-35B-A3B, no gating | Isolates benefit of reliability head |
| Baseline | TriSense-style query-conditioned connector | Tests our claim that calibration beats query reweighting |
| Baseline | Uniform fusion over available modalities | Tests whether gating beats naive averaging |
| Ablation-of-ours | CMRG without ECE calibration (CE-only head) | Isolates calibration objective from gating mechanism |
| Ablation-of-ours | CMRG without leave-one-modality-out target (ECE-only head) | Isolates direct informativeness target from calibration |
| Ablation-of-ours | CMRG with masking-only dropout (no desync/mismatch) | Tests dropout-to-test-corruption equivalence assumption |
| Diagnostic | Spearman correlation between $r_m$ and ground-truth informativeness $g_m$ across corruption severity sweep | Directly tests monotonicity claim |

### Why this setup validates the claim

The central claim is that a calibrated, dropout-trained reliability head on a frozen omni-modal backbone yields graceful degradation when a stream is corrupted or missing, and that this degradation is better than what query-conditioned reweighting or uniform fusion provides. The frozen-backbone baseline tests the null hypothesis that gating adds nothing over the backbone's native fusion, so any gap attributable to CMRG isolates the head's contribution. The TriSense-style connector tests the specific sub-claim that calibration (ECE on dropout validation) matters beyond query relevance: if query-conditioned reweighting already handles corruption, CMRG's calibration objective is redundant. The uniform-fusion baseline tests whether the gain is simply from down-weighting rather than from calibrated down-weighting, and the CE-only ablation separates the calibration term from the gating mechanism itself. The ECE-only ablation isolates the contribution of the direct leave-one-modality-out informativeness target from the ECE term. The masking-only ablation tests the assumption that the training dropout distribution covers the test-time corruption manifold, including temporal desync and cross-modal mismatch. The Spearman correlation diagnostic directly measures whether $r_m$ is monotone in ground-truth informativeness across corruption severity, substantiating the 'monotone by construction' claim. Calibration-aware accuracy is the right primary metric because the predicted effect is a shift in the confidence-accuracy relationship under corruption, not raw accuracy on clean data; a method that improves clean accuracy but not corruption-subset ECE would not validate the reliability claim.

### Expected outcome and causal chain

**vs. Frozen WeMM-Video-35B-A3B, no gating** — On a case where the audio stream is replaced with Gaussian noise while vision and speech remain informative, the ungated backbone produces a confident but wrong answer because its fusion layer has no signal distinguishing corrupted from clean embeddings, so noise is averaged in at full weight. Our method instead assigns low $r_a$ to the corrupted audio via the dropout-trained head, so the fused representation $F$ is dominated by $h_v$ and $h_s$, and we expect a noticeable gap on the corrupted-audio subset with near-parity on clean subsets.

**vs. TriSense-style query-conditioned connector** — On a case where the query asks about a visual detail but the audio is clean and the vision stream is corrupted, the query-conditioned connector up-weights vision because it conflates query relevance with reliability, producing a confident wrong answer grounded in corrupted visual features. Our method instead assigns low $r_v$ from the calibration objective regardless of query, so the answer draws on clean audio and speech, and we expect a gap concentrated on query-relevance/corruption-mismatch cases rather than uniform improvement.

**vs. Uniform fusion over available modalities** — On a case where speech is missing and audio is mildly corrupted but vision is clean, uniform fusion dilutes the clean visual signal with corrupted audio at equal weight, producing degraded but not catastrophic answers. Our method instead sets $r_s = 0$ and down-weights $r_a$ proportionally to its calibrated unreliability, so we expect a moderate gap on the missing-speech-plus-corrupted-audio subset and near-parity on fully clean subsets, with the gap scaling with corruption severity.

### What would falsify this idea

If CMRG's gain over uniform fusion is uniform across clean and corrupted subsets rather than concentrated where the predicted failure mode occurs, or if the CE-only ablation matches the calibrated head on corruption-subset ECE, then the calibration objective is not doing the work the claim attributes to it and the central reliability argument is wrong. Additionally, if the Spearman correlation between $r_m$ and ground-truth informativeness $g_m$ is not significantly positive across the corruption severity sweep, or if the masking-only ablation matches the full corruption-augmentation head on temporal-desync subsets, then the dropout-to-test-corruption equivalence assumption fails and the reliability scores do not reflect true informativeness.

## References

1. WeMM-Video: Pushing the Frontiers of Open-Source Omni-Modal Video Understanding
