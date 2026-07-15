# SubBandConsistency: Self-Supervised Reward via Frequency-Subband Masking for RL Post-Training in Multi-Instrument Transcription

## Motivation

MuScriptor uses reinforcement learning post-training with a reward that relies on instrument presence conditioning, but this fails when instrument labels are missing in dense polyphony, as is common in real-world recordings. Existing RL-based transcription methods require ground-truth instrument annotations for reward computation, creating a bottleneck that prevents generalization to unlabeled data. To enable RL post-training without instrument labels, a self-supervised reward signal that captures polyphonic accuracy is needed.

## Key Insight

Frequency-subband consistency provides a self-supervised signal because the model must reconstruct audio from partial spectral views, inherently capturing polyphonic interactions without requiring instrument labels.

## Method

(A) **What it is**: SubBandConsistency is a self-supervised reward function for RL post-training of multi-instrument transcription models. It takes a predicted multi-instrument piano roll, passes it through a differentiable synthesizer to produce a predicted spectrogram, and computes a reconstruction loss against the original audio after a random frequency-subband mask is applied. The loss is computed only on the masked subbands, forcing the model to infer spectral content from context.

(B) **How it works**:
```python
def subband_consistency_reward(audio_waveform, predicted_piano_roll, synthesizer, n_subbands=8, mask_ratio=0.5):
    # audio_waveform: (T,) waveform
    # predicted_piano_roll: (T, F, K) where K instruments
    S_orig = stft_magnitude(audio_waveform)  # (T, F)
    F = S_orig.shape[1]
    # Divide frequency bins into n_subbands randomly
    subband_indices = np.random.permutation(F).reshape(n_subbands, -1)
    # Randomly select mask_ratio of subbands to mask
    masked_subbands = np.random.choice(n_subbands, size=int(n_subbands*mask_ratio), replace=False)
    mask = np.ones(F, dtype=bool)
    mask[np.concatenate(subband_indices[masked_subbands])] = False
    S_masked = S_orig * mask  # zero out masked bins
    # Synthesize predicted audio
    pred_waveform = synthesizer(predicted_piano_roll)  # differentiable
    S_pred = stft_magnitude(pred_waveform)
    # Compute MSE only on masked bins
    loss = ((S_pred[~mask] - S_orig[~mask])**2).mean()
    return -loss  # reward
```
Hyperparameters: `n_subbands=8`, `mask_ratio=0.5`, synthesizer = pre-trained differentiable additive synthesizer (e.g., a harmonic model with learnable amplitudes, from [DDSP](https://github.com/magenta/ddsp)).

(C) **Why this design**: We chose random subband masking over fixed-frequency masking because it forces the model to handle diverse spectral contexts, reducing overfitting to specific patterns; the trade-off is that some masks may be trivially easy (e.g., masking only silence) or impossible (e.g., masking a fundamental of all instruments), but averaging over many episodes yields a robust gradient. We used a differentiable synthesizer (additive harmonic model) rather than a non-differentiable one (e.g., FDSP) to allow reward gradient propagation through the synthesis step; the trade-off is timbre fidelity, as additive models cannot capture noise or spectral variation—however, note-level accuracy is less dependent on timbre. We computed loss only on masked subbands rather than the full spectrogram to avoid the model trivially copying unmasked regions; this incentivizes the model to actually infer missing content, at the cost of ignoring spectral regions that may contain valuable training signal. We chose magnitude spectrogram over complex to avoid phase reconstruction difficulty; the trade-off is that phase errors are ignored, but phase is perceptually less critical for transcription tasks. Finally, we used MSE loss over L1 because it penalizes large deviations more heavily, encouraging sharp reconstruction which aligns with accurate pitch predictions.

(D) **Why it measures what we claim**: The reconstruction error on masked subbands measures **polyphonic accuracy** because to correctly reconstruct a masked subband, the model must correctly predict the pitches and amplitudes of all instruments contributing to that frequency region; if any instrument is wrong, the error increases. The assumption is that the masked subband contains contributions from multiple instruments; this assumption fails when a subband is dominated by a single instrument at a narrow frequency, in which case the error primarily measures that instrument's accuracy rather than polyphonic precision. The reward also measures **robustness to missing instrument labels** because the mask removes spectral information arbitrarily, and the model must infer the missing parts without knowing which instruments are absent; this assumption holds when the mask covers frequency regions that are not instrument-specific, but fails if a mask entirely removes a band where only one instrument plays, then the model must rely on context from other instruments' harmonics, still testing polyphonic inference. The differentiable synthesizer's ability to produce realistic audio from piano rolls is critical: the reward assumes that a correct transcription leads to a low reconstruction error; if the synthesizer is poor (e.g., cannot reproduce harmonic richness), the reward will penalize accurate transcriptions with unrealistic timbre, biasing the model to match the synthesizer's timbre rather than true audio. Thus, the method requires a synthesizer that is both differentiable and sufficiently realistic—a limitation we explicitly accept.

## Contribution

(1) A self-supervised reward function, SubBandConsistency, that uses random frequency-subband masking to provide a training signal for reinforcement learning post-training in multi-instrument transcription without requiring instrument labels. (2) A demonstration that subband consistency forces the model to infer missing spectral content from context, implicitly capturing polyphonic interactions among instruments. (3) An integration of a differentiable additive synthesizer into the RL reward path, enabling gradient-based optimization from audio-level feedback.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | Slakh2100 | Multi-instrument, high-quality, popular benchmark |
| Primary metric | Polyphonic note F1 | Captures instrument correctness in polyphony |
| Baseline: MT3 | Google’s MT3 | Strong multi-instrument baseline |
| Baseline: Perceiver TF | Perceiver TF from [Multitrack Music Transcription] | Recent strong architecture with time-frequency |
| Ablation (ours) | Full spectrogram loss (no masking) | Isolates effect of subband masking |

### Why this setup validates the claim
This design tests the central claim that our subband consistency reward improves multi-instrument transcription by forcing the model to infer missing spectral content. Slakh2100 provides mixed-instrument audio with complete pitch labels, enabling fair comparison. MT3 and Perceiver TF are strong baselines representing different architectural families; outperforming them on polyphonic F1 would demonstrate that self-supervised spectral inference yields better note accuracy. The ablation (full spectrogram loss) isolates the benefit of masked reconstruction: if our gain disappears without masking, the masking mechanism is causal. The metric (polyphonic F1) directly measures note-level correctness across instruments, aligning with the reward’s intended effect.

### Expected outcome and causal chain

**vs. MT3** — On a passage where piano and guitar share the same octave (e.g., C4–C5), MT3 often misses one instrument because its supervised loss lacks spectral consistency constraints, causing it to collapse overlapping harmonics to one source. Our method’s masked reconstruction compels the model to infer the missing subband’s content, preserving both instruments. We expect a noticeable F1 gap (e.g., ~5 points) on high-overlap segments, with parity on solo passages.

**vs. Perceiver TF** — On a recording from an unseen genre where some instruments are weakly present (e.g., a synth pad mixed low), Perceiver TF fails to transcribe it because its training only covers labeled instruments. Our method’s reward doesn’t require instrument labels; the model learns to reconstruct all spectral content from context, recovering soft instruments. We expect a larger recall gain (e.g., 3–4 points) on under-represented instruments.

### What would falsify this idea
If the full spectrogram ablation matches or surpasses our method’s performance, the masking mechanism is unnecessary, invalidating the core claim. Alternatively, if our gain is uniform across all density levels (not concentrated on dense polyphony), then the reward doesn’t specifically improve handling of overlapping instruments.

## References

1. MuScriptor: An Open Model for Multi-Instrument Music Transcription
2. Unaligned Supervision For Automatic Music Transcription in The Wild
3. YourMT3+: Multi-Instrument Music Transcription with Enhanced Transformer Architectures and Cross-Dataset STEM Augmentation
4. Multitrack Music Transcription with a Time-Frequency Perceiver
5. Automatic Piano Transcription with Hierarchical Frequency-Time Transformer
6. HPPNet: Modeling the Harmonic Structure and Pitch Invariance in Piano Transcription
7. Towards Automatic Transcription of Polyphonic Electric Guitar Music: A New Dataset and a Multi-Loss Transformer Model
8. Jointist: Joint Learning for Multi-instrument Transcription and Its Applications
