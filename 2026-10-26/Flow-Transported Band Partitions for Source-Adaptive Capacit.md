# Flow-Transported Band Partitions for Source-Adaptive Capacity in Flow-Matching Music Separation

## Motivation

The field's trajectory has removed the deterministic subband encoder that once provided source-adaptive spectral compression, leaving full-band flow-matching separators such as FLOMSS to allocate generative capacity uniformly across all frequencies and all sources. Because music sources are spectrally heterogeneous — bass energy concentrated below 250 Hz, cymbals and vocals broadband — this uniform allocation wastes modeling budget on regions that need little generative resolution and blurs source-specific fine structure where it matters. The new bottleneck is therefore to recover source-adaptive spectral compression inside the generative model itself, without reintroducing a hand-designed subband tree or a separate encoder.

## Key Insight

If the band-partition boundaries are treated as part of the flow state and transported by the same ODE that transports the audio, then the flow-matching regression loss simultaneously supervises the vector field and the partition, so spectral capacity is allocated by the same objective that generates the signal rather than by a hand-designed or separately trained encoder.

## Method

**What it is.** Flow-Transported Band Partitioning (FTBP) is a flow-matching separator whose vector field is factorized over a learned, per-source, per-example set of K=8 frequency bands. The band boundaries are state variables of the same ODE as the audio, so a single conditional flow-matching loss trains both the audio vector field and the partition. Inputs: a mixture spectrogram (magnitude, 1024 frequency bins) and source index (4-way one-hot); outputs: the separated source spectrogram and the transported band boundaries.

**How it works.** The mechanism has three coupled pieces: (i) a differentiable band-partition operator that maps a full-band spectrogram into K=8 soft band assignments from boundary positions; (ii) a band-conditioned vector field that predicts the flow velocity per band; (iii) a boundary-transport term that moves the boundaries along the same flow. The operator is soft: for frequency bin f and band k with boundaries b_{k-1}, b_k, the assignment weight is a smooth indicator w_k(f) = sigmoid((f - b_{k-1})/tau) * sigmoid((b_k - f)/tau), with temperature tau annealed from 8 bins to 1 bin over the first 50% of training (linear schedule), then held at 1 bin. The vector field v_theta(x_t, t, c, {w_k}) is a U-Net (6 down/up blocks, base channels 64, 2 residual blocks per level, self-attention at 16x16 resolution) whose per-band output heads are gated by w_k, so each band has its own capacity. The boundary-transport term is derived from the same flow: the instantaneous frequency velocity at the boundary is the weighted average of the predicted per-bin velocities in the two adjacent bands, and the boundary ODE is db_k/dt = sum_f w'_k(f) * v_theta(x_t)_f / (sum_f w'_k(f) + eps), where w'_k is the boundary-derivative of the assignment weight and eps=1e-6 prevents division by zero. Training minimizes L = E_{t,x_0,x_1} || v_theta(x_t,t) - (x_1 - x_0) ||^2 + lambda * E_{t,x_0,x_1} || db/dt - v_b ||^2 with lambda = 0.1, plus a mixture-consistency penalty on the sum of separated sources (weight 0.05). At inference, the audio ODE and the boundary ODE are integrated jointly with a shared RK4 solver, 32 steps, and the final partition is read out per source.

**Load-bearing assumption (explicit).** The method assumes that the flow-matching regression loss on the audio signal provides a meaningful, non-degenerate gradient that drives the band boundaries to source-adaptive positions, i.e., that the boundary ODE db_k/dt = sum_f w'_k(f) v_theta(x_t)_f / (sum_f w'_k(f) + eps) is a well-posed transport whose fixed points coincide with capacity-optimal partitions. This assumption is load-bearing because the only signal tying boundaries to the audio loss is the lambda-weighted boundary-transport term, whose target v_b is itself defined from v_theta; if the gradient is degenerate, the boundary loss can be driven to zero by a constant partition regardless of audio SDR. To verify this assumption, we add a diagnostic (see experiment) measuring the correlation between w_k and per-band flow-matching residual magnitude on held-out data; if the correlation is below 0.3, the capacity-allocation premise is unsupported and the soft-indicator design must be revised.

**Why this design.** We chose a soft differentiable partition over a hard subband tree (the design removed in branch C) because a hard tree cannot be optimized by the flow-matching regression gradient, accepting the cost that the soft assignment adds a temperature hyperparameter and a small bias near boundaries at high tau. We chose to transport the boundaries with the same ODE rather than learning a separate boundary predictor because a separate predictor would be a second training signal that can disagree with the audio flow, accepting the cost that boundary transport inherits any error in the audio velocity field. We chose per-source partitions over a shared partition because bass and cymbals occupy disjoint spectral regions, accepting the cost that the model must condition the partition on the source index and that multi-source joint training couples the partitions through the mixture-consistency term. A domain expert would not describe this as FLOMSS plus a learned filterbank: FLOMSS has no partition state at all, and a learned filterbank is a fixed front-end, whereas here the partition is a state variable of the generative ODE and is supervised by the same regression target as the audio.

**Why it measures what we claim.** The per-band assignment weight w_k(f) measures the claim that frequency region f needs generative capacity in band k because the soft-indicator assumption makes w_k equivalent to the fraction of the flow-matching residual that band k must explain; this assumption fails when two adjacent bands have nearly equal residual magnitude, in which case w_k reflects boundary temperature rather than true capacity demand. The boundary velocity db_k/dt measures the claim that the partition adapts to the source because the weighted-average assumption makes the boundary move with the local spectral flow; this assumption fails when the velocity field is locally non-smooth across the boundary, in which case db_k/dt reflects solver noise rather than source-adaptive motion. The per-band regression loss measures the claim that capacity is allocated where it is needed because the band-gated head assumption makes the loss on band k equal to the residual left unexplained by the other bands; this assumption fails when bands overlap strongly at high tau, in which case the loss reflects cross-band leakage.

## Contribution

['A flow-matching separator (FTBP) whose frequency-band partition is a state variable of the generative ODE and is trained jointly with the audio vector field by the same regression loss, eliminating the need for a separate subband encoder or hand-designed tree.', 'A differentiable soft band-partition operator with annealed temperature that makes the partition boundaries transportable by the flow, together with a boundary-transport term derived from the per-bin velocity field.', 'The design principle that source-adaptive spectral capacity can be recovered inside a full-band generative separator by making the partition a transported state rather than a fixed front-end, demonstrated per source on MUSDB18-HQ.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | MUSDB18-HQ | Standard multi-source benchmark with real spectral overlap |
| Primary metric | SDR per source | Directly measures separation fidelity per source |
| Baseline | FLOMSS | Same generative family, no partition state |
| Baseline | Demucs | Strong discriminative separator, no adaptive capacity |
| Baseline | Open-Unmix | Fixed-bandwidth baseline, tests capacity allocation |
| Baseline | Learned fixed filterbank front-end (K=8 bands, same U-Net, trained with same flow-matching loss) | Tests whether ODE-transport of boundaries uniquely helps |
| Baseline | Hard subband tree (branch C, K=8 bands, same U-Net, trained with same flow-matching loss) | Tests against removed hand-designed partition |
| Ablation-of-ours | FTBP with shared (non-per-source) partition | Isolates benefit of per-source band adaptation |
| Ablation-of-ours | FTBP with uniform fixed partition (no boundary transport) | Isolates benefit of transported boundaries |
| Diagnostic | Correlation between w_k and per-band flow-matching residual magnitude on held-out data | Tests load-bearing assumption directly |

### Why this setup validates the claim

The central claim is that making the frequency partition a transported state variable of the flow ODE lets generative capacity concentrate where each source actually demands it, which should show up as a separation gain concentrated on sources whose energy is spectrally structured and disjoint from the rest of the mixture. MUSDB18-HQ is the right testbed because it contains real recordings where bass, drums, vocals, and other overlap in ways that a fixed partition cannot resolve, so a capacity-allocation advantage has room to manifest. SDR per source is the right metric because it penalizes exactly the residual that the band-gated heads are supposed to absorb: if band capacity is misallocated, the unexplained residual rises and SDR drops, so the metric is causally tied to the mechanism rather than to perceptual proxies. FLOMSS tests the sub-claim that partition state adds value beyond flow-matching separation itself; Demucs and Open-Unmix test the sub-claim against fixed-bandwidth and discriminative capacity, respectively; the learned fixed filterbank and hard subband tree baselines test whether the ODE-transport of boundaries uniquely buys anything beyond any learned partition at matched parameter count; the shared-partition and uniform-partition ablations test whether the per-source conditioning and the transport are what produce the gain or whether any soft partition would do. The diagnostic correlation directly tests the load-bearing assumption that w_k measures capacity demand; if it is weak (<0.3), the capacity-allocation premise is unsupported. If the gain appears only when the partition is per-source and transported, the claim survives; if the shared or uniform ablation matches the full model, the per-source or transport claim is dead.

### Expected outcome and causal chain

**vs. FLOMSS** — On a case where a sustained bass note and low-frequency kick overlap, FLOMSS with a fixed capacity allocation produces residual bleed between the two sources because its vector field has no mechanism to shift generative capacity toward the contested band. Our method instead transports the band boundary toward the contested region because the boundary ODE follows the local spectral flow, so the band-gated heads reallocate capacity there. We expect a noticeable SDR gap on bass and drums (≥0.5 dB) but near-parity on vocals, whose energy sits in a region neither method contests.

**vs. Demucs** — On a case where a source has a narrow, transient spectral signature (e.g., a cymbal crash buried under dense mid-range), Demucs produces smeared or attenuated transients because its fixed-bandwidth discriminative filters cannot locally increase resolution. Our method instead sharpens the partition around the transient because the boundary velocity responds to the local flow magnitude, giving that band dedicated capacity. We expect a concentrated gain on percussive subsets (≥0.3 dB) and parity on sustained harmonic content.

**vs. Open-Unmix** — On a case where two sources share a broad spectral region but differ in fine structure, Open-Unmix produces cross-source leakage because its fixed basis cannot separate within-region structure. Our method instead splits the shared region into per-source bands because each source index conditions its own partition, so the two partitions can disagree about where the boundary belongs. We expect a modest but consistent gain (≥0.2 dB) on overlapping-region sources and little change elsewhere.

**vs. Learned fixed filterbank front-end** — On a case where the optimal band boundaries differ between bass and cymbals, the fixed filterbank cannot adapt its partition per source or per example, so its capacity allocation is suboptimal for at least one source. Our method transports boundaries per source, so we expect a gain (≥0.3 dB) on sources whose optimal partition deviates most from the fixed filterbank's learned average, and parity elsewhere.

**vs. Hard subband tree (branch C)** — On a case where the hand-designed tree's boundaries fall in the middle of a source's critical band, the hard tree cannot move them, causing capacity misallocation. Our method's soft, transported boundaries can shift to the source-adaptive position, so we expect a gain (≥0.3 dB) on sources whose critical bands are misaligned with the tree, and parity on sources that happen to align.

**vs. FTBP with shared partition** — On a case where bass and cymbals need different boundaries, the shared partition must compromise, allocating capacity suboptimally for both. Our per-source partition lets each source choose its own boundaries, so we expect a gain (≥0.3 dB) on sources with disjoint spectral needs, and parity on sources with similar needs.

**vs. FTBP with uniform fixed partition** — On a case where a source's energy is concentrated in a narrow band, the uniform partition spreads capacity evenly, wasting it on empty regions. Our transported boundaries concentrate capacity where the flow demands it, so we expect a gain (≥0.3 dB) on sources with concentrated spectra, and parity on broadband sources.

**Diagnostic** — We will compute the Pearson correlation between w_k(f) and the per-band flow-matching residual magnitude |v_theta(x_t)_f - (x_1 - x_0)_f| on a held-out set of 512 examples. If the correlation is ≥0.5, the load-bearing assumption is supported; if <0.3, it is unsupported and the soft-indicator design must be revised.

### What would falsify this idea

If the per-source transported partition yields a gain that is uniform across all sources and subsets rather than concentrated where spectral overlap and transient structure predict it, or if the shared-partition or uniform-partition ablation matches the full model, or if the diagnostic correlation between w_k and per-band residual magnitude is below 0.3, then the partition is not doing the capacity-allocation work the claim requires.

## References

1. FLOMSS: Generative Music Source Separation by Flow Matching
2. Multi-Source Diffusion Models for Simultaneous Music Generation and Separation
3. Improving and generalizing flow-based generative models with minibatch optimal transport
4. Flow Matching for Generative Modeling
