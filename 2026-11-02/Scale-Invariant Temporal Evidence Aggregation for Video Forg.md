# Scale-Invariant Temporal Evidence Aggregation for Video Forgery Detection

## Motivation

Current video forgery detectors, including the unified Venus-DefakerOne2 foundation model, assume a fixed temporal window when accumulating spatial-temporal anomaly evidence, so a single forgery artifact that is physically consistent at one frame rate or clip length can appear inconsistent—or vanish—at another. This failure is structural: a foundation model trained on static images (as Venus-DefakerOne2 is) has no mechanism to certify that the same forgery evidence persists across temporal scales, so the fixed window silently encodes the training distribution's scale rather than the forgery's intrinsic temporal signature. The field's trajectory toward unified forensic foundation models makes this bottleneck shared across branches (optical-flow PAD, spectral augmentation, spatial-temporal anomaly), since each inherits the same fixed-window assumption when ported to video.

## Key Insight

Forgery evidence that is causally attributable to manipulation must remain rank-consistent across nested temporal windows, so a learned attention over window-wise evidence scores—trained with a scale-consistency objective—recovers a temporal-scale-invariant forgery signature rather than memorizing the training window length.

## Method

**A. What it is.** We introduce **STAIN** (Scale-invariant Temporal Aggregation of Invariant-evidence Networks), a drop-in head replacing the fixed temporal window of Tide-Style-Tuning in Venus-DefakerOne2. Input: a video clip's per-frame spatial-forensic feature sequence from the frozen backbone; output: a single forgery logit plus the aggregated multi-scale evidence vector.

**B. How it works.**
```pseudocode
# Given: backbone features F[1..T, D] for a clip of T frames
# Hyperparams: window set W = {2,4,8,16,32} (frames), temperature tau=0.1,
#              consistency weight lambda=0.5, entropy-adapt threshold rho=0.7
#              (pre-registered via validation sweep over W in {2,4,8,16,32} and lambda in {0.1,0.5,1.0})

1. For each w in W:
     E_w[t] = ForgeryScorer_w( mean_pool(F[t-w/2 : t+w/2]) )   # scalar evidence per center frame
     s_w    = Readout_w( E_w )                                  # window-level forgery score
     # Codec-noise baseline estimation (per window, from calibration clips):
     #   b_w[t] = mean over C=512 pristine/recompressed calibration clips of E_w[t]
     #   (calibration clips are drawn from the same benchmark suite, held out from training)
     #   E_w^res[t] = E_w[t] - b_w[t]   # noise-residualized evidence
2. Attention aggregation:
     a_w = softmax_w( MLP( [s_w, Var_t(E_w^res), w] ) / tau )    # learned scale attention
     s_agg = sum_w a_w * s_w
3. Scale-consistency objective (training):
     # Applied to noise-residualized evidence distributions:
     L_cons = mean_{w<w'} KL( softmax(E_w^res) || softmax(E_{w'}^res) )
     L = BCE(s_agg, y) + lambda * L_cons
4. Entropy-based Test Adaptation (inherited):
     if entropy(softmax([s_agg, s_w...])) > rho: pick argmax_w s_w via held-out
     else: output s_agg
5. Return s_agg (or selected s_w) and evidence vector {a_w, E_w^res}
```

**C. Why this design.** We chose a *nested* window set W={2,4,8,16,32} over a learned continuous scale because nested windows guarantee that E_w and E_{w'} share center frames, making the KL consistency term a well-posed constraint rather than a comparison of misaligned evidence—accepting the cost that the largest window bounds the detectable temporal extent. We chose a learned attention over window scores (with variance and window size as inputs) rather than uniform averaging because forgery artifacts at different scales have different reliability (e.g., compression artifacts dominate small windows), accepting the cost of extra parameters that must be regularized by the consistency loss. We chose the KL consistency objective over a contrastive or adversarial invariance loss because KL directly penalizes rank divergence between window evidence distributions, which is exactly the scale-invariance property we need, accepting the cost that it can over-smooth genuinely scale-dependent forgeries (e.g., frame-rate-dependent flicker). A domain expert would *not* describe this as a variant of the fixed-window spatial-temporal anomaly detectors used in Venus-DefakerOne2's branches, because those detectors score evidence within one window and never enforce cross-window rank consistency; STAIN's load-bearing component is the consistency constraint, not the aggregator. To guard against the confound that nested windows share correlated codec noise, we residualize evidence against a per-window compression-noise baseline estimated from C=512 calibration clips before computing both the attention variance and the KL term; this makes L_cons a constraint on the manipulation-attributable component rather than on shared-window artifact noise.

**D. Why it measures what we claim.** The attention weight a_w measures *scale reliability* of forgery evidence because we assume that a window whose noise-residualized evidence distribution is rank-consistent with other windows is more likely to reflect the forgery rather than window-specific noise; this assumption fails when a forgery is genuinely scale-localized (e.g., a single-frame splice), in which case a_w reflects window-specific artifact saliency rather than forgery attribution. The KL divergence L_cons measures *temporal-scale invariance* because we assume that rank-preserving noise-residualized evidence across nested windows is equivalent to a scale-invariant forgery signature; this assumption fails when nested windows share correlated compression noise that survives residualization (e.g., under heavy recompression), in which case L_cons reflects shared codec artifacts rather than forgery persistence. The variance term Var_t(E_w^res) measures *evidence stability within a window* because we assume that stable per-frame residualized evidence indicates a persistent manipulation rather than transient motion; this assumption fails under fast camera motion, in which case Var_t reflects motion-induced feature drift rather than forgery instability.

## Contribution

(1) STAIN, a scale-invariant temporal evidence aggregator that replaces fixed-window anomaly scoring with nested-window evidence and a rank-consistency training objective, integrable into Tide-Style-Tuning. (2) The finding that cross-window rank consistency is a necessary training signal—not merely an aggregation choice—for temporal-scale invariance in video forgery detection. (3) An entropy-gated test-time selection rule over window-wise scores that degrades gracefully when the largest window exceeds the clip length.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | Venus-DefakerOne2 forensic benchmark suite | Spans forgery forms and media types |
| Primary metric | AUC on forged-vs-pristine clips | Threshold-free, sensitive to rank shifts |
| Baseline | Venus-DefakerOne2 fixed-window head | Isolates our head-replacement claim |
| Baseline | Uniform multi-window averaging | Tests learned attention necessity |
| Baseline | Single-scale scorer (w=8 only) | Tests cross-window consistency necessity |
| Baseline | Temporal pyramid with feature-level consistency loss (multi-scale detector) | Tests whether rank-consistency on evidence distributions, not feature-level consistency, drives the gain |
| Ablation-of-ours | STAIN minus KL consistency loss | Isolates load-bearing component |
| Ablation-of-ours | STAIN with raw-evidence KL (no codec-noise residualization) | Tests whether residualization is necessary to avoid shared codec noise |
| Ablation-of-ours | STAIN with KL restricted to non-overlapping window pairs | Tests whether shared center-frame codec noise trivially satisfies the constraint |
| Ablation-of-ours | STAIN minus KL, evaluated on high-compression subset | Compression-stratified test of whether the gain survives on high-compression subsets |
| Controlled injection | Synthetic experiment: known forgery injected at controlled temporal extents (2, 4, 8, 16, 32 frames) and frame rates (15, 30, 60 fps) | Grounds the KL-to-invariance claim in a controlled causal manipulation |
| Transfer | STAIN head applied to two additional frozen backbones: a video-native model and a second image foundation model | Establishes backbone-agnosticism of the scale-invariance principle |

### Why this setup validates the claim

The central claim is that cross-window rank consistency, not the aggregator, is what makes scale-invariance work. The design tests this by holding the backbone frozen and varying only the head, so any AUC difference is attributable to the temporal-aggregation logic rather than representation learning. The fixed-window baseline tests whether multi-scale windows help at all; the uniform-averaging baseline tests whether learned scale attention adds anything beyond naive pooling; the single-scale baseline tests whether the nested-window structure matters when consistency is removed. The temporal-pyramid-with-feature-level-consistency baseline tests whether STAIN's rank-consistency on evidence distributions, rather than feature-level consistency, is what drives the gain—if the feature-level baseline matches STAIN, the novelty collapses into known ensembling. The KL ablation is the decisive comparison: if removing L_cons collapses performance to the uniform-averaging level, the consistency constraint is confirmed as load-bearing rather than decorative. The raw-evidence KL ablation and the non-overlapping-window KL ablation test whether the codec-noise residualization and the nested-window design are necessary to prevent shared codec noise from trivially satisfying the constraint. The compression-stratified ablation tests whether the gain survives on high-compression subsets, where shared blocking artifacts are most likely to dominate. The controlled injection experiment directly measures whether L_cons decreases as the injected signature becomes scale-invariant, grounding the KL-to-invariance claim in a causal manipulation rather than observational AUC. The transfer experiment tests whether the scale-invariance principle is backbone-agnostic. AUC is the right metric because the predicted effect is a rank-reordering of forged vs. pristine clips under scale-mismatched forgeries, which threshold-tuned accuracy would mask.

### Expected outcome and causal chain

**vs. Venus-DefakerOne2 fixed-window head** — On a case where a splice persists across 20+ frames but the fixed window is 8 frames, the baseline produces a confident score from a sub-window that clips the manipulation boundary because its scorer only sees evidence within one window and cannot reconcile boundary-truncated with boundary-inclusive views. Our method instead aggregates over nested windows {2,4,8,16,32} with KL consistency forcing the boundary-truncated and boundary-inclusive noise-residualized evidence distributions to agree, so we expect a noticeable AUC gap concentrated on long-duration forgeries and near-parity on short splices where the fixed window already suffices.

**vs. Uniform multi-window averaging** — On a case where small windows are dominated by codec blocking artifacts and large windows by genuine tampering, the baseline produces a diluted score because equal weighting lets artifact-heavy small windows drag the average toward pristine. Our method instead learns a_w from variance and window size, down-weighting high-variance artifact-dominated windows, so we expect a clear gain on compression-heavy subsets and parity on clean, low-compression subsets where uniform averaging is already unbiased.

**vs. Single-scale scorer (w=8 only)** — On a case where forgery evidence only stabilizes at w=32 (slow, persistent manipulation) or only at w=2 (single-frame splice), the baseline produces a near-chance score because a single scale cannot span both regimes. Our method instead selects or aggregates across scales via entropy-triggered adaptation, so we expect a bimodal gain pattern — improvement concentrated on the extreme-scale subsets and flat elsewhere — which is exactly the signature of scale coverage rather than general capacity.

**vs. Temporal pyramid with feature-level consistency loss** — On a case where forgery evidence is rank-consistent across scales but feature-level representations are not (e.g., due to backbone feature drift under compression), the baseline produces a weaker score because its consistency loss operates on features rather than on evidence distributions. Our method instead enforces rank consistency on noise-residualized evidence distributions, so we expect STAIN to outperform the feature-level baseline on compression-heavy subsets and on long-duration forgeries, with parity on clean subsets where feature-level consistency is already sufficient.

**vs. STAIN minus KL consistency loss** — On a case where forgery evidence is scale-invariant, the ablation produces a less stable score because without L_cons the attention weights are free to overfit to window-specific noise. Our method instead enforces cross-window rank consistency on residualized evidence, so we expect a clear AUC gap concentrated on scale-mismatched forgeries, with parity on cases where a single window already captures the forgery.

**vs. STAIN with raw-evidence KL (no codec-noise residualization)** — On a high-compression subset, the raw-evidence KL baseline produces a spuriously high consistency because shared codec noise trivially satisfies the KL constraint, leading to overconfident but artifact-driven scores. Our method instead residualizes evidence against a per-window compression-noise baseline, so we expect STAIN to outperform the raw-evidence KL baseline specifically on high-compression subsets, with parity on low-compression subsets where codec noise is negligible.

**vs. STAIN with KL restricted to non-overlapping window pairs** — On a case where nested windows share center-frame codec noise, the non-overlapping KL baseline avoids trivial satisfaction but loses the well-posedness of comparing aligned evidence. Our method instead uses nested windows with residualization, so we expect STAIN to outperform the non-overlapping baseline on long-duration forgeries where aligned evidence across scales is informative, with parity on short splices.

**vs. STAIN minus KL, high-compression subset** — On a high-compression subset, removing L_cons should not change performance if the gain is artifact-driven, but should degrade performance if the gain is forgery-driven. We expect the full STAIN to outperform the ablation on high-compression subsets, confirming that the consistency constraint captures forgery persistence rather than shared codec noise.

**vs. Controlled injection experiment** — We expect L_cons to decrease monotonically as the injected forgery's temporal extent increases and as frame rate increases (making the signature more scale-invariant), and we expect this decrease to be absent or weaker for the raw-evidence KL variant, grounding the KL-to-invariance claim.

**vs. Transfer to additional backbones** — We expect STAIN to improve over the fixed-window head on both additional frozen backbones, with gains concentrated on long-duration and scale-mismatched forgeries, establishing that the scale-invariance principle is backbone-agnostic.

### What would falsify this idea

If removing the KL consistency loss leaves AUC essentially unchanged relative to full STAIN, or if STAIN's gain over the fixed-window baseline is uniform across all subsets rather than concentrated on long-duration and scale-mismatched forgeries, or if the raw-evidence KL variant matches STAIN on high-compression subsets, then the central claim that cross-window rank consistency is load-bearing is wrong.

## References

1. Venus-DefakerOne2: A Vision-Forensics Foundation Model with Broader Coverage
