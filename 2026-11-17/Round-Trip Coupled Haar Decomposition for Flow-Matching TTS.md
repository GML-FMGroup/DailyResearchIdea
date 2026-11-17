# Round-Trip Coupled Haar Decomposition for Flow-Matching TTS

## Motivation

VoxMel TTS Technical Report decomposes mel spectrograms with a fixed two-axis Haar transform into Coarse-Mel, temporal residual, and frequency residual, then generates each tier with separate predictors (LM+flow head for Coarse-Mel, non-autoregressive heads for residuals) without ever feeding residuals back or enforcing that the generated sub-bands invert to a valid mel spectrogram. This is a structural failure of the decomposition-as-front-end paradigm: because no branch is trained to satisfy the Haar inverse, the generated tiers can drift into a subspace that is not in the range of the forward transform, so reconstruction error is uncharacterized and generation fidelity is unanchored. The same gap recurs across the flow-matching TTS line — Shallow Flow Matching (E1) and DMOSpeech (E5) both train flow heads on coarse conditions without any invertibility constraint tying the generated intermediate states back to the target spectrogram, so the field has no certified front-end guarantee for coarse-to-fine TTS.

## Key Insight

The two-axis Haar transform is an orthogonal involution whose inverse is its own transpose, so a single differentiable round-trip loss on the generated sub-bands is an exact, zero-cost certificate of whether the flow head's output lies in the transform's range — a constraint that pure flow-matching objectives (E1, E5) cannot express because they never apply the inverse.

## Method

**What it is.** We introduce **RCH-VoxMel** (Round-trip Coupled Haar VoxMel), a training framework that augments VoxMel's three-tier generator with an explicit inverse-Haar coupling loss and a Parseval energy-conservation penalty on the flow-matching head, so that every generated tier is trained to lie in the range of the fixed Haar operator. Inputs: mel spectrogram $X \in \mathbb{R}^{F\times T}$, text $c$. Outputs: generated Coarse-Mel $\hat{C}$, temporal residual $\hat{R}_t$, frequency residual $\hat{R}_f$, and a reconstruction $\hat{X} = H^{-1}(\hat{C}, \hat{R}_t, \hat{R}_f)$ that is optimized to match $X$.

**Load-bearing assumption (stated explicitly).** The method assumes that the fixed two-axis Haar transform is an exact orthonormal involution on the actual $F\times T$ mel grid used, so that $H_f^T H_f = I_F$ and $H_t^T H_t = I_T$ to machine precision, and that the residuals $R_f, R_t$ are defined as the orthogonal projections of $X$ and $A_f$ onto the detail subspaces (not as algebraic complements). Under this assumption, a nonzero round-trip error $\|H^{-1}(\hat{C},\hat{R}_t,\hat{R}_f) - X\|_1$ is attributable solely to generated sub-bands leaving the range of the fixed Haar operator. We verify this assumption empirically on the actual grid and report the residual $\|H_f^T H_f - I_F\|_F^2 + \|H_t^T H_t - I_T\|_F^2$ as a calibration number (expected $<10^{-12}$ for dyadic $F,T$; for non-dyadic dimensions we pad $X$ to the next power of two with symmetric extension and crop after inverse, and report the padding-induced reconstruction error on $X$ itself as a separate calibration baseline). If the residuals were instead defined as algebraic complements, the round-trip loss would be trivially zero for any tiers satisfying the complement identities; we therefore use the orthogonal-projection definition below.

**How it works.**

```pseudocode
# Fixed two-axis Haar operators (no learned parameters)
H_f, H_t          # orthonormal Haar matrices along frequency, time (dyadic; pad X if needed)
H_f^{-1} = H_f^T ; H_t^{-1} = H_t^T

# Forward decomposition (targets, computed once per batch)
A_f = H_f^T @ X                    # frequency-averaged mel (low-pass along F)
R_f = X - H_f @ A_f                # frequency residual = projection onto detail subspace
C   = A_f @ H_t^T                  # Coarse-Mel (time-averaged)
R_t = A_f - C @ H_t                # temporal residual = projection onto temporal detail subspace

# Generation
C_hat = LM_flow(c)                 # autoregressive LM + flow head
R_t_hat = NAR_t(C_hat)             # temporal residual tier
R_f_hat = NAR_f(H_f @ (C_hat @ H_t + R_t_hat))   # freq residual tier

# Inverse Haar reconstruction
A_f_hat = C_hat @ H_t + R_t_hat
X_hat   = H_f @ A_f_hat + R_f_hat

# Losses
L_flow  = flow_matching(C_hat, C) + flow_matching(R_t_hat, R_t)
L_rt    = || X_hat - X ||_1 + || H_f^T X_hat - A_f_hat ||_2^2   # round-trip
L_energy= | ||X||_2^2 - (||C_hat||_2^2 + ||R_t_hat||_2^2 + ||R_f_hat||_2^2) |  # Parseval
L = L_flow + λ_rt * L_rt + λ_energy * L_energy
# λ_rt = 0.5, λ_energy = 0.1 (tuned on dev); L_orth dropped (constant for fixed orthonormal H)
```

**Why this design.** We chose an *explicit inverse-transform loss* over a learned discriminator or a post-hoc correction because the Haar operator is orthonormal and its inverse is exactly $H^T$, so the round-trip is a deterministic, differentiable function of the generated tiers — accepting the cost that the loss is only as informative as the fixed transform's range, which is a design choice we make deliberately rather than learning the transform. We chose to penalize *Parseval energy* separately from the round-trip $L_1$ because a pure reconstruction loss can be satisfied by compensating errors across tiers (a large $R_f$ offsetting a small $C$ error), whereas the energy term forces the tier set to individually respect the transform's norm invariant; the trade-off is that this penalty adds one hyperparameter and can slow early training. We dropped the orthogonality penalty $\|H^TH-I\|_F^2$ because for a fixed orthonormal Haar matrix it is identically zero and carries no gradient to any learned parameter; instead we verify $H^TH=I$ to machine precision as a calibration number. We chose to apply the round-trip loss to *all three tiers jointly* rather than only to Coarse-Mel because VoxMel's residual tiers are non-autoregressive and never see feedback, so the only way to anchor them to the transform's range is through the shared reconstruction; this differs from Shallow Flow Matching (E1), which trains its flow head on coarse conditions with no inverse-Haar coupling, and from DMOSpeech (E5), which optimizes perceptual metrics but never applies the transform inverse — a domain expert would not describe RCH-VoxMel as a variant of either because the load-bearing signal is the *inverse operator applied to generated sub-bands*, not a better flow objective or a distillation target. We chose a fixed Haar basis over a learned invertible transform because the fixed basis gives a closed-form inverse (no separate invertibility network), accepting the cost that the decomposition cannot adapt to speech statistics.

**Why it measures what we claim.** The round-trip $L_1$ $\|\hat{X}-X\|_1$ measures *reconstruction-consistency closure* because the Haar inverse is exact on the padded dyadic grid, so any nonzero value is attributable solely to generated sub-bands leaving the transform's range; this assumption fails when the target $X$ itself is not exactly representable (e.g., mel filterbank edge effects or padding-induced truncation), in which case the loss reflects transform truncation rather than generation drift — we report the padding-induced reconstruction error on $X$ as a calibration baseline to bound this. The Parseval energy term $|\|X\|^2 - \sum\|\text{tier}\|^2|$ measures *energy conservation across tiers* because orthonormal decomposition preserves the $\ell_2$ norm exactly; this assumption fails when the flow head's output distribution has heavier tails than the target (e.g., during early training), in which case the energy term reflects distributional mismatch rather than a transform violation. The calibration number $\|H_f^T H_f - I_F\|_F^2 + \|H_t^T H_t - I_T\|_F^2$ measures *Haar orthogonality* because the Frobenius norm of the deviation from identity is zero iff the operator is orthonormal; this assumption fails when numerical precision in the matrix multiply introduces $O(\epsilon)$ residuals, in which case the number reflects floating-point noise rather than a structural violation — we report it as a diagnostic, not a loss.

## Contribution

["A round-trip coupled training framework (RCH-VoxMel) that adds an explicit inverse-Haar reconstruction loss and orthogonality/Parseval penalties to VoxMel's three-tier generator, making every generated sub-band trainable against the transform's range without changing the inference architecture.", 'A design principle for coarse-to-fine TTS front-ends: orthonormal decompositions admit exact differentiable round-trip certificates, so reconstruction consistency can be enforced as a training signal rather than checked post-hoc — a property that flow-matching TTS heads (E1, E5) currently lack.', 'An empirical characterization of how round-trip and energy penalties trade off against flow-matching loss during training, identifying the regime where reconstruction closure improves generation fidelity versus where it over-constrains the flow head.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | VoxMel-style multi-speaker TTS benchmark (single public corpus, e.g., LibriTTS train-clean-100/360, 80-bin mel, 22.05 kHz) | Matches the three-tier generator being augmented |
| Primary metric | Reconstruction-consistency error on held-out mel: $\|\hat{X}-X\|_1$ and $\|H_f^T\hat{X}-\hat{A}_f\|_2^2$, reported separately for short (≤3 s) and long (≥10 s) utterances | Directly measures inverse-Haar closure claim |
| Secondary metric | UTMOS on long utterances (≥10 s) and CMOS on 50 paired long utterances | Ties closure to perceptual target |
| Diagnostic metric | Out-of-range residual magnitude of generated tiers on a trained VoxMel baseline: $\|(I-H_fH_f^T)\hat{R}_f\|_F$ and $\|(I-H_tH_t^T)\hat{R}_t\|_F$ | Empirically establishes drift premise before claiming coupling addresses it |
| Baseline | VoxMel (no coupling loss) | Tests whether inverse-Haar coupling adds anything |
| Baseline | Shallow Flow Matching head (E1) | Tests whether coupling beats better flow objectives |
| Baseline | DMOSpeech-style perceptual distillation (E5) | Tests whether metric optimization substitutes for coupling |
| Ablation-of-ours | RCH-VoxMel without L_energy | Isolates the energy-penalty contribution |
| Ablation-of-ours | RCH-VoxMel with algebraic-complement residuals (original pseudocode) | Tests whether orthogonal-projection definition is load-bearing |

### Why this setup validates the claim

The central claim is that an explicit inverse-Haar round-trip loss, coupled with a Parseval energy penalty, anchors each generated tier to the range of the fixed Haar operator and thereby improves reconstruction consistency without degrading naturalness. The dataset must be one where VoxMel's coarse-to-fine decomposition is already defined, so the only variable is the coupling loss; we scope to a single public multi-speaker corpus and a single reproduced VoxMel baseline first, pre-registering the long-vs-short utterance split (≤3 s vs ≥10 s) so the falsification test is runnable without re-implementing full SFM and DMOSpeech systems. The primary metric — reconstruction-consistency error between $\hat{X}$ and $X$ on held-out mel, reported separately for short and long utterances — is the right detector because, under an exact orthonormal inverse on the padded dyadic grid, any residual is attributable to generated sub-bands leaving the transform's range; a metric like MOS alone would conflate reconstruction drift with perceptual preference and could not distinguish our mechanism from a better flow objective. The diagnostic metric on a trained VoxMel baseline measures the actual out-of-range residual magnitude of generated tiers (projection onto the orthogonal complement of the Haar range) to empirically establish the drift premise before claiming the coupling loss addresses it. VoxMel-no-coupling tests the base sub-claim that coupling adds signal beyond the generator. Shallow Flow Matching (E1) tests whether a superior flow objective alone, without inverse coupling, captures the same gain — if it does, our load-bearing signal is redundant. DMOSpeech (E5) tests whether direct perceptual-metric optimization substitutes for transform-range anchoring. The ablation without L_energy isolates whether the energy penalty, rather than the round-trip L1 alone, is what forces tier-set norm compliance, since a pure reconstruction loss can be satisfied by compensating cross-tier errors. The ablation with algebraic-complement residuals tests whether the orthogonal-projection definition is load-bearing, since with complements the round-trip loss is trivially zero for any tiers satisfying the complement identities.

### Expected outcome and causal chain

**vs. VoxMel (no coupling loss)** — On a case where the non-autoregressive residual tiers drift from the transform's range (e.g., a long utterance where $\hat{R}_f$ accumulates a low-frequency offset), the baseline produces a reconstruction $\hat{X}$ whose inverse-Haar image is inconsistent with $\hat{A}_f$ because its residual tiers never receive feedback. Our method instead applies the round-trip L1 and the Parseval penalty jointly across all three tiers, so compensating cross-tier errors are penalized. We expect a clear reconstruction-error gap on long utterances (≥10 s) and near-parity on short ones (≤3 s), and a positive UTMOS/CMOS gap on long utterances.

**vs. Shallow Flow Matching head (E1)** — On a case where the flow head's intermediate state is initialized from coarse conditions that are themselves slightly off-range, the baseline produces a natural-sounding but range-violating coarse tier because SFM improves the flow path without ever applying the inverse operator to generated sub-bands. Our method instead couples the flow output to the inverse-Haar reconstruction, so coarse-tier drift is directly penalized. We expect our gain to be concentrated in reconstruction consistency, with SFM possibly matching or slightly beating us on naturalness.

**vs. DMOSpeech-style perceptual distillation (E5)** — On a case where perceptual metrics favor a plausible but energy-inconsistent decomposition (e.g., a tier pair whose summed energy exceeds $\|X\|^2$), the baseline produces high-MOS output whose tiers violate Parseval because its objective never constrains the transform invariants. Our method instead adds the explicit energy term, so such decompositions are penalized. We expect a visible energy-conservation gap on high-variance utterances while MOS remains comparable.

**vs. RCH-VoxMel without L_energy** — On a case where compensating cross-tier errors satisfy the round-trip L1 (e.g., a large $\hat{R}_f$ offsetting a small $\hat{C}$ error), the ablation produces low reconstruction L1 but a tier-set whose summed energy deviates from $\|X\|^2$. Our method instead penalizes the energy deviation directly, so we expect a smaller energy-conservation gap with comparable reconstruction L1.

**vs. RCH-VoxMel with algebraic-complement residuals** — On any utterance, the algebraic-complement variant produces a round-trip loss that is trivially zero for any tiers satisfying the complement identities, so its reconstruction L1 is not a certificate of range compliance. Our method instead uses orthogonal-projection residuals, so the round-trip loss is nonzero exactly when generated coefficients leave the transform's range. We expect the complement variant to show near-zero training L_rt but no reconstruction-consistency gain over VoxMel-no-coupling on held-out mel.

### What would falsify this idea

If the reconstruction-consistency gain over VoxMel-no-coupling is uniform across short and long utterances and across high- and low-variance subsets, rather than concentrated where residual-tier drift is predicted, the claim that inverse-Haar coupling anchors the tiers is wrong. Likewise, if removing L_energy leaves the gain intact, the energy penalty is not the load-bearing signal; and if the algebraic-complement variant matches our reconstruction-consistency gain, the orthogonal-projection definition is not load-bearing.

## References

1. VoxMel TTS Technical Report
