# Modular Shortcut Auditing for Audio-Visual Reasoning Benchmarks

## Motivation

Evidence-grounded audio-visual benchmarks such as EvaOmni assume that human-annotated evidence plus video-based distractors suppress all shortcuts, yet they provide no systematic, training-free way to verify this assumption, so model-specific shortcuts can survive undetected and high scores cannot be attributed to joint audio-visual reasoning. The root cause is structural: a benchmark that only reports a single full-model score has no reference point separating genuine cross-modal integration from unimodal competence or shallow temporal alignment, and the full multimodal model's own score cannot serve as that reference because it is exactly the quantity under suspicion.

## Key Insight

If a training-free composition of unimodal and alignment-only modules reproduces the full multimodal model's score on a capability, then that capability's score is fully explained by the modular components' information channels, so any residual advantage of the full model is the only evidence of genuine joint reasoning.

## Method

**What it is.** MOSAIC (Modular Shortcut Auditor for Integrated Capabilities) is a training-free diagnostic harness that takes an existing audio-visual benchmark (EvaOmni-style QA pairs with evidence annotations) and a target full multimodal model M, and outputs a per-capability shortcut profile: for each of the 15 capabilities, the score of M alongside the scores of three internal counterfactual ablations of M itself (audio-only sub-path, visual-only sub-path, alignment-only sub-path) and a derived shortcut gap. No parameters are trained; all components are frozen pretrained models. The load-bearing assumption is stated explicitly: the internal counterfactual envelope `env[c] = max(acc_A^M[c], acc_V^M[c], acc_T^M[c])` computed by ablating M's own cross-modal pathway is an upper bound on what M could achieve without genuine cross-modal integration, because it uses M's own unimodal sub-path rather than an independently trained unimodal model. This assumption fails when M's unimodal sub-path is itself weaker than an external frozen unimodal baseline (e.g., because joint training degraded a modality), in which case the envelope is a lower bound and gap[c] conflates fusion with M's internal unimodal weakness; we therefore report external frozen baselines only as a secondary sanity check and never use them to compute gap[c] or flag[c].

**How it works.**

```pseudocode
Input: benchmark B = {(v_i, q_i, a_i, E_audio_i, E_visual_i)}_{i=1..N}
       full model M; frozen external audio-only model A_ext; frozen external visual-only model V_ext
       frozen temporal-alignment scorer T (e.g. CLAP-style audio-visual sync head)
       capability labels c(q_i) in {1..15}
Output: per-capability shortcut profile

# Phase 1: build three internal counterfactual answerers by ablating M's own cross-modal pathway
for each item i:
    # (a) audio-only sub-path of M: mask visual tokens at M's fusion layers to a matched-distribution null
    y_A[i]  = M(v_i.audio, q_i, visual_tokens := null_visual)   # null_visual = per-dimension mean over dataset visual tokens
    # (b) visual-only sub-path of M: mask audio tokens at M's fusion layers to a matched-distribution null
    y_V[i]  = M(v_i.frames, q_i, audio_tokens := null_audio)    # null_audio = per-dimension mean over dataset audio tokens
    # (c) alignment-only sub-path of M: replace content tokens with sync scalar, no content semantics
    s_T[i]  = T(v_i.audio, v_i.frames)                          # scalar sync score per item
    y_T[i]  = answer selected by nearest neighbor over s_T[i] within the
              candidate answer set of q_i (tie broken by shortest answer)

# Phase 1b: external frozen baselines as secondary sanity check (NOT used in gap/flag)
for each item i:
    y_A_ext[i] = A_ext(v_i.audio, q_i)   # visual input masked to silence/blank
    y_V_ext[i] = V_ext(v_i.frames, q_i)  # audio input masked to silence

# Phase 2: score everything on the same items
for each capability c:
    acc_M[c]     = mean_i[ c(q_i)=c ] 1[ y_M[i] == a_i ]
    acc_A[c], acc_V[c], acc_T[c] analogously from y_A, y_V, y_T (internal ablations)
    acc_A_ext[c], acc_V_ext[c] analogously from external baselines (sanity check only)
    # modular envelope: best achievable by M without joint reasoning
    env[c]   = max(acc_A[c], acc_V[c], acc_T[c])
    # shortcut gap: how much the full model exceeds its own non-joint envelope
    gap[c]   = acc_M[c] - env[c]
    # shortcut flag: capability is suspicious if the envelope already matches M
    flag[c]  = 1[ gap[c] <= tau ]                # tau = 0.02 (2 points)
    # sanity check: external baselines should not exceed internal envelope by more than delta
    sanity[c] = 1[ max(acc_A_ext[c], acc_V_ext[c]) <= env[c] + delta ]  # delta = 0.05

# Phase 3: evidence-consistency check via model-agnostic occlusion (no attention required)
for each item i where y_M[i] == a_i:
    # occlude annotated audio span and measure answer flip
    y_M_occ_audio[i] = M(v_i with E_audio_i replaced by silence, q_i)
    flip_audio[i]    = 1[ y_M_occ_audio[i] != y_M[i] ]
    # occlude annotated visual span and measure answer flip
    y_M_occ_visual[i] = M(v_i with E_visual_i replaced by black frames, q_i)
    flip_visual[i]    = 1[ y_M_occ_visual[i] != y_M[i] ]
    joint_evidence[i] = 1[ flip_audio[i] == 1 and flip_visual[i] == 1 ]

# Phase 4: report
return {acc_M[c], env[c], gap[c], flag[c], mean_joint_evidence[c], sanity[c]}_c
```

Hyperparameters: `tau = 0.02` (shortcut flag threshold, chosen to sit inside the binomial standard error at N≈68 items per capability; pre-registered against a binomial power calculation: with N=68 and p=0.5, SE≈0.061, so tau=0.02 is conservative and flags only when gap is statistically indistinguishable from zero); `delta = 0.05` (external-baseline sanity tolerance, chosen as one binomial SE at N≈68); occlusion uses the annotated evidence spans directly (no IoU threshold needed since spans are replaced wholesale); the alignment-only nearest-neighbor tie-break is deterministic (shortest answer) to keep the baseline training-free and reproducible.

**Why this design.** We chose an internal counterfactual envelope over external frozen unimodal baselines because the literature on modality gaps (Gat et al., CVPR 2024; Frank et al., EMNLP 2021) shows that a jointly trained multimodal model's unimodal sub-path is typically stronger than an independently trained unimodal model, so external baselines are not upper bounds on M's non-joint competence; the internal ablation makes env[c] an upper bound by construction, accepting the cost that it requires M to expose a maskable fusion interface. We chose `max` over the three internal ablation scores rather than an average because the question is whether *any* single non-joint channel suffices to explain M's score, and averaging would dilute a dominant shortcut channel, accepting the cost that the envelope is optimistic and thus conservative about flagging. We chose the alignment-only path scored by nearest-neighbor over a sync scalar rather than a learned temporal-reasoning head because a learned head would reintroduce the joint reasoning we are trying to rule out, accepting the cost that the alignment baseline is weak and will rarely be the binding envelope term. We included the occlusion-based evidence-consistency check because a capability can pass the envelope test while M still answers via a non-evidence cue, and occlusion is model-agnostic (no attention exposure required), accepting the cost of two extra forward passes per correct item. A domain expert would not describe this as a variant of Toolformer-style retrieval or a router/controller system: there is no learned selection among modules and no retrieval library — the method computes a *counterfactual envelope* (what M itself could score with only channel X) and compares it against M, which is an intervention on the information available rather than a nearest-neighbor lookup.

**Why it measures what we claim.** The quantity `gap[c] = acc_M[c] - max(acc_A[c], acc_V[c], acc_T[c])` measures *genuine joint-reasoning advantage* because it assumes that if M's score is fully attainable by M's own single-modality or alignment sub-path, then M's cross-modal parameters add no information beyond that channel; this assumption fails when M's advantage comes from better unimodal extraction than its own ablated sub-path (e.g., the ablation damages representations), in which case gap reflects ablation damage rather than joint reasoning. The quantity `flag[c] = 1[gap[c] <= tau]` measures *shortcut reliance* because it assumes that M's own non-joint sub-path matching M implies M is using the same information channel; this assumption fails when the ablation is imperfect (e.g., null tokens leak information), in which case the flag reflects ablation leakage rather than shortcut use. The quantity `joint_evidence[c]` measures *evidence-grounded reasoning* because it assumes that occluding both annotated evidence sets flips M's answer implies M used both modalities; this assumption fails when occlusion creates out-of-distribution inputs that flip answers for spurious reasons, in which case it reflects occlusion sensitivity rather than causal evidence use. The quantity `sanity[c]` measures *envelope validity* because it assumes that external frozen baselines should not exceed the internal envelope by more than delta; this assumption fails when the external baseline is a stronger unimodal model than M's internal encoder, in which case sanity[c]=0 signals that gap[c] may be confounded by M's internal unimodal weakness and should be interpreted with caution.

## Contribution

(1) MOSAIC, a training-free modular diagnostic harness that produces a per-capability shortcut profile by comparing a full multimodal model against a max-envelope of frozen audio-only, visual-only, and temporal-alignment baselines. (2) The shortcut-gap design principle: capabilities where the modular envelope matches the full model are flagged as explained by non-joint channels, giving a capability-level rather than benchmark-level attribution of scores to joint reasoning. (3) An evidence-consistency check that couples the envelope test with EvaOmni's human-annotated audio and visual evidence, yielding a joint-evidence rate per capability as a second, independent shortcut signal.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | EvaOmni (15 capabilities, 3 reasoning levels) | Per-capability labels enable shortcut profiling |
| Primary metric | Shortcut gap `gap[c] = acc_M[c] − max(acc_A, acc_V, acc_T)` where acc_A, acc_V, acc_T are from internal ablations of M | Directly quantifies joint-reasoning advantage per capability without cross-model confound |
| Baseline (internal envelope) | Audio-only sub-path of M (visual tokens masked to null) | Tests whether M's audio channel alone explains M |
| Baseline (internal envelope) | Visual-only sub-path of M (audio tokens masked to null) | Tests whether M's visual channel alone explains M |
| Baseline (internal envelope) | Alignment-only sync scorer T (CLAP-style NN) | Tests whether sync cue alone explains M |
| Secondary sanity check | External frozen audio-only model A_ext | Checks whether external baseline exceeds internal envelope (sanity[c]) |
| Secondary sanity check | External frozen visual-only model V_ext | Checks whether external baseline exceeds internal envelope (sanity[c]) |
| Ablation-of-ours | Envelope with `mean` instead of `max` | Tests whether dilution masks a dominant shortcut |
| Calibration | Known-shortcut model M_shortcut (fine-tuned to answer from transcript leakage) injected into benchmark | Verifies gap[c] and flag[c] fire on exactly the injected capabilities |
| Control | Stronger external unimodal model swapped in for A_ext/V_ext | Tests whether flags change when external baseline strength changes |

### Why this setup validates the claim

The central claim is that a full multimodal model's score on a capability is often attainable by its own single non-joint channel, and that the residual `gap[c]` isolates genuine cross-modal reasoning. Each internal ablation tests one sub-claim: the audio-only sub-path asks whether M's visual stream is load-bearing, the visual-only sub-path asks whether M's audio stream is load-bearing, and the alignment-only path asks whether temporal synchrony without content semantics suffices. Because all three are computed from M itself with frozen weights, they cannot "learn their way" to the answer, and the envelope is an upper bound on M's non-joint competence by construction rather than by cross-model comparison. The external frozen baselines are retained only as a sanity check: if `sanity[c]=0` (external baseline exceeds internal envelope by more than delta=0.05), we flag that gap[c] may be confounded by M's internal unimodal weakness and interpret the capability with caution. The `mean`-vs-`max` ablation is the falsification lever: if switching to `mean` collapses flags, the effect was dilution, not shortcut dominance. The known-shortcut injection calibration establishes that the envelope detects a ground-truth shortcut rather than merely correlating with baseline strength: M_shortcut is fine-tuned on a synthetic transcript-leakage task for the injected capabilities, and we verify that flag[c]=1 on exactly those capabilities and flag[c]=0 elsewhere. The stronger-unimodal control swaps in a stronger external model for A_ext/V_ext; if flags change, the envelope is sensitive to external baseline strength and the internal ablation is doing the diagnostic work.

### Expected outcome and causal chain

**vs. Audio-only sub-path of M** — On a capability like "visual-object-identification-under-speech," the audio-only sub-path sees only the audio tokens and answers from speech content, producing the correct answer whenever the spoken words leak the label, because its mechanism has no access to frames but the answer is recoverable from audio alone. Our method computes `acc_A` on the same items and finds `env[c] ≥ acc_M[c]`, because M's score is fully explained by its own audio channel; we expect a near-zero gap concentrated on capabilities where audio semantics dominate, with the flag firing on roughly the subset of capabilities whose answers are audio-recoverable and parity elsewhere.

**vs. Visual-only sub-path of M** — On a capability like "off-screen-sound-source-localization," the visual-only sub-path sees frames with muted audio and answers from visible scene layout, producing the correct answer when the source is on-screen, because its mechanism lacks any acoustic cue yet the visual context alone disambiguates. Our method's envelope absorbs this: `env[c]` matches `acc_M[c]` on on-screen items, so the gap is near zero, and we expect the flag to fire on the on-screen-heavy capabilities while the gap re-emerges on capabilities requiring off-screen acoustic evidence — a split pattern, not a uniform one.

**vs. Alignment-only sync scorer T** — On a capability like "temporal-event-ordering," the alignment-only baseline uses only the scalar audio-visual sync score and picks the nearest-neighbor answer, producing the correct answer whenever event order correlates with sync strength, because its mechanism captures timing but not content. Our method includes this weak channel in the envelope; we expect `acc_T` to rarely be the binding term (so the alignment baseline rarely drives flags) but to occasionally match M on pure-timing capabilities, and the observable signal is that flags on timing capabilities coincide with high `acc_T` while content-heavy capabilities show a positive gap — confirming the envelope is conservative rather than over-flagging.

**vs. External frozen audio-only model A_ext (sanity check)** — On the same "visual-object-identification-under-speech" capability, A_ext may be a stronger unimodal model than M's internal audio sub-path, producing `acc_A_ext > acc_A`. Our method does not use A_ext in gap or flag; instead it computes `sanity[c] = 1[ max(acc_A_ext, acc_V_ext) <= env[c] + delta ]`. If `sanity[c]=0`, we report that gap[c] may be confounded by M's internal unimodal weakness and interpret the capability with caution, rather than attributing the gap to joint reasoning.

**vs. External frozen visual-only model V_ext (sanity check)** — On "off-screen-sound-source-localization," V_ext may exceed M's internal visual sub-path on on-screen items. Again, `sanity[c]` flags this, and we do not use V_ext to compute gap or flag; the internal envelope remains the primary diagnostic.

**vs. Known-shortcut model M_shortcut (calibration)** — M_shortcut is fine-tuned to answer from transcript leakage on a designated subset of capabilities (e.g., 3 of 15). We expect `flag[c]=1` on exactly those 3 capabilities and `flag[c]=0` on the other 12, with `gap[c] <= tau` on the injected ones and `gap[c] > tau` elsewhere. If flags do not align with the injected capabilities, the envelope is not detecting ground-truth shortcuts.

**vs. Stronger-unimodal control** — We swap in a stronger external unimodal model for A_ext/V_ext and recompute `sanity[c]`. If `sanity[c]` changes from 1 to 0 on some capabilities, the envelope is sensitive to external baseline strength; we report this as a limitation and rely on the internal ablation for the primary diagnostic.

### What would falsify this idea

If `gap[c]` is uniformly positive across all 15 capabilities — i.e., M exceeds its own internal modular envelope everywhere by a similar margin — then the claim that shortcuts dominate specific capabilities is wrong, since a genuine joint-reasoning advantage would concentrate where unimodal channels are insufficient rather than spread evenly. Equivalently, if the `mean`-vs-`max` ablation produces identical flags, the envelope construction is not doing the diagnostic work we claim; and if the known-shortcut injection calibration fails to fire flags on the injected capabilities, the envelope does not detect ground-truth shortcuts.

## References

1. EvaOmni: An Evidence-Grounded Benchmark for Audio-Visual Reasoning
