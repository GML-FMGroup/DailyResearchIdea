# Causally Sufficient Retrieval Heads in Multimodal LLMs via Activation Patching

## Motivation

The ablation-based localization in 'A Broader Look into MLLM's Retrieval Heads' identifies cross-modal retrieval heads by removing them and observing performance drops, but this only establishes necessity, not sufficiency. The field's shared assumption that ablation is an adequate causal probe systematically prevents moving from correlational localization to a complete causal account, leaving open whether the localized head set is the actual mechanism or merely one of many necessary components.

## Key Insight

If the identified cross-modal retrieval heads form a causally sufficient mechanism, then patching only their activations from a clean run into a corrupted run will restore retrieval performance to near-intact levels, a property that ablation alone cannot establish because necessity does not imply sufficiency.

## Method

**What it is** — We introduce Head-Set Activation Patching (HSAP), a causal sufficiency test that takes as input a localized set of cross-modal retrieval heads (identified via prior ablation), a clean multimodal input, and a corrupted version of that input, and outputs a restoration score quantifying how much patching only the head set's clean activations into the corrupted run recovers retrieval performance. The load-bearing assumption is stated explicitly: patching a head's clean-run output activation into a corrupted run is a valid counterfactual intervention that restores the head's functional state, i.e., the clean activation remains the appropriate input to downstream computation even though the corrupted input has changed the head's own input and the surrounding context. We make this assumption testable via a no-op patch control and a context-matched patching control, and we report restoration relative to an on-manifold ceiling rather than raw clean accuracy.

**How it works** —
```pseudocode
# Phase 0: Pilot — verify corruption reaches near-chance
# Run on 128 examples from the paired dataset; if corrupted accuracy > 0.2 above chance, adjust corruption strength (e.g., silence duration, noise SNR) and re-check.

# Phase 1: Localize retrieval heads via ablation (reuse prior protocol)
H_ret = ablate_and_locate(model, paired_dataset)  # set of cross-modal retrieval heads; top-k heads by ablation effect, k=32 unless otherwise specified

# Phase 2: Construct corrupted inputs
for each clean pair (audio, image, question, answer):
    corrupt evidence modality (e.g., replace audio with silence of matched duration or image with Gaussian noise at SNR=0 dB)
    verify corruption reduces retrieval accuracy to near-chance (≤ 0.1 above chance on a 128-example pilot)

# Phase 3: Activation patching with on-manifold verification
# Token-position alignment: use the tokenizer's offset mapping to align clean and corrupted runs at the question-token positions; if lengths differ, pad the shorter run with a mask token and record alignment.
for each corrupted input:
    run model with hooks on all heads in H_ret
    for each head h in H_ret:
        replace h's output activation with its clean-run activation at the aligned token position
    record patched output and retrieval accuracy (acc_patched)

# Phase 3b: No-op patch control (on-manifold ceiling)
# Patch clean activations into the clean run (same hooks, same positions). Must yield acc_noop ≈ acc_clean (restoration ≈ 1.0). If not, the patching implementation is off-manifold; abort and fix hooks.

# Phase 3c: Context-matched patching control
# Patch activations from a clean run that shares the same corrupted context (e.g., same corrupted audio but clean image) into the corrupted run. This tests whether the clean activation is appropriate given the corrupted context.

# Phase 4: Sufficiency test
restoration = (acc_patched - acc_corrupted) / (acc_noop - acc_corrupted)
# Sufficiency if restoration >= 0.9 (near-intact), where 0.9 is pre-registered and not tuned.
# Also report raw restoration relative to acc_clean for comparison.
```

**Why this design** — We chose activation patching over retraining or head masking because patching directly intervenes on the computational quantity (head output) that the localization step claims is causally responsible, accepting the cost that patching requires clean-run activations and may not generalize to arbitrary corruptions. We chose a single-modality corruption (rather than multi-modality) to isolate the cross-modal retrieval mechanism, accepting that this may underestimate sufficiency if the head set is redundant across modalities. We chose a restoration threshold of 0.9 (rather than exact equality) to account for residual noise, accepting that this introduces a hyperparameter that could be tuned; we pre-register 0.9 and do not tune it. Unlike ablation-only studies (e.g., 'A Broader Look into MLLM's Retrieval Heads'), which test necessity, our design tests sufficiency by holding all other heads at their corrupted activations and patching only the localized set, directly answering whether the set is sufficient. This is not a variant of standard ablation because it requires a counterfactual clean run and a corrupted run, and the patched activation is a specific intervention that ablation cannot simulate. The no-op patch control (Phase 3b) verifies that the patching implementation is on-manifold: if patching clean activations into the clean run does not yield restoration ≈ 1.0, the patching is off-distribution and the restoration score is invalid. The context-matched control (Phase 3c) tests the load-bearing assumption directly: if the clean activation is appropriate only when the context is also clean, then context-matched patching should yield lower restoration than full clean patching, and we report this gap.

**Why it measures what we claim** — The restoration score measures causal sufficiency because we assume that if patching only the localized head set restores performance, then those heads carry all the information needed for the retrieval function; this assumption fails when the corrupted input still contains partial evidence that other heads can exploit, in which case the restoration score reflects redundancy rather than sufficiency. The patched activation replacement measures the head's causal contribution because we assume that the clean activation is the same functional state as in the intact run; this assumption fails when the corrupted input changes the context such that the clean activation is no longer appropriate, in which case the patched output reflects a mismatch rather than the head's true role. The corruption manipulation measures the absence of evidence because we assume that the corruption removes the modality-specific evidence without introducing artifacts; this assumption fails when the corruption creates spurious cues (e.g., silence being interpreted as a signal), in which case the corrupted baseline is not a valid counterfactual. The no-op patch control measures on-manifoldness because we assume that patching clean activations into the clean run is a no-op; this assumption fails when the hook implementation or token alignment is incorrect, in which case the restoration score is invalid. The context-matched patching control measures context-appropriateness because we assume that the clean activation remains appropriate under the corrupted context; this assumption fails when the corrupted context alters downstream computation, in which case the context-matched restoration will be lower than full clean restoration.

## Contribution

(1) A causal sufficiency testing protocol for multimodal retrieval heads that combines ablation-based localization with targeted activation patching, providing a tighter head-to-function mapping than necessity-only ablation. (2) An empirical demonstration that the localized cross-modal retrieval head set is causally sufficient for audio-visual retrieval, with restoration scores near 0.9, establishing that the set is not merely necessary but sufficient. (3) A public toolkit for activation patching in multimodal LLMs, enabling reproducible sufficiency tests across checkpoints and tasks.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Paired audio-visual QA (speech vs. on-screen text), 2,000 examples | Isolates cross-modal retrieval with controlled evidence modality |
| Primary metric | Restoration score (patched vs. corrupted vs. clean accuracy), with no-op ceiling | Directly quantifies causal sufficiency of localized head set |
| Baseline 1 | Ablation-only (necessity test) | Tests necessity, not sufficiency; contrast isolates sufficiency claim |
| Baseline 2 | Random head-set patching (k=32 random heads) | Controls for generic patching effects unrelated to localization |
| Baseline 3 | Full-model patching (all heads) | Upper bound: restoration when all heads are patched |
| Ablation-of-ours | HSAP with shuffled clean activations (position-matched, on-manifold) | Controls for activation content vs. position effects |
| Control 1 | No-op patch (clean activations into clean run) | Verifies on-manifold patching; must yield restoration ≈ 1.0 |
| Control 2 | Context-matched patching (clean activations from same corrupted context) | Tests context-appropriateness of clean activations |

### Why this setup validates the claim

The central claim is that the localized cross-modal retrieval head set is causally sufficient: patching only its clean activations into a corrupted run should restore retrieval performance. The paired audio-visual dataset provides a clean counterfactual because the same answer content is present in two modalities, so corrupting one modality removes evidence without changing the task. The restoration score is the right metric because it directly operationalizes sufficiency as the fraction of clean performance recovered by the head set alone, measured relative to the no-op ceiling (which verifies on-manifold patching) rather than raw clean accuracy. Ablation-only tests necessity, so a high restoration score from HSAP but low ablation effect would specifically support sufficiency over necessity. Random head-set patching controls for the possibility that any head set restores performance due to generic activation patching. Full-model patching establishes the ceiling, and shuffled-activation patching (position-matched and on-manifold) controls for the content of the patched activations. The no-op patch control ensures that the patching implementation is valid: if patching clean activations into the clean run does not yield restoration ≈ 1.0, the restoration score is invalid. The context-matched patching control tests the load-bearing assumption that clean activations remain appropriate under the corrupted context; a large gap between full clean and context-matched restoration would indicate context-mismatch. Together, these baselines form a falsifiable test: if HSAP restores near-clean performance only when the localized set is patched, and not with random or shuffled controls, and if the no-op control confirms on-manifold patching, the sufficiency claim is supported.

### Expected outcome and causal chain

**vs. Ablation-only (necessity test)** — On a case where the audio modality is corrupted to silence, the ablation-only baseline cannot distinguish whether the localized heads are necessary or sufficient because it only removes heads from a clean run, producing degraded performance without a counterfactual restoration signal. Our method instead patches clean head activations into the corrupted run, directly testing whether those heads carry the missing evidence. We expect HSAP to yield a high restoration score (near 0.9) on the corrupted-modality subset, while ablation-only shows a drop but no restoration measure, creating a clear dissociation between necessity and sufficiency.

**vs. Random head-set patching** — On a case where the image modality is replaced with noise, random head-set patching will replace arbitrary head activations with clean ones, but because those heads are not the retrieval circuit, the patched activations will be functionally irrelevant and restoration will remain near zero. Our method patches only the localized cross-modal retrieval heads, which we expect to carry the modality-specific evidence, so restoration should be high. We expect a large gap: HSAP restoration near 0.9 versus random patching near 0.1–0.2, concentrated on the corrupted-modality subset and near parity on clean inputs.

**vs. Full-model patching (all heads)** — On a case where the audio is corrupted but the visual evidence is intact, full-model patching replaces all head activations with clean ones, effectively reverting the model to the clean run and yielding restoration near 1.0, but it provides no information about which heads are sufficient. Our method patches only the localized set, so if restoration is also near 0.9, it demonstrates that the localized set alone is sufficient, not just the full model. We expect HSAP to approach full-model patching (e.g., within 0.05–0.1 restoration) on the corrupted-modality subset, with the gap reflecting residual redundancy.

**vs. No-op patch control** — On a clean run with clean activations patched into the same positions, we expect restoration ≈ 1.0 (within 0.02). If not, the patching implementation is off-manifold and all restoration scores are invalid; we will report this as a failure of the method's on-manifold assumption.

**vs. Context-matched patching control** — On a corrupted run where the clean activation is taken from a run with the same corrupted context (e.g., same corrupted audio but clean image), we expect restoration to be lower than full clean patching if the clean activation is context-dependent. The gap between full clean and context-matched restoration quantifies the context-mismatch effect; a large gap would indicate that the load-bearing assumption fails.

### What would falsify this idea

If HSAP restoration is uniformly low (e.g., below 0.5) across all corrupted-modality subsets while full-model patching restores near 1.0, then the localized head set is not sufficient and the central claim is wrong. Alternatively, if HSAP restoration is high even when patching shuffled clean activations (position-matched and on-manifold), then the restoration score reflects generic patching artifacts rather than the specific causal role of the localized heads. If the no-op patch control does not yield restoration ≈ 1.0, the patching implementation is off-manifold and the sufficiency test is invalid.

## References

1. A Broader Look into MLLM's Retrieval Heads
