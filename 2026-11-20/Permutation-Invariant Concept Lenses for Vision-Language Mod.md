# Permutation-Invariant Concept Lenses for Vision-Language Models

## Motivation

Mechanistic interpretability of VLMs currently reads out concept information using token-anchored carriers whose identity is tied to a fixed input layout, so it is unknown whether an identified concept's causal role reflects the concept or merely the position it happens to occupy. Evidence from cross-modal information flow work (E1, E4) shows that visual information is deliberately routed into specific question-token positions in lower and middle layers and then propagated to the final position, meaning readout position and information content are structurally entangled in exactly the layers interpretability methods probe. MC-Lens (E2, E3) extends readout carriers from a single name token to a token set but does not test whether its identifiability and causal-role findings survive reordering of the multimodal input sequence, leaving a shared gap across both branches: neither the layer-band transfer account nor the lens formulation separates genuine concept causal role from positional/attention-order confounds.

## Key Insight

Because cross-modal information is routed to layout-dependent positions (E1, E4), a concept readout is only causally meaningful if it is invariant under permutation of the multimodal input sequence; enforcing this invariance via content-addressed rather than position-addressed carriers is the structural property that separates concept identity from layout, which neither the fixed-layout MC-Lens nor the layer-band transfer account establishes.

## Method

**A. What it is.** We introduce PIL (Permutation-Invariant Lens), a readout-and-intervention framework that produces concept identifiability and causal-role scores for a VLM that are invariant to the ordering of the multimodal input sequence. Inputs: a VLM, an image-question pair, and a set of target concepts. Outputs: per-layer concept identifiability scores, per-layer causal-role scores (reported as a lower bound with a redundancy diagnostic), and a permutation-stability diagnostic that decomposes each score into a layout-invariant component and a positional-residual component.

**Load-bearing assumption (stated explicitly).** PIL assumes that a concept's representation at layer m is localized to a content-addressable token set (the carrier) such that replacing that carrier's content removes the concept's contribution, so that cross-layout agreement of the carrier readout measures concept identity rather than the carrier-selection procedure itself. This assumption is load-bearing and is not verified by construction; we therefore (i) select carriers from an independent causal screen rather than from the decoder-lens argmax used for scoring, (ii) replace the single-carrier content-replacement intervention with a distributed intervention over the top-k carrier set plus a redundancy control, and (iii) report causal_role as an explicit lower bound with a redundancy diagnostic rather than as a point estimate of concept necessity.

**B. How it works.**

```pseudocode
# Phase 1: layout ensemble construction
inputs: image I, question Q, concept set C, layouts L = {image-first, question-first, k=8 seeded interleavings}
for each layout l in L:
    seq_l = arrange(I, Q, l)              # same semantic content, permuted positions
    record position map pos_l: token -> index

# Phase 2: content-addressed readout carriers (independent causal screen)
for each layer layer m in 1..M:
    H_m = hidden_states(seq_l, m)          # [n_tokens, d]
    for each concept c in C:
        # carrier = token set selected by an INDEPENDENT causal screen, not by the decoder-lens argmax used for scoring
        # causal screen: activation patching over all token positions, ranking positions by |delta logit_c| when patched with mean activation
        patch_scores = causal_screen(H_m, c)   # [n_tokens], computed on a held-out probe set of 64 examples
        carrier_c = top-k indices by patch_scores, k = min(16, #positions with patch_score > 95th pct of shuffled-concept null)
        if |carrier_c| == 0:
            # explicit empty-carrier policy: exclude this (m,c) cell from scoring and record coverage rate
            coverage[m,c] = 0
            continue
        coverage[m,c] = 1
        readout[m,c,l] = mean_{i in carrier_c} H_m[i]

# Phase 3: permutation-invariance test
for each (m, c) with coverage[m,c] == 1:
    identifiability[m,c] = agreement of readout[m,c,l] across l   # cosine consistency
    # causal role via DISTRIBUTED counterfactual intervention on the CONTENT, not the index
    for each l:
        # distributed intervention: mean-ablation over top-k carrier set
        intervene: replace {H_m[i] : i in carrier_c} with mean over other concepts (mean-ablation)
        delta[m,c,l] = change in target-concept logit at final position
        # redundancy control: ablate carrier and measure residual concept logit
        residual_logit[m,c,l] = concept logit after carrier ablation
    causal_role[m,c]     = mean_l delta[m,c,l]          # reported as LOWER BOUND
    redundancy[m,c]      = mean_l residual_logit[m,c,l] # diagnostic: high => concept redundantly distributed
    positional_residual[m,c] = variance_l delta[m,c,l]

# Phase 4: layer-band extraction
band = contiguous layers where identifiability and causal_role exceed
       null-calibrated thresholds (shuffled-concept controls, 95th pct)
report band stability across layouts as Jaccard overlap of band layer-sets
```

Hyperparameters: 10 layouts (2 canonical + 8 seeded interleavings), null calibration at 95th percentile over 50 stratified shuffled-concept controls per layer with bootstrap CIs (1000 resamples), carrier size capped at 16 tokens, causal screen on 64 held-out probe examples, empty-carrier cells excluded and coverage rate reported.

**C. Why this design.** We chose content-addressed carriers over the token-anchored carriers used by MC-Lens (E2, E3) because anchoring a carrier to a fixed index confounds concept identity with the position that cross-modal routing happens to write into (E1, E4), accepting the cost that content-addressing can return empty carrier sets for weakly-decoded concepts and requires an explicit empty-carrier exclusion policy with reported coverage. We chose a layout ensemble with seeded interleavings over a single image-first vs. question-first comparison because two layouts cannot distinguish a genuine positional effect from a layout-specific artifact, accepting higher compute cost per concept. We chose counterfactual intervention on carrier content rather than on carrier position because position interventions change two things at once (content and attention routing), accepting the cost that content replacement is a coarser intervention. We chose Jaccard overlap of layer-band sets as the stability metric over per-layer correlation because the claim under test is about band boundaries, not smooth score profiles, accepting the cost that Jaccard is insensitive to small shifts within a band. A domain expert would not describe this as a variant of MC-Lens: MC-Lens fixes the readout carrier and varies the concept, whereas PIL varies the layout while holding the concept fixed, which is the only way to separate the two. To break the circularity between carrier selection and scoring, we select carriers from an independent causal screen (activation patching over all positions on a held-out probe set) rather than from the decoder-lens argmax used for scoring, and we replace the single-carrier intervention with a distributed mean-ablation over the top-k carrier set plus a redundancy control, reporting causal_role as a lower bound with a redundancy diagnostic.

**D. Why it measures what we claim.** `identifiability[m,c]` measures concept identity stability because cosine agreement of readouts across layouts is equivalent to the concept being recoverable independently of position, under the assumption that the layout ensemble spans the routing configurations the model actually uses; this assumption fails when a layout produces out-of-distribution attention patterns, in which case the metric reflects layout-induced representational drift rather than concept instability. `causal_role[m,c]` measures concept causal necessity because replacing the carrier content with other-concept content is equivalent to removing the concept's contribution, under the assumption that the carrier set is the sole locus of that concept's representation at layer m; this assumption fails when concept information is redundantly distributed outside the carrier, in which case the metric reflects a lower bound on causal role. `positional_residual[m,c]` measures layout confound because variance of the intervention effect across layouts is equivalent to the portion of causal role attributable to positional routing, under the assumption that layouts differ only in position and not in semantic content; this assumption fails when interleaving changes the model's interpretation of the question, in which case the residual reflects semantic ambiguity rather than pure positional confound. Band Jaccard overlap measures transfer-band stability because overlap of layer sets across layouts is equivalent to the band being a property of the computation rather than of the layout, under the assumption that band thresholds are calibrated on the same null distribution for all layouts; this assumption fails when null distributions differ by layout, in which case overlap reflects threshold miscalibration. `redundancy[m,c]` measures the residual concept logit after carrier ablation and thus bounds the lower-bound bias of `causal_role[m,c]`: high redundancy indicates the carrier is not the sole locus, so the reported causal role understates true necessity.

## Contribution

(1) A permutation-invariant lens framework (PIL) with content-addressed readout carriers and a layout-ensemble protocol that decomposes concept identifiability and causal role into layout-invariant and positional-residual components. (2) A design principle: concept causal-role claims in VLMs should be reported alongside a permutation-stability diagnostic, since position-addressed carriers can inflate apparent causal role through routing confounds. (3) A permutation-stability benchmark and null-calibrated band-extraction procedure for comparing lens formulations across layouts.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | VQA pairs with object-centric questions, multiple canonical layouts (e.g., 500 pairs from a standard VQA benchmark, each rendered under 10 layouts) | Needs layout variation and localized concepts |
| Primary metric | Layout-invariant causal-role score (positional residual fraction) | Directly quantifies the confound we claim to remove |
| Baseline | MC-Lens token-anchored carrier | Tests whether fixed-index carriers confound identity with position |
| Baseline | Single image-first vs. question-first comparison | Tests whether two layouts suffice to isolate positional effects |
| Baseline | Position-intervention counterfactual | Tests whether content replacement is necessary for clean causal signal |
| Ablation-of-ours | PIL without layout ensemble (single layout) | Isolates contribution of the permutation ensemble itself |
| Ablation-of-ours | PIL with decoder-lens argmax carrier (circular selection) | Tests whether independent causal screen breaks circularity |
| Ablation-of-ours | PIL with single-carrier intervention (no distributed mean-ablation) | Tests whether distributed intervention reduces redundancy bias |
| Diagnostic | Redundancy probe (residual concept logit after carrier ablation) | Bounds lower-bound bias of causal_role |
| Diagnostic | Attention entropy across layouts | Controls for layout-induced OOD attention drift |

### Why this setup validates the claim

The central claim is that concept identifiability and causal role in VLMs are partly artifacts of input layout, and that PIL separates a layout-invariant component from a positional residual. The dataset must contain questions where object concepts are localized yet routing depends on token order, so VQA pairs with object-centric questions and multiple canonical layouts are the right substrate: they let the same semantic content appear under different orderings. MC-Lens is the key baseline because it holds layout fixed and varies concepts, so if it produces concept scores that shift under reordering, that is direct evidence its token-anchored carrier conflated identity with position — exactly the sub-claim PIL attacks. The two-layout baseline tests the weaker sub-claim that a single image-first vs. question-first contrast is insufficient to distinguish genuine positional effects from layout-specific artifacts. The position-intervention baseline tests whether intervening on content rather than index gives a cleaner causal signal. The ablation removes the layout ensemble and should collapse the invariant/residual decomposition, showing the ensemble is doing the work. The additional ablations test the circularity repair (independent causal screen vs. decoder-lens argmax) and the distributed intervention (mean-ablation vs. single-carrier). The redundancy probe and attention-entropy diagnostic bound the lower-bound bias and OOD drift. The primary metric — the positional-residual fraction of causal role — is the right detector because it is defined as the layout-attributable variance; if layout truly confounds baseline readouts, this fraction should be large for baselines and small for PIL on the same concepts.

### Expected outcome and causal chain

**vs. MC-Lens token-anchored carrier** — On a case where the same object concept is routed to different token positions under image-first versus interleaved layouts, MC-Lens fixes its carrier at one index, so its readout samples a position that cross-modal routing only sometimes writes into, producing concept scores that drift with layout because its mechanism gap is index-anchoring. Our method instead re-derives the carrier from an independent causal screen per layout, so the same concept is sampled wherever it actually lands, because content-addressing follows the representation rather than the slot. We expect a noticeable reduction in score variance across layouts for object concepts that are positionally mobile, with rough parity on concepts that happen to be positionally stable — the gap should concentrate exactly on the mobile subset.

**vs. Single image-first vs. question-first comparison** — On a case where both canonical layouts happen to route the concept to similar positions, the two-layout baseline sees near-zero difference and concludes there is no positional confound, because its mechanism gap is too few samples to distinguish a genuine null from a layout-specific coincidence. Our method instead draws on eight seeded interleavings that probe intermediate routing configurations, so a confound that is invisible at the two extremes surfaces in the interior. We expect the two-layout baseline to report a near-flat positional residual while PIL reports a clearly non-trivial residual on concepts whose routing is order-sensitive, with agreement on concepts that are genuinely layout-invariant.

**vs. Position-intervention counterfactual** — On a case where moving a carrier token changes both the content at that slot and the attention pattern into it, the position-intervention baseline attributes the resulting logit change to content alone, because its mechanism gap is that position and routing move together. Our method instead replaces carrier content while leaving positions fixed, so the measured delta reflects concept removal rather than a joint content-and-routing perturbation. We expect position-intervention to overstate causal role on concepts whose slots are attention hubs, while PIL's causal-role estimates stay closer to the content-replacement ablation and show a cleaner separation from shuffled-concept controls.

**vs. PIL without layout ensemble (ablation)** — On a case where a single layout is used, the ablation cannot compute cross-layout agreement or variance, so its identifiability and residual scores degenerate to a single-layout point estimate, because its mechanism gap is the missing ensemble. Our method instead decomposes each score into invariant and residual components across layouts, so the ablation should show near-zero residual by construction and inflated identifiability, confirming that the ensemble — not the carrier design alone — is what enables the separation. We expect the ablation to look artificially stable, with band Jaccard overlap near one by construction, versus PIL's more informative band boundaries.

**vs. PIL with decoder-lens argmax carrier (circular selection)** — On a case where the carrier is selected by the same decoder-lens argmax used for scoring, the circular ablation will show inflated identifiability because the carrier is chosen to maximize the readout it is then used to validate, because its mechanism gap is circularity. Our method instead selects carriers from an independent causal screen, so identifiability reflects concept recoverability rather than selection bias. We expect the circular ablation to report higher identifiability and lower positional residual than PIL, with the gap largest on concepts where the decoder lens is noisy.

**vs. PIL with single-carrier intervention (no distributed mean-ablation)** — On a case where the concept is redundantly distributed across the top-k carrier set, the single-carrier ablation will underestimate causal role because it removes only one locus, because its mechanism gap is the single-locus assumption. Our method instead uses distributed mean-ablation over the top-k set, so causal_role captures more of the concept's contribution. We expect the single-carrier ablation to show lower causal_role and higher redundancy residual than PIL, with the gap concentrated on concepts with high redundancy diagnostic.

### What would falsify this idea

If PIL's reduction in layout-induced score variance is uniform across all concepts rather than concentrated on the positionally mobile subset predicted by the routing-confound account, the central claim that layout confounds baseline readouts would be wrong — the gain would instead reflect generic smoothing. Similarly, if band Jaccard overlap is no higher for PIL than for the single-layout ablation once thresholds are matched, the permutation-invariance claim fails. Additionally, if the redundancy diagnostic shows high residual concept logit after carrier ablation for most concepts, the causal-role lower bound is too weak to support the claimed separation.

## References

1. Probing the "Conscious" Mind: Selective Access and Global Workspace in Vision-Language Models
