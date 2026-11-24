# Stage-Monotone Modality-Attribution Routing for Multimodal Continued Pre-Training

## Motivation

Multimodal continued pre-training pipelines such as WeMM-Ecom's two-stage schedule optimize fusion end-to-end, with no invariant tying the intermediate stages at which visual evidence enters question-token representations to the final output decision. Diagnostic work (Cross-modal Information Flow in MLLMs) shows that LLaVA-series models transfer general image features into question tokens in lower layers and object-specific visual evidence in middle layers before propagating a multimodal representation to the prediction position, yet no training objective constrains that transfer trajectory to remain consistent across distribution shift — e.g., when WeMM-Ecom's e-commerce data proportion is raised in the second pre-training stage, the layer-wise attribution can drift while end-to-end loss still decreases, so the model can pass the benchmark while its modality routing is no longer the routing that produced the correct answer.

## Key Insight

Because the layer-wise image-to-question-token transfer is a measurable, ordered quantity, it can be made a training invariant: requiring the modality-attribution map to be monotone across pre-training stages and to be causally necessary for the logits (via a swapped-attribution counterfactual) forces the routing that the final prediction depends on to be the same routing diagnosed at each stage, which end-to-end loss alone cannot enforce.

## Method

**What it is.** SMAR (Stage-Monotone Attribution Routing) is a regularizer added to WeMM-Ecom's two-stage continued pre-training. Its inputs are, per training example, the image, the question tokens, and the model's layer-wise attention/attribution over image patches; its outputs are two scalar penalties added to the language-modeling loss plus a per-stage attribution map that must remain consistent across stages and causally necessary for the prediction.

**Load-bearing assumption (stated explicitly).** SMAR assumes that the layer-wise patch-to-question-token attribution profile has a well-defined, per-token transfer stage that is stable across pre-training stages and causally tied to answer correctness. Concretely, we assume (A1) the attribution profile for a token is sufficiently peaked that its centroid over the lower+middle bands is a meaningful stage index rather than attention noise, and (A2) noising the full lower+middle attribution band removes exactly the visual evidence the routing claims to carry, without redundant pathways recovering it. If A1 fails (diffuse profile), the routing summary reflects noise; if A2 fails (redundant pathways), the logit drop reflects redundancy rather than non-necessity. Both are checked empirically in the calibration step below, and tokens failing A1 are excluded from the monotonicity penalty rather than treated as transfer stages.

**How it works.**

```pseudocode
# Per example, per pre-training stage s in {stage1, stage2}
# 1. Attribution map A_s[l, i, j] = contribution of image patch i to question token j at layer l
#    (attention-weighted value-norm, aggregated over heads; normalized per layer)
A_s = attribution_map(model, image, question)          # shape [L, P, Q]

# 2. Band-level routing summary over the lower+middle bands B = layers [L_lo, L_mid]
#    (L_lo = 0, L_mid = floor(0.6 * L) for LLaVA-style 32-layer models; i.e. layers 0-19)
#    r_s[j] = centroid of the attribution profile over B, weighted by per-layer mass:
#             r_s[j] = sum_{l in B} l * a_s[l, j] / sum_{l in B} a_s[l, j],
#             where a_s[l, j] = sum_i A_s[l, i, j]
#    m_s[j] = sum_{l in B} a_s[l, j]                    # total band mass
#    peakedness p_s[j] = 1 - H(a_s[:, j]) / log(|B|)    # normalized 1 - entropy over B
r_s, m_s, p_s = routing_summary(A_s)                   # per question token

# 3. Stage-monotonicity penalty, gated on peakedness:
#    only tokens with p_1[j] >= tau_p (tau_p = 0.5) and m_1[j] >= tau_m (tau_m = 0.01)
#    are included; diffuse tokens are excluded rather than penalized.
L_mono = mean_{j in S} max(0, r_1[j] - r_2[j] + delta)  # delta = 1 layer margin
#    S = { j : p_1[j] >= 0.5 and m_1[j] >= 0.01 }

# 4. Causal-necessity penalty via band-level swapped-attribution counterfactual:
#    replace image-patch attribution across the full lower+middle band B with a
#    matched-noise map (Gaussian noise matched to per-layer mean and variance of A_s),
#    recompute logits, require the correct-answer logit to drop.
logits      = model(image, question)
logits_swap = model_with_band_attribution_swap(image, question, band=B, mode='matched-noise')
L_causal = mean max(0, logits_swap[correct] - logits[correct] + gamma)  # gamma = 0.5 nats

# 5. Total loss
L = L_lm + lambda_mono * L_mono + lambda_causal * L_causal
# lambda_mono = 0.1, lambda_causal = 0.1, annealed linearly to 0 over last 10% of steps
```

Anchor examples are a fixed 5% replay buffer drawn from stage-1 data and re-scored at stage 2, so the monotonicity term compares the same inputs before and after the e-commerce data-proportion increase.

**Calibration and verification of the load-bearing assumption (run before the main training).** On the WeMM-Ecom-35B-A3B model itself, we reproduce the Cross-modal Information Flow protocol on a held-out set of 2,048 image-question pairs from stage-1 data: for each question token we compute the attribution profile a_s[l, j] over all layers and report (i) the distribution of peakedness p_s[j], (ii) the fraction of tokens with p_s[j] >= 0.5, and (iii) the centroid r_s[j] distribution per token type (attribute, object, yes/no). If the peakedness distribution has no mode above 0.5 (i.e., the profile is diffuse for most tokens), the routing summary is not a valid stage measure for this architecture and we report that as a negative result rather than proceeding. For the causal term, we additionally run a non-peak-layer control: swapping attribution at a randomly chosen layer outside the band B (e.g., a high layer in [L_mid+1, L-1]) and confirming that the logit drop is smaller than the band-level swap; if the non-peak control produces a comparable drop, the band-level swap is not specific to the diagnosed routing and we report that as a failure of A2.

**Why this design.** We chose a band-level centroid routing summary rather than a full attribution-map L2 penalty because the diagnostic evidence (Cross-modal Information Flow in MLLMs) shows the transfer is *ordered in depth* and distributed across lower and middle bands, not a fixed tensor or a single peak layer, so penalizing centroid regression over the band targets the property that actually shifts while excluding diffuse tokens via the peakedness gate, accepting that we lose fine-grained patch-level fidelity and that the gate discards some tokens. We chose a band-level matched-noise attribution swap over input masking because masking the image would change the input distribution and confound the causal test with out-of-distribution effects, and because a single-layer swap is recoverable by redundant pathways (as the mechanistic-interpretability literature shows), accepting the cost that the band-level swap is only an approximation of a true intervention and that it may still be partially recovered. We chose to anneal both penalties to zero in the last 10% of steps so the final model is not over-constrained by the regularizer at convergence, accepting that late-stage drift is only checked, not corrected. A domain expert would not describe this as a WeMM-Ecom variant with an extra loss: WeMM-Ecom's pipeline (two-stage pre-training plus hybrid-thinking post-training) is unchanged, and the contribution is the invariant itself — a stage-monotone, causally-necessary routing condition — which no prior multimodal pre-training objective, including SymDPO's preference-based demonstration alignment, enforces.

**Why it measures what we claim.** The band-level routing summary `r_s[j]` measures *modality-transfer stage* because the centroid of patch-to-token attribution over the lower+middle bands is assumed to be the depth at which visual evidence is integrated for token j (the assumption validated by Cross-modal Information Flow in MLLMs' layer-wise analysis and checked by our peakedness calibration); this assumption fails when a token's attribution is diffuse across many layers with no clear mode, in which case `r_s[j]` reflects attention noise rather than a transfer stage — which is why we gate the monotonicity penalty on `p_s[j] >= 0.5` and exclude diffuse tokens. The band-level swapped-attribution logit drop `logits[correct] - logits_swap[correct]` measures *causal necessity of the diagnosed routing* because replacing the full lower+middle band attribution with matched noise is assumed to be a valid intervention that removes exactly the visual evidence the routing claims to carry without redundant pathways recovering it; this assumption fails when the model has redundant visual pathways that recover the information from other layers, in which case the drop reflects redundancy rather than non-necessity — which is why we include a non-peak-layer control swap to confirm the drop is specific to the diagnosed band. The monotonicity margin `delta` measures *stage consistency* because the same anchor examples are replayed across stages, so any regression in `r` is attributable to the stage-2 data shift rather than to input variation; this assumption fails when the anchor replay buffer is not representative of the stage-1 distribution, in which case the penalty reflects sampling bias — which is why the buffer is a fixed 5% random sample of stage-1 data and we report its coverage of stage-1 token types.

## Contribution

(1) A stage-monotone modality-attribution routing regularizer (SMAR) that couples layer-wise image-to-question-token transfer to final logits via a matched-noise attribution-swap counterfactual, added to a two-stage multimodal continued pre-training schedule. (2) A design principle: routing consistency across pre-training stages is a trainable invariant that end-to-end loss does not enforce, operationalized as a per-token peak-layer summary plus a causal-necessity margin. (3) A diagnostic protocol that reports per-stage routing maps alongside benchmark accuracy, enabling detection of routing drift that accuracy alone masks.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | WeMM-Ecom 28 e-commerce + 64 general benchmarks | Covers both stage-2 domain shift and general retention |
| Primary metric | Accuracy on stage-2 e-commerce suite | Directly reflects cross-stage routing stability claim |
| Baseline | WeMM-Ecom-35B-A3B (no regularizer) | Isolates our added objective from pipeline gains |
| Baseline | SymDPO-style preference alignment | Tests whether preference alignment alone yields stage-monotone routing |
| Baseline | Qwen3.5 matched-size | Tests whether stage-monotone routing is architecture-agnostic |
| Baseline | Full-attribution-map L2 drift penalty (no routing summary, no causal term) | Tests whether stage-monotone summary + causal term beats trivial consistency |
| Ablation-of-ours | SMAR minus causal-necessity penalty | Isolates monotonicity term from causal term |
| Ablation-of-ours | SMAR with non-peak-layer swap control | Confirms causal drop is specific to diagnosed band |
| Ablation-of-ours | SMAR with peakedness gate removed | Tests whether diffuse-token exclusion matters |
| Feasibility scale | WeMM-Ecom-2B dense variant with full SMAR | Demonstrates effect at feasible scale before 35B run |

### Why this setup validates the claim

The central claim is that a stage-monotone, causally-necessary routing condition yields gains concentrated where stage-2 e-commerce data reshapes visual evidence integration. The dataset choice matters because the two-stage pre-training itself creates the stage shift: stage-1 general data builds routing, and stage-2 e-commerce data risks regressing it, so only a benchmark spanning both regimes can detect the predicted effect. The unregularized WeMM-Ecom baseline tests the core sub-claim that the stage shift alone degrades routing; SymDPO tests whether preference-based demonstration alignment, which operates on outputs rather than internal routing, could incidentally produce the same invariant; Qwen3.5 tests whether the invariant is specific to our pipeline or a general property. The full-attribution-map L2 drift baseline is the sharpest novelty test: if a trivial consistency regularizer matches SMAR, the routing summary and causal term add nothing. The ablation-of-ours without the causal-necessity penalty isolates whether the causal term, not merely the stage comparison, is doing the work; the non-peak-layer swap control confirms the causal drop is specific to the diagnosed band rather than a generic perturbation effect; and the peakedness-gate ablation tests whether excluding diffuse tokens is necessary. The 2B feasibility run demonstrates the effect at a scale where full SMAR is affordable before committing to the 35B run. Accuracy on the stage-2 suite is the right primary metric because the predicted effect is a capability loss under domain shift, and accuracy on that suite is exactly the observable that a routing regression would suppress; a uniform accuracy gain across subsets would be inconsistent with our mechanism story.

### Expected outcome and causal chain

**vs. WeMM-Ecom-35B-A3B (no regularizer)** — On an e-commerce case where a product image and a question about a specific attribute (e.g., material) require late-layer visual grounding, the unregularized baseline's routing drifts earlier after stage-2 data proportion increases, so early layers absorb generic visual features and the answer logit relies on textual priors, producing attribute confusions. Our method replays stage-1 anchors and penalizes band-centroid regression, keeping the centroid at or beyond its stage-1 position, so specific-object evidence reaches the question tokens at the relevant depth. We expect a noticeable accuracy gap concentrated on attribute-specific e-commerce subsets, with near-parity on text-dominant subsets.

**vs. SymDPO-style preference alignment** — On a multimodal in-context demonstration where the correct answer depends on reading the demonstration image rather than a textual pattern, SymDPO aligns outputs to symbol-conditioned preferences but leaves the layer-wise attribution path unconstrained, so the model can satisfy preferences while still routing through shallow, pattern-matching layers. Our method directly constrains the band-level routing summary and enforces causal necessity via band-level swapped attribution, so the band-level evidence is both preserved across stages and required for the prediction. We expect SymDPO to improve demonstration-following but show little gain on stage-2 attribute subsets, whereas ours shows gains on both, with the largest separation on cases where the demonstration image contradicts the textual prior.

**vs. Qwen3.5 matched-size** — On a general-purpose case requiring fine-grained visual detail after heavy e-commerce pre-training, Qwen3.5 has no stage-monotone constraint and its routing is shaped only by its own data mixture, so on fused-modal retrieval-style queries it may peak early and miss detail. Our method's monotonicity term is anchored to our stage-1 distribution, so it preserves the depth at which detail is integrated while still adapting to e-commerce data. We expect our model to surpass its Qwen3.5 counterpart on general benchmarks, with the advantage largest on detail-sensitive subsets and smaller on coarse-category subsets.

**vs. Full-attribution-map L2 drift penalty** — This baseline penalizes L2 drift of the full attribution map across stages without a routing summary or causal term. It may reduce raw map drift but does not target the ordered depth of transfer or require the diagnosed routing to be causally necessary, so we expect it to yield smaller gains than SMAR on attribute-specific e-commerce subsets and to show no advantage on the non-peak-layer control. If this baseline matches SMAR, the routing summary and causal term add nothing and the novelty claim collapses.

**vs. SMAR minus causal-necessity penalty** — This ablation keeps the stage-monotone band-centroid penalty but removes the band-level swapped-attribution term. If the gain over the unregularized baseline is driven by the causal term, this ablation should show a substantial drop on attribute-specific subsets, confirming that causal necessity, not merely stage comparison, is doing the work.

**vs. SMAR with non-peak-layer swap control** — This ablation swaps attribution at a randomly chosen layer outside the lower+middle band instead of the full band. If the logit drop from the non-peak control is comparable to the band-level swap, the causal term is not specific to the diagnosed routing and the causal-necessity claim is unsupported.

**vs. SMAR with peakedness gate removed** — This ablation includes diffuse tokens in the monotonicity penalty. If performance degrades relative to full SMAR, the peakedness gate is necessary to avoid penalizing attention noise; if performance is unchanged, the gate is not load-bearing.

**vs. WeMM-Ecom-2B dense variant with full SMAR** — This feasibility run applies the full SMAR pipeline at 2B scale. We expect the same directional effect (gain concentrated on attribute-specific e-commerce subsets) with reported wall-clock and memory overhead per step, demonstrating the effect is not an artifact of the 35B scale.

### What would falsify this idea

If the gain over the unregularized WeMM-Ecom baseline is uniform across e-commerce and general subsets rather than concentrated on attribute- and detail-sensitive subsets, or if removing the causal-necessity penalty leaves the stage-2 accuracy unchanged, or if the full-attribution-map L2 drift baseline matches SMAR, or if the non-peak-layer swap control produces a comparable logit drop to the band-level swap, then the routing invariant is not the causal driver and the central claim is wrong.

## References

1. WeMM-Ecom: Advancing Multimodal Content Understanding for WeChat E-Commerce
