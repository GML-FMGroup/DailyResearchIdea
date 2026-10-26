# Supervised Evidence-Prior Disentanglement for Omni LLM Hallucination Correction

## Motivation

Omni-Woodpecker (OWP) corrects omni-LLM hallucinations by training-free orthogonal decomposition of fused hidden states into evidence and prior components, but this assumes the two factors are linearly separable in Qwen2.5-Omni's end-to-end trained representations. Because the Qwen2.5-Omni Thinker is trained end-to-end with interleaved TMRoPE-positioned audio-video tokens, grounding evidence and misleading prior are entangled nonlinearly, so a fixed projection cannot isolate them and OWP's rebalancing operates on a subspace that mixes both factors. The structural gap is that no existing mitigation enforces the separability property that projection-based correction silently presupposes.

## Key Insight

A learned decomposition trained with counterfactual evidence-swap supervision yields a subspace in which the evidence direction is identifiable as the unique direction whose projection flips sign under evidence perturbation while the prior direction is invariant — a property no fixed orthogonal projection can guarantee on entangled end-to-end representations.

## Method

**What it is.** EPD (Evidence-Prior Disentangler) is a lightweight learned module inserted at one Thinker layer of Qwen2.5-Omni. It takes the fused hidden state h_t and outputs two additive components h_t = e_t + p_t (evidence and prior), plus a scalar conflict gate g_t. OWP's conflict-conditioned rebalancing is then applied on (e_t, p_t) rather than on an orthogonal projection.

**How it works.**

```pseudocode
# Training data construction (per training sample)
for each training triple (audio, video, question):
    h   = Thinker_hidden(layer=L)          # fused state, dim d=3584
    h_cf = Thinker_hidden(layer=L) with evidence span replaced by a
           counterfactually-consistent span (same entities, altered grounding cue)
    h_pr = Thinker_hidden(layer=L) with prior-inducing context (e.g. shuffled
           audio track) while evidence tokens held fixed

# Module: two linear maps + layernorm, rank r=64
E = W_e @ LN(h)      # evidence component, dim d
P = W_p @ LN(h)      # prior component, dim d
g = sigmoid(w_g . LN(h))   # conflict gate

# Disentanglement losses (weights: 1.0, 0.5, 0.1)
L_rec  = ||h - (E+P)||^2
L_swap = ||E(h_cf) - E(h)||^2 / (||E(h_pr) - E(h)||^2 + eps)   # evidence moves under evidence swap, prior does not
L_orth = ||E^T P||_F^2 / d                                      # soft decorrelation
L_task = CE(answer | rebalanced (E,P))

# Inference: rebalance then decode
E,P,g = EPD(h)
h_corr = E + (1 - g) * P     # suppress prior when conflict gate high
logits = Thinker_head(h_corr)
```

Hyperparameters: layer L chosen by a held-out conflict-probe sweep (we commit to layer 14 of 28); rank r=64; loss weights fixed as above; gate initialized to 0.5. Training freezes the Qwen2.5-Omni backbone and updates only EPD (≈2.1M params) for 3 epochs on 40K constructed triples.

**Swap-pair purity assumption and verification.** The ratio loss L_swap assumes that h_cf differs from h only in the grounding/evidence factor and h_pr differs only in the prior factor. To make this assumption explicit and testable, we add a held-out purity filter: for each constructed pair, we train two linear probes on a disjoint calibration set of 2,000 triples — one probe predicts the grounding answer from h, the other predicts the prior-inducing context label from h. A pair (h, h_cf) is accepted only if the evidence probe's prediction changes (Δlogit > 0.5) and the prior probe's prediction is stable (Δlogit < 0.1); a pair (h, h_pr) is accepted only if the prior probe changes and the evidence probe is stable. We report the yield rate of this filter (expected 60–70%) and use only accepted pairs for L_swap. This operationalizes the assumption that the swap edit is factor-pure and prevents the ratio from being minimized by inflating the denominator via spurious surface statistics.

**Why this design.** We chose a learned linear-plus-gate module over OWP's fixed orthogonal projection because a fixed basis cannot adapt to the nonlinear entanglement introduced by Qwen2.5-Omni's end-to-end Thinker training, accepting the cost of a small amount of task-specific supervision. We chose counterfactual evidence-swap pairs over generic evidence-span annotations because swap pairs directly operationalize the invariance we need (evidence direction must move, prior direction must not), accepting the cost of constructing paired data. We chose a soft decorrelation penalty over a hard orthogonality constraint because enforcing exact orthogonality on a low-rank subspace over-constrains the evidence direction when the true factors are only approximately separable, accepting residual correlation as the price of preserving evidence fidelity. A domain expert would not describe this as a variant of OWP because OWP never trains a decomposition — the mechanism here is the supervised invariance objective, not the rebalancing step, which we reuse unchanged.

**Why this measures what we claim.** The quantity `||E(h_cf) - E(h)||^2 / ||E(h_pr) - E(h)||^2` measures evidence-prior separability because it assumes that counterfactual evidence swaps change only the grounding factor and prior-inducing perturbations change only the prior factor; this assumption fails when the swap inadvertently alters prior-relevant surface statistics (e.g., entity frequency), in which case the ratio reflects spurious surface sensitivity rather than factor separation. The gate `g = sigmoid(w_g . LN(h))` measures cross-modal conflict because it assumes conflict is linearly decodable from the fused state at layer L; this assumption fails when conflict is encoded nonlinearly across layers, in which case g reflects a layer-local proxy rather than global conflict. The decorrelation term `||E^T P||_F^2` measures factor independence because it assumes linear independence is sufficient for the additive decomposition to be identifiable; this fails when E and P share a nonlinear common mode, in which case the term reflects only second-order decorrelation.

## Contribution

(1) A learned evidence-prior decomposition module (EPD) with a counterfactual-swap disentanglement objective that enforces linear separability of grounding evidence and misleading prior in Qwen2.5-Omni's fused hidden states, replacing OWP's fixed orthogonal projection. (2) A design principle: evidence-direction identifiability under counterfactual evidence perturbation is the property projection-based correction implicitly requires, and it can be trained rather than assumed. (3) A reusable construction procedure for evidence-swap and prior-perturbation training triples from audio-visual hallucination benchmarks.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | CMM + AVHBench hallucination sets | Directly stress audio-visual conflict grounding |
| Primary metric | Hallucination rate on conflict subset | Isolates the predicted failure mode |
| Baseline | Unmitigated Qwen2.5-Omni-7B | Measures raw backbone hallucination floor |
| Baseline | OWP fixed orthogonal projection | Tests whether learned decomposition beats fixed basis |
| Baseline | Output-level post-hoc mitigation | Tests whether correction must be internal, not output-level |
| Ablation-of-ours | EPD without gate (g fixed 0.5) | Isolates contribution of conflict-conditioned rebalancing |
| Ablation-of-ours | EPD without purity filter (all pairs) | Tests whether swap-pair purity assumption is load-bearing |

### Why this setup validates the claim

The central claim is that a supervised, learned evidence-prior decomposition beats a fixed orthogonal projection precisely where evidence and prior are nonlinearly entangled, i.e. on conflict instances. CMM and AVHBench supply those conflict instances, so the conflict subset is where the mechanism must show its advantage; measuring hallucination rate there (rather than aggregate accuracy) directly detects the predicted effect. The unmitigated backbone establishes the floor any mitigation must beat, ruling out the possibility that reported gains are just backbone capability. OWP's fixed orthogonal projection is the decisive baseline: it shares our rebalancing step but lacks our learned invariance objective, so a gap between EPD and OWP isolates the decomposition-learning sub-claim from the rebalancing sub-claim. The output-level baseline tests the separate sub-claim that correction must operate on internal factors rather than on generated text. The gate-free ablation is the causal hinge: if the gate is doing real work, removing it should degrade conflict instances specifically while leaving non-conflict instances near parity. The no-purity-filter ablation directly tests the load-bearing assumption: if the purity filter is necessary for EPD's advantage, then removing it should reduce the gap over OWP, confirming that swap-pair purity is not a mere implementation detail but a requirement for the separability signal to be valid. Together these form a falsifiable test: the claim predicts a concentrated, subset-specific advantage, not a uniform lift.

### Expected outcome and causal chain

**vs. Unmitigated Qwen2.5-Omni-7B** — On a case where the audio track says "no" while the visual scene shows the named object present, the baseline follows the dominant prior (visual co-occurrence) and asserts the object, because its fused Thinker state carries no explicit separation between grounding evidence and learned prior. Our method instead down-weights the prior via a high conflict gate, so the rebalanced state is dominated by the evidence component, and we expect a clear hallucination-rate drop concentrated on conflict instances with near-parity on non-conflict instances, since the gate stays low there.

**vs. OWP fixed orthogonal projection** — On a case where the grounding cue is entangled with prior-relevant surface statistics (entity frequency, speaker accent), the fixed orthogonal basis cannot rotate to follow the entanglement, so it projects out part of the evidence direction and still hallucinates on the harder conflict cases. Our method instead learns the evidence direction from counterfactual swap pairs, so the evidence component moves under evidence swaps and stays put under prior perturbations, and we expect a moderate but consistent gap over OWP that widens on the higher-entanglement conflict subset while shrinking to parity on clean instances.

**vs. Output-level post-hoc mitigation** — On a case where the wrong entity is already committed early in decoding, the output-level baseline can only rewrite the surface string, so it corrects wording but not the underlying grounding error and often leaves the wrong entity intact. Our method instead suppresses the prior before decoding, so the wrong entity is never favored, and we expect a larger drop in entity-level hallucination than in fluency-style errors, with the advantage concentrated where the prior is strongly misleading.

**vs. EPD without gate (g fixed 0.5)** — On a case where conflict is present but the gate is disabled, the rebalancing applies a constant prior discount, so it under-corrects high-conflict cases and over-corrects low-conflict ones. Our method instead modulates the prior discount by the conflict gate, and we expect the gated version to beat the ungated ablation specifically on high-conflict instances while matching it on low-conflict ones, which is the signature that the gate encodes real conflict rather than a constant bias.

**vs. EPD without purity filter** — On a case where the swap pair inadvertently changes prior-relevant surface statistics (e.g., entity frequency), the ratio loss can be minimized by inflating the denominator (making E sensitive to prior perturbations) rather than by making E invariant, leading to a degenerate evidence direction that latches onto spurious correlates. Our method instead uses only pairs that pass the held-out purity filter, so the evidence direction is trained on factor-pure edits. We expect the no-filter ablation to show a smaller gap over OWP and higher variance across seeds, confirming that the purity filter is necessary for the separability signal to be valid.

### What would falsify this idea

If EPD's advantage over OWP is uniform across conflict and non-conflict subsets rather than concentrated on the predicted high-entanglement conflict cases, the claim that learned decomposition captures nonlinear entanglement is wrong. Likewise, if removing the gate leaves conflict-subset performance unchanged, the conflict-conditioned rebalancing mechanism is not doing the causal work we attribute to it. Finally, if the no-purity-filter ablation matches the full EPD, then the swap-pair purity assumption is not load-bearing and the ratio loss is measuring something other than factor separation.

## References

1. Omni-Woodpecker: Hallucination Correction via Evidence-Prior Rebalancing for Omni LLMs
2. Qwen2.5-Omni Technical Report
3. CosyVoice 2: Scalable Streaming Speech Synthesis with Large Language Models
4. MIO: A Foundation Model on Multimodal Tokens
5. F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching
