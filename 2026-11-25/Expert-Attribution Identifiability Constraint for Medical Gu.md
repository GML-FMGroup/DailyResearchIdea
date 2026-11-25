# Expert-Attribution Identifiability Constraint for Medical Guardrail Distillation

## Motivation

Distilling multiple medical capability experts into a single guardrail (as in LingSante MedGuard, E1) compresses expert-specific supervision into shared parameters, so the resulting explanation cannot be traced to which expert signal actually drove a risk decision; the explanation is faithful to the fused model but not identifiable with respect to expert provenance. This is a structural consequence of standard multi-teacher on-policy distillation, which optimizes student outputs to match teacher feedback without ever requiring that expert identity remain recoverable from the model's internal state. The same compression problem recurs across guardrail distillation pipelines (e.g., SafeWatch's policy-chunk encoding, SafetyAnalyst's weight aggregation), but MedGuard is the concrete instance where six heterogeneous medical experts are merged, making expert-provenance loss most consequential for clinical accountability.

## Key Insight

If each expert's supervision is routed through a distinct, sparse subspace of the student's residual stream and an attribution head is trained to recover the expert label from that subspace under a counterfactual expert-ablation intervention, then expert provenance remains identifiable after distillation because the counterfactual intervention breaks the confound between expert signal and shared task difficulty.

## Method

```pseudocode
# Phase 1: Expert-routed subspace construction
Input: guardrail teacher T, experts E1..E6 (numerical reasoning, multimodal hallucination, rule understanding, agentic safety, risk localization, entity understanding)
For each expert Ei: collect on-policy student responses Ri and expert feedback Fi
Initialize student S (MedGuard-style base) with residual stream dimension d (d=4096 for a 7B base; layer fixed at the final transformer block, layer L=32)
Assign each expert a fixed sparse mask Mi over d channels (disjoint support, |Mi| = d/12 = 341 channels, selected as top-k by mean |gradient| of the distillation loss on Fi, computed over the first 500 on-policy steps at layer L)
# Load-bearing assumption (stated explicitly): the disjoint mask Mi is assumed to be a sufficient statistic for expert Ei's supervision signal, so that zeroing h[Mi] is a valid expert-specific intervention rather than generic degradation. This assumption is validated post hoc in Phase 3 (disjointness probe + matched-control ablation), not asserted.

# Phase 2: Attribution head training
Attribution head A: 2-layer MLP (input 341 -> hidden 256, GeLU -> 6-way logits), applied to residual stream slice (Mi)
For each training step:
  h = S.residual(response) at layer L
  z_i = A(h[Mi])                       # expert-specific slice
  L_attr = CE(z_i, i)                  # must recover true expert label
  # Counterfactual expert ablation (matched-control):
  h_abl = h with h[Mi] zeroed
  h_rand = h with a random same-size slice (341 channels, disjoint from all Mj) zeroed
  h_other = h with h[Mj] zeroed for a randomly chosen j != i
  L_cf = CE(A(h_abl), i) + lambda * KL(S(h_abl) || S(h))   # lambda=0.5
  L_ctrl = KL(S(h_rand) || S(h)) + KL(S(h_other) || S(h))  # matched-control KLs
  L_sel = relu(L_ctrl - L_cf + margin)  # margin=0.1; requires expert-selective KL, not merely nonzero
  # Explanation consistency:
  explanation = S.explain(response)    # generated explanation tokens
  L_exp = -log P(explanation mentions expert Ei | z_i)  # via constrained decoding to expert-name tokens
  L_total = L_distill + alpha*L_attr + beta*(L_cf + L_sel) + gamma*L_exp  # alpha=1.0, beta=0.3, gamma=0.5

# Phase 3: Post-hoc validation of the disjoint-mask assumption (run once after Phase 2)
# (a) Disjointness probe: for each expert label Ei, train a linear probe on every other expert's mask support Mj (j != i); report cross-expert probe accuracy and overlap rate on MedGuardBench.
# (b) Matched-control ablation diagnostic: for each expert Ei, compare KL(S(h_abl_i)||S(h)) against KL(S(h_rand)||S(h)) and KL(S(h_abl_j)||S(h)); report the expert-selectivity ratio (KL_abl_i / KL_ctrl) per expert.
# (c) Residualized attribution target: regress expert label on task-difficulty/overlap features (response length, number of risk categories flagged, overlap flag from (a)), take residual as the attribution target for a secondary attribution head A_res; report attribution accuracy for both A and A_res.
```

**What it is**: EARL (Expert-Attribution Routing Loss) augments multi-teacher distillation with a sparse expert-subspace mask, an attribution head trained on counterfactual expert ablation with matched controls, and an explanation-consistency loss that forces generated explanations to name the causally responsible expert. Inputs: student responses, expert feedback, expert labels. Outputs: distilled guardrail with identifiable expert routing and attribution-consistent explanations, plus a post-hoc disjointness/selectivity report that quantifies whether the mask assumption holds.

**Why this design**: We chose fixed sparse disjoint masks over learned soft attention because soft routing collapses to a mixture that cannot support counterfactual ablation (zeroing a soft slice changes all experts simultaneously), accepting the cost that mask support must be pre-allocated and may underfit experts with overlapping feature needs. We chose a counterfactual ablation loss over a pure classification loss on the attribution head because classification alone can be gamed by shared task-difficulty features that correlate with expert labels (e.g., all hard cases get numerical-reasoning labels); ablation forces the head to rely on the expert-specific slice, accepting the cost of a second forward pass. We added matched-control ablations (random same-size slice, other-expert slice) because a nonzero ablation KL alone is consistent with generic degradation; the selectivity term L_sel requires the expert slice's KL to exceed the control KLs by a margin, accepting the cost of two additional forward passes per step. We chose constrained decoding for explanation consistency over post-hoc explanation editing because post-hoc editing decouples explanation from the model's actual computation, accepting the cost that the explanation vocabulary is restricted to expert-name tokens. A domain expert would not describe this as a variant of standard multi-teacher distillation (e.g., MedGuard's on-policy distillation) because the attribution head and counterfactual ablation introduce a causal-necessity requirement that standard distillation never enforces; nor is it a variant of SafetyAnalyst's interpretable weight aggregation, which interprets post-hoc harm-benefit trees rather than expert provenance during distillation.

**Why it measures what we claim**: The attribution head's accuracy on the held-out expert label measures expert-provenance identifiability because the sparse disjoint mask assumption makes the expert-specific slice a sufficient statistic for expert identity; this assumption fails when two experts supervise overlapping response types (e.g., a numerical error that is also a hallucination), in which case the head reflects the dominant expert's label rather than a unique cause — quantified by the Phase 3(a) cross-expert probe accuracy and addressed by the Phase 3(c) residualized target. The counterfactual ablation KL divergence measures causal necessity of the expert slice for the risk decision because zeroing the slice is a valid intervention on the expert signal under the assumption that the mask support is disjoint from other experts' supports; this assumption fails when the student's residual stream has cross-expert superposition, in which case the KL reflects general degradation rather than expert-specific causal contribution — which is why the matched-control term L_sel requires the expert-slice KL to exceed random-slice and other-expert-slice KLs by margin=0.1, and Phase 3(b) reports the selectivity ratio per expert. The explanation-consistency loss measures faithfulness of the explanation to the attributed expert because constrained decoding forces the explanation to name the expert that the attribution head identifies; this assumption fails when the attribution head is wrong, in which case the explanation is consistently wrong rather than unfaithful — reported as explanation-expert agreement conditional on attribution correctness.

## Contribution

['A distillation framework (EARL) that preserves expert-provenance identifiability by pairing fixed sparse expert-subspace masks with a counterfactual-ablation-trained attribution head.', 'A design principle: expert-attribution identifiability in distilled guardrails requires a causal-necessity signal (counterfactual ablation) rather than a correlational classification signal, because shared task difficulty confounds expert labels.', 'An explanation-consistency mechanism that constrains generated explanations to name the causally attributed expert, making explanation faithfulness checkable against the attribution head.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | LingSante MedGuardBench (11 risk-detection tasks); split into single-expert subset and overlapping-expert subset (cases flagged by Phase 3(a) cross-expert probe accuracy > 0.5 on a non-native mask) | Medical-domain risk tasks match guardrail target; the split directly tests the overlapping-supervision failure mode. |
| Primary metric | Expert-attribution accuracy on held-out expert labels, reported separately for single-expert and overlapping-expert subsets | Directly tests provenance identifiability claim and its stated failure mode. |
| Secondary metric | Expert-selectivity ratio = KL(S(h_abl_i)||S(h)) / mean(KL(S(h_rand)||S(h)), KL(S(h_abl_j)||S(h))) per expert | Tests whether ablation is expert-specific rather than generic degradation. |
| Secondary metric | Explanation-expert agreement (constrained-decoding match rate), conditional on attribution correctness | Tests faithfulness of explanation to attributed expert. |
| Diagnostic metric | Cross-expert linear-probe accuracy (probe for Ei trained on Mj, j != i) and overlap rate on MedGuardBench | Quantifies the disjointness assumption rather than asserting it. |
| Baseline | MedGuard-style multi-teacher distillation | Isolates value of attribution/ablation over standard distillation. |
| Baseline | SafetyAnalyst-style post-hoc harm-benefit moderation | Tests whether causal slice attribution beats post-hoc interpretation. |
| Baseline | Soft-routing mixture guardrail (learned attention) | Tests disjoint-mask necessity for counterfactual ablation. |
| Baseline | Learned sparse-routing guardrail (top-k gating with load-balancing loss, k=341) | Tests whether the counterfactual-ablation signal, not merely sparsity, drives attribution accuracy. |
| Baseline | Linear probe on frozen MedGuard-style student residual stream | Tests whether the attribution head + ablation training adds anything over a post-hoc probe. |
| Ablation-of-ours | EARL minus counterfactual ablation loss (L_cf + L_sel removed) | Isolates causal-necessity term from attribution classification. |
| Ablation-of-ours | EARL minus matched-control selectivity term L_sel (L_cf retained) | Isolates the selectivity requirement from the raw ablation KL. |
| Ablation-of-ours | EARL with residualized attribution target A_res (Phase 3c) | Tests whether provenance survives under superposition when task-difficulty is regressed out. |

### Why this setup validates the claim

The central claim is that a sparse disjoint expert-subspace plus a counterfactual-ablation-trained attribution head yields expert provenance that is both identifiable and causally necessary, and that explanations name that expert. MedGuard-style distillation is the right comparator because it shares the same teacher/expert supervision and student base, so any gain isolates the attribution machinery rather than data or capacity. SafetyAnalyst tests the competing interpretability route: post-hoc harm-benefit aggregation can score harm but cannot attribute which expert's signal drove the decision, so it should trail on attribution accuracy while possibly matching on raw risk classification. The soft-routing baseline tests the design's core wager: if learned soft routing also supports ablation, the disjoint-mask assumption is unnecessary and our rationale weakens. The learned sparse-routing baseline (top-k gating, k=341) and the linear-probe-on-frozen-student baseline separate the two ingredients the design bundles — sparsity and the counterfactual-ablation training signal — so that if attribution accuracy is driven by sparsity alone, the ablation term is doing no work. The ablation-of-ours removes only the counterfactual term, so if attribution accuracy is unchanged without it, the causal-necessity mechanism is doing no work; the L_sel-only ablation isolates the selectivity requirement from the raw KL. Expert-attribution accuracy is the right primary metric because it is the quantity the mask-and-ablation design is constructed to make identifiable; a benchmark-level F1 could improve for reasons unrelated to provenance. The Phase 3(a) cross-expert probe and Phase 3(b) selectivity ratio make the load-bearing disjointness assumption a measured quantity rather than an assertion, so the claim is falsifiable on its own terms.

### Expected outcome and causal chain

**vs. MedGuard-style multi-teacher distillation** — On a case where a numerical-reasoning error also looks like a hallucination, the baseline produces a correct risk label but no recoverable provenance, because its distillation objective never separates expert signals in the residual stream. Our method instead routes the response through the numerical-reasoning slice, and the attribution head recovers that label because the disjoint mask makes the slice a sufficient statistic. We expect a clear attribution-accuracy gap on the overlapping-expert subset, with near-parity on the single-expert subset.

**vs. SafetyAnalyst-style post-hoc harm-benefit moderation** — On a case where a compliance violation is flagged correctly, the baseline produces an interpretable harm tree that describes consequences but cannot name which expert's supervision caused the decision, because its weights aggregate effects rather than expert provenance. Our method instead ties the decision to the masked expert slice via the ablation KL and the matched-control selectivity term, so the explanation names the causally responsible expert. We expect SafetyAnalyst to match on risk classification but trail noticeably on attribution accuracy and explanation-expert agreement.

**vs. Soft-routing mixture guardrail** — On a case where zeroing the numerical slice should selectively degrade numerical reasoning, the baseline produces diffuse degradation across all expert behaviors, because its soft mixture spreads each expert's signal over shared channels. Our method instead confines the intervention to the disjoint support, so ablation degrades only the targeted expert's decisions. We expect our expert-selectivity ratio to be concentrated on the ablated expert's subset (ratio > 1 by margin) while the soft-routing baseline shows uniform degradation (ratio ≈ 1), a diagnostic separation rather than a headline gain.

**vs. Learned sparse-routing guardrail (top-k gating, k=341)** — On overlapping-expert cases, the learned sparse router may allocate channels by gradient magnitude without enforcing disjointness, so its supports may overlap and its attribution accuracy may approach ours on single-expert cases but degrade on overlapping cases. We expect our fixed-disjoint-mask attribution accuracy to exceed the learned-sparse baseline specifically on the overlapping-expert subset, and to be comparable on single-expert cases; if the learned-sparse baseline matches us on both, the disjointness constraint is not the active ingredient.

**vs. Linear probe on frozen MedGuard-style student** — A post-hoc linear probe on the frozen student's residual stream can recover expert labels only from whatever expert signal the distillation objective left in the shared representation. We expect the probe to trail our attribution head on the overlapping-expert subset, because our head is trained with the counterfactual-ablation signal that forces reliance on the expert slice; if the probe matches us, the ablation training signal is doing no work beyond what standard distillation already leaves in the residual stream.

### What would falsify this idea

If attribution accuracy is uniform across overlapping-expert and single-expert subsets, or if removing the counterfactual-ablation loss (or only the L_sel selectivity term) leaves attribution accuracy unchanged, then the causal-necessity mechanism is not what produces provenance and the central claim is wrong. Likewise, if the soft-routing baseline matches our expert-selectivity ratio pattern, or if the Phase 3(a) cross-expert probe shows near-chance accuracy on non-native masks (i.e., the disjointness assumption holds trivially), the disjoint-mask assumption is either unnecessary or untested and the rationale weakens.

## References

1. LingSante MedGuard Technical Report
