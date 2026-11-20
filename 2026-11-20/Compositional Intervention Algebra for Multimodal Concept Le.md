# Compositional Intervention Algebra for Multimodal Concept Lenses

## Motivation

MC-Lens (arxiv:2611.26590) validates that accessed concept information 'participates' in downstream computation only through single-site counterfactual edits of token-set readout carriers, so its intervention formalism implicitly assumes that concept edits are independent, additive perturbations. Structurally, this assumption cannot be certified by the formalism itself: because each edit is applied in isolation, any interaction term between two concepts is invisible to the measurement, which means the field's growing practice of ablating 'one concept at a time' to localize multimodal computation (echoed by the layer-wise single-modality flow findings of f88c232e59406b537070e70b004bd46d2cc2bb2f) rests on an untested additivity premise rather than a measured one.

## Key Insight

Simultaneous multi-carrier edits expose a second-order interaction tensor whose nonzero entries are exactly the concepts whose downstream roles are coupled, so additivity becomes a measurable property of the model rather than a silent assumption baked into the intervention design.

## Method

**What it is.** We introduce CIA (Compositional Intervention Algebra), a measurement framework that takes an MC-Lens-style vision-language model plus a set of K concept token-set readout carriers and returns, for every subset S of those carriers, the downstream effect of jointly editing S, decomposed into single-site main effects plus pairwise and higher-order interaction terms on a shared output-vocabulary semantic space. The load-bearing assumption, stated plainly: the Möbius interaction term I_S computed from joint edits of token-set readout carriers measures genuine coupling between concept roles in the VLM, rather than the nonlinearity/miscalibration of the lens readout and the non-commutativity of simultaneous activation edits — i.e., that the downstream readout is a smooth, well-calibrated function of the edited activations over the full range of the 2^K factorial design. This assumption is not left implicit: it is operationalized and bounded by the displacement-matched null and the calibration gate in PHASE 0 and PHASE 3b below.

**How it works.**
```pseudocode
INPUT: VLM M, MC-Lens lenses (vision encoder lens Lv, decoder lens Ld),
       concept set C = {c1..cK} each with token-set carrier T_ci,
       probe inputs D (image-question pairs), downstream readout R (final-token logits)

PHASE 0  Calibration gate (pilot, run before the 2^K design)
  # 0a. Reproduce MC-Lens single-site effects on a pilot probe set D_pilot (|D_pilot| = 64)
  for each ci in C, x in D_pilot:
      y_i[x] = R(M(x | T_ci <- c'_i));  delta_i[x] = y_i[x] - y0[x]
  check: sign and top-5 vocabulary support of delta_i match the MC-Lens paper's reported
         single-site effect for the same concept on the same model family (agreement >= 80% of probes)
  # 0b. Readout-fidelity vs edit magnitude curve (per concept, per layer)
  for edit magnitude m in {0.25, 0.5, 0.75, 1.0, 1.5, 2.0} * ||c'_i - c_i||:
      measure single-site readout fidelity = cosine(Ld(h_edited), Ld(h_clean)) at matched layer
  cap c'_i at the largest m whose fidelity >= 0.95 (pre-registered threshold); record m_cap(ci)
  # 0c. Pre-register contrast c'_i: one fixed alternate concept per ci, chosen as the
  #      nearest neighbor in the decoder output-vocabulary semantic space that is NOT a
  #      synonym/hypernym of c_i; sensitivity check over a second contrast c''_i (next-nearest)

PHASE 1  Baseline and single-site effects
  for each x in D:
      y0[x] = R(M(x))                                  # unedited reference
      for each ci in C:
          y_i[x] = R(M(x | T_ci <- c'_i))              # counterfactual edit, c'_i = alternate concept, magnitude capped at m_cap(ci)
          delta_i[x] = y_i[x] - y0[x]                   # main effect vector

PHASE 2  Multi-site joint edits (full 2^K design, K <= 6)
  for each subset S of C with |S| >= 2:
      y_S[x] = R(M(x | {T_ci <- c'_i for ci in S}))     # simultaneous edits, each at its m_cap
      delta_S[x] = y_S[x] - y0[x]

PHASE 3  Interaction decomposition (Mobius / inclusion-exclusion)
  for each S:
      I_S[x] = delta_S[x] - sum_{T proper subset of S} I_T[x]   # I_{ci} = delta_i
      # I_{ci,cj}[x] is the pairwise interaction vector
  aggregate: mean and bootstrap CI over D per (S, output-dim)

PHASE 3b  Displacement-matched null (curvature correction)
  # Construct a matched null by applying the SAME multi-site edit magnitudes to
  # semantically inert / orthogonal carrier directions (or to shuffled carriers at
  # matched activation displacement ||delta_h||_2 within 5%).
  for each S, x:
      y_S^null[x] = R(M(x | {T_ci <- c'_i^null for ci in S}))   # c'_i^null = orthogonal direction at matched ||delta_h||
      I_S^null[x] = Mobius residual of y_S^null (same estimator as PHASE 3)
  report I_S^excess[x] = I_S[x] - mean_x'(I_S^null[x'])   # excess over measured readout nonlinearity

PHASE 4  Additivity certification
  additivity_ratio(S) = ||sum_{ci in S} delta_i|| / ||delta_S||
  interaction_norm(S) = ||I_S^excess||
  flag S as NON-ADDITIVE if bootstrap CI of interaction_norm excludes 0 AND
       interaction_norm(S) > 2 * ||I_S^null||_mean (displacement-matched null band)

PHASE 5  Compositional lens report
  return {main effects, interaction tensor I_S^excess, additivity_ratio per S,
          per-layer localization of nonzero I_S^excess, m_cap per concept,
          null-band width per S}
```
Hyperparameters: K ≤ 6 (full factorial, 64 subsets max per probe); bootstrap resamples B = 1000; alternate concept c'_i drawn from the decoder output vocabulary via the lens's semantic space (one fixed contrast per concept, pre-registered, with a second-contrast sensitivity check); edit magnitude capped at m_cap(ci) from PHASE 0b; displacement-matched null uses orthogonal carrier directions at ||delta_h|| within 5% of the real edit.

**Why this design.** We chose a full factorial 2^K design over a random subset of edits because the Mobius decomposition requires every subset to estimate each interaction term without confounding, accepting the cost that K is capped at roughly 6 concepts per experiment. We chose the Mobius inclusion-exclusion estimator over a learned interaction head because the interaction terms must be functions of the intervention itself, not of a trained proxy, accepting the cost that we cannot scale to large concept vocabularies in one pass. We chose to keep MC-Lens's token-set readout carriers rather than single name tokens because the prior work explicitly extended carriers to token sets to capture concept information, accepting that our interaction estimates inherit whatever noise the token-set carrier carries. We added the PHASE 0 calibration gate and the PHASE 3b displacement-matched null because the reviewer's adversarial alert identifies the load-bearing assumption — that I_S measures concept coupling rather than lens miscalibration — as unverified; the gate caps edits to the regime where single-site readout fidelity is ≥ 0.95, and the null subtracts the curvature that the same edit magnitudes produce on semantically inert directions, so I_S^excess is a contrast against measured readout nonlinearity rather than an absolute residual. A domain expert would not describe this as a variant of standard ablation or of MC-Lens: MC-Lens performs independent single-site counterfactuals, whereas CIA's load-bearing object is the off-diagonal interaction tensor I_{ci,cj}^excess, which single-site ablation cannot even represent.

**Why it measures what we claim.** The Mobius interaction term I_S measures the non-additivity of concept roles because it is defined as the residual of the joint edit after removing all lower-order effects, under the assumption that the downstream readout is a smooth function of the edited activations so that the Mobius expansion is exact; this assumption fails when the edit pushes the model off the manifold where the lens is calibrated, in which case I_S reflects lens miscalibration rather than genuine concept coupling — which is why PHASE 3b reports I_S^excess as the excess over a displacement-matched null and PHASE 0b caps edits at m_cap where single-site readout fidelity is ≥ 0.95. The additivity_ratio measures whether single-site ablations can be composed because it compares the norm of the summed main effects to the joint effect, under the assumption that the output-vocabulary semantic space is metrically meaningful across edits; this assumption fails when two concepts map to overlapping vocabulary regions, in which case the ratio reflects vocabulary collision rather than interaction. The bootstrap CI flag measures statistical reliability of non-additivity because it captures sampling variance over D, under the assumption that D is representative of the task distribution; this assumption fails under distribution shift, in which case the flag reflects probe-set idiosyncrasy. The displacement-matched null band operationalizes the bound on the failure mode: any I_S^excess inside 2× the null band is attributed to readout curvature or vocabulary collision, not to concept coupling.

## Contribution

(1) A compositional intervention algebra (CIA) that extends MC-Lens-style single-site counterfactual editing to simultaneous multi-carrier edits with an explicit Mobius decomposition into main effects and pairwise/higher-order interaction terms. (2) A measured additivity certification procedure with bootstrap confidence intervals that converts the previously assumed independence of concept edits into a testable property of a given vision-language model. (3) An interaction tensor report that localizes which concept pairs have coupled downstream roles and at which model layers the coupling emerges.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Visual question answering image-question pairs (LLaVA-style probes); pilot set |D_pilot| = 64, main set |D| = 512 | Matches prior cross-modal flow and lens work; pilot gates calibration |
| Primary metric | Interaction-norm ||I_S^excess|| vs displacement-matched null band, per concept subset | Directly quantifies non-additivity claim against measured curvature |
| Baseline 1 | Independent single-site MC-Lens counterfactuals | Tests whether composition is needed at all |
| Baseline 2 | Additive reconstruction from main effects only | Tests whether off-diagonal terms carry signal |
| Baseline 3 | Random concept-pair edits (null interaction) | Calibrates what spurious interaction looks like |
| Baseline 4 | Shapley interaction indices on the same carriers (K ≤ 6, exact enumeration) | Tests whether Mobius tensor reveals structure Shapley cannot |
| Ablation-of-ours | CIA with shuffled carrier-to-concept assignment | Isolates whether coupling is concept-specific |
| Ablation-of-ours | CIA without PHASE 3b displacement-matched null (raw Mobius residual) | Isolates curvature correction contribution |
| Synthetic ground-truth | Controlled model with injected known pairwise interaction (analytic I_S) | Bounds off-manifold miscalibration confound |

### Why this setup validates the claim

The central claim is that concept roles in VLMs are non-additive and that this non-additivity is measurable via a Mobius interaction tensor over token-set carriers, with the load-bearing assumption that I_S^excess reflects genuine coupling rather than lens miscalibration. The independent single-site baseline tests the sub-claim that composition is necessary: if single-site effects summed reconstruct joint edits, the interaction tensor is trivially zero and the framework is unneeded. The additive-reconstruction baseline tests whether the off-diagonal terms carry genuine signal beyond main effects. The random-pair baseline calibrates the null: it tells us what interaction magnitude arises from lens noise and vocabulary collision alone, so any excess in real concept pairs is attributable to coupling. The Shapley-interaction baseline tests whether CIA's Mobius tensor reveals structure that an existing interaction-attribution method cannot, establishing a genuine mechanistic delta rather than a re-labeling. The shuffled-carrier ablation tests concept-specificity: if interaction norms persist under shuffled assignments, the effect is a carrier artifact, not concept structure. The no-null ablation isolates the contribution of the curvature correction. The synthetic ground-truth test bounds the off-manifold miscalibration confound by reporting recovery error of I_S as a function of edit magnitude on a model where the true pairwise interaction is known analytically. The interaction-norm metric is right because it is exactly the quantity the claim asserts is nonzero and structured, so it detects the predicted effect directly rather than through a downstream proxy.

### Expected outcome and causal chain

**vs. Independent single-site MC-Lens counterfactuals** — On a case where two concepts jointly determine the answer (e.g., object identity and its attribute both feed the prediction), the baseline produces a prediction from summing independent edits because it never models the joint edit, so it cannot represent the interaction at all. Our method instead estimates delta_S for the joint edit and subtracts all lower-order terms, exposing the residual coupling. We expect the additive reconstruction to show a visible error concentrated on concept pairs whose joint edit diverges from the sum, with parity on pairs that are genuinely independent.

**vs. Additive reconstruction from main effects only** — On a case where concept edits interact super-additively, the additive baseline predicts the joint effect as the sum of main effects because its mechanism assumes separability. Our method's Mobius term captures the residual, so we expect the interaction-norm to be markedly above the random-pair null specifically on coupled pairs, and near the null on decoupled pairs — a structured, not uniform, gap.

**vs. Random concept-pair edits (null interaction)** — On a case where concept pairs are semantically unrelated, the random baseline produces interaction magnitudes driven by lens miscalibration and vocabulary collision because it has no real coupling to measure. Our method should show real concept pairs exceeding this null floor while unrelated pairs sit at it, so we expect a bimodal separation: coupled pairs above the null band, decoupled pairs inside it.

**vs. Shapley interaction indices on the same carriers** — On a case where three or more concepts interact, Shapley interaction indices distribute the higher-order effect across pairs because they are defined as marginal contributions over coalitions, whereas the Mobius tensor isolates the exact subset S. We expect CIA to show nonzero I_S^excess on the specific triple that Shapley smears across its pairwise terms, with the two methods agreeing on pure pairwise cases — a difference in localization, not in overall magnitude.

**vs. CIA without PHASE 3b displacement-matched null** — On a case where the edit magnitude is near m_cap, the raw Mobius residual includes readout curvature because it subtracts only lower-order edits, not the nonlinearity of the lens at that displacement. Our method's excess-over-null should shrink the interaction norm on decoupled pairs toward zero while preserving it on coupled pairs, so we expect the no-null ablation to show inflated interaction norms on the random-pair null and the full method to close that gap.

**vs. Synthetic ground-truth model** — On a controlled model with an injected known pairwise interaction, the recovery error of I_S should increase with edit magnitude because off-manifold drift grows; we expect the error to stay below 10% of the true interaction norm for edits within m_cap and to rise sharply beyond it, empirically bounding the regime where CIA's estimates are trustworthy.

### What would falsify this idea

If interaction norms for real concept pairs were statistically indistinguishable from the shuffled-carrier ablation and the random-pair null across all subsets, or if the gain over additive reconstruction were uniform rather than concentrated on predicted coupled pairs, or if the synthetic ground-truth recovery error exceeded the null band within the m_cap regime, the claim that concept roles are non-additive and measurable would be wrong.

## References

1. Probing the "Conscious" Mind: Selective Access and Global Workspace in Vision-Language Models
