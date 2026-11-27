# Overlap-Monotone Confidence: Coupling Anatomical Grounding and Calibrated Confidence via Counterfactual Region Identifiability

## Motivation

Medical vision-language models increasingly localize evidence (MIMO's pixel grounding, MedRegA's region-centric tasks, Radev's anatomical-bounds selection) and separately emit confidence, but the two outputs are produced by disjoint heads with no shared constraint, so a model can be highly confident while its selected region is spatially misgrounded. Radev's counterfactual-calibrated training already intervenes on organ representations to shape confidence, yet it never forces confidence to be a function of whether the *selected* region is the causally relevant one — the grounding head and confidence head can drift independently. This is the meta-level gap: across VILA-M3, MIMO, MedRegA and Radev, region selection is never tied to a calibrated confidence that the selected region is the causally necessary evidence.

## Key Insight

If confidence is trained to be monotone in the measured overlap between the model's selected evidence region and the region whose removal actually degrades the decision, then a spatially misgrounded prediction is structurally forced to carry low confidence, because the overlap term that gates confidence is computed from the same causal intervention that defines correctness of grounding.

## Method

**What it is.** We propose OMIC (Overlap-Monotone Identifiability Coupling), a training objective and inference head that take a volumetric medical scan plus a clinical query and output (i) an anatomical-bounds evidence region, (ii) a candidate decision, and (iii) a confidence that is monotone (in the training target) in the overlap between the selected region and a counterfactually validated causal region.

**Load-bearing assumption (stated explicitly).** The method assumes that the decision's causal support is identifiable from organ-level counterfactual interventions with a fixed threshold tau, i.e., that the set of organs whose removal changes the decision distribution is well-defined and stable enough to serve as a hard ceiling on confidence. This assumption fails under organ redundancy or interaction: if two organs are redundant (removing either alone leaves the decision unchanged) or synergistic (neither alone exceeds tau but their joint removal does), then marginal single-organ ablation under-covers the causal support and the resulting r_star is mis-specified. We therefore compute r_star from a redundancy-robust coalition score (below) and add an explicit 'unidentifiable' abstain state when no coalition passes tau.

**How it works.**

```pseudocode
Input: scan x (3D CT), query q, candidate set C, frozen 3D organ encoder E_3d, 2D slice encoder E_2d
Hyperparameters: tau=0.1 (KL threshold), coalition size K=3, Shapley samples M=32, alpha=4.0, beta=-2.0 (shared grid), lambda_1=1.0, lambda_2=0.5

# Phase 1: grounding head (inherited from Radev-style anatomy-guided selection)
b_hat = GroundingHead(q, E_3d(x))          # predicted anatomical bounds, organ-level
v     = CrossAttend(q, E_3d(x)[b_hat], E_2d(x)[b_hat])   # compact visual tokens
p     = LanguageBackbone(v, q)             # candidate distribution over C, single forward pass

# Phase 2: redundancy-robust causal-region construction (training only)
# Cache frozen-encoder organ features once per volume: f_o = E_3d(x)[o] for o in OrganSet(x)
# For each organ coalition S (|S| <= K), compute p_minus_S via masked attention over cached organ tokens
# (no full re-forward pass; masked attention over cached f_o).
for each coalition S in SampledCoalitions(OrganSet(x), K, M):
    p_minus_S = ForwardMasked(cached_features, mask=S, q)   # same frozen pipeline, masked organ tokens
    delta_S   = KL(p || p_minus_S)                          # decision degradation attributable to coalition S
# Shapley-style marginal contribution per organ:
phi_o = (1/M) * sum_{S not containing o} [delta_{S union {o}} - delta_S] / C(|OrganSet|-1, |S|)
r_star = { o : phi_o > tau }                  # causal region: organs with positive Shapley contribution above tau
if r_star is empty:                           # unidentifiable state
    r_star = None                             # ceiling defaults to low-confidence band (see Phase 3)

# Phase 3: shared identifiability objective
if r_star is not None:
    overlap = IoU(b_hat, r_star)              # spatial overlap between selected and causal region
    ceiling = sigma(alpha * overlap + beta)   # monotone ceiling on confidence
else:
    overlap = 0.0                             # unidentifiable -> no causal support
    ceiling = sigma(beta)                     # low-confidence band (sigma(-2.0) ≈ 0.12)
c_hat   = ConfidenceHead(v, b_hat)            # 2-layer MLP, hidden=256, GeLU, sigmoid output
L_mono  = max(0, c_hat - ceiling)^2           # c_hat must not exceed the overlap-implied ceiling
L_conf  = BCE(c_hat, Correct(p, label))       # standard calibration
L_ground= Dice(b_hat, r_star) if r_star is not None else 0.0   # grounding supervision from causal region
L_total = L_conf + lambda_1 * L_mono + lambda_2 * L_ground   # lambda_1=1.0, lambda_2=0.5

# Inference: c_hat is emitted alongside b_hat; no intervention needed at test time
# Inference-time diagnostic: recompute overlap on a held-out intervention set (512 volumes)
# to verify the ceiling is binding (mean c_hat <= mean ceiling) rather than vacuous.
```

**Why this design.** We chose a hard monotonicity ceiling `sigma(alpha*overlap+beta)` over a learned auxiliary consistency loss because a learned loss can be satisfied by the confidence head ignoring overlap entirely (the two heads remain free to drift), whereas a ceiling makes misgrounding *structurally* incompatible with high confidence, accepting the cost that the ceiling parameters alpha, beta must be tuned per modality (we pre-register a single shared grid alpha in {2,4,8}, beta in {-2,-1,0} and report a sensitivity plot rather than per-modality tuning). We chose to compute `r_star` from Shapley-style coalition ablation over cached organ features rather than from marginal single-organ removal because marginal removal cannot identify causal necessity under redundancy (Lundberg & Lee, 2017; Hooker et al., 2021) and produces off-manifold volumes; the coalition score is redundancy-robust and the masked-attention implementation avoids O(|OrganSet|) full re-forward passes by caching frozen-encoder organ features once per volume. We chose to supervise grounding with Dice against `r_star` (a causal target) rather than against human bounding-box annotations because annotations encode what radiologists *look at*, not what the decision *depends on*, accepting noisier targets when the organ set is coarse. We chose a single-forward-pass confidence head over autoregressive token log-probs because log-probs are miscalibrated for decision correctness, accepting that the head must be trained jointly rather than read off for free. This differs from Radev's CCT: CCT shapes confidence via intact-vs-intervened *representation* comparisons but never constrains the grounding head, and it is not a variant of CCT because the coupling term is a spatial-overlap ceiling linking two heads that CCT treats as independent.

**Why it measures what we claim.** The quantity `overlap = IoU(b_hat, r_star)` measures *causal necessity of the selected region* because we assume that the Shapley-style coalition score phi_o > tau identifies the organs the decision causally depends on under redundancy; this assumption fails when the coalition sample size M is too small or when interactions exceed the coalition size K, in which case `r_star` under-covers and overlap reflects only the sampled non-redundant subset. The ceiling `sigma(alpha*overlap+beta)` measures *confidence identifiability* because we assume confidence should not exceed the degree to which the evidence region is causally validated; this assumption fails when the model's decision is correct for a reason outside the organ set (e.g., a scanner artifact), in which case the ceiling penalizes a correct-but-differently-grounded prediction, and it also fails under train-to-test distribution shift in organ segmentation or query mix, in which case b_hat drifts while alpha, beta stay fixed and the ceiling may become vacuous (we therefore add the held-out intervention diagnostic). `L_conf = BCE(c_hat, Correct(p,label))` measures *empirical calibration* because we assume the held-out correctness label is the target of confidence; this fails under label noise, in which case c_hat tracks annotator agreement rather than decision reliability. The Dice term measures *grounding fidelity to causal evidence* under the same redundancy assumption as overlap; when it fails, Dice rewards matching a partial causal region and under-penalizes over-selection.

## Contribution

(1) A joint training objective and inference head (OMIC) that enforces a monotonicity ceiling linking an anatomical-bounds grounding head to a confidence head via a counterfactually constructed causal region, making high confidence structurally incompatible with spatially misgrounded evidence. (2) A design principle — overlap-monotone confidence — showing that grounding and calibration must share a single identifiability constraint rather than being trained as separate heads, instantiated as a ceiling function whose parameters are modality-tunable. (3) A counterfactual causal-region construction procedure (organ-removal KL attribution with threshold tau) that supplies grounding supervision without requiring radiologist bounding boxes.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | 3D CT with organ-level counterfactual labels (Radev-style), plus a chest X-ray subset with region-level counterfactual masks from MIMIC-CXR | 3D CT is the only setting where causal region r_star is constructible via coalition ablation; MIMIC-CXR tests whether the ceiling transfers to a second modality |
| Primary metric | Selective-prediction risk at matched coverage (coverage levels 50%, 70%, 90%) | Directly tests whether confidence tracks causal grounding |
| Secondary metric | ECE (15 bins) and Brier score on held-out set | Tests calibration quality alongside selective risk |
| Baseline | Radev / CCT decision-native model | Tests whether overlap-ceiling coupling beats representation-only calibration |
| Baseline | MedRegA-style attention grounding | Tests whether causal targets beat attention saliency |
| Baseline | Standard post-hoc calibration (temperature scaling, ECE) | Tests whether coupling beats read-off confidence |
| Ablation-of-ours | OMIC without L_mono ceiling | Isolates the structural coupling term |
| Ablation-of-ours | OMIC with marginal single-organ ablation (original r_star) | Tests whether redundancy-robust coalition score is necessary |
| Diagnostic | Held-out intervention set (512 volumes) recomputing overlap | Verifies ceiling is binding (mean c_hat <= mean ceiling) rather than vacuous |

### Why this setup validates the claim

The central claim is that confidence should be structurally capped by the causal validity of the selected evidence region, so misgrounded-but-confident predictions become impossible rather than merely discouraged. The dataset choice is forced: only a volumetric setting with organ-level interventions lets us construct `r_star` and thus compute `overlap` at all; the MIMIC-CXR subset tests whether the same ceiling transfers to a second modality without per-modality tuning. Selective-prediction risk at matched coverage is the right metric because it conditions on confidence — if our ceiling works, the high-confidence stratum should be enriched in causally grounded cases, so risk at fixed coverage should drop without sacrificing coverage. Each baseline isolates one sub-claim: Radev/CCT tests whether a representation-level calibration term suffices when the grounding head is left unconstrained; MedRegA tests whether attention-derived grounding is a viable causal target; post-hoc calibration tests whether the coupling adds anything beyond rescaling log-probs. The ablation (no ceiling) is the critical control — it holds the causal-target supervision fixed while removing only the monotonicity link, so any gap between full OMIC and the ablation is attributable to the coupling itself, not to better grounding targets. The marginal-ablation ablation tests whether the redundancy-robust coalition score is necessary for the ceiling to be well-defined. The held-out intervention diagnostic directly checks the train-to-test stationarity assumption: if mean c_hat exceeds mean ceiling on held-out interventions, the ceiling is vacuous and the coupling claim is unsupported.

### Expected outcome and causal chain

**vs. Radev / CCT** — On a case where the grounding head selects a non-causal organ (e.g., a redundant organ whose removal leaves the decision unchanged) but the decision is nonetheless correct, the baseline produces high confidence because its CCT term only compares intact-vs-intervened representations and never penalizes the grounding head for selecting the wrong region. Our method instead caps confidence at `sigma(alpha*overlap+beta)`, which is low when `b_hat` misses `r_star`, so confidence is dragged down on exactly these misgrounded-but-correct cases. We expect a concentrated risk reduction in the high-confidence stratum on the redundant-organ subset, with near-parity on single-causal-organ cases.

**vs. MedRegA-style attention grounding** — On a case where the model attends to a salient but non-necessary structure (e.g., a bright artifact-adjacent organ that draws attention without driving the decision), the baseline produces a grounding region aligned with saliency and thus a confidence that is high despite the region not being causally necessary. Our method instead supervises `b_hat` with Dice against the counterfactually validated `r_star`, so the grounding head is pulled toward causal organs and the overlap ceiling stays high only when it succeeds. We expect a larger gap on cases with salient distractors and parity on clean single-organ cases.

**vs. Standard post-hoc calibration** — On a case where the model is correct for a reason outside the organ set (e.g., scanner artifact), the baseline produces a well-calibrated confidence by rescaling log-probs, but that confidence carries no information about whether the evidence region is causally valid — it tracks correctness, not grounding. Our method instead ties confidence to overlap, so on artifact-driven-correct cases it will *under*-confidently flag them (low overlap, low ceiling), which is the intended behavior. We expect the ablation to match post-hoc calibration on overall ECE but to lose the high-confidence-stratum risk advantage, confirming the ceiling — not the calibration term — drives the effect.

**vs. OMIC with marginal single-organ ablation** — On a case with redundant organs (e.g., contralateral kidney hypertrophy), the marginal-ablation variant under-covers `r_star` because neither organ alone exceeds tau, so overlap is low even when the grounding head selects the correct organ, and the ceiling incorrectly suppresses confidence. Our method instead uses the Shapley-style coalition score, which identifies the redundant pair as jointly causal, so overlap stays high and confidence is not spuriously suppressed. We expect the marginal-ablation variant to show a larger risk gap on the redundant-organ subset and to underperform full OMIC on selective-prediction risk at 70% coverage.

### What would falsify this idea

If the full OMIC model's selective-prediction gain over the no-ceiling ablation is uniform across the redundant-organ, distractor, and clean subsets rather than concentrated where the predicted misgrounding failure mode occurs, then the ceiling is not doing the causal work we claim and the central coupling argument is wrong. A second falsifier: if confidence under OMIC fails to drop on artifact-driven-correct cases (where overlap is low by construction), or if the held-out intervention diagnostic shows mean c_hat exceeding mean ceiling, the ceiling is not actually binding and the monotonicity claim is vacuous.

## References

1. Decision-Native Multimodal Clinical Modeling via Anatomical Grounding and Confidence Calibration
