# Geometry-Entailment Verification for 3D Vision-Language Spatial Reasoning

## Motivation

Current 3D vision-language models such as StereoVLM fuse geometry-derived tokens into a multimodal LLM that can still answer spatial questions from semantic priors alone, because no component checks whether the answer is entailed by the model's own estimated 3D geometry. This structural gap means a correct-looking answer may be semantically plausible but geometrically unsupported, and at inference there is no signal distinguishing geometry-grounded from prior-driven answers. The problem recurs across 3D-VL architectures that add geometric encoders without a consistency check between the reasoning head and the geometric evidence.

## Key Insight

A spatial answer is geometry-grounded exactly when it is the unique answer consistent with the model's estimated depth/disparity geometry under a differentiable rendering of the scene, so training a verifier on counterfactual geometry perturbations yields a decision boundary that semantic priors cannot cross.

## Method

**What it is.** GEVer (Geometry-Entailment Verifier) takes a 3D-VL model's spatial-reasoning answer, its intermediate geometric estimate (per-pixel depth/disparity plus object positions), and the question, and outputs an entailment score in [0,1] plus a gradient that penalizes answers not entailed by the geometry. It is trained jointly with the reasoning head via a differentiable consistency loss. The load-bearing assumption, stated plainly: the geometry head's own depth/disparity estimate is faithful enough that predicate truth rendered from it is a valid entailment target, and the parsed predicate faithfully captures the answer's spatial content. GEVer does not assume this silently — it measures both and gates the loss on them (steps 2b and 3b below).

**How it works.**
```pseudocode
# Inputs: question q, image/stereo pair I, model M with reasoning head R and geometry head G
# Hyperparams: lambda_cons=0.5, tau=0.1, K=4 counterfactual perturbations, margin=0.3,
#              N_geo=3 geometry-head ensemble members, gamma_parse=0.6, gamma_stab=0.7

def GEVer(q, I, M):
    # 1. Geometry ensemble: N_geo=3 independently-perturbed geometry heads
    #    (same architecture, different init / dropout masks / input crops)
    geo = [G_n(I) for n in 1..N_geo]          # each yields depth D_n, positions P_n
    D, P = mean_depth(geo), mean_positions(geo)
    # 2. Parse answer into a spatial relation predicate over objects in P
    pred, c_parse = parse_spatial_predicate(a_hat, P)   # c_parse in [0,1] = parser confidence
    # 2b. Parser-fidelity gate: exclude vague/ambiguous parses from the entailment target
    if c_parse < gamma_parse: return a_hat, None, 0.0   # no consistency loss on this instance
    # 3. Differentiable geometry renderer: predicate truth from D,P
    t_geo = render_predicate_truth(pred, D, P)          # in [0,1], differentiable in D,P
    # 3b. Cross-ensemble stability gate: truth must be stable across the N_geo heads
    t_ens = [render_predicate_truth(pred, D_n, P_n) for n in 1..N_geo]
    stab = 1 - std(t_ens) / (mean(t_ens) + 1e-6)
    if stab < gamma_stab: return a_hat, None, 0.0       # geometry head unreliable here
    # 4. Counterfactual geometry perturbations: K=4 random rigid transforms of P
    #    and corresponding depth shifts; recompute truth under each
    t_cf = [render_predicate_truth(pred, D_k, P_k) for k in 1..K]
    # 4b. Perturbation-validity audit: fraction of K that actually flip the relation
    flip_rate = mean(1[t_cf_k < 0.5] for k in 1..K)     # logged, not backpropagated
    # 5. Entailment score: answer must be true in original geometry and
    #    flip under perturbations that change the relation
    s_ent = sigmoid((t_geo - mean(t_cf)) / tau)
    # 6. Consistency loss (differentiable) between reasoning head logits and s_ent
    L_cons = max(0, margin - (s_ent - s_sem))           # s_sem = reasoning head confidence
    # 7. Verifier training: reject answers supported only by semantic priors
    L_ver = BCE(s_ent, label)  # label=1 if t_geo high & stable, 0 if t_cf high
    return a_hat, s_ent, L_cons + lambda_cons * L_ver
```

**Why this design.** We chose a differentiable renderer over a learned classifier because the renderer's output is an explicit function of the geometric estimate, so the consistency loss propagates gradients into both the geometry head and the reasoning head; a learned classifier would collapse onto the same semantic-prior shortcut it is meant to detect. We chose counterfactual geometry perturbations over a single forward check because entailment requires that the answer be sensitive to the geometry that supposedly supports it — a semantically plausible answer that stays true under all perturbations is flagged as prior-driven. We chose to train the verifier jointly with the reasoning head rather than post-hoc because post-hoc calibration cannot change the reasoning head's behavior, accepting the cost that joint training may destabilize early optimization (mitigated by the margin-based loss). We added the N_geo=3 ensemble and the two gates (c_parse ≥ 0.6, stab ≥ 0.7) because the renderer inherits the geometry head's errors: if the depth map is itself prior-contaminated, the counterfactual gap would measure noise in G rather than grounding in R, and gating on parser confidence and cross-ensemble stability excludes exactly those instances from the entailment target. A domain expert would not describe this as a variant of StereoVLM: StereoVLM adds a geometry encoder but never checks answer-geometry consistency, whereas GEVer's core mechanism is the counterfactual entailment test that StereoVLM lacks.

**Why it measures what we claim.** The entailment score `s_ent` measures geometry-grounding because the differentiable renderer's predicate truth is equivalent to logical entailment under the assumption that the parsed spatial predicate faithfully captures the answer's spatial content; this assumption fails when the answer uses vague or context-dependent language (e.g., "near"), in which case `s_ent` reflects parser ambiguity rather than grounding — which is why we gate on c_parse ≥ 0.6 and report s_ent conditioned on parser confidence. The counterfactual gap `t_geo - mean(t_cf)` measures prior-independence because a geometry-grounded answer must flip when the supporting geometry is perturbed, under the assumption that the perturbations change the relevant spatial relation without changing semantics; this fails when perturbations accidentally preserve the relation, in which case the gap underestimates grounding — which is why we log flip_rate and report s_ent conditioned on it. The cross-ensemble stability `stab` measures whether the geometry head's estimate is reliable enough to serve as an entailment target because independent geometry heads should agree where the geometry is visually determined; this fails when all heads share the same prior contamination, in which case stab is high but uninformative, and we report the fraction of instances passing the stab gate. The consistency loss `L_cons` measures the alignment between the reasoning head's confidence and geometric entailment because the margin enforces that semantic confidence cannot exceed entailment by more than `margin`; this fails when the reasoning head's confidence is uncalibrated, in which case the loss reflects calibration error rather than grounding.

## Contribution

['A geometry-entailment verifier (GEVer) that uses a differentiable predicate renderer and counterfactual geometry perturbations to produce a training signal distinguishing geometry-grounded from semantically-prior-driven spatial answers.', "A joint training objective that couples the reasoning head's confidence to the verifier's entailment score via a margin-based consistency loss, making the two quantities distinguishable at inference.", "An inference-time entailment score that can be used to reject or flag answers not supported by the model's estimated 3D geometry."]

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Beacon3D object-centric 3D-VL benchmark | Object-centric tests isolate grounding from language variation |
| Dataset (transfer) | VSI-Bench (metric-spatial QA, as used by VG LLM) | Tests whether verifier transfers beyond Beacon3D |
| Primary metric | Grounding-QA coherence under counterfactual geometry swaps | Directly detects prior-driven versus geometry-grounded answers |
| Secondary metric | s_ent conditioned on parser confidence (c_parse ≥ 0.6) and flip_rate | Ties grounding claim to measured quantities |
| Secondary metric | Perturbation-validity rate (fraction of K=4 that flip the relation) | Audits the counterfactual assumption |
| Baseline | StereoVLM (geometry encoder, no consistency check) | Tests whether geometry encoding alone suffices |
| Baseline | 2D-only cdViews / LLaVA-OV zero-shot | Tests whether semantic prior alone suffices |
| Baseline | Vanilla reasoning head without verifier | Tests whether joint verifier training is the active ingredient |
| Baseline | Learned entailment classifier trained on same counterfactual labels | Tests whether renderer mechanism, not supervision, drives gain |
| Baseline | Post-hoc calibration of reasoning-head confidence | Tests whether joint training is necessary |
| Ablation-of-ours | GEVer with single forward check (no counterfactuals) | Isolates the counterfactual entailment test |
| Ablation-of-ours | GEVer without ensemble/gates (N_geo=1, no c_parse/stab gate) | Isolates the load-bearing-assumption repair |

### Why this setup validates the claim

The central claim is that a differentiable, counterfactual geometry-entailment verifier forces a 3D-VL model's answers to be grounded in its own geometric estimate rather than in semantic priors. Beacon3D is the right testbed because its object-centric, multi-test-per-object design and chain-of-analysis paradigm expose exactly the failure the claim targets: plausible answers that do not cohere with grounding. VSI-Bench is added because it is a metric-spatial benchmark used by VG LLM, so it tests whether the entailment signal transfers beyond a single architecture and dataset. The primary metric — coherence between grounding and QA under geometry-preserving versus geometry-changing perturbations — is chosen because it is the only signal that separates a geometry-grounded answer from a semantically fluent one; plain QA accuracy cannot, since both can be correct on unperturbed instances. The secondary metrics are chosen to make the load-bearing assumption measurable: s_ent is reported conditioned on parser confidence (c_parse ≥ 0.6) and on the perturbation-validity rate, so the grounding claim is tied to measured quantities rather than assumed. Each baseline removes one ingredient: StereoVLM has geometry but no consistency check, cdViews has semantics but no explicit geometry, the vanilla head has both heads but no joint verifier, the learned entailment classifier has the counterfactual supervision but not the differentiable renderer, and post-hoc calibration has the confidence signal but not joint training. The ablation removes the counterfactual gap, so if the gain survives it, the counterfactual mechanism is not doing the work; the second ablation removes the ensemble/gates, so if the gain survives it, the load-bearing-assumption repair is not doing the work. Together these form a falsifiable contrast: if the verifier is the cause, gains must concentrate where priors and geometry disagree.

### Expected outcome and causal chain

**vs. StereoVLM** — On a case where the question places an object "left of" another and the binocular geometry clearly contradicts a semantic prior (e.g., a chair typically on the right), StereoVLM produces the prior-consistent answer because its geometry encoder feeds features forward but never checks the answer against the recovered metric structure. Our method instead renders the predicate truth from the depth/disparity map and penalizes the answer when it fails the counterfactual flip test, so we expect a noticeable coherence gap on prior-conflicting subsets but near-parity on prior-consistent subsets.

**vs. 2D-only cdViews / LLaVA-OV zero-shot** — On a case where two sampled views are ambiguous about relative depth, the 2D pipeline produces a confident but ungrounded answer because it has no explicit metric geometry to be consistent with. Our method instead grounds the answer in the geometry head's depth and object positions and rejects answers that stay true under perturbation, so we expect the largest gains on depth-ambiguous QA subsets and little change on unambiguous ones.

**vs. Vanilla reasoning head without verifier** — On a case where the reasoning head's confidence is high but the geometry contradicts the answer, the vanilla head keeps its confident wrong answer because nothing couples confidence to geometry. Our method instead applies the margin loss so semantic confidence cannot exceed entailment, so we expect reduced overconfident errors and a tighter confidence-entailment correlation, concentrated on instances where the two heads initially disagree.

**vs. Learned entailment classifier trained on same counterfactual labels** — On a case where the counterfactual labels are available but the classifier can memorize semantic shortcuts, the learned classifier should match GEVer on prior-consistent subsets but lose on prior-conflicting subsets because it has no explicit geometric function to differentiate through. We expect GEVer's gain over this baseline to concentrate on prior-conflicting instances, which is the test that the renderer mechanism, not the supervision signal, is the source of the gain.

**vs. Post-hoc calibration of reasoning-head confidence** — On a case where the reasoning head is confidently wrong, post-hoc calibration can rescale confidence but cannot change the answer, so it should reduce overconfidence without improving coherence. We expect GEVer to beat post-hoc calibration on grounding-QA coherence, confirming that joint training is necessary.

**vs. Single-forward-check ablation** — On a case where a semantically plausible answer happens to be true in the original geometry, the single-check variant accepts it because it never tests sensitivity. Our method instead flags it as prior-driven when it survives counterfactual perturbations, so we expect the ablation to recover most of the unperturbed accuracy but lose the prior-conflicting subset gain.

**vs. GEVer without ensemble/gates** — On a case where the geometry head's depth is prior-contaminated, the ungated variant renders predicate truth from a noisy estimate and the counterfactual gap measures noise in G rather than grounding in R. Our method instead requires stability across N_geo=3 heads and gates on parser confidence, so we expect the gated variant to show higher s_ent separation between prior-conflicting and prior-consistent subsets, with the gap largest on instances that fail the stab gate in the ungated variant.

### What would falsify this idea

If the gain over baselines is uniform across prior-conflicting and prior-consistent subsets rather than concentrated where geometry and semantics disagree, the verifier is not detecting prior-driven answers and the central claim is wrong. Likewise, if the single-forward-check ablation matches full GEVer, the counterfactual entailment mechanism is not the cause of any improvement; and if the learned entailment classifier matches GEVer, the differentiable renderer is not the cause.

## References

1. StereoVLM: Binocular Vision-Language Models for Metric 3D Spatial Reasoning
