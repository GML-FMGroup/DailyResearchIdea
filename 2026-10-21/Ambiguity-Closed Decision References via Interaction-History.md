# Ambiguity-Closed Decision References via Interaction-History Consensus for GUI Agents

## Motivation

Relevance-based GUI focusing methods such as FocusGUI estimate which multimodal evidence is action-sufficient by comparing channels against a step-specific decision reference, but they inherit the benchmark's assumption that tasks arrive pre-segmented into clean steps with externally specified targets. When step boundaries are missing or ambiguous — the realistic case for continuous GUI interaction — there is no mechanism to derive the reference, so the relevance estimator's target is undefined and the focusing signal silently degrades to a fixed tutorial-derived label that may not match the agent's actual decision. This is a closure failure: the estimator's target is defined outside the agent's own observable interaction, so the method cannot operate on unsegmented tasks.

## Key Insight

When a step's decision reference is unobservable, the agent's own interaction history supplies a set of mutually-consistent candidate references whose agreement (low entropy) is itself evidence of which reference is correct, so the target can be a derived distribution over candidates rather than a single external label — closing the derivation under ambiguity without external segmentation.

## Method

**What it is.** We introduce ACID (Ambiguity-Closed Interaction-Derived references), a module that replaces FocusGUI's externally-specified step-specific decision reference with a distribution over candidate references derived from the agent's own interaction history, and trains the relevance estimator against that distribution (soft-weighting) instead of a single label. ACID's central assumption is stated explicitly: **pairwise agreement among history-derived candidate references is a valid proxy for reference correctness** — i.e., when the true step reference is well-defined, independent projections of it will agree more than consistent-error candidates. This assumption is load-bearing and is not verified by consensus alone; ACID therefore includes a non-self-referential anchor (Phase 2b) and a decorrelation step (Phase 1b) to bound the failure mode where correlated erroneous history produces tight low-entropy consensus on the wrong reference.

**How it works.**

```pseudocode
Input: frozen omni-modal backbone B, interaction history H = [(s_0,a_0),...,(s_t,a_t)],
       current screen s_t, task instruction I
Hyperparams: window W=5, temperature tau=0.5, consensus floor kappa=0.6,
             decorrelation threshold rho=0.8, anchor weight lambda=0.5,
             calibration set size N_cal=512, abstain threshold alpha=0.3

# Phase 1: candidate reference construction from interaction history
C = []
for lag in 1..W:
    # reference implied by the action taken lag steps ago, projected to current screen
    r_lag = project(a_{t-lag}, s_{t-lag} -> s_t)   # element/region descriptor
    C.append(r_lag)
C.append(parse_instruction_step(I, s_t))          # instruction-derived candidate

# Phase 1b: decorrelation — drop candidates sharing the same source or action type
# (e.g., all from instruction, or all click-type) when pairwise source correlation > rho
# Source correlation is measured as the fraction of candidates sharing the same
# (source_type, action_type) pair; if > rho, keep only the highest-scoring candidate
# from that source group and drop the rest.
C = decorrelate(C, rho)

# Phase 2: consensus scoring (entropy over candidate agreement)
for each r in C:
    score[r] = mean_{r' in C, r'!=r} sim( B.embed(r), B.embed(r') )   # pairwise agreement
w = softmax(score / tau)                          # distribution over candidates
H_norm = -sum_r w[r] * log w[r] / log|C|          # normalized entropy in [0,1]

# Phase 2b: non-self-referential anchor — gate candidate weight by an outcome/verification signal
# Anchor signal a[r] in [0,1] is computed from post-action state change: for each candidate r,
# a[r] = 1 if the action associated with r produced a measurable screen change (pixel diff > 5%)
# or task-progress predictor score > 0.5; else 0. This signal is independent of the agent's
# action distribution because it is derived from environment feedback, not from B.
for each r in C:
    w_anchored[r] = w[r] * (1 + lambda * a[r]) / Z   # Z normalizes to sum to 1
w = w_anchored

# Phase 3: ambiguity-aware target with calibration-based abstention
# Calibration: on a held-out set of N_cal=512 trajectories with known step references,
# compute the empirical accuracy of the consensus argmax as a function of H_norm.
# If calibration shows that for H_norm < alpha the consensus argmax accuracy is below
# chance (i.e., consensus is anti-correlated with correctness), abstain: fall back to
# the instruction-derived candidate (parse_instruction_step(I, s_t)) as a sharp target.
if calibration_anti_correlated(H_norm < alpha):
    p_ref = onehot(parse_instruction_step(I, s_t))   # abstain to instruction
elif 1 - H_norm >= kappa:                           # consensus reached
    p_ref = onehot(argmax_r w[r])                 # sharp target
else:
    p_ref = w                                     # soft-weighted target

# Phase 4: train relevance estimator against distribution
for each modality channel m in {screenshot, text, layout}:
    rel[m] = relevance_head( B.embed(m), p_ref )  # cross-attention to reference dist
L = -sum_r p_ref[r] * log rel[r]                  # distributional (soft) cross-entropy
# FocusGUI latent injection unchanged, but conditioned on rel[m]
```

**Why this design.** We chose to derive candidates from the agent's *own past actions* rather than from external segmentation because the benchmark's segmentation is exactly what is missing at deployment, accepting the cost that early in an episode (t < W) the candidate set is sparse and consensus is weak. We chose a normalized-entropy consensus criterion over a learned ambiguity classifier because entropy over pairwise agreement has no parameters to overfit to the benchmark's segmentation artifacts, accepting the cost that it conflates genuine multi-step ambiguity with noisy candidate projections. We chose soft distributional cross-entropy over hard argmax targets because a hard target on an ambiguous step injects a false label that the relevance head will faithfully memorize, accepting the cost that soft targets slow convergence when consensus is actually clear. We added the decorrelation step (Phase 1b) and the non-self-referential anchor (Phase 2b) because the consensus score alone is self-referential: when the agent's past actions are erroneous or habitual, independent projections of the true reference may not out-agree the consistent-error candidates. The anchor uses environment feedback (post-action state change) that is independent of the agent's action distribution, and the calibration-based abstention (Phase 3) provides a fallback to the instruction-derived reference when consensus is empirically anti-correlated with correctness. A domain expert would not describe this as FocusGUI plus a retrieval step: FocusGUI's target is given, whereas ACID's target is *computed by the agent and self-validated through consensus plus an external anchor*, which is the closure property FocusGUI structurally lacks; nor is it a retrieval scheme like Toolformer/Gorilla, because the candidate set is generated by intervening on the agent's own trajectory, not looked up from a library.

**Why it measures what we claim.** The pairwise agreement score `score[r]` measures *reference correctness* because we assume that when the true step reference is well-defined, independent history-derived projections of it will agree; this assumption fails when the agent's past actions were themselves erroneous, in which case agreement reflects *consistent error* rather than correctness. The decorrelation step (Phase 1b) and the anchor (Phase 2b) are designed to mitigate this failure by dropping redundant candidates and requiring corroboration from an environment-feedback signal. The normalized entropy `H_norm` measures *task ambiguity* because we assume ambiguity is equivalent to disagreement among independently-derived candidates; this assumption fails when candidates are derived from a single correlated source (e.g., all from the instruction), in which case low entropy reflects *source redundancy* rather than genuine consensus. The calibration-based abstention (Phase 3) detects when this failure mode makes consensus anti-correlated with correctness and falls back to the instruction-derived reference. The soft target `p_ref` measures *the relevance estimator's correct supervision signal* because we assume the consensus-weighted distribution concentrates on the action-sufficient evidence; this assumption fails when the majority candidate is a distractor that the agent repeatedly attended to, in which case `p_ref` reflects *habitual attention* rather than sufficiency. The anchor signal `a[r]` (post-action state change) provides a sufficiency check: a candidate whose associated action produced no measurable screen change is down-weighted, reducing the chance that habitual distractor attention dominates the target.

## Contribution

(1) ACID, an ambiguity-aware decision-reference derivation module that constructs candidate references from the agent's own interaction history and selects/soft-weights them by a normalized-entropy consensus criterion, closing the reference derivation under missing step boundaries. (2) A distributional training objective for the relevance estimator that replaces FocusGUI's single external target with a derived reference distribution, with a consensus floor that switches between sharp and soft targets. (3) A design principle — that interaction-history consensus is a valid substitute for external segmentation — demonstrated by the module operating without any pre-segmented step labels.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Omni-modal GUI agent benchmark (screenshot+text+layout) | Matches the three modality channels ACID reasons over |
| Primary metric | Step-level task success rate | Directly reflects correct reference selection per step |
| Baseline | Frozen omni-modal backbone (no FocusGUI) | Tests whether any reference conditioning helps at all |
| Baseline | FocusGUI with given step reference | Tests whether self-derived references beat externally-given ones |
| Baseline | FocusGUI + hard argmax self-reference | Tests whether soft consensus targets beat hard targets |
| Baseline | Generic self-consistency: sample K=5 references from backbone, majority/entropy-vote | Tests whether interaction-history intervention beats generic consensus |
| Baseline | Learned ambiguity classifier (2-layer MLP on B.embed(s_t), B.embed(I)) | Tests whether entropy consensus beats a learned classifier |
| Ablation-of-ours | ACID with uniform p_ref (no consensus weighting) | Isolates the contribution of consensus-based soft targets |
| Ablation-of-ours | ACID without decorrelation (Phase 1b disabled) | Isolates the contribution of decorrelation |
| Ablation-of-ours | ACID without anchor (Phase 2b disabled) | Isolates the contribution of the non-self-referential anchor |
| Ablation-of-ours | ACID without calibration abstention (Phase 3 disabled) | Isolates the contribution of calibration-based fallback |

### Why this setup validates the claim

The central claim is that ACID's self-derived, consensus-validated reference distribution gives a better supervision signal than an externally-supplied step reference, precisely because the external reference is unavailable at deployment. The frozen-backbone baseline establishes the floor: if ACID cannot beat it, the reference conditioning is inert. FocusGUI-with-given-reference is the sharpest test of the sub-claim that *derivation* rather than *conditioning* is what matters — it has the same adapter and latent injection, so any gap between it and ACID isolates the source of the reference. FocusGUI-with-hard-argmax isolates the soft-weighting sub-claim: same candidate construction, different target shape. The generic self-consistency baseline (sample K=5 references from the backbone and majority/entropy-vote them) tests whether ACID's interaction-history intervention is what produces the gain, not the consensus trick itself; if ACID does not beat this baseline, the contribution reduces to generic self-consistency. The learned ambiguity classifier baseline tests whether the parameter-free entropy consensus is preferable to a learned classifier that could overfit to benchmark segmentation artifacts. The uniform-p_ref ablation isolates the consensus scoring itself, separating "distributional targets help" from "consensus-weighted distributional targets help." The decorrelation, anchor, and calibration-abstention ablations isolate the contributions of the three mechanisms added to address the load-bearing assumption. Step-level success is the right metric because the mechanism is per-step reference selection; a task-level metric would dilute the signal across steps where consensus is trivially clear. The predicted effect is a *subset-concentrated* gain — on ambiguous steps where external references are missing or wrong — so this design is falsifiable: uniform gains or gains on unambiguous steps would contradict the causal story.

### Expected outcome and causal chain

**vs. Frozen omni-modal backbone (no FocusGUI)** — On a case where a task step requires attending to a small layout region while the instruction text is generic ("click the confirm button"), the frozen backbone produces a diffuse attention over the screenshot because nothing tells it which evidence is action-sufficient. Our method instead conditions the relevance head on a consensus-derived reference distribution that concentrates on the projected element region, so we expect a moderate overall gain but a much larger gap on the ambiguous-instruction subset, with near-parity on steps whose instructions name the target explicitly.

**vs. FocusGUI with given step reference** — On a case where the benchmark's segmentation labels a region that is correct for the *annotated* trajectory but the agent's actual prior action diverged (e.g., it opened a dialog the annotation did not anticipate), FocusGUI produces a reference pointing at the wrong element and faithfully memorizes it, because its target is externally fixed and cannot reflect the agent's own state. Our method instead recomputes the reference from the agent's interaction history and down-weights it via entropy when candidates disagree, so we expect parity on well-annotated steps and a visible gap concentrated on steps where the agent's trajectory deviates from the annotation.

**vs. FocusGUI + hard argmax self-reference** — On a genuinely ambiguous step with two plausible targets, the hard-argmax variant produces a confident single label that the relevance head memorizes, because argmax discards the disagreement structure. Our method instead emits the soft consensus distribution as the target, so the relevance head learns to hedge across both candidates; we expect the hard-argmax variant to match us on clear-consensus steps but to degrade sharply on high-entropy steps, visible as a gap that grows with the fraction of ambiguous steps in the evaluation slice.

**vs. Generic self-consistency (sample K=5 references from backbone, majority/entropy-vote)** — On a step where the backbone's samples are correlated by construction (same frozen model), the generic self-consistency baseline produces a tight vote that reflects the backbone's mode, not the agent's actual interaction history. Our method instead derives candidates from the agent's own past actions and projects them to the current screen, so we expect ACID to outperform this baseline specifically on steps where the agent's trajectory deviates from the backbone's prior, with parity on steps where the backbone's mode is correct.

**vs. Learned ambiguity classifier (2-layer MLP on B.embed(s_t), B.embed(I))** — On a step where the classifier overfits to benchmark segmentation artifacts (e.g., learning that certain screen regions are always targets), it will misclassify ambiguous steps. Our method instead uses a parameter-free entropy criterion that has no parameters to overfit, so we expect ACID to match or beat the classifier on ambiguous steps and to be more robust to distribution shift in segmentation patterns.

**vs. ACID with uniform p_ref (no consensus weighting)** — On a case where history-derived candidates agree strongly, the uniform variant produces a flat target that dilutes the correct candidate with noisy projections, because it ignores pairwise agreement. Our method instead sharpens toward the agreeing candidate via the entropy-gated softmax, so we expect the uniform ablation to trail us most on clear-consensus steps and to close the gap on ambiguous steps — the reverse of the pattern predicted for the hard-argmax baseline, which is itself a diagnostic.

**vs. ACID without decorrelation (Phase 1b disabled)** — On a case where all candidates come from the same source (e.g., all instruction-derived), the decorrelation step would drop redundant candidates and prevent source redundancy from masquerading as consensus. Without decorrelation, the method may produce low entropy due to source redundancy, leading to a sharp wrong target. We expect the full ACID to outperform this ablation on steps with correlated candidate sources, with parity on steps with diverse sources.

**vs. ACID without anchor (Phase 2b disabled)** — On a case where the agent habitually attends to a distractor, the anchor signal (post-action state change) would down-weight candidates that produced no measurable screen change. Without the anchor, the consensus may concentrate on the distractor, leading to a confident wrong target. We expect the full ACID to outperform this ablation on steps with habitual distractor attention, with parity on steps where the agent's actions are correct.

**vs. ACID without calibration abstention (Phase 3 disabled)** — On a case where consensus is anti-correlated with correctness (e.g., early in training when the agent's actions are noisy), the calibration-based abstention would fall back to the instruction-derived reference. Without abstention, the method may emit a sharp wrong target. We expect the full ACID to outperform this ablation on steps where consensus is unreliable, with parity on steps where consensus is reliable.

### What would falsify this idea

If ACID's gain over FocusGUI-with-given-reference were uniform across the ambiguous and unambiguous subsets rather than concentrated on steps where external references are missing or misaligned, the claim that self-derived consensus references close a structural gap would be wrong. Likewise, if the uniform-p_ref ablation matched ACID on clear-consensus steps, the consensus scoring would be doing no work and the mechanism would reduce to "any distributional target helps." Additionally, if ACID did not outperform the generic self-consistency baseline on steps where the agent's trajectory deviates from the backbone's prior, the interaction-history intervention would not be the source of the gain.

## References

1. Focused Omni-Modal GUI Agents
2. Navigating the Digital World as Humans Do: Universal Visual Grounding for GUI Agents
3. Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond
4. CogAgent: A Visual Language Model for GUI Agents
5. WebShop: Towards Scalable Real-World Web Interaction with Grounded Language Agents
6. Pix2Struct: Screenshot Parsing as Pretraining for Visual Language Understanding
