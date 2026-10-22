# Rubric-Verification Identifiability: Calibrating Multimodal Code Rubric Rewards Against Rendered-Visual Oracles

## Motivation

Across the multimodal code generation literature, the training signal is never structurally tied to the property it is supposed to certify: VisCodex merges vision and coding models and evaluates on InfiBench-V but relies on benchmark accuracy as a proxy for visual correctness, while PrismaCoder synthesizes a decomposed rubric spanning program, construction, and visual correctness and uses it directly as an RL reward without any calibration between the rubric score and what a renderer would actually show. The root cause is that the rubric is a text-only scorer over an artifact it never renders, so its score is an uncalibrated surrogate whose relationship to rendered-visual correctness is assumed rather than identified, and no existing branch flags when the surrogate has silently decoupled from the oracle.

## Key Insight

When a rubric criterion is paired with a differentiable-renderer oracle whose pixel-consistency verdict is treated as the ground-truth label, the rubric score becomes a sufficient statistic for that verdict only if a calibration head is trained to make the score predict the oracle label across artifact types—and this sufficiency condition is exactly what turns an uncalibrated reward into an identifiable one, flagging ambiguous artifacts instead of scoring them blindly.

## Method

**What it is.** We introduce RVIH (Rubric-Verification Identifiability Head), a calibration layer bolted onto PrismaCoder's rubric reward. Its inputs are (i) a synthesized rubric with per-criterion scores for a candidate artifact and (ii) a rendered-visual verification oracle O that returns a binary visual-correctness label y_vis and a pixel-consistency margin m. Its outputs are a calibrated rubric reward r̂ and an ambiguity flag a ∈ {0,1} that routes flagged artifacts to O rather than trusting the rubric alone.

**Load-bearing assumption (stated explicitly).** RVIH assumes that the oracle's label y_vis is a valid ground-truth label for visual correctness, so that a calibration head trained to predict y_vis yields a rubric score that is a sufficient statistic for visual correctness. This assumption is load-bearing: if the oracle is systematically wrong (e.g., a font-fallback bug in the renderer), the head's sufficiency is sufficiency for the renderer, not for visual correctness. To make this assumption testable rather than assumed, we quantify oracle reliability per artifact type (Phase 1b) and train the head against a soft, uncertainty-aware oracle label (Phase 3) with an explicit oracle-noise term (Phase 4). The sufficiency claim is therefore stated relative to a measured oracle reliability, not an assumed-perfect renderer.

**How it works.**

```pseudocode
# Phase 1: oracle construction (per artifact type)
for artifact x in corpus:
    render x with differentiable renderer R  -> image I_x
    reference image I_ref (from task spec)
    c     = pixel_consistency(I_x, I_ref)   # continuous in [0,1], e.g., 1 - normalized L2
    y_vis = 1[ c > tau_pix ]                # tau_pix = 0.92
    m     = c - tau_pix

# Phase 1b: oracle reliability estimation (per artifact type t)
#   human-annotated audit set A_t of size 200 per type (charts, HTML, diagrams)
#   y_human in {0,1} from a single expert annotator; inter-annotator agreement
#   measured on a 50-example overlap (Cohen's kappa reported)
#   reliability rho_t = agreement(y_vis, y_human) on A_t
#   oracle-noise floor eps_t = 1 - rho_t
#   soft oracle label: y_tilde = rho_t * y_vis + (1 - rho_t) * (1 - y_vis)
#   (i.e., flip probability eps_t; y_tilde in [eps_t, 1-eps_t])

# Phase 2: rubric scoring (PrismaCoder's existing head)
s = RubricScore(x)                        # vector over criteria

# Phase 3: calibration head (our contribution)
z   = MLP_phi( [s ; type_embed(x)] )      # 2-layer MLP, hidden 256, GeLU, dropout 0.1
p   = sigmoid(z)                          # predicted P(y_vis=1 | s, type)
r_hat = p                                 # calibrated reward

# Phase 4: sufficiency loss (the identifiability objective)
L_suff = BCE(p, y_tilde)                  # calibration against soft oracle label
L_amb  = -log( 1 - |2p - 1| )             # penalize p near 0.5 (ambiguous)
L      = L_suff + lambda_amb * L_amb      # lambda_amb = 0.1

# Phase 5: ambiguity routing at inference
a = 1[ |2p - 1| < delta  OR  eps_{type(x)} > eps_max ]   # delta = 0.3, eps_max = 0.15
if a == 1:  r_hat = O(x)                  # route to oracle, not rubric
else:       r_hat = p                     # trust calibrated rubric

# Phase 6: RL reward uses r_hat (replaces raw rubric score)
```

**Why this design.** We chose a learned calibration head over simply thresholding the raw rubric score because the raw score has no monotone relationship to rendered correctness across artifact types (a chart and an HTML page have different rubric scales), accepting the cost that the head adds parameters that must be trained on oracle labels. We chose a differentiable renderer plus pixel-consistency check as the oracle over a human-annotated visual-correctness set because it gives a dense, per-artifact label at scale, accepting the cost that the renderer must be differentiable and the reference image must exist. We chose a sufficiency loss (BCE against the oracle label) plus an explicit ambiguity penalty over a standard reward-model loss because we need the head to be confident exactly when the rubric is informative, accepting the cost that near-0.5 predictions are pushed away from the decision boundary and may over-flag. We chose a hard routing rule (flag when |2p−1| < δ or when the oracle's own estimated error rate for that artifact type exceeds ε_max) over a soft blend because routing must be auditable, accepting the cost that δ and ε_max are fixed hyperparameters. A domain expert would not describe this as a variant of PrismaCoder's rubric reward or of VisCodex's model merging: those treat the rubric score as a reward and the benchmark as a check, whereas RVIH trains the score to be a sufficient statistic for a rendered-visual oracle and uses the residual ambiguity as a routing signal—a property neither prior method establishes.

**Why it measures what we claim.** The calibration head's output p measures the probability that the rendered artifact is visually correct because the BCE loss forces p to match the soft oracle label y_tilde across artifact types, where y_tilde incorporates the measured oracle flip probability ε_t; this assumption fails when the renderer itself is wrong in a way not captured by the audit set (e.g., a font-fallback bug that the audit set does not exercise), in which case p reflects renderer fidelity rather than visual correctness. The ambiguity flag a measures rubric-oracle disagreement because |2p−1| is small exactly when the rubric score is not sufficient for the oracle verdict, and a also fires when the oracle's own reliability ε_t is high; this assumption fails when the rubric is systematically biased in a way that still yields confident but wrong p, in which case a reflects calibration-head overconfidence rather than true ambiguity. The sufficiency loss L_suff measures identifiability because minimizing BCE over the soft oracle label is equivalent to enforcing that s is a sufficient statistic for y_vis under the assumption that the MLP can represent the conditional P(y_vis|s) and that the rubric criteria span all renderer-visible factors; this assumption fails when the rubric criteria omit a visual factor the renderer captures, in which case L_suff measures head capacity rather than rubric completeness.

## Contribution

["RVIH, a rubric-verification identifiability head that pairs each synthesized rubric criterion with a differentiable-renderer pixel-consistency oracle and trains a calibration head with a sufficiency loss so the rubric reward becomes a sufficient statistic for the oracle's visual-correctness verdict.", 'An ambiguity-routing mechanism that flags artifacts where the calibrated rubric is uninformative (|2p−1| < δ) and routes them to the oracle instead of scoring them by the rubric alone, turning calibration residual into an auditable data-selection signal.', "A concrete instantiation on PrismaCoder's PrismaForge pipeline showing how the calibrated reward replaces the raw rubric score in rubric-driven RL while preserving the single-rubric design."]

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | MCD (multimodal code generation) | Matches PrismaCoder/VisCodex artifact types and rubric scales. |
| Primary metric | Reward–oracle agreement on held-out artifacts | Directly tests whether calibrated reward tracks visual correctness. |
| Secondary metric | End-task visual-correctness on held-out MCD benchmark after RL training with r_hat vs raw rubric | Ties significance to downstream artifact quality. |
| Baseline | PrismaCoder raw rubric reward | Tests sub-claim that raw rubric is not sufficient for visual correctness. |
| Baseline | VisCodex merged-model reward | Tests sub-claim that model merging lacks oracle identifiability. |
| Baseline | Fixed-threshold rubric routing | Tests sub-claim that learned calibration beats naive thresholding. |
| Baseline | Direct supervised classifier on rubric scores (no calibration framing) | Tests whether calibration framing yields more than a plain classifier. |
| Ablation-of-ours | RVIH without L_amb (no ambiguity penalty) | Isolates whether the ambiguity term drives routing gains. |
| Ablation-of-ours | RVIH with hard-threshold oracle label (no soft label / no ε_t) | Isolates contribution of oracle-reliability calibration. |
| Diagnostic | Rubric-completeness probe: hold out a renderer-visible factor (e.g., color contrast) from rubric criteria and measure L_suff degradation | Quantifies gap between head capacity and rubric completeness. |
| Diagnostic | Bias-induction test: perturb rubric criteria to induce systematic bias and check whether ambiguity flag fires | Tests whether confident-but-wrong p is distinguishable from true ambiguity. |

### Why this setup validates the claim

The central claim is that a rubric score can be made a sufficient statistic for a rendered-visual oracle, and that residual ambiguity from that calibration is a usable routing signal. MCD is the right dataset because it spans multiple artifact types (charts, HTML, diagrams) whose rubric scales differ, which is exactly the condition under which the raw rubric is claimed to be non-monotone with visual correctness. The primary metric — agreement between the final reward and the oracle label on held-out artifacts — directly operationalizes identifiability: if the calibrated reward does not agree with the oracle more often than the raw rubric does, the sufficiency claim is false. The secondary metric — end-task visual-correctness after RL training with r_hat versus raw rubric — ties the identifiability claim to downstream artifact quality, so significance is not only reward-oracle agreement. The three baselines each probe a distinct sub-claim: PrismaCoder's raw rubric tests whether calibration is needed at all, VisCodex's merged reward tests whether a stronger generative model already encodes visual correctness, and fixed-threshold routing tests whether the learned head adds anything over a hand-tuned cutoff. The direct supervised classifier baseline tests whether the calibration framing (soft oracle label, ambiguity penalty, routing) yields something a plain classifier on rubric scores does not. The no-L_amb ablation is the decisive control: it removes only the ambiguity penalty, so any gap between it and full RVIH isolates the contribution of the routing signal rather than of calibration in general. The hard-threshold-oracle ablation isolates the contribution of oracle-reliability calibration. The rubric-completeness probe directly measures whether L_suff degrades when a renderer-visible factor is omitted from the rubric, quantifying the gap between head capacity and rubric completeness. The bias-induction test perturbs rubric criteria to induce systematic bias and checks whether the ambiguity flag fires, testing whether confident-but-wrong p is distinguishable from true ambiguity. Together these form a falsifiable test because each baseline and diagnostic predicts a specific pattern of where our gain should and should not appear.

### Expected outcome and causal chain

**vs. PrismaCoder raw rubric reward** — On a case where a chart artifact scores high on rubric criteria (axis labels, legend, title) but renders with a misaligned y-axis due to a scaling bug, the raw rubric produces a high reward because its criteria never inspect rendered pixels. Our method instead learns p from rubric scores to oracle labels, so the scaling bug drives p downward and the artifact is flagged and routed to the oracle. We expect a noticeable agreement gap concentrated on artifacts whose rubric criteria are visually under-specified, with parity on artifacts where rubric criteria already encode the visual factor.

**vs. VisCodex merged-model reward** — On a case where an HTML page is semantically correct but a font-fallback causes layout overflow, the merged model's reward reflects its generative prior rather than a rendered check, so it assigns a confident high score because the code looks idiomatic. Our method instead compares rendered pixels to the reference and pushes p toward the oracle's low label, so the overflow is caught. We expect our advantage to concentrate on renderer-detectable defects that are invisible in source text, and to shrink to parity on defects that are also visible in code.

**vs. Fixed-threshold rubric routing** — On a case where a diagram's rubric score is mid-range because criteria are partially satisfied, a fixed threshold either always trusts the rubric or always routes to the oracle, so it mis-handles one side of the boundary. Our method instead learns the per-type mapping from rubric scores to oracle labels, so the same mid-range score is interpreted differently for a chart versus an HTML page. We expect a gap concentrated near the decision boundary and on artifact types with unusual rubric scales, with parity on extreme scores.

**vs. Direct supervised classifier on rubric scores** — On a case where the rubric score is high but the oracle label is low due to a renderer-visible defect, a plain classifier trained on rubric scores without the calibration framing (soft oracle label, ambiguity penalty, routing) may overfit to the rubric scale and produce confident wrong predictions. Our method instead uses the soft oracle label and ambiguity penalty to produce calibrated probabilities and a routing signal. We expect our advantage to concentrate on artifacts where the rubric is systematically biased or where oracle reliability is low, and to shrink to parity on artifacts where the rubric is already a sufficient statistic.

### What would falsify this idea

If RVIH's gain over the raw rubric is uniform across all artifact types and score ranges rather than concentrated where rubric criteria are visually under-specified or where scores sit near the decision boundary, then the calibration-and-routing mechanism is not the cause of the improvement and the identifiability claim is wrong. Likewise, if the no-L_amb ablation matches full RVIH everywhere, the ambiguity signal is not doing the work we claim; and if the bias-induction test shows the ambiguity flag does not fire under induced rubric bias, then confident-but-wrong p is indistinguishable from true ambiguity and the routing signal is not identifiable.

## References

1. PrismaCoder: Pushing the Boundaries of Open Multimodal Code Intelligence through Decomposed Rubric Synthesis
2. VisCodex: Unified Multimodal Code Generation via Merging Vision and Coding Models
