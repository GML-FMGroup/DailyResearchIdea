# Execution-Grounded Action-Region Verification for GUI Grounding in Synthesized Computer-Use Environments

## Motivation

Environment-synthesis pipelines for computer-use agents, such as WildAWM, validate tasks only at the outcome level: an agent rollout either solves the task or not, and cross-judge consistency checks whether a verifier agrees on that outcome. This means the grounding mechanism — the mapping from a GUI observation to the screen region an action actually manipulates — is never validated at the action-region level; functional equivalence of different grounding mechanisms is simply assumed because a shared residual gap appears across the synthesized environments. The root cause is structural: outcome-level gates cannot localize which action's grounding was wrong, so a task can pass while the agent's region prediction is systematically misaligned, and no learned verifier exists that ties a predicted region to the region the execution actually affected.

## Key Insight

Because every GUI action's execution deterministically induces a screen-region change that is observable in the post-action frame, the executed-region mask is a causal, action-conditioned sufficient statistic for grounding correctness, letting a verifier be trained against execution-derived regions rather than task outcomes.

## Method

**What it is.** EXARG (Execution-grounded Action-Region Grounding verifier) is a training and acceptance layer that, for each GUI action in a synthesized task, extracts the actual screen region affected by executing that action, trains a region-prediction verifier to map the pre-action observation plus action intent to that region, and accepts a grounding mechanism only when its predicted region overlaps the executed region above a threshold. Inputs: synthesized environment rollouts (pre/post frames, action metadata, environment hit-test/accessibility tree, and counterfactual probe rollouts). Outputs: a trained region verifier and a per-action acceptance decision.

**Load-bearing assumption (stated explicitly).** The executed-region mask M_exec is a faithful, action-conditioned sufficient statistic for the region the action functionally manipulated — i.e., M_exec equals the widget's functional hit-test region, so IoU against it is monotone in grounding correctness. This assumption fails when a widget's clickable area differs from its visually-changed extent (invisible padding/hit-slop, transparent overlays, parent-container event capture, hover-only or delayed state changes) or when the cursor sprite and incidental animation contaminate the difference mask. To make the assumption testable rather than assumed, M_exec is derived from coordinate-independent execution evidence (environment hit-testing plus counterfactual probing) and cursor/animation pixels are suppressed, as specified below.

**How it works.**

```pseudocode
# Phase 0: Animation/cursor suppression (per rollout R)
# Capture frames with CSS animations paused and cursor rendering disabled where the
# environment exposes these hooks; otherwise subtract a no-op control frame.
I_noop_pre, I_noop_post = replay_noop(R)          # pointer-move / no-op control
D_control = abs(I_noop_post - I_noop_pre)         # incidental-change footprint

# Phase 1: Execution-region extraction (per action a in rollout R)
for each action a with pre-frame I_pre, post-frame I_post, action coords c_a:
    D = abs(I_post - I_pre)                      # per-pixel change magnitude
    M_raw = (D > tau_pix) AND (D_control <= tau_pix)   # tau_pix = 8/255; suppress incidental
    M = morphological_open(M_raw, k=3)            # remove anti-aliasing noise
    # coordinate-independent functional attribution:
    H = hit_test_region(I_pre, c_a)               # from env hit-test/accessibility tree
    if H is available:
        M_func = H                                # functional hit region (not a disk around c_a)
    else:
        # counterfactual probing: replay action at perturbed coords c_a + delta, |delta|<=16px
        # in a grid of 8 offsets; record which perturbations change state
        P = { delta : state_changes(replay(a, c_a + delta)) }
        M_func = union of hit regions implied by P   # coordinate-independent attribution
    M_exec = M_func AND M                         # widget state change only, cursor removed
    if area(M_exec) < tau_area (0.01% of screen): discard a  # no observable effect
    store (I_pre, action_intent, M_exec)

# Phase 2: Train region verifier V_theta
# V_theta: (I_pre, action_intent) -> region mask; 2-layer MLP head (hidden=256, GeLU)
# on a frozen ViT-B/16 image encoder; action_intent embedded via a 2-layer MLP (hidden=256)
# and concatenated with image features before the mask head.
for each stored tuple:
    M_hat = V_theta(I_pre, action_intent)
    loss = BCE(M_hat, M_exec) + lambda * (1 - soft_IoU(M_hat, M_exec))   # lambda = 1.0
    update theta

# Phase 3: Acceptance gate for a grounding mechanism G
for each action a in held-out synthesized task:
    region_pred = G(I_pre, action_intent)
    region_exec = execution_region(a)             # Phase 1 (M_exec)
    iou = IoU(region_pred, region_exec)
    accept a if iou >= tau_iou (0.5) and V_theta(I_pre, action_intent) agrees with region_pred
accept grounding mechanism G if acceptance_rate >= tau_task (0.9)
```

**Why this design.** We chose execution-difference masks over human-annotated bounding boxes because synthesized environments produce thousands of actions per hour and annotations cannot scale, accepting the cost that difference masks conflate the action's region with incidental screen changes (cursor blink, animation) and must be filtered by the causal-locus intersection. We chose a learned verifier V_theta over a fixed IoU threshold alone because a threshold cannot distinguish a genuinely correct region from a coincidentally overlapping one when the target widget is large, accepting the cost of training a model that may inherit the environment's rendering biases. We chose to condition V_theta on action intent rather than the raw action coordinates because the verifier must generalize to unseen tasks where coordinates differ, accepting the cost that intent descriptions must be normalized across the synthesis pipeline. We chose per-action acceptance with a task-level aggregate rather than per-task outcome gates because the motivation requires localizing which grounding failed; this is not a variant of outcome-level verifier training as in SWE-Gym or R2E-Gym, where the verifier scores a final solution, because here the verifier scores an intermediate region against an execution-derived causal mask. We chose coordinate-independent functional attribution (hit-test tree or counterfactual probing) over a disk centered on the predicted coordinates because a disk around c_a is conditioned on the very grounding being evaluated and makes M_exec non-identifying for nearby errors; the cost is that hit-test trees are not exposed in all synthesized environments, so counterfactual probing (8 perturbations, |delta|≤16px) is the fallback.

**Why it measures what we claim.** The executed-region mask M_exec measures the action's true affected region because the assumption that frame differencing isolates the action's effect holds when the environment is deterministic and the action is the only state-changing event between frames; this assumption fails when background animation or asynchronous updates change pixels, in which case M_exec reflects incidental change and the control-frame subtraction plus the functional hit-region intersection is what restores attribution. The IoU between predicted and executed region measures grounding correctness because the assumption that region overlap is monotone in functional action success holds when the environment's hit-testing is region-based; this assumption fails when a widget's clickable area differs from its visual extent, in which case IoU reflects visual alignment rather than functional clickability, and the hit-test-tree / counterfactual-probe attribution is what makes M_exec track the functional region rather than the visual one. The verifier's agreement with the grounding mechanism measures consistency because the assumption that V_theta learned the execution-region distribution rather than memorizing training tasks holds when held-out tasks are drawn from unseen environments; this assumption fails under environment leakage, in which case agreement reflects memorization.

## Contribution

(1) A method, EXARG, that derives action-region ground truth from execution frame differences with causal-locus attribution and trains a region verifier against it. (2) An acceptance protocol that gates GUI grounding mechanisms at the action-region level using overlap between predicted and execution-derived regions, replacing outcome-level validation. (3) An analysis of when frame-difference regions are valid ground truth for grounding, including the widget-extent versus visual-extent failure mode.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Synthesized GUI rollouts with held-out unseen environments; plus a diagnostic subset of widgets whose hit-test box is deliberately offset from their visual box by 8–24px | Tests generalization and the hit-test-vs-visual-extent assumption |
| Primary metric | Task-level acceptance rate under IoU≥0.5 gate | Directly measures the causal-region grounding claim |
| Secondary metric | Hit-test-vs-visual-extent divergence rate on the diagnostic subset (fraction of widgets where IoU(hit-test box, visual box) < 0.5) | Quantifies how often the load-bearing assumption is violated |
| Baseline | Fixed-IoU threshold without learned verifier | Isolates value of the verifier over threshold alone |
| Baseline | Coordinate-conditioned grounding mechanism | Tests whether intent conditioning beats raw coordinates |
| Baseline | Raw difference-mask grounding (no causal-locus filter) | Tests whether filtering incidental change matters |
| Baseline | SAM-style promptable segmentation seeded by the action coordinate (ViT-B backbone, point prompt at c_a) | Tests whether off-the-shelf region proposal suffices |
| Ablation-of-ours | EXARG without V_theta agreement check | Isolates the verifier-agreement sub-claim |
| Ablation-of-ours | EXARG with disk(c_a, r=32) causal locus instead of hit-test/probe attribution | Isolates the coordinate-independent attribution sub-claim |

### Why this setup validates the claim

The central claim is that execution-derived causal masks plus a learned intent-conditioned verifier accept correct grounding mechanisms and reject coincidentally-overlapping ones, and that this acceptance transfers to unseen environments. The fixed-IoU baseline tests the sub-claim that a threshold alone cannot distinguish genuine overlap from coincidence on large widgets; the coordinate-conditioned baseline tests whether intent conditioning is what enables generalization across differing coordinates; the raw-difference baseline tests whether the causal-locus intersection is what restores attribution under incidental screen change; the SAM-style baseline tests whether EXARG's attribution and intent conditioning add anything beyond an off-the-shelf promptable segmenter. The ablation removing V_theta agreement tests whether the learned verifier adds discrimination beyond the executed-region IoU gate itself; the ablation replacing hit-test/probe attribution with a disk around c_a tests whether coordinate-independent attribution is what makes M_exec identifying for nearby errors. The primary metric — task-level acceptance rate gated on per-action IoU — is the right detector because the claim is about acceptance decisions, not raw segmentation quality: a method could have high mean IoU yet accept a broken grounding mechanism, and only the acceptance-rate metric exposes that. The diagnostic subset directly measures the load-bearing assumption: if hit-test-vs-visual-extent divergence is frequent, the IoU proxy is measuring visual alignment rather than functional clickability, and the acceptance-rate gap on that subset is the falsification signal. Held-out unseen environments make the test falsifiable by removing memorization as an explanation for agreement.

### Expected outcome and causal chain

**vs. Fixed-IoU threshold without learned verifier** — On a case where a large target widget makes a wrong grounding region overlap the executed mask above 0.5 by coincidence, the baseline accepts it because a scalar threshold cannot encode which pixels are causally attributable. Our method instead rejects it because V_theta predicts a tighter causal region and disagrees with the coincidental overlap. We expect a noticeable acceptance-rate gap concentrated on large-widget and dense-layout subsets, with parity on small isolated widgets.

**vs. Coordinate-conditioned grounding mechanism** — On a held-out task where the same intent appears at shifted coordinates, the baseline mispredicts the region because it memorized coordinate-to-region mappings from training layouts. Our method instead generalizes because V_theta is conditioned on normalized action intent, which is layout-invariant. We expect a clear gap on the unseen-environment split and near-parity on in-distribution tasks, which is the signature of intent conditioning rather than rendering memorization.

**vs. Raw difference-mask grounding (no causal-locus filter)** — On a case where a cursor blink or background animation changes pixels far from the action, the baseline treats those incidental pixels as part of the executed region, inflating the mask and inflating IoU for any broad prediction. Our method instead intersects with the functional hit region and subtracts the control-frame footprint, restoring a tight mask that punishes broad predictions. We expect the baseline to show inflated acceptance for sloppy mechanisms and our method to show lower acceptance for those same mechanisms while matching on clean, animation-free subsets.

**vs. SAM-style promptable segmentation seeded by the action coordinate** — On a case where the target widget has invisible padding or a transparent overlay, the baseline segments the visual extent rather than the functional hit region, so its predicted region overlaps the executed mask only when the two coincide. Our method instead uses the hit-test/probe-derived M_exec, so the acceptance decision tracks functional clickability. We expect the baseline to show inflated acceptance on widgets where visual and hit-test extents diverge, and our method to show a gap concentrated on the diagnostic offset subset.

**vs. EXARG without V_theta agreement check (ablation)** — On a case where a broad prediction overlaps the executed mask above 0.5 by coincidence, the ablation accepts it because the IoU gate alone cannot distinguish genuine from coincidental overlap. Our method instead rejects it because V_theta disagrees. We expect the ablation to show higher acceptance of sloppy mechanisms on large-widget subsets, confirming the verifier-agreement sub-claim.

**vs. EXARG with disk(c_a, r=32) causal locus (ablation)** — On a case where a wrong-but-nearby prediction lands inside the same 32px disk as the correct one, the ablation produces identical M_exec for both and cannot localize the error. Our method instead uses coordinate-independent attribution, so the two predictions yield different M_exec. We expect the ablation to show inflated acceptance for nearby-wrong predictions on dense-layout subsets, confirming the attribution sub-claim.

### What would falsify this idea

If our acceptance-rate gain is uniform across large-widget, unseen-environment, animation-heavy, and hit-test-offset subsets rather than concentrated where the predicted failure modes occur, the causal story is wrong and the gain is not attributable to execution-grounded region verification. Equivalently, if the V_theta ablation or the disk-locus ablation shows no drop in acceptance discrimination, the learned verifier and coordinate-independent attribution add nothing beyond the IoU gate.

## References

1. Agent World Model in the Wild: Computer-Use Agent Environment Synthesis
2. R2E-Gym: Procedural Environments and Hybrid Verifiers for Scaling Open-Weights SWE Agents
3. Generalizable End-to-End Tool-Use RL with Synthetic CodeGym
4. Training Software Engineering Agents and Verifiers with SWE-Gym
5. APIGen: Automated Pipeline for Generating Verifiable and Diverse Function-Calling Datasets
6. FireAct: Toward Language Agent Fine-tuning
7. Self-Instruct: Aligning Language Models with Self-Generated Instructions
