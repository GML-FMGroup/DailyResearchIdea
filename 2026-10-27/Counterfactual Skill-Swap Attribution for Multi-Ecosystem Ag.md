# Counterfactual Skill-Swap Attribution for Multi-Ecosystem Agent Tool Use

## Motivation

End-to-end success is the only observable in agent tool-use benchmarks, so a failed trajectory cannot be attributed to planning, retrieval, or skill invalidation. This structural limitation recurs across prior work: Δτ-Bench reports only task success across six ecosystem states and finds that skill-learning gains do not consistently transfer, while T-Eval decomposes tool use into sub-processes but evaluates them in isolation rather than attributing failure within a single coupled trajectory. Without an identifiable mapping from failure to subprocess, neither benchmark can diagnose why adaptation fails when the same skill is retrieved across states.

## Key Insight

The capability-preserving transformations that define Δτ-Bench's ecosystem states guarantee that a valid skill's semantics are invariant across states, so swapping a retrieved skill for its cross-state twin changes only the retrieval-to-skill binding and makes the resulting success delta a valid counterfactual probe for attribution.

## Method

**What it is.** CASSA (Counterfactual Skill-Swap Attribution) takes a failed trajectory from Δτ-Bench, the task specification, and the skill library of the state in which it failed, and outputs a three-way attribution label — planning failure, retrieval failure, or skill invalidation — together with a per-label counterfactual success delta. It requires no learned disentanglement head; the attribution is read off designed interventions on the skill library.

**Load-bearing assumption (stated explicitly).** The capability-preserving transformations T_{i→j} that define Δτ-Bench's six ecosystem states are assumed to preserve skill semantics exactly enough that a swapped twin k*' is behaviorally interchangeable with k* on the target task, so any success delta from the swap is attributable solely to the retrieval-to-skill binding rather than to residual capability differences between states. This assumption is not taken on faith: it is measured per twin via a calibrated equivalence test (step 2b) and its residual is subtracted from Δ_swap (step 4), so imperfect transformations degrade the delta rather than masquerading as retrieval credit.

**How it works.**

```pseudocode
Input: failed trajectory τ for task t in ecosystem state s_i
       skill library L_i, capability-preserving transformation T_{i→j} mapping s_i to s_j
       retrieved skill k* used in τ, planner/harness H
Output: attribution label a ∈ {planning, retrieval, skill-invalidation}, deltas Δ_p, Δ_r, Δ_inv

1.  Identify the binding step b in τ where k* was retrieved and invoked.
2.  Construct the capability-preserving twin k*' = T_{i→j}(k*) in state s_j,
    for each j ≠ i (six states ⇒ up to five twins).
2a. Twin validity gate: run the capability-preservation check (same
    input/output schema and same postcondition on a held-out probe set
    of 32 input/output pairs per skill); discard any twin that fails.
2b. Calibrated equivalence test (replaces binary gate): for each twin
    that passes 2a, estimate its residual capability gap on matched
    non-failed control trajectories in state s_j. Draw N_ctrl = 64
    control trajectories from s_j that (i) are labeled success under
    the original skill k* in s_i, (ii) share k*'s capability signature,
    and (iii) are matched to τ on task family and step count (±1 step).
    Replay each control under H with k*' bound at the corresponding
    step; record success rate p_ctrl(k*') ∈ [0,1]. The residual gap is
    g(j) = 1 − p_ctrl(k*'). If g(j) > 0.25 (twin fails on more than a
    quarter of matched controls), mark twin j as unreliable and exclude
    it from attribution; report it in the twin-validity table.
3.  REPLAY τ under H with the ONLY change being the binding at step b:
    k* replaced by k*'. Keep task t, planner H, prompt, and all other
    retrieved skills fixed. Record success y_swap(j) ∈ {0,1}.
    Also record baseline y_orig = 0 (τ failed).
4.  Compute calibrated swap delta for each reliable twin j:
    Δ_swap(j) = y_swap(j) − y_orig − g(j) = y_swap(j) − g(j).
    (For unreliable twins, Δ_swap(j) is reported but flagged and not
    used in step 5.)
5.  Attribute using confidence-weighted rule:
    Let Δ*_r = max_{j reliable} Δ_swap(j).
    if Δ*_r ≥ 0.5  → retrieval failure (a semantics-equivalent skill
        from another state succeeds where the retrieved one failed,
        after subtracting the twin's residual gap)
    elif the skill library L_i contains an alternative skill k_alt with
        the same capability signature as k* and replaying with k_alt
        succeeds (y_alt = 1) → skill invalidation (the specific skill
        instance, not retrieval binding, is at fault)
    else → planning failure (no semantics-preserving swap recovers
        success after calibration; the failure lies in the planner's
        action sequence)
    Confidence c = |Δ*_r| if retrieval branch taken; c = 1 − g(j*) for
    skill-invalidation branch (j* = twin used for k_alt match);
    c = 1 − max_j g(j) for planning branch. Return (a, c).
6.  Return a, c, {Δ_swap(j)}, {g(j)}, and the set of reliable twins used.
```

Hyperparameters: the capability-preservation check uses a held-out probe set of 32 input/output pairs per skill; the calibrated equivalence test uses N_ctrl = 64 matched control trajectories per twin and a reliability threshold g(j) ≤ 0.25; the number of twins per skill is bounded by the number of ecosystem states (six, so at most five swaps per failed trajectory); replay uses the same planner and prompt as the original run with temperature 0 to make the swap comparison deterministic.

**Why this design.** We chose a designed invariance (capability-preserving transformations) over a learned disentanglement head because planning and retrieval are operationally coupled in a single trajectory — a learned head would have to manufacture independence the data does not contain, which is exactly the failure mode Anti-Pattern 1 names and which prior work such as T-Eval avoids only by evaluating sub-processes in isolation rather than within a coupled trajectory. We chose to replay under a fixed planner and fixed prompt rather than re-planning after the swap because re-planning would confound the retrieval-to-skill binding with planner adaptation, accepting the cost that a planner that would have recovered under a different plan is attributed to planning failure rather than to a missed retrieval. We chose a three-way attribution rule with a capability-signature check for skill invalidation over a scalar success delta alone because a scalar delta cannot distinguish 'the retrieved skill was wrong for this state' from 'the skill instance itself is broken', accepting the cost of requiring a capability-signature index over the skill library. We chose to bound the number of swaps by the number of ecosystem states rather than by a learned search over the skill library because the six-state structure of Δτ-Bench already provides the invariance needed, accepting the cost that attribution is coarser when no twin exists for a given skill. We chose to calibrate the swap delta against a measured residual capability gap on matched controls rather than trust the binary twin-validity gate, because the benchmark's own reported inconsistency of cross-state skill transfer shows that nominally capability-preserving transformations leave residual behavioral differences; the calibration makes those differences explicit and subtracts them, accepting the cost of 64 control replays per twin. A domain expert would not describe this as a variant of Toolformer or Gorilla retrieval because the mechanism is an intervention on the skill library that produces a counterfactual success delta, not a nearest-neighbor match over descriptors.

**Why it measures what we claim.** The calibrated counterfactual success delta Δ_swap(j) = y_swap(j) − g(j) measures retrieval credit because the capability-preserving transformation T_{i→j} is assumed to preserve skill semantics exactly, so any change in success under the swap is attributable to the retrieval-to-skill binding rather than to the skill's capability; this assumption fails when the transformation is imperfect (e.g., a renamed tool with a subtly different postcondition), in which case the residual gap g(j) estimated on matched controls absorbs the capability difference and Δ_swap reflects the binding effect net of that difference, and if g(j) > 0.25 the twin is excluded rather than allowed to masquerade as retrieval credit. The capability-signature check measures skill invalidation because two skills sharing a signature are assumed to be interchangeable in capability, so a success under k_alt but not under k* isolates the instance; this assumption fails when the signature is too coarse and k_alt differs in efficiency or side effects, in which case the label reflects a capability mismatch rather than invalidation, and the confidence c = 1 − g(j*) reports how much residual gap the matched twin carried. The residual label of planning failure measures planner credit because the swap holds the planner, prompt, and task fixed and varies only the skill binding, so a null calibrated swap delta is assumed to mean the planner's action sequence is the binding constraint; this assumption fails when the planner would have recovered under a different plan, in which case the residual reflects planner rigidity rather than a pure planning error, and the confidence c = 1 − max_j g(j) reports how much of the null could be due to unreliable twins rather than planner rigidity.

## Contribution

(1) A counterfactual skill-swap attribution protocol (CASSA) that uses the capability-preserving invariance across Δτ-Bench's six ecosystem states to produce an identifiable three-way failure attribution without a learned disentanglement head. (2) A capability-signature index and twin-validity check that operationalizes skill invalidation as a distinct attribution label separable from retrieval failure. (3) An empirical finding on Δτ-Bench characterizing which subprocess dominates failures as the ecosystem state advances, and how the attribution distribution shifts across models and harnesses.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | Δτ-Bench failed trajectories (six states) | Provides built-in capability-preserving invariance for swaps |
| Primary metric | Three-way attribution accuracy vs. expert labels | Directly tests the causal claim, not just success |
| Secondary metric | Calibrated confidence c (mean and calibration error vs. label correctness) | Tests whether the residual-gap calibration produces meaningful confidence |
| Baseline | Steward | Strongest skill-maintenance method on same benchmark |
| Baseline | T-Eval-style isolated sub-process scoring | Tests whether coupled-trajectory attribution beats isolation |
| Baseline | Scalar success-delta-only attribution | Tests whether three-way rule beats a single delta |
| Baseline | Supervised disentanglement head (2-layer MLP, hidden=256, GeLU, trained on the same replay outcomes with matched label budget) | Tests whether the designed intervention beats a learned decision rule on the same data |
| Ablation-of-ours | CASSA with swaps only, no capability-signature check | Isolates contribution of the skill-invalidation branch |
| Ablation-of-ours | CASSA with binary twin-validity gate, no calibrated equivalence test (g(j) ≡ 0) | Isolates contribution of the residual-gap calibration |
| Ablation-of-ours | CASSA with re-planning control (re-plan after swap) | Bounds the planner-rigidity failure mode F |
| Diagnostic | Twin-validity table: fraction of twins passing 32-probe check, false-preservation rate on adversarially perturbed postconditions, attribution accuracy vs. probe-set size (8, 16, 32, 64, 128) | Empirically bounds the invariance assumption |
| Diagnostic | Inter-annotator agreement: two experts label 200 failed trajectories, report Cohen's kappa and adjudication protocol | Establishes reliability of the primary metric |

### Why this setup validates the claim

The central claim is that a designed capability-preserving intervention on the skill library yields correct three-way attribution of failure cause, and that this beats both a strong skill-maintenance baseline and sub-process-isolation scoring. Δτ-Bench is the right dataset because its six states are *constructed* via capability-preserving transformations, so the swap in CASSA is grounded in the benchmark's own invariance rather than an assumed one — this is what makes the counterfactual delta interpretable. Attribution accuracy against expert labels is the right primary metric because the method's whole job is to output a label; measuring task success instead would not test attribution at all. Steward tests the sub-claim that attribution is not already solved by strong skill maintenance. The T-Eval-style baseline tests the sub-claim that coupled-trajectory attribution is necessary — if isolated scoring matches CASSA, the coupling argument fails. The scalar-delta baseline tests the sub-claim that the capability-signature branch adds discriminative power. The supervised disentanglement head tests the sub-claim that the designed intervention, not merely the decision rule, is what recovers the label: it is trained on the same replay outcomes with the same label budget, so any accuracy gap isolates the intervention. The calibrated-equivalence ablation tests the load-bearing assumption directly: if removing the residual-gap subtraction does not change attribution accuracy, the calibration is unnecessary; if it does, the binary gate was insufficient. The re-planning control bounds the planner-rigidity failure mode F by measuring how often a planner that would have recovered under a different plan is mislabeled as planning failure. The twin-validity table and probe-set-size sensitivity analysis empirically bound the invariance assumption rather than asserting it, and the inter-annotator agreement study establishes that the primary metric is reliable. Together these form a falsifiable test: CASSA must beat each baseline *on the subset where that baseline's mechanism gap predicts failure*, not uniformly.

### Expected outcome and causal chain

**vs. Steward** — On a case where the planner correctly sequences actions but the retrieved skill `k*` is stale for state `s_i`, Steward produces a repaired skill library yet still mislabels the failure as a planning defect because its maintenance loop optimizes downstream success without counterfactual isolation of the binding step. Our method instead swaps in the twin `k*' = T_{i→j}(k*)`, subtracts the twin's residual gap `g(j)` measured on 64 matched controls, and observes calibrated `Δ_swap = 1 − g(j) ≥ 0.5`, attributing retrieval failure because the swap holds planner, prompt, and task fixed. We expect a noticeable attribution-accuracy gap concentrated on retrieval-failure cases, with near-parity on pure planning-failure cases where no twin recovers success.

**vs. T-Eval-style isolated sub-process scoring** — On a case where the planner's action sequence is only wrong *given* the retrieved skill (the plan would be correct under a different binding), isolated scoring produces a planning label because it scores the planner in a skill-free harness where the coupling is absent. Our method instead replays the full coupled trajectory with only the binding changed, so a successful calibrated swap correctly surfaces retrieval failure. We expect the gap to concentrate on trajectories where planning and retrieval are entangled, and to shrink toward parity on trajectories where the two are separable.

**vs. Scalar success-delta-only attribution** — On a case where the retrieved skill `k*` is genuinely broken but an alternative `k_alt` with the same capability signature exists in `L_i`, the scalar baseline produces a retrieval label because any successful swap yields a positive delta and it cannot distinguish instance-brokenness from wrong-state retrieval. Our method instead checks the capability-signature index and, finding `k_alt` succeeds while `k*` fails, labels skill invalidation with confidence `c = 1 − g(j*)`. We expect a gap concentrated on skill-invalidation cases and parity elsewhere, which is exactly the diagnostic pattern the ablation should reproduce.

**vs. Supervised disentanglement head** — On the same replay outcomes and matched label budget, the learned head must infer the binding effect from features without an explicit counterfactual, so its attribution variance is expected to be higher on retrieval-failure cases where the swap delta is the only signal. Our method reads the label off the calibrated delta directly, so we expect lower variance and higher accuracy on retrieval and skill-invalidation subsets, with parity on planning-failure cases where both methods see a null signal.

**vs. CASSA with binary twin-validity gate (no calibration)** — On a case where the twin passes the 32-probe check but carries a residual gap `g(j) = 0.3` on matched controls, the binary gate admits the twin and a successful swap yields `Δ_swap = 1`, producing a retrieval label even though the twin's capability differs. Our calibrated method subtracts `g(j)` and, since `g(j) > 0.25`, excludes the twin, falling through to the planning or skill-invalidation branch. We expect the calibration ablation to show inflated retrieval labels on twins with high residual gap, with the gap concentrated on skills whose postconditions branch or have stateful side effects.

**vs. CASSA with re-planning control** — On a case where the planner would have recovered under a different plan after the swap, the fixed-planner replay yields `y_swap = 0` and a planning label, while the re-planning control yields success and reveals planner rigidity. We expect the re-planning control to recover a fraction of planning-labeled cases, bounding failure mode F; the difference between fixed-planner and re-planning attribution accuracy quantifies how much planner rigidity contaminates the planning label.

### What would falsify this idea

If CASSA's attribution-accuracy gain over the scalar-delta and T-Eval-style baselines is uniform across all three failure-label subsets rather than concentrated on retrieval and skill-invalidation cases where those baselines' mechanisms are predicted to break, the claim that designed interventions recover the correct causal label is wrong. Equally fatal: if the swap-only ablation matches full CASSA on skill-invalidation cases, the capability-signature branch contributes nothing and the three-way rule collapses to a two-way one; and if the calibrated-equivalence ablation matches full CASSA on twins with residual gap `g(j) > 0.25`, the load-bearing invariance assumption is not actually load-bearing and the calibration is unnecessary.

## References

1. Δτ-Bench: Pushing Agents Toward a Moving Capability
2. Towards General Agentic Intelligence via Environment Scaling
3. T-Eval: Evaluating the Tool Utilization Capability of Large Language Models Step by Step
4. AutoGuide: Automated Generation and Selection of Context-Aware Guidelines for Large Language Model Agents
5. ReAct: Synergizing Reasoning and Acting in Language Models
6. xLAM: A Family of Large Action Models to Empower AI Agent Systems
7. ExpeL: LLM Agents Are Experiential Learners
8. AgentTuning: Enabling Generalized Agent Abilities for LLMs
