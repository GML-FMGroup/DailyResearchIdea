# First-Appearance Grounding Contracts: A Quantitative Benchmark and Identifiability Test for Rule-to-Visual Realization on Unseen Objects

## Motivation

Configurable interactive world models must not only execute user-defined rules symbolically but visibly realize them on the objects those rules designate — yet across this literature the realization step is asserted rather than measured. WorldRuler introduces dynamic object binding and stored-state constraints precisely to ground rule-specified entities in visual instances, including objects unseen during training, but its abstract reports no datasets, baselines, or quantitative realization accuracy, so it is impossible to tell whether a rendered attribute change is caused by the rule or merely correlated with it. The structural root cause is that rule-to-visual realization has no operationalization: there is no contract fixing which pixels are accountable for a rule, no intervention that isolates the rule's causal effect on those pixels, and no test that the binding is identifiable for objects the model never saw. This is a meta-level gap recurring across interactive-world-model branches — the exploration branch (GenieRedux-style pretraining, E1) does not address rule-conditioned generation at all, so the field currently lacks any shared measurement of whether rules are visibly realized.

## Key Insight

Rule-to-visual realization becomes measurable and identifiable exactly when the binding between a rule-specified entity and its visual instance is frozen at first appearance and the rule is then intervened on, because this contract makes the designated pixels a sufficient statistic for the rule's causal effect and turns the unseen-object case into an out-of-support test of the binding rather than of the renderer.

## Method

**What it is.** We introduce **FIRST-BIND**, a benchmark-plus-protocol that takes as input (i) a WorldRuler-style executable 3D rule graph with an object-attribute state-replacement operator, (ii) a video world model conditioned on rule executions, and (iii) a held-out object split, and outputs a per-rule *Realization Score* together with an *Identifiability Margin* that quantifies whether the rendered change is attributable to the rule rather than to the renderer.

**Load-bearing assumption (stated explicitly).** The paired-null counterfactual is valid only if, under identical seeds, prompts, and camera trajectory, V^+ and V^- differ *only* in the rule-specified attribute state, so any masked LPIPS/attribute-match difference is attributable to the rule rather than to the renderer's own stochasticity or to the rule's indirect effects on the rest of the rollout. Autoregressive video world models are chaotically sensitive to conditioning perturbations (E3: compounding error is 'inherently irreducible in autoregressive video generation'; current video models have 'limited in-context learning capabilities'), so a one-attribute state change at t* is exactly the kind of small conditioning perturbation that can diverge globally within a few frames. FIRST-BIND therefore does not assume this away: it *measures* the divergence and subtracts it via a matched-perturbation control, and it restricts scoring to a short post-t* horizon whose length is set by a measured divergence-onset estimate. The assumption is verified per scene by the seed-variance and control-perturbation checks below; when those checks fail, the scene is flagged and its Realization/Margin are reported as confounded rather than as binding evidence.

**How it works.**

```pseudocode
INPUT: rule graph G (rules R, dependencies, object states S)
       video world model M (rule-conditioned generator)
       scene set D = D_seen ∪ D_unseen   # unseen = object identity absent at train
       intervention budget K = 3, tie-break tolerance tau = 0.05
       post-t* scoring horizon H = 8 frames (set by divergence-onset estimate, below)

PHASE 1 — First-Appearance Grounding Contract (FAGC)
  for each scene d in D:
    roll out M under the *null* rule set (no rule fires) for T0 = 16 frames
    at the first frame t* where object o first appears:
      freeze a pixel mask m_o  <- SAM2 mask prompted by the rule-graph
                                   entity handle for o at frame t*
      record anchor state s_o* = (pose, material, color) from G
      store contract C_o = (t*, m_o, s_o*)
    # m_o is FROZEN for the rest of the episode; later masks are never re-derived

PHASE 2 — Rule Intervention with Matched-Perturbation Control
  for each rule r in R designating object o with attribute a:
    s_o^+ = apply_attribute_replacement(s_o*, r)   # from G
    s_o^- = s_o*                                    # null counterfactual
    s_o^0 = apply_null_perturbation(s_o*, a)        # semantically null but
                                                    # equal-magnitude change to a
                                                    # (e.g. equal-ΔE color shift)
    V^+ = M(G with s_o = s_o^+, contract C_o)       # frames t*..t*+H
    V^- = M(G with s_o = s_o^-, contract C_o)
    V^0 = M(G with s_o = s_o^0, contract C_o)       # matched-perturbation control
    # all three rollouts share identical seeds, prompts, and camera trajectory

PHASE 3 — Closed-Loop Perceptual Realization Score (difference-of-differences)
  for each frame t in [t*, t*+H]:
    delta_t   = LPIPS(V^+[t] ⊙ m_o, V^-[t] ⊙ m_o)      # change INSIDE mask, rule
    delta0_t  = LPIPS(V^0[t] ⊙ m_o, V^-[t] ⊙ m_o)      # change INSIDE mask, control
    leak_t    = LPIPS(V^+[t] ⊙ ¬m_o, V^-[t] ⊙ ¬m_o)    # change OUTSIDE mask, rule
    leak0_t   = LPIPS(V^0[t] ⊙ ¬m_o, V^-[t] ⊙ ¬m_o)    # change OUTSIDE mask, control
    match_t   = 1 - d_attr(phi(V^+[t] ⊙ m_o), s_o^+)    # phi = CLIP + attr probe
    match0_t  = 1 - d_attr(phi(V^0[t] ⊙ m_o), s_o^0)
    # difference-of-differences subtracts renderer divergence from ANY
    # equal-magnitude conditioning change
    Realization(r,o) = mean_t [ match_t - match0_t ]
    Leakage(r,o)     = mean_t [ (leak_t - leak0_t) / (delta_t - delta0_t + eps) ]
  # feedback: feed (match_t, leak_t) back into G to re-execute r

PHASE 3b — Divergence-Onset Estimate and Seed-Variance Check
  for each scene d:
    # divergence onset: first frame t where LPIPS(V^+[t] ⊙ ¬m_o, V^-[t] ⊙ ¬m_o)
    # exceeds the 95th percentile of the intra-seed LPIPS distribution
    # (computed from K=3 repeat rollouts of V^- with identical inputs)
    H_d = min(t_div - t*, H_max=8)   # per-scene horizon; H_d >= 1 required
    # seed-variance check: intra-seed Realization variance across K=3 repeats
    # must be < tau = 0.05; scenes failing this are flagged 'seed-unstable'
    # and excluded from the primary Margin ranking (reported separately)

PHASE 4 — Identifiability Margin (unseen-object test)
  for o in D_unseen:
    bind_acc(o) = IoU(m_o, GT_mask(o)) at t*
    Marg(o) = Realization(r,o) - Realization(r, o_perm)
    # o_perm = same rule applied to a randomly re-bound WRONG object
    #          (permuted binding, same matched-perturbation control applied)
  report Marg_seen vs Marg_unseen; flag scenes where Marg < tau
```

Hyperparameters: T0 = 16 warm-up frames before first appearance; K = 3 intervention seeds (reduced from 8, with variance-based early stopping: if intra-seed Realization variance < tau/2 after K=2, stop); tau = 0.05 margin threshold; H_max = 8 post-t* scoring frames; mask backbone fixed to SAM2 (not swapped per scene); matched-perturbation control uses equal-ΔE color shift for color rules and equal-magnitude pose jitter for pose rules.

**Why this design.** We chose a *frozen* first-appearance mask over re-deriving the mask every frame because re-derivation lets the renderer's own drift redefine which pixels are accountable, which would let a model that fails to bind still score well by moving the mask onto whatever it changed — accepting the cost that a genuinely occluded object will be penalized even when the renderer is correct. We chose a *paired null counterfactual with a matched-perturbation control* (V^+ vs V^- vs V^0 under identical seeds) over comparing against a ground-truth video because ground-truth video does not exist for user-defined rules, a paired rollout cancels renderer stochasticity, and the V^0 control subtracts the divergence that any equal-magnitude conditioning change induces in an autoregressive model — accepting the cost that the benchmark requires the world model to be seedable, deterministic, and to accept a semantically null perturbation. We chose the *leakage ratio* as a separate term rather than folding it into Realization because a model can realize the rule on the designated object while also repainting the whole scene, and a single score would hide this — accepting the cost of a two-number report that is harder to rank on. We chose the *permuted-object margin* over a simple seen/unseen accuracy split because permutation directly tests whether the binding is causal rather than coincidental, accepting the cost of K extra rollouts per rule. We chose a *short post-t* horizon with a measured divergence-onset estimate* over averaging over [t*, T] because the autoregressive confound grows with rollout length, and a fixed horizon would either truncate informative frames or admit divergent ones — accepting the cost that long-horizon rule effects are not scored. A domain expert would not describe FIRST-BIND as a variant of WorldRuler: WorldRuler proposes a generation mechanism (the rule graph and closed loop), whereas FIRST-BIND proposes a measurement contract and identifiability test that WorldRuler itself does not contain, and could be applied to any rule-conditioned world model including a GenieRedux-style baseline (E1) that has no rule graph at all.

**Why it measures what we claim.** `Realization(r,o)` measures rule-to-visual realization because the paired-null-plus-matched-control assumption that V^+ and V^- differ only in the rule-specified attribute state, after subtracting the equal-magnitude control V^0, makes the masked LPIPS-plus-attribute-match difference-of-differences attributable to the rule; this assumption fails when the world model's seed does not fully control rollout stochasticity or when the control perturbation is not matched in magnitude, in which case Realization reflects renderer noise rather than rule realization (and the seed-variance check flags the scene). `Leakage(r,o)` measures rule locality because the assumption that a correctly bound rule should not alter pixels outside its frozen mask, beyond what the matched control alters, makes out-of-mask difference-of-differences a direct signal of mis-binding; this assumption fails when a rule legitimately triggers a physically coupled secondary effect (e.g., a pushed object occluding a neighbor), in which case Leakage reflects correct physics rather than binding error. `Marg(o)` measures identifiability of the binding because the assumption that a permuted wrong-object binding should score strictly lower makes the margin a causal test of the contract; this assumption fails when the renderer ignores the binding entirely and produces the same output for both bindings, in which case Marg collapses to zero and reflects renderer insensitivity rather than binding failure. `bind_acc(o)` measures first-appearance grounding because the assumption that the GT mask at t* is the unique correct referent makes IoU a valid binding score; this assumption fails under heavy occlusion at first appearance, in which case bind_acc reflects mask ambiguity rather than a grounding error.

## Contribution

(1) FIRST-BIND, a benchmark and protocol that operationalizes rule-to-visual realization via a frozen first-appearance grounding contract, paired-null rule intervention, and a closed-loop realization/leakage score. (2) An identifiability margin based on permuted-object binding that separates 'the rule was realized on the right object' from 'the renderer changed something', yielding a design principle: realization accuracy must be reported jointly with leakage and margin, since any one alone is gameable. (3) A held-out unseen-object split and evaluation harness applicable to any rule-conditioned interactive world model, including models without an explicit rule graph.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | WorldRuler rule-graph scenes with held-out object split | Directly supplies rule graphs, state replacement, unseen objects |
| Synthetic harness | Scripted renderer with known single-attribute causal effect | Ground-truth effect size known by construction |
| Primary metric | Identifiability Margin (seen vs unseen) | Causal binding test, not just rendering fidelity |
| Secondary metric | Realization (difference-of-differences vs matched control) | Isolates rule effect from renderer divergence |
| Secondary metric | Intra-seed Realization variance (K=3) | Verifies seed-determinism assumption empirically |
| Baseline | WorldRuler closed-loop generator | Tests whether our contract adds measurement power beyond mechanism |
| Baseline | GenieRedux-style rule-free world model | Tests applicability to models lacking an explicit rule graph |
| Baseline | Naive seen/unseen realization split | Tests whether permutation margin is necessary vs accuracy |
| Ablation-of-ours | Re-derived per-frame masks (no frozen contract) | Isolates contribution of the first-appearance freeze |
| Ablation-of-ours | No matched-perturbation control (V^+ vs V^- only) | Isolates contribution of divergence subtraction |

### Why this setup validates the claim

The central claim is that FIRST-BIND measures rule-to-visual realization *causally* — that a rule-conditioned world model actually binds the rule to the designated object rather than merely repainting something plausible. The synthetic harness is the first line of validation: a scripted renderer with a single known attribute change lets us check that Realization and Margin recover the known effect size within tau = 0.05, grounding the sufficiency claim empirically rather than by assertion. The WorldRuler baseline tests the sub-claim that a generation mechanism alone does not guarantee measurable, attributable binding: if WorldRuler scores high on Realization but shows a small Identifiability Margin, the mechanism exists but the binding is not causal, which is exactly what our metric is built to expose. The GenieRedux-style baseline tests generality: it has no rule graph, so it should fail the FAGC contract outright, confirming the benchmark discriminates models by binding capability rather than by raw video quality. The naive seen/unseen realization split is included because it is the obvious alternative measurement; if it produces the same ranking as the permuted margin, our permutation machinery is unnecessary. The frozen-mask ablation isolates the one design choice we argue is load-bearing: re-deriving masks per frame should inflate scores precisely on scenes where the renderer drifts, so the ablation should show inflated Realization with collapsed Margin. The no-control ablation isolates the divergence-subtraction contribution: without V^0, Leakage should be dominated by renderer divergence and Margin should be inflated on unseen objects where the conditioning perturbation is largest. The intra-seed variance metric directly verifies the seed-determinism assumption: scenes with variance above tau are flagged and reported separately, so the primary ranking rests only on scenes where the paired-null assumption holds. The primary metric is the right detector because it pairs a realization score with a causal contrast, so a model that renders beautifully but ignores the binding cannot hide behind fidelity.

### Expected outcome and causal chain

**vs. WorldRuler closed-loop generator** — On a case where a rule changes an unseen object's material, WorldRuler's closed loop re-executes and repaints, but its perceptual feedback can converge on any visually plausible change, including changes to neighboring objects, because it has no frozen accountability region. Our method instead freezes the first-appearance mask and pairs the rollout against a null counterfactual and a matched control, so the change is attributed only inside the contract and only beyond what an equal-magnitude null perturbation induces. We expect WorldRuler to show high Realization but a small or near-zero Identifiability Margin on unseen objects, with the gap concentrated on unseen-object scenes and parity on seen-object scenes.

**vs. GenieRedux-style rule-free world model** — On a case where a rule specifies an attribute replacement, the rule-free model has no mechanism to consume the rule graph, so its V^+, V^-, and V^0 rollouts are driven only by prompt and seed, producing near-identical outputs inside the mask. Our method still runs the contract and produces a measurable margin because the rule-conditioned generator receives the state replacement. We expect the rule-free baseline to show near-zero delta_t - delta0_t and near-zero Margin across all subsets, confirming the benchmark is measuring binding rather than generic video quality.

**vs. Naive seen/unseen realization split** — On a case where the renderer is insensitive to binding, both the correct and permuted bindings produce similar outputs, so the naive split reports a high realization score for both seen and unseen objects and cannot distinguish a bound from an unbound model. Our method's permuted-object margin collapses to zero in exactly this case, flagging renderer insensitivity. We expect the naive split to rank models similarly regardless of binding quality, while our margin separates them, with the divergence largest on unseen objects.

**vs. Synthetic harness (known causal effect)** — On a scripted renderer where the rule's effect on the designated pixels is known by construction, Realization should recover the known effect size within tau = 0.05 and Margin should be positive and stable across K=3 seeds. If Realization deviates from the known effect by more than tau, the difference-of-differences estimator is miscalibrated and the primary metric is not trustworthy.

### What would falsify this idea

If our Identifiability Margin gain over the naive split is uniform across seen and unseen subsets rather than concentrated where the predicted binding failure occurs, or if the frozen-mask ablation shows no inflation of Realization on drift-prone scenes, or if the synthetic harness fails to recover the known causal effect within tau, then the claim that the frozen contract and permutation test capture causal binding is wrong.

## References

1. WorldRuler: Interactive World Modeling under Configurable Rules
