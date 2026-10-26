# GenExec: Generative Action-Sequence Synthesis with Simulator-Verified Execution for Multimodal Agent and World-Model Evaluation

## Motivation

Across the entire multimodal-agent and world-model evaluation lineage, competence is operationalized as discriminative selection among provided counterfactual candidates — WorldPrediction asks models to distinguish the correct action or action ordering from distractors, and Dual Exam's World Model track fixes action interventions and scores state-evolution prediction, while its Agent track fixes environment responses and scores action selection. This shared protocol structurally cannot test the missing property: whether an agent can generate an executable action sequence and whether a world model's predicted outcome matches what actually happens when that sequence is executed. Because no candidate set is supplied and no execution is performed, neither benchmark can distinguish a model that recognizes correct actions from one that can synthesize and close the loop on them.

## Key Insight

Executability in a deterministic simulator is a verifiable, non-learned ground-truth signal, so requiring a synthesized action sequence to actually run to the goal and requiring the world model's predicted terminal state to match the simulator's executed terminal state turns evaluation into a closed generative-execution loop that discriminative candidate selection cannot approximate.

## Method

## GenExec: Generative Action Closure under Execution

**(A) What it is.** GenExec is an evaluation protocol (with an accompanying harness) that takes as input a scenario specification `S = (initial_state s0, goal predicate g, simulator M, action vocabulary A, horizon H)` and two systems — an *agent* that emits an action sequence and a *world model* that predicts its outcome — and outputs three coupled scores: Executability (E), Outcome Fidelity (F), and Closure (C). No candidate action sets are ever shown to either system.

**(B) How it works.**

```pseudocode
Input: scenario S = (s0, g, simulator M, action vocab A, horizon H)
Input: agent AG, world model WM
Input: calibration set S_cal = {S_1..S_512} (disjoint from evaluation scenarios)

Phase 0 — Marginal calibration (bias correction for closure)
  # Estimate each system's marginal terminal-state distribution on S_cal
  # by querying with the true executed sequence from M.
  For each S_i in S_cal:
      a_1..a_k = M.oracle_plan(S_i)          # simulator's own solution
      s_T_i = M.execute(s0_i, a_1..a_k)      # ground-truth terminal
      s_hat_T_i = AG.predict_terminal(s0_i, a_1..a_k)
      s_tilde_T_i = WM.rollout(s0_i, a_1..a_k)
  Fit marginal distributions P_AG(s_hat_T) and P_WM(s_tilde_T) over S_cal
  (empirical histograms over discretized state space; 512 samples)

Phase 1 — Generative action synthesis (no candidates shown)
  a_1..a_k = AG.synthesize(s0, g, A)          # k <= H, agent's own stopping
  s_hat_T  = AG.predict_terminal(s0, a_1..a_k)

Phase 2 — Executability verification in simulator M
  s = s0
  for t in 1..k:
      if a_t not in A:            return E=0, reason="invalid_action"
      if not M.precondition_ok(s, a_t): return E=0, reason="precondition"
      s = M.step(s, a_t)          # deterministic transition
  E = 1 if g(s) else 0            # reached goal after execution
  s_T = s                         # executed terminal state (ground truth)

Phase 3 — World-model outcome prediction on the SAME sequence
  s_tilde_T = WM.rollout(s0, a_1..a_k)
  F = sim(s_T, s_tilde_T)         # task-relevant state similarity in [0,1]

Phase 4 — Generative action closure (bias-corrected)
  # Raw agreement between agent's self-report and WM's prediction
  raw_agree = 1[sim(s_hat_T, s_tilde_T) >= tau]   # tau = 0.9
  # Chance agreement under independent marginals from Phase 0
  p_agree_chance = sum_{x} P_AG(x) * P_WM(x)      # product of marginals
  # Bias-corrected agreement (Cohen's kappa style)
  kappa = (raw_agree - p_agree_chance) / (1 - p_agree_chance + 1e-8)
  C = 1[E=1] * max(0, kappa) * F

Return (E, F, C) and per-scenario diagnostics
```

Aggregation: report E, F, C macro-averaged over scenarios, plus a *closure gap* `G = E - C` that isolates cases where the agent executed correctly but the agent–WM pair disagreed about the outcome. Hyperparameters: horizon `H` set per scenario family (physical 12, digital 20, social 8, scientific 15); `tau = 0.9` fixed before any model is run; `sim` is task-relevant state equality (exact match on symbolic state, IoU on object poses); calibration set size 512 scenarios per domain, drawn from the same scenario generator but with disjoint seeds.

**(C) Why this design.**
We chose *generative synthesis with no candidate set over discriminative selection* because a supplied distractor set leaks the answer distribution and lets a model score well by recognizing plausible actions rather than by planning — this is exactly the limitation of WorldPrediction, whose WM/PP tasks are forced-choice among counterfactuals; accepting the cost that GenExec cannot reuse existing multiple-choice datasets and must build simulator-backed scenarios. We chose *simulator execution as the ground-truth verifier over a learned reward or LLM judge* because a deterministic simulator gives an assumption-free, reproducible executability signal, accepting the cost that scenarios are restricted to domains with a faithful simulator (physical/digital/social/scientific sandboxes) and cannot cover open-ended real-world video. We chose *three coupled scores (E, F, C) over a single aggregate* because collapsing executability and outcome fidelity into one number hides the failure mode Dual Exam already observed (component scores not transferring to coupled success), accepting the cost of a richer reporting surface. We chose *feeding the executed sequence to the world model rather than the agent's intended sequence* because it isolates world-model error from agent error on identical inputs, accepting the cost that the WM is evaluated off-policy relative to its own generation. We chose *bias-corrected closure via marginal calibration* because raw agreement between two models from the same pretraining prior can be inflated by shared bias; the calibration set estimates each system's marginal terminal-state distribution, and the kappa-style correction discounts agreement that would be expected by chance, accepting the cost of a 512-scenario calibration set per domain and the assumption that the calibration scenarios are representative of the evaluation scenarios. A domain expert would not describe GenExec as a variant of WorldPrediction or Dual Exam: those methods score selection among provided candidates under fixed interventions, whereas GenExec requires the agent to produce a sequence that must run to the goal and requires the WM to match the simulator's executed terminal state — a property neither benchmark's mechanism can produce.

**(D) Why it measures what we claim.**
Executability `E` measures *generative action competence* because the simulator's precondition and goal checks are the operational definition of an executable plan, under the assumption that the simulator faithfully encodes the scenario's causal dynamics; this assumption fails when the simulator is under-specified (e.g., missing a physical constraint), in which case `E` reflects simulator coverage rather than agent competence. Outcome Fidelity `F` measures *world-model predictive faithfulness* because comparing `WM.rollout(s0, a_1..a_k)` to `M.step`-chained `s_T` on the *same* sequence holds the input fixed and isolates prediction error, under the assumption that `sim` captures task-relevant state; this assumption fails when `sim` ignores a state dimension the goal depends on, in which case `F` reflects surface similarity rather than causal state tracking. Closure `C` measures *agent–world-model agreement under execution* because it requires the agent's self-reported terminal state and the WM's predicted terminal state to agree *and* the executed goal to be reached, under the assumption that agreement on the executed sequence is necessary for sustained reciprocal improvement; this assumption fails when both systems share a correlated bias (e.g., same pretraining prior), in which case `C` reflects shared bias rather than genuine closure. The bias-corrected kappa term in Phase 4 explicitly discounts agreement expected from the product of marginals, so `C` measures *agreement beyond chance* under the assumption that the calibration set's marginals are representative; this assumption fails when the evaluation scenarios have systematically different terminal-state distributions from the calibration set, in which case `C` reflects calibration mismatch rather than coupling. The closure gap `G = E - C` measures *the coupling failure* Dual Exam's pilot could not diagnose, under the assumption that `E=1, C=0` isolates disagreement rather than simulator noise; this assumption fails when execution is stochastic, in which case `G` reflects variance rather than disagreement.

## Contribution

(1) GenExec, a generative-execution evaluation protocol that replaces discriminative counterfactual selection with agent-synthesized action sequences verified by a deterministic simulator, yielding Executability, Outcome Fidelity, and Closure scores plus a closure-gap diagnostic. (2) A design principle: feeding the *executed* sequence to the world model isolates world-model error from agent error on identical inputs, and the closure gap operationalizes the component-to-coupled transfer failure that Dual Exam's zero-success pilot could only observe. (3) A simulator-backed scenario suite spanning physical, digital, social, and scientific domains with fixed horizons and a pre-registered similarity threshold, enabling reproducible generative-execution evaluation of arbitrary agent and world-model configurations.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Simulator-backed scenarios across four domains (physical, digital, social, scientific); 200 evaluation scenarios per domain, 512 calibration scenarios per domain (disjoint seeds) | Deterministic ground truth enables execution verification without judge assumptions |
| Primary metric | Closure gap G = E − C | Isolates agent–WM coupling failure from raw competence |
| Baseline | WorldPrediction (discriminative WM/PP) | Tests whether candidate selection masks planning gaps |
| Baseline | Dual Exam (component-scored tracks) | Tests whether decoupled scores fail to predict coupled success |
| Baseline | Single-agent execution-only (no WM score) | Tests whether closure adds signal beyond executability |
| Ablation-of-ours | GenExec with WM fed agent's intended sequence | Tests whether off-policy WM input is necessary for closure |
| Ablation-of-ours | GenExec with shuffled self-report (s_hat_T replaced by random terminal claim from calibration set) | Tests whether bias correction isolates genuine coupling from correlated bias |
| Sensitivity | tau in {0.7, 0.8, 0.9, 0.95, 0.99} and sim in {exact-match, IoU, learned embedding} | Justifies operating point empirically |

### Why this setup validates the claim

The central claim is that generative synthesis plus simulator execution plus a coupled closure score exposes a failure mode—agent–world-model disagreement under execution—that neither discriminative benchmarks nor decoupled component scoring can detect. The dataset choice is load-bearing: only a faithful deterministic simulator can supply an assumption-free executability signal and a ground-truth terminal state to compare against the world model, so a positive result cannot be attributed to a lenient judge. WorldPrediction tests the sub-claim that candidate selection inflates apparent competence; if GenExec produces lower executability on scenarios where WorldPrediction scores high, the leak hypothesis is supported. Dual Exam tests the sub-claim that component scores fail to transfer to coupled success; if Dual Exam's strong component scores coexist with a large closure gap while GenExec's coupled score tracks executability, the coupling claim is supported. The execution-only baseline tests whether closure is redundant with executability. The ablation tests whether the *off-policy* WM input is what makes closure diagnostic rather than a restatement of F. The shuffled self-report ablation tests whether the bias-corrected closure term discounts shared-prior agreement: if C drops to near zero when s_hat_T is randomized, the kappa correction is isolating genuine coupling rather than correlated bias. The sensitivity analysis over tau and sim definitions tests whether the operating point is robust. The closure gap is the right metric because it is a difference of coupled quantities: it can only be large when execution succeeds but agreement fails, which is precisely the predicted effect and cannot be produced by a single-score benchmark.

### Expected outcome and causal chain

**vs. WorldPrediction** — On a case where the correct action sequence is long and the counterfactual distractors are superficially plausible, WorldPrediction's model produces the right selection because distractor elimination is easier than generation. Our method instead requires the agent to emit the sequence with no candidates and run it to the goal, so the same model must plan rather than recognize, and we expect a noticeable executability drop on long-horizon subsets with parity on short ones, concentrated where distractor plausibility is highest.

**vs. Dual Exam** — On a case where the agent reaches the goal but its self-reported terminal state diverges from what the world model predicts on the executed sequence, Dual Exam reports strong agent and WM component scores because the tracks are scored separately. Our method instead computes closure on the same executed sequence with bias correction, so the disagreement surfaces as a large closure gap G on exactly those cases, and we expect the gap to be concentrated in scenarios whose terminal state depends on a constraint the two systems encode differently.

**vs. Single-agent execution-only** — On a case where the agent executes correctly but the world model's rollout diverges from the simulator's terminal state, an execution-only score reports full success and hides the WM error. Our method instead requires the WM to match the executed terminal state before awarding closure, so we expect closure to fall well below executability on subsets where WM rollout error is large, with the gap shrinking to near zero where the WM is faithful.

**vs. GenExec with intended-sequence WM input (ablation)** — On a case where the agent's intended sequence differs from what it actually emitted, feeding the intended sequence to the WM lets the WM predict the agent's claim rather than the executed outcome, inflating closure. Our method instead feeds the executed sequence, so closure reflects genuine agreement on what ran, and we expect the ablation to show inflated closure on cases with emission drift and parity elsewhere, confirming that off-policy input is what makes closure diagnostic.

**vs. GenExec with shuffled self-report (ablation)** — On a case where both the agent and the world model share a pretraining prior that biases them toward a salient terminal state, raw agreement would be high even if both are wrong. Our method instead replaces s_hat_T with a random terminal claim drawn from the calibration marginal, so the kappa-corrected closure should drop to near zero if the correction is working; we expect C to be substantially lower than raw agreement on all subsets, with the largest drop on scenarios where the marginal distributions are most concentrated, confirming that bias correction isolates genuine coupling.

### What would falsify this idea

If the closure gap G is uniformly small across all subsets—including those where the predicted failure mode (agent–WM disagreement on executed sequences) should occur—then the coupled score adds nothing over executability and the central claim fails. Likewise, if GenExec's gain over WorldPrediction and Dual Exam is uniform rather than concentrated on the long-horizon and terminal-state-divergence subsets where the mechanism predicts it, the effect is not attributable to generative closure. Additionally, if the shuffled self-report ablation does not reduce C relative to raw agreement, the bias correction is not discounting shared-prior agreement and the closure score remains confounded.

## References

1. Dual Exam: Controlled Evaluation of Multimodal Agents and World Models
2. WorldPrediction: A Benchmark for High-level World Modeling and Long-horizon Procedural Planning
