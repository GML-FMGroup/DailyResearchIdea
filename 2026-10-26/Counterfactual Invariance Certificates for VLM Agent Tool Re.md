# Counterfactual Invariance Certificates for VLM Agent Tool Repair and Task Success

## Motivation

Across both HexaShell-style robot coding agents and single-forward-pass VLM navigation agents, the process that produces an action or a repaired tool is also the sole oracle that declares it correct, so the generator's blind spots are inherited by the success signal and no independent validity is ever established. In HexaShell the VLM reads its own logs, diagnoses the failure, writes the repair, and then its frozen-tool test on ten new layouts is still judged by the same harness the agent authored; in AutoEval the automatic success detector replaces human judgment but is itself a learned readout that shares the policy's perceptual biases. The structural problem is self-certification, and no existing verification scheme (process reward models, self-consistency, learned verifiers) escapes it because every such verifier is trained on or reads the generator's own distribution.

## Key Insight

Correctness of a repair or a trajectory is a property of the environment's response to semantically irrelevant interventions, so an evaluator that only observes outcome stability under independently generated equivalence-preserving perturbations of the execution context is structurally independent of the generator even though it never inspects the generator's state.

## Method

```pseudocode
# CIC: Counterfactual Invariance Certificate
# Inputs: agent's claim C (a repaired tool T, or a claimed-success trajectory tau)
#         execution context E (logs, tool args, sensor stream, scene state)
#         perturbation generator G (independent of agent), evaluator O (never sees agent internals)
#         golden set D_gold = {(artifact, ground_truth_label)} of size N_gold=40, labeled by an
#           independent ground-truth signal (human expert OR a physically distinct simulator)
#           that shares no code with the agent's harness
#
# STATED ASSUMPTION (load-bearing): the four perturbation families are true semantic-equivalence
# transformations of E, so a correct artifact must produce identical behavioral signatures across
# all 32 perturbed contexts while an incorrect one must produce a detectable difference. This
# assumption is NOT asserted; it is empirically calibrated and falsification-tested in Phase 0.

PHASE 0 - EQUIVALENCE-CLASS CALIBRATION AND FALSIFICATION (runs once, before any agent claim):
  # 0a. Per-family natural behavioral variance from the golden set
  for each perturbation family f in {log-reorder, arg-rename, sensor-noise-replay, layout-jitter}:
     for each known-correct artifact a in D_gold (label==correct):
        run a under E and under K_f=8 perturbed contexts E_f^1..E_f^8
        record stability_f(a) = fraction of k with sig_k == sig_0
     mu_f = mean over correct a of stability_f(a)
     sigma_f = std  over correct a of stability_f(a)
     # per-family acceptance threshold: lower 5th percentile of correct-artifact stability,
     # floored at 0.75 to avoid degenerate thresholds
     theta_f = max(0.75, mu_f - 1.645 * sigma_f)
  # 0b. Invariant-checker falsification test
  #     inject known-semantics-CHANGING perturbations (e.g., swap two tool args non-bijectively,
  #     reorder log lines that carry causal order, jitter layout beyond declared tolerance)
  for each family f:
     for each injected semantics-changing perturbation p of family f:
        assert invariant_checker(E, p(E)) == False   # must be REJECTED
     false_equivalence_rate_f = fraction of injected p wrongly accepted
     assert false_equivalence_rate_f <= 0.05
  # 0c. Per-family false-reject / false-certify rate against golden set
  for each family f:
     false_reject_f   = fraction of correct a in D_gold with stability_f(a) < theta_f
     false_certify_f  = fraction of incorrect a in D_gold with stability_f(a) >= theta_f
  # 0d. Coverage audit (assumption A: families jointly span the semantic-equivalence class)
  enumerate correctness-relevant axes of the task (e.g., coordinate frame, argument identity,
    sensor noise, spatial layout, temporal event ordering, resource accounting)
  map each axis to a family; flag any axis with no covering family as UNCOVERED and report it
  # 0e. Non-redundancy check (assumption: families are mutually non-redundant)
  compute pairwise correlation of per-family stability vectors over D_gold; flag pairs with r>0.8

PHASE 1 - EQUIVALENCE CLASS CONSTRUCTION (generator G):
  for each perturbation family f in {log-reorder, arg-rename, sensor-noise-replay, layout-jitter}:
     produce K_f = 8 perturbed contexts E_f^1..E_f^8 by applying f to E
     # each f is certified semantically-equivalent by a domain invariant checker:
     #   log-reorder: multiset of log lines preserved
     #   arg-rename: bijective renaming of tool argument identifiers, applied consistently
     #   sensor-noise-replay: re-draw noise from the SAME calibrated sensor model, seed varies
     #   layout-jitter: rigid transform within the declared task-equivalence tolerance
     assert invariant_checker(E, E_f^k) == True for all k
     # perturbation seeds are drawn AFTER C is frozen (held-out seeds, not visible to agent)

PHASE 2 - OUTCOME PROBING (evaluator O, blind to agent):
  for each E_f^k:
     if C is a repaired tool T:
        run T on E_f^k in the simulator; record behavioral signature sig_k = (success, output_hash, resource_usage)
     if C is a trajectory tau:
        replay tau under E_f^k; record terminal_success_k in {0,1}
  baseline: run T / tau under the UNPERTURBED E to get sig_0 / success_0

PHASE 3 - CERTIFICATE COMPUTATION (evaluator O):
  invariance_score = fraction of k with sig_k == sig_0   (over all families, |K|=32)
  per_family_score = min over f of (fraction stable within family f)
  # thresholds are CALIBRATED per family in Phase 0, not fixed constants:
  #   aggregate threshold theta_agg = mean_f theta_f  (replaces fixed 0.94)
  #   per-family threshold theta_f as computed in Phase 0a (replaces fixed 0.875)
  CERTIFY(C) = 1 iff invariance_score >= theta_agg AND per_family_score >= min_f theta_f
  else REJECT and return the failing family f* + the perturbed context E_f*^k that broke it
  # certificate is reported as a calibrated statistical test: alongside CERTIFY, report
  # (false_reject_f*, false_certify_f*) from Phase 0c for the deciding family

PHASE 4 - FEEDBACK (only failing family id + context returned; never the agent's diagnosis):
  agent receives (f*, E_f*^k) and must revise C; loop to Phase 1, max 3 rounds
```

**Why this design.** We chose a separate evaluator that only runs the artifact under perturbed contexts over a learned verifier trained on the agent's outputs because a learned verifier inherits exactly the generator distribution it was trained on (the failure mode of process reward models and self-consistency), accepting the cost that our evaluator can only certify properties the perturbation families can actually probe. We chose to require both an aggregate invariance score and a worst-family minimum over a single averaged score because a tool can pass on average while being silently brittle to one equivalence class (e.g., argument renaming), accepting the cost of a stricter acceptance threshold and more agent revisions. We chose to return only the failing family identifier and the breaking context rather than the full perturbation set because exposing the whole set would let the agent overfit to the specific perturbations (turning the certificate into another self-certification channel), accepting the cost of slower convergence across rounds. Unlike AlphaEvolve, whose evaluators define the search objective and are supplied by the problem author, our evaluator is not an objective the generator optimizes against directly — it is a held-out invariance test whose perturbation seeds are drawn after the artifact is frozen, so a domain expert would not describe CIC as an evaluator-in-the-loop evolutionary search. We added Phase 0 because the equivalence-class definition is itself a hand-authored oracle; calibrating per-family thresholds from a golden set and falsification-testing the invariant checker makes the certificate a validated statistical test rather than an asserted one, accepting the cost of a one-time calibration run over N_gold=40 artifacts.

**Why it measures what we claim.** The aggregate invariance score measures *structural correctness of the repair* because it assumes the four perturbation families are true semantic-equivalence transformations of the execution context (a tool that is correct must produce identical behavior on any member of the equivalence class); this assumption fails when a perturbation family accidentally changes task semantics (e.g., layout-jitter exceeds the task's true tolerance), in which case the score reflects sensitivity to a genuine semantic change rather than brittleness — Phase 0b's falsification test and Phase 0c's false-equivalence rate bound this failure. The worst-family minimum measures *robustness of the success claim across independent failure modes* because it assumes the four families are mutually non-redundant probes (each targets a different axis the generator could have overfit); this assumption fails when two families are correlated (e.g., arg-rename and log-reorder both touch the same log format), in which case the minimum under-weights the untouched axis — Phase 0e's pairwise correlation check flags such redundancy. The blindness of the evaluator to agent internals measures *independence from the generator* because it assumes the perturbation generator G shares no parameters or prompts with the agent; this assumption fails when G is itself an LLM prompted with the agent's own logs, in which case the certificate degrades toward self-certification. The certificate threshold measures *acceptance reliability* because it assumes the 32 perturbed contexts are a representative sample of the equivalence class; this assumption fails for high-variance environments (e.g., stochastic contact dynamics), in which case the score reflects simulator noise rather than the artifact's invariance — Phase 0a's per-family variance estimate and Phase 0c's false-reject rate quantify this. The coverage audit in Phase 0d measures *span of the equivalence class* because it assumes the four families jointly cover every correctness-relevant axis; this assumption fails when a correctness-relevant axis (e.g., temporal ordering of sensor events) is covered by no family, in which case the certificate is silent on that axis and the audit flags it as UNCOVERED.

## Contribution

(1) A counterfactual-invariance certificate protocol that decouples the generator from the oracle by making acceptance a property of the environment's response to independently generated, invariant-checked perturbations of the execution context, with a worst-family stability criterion that a single averaged score cannot provide. (2) A finding that worst-family stability (rather than aggregate stability) is the discriminating signal between genuinely repaired tools and tools that merely pass the agent's own frozen-layout test, because aggregate scores are dominated by the easiest perturbation family. (3) An open perturbation-family library with per-family invariant checkers (log multiset, argument bijection, sensor-noise replay, layout rigid transform) that makes the certificate reproducible across HexaShell-style robot coding and replayable-log navigation settings.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Robot tool-repair tasks across 10 train / 10 held-out layouts (HexaShell) | Held-out layouts test generalization beyond fitting layouts. |
| Dataset (calibration) | Golden set D_gold of N_gold=40 artifacts (20 correct, 20 incorrect) labeled by human expert or physically distinct simulator | Calibrates per-family thresholds and false-reject/certify rates. |
| Primary metric | Held-out-layout success rate under CIC-certified artifacts | Directly measures invariance transfer, not in-distribution fit. |
| Secondary metric | Per-family false-reject / false-certify rate vs. D_gold | Reports certificate as calibrated statistical test. |
| Baseline | HexaShell frozen-tool generalization | Tests whether CIC adds value over layout-split freezing. |
| Baseline | AlphaEvolve-style evaluator-in-the-loop search | Tests whether held-out perturbation beats objective optimization. |
| Baseline | Learned verifier / process reward model | Tests whether blind invariance beats learned self-certification. |
| Ablation-of-ours | CIC with aggregate score only (no worst-family min) | Isolates the worst-family requirement's contribution. |
| Ablation-of-ours | CIC with fixed thresholds (0.94/0.875) instead of Phase-0-calibrated thresholds | Isolates the calibration contribution. |
| Ablation-of-ours | CIC with invariant checker not falsification-tested (Phase 0b skipped) | Isolates the equivalence-class validation contribution. |

### Why this setup validates the claim

The central claim is that a blind, held-out invariance certificate over semantic-equivalence perturbations produces repairs and success claims that transfer to genuinely new layouts, whereas objective-optimizing and learned-verifier baselines overfit the distribution they were scored against. The dataset choice is load-bearing: HexaShell's train/held-out layout split is the natural proxy for an equivalence class whose members the agent never saw, so held-out-layout success is the only metric that can distinguish true structural correctness from memorized layout specifics. The golden set D_gold is load-bearing for the certificate's validity: it supplies the per-family natural behavioral variance from which Phase 0a sets theta_f, and the false-reject/false-certify rates from Phase 0c that turn the certificate from a bare threshold into a calibrated statistical test. Each baseline tests a distinct sub-claim. HexaShell's frozen-tool protocol tests whether CIC's perturbation probing adds anything beyond simply freezing after training. AlphaEvolve-style search tests whether optimizing against an explicit evaluator, as opposed to being tested by a held-out one, produces brittle artifacts. The learned verifier tests whether a verifier trained on the agent's own output distribution inherits that distribution's blind spots. The aggregate-only ablation is the crucial control: if removing the worst-family minimum does not hurt, the design's stated rationale for that requirement is wrong. The fixed-threshold and un-falsification-tested ablations are the calibration controls: if they match full CIC, Phase 0 is doing no work and the equivalence-class assumption is unvalidated. The metric is right because invariance failures should manifest as layout-specific brittleness, which a held-out-layout success rate detects and an in-distribution score would mask.

### Expected outcome and causal chain

**vs. HexaShell frozen-tool generalization** — On a case where a repaired tool hardcodes a coordinate assumption that holds across the ten training layouts but breaks when the held-out layout places the target near a boundary, the frozen baseline produces a confident but geometrically wrong grasp because its repair was validated only on layouts it had seen. Our method instead rejects that repair in Phase 3 because layout-jitter perturbations within the declared tolerance already expose the hardcoded assumption, forcing the agent to revise toward a relative-coordinate repair. We expect a clear gap on held-out layouts with boundary-adjacent targets and near-parity on interior targets, with the gap concentrated exactly where the latent assumption is triggered.

**vs. AlphaEvolve-style evaluator-in-the-loop search** — On a case where the agent's tool succeeds on the training layouts by exploiting a simulator quirk in resource accounting, the AlphaEvolve-style baseline produces a high-scoring artifact because its evaluator rewards the score the artifact was evolved to maximize, and the quirk is inside that objective. Our method instead certifies against a held-out invariance test whose perturbation seeds are drawn after the artifact is frozen, so the quirk shows up as instability under sensor-noise-replay and arg-rename. We expect the baseline to match or beat us on the training-layout score while losing noticeably on held-out layouts, a crossover pattern rather than a uniform gap.

**vs. Learned verifier / process reward model** — On a case where a success trajectory passes only because the agent's own reasoning trace rationalizes a skipped preconditions check, the learned verifier produces a high acceptance probability because it was trained on the same generator distribution that produced the rationalization. Our method instead replays the trajectory under reordered logs and renamed arguments, where the skipped precondition becomes observable as a terminal-success change. We expect the verifier to accept a substantial fraction of trajectories that our certificate rejects, and we expect those rejected trajectories to be the ones that fail on held-out layouts, so the baseline's in-distribution acceptance advantage reverses on transfer.

**vs. CIC with fixed thresholds (0.94/0.875)** — On a case where a correct tool legitimately varies under sensor-noise-replay because the calibrated sensor model has high variance, the fixed-threshold variant falsely rejects it, while Phase-0-calibrated theta_f (lower 5th percentile of correct-artifact stability) accepts it. We expect the calibrated variant to have lower false-reject rate on D_gold and higher held-out-layout success, with the gap concentrated on the high-variance family.

**vs. CIC with invariant checker not falsification-tested (Phase 0b skipped)** — On a case where a perturbation family accidentally changes task semantics (e.g., layout-jitter exceeds true tolerance), the un-falsification-tested variant certifies a brittle tool because the semantics-changing perturbation is wrongly accepted as equivalent. We expect the falsification-tested variant to reject such tools and show higher held-out-layout success, with the gap concentrated on the family whose checker fails Phase 0b.

### What would falsify this idea

If the aggregate-only ablation matches the full CIC on held-out-layout success, the worst-family requirement is doing no work and the design's core rationale collapses. If the fixed-threshold and un-falsification-tested ablations match full CIC, Phase 0 calibration is doing no work and the equivalence-class assumption is unvalidated. If CIC's gain over the learned verifier is uniform across all subsets rather than concentrated on the perturbation families predicted to expose each baseline's failure mode, the mechanism story is wrong even if the headline number is higher.

## References

1. HexaShell: A Thin Harness for Exploring the Limits of VLM Agents as Robot Coding Agents
2. AlphaEvolve: A coding agent for scientific and algorithmic discovery
3. MoGe-2: Accurate Monocular Geometry with Metric Scale and Sharp Details
4. AutoEval: Autonomous Evaluation of Generalist Robot Manipulation Policies in the Real World
5. MoGe: Unlocking Accurate Monocular Geometry Estimation for Open-Domain Images with Optimal Training Supervision
6. Multi-objective Evolution of Heuristic Using Large Language Model
7. Amplifying human performance in combinatorial competitive programming
8. Autonomous Improvement of Instruction Following Skills via Foundation Models
