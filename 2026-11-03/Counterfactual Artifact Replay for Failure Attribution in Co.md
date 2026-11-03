# Counterfactual Artifact Replay for Failure Attribution in Computer-Use Agents

## Motivation

Retrieval-based computer-use agents such as VISUALSKILL and Agent Workflow Memory certify their retrieved skills/artifacts only at construction time (via authored documentation plus live UI exploration, or via induced workflows), so at runtime a failed trajectory is a mixture of a stale/mis-specified artifact and a live grounding/planning error, and neither construction-time gate can separate the two because both assume the artifact is correct-by-construction. The field's convergent trajectory toward precomputed retrievable artifacts (VISUALSKILL's load_topic skills, ASI's programmatic skills, AWM's workflows) makes this mixture the new bottleneck: every tree's execution-based gate certifies the artifact offline but says nothing about whether a given runtime failure came from the artifact or from the live step that consumed it.

## Key Insight

Because artifacts are offline-materialized and therefore re-executable, intervening on the artifact while holding the live context fixed (and vice versa) yields a counterfactual contrast whose sign identifies which component is causally responsible, a contrast that retrieval-based attribution structurally cannot compute since it never re-instantiates the artifact.

## Method

**What it is.** CARA (Counterfactual Artifact Replay Attribution) takes as input a failed trajectory τ = (o_0, a_0, ..., o_T) produced by a computer-use agent, the retrieved skill/environment artifact A it loaded (e.g., a VISUALSKILL topic file or an ASI program), and the live grounding/planning context C (screenshots, plan state). It outputs a scalar attribution score s ∈ [−1, +1] where s > 0 means the artifact is implicated (stale/mis-specified) and s < 0 means the live grounding/planning step is implicated, plus a step-level localization of the first divergent action.

**Load-bearing assumption (stated explicitly).** CARA assumes that perturbing the artifact while holding the live context fixed (and vice versa) yields a replay outcome whose change is attributable solely to the perturbed component — i.e., that A and C are separable causes whose effects on replay success are independent and comparable in magnitude. This assumption is not assumed silently: Phase 4 measures the interaction term directly on a 2×2 intervention grid, and Phase 2b calibrates perturbation strengths on successful trajectories so the two families have matched marginal disruption before any ratio is formed. If the interaction term is large, CARA returns a set-valued attribution (artifact-implicated, live-implicated, or both) rather than a signed scalar.

**How it works.**

```pseudocode
Input: failed trajectory τ (length T), artifact A, live context C, replay budget K=5 per cell, calibration set S+ of 64 successful trajectories
Output: attribution score s (or set-valued label if interaction large), localized step t*, interaction term I

# Phase 1: identify the artifact-conditioned decision points
D = {t : action a_t was produced by consuming A}   # via load_topic / skill-call trace
if D empty: return s = -1 (pure live failure, no artifact consumed)

# Phase 2a: artifact-side counterfactual — perturb A, hold C fixed
for k in 1..K:
    A_k = perturb(A, mode_k)   # mode in {drop-topic, swap-topic, stale-screenshot-sub, reorder-steps}
    τ_k = replay(agent, C, A_k)   # re-execute from the first artifact-consuming step
    r_k = outcome(τ_k)            # binary success/failure of replay

# Phase 2b: live-side counterfactual — hold A fixed, perturb C
for k in 1..K:
    C_k = perturb(C, mode_k)   # mode in {re-ground-element, re-plan-subgoal, resample-plan-temperature}
    τ'_k = replay(agent, C_k, A)
    r'_k = outcome(τ'_k)

# Phase 2c: joint 2x2 intervention grid (measures interaction, does not assume zero)
for k in 1..K:
    τ''_k = replay(agent, C_k, A_k)   # both perturbed
    r''_k = outcome(τ''_k)
# neither-perturbed cell is the observed failure: r_0 = 0 by construction

# Phase 2d: perturbation-strength calibration on successful trajectories S+
# For each family, choose the perturbation magnitude (e.g., number of topics dropped,
# number of elements re-grounded) so that the marginal success-rate drop on S+ is equal:
#   Δ_A = mean_{τ+ in S+} [1 - mean_k r_k(τ+)]
#   Δ_C = mean_{τ+ in S+} [1 - mean_k r'_k(τ+)]
# Scale perturbation strength per family until |Δ_A - Δ_C| < 0.05.

# Phase 3: causal contrast with interaction term
p_A = mean(r_k)          # success rate under artifact perturbation only
p_C = mean(r'_k)         # success rate under live perturbation only
p_AC = mean(r''_k)       # success rate under both perturbations
p_0 = 0                  # observed failure
# Interaction term (deviation from additive separability):
I = p_AC - (p_A + p_C - p_0)   # if I ≈ 0, effects are additive; large |I| flags non-separability

# Phase 4: attribution output
if |I| > 0.2:   # threshold chosen on calibration set S+ where I should be near 0
    # set-valued attribution: report which components are implicated
    implicated = {}
    if p_A > 0.5: implicated.add('artifact')
    if p_C > 0.5: implicated.add('live')
    return implicated, t*, I
else:
    s = (p_A - p_C) / (p_A + p_C + eps)   # eps=1e-6
    return s, t*, I

# Phase 5: step localization
t* = argmin_t  KL( a_t(τ_k) || a_t(τ) )  over k with r_k=1
return s (or implicated set), t*, I
```

**Why this design.** We chose to perturb the artifact and re-execute against the *original* live context rather than re-run the whole task from scratch, because re-running from scratch confounds artifact staleness with environment drift, accepting the cost that replay must be able to re-instantiate the artifact in a sandboxed environment (VISUALSKILL's offline-materialized topic files and ASI's programs both satisfy this). We chose four structured perturbation modes (drop-topic, swap-topic, stale-screenshot substitution, step reordering) over random noise injection because random noise produces off-manifold artifacts whose replay outcome is uninformative, accepting the cost that the perturbation set must be hand-curated per artifact type. We chose the normalized contrast (p_A − p_C)/(p_A + p_C) over a raw difference because raw differences are not comparable across tasks with different replay success rates, accepting the cost that the score is undefined when both perturbation families yield zero success (handled by returning a tie and flagging for human review). We chose KL-divergence on the action distribution at each step for localization over attention-weight inspection because attention weights are not causally validated, accepting the cost of one extra forward pass per step. A domain expert would not describe CARA as a variant of RAG re-ranking or of Toolformer-style tool selection because those methods score artifacts by similarity to the query without ever re-instantiating them; CARA's signal comes from the *execution outcome* of a perturbed artifact, which is a quantity retrieval never computes. The 2×2 grid and calibration step were added because the original normalized contrast assumed symmetric, calibrated perturbation strengths and zero interaction; measuring I and equalizing Δ_A, Δ_C on successful trajectories makes the separability assumption testable rather than assumed.

**Why it measures what we claim.** The quantity p_A (success rate under artifact perturbation with C fixed) measures *artifact causal necessity* because the replay holds all live-context variables fixed by construction (same screenshots, same plan state, same model weights), so any change in outcome must flow through the artifact; this assumption fails when the agent's replay is non-deterministic (e.g., temperature > 0 sampling), in which case p_A reflects sampling noise rather than artifact necessity — mitigated by K=5 replays and reporting the variance. The quantity p_C (success rate under live perturbation with A fixed) measures *live-step causal sufficiency* because the artifact is held byte-identical across replays, so outcome changes must flow through the grounding/planning step; this assumption fails when the perturbation mode changes the task semantics (e.g., re-planning to a different subgoal), in which case p_C reflects task-difficulty change rather than live-step error. The normalized contrast s measures *relative attribution* because it is the ratio of the two causal effects, which equals the posterior odds of artifact-vs-live responsibility under a symmetric prior over the two failure modes; this assumption fails when the two perturbation families have systematically different effect sizes (e.g., artifact perturbations are always more disruptive), in which case s is biased toward artifact attribution — mitigated by calibrating perturbation strength on a held-out set of *successful* trajectories where s should be near zero. The interaction term I measures *non-separability* because it is the deviation of the joint-perturbation success rate from the additive prediction p_A + p_C − p_0; this assumption fails when the perturbation families interact (e.g., a stale artifact makes the live grounding more brittle), in which case I is large and CARA returns a set-valued attribution rather than a scalar.

## Contribution

(1) A counterfactual replay mechanism (CARA) that localizes runtime failures in computer-use agents to artifact staleness versus live grounding/planning error by exploiting the offline re-executability of retrieved artifacts, producing a normalized attribution score and a step-level localization without ground-truth failure labels. (2) A design principle: attribution between a retrieved artifact and the live step that consumed it is identifiable when the artifact is offline-materialized and re-executable, because the artifact can be intervened on while the live context is held fixed — a counterfactual that retrieval-based attribution cannot perform. (3) A calibration protocol that validates the attribution score on successful trajectories (where it should be near zero) and on synthetically corrupted artifacts (where it should be positive), providing a label-free sanity check for the mechanism.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | CUA-World failure trajectories with VISUALSKILL artifacts | Artifacts are offline-materializable, so replay is feasible |
| Primary metric | Attribution accuracy vs. human-labeled failure cause | Directly tests the artifact-vs-live localization claim |
| Secondary metric | Interaction term I and per-family effect sizes (Δ_A, Δ_C) | Tests the separability assumption directly |
| Baseline | No-skill agent (0.303) | Tests whether artifact attribution is even needed |
| Baseline | Matched text-only skill agent (0.373) | Tests attribution under text artifacts |
| Baseline | Agent S2 on OSWorld horizons | Tests attribution under a strong generalist agent |
| Ablation-of-ours | CARA with random-noise artifact perturbation | Isolates value of structured perturbation modes |
| Ablation-of-ours | CARA without Phase 2b calibration (uncalibrated Δ_A, Δ_C) | Tests whether calibration reduces bias toward artifact attribution |
| Ablation-of-ours | CARA without 2×2 grid (assumes I=0) | Tests whether measuring interaction improves attribution accuracy |

### Why this setup validates the claim

The central claim is that artifact causal necessity (p_A) and live-step causal sufficiency (p_C) are separable signals recoverable only by re-instantiating the artifact and replaying against a fixed live context. The dataset choice matters because CUA-World failures paired with VISUALSKILL topic files give us ground-truth artifact identities that can be sandboxed and perturbed — without materializable artifacts, replay is impossible and the claim untestable. Human-labeled failure causes (artifact-stale vs. live-grounding) provide the external criterion for the attribution score, so the primary metric is attribution accuracy, not task success: success rate would confound attribution quality with the agent's underlying competence. The no-skill baseline establishes the floor where no artifact is consumed, so CARA must return s = −1 by construction; this tests the Phase-1 gate. The text-only skill baseline tests whether attribution works when artifacts are non-visual, checking that the mechanism is not overfit to screenshot-bearing topics. The Agent S2 baseline tests attribution against a strong compositional agent whose failures may be dominated by live grounding, checking that CARA does not over-attribute to artifacts when the artifact is well-formed. The random-noise ablation is the decisive control: if structured perturbation modes carry the signal, the ablation should degrade attribution accuracy substantially, whereas if any perturbation works equally well, the design rationale for hand-curated modes is wrong. The calibration ablation tests whether equalizing Δ_A and Δ_C on successful trajectories reduces the bias toward artifact attribution that the original normalized contrast would exhibit when artifact perturbations are more disruptive. The 2×2-grid ablation tests whether measuring the interaction term I improves attribution accuracy on cases where the separability assumption fails.

### Expected outcome and causal chain

**vs. No-skill agent (0.303)** — On a case where the agent fails with no artifact loaded (e.g., a CUA-World task requiring a multi-step form fill with no VISUALSKILL topic), the no-skill baseline produces a wrong action because it has no procedural knowledge to retrieve, and any attribution method that scores artifacts by similarity would still assign nonzero blame. Our method instead returns s = −1 immediately because Phase 1 finds D empty — no action consumed an artifact. We expect near-perfect agreement with human labels on the no-artifact subset, with the gap between CARA and similarity-based attribution concentrated exactly on these no-artifact cases.

**vs. Matched text-only skill agent (0.373)** — On a case where a text-only skill describes a UI element that has since moved (e.g., a "Settings" button relocated), the text-skill baseline produces a click at the stale coordinates because its artifact encodes position implicitly and it cannot re-ground visually, and a RAG-reranker would still score the skill as relevant because the text matches the query. Our method instead perturbs the topic (drop/swap/stale-screenshot substitution) and observes that replay success changes, yielding s > 0 and localizing t* at the first divergent click. We expect a clear separation in attribution accuracy on the stale-artifact subset, with parity on the well-formed-artifact subset.

**vs. Agent S2 on OSWorld horizons** — On a case where Agent S2 fails at step 15 on a long-horizon task, the failure often stems from live re-grounding drift rather than a bad skill, but a naive attribution method that inspects attention over the skill would blame the skill because it was attended to. Our method instead holds the artifact byte-identical and perturbs the live context (re-ground, re-plan), observing that replay success rises, yielding s < 0. We expect CARA to correctly attribute live failures at a higher rate than attention-based attribution, with the advantage growing at the 50-step horizon where grounding drift compounds.

**vs. CARA without Phase 2b calibration** — On a case where artifact perturbations are more disruptive than live perturbations (e.g., dropping a topic removes all procedural knowledge), the uncalibrated contrast yields s > 0 even when the true failure is live grounding. Our method calibrates Δ_A and Δ_C on 64 successful trajectories so that |Δ_A − Δ_C| < 0.05, reducing this bias. We expect the calibrated version to have higher attribution accuracy on the live-grounding subset, with no loss on the artifact-stale subset.

**vs. CARA without 2×2 grid** — On a case where the artifact and live context interact (e.g., a stale artifact makes the live grounding more brittle), the additive assumption I=0 yields a scalar s that misattributes. Our method measures I and returns a set-valued attribution when |I| > 0.2. We expect the 2×2 version to have higher accuracy on the interaction-heavy subset, identified by high I values.

### What would falsify this idea

If CARA's attribution accuracy gain over the random-noise ablation is uniform across all failure subsets rather than concentrated on the stale-artifact and live-grounding-drift subsets where the causal contrast is predicted to operate, then the replay-based causal signal is not what drives the improvement. Equivalently, if the random-noise ablation matches structured perturbations on attribution accuracy, the design claim that off-manifold artifacts yield uninformative replay outcomes is wrong.

## References

1. VISUALSKILL: Multimodal Skills for Computer-Use Agents
2. Inducing Programmatic Skills for Agentic Tasks
3. Agent S2: A Compositional Generalist-Specialist Framework for Computer Use Agents
4. Agent Workflow Memory
5. Proposer-Agent-Evaluator(PAE): Autonomous Skill Discovery For Foundation Model Internet Agents
6. Agent S: An Open Agentic Framework that Uses Computers Like a Human
7. AgentStore: Scalable Integration of Heterogeneous Agents As Specialized Generalist Computer Assistant
8. OS-ATLAS: A Foundation Action Model for Generalist GUI Agents
