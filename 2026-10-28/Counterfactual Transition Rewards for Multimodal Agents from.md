# Counterfactual Transition Rewards for Multimodal Agents from Environment State Deltas

## Motivation

Across the current trajectory of multimodal agent self-improvement, the learning signal is a proxy that is never checked against the environment: TARL (Process-Supervised RL for Interactive Multimodal Tool-Use Agents) asks an LLM judge to assign turn-level credit, and VISUALSKILL asks humans to author per-application skill files, so in both cases the reward or skill content is validated only by another model's opinion or by manual documentation rather than by what actually changed in the environment. This is structurally fragile because the judge can be confidently wrong about a turn that silently corrupted a database or left a UI in an inconsistent state, and an authored skill can encode a procedure that no longer matches the live application. The field therefore lacks a reward/skill source whose correctness is self-verifying against the cheap, causally downstream state transitions that every interactive agent already logs.

## Key Insight

Because environment state transitions are deterministic functions of the agent's action given the pre-action state, a turn's contribution is identified by the change in the state delta when that action is replaced by the action the agent's own history predicts it would have taken under a matched pre-state — a within-trajectory contrast that needs no external judge and is falsifiable against the logged delta.

## Method

**What it is.** CTR-Delta (Counterfactual Transition Reward from Deltas) is a self-supervised procedure that takes an agent's logged interaction history H = {(s_t, a_t, s_{t+1})} over a multimodal environment (database rows, file tree, UI widget states) and outputs (i) a per-turn reward r_t and (ii) a set of skill snippets, both derived from comparing the observed state delta Δ_t = s_{t+1} − s_t against the counterfactual delta Δ_t^cf implied by a matched alternative action.

**Load-bearing assumption (stated explicitly).** The reward `sim(Δ_t, Δ^goal) − sim(Δ_t^cf, Δ^goal)` measures turn-level causal credit only under the assumption A that the matched transition (s_{j*}, a_{j*}, s_{j*+1}) is a valid substitute for the counterfactual outcome at s_t under a_t^cf, i.e., that the environment's transition function is effectively Markov in the typed fingerprint φ(s) so that matching on φ(s) removes all path dependence. This assumption fails (failure mode F) when the environment is path-dependent or partially observed: the same φ(s) can be reached with different hidden modal state, focus, scroll, transaction/lock state, or undo history, so a click that is idempotent from one pre-state is destructive from another. In that case the residual reflects pre-state mismatch rather than the action's contribution. The shrinkage term `(1 − λ·dist_t/k)` does not repair this because distance is measured in the same φ space that omits the confounder, so a path-dependent match can have dist=0 and receive full weight. To make the reward a certified interval rather than a point estimate, we augment φ with a logged hidden-state summary h(s) (a hash of the full UI accessibility tree / open-modal stack / DB transaction id) and emit r_t only when the matched set's counterfactual deltas are consistent (low variance across the k candidates); otherwise we fall back to a conservative lower-confidence reward (the minimum over candidates) rather than the single nearest neighbor.

**How it works.**

```pseudocode
Input: history H of (s_t, a_t, s_{t+1}); task goal g; state-delta encoder phi (typed diff over DB rows / files / UI widget attrs); hidden-state summary h(s) = hash(accessibility tree, open-modal stack, DB transaction id)
Hyperparams: match radius k=5, shrinkage lambda=0.3, skill support threshold tau=3, consistency threshold sigma=0.1

# Phase 1: build the counterfactual transition graph over the agent's OWN history
for each turn t:
    key_t   = (phi(s_t), h(s_t))          # typed state fingerprint + hidden-state summary
    cand_t  = { (s_j, a_j, s_{j+1}) in H : j != t, ||phi(s_j) - phi(s_t)||_1 <= k and h(s_j) == h(s_t) }
    # matched alternative action = the action the agent itself took from the nearest pre-state
    a_t^cf  = a_{j*} where j* = argmin_{j in cand_t} ||phi(s_j) - phi(s_t)||_1
    # counterfactual delta = observed delta of that matched transition, re-anchored on s_t
    Delta_t^cf = phi(s_{j*+1}) - phi(s_t)
    # consistency check over the matched set
    deltas_cf = { phi(s_{j+1}) - phi(s_t) : j in cand_t }
    if var(deltas_cf) <= sigma:
        Delta_t^cf_used = Delta_t^cf
    else:
        Delta_t^cf_used = min_{d in deltas_cf} sim(d, Delta_t^goal)   # conservative lower-confidence reward

# Phase 2: turn reward = difference between observed and counterfactual state deltas
r_t = sim(Delta_t, Delta_t^goal) - sim(Delta_t^cf_used, Delta_t^goal)
    where sim(d, d_goal) = cosine over the typed-diff vector, Delta_t^goal = phi(s_goal) - phi(s_t)
    # shrinkage toward zero when the match is weak (few candidates / large distance)
    r_t <- (1 - lambda * min(1, dist_t / k)) * r_t

# Phase 3: skill extraction from turns whose delta is causally necessary
actionable_t = 1[ r_t > 0 and |Delta_t| > 0 and Delta_t is not reproduced by any a_t^cf ]
skill_snippets = cluster actionable_t by (phi(s_t) region, a_t schema); keep clusters with support >= tau
    # each snippet = (pre-state predicate, action schema, expected delta predicate)

# Phase 4: train policy pi_theta with per-turn rewards
for each rollout: A_t = sum_{t'>=t} gamma^{t'-t} r_{t'};  update pi_theta with policy gradient on A_t
```

**Why this design.** We chose to build the counterfactual from the agent's *own* logged transitions rather than from a learned dynamics model because a learned model introduces exactly the unverified proxy we are trying to eliminate, accepting the cost that matches are only available where the history revisits similar pre-states. We chose a *typed state-delta encoder* (explicit diff over DB rows, file hashes, and UI widget attributes) over raw screenshot or embedding similarity because the reward must be falsifiable against a concrete delta, accepting that environments without structured state logging are out of scope. We chose a *difference-of-similarities* reward over an absolute progress score because the counterfactual baseline absorbs task difficulty and pre-state effects, accepting higher variance when the match radius k is large (hence the shrinkage term). We chose *clustered actionable turns* over per-turn skill authoring because a skill should only be emitted when the delta is causally necessary, accepting that rare-but-correct actions may fall below the support threshold tau. A domain expert would not describe this as a variant of TARL: TARL's reward is an LLM judge score with no environment term, whereas here the reward is literally the residual between two logged state deltas, and the judge is absent by construction.

**Why it measures what we claim.** The quantity `sim(Delta_t, Delta_t^goal) - sim(Delta_t^cf, Delta_t^goal)` measures *turn-level causal credit* because the assumption is that the matched transition (s_{j*}, a_{j*}, s_{j*+1}) is a valid substitute for what would have happened at s_t under the alternative action a_t^cf; this assumption fails when the environment is path-dependent (e.g., a dialog box opened earlier changes what a click does), in which case the residual reflects the pre-state mismatch rather than the action's contribution. The quantity `actionable_t = 1[r_t > 0 and Delta_t not reproduced by any a_t^cf]` measures *skill necessity* because the assumption is that a delta not reproducible by any matched alternative action is caused by the specific action schema; this fails when the history is too sparse to contain a reproducing counterfactual, in which case the flag reflects coverage of the history rather than true necessity. The shrinkage factor `(1 - lambda * dist_t / k)` measures *match reliability* because the assumption is that counterfactual validity degrades smoothly with pre-state distance; this fails when distance is dominated by irrelevant state fields, in which case it under-weights valid matches. The cluster support threshold tau measures *skill generality* because the assumption is that a snippet recurring across >= tau distinct pre-states transfers; this fails when the history over-represents one task, in which case tau reflects task frequency rather than generality. The consistency check `var(deltas_cf) <= sigma` measures *counterfactual validity* because the assumption is that low variance across matched candidates indicates the hidden-state summary h(s) has removed path dependence; this fails when the matched set is small (|cand_t| < 2), in which case the check is vacuous and the conservative fallback is used.

## Contribution

(1) CTR-Delta, a reward and skill-extraction procedure that derives turn-level credit and skill snippets purely from the residual between observed and counterfactually-implied environment state deltas over the agent's own history, eliminating both the LLM judge and authored skill files. (2) A design principle — typed state-delta encoding plus within-history counterfactual matching — that makes the learning signal falsifiable against logged transitions, with a shrinkage rule that trades counterfactual validity against match coverage. (3) An actionable-turn clustering scheme that emits skill snippets as (pre-state predicate, action schema, expected delta predicate) triples only when the delta is causally necessary and recurrent.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | CUA-World + OSExpert-Eval | Structured UI-state logging enables typed deltas |
| Primary metric | Task success rate | Directly reflects goal-directed delta progress |
| Baseline | No-skill agent | Isolates value of extracted skills |
| Baseline | Matched text-only skill | Tests whether delta-grounded skills beat prose skills |
| Baseline | Standard RL (no turn-level reward) | Tests value of counterfactual per-turn credit |
| Ablation-of-ours | CTR-Delta w/o counterfactual (absolute sim only) | Isolates the counterfactual baseline term |
| Ablation-of-ours | CTR-Delta w/o hidden-state summary h(s) (φ only) | Tests the path-independence assumption A |
| Diagnostic | Synthetic environment with known ground-truth per-turn causal effects | Grounds that r_t measures causal credit, not pre-state mismatch |

### Why this setup validates the claim

The central claim is that a reward formed as the residual between an observed state delta and a matched counterfactual delta yields turn-level credit that is causally attributable to the action, and that clustering turns where this residual is positive and non-reproducible yields transferable skills. Each baseline targets one link in that chain. The no-skill baseline establishes the headroom from skills at all. The matched text-only skill tests whether grounding skills in typed deltas (pre-state predicate, action schema, expected delta predicate) beats prose descriptions of the same skill — if our skills only help as much as text skills, the delta grounding is doing no work. Standard RL without turn-level adjudication tests the reward sub-claim: if per-turn counterfactual credit is real, sparse task-level reward should underperform on long-horizon tasks where credit must be assigned across many turns. The ablation removing the counterfactual term is the sharpest test: it keeps the typed-delta encoder and skill extraction but replaces the difference-of-similarities with absolute progress, so any gain attributable to the counterfactual baseline itself is isolated. The ablation removing the hidden-state summary h(s) tests the load-bearing assumption A directly: if path dependence is present, matching on φ alone should degrade the reward and increase the gap to the full method. The synthetic diagnostic with known ground-truth per-turn causal effects provides a controlled check that r_t correlates with the true effect rather than with pre-state mismatch. Task success rate is the right primary metric because the method claims to improve goal-directed behavior end-to-end; a reward-shaping artifact that improves intermediate sim scores without raising success would be a failure, so success is the falsifiable endpoint.

### Expected outcome and causal chain

**vs. No-skill agent** — On a case where a task requires a multi-step UI sequence (open dialog, edit a DB field, confirm), the no-skill agent re-derives the sequence each episode and frequently omits a confirmation step because nothing in its context marks that step as causally necessary. Our method flags the confirmation turn as actionable (its delta is not reproduced by any matched alternative action) and emits a skill snippet whose expected-delta predicate names the post-confirmation state, so the policy retrieves the full sequence. We expect a clear success-rate gap concentrated on tasks whose logs contain such non-reproducible deltas, with near-parity on single-turn tasks where no skill is needed.

**vs. Matched text-only skill** — On a case where a skill's prose says "select the correct row before editing," a text-only skill leaves the agent to infer which row from the current screen, and it sometimes edits an adjacent row whose widget attributes look similar. Our method's skill carries a typed pre-state predicate over the specific row key and widget attributes, so retrieval is gated on the fingerprint matching, and the agent edits the correct row. We expect a gap on tasks with visually similar sibling widgets (tables, file lists) and parity on tasks with unique targets — the delta grounding should pay off exactly where text descriptions are ambiguous.

**vs. Standard RL (no turn-level reward)** — On a case where a long rollout contains one decisive action early and many neutral actions after, standard RL spreads a sparse terminal reward across the whole trajectory, so the decisive action receives near-zero advantage and is not reinforced. Our method assigns it a large positive residual because its delta advances toward the goal while matched alternatives do not, and the neutral turns get near-zero residuals, so the advantage concentrates on the decisive turn. We expect the gap to widen with rollout length and to be small on short tasks — the signature of correct credit assignment rather than a general policy improvement.

**vs. CTR-Delta w/o counterfactual (absolute sim only)** — On a case where the pre-state already has high similarity to the goal (e.g., a nearly complete form), the absolute-progress reward gives a large positive score to any turn that increases similarity, even if the turn is a neutral scroll or a redundant edit that a matched alternative would also have produced. Our method's residual subtracts the counterfactual delta, so such turns receive near-zero reward and the advantage concentrates on the turns that are causally necessary. We expect the full method to outperform the ablation on long-horizon tasks with many neutral turns, and to match it on short tasks where every turn changes the state.

**vs. CTR-Delta w/o hidden-state summary h(s) (φ only)** — On a case where the same φ(s) is reached with different hidden modal state (e.g., a dialog box open vs. closed), the φ-only match selects a transition that is not a valid substitute, so the residual reflects pre-state mismatch. The full method conditions on h(s) and emits a conservative lower-confidence reward when the matched set is inconsistent, so the reward remains a certified interval. We expect the full method to outperform the φ-only ablation on tasks with path-dependent UI state, and to match it on tasks where the hidden state is constant.

**vs. Synthetic diagnostic with known ground-truth per-turn causal effects** — On a synthetic environment where the true per-turn causal effect is known, we compute the correlation between r_t and the true effect. If the residual measures causal credit, the correlation should be high (e.g., Pearson r > 0.7) and should drop when h(s) is removed, confirming that the hidden-state summary is necessary for the estimator to track the true effect rather than pre-state mismatch.

### What would falsify this idea

If the gain over the text-only skill baseline is uniform across tasks rather than concentrated on tasks with visually similar sibling widgets and non-reproducible deltas, the delta-grounding mechanism is not what is driving the improvement. Equally, if the ablation without the counterfactual term matches our full method on long-horizon tasks, the counterfactual baseline is doing no causal work and the reward is just absolute progress in disguise. If the ablation without the hidden-state summary h(s) matches the full method on path-dependent tasks, the load-bearing assumption A is not violated in practice and the consistency check is unnecessary.

## References

1. Process-Supervised Reinforcement Learning for Interactive Multimodal Tool-Use Agents
2. VISUALSKILL: Multimodal Skills for Computer-Use Agents
3. Agent S2: A Compositional Generalist-Specialist Framework for Computer Use Agents
4. OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments
5. Agent S: An Open Agentic Framework that Uses Computers Like a Human
6. AgentStore: Scalable Integration of Heterogeneous Agents As Specialized Generalist Computer Assistant
7. OS-ATLAS: A Foundation Action Model for Generalist GUI Agents
8. OSCAR: Operating System Control via State-Aware Reasoning and Re-Planning
