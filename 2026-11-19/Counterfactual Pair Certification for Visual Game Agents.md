# Counterfactual Pair Certification for Visual Game Agents

## Motivation

Across the game-playing visual-agent tree, success is defined as a predicate over exposed internal state rather than as a property of the perception-to-action mapping: Game2Player's Builder verifies tasks using state-based progress measures and success checks, and LVLM-Playground's four-task diagnostics (perceiving, QA, rule following, end-to-end) report aggregate scores without any per-step verifiability invariant tying a success to the visual evidence the agent actually used. The structural consequence is that an agent can satisfy a state predicate via incidental state progress — a lucky action sequence, a memorized layout, or a cue the Builder never intended to be rule-relevant — and progress-weighted SFT then up-weights those steps as if they reflected the intended visual reasoning. The missing closure is identifiability of the task's latent goal under partial visual observation, which neither branch supplies.

## Key Insight

If two task variants are constructed to share every exposed state variable and differ only in a rule-relevant visual cue, then any trajectory that succeeds on both must have conditioned its action on that cue, so pairwise discrimination is a sufficient condition for perception-conditional success rather than incidental state progress.

## Method

**What it is.** We introduce CPC (Counterfactual Pair Certification), an intervention-based identifiability layer that sits on top of Game2Player's Builder. Inputs: a Builder-adapted web game with exposed internal state and a base task with state-based progress measures; outputs: a set of certified counterfactual task pairs, per-trajectory perception-conditional success certificates, and a re-weighted SFT dataset in which only certified steps receive progress weight.

**Load-bearing assumption (stated explicitly).** CPC's attribution claim rests on the assumption that the two counterfactual variants are observationally identical to the VLM player everywhere except at the designated cue region, so that any action divergence at the cue-relevant step is causally attributable to the cue rather than to unexposed state, rendering noise, or stochastic dynamics. This assumption is not guaranteed by the Builder's `state_signature` equality alone (which is an abstraction over exposed internal state, not over pixels); it is verified per pair by the observation-equivalence check in Phase 1 and the repeated-rollout divergence test in Phase 3, and pairs that fail either check are rejected rather than certified.

**How it works.**

```pseudocode
# Phase 1: Builder synthesizes a counterfactual pair for each base task
for each base_task T in Builder_task_set:
    cue = Builder.select_rule_relevant_cue(T)        # e.g. color/shape of a key object
    T0  = Builder.clone(T)                            # reference variant, cue unchanged
    T1  = Builder.apply_intervention(T, cue)          # SAME state layout, cue flipped
    assert Builder.state_signature(T0) == Builder.state_signature(T1)   # state-identity check
    assert Builder.success_predicate(T0) == Builder.success_predicate(T1) # predicate-identity
    # observation-equivalence check (smallest repair): render both variants from the same
    # seed and verify pixel-level identity outside the cue region; the cue region must be
    # the sole differing support.
    obs0 = Builder.render(T0, seed=0); obs1 = Builder.render(T1, seed=0)
    if not Builder.observation_identical_outside_cue(obs0, obs1, cue_region, tol=0):
        Builder.register_pair(pair, status=REJECT_OBSERVATION_MISMATCH)
        continue
    pair = (T0, T1); Builder.register_pair(pair, status=OK)

# Phase 2: VLM Player attempts both variants; record trajectories
#   multiple stochastic rollouts per variant (N=8) so that noise-driven divergence
#   is rejected rather than certified
for pair (T0, T1) in registered_pairs with status=OK:
    taus0 = [VLMPlayer.rollout(T0, seed=k) for k in 1..N]   # tau = (o_t, a_t, s_t, progress_t)
    taus1 = [VLMPlayer.rollout(T1, seed=k) for k in 1..N]
    cert = certify(taus0, taus1, T0, T1)

# Phase 3: Certification rule (per pair)
def certify(taus0, taus1, T0, T1):
    succ0 = all(Builder.success_check(T0, tau) for tau in taus0)   # state predicate
    succ1 = all(Builder.success_check(T1, tau) for tau in taus1)
    if not (succ0 and succ1):
        return REJECT                       # not both solved -> no certificate
    # repeated-rollout divergence test: action distribution at t_star must shift in the
    # rule-implied direction with statistical significance (binomial test, alpha=0.05)
    t_star = Builder.cue_relevant_step(T0)  # step index where cue is observable
    a0_dist = [tau.action_at(t_star) for tau in taus0]
    a1_dist = [tau.action_at(t_star) for tau in taus1]
    a_expected = Builder.expected_action_under_cue(T1, cue)
    # divergence: fraction of rollouts taking a_expected must be significantly higher in T1
    p1 = mean(a == a_expected for a in a1_dist)
    p0 = mean(a == a_expected for a in a0_dist)
    if not binomial_test(p1 > p0, n=N, alpha=0.05):
        return REJECT                       # divergence not statistically significant
    # counterfactual consistency: action must be the one the flipped cue implies
    if p1 < 0.5:                            # majority of T1 rollouts must match expected
        return REJECT
    return CERTIFIED

# Phase 4: Progress-weighted SFT restricted to certified steps
for trajectory tau in certified_trajectories:
    for step t in tau:
        w_t = 1.0 if (t == t_star and certified) else 0.0   # only cue-attributable steps weighted
        # alternatively: w_t = progress_t * 1[certified] for non-cue steps
        loss += w_t * CE(policy(o_t), a_t)
```

**Why this design.** We chose to have the Builder synthesize paired counterfactual variants rather than to learn a disentanglement head over trajectories, because the state-identity check (`state_signature(T0) == state_signature(T1)`) plus the observation-equivalence check gives a structural guarantee that the two variants share every exposed state variable and are pixel-identical outside the cue region, so any behavioral difference is forced to flow through the visual cue — a learned head cannot manufacture that guarantee from coupled data. We chose a discrete, closed-form certification rule (repeated-rollout divergence at the cue-relevant step plus expected-action match) over a learned verifier because the rule's decision surface is inspectable and its failure mode is explicit; the cost is that it only applies when the Builder can name a single cue-relevant step, which we accept by scoping to tasks with localized rule-relevant cues. We chose to zero out progress weight on non-certified steps rather than to down-weight them smoothly, because the motivation is that incidental state progress is not evidence of visual reasoning; the cost is a smaller effective training set, which we accept as the price of attributability. This is not a variant of Game2Player's progress-weighted SFT: Game2Player weights every step by state-progress delta, whereas CPC weights only steps whose advantage survives a counterfactual intervention, so the two signals are structurally different (state-progress vs. cue-attributable discrimination).

**Why it measures what we claim.** The quantity `p1 > p0` (repeated-rollout divergence at `t_star`) measures **perception-conditional success** because the assumption is that the Builder's `state_signature` equality plus the observation-equivalence check makes the two variants observationally identical except at the cue, so any statistically significant action-distribution shift at `t_star` is causally attributable to the cue; this assumption fails when the game has stochastic dynamics or hidden state not captured by `state_signature` or when rendering noise outside the cue region differs, in which case the divergence reflects noise rather than perception. The quantity `p1 >= 0.5` (majority of T1 rollouts match `a_expected`) measures **rule-following under the flipped cue** because the assumption is that the Builder's `expected_action_under_cue` correctly encodes the rule semantics; this assumption fails when the Builder's rule encoding is itself wrong, in which case the certificate reflects Builder error rather than player error. The weight `w_t = 1.0` on certified steps measures **attributable credit** because the assumption is that certification is a sufficient condition for the step's advantage to be due to visual reasoning; this assumption fails when the agent solves the pair via a non-cue shortcut that happens to align with the expected action, in which case the weight reflects a coincidence rather than genuine attribution.

## Contribution

["CPC, an intervention-based identifiability layer that couples Game2Player's Builder with paired counterfactual task variants differing only in a rule-relevant visual cue, producing a per-trajectory perception-conditional success certificate rather than a state-predicate success check.", 'A certification rule with an explicit state-identity invariant (state_signature equality) and a closed-form discrimination test, which yields a training signal that is restricted to steps whose advantage survives the counterfactual intervention.', "A re-weighted SFT procedure that zeroes progress weight on non-certified steps, converting Game2Player's state-progress weighting into cue-attributable weighting."]

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Game2Player verified games + PlayerBench holdout | Provides exposed internal state and state-based success checks. |
| Primary metric | Held-out success on cue-flipped variants | Directly probes perception-conditional rule use. |
| Baseline | Game2Player progress-weighted SFT | Tests whether state-progress weighting alone suffices. |
| Baseline | Uniform SFT on all collected trajectories | Tests whether attribution filtering adds anything over plain imitation. |
| Baseline | REGENT retrieval-augmented in-context player | Tests whether adaptation without certification closes the gap. |
| Ablation-of-ours | CPC without the expected-action check | Isolates contribution of rule-consistency vs. mere action divergence. |
| Diagnostic | Observation-equivalence rate | Measures fraction of pairs where state_signature equality implies observation identity. |
| Diagnostic | Certificate inflation under injected stochasticity | Measures how many spurious certificates arise when dynamics are stochastic. |

### Why this setup validates the claim

The central claim is that weighting only counterfactually certified steps yields agents whose success is attributable to the visual cue rather than to incidental state progress. The held-out cue-flipped success metric is the right detector because it can only be passed by an agent that actually reads the cue; an agent that exploits state-progress shortcuts should still solve the original variants but collapse on the flipped ones. Game2Player progress-weighted SFT tests the sub-claim that state-progress weighting is insufficient — if it matched CPC on flipped variants, our attribution story would be redundant. Uniform SFT tests whether certification filtering, not merely more data, drives the effect. REGENT tests whether in-context retrieval adaptation substitutes for certification. The no-expected-action ablation separates "action differs" from "action is the rule-correct one," so a gain that survives without the rule check would reveal our discrimination test is doing the work rather than rule attribution. The observation-equivalence diagnostic directly bounds the load-bearing assumption: if the fraction of pairs passing the pixel-identity check is low, the attribution claim is restricted to that subset. The stochasticity-injection diagnostic measures certificate inflation under the failure mode where noise-driven divergence is mistaken for perception. Together these form a falsifiable contrast: CPC must beat each baseline specifically on cue-flipped subsets while staying at parity on cue-irrelevant subsets, and the diagnostics must show that the certified subset is non-trivial and that inflation is controlled.

### Expected outcome and causal chain

**vs. Game2Player progress-weighted SFT** — On a case where a key object's color determines which door is correct and the layout is otherwise identical, the baseline produces correct play on the original variant but wrong-door choices on the flipped variant because its progress weights reward any state-advancing step, including steps taken for the wrong reason. Our method instead trains only on steps whose action distribution shifts with the cue and matches the rule-implied action, because certification filters out progress that is not cue-attributable. We expect a noticeable gap on cue-flipped variants and near-parity on cue-irrelevant control variants.

**vs. Uniform SFT on all trajectories** — On a case where most trajectories succeed via a shortcut that happens to coincide with the rule, the baseline produces high original-variant accuracy but near-chance flipped accuracy because it imitates shortcut actions with equal weight. Our method instead zeroes weight on non-certified steps, so the shortcut is never reinforced as visual reasoning. We expect uniform SFT to look competitive on originals yet degrade sharply on flipped pairs, while CPC holds flipped accuracy close to its original accuracy.

**vs. REGENT retrieval-augmented in-context player** — On a case where the retrieved neighbor comes from a game with a similar layout but a different cue-to-action mapping, the baseline produces the neighbor's action because retrieval matches on state appearance, not on rule semantics. Our method instead conditions the trained policy on certified cue-action pairs, so the flipped cue maps to the expected action regardless of nearest-neighbor similarity. We expect REGENT to be strong on same-rule neighbors but to show a clear deficit on cross-rule flipped pairs where CPC remains stable.

**vs. CPC without expected-action check** — On a case where the agent diverges at the cue step for a spurious reason that coincidentally differs across variants, the ablation produces a certificate because it only requires action divergence. Our method instead rejects such pairs because the flipped action does not match the rule-implied action. We expect the ablation to gain on flipped variants relative to baselines but to show inflated certificates and weaker rule-consistency, isolating the expected-action check as the source of CPC's advantage.

**Diagnostic: observation-equivalence rate** — We expect a non-trivial fraction of Game2Player tasks to fail the pixel-identity check outside the cue region due to rendering variation (anti-aliasing, animation phase, HUD timers). The reported rate quantifies the subset of tasks to which CPC's attribution claim applies, and we expect this subset to be large enough (e.g., >50% of tasks with localized cues) to support the main experiments.

**Diagnostic: certificate inflation under injected stochasticity** — We inject controlled stochasticity into game dynamics (e.g., random action delays or state perturbations) and measure the fraction of pairs that receive a certificate despite the agent's action being noise-driven. We expect the repeated-rollout divergence test to keep inflation low (e.g., <10% of certificates) compared to a single-rollout variant, confirming that the statistical test is necessary.

### What would falsify this idea

If CPC's gain over Game2Player progress-weighted SFT were uniform across cue-flipped and cue-irrelevant subsets rather than concentrated on the flipped subset, the attribution claim would be wrong. Likewise, if the no-expected-action ablation matched full CPC everywhere, the rule-consistency component would be doing no work and the certification story would collapse to mere action-divergence filtering. Additionally, if the observation-equivalence rate were near zero or certificate inflation under stochasticity were high, the load-bearing assumption would be empirically violated and the method would need to fall back to a paired-difference-in-differences estimator over many pairs.

## References

1. Game2Player: Scaling Gameplay Experience for Visual Agents through Environment-Player Co-Development
