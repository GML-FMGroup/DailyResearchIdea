# Adversarial Counterfactual Negative Synthesis for Self-Supervised Agent Preference Training

## Motivation

Preference-pair construction for agent self-training harvests negatives exclusively from the target agent's own rollouts, so the negative distribution equals the agent's current failure distribution and is never closed under substitution of failure modes: DeLoop pairs looping actions against the first later non-looping action in the same rollout, and Co-Evolving Agents trains a failure agent only on trajectories already produced by the target and its co-evolver. This creates a fixed point — the agent cannot learn to avoid failures it never produces, and it never produces failures it cannot yet exhibit — so informative out-of-support negatives are structurally absent from every training set built this way. The root cause is that the negative generator and the target policy share support by construction, so no amount of reweighting or iterative co-evolution can escape the target's native failure manifold.

## Key Insight

Constraining the synthesizer to maximize divergence from the target's native failure distribution subject to a reachability constraint enforced by an independent executor yields a negative set that is provably not a subset of the target's failure distribution, breaking the fixed point without requiring the target to first produce those failures.

## Method

**What it is.** ACNS (Adversarial Counterfactual Negative Synthesis) is a training-data construction procedure that outputs, for each task, a set of negative trajectories guaranteed to lie outside the target agent's native rollout support while remaining realizable by some policy in the environment. Inputs: a task $x$, the frozen target policy $\pi_\theta$, an environment simulator; outputs: preference pairs $(a^+ \succ a^-)$ where $a^-$ is an ACNS-synthesized failure.

**How it works.**

```pseudocode
Input: task x, frozen target pi_theta, env simulator E, synthesizer S_phi (policy over actions),
       executor ensemble M = {mu_1, ..., mu_M} (M=3 independent fixed checkpoints),
       divergence weight lambda, reachability budget K, confidence threshold alpha=0.5

# Phase 1: characterize target's native failure support
Rollout N=8 trajectories of pi_theta on x under E -> D_native
F_native <- {failure steps in D_native}           # e.g., looping actions, as in DeLoop

# Phase 2: synthesize out-of-support failures
for k in 1..K:
    tau_k <- rollout of S_phi on x under E       # synthesizer's own trajectory
    if tau_k is a failure AND tau_k not in F_native:
        keep tau_k

# Phase 3: coverage-calibrated reachability filter via independent executor ensemble
# Each mu_j is a fixed, separately-trained policy (NOT pi_theta); it must reproduce tau_k's
# failure step from the same prefix state s_t. We accept if at least ceil(alpha * M) = 2
# executors reproduce a_t^- under environment replay (stochastic sampling with confidence threshold).
for each kept tau_k with failure step (s_t, a_t^-):
    votes <- 0
    for j in 1..M:
        if mu_j(s_t) == a_t^- under environment replay:
            votes += 1
    if votes >= ceil(alpha * M):
        keep tau_k                          # reachable under union of executor supports
    else:
        discard tau_k                       # unreachable under all executors

# Phase 3b: calibrate executor false-negative rate on held-out known-reachable failures
# Held-out set H of 512 failure steps known reachable (from a separate policy or human labels).
# For each (s_t, a_t^-) in H, compute votes as above; false-negative rate = fraction with votes < ceil(alpha*M).
# Report this rate as a bound; if > 0.2, increase M or lower alpha.

# Phase 4: build preference pairs
for each reachable tau_k:
    a^- <- failure action a_t^- at state s_t
    a^+ <- argmax_{a != a^-} pi_theta(a | s_t)  # target's own preferred alternative
    emit pair (a^+, a^-)

# Phase 5: train target with DPO on emitted pairs, then update S_phi to
# maximize divergence from F_native subject to reachability (adversarial loop).
# Divergence reward: r = lambda * D_KL(hist(S_phi) || hist(pi_theta)) - reachability_penalty
# where reachability_penalty = 0 if votes >= ceil(alpha*M) else +inf.
# Additionally, penalize failures that are only reachable via mu-specific actions:
# compute a mu-specificity score = 1 - (votes / M); add penalty beta * mu_specificity to r,
# with beta=0.1, to decouple the divergence reward from mu's support.
```

The synthesizer $S_\phi$ is trained with a reward $r = \lambda \cdot D_{\text{KL}}(\text{hist}(S_\phi) \,\|\, \text{hist}(\pi_\theta)) - \text{reachability\_penalty} - \beta \cdot \text{mu\_specificity}$, where the reachability penalty is 0 if at least $\lceil \alpha M \rceil$ independent executors reproduce the failure and $+\infty$ otherwise, and mu_specificity $= 1 - \text{votes}/M$ penalizes failures that only a single executor emits. The executor ensemble $\mathcal{M} = \{\mu_1, \mu_2, \mu_3\}$ consists of three frozen checkpoints of *different* models (e.g., a 1.5B, a 3B, and a 7B model trained on the same environment), so their action distributions are structurally decoupled from $\pi_\theta$ and from each other. We calibrate the ensemble's false-negative rate on a held-out set of 512 known-reachable failure steps (collected from a separate policy or human labels) and report it as a bound; if the rate exceeds 0.2, we increase $M$ to 5 or lower $\alpha$ to 0.4.

**Why this design.** We chose an independent executor ensemble $\mathcal{M}$ over a learned discriminator because a discriminator trained on $\pi_\theta$'s rollouts inherits $\pi_\theta$'s support and collapses to mode-seeking, while $\mathcal{M}$ provides a reachability certificate that does not depend on $\pi_\theta$ — accepting the cost that $\mathcal{M}$ may reject some genuinely reachable failures (false negatives), which we mitigate by using $\mathcal{M}$ only as a filter, not as a generator, and by calibrating its false-negative rate on a held-out set. We chose divergence maximization over uniform sampling from $S_\phi$ because uniform sampling produces failures that are trivially easy (low divergence) and thus uninformative — accepting the cost that $\lambda$ must be tuned to avoid synthesizing failures that are reachable by $\mathcal{M}$ but semantically nonsensical. We chose DPO over PPO because the preference pairs are constructed offline and DPO's closed-form objective does not require a reward model, consistent with DeLoop's design; accepting the cost that DPO cannot weight pairs by their divergence magnitude. The closest prior method is Co-Evolving Agents, which trains a failure agent on the target's own failed trajectories — a domain expert would not describe ACNS as a variant of it because ACNS's failure agent is trained against a *frozen independent executor ensemble* and a *divergence objective from the target's native failures*, not on the target's failure log, so its output distribution is not a subset of the target's.

**Why it measures what we claim.** The divergence term $D_{\text{KL}}(\text{hist}(S_\phi) \| \text{hist}(\pi_\theta))$ measures *out-of-support-ness* of synthesized failures because it is computed between the synthesizer's trajectory histogram and the target's native rollout histogram, under the assumption that histogram divergence is a sufficient statistic for support mismatch; this assumption fails when the target's rollout distribution is multi-modal and the synthesizer's histogram diverges only in a low-probability mode, in which case the divergence reflects mode coverage rather than genuine support escape. The reachability filter (executor ensemble $\mathcal{M}$ reproducing the failure action) measures *counterfactual realizability* because $\mathcal{M}$ is an independent ensemble whose action at state $s_t$ is a witness that some policy in the environment can produce $a_t^-$; this assumption fails when $\mathcal{M}$ is systematically weaker than any realizable policy (e.g., $\mathcal{M}$ cannot represent the needed action), in which case the filter reflects $\mathcal{M}$'s capacity rather than true reachability. The preference pair $(a^+, a^-)$ measures *informative negative* because $a^+$ is the target's own preferred action at the same state, so the pair isolates the failure action as the sole difference; this assumption fails when the target's preferred action $a^+$ is itself a failure mode (the target is confidently wrong), in which case the pair teaches the target to prefer one failure over another.

## Contribution

(1) ACNS, a preference-pair construction framework that combines a divergence-maximizing synthesizer with an independent-executor reachability filter to produce negatives provably outside the target's native failure distribution. (2) The finding that reachability filtering via a frozen independent policy is sufficient to prevent the mode-seeking collapse that plagues naive adversarial negative generation, without requiring a learned discriminator. (3) A training procedure that closes the negative distribution under substitution of failure modes, enabling self-supervised agent improvement on failures the agent cannot currently produce.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | CodeGym under history compression | Matches DeLoop; exposes looping failures |
| Primary metric | Task success rate | Directly measures whether failures were corrected |
| Baseline | DeLoop loop/non-loop DPO | Tests whether native-support negatives suffice |
| Baseline | Co-Evolving Agents failure agent | Tests whether target-derived failures suffice |
| Baseline | Untrained Qwen3-8B agent | Floor reference for loop frequency |
| Ablation-of-ours | ACNS without reachability filter | Tests the executor-certificate sub-claim |
| Ablation-of-ours | ACNS with single executor (M=1) vs. ensemble (M=3) | Tests whether coverage calibration matters |
| Ablation-of-ours | ACNS with mu-specificity penalty (beta=0.1) vs. without | Tests whether decoupling from mu's support prevents collapse |
| Diagnostic | Executor false-negative rate on 512 held-out known-reachable failures | Calibrates the reachability certificate |

### Why this setup validates the claim

The central claim is that negatives lying outside the target's native rollout support, yet certified reachable by an independent executor ensemble, teach the target more than negatives drawn from its own failure log. CodeGym under compression is the right testbed because it reliably produces the looping failure mode DeLoop already isolates, giving us a known failure class to escape. DeLoop is the decisive baseline: it constructs pairs from the target's own loops, so if ACNS's out-of-support negatives add nothing, ACNS and DeLoop should be statistically indistinguishable. Co-Evolving Agents tests the independence claim — its failure agent is trained on the target's log, so if independence from the target's support is what matters, ACNS should pull ahead precisely on tasks where the target loops. The no-filter ablation isolates the reachability sub-claim: if the executor certificate is doing real work, removing it should degrade success on reachable-but-out-of-support cases while leaving trivially reachable failures untouched. The single-executor vs. ensemble ablation tests the coverage-calibration repair: if the ensemble's union of supports matters, M=3 should outperform M=1 on tasks where the single executor has high false-negative rate. The mu-specificity penalty ablation tests whether the adversarial loop collapses back onto mu's manifold: without the penalty, we expect the synthesizer to converge to mu-specific failures, reducing success-rate gain. Task success rate is the right primary metric because the whole causal story terminates in the agent no longer committing the failure; loop frequency is a mechanism check, not the outcome.

### Expected outcome and causal chain

**vs. DeLoop loop/non-loop DPO** — On a task where the target enters a compression-induced action loop, DeLoop builds its negative from that same loop, so the pair's negative is already inside the target's support and the gradient mostly reshapes probability mass among actions the target already considers. Our method instead synthesizes a failure the target never emits, certified by the independent executor ensemble, so the pair carries genuinely new information. We expect a noticeable success-rate gap concentrated on tasks where the untrained agent loops, with near-parity on tasks that never trigger loops.

**vs. Co-Evolving Agents failure agent** — On a case where the target's failure log is dominated by one shallow failure mode, the co-evolving failure agent learns to reproduce that mode, so its negatives stay a subset of the target's failures and the target is never pushed toward unseen failure regions. Our synthesizer is rewarded for divergence from the target's native failure histogram under the reachability constraint, so it surfaces failures the target's log cannot contain. We expect ACNS to win on the subset of tasks with heterogeneous failure modes, with a smaller gap where the failure space is unimodal.

**vs. Untrained Qwen3-8B agent** — On a long-horizon task, the untrained agent loops because compression erases the signal that a prior action was already attempted, so it repeats indefinitely and burns the step budget. DeLoop-style training reduces loops but only for patterns present in its own rollouts; ACNS additionally supplies out-of-support failure pairs at the exact states where loops begin. We expect both trained variants to beat the untrained floor on loop frequency, with ACNS showing a further step-count reduction on the hardest tasks.

**vs. ACNS without reachability filter** — Without the filter, the synthesizer is free to produce failures that no policy in the environment can realize, so the preference pairs may include unreachable negatives that teach the target to avoid actions it could never take, wasting capacity. With the coverage-calibrated ensemble filter, we expect higher success on reachable-but-out-of-support cases and no degradation on trivially reachable failures.

**vs. ACNS with single executor (M=1)** — A single executor has a restricted support and a higher false-negative rate, so it rejects more genuinely reachable failures and the adversarial loop may collapse onto that executor's manifold. The ensemble (M=3) with confidence threshold alpha=0.5 certifies membership in a union of supports, reducing false negatives and preventing collapse. We expect M=3 to outperform M=1 on tasks where the single executor's false-negative rate exceeds 0.2.

**vs. ACNS without mu-specificity penalty** — Without the penalty, the divergence reward can be maximized by failures that only one executor emits, reintroducing the shared-support fixed point with mu instead of pi_theta. The penalty beta=0.1 discourages such mu-specific failures. We expect the penalty to improve success-rate gain on tasks with heterogeneous failure modes by keeping the synthesizer away from mu's idiosyncratic actions.

### What would falsify this idea

If ACNS's gain over DeLoop is uniform across all task subsets rather than concentrated on tasks where the target natively loops, the out-of-support-negative story is wrong and the improvement is coming from generic preference-pair regularization. Likewise, if the no-filter ablation matches full ACNS, the independent-executor reachability certificate is not doing the causal work we claim; and if the ensemble (M=3) does not outperform the single executor (M=1) on tasks with high executor false-negative rate, the coverage-calibration repair is not effective.

## References

1. DeLoop: Repeated Actions as Self-Supervision for Long-Horizon Agents
2. Context as a Tool: Context Management for Long-Horizon SWE-Agents
3. Encouraging Good Processes Without the Need for Good Answers: Reinforcement Learning for LLM Agent Planning
4. AgenTracer: Who Is Inducing Failure in the LLM Agentic Systems?
5. Scaling Long-Horizon LLM Agent via Context-Folding
6. Co-Evolving Agents: Learning from Failures as Hard Negatives
7. Generalizable End-to-End Tool-Use RL with Synthetic CodeGym
8. AgentGen: Enhancing Planning Abilities for Large Language Model based Agent via Environment and Task Generation
