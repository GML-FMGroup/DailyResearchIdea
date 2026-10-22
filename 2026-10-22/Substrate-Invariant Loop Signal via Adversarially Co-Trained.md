# Substrate-Invariant Loop Signal via Adversarially Co-Trained History Compression

## Motivation

Existing long-horizon agent methods — DeLoop's repeated-tool-call-and-identical-observation detector, Context-Folding's fold-when-summary-complete heuristic, and CAT's separately trained SWE-Compressor — all inherit the same structural assumption: the environment substrate (CodeGym's tool surface, fixed step indexing, observation ordering) and the history-processing architecture are held fixed, and the loop-freeness signal is computed on that fixed surface by a compression stage trained independently of the action policy. This is why the signal does not transport across substrates and why compression never co-adapts with action selection: the loop detector is a function of surface form (a repeated call/observation pair) rather than of task progress, so renaming a tool or reordering observations silently breaks it, and the separable compressor cannot learn which distinctions the action policy actually needs to avoid looping.

## Key Insight

If the loop signal is defined as the action's effect on a compressed sufficient statistic of task progress that is provably invariant to a substrate-perturbation family, then two actions that look different on the surface but leave progress unchanged are identified as loops, and the compression policy and action policy must share that statistic because neither can be optimized without the other.

## Method

**What it is.** SILC (Substrate-Invariant Loop signal via Co-training) jointly trains two heads over a shared encoder: a *progress statistic* z_t = C(h_t) ∈ R^d (a compression of the agent's full history h_t) and an *action policy* π(a | h_t, z_t). The loop signal for a candidate action a is the predicted change in z under the counterfactual continuation, and the pair (C, π) is optimized so that z is invariant to a substrate-perturbation family T while remaining maximally predictive of task progress.

**Load-bearing assumption (stated explicitly).** The perturbation family T contains only semantics-preserving transformations — renaming tools, reordering observations, rescaling step indices — so that enforcing z(T(τ)) = z(τ) forces the progress statistic to depend on task semantics alone rather than discarding genuinely progress-relevant information. This assumption is load-bearing because z is simultaneously the loop signal and the DPO conditioning variable; any semantic information the adversary strips is stripped from the loop detector itself. To keep the assumption true rather than merely asserted, the adversarial inner loop is constrained (step 3 below) so that it cannot increase L_progress, and a sufficiency term is added to the objective (step 6) so the encoder cannot satisfy L_inv by collapsing progress-relevant dimensions.

**How it works.**

```pseudocode
Input: rollout buffer D, perturbation family T = {rename tools, reorder observations, rescale step indices}
Shared encoder E_theta (2-layer MLP, hidden=256, GeLU); progress head C_psi: E_theta(h) -> z in R^d (d=64, linear); policy pi_phi(a | E_theta(h), z) (Qwen3-8B with LoRA rank 16, alpha 32, lr 1e-4)
Reward r_t from environment (task success / verifiable outcome), not from a judge
Calibration set: 512 non-terminal transitions from a held-out set of tasks, used to set epsilon and to gate T

for each training round (batch size 32 trajectories, 200 rounds):
    # 1. Rollout with current policy, storing (h_t, a_t, o_{t+1}, r_t)
    D <- Rollout(pi_phi, C_psi, tasks)

    # 2. Progress-consistency target: z must predict cumulative reward-to-go
    #    via a linear probe g: z -> V_hat (linear, d=64 -> 1), trained with MSE to Monte-Carlo return
    L_progress = E_{(h_t, G_t)~D} [ (g(C_psi(E_theta(h_t))) - G_t)^2 ]

    # 3. Substrate-invariance: constrained adversarial perturbation of surface form
    #    T maps a trajectory to an equivalent trajectory with renamed tools,
    #    reordered observations, rescaled step indices, preserving z
    #    Adversary: a learned perturbation policy q_omega over the discrete choices in T
    #    (tool-rename map, observation permutation, step-index scale), trained with REINFORCE
    #    to maximize L_inv subject to the constraint that L_progress does not increase:
    #      max_omega E_{T~q_omega}[ L_inv(T) ]  s.t.  L_progress(T) <= L_progress(identity) + 0.01
    #    Implemented as a Lagrangian: L_adv = L_inv(T) - beta * max(0, L_progress(T) - L_progress(identity) - 0.01),
    #    with beta=1.0 fixed. q_omega is a 3-layer MLP (hidden=128) over a one-hot encoding of the
    #    current surface form; REINFORCE with baseline = mean L_inv over the batch, 4 inner steps per round.
    L_inv = E_{tau~D, T~q_omega} [ || C_psi(E_theta(T(tau))) - C_psi(E_theta(tau)) ||^2 ]

    # 4. Loop signal from progress statistic:
    #    For each stored action a_t, compute counterfactual z-change
    #    delta_t = || C_psi(E_theta(h_t + a_t + o_{t+1})) - C_psi(E_theta(h_t)) ||
    #    A loop is an action with delta_t below threshold epsilon (set to 5th percentile of
    #    delta over non-terminal steps in the 512-transition calibration set)

    # 5. Preference pairs from the loop signal (as in DeLoop, but on delta not on surface repetition)
    #    negative = looping action, positive = first subsequent action with delta > epsilon
    #    DPO loss on pi_phi with these pairs, with z as additional conditioning (beta_DPO=0.1)

    # 6. Joint update: minimize L_progress + lambda * L_inv + DPO loss + mu * L_suff
    #    L_suff = E_{(h_t, G_t)~D, T~q_omega} [ (g(C_psi(E_theta(T(h_t)))) - G_t)^2 ]
    #    (sufficiency under the perturbed view: z must still predict G_t after T is applied)
    #    lambda annealed from 0.1 to 1.0 over training; mu=0.5 fixed
    theta, psi, phi <- Adam update on combined loss (lr=3e-4 for E_theta, C_psi, g; 1e-4 for pi_phi)
```

**Why this design.** We chose a *shared encoder with two heads* over DeLoop's separable loop detector plus independent compressor because the structural reason the signal fails to transport is precisely that the compressor never sees the action policy's gradient; sharing E_theta forces the progress statistic to retain exactly the distinctions the policy needs to avoid loops, accepting the cost that the encoder cannot specialize to compression alone and may under-compress. We chose *adversarial* perturbation over a fixed augmentation set because a fixed set (e.g., always rename tools) is a finite list that the encoder can memorize, so invariance would not generalize to a held-out substrate; adversarial selection within the family forces the encoder to find the invariant subspace, accepting the cost of an inner optimization loop per round (4 REINFORCE steps over a 3-layer MLP, ~0.3 s per round on one A100). We chose a *counterfactual z-change* as the loop signal over DeLoop's repeated-tool-call rule because the latter is a function of surface form and breaks under renaming, whereas z-change is defined on the compressed progress statistic; this accepts the cost that z-change requires a forward pass through C for each candidate action, roughly doubling rollout compute. A domain expert would not describe this as DeLoop-plus-augmentation because the loop label is not computed from the trajectory surface at all — it is computed from the geometry of the learned progress manifold, and the DPO pairs are generated from that geometry, so the method's core operation (counterfactual z-change under an invariance constraint) is absent from DeLoop, Context-Folding, and CAT.

**Why it measures what we claim.** The counterfactual z-change `delta_t = ||C(h_t + a_t + o_{t+1}) - C(h_t)||` measures *loop-ness* (action fails to advance task progress) because the assumption that z is a sufficient statistic for return-to-go makes zero z-change equivalent to zero expected progress; this assumption fails when the task has progress that is real but not return-relevant within the horizon (e.g., a correct intermediate refactor that only pays off later), in which case delta_t reflects local return-irrelevance rather than a true loop. The adversarial invariance loss `L_inv` measures *substrate-invariance* because the assumption that the perturbation family T spans all surface-form variation that does not change task semantics makes z(T(tau)) = z(tau) equivalent to z being a function of semantics only; this assumption fails when a perturbation in T accidentally changes semantics (e.g., reordering observations that are order-dependent), in which case the invariance loss pushes the encoder to discard a genuinely progress-relevant distinction — which is why the adversary is constrained so that L_progress cannot increase, and why L_suff is added to the objective. The linear probe `g` measures *progress-predictiveness* because the assumption that return-to-go is linearly decodable from a d-dimensional statistic makes the MSE a valid proxy for sufficiency; this assumption fails when the progress manifold is curved enough that a linear probe underfits, in which case L_progress reflects probe capacity rather than statistic quality. The threshold `epsilon` measures *the boundary between loop and non-loop* because the assumption that the 5th percentile of delta on calibration non-terminal steps separates stalled from advancing actions makes epsilon a calibrated cut; this assumption fails on tasks with unusually many near-zero-progress-but-legitimate steps, in which case epsilon is too high and flags legitimate slow progress as loops.

## Contribution

(1) SILC, a co-training framework in which a history-compression policy and an action policy share an encoder and are jointly optimized under an adversarial substrate-invariance constraint, yielding a loop signal defined on a compressed progress statistic rather than on surface repetition. (2) The finding that loop labels derived from counterfactual z-change under an invariance constraint produce preference pairs that transfer across renamed/reordered substrate variants where surface-repetition labels (DeLoop) degrade, establishing co-adaptation of compression and action selection as the mechanism. (3) A calibration procedure for the loop threshold epsilon based on the delta distribution over non-terminal steps, and an analysis of when the sufficiency assumption for z breaks (return-irrelevant progress).

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | CodeGym under history compression | Matches DeLoop; exposes loops under compression |
| Dataset (held-out substrate) | CodeGym with renamed tools, reordered observations, rescaled step indices | Directly probes transport across surface-form variation |
| Primary metric | Task success rate at fixed step budget (budget = 50 steps) | Success is the outcome loops actually destroy |
| Secondary metric | Loop frequency (fraction of steps with delta_t < epsilon) | Complements success; guards against flagging slow progress |
| Secondary metric | Probe R² of z for return-to-go vs. probe R² of z for surface-form classifier | Measures sufficiency vs. invariance directly |
| Baseline | DeLoop (repeated-action pairs) | Tests whether surface-loop signal transports |
| Baseline | Untrained Qwen3-8B agent | Establishes loop-induced ceiling without training |
| Baseline | Context-Folding / CAT-style compression | Tests substrate-robustness of competing management |
| Ablation-of-ours | SILC with fixed (non-adversarial) perturbation set | Isolates adversarial invariance from augmentation |
| Ablation-of-ours | SILC with stop-gradient from DPO loss into C (and from L_progress into π) | Tests whether shared-gradient co-adaptation is load-bearing |
| Ablation-of-ours | SILC with unconstrained adversary (no L_progress constraint, no L_suff) | Tests the load-bearing assumption that T is semantics-preserving |

### Why this setup validates the claim

The central claim is that a loop signal defined on a substrate-invariant progress manifold transports across surface-form variation where a surface-defined signal (DeLoop's repeated-tool-call rule) does not. CodeGym under compression is the right testbed because it is exactly the regime DeLoop was built for, so a win there is not a change of subject; but we evaluate on a *held-out substrate* — renamed tools, reordered observations, rescaled step indices — so the comparison directly probes transport rather than in-distribution fit. DeLoop tests the sub-claim that surface-loop labels fail to generalize; the untrained agent tests the sub-claim that loops, not raw capability, are the binding constraint; the CAT-style compressor tests the sub-claim that our gain is not merely from compression. The fixed-perturbation ablation is the load-bearing one: if our gain survives only when perturbation is adversarial, the mechanism is invariance-finding, not augmentation-memorization. The stop-gradient ablation tests the co-adaptation claim directly: if removing the shared-gradient path between C and π does not degrade loop-label quality on the held-out substrate, the novelty reduces to DeLoop plus an invariance regularizer. The unconstrained-adversary ablation tests the load-bearing assumption: if the unconstrained version matches the constrained one, the semantics-preserving constraint is doing no work. The probe-R² diagnostic converts the asserted 'invariant subspace' claim into a measured quantity: we report whether the adversarial objective drives the surface-form probe R² toward chance while preserving the return-to-go probe R². Task success at a fixed step budget is the correct primary metric because a loop signal is only valuable insofar as it converts wasted steps into completed tasks; loop frequency alone would reward a detector that flags legitimate slow progress.

### Expected outcome and causal chain

**vs. DeLoop (repeated-action pairs)** — On a case where the agent renames a tool between two functionally identical calls (e.g., `read_file` then `open_file`), DeLoop produces no loop label because its rule keys on surface repetition, so the DPO pairs omit the loop and the policy keeps wasting steps; our method computes counterfactual z-change, which is near-zero for both calls because neither advances the progress statistic, so the pair is generated and the policy is pushed off the loop. We expect a noticeable success gap on the renamed-tool subset and near-parity on the canonical-naming subset, with the gap widening as step budget tightens.

**vs. Untrained Qwen3-8B agent** — On a case where the compressed history drops the observation that a prior attempt failed, the untrained agent produces a re-attempt of the same failed action because nothing in its context marks the attempt as exhausted; our method instead conditions on z, which encodes cumulative-return-relevant history and is trained to be invariant to the compression-induced surface loss, so the re-attempt is labeled a loop and suppressed. We expect the trained agent to show a marked reduction in steps-per-task on tasks requiring more than one compression event, with parity on short tasks where no compression fires.

**vs. Context-Folding / CAT-style compression** — On a case where the compressor folds a sub-trajectory whose surface form was renamed mid-episode, the CAT-style baseline produces a fold decision keyed on surface tokens that no longer match, so it either over-folds (losing progress) or under-folds (context bloat) because its salience heuristic is substrate-dependent; our method instead makes the fold-relevant statistic invariant by construction, so the same z is recovered under renaming and the loop signal remains calibrated. We expect a modest but consistent success gain concentrated on long episodes with multiple substrate perturbations, and parity on single-perturbation episodes.

**vs. SILC with fixed perturbation set** — On a case where the held-out substrate uses a renaming convention absent from the fixed augmentation list, the fixed-set ablation produces invariance that does not transfer because the encoder memorized the finite list, so loop labels degrade to DeLoop-like surface matching; our adversarial version instead searches within the family each round, forcing the encoder toward the invariant subspace. We expect the ablation to match us on seen perturbations and fall to near-DeLoop levels on the held-out convention — the diagnostic signature of memorization versus invariance.

**vs. SILC with stop-gradient (no co-adaptation)** — On a case where the policy needs a distinction that the compressor would not retain on its own (e.g., a subtle progress cue that only matters for avoiding a specific loop), the stop-gradient variant's C never receives the DPO gradient, so it drops that distinction and the loop label is miscalibrated; our shared-gradient version retains it because the policy's gradient flows into C. We expect the stop-gradient variant to show lower loop-label precision (fraction of flagged loops that are true loops, judged by held-out ground-truth progress) on the held-out substrate, with the gap concentrated on tasks where the loop-avoidance distinction is not return-relevant in isolation.

**vs. SILC with unconstrained adversary** — On a case where the adversary can find a semantics-breaking perturbation (e.g., reordering order-dependent observations), the unconstrained version maximizes L_inv by discarding progress-relevant dimensions, so z collapses and the loop signal degrades; our constrained version (L_progress constraint + L_suff) prevents this. We expect the unconstrained variant to show lower return-to-go probe R² and higher loop frequency (false positives) on tasks with order-dependent observations, with parity on tasks where T is naturally semantics-preserving.

### What would falsify this idea

If our gain over DeLoop is uniform across the renamed-tool and canonical-naming subsets rather than concentrated on the renamed subset, then the loop signal is not transporting via invariance and the central claim is wrong. Likewise, if the fixed-perturbation ablation matches the adversarial version on the held-out convention, adversarial selection is doing no work and the mechanism story collapses. Additionally, if the stop-gradient ablation matches the shared-gradient version on loop-label precision, the co-adaptation claim is false; and if the unconstrained adversary matches the constrained version on return-to-go probe R², the load-bearing assumption that T is semantics-preserving is not binding.

## References

1. DeLoop: Repeated Actions as Self-Supervision for Long-Horizon Agents
2. Context as a Tool: Context Management for Long-Horizon SWE-Agents
3. Encouraging Good Processes Without the Need for Good Answers: Reinforcement Learning for LLM Agent Planning
4. AgenTracer: Who Is Inducing Failure in the LLM Agentic Systems?
5. Scaling Long-Horizon LLM Agent via Context-Folding
6. Co-Evolving Agents: Learning from Failures as Hard Negatives
7. Generalizable End-to-End Tool-Use RL with Synthetic CodeGym
8. AgentGen: Enhancing Planning Abilities for Large Language Model based Agent via Environment and Task Generation
