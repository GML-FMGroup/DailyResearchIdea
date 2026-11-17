# Failure-Substitution Closure for Universal Action Interfaces in Embodied Foundation Models

## Motivation

Agentic robotics harnesses (E4) treat skill invocation as an atomic operation: the harness invokes a skill, verifies the outcome, and on failure re-plans at the task level, discarding the partial trajectory. This atomicity is structurally forced by the action interfaces they consume — UniAct (E2, E6) maps heterogeneous embodiments into a Universal Action Space but provides no semantics for partially-executed universal actions, and EnerVerse (E5, E9) emits fixed-length action chunks via its policy head with no notion of substituting a suffix of a chunk under a mid-trajectory failure signal. Because neither interface exposes a compositional closure under failure, the harness must fall back to task-level re-planning, so recovery cannot preserve the cross-embodiment semantics that the universal action representation was designed to carry.

## Key Insight

A Universal Action Space admits a substitution algebra — a partial monoid whose composition is defined exactly when the residual goal-state distance contracts — so that failure recovery becomes an algebraic re-composition of universal actions rather than a task-level re-plan, and this closure is embodiment-invariant because it is defined on the universal action rather than on any robot's command space.

## Method

**What it is.** FSC (Failure-Substitution Closure) is a wrapper around a UniAct-style universal-action encoder and an EnerVerse-style action-chunk policy head that turns skill invocation into a composable operation. Its inputs are (i) the current universal action segment $u_{t:t+k}$ emitted by the policy head, (ii) a mid-trajectory failure signal $f$ from the harness's verify stage, and (iii) the grounded goal state $g$ in the world-model state space; its output is a substitution sequence $\sigma = (u^{(1)}, u^{(2)}, \dots)$ of universal action segments, each with an embodiment-specific decoder tag, that the harness can invoke as a recoverable chain.

**How it works.**

```pseudocode
# Inputs: universal action chunk u[0:k], failure signal f at step j<k,
#         grounded goal g (world-model state), WM rollout M, decoder D
# Hyperparams: contraction threshold tau=0.15, max substitutions S=3,
#              chunk length k=8, rollouts per candidate R=4,
#              calibration buffer size B=512, confidence level 1-alpha=0.95

FSC(u[0:k], f, g):
    # 1. Locate the failure boundary in universal-action coordinates
    j = first step where f fires
    prefix = u[0:j]                      # executed, kept as-is
    residual_goal = M.rollout(prefix) -> g_hat   # predicted state after prefix
    delta_0 = M.temporal_distance(g_hat, g)     # scalar, in WM units

    # 2. Build the substitution candidate set by *intervening* on the
    #    universal action, not by retrieving from a library
    candidates = []
    for s in 1..S:
        # re-encode the residual goal into universal-action space
        u_s = UniAct.encode_residual(g_hat, g)   # universal action, no embodiment
        # compose: prefix ++ u_s ++ learned tail, all in universal space
        tail = PolicyHead.continue_from(g_hat, u_s, length=k-j)
        candidates.append( compose(prefix, u_s, tail) )

    # 3. Contraction test: a substitution is admissible iff the
    #    predicted residual distance contracts by >= (1-tau)
    #    AND the contraction is validated against an empirically calibrated
    #    recoverability estimate on the target embodiment.
    admissible = []
    for cand in candidates:
        g_c = M.rollout(cand)
        delta_c = M.temporal_distance(g_c, g)
        if delta_c <= (1 - tau) * delta_0:
            # 3a. Calibration check: use a held-out buffer of executed
            #     (prefix, substitution, outcome) tuples on the target embodiment
            #     to estimate the true recoverability probability p_hat.
            #     Admit only if the lower confidence bound of p_hat exceeds 0.5.
            p_hat, lcb = calibrate(cand, embodiment, buffer)
            if lcb > 0.5:
                admissible.append((cand, delta_c))

    # 4. Pick the admissible substitution with the smallest delta_c;
    #    if none, return the empty substitution (harness re-plans)
    if admissible is empty: return []
    best = argmin over delta_c
    # 5. Decode to embodiment-specific commands and emit as a chain
    return [ D.decode(seg) for seg in best ]
```

A **closure operator** $C$ wraps this: given a substitution sequence, $C(\sigma)$ re-runs step 3 on the *composed* sequence, so that a substitution may itself be substituted. The operator is idempotent on admissible sequences by construction (the contraction test is monotone in residual distance), and it is exposed to the harness's invoke stage as a single call `invoke_with_closure(skill, goal, f)`.

**Calibration procedure.** The calibration check in step 3a uses a replay buffer of size $B=512$ per embodiment, populated during training by executing candidate substitutions and recording binary success (whether the residual distance to $g$ actually decreased by at least $(1-\tau)$ in the real environment). For a candidate $cand$ on embodiment $e$, we compute the empirical success rate $\hat{p}$ over the $n_e$ most similar buffered tuples (cosine similarity in universal-action space, $n_e=32$), and the lower confidence bound $lcb = \hat{p} - z_{1-\alpha} \sqrt{\hat{p}(1-\hat{p})/n_e}$ with $\alpha=0.05$. A candidate is admitted only if $lcb > 0.5$. This makes the definedness condition depend on an empirically calibrated recoverability estimate rather than the raw WM temporal-distance ratio alone; the WM distance is used only to rank candidates that pass the calibration check.

**Why this design.**

We chose an *algebraic contraction test on the universal action* over a learned failure classifier because the classifier would be a log-prob gating signal (Anti-Pattern 3) with no calibration story for task correctness, whereas the contraction test is a closed-form predicate on a quantity — temporal distance to the grounded goal in the world model — that UniAct's universal action already encodes; the cost is that the test is only as good as the world model's distance estimate, which we accept because EnerVerse's EnerVerse-A head already produces this estimate as a byproduct of its denoising rollout. We chose to build substitutions by *intervening on the universal action* (`encode_residual`) rather than retrieving a nearest skill from a library, because retrieval (Toolformer/Gorilla/RAG-style k-NN over skill descriptors) cannot produce a segment that was never in the library, and the residual goal typically lies off the library manifold; the cost is that `encode_residual` must be trained, which we do by distilling the world model's goal-conditioned distance into the universal-action encoder. We chose a bounded substitution depth $S=3$ with early exit over unbounded recursion because the harness has a reaction-time budget (E4), accepting that some deep failures will fall through to task-level re-planning. This is not a variant of UniAct+EnerVerse composition (Anti-Pattern 5): the composition alone produces a cross-embodiment policy but no closure under failure, because neither interface defines what a *partially executed* universal action means; FSC adds exactly that semantics via the contraction predicate, and a domain expert would describe it as a substitution algebra on the action interface, not as a pipeline glue.

**Why it measures what we claim.**

The contraction ratio $\delta_c / \delta_0$ measures *recoverability of the skill invocation* because we assume the world model's temporal distance is a sufficient statistic for goal progress along the universal action manifold — an assumption that holds when the WM is trained on the same embodiment distribution as the policy head, and that fails when the WM is queried on an out-of-distribution embodiment, in which case the ratio reflects WM extrapolation error rather than recoverability. The admissibility predicate `delta_c <= (1-tau)*delta_0` measures *compositional closure* because we assume the universal action space is a partial monoid under `compose` with the contraction predicate as its definedness condition; this assumption fails when two universal actions share no structural feature (e.g., a locomotion and a manipulation skill with disjoint embodiment supports), in which case `compose` returns a segment that decodes to an incoherent command and the predicate reflects decoder incoherence rather than non-closure. The closure operator $C$'s idempotence measures *stability of recovery* because we assume the contraction test is monotone in residual distance, which fails under stochastic WM rollouts with high variance, in which case idempotence reflects rollout noise rather than algebraic closure. The calibration check (step 3a) measures *empirical recoverability* because we assume the held-out buffer is representative of the target embodiment's execution outcomes; this assumption fails when the buffer is small or biased (e.g., only successful substitutions recorded), in which case the lower confidence bound reflects buffer coverage rather than true recoverability.

## Contribution

['A failure-substitution algebra over the Universal Action Space: a partial monoid with a contraction-predicate definedness condition and an idempotent closure operator that re-composes universal actions under mid-trajectory failure while preserving cross-embodiment semantics.', "A concrete instantiation, FSC, that exposes this algebra to an agentic harness's invoke stage as a recoverable, composable skill-invocation primitive, replacing atomic invocation with a substitution chain whose admissibility is a closed-form test on world-model temporal distance.", 'A design principle: cross-embodiment recovery is achievable without task-level re-planning precisely when the failure signal is lifted into universal-action coordinates before substitution, because embodiment-specific command space is where composability is lost.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Two-embodiment manipulation suite (e.g., Franka Panda and UR5) from an existing multi-task benchmark (e.g., RoboSuite or Meta-World) with programmatic mid-trajectory perturbation injection (random joint offset, object slip) | Tests failure recovery across heterogeneous embodiments with controlled perturbations |
| Primary metric | Closed-loop task success rate under injected failures | Directly measures recoverability claim, not imitation fidelity |
| Secondary metric | Harness-level re-plan frequency (number of task-level re-plans per episode) | Measures reduction in costly global re-planning |
| Baseline | UniAct + EnerVerse pipeline, no closure | Isolates whether closure adds anything beyond composition |
| Baseline | Fine-tuned UniAct policy (per-embodiment) | Tests cross-embodiment generalization of recovery without retraining from scratch |
| Baseline | Harness re-plan from scratch on failure | Tests whether substitution beats global replanning |
| Ablation-of-ours | FSC with retrieval-based substitution instead of encode_residual | Tests whether intervention beats library lookup |
| Ablation-of-ours | FSC with WM-distance-only admissibility (no calibration check) | Tests whether calibration is necessary for recoverability |
| Calibration test | Rank correlation (Spearman) between WM temporal distance and true goal progress along decoded universal-action segments, across both embodiments | Validates monotonicity assumption underlying contraction predicate |

### Why this setup validates the claim

The central claim is that FSC's contraction test on universal-action coordinates yields recoverable substitution chains where naive composition, task-specific recovery, and global re-planning do not. The dataset choice is load-bearing: injecting mid-trajectory failures on a two-embodiment suite forces the policy to encounter the partially-executed-action state that FSC claims to define, and it does so across embodiments so that the cross-embodiment claim is exercised rather than assumed. The primary metric — closed-loop success under injected failure — is the right detector because it is sensitive exactly to the recoverability property the contraction ratio is supposed to measure; a method that merely imitates well but cannot recover will show clean-rollout parity and failure-rollout gap. The UniAct+EnerVerse baseline tests the composition-without-closure sub-claim: if FSC's gain over it is real, closure must be doing work that the pipeline glue cannot. The fine-tuned UniAct policy baseline tests the cross-embodiment sub-claim, since it is strong per-embodiment but cannot transfer without retraining. The re-plan-from-scratch baseline tests the substitution-beats-replanning sub-claim, since it is a legitimate alternative recovery mechanism rather than a strawman. The retrieval ablation is the diagnostic that separates intervention from lookup: if retrieval matches encode_residual, the intervention story is wrong and the method reduces to a skill library with extra steps. The WM-distance-only ablation tests whether the calibration check is necessary: if it matches full FSC, the calibration step is redundant; if it underperforms, the calibration is doing the work of correcting WM extrapolation error. The calibration test directly measures the monotonicity assumption: if Spearman correlation is low or negative on either embodiment, the contraction predicate is unfounded and any success-rate gain must be attributed to other factors.

### Expected outcome and causal chain

**vs. UniAct + EnerVerse pipeline (no closure)** — On a case where the policy head emits a chunk that drifts off the goal manifold at step j and the harness's verify stage fires, the baseline produces the executed prefix followed by an unchanged tail that continues the drifted trajectory, because neither interface defines what a partially executed universal action means and there is no predicate to reject the tail. Our method instead re-encodes the residual goal into universal-action space, composes a candidate, and admits it only if the world-model temporal distance contracts by at least tau and the calibrated recoverability lower bound exceeds 0.5, because the contraction predicate plus calibration supplies exactly the definedness condition the pipeline lacks. We expect a noticeable success-rate gap concentrated on the failure-injected subset with near-parity on clean rollouts.

**vs. Fine-tuned UniAct policy** — On a case where a failure fires on an embodiment whose recovery data was thin, the baseline produces a recovery behavior tuned to the embodiments it saw during fine-tuning and mis-executes on the held-out embodiment, because its recovery is indexed to embodiment-specific control interfaces and does not transfer. Our method instead builds the substitution in universal-action coordinates and only decodes to embodiment-specific commands at the final step, because the universal action space abstracts away the interface heterogeneity that blocks transfer. We expect the gap to widen as embodiment distance from the training distribution grows, with parity on the most-represented embodiment.

**vs. Harness re-plan from scratch** — On a case where a failure fires late in a long-horizon task after substantial progress, the baseline produces a full task-level re-plan that discards the executed prefix and re-attempts earlier subgoals, because it has no notion of preserving the prefix as a committed partial execution. Our method instead keeps the executed prefix as-is and only substitutes the residual segment, because the substitution algebra is defined relative to the failure boundary j rather than the task start. We expect a gap that grows with prefix length and shrinks on short-horizon tasks where re-planning is cheap, and we expect FSC to show a lower harness-level re-plan frequency.

**vs. Retrieval-based substitution ablation** — On a case where the residual goal lies off the skill-library manifold, the retrieval ablation produces the nearest library segment, which decodes to a command that does not reduce the residual distance, because k-NN over skill descriptors cannot synthesize a segment that was never in the library. Our method instead synthesizes the substitution by intervening on the universal action via encode_residual, because the encoder is distilled to map residual goals into universal-action space rather than to match descriptors. We expect the ablation to match us on in-library residuals and fall behind specifically on off-manifold residuals, which is the diagnostic that the intervention is doing the work.

**vs. FSC with WM-distance-only admissibility (no calibration)** — On a case where the WM is queried on an out-of-distribution embodiment, the WM-distance-only ablation admits candidates that appear to contract in predicted distance but fail to execute, because the WM distance is systematically overoptimistic off-distribution. Our method instead requires the calibrated recoverability lower bound to exceed 0.5, which corrects for WM extrapolation error. We expect the ablation to underperform on OOD embodiments and match on in-distribution embodiments, confirming the calibration check is necessary.

### What would falsify this idea

If FSC's gain over the no-closure pipeline is uniform across clean and failure-injected subsets rather than concentrated on the failure-injected subset, the contraction test is not doing what we claim and the gain is attributable to something else. If the retrieval ablation matches FSC specifically on off-manifold residuals, the intervention story is wrong and the method reduces to a skill library with extra machinery. If the calibration test shows low or negative Spearman correlation between WM temporal distance and true goal progress on either embodiment, the contraction predicate is unfounded and the method's admissibility criterion is invalid.

## References

1. Envisioning Agentic Robotics: A New Era of Robotic Systems
