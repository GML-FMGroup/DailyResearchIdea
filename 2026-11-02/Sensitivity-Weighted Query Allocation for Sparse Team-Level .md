# Sensitivity-Weighted Query Allocation for Sparse Team-Level Human Feedback

## Motivation

Learning from sparse aggregate team-level human feedback faces a feedback-budget closure problem: as team size and modality count grow, a fixed budget of feedback queries can no longer constrain each agent's policy independently. Prior work such as Multi-Robot Learning from Human Feedback outlines deployment strategies for aligning supervisor intent with robot teams but does not address how the aggregate signal's Fisher information about any single agent's policy decays with team size, so uniform or variance-based per-agent credit assignment leaves alignment error growing with the team. The structural cause is that existing allocation rules treat the aggregate feedback operator as isotropic across agents, ignoring that its sensitivity to individual policies and the cross-agent coupling induced by shared modalities are separable quantities that can be computed and exploited.

## Key Insight

The aggregate team-feedback operator decomposes into a per-agent sensitivity term (the Fisher information each agent's policy contributes to the aggregate signal) times a cross-agent coupling term induced by shared modalities, and querying agents in proportion to the sensitivity-to-coupling ratio keeps per-agent alignment error bounded under a fixed budget regardless of team size.

## Method

**What it is**
SQUAD (Sensitivity-Weighted Query Allocation under Decomposition) is a query-allocation rule for sparse team-level human feedback. Its inputs are the current team policy parameters, the shared-modality coupling graph, and a fixed feedback budget B; its output is a per-agent query count n_i summing to B and an updated credit-assignment weight per agent used in the downstream policy update.

**How it works**

```pseudocode
Input: team policies {pi_i}, shared-modality coupling graph G, budget B, temperature tau=0.1, eps=1e-6, recompute interval K=10
# Phase 1: per-agent sensitivity of the aggregate feedback operator
for each agent i:
    s_i = || d E[F(team)] / d theta_i ||_2      # gradient of expected aggregate feedback wrt agent i's params
# Phase 2: cross-agent coupling induced by shared modalities
for each pair (i,j) sharing a modality in G:
    C_ij = || d^2 E[F(team)] / (d theta_i d theta_j) ||_F   # Hessian cross-block norm, computed via double-backward (functorch)
c_i = sum_j C_ij                                  # coupling load of agent i
# Phase 3: sensitivity-to-coupling allocation
r_i = s_i / (c_i + eps)                            # eps = 1e-6
n_i = B * softmax(r_i / tau)                       # tau = 0.1, fixed budget split
# Phase 4: credit weight for downstream update
w_i = s_i / sum_k s_k
return {n_i}, {w_i}
```

**Assumption A (stated explicitly):** The per-agent alignment error under a fixed query budget scales as c_i/s_i, i.e., sensitivity s_i (gradient norm of expected aggregate feedback) and cross-agent coupling c_i (Hessian cross-block norm) are separable, independently estimable quantities whose ratio r_i = s_i/c_i is the correct marginal-error-reduction-per-query ordering across agents. This holds under the local-quadratic approximation of the feedback operator and when coupling is direct (via shared modalities) rather than mediated by a third agent. It fails when the feedback operator is strongly non-quadratic (e.g., preference feedback with near-ties), in which case s_i reflects local slope rather than Fisher information, and when coupling is mediated, in which case C_ij under-counts the coupling load.

**Calibration check (verification of Assumption A):** Every K=10 feedback steps, we hold out a calibration batch of 512 team-feedback queries and compute the empirical per-agent alignment error e_i (mean squared error between the aggregate feedback and the team policy's predicted feedback, projected onto agent i's parameter block). We regress e_i against c_i/s_i across agents using ordinary least squares and record the R^2. If R^2 < 0.5, Assumption A is flagged as violated for the current task, and the allocation falls back to uniform (n_i = B/N) for the next K steps. This calibration uses the same backward passes as Phase 1 and Phase 2, adding no extra gradient computations.

**Why this design**
We chose to compute the per-agent sensitivity s_i as the norm of the gradient of the expected aggregate feedback rather than a variance-based or uniform allocation because the aggregate signal's Fisher information about agent i is exactly this gradient norm under the local-quadratic approximation, and variance-based rules conflate coupling with sensitivity; the cost is that s_i requires one backward pass per agent per allocation round, which we amortize by recomputing only every K=10 feedback steps. We chose the Hessian cross-block norm C_ij over a simpler correlation proxy because shared modalities make agents' effects on the aggregate feedback non-additive, and only the cross-block captures that non-additivity; the cost is a per-pair Hessian-vector product, which we limit to pairs that share a modality in G. We chose a softmax with temperature tau over a hard argmax allocation because hard allocation starves agents whose sensitivity is temporarily low but whose coupling load will rise, and softmax guarantees every agent receives at least a floor query count; the cost is that the floor dilutes the budget available to high-sensitivity agents, which we bound by setting tau small (0.1). Unlike the uniform and variance-based credit assignment discussed in Multi-Robot Learning from Human Feedback, and unlike the trajectory-level optimization of StarPO in RAGEN, SQUAD does not optimize a policy directly; it produces an allocation whose ratio s_i/c_i is the quantity that governs per-agent error under sparse aggregate feedback, so a domain expert would not describe it as a variant of either.

**Why it measures what we claim**
The gradient norm s_i measures per-agent sensitivity of the aggregate feedback because under the local-quadratic assumption the expected feedback's curvature with respect to theta_i equals the Fisher information that feedback carries about agent i's policy; this assumption fails when the feedback operator is strongly non-quadratic (e.g., preference feedback with near-ties), in which case s_i reflects local slope rather than information and the allocation over-queries agents with high slope but low information. The cross-block norm C_ij measures cross-agent coupling induced by shared modalities because the mixed second derivative is nonzero exactly when agents' policy changes interact in the aggregate signal, which is the definition of coupling through a shared modality; this assumption fails when coupling is mediated by a third agent rather than a direct shared modality, in which case C_ij under-counts the coupling load and the allocation under-queries the mediator. The ratio r_i = s_i/c_i measures the marginal error reduction per query because the per-agent alignment error under a fixed budget scales with c_i/s_i when sensitivity and coupling are separable; this assumption fails when sensitivity and coupling are correlated across agents (e.g., a single dominant agent drives both), in which case the ratio saturates and the softmax allocation approaches uniform, which is the correct fallback but loses the scaling benefit. The calibration R^2 measures whether Assumption A holds on the current task because it directly tests whether empirical per-agent alignment error tracks c_i/s_i; this check fails when the feedback operator is non-quadratic or coupling is mediated, in which case the fallback to uniform allocation is triggered.

## Contribution

(1) SQUAD, a query-allocation rule for sparse team-level human feedback that decomposes the aggregate feedback operator into a per-agent sensitivity term and a shared-modality coupling term and allocates a fixed budget by their ratio. (2) The design principle that sensitivity-to-coupling ratio, not raw sensitivity or variance, is the quantity that governs per-agent alignment error under sparse aggregate feedback, yielding a softmax allocation that degrades gracefully to uniform when the two quantities are correlated. (3) A coupling-graph construction from shared modalities that makes the cross-agent Hessian blocks computable with a bounded number of Hessian-vector products per allocation round.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Cooperative shared-modality MARL tasks with sparse team reward (e.g., Multi-Agent Particle Environment with a shared vision cone, team sizes N ∈ {2,4,8,16}) | Shared modality induces coupling the method targets |
| Primary metric | Team return at fixed feedback budget B (B=1000 queries) | Isolates allocation efficiency under fixed budget |
| Secondary metric | Per-agent alignment error e_i (MSE between aggregate feedback and predicted feedback projected onto agent i's parameter block) vs. team size N | Directly tests claimed team-size-independent error bound |
| Baseline | Uniform query allocation | Tests whether sensitivity-weighting beats equal split |
| Baseline | Variance-based credit allocation | Tests whether coupling term adds beyond variance proxy |
| Baseline | StarPO-style trajectory-level optimization | Tests allocation rule against direct policy optimization |
| Baseline | Fisher-ratio allocation (D-optimality on diagonal of inverse FIM) | Tests whether separable ratio is a relabeling of information-theoretic rule |
| Ablation-of-ours | SQUAD with c_i fixed to constant | Isolates contribution of coupling load term |
| Ablation-of-ours | SQUAD with calibration fallback disabled (always use r_i) | Tests whether calibration check is necessary |
| Ablation-of-ours | SQUAD with near-tie preference feedback variant (feedback operator with near-ties) | Tests failure mode F of Assumption A |

### Why this setup validates the claim

The central claim is that allocating queries by the sensitivity-to-coupling ratio r_i = s_i/c_i reduces per-agent alignment error under a fixed budget more than allocation rules that ignore either sensitivity or coupling. The uniform baseline tests the sub-claim that non-uniform allocation matters at all: if SQUAD only matches uniform, sensitivity-weighting is inert. The variance-based baseline tests the sub-claim that the coupling term c_i carries information variance conflates away: if SQUAD only matches variance-based allocation, the Hessian cross-block is decorative. The StarPO-style baseline tests whether an allocation rule can compete with directly optimizing a trajectory-level policy, which addresses the natural objection that allocation is a weaker lever than optimization. The Fisher-ratio baseline tests whether the separable ratio s_i/c_i is merely a relabeling of a standard information-theoretic allocation (D-optimality on the diagonal of the inverse Fisher information matrix); if SQUAD matches Fisher-ratio, the decomposition adds no new ordering. The ablation (c_i constant) is the decisive test: it removes exactly the coupling mechanism while preserving sensitivity, so any gain that survives ablation cannot be attributed to coupling. The calibration-fallback ablation tests whether the Assumption A check is load-bearing: if disabling it does not hurt, the check is unnecessary. The near-tie preference variant directly probes failure mode F: if SQUAD's advantage disappears or reverses there, Assumption A is correctly identified as the boundary. Fixed-budget team return is the right metric because it measures the downstream consequence of allocation rather than the allocation itself; an allocation rule that looks well-shaped but does not convert into return at equal budget has not validated the claim. The per-agent alignment error vs. team size N directly tests the claimed team-size-independent error bound: if error grows with N for SQUAD but not for baselines, the bound is false.

### Expected outcome and causal chain

**vs. Uniform query allocation** — On a case where two agents share a visual modality and one agent's policy changes dominate the aggregate feedback while the other's are near-flat, uniform allocation spends half the budget on the near-flat agent, so its queries return low-information gradients and the shared-modality agent's alignment lags. Our method computes s_i per agent and shifts queries toward the high-sensitivity agent, so we expect a clear return gap on shared-modality task subsets and near-parity on independent-modality subsets where sensitivities are similar. Additionally, we expect per-agent alignment error to remain flat as team size N grows from 2 to 16 for SQUAD, while uniform allocation's error grows linearly with N.

**vs. Variance-based credit allocation** — On a case where a shared modality makes two agents' effects non-additive, variance-based allocation assigns credit by per-agent reward variance, which conflates a genuinely coupled agent with a noisy but uncoupled one, over-querying the noisy agent. Our method uses the Hessian cross-block norm C_ij to identify the coupled pair and down-weights its coupling load in r_i, so we expect the gap to concentrate on task variants with strong cross-agent interaction and to shrink on variants with weak interaction.

**vs. StarPO-style trajectory-level optimization** — On a case with a very sparse team reward and a tight feedback budget, trajectory-level optimization must spread signal across a long rollout, so per-agent credit is diluted and learning stalls on the agent whose contribution is masked by teammates. Our method allocates queries by r_i before the update, so the masked agent receives enough queries to resolve its gradient, and we expect faster return growth early in training at equal budget, with the advantage narrowing once the budget is large enough that dilution is no longer binding.

**vs. Fisher-ratio allocation (D-optimality on diagonal of inverse FIM)** — On a case where sensitivity and coupling are correlated across agents (e.g., a single dominant agent drives both), the separable ratio r_i = s_i/c_i saturates and softmax approaches uniform, whereas Fisher-ratio allocation uses the full diagonal of the inverse FIM to account for correlation. We expect SQUAD to match Fisher-ratio on tasks where Assumption A holds (calibration R^2 > 0.5) and to underperform on tasks where Assumption A is violated (calibration R^2 < 0.5), demonstrating that the decomposition is not a relabeling but a simplification with a measurable boundary.

### What would falsify this idea

If SQUAD's gain over uniform allocation is uniform across shared-modality and independent-modality subsets, rather than concentrated where coupling load c_i is high, then the coupling mechanism is not doing the work the claim assigns it. Likewise, if the c_i-constant ablation matches full SQUAD on the high-coupling subsets, the Hessian cross-block term is not the causal driver and the central claim is wrong. Additionally, if per-agent alignment error for SQUAD grows with team size N at the same rate as uniform allocation, the claimed team-size-independent error bound is false. If SQUAD matches Fisher-ratio allocation on all tasks including those where calibration R^2 < 0.5, the separable ratio is a relabeling and the decomposition adds no new ordering.

## References

1. Multi-Robot Learning from Human Feedback
2. Discovery and Deployment of Emergent Robot Swarm Behaviors via Representation Learning and Real2Sim2Real Transfer
3. Zespol: A Lightweight Environment for Training Swarming Agents
4. Optimising autonomous robot swarm parameters for stable formation design
5. RAGEN: Understanding Self-Evolution in LLM Agents via Multi-Turn Reinforcement Learning
6. Plan-and-Act: Improving Planning of Agents for Long-Horizon Tasks
7. Training Language Models to Self-Correct via Reinforcement Learning
8. Is Your LLM Secretly a World Model of the Internet? Model-Based Planning for Web Agents
