# Topological Invariance Credit Assignment: Identifiable Step-Level Credit from Trajectory-Graph Re-Timing

## Motivation

Across the agentic RL lineage, step-level policies are optimized against a scalar reward whose granularity does not match the optimization unit, and neither Agent-R1's step-level trajectory representation nor Math-Shepherd's Monte-Carlo-rollout step rewards ever formalizes when a step-level credit signal is identifiable from trajectory structure alone — both silently assume an externally supplied, granularity-matched signal (a judge, a ground-truth final answer, or an outcome broadcast). The structural root cause is that credit is anchored to an external scalar rather than to a property of the trajectory DAG itself, so the step signal has no calibration target independent of the reward's source and inherits the reward's granularity mismatch. We reframe the problem at the meta level: a valid step-credit assignment must be invariant to re-timing of causally-equivalent sub-trajectories, an invariance that is checkable purely on the graph and therefore pins credit down from topology rather than from an external judge.

## Key Insight

Step credit is identifiable from the trajectory DAG's topology because any valid credit assignment must assign equal total credit to sub-trajectories that are causally equivalent under node-splitting (series-parallel re-timing), and this invariance constraint alone fixes the credit up to a single per-trajectory scale that the outcome reward then calibrates.

## Method

**FARA (Flow-Invariant credit Assignment)** takes as input a set of step-level trajectories represented as DAGs (nodes = interaction steps with state/action/observation, edges = causal transitions) plus a single scalar outcome reward per trajectory, and outputs a per-node credit vector whose within-trajectory proportions are determined by topology plus a stated parallel-node tie-breaking axiom, and whose scale is set by the outcome.

**(A) What it is.** FARA is a constrained credit-assignment operator: it solves for node credits that (i) are consistent with the outcome reward when summed along any root-to-terminal path, (ii) are invariant under a set of graph re-timing transformations that generate causally-equivalent sub-trajectories, and (iii) resolve credit among incomparable (parallel) nodes by an explicit equal-split-among-concurrent-branches axiom. The invariance constraint plus the parallel-node axiom are the identification device; the outcome reward only fixes scale. The identifiability claim is stated as: credit is identifiable up to the stated parallel-node axiom and a single per-trajectory scale, not fixed by topology alone.

**(B) How it works.**

```pseudocode
Input: trajectory DAG G=(V,E), outcome reward R, invariance tolerance eps,
       orbit sample size K=32, Tikhonov lambda=1e-3
Output: credit c_v for each node v in V, invariance violation V(G),
        null-space rank r_null

1. BUILD re-timing orbit (sampled, polynomial time).
   For each maximal series-parallel sub-DAG S of G (found by the standard
   SP decomposition in O(|V|+|E|)):
   - sample K=32 re-timings tau uniformly from the orbit of S by applying
     random node-splitting/node-merging moves that preserve the reachability
     relation on V (each move O(|V|); total O(K*|V|) per sub-DAG)
   - each tau yields a transformed DAG G_tau with the same terminal set

2. WRITE invariance constraints. For every sampled re-timing tau and every
   pair of nodes (u,v) whose causal order is preserved by tau:
       c_u(G) - c_v(G) = c_{tau(u)}(G_tau) - c_{tau(v)}(G_tau)      (INV)
   i.e. relative credit between causally-ordered nodes is re-timing invariant.

3. WRITE parallel-node axiom. For every pair of incomparable nodes (u,v)
   that are concurrent (neither reaches the other) and share the same
   immediate predecessor and successor sets in the SP sub-DAG:
       c_u(G) = c_v(G)                                          (PAR)
   i.e. equal-split-among-concurrent-branches: concurrent parallel branches
   receive equal credit. This is the stated tie-breaking axiom that resolves
   the parallel-node null space left open by INV.

4. WRITE path-sum constraints. For every root-to-terminal path P in G:
       sum_{v in P} c_v = R                                     (SCALE)

5. SOLVE the linear system [INV; PAR; SCALE] for c by minimum-norm least
   squares with Tikhonov regularization lambda=1e-3 on ||c||^2. Report the
   numerical rank r_null of the stacked constraint matrix (SVD, tolerance
   1e-6) alongside c; r_null is logged per trajectory.

6. CALIBRATION CHECK. Compute invariance violation
       V(G) = mean_{tau} || c(G) - push_tau(c(G_tau)) ||_2
   If V(G) > eps (eps=0.05 * |R|), flag trajectory as miscalibrated and
   down-weight its contribution to the policy-gradient batch by 1/(1+V(G)).

7. RETURN c, V(G), r_null.
```

Credits c feed a step-level policy-gradient estimator: the advantage of node v is A_v = c_v - baseline(state_v), where baseline is an exponential moving average (decay 0.99) of c_v over nodes sharing the same state-cluster (cluster by observation embedding, k=32 via k-means). The invariance-violation score V(G) is used only as a batch-level reliability weight, never as a per-node gate. The reported r_null is logged for every trajectory and used to stratify the synthetic ground-truth credit-recovery analysis (see experiment).

**(C) Why this design.** We chose a linear invariance-plus-scale system over a learned credit head because the identification claim is a property of the graph, not of a function approximator: a learned head (as in Math-Shepherd's step reward model) can fit any labeling and therefore cannot certify identifiability, accepting the cost that FARA cannot extrapolate credit to node types absent from the re-timing orbit. We chose minimum-norm least squares over an unregularized solve because the INV constraints leave a null space that is one-dimensional only after the PAR axiom is imposed on concurrent parallel branches; pinning that scale with SCALE while regularizing the rest gives a unique, reproducible credit vector, accepting the cost that lambda must be tuned per environment scale and that the identifiability claim is conditional on the PAR axiom rather than on topology alone. We chose to enumerate re-timings only over maximal series-parallel sub-DAGs rather than the full graph because series-parallel structure is exactly where causal equivalence under re-timing is well-defined, accepting the cost that graphs with irreducible cross-edges (non-series-parallel) receive weaker constraints and rely more on SCALE. A domain expert would not describe FARA as a variant of VinePPO or Process Reward Models: those estimate per-step value by Monte-Carlo rollouts or supervised step labels, whereas FARA never rolls out and never labels — it solves a graph-constrained linear system whose solution is fixed by topology plus the stated PAR axiom. Nor is it 'X+Y in Z': the invariance constraint is not present in any composition of GRPO/PPO with step rewards.

**(D) Why it measures what we claim.** The invariance-violation score V(G) measures miscalibration of the step-credit signal because the assumption that causally-equivalent re-timings of a series-parallel sub-DAG must receive equal relative credit makes V(G) zero exactly when credit is a function of causal position rather than of timing; this assumption fails when the environment's transition kernel is timing-dependent (e.g., a tool with a rate limit makes two causally-equivalent orders non-equivalent), in which case V(G) reflects genuine timing effects rather than miscalibration. The minimum-norm credit c_v measures causal contribution of node v because the assumption that the outcome reward decomposes additively along root-to-terminal paths (SCALE) plus the PAR axiom on concurrent parallel branches makes the path-sum the only scale constraint and INV+PAR the only shape constraints; this assumption fails when reward is super-additive across branches (e.g., two tool calls jointly enabling a capability neither enables alone), in which case c_v reflects an average over decompositions rather than a unique contribution. The batch weight 1/(1+V(G)) measures trajectory reliability because the assumption that high invariance violation indicates a credit signal inconsistent with the graph structure makes the weight monotone in signal trustworthiness; this assumption fails when V(G) is high due to a genuinely non-series-parallel trajectory rather than miscalibration, in which case the weight down-weights a valid but structurally unconstrained trajectory. The reported null-space rank r_null measures how much of the credit vector is fixed by the constraints versus by the minimum-norm convention: r_null=1 indicates the constraints pin credit up to the per-trajectory scale, while r_null>1 indicates the PAR axiom is insufficient for that trajectory's parallel structure and the minimum-norm solve is doing tie-breaking work.

## Contribution

(1) A graph-constrained credit-assignment operator (FARA) that identifies step credit from trajectory-DAG topology via a re-timing invariance constraint plus a path-sum scale constraint, solved as a regularized linear system with no learned credit head and no external judge. (2) A label-free calibration statistic — the invariance-violation score — that is computable on any step-level trajectory artifact (including Agent-R1's step representation) and serves as a batch-level reliability weight for policy-gradient training. (3) A concrete characterization of when step credit is identifiable from structure alone (series-parallel sub-DAGs) versus when it degrades to an averaged decomposition, giving practitioners a checkable precondition on their trajectory graphs.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Multi-step tool-use agent tasks with DAG-able traces | Provides series-parallel structure FARA assumes |
| Dataset (real benchmark) | ALFWorld (134 unseen eval tasks; report fraction of trajectories that are series-parallel) | Tests applicability beyond synthetic DAG-able tasks |
| Dataset (synthetic ground-truth) | Controlled DAG environment with planted decisive nodes (N=500 trajectories; series-parallelism fraction swept over {0.25, 0.5, 0.75, 1.0}; reward super-additivity swept over {0, 0.5, 1.0}) | Grounds identifiability claim against known true credit |
| Primary metric | Success rate on held-out tasks | Directly reflects credit quality via policy improvement |
| Secondary metric (synthetic) | Credit-recovery error: mean absolute error between FARA c_v and planted true credit, stratified by series-parallelism and super-additivity | Directly tests identifiability claim |
| Secondary metric (diagnostic) | Reported null-space rank r_null per trajectory (SVD tolerance 1e-6) | Tests whether INV+PAR leaves one-dimensional null space |
| Baseline | GRPO with trajectory-level advantage | Tests whether step credit beats outcome-only credit |
| Baseline | VinePPO-style Monte-Carlo step values (N=16 rollouts per step) | Tests whether graph-solve beats rollout estimation |
| Baseline | Math-Shepherd-style learned step reward (2-layer MLP, hidden=256, GeLU) | Tests whether topology certification beats learned head |
| Baseline | Shapley-value credit over trajectory coalitions (exact for |V|≤12, sampled otherwise) | Tests whether FARA is a known allocation rule in disguise |
| Ablation-of-ours | FARA with INV constraints removed (SCALE+PAR only) | Isolates contribution of invariance identification |
| Ablation-of-ours | FARA with PAR axiom removed (INV+SCALE only) | Isolates contribution of parallel-node tie-breaking |
| Ablation-of-ours | FARA with constant baseline (replace k-means EMA baseline) | Isolates credit-shape contribution from baseline heuristic |

### Why this setup validates the claim

The central claim is that credit fixed by graph topology plus the stated PAR axiom, not by a function approximator or Monte-Carlo rollouts, yields a more identifiable step-level signal and thus better policy learning on multi-step agent tasks. The dataset must contain trajectories whose interaction structure is genuinely series-parallel, since that is exactly where FARA's re-timing orbit is well-defined and where the invariance constraint carries information; tasks with irreducible cross-edges would weaken the test by construction, so we report the series-parallel fraction on ALFWorld and condition the success-rate gain on that fraction. Success rate is the right primary metric because credit assignment is only instrumentally valuable — it must translate into policy improvement, and a credit vector that is topologically elegant but useless for learning would be exposed by flat success rate. The synthetic ground-truth experiment is the direct test of identifiability: because the true per-node credit is known by construction, we can measure recovery error as a function of series-parallelism and reward super-additivity, and we can report the null-space rank r_null to check whether INV+PAR actually leaves a one-dimensional null space or whether the minimum-norm solve is doing arbitrary tie-breaking. Each baseline tests a distinct sub-claim: GRPO-with-trajectory-advantage tests whether step-level credit matters at all; VinePPO tests whether solving a linear system beats Monte-Carlo step-value estimation; Math-Shepherd tests whether a topology-certified signal beats a learned step reward that can fit any labeling; Shapley-value credit tests whether FARA is a known allocation rule in disguise. The ablations are the sharpest tests: removing INV leaves only SCALE+PAR, so if FARA's gain comes from the invariance identification device rather than from merely having per-step credits, the ablation should recover most of the baseline behavior and lose most of FARA's gain; removing PAR isolates the parallel-node tie-breaking axiom; replacing the k-means EMA baseline with a constant baseline isolates the credit-shape contribution from the baseline heuristic. Together these form a falsifiable design because the predicted effect is localized: FARA should win specifically on trajectories with rich series-parallel structure and on tasks where timing-equivalent reorderings are genuinely equivalent, should show parity elsewhere, and should show credit-recovery error increasing with reward super-additivity.

### Expected outcome and causal chain

**vs. GRPO with trajectory-level advantage** — On a case where a long trajectory contains one decisive correct tool call followed by several neutral steps, GRPO assigns the same outcome-derived advantage to every step because it has no within-trajectory shape information. Our method instead solves the INV+PAR+SCALE system so the decisive node receives proportionally larger credit, because the path-sum constraint forces the outcome to be distributed according to causal position rather than uniformly. We expect a noticeable success-rate gap concentrated on long trajectories with sparse decisive steps, and near-parity on short trajectories where uniform credit is already approximately correct.

**vs. VinePPO-style Monte-Carlo step values** — On a case where a step's value estimate requires many rollouts to stabilize and the environment has high branching, VinePPO produces noisy per-step values because its estimate is a finite-sample average over sampled continuations. Our method instead computes credit by solving a deterministic linear system on the observed DAG, because topology plus the outcome reward plus the PAR axiom pin the credit shape without sampling. We expect FARA to match VinePPO on low-branching tasks but show a clear advantage on high-branching tasks where rollout variance dominates, with the gap widening as branching factor increases.

**vs. Math-Shepherd-style learned step reward** — On a case where the training distribution contains few examples of a particular node type, the learned step reward produces confidently wrong step labels because a function approximator can fit any labeling and has no identifiability guarantee off-distribution. Our method instead derives credit from the graph structure of each trajectory, because the invariance constraint plus the PAR axiom certify credit as a function of causal position rather than of learned weights. We expect FARA to hold steady on rare node types where the learned baseline degrades, producing a gap that concentrates on low-frequency step categories and vanishes on common ones.

**vs. Shapley-value credit** — On a case where the trajectory DAG has branching and re-convergence, Shapley-value credit requires marginal contributions over all coalitions, which is exponential in |V| and must be sampled for |V|>12, introducing estimation variance. Our method instead solves a polynomial-time linear system (O(K*|V|) orbit sampling plus least squares) whose solution is fixed by INV+PAR+SCALE. We expect FARA to match Shapley on series-parallel trajectories with low super-additivity (where the PAR axiom coincides with Shapley symmetry) but to diverge on trajectories with high super-additivity, where Shapley captures interaction effects that FARA's additive path-sum cannot; we report this divergence as a diagnostic, not as a win.

### What would falsify this idea

If FARA's gain over the INV-removed ablation is uniform across all trajectory subsets rather than concentrated on trajectories with rich series-parallel structure and timing-equivalent reorderings, then the invariance constraint is not doing the identification work we claim and the central mechanism is wrong. Likewise, if the reported null-space rank r_null is consistently greater than 1 even after the PAR axiom, or if FARA's credit-recovery error on the synthetic ground-truth experiment does not increase with reward super-additivity, then the identifiability claim is false and the minimum-norm solve is doing arbitrary tie-breaking rather than recovering causal contribution.

## References

1. Agent-R1: A Unified and Modular Framework for Agentic Reinforcement Learning
2. HybridFlow: A Flexible and Efficient RLHF Framework
3. DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models
4. Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations
5. OpenWebMath: An Open Dataset of High-Quality Mathematical Web Text
6. Improving Generalization of Alignment with Human Preferences through Group Invariant Learning
7. Oobleck: Resilient Distributed Training of Large Models Using Pipeline Templates
8. GEAR: A GPU-Centric Experience Replay System for Large Reinforcement Learning Models
