# Feedback-Sufficiency Certification for Decentralized Multi-Robot Policies

## Motivation

Existing multi-robot learning-from-human-feedback methods, such as R2BC (round-robin single-agent demonstrations) and MAPF-GPT (imitation from a fixed expert dataset), implicitly assume that the collected team-level feedback uniquely determines a decentralized joint policy. Because decentralized execution induces local-observation aliasing—different joint policies can produce identical local observation-action histories—team-level feedback is often consistent with an equivalence class of joint policies, so feedback scarcity or ambiguity silently yields under-determined teams that appear trained but fail in deployment. No prior work defines a computable condition under which team-level demonstrations, corrections, or preferences are sufficient to identify the joint policy, leaving a structural meta-gap across all human-in-the-loop branches.

## Key Insight

The set of decentralized joint policies consistent with observed team-level feedback forms an equivalence class whose local dimension can be computed from the feedback stream via a local-observation aliasing operator, and this dimension yields a feedback-sufficiency certificate that is both necessary and sufficient for unique identification within the decentralized policy class.

## Method

**What it is.** FSC (Feedback-Sufficiency Certificate) takes as input a decentralized policy class (finite set of local observation-based policies), a team-level feedback stream (demonstrations, corrections, or pairwise preferences over team rollouts), and a candidate joint policy; it outputs a certificate score κ ∈ [0,1] and, when κ < 1, a minimal repair set of local state-action pairs whose additional feedback would restore sufficiency.

**Load-bearing assumption (stated explicitly).** The certificate is valid under the assumption A that the finite codebook class Π (product of per-robot codebooks of size M_i = 64) is behaviorally representative of the continuous decentralized policy space, and that the aliasing relation ~_alias captures all observational indistinguishability relevant to identification. The failure mode F is off-support policies that differ on local observations never visited under the candidate policy: if ~_alias is defined only on the candidate's support, such policies are counted as distinguishable, inflating |C(F)/~_alias| and biasing κ upward. To close this gap, ~_alias is defined over a reachability closure (step 1) and the cardinality is replaced by a behavior-weighted effective count (step 3), making κ invariant to codebook resolution M_i.

**How it works.**
```pseudocode
Input: decentralized policy class Π = Π_1 × ... × Π_n (each Π_i finite, size M_i = 64),
       candidate joint policy π = (π_1,...,π_n) ∈ Π,
       feedback stream F = { (o_i^t, a_i^t) } or preferences {σ ≻ σ'},
       aliasing tolerance ε (default 0.05),
       reference visitation distribution μ_ref (estimated from F, see step 1b)
Output: certificate κ ∈ [0,1], repair set R ⊆ local state-action pairs

1. Build the local-observation aliasing equivalence relation on Π over a reachability closure:
   a. Reachability closure: starting from the local observations in F, compute the set O*_i
      of local observations reachable under any policy in Π_i by forward simulation of the
      team's joint observation model for at most H = 50 steps (or, if no simulator is
      available, use the union of local observations in F plus all local observations in the
      codebook's declared support).
   b. Reference distribution: estimate μ_ref over O*_1 × ... × O*_n from the empirical
      visitation frequencies in F, smoothed with a Dirichlet prior (α = 1) over the codebook
      support; if F is empty, μ_ref is uniform over O*.
   c. For each pair (π, π') ∈ Π × Π, define
        π ~_alias π'  iff  E_{o ~ μ_ref}[ 1{ π_i(o_i) ≠ π'_i(o_i) for some i } ] ≤ ε.
      Represent each equivalence class by its local action profile signature on O*.

2. Estimate the feedback-consistent set C(F) ⊆ Π:
   - For demonstrations/corrections: C(F) = { π' ∈ Π : π' reproduces every
     demonstrated local action on the observed local observation, up to ε }.
   - For preferences: C(F) = { π' ∈ Π : π' ranks the preferred rollout higher
     under the Bradley-Terry likelihood with temperature 1 }.

3. Compute the behavior-weighted local identifiability dimension:
   For each aliasing class c ∈ Π / ~_alias, let w(c) = μ_ref-mass of the local observation
   profiles on which c's action profile differs from the candidate π (0 if identical).
   Define the effective number of distinguishable classes as
     N_eff(F) = ( Σ_{c ∈ C(F)/~_alias} w(c) ) / ( Σ_{c ∈ Π/~_alias} w(c) )
   and set
     d_local(F) = log( N_eff(F) ) / log( N_eff(Π) ),
   where N_eff(Π) is computed with C(F) = Π. κ = 1 - d_local(F).
   (When all w(c) are equal, this reduces to the original cardinality ratio; the weighting
   makes κ invariant to codebook size M_i because adding ε-close codebook entries does not
   change the weighted mass.)

4. If κ < 1 - ε: extract a repair set R by solving a greedy set-cover over
   the local state-action pairs that discriminate the largest number of
   policy pairs within C(F) under μ_ref; return R as the minimal feedback query set.

5. Return (κ, R).
```

**Why this design.** We chose a finite policy class with an explicit aliasing equivalence relation over a continuous policy parameterization because the certificate must be computable from the feedback stream alone without access to a simulator or privileged joint-state information, accepting the cost that the class must be discretized (we use a product of per-robot behavior-cloning policy heads with a fixed codebook of local action profiles, size M_i = 64 per robot). We chose a cardinality-based dimension d_local over a Fisher-information or Hessian-based local sensitivity measure because aliasing is a global equivalence-class property, not a local curvature property, accepting the cost that the certificate is conservative when the class is large. We chose greedy set-cover for repair over an exact minimum-query optimization because the exact problem is NP-hard and the greedy solution has a logarithmic approximation that is sufficient for deployment, accepting the cost that the repair set may be up to log-factor larger than optimal. We chose to separate the demonstration/correction case from the preference case in step 2 rather than unify them under a single likelihood because preferences do not provide pointwise local actions and require a ranking model, accepting the cost of two code paths. A domain expert would not describe this method as a variant of R2BC or MAPF-GPT: those methods train a policy from feedback and never compute an identifiability certificate over the policy class, so they cannot detect when the feedback is insufficient; FSC operates on the feedback stream as a meta-level diagnostic, not as a training procedure.

**Why it measures what we claim.** The quantity N_eff(F) measures feedback sufficiency for unique identification because the assumption that the feedback stream is a complete local-action observation under the candidate policy makes the feedback-consistent set equal to the set of policies that reproduce the observed local behavior; this assumption fails when the feedback is sparse or noisy, in which case N_eff(F) reflects the volume of policies consistent with the incomplete feedback rather than the true identifiability gap. The quantity d_local(F) measures the local identifiability dimension because the assumption that the aliasing equivalence relation captures all observationally indistinguishable joint policies over the reachability closure makes the weighted quotient a faithful count of distinguishable policies; this assumption fails when two policies differ only on local observations outside the reachability closure, in which case d_local(F) reflects an over-count of distinguishable policies. The certificate κ = 1 - d_local(F) measures feedback sufficiency because the assumption that the policy class is finite and the feedback is consistent makes κ = 1 exactly when the feedback-consistent set collapses to a single aliasing class; this assumption fails when the feedback stream is contradictory, in which case κ reflects the inconsistency of the feedback rather than sufficiency. The repair set R measures the minimal additional feedback needed because the greedy set-cover assumption that each local state-action pair discriminates a known set of policy pairs makes the cover a valid query set; this assumption fails when the discrimination sets overlap in unmodeled ways, in which case R reflects an over-estimate of the minimal query set.

## Contribution

['A computable feedback-sufficiency certificate (FSC) that couples team-level human feedback informativeness to the identifiability of a decentralized joint policy via a local-observation aliasing equivalence relation, yielding a scalar κ ∈ [0,1] and a minimal repair set of local state-action queries.', 'A design principle for human-in-the-loop multi-robot learning: feedback should be collected until κ reaches 1 - ε, and the repair set identifies which local state-action pairs to query, converting feedback scarcity from a silent failure into a detectable and repairable condition.', 'An open-source implementation of the certificate and repair-set computation for finite decentralized policy classes, compatible with demonstrations, corrections, and pairwise preferences.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | Multi-agent task suite with sparse, per-robot sequential demos (3 tasks: cooperative navigation, predator-prey, traffic junction; 200 episodes per task, 3–5 robots) | Sparse per-robot feedback is exactly where sufficiency is in doubt |
| Primary metric | κ calibration: Spearman rank correlation between κ and true identifiability gap (estimated by enumerating the discretized class and counting behaviorally distinct policies consistent with feedback, using the same reachability closure and μ_ref as FSC) | Directly tests whether the certificate tracks the real gap |
| Secondary metric | Task success rate on aliasing subset vs. non-aliasing subset (defined by whether the candidate policy's support contains a local observation with ≥2 aliasing-distinct joint configurations) | Tests whether low κ predicts deployment failure |
| Baseline | Oracle joint-action behavior cloning | Upper bound with privileged synchronized feedback |
| Baseline | Independent behavior cloning (no round-robin) | Tests whether naive per-robot feedback looks sufficient |
| Baseline | R2BC-style round-robin imitation | Tests whether a feedback-training method masks insufficiency |
| Ablation-of-ours | FSC without aliasing quotient (raw |C(F)|/|Π|) | Isolates whether the aliasing relation is doing the work |
| Ablation-of-ours | FSC with codebook size M_i ∈ {16, 64, 256} | Tests invariance of κ to codebook resolution |
| Ablation-of-ours | FSC with μ_ref uniform vs. estimated from F | Tests sensitivity to reference distribution |
| Active-query baseline | Random-query selection using the same query budget as repair set R | Tests whether the repair set changes downstream task success |

### Why this setup validates the claim

The central claim is that FSC detects when a team-level feedback stream is insufficient to pin down a decentralized policy, and that this detection is what feedback-training baselines lack. The oracle joint-action baseline establishes the ceiling: if feedback were fully sufficient, its performance is what a correctly identified policy should approach, so it anchors the meaning of "sufficient." The independent and R2BC-style baselines test the opposite sub-claim: that a training procedure can consume sparse per-robot feedback and appear to converge while actually landing in a large aliasing class, i.e. matching behavior on visited observations but diverging off-support. The aliasing-stripped ablation is the key internal control: if κ's predictive power survives removing the quotient, the aliasing mechanism is not what makes the certificate work and the design rationale collapses. The codebook-size ablation directly tests the load-bearing assumption that κ is invariant to discretization: if κ's rank correlation changes with M_i, the certificate is measuring codebook granularity rather than sufficiency. The active-query comparison tests whether the repair set R is actionable: if querying R recovers more task success than random queries at the same budget, the certificate changes a downstream decision. The primary metric — rank correlation between κ and the measured true identifiability gap (estimated by enumerating the discretized class and counting behaviorally distinct policies consistent with feedback, using the same reachability closure and μ_ref as FSC) — is the right one because it is a monotone, scale-free test of whether the certificate orders feedback streams by actual sufficiency, which is exactly the claim; raw task performance would confound identification with policy quality.

### Expected outcome and causal chain

**vs. Oracle joint-action behavior cloning** — On a case where per-robot demonstrations are collected sequentially and robot 2's actions are conditioned on robot 1's unobserved-by-2 state, the oracle baseline never fails because it sees synchronized joint actions and its consistent set is a single aliasing class. Our method assigns κ ≈ 1 to that stream and a lower κ to the sequential stream, so we expect near-perfect rank correlation on oracle streams but a clear κ drop on sequential streams, with the oracle's task performance unreachable by any method trained only on the sequential stream.

**vs. Independent behavior cloning (no round-robin)** — On a case where each robot's local observation aliases two joint configurations requiring different coordinated actions, independent BC fits the marginal local action distribution and produces policies that look correct on the visited support. Our method computes N_eff(F) and finds many aliasing-distinct policies consistent with the same local actions, so κ falls well below 1 and the repair set names the aliasing local state-action pairs. We expect independent BC to show a large on-support/off-support performance gap while FSC predicts that gap in advance via low κ, giving a noticeable κ–gap correlation where the baseline provides no signal at all.

**vs. R2BC-style round-robin imitation** — On a case where round-robin feedback covers each robot's actions but never jointly disambiguates a coordination-relevant alias, R2BC converges to a policy that reproduces all demonstrated local actions yet fails on team coordination at test time. Our method's C(F) contains the aliasing-equivalent policies R2BC cannot distinguish, so κ stays below 1 and the repair set flags the missing joint-discriminating queries. We expect R2BC and FSC to have parity on task success on non-aliasing subsets but a pronounced gap on the aliasing subset, with FSC's repair set predicting exactly which episodes fail.

**vs. FSC without aliasing quotient (raw |C(F)|/|Π|)** — On the same aliasing subset, the raw ratio counts policies that differ on unvisited local observations as distinguishable, so it assigns a higher κ than full FSC. We expect the raw ratio's rank correlation with the true gap to be lower than full FSC's, and its κ to be less predictive of the on-support/off-support performance gap, confirming that the aliasing quotient is doing the work.

**vs. FSC with codebook size M_i ∈ {16, 64, 256}** — If κ is invariant to codebook resolution, the rank correlation and the κ values on a fixed set of feedback streams should be stable across M_i. We expect the behavior-weighted N_eff(F) to yield κ differences of at most 0.05 across M_i, whereas the raw cardinality ratio would shift by more than 0.2, demonstrating that the weighting closes the discretization gap.

**vs. FSC with μ_ref uniform vs. estimated from F** — On streams where the empirical visitation is highly non-uniform, the uniform reference over-weights rarely visited local observations. We expect the estimated μ_ref to yield higher rank correlation with the true gap (which is computed under the same μ_ref) and to produce repair sets that are smaller and more targeted, with the gap between the two variants growing as the feedback stream becomes sparser.

**vs. Random-query selection (same budget as repair set R)** — On streams where κ < 1, we query the local state-action pairs in R and retrain, versus querying the same number of pairs uniformly at random. We expect the R-query condition to recover at least 80% of the oracle's task success on the aliasing subset, while random queries recover at most 40%, showing that the repair set changes a downstream decision.

### What would falsify this idea

If κ's rank correlation with the true identifiability gap is uniform across aliasing and non-aliasing subsets — i.e. no concentration of predictive power where the aliasing mechanism is supposed to act — then the certificate is not measuring feedback sufficiency but some generic property of the feedback volume, and the central claim is wrong. Equivalently, if the aliasing-stripped ablation matches full FSC's correlation, or if κ's rank correlation varies by more than 0.1 across codebook sizes M_i ∈ {16, 64, 256}, the quotient or the behavior weighting is not doing the work and the load-bearing assumption fails.

## References

1. Multi-Robot Learning from Human Feedback
2. R2BC: Multi-Agent Imitation Learning from Single-Agent Demonstrations
3. MAPF-GPT: Imitation Learning for Multi-Agent Pathfinding at Scale
4. Transformer-Based Imitative Reinforcement Learning for Multirobot Path Planning
