# Certified Coverage Auditing for LLM-Agent Skill Bases via Constraint-Checked Rejection Sampling

## Motivation

Across both automated skill discovery (EXIF) and constraint-following benchmarks (AgentIF, CFBench, SysBench, FollowBench), a single generator's support—explorer trajectories or human annotation—is treated as a complete basis for the admissible behavior space, with coverage asserted rather than certified. This structural meta-gap means downstream training and evaluation inherit an unquantified blind spot: EXIF trains Bob on Alice's retrospectively generated skills without bounding the probability mass of admissible behaviors Alice never visited, and constraint benchmarks similarly assume their annotated constraint sets exhaust the space of constraint-satisfying behaviors.

## Key Insight

Because admissibility is defined by constraints that are checkable on any candidate behavior, the uncovered admissible mass can be estimated distribution-free by rejection sampling against the constraint basis using only the trajectory set—without access to the true admissible space—yielding a finite-sample upper bound that serves as a training and evaluation gate.

## Method

**A. What it is**

COCOA (Certified COverage Certificate for Agents) takes as input a set of explorer trajectories `T = {τ_1, ..., τ_n}` and a constraint basis `C = {c_1, ..., c_m}` (each constraint a programmatic checker `c_j: Behavior → {0,1}`), and outputs a certified upper bound `Û_α` on the probability mass of admissible (constraint-satisfying) behaviors outside the support of `T`, together with a gate decision (PASS/FAIL) at a user-specified tolerance `ε`. The bound is explicitly decomposed into a certifiable within-proposal-support term and an uncertifiable out-of-proposal-support term, and the gate is only issued when the out-of-support term is empirically bounded below a user-specified caveat threshold `γ`.

**B. How it works**

```python
def COCOA(T, C, ε, δ, γ=0.05, B=1000, n_cal=200, n_hold=500):
    # Phase 1: Extract support skeleton from trajectories
    S = set()
    for τ in T:
        for step in τ:
            S.add(canonicalize(step))          # canonical form: (action_type, args_hash)
    S = prune_redundant(S, coverage_radius=0.1)  # deduplicate near-identical steps

    # Phase 2: Fit a proposal distribution q over behaviors
    #   q is a smoothed empirical distribution over S, with a uniform floor
    q = smoothed_empirical(S, floor=0.05)

    # Phase 3: Rejection sampling against C
    #   Draw candidate behaviors from q, keep those satisfying all c_j
    accepted = []
    for _ in range(B * n_cal):
        b = sample(q)
        if all(c_j(b) == 1 for c_j in C):
            accepted.append(b)
            if len(accepted) >= n_cal:
                break

    # Phase 4: Estimate uncovered mass via importance weighting
    #   For each accepted b, compute whether b lies in S (support)
    #   Uncovered mass = P_q(admissible & not in S) / P_q(admissible)
    #   Use Clopper-Pearson to get a finite-sample upper bound
    n_acc = len(accepted)
    if n_acc < n_cal:
        return FAIL(reason="insufficient admissible samples")
    n_out = sum(1 for b in accepted if not in_support(b, S))
    # Finite-sample upper bound on p = P(not in S | admissible)
    p_hat = n_out / n_acc
    p_upper = clopper_pearson_upper(n_out, n_acc, delta=δ)

    # Phase 4b: Empirical bound on out-of-proposal-support mass
    #   Draw n_hold fresh admissible behaviors from q (same rejection sampler)
    #   and count how many have q(b) < 1e-6 (near-zero proposal density)
    hold = []
    for _ in range(B * n_hold):
        b = sample(q)
        if all(c_j(b) == 1 for c_j in C):
            hold.append(b)
            if len(hold) >= n_hold:
                break
    n_near_zero = sum(1 for b in hold if q_density(b) < 1e-6)
    p_out_upper = clopper_pearson_upper(n_near_zero, len(hold), delta=δ)

    # Phase 5: Gate decision
    #   Gate only if out-of-support caveat is below γ
    if p_out_upper > γ:
        return FAIL(reason="out-of-proposal-support mass exceeds caveat threshold", bound=p_upper, caveat=p_out_upper)
    if p_upper <= ε:
        return PASS(bound=p_upper, caveat=p_out_upper)
    else:
        return FAIL(bound=p_upper, caveat=p_out_upper)
```

**C. Why this design**

We chose rejection sampling against the constraint basis over training a discriminator to distinguish admissible from inadmissible behaviors because the constraint basis is programmatic and checkable on any candidate, making rejection sampling exact and distribution-free, accepting the cost that the proposal distribution `q` must have sufficient overlap with the admissible set or the acceptance rate becomes prohibitively low. We chose a smoothed empirical distribution over the support skeleton as the proposal `q` rather than a uniform distribution over all possible behaviors because the support skeleton concentrates mass near explored regions, making the importance-weighting estimate of uncovered mass more efficient, accepting the cost that `q` may under-cover distant admissible regions and thus the bound is conservative (upper bound) rather than tight. We chose Clopper-Pearson over a normal approximation for the finite-sample upper bound because it provides exact coverage guarantees even for small `n_acc`, accepting the cost that the bound is wider (more conservative) than a Wald interval. We chose to gate on the upper bound `p_upper` rather than the point estimate `p_hat` because the certificate must be valid with probability at least `1-δ` under the sampling randomness, accepting the cost that the gate is more stringent and may require larger `B` or `n_cal` to pass. We added the out-of-proposal-support caveat term `p_out_upper` and the gate condition `p_out_upper ≤ γ` because the original certificate's validity depends on `q` having full support on the admissible set, and this condition makes that dependence explicit and testable rather than assumed. A domain expert would NOT describe this as a variant of EXIF because EXIF uses Alice's trajectories as a training set without any coverage certification; COCOA treats the trajectory set as a proposal for rejection sampling and computes a certified residual mass, which is a fundamentally different operation on the same input.

**D. Why it measures what we claim**

The computational quantity `p_upper` (the Clopper-Pearson upper confidence bound on the fraction of admissible samples not in the support) measures the true uncovered admissible mass because the assumption that rejection sampling from `q` yields i.i.d. samples from the conditional distribution `P(· | admissible)` holds when `q` has full support on the admissible set; this assumption fails when `q` assigns zero probability to some admissible region (e.g., a behavior requiring a tool not present in any trajectory), in which case `p_upper` reflects only the uncovered mass within the support of `q` and underestimates the true uncovered mass. The support skeleton `S` measures the explorer's coverage because the assumption that canonicalized steps capture all behaviorally relevant distinctions holds when the canonicalization is a sufficient statistic for constraint satisfaction; this assumption fails when two behaviors have identical canonical forms but differ in constraint satisfaction (e.g., different argument values that trigger different constraints), in which case `S` overestimates coverage and `p_upper` underestimates uncovered mass. The gate decision at tolerance `ε` measures whether the skill base is certified complete because the assumption that `ε` is a meaningful threshold for downstream training/evaluation holds when the downstream task's sensitivity to uncovered mass is known; this assumption fails when the downstream task is highly sensitive to rare behaviors, in which case a PASS at `ε` may still leave critical blind spots. The out-of-proposal-support caveat `p_out_upper` measures the fraction of admissible behaviors that `q` assigns near-zero probability because the assumption that the held-out admissible samples from `q` are representative of the admissible set holds when `q` has full support; this assumption fails when `q` has zero mass on some admissible region, in which case `p_out_upper` underestimates the true out-of-support mass and the gate condition `p_out_upper ≤ γ` may pass even though the certificate's scope is limited to the reachable proposal support.

## Contribution

1. We introduce COCOA, the first framework that computes a distribution-free finite-sample upper bound on the probability mass of admissible behaviors outside an explorer's trajectory support, using only the trajectory set and a programmatic constraint basis, without access to the true admissible space. 2. We provide a design principle for using this bound as a training and evaluation gate: the certificate enables practitioners to decide whether a skill base is sufficiently complete for downstream use, and to quantify the residual blind spot when it is not. 3. We release an open-source implementation of COCOA with support for canonicalization, proposal smoothing, and Clopper-Pearson bounds, along with a diagnostic that reports which constraint types contribute most to uncovered mass.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | AGENTIF constraint-annotated instructions | Human-annotated constraints give ground-truth admissibility checks |
| Primary metric | Certified uncovered-mass bound vs. realized violation rate | Directly tests whether certificate tracks true blind spots |
| Baseline | EXIF (trajectory-as-training-set) | Tests whether certification adds value beyond raw exploration |
| Baseline | Uniform-proposal rejection sampler | Tests whether support-skeleton proposal drives the gain |
| Baseline | Point-estimate gate (p_hat, no CP) | Tests whether the finite-sample upper bound is necessary |
| Ablation-of-ours | COCOA without canonicalization pruning | Tests whether support skeleton fidelity matters |
| Ablation-of-ours | COCOA without out-of-support caveat (γ=1) | Tests whether the caveat term is necessary for validity |

### Why this setup validates the claim

The central claim is that COCOA's certified upper bound `p_upper` is a *valid* (conservative) estimate of uncovered admissible mass and that gating on it correctly flags incomplete skill bases. AGENTIF is the right dataset because each instruction carries explicit, programmatically checkable constraints, so we can compute the *true* uncovered admissible mass on a held-out set of admissible behaviors and compare it against `p_upper`. The EXIF baseline tests the sub-claim that certification is more than exploration: if EXIF's raw trajectory coverage already predicts violations as well as `p_upper`, the certificate adds nothing. The uniform-proposal baseline tests the sub-claim that the support-skeleton proposal is what makes rejection sampling efficient enough to produce a usable bound. The point-estimate baseline tests the sub-claim that the Clopper-Pearson upper bound (rather than `p_hat`) is what makes the gate valid at confidence `1-δ`. The pruning ablation tests the sub-claim that the support skeleton's canonicalization is a faithful coverage proxy. The out-of-support caveat ablation tests the sub-claim that the caveat term `p_out_upper` is necessary to prevent false PASSes when `q` lacks support on admissible regions. The primary metric — the relationship between certified bound and realized violation rate on held-out admissible behaviors — is the right one because it directly measures calibration of the certificate, which is the entire claim; a method that under-covers but reports a low bound would be caught as a bound below the realized rate.

### Expected outcome and causal chain

**vs. EXIF** — On a case where the explorer's trajectories miss a rare admissible behavior (e.g., a tool combination never exercised), EXIF produces a skill dataset with no signal about the gap because it treats trajectories as a training set with no residual-mass estimate. Our method instead computes `p_upper` over the admissible conditional and flags the gap, because rejection sampling against the constraint basis surfaces admissible-but-unexplored behaviors. We expect EXIF's realized violation rate on held-out admissible behaviors to exceed its (nonexistent) coverage claim, while COCOA's `p_upper` sits above the realized rate — a noticeable gap on rare-behavior subsets but parity on dense subsets. Additionally, we expect COCOA's caveat term `p_out_upper` to be small (≤0.05) on instances where the explorer's trajectories cover all tools, and larger on instances with unseen tool combinations, correctly triggering FAIL when `p_out_upper > γ`.

**vs. Uniform-proposal rejection sampler** — On a case where admissible behaviors are concentrated in a small region of behavior space, the uniform proposal produces almost no accepted samples because acceptance rate collapses, yielding either FAIL-on-insufficient-samples or a wildly wide bound. Our method instead draws from the smoothed empirical skeleton, so accepted samples concentrate near explored admissible regions and the bound is usable. We expect uniform-proposal to either fail to certify or return bounds an order of magnitude looser, while COCOA returns bounds within a small factor of the realized rate on the same instances.

**vs. Point-estimate gate** — On a case where `n_acc` is small (a rare constraint combination), the point-estimate gate passes because `p_hat` happens to land below `ε`, but the true uncovered mass exceeds `ε` on held-out admissible behaviors. Our method instead gates on the Clopper-Pearson upper bound, which widens under small `n_acc` and refuses to certify. We expect the point-estimate gate to show a cluster of false PASSes on small-`n_acc` instances, while COCOA's PASS set has realized violation rate at or below `ε` — a gap concentrated exactly where `n_acc` is small.

**vs. COCOA without out-of-support caveat (γ=1)** — On a case where the explorer's trajectories miss an admissible behavior requiring an unseen tool, the proposal `q` assigns near-zero probability to that region, so `p_upper` underestimates the true uncovered mass and the gate without caveat may PASS incorrectly. Our method with caveat computes `p_out_upper` on held-out admissible samples and refuses to certify when `p_out_upper > γ`. We expect the no-caveat variant to show false PASSes on instances with unseen tools, while the full COCOA correctly FAILs or reports a caveat term that bounds the underestimation.

### What would falsify this idea

If COCOA's certified `p_upper` falls *below* the realized violation rate on held-out admissible behaviors — i.e., the certificate systematically under-covers — then the central validity claim is wrong. Equivalently, if the point-estimate gate and COCOA's gate produce identical PASS/FAIL decisions across instances (no benefit from the finite-sample bound), the claimed role of Clopper-Pearson is falsified.

## References

1. Automated Skill Discovery for Language Agents through Exploration and Iterative Feedback
2. AGENTIF: Benchmarking Instruction Following of Large Language Models in Agentic Scenarios
3. SysBench: Can Large Language Models Follow System Messages?
4. CFBench: A Comprehensive Constraints-Following Benchmark for LLMs
5. Instruction-Following Evaluation for Large Language Models
6. FollowBench: A Multi-level Fine-grained Constraints Following Benchmark for Large Language Models
7. Can LLMs Follow Simple Rules?
8. Can Large Language Models Understand Real-World Complex Instructions?
