# Failure-Substitution Closure Testing for LLM Operations Research Agents

## Motivation

Existing agent benchmarks such as OR-License assume that simulated client feedback is always actionable and cooperative, so they never test whether an agent's action set remains closed under rejection — i.e., whether after a client rejects a proposed formulation, the agent can produce at least one valid alternative that satisfies the stated feedback. This is a structural gap in interactive optimization evaluation: ORSim's rejection path records only the rejection, not the agent's recovery trajectory, so an agent that silently stalls or repeats its rejected formulation is scored identically to one that successfully substitutes a valid alternative.

## Key Insight

Recovery from rejection is a closure property of the agent's action set under a feedback-induced substitution operator, and it can be tested without a learned judge by precomputing, per rejection scenario, the set of formulations that provably satisfy the client's stated constraints and checking membership of the agent's subsequent trajectory in that set.

## Method

**What it is.** We introduce FSC (Failure-Substitution Closure), a benchmark layer that augments ORSim with rejection scenarios and a precomputed valid-substitute oracle. Inputs: an OR-License instance, a rejection scenario (client feedback string plus the rejected formulation), and the agent's post-rejection trajectory. Output: a binary closure score and a per-scenario recovery trace.

**How it works.**

```pseudocode
# Phase 1: scenario construction (offline, per OR-License instance)
for each instance I in OR-License (31 instances):
    F* = expert_accepted_formulation(I)              # from OR-License hidden checks
    for each constraint c in client_requirements(I):  # e.g. budget, service level, horizon
        r = rejection_template(c, F*)                # client rejects F* on constraint c
        S_r = enumerate_substitutes(I, c, budget=B)  # B = 200 candidates, see Phase 2
        store(r, S_r)

# Phase 2: substitute-set enumeration (offline)
# For each rejection r, S_r is the set of formulations F' such that:
#   (a) F' satisfies all client requirements in I except possibly c
#   (b) F' satisfies c strictly (constraint c is the feedback target)
#   (c) F' is feasible under hidden checks of I
# Enumerated by perturbing the formulation family (e.g. swapping objective
# weighting, tightening a bound, changing a policy class) and filtering
# through the OR-License feasibility checker; B=200 caps cost.
# Fallback: if perturbation yields <10 substitutes for an instance, we run a
# constraint-solver-based generation (using the OR-License feasibility checker
# as a black-box predicate) to sample additional candidates up to B=200.

# Phase 3: online closure test
for each scenario (r, S_r):
    agent.reset(I); agent.propose(F0)                # F0 = agent's first formulation
    client.reject(r)                                 # ORSim emits feedback r
    T = agent.trajectory_after_rejection(max_steps=K)  # K = 20 steps
    hit = any(F_t in S_r for F_t in T)               # membership check
    closure_score += hit
closure = closure_score / num_scenarios
```

**Two-stage closure certification (replaces binary membership).** For each scenario, we compute a certified closure score:

```pseudocode
for each scenario (r, S_r):
    agent.reset(I); agent.propose(F0)
    client.reject(r)
    T = agent.trajectory_after_rejection(max_steps=K)  # K = 20 steps
    certified = False
    for F_t in T:
        if F_t in S_r:
            certified = True; break
        else:
            # out-of-set: directly check constraint c and feasibility
            if satisfies_constraint(F_t, c) and feasible_under_hidden_checks(F_t, I):
                certified = True; break
    closure_score += certified
closure = closure_score / num_scenarios
```

We report two diagnostics alongside closure: (i) enumeration coverage = |S_r| / B per scenario, and (ii) out-of-set certification rate = fraction of certified hits that were not in S_r. These are reported separately and do not enter the closure denominator.

**Feedback-randomization control.** To test whether in-set hits reflect feedback-driven recovery rather than blind search, we run a control condition where the rejection feedback r is replaced by a random rejection template r' drawn uniformly from the set of all rejection templates for instance I (including templates targeting constraints other than c). We compute closure under this randomized-feedback condition and compare to the original closure. If closure under randomized feedback is statistically indistinguishable from original closure, the hits are attributable to blind search rather than feedback use.

**Why this design.** We chose precomputed substitute sets over an LLM judge because an LLM judge would reintroduce the very calibration problem the benchmark is meant to avoid, accepting the cost that substitute enumeration is bounded by B=200 and may miss exotic valid formulations; we mitigate this by (a) the two-stage certification that checks out-of-set formulations directly against the constraint predicate and feasibility checker, so the oracle is defined by constraint satisfaction rather than by the finite enumeration, and (b) reporting enumeration coverage and out-of-set certification rate as diagnostics. We chose constraint-targeted rejection templates over free-form client feedback because free-form feedback is not reproducible across agents and would make closure scores incomparable, accepting the cost that templated rejections are less naturalistic than OR-License's original ORSim dialogue. We chose trajectory membership over final-answer-only checking because an agent that stumbles onto a valid substitute after several invalid attempts still demonstrates closure, accepting the cost that K=20 step truncation may miss late recoveries. Unlike OR-License, which evaluates only the terminal deliverable via DAQ, and unlike Toolformer/Gorilla-style retrieval benchmarks that score single-shot tool selection, FSC scores a property of the agent's sequential action set under an intervention (rejection); a domain expert would not describe FSC as a variant of retrieval because the substitute set is defined by constraint satisfaction, not by similarity to a query.

**Why it measures what we claim.** The computational quantity `certified = (F_t in S_r) or (satisfies_constraint(F_t, c) and feasible_under_hidden_checks(F_t, I))` measures failure-substitution closure because we assume that the constraint predicate and feasibility checker exactly characterize the formulations satisfying the client's rejection feedback; this assumption fails when the client's feedback is under-specified (e.g., 'make it cheaper' without a budget), in which case the predicate reflects our enumerated interpretation rather than the client's true intent, and certified measures closure relative to that interpretation. The quantity `closure = closure_score / num_scenarios` measures robustness of the agent's action set across rejection types because we assume scenarios are sampled to cover the constraint space of each instance; this fails when an instance has only one binding constraint, in which case closure reflects recovery on that single constraint rather than breadth. The quantity `K = 20` bounds the recovery window and measures whether closure is achieved within a realistic interaction budget because we assume OR-License clients tolerate at most 20 post-rejection exchanges; this fails for high-dimensional instances where valid substitutes require more exploration, in which case certified measures early-recovery ability rather than closure proper. The feedback-randomization control measures whether certified hits are attributable to feedback use because we assume that randomizing the feedback content should reduce closure if agents are using the feedback; this fails when agents ignore feedback and search blindly, in which case closure under randomized feedback equals original closure and the metric reflects search ability rather than feedback-driven recovery.

## Contribution

['A benchmark layer, FSC, that augments ORSim with rejection scenarios and precomputed valid-substitute sets, turning OR-License from a terminal-deliverable benchmark into one that also scores recovery trajectories.', 'A constraint-targeted rejection template family and substitute-enumeration procedure that make closure scores reproducible across agents without an LLM judge.', 'An analysis of which constraint types (budget, service level, horizon) produce the largest closure gaps across frontier LLM agents, identifying where interactive OR agents fail to substitute rather than fail to solve.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | OR-License (31 instances, constraint-targeted rejection scenarios) | Provides hidden checks and expert formulations for substitute enumeration. |
| Primary metric | FSC certified closure score (per constraint type and instance) | Directly measures post-rejection recovery via constraint predicate, not just enumeration. |
| Baseline | ORSim terminal DAQ (OR-License default) | Tests whether terminal-only scoring misses recovery behavior. |
| Baseline | Final-answer-only closure | Tests whether trajectory membership adds signal over endpoint matching. |
| Baseline | LLM-judge closure | Tests whether judge calibration reintroduces the problem FSC avoids. |
| Ablation-of-ours | FSC with free-form rejection feedback | Tests whether templated rejections are necessary for comparability. |
| Control | FSC with randomized rejection feedback | Tests whether certified hits depend on feedback content vs. blind search. |
| Diagnostic | Enumeration coverage (|S_r|/B) and out-of-set certification rate | Quantifies oracle completeness and reliance on direct predicate checks. |

### Why this setup validates the claim

The design is falsifiable because each baseline and control isolates one sub-claim of FSC. ORSim terminal DAQ tests the claim that terminal-only evaluation cannot distinguish agents that recover from rejection from agents that never recover; if terminal DAQ and FSC certified closure rank agents identically, the trajectory-level signal is redundant. Final-answer-only closure tests the claim that intermediate formulation attempts matter; if endpoint matching equals trajectory membership, the K-step window and trajectory machinery are unjustified. The LLM-judge baseline tests the claim that precomputed substitute sets plus direct constraint checks avoid calibration drift; if judge-based closure correlates near-perfectly with FSC certified closure across constraints, the oracle is unnecessary overhead. The free-form-rejection ablation tests whether templated, reproducible feedback is what makes closure comparable across agents; if free-form yields the same per-constraint pattern, the template cost is not load-bearing. The randomized-feedback control tests whether certified hits reflect feedback-driven recovery rather than blind search; if closure under randomized feedback is statistically indistinguishable from original closure, the metric is not measuring feedback use. The primary metric is the right detector because certified closure is defined as satisfying the constraint predicate and feasibility checks, so it should rise specifically where the predicted failure mode (constraint mis-targeting after rejection) occurs, and stay flat elsewhere. The diagnostics (enumeration coverage and out-of-set certification rate) quantify how often the finite enumeration S_r is bypassed by direct predicate checks, directly addressing the closed-world oracle concern.

### Expected outcome and causal chain

**vs. ORSim terminal DAQ (OR-License default)** — On a case where the client rejects the expert formulation on a service-level constraint, the baseline produces a low DAQ score because it only inspects the terminal deliverable and cannot tell whether the agent explored a feasible substitute before settling. Our method instead checks whether any intermediate formulation satisfies the constraint predicate (either in S_r or certified out-of-set), because it scores the sequential action set under the rejection intervention. We expect a noticeable gap concentrated on instances with multiple binding constraints, with near-parity on single-constraint instances where terminal and trajectory signals coincide.

**vs. Final-answer-only closure** — On a case where the agent stumbles onto a valid substitute at step 6 but then reverts to an infeasible formulation by step 20, the baseline scores zero because its mechanism inspects only the endpoint. Our method instead credits the step-6 hit via trajectory membership (certified by constraint predicate), because it treats recovery as a property of the action set rather than the final answer. We expect the gap to concentrate on high-dimensional instances requiring exploration, with parity on easy instances where the first post-rejection proposal is already valid.

**vs. LLM-judge closure** — On a case where the rejection is under-specified and the judge must infer intent, the baseline produces unstable closure scores because its mechanism relies on an uncalibrated judge whose acceptance threshold drifts across constraint types. Our method instead uses a precomputed substitute set plus direct constraint-predicate certification, because membership is defined by constraint satisfaction (via the predicate and feasibility checker) rather than judge opinion. We expect our scores to be stable across reruns and to diverge from the judge specifically on ambiguous rejections, while agreeing on unambiguous ones.

**vs. Randomized-feedback control** — On a case where the agent receives a rejection targeting a different constraint than the one it violated, the control condition produces lower certified closure if the agent is using feedback to guide substitution. Our method's original closure should be significantly higher than randomized-feedback closure, because the agent's recovery is feedback-driven. We expect the gap to be largest on instances with multiple binding constraints and near-zero on single-constraint instances where any rejection points to the same constraint.

### What would falsify this idea

If FSC certified closure gains over terminal DAQ and final-answer-only baselines are uniform across all constraint types and instances rather than concentrated where the predicted constraint-mis-targeting failure occurs, the central claim that FSC captures a distinct recovery property is wrong. Likewise, if the free-form-rejection ablation reproduces the same per-constraint pattern as templated rejections, the comparability argument for templating collapses. Additionally, if certified closure under randomized feedback is statistically indistinguishable from original closure, the metric reflects blind search rather than feedback-driven recovery.

## References

1. OR-License: Evaluating LLM Agents as Real-World Operations Research Practitioners
