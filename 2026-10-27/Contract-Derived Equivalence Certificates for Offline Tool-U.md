# Contract-Derived Equivalence Certificates for Offline Tool-Use Policy Learning

## Motivation

Across the tool-use agent literature, synthesized or simulated interactions are treated as supervision equivalent to live API execution without any per-decision check that the simulated outcome actually matches live tool semantics: Trajectory2Gym grounds decisions only against learner-visible history, DreamGym substitutes a reasoning-based experience model for real rollouts, and WebDreamer commits to actions based on LLM world-model simulations, all assuming fidelity they never establish. Structurally, none of these methods can certify a synthesized transition because they represent tool behavior as free-form generated text rather than as a position in a declared I/O contract, so there is no object against which simulated and live outcomes could be compared offline. The result is that offline learnability and bias-independence of synthesized supervision are assumed rather than established, and policies silently absorb transitions whose simulated outcome diverges from what the live API would return.

## Key Insight

A tool's declared I/O contract partitions all possible call outcomes into equivalence classes, and a synthesized transition is safe to learn from exactly when its simulated outcome lands in the same class as the live outcome would — a condition checkable offline from the contract alone, making the certificate a decision-theoretic bias bound rather than a confidence score.

## Method

**What it is.** CONCERT (Contract-derived Equivalence Certificate for offline tool-use learning) takes as input a set of synthesized executable interactions (each a sequence of tool calls with simulated observations), the tools' declared I/O contracts (typed argument schemas, return schemas, declared error/precondition clauses), a small held-out calibration set of live executions (default 512 calls sampled uniformly over tools and argument branches), and a candidate policy; it outputs a filtered, reweighted training set in which every retained transition carries a per-transition certificate, plus the trained policy.

**Load-bearing assumption (stated explicitly).** CONCERT's certificate is valid only under the assumption that a tool's declared I/O contract is complete and sound enough that the live outcome for any call falls inside the same contract-derived equivalence class as the simulated outcome. When this assumption holds, the offline class-level gradient-discrepancy bound B_t upper-bounds the policy bias injected by the synthesized transition. When it fails (undocumented side effects, over-specified preconditions, hidden state dependencies), B_t bounds only the documented-class discrepancy and is silent on cross-class error. CONCERT therefore does not assume completeness; it treats the contract-derived class as a hypothesis and estimates its coverage empirically via the calibration set, inflating B_t by a measured cross-class residual and abstaining on tools whose live outcomes fall outside the declared partition.

**How it works.**

```pseudocode
Input: synthesized trajectories T = {(s_t, a_t, o_t_sim)}, tool contracts C = {c_k},
       live calibration set L = {(s_i, a_i, o_i_live)} of size N_cal = 512,
       policy pi, bias tolerance delta = 1e-2, weight floor w_min = 0.1,
       coverage threshold tau_cov = 0.95
Output: certified dataset D_cert, trained policy pi*

# Phase 1: build equivalence classes from the contract (offline, no execution)
for each tool k with contract c_k:
    E_k = partition_outcome_space(c_k)
    # classes keyed by (return-schema variant, declared error clause, precondition branch)
    # e.g. "success with field X present", "declared 4xx precondition violation", "declared 5xx retryable"

# Phase 1b: empirical coverage calibration (offline, uses L)
for each tool k:
    n_k = number of calibration calls to tool k
    n_in = number of those whose live outcome classifies into some class in E_k
    cov_k = n_in / max(n_k, 1)                      # empirical coverage of declared partition
    # cross-class residual: max gradient discrepancy between live outcome and its
    # nearest declared class, over calibration calls that fell outside E_k
    R_k = max_{i in L_k, cls(o_i_live) not in E_k} min_{c in E_k} || grad_ell(pi, s_i, a_i, o_i_live) - grad_ell(pi, s_i, a_i, o_c) ||
    # o_c = canonical representative observation of class c
    if cov_k < tau_cov:
        mark tool k as ABSTAIN               # certificate undefined; transitions dropped

# Phase 2: per-transition certificate (no live call)
for each transition (s_t, a_t, o_t_sim) in T:
    k = tool_of(a_t)
    if k is ABSTAIN: drop transition
    cls_sim = classify(o_t_sim, E_k)
    # admissible classes: those reachable from a_t's arguments under c_k
    A = admissible_classes(a_t, c_k)
    if cls_sim not in A:
        drop transition                       # out-of-contract simulation
    # bias bound: max policy-gradient discrepancy over the class, inflated by measured residual
    B_t = sup_{o in cls_sim, o' in cls_sim} || grad_ell(pi, s_t, a_t, o) - grad_ell(pi, s_t, a_t, o') || + R_k
    cert_t = 1[ B_t <= delta ]               # binary certificate, delta fixed at 1e-2
    w_t = max(w_min, 1 - B_t / delta)        # continuous reweight, floor w_min = 0.1

# Phase 3: certified offline policy learning
D_cert = {(s_t, a_t, o_t_sim, w_t) : cert_t = 1}
pi* = argmin_pi  sum_{(s,a,o,w) in D_cert}  w * ell(pi, s, a, o)
     # ell = parent-normalized SFT loss (interaction as unit), as in Trajectory2Gym
```

**Why this design.** We chose a *contract-derived partition* over a learned outcome predictor (as in DreamGym's reasoning experience model) because the contract is the only object that is both available offline and causally upstream of the live outcome, accepting the cost that tools with weak or absent contracts cannot be certified and must be excluded. We chose a *decision-theoretic gradient-discrepancy bound* B_t over a scalar confidence score because a confidence score conflates simulation likelihood with policy-bias, while B_t directly upper-bounds how much a transition can move the policy; the cost is that B_t requires a differentiable loss and is vacuous for non-differentiable rewards, so we fall back to the binary certificate there. We chose *per-transition binary gating plus continuous reweighting* over pure filtering because pure filtering discards near-boundary transitions that still carry signal, accepting the cost of a tunable weight floor w_min. We added the *empirical coverage calibration* (Phase 1b) because the contract is a hypothesis, not a guarantee: the calibration set measures how often live outcomes actually fall inside the declared partition and inflates B_t by the observed cross-class residual, so the certificate becomes 'contract class plus empirically estimated residual' and the method abstains on tools whose contracts are too incomplete to certify. A domain expert would not describe CONCERT as a variant of Trajectory2Gym: Trajectory2Gym decides *which structures to synthesize* and *how to weight interactions*, but never checks whether a synthesized observation is in the same contract class as the live one; CONCERT's load-bearing object is the equivalence class and the bias bound, not synthesis allocation.

**Why it measures what we claim.** The quantity `B_t` measures the *policy-bias a synthesized transition can inject* because the gradient-discrepancy over an equivalence class upper-bounds the change in the SFT objective when the simulated observation is swapped for any other observation in the same class, with the measured cross-class residual R_k added to account for contract incompleteness; the named assumption is that the live outcome lies in the simulated outcome's class, which holds when the contract's declared clauses are complete and the calibration coverage cov_k >= tau_cov; this assumption fails when a tool has undocumented side effects (e.g., a silently created resource), in which case B_t reflects only the documented-class bias plus the measured residual and under-counts the true discrepancy, and the tool is marked ABSTAIN if coverage falls below tau_cov. The quantity `cls_sim in A` measures *contract admissibility* because the admissible-class set A is computed from the argument schema's precondition branches, which is equivalent to the set of outcomes the live API can return for those arguments only under the assumption that the contract's precondition clauses are sound; when a contract is over-specified (declares a precondition the API does not enforce), A is too small and valid transitions are wrongly dropped. The quantity `w_t = 1 - B_t/delta` measures *how much trust a transition deserves* because the linear map from bias bound to weight is monotone in B_t, which is equivalent to a trust score only under the assumption that bias scales linearly with gradient discrepancy; when the loss is highly non-convex near s_t, the linear map misranks transitions and w_t reflects local curvature rather than global trust. The binary certificate `1[B_t <= delta]` measures *provable indistinguishability* because delta is a fixed tolerance on the gradient discrepancy, which is equivalent to indistinguishability of the learned policy under the two outcomes only under the assumption that the SFT objective is Lipschitz in the observation with constant 1; when the observation enters through a discontinuous feature (e.g., a categorical flag), the certificate can pass while the policy output changes discretely, in which case the certificate reflects a smoothed surrogate rather than true indistinguishability.

## Contribution

(1) CONCERT, a per-transition certificate mechanism that derives tool-semantics equivalence classes from declared I/O contracts and gates synthesized transitions by a decision-theoretic gradient-discrepancy bound computed entirely offline, without executing the live tool. (2) A design principle showing that contract-derived class membership — not simulation likelihood — is the correct admissibility criterion for offline tool-use supervision, together with the boundary conditions (contract completeness, loss differentiability) under which the certificate is valid. (3) A certified offline training pipeline that composes binary gating with continuous reweighting and parent-normalized interaction weighting, yielding a reusable recipe for filtering synthesized tool-use data.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Tau1 + BFCL synthesized interactions, with a 512-call live calibration split per tool | Compositional tool calls with declared I/O contracts and live ground truth |
| Primary metric | Success rate on live tool execution | Directly tests whether certified transitions transfer |
| Secondary metric | Certificate failure rate under injected undocumented side effects | Measures impact of contract incompleteness |
| Secondary metric | Empirical vs. actual gradient discrepancy (B_t vs. swap discrepancy) | Validates the bias bound and Lipschitz assumption |
| Baseline | Trajectory2Gym SFT | Tests synthesis allocation without contract certification |
| Baseline | DreamGym-style learned outcome predictor | Tests learned confidence vs contract-derived equivalence |
| Baseline | Unfiltered SFT on synthesized trajectories | Tests whether certification itself carries the signal |
| Baseline | Schema-validation-only filter (no bias bound) | Tests whether standard contract checking suffices |
| Ablation-of-ours | CONCERT with binary gating only (no reweighting) | Isolates contribution of continuous weight floor |
| Ablation-of-ours | CONCERT without Phase 1b calibration (R_k = 0, no abstain) | Isolates contribution of empirical coverage calibration |

### Why this setup validates the claim

The central claim is that contract-derived equivalence classes plus a gradient-discrepancy bias bound yield a training set whose simulated observations are policy-indistinguishable from live outcomes, so a policy trained on it transfers to live tools. The dataset choice matters because Tau1 and BFCL contain tools with genuinely declared schemas and error clauses, which is exactly the regime where the contract-derived partition is defined; on tools with no contracts the method abstains, so a dataset with rich contracts is the fair test. The primary metric, live success rate, is the right detector because the entire mechanism is built to prevent simulation-induced policy bias, and such bias manifests only as live-execution failure, not as offline loss. The secondary metrics directly test the load-bearing assumption: we compare B_t to the actual policy-gradient discrepancy observed when swapping simulated and live outcomes on the calibration set, and we inject undocumented side effects to measure how often the certificate fails and how much the coverage calibration recovers. Each baseline isolates one sub-claim: unfiltered SFT tests whether any certification is needed, Trajectory2Gym tests whether synthesis allocation alone suffices without checking outcome-class membership, the learned-predictor baseline tests whether a learned confidence score can substitute for the contract-derived bound, and the schema-validation-only baseline tests whether standard contract checking suffices without the bias bound. The binary-gating ablation tests whether the continuous reweighting contributes beyond filtering, and the no-calibration ablation tests whether empirical coverage calibration is load-bearing. Together these form a falsifiable lattice: if certification is load-bearing, gains must concentrate where simulated observations drift out of contract class, not everywhere.

### Expected outcome and causal chain

**vs. Trajectory2Gym SFT** — On a case where a synthesized trajectory composes two tools and the second tool's simulated return omits a field that the live API includes only on success, Trajectory2Gym retains it because it allocates synthesis to compositional structure but never checks outcome-class membership, so the policy learns to condition on a class the live tool never produces and fails at composition time. Our method classifies the simulated observation into the contract's return-schema variants, finds it admissible but with a large gradient discrepancy against the success variant, and either drops or down-weights it, so we expect a noticeable gap on multi-tool compositional subsets and near-parity on single-tool subsets.

**vs. DreamGym-style learned outcome predictor** — On a case where a tool has an undocumented but consistent error path, the learned predictor assigns high confidence because the synthesized observation resembles its training distribution, yet the live outcome lands in a different contract class, so the baseline admits a biased transition on the strength of a score that conflates likelihood with policy bias. Our method instead bounds the gradient discrepancy over the declared class and, where the class is complete, certifies the transition; where the class is incomplete, the calibration set measures the cross-class residual R_k and inflates B_t, and if coverage falls below tau_cov the tool is marked ABSTAIN, so we expect the baseline's advantage to vanish specifically on tools with documented error clauses and to persist only on contract-poor tools, giving us a gap concentrated on the contract-rich subset.

**vs. Unfiltered SFT on synthesized trajectories** — On a case where the simulator returns a plausible but out-of-contract observation, such as a success payload for arguments that violate a declared precondition, unfiltered SFT trains the policy to emit those arguments and to expect success, so live execution returns the declared 4xx and the episode fails. Our method computes the admissible-class set from the precondition branches and drops the transition before it enters the loss, so we expect unfiltered SFT to show a large live-failure rate on precondition-violating subsets while CONCERT is near the ceiling there, with the two converging on subsets where all synthesized observations happen to be admissible.

**vs. Schema-validation-only filter** — On a case where a simulated observation passes the return schema but lands in a different error class than the live outcome (e.g., a retryable 5xx simulated as a terminal 4xx), schema validation accepts it because the payload is well-typed, while our bias bound B_t detects the large gradient discrepancy between the two classes and drops or down-weights it. We expect the schema-only baseline to match CONCERT on well-typed but class-mismatched transitions only when the gradient discrepancy is small, and to fall behind on subsets where class membership changes the policy gradient, demonstrating that the bias bound is not a repackaging of schema validation.

### What would falsify this idea

If CONCERT's gain over Trajectory2Gym and unfiltered SFT were uniform across all subsets rather than concentrated on transitions whose simulated observations sit near a contract-class boundary, then the certification would not be doing the causal work and the result would be attributable to generic data cleaning. Likewise, if the binary-gating ablation matched full CONCERT everywhere, the continuous reweighting would be shown inert; and if the no-calibration ablation matched full CONCERT even on tools with injected undocumented side effects, the empirical coverage calibration would be shown unnecessary.

## References

1. Trajectory2Gym: Decision-Grounded Compositional Interaction Synthesis for Tool-Use Agents
2. Scaling Agent Learning via Experience Synthesis
3. ReTool: Reinforcement Learning for Strategic Tool Use in LLMs
4. Is Your LLM Secretly a World Model of the Internet? Model-Based Planning for Web Agents
5. WebRL: Training LLM Web Agents via Self-Evolving Online Curriculum Reinforcement Learning
