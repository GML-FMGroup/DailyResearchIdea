# Script-Perturbation Identifiability for Collaboration Benchmarks

## Motivation

Collaboration benchmarks such as RealCLI measure collaboration quality against a fixed, pre-authored user script, so the resulting score entangles the agent's collaboration skill with the idiosyncrasies of that one artifact; because the script is held fixed and the agent is observed only once per task, no amount of rubric refinement, learned disentanglement heads, or verifier stacking can separate agent-attributable from script-attributable variance. The same structural confound recurs in SWE-bench, where a fixed test suite stands in for the spec, so the measured signal is never identifiable independent of the artifact. This is a meta-level identifiability failure, not a local limitation of any single benchmark.

## Key Insight

If the same task is instantiated under a family of counterfactually perturbed user scripts that are provably behaviorally equivalent for a script-following agent, then collaboration ability is exactly the outcome component that is invariant across the family, and the script-attributable component is the residual that varies with the script — turning the artifact from a confound into an instrument.

## Method

## SCRIPT-PIN: Script-Perturbation Identifiability for Collaboration Benchmarks

**(A) What it is.** SCRIPT-PIN takes an existing scripted collaboration task (a task spec plus its pre-authored user script from RealCLI-COLLAB) and a target agent, and outputs a decomposition of the agent's outcome into a *collaboration-invariant* component and a *script-attributable* residual. It does this by running the agent on a family of K counterfactually perturbed scripts that are constructed to satisfy an **agent-conditional behavioral-invariance constraint** (defined below) for the target agent, then reading the variance structure across the family. The load-bearing assumption, stated plainly: *the perturbation family spans the surface-realization space to which the target agent is sensitive, and within that family the agent's intermediate decisions are invariant.* This assumption is not asserted — it is tested per task by the agent-conditional invariance gate, and the fraction of the script space for which it holds is reported as a first-class output (the *coverage diagnostic*).

**(B) How it works.**

```pseudocode
INPUT: task T, base user script S0, agent A, K perturbations,
       reference oracle O (deterministic script-follower),
       agent-conditional invariance checker E_A, R repeat samples per script

# Phase 1: build the perturbation family (surface-only modes)
modes = [reorder_nonblocking_turns, paraphrase_same_content,
         relocate_clarification_turn_index, substitute_equivalent_authorization]
family = []
for k in 1..K:
    Sk = perturb(S0, mode=modes[k % len(modes)])
    # Gate 1 (oracle sanity): reject perturbations that change information
    # content, blocking structure, or authorization decisions.
    assert E_O(S0, Sk) == EQUIVALENT   # oracle-trace identity
    family.append(Sk)

# Phase 2: run the agent R times per script (K+1 scripts, R repeats)
# R=3 by default; used to separate surface sensitivity from run-to-run noise.
outcomes = {}   # outcomes[k][r]
for Sk in [S0] + family:
    for r in 1..R:
        trace = run_agent(A, T, user=Sk)   # existing RealCLI-COLLAB harness
        outcomes[Sk][r] = (score(trace), decisions(trace))
        # decisions(trace) = ordered list of (tool_call, info_acquired,
        #                    blocking_decision) tuples extracted from trace

# Phase 3: agent-conditional invariance gate E_A (per task)
# For each perturbed script Sk, test whether the agent's own intermediate
# decisions are invariant to Sk up to run-to-run stochasticity.
invariant_family = []
for Sk in family:
    # decision-identity: compare decisions(trace) across the R repeats of
    # S0 and the R repeats of Sk; a script passes iff the multiset of
    # decision tuples for Sk is contained in the S0 decision distribution
    # at tolerance tau (default tau = 0.0 exact-match on tool_call and
    # blocking_decision; info_acquired compared as set equality).
    if E_A(decisions(S0), decisions(Sk), tau) == INVARIANT:
        invariant_family.append(Sk)

coverage = len(invariant_family) / K   # first-class output: fraction of
                                       # script space where agent-invariance holds

# Phase 4: decompose the outcome over the invariant sub-family only
scripts_used = [S0] + invariant_family
mu       = mean over scripts_used and repeats of score
var_total = var over scripts_used of (mean over repeats of score)
collab_invariant = mu          # component stable across the invariant family
script_residual  = var_total   # component that moves with the script
# per-dimension decomposition of the RealCLI five-dimensional rubric:
for dim d in five_dimensional_framework:
    collab_invariant[d] = mean_d over scripts_used and repeats
    script_residual[d]  = var_d over scripts_used of (mean over repeats)

OUTPUT: (collab_invariant, script_residual) per dimension and overall,
        plus coverage (fraction of K perturbations passing E_A)
```

The **agent-conditional invariance gate `E_A`** is the load-bearing component: a perturbation enters the decomposition only if the *target agent's own* intermediate decisions (tool calls, information acquired, blocking decisions) are invariant to it across R=3 repeats, at tolerance tau=0.0 exact-match on tool calls and blocking decisions and set-equality on acquired information. The oracle gate `E_O` is retained as a *sanity filter* (it rejects perturbations that change information content, blocking structure, or authorization decisions) but is no longer treated as sufficient: `E_O` certifies equivalence for the reference policy only, and LLM agents are documented to be surface-sensitive (Zhao et al. 2021; Lu et al. 2022; Sclar et al. 2024), so the equivalence relation is made a property of the agent's policy class via `E_A`. The **coverage diagnostic** `coverage = |invariant_family| / K` quantifies how much of the perturbation family the agent is actually invariant to; the decomposition is reported only over the sub-family where `E_A` holds, and `coverage` is reported alongside every decomposition so a reader can see how narrow the agent's equivalence class is.

**(C) Why this design.** We chose *counterfactual script perturbation* over a learned disentanglement head because the script and the agent are operationally coupled in a single rollout, so no learned independence assumption can manufacture the separation the data does not contain — the coupling is a property of the observation design, not of the estimator. We chose an *agent-conditional invariance gate* over an oracle-only equivalence gate because trace-identity under one deterministic policy does not imply invariance of a different policy's behavior: LLM agents are demonstrably sensitive to semantically irrelevant surface form (Zhao et al. 2021 report up to ~30-point swings from demonstration-order permutation; Sclar et al. 2024 report up to 76-point drops from formatting-only changes), so the oracle-equivalence class is not the agent's equivalence class and the gate must be evaluated on the target agent's own decisions. We chose *variance across the invariant family* as the residual estimator over a single-run residual because a single run cannot distinguish agent noise from script sensitivity; the cost is K+1 scripts × R repeats per task, roughly a K·R-fold compute increase, and the coverage diagnostic makes the narrowness of the surviving family visible rather than hidden. A domain expert would not describe SCRIPT-PIN as a variant of learned disentanglement (e.g., the independence-head approach in the disentanglement literature) because SCRIPT-PIN never estimates a joint distribution over agent and script — it *intervenes on the script* and reads the invariance structure, which is an experimental-design operation, not a statistical-independence assumption.

**(D) Why it measures what we claim.** The quantity `collab_invariant = mean(outcomes)` measures *agent-attributable collaboration ability* because the agent-conditional gate `E_A` restricts the family to perturbations under which the target agent's own intermediate decisions are invariant across R=3 repeats, so within the surviving family variation in outcomes can only come from the agent's departure from script-following; this assumption fails when the agent is surface-sensitive in a way that survives `E_A` at tau=0.0 (e.g., a paraphrase cue that changes a tool call only on some repeats, which the repeat-based gate would catch, or a cue that changes a decision deterministically and is therefore excluded from the family, which shrinks coverage), in which case the mean reflects a mixture of true collaboration skill and residual script sensitivity and the reported coverage flags how much of the script space was excluded. The quantity `script_residual = var(outcomes)` measures *script-attributable variance* because under `E_A` the only source of cross-script outcome variation within the invariant family is the agent's differential response to surface realization; this assumption fails when the perturbation family is too narrow to span the script space (low coverage), in which case the residual under-estimates the true script-attributable component and the coverage diagnostic quantifies the shortfall. The per-dimension decomposition `mean_d` and `var_d` measure *dimension-specific collaboration skill and dimension-specific script sensitivity* because the RealCLI five-dimensional rubric scores each dimension independently; this assumption fails when rubric dimensions are correlated in the scoring function, in which case per-dimension residuals leak across dimensions.

**(E) Scope.** SCRIPT-PIN applies to any scripted interactive benchmark with a deterministic script-following oracle; it requires K+1 scripts × R=3 repeats per task, a perturbation generator whose outputs pass the oracle sanity gate `E_O`, and an agent-conditional invariance check `E_A` that compares the target agent's intermediate decisions across repeats. The coverage diagnostic `coverage = |invariant_family| / K` is reported with every decomposition and is the first-class measure of how much of the script space the agent's equivalence class spans.

## Contribution

(1) A script-perturbation identifiability framework (SCRIPT-PIN) that decomposes collaboration-benchmark outcomes into an agent-attributable invariant and a script-attributable residual, with a behavioral-equivalence gate that makes the decomposition well-posed. (2) A design principle: the pre-authored script becomes an instrument rather than a confound when the observation design includes a counterfactual family, so identifiability is achieved by intervention rather than by learned independence. (3) A perturbation-family construction recipe layered on top of existing RealCLI-COLLAB infrastructure, requiring no new task authoring.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | RealCLI-COLLAB scripted tasks (full set + 20-task pilot subset) | Provides pre-authored user scripts plus five-dim rubric |
| Primary metric | Script-residual variance share vs. invariant mean, reported with coverage | Directly reads the decomposition the method outputs |
| Baseline | RealCLI single-run collab score | Tests whether variance decomposition adds signal |
| Baseline | Unconstrained-paraphrase family (no E_O gate) | Tests whether oracle sanity gate is load-bearing |
| Baseline | Oracle-only equivalence gate (E_O, no E_A) | Tests whether agent-conditional gate is load-bearing |
| Baseline | Agent-identity shuffle null | Tests whether residual reflects script not agent noise |
| Ablation-of-ours | SCRIPT-PIN without agent-conditional gate E_A (oracle gate only) | Isolates contribution of the agent-conditional constraint |
| Ablation-of-ours | SCRIPT-PIN with R=1 (no repeat-based noise separation) | Isolates contribution of repeat sampling to E_A |
| Grounding check | Synthetic agent with injected script sensitivity (known ground truth) | Verifies estimator recovers injected components within tolerance |
| Transfer check | SWE-bench-style fixed-test-suite task subset (perturb test-suite surface form) | Tests identifiability framing beyond RealCLI-COLLAB |

### Why this setup validates the claim

The central claim is that SCRIPT-PIN separates agent-attributable collaboration ability from script-attributable variance, and that this separation is real rather than an artifact of averaging or of unconstrained perturbation. RealCLI-COLLAB is the right dataset because it already pairs a task spec with a pre-authored deterministic user script and a five-dimensional rubric, so the oracle sanity gate E_O has a concrete oracle to check against and the decomposition lands on interpretable dimensions. The primary metric — the share of outcome variance attributable to the script residual versus the invariant mean, reported alongside the coverage diagnostic — is exactly the quantity the method outputs, so a baseline that reports only a single-run collaboration score cannot express it; the comparison therefore tests whether the decomposition reveals structure the aggregate hides. The unconstrained-paraphrase baseline tests the sub-claim that the oracle sanity gate is load-bearing: if removing E_O produces the same residual pattern, the gate is doing nothing. The oracle-only baseline (E_O without E_A) is the decisive control for the adversarial concern: if the oracle-only gate produces the same residual structure as the full agent-conditional gate, then the agent-conditional gate is not load-bearing and the claimed separation is an artifact of the oracle-equivalence class rather than of the agent's own invariance. The agent-identity shuffle null tests the sub-claim that the residual tracks script sensitivity rather than run-to-run agent noise: if shuffling which agent produced which outcome leaves the residual unchanged, the residual is noise. The R=1 ablation isolates the contribution of repeat sampling to E_A. The synthetic grounding check verifies the estimator recovers injected script sensitivity within a stated tolerance before applying it to real agents. The SWE-bench-style transfer check tests whether the identifiability framing generalizes beyond a single benchmark. Together these form a falsifiable test: the method must show a residual structure that (i) the single-run baseline cannot see, (ii) the unconstrained baseline distorts, (iii) the oracle-only baseline does not reproduce, and (iv) the shuffle null fails to reproduce, while (v) the synthetic grounding check recovers known components within tolerance and (vi) coverage is reported and non-trivial.

### Expected outcome and causal chain

**vs. RealCLI single-run collab score** — On a case where the agent succeeds under the base script S0 but the rubric credits several clarification turns, the single-run baseline produces one aggregate score that folds script idiosyncrasy into apparent collaboration skill, because it has no counterfactual family to separate the two. Our method instead runs the K perturbed scripts with R=3 repeats and reads the variance over the agent-invariant sub-family: the invariant mean stays near the base outcome while the residual exposes how much of the score moved with surface realization, and coverage reports how much of the family survived E_A. We expect a noticeable divergence between invariant mean and single-run score on tasks whose scripts contain many non-blocking turns, but near-parity on tasks with short, blocking-heavy scripts where there is little surface variance to exploit.

**vs. Unconstrained-paraphrase family (no E_O gate)** — On a case where an unconstrained paraphrase accidentally changes the information content of a clarification (e.g., drops a constraint), the no-gate baseline produces a large residual that conflates script idiosyncrasy with genuine task-difficulty change, because it never checks behavioral equivalence against the oracle. Our method instead rejects that perturbation at the E_O gate and keeps the family narrow, so its residual reflects only surface sensitivity. We expect the no-gate baseline to show inflated residual variance concentrated on tasks with dense clarification content, while SCRIPT-PIN's residual stays bounded and its invariant mean tracks the base-script outcome closely.

**vs. Oracle-only equivalence gate (E_O, no E_A)** — On a case where the agent is surface-sensitive to a paraphrase that the oracle treats as equivalent (e.g., a reordering that changes the agent's tool-call order), the oracle-only baseline admits that perturbation into the family and attributes the resulting outcome swing to script-attributable residual, because it never checks the agent's own decisions. Our method instead rejects that perturbation at E_A and reports a lower coverage, so its residual reflects only perturbations the agent is actually invariant to. We expect the oracle-only baseline to show inflated residual and coverage near 1.0 (it admits everything E_O passes), while SCRIPT-PIN's coverage is lower and its residual is bounded; the gap between the two residuals is the size of the agent-attributable surface sensitivity that the oracle-only gate mislabels as script-attributable.

**vs. Agent-identity shuffle null** — On a case where two agents with different base-script outcomes are pooled, the shuffle null produces a residual of similar magnitude to the real decomposition, because pooling across agents reintroduces agent-driven variance that mimics script sensitivity. Our method instead computes the residual within a single agent across the invariant family, so the shuffle should collapse the residual toward the run-to-run noise floor. We expect a clear separation: real SCRIPT-PIN residuals noticeably exceed the shuffle null on tasks with high script surface variability, but converge to the null on tasks with minimal surface variability.

**vs. Synthetic agent with injected script sensitivity (grounding check)** — On a synthetic agent constructed with a known script-sensitivity parameter (e.g., a policy that flips a blocking decision with probability p when a clarification is relocated), the estimator should recover the injected invariant component and residual within a stated tolerance (e.g., ±10% relative error on the residual, ±5% on the invariant mean) across 50 synthetic tasks. If it does not, the estimator is not grounded and the real-agent results are uninterpretable.

**vs. SWE-bench-style fixed-test-suite transfer check** — On a subset of SWE-bench-style tasks where the test suite is perturbed in surface form (reordering tests, renaming test functions, reformatting assertions) while preserving pass/fail semantics for a reference solution, the same decomposition should yield a coverage diagnostic and residual that track the agent's sensitivity to test-suite surface form. We expect coverage to be high (test suites are more rigid than user scripts) and residual to be small, demonstrating the identifiability framing generalizes rather than being a RealCLI-COLLAB artifact.

### What would falsify this idea

If the oracle-only baseline (E_O without E_A) produces the same residual structure and the same invariant-mean ordering as the full agent-conditional method, then the agent-conditional gate is not load-bearing and the claimed causal separation is an artifact of the oracle-equivalence class rather than of the agent's own invariance. Equivalently, if the real residual is indistinguishable from the agent-identity shuffle null across all task subsets rather than concentrated where surface variability is high, or if the synthetic grounding check fails to recover injected components within tolerance, the residual is measuring agent noise or the estimator is ungrounded, not script sensitivity.

## References

1. RealCLI: Toward a Unified Evaluation of Human-Agent Collaboration for CLI Agents
2. SWE-bench: Can Language Models Resolve Real-World GitHub Issues?
