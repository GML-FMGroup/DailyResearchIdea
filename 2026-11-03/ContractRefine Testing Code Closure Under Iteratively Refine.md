# ContractRefine: Testing Code Closure Under Iteratively Refined Cross-Repository Interface Contracts

## Motivation

CrossSWE freezes inter-repository API contracts at instance creation, so an agent's generated code is only ever tested against a single static snapshot of the interface it must satisfy; the benchmark therefore never measures whether generated code remains correct when the contract itself evolves, which is the normal condition of real cross-repository development. This closure-under-refinement gap is structural, not incidental: because the contract is frozen, CrossSWE's 44.2% cross-repository-inconsistency failure mode is observed but never stress-tested against the versioned interface changes that actually produce such inconsistencies in practice, and the same static-context assumption is shared by SWE-Exp, Fea-Bench, and the continuous-learning branch, so all of them inherit the untested assumption that repository interfaces are static.

## Key Insight

A generated cross-repository implementation is genuinely interface-closed if and only if it can be re-derived to satisfy a contract that has been perturbed along a semantically-preserving refinement direction while its inter-repository behavioral obligations are held fixed, so closure is measurable as the stability of pass/fail outcomes across a contract-refinement trajectory rather than as a single static pass.

## Method

**What it is.** ContractRefine is a benchmark-construction and evaluation procedure that wraps each CrossSWE instance in an iteratively refined inter-repository API contract. Its inputs are a CrossSWE instance (repositories R_A, R_B, initial contract C_0, tests T) and an agent; its outputs are a refinement trajectory {C_0, C_1, ..., C_k} plus, for each step, a closure verdict (whether the agent's regenerated code passes T under C_i) and a contract-drift attribution.

**Load-bearing assumption (stated explicitly).** CLOSURE_SCORE is interpretable as interface closure only under assumption A: that each operator's equivalence witness, validated on the reference implementation, implies behavioral equivalence for *arbitrary agent code* on the refined edge — i.e., that T is behaviorally complete for that edge. A fails when T under-covers the edge (the equivalent-mutant problem: Jia & Harman, IEEE TSE 2011), in which case a pass-to-fail transition may reflect a coverage gap rather than a closure failure. We therefore (i) strengthen the witness check with a differential-equivalence oracle over held-out inputs disjoint from T, (ii) audit T's coverage per operator, and (iii) report CLOSURE_SCORE only on operators whose witness survives the stronger oracle, with a per-operator coverage bound.

**How it works.**

```pseudocode
INPUT: instance (R_A, R_B, C_0, T), agent A, refinement budget k=4
OUTPUT: trajectory {(C_i, code_i, pass_i, attribution_i, cov_i)}

0. WITNESS CALIBRATION (per operator, run once before the trajectory):
   For each operator op in the menu:
     a. Apply op to R_A's interface with its witness, producing C_op and
        witness-rewritten reference implementation ref_op.
     b. DIFFERENTIAL-EQUIVALENCE ORACLE: generate N_fuzz = 2000 property-based
        inputs per refined edge from a generator disjoint from T (Hypothesis-style:
        type-directed random values, boundary values, and 10% adversarial
        edge cases per parameter type). Run ref_op and ref_0 on all N_fuzz inputs;
        op passes the oracle iff outputs agree on 100% of inputs (bitwise for
        scalars, structural equality for containers).
     c. TEST-SUITE ADEQUACY: mutate ref_op with 20 mutation-testing operators
        (arithmetic, relational, boundary, negation) restricted to the refined
        edge's body; compute edge-coverage adequacy = (mutants killed by T) /
        (mutants killed by T ∪ N_fuzz). op passes adequacy iff adequacy ≥ 0.90.
     d. op is ADMITTED to the menu iff it passes both (b) and (c); otherwise
        it is DROPPED and logged with its oracle/adequacy scores.
   The admitted menu is fixed for all instances; the scheduler cycles only
   over admitted operators.

1. PARSE C_0 into a typed interface graph G_0:
     nodes = exported symbols in R_A consumed by R_B
     edges = (symbol, signature, semantic obligation) where obligation
             is the behavioral assertion in T that exercises the symbol

2. For each refinement step i in 1..k:
   a. SELECT the refinement operator op_i from the ADMITTED menu applied to
      exactly one edge e in G_{i-1}, chosen by a deterministic scheduler
      that cycles: RENAME_PARAM -> ADD_OPTIONAL_PARAM -> WIDEN_TYPE
      -> SPLIT_FUNCTION -> REORDER_ARGS (skipping dropped operators). Each
      operator carries a machine-checkable equivalence witness (e.g., for
      RENAME_PARAM, a bijection on parameter positions; for WIDEN_TYPE, a
      subtyping certificate; for SPLIT_FUNCTION, a call-site rewrite rule).
   b. APPLY op_i to R_A's interface, producing C_i, and mechanically
      rewrite all R_A-internal call sites and R_B call sites via the
      witness, so the *behavioral* obligation in T is preserved by
      construction. VERIFY via the step-0 oracle: run the N_fuzz inputs on
      the witness-rewritten reference implementation under C_i; if any
      disagreement with C_0, op_i is rejected and the scheduler advances.
      Also record cov_i = edge-coverage adequacy of T under C_i (same
      mutation procedure as step 0c) for the refined edge.
   c. PROMPT agent A with (R_A at C_i, R_B at C_{i-1}) and the task
      description, requesting coordinated edits; collect code_i.
   d. RUN T against code_i; record pass_i in {0,1} and, on failure,
      attribute the failure to a node in G_i by diffing the failing
      assertion's exercised symbol against the changed edge e.

3. CLOSURE_SCORE = (1/|S|) * sum_{i in S} pass_i, where S = {i : cov_i ≥ 0.90}
   (fraction of refinement steps at which regenerated code remains correct,
   restricted to steps whose refined edge is adequately covered by T).
   Report |S|/k as the COVERAGE-YIELD of the trajectory.
4. DRIFT SCORE = fraction of failures attributed to the refined edge e
   rather than to unrelated nodes (isolates contract-induced breakage).
5. Report (CLOSURE_SCORE, DRIFT_SCORE, COVERAGE-YIELD) per agent, plus a
   per-operator breakdown over the admitted operators, with each operator's
   oracle-agreement and adequacy scores from step 0.
```

Hyperparameters fixed inline: refinement budget k=4; operator menu of 5 (subject to step-0 admission); scheduler is deterministic round-robin (not learned); equivalence witness checked by the differential-equivalence oracle (N_fuzz=2000 held-out inputs per edge) plus mutation-testing adequacy (20 mutants per edge, threshold 0.90) before each step; rejection rate and per-operator oracle/adequacy scores logged.

**Why this design.** We chose a fixed, mechanically-witnessed operator menu over free-form LLM-generated contract edits because an unwitnessed edit cannot distinguish "the agent's code broke" from "the contract edit itself broke behavior," accepting the cost that our refinements cover only five syntactic-semantic perturbation classes rather than the full space of real interface evolution. We chose a deterministic round-robin scheduler over a difficulty-adaptive one because adaptivity would confound closure with scheduler policy, accepting the cost that some refinement steps may be trivially easy for strong agents. We chose to regenerate code from scratch at each step rather than incrementally patch the previous step's code because incremental patching would let the agent amortize reasoning across steps and hide closure failures, accepting the cost that we measure a harder, less realistic workflow. We chose failure attribution by symbol-diff against the changed edge over LLM-judged attribution because the latter is unverifiable, accepting the cost that attribution is coarse when a single edge is exercised by multiple assertions. We chose the step-0 differential-equivalence oracle and adequacy audit over a single T-run because the single T-run is exactly the coverage condition under which the equivalent-mutant problem bites, accepting the cost that operators failing the oracle are dropped from the menu and the trajectory may cover fewer than five classes. A domain expert would not describe this as a variant of CrossSWE: CrossSWE fixes the contract and varies the task, whereas ContractRefine fixes the behavioral obligation and varies the contract, so the measured quantity (stability of correctness under witnessed interface refinement) does not exist in CrossSWE's evaluation surface.

**Why this measures what we claim.** CLOSURE_SCORE measures interface closure of generated code because the equivalence-witness assumption makes each C_i behaviorally equivalent to C_0 for the reference implementation, so any pass-to-fail transition must originate in the agent's code rather than in the contract edit; this assumption fails when an operator's witness is too weak to preserve behavior for edge cases not covered by T, in which case CLOSURE_SCORE reflects test-suite coverage gaps rather than closure — which is why CLOSURE_SCORE is restricted to steps with cov_i ≥ 0.90 and COVERAGE-YIELD is reported alongside it. DRIFT_SCORE measures contract-induced breakage because attributing a failure to the refined edge via symbol-diff assumes the failing assertion exercises exactly the refined symbol; this assumption fails when an assertion exercises multiple symbols and the true cause is an unrefined node, in which case DRIFT_SCORE reflects attribution coarseness rather than causal locality. The per-operator breakdown measures robustness to refinement *class* because each operator is a distinct, named perturbation with its own witness type; this assumption fails when two operators interact across steps (e.g., a rename followed by a split), in which case the breakdown reflects operator composition rather than isolated operator difficulty.

## Contribution

(1) A benchmark-construction procedure and evaluation protocol that turns any CrossSWE-style instance into a contract-refinement trajectory using five mechanically-witnessed interface operators, yielding two new metrics (CLOSURE_SCORE and DRIFT_SCORE) that quantify code stability under evolving cross-repository contracts. (2) A design finding that closure under witnessed interface refinement is separable from static cross-repository correctness, because the operator menu isolates contract-induced breakage from task-intrinsic difficulty. (3) An open trajectory corpus of refinement steps with per-operator closure and drift labels, enabling diagnosis of which interface-change classes agents fail to propagate across repository boundaries.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | CrossSWE instances with witnessed contracts | Fixed behavioral obligation lets contract vary, not task. |
| Primary metric | CLOSURE_SCORE across refinement steps (restricted to cov_i ≥ 0.90) | Directly measures correctness stability under interface refinement. |
| Secondary metric | COVERAGE-YIELD (|S|/k) and per-operator adequacy | Separates closure signal from test-suite coverage gaps. |
| Baseline | GPT-6.1 Sol (CrossSWE best) | Strongest fixed-contract agent establishes ceiling. |
| Baseline | Static-contract agent (C_0 only) | Isolates refinement-induced breakage from base competence. |
| Baseline | Free-form LLM contract editor | Tests whether witness discipline, not editing, drives gains. |
| Ablation-of-ours | ContractRefine without DRIFT attribution | Tests whether attribution, not closure, carries the signal. |
| Ablation-of-ours | ContractRefine with single T-run witness (no oracle/adequacy) | Tests whether the stronger witness check changes CLOSURE_SCORE. |

### Why this setup validates the claim

The central claim is that correctness stability under witnessed interface refinement is a measurable, agent-distinguishing quantity that CrossSWE's fixed-contract surface cannot expose. The static-contract baseline tests the sub-claim that refinement actually stresses agents beyond C_0: if CLOSURE_SCORE for static-contract equals ContractRefine's, refinement adds no signal. GPT-6.1 Sol tests the sub-claim that the quantity is not trivially saturated by a strong model: if Sol closes every step, the benchmark is too easy; if it drops specifically on witnessed operators, the surface is real. The free-form LLM contract editor tests the sub-claim that mechanical witnesses, not the act of editing, are what make failures attributable — without witnesses, pass-to-fail transitions cannot be localized. The ablation without DRIFT attribution tests whether DRIFT_SCORE carries independent information or merely mirrors CLOSURE_SCORE. The ablation with the single T-run witness tests whether the step-0 differential-equivalence oracle and adequacy audit materially change CLOSURE_SCORE: if the two scores diverge, the original single-T-run witness was admitting coverage-gap noise, which is exactly the load-bearing assumption A made explicit in the method. CLOSURE_SCORE is the right primary metric because the equivalence-witness construction makes each C_i behaviorally equivalent to C_0 for the reference implementation, so any pass-to-fail transition must originate in agent code; a metric of raw Pass@1 at a single contract would not detect the predicted effect because it has no within-instance variation to attribute. COVERAGE-YIELD and per-operator adequacy are reported alongside CLOSURE_SCORE so the reader can see how many refinement steps survive the coverage bound and whether the closure gap is concentrated on operators whose T-coverage is weakest.

### Expected outcome and causal chain

**vs. GPT-6.1 Sol (CrossSWE best)** — On a case where the refined operator is SPLIT_FUNCTION and the agent must propagate a call-site rewrite across R_A and R_B, the baseline produces a locally correct edit that misses one internal R_A call site because its fixed-contract training never required tracking a symbol through a signature-preserving split. Our method instead forces the agent to regenerate against C_i with the split already applied to R_A's interface, so the missing call site surfaces as a compile-or-test failure attributable to the split edge. We expect Sol to show high CLOSURE_SCORE on RENAME_PARAM and ADD_OPTIONAL_PARAM but a noticeable drop concentrated on SPLIT_FUNCTION and REORDER_ARGS, with parity on the easy operators; we expect COVERAGE-YIELD to be near 1.0 for the three mechanically trivial operators and lower for SPLIT_FUNCTION and WIDEN_TYPE, and we report CLOSURE_SCORE only on the steps that survive the cov_i ≥ 0.90 bound.

**vs. Static-contract agent (C_0 only)** — On a case where the agent's code passes T at C_0 but the contract is refined by WIDEN_TYPE, the baseline produces code that never sees the widened signature and therefore cannot exhibit a closure failure, so its CLOSURE_SCORE is pinned near 1.0 by construction. Our method instead re-prompts at each C_i, so the same agent must re-derive correctness under a changed obligation and its failures become visible. We expect the static baseline to look artificially perfect while ContractRefine reveals a substantial closure gap, with the gap widening as refinement steps accumulate; the gap is reported only on steps with cov_i ≥ 0.90, and COVERAGE-YIELD quantifies how many steps enter that comparison.

**vs. Free-form LLM contract editor** — On a case where the editor rewrites an interface in a way that silently breaks an edge-case behavior not covered by T, the baseline produces a pass-to-fail transition that cannot be attributed because no equivalence witness exists, so its DRIFT_SCORE is near-uniform across nodes. Our method instead rejects the edit at witness-check time (differential-equivalence oracle plus adequacy audit) or attributes the failure to the refined edge via symbol-diff, so DRIFT_SCORE concentrates on the changed edge. We expect the free-form editor to show comparable CLOSURE_SCORE but a markedly flatter DRIFT_SCORE distribution, indicating unattributable breakage; we also expect the free-form editor to have no COVERAGE-YIELD analogue because its edits are not certified against held-out inputs.

**vs. ContractRefine without DRIFT attribution** — On a case where the agent's code fails T at C_i, the ablation records pass_i but cannot localize the failure to the refined edge, so its per-operator breakdown is flat by construction. Our method attributes the failure via symbol-diff and reports DRIFT_SCORE, so the per-operator breakdown concentrates on the refined edge. We expect the ablation to match CLOSURE_SCORE but to lack the per-operator concentration, confirming that DRIFT_SCORE carries independent information.

**vs. ContractRefine with single T-run witness** — On a case where an operator's witness is behaviorally equivalent on T but not on held-out inputs (the equivalent-mutant case), the single-T-run ablation admits the operator and records a pass-to-fail transition that is actually a coverage gap, inflating the apparent closure gap. Our method's step-0 oracle and adequacy audit either drop the operator or restrict CLOSURE_SCORE to steps with cov_i ≥ 0.90, so the reported closure gap is not confounded by coverage. We expect the single-T-run ablation to show a larger apparent closure gap than the full method, with the difference concentrated on WIDEN_TYPE and ADD_OPTIONAL_PARAM, the operators whose observable interface contract changes most.

### What would falsify this idea

If CLOSURE_SCORE for ContractRefine is statistically indistinguishable from the static-contract baseline across all admitted operators, or if the free-form editor's DRIFT_SCORE is as concentrated on the refined edge as ours, then the witnessed-refinement surface is not measuring interface closure and the central claim is wrong. A uniform gain across operators rather than a concentration on SPLIT_FUNCTION and REORDER_ARGS would also falsify the predicted causal chain, as would a COVERAGE-YIELD near zero (no operator survives the step-0 oracle and adequacy audit), which would indicate the refinement menu cannot be certified as behavior-preserving under the stronger witness check.

## References

1. CrossSWE: Benchmarking Cross-Repository Joint Development for Coding Agents
