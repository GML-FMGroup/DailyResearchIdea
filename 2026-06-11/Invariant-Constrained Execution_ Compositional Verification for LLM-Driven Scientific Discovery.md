# Invariant-Constrained Execution: Compositional Verification for LLM-Driven Scientific Discovery

## Motivation

Existing methods like Popper rely on LLMs to generate correct implications and experiments without formal guarantees across interdependent steps, leading to cumulative errors that single-step verification cannot detect. The root cause is that each LLM output is verified independently, ignoring domain invariants (e.g., conservation laws, dimensional consistency) that couple outputs across steps. We address this by enforcing these invariants as compositional constraints.

## Key Insight

Scientific invariants provide a closed-form consistency condition across LLM outputs that is independent of the LLM's reasoning path, enabling formal multi-step coherence guarantees that single-step verifiers cannot achieve.

## Method

(A) **What it is**: We introduce the **Invariant-Constrained Execution Framework (ICEF)**. ICEF takes as input a sequence of LLM-generated scientific outputs (hypothesis, experimental procedure, data interpretation) and a set of domain-specific invariants (e.g., conservation laws, unit consistency, logical consistency rules). It outputs a verified sequence that satisfies all invariants, or a rejection with diagnostics.

(B) **How it works**:
```
Input: LLM-generated outputs O = {o_1, o_2, ..., o_T} (each step's output)
       Invariant set I = {i_1, i_2, ..., i_K} (each invariant is a predicate over a subset of outputs)
       Max search depth D = 10 (hyperparameter, adaptive backtracking until fixed point or D exceeded)
Output: Verified sequence O' or rejection signal

1. For each invariant i_k in I:
   - Identify relevant outputs O_k ⊆ O (can span multiple steps)
2. For each invariant i_k:
   - Check constraint i_k(O_k). If satisfied, continue.
   - If violated, attempt local repair:
       a. Identify the earliest output in O_k that can be adjusted without affecting later steps (use dependency graph D).
       b. Query LLM to regenerate that output with the constraint i_k as a prompt suffix.
       c. Re-check i_k. If still violated, backtrack: revert to previous step and regenerate with additional context (up to D times).
   - If after D attempts no fixed point is reached (no new violations after repair), reject sequence.
3. Return the repaired sequence O'.
```
Hyperparameters: Max search depth D=10, dependency graph D constructed from step order and variable references.

(C) **Why this design**: We chose a constraint-check-and-repair loop over single-pass verification because (1) repair leverages the LLM's generative ability to correct errors when given explicit constraints, avoiding the need for a separate correction model; this accepts the cost of increased latency from multiple LLM calls. (2) We prioritize repairing the earliest violating output to minimize cascading conflicts, accepting that earlier outputs may be harder to change due to downstream dependencies. (3) We use adaptive backtracking (max depth D=10) over fixed backtracking because scientific invariants may require variable repair depth; the trade-off is increased inference cost but better success on deep interactions. (4) The dependency graph is built from explicit step ordering and variable names rather than learned, which is simple but requires manual specification; a learned dependency would be more flexible but risk spurious edges. This design avoids the anti-pattern of a learned controller by using deterministic constraint satisfaction: the decision to backtrack is based on a formal invariant violation, not a learned gating signal.

(D) **Why it measures what we claim**: The invariant compliance rate measures compositional consistency directly because each invariant is a formal predicate that must hold across multiple outputs; this assumes the invariant set is correct and complete for the domain, which fails when an important invariant is missing, in which case the measure reflects only the subset of relationships encoded. The backtrack depth D operationalizes effort-bounded verifiability because it caps the number of repair attempts while still providing a probabilistic guarantee of success if the LLM can correct errors with high probability given a constraint; this assumption fails when errors require multi-step reasoning beyond D iterations, in which case the method rejects a potentially correctable sequence. The dependency graph D measures causal coupling between outputs because it encodes which outputs directly affect others based on step order and variable references; this assumes a linear flow of information, which fails when outputs have nonlinear dependencies (e.g., cyclic reasoning), in which case the graph may be incomplete.

## Contribution

(1) A framework (ICEF) that enforces domain-specific scientific invariants as compositional constraints for LLM output verification in multi-step workflows. (2) A design principle: using formal invariants from scientific domains provides a closed-form verification signal that is more reliable than learned verifiers because invariants are independent of LLM calibration. (3) A repair strategy that combines local constraint violation detection with LLM regeneration, guided by a dependency graph and backtracking.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Compositional Science Benchmark: 500 multi-step problems from physics (conservation laws), chemistry (stoichiometry), biology (gene regulation), each with 3-5 steps and 2-3 invariants | Multi-step scientific outputs with invariants. |
| Primary metric | Invariant compliance rate (fraction of problems where all invariants hold after verification) | Directly measures compositional consistency. |
| Baseline 1 | No verification: raw GPT-4 without constraint checking | Shows baseline generation quality. |
| Baseline 2 | Single-pass verification: GPT-4 with constraint checking but no repair, reject on violation | Isolates repair effect. |
| Baseline 3 | Full constraint propagation: systematic backtracking with no depth limit, using a SAT solver for invariants | Benchmark for optimal repair. |
| Ablation-of-ours | ICEF w/o dependency graph: repair uses random output selection instead of earliest | Tests impact of causal ordering. |

### Why this setup validates the claim
This experimental design tests the central claim that ICEF improves compositional consistency via constraint-check-and-repair. The dataset provides multi-step scientific outputs with known invariants, enabling direct measurement of consistency. The no-verification baseline establishes the LLM's intrinsic error rate; single-pass verification tests whether detection alone suffices; full constraint propagation offers an upper bound. The ablation removes the dependency graph to assess its contribution to targeted repair. The primary metric—invariant compliance rate—directly captures the key outcome, ensuring that any improvement is attributable to the method's core mechanism. If ICEF outperforms no-verification and single-pass, and approaches full propagation, the claim is supported; if not, the design falsifies it by showing no benefit from the repair loop.

### Expected outcome and causal chain

**vs. No verification (raw LLM)** — On a case where a generated hypothesis violates a conservation law (e.g., energy non-conservation in a physics derivation), the raw LLM produces the invalid sequence because it lacks any consistency enforcement. Our method instead detects the violation via invariant i_k, identifies the earliest relevant output (hypothesis), and queries the LLM to regenerate it with the conservation constraint, so we expect a substantial gap in compliance rate (e.g., ICEF >80% vs raw <30%) on invariants that are explicit but often missed by the LLM.

**vs. Single-pass verification (no repair)** — On a case where a data interpretation step misaligns units (e.g., meters vs inches), the baseline rejects the entire sequence after detecting the invariant violation, yielding zero correct completions for that instance. Our method instead repairs the offending output step by step (e.g., correcting the unit conversion in the data interpretation) while keeping earlier valid outputs, so we expect single-pass to have lower coverage (many rejected but correctable sequences) while ICEF accepts them, leading to a higher overall compliance rate (e.g., ICEF 80% vs single-pass 50%).

**vs. Full constraint propagation (exhaustive)** — On a case with multiple interacting invariants (e.g., both unit consistency and logical progression), full propagation solves all conflicts using a SAT solver with exponential time, while ICEF with adaptive backtracking (max depth D=10) may succeed on most but might hit depth limit on extremely entangled violations, producing a rejection where full propagation would succeed. Thus we expect full propagation to achieve perfect compliance (100% on solvable instances), while ICEF achieves high but slightly lower compliance (e.g., 95%) due to occasional failures on deeply entangled interactions—exactly the trade-off predicted by the adaptive backtracking design.

**vs. ICEF w/o dependency graph** — On a case where a later output depends on an earlier one (e.g., data interpretation references hypothesis variable), random repair order may fix a later output first, causing cascading conflicts. Our method's dependency graph prioritizes the earliest violating output, reducing such conflicts. We expect ICEF with dependency graph to achieve higher compliance (e.g., 80% vs 60% without graph) on problems with strong causal ordering.

### What would falsify this idea
If ICEF's compliance rate is not significantly higher than single-pass verification across all invariant types, or if the improvement is uniform across easy and hard cases rather than concentrated on instances where invariant violations are correctable via few repairs, then the central claim that the repair loop effectively leverages the LLM's generative ability fails.

## References

1. Automated Hypothesis Validation with Agentic Sequential Falsifications
