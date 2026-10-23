# Conflict-Induced Dominance Learning for Calibrated Constraint Relaxation in LLM Agents

## Motivation

Every agentic instruction-following benchmark in the Idea Cards — AGENTIF, CFBench, FollowBench, SysBench, CELLO, COLLIE, RuLES — reduces constraint satisfaction to an unweighted conjunction of independently checkable predicates, so when constraints are mutually unsatisfiable (e.g., 'be concise' vs. 'include all requested fields') the benchmarks can only report a violation count, never which constraint a rational annotator would have relaxed. This is the shared meta-gap: importance and dominance are latent in the conflict resolutions that annotators and environments implicitly encode, but no existing method extracts them, because the benchmarks never force the agent to commit to a relaxation and therefore never expose the signal. AGENTIF's 11.9 constraints per instruction are precisely the regime where these conflicts are dense, yet its evaluation protocol treats each constraint as a binary check with no priority structure.

## Key Insight

A constraint's dominance weight is identifiable from the agent's own executed relaxation choices if and only if those choices are elicited under a forced-conflict intervention that holds the constraint set fixed and varies only which constraint must yield — making the pairwise preference a revealed-preference signal rather than a human label.

## Method

**What it is.** We introduce CIDL (Conflict-Induced Dominance Learning), which takes as input an annotated constraint set C = {c_1..c_n} (in the AGENTIF/CFBench style) and a task environment E, and outputs a weighted partial order (a dominance DAG with edge weights w_ij ∈ [0,1] and node importances) over C. The DAG is then used at inference to select a maximal satisfiable relaxation when C is unsatisfiable.

**Load-bearing assumption (stated explicitly).** The constraint the executor violates under the twin-task forced-conflict intervention is a stable, order-invariant readout of the executor's implicit priority over constraints, so that a consistent violation pattern across prompt orderings can be interpreted as a revealed dominance edge rather than as an artifact of prompt position, phrasing, or the executor's inability to satisfy the constraint at all. Phase 1 below makes this assumption conditional on demonstrated feasibility and order-invariance via two controls: a capability probe (each constraint alone) and an order-permuted null pair (no real conflict).

**How it works.**

```pseudocode
INPUT: constraint set C, environment E, agent policy pi, annotator/verifier V
OUTPUT: dominance DAG G=(C,E_dom,w), node importance alpha

PHASE 0 — Capability probe (feasibility control)
  for each c_i in C:
     run pi on E with ONLY c_i active (single-constraint task)
     V labels satisfied/violated
     if violated: mark c_i INFEASIBLE for pi on E; exclude c_i from all dominance edges
  (This separates 'dominated' from 'infeasible for this policy'.)

PHASE 1 — Forced-conflict elicitation (build the preference signal)
  for each pair (c_i, c_j) that the executor finds jointly unsatisfiable on E
     AND both c_i, c_j are FEASIBLE per Phase 0:
     construct two twin tasks T_ij and T_ji that are identical except
     the prompt orders c_i and c_j first (order is the intervention)
     run pi on both; V labels which constraint was violated in each
     if pi violates c_j in T_ij AND violates c_i in T_ji:
         record a genuine conflict (order-invariant)
     if pi violates the SAME constraint in both:
         record a revealed dominance edge c_other > c_violated
         with weight = 1 (deterministic) or a soft count over K=5 reruns
     else (violation pattern is order-sensitive):
         mark pair UNRANKED (positional bias suspected)
  # Order-permuted null pairs (position-bias control):
  for each pair (c_i, c_j) with NO real conflict (both jointly satisfiable):
     construct twin tasks T_ij and T_ji as above
     run pi on both; V labels violations
     if pi violates the SAME constraint in both orderings:
         record a position-bias count for that constraint (used to down-weight
         any dominance edge whose violated constraint matches the bias pattern)

PHASE 2 — Fit the partial order
  build a directed graph over C from revealed edges (excluding INFEASIBLE nodes
  and edges down-weighted by the null-pair position-bias count)
  run a topological sort with cycle-breaking by minimum feedback-arc-set
  (exact ILP for |C|<=20, greedy otherwise)
  fit node importance alpha by Bradley-Terry MLE over all pairwise edges:
     P(c_i > c_j) = sigmoid(alpha_i - alpha_j)
  edge weights w_ij = empirical win rate of c_i over c_j on E

PHASE 3 — Calibrated relaxation at inference
  when C is unsatisfiable on a new task:
     find maximal satisfiable subset S* maximizing sum_{c in S} alpha_c
     subject to S being a down-set of the dominance DAG
     solve by greedy peeling in topological order (O(n log n))
     return S* and the relaxed set C \ S*
```

Hyperparameters: K=5 reruns per twin for weight stability; cycle-breaking via exact ILP up to |C|=20; Bradley-Terry fit with L2 regularization λ=1e-2; the down-set constraint is what makes the relaxation 'rational' rather than arbitrary. Phase 0 capability probe and Phase 1 null pairs add 2n + 2|pairs| extra runs per instruction (n≈12, so ~24 + ~132 runs per instruction for full pairwise coverage).

**Why this design.** We chose forced-conflict twin elicitation over asking the annotator for importance labels because AGENTIF and CFBench show that annotators cannot reliably rank 11.9 constraints by importance without seeing a conflict, accepting the cost that we only recover dominance for pairs the executor actually makes conflict on (unexercised pairs remain unranked, which is why we output a partial order rather than a total order). We chose a Bradley-Terry fit over a learned neural scorer because with n≈12 constraints per instruction the pairwise signal is sparse and a parametric model with n parameters is identifiable where a neural head is not, accepting the cost that we cannot generalize dominance across constraint types without a shared embedding (we add a constraint-type prior to partially mitigate this). We chose greedy topological peeling over ILP for the inference-time relaxation because the down-set constraint makes the greedy choice optimal for the maximal-weight down-set problem on a DAG, accepting the cost that if the DAG has cycles from noisy edges the greedy result may be locally suboptimal — which is why Phase 2 breaks cycles before Phase 3. This is not a variant of retrieval-based tool selection (Toolformer/Gorilla) because the dominance edges are computed by intervening on the constraint ordering and observing which constraint the executor violates — a counterfactual that retrieval cannot produce — and it is not a log-prob gating scheme because the signal is a revealed-preference count, not a model confidence.

**Why this measures what we claim.** The empirical win rate w_ij measures the dominance of c_i over c_j because the twin-task design holds the constraint content and environment fixed and varies only the prompt ordering, so any order-invariant violation pattern is attributable to the executor's implicit priority; this assumption fails when the executor's behavior is order-sensitive for reasons unrelated to priority (e.g., recency bias in long prompts), in which case w_ij reflects positional bias rather than dominance — which is why Phase 1 marks order-sensitive pairs UNRANKED and Phase 0 excludes constraints the executor cannot satisfy alone. The Bradley-Terry importance alpha_i measures the rational annotator's endorsed relaxation because the MLE is consistent for the true ranking under the assumption that pairwise preferences are transitive and generated by a single latent scale; this assumption fails when constraints are incommensurable (e.g., a style constraint vs. a safety constraint), in which case alpha_i collapses two incomparable dimensions onto one axis and the relaxation may violate a constraint the annotator would have protected. The down-set constraint in Phase 3 measures 'rational relaxation' because a down-set of the dominance DAG is exactly the set of constraints closed under 'dominates', so relaxing a constraint never relaxes something it dominates; this assumption fails when the DAG is a forest with disconnected components, in which case any union of down-sets is a valid relaxation and the choice among them is underdetermined by dominance alone.

## Contribution

['A method (CIDL) that recovers a weighted partial order over agentic constraints from forced-conflict twin-task interventions rather than human importance labels, producing a dominance DAG with Bradley-Terry-calibrated node importances.', "A design principle: maximal satisfiable relaxation under a down-set constraint of the dominance DAG, which makes the agent's relaxation choice rational by construction rather than by post-hoc scoring.", 'A procedure for cycle-breaking and partial-order completion that handles the sparse, order-sensitive preference signal that pairwise conflict elicitation produces on AGENTIF-style 11.9-constraint instructions.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | AGENTIF + CFBench conflict subsets | Real unsat constraint sets with annotated priorities |
| Primary metric | Relaxation agreement with annotator-endorsed maximal subset | Directly tests rational relaxation, not raw task success |
| Baseline | Order-insensitive log-prob priority scorer | Tests whether counterfactual elicitation beats confidence gating |
| Baseline | Annotator importance ranking (no conflict) | Tests whether forced-conflict signal beats static labels |
| Baseline | Random / uniform-priority down-set relaxation | Tests whether revealed dominance structure matters at all |
| Baseline | Single-task 'ask which constraint you would drop' probe | Tests whether ordering intervention specifically beats direct elicitation |
| Baseline | Paraphrase-based intervention (vary phrasing, not order) | Tests whether order intervention specifically beats phrasing intervention |
| Ablation-of-ours | CIDL without down-set constraint (ILP max-weight subset) | Isolates contribution of the dominance-closure constraint |
| Ablation-of-ours | CIDL without Phase 0 capability probe | Isolates contribution of feasibility control |
| Ablation-of-ours | CIDL without Phase 1 null-pair position-bias control | Isolates contribution of order-invariance control |
| Diagnostic | Per-pair order-sensitivity rate (fraction of pairs marked UNRANKED) | Measures how often the load-bearing assumption is violated |

### Why this setup validates the claim

The central claim is that dominance recovered by intervening on constraint ordering yields relaxations that match what a rational annotator would endorse, and that this signal is not obtainable from confidence scores or static importance labels. The log-prob baseline tests the sub-claim that counterfactual order intervention beats model-confidence gating: if CIDL's gain vanishes against it, the intervention adds nothing over reading the executor's own probabilities. The static-importance baseline tests the sub-claim that revealed-preference-under-conflict beats asking annotators to rank constraints, which AGENTIF/CFBench suggest is unreliable. The random-down-set baseline is the floor: if CIDL does not beat it, the DAG structure is noise. The single-task 'ask which constraint you would drop' baseline tests whether the ordering intervention specifically, rather than elicitation generally, produces the dominance signal: if a direct probe matches CIDL, the twin-task design is unnecessary. The paraphrase-based intervention baseline tests whether varying constraint phrasing rather than order produces the same signal: if paraphrase matches CIDL, the order intervention is not the active ingredient. The ablation isolates the single most load-bearing design choice — the down-set constraint — so that any gain can be attributed to rational closure rather than mere weight maximization. The Phase 0 and Phase 1 control ablations isolate the feasibility and order-invariance controls that make the load-bearing assumption conditional rather than assumed. The per-pair order-sensitivity diagnostic directly measures the fraction of pairs where the assumption fails, so the method's validity is reported rather than assumed. Relaxation agreement with the annotator-endorsed maximal satisfiable subset is the right primary metric because it directly operationalizes 'rational relaxation' and is sensitive to exactly the failure modes (positional bias, incommensurable constraints, disconnected components) the method's assumptions flag.

### Expected outcome and causal chain

**vs. Order-insensitive log-prob priority scorer** — On a case where the executor's token-level confidence is high on a style constraint but its actual behavior sacrifices that constraint when a safety constraint conflicts, the log-prob scorer ranks the style constraint high and relaxes the safety one, because confidence reflects fluency rather than revealed priority. Our method instead forces a twin-task conflict, observes that the executor consistently drops the style constraint regardless of prompt order, and records a dominance edge safety > style. We expect a clear gap concentrated on the conflict subset where confidence and revealed violation disagree, with parity on non-conflicting constraints.

**vs. Annotator importance ranking (no conflict)** — On a case with ~12 constraints where two are near-tied in annotator-assigned importance, the static ranking produces an arbitrary tie-break and relaxes whichever the annotator happened to label lower, because annotators cannot reliably rank many constraints without a concrete conflict. Our method instead elicits the preference only for pairs the executor actually makes conflict on, recovering a partial order that is silent on unexercised pairs rather than guessing. We expect the gain to concentrate on high-constraint-count instances and on near-tie pairs, with near-parity on instances with few constraints.

**vs. Random / uniform-priority down-set relaxation** — On a case where the dominance DAG has a clear chain (safety > format > style), the random down-set relaxation picks among all valid down-sets uniformly and frequently relaxes a constraint that dominates others. Our method instead maximizes sum of Bradley-Terry importances over down-sets, so it relaxes the topological bottom first. We expect a large gap on instances with deep dominance chains and a smaller gap on shallow or disconnected DAGs, which is the predicted signature.

**vs. Single-task 'ask which constraint you would drop' probe** — On a case where the executor's self-report is miscalibrated (it says it would drop a style constraint but actually drops a safety constraint under conflict), the direct probe returns the self-report and relaxes the wrong constraint. Our method instead observes the executed violation under the twin-task intervention, so the dominance edge reflects behavior rather than stated preference. We expect a gap concentrated on pairs where self-report and executed violation disagree, with parity where they agree.

**vs. Paraphrase-based intervention (vary phrasing, not order)** — On a case where paraphrasing a constraint changes its surface form but not its priority, the paraphrase intervention produces no consistent violation pattern and records noisy or no dominance edges. Our method instead holds content fixed and varies only order, so any order-invariant violation pattern is attributable to priority. We expect CIDL to recover more edges and higher agreement than paraphrase, with the gap largest on pairs where paraphrase changes fluency but not semantics.

**vs. CIDL without down-set constraint** — On a case where the max-weight subset violates closure (relaxing a dominating constraint while keeping a dominated one), the ablation returns an incoherent relaxation that the annotator would reject. Our method's down-set constraint forbids this by construction. We expect the ablation to match CIDL on simple chains but diverge sharply on branching DAGs, isolating the closure contribution.

**vs. CIDL without Phase 0 capability probe** — On a case where a constraint is individually infeasible for the executor (capability failure), the no-probe variant records a dominance edge against it because it is violated in both orderings, conflating 'dominated' with 'infeasible'. Our method excludes infeasible constraints from all edges. We expect the no-probe variant to over-relax infeasible constraints and show lower agreement on instances containing such constraints, with parity elsewhere.

**vs. CIDL without Phase 1 null-pair position-bias control** — On a case where recency bias causes the executor to violate the same constraint in both orderings regardless of priority, the no-control variant records a spurious dominance edge. Our method down-weights edges matching the null-pair bias pattern. We expect the no-control variant to show lower agreement on instances with high per-pair order-sensitivity, with parity on low-sensitivity instances.

### What would falsify this idea

If CIDL's gain over the log-prob, static-importance, single-task-probe, and paraphrase baselines is uniform across all subsets rather than concentrated on the predicted conflict, near-tie, deep-chain, self-report-disagreement, and high-order-sensitivity subsets, then the mechanism is not doing what we claim and the improvement is an artifact. Equally fatal: if the per-pair order-sensitivity rate is high (most pairs marked UNRANKED) or the Phase 0 capability probe shows most constraints are individually infeasible, then the revealed-preference signal is too sparse to support a partial order and the method is uninformative.

## References

1. AGENTIF: Benchmarking Instruction Following of Large Language Models in Agentic Scenarios
2. SysBench: Can Large Language Models Follow System Messages?
3. CFBench: A Comprehensive Constraints-Following Benchmark for LLMs
4. Instruction-Following Evaluation for Large Language Models
5. FollowBench: A Multi-level Fine-grained Constraints Following Benchmark for Large Language Models
6. Can LLMs Follow Simple Rules?
7. Can Large Language Models Understand Real-World Complex Instructions?
8. COLLIE: Systematic Construction of Constrained Text Generation Tasks
