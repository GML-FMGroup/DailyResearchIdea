# Monotone Support-Graph Revision for Probe-Driven Task Repair

## Motivation

WorkFactory's probe-driven repair treats the information support graph as freely editable: when a probe reveals a missing support, the repair operator can delete or rewire dependencies in ways that invalidate tasks whose executability was already verified, so the executable task closure is not preserved and teacher trajectories may be collected on silently broken tasks. This is a structural instance of a meta-level failure recurring across agent data-synthesis pipelines — R2E-Gym and WorkFactory both treat environment support as a fixed input rather than a dynamically revised object carrying an invariant — so the field currently has no notion of a repair that is *safe* with respect to previously established executability.

## Key Insight

Every legitimate repair is a refinement on the support-graph's dependency partial order — it may only add support edges or replace a failed node by a strictly more specific supported one — so executability is monotone under refinement and can be maintained by incrementally checking only the newly added edges against the already-verified closure.

## Method

**What it is.** We introduce *RefineRepair*, a probe-driven repair operator for WorkFactory that replaces free-form graph edits with a monotone revision operator over the information support graph. Its inputs are the current support graph G=(V,E), the verified executable task set C (a closure), and a probe failure report; its output is a refined graph G' together with a machine-checkable refinement certificate and an updated closure C' ⊇ C.

**Load-bearing assumption (stated explicitly).** Task executability is monotone under support-graph refinement — adding a positive support edge or specializing a node to a lattice descendant cannot invalidate an already-verified task — *provided* the support graph is annotated with edge polarity and the incremental check covers the transitive resource-conflict neighborhood of changed nodes. This assumption fails under negative dependencies (outputs that must NOT exist, tools that must be absent) and under non-local shared mutable resources; the method therefore restricts the monotone filter and incremental check to the positive-local subgraph (defined below) and reports coverage as a measured quantity.

**How it works.**

```pseudocode
INPUT: support graph G=(V,E), verified closure C, probe failure (t, missing support s)
       edge polarity annotation pol: E -> {+, -}  (static classifier, see below)
       task locality label loc: V -> {local, non-local}  (static classifier)
OUTPUT: refined G'=(V',E'), certificate cert, closure C' with C ⊆ C'

0. SCOPE RESTRICTION
   E+_local = { e in E : pol(e)=+ and loc(head(e))=local and loc(tail(e))=local }
   E-_all   = { e in E : pol(e)=- }   (negative edges anywhere)
   resource_claims = { (r, task) : task claims shared resource r }  (files, ports, packages)

1. CANDIDATE REPAIRS  R = { add_support_edge(s -> t) with polarity pol(s->t),
                          specialize(failed_node v -> v') for v' in subnodes(v) }

2. MONOTONE FILTER    keep r in R iff r is a refinement of E under polarity:
                        - add_support_edge with pol=+: allowed iff E+_local ∪ {e}
                          is a refinement (E+_local ⊆ E+_local ∪ {e}) AND e does not
                          conflict with any negative precondition in E-_all or any
                          resource_claim in the transitive resource-conflict
                          neighborhood of t
                        - add_support_edge with pol=-: allowed only if it does not
                          remove any existing positive support path for a task in C
                        - specialize: allowed only if v' is a strict descendant of v
                          in the type/subnode lattice (depth ≤3), so support set
                          shrinks (stronger), AND v' does not introduce a new
                          negative precondition or resource claim conflicting with C

3. CERTIFICATE        cert = (old_edge_set, new_edge_set, refinement_witness,
                               conflict_freedom_witness)
                        refinement_witness = for each changed node the lattice step v -> v'
                        conflict_freedom_witness = for each added edge, the list of
                          negative preconditions and resource claims checked, with
                          the resource-conflict neighborhood radius used (default 2 hops)

4. INCREMENTAL CHECK  C' = C;  frontier = {t} ∪ {u in C : u in resource-conflict
                        neighborhood of any changed node (radius 2)}
                        for each task u in frontier reachable from t under E':
                          if u not in C: verify_executable(u, G'); if ok: C'.add(u)
                        assert every u in C remains executable: re-check only edges
                          incident to changed nodes AND edges in the resource-conflict
                          neighborhood of changed nodes (|changed| + |neighborhood| << |E|)

5. REJECT if any u in C fails re-check -> discard repair, fall back to next candidate

6. RETURN G', cert, C'
```

Hyperparameters: refinement lattice depth ≤3 for `specialize`; incremental re-check budget = edges incident to changed nodes plus edges within 2-hop resource-conflict neighborhood; polarity classifier threshold = 0.5 on a logistic regression over edge text features (tool name, argument type, precondition keyword); locality classifier = rule-based: a task is `non-local` if it writes to any path under `/tmp`, `/usr`, or a shared package directory, or binds a port, else `local`.

**Why this design.** We chose a partial-order refinement constraint over WorkFactory's unconstrained graph edit because the failure mode is precisely that arbitrary edits shrink the executable closure, accepting the cost that some repairs which would fix a probe failure are rejected as non-monotone — this is deliberate, since a rejected repair leaves a task uninstantiated rather than silently corrupting verified ones. We chose a *certificate* (old/new edge sets plus a lattice-step witness per changed node plus a conflict-freedom witness) over a boolean 'monotone' flag because the certificate is independently checkable without re-running probes, accepting extra bookkeeping to make audit possible. We chose incremental closure verification restricted to edges incident to changed nodes and their 2-hop resource-conflict neighborhood over full re-verification of C because full re-verification is O(|C|) probe executions per repair and would dominate cost, accepting the risk that a non-local dependency outside the 2-hop neighborhood invalidates a distant task — which step 5 catches by re-checking the frontier reachable from t and the resource-conflict neighborhood. We chose lattice-based `specialize` over free node substitution because it makes 'strictly more specific' a decidable relation, accepting that the type/subnode lattice must be authored per occupation domain. A domain expert would not describe this as a variant of R2E-Gym's environment-repair loop, because R2E-Gym's repairs are unconstrained environment mutations with no order-theoretic invariant and no closure-preservation certificate.

**Why it measures what we claim.** The refinement relation `E+_local ⊆ E'+_local` (with specialize as lattice-step and conflict-freedom against negative preconditions and resource claims) measures *monotone preservation of executability* because the assumption is that a task's executability depends only on the existence of its positive support edges and the absence of conflicting negative preconditions, so adding positive edges or specializing a node cannot remove a support path and cannot introduce a conflict; this assumption fails when a task's executor has a negative dependency not captured by the polarity classifier or a shared resource outside the 2-hop neighborhood, in which case refinement can break it and the certificate reflects only positive-local monotonicity. The incremental closure check `C' ⊇ C` measures *closure preservation* because the assumption is that any task whose incident edges and 2-hop resource-conflict neighborhood are unchanged remains executable when only those tasks are re-probed; this fails when support is non-local beyond 2 hops (a task depends on a shared resource mutated by a distant task), in which case C' reflects local re-verification rather than global closure. The refinement certificate's `refinement_witness` and `conflict_freedom_witness` measure *auditability of the repair* because the assumption is that the lattice step v→v' and the checked conflict list are the only semantic changes, so a checker can replay them; this fails when a repair bundles multiple lattice steps or when the conflict list omits a resource claim, in which case the witness reflects a composed or incomplete step rather than a single one.

## Contribution

(1) RefineRepair, a monotone support-graph revision operator with a machine-checkable refinement certificate that constrains probe-driven repair to order-preserving edits and guarantees the verified executable closure is non-decreasing. (2) A design principle: incremental closure verification restricted to edges incident to changed nodes suffices to preserve executability under refinement, with a frontier re-check that catches non-local violations. (3) A certificate format and checker that make agent-data-synthesis repairs auditable independent of re-running probes.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | WorkFactory 112,686-task corpus, stratified sample of 40 occupations (≈7,400 tasks) | Matches the domain where support graphs are authored; sample size chosen to make lattice-authoring effort (≈2 hours per occupation, 80 hours total) and inter-annotator agreement (Cohen's κ on lattice edges, target ≥0.7) measurable within 3–6 months |
| Primary metric | Executable-closure retention rate under probe failures | Directly measures monotone preservation claim |
| Secondary metric | Fraction of tasks eligible for the positive-local guarantee (coverage) | Makes the guarantee's scope a measured quantity rather than a caveat |
| Secondary metric | Audit cost (probe executions per repair) and fraction of repairs surviving independent certificate checking | Isolates certificate value |
| Baseline | WorkFactory unconstrained graph-edit repair | Tests whether order-theoretic invariant matters |
| Baseline | R2E-Gym-style environment mutation loop | Tests whether closure certification matters |
| Baseline | No-repair / task-drop fallback | Tests whether repair beats abstention |
| Ablation-of-ours | RefineRepair minus refinement certificate | Isolates auditability from repair efficacy |
| Ablation-of-ours | RefineRepair minus polarity/locality classifier (treat all edges as positive-local) | Measures how often the load-bearing assumption fires (F) |
| Injection experiment | Inject non-local shared-resource dependencies (e.g., two tasks writing to the same `/tmp` file) into 10% of sampled tasks | Measures how often the 2-hop neighborhood misses a distant conflict |

### Why this setup validates the claim

The central claim is that constraining repairs to a partial-order refinement over the support graph preserves the verified executable closure, whereas unconstrained editing silently corrupts it. To falsify this, we need a setting where repairs are actually triggered and where corruption is observable. WorkFactory's 112,686 tasks across 612 occupations provide exactly that: probe failures are common enough to force many repair decisions, and the closure is large enough that corruption is measurable as a rate rather than an anecdote. The primary metric — executable-closure retention under probe failures — is the right detector because the claim is precisely about closure preservation, not raw task yield; a method that fixes probes but drops verified tasks must be penalized, and this metric does that. The unconstrained-edit baseline tests the order-theoretic sub-claim: if refinement constraints are doing real work, that baseline should show closure erosion on the same failures. The R2E-Gym-style baseline tests the certificate sub-claim: without an order invariant, its repairs should be harder to audit and more likely to admit non-monotone edits. The no-repair baseline tests whether repair beats abstention at all, guarding against the trivial outcome that rejecting everything is fine. The certificate ablation isolates whether the auditability mechanism is load-bearing or merely bookkeeping — if removing it changes nothing, the design choice was unjustified. The polarity/locality ablation and the non-local injection experiment directly test the load-bearing assumption: if the classifier is accurate and the 2-hop neighborhood catches conflicts, the ablation should match full RefineRepair; if not, the gap quantifies the assumption's failure rate.

### Expected outcome and causal chain

**vs. WorkFactory unconstrained graph-edit repair** — On a case where a probe fails because a support edge is missing and the unconstrained editor also rewrites an adjacent node to a broader type, the baseline produces a locally fixed probe but a shrunken closure, because it has no order invariant preventing support-set expansion. Our method instead admits only refinement steps (added edges or lattice-descendant specialization) that pass the conflict-freedom check, so the closure cannot shrink, and we expect a noticeable retention gap concentrated on failures where the baseline's edit touches nodes shared by many verified tasks, with parity on isolated single-edge failures. We also expect the baseline to show a measurable closure-erosion rate (target: quantify as a percentage of repairs on the 40-occupation sample) to establish failure prevalence.

**vs. R2E-Gym-style environment mutation loop** — On a case where a probe failure requires specializing a node and the mutation loop substitutes an arbitrary compatible node, the baseline produces a repair that happens to pass the probe but bundles several semantic changes with no replayable witness, so a later audit cannot tell whether the closure was preserved. Our method instead emits a lattice-step witness per changed node plus a conflict-freedom witness, making the repair independently checkable, and we expect the gap to appear as a higher fraction of auditable repairs and fewer downstream re-probe failures, not as a raw yield advantage.

**vs. No-repair / task-drop fallback** — On a case where a probe failure is repairable by adding a single support edge, the fallback produces an uninstantiated task and leaves the closure unchanged but smaller than achievable. Our method instead adds the edge and incrementally re-verifies only incident tasks and their 2-hop resource-conflict neighborhood, so we expect a clear task-yield advantage over abstention on repairable failures, with parity on failures our monotone filter rejects — which is the intended trade.

**vs. RefineRepair minus refinement certificate (ablation)** — On a case where a repair is later contested, the ablated variant produces the same graph edit but no replayable witness, so a checker must re-run probes to decide monotonicity. Our method instead returns a certificate whose old/new edge sets, lattice steps, and conflict-freedom list can be checked without probes, and we expect the ablation to match our closure retention but lose on audit cost (probe executions per repair) and on the fraction of repairs that survive independent checking.

**vs. RefineRepair minus polarity/locality classifier (ablation)** — On a case where a repair adds a positive edge that conflicts with a negative precondition, the ablated variant treats the edge as positive-local and admits the repair, silently breaking a verified task. Our method instead rejects the repair via the conflict-freedom check, and we expect the ablation to show a higher rate of closure erosion on the injected non-local subset, quantifying how often the load-bearing assumption fires.

**Injection experiment (non-local shared-resource dependencies)** — On 10% of sampled tasks we inject a shared `/tmp` file write that creates a conflict between two tasks more than 2 hops apart in the support graph. We expect the 2-hop neighborhood to miss some fraction of these conflicts, and we report that fraction as the empirical failure rate of the locality assumption, with the full RefineRepair (2-hop) versus a 4-hop variant to show sensitivity.

### What would falsify this idea

If refinement-constrained repair shows uniform closure-retention gains across all failure subsets rather than being concentrated on failures where unconstrained edits touch shared or negatively-depended nodes, the order-theoretic invariant is not the causal mechanism. Likewise, if the certificate ablation matches full RefineRepair on auditability and downstream re-probe failures, the certificate is bookkeeping rather than a load-bearing design choice. If the polarity/locality ablation matches full RefineRepair on the injected non-local subset, the load-bearing assumption is not doing work and the monotone-closure claim must be dropped.

## References

1. WorkFactory: Turning Professional Intents into Executable Work for Scalable Agent Training
