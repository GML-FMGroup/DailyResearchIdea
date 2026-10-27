# Failure-Triggered Action Substitution with Openness-Preserving Closure for Spreadsheet Agents

## Motivation

Existing spreadsheet agents like SheetCopilot commit to a fixed atomic action set, and even broader frameworks such as Work Agents define capability families without any rule for expanding the action set when an atomic action fails; as a result, high-openness spreadsheet work — where the workspace admits operations not enumerated at design time — cannot recover from execution errors because the agent has no licensed way to substitute a failed action with an alternative or a composition of primitives. This is a structural gap: the harness's action vocabulary is closed by construction, while the workspace's affordances are open, so failure recovery is undefined precisely where it is most needed.

## Key Insight

A substitution operator that expands the action set only when an atomic action fails, and that composes new actions from primitives while preserving the closure level of the workspace, restores the invariant that the reachable action set is closed under the workspace's openness — an invariant that fixed-action-set agents violate by construction.

## Method

**FARA** (Failure-triggered Action-set Replacement with Axiomatic closure) takes as input a spreadsheet workspace state $s$, a goal $g$, a base atomic action set $A_0$, and an openness level $k \in \{0,1,2\}$ (0 = closed, 1 = open under composition, 2 = open under composition and external calls). It outputs either a successful action sequence or a substitution certificate.

**Load-bearing assumption (stated explicitly).** FARA assumes that the primitive set $\text{PRIMITIVES}(A)$ plus bounded composition of depth $m \le k+1$ spans the workspace's affordances at openness level $k$, so that a non-empty filtered $C$ is equivalent to the existence of a licensed recovery action and an empty filtered $C$ certifies irrecoverability *within the declared primitive basis and composition budget*. This is a closed-world assumption over the primitive vocabulary: it does not claim that the workspace has no other affordances, only that no recovery is expressible from the current primitives at the current depth. The certificate is therefore a certificate of bounded irrecoverability, not of absolute irrecoverability.

```pseudocode
FARA(s, g, A_0, k):
  A = A_0
  depth = k + 1                     # initial composition bound
  loop:
    a = LLM.plan(s, g, A)            # propose next action from current set
    try:
      s' = EXECUTE(a, s)
      if VERIFY(s', g): return s'
      s = s'
    except ActionFailure as f:
      # substitution closure step with iterative deepening
      C = COMPOSE(A, f, depth)       # candidate new actions from primitives
      C = FILTER_CLOSURE(C, A, k)    # keep only those preserving closure level k
      if C empty and depth < MAX_DEPTH:   # MAX_DEPTH = 5 (calibrated, see below)
        depth = depth + 1            # deepen once, then retry
        continue
      if C empty: return CERTIFICATE(f, A, k, depth)
      A = A ∪ C                      # expand action set, not replace
      continue

COMPOSE(A, f, depth):
  prims = PRIMITIVES(A)              # atomic actions and their inverses
  return { compose(p1,...,pm) : m ≤ depth, pi ∈ prims, signature matches f }

FILTER_CLOSURE(C, A, k):
  return { c ∈ C : ∀ s, EXECUTE(c, s) ∈ DOMAIN_k }   # stays inside openness level
```

**Calibration of MAX_DEPTH.** MAX_DEPTH is set to 5 by a grounding study on a 200-failure calibration split (disjoint from test): for each injected failure we compute the minimum composition depth $m^*$ at which a signature-matching composition exists in the primitive basis; we set MAX_DEPTH to the 95th percentile of $m^*$ over recoverable failures, rounded up. This makes the certificate's boundedness explicit and data-driven rather than an arbitrary constant.

**Why this design.** We chose failure-triggered expansion over always-on open action generation because always-on generation makes the planner's search space unbounded and empirically degrades plan quality (the planner spends budget on irrelevant compositions), accepting the cost that substitution cannot preempt a failure. We chose composition from primitives over retrieval of similar past actions because retrieval cannot produce an action that was never in the library — the failure mode FARA targets is precisely the absence of a suitable action — accepting the cost that composition must be checked for closure. We chose a closure filter with an explicit openness level $k$ over an unconstrained composition grammar because unconstrained composition can silently change the workspace's openness (e.g., by invoking an external service in a closed workspace), accepting the cost that some failures remain unrecoverable and are reported as certificates. A domain expert would not describe FARA as a variant of Toolformer-style tool retrieval or of a fixed-action spreadsheet agent such as SheetCopilot, because the load-bearing operation is set expansion under a closure invariant, not selection from a fixed or retrieved set.

**Why this measures what we claim.** The computational quantity `C = COMPOSE(A, f, depth)` measures *recoverability within the primitive basis* because the assumption that the primitive set plus bounded composition spans the workspace's affordances at level $k$ makes a non-empty filtered $C$ equivalent to the existence of a licensed recovery action; this assumption fails when the workspace contains an affordance not expressible as a composition of primitives (e.g., a new external API in a level-2 workspace, or a spreadsheet function absent from PRIMITIVES(A)), in which case `C` reflects only primitive-composition recoverability, not full workspace recoverability. The quantity `FILTER_CLOSURE(C, A, k)` measures *static closure preservation* because the assumption that every action's effect stays in the domain defined by $k$ makes the filter equivalent to preserving the openness invariant; this assumption fails when an action's effect depends on runtime state not captured in the domain definition (e.g., a formula whose result depends on cell values that change the effective function surface), in which case the filter reflects static closure, not dynamic closure. The `CERTIFICATE(f, A, k, depth)` output measures *bounded irrecoverability under the declared primitive basis and depth* because the assumption that `COMPOSE` enumerates all licensed compositions up to `depth` makes an empty filtered set equivalent to no licensed recovery within that basis and budget; this assumption fails when the composition bound `depth` is too small or when the needed affordance is outside PRIMITIVES(A), in which case the certificate reflects bounded irrecoverability, not absolute irrecoverability.

## Contribution

(1) A failure-triggered action-substitution operator with an explicit openness-preserving closure filter, formalized as a set-expansion rule over a base atomic action set and a workspace openness level. (2) A substitution certificate that distinguishes irrecoverable failures from recoverable ones under a declared openness level, enabling the harness to report when its action vocabulary is the binding constraint. (3) An instantiation of the rule as a spreadsheet agent harness that composes primitive spreadsheet operations (read, write, filter, pivot, formula injection) into substitute actions on failure, demonstrating that closure-preserving substitution recovers tasks that fixed-action-set agents abandon.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Spreadsheet manipulation tasks with injected action failures (200 calibration + 800 test failures, disjoint) | Failures must be triggered to exercise substitution closure |
| Primary metric | Task success rate under failure injection | Directly measures recoverability claim, not raw planning quality |
| Secondary metric | Openness-violation rate (fraction of recovered tasks whose action sequence leaves DOMAIN_k) | Tests closure preservation directly |
| Secondary metric | Primitive-composition recoverability fraction (fraction of injected failures with a signature-matching composition at depth ≤ MAX_DEPTH) | Quantifies the span assumption empirically |
| Baseline | Always-on open action generation | Tests whether failure-triggering beats unbounded search |
| Baseline | Retrieval-based action substitution | Tests whether composition beats selection from library |
| Baseline | Fixed-action spreadsheet agent | Tests whether expansion beats closed action set |
| Ablation-of-ours | FARA without FILTER_CLOSURE | Isolates closure-preservation contribution from expansion |
| Ablation-of-ours | FARA with fixed depth = k+1 (no iterative deepening) | Isolates the effect of adaptive depth on certificate soundness |

### Why this setup validates the claim

The central claim is that failure-triggered action-set expansion under a closure invariant recovers failures that selection-based and always-open methods cannot, while preserving workspace openness. The dataset must therefore contain tasks where the base action set is genuinely insufficient, so that recovery requires generating an action never present in any library; otherwise all baselines succeed and the claim is untested. The always-on baseline tests the sub-claim that failure-triggering is preferable to unbounded generation: if open generation were equally good, our triggering mechanism adds nothing. The retrieval baseline tests the sub-claim that composition beats selection: it can only recombine existing library actions, so on instances where the needed action is absent it must fail. The fixed-action baseline tests the sub-claim that expansion beats a closed set, which is the trivial floor. The ablation isolates the closure filter: without it, expansion may succeed on some tasks but should violate openness on level-0 and level-1 workspaces, so the primary metric must be conditioned on failure-injected tasks where recovery is required. The primitive-composition recoverability fraction is measured on the calibration split to quantify the span assumption: it reports the fraction of injected failures whose required recovery action is expressible as a composition of at most MAX_DEPTH primitives, thereby grounding the certificate's boundedness claim. Success rate under injection is the right primary metric because it directly detects the predicted effect — gains concentrated exactly where base actions are insufficient — whereas raw plan quality would be confounded by planner budget.

### Expected outcome and causal chain

**vs. Always-on open action generation** — On a case where the goal requires a two-step composition that is irrelevant to the current subgoal, the always-on baseline generates many candidate compositions each step, and the planner's budget is spent scoring irrelevant actions, producing either a wrong action choice or budget exhaustion before the needed composition appears. Our method keeps $A$ at $A_0$ until a failure fires, so the planner searches a small set and only expands when a concrete failure names the missing signature. We expect a noticeable gap on failure-injected tasks with many distractor compositions, but near-parity on tasks where $A_0$ already suffices.

**vs. Retrieval-based action substitution** — On a case where a failure requires an action that was never in the library (e.g., a composed inverse-then-forward primitive sequence), the retrieval baseline returns the nearest past action, which has the wrong signature and fails again, because retrieval cannot synthesize an action absent from its index. Our method instead composes from primitives to match the failure signature, so the needed action is constructed rather than retrieved. We expect a clear gap concentrated on the subset where the required action is absent from all libraries, and parity on the subset where a similar past action exists.

**vs. Fixed-action spreadsheet agent** — On a case where the workspace affordance needed for recovery is simply not in the fixed action set, the fixed-action baseline loops or terminates with a wrong state because it has no mechanism to grow its action set. Our method expands $A$ via COMPOSE on failure, so a licensed recovery action enters the set. We expect the largest gap on failure-injected tasks, with the fixed baseline near-zero recovery, and we expect the ablation without FILTER_CLOSURE to recover more often but to break openness on level-0/1 workspaces, showing closure preservation is load-bearing.

**vs. FARA without FILTER_CLOSURE** — On level-0 and level-1 workspaces, the no-filter ablation may recover failures by composing actions that invoke external calls or leave the closed domain, so its success rate can be higher on some tasks but its openness-violation rate should be substantially higher. We expect full FARA to match or exceed the ablation on success rate while keeping openness-violation rate near zero, demonstrating that the closure filter is what preserves the invariant.

**vs. FARA with fixed depth = k+1** — On failures whose minimal recovery composition depth exceeds k+1, the fixed-depth variant returns a certificate (false negative) while full FARA deepens and recovers. We expect full FARA to have a higher success rate on the subset of failures with $m^* > k+1$, and the calibration study to show that MAX_DEPTH=5 covers ≥95% of recoverable failures.

### What would falsify this idea

If our gain over the retrieval and fixed-action baselines is uniform across tasks rather than concentrated on the subset where the required action is absent from the library, then the composition mechanism is not what drives recovery and the central claim is wrong. Likewise, if the no-FILTER_CLOSURE ablation matches full FARA on openness-violation rate, the closure invariant is doing no work; and if the primitive-composition recoverability fraction on the calibration split is low (e.g., <50%), the span assumption is empirically weak and the certificate's boundedness claim is undermined.

## References

1. Work Agents: Systems, Capabilities, and Optimization
2. ScienceAgentBench: Toward Rigorous Assessment of Language Agents for Data-Driven Scientific Discovery
3. SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering
