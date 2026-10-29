# Protocol Closure Scoring for Human-Agent Collaboration with Out-of-Protocol Redirections

## Motivation

Existing agent evaluation benchmarks, including RealCLI's scripted-user oracle and SWE-bench's fixed test suites, assume a pre-declared action alphabet and a fixed interaction protocol, so any human action outside that alphabet (e.g., a novel constraint revision or an unanticipated authorization) has no defined evaluation semantics. This structural limitation means agents are never scored on their ability to incorporate or reject out-of-protocol human redirections, leaving a core aspect of interactive adaptation unmeasured.

## Key Insight

Treating the interaction protocol as a partial algebra and scoring the minimal cost of closing it under a human-initiated redirection operator yields a structural invariant that is computable from the protocol's syntax and semantics, independent of task success or learned signals.

## Method

**What it is** — We introduce Protocol Closure Scoring (PCS), a benchmark evaluation mechanism that takes as input a base protocol (a partial algebra of agent actions with pre/post-conditions) and a human redirection action drawn from an open set, and outputs a score equal to the minimal extension cost of closing the protocol under that redirection, or a certified refusal cost.

**How it works** — PCS operates in three phases, shown as pseudocode below.

```pseudocode
# Input: base protocol P = (A, Σ, δ, pre, post) where A is agent action set,
# Σ is state space, δ: A × Σ → Σ is partial transition function,
# pre: A × Σ → Bool, post: A × Σ × Σ → Bool.
# Human redirection r is an unenumerated action with observed pre-state σ and desired post-state σ'.
# Hyperparameters: cost function c: Extension → ℝ≥0 (default: number of new action symbols + guard clauses).

Phase 1: Parse redirection r
  Extract from r a tuple (name, args, pre_r, post_r) using a rule-based semantic parser
  (regex + typed slots; no learned model).
  To guard against parser artifacts, the parsed tuple is treated as an explicit, human-verifiable intermediate artifact:
  - The redirection is presented to the agent as (or alongside) a structured pre/post-condition spec
    (e.g., a JSON object with fields name, args, pre, post) whenever the source benchmark provides one;
    otherwise, the rule-based parser produces the tuple from natural language.
  - A parser-agnostic equivalence check is applied: two independent parses (e.g., the rule-based parser
    and a second, independently written rule-based parser with different regex patterns) must agree
    on the tuple (name, args, pre_r, post_r) up to variable renaming.
  - If the two parses disagree, the redirection is placed in an 'unparseable/ambiguous' bucket,
    excluded from the closure metric, and reported separately.
  If parsing fails (both parsers fail or produce no tuple) → return REFUSE with certified reason "ill-formed redirection".

Phase 2: Check admissibility
  If pre_r(σ) is false → return REFUSE with certified reason "precondition violated".
  If post_r(σ') is inconsistent with P's invariants → return REFUSE with certified reason "invariant conflict".
  Else proceed to closure.

Phase 3: Compute minimal extension
  Let E be the set of candidate extensions: each extension adds a new action symbol a_r to A,
  defines δ(a_r, σ) = σ', and adds guard clauses to existing actions to preserve partiality.
  For each candidate E, verify that the extended algebra P_E is well-formed:
    - δ remains a partial function (no two actions map same state to different next states),
    - all pre/post-conditions are satisfiable,
    - no cycles in the dependency graph of actions,
    - non-triviality constraint: the added action a_r must not be a no-op; formally, there must exist
      at least one state s in Σ such that δ(a_r, s) ≠ s, and the post-condition post_r must not be
      trivially satisfiable by the identity transition (i.e., σ' ≠ σ or the guard clauses must
      enforce a change in at least one observable variable).
  If no well-formed extension exists → return REFUSE with certified reason "no closure".
  Else compute cost c(E) = |new symbols| + |new guard clauses| + |removed transitions|.
  Return min_E c(E) as the closure score.
```

**Why this design** — We chose a rule-based semantic parser over a learned parser because the evaluation must be reproducible and independent of any particular LLM's biases, accepting the cost that the parser may fail on highly ambiguous natural-language redirections (which then yield a certified refusal rather than a spurious closure). To mitigate parser brittleness, we require the parsed tuple to be an explicit, human-verifiable artifact and enforce agreement between two independent rule-based parsers; disagreements are excluded from the closure metric and reported as an 'unparseable/ambiguous' bucket. We chose a minimal-extension cost over a binary success/failure because the cost provides a graded signal that distinguishes agents that can incorporate redirections with small protocol changes from those that require large, disruptive extensions, accepting the cost that the cost function must be manually specified (we use a simple syntactic count). We chose to certify refusals with explicit reasons over silent failure because the evaluation must distinguish an agent that correctly refuses an inadmissible redirection from one that fails to parse it, accepting the cost that the refusal reasons must be manually validated. This design is not a variant of RealCLI's scripted-user evaluation: RealCLI uses a fixed template of user actions and scores task success plus collaboration quality; PCS instead treats the protocol as a partial algebra and scores the minimal cost of closing it under an open-set redirection, which is a structural invariant not present in any retrieval-based or template-based benchmark.

**Why it measures what we claim** — The computational quantity `min_E c(E)` measures the agent's ability to compute protocol closure because we assume that the minimal extension cost is a monotone function of the agent's understanding of the protocol's semantics; this assumption fails when the agent can produce a syntactically valid but semantically vacuous extension (e.g., adding a no-op action), in which case the score reflects syntactic manipulation rather than semantic closure. The certified refusal reason measures the agent's ability to recognize inadmissibility because we assume that the reason correctly identifies the violated precondition or invariant; this assumption fails when the agent's parser misidentifies the violated condition, in which case the refusal reason reflects parser error rather than protocol understanding. The admissibility check measures the agent's ability to detect conflicts because we assume that the invariants are complete and consistent; this assumption fails when the invariants are incomplete, in which case the check may miss conflicts or flag spurious ones. The non-triviality constraint in Phase 3 measures the agent's ability to avoid vacuous extensions because we assume that any extension satisfying the constraint represents a genuine semantic change; this assumption fails when the constraint is satisfied by a syntactically valid but semantically meaningless change (e.g., renaming a variable), in which case the score reflects syntactic compliance rather than semantic closure.

## Contribution

(1) A novel evaluation mechanism, Protocol Closure Scoring (PCS), that treats the interaction protocol as a partial algebra and scores agents on the minimal cost of closing it under open-set human redirections. (2) A design principle: protocol closure is a structural invariant that can be computed from syntax and semantics alone, enabling reproducible evaluation of interactive adaptation without learned signals. (3) A set of certified refusal reasons and a cost function that together provide a graded, interpretable score for out-of-protocol human actions.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Protocol-redirection suite (base protocols + open-set redirections) | Directly exercises partial-algebra closure, not task success |
| Primary metric | Closure score `min_E c(E)` (lower = better agent) | Graded structural signal distinguishing small vs disruptive extensions |
| Baseline | RealCLI-style scripted-user evaluation | Tests whether template-based scoring captures closure structure |
| Baseline | SWE-bench pass-rate harness | Tests whether binary test-suite success proxies closure cost |
| Baseline | LLM-judge semantic similarity | Tests whether surface plausibility substitutes for certified closure |
| Ablation-of-ours | PCS with binary success/failure (no cost grading) | Isolates whether the graded cost signal carries the claim |
| Ablation-of-ours | PCS without non-triviality constraint | Tests whether the non-triviality constraint is necessary to reject vacuous extensions |

### Why this setup validates the claim

The central claim is that minimal extension cost under an open-set redirection is a structural invariant that discriminates genuine protocol understanding from plausible-looking but semantically vacuous behavior. Each baseline probes a different competing hypothesis: RealCLI's scripted-user template tests whether fixed action templates already capture closure structure (if so, PCS adds nothing); SWE-bench's binary pass-rate tests whether a coarser success signal is sufficient (if so, the graded cost is redundant); the LLM-judge similarity baseline tests whether surface plausibility correlates with closure cost (if so, the structural computation is unnecessary). The ablation removes cost grading while keeping the algebra, isolating whether the *graded* signal—not merely the well-formedness check—is what produces the predicted discrimination. The additional ablation removes the non-triviality constraint, isolating whether that constraint is necessary to reject vacuous extensions. The closure score is the right metric because it is the only quantity that directly operationalizes the claimed invariant: it varies continuously with how disruptive a redirection is to the protocol, so a method that genuinely reasons about closure should show cost stratification that the baselines cannot produce. If the predicted stratification fails to appear, the invariant claim is unsupported regardless of aggregate accuracy.

### Expected outcome and causal chain

**vs. RealCLI-style scripted-user evaluation** — On a case where a redirection requires adding a new action symbol with a guard clause that partially disables an existing transition, the RealCLI baseline maps the redirection onto its nearest scripted template and scores task completion plus collaboration quality, so it reports near-success because the template's fixed action sequence still executes; its mechanism gap is that templates cannot represent protocol extension cost. Our method instead parses the redirection, detects the required guard clause, and charges the minimal extension cost, so we expect a clear cost gap between redirections that are template-expressible and those that are not, with the baseline nearly flat across that split.

**vs. SWE-bench pass-rate harness** — On a case where a redirection is admissible but demands a disruptive extension (removing several transitions to preserve partiality), the SWE-bench-style harness runs a test suite that passes because the redirection's target post-state is reachable, collapsing a disruptive closure into a binary success; its mechanism gap is that test suites do not measure structural disruption. Our method instead returns a high closure cost for the disruptive extension and a low cost for a cheap one, so we expect the baseline to show parity between cheap and disruptive redirections while our score separates them by a wide margin concentrated exactly on the disruptive subset.

**vs. LLM-judge semantic similarity** — On a case where an agent proposes a syntactically valid but semantically vacuous extension (a no-op action satisfying the post-state trivially), the LLM-judge baseline scores it as highly plausible because the proposal reads fluently; its mechanism gap is that surface plausibility does not enforce well-formedness or partiality. Our method instead rejects or penalizes the vacuous extension via the well-formedness and partial-function checks, so we expect the judge baseline to inflate scores on the vacuous-extension subset while our score stays low there, with the two converging only on genuinely well-formed closures.

**vs. PCS without non-triviality constraint** — On a case where an agent proposes a no-op action that satisfies the post-state trivially, the ablation without the non-triviality constraint accepts the extension as well-formed and assigns a low cost, whereas the full PCS rejects it or assigns a high cost due to the non-triviality constraint. We expect the ablation to show inflated acceptance of vacuous extensions and lower closure scores on the vacuous-extension subset, confirming that the non-triviality constraint is necessary to prevent syntactic manipulation.

### What would falsify this idea

If our closure score's advantage over the baselines were uniform across all redirection subsets—rather than concentrated on the disruptive-extension and vacuous-extension subsets where the predicted mechanism should bite—the claim that minimal extension cost captures structural closure would be unsupported. Likewise, if the binary-success ablation matched the graded cost's discrimination pattern, the graded signal would be doing no work; and if the ablation without the non-triviality constraint matched the full PCS on the vacuous-extension subset, the non-triviality constraint would be unnecessary.

## References

1. RealCLI: Toward a Unified Evaluation of Human-Agent Collaboration for CLI Agents
2. SWE-bench: Can Language Models Resolve Real-World GitHub Issues?
