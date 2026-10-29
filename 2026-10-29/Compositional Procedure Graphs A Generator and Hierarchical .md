# Compositional Procedure Graphs: A Generator and Hierarchical Scorer for Tool-Using Agents

## Motivation

Existing procedure-aware benchmarks treat procedure specifications as flat, fixed lists of required steps. TauIndianBankBench, for instance, annotates each of its 1,000 banking tasks with a single-domain procedure of required steps and constraints, so the specification is not closed under composition; it cannot express parallel branches that must all complete, nested sub-procedures with their own internal dependencies, or alternative valid orderings. As a result, the field cannot systematically vary trajectory breadth (number of parallel branches) and depth (nesting levels) while keeping the procedure specification faithful, which is exactly the variation needed to expose where agents fail as procedural complexity grows. The root cause is structural: a flat step list has no algebra, so it cannot be composed, nested, or parameterized without losing the very dependency information the scorer must check.

## Key Insight

A procedure specification expressed as a typed hierarchical graph (sequence, parallel, choice, and atomic-step nodes) is closed under composition, so a single generator can produce tasks whose required-step graphs vary breadth and depth independently while the dual-check scorer evaluates any such graph by a recursive graph-traversal that is exact for the same reason a tree automaton is exact on its language.

## Method

**What it is.** CPG-Gen (Compositional Procedure Graph Generator) takes a library of atomic tool-action schemas and a set of procedure templates written in a small typed graph language, and outputs tasks each paired with a hierarchical required-step graph (RSG). CPG-Score takes an agent trajectory and an RSG and returns a dual check: (i) outcome correctness against the final record, and (ii) hierarchical procedure adherence computed by a recursive traversal of the RSG over the trajectory's action sequence.

**How it works.** The procedure specification language is a small typed term algebra. A procedure term is one of: `Atom(tool, args, effect)`, `Seq(p1,...,pk)`, `Par(p1,...,pk)`, `Choice(p1,...,pk)`, or `Nest(guard, p)`. Each term compiles to an RSG node with a node type, a set of child edges, and a `required` flag; `Par` children are unordered and all required; `Choice` children are alternatives of which exactly one must be satisfied; `Nest` attaches a guard predicate (a state condition) that must hold before its subtree is traversed.

```pseudocode
# Generation
generate_task(template, breadth_range, depth_range):
    rsg = instantiate(template)                       # fills tool slots from schema library
    rsg = expand_parallel(rsg, b ~ Uniform(breadth_range))   # replace Par placeholders with b branches
    rsg = expand_nest(rsg, d ~ Uniform(depth_range))         # wrap d sub-procedures in Nest guards
    task = render_dialogue(rsg)                        # simulated customer utterance per leaf
    return task, rsg

# Scoring
score(trajectory, rsg):
    outcome_ok = check_final_record(trajectory.final_state, rsg.expected_record)
    matched = consuming_match(trajectory.actions, rsg)   # returns set of satisfied RSG nodes and consumed action indices
    adherence_ok = all_required_satisfied(rsg, matched) and no_unrequested_change(trajectory)
    return {outcome_ok, adherence_ok, matched}

# Consuming recursive match over RSG nodes (each action used at most once)
consuming_match(actions, node):
    if node.type == Atom:
        # find first unconsumed action a such that atom_equiv(a, node)
        for i, a in enumerate(actions):
            if i not in consumed and atom_equiv(a, node):
                consumed.add(i)
                return {node}, {i}
        return {}, {}
    if node.type == Seq:
        return seq_match_consuming(actions, node.children)   # contiguous-subsequence DP with consumption
    if node.type == Par:
        # all children must be satisfied on disjoint action subsets
        return par_match_consuming(actions, node.children)
    if node.type == Choice:
        # exactly one child subtree must be satisfied
        return choice_match_consuming(actions, node.children)
    if node.type == Nest:
        if guard_holds(node.guard, actions):
            return consuming_match(actions, node.child)
        else:
            return {}, {}

# atom_equiv: typed semantic equivalence over (tool, args)
atom_equiv(a, node):
    if a.tool != node.tool: return False
    # canonicalize argument values by type: numeric tolerance 1e-6, string case-fold and trim,
    # unordered collections compared as multisets, optional fields ignored if absent in both
    return canonicalize(a.args) == canonicalize(node.args)

# seq_match_consuming: DP over (action index, child index) requiring children satisfied in order,
# allowing intervening actions, and consuming actions used by earlier children.
# par_match_consuming: backtracking search over assignments of disjoint action subsets to children;
# succeeds iff all children matched.
# choice_match_consuming: try each child; succeed if exactly one child matches.
```

`seq_match_consuming` is a dynamic program over (action index, child index) that requires children to be satisfied in order but allows intervening actions, and marks actions consumed by earlier children so they cannot be reused. `par_match_consuming` performs a backtracking search that assigns each child a disjoint subset of actions; it succeeds only if all children are matched. `choice_match_consuming` tries each child and succeeds if exactly one child subtree is satisfied. `no_unrequested_change` compares the final record to the expected record and flags any field changed that is not the target of a required atom.

**Why this design.** We chose a typed term algebra with explicit `Par`/`Choice`/`Nest` constructors over a flat required-step list (as in TauIndianBankBench) because flat lists cannot express unordered parallel completion or guarded sub-procedures, accepting that the generator must maintain a schema library and guard predicates per domain. We chose recursive ordered matching with a DP over `Seq` nodes over a set-based coverage check (as in ToolSandbox's milestone scoring) because set coverage collapses `Seq` ordering, accepting a slightly higher scoring cost per trajectory in exchange for detecting order violations that set-based checks miss. We chose to generate tasks from templates with explicit breadth and depth parameters over sampling from a fixed suite (as in TRAJECT-Bench's synthesized trajectories) because template instantiation lets us hold the atomic-step vocabulary fixed while varying only structural complexity, accepting that template coverage bounds the diversity of generated procedures. A domain expert would not describe this as a variant of TRAJECT-Bench because TRAJECT-Bench varies trajectory breadth and depth but scores against a fixed, non-compositional task specification; here the specification itself is a composable graph and the scorer is a graph automaton, which is the object that is closed under composition.

**Why it measures what we claim.** The `matched` set from `consuming_match` measures procedure adherence because the recursive traversal is a tree-automaton acceptance check with action consumption: a trajectory satisfies an RSG node exactly when the automaton reaches an accepting state using each action at most once, under the assumption that tool-action matching is semantically exact (an action matches an atom iff tool name and canonicalized argument values are equal); this assumption fails when an agent reaches the same record via a semantically equivalent but syntactically different action (e.g., a different argument ordering or numeric formatting), in which case `matched` reflects syntactic non-coverage rather than procedural failure. The `breadth_range` and `depth_range` parameters measure trajectory structural complexity because `expand_parallel` and `expand_nest` directly set the number of `Par` children and `Nest` levels in the RSG, under the assumption that RSG node count is a faithful proxy for the number of required tool calls; this assumption fails when a `Nest` guard is never triggered, in which case the realized trajectory depth is lower than the nominal depth. The `no_unrequested_change` flag measures constraint satisfaction because it compares the final record field-by-field against the expected record, under the assumption that the expected record fully specifies which fields should change; this assumption fails when the task legitimately permits an optional field change, in which case the flag reports a false violation.

## Contribution

['A typed compositional procedure specification language (CPG) with `Seq`, `Par`, `Choice`, and `Nest` constructors, together with a generative task generator that instantiates CPG templates with independently controlled breadth and depth parameters.', 'A hierarchical dual-check scorer (CPG-Score) that evaluates agent trajectories against a required-step graph via recursive ordered matching, extending flat procedure scoring to nested and parallel dependency structures.', 'An empirical finding that procedure-aware scoring under CPG reveals order and coverage violations that outcome-only scoring misses, and that agent success degrades as a function of RSG breadth and depth independently.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | CPG-Gen generated RSG tasks (banking schema library) | Structural complexity sweep with ground-truth graphs |
| Primary metric | Hierarchical procedure-adherence rate (RSG acceptance) | Directly measures the compositional claim |
| Baseline | Outcome-only scorer (TauIndianBankBench-style) | Tests whether procedure structure adds signal |
| Baseline | Set-based milestone coverage (ToolSandbox-style) | Tests whether Seq ordering matters |
| Baseline | Flat required-step list scorer | Tests whether Par/Choice/Nest composition matters |
| Ablation-of-ours | CPG-Score w/o Nest guards (flat Seq/Par/Choice) | Isolates contribution of guarded subtrees |
| Ablation-of-ours | CPG-Score with non-consuming matcher (original union-based) | Isolates contribution of action consumption and Par/Choice disambiguation |
| Diagnostic | Syntactic-matching false-negative rate on 200 hand-labeled semantically equivalent trajectories | Quantifies how much adherence gap is an artifact of matching strictness |
| Diagnostic | Realized vs. nominal depth distribution (mean, std, % untriggered Nest guards) | Tests whether nominal depth equals realized trajectory complexity |

### Why this setup validates the claim

The central claim is that a composable required-step graph plus a recursive tree-automaton scorer with action consumption detects procedural failures that flat or set-based specifications structurally cannot. Each baseline targets one sub-claim: the outcome-only scorer tests whether procedure adherence carries information beyond final-record correctness; the set-based milestone baseline tests whether `Seq` ordering is a real failure axis rather than noise; the flat required-step baseline tests whether `Par`/`Choice`/`Nest` expressiveness changes what the scorer can discriminate. The ablation removes only `Nest` guards, so any drop relative to full CPG-Score is attributable to guarded subtrees rather than to generator complexity or schema vocabulary. The non-consuming ablation isolates the effect of action consumption and Par/Choice disambiguation, directly testing the load-bearing assumption that the matcher is a genuine consuming automaton. The syntactic-matching false-negative diagnostic measures the gap between syntactic and semantic atom matching on a hand-labeled set, and the realized-vs-nominal depth diagnostic measures whether nominal breadth/depth parameters faithfully control trajectory complexity. Hierarchical procedure-adherence rate is the right primary metric because it is exactly the quantity the consuming tree-automaton acceptance check computes, and it decomposes by RSG node type, letting us see whether gains concentrate on the structural constructs each baseline cannot represent. Falsification is therefore built into the design: if gains are uniform across node types, the compositionality story is wrong.

### Expected outcome and causal chain

**vs. Outcome-only scorer (TauIndianBankBench-style)** — On a case where the agent reaches the correct final record by reordering two dependent tool calls (e.g., reading balance after a transfer instead of before), the baseline marks it correct because it inspects only the final record. Our method instead runs `seq_match_consuming` over the `Seq` node and finds the ordered subsequence violated, so adherence fails while outcome passes. We expect a noticeable gap between outcome accuracy and adherence rate concentrated on order-sensitive subsets, with near-parity on single-atom tasks.

**vs. Set-based milestone coverage (ToolSandbox-style)** — On a case where a `Par` node requires three unordered branches all satisfied and the agent completes only two before terminating, the baseline's set coverage over milestones can still report high partial credit and, depending on thresholding, accept the trajectory. Our method's `all_required_satisfied` check on `Par` children marks the node unsatisfied because all `Par` children are required. We expect our adherence rate to drop sharply on high-breadth `Par` subsets where the baseline stays flat, producing a widening gap as `breadth_range` increases.

**vs. Flat required-step list scorer** — On a case with a `Nest` guard whose predicate is false (a sub-procedure that should never fire), the flat scorer still expects the nested atoms and penalizes the agent for correctly skipping them, producing a false violation. Our method traverses the `Nest` subtree only when `guard_holds`, so the correctly-skipped branch is not counted as a miss. We expect the flat baseline to under-report adherence on `Nest`-heavy subsets while our method stays stable, and the no-Nest ablation to reproduce the flat baseline's pattern.

**vs. CPG-Score with non-consuming matcher (ablation)** — On a case where the same action (e.g., a balance check) is used to satisfy two different atoms in a `Par` node, the non-consuming matcher's union over children accepts the trajectory because both atoms are matched by the same action. Our consuming matcher assigns each action to at most one atom, so the `Par` node fails because the children cannot be matched on disjoint action subsets. We expect the non-consuming ablation to over-report adherence on high-breadth `Par` subsets and on trajectories with repeated tool calls, with the gap widening as `breadth_range` increases.

**Diagnostic: syntactic-matching false-negative rate** — On a hand-labeled set of 200 trajectories where the agent reaches the correct record via semantically equivalent but syntactically different actions (e.g., different argument ordering, numeric formatting, or optional field omission), we compute the rate at which exact syntactic matching (tool name and raw argument equality) marks an atom as unmatched. We expect this rate to be non-zero (target < 5% after canonicalization) and report how much of the adherence gap versus baselines is attributable to matching strictness rather than procedural failure.

**Diagnostic: realized vs. nominal depth** — For each generated task, we record the nominal depth (number of `Nest` levels) and the realized depth (number of `Nest` guards actually triggered during a reference trajectory). We report the mean and standard deviation of the difference and the percentage of untriggered guards. We expect realized depth to be lower than nominal depth on a non-trivial fraction of tasks (target < 10% untriggered), and we report this distribution to bound the assumption that nominal depth equals realized complexity.

### What would falsify this idea

If CPG-Score's advantage over the set-based and flat baselines were uniform across RSG node types rather than concentrated on `Seq`-ordering, high-breadth `Par`, and triggered-`Nest` subsets, then the compositional-graph mechanism would not be the cause of the gain. Likewise, if the no-Nest ablation matched full CPG-Score everywhere, guarded subtrees would carry no measurable signal and the term-algebra design would be unjustified.

## References

1. TauIndianBankBench: Procedure-Aware, Action-Oriented Evaluation of Tool-Using Conversational Agents
2. Beyond the Final Answer: Evaluating the Reasoning Trajectories of Tool-Augmented Agents
3. TRAJECT-Bench:A Trajectory-Aware Benchmark for Evaluating Agentic Tool Use
4. CRMArena-Pro: Holistic Assessment of LLM Agents Across Diverse Business Scenarios and Interactions
5. ACEBench: Who Wins the Match Point in Tool Usage?
6. τ2-Bench: Evaluating Conversational Agents in a Dual-Control Environment
7. Non-Collaborative User Simulators for Tool Agents
8. ToolSandbox: A Stateful, Conversational, Interactive Evaluation Benchmark for LLM Tool Use Capabilities
