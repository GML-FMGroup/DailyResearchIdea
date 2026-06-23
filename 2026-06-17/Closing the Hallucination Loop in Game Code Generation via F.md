# Closing the Hallucination Loop in Game Code Generation via Finite-State Machine Induction and Abstract Interpretation

## Motivation

Existing LLM-based code generation for games lacks a verification mechanism without a complete test suite, leading to hallucinated behavior that violates implicit state invariants. Prior work such as 'Empirical Research on Utilizing LLM-based Agents for Automated Bug Fixing via LangGraph' relies on iterative bug fixing with execution feedback but assumes a test oracle, which is unavailable for novel game code. The root cause is that LLMs generate code from static context without reasoning about the game's dynamic state space, causing state-dependent hallucinations that are missed by single-pass generation.

## Key Insight

Games have deterministic, finite state spaces under typical interactions, making it possible to induct a faithful finite-state machine from a small set of random playthroughs and then use abstract interpretation to provably check whether generated code respects the state invariants.

## Method

### (A) What it is
**FSM-GenVer** is a method that, given a game description, automatically infers a finite-state machine (FSM) from a small number of coverage-guided playthroughs and then verifies LLM-generated game code against that FSM using abstract interpretation. When violations are detected, it performs targeted re-generation of the offending code block guided by the FSM constraints.

### (B) How it works
```pseudocode
Input: Game description D (natural language + reward function), LLM G, number of playthroughs N=50, abstraction bound k=3, coverage threshold θ=95%
Output: Verified game code C

1. // Phase 1: FSM Induction with Coverage-Guided Exploration
   s0 = initial game state
   traces = []
   state_coverage = set()
   while |state_coverage| < threshold or len(traces) < N:
       s = select_state_to_explore(state_coverage)  // e.g., least visited
       trace = play_with_coverage_guidance(s, max_steps=100)  // uses BFS-like expansion
       traces.append(trace)
       state_coverage.update(states_in_trace)
   FSM = induct_FSM(traces, algorithm='L* with random walk equivalence oracle')  
   // FSM = (S, A, δ, s0, F) where S=states, A=actions, δ=transition function, F=final states
   // Verify FSM fidelity: sample 10 held-out coverage-guided traces, compute transition match rate; if <90%, increase N or continue exploration.

2. // Phase 2: Generate candidate code
   C = G(D, "Write game logic that implements the described mechanics.")

3. // Phase 3: Abstract Interpretation
   // Define abstract domain: for numeric variables, use k-interval abstraction (k=3 subdivide each variable into intervals of equal width based on observed min/max)
   // For enum variables, use explicit set abstraction.
   abstract_interpreter = build_abstract_interpreter(C, abstraction='k-interval (k=3) + explicit set', library='custom Python AST interpreter using z3 for constraint solving')
   for each state s in S of FSM:
       for each action a in A applicable to s:
           // Compute abstract post-state from concrete transition δ(s,a)
           concrete_post = δ(s,a)
           // Compute abstract post-state from code execution
           abstract_post = abstract_interpreter.run(C, s, a)  // sound overapproximation
           // Check inclusion: concrete_post ∈ abstract_post (i.e., code's behavior covers the FSM transition)
           if not check_inclusion(concrete_post, abstract_post):
               violation = (s, a, concrete_post, abstract_post)
               add_to_violations(violation)

4. // Phase 4: Targeted Re-generation
   if violations is non-empty:
       select most frequent violating state-action pair (s_v, a_v)
       prompt = f"The following block of code for state {repr(s_v)} and action {a_v} does not produce the expected next state {repr(concrete_post)}. Please rewrite the corresponding code block to make the transition correct. Reference: the variable values in state {repr(s_v)} are {repr(s_v.variables)}."
       C = G(prompt, reference_code=C)
       goto Step 3  // re-verify
   else:
       output C
```
Hyperparameters: N=50 (playthroughs), k=3 (abstraction bound for intervals), coverage threshold θ=95%.

### (C) Why this design
We chose FSM induction over learning a neural state predictor because FSMs provide a finite, verifiable representation that abstract interpretation can soundly check; the trade-off is that FSM induction may miss rare states with few playthroughs, accepting that we may need to increase N or rely on coverage-guided exploration to ensure coverage. We chose abstract interpretation over runtime testing because it gives a static guarantee without executing the code on every possible input, but at the cost of precision loss due to overapproximation; however, for game state spaces that are discrete and bounded, this overapproximation is manageable. We chose targeted re-generation based on the most frequent violation rather than brute-force all violations because it reduces LLM calls while iteratively fixing the most impactful issues, but this may fail if violations are interdependent and fixing one introduces another. Finally, we used L* with coverage-guided traces as the equivalence oracle because game states are finite and reachable via targeted exploration, but this assumes the exploration strategy covers all relevant state-action pairs, which may not hold for adversarial sequences; we accept this as a pragmatic choice given no formal specification.

### (D) Why it measures what we claim
The computational quantity `concrete_post = δ(s,a)` from the induced FSM measures the ground-truth state transition expected by the game design, because the FSM is inducted from coverage-guided playthroughs that aim to cover all state-action pairs; this assumption fails when coverage is incomplete (e.g., a rare state is missed), in which case the FSM may omit transitions, causing false negatives in verification. The quantity `abstract_post = abstract_interpreter.run(C, s, a)` measures the set of possible next states the generated code can produce from state s under action a, because abstract interpretation provides a sound overapproximation of the code's semantics; this assumption fails when the abstraction bound k is too coarse (e.g., intervals ignore correlation between variables), causing overapproximation that may pass incorrect code (false negatives) or reject correct code (false positives). The check `check_inclusion(concrete_post, abstract_post)` measures whether the FSM's expected transition is within the code's guaranteed behavior; this equivalence holds under the assumption that the FSM is correct and the abstract interpreter is sound. By iterating until all violations are resolved, the method ensures the generated code is closed under hallucination with respect to the learned FSM, because the FSM captures all reachable states under coverage-guided exploration (the closure property).

## Contribution

(1) A method for verifying LLM-generated game code without a test suite by inducting a finite-state machine from random playthroughs and applying abstract interpretation. (2) A targeted re-generation mechanism that leverages FSM constraint violations to guide code fixing, reducing hallucinated behaviors. (3) A demonstration that game code generation can benefit from state-space learning and static verification, bridging the gap between unstructured generation and formal methods.

## Experiment

### Evaluation Setup
| Role | Choice | Rationale (≤12 words) |
|------|--------|----------------------|
| Dataset | GameBench: 20 text-based games with ground truth FSM | Tests state-based mechanics systematically |
| Primary metric | FSM violation rate (lower is better) | Directly measures code adherence to induced FSM |
| Baseline 1 | Direct GPT-4 generation (no verification) | Baseline without any constraint |
| Baseline 2 | LLM + random testing (100 playthroughs) | Tests without abstract interpretation |
| Baseline 3 | LangGraph iterative bug fixing (from related work) | State-of-the-art iterative method |
| Baseline 4 (new) | LLM + neural state predictor (trained on same playthroughs) | Compares FSM induction to neural prediction |
| Ablation-of-ours | Ours without abstract interpretation (only random testing) | Isolates contribution of abstract interpretation |
| Hyperparameter ablation | Vary N ∈ {20,50,100} and k ∈ {1,2,3,5} | Measure sensitivity to hyperparameters |

Additionally, conduct a case study on a stochastic game (e.g., simplified Blackjack) to test generality.

### Why this setup validates the claim
This design compares FSM-GenVer against baselines that lack its core components: direct generation (no verification), random testing (no static analysis), a prior iterative method (LangGraph), and a neural state predictor (no formal FSM). The ablation removes abstract interpretation to isolate its effect. The metric—FSM violation rate—directly tests the claim that the method reduces hallucinations by ensuring code respects the inducted FSM. If the method’s advantage stems from the combination of FSM induction and abstract interpretation, it should outperform all baselines on complex games with rare states, while the ablation should underperform the full method. The hyperparameter ablation (vary N and k) will show how sensitive the method is to these choices. This setup provides a falsifiable test: if the improvement is uniform across games or if the ablation matches the full method, the core mechanism is disconfirmed.

### Expected outcome and causal chain
**vs. Direct GPT-4** — On a case where the game requires a subtle state transition (e.g., an inventory combine that changes two variables), Direct GPT-4 often generates code that misses the combined effect, because it has no way to check consistency. Our method inducts an FSM from coverage-guided playthroughs; if the transition is covered, abstract interpretation will catch the omission and trigger regeneration. We expect a noticeably lower violation rate on complex games, with parity on simple ones.

**vs. LLM + random testing** — On a case where a rare state is only reachable via a specific action sequence, random testing may never hit that branch, so the code passes but contains a hallucination. Our method uses abstract interpretation to overapproximate the code’s behavior, covering that branch statically. We expect our method to have fewer false negatives, especially on games with conditional paths, while random testing may miss violations in those subsets.

**vs. LangGraph** — On a case where the game logic is syntactically correct but semantically wrong (e.g., an action decrements a counter when it should not), LangGraph relies on runtime errors or crashes to trigger bug fixing, so silent violations persist. Our method explicitly checks each transition against the inducted FSM, catching semantic mismatches. We expect our method to achieve higher FSM conformance, with a significant gap on games where bugs do not cause crashes.

**vs. Neural state predictor** — On a case where the neural predictor overfits or misses rare transitions due to insufficient data, the FSM induction (with coverage guidance) will have a more precise representation, leading to better verification. We expect our method to have lower violation rates, especially on games with complex state spaces.

**vs. Our ablation (no abstract interpretation)** — On a case with deep conditional branches, the ablation relies on random testing and may miss violations in uncharted states, while the full method’s abstract interpretation covers all branches. We expect the full method to have lower violation rates, demonstrating the value of static analysis.

**Hyperparameter sensitivity** — For N=20, FSM may be incomplete, leading to higher false negatives; for N=100, FSM is more accurate but induction cost increases. For k=1, abstraction is coarse, causing many false positives; for k=5, abstraction is finer but may not scale. We expect k=3 to be a good balance.

### What would falsify this idea
If our method does not significantly reduce FSM violation rates compared to the baselines, or if the reduction is uniform across all game complexities (rather than concentrated on games with rare or complex state transitions), then the core mechanism of FSM induction plus abstract interpretation is not effective. Specifically, if the ablation performs similarly to the full method, abstract interpretation adds no value. Also, if the neural state predictor baseline performs comparably or better, then FSM induction is not necessary.

## References

1. Empirical Research on Utilizing LLM-based Agents for Automated Bug Fixing via LangGraph
2. Hallucination by Code Generation LLMs: Taxonomy, Benchmarks, Mitigation, and Challenges
3. On Mitigating Code LLM Hallucinations with API Documentation
4. We Have a Package for You! A Comprehensive Analysis of Package Hallucinations by Code Generating LLMs
5. Collu-Bench: A Benchmark for Predicting Language Model Hallucinations in Code
6. LLM Hallucinations in Practical Code Generation: Phenomena, Mechanism, and Mitigation
7. What is wrong with your code generated by large language models? An extensive study
8. ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs
