# Adversarial Benchmark Synthesis: Generating Valid Agent Benchmarks from Formal Domain Specifications

## Motivation

Existing automated benchmark design methods such as BeTaL rely on human-provided base templates, and best-practice checklists for agentic benchmarks are manually curated. This reliance on human foundations creates a structural gap: no mechanism exists to derive a benchmark template or checklist from first principles, forcing dependence on human intuition. To close this gap, we need a method that starts from a formal specification of the task domain (actions, observations, rewards) and automatically produces both valid benchmark instances and a validated checklist, eliminating the need for human-provided foundations.

## Key Insight

Because the formal domain specification defines the complete space of possible agent interactions, adversarial probing can provably expose any failure mode that a benchmark instance might miss, ensuring validity without human templates.

## Method

### (A) What it is
Adversarial Benchmark Synthesis (ABS) is a generative-adversarial framework that takes a formal domain specification (a tuple ⟨S, A, O, R, T⟩) and outputs a set of benchmark instances and a failure-mode checklist. The generator proposes candidate benchmarks within the spec, and probe agents attempt to find policy violations; failures are used to refine the benchmark and populate the checklist. The probe set is not fixed: starting from an initial set of five probe types, new probe strategies are synthesized from failure descriptions to ensure coverage of unseen failure modes.

### (B) How it works
```python
Input: DomainSpec = (S, A, O, R, T)   # state space, actions, observations, reward, transition
Parameters:
  initial_probe_types = ['random', 'adversarial_llm', 'greedy', 'bounded_optimal', 'curriculum']
  probe_types = initial_probe_types.copy()
  max_iterations = 100
  violation_threshold = 0.1  # reward threshold below which considered violation
  high_threshold = 0.95 * max_theoretical_reward   # e.g., 0.95 of maximum possible reward derived from spec
  low_threshold = 0.1 * max_theoretical_reward
  probe_synthesis_llm = initialize_llm(spec)  # LLM to generate new probes from violation descriptions

Initialize:
  benchmark_set = []           # list of (initial_state_sampler, reward_modifier, termination_condition)
  checklist = []               # list of failure_mode descriptions
  generator = initialize_llm_prompt(spec)   # or random sampler

for iteration in range(max_iterations):
  candidate = generator.propose()   # e.g., by sampling initial state distribution, or modifying reward function within spec
  for probe_type in probe_types:
    probe_agent = create_probe(probe_type, spec)
    performance = evaluate(probe_agent, candidate, num_episodes=50)
    if performance > high_threshold or performance < low_threshold:
      violation_description = describe_violation(probe_agent, candidate, performance)
      checklist.append(violation_description)
      # Synthesize a new probe type from the violation description and add to probe_types
      new_probe_type = probe_synthesis_llm.generate(description=violation_description, existing_probes=probe_types)
      probe_types.append(new_probe_type)
      generator.update(negative_feedback=candidate, violation_description)
      break  # move to next candidate
  if no violation detected after processing all probe_types:
    benchmark_set.append(candidate)
    # optionally add positive checklist items
```

Assumption: The initial probe types capture a useful set of failure modes, and the adaptive synthesis mechanism ensures that any failure mode missed by the initial set will be discovered and encoded as a new probe type, thereby achieving coverage of all possible failure modes expressible in the domain specification.

### (C) Why this design
We chose adversarial probing over simple random sampling because probes can systematically exploit loopholes that random agents might miss. We used multiple probe types (random, adversarial LLM, greedy, bounded optimal, curriculum) to cover diverse failure modes — greedy probes detect reward hacking, bounded optimal probes detect suboptimality, adversarial LLM probes can propose creative exploits. We added adaptive probe generation to address the limitation that a fixed set may miss emergent exploits; by synthesizing new probes from detected failures, we iteratively expand coverage. The trade-off is computational cost: running many probes on each candidate increases runtime, but this is acceptable for offline benchmark generation. We chose LLM-based generator over rule-based generation because the formal spec is high-level and LLMs can propose diverse initial conditions. However, LLMs may hallucinate out-of-spec elements, so we enforce spec consistency via a validator. The iterative refinement loop with negative feedback ensures the generator learns from failures, akin to adversarial training. We avoided a single monolithic probe because different agents have complementary blind spots; using five distinct types increases coverage. This design differs from LLM-POET, which uses co-evolution of environments and agents without a formal spec or a fixed validity target; our method explicitly validates against a spec and produces a checklist.

### (D) Why it measures what we claim
The number of probes that fail to detect a violation (false negatives) measures the completeness of the benchmark because each probe type is designed to expose a specific class of failure; if all probes pass, the benchmark is likely valid. The generator's iterative refinement based on violation descriptions ensures that the final benchmark set contains only instances that are robust to all probe strategies, operationalizing the concept of "no human-provided foundations" by replacing human intuition with exhaustive adversarial testing. Specifically, computational quantity `performance > high_threshold` measures unintended exploit because the assumption is that a probe agent that achieves reward significantly above the intended maximum indicates a loophole; this assumption fails when the reward function is poorly specified, in which case high performance may actually reflect intended behavior. Similarly, `performance < low_threshold` measures unintended difficulty because the assumption is that a probe agent should achieve at least some minimal reasonable performance if the benchmark is solvable; this assumption fails when the probe strategy is fundamentally mismatched, in which case low performance reflects probe incompetence rather than benchmark flaw. `checklist.append(violation_description)` measures the set of failure modes because each violation description is derived from a concrete interaction; the assumption is that the probe types collectively cover all possible failure modes expressible in the domain, which fails when a failure mode requires a combination of strategies not captured by any single probe type, in which case that failure mode is missed. The adaptive probe synthesis mechanism aims to close this gap by generating probes that combine strategies.

## Contribution

(1) A novel adversarial benchmark synthesis framework that generates benchmark instances and failure-mode checklists from formal domain specifications, eliminating the need for human-provided templates or checklists. (2) A design principle that adversarial probing with diverse agent strategies can provably validate benchmark correctness within the spec, establishing a foundation for automated benchmark generation. (3) A methodology for deriving best-practice checklists as a byproduct of the adversarial refinement process, providing a systematic alternative to manual curation.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|------------------------|
| Dataset | Three formal domain specs | Diverse domains test generality. |
| Primary metric | False negative rate of violations | Directly measures completeness claim. |
| Baseline 1 | Random parameter sampling | No adversarial probing. |
| Baseline 2 | LLM-POET co-evolution | No formal spec enforcement. |
| Baseline 3 | Single probe type (greedy) | Tests need for multiple probes. |
| Ablation | ABS without iterative refinement | Tests refinement loop vs. single pass. |

### Why this setup validates the claim

This experimental design directly tests the central claim that ABS generates robust benchmarks with minimal false negatives. By evaluating across three distinct domains, we assess generality. The primary metric, false negative rate, quantifies the core promise: that adversarial probing catches all violations. Random sampling baseline isolates the value of probing; LLM-POET baseline tests the necessity of formal spec validation; single-probe baseline tests the need for diverse probe types. The ablation of iterative refinement reveals whether the adversarial training loop is essential. Together, these comparisons create a falsifiable test: if ABS outperforms all baselines on false negative rate, the claim is supported; otherwise, the mechanism is incomplete.

### Expected outcome and causal chain

**vs. Random parameter sampling** — On a case where the generator proposes a candidate with a subtle reward-hacking opportunity (e.g., a state where an agent can get infinite reward by looping), random sampling will likely miss this violation because random agents do not systematically exploit loops. Our method uses an adversarial LLM probe that creatively explores such exploits, so we expect ABS to have a noticeably lower false negative rate (e.g., 10% vs 60%) on such reward-hacking instances.

**vs. LLM-POET co-evolution** — On a case where the environment is complex (e.g., multi-step assembly task), LLM-POET may generate unsolvable instances because it lacks a formal spec to enforce solvability; its co-evolution may produce environments that are too hard. Our method validates each candidate against the spec using bounded-optimal probes, ensuring solvability. Thus we expect ABS to have a lower false negative rate on unsolvable instances (e.g., 5% vs 40%) while maintaining similar or better difficulty targeting.

**vs. Single probe type (greedy)** — On a case where a benchmark exploits exploratory failure (e.g., requires deep exploration to find reward), a greedy probe will achieve low performance and misclassify it as unsolvable, but a curriculum probe can gradually learn. Our method uses multiple probe types including curriculum, so it correctly identifies solvable benchmarks. We expect ABS to have a lower false negative rate on exploration-dependent instances (e.g., 15% vs 50%).

### What would falsify this idea
If ABS does not achieve a substantially lower false negative rate than the single-probe baseline on instances requiring diverse failure modes (e.g., exploration and reward hacking simultaneously), or if iterative refinement does not improve over a single pass, then the central claim—that adversarial probing with multiple types and refinement ensures benchmark completeness—is false.

## References

1. Automating Benchmark Design
2. LLM-POET: Evolving Complex Environments using Large Language Models
3. Automatic Generation of Benchmarks and Reliable LLM Judgment for Code Tasks
4. APIGen: Automated Pipeline for Generating Verifiable and Diverse Function-Calling Datasets
5. CrossCodeEval: A Diverse and Multilingual Benchmark for Cross-File Code Completion
6. SWE-bench: Can Language Models Resolve Real-World GitHub Issues?
7. Evolution Gym: A Large-Scale Benchmark for Evolving Soft Robots
8. MarioGPT: Open-Ended Text2Level Generation through Large Language Models
