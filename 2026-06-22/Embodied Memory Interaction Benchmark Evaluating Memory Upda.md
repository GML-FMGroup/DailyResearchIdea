# Embodied Memory Interaction Benchmark: Evaluating Memory Updates Through Action Success in Interactive Environments

## Motivation

Existing memory benchmarks, such as those used in OAgents, evaluate memory components in static settings where memory is updated only via text or passive observation, ignoring the embodied requirement that memory must be updated through physical actions that reveal hidden state. This structural gap prevents assessing how well agents ground memory in sensorimotor feedback, limiting progress toward interactive agents. We need a benchmark that directly couples memory accuracy with action success in environments where actions causally update memory.

## Key Insight

The causal necessity of memory for task success—where achieving a goal uniquely depends on correctly remembering a state that was only revealed through prior physical action—creates a natural evaluation signal that distinguishes grounded memory from memorization or heuristics.

## Method

**Embodied Memory Interaction Benchmark (EMIB)**

(A) **What it is**: EMIB is a benchmark protocol for evaluating memory in embodied agents. It measures memory through task success in simulated environments where object properties (e.g., color, shape, weight) are initially hidden and must be revealed via physical interactions (e.g., pushing, lifting, opening). Memory queries are interleaved with actions, and final task success requires accurate memory of revealed states. Tasks are pre-validated using an automatic procedure that exhaustively tests all possible non-memory strategies (random, heuristic, brute-force action sequences); only tasks where the success rate of any such strategy is below 5% are retained.

(B) **How it works** (pseudocode):
```
# Pre-validation: all tasks in EMIB-TaskSet pass the non-memory strategy check
for each episode:
  # Initialize environment with hidden object properties
  env = HabitatManipulaTHOR(seed)
  objects = generate_objects(seed, num_objects=3, hidden_prop=color, dynamic_updates=True)  # dynamic: property may change after action sequences (e.g., two pushes)
  # Example task: "Push the blue object, then pick up the object that became red."
  task = sample_task(seed)
  
  # Agent interacts for a fixed number of steps (max_steps=20)
  for step in range(max_steps):
    action = agent.get_action(observation, memory)
    observation, reward, done = env.step(action)
    # Action reveals hidden property if applicable
    if action.type == 'push' and action.target in objects:
      object[action.target].color = reveal_color(observation)
    # Dynamic property update: after two pushes on same object, color toggles
    if action.type == 'push' and action.target in objects:
      object[action.target].push_count += 1
      if object[action.target].push_count == 2:
        object[action.target].color = toggle_color(object[action.target].color)
    # Optionally interleave memory query (every 3 steps)
    if step % 3 == 0:
      query = sample_memory_query(objects)  # e.g., "What color is object A?"
      answer = agent.memory_query(query)
      # Answer not used for scoring but can be recorded
    if done: break
  # Final step: agent must act based on memory (e.g., pick object with specific color)
  final_action = agent.get_action(observation, memory)
  success = evaluate_task_success(task, final_action, env)
  record(episode_success=success, memory_accuracy=optional)
```
Hyperparameters: `max_steps=20`, `memory_query_interval=3`, `num_objects=3`, `dynamic_updates=True`.

(C) **Why this design**: We chose task success as the primary metric over direct memory recall accuracy because recall can be gamed by statistical guessing, whereas task success requires memory to be causally integrated into action; the trade-off is that success conflates memory with planning ability, but we mitigate by designing tasks where success hinges on a single memory-dependent decision (e.g., picking the right object). We chose interleaved memory queries (rather than post-hoc questioning) to test memory update dynamics—this adds realism but risks interfering with the agent's action stream; we mitigate by making queries optional and not scored. We chose simulation over real-world deployment for controlled reproducibility; the trade-off is sim-to-real gap, but we use high-fidelity simulators (ManipulaTHOR) with physics and partial observability, which retain key challenges. Task pre-validation ensures that success cannot be achieved by non-memory strategies, strengthening the causal link.

(D) **Why it measures what we claim**: The computational quantity `episode_success` measures memory grounding (the motivation-level concept) because success requires the agent to maintain a memory representation that is causally necessary for the correct final action; this assumption fails when the agent uses a heuristic (e.g., always picks the first object), in which case `episode_success` measures heuristic adherence, not memory. The `memory_query` responses (optional) measure memory update correctness, but they are not part of the primary score to avoid rewarding recall without grounding—they serve as diagnostic. The `task_success` metric's causal link to memory relies on the task design where hidden properties are only revealed through specific actions, and the final action uniquely depends on that revealed property; this assumption fails when alternative strategies (e.g., brute-force trial and error) can bypass memory, so we design tasks where such strategies are impossible (e.g., objects disappear after one interaction). The pre-validation step explicitly checks that no non-memory strategy exceeds a 5% success rate, ensuring the metric reflects memory grounding.

## Contribution

(1) The Embodied Memory Interaction Benchmark (EMIB), a protocol that evaluates agent memory through success on tasks where memory is causally required for action, integrating physical interaction loops. (2) A design principle for embodied memory evaluation: memory should be measured by its functional utility in achieving goals, not by recall accuracy alone. (3) A set of 50 benchmark tasks in ManipulaTHOR with hidden object properties, along with evaluation code and human baselines.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|------------------------|
| Dataset | EMIB-TaskSet (ManipulaTHOR) | Tests memory via physical interaction |
| Primary metric | Task success rate | Directly measures memory grounding |
| Baseline1 | Random agent | Lower bound, no memory |
| Baseline2 | Heuristic (always pick first) | Failure on memory-dependent tasks |
| Ablation-of-ours | EMIB w/o interleaved queries | Isolates querying effect on memory |
| Ablation2 (controlled corruption) | EMIB with corrupted memory (p=0.3) | Verifies memory necessity by degrading |

### Why this setup validates the claim

The EMIB-TaskSet is designed such that hidden object properties (e.g., color, shape) are revealed only through specific actions, and the final task requires recalling those properties. Task success rate is the primary metric because it directly tests whether memory is causally integrated into action, unlike raw recall which can be gamed. The random baseline provides a lower bound, showing that memory is necessary. The heuristic baseline (always picking the first object) fails when the correct object is not first, isolating memory requirement. The ablation (removing interleaved memory queries) tests whether periodic queries aid memory updates; if our method outperforms this ablation, it confirms that interleaved queries improve memory maintenance. Additionally, the controlled memory corruption ablation (randomly corrupting stored memory with 30% probability during test) provides a direct causal probe: if task success drops significantly under corruption, it confirms that success relies on accurate memory. This combination forms a falsifiable test: if our method does not outperform the heuristic on tasks where memory is crucial, the central claim that memory grounding is effective fails. The metric also ensures that success is not due to heuristics or brute force, reinforced by pre-validation of tasks.

### Expected outcome and causal chain

**vs. Random agent** — On a case where the agent must push an object to reveal its color, then later pick that color, the random agent fails because it has no memory (chooses randomly). Our method succeeds because it updates memory after push and recalls the color at decision time. We expect a large gap (e.g., >80% vs. ~30%) on tasks requiring memory, with parity on tasks not requiring memory.

**vs. Heuristic (always pick first)** — On a case where the correct object is not first (e.g., second object revealed as red), the heuristic fails because it ignores memory. Our method remembers the revealed color and selects correctly. We expect a noticeable gap (e.g., >80% vs. ~50%) on tasks where correct object is non-first, and parity otherwise.

**vs. Ablation2 (corrupted memory)** — On tasks requiring memory, corrupting memory should cause a significant drop (e.g., from >80% to ~40%), confirming that memory is causally responsible for success.

### What would falsify this idea

If our method's task success rate is similar to the heuristic baseline on tasks where the correct object is not first, or if the ablation shows equal performance to our full method, or if the memory corruption ablation does not degrade performance, then the central claim that interleaved memory queries and memory grounding are necessary would be falsified.

## References

1. OAgents: An Empirical Study of Building Effective Agents
2. OS-Copilot: Towards Generalist Computer Agents with Self-Improvement
3. CogAgent: A Visual Language Model for GUI Agents
4. OpenAgents: An Open Platform for Language Agents in the Wild
5. Pix2Struct: Screenshot Parsing as Pretraining for Visual Language Understanding
6. WebShop: Towards Scalable Real-World Web Interaction with Grounded Language Agents
