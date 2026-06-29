# Tool-Discovery Arena: An Open-Ended Benchmark for Tool-Set Independent Computer-Use Agents

## Motivation

Both API-Bank (fixed 73-API set) and the GUI vs CLI benchmark (predefined desktop skills) evaluate agents on a closed tool set, structurally preventing assessment of tool discovery—the ability to identify and integrate novel tools during task execution. This closed-world assumption leads to agents that memorize API calls rather than adapt to dynamic environments. For example, the GUI vs CLI benchmark's skill augmentation is manual and only adds missing CLI skills, never requiring the agent to discover a tool from scratch.

## Key Insight

Task success becomes independent of tool set memorization when the mapping from environmental observations to tool utility must be inferred by the agent via descriptive queries to a continuously expanding library, rather than selected from a fixed list.

## Method

# Tool-Discovery Arena: An Open-Ended Benchmark

**Load-bearing assumption:** The agent's descriptive query, generated from visual observation of a novel object, reliably captures the object's functional affordance such that a dense retriever can return a matching skill description from the library.

## (A) What it is

Tool-Discovery Arena (TDA) is a benchmark protocol and evaluation framework consisting of: (i) a task generator that creates tasks requiring novel, unseen tools; (ii) a simulated environment where tools appear as interactive but initially unlabeled objects; (iii) a retrieval-augmented skill library (RASL) that starts empty and grows when agents successfully use new tools; (iv) a verifier that checks task completion independent of tool use. Input: a textual task description and initial environment state. Output: a pass/fail verdict and an updated RASL.

## (B) How it works (pseudocode)

```python
def TDA_episode(agent, env, RASL, task_generator):
    # Phase 1: Task generation with novel tool
    task_description, required_novel_tool = task_generator.sample()
    env.reset(novel_tool=required_novel_tool)  # unknown to agent
    
    # Phase 2: Agent execution loop
    while not env.task_complete() and steps < max_steps:
        observation = env.get_observation()  # screenshot or state
        
        # Agent decides action: either interact with object or retrieve from RASL
        if agent.detect_novel_object(observation):
            # Generate descriptive query from observation using frozen LLM (e.g., GPT-4, temperature=0.2)
            query = agent.describe_object(observation["novel_object"])
            retrieved_skills = RASL.retrieve(query, top_k=3)  # standard dense retrieval (e.g., using Sentence-BERT)
            # **Confirmation step: test retrieved skill before use**
            action = None
            for skill in retrieved_skills:
                # Simulate skill execution in a sandboxed environment (or ask LLM to verify)
                if agent.confirm_skill(skill, observation):
                    action = skill
                    break
            if action is None:
                # Fall back to direct interaction (exploratory actions)
                action = agent.direct_interaction(observation)  # click, type, etc.
        else:
            action = agent.direct_interaction(observation)
        
        env_step_result = env.step(action)
        
        # Phase 3: Skill addition on successful novel tool use
        if env_step_result.succeeded and action.used_novel_tool:
            skill_description = agent.describe_skill(action, env_step_result)
            # Add to RASL only if skill is non-redundant (e.g., similarity < 0.9 with existing entries)
            RASL.add(skill_description)
    
    return env.task_complete()  # verifier checks final state
```
Hyperparameters: `top_k=3` (retrieval count), `max_steps=100`, query generation uses a frozen LLM (e.g., GPT-4) with temperature 0.2, confirmation step uses the same LLM with a prompt to verify functional fit, skill addition threshold cosine similarity < 0.9 to avoid duplicates.

## (C) Why this design

We chose to make the skill library start empty and grow through agent discovery rather than pre-populated because an empty initial library forces genuine exploration—agents cannot rely on a fixed skill set, directly addressing the tool-set independence gap. This design accepts the cost that early tasks may have low success rates as agents learn to generate effective queries. We chose to have the agent generate a descriptive query from the novel object observation instead of using the object's identifier (which is unknown) because the description acts as a bridge between visual affordance and functional skill, whereas an identifier would require a pre-existing mapping. The trade-off is that query quality depends on the agent's ability to articulate relevant features; poor queries reduce retrieval accuracy. We added a confirmation step (testing retrieved skill before use) to mitigate retrieval failures arising from the load-bearing assumption that descriptions capture affordances. We chose to add skills only upon successful use of a novel tool (not after any attempt) to avoid polluting the library with erroneous or incomplete skill descriptions, at the cost that some skills may be learned slowly if failures go unrecorded. This design contrasts with standard retriever-based tool selection (e.g., Toolformer) where the tool set is fixed and retrieved identifiers; here, retrieval queries are generated from environmental perception, not from task context alone, which is a structural difference preventing collapse to prior retrieval-based methods.

## (D) Why it measures what we claim

The central concept to measure is tool-set independence—the extent to which an agent's performance depends on pre-existing knowledge of tools versus its ability to discover them. `RASL.retrieve(query, top_k=3)` measures tool discovery skill because the query is derived from a novel object the agent has never seen; the assumption is that the agent's description captures the tool's functional affordance, allowing retrieval of a matching skill description from the library (even if the library contains only previously discovered skills). This assumption fails when the agent's description is too vague or focuses on irrelevant visual features, in which case retrieval measures only the similarity between the description and stored skill texts, not actual functional match. The confirmation step partially addresses this by testing retrieved skills before use, making success contingent on functional match. `env_step_result.used_novel_tool` and success condition measure whether the agent applied a tool in a way that solved the task; the assumption is that any successful novel tool use implies the agent discovered and utilized a new capability. This assumption fails when the novel tool is accidentally used (e.g., random click that completes the task), in which case success reflects luck rather than discovery. To control for this, we require multiple successful uses (at least 2) of the same novel tool within the episode to count as true discovery. The growing RASL measures cumulative discovery breadth; the assumption is that each addition represents a distinct, reusable skill. This fails when two very similar tools lead to duplicate entries, overstating diversity; we mitigate this by deduplication (cosine similarity < 0.9 threshold). Together, these components operationalize tool discovery and independence by tracking query generation quality, successful application, and library growth, with explicit failure modes and controls for each proxy.

## Contribution

(1) A benchmark protocol that generates tasks with previously unseen tools, requiring agents to discover and integrate them during execution, unlike prior fixed-tool benchmarks. (2) A retrieval-augmented skill library (RASL) that grows via a novel tool-to-skill mapping from agent observations, enabling evaluation of tool-set independence. (3) An analysis framework to measure the gap between agent performance in fixed vs. open tool sets, providing a new dimension for agent evaluation.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|----------------------|
| Dataset | TDA-constructed tasks with 5–10 tool types (initial) expanding to 50+ | Tests tool discovery under controlled novelty; gradual expansion ensures feasibility |
| Primary metric | Task success rate on novel-tool tasks (averaged over 100 episodes) | Measures ability to use unknown tools |
| Baseline 1 | GUI agent (screen-only) clicks/typing, no tool abstraction | No tool knowledge, relies on direct interaction |
| Baseline 2 | Pre-populated skill library agent (fixed set of 20 skills) | Fixed tool set, no discovery required |
| Ablation-of-ours | TDA without RASL (no skill library) | Isolates effect of growing skill library |
| Ablation: query isolation | TDA with frozen query generation but no confirmation step | Isolates query generation vs. confirmation |
| Ablation: retrieval isolation | TDA with random skill selection instead of retrieval | Isolates retrieval quality |

### Why this setup validates the claim

This setup validates the claim by directly comparing agents along the tool-dependency axis. The dataset ensures tasks require truly novel tools, measuring discovery. The GUI baseline lacks any tool abstraction, testing raw interaction. The pre-populated baseline tests reliance on fixed tool knowledge. Our ablation isolates the skill library's contribution. Additional ablations (query isolation, retrieval isolation) decompose the method's components to measure each's contribution to success. The metric—success on novel-tool tasks—captures the core capability. Differences between these conditions reveal whether tool-set independence arises from the growing library mechanism or from other factors, forming a falsifiable test. To control for accidental success, we require at least 2 successful uses of the same novel tool in an episode for it to count as discovered (relaxed to 1 for known tools).

### Expected outcome and causal chain

**vs. GUI agent (screen-only)** — On a case requiring a novel tool (e.g., drag slider), GUI agent randomly interacts, failing because it lacks tool abstraction. Our method detects the object, generates a descriptive query, retrieves or learns a skill, and succeeds. We expect our method to achieve >50% success vs GUI's <20%, with gap increasing as library grows. On the subset of tasks requiring multiple novel tools (e.g., 3 tools), the GUI agent's success drops to ~5%, while ours maintains >30%.

**vs. Pre-populated skill library agent** — On a case with a tool not in its fixed set (e.g., a novel gesture), the pre-populated agent tries to apply known tools incorrectly, causing failure. Our method discovers the tool and adds it to RASL, enabling future use. We expect our method to outperform notably on truly novel tools (e.g., 2× success), but perform similarly on tools already in the pre-populated set (within 5% margin). The observable signal: a large gap on novel-tool tasks (e.g., 45% vs 22%) but near parity on known-tool tasks (e.g., 80% vs 78%).

**Ablation: w/o RASL** — Without the skill library, performance falls to GUI levels on novel tools; on easy tasks it may match, but overall success rate is ≤20% on novel tasks. This confirms the library's role.

**Ablation: query isolation** — Without confirmation step, success on novel-tool tasks drops by ~10–15% due to retrieval failures, showing the importance of verification.

**Ablation: retrieval isolation** — Random skill selection yields near-zero success on novel tools, confirming that retrieval is essential.

### What would falsify this idea

If our method's success rate on novel-tool tasks is not significantly higher than the GUI baseline (i.e., <10% absolute improvement), or if the improvement is uniform across all tasks rather than concentrated on novel-tool tasks (e.g., similar gap on known-tool tasks), then the central claim of tool discovery is unsupported. Additionally, if the ablation without RASL shows comparable performance to the full method on novel tools, the library is not causal.

## References

1. GUI vs. CLI: Execution Bottlenecks in Screen-Only and Skill-Mediated Computer-Use Agents
2. API-Bank: A Comprehensive Benchmark for Tool-Augmented LLMs
3. Few-shot Learning with Retrieval Augmented Language Models
