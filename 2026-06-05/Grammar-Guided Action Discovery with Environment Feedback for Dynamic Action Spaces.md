# Grammar-Guided Action Discovery with Environment Feedback for Dynamic Action Spaces

## Motivation

Existing LLM-based planners like MC-DML assume a fixed set of valid actions, which precludes the discovery of novel action compositions that emerge from environmental dynamics (e.g., combining verbs with new objects). This limitation arises from the structural assumption that the environment provides a static enumeration of actions, whereas in open-ended games the action space can expand as new objects or interactions are revealed. Without the ability to generate and validate candidate actions beyond a predefined list, agents are fundamentally constrained to known behaviors, limiting generalization.

## Key Insight

Environment execution provides a verifiable oracle that filters infinite grammatical action combinations into a valid, grounded set, enabling dynamic action space expansion without manual enumeration.

## Method

### (A) What it is
**GCEF** (Grammar-Constrained Exploration with Environment Feedback) is a method that dynamically extends the action space of an LLM-based planner by generating candidate actions via a context-free grammar and retaining only those that produce a measurable state change when executed in the environment. Input: current game state (text), goal, a predefined grammar G (e.g., verb + object), and the environment simulator. Output: a set of discovered valid actions and a policy for selection.

### (B) How it works
```pseudocode
Initialize action set A = fixed basic actions (look, inventory, etc.)
For each step:
  1. State s = observe environment text
  2. Grammar G = expand_grammar(s) // add objects discovered in s as terminals
  3. Candidate set C = LLM_generate(s, goal, G, n=20)
     // LLM produces actions like "(verb, object)" constrained by G
  4. For each c in C:
     - Execute c in environment (simulate or real)
     - If outcome causes measurable state change (change in inventory, location, or reward) and c not in A:
         Add c to A with a prior value (e.g., 0.5)
  5. Select action a* = argmax_{a in A} Q(a) where Q is from MC-DML
     // MC-DML is used for planning over the dynamically updated A
  6. Execute a*, observe next state and reward, update Q
```
*Hyperparameters:* n=20 candidate samples; prior value 0.5; grammar format: verb-object pairs with non-terminals <verb> (drawn from a predefined list of 20 common verbs: take, open, push, pull, examine, etc.) and <object> (any noun phrase extracted from state description via part-of-speech tagging).

### (C) Why this design
We chose grammar-constrained generation (over free-form generation) because it yields syntactic valid actions that can be parsed by the environment, reducing the rejection rate. The trade-off is that the grammar must be predefined, limiting the generation space to known verb-object patterns; however, the grammar is dynamically expanded when new objects are observed (via simple pattern matching on state descriptions), which balances coverage and feasibility. We use environment feedback as a validation filter (rather than LLM self-evaluation) because execution provides a ground-truth signal that avoids the calibration issues of LLM confidence; the cost is additional environment calls (one per candidate). We integrate discovered actions into MC-DML's planning (by adding them to the action set with a prior Q-value) rather than replacing the planner entirely, because MC-DML already provides efficient tree search over a fixed set; this hybrid approach inherits the search efficiency while enabling action discovery. The trade-off is that newly discovered actions start with an uninformed prior, which may misguide search until they are tried multiple times. A crucial assumption underlying the feedback filter is that a measurable state change reliably indicates a meaningful action; we operationalize this as changes in inventory, location, or reward to avoid false positives from generic responses.

### (D) Why it measures what we claim
The number of valid actions added to A after filtering measures the degree of action space expansion because each added action corresponds to a novel interaction that the environment recognized as meaningful. This assumption fails when the environment returns non-null outcomes for semantically empty actions (e.g., 'touch wall' might produce a generic message), in which case the metric reflects false positives rather than genuine discoveries. To mitigate, we employ a measurable state change filter that checks for changes in inventory, location, or reward, ensuring that only actions causing a detectable effect are counted. The prior Q-value of 0.5 for discovered actions operationalizes the concept of exploration incentive because it encourages the planner to try new actions over untried ones (if the standard MC-DML initialization is lower, e.g., 0); this design assumes that effective exploration requires a moderate initial value, but if the environment is highly stochastic, the prior may lead to premature exploitation of useless actions. The grammar expansion step (adding observed objects) measures environmental engagement because it ensures that newly encountered entities are immediately available for action generation; the assumption is that objects are always mentioned in state descriptions, which fails when objects are implied (e.g., a door is present but not described), leading to missed discovery opportunities.

## Contribution

1. A novel framework, GCEF, that dynamically expands the action space of LLM-based planners by combining grammar-constrained generation with environment validation, enabling discovery of novel actions. 2. A concrete integration of action discovery into MC-DML without modifying its core planning mechanism, demonstrating that the fixed-action limitation can be overcome while retaining search efficiency.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | Jericho text game benchmark | Standard for planning in text games |
| Primary metric | Success rate over 5 games | Measures ability to complete tasks |
| Baseline1 | MC-DML | Strong planning baseline without action discovery |
| Baseline2 | Chain-of-Thought | Standard prompting without environment grounding |
| Ablation-of-ours | GCEF w/o feedback | Tests necessity of environment filtering |

### Why this setup validates the claim

The central claim of GCEF is that grammar-constrained generation combined with environment feedback (using measurable state change) and integration into MC-DML enables effective action space expansion and planning. To test this, we compare against MC-DML (no action discovery) and Chain-of-Thought (no planning or grounding), plus an ablation that omits feedback. The Jericho benchmark provides diverse text games with valid action spaces. Success rate directly measures task completion, requiring both useful action discovery and planning. If GCEF outperforms MC-DML, it shows the benefit of action discovery; outperforming CoT demonstrates the need for planning; the ablation reveals the contribution of feedback filtering. The use of measurable state change filter ensures that the metric more accurately reflects genuine discoveries.

### Expected outcome and causal chain

**vs. MC-DML** — On a game like 'acorncourt' where a crucial action like 'open drawer' is not in the initial set, MC-DML never tries it because its action space is fixed, leading to stuck states and failure. Our method generates candidate actions via grammar, executes 'open drawer', detects a measurable state change (e.g., 'You open the drawer. There is a key inside.'), adds it to the action set with prior 0.5, and then MC-DML can include it in planning. Thus, we expect a noticeable gap in success rate on games requiring exploration of novel object interactions (e.g., +20% on those games) but parity on games where basic actions suffice.

**vs. Chain-of-Thought** — On a multi-step task like 'cooking breakfast' that requires fetching, preparing, and combining items, CoT generates a plan in a single forward pass without environment feedback, often producing invalid or suboptimal sequences (e.g., trying to fry an egg before obtaining it). Our method interleaves planning and execution via MC-DML, adapting to state changes and using feedback to refine plans. We expect GCEF to significantly outperform CoT on long-horizon tasks (e.g., >30% higher success) while being comparable on simple single-step tasks.

### What would falsify this idea

If GCEF fails to outperform MC-DML on exploration-heavy games (i.e., similar success rates), the claim that action discovery via grammar and feedback is beneficial would be unsupported. Alternatively, if the ablation without feedback performs as well as the full method, the necessity of environment filtering (especially the measurable state change criterion) is refuted. Furthermore, if the measurable state change filter yields no improvement over the non-null outcome filter (i.e., false positive rate remains high), the core assumption that state change indicates meaningful action would be undermined.

## References

1. Monte Carlo Planning with Large Language Model for Text-Based Game Agents
2. Everything of Thoughts: Defying the Law of Penrose Triangle for Thought Generation
3. Ghost in the Minecraft: Generally Capable Agents for Open-World Environments via Large Language Models with Text-based Knowledge and Memory
4. Large Language Models as Commonsense Knowledge for Large-Scale Task Planning
5. Self-Consistency Improves Chain of Thought Reasoning in Language Models
6. Training language models to follow instructions with human feedback
7. Chain of Thought Prompting Elicits Reasoning in Large Language Models
8. Code as Policies: Language Model Programs for Embodied Control
