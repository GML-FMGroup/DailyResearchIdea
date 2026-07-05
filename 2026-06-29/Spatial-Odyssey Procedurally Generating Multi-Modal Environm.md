# Spatial-Odyssey: Procedurally Generating Multi-Modal Environments for Sensorimotor Continual Learning

## Motivation

AgentOdyssey (2026) generates open-ended text games for continual learning but remains text-only, failing to test sensorimotor abilities essential for embodied agents. This limitation arises because the grammar-based generation lacks spatial structure: entity interactions are purely symbolic without positions, movement, or collision, making it impossible to evaluate policies that require spatial reasoning or physical action. As a result, agent evaluation in this paradigm cannot distinguish between improvements in language understanding and genuine sensorimotor adaptation.

## Key Insight

By decoupling the underlying physical state from the observation modality (text or map), a single procedural generation engine can produce diverse sensorimotor tasks without compromising scalability, because the spatial grammar provides a deterministic world model that both text and map observations are derived from.

## Method

**What it is.** Spatial-Odyssey is a procedural generation framework that extends AgentOdyssey's grammar to include spatial primitives (positions, adjacency, collision, movement), producing multi-modal environments where a consistent 2D grid world underlies both natural language descriptions and optional map representations. The framework generates long-horizon tasks (e.g., navigate to object X, collect keys, avoid obstacles) with procedurally arranged rooms and entities.

**How it works.** The generation follows three phases:

```pseudocode
# Phase 1: Spatial Graph Construction
function generate_spatial_grammar(params):
    rooms = generate_rooms(num_rooms, min_size, max_size)  # e.g., 5 rooms, size 4x4-8x8
    adjacency = generate_adjacency(rooms, connectivity=0.6)  # random undirected edges
    for each room: place_entities(entities_from_agentodyssey_grammar, positions_within_room)
    return spatial_graph(rooms, adjacency, entities_with_positions)

# Phase 2: Observation Templates
function derive_observations(spatial_graph, agent_position):
    current_room = room_containing(agent_position)
    # Text: template-based description from entities, positions, exits
    text = f"You are in {current_room.name}. You see {list_entities(current_room)}. Exits: {list_adjacent_rooms(current_room)}."
    # Map: 2D integer grid (0=empty, 1=entity, 2=obstacle, 3=exit) centered at agent
    map = render_2d_grid(spatial_graph, view_radius=5, agent_position)  # partial observability
    return (text, map)

# Phase 3: Action Effects (deterministic)
function apply_action(action, state):
    if action == "move N": new_pos = (agent.x, agent.y-1) if not blocked
    elif action == "move E": ...
    elif action == "interact": trigger_entity_effect(entity at position)
    update_state(new_pos, entity_states)
    return new_text_and_map
```

**Why this design.** We chose spatial graph over continuous physics because it balances expressiveness (rooms, paths, positioning) with scalability (states are discrete and countable, avoiding floating-point drift). We chose template-based text instead of free-form LLM-generated descriptions to ensure that observations are perfectly grounded in the underlying state, preventing hallucination that would confuse agents. We chose partial observability (view radius) because real environments are never fully visible, forcing agents to explore and maintain memory. Accepting the cost that templates produce repetitive language, we prioritize diagnostic clarity over linguistic variety. We chose deterministic action effects to make learning credit assignment straightforward, at the cost that the world is less unpredictable than real physics.

**Why it measures what we claim.** The spatial graph `adjacency` and `entity_positions` operationalize the environment's physical structure; `view_radius` controls the degree of partial observability, which directly measures **spatial exploration coverage** (fraction of cells visited) because the agent must actively move to expand its map. The `text` observation, when templated, ensures a bijection between language tokens and world state (one entity per noun phrase, one room per prepositional phrase), so **action diversity** in the text modality reflects true behavioral variation rather than linguistic paraphrase. The assumption that template accuracy holds fails when the spatial grammar produces ambiguous layouts (e.g., two entities with the same name in one room), in which case `text` contains lists that do not disambiguate referents, and action diversity then reflects lexical synonymy rather than exploration. The deterministic `apply_action` guarantees that **world knowledge acquisition** (e.g., answering 'where is key?') is a function of visited cells and memory, not stochastic luck; this assumption fails when entities move (not included here), in which case knowledge requires tracking dynamics. Thus each computational component (graph, templates, deterministic update) is explicitly linked to a motivation-level metric (exploration, behavioral diversity, world knowledge) through stated assumptions that define when the metric is valid.

## Contribution

(1) Spatial-Odyssey, a procedural generation framework that extends text-game grammars to produce multi-modal (text + 2D grid map) environments with physically grounded actions, enabling sensorimotor continual learning evaluation. (2) Demonstration that agents can learn navigation and manipulation policies entirely from text-and-map observations in procedurally generated worlds, with metrics (spatial coverage, navigation efficiency) that isolate sensorimotor adaptation from language comprehension. (3) A design principle: decoupling underlying physical state from observation modality allows the same procedural engine to serve both text-only and multi-modal agents, ensuring fair comparisons without re-engineering the environment.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Spatial-Odyssey procedural environments | Tests spatial exploration and grounding |
| Primary metric | Spatial exploration coverage | Directly measures spatial reasoning requirement |
| Baseline 1 | Static policy (no learning) | Tests necessity of any adaptation |
| Baseline 2 | Inference-only (no memory update) | Tests value of persistent memory |
| Baseline 3 | Random exploration | Tests upper bound of uninformed search |
| Ablation | Text-only (remove spatial graph) | Isolates effect of spatial structure |

### Why this setup validates the claim

The experimental design forms a falsifiable test of whether Spatial-Odyssey's spatial graph and grounded observations effectively measure spatial continual learning. By comparing against baselines that lack learning (static), lack memory (inference-only), or lack strategy (random), we isolate the specific contribution of structured spatial exploration to task performance. The ablation removes the spatial graph, turning environments into flat text descriptions, which tests whether the spatial structure itself drives any observed advantage. Spatial exploration coverage is the right metric because it directly reflects the agent's ability to build and use a mental map, which is the core claim of the framework. If our method only impacts speed but not coverage, the claim is weakened.

### Expected outcome and causal chain

**vs. Static policy** — On a case where the agent must navigate through multiple rooms to reach a goal, the static policy repeats the same action pattern without adapting to new layouts, so it repeatedly fails to explore beyond its initial room. Our method instead builds a spatial memory from observed positions and uses it to plan paths, so we expect a large gap in coverage (e.g., static near 0%, ours >60%) in multi-room tasks.

**vs. Inference-only** — On a case where a key is in an earlier room but the agent must return later to use it, the inference-only agent lacks memory updates so forgets the key's location after leaving the room. Our method updates its internal map and can backtrack efficiently, so we expect a noticeable gap on return-dependent tasks (e.g., inference-only success <30%, ours >70%).

**vs. Random exploration** — On a case where rooms are arranged in a dead-end structure, random exploration wanders aimlessly and may revisit cells many times. Our method plans to cover uncovered areas, so we expect higher coverage per step (e.g., random 20% coverage at 100 steps, ours 70%).

### What would falsify this idea

If our method shows no advantage over inference-only on tasks requiring memory (e.g., comparable success on return-dependent tasks), the claim that spatial memory is effectively acquired is false. Also, if the ablation (text-only) performs as well as the full method on spatial exploration coverage, then the spatial graph is not causally necessary.

## References

1. AgentOdyssey: Open-Ended Long-Horizon Text Game Generation for Test-Time Continual Learning Agents
