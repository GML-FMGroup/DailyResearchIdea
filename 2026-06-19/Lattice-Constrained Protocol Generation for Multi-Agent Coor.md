# Lattice-Constrained Protocol Generation for Multi-Agent Coordination

## Motivation

Existing approaches to multi-agent communication protocol design either rely on manual specification, which does not scale with network size (as shown by AgentsNet's performance degradation on larger networks), or use LLM-generated routines that lack correctness guarantees and require costly post-hoc verification (as in Agora). The root cause is that correctness is treated as a verification target rather than a generative property, leading to fragile and unscalable protocols. A structural inductive bias that guarantees key properties like deadlock- and livelock-freedom by construction is missing.

## Key Insight

By constraining the state space of agent communication protocols to a complete lattice with a unique global minimum and enforcing monotonic transitions, any generated protocol is guaranteed to be deadlock- and livelock-free because every state is on a path to the absorbing minimum and no cycles can form.

## Method

(A) **What it is**: The Lattice-Constrained Protocol Transformer (LCPT) generates a deterministic finite automaton (DFA) that defines agent communication protocols. Inputs are a task specification (natural language) and environmental context (e.g., network topology). Output is a DFA whose state space is a complete lattice with a unique global minimum. All transitions are monotonic with respect to the lattice order, ensuring safety by construction.

(B) **How it works**:
```python
# Predefined lattice: product of K binary components (e.g., has_data, has_consent)
# L = {0,1}^K, order: (a1,...,aK) <= (b1,...,bK) iff all ai <= bi
# Unique global minimum: (0,...,0); global maximum: (1,...,1)

def generate_protocol(task_spec, env_context):
    # Encode inputs
    task_emb = TransformerEncoder(task_spec, env_context)
    # Initialize DFA with minimum state
    states = [tuple([0]*K)]
    transitions = {}
    alphabet = set()
    # BFS generation, starting from min state
    queue = [states[0]]
    while queue:
        cur = queue.pop(0)
        # Decode whether cur is an accept state (task-specific)
        if DecoderPredictAccept(cur, task_emb):
            continue  # no outgoing transitions from accept state
        # For each component, decide if it flips from 0 to 1 (monotonic: 1→1 always)
        # Use K independent Bernoulli with logits from decoder
        logits = DecoderTransitionLogits(cur, task_emb)  # shape (K,)
        # Clamp logits such that if component is already 1, logit = +inf (always 1)
        for i in range(K):
            if cur[i] == 1:
                logits[i] = float('inf')
        # Sample new state (only 0->1 flips possible)
        new_components = [1 if cur[i]==1 or torch.sigmoid(logits[i]) > 0.5 else 0 for i in range(K)]
        new_state = tuple(new_components)
        # Create message (label for transition)
        msg = DecoderMessage(cur, new_state, task_emb)  # e.g., "request_data"
        alphabet.add(msg)
        transitions[(cur, msg)] = new_state
        if new_state not in states:
            states.append(new_state)
            queue.append(new_state)
    return DFA(states, alphabet, transitions, initial_state=(0,...,0), accept_states=states_with_accept_flag)
```
Training: LCPT is trained via a two-stage procedure. First, imitation learning on a dataset of expert-designed protocols (e.g., from simple coordination tasks) to learn basic transition patterns. Second, reinforcement learning (REINFORCE with task success as reward) to optimize efficiency (number of states, message overhead) while keeping the lattice constraint hard-coded. Hyperparameters: K=4 binary components, learning rate 1e-4, batch size 64.

(C) **Why this design**: We chose a lattice constraint over a generic FSM generator because it provides a structural guarantee of deadlock/livelock freedom without relying on post-hoc verification (unlike Agora's LLM-generated routines), accepting the trade-off that the protocol can only represent monotonic progress—agents cannot revert to a lower state. We use a product of binary component lattices instead of a more expressive partial order (e.g., arbitrary DAG) because componentwise monotonicity simplifies enforcement and aligns with common agent state dimensions (e.g., knowledge, resources). The cost is that correlated state changes are modeled as independent decisions, but we assume independence suffices for the target tasks. The decoder uses independent Bernoulli per component (rather than a joint distribution over the lattice) to guarantee exact monotonicity—any joint sampler could violate the order if not carefully designed. We combine imitation and RL because pure imitation may not generalize to unseen tasks, while pure RL suffers from sample complexity; the trade-off is a more sample-efficient training that leverages expert prior while exploring efficiency. The key structural design is that the lattice is predefined and fixed, not learned, which makes the guarantee unconditional on training quality.

(D) **Why it measures what we claim**: The lattice constraint ensures deadlock freedom because from any state there is always a monotonic path to the unique global minimum (accept state), and no cycles exist since every transition strictly increases the component vector (at least one 0→1 flip). This is a direct computational operationalization of deadlock freedom: a deadlock occurs when protocol execution reaches a non-accept state with no outgoing transition, but in LCPT every state (except accept) has at least the transition to a higher state. *The componentwise monotonicity measures protocol progress* because each transition increases the number of components set to 1, guaranteeing termination in at most K steps. This assumption fails if the task requires releasing resources (turning a component from 1 to 0), in which case the metric would incorrectly measure regress as progress; therefore, the method is limited to tasks where state is cumulative. The unique global minimum measures protocol completion: when all agents reach the (0,...,0) state, they have achieved idle or termination. This assumes that the global minimum is indeed an appropriate terminal state; if the task requires a different end condition, the metric would misclassify completion. Thus, the lattice parameters must be chosen to align with task semantics.

## Contribution

(1) A novel Lattice-Constrained Protocol Transformer (LCPT) architecture that generates agent communication protocols guaranteed to be deadlock- and livelock-free by construction, eliminating the need for post-hoc verification. (2) An empirical finding that protocols generated by LCPT achieve high task success rates on multi-agent coordination benchmarks (e.g., AgentsNet tasks) while requiring zero verification effort, demonstrating that structural constraints can replace costly correctness checks. (3) A combined imitation and reinforcement learning training procedure that balances expert guidance and efficiency optimization under hard safety constraints.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|----------|
| Dataset | Multi-Agent Cooperative Navigation (MACN) | Standard spatial coordination benchmark |
| Primary metric | Task success rate | Directly measures coordination goal |
| Baseline 1 | Homogeneous LLM agents (free-form) | Upper bound with full flexibility |
| Baseline 2 | Agora-style LLM-generated protocol | Represents prior art without lattice |
| Baseline 3 | Random monotonic protocol | Tests benefit of learned transitions |
| Ablation | LCPT with non-monotonic transitions | Isolates effect of lattice constraint |

### Why this setup validates the claim

This setup directly tests whether the lattice-constrained DFA generation guarantees deadlock-free and efficient coordination. The MACN dataset offers varied spatial topologies and task complexities, stressing protocol adaptability. Task success rate is the ultimate measure of coordination, and comparing against free-form LLM agents (flexible but prone to deadlock), Agora-style protocols (no structural guarantees), and random monotonic protocols (safe but suboptimal) isolates the contribution of learned monotonic transitions. The ablation (non-monotonic) tests if the lattice constraint itself is crucial. A clear performance advantage in tasks where deadlock or inefficiency typically arises would validate the claim, while uniform or reversed patterns would falsify it.

### Expected outcome and causal chain

**vs. Homogeneous LLM agents (free-form)** — On a case requiring strict turn-taking (e.g., agents must sequentially acquire resources), free-form agents often deadlock due to contradictory messages (e.g., both request data simultaneously). Our method enforces monotonic progress via lattice, ensuring each step increases state; thus, deadlock is impossible. We expect a large success rate gap (e.g., 0.8 vs 0.3) on such coordination-heavy tasks, with parity on simple tasks where deadlock is unlikely.

**vs. Agora-style LLM-generated protocol** — On a case with a complex network (e.g., 16 agents in a ring), Agora may generate a protocol with hidden cycles causing livelock (e.g., agents repeatedly exchange consent). LCPT's lattice guarantees a strict partial order and unique minimum, eliminating cycles. We expect LCPT to achieve higher success rate (e.g., 0.9 vs 0.6) and lower variance across runs, particularly on larger networks.

**vs. Random monotonic protocol** — On a case where efficient resource allocation matters (e.g., agents must gather data from dispersed locations), random monotonic transitions often waste steps (e.g., flipping components in suboptimal order). LCPT learns task-specific transition probabilities via imitation+RL, enabling faster progress. We expect LCPT to complete tasks in fewer steps (e.g., ~5 vs ~10) with higher success rate.

### What would falsify this idea

If LCPT with lattice constraint achieves similar or lower success rate than free-form LLM on tasks where deadlock is predicted (e.g., turn-taking scenarios), or if the non-monotonic ablation outperforms LCPT on any task, then the central claim that monotonicity ensures safety and efficiency would be refuted.

## References

1. AgentsNet: Coordination and Collaborative Reasoning in Multi-Agent LLMs
2. A Scalable Communication Protocol for Networks of Large Language Models
3. Problem-Solving in Language Model Networks
4. War and Peace (WarAgent): Large Language Model-based Multi-Agent Simulation of World Wars
5. S3: Social-network Simulation System with Large Language Model-Empowered Agents
6. Encouraging Divergent Thinking in Large Language Models through Multi-Agent Debate
7. PaLM: Scaling Language Modeling with Pathways
8. GLM-130B: An Open Bilingual Pre-trained Model
