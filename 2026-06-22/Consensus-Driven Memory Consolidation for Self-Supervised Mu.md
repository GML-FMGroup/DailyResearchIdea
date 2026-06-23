# Consensus-Driven Memory Consolidation for Self-Supervised Multi-Agent Systems

## Motivation

Existing multi-agent memory systems, such as G-Memory's hierarchical graph, rely on external feedback signals (e.g., task success) to decide which memories to store or update. However, in open-ended or unsupervised environments external signals are unavailable, preventing autonomous learning. This dependence on external verifiers is a structural limitation that recurs across frameworks like EvoMAC and Self-evolving agents, which assume a reliable target proxy. To overcome this, we need a self-supervised signal intrinsic to the multi-agent system.

## Key Insight

The variance of predictions across multiple independently-initialized agents provides a natural proxy for epistemic uncertainty, enabling memory consolidation decisions without external feedback.

## Method

### (A) What it is
C-Memory (Consensus Memory) is a self-supervised memory consolidation mechanism for multi-agent systems. It takes as input a stream of observations, the set of agent predictions on those observations, and a local memory buffer per agent; it outputs consolidated long-term memories and updated short-term buffers.

### (B) How it works

**Load-bearing assumption (must hold for method to work):** High pairwise agreement among independently-initialized agents implies that the aggregated prediction is accurate (corresponds to ground truth). This assumption is justified only when agents are initialized with independent random seeds and trained on sufficiently diverse data; if agents share systematic biases, high agreement may reflect common error rather than correctness.

```
Initialize: long_term_memory = {}, short_term_buffers = {agent_i: [] for all i}
Parameters: threshold_high = 0.85, threshold_low = 0.2, decay_rate = 0.9, max_capacity = 50
# Optional calibration: if a small labeled set (e.g., 512 examples) is available periodically,
# check correlation between consensus_score and accuracy; if correlation < 0.5, adjust thresholds or add diversity regularization.

for each observation o_t:
   predictions = {agent_i.predict(o_t) for all i}  # each prediction is a probability distribution over classes
   consensus_score = mean_pairwise_agreement(predictions)  # computed as average of pairwise normalized mutual information between prediction distributions
   
   if consensus_score > threshold_high:
        # High consensus: consolidate into long-term memory
        long_term_memory.store(o_t, aggregate_prediction = majority_vote(predictions))
        # Update short-term buffers: reset with consolidated info
        for agent_i:
            short_term_buffers[agent_i].append((o_t, aggregate_prediction))
            if len(short_term_buffers[agent_i]) > max_capacity:
                short_term_buffers[agent_i].pop(0)  # FIFO
   elif consensus_score < threshold_low:
        # Low consensus: store tentatively in short-term buffer with decay
        for agent_i:
            short_term_buffers[agent_i].append((o_t, predictions[agent_i]))
            apply_decay(short_term_buffers[agent_i], decay_rate)  # multiply each buffer entry weight by decay_rate
   else:
        # Moderate consensus: keep in short-term, no consolidation
        for agent_i:
            short_term_buffers[agent_i].append((o_t, predictions[agent_i]))
            if len(short_term_buffers[agent_i]) > max_capacity:
                short_term_buffers[agent_i].pop(0)
```

### (C) Why this design
We chose pairwise agreement over entropy-based measures because agreement is more interpretable and robust to varying numbers of agents, accepting the cost that it ignores the diversity of prediction distributions. We used a dual-threshold decision (high/low/moderate) rather than a single threshold to avoid premature consolidation of borderline cases, trading off increased complexity for finer-grained memory management. We applied FIFO eviction for short-term buffers rather than importance-weighted retention because the consensus signal already encapsulates reliability, and FIFO is simpler and does not require an additional scoring function. We opted for majority voting for aggregation in high-consensus cases over averaging of probabilistic predictions because majority vote is less sensitive to outlier agent outputs, though it discards confidence information.

### (D) Why it measures what we claim
Consensus score (pairwise agreement) measures reliability of the information because, under the assumption that agents are initialized with independent random seeds and receive the same observation, high agreement implies that the true underlying state is uniquely identifiable (Condorcet jury theorem); this assumption fails when agents share systematic biases (e.g., all trained on the same flawed data), in which case high agreement reflects common error rather than ground truth. The threshold_high parameter operationalizes the confidence level needed for long-term storage; it measures the system's tolerance for false positives, and its optimal value depends on the diversity of agents. The threshold_low parameter operationalizes the level of below which information is considered too unreliable for retention; it measures the system's sensitivity to noise, with lower values allowing more tentative storage. Decay_rate operationalizes the forgetting curve for unreliable information; it measures how quickly low-consensus predictions are discarded, assuming that unreliability correlates with temporariness, which fails if initially disputed information later becomes valuable. The optional calibration step operationalizes a verification of the core assumption; if the correlation between consensus and accuracy drops below a threshold (e.g., 0.5), the system flags a need for diversity regularization or threshold adjustment, measuring the validity of the independence assumption.

## Contribution

(1) A self-supervised memory consolidation mechanism for multi-agent systems that uses prediction consensus among agents as an intrinsic feedback signal, eliminating the need for external verifiers. (2) A theoretical justification based on Condorcet's jury theorem that links consensus thresholds to bounds on false positive storage rates under agent diversity assumptions. (3) An analysis of the memory-retention vs. consensus-tradeoff, showing that the dual-threshold design reduces storage of incorrect information compared to single-threshold baselines.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Multi-agent MMLU (1000 questions, each agent receives same question with different random seed) | Tests reasoning with conflicting information from independent seeds. |
| Primary metric | Accuracy | Direct measure of decision correctness. |
| Baseline 1 | Individual memory (each agent FIFO buffer size 10, no sharing) | No inter-agent memory consolidation. |
| Baseline 2 | Shared memory (global FIFO buffer size 40, no consensus filtering) | Global buffer without consensus filtering. |
| Baseline 3 | G-Memory (hierarchical graph with 2 levels, no consensus trigger) | Hierarchical memory but no consensus trigger. |
| Ablation | C-Memory without decay (decay_rate = 1.0) | Tests necessity of temporal forgetting. |

### Why this setup validates the claim

The multi-agent MMLU dataset provides claims with varying consensus levels among agents (due to random seeds), directly testing the method's core mechanism: using consensus to decide memory consolidation. Baseline 1 (individual memory) tests the claim that inter-agent coordination improves reliability; if C-Memory outperforms it, the benefit of shared memory is confirmed. Baseline 2 (shared memory) isolates the effect of consensus-based filtering vs. simple averaging; superiority of C-Memory would show the value of selective consolidation. Baseline 3 (G-Memory) checks against a structured but non-consensus-based memory. The ablation (no decay) tests the importance of forgetting tentative information. Accuracy is a clear, interpretable metric that reflects correct answers, making predicted effects directly observable. This setup forms a falsifiable test: if C-Memory does not outperform these baselines on accuracy, the central claim (consensus improves reliability) is unsupported.

### Expected outcome and causal chain

**vs. Individual memory** — On a question where three out of four agents agree (high consensus, e.g., correct answer A) but the fourth agent independently guesses B, individual memory does not share the correct answer, so the fourth agent may answer B, lowering group accuracy. Our method consolidates the consensus (A) into long-term memory and updates all agents' buffers, so the fourth agent also answers A. We expect a noticeable gain on high-consensus questions (roughly +5-10% accuracy) and parity on low-consensus ones.

**vs. Shared memory** — On a question where two agents predict A (correct) and two predict B (no consensus), shared memory averages all predictions (e.g., output A with 0.5 probability), leading to ambiguous decisions. Our method detects low consensus (< threshold_low) and keeps predictions in short-term buffers with decay, avoiding the wrong consolidation; later, if new evidence tips consensus, our method consolidates. We expect our method to have higher accuracy on ambiguous questions (roughly +3-5%) and similar performance on high-consensus ones.

**vs. G-Memory** — On a question where different hierarchical levels store conflicting information (e.g., lower-level memory retains a wrong answer from early trials), G-Memory may retrieve outdated data. Our method uses real-time consensus to decide what to store, so irrelevant low-consensus information decays. We expect our method to be more robust to temporal noise, with accuracy gains on questions where recent consensus contradicts earlier patterns (roughly +2-4%).

### What would falsify this idea
If C-Memory's accuracy improvement over baselines is uniform across all consensus levels (e.g., +2% on both high and low consensus), then the consensus mechanism is not the cause; the idea is falsified. Alternatively, if the ablation without decay matches or exceeds C-Memory's performance, the decay hypothesis is wrong.

## References

1. G-Memory: Tracing Hierarchical Memory for Multi-Agent Systems
2. G-Designer: Architecting Multi-agent Communication Topologies via Graph Neural Networks
3. Self-Evolving Multi-Agent Collaboration Networks for Software Development
4. Self-evolving Agents with reflective and memory-augmented abilities
5. Can Large Language Models Really Improve by Self-critiquing Their Own Plans?
6. AgentVerse: Facilitating Multi-Agent Collaboration and Exploring Emergent Behaviors in Agents
7. Towards an understanding of large language models in software engineering tasks
8. Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection
