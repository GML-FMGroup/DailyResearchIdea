# Integrated Memory Networks: Emergent Memory Synergies via Non-Modular High-Dimensional Storage

## Motivation

Existing agent memory systems, as analyzed in 'Are We Ready For An Agent-Native Memory System?' (2026), typically decompose memory into separate modules for representation, extraction, retrieval, and maintenance. This modularity imposes a fixed separation of concerns that prevents the agent from learning cross-functional synergies—for example, the representation may be suboptimal for retrieval because it is optimized independently. In complex tasks requiring tight coordination between memory functions (e.g., multi-turn planning with dynamic knowledge), this modular bottleneck leads to performance ceilings that cannot be overcome by tuning individual modules alone.

## Key Insight

A single high-dimensional representation space implicitly enforces that memory storage, retrieval, and maintenance optimize a common objective, enabling emergent synergies that are inaccessible when these functions are modularly separated.

## Method

```pseudocode
# Integrated Memory Network (IMN)
Input: observation o_t, previous memory state M_{t-1} (N x d matrix), agent state s_{t-1}
Output: action a_t, new memory state M_t

Parameters: encoder E (d_input -> d), query network Q (d_state -> d), attention temperature tau=1.0, learning rate lr_mem=1e-3, regularization coefficient lambda=0.1 (tuned on calibration set of 512 examples to ensure memory retention)

# 1. Encode observation
v_t = E(o_t)  # d-dimensional vector (d=1024)

# 2. Retrieve relevant memories via attention
q_t = Q(s_{t-1})  # query vector
alpha_i = softmax(q_t^T M_{t-1}[i] / tau) for i=1..N
retrieved = sum_i alpha_i * M_{t-1}[i]  # weighted sum

# 3. Update memory: write new vector and maintenance via gradient-based optimization
# We treat memory update as a gradient step on a holistic loss L_holistic that combines:
#   - Retrieval loss: negative log-prob of correct retrieval for current (o_t, s_{t-1}) if supervised
#   - Storage preservation: minimize interference with previous memories
L_holistic = - log softmax(q_t^T v_t) + lambda * sum_i (M_t[i] - M_{t-1}[i])^2 / N
M_t = M_{t-1} - lr_mem * grad_M L_holistic  # update all memory vectors (including potential insertion of v_t)

# Verification step: monitor variance of old memory vectors (first 10% of slots) across updates; if variance drops >20% on calibration set, increase lambda by 0.1 (up to 1.0) to prevent catastrophic forgetting.

# 4. Generate action
a_t = policy_network(s_{t-1}, retrieved, v_t)
```

**Why this design:** We chose a gradient-based update for the entire memory matrix over a modular write-and-forget mechanism because it allows all memory items to be jointly optimized for both future retrieval and minimal interference. This decision accepts the trade-off of higher computational cost per step (matrix gradient computation) but avoids the need for hand-designed maintenance policies (e.g., LRU, FIFO) that would break the holistic optimization. Second, we use a single query network for retrieval rather than separate retrieval and routing modules; this forces the same representation to be useful for both recognizing relevant past and informing the current decision, capturing the synergy between extraction and retrieval. Third, we keep the memory dimension d high (e.g., 1024) to provide ample capacity for emergent structure; the cost is increased memory footprint and slower attention, but it prevents the need for explicit decomposition into slots or categories. Finally, we include a regularization term in the holistic loss to avoid catastrophic forgetting, a direct consequence of treating all memories as equally malleable; the trade-off is that very old memories may degrade, but this is bounded by the L2 penalty. **Load-bearing assumption:** We assume that the L2 regularization term with fixed lambda prevents catastrophic forgetting, but this may not hold for long sequences; we verify this empirically via monitoring memory vector variance on a calibration set and adjusting lambda if needed.

**Why it measures what we claim:** The holistic loss gradient L_holistic operationalizes the concept of emergent synergy because it simultaneously optimizes retrieval accuracy (log softmax term) and memory stability (L2 regularization term); the assumption is that minimizing this combined loss forces the memory representation to balance storage and retrieval demands, producing an integrated solution. This assumption fails when the retrieval and storage objectives are fundamentally incommensurable (e.g., one-time retrieval of a single fact vs. long-term accumulation), in which case the gradient may produce a blurry average that serves neither goal well. The query-vector dot product measures relevance similarity between the current agent state and stored memories, assuming that cosine similarity in the learned space corresponds to functional relevance; this assumption fails when the state representation is poor, in which case the attention weights reflect spurious correlations. The L2 regularization term measures the degree of interference between old and new memories, assuming that Euclidean distance preserves discriminability; this assumption fails if the space is not contractive, leading to memory drift. **Additional assumption A:** Gradient descent on this weighted sum forces a Pareto-optimal trade-off between retrieval accuracy and memory stability. **Failure mode F:** The weight lambda is chosen arbitrarily; if too small, catastrophic forgetting occurs; if too large, retrieval adaptation is severely limited. We mitigate this by tuning lambda on a small calibration set and monitoring memory variance.

## Contribution

(1) A novel agent memory architecture, Integrated Memory Network (IMN), that abandons explicit modular decomposition and instead uses a single high-dimensional memory matrix updated via a holistic gradient-based loss, allowing emergent synergies across memory functions. (2) Empirical demonstration that on complex tasks requiring coordination between memory operations (e.g., multi-hop reasoning with dynamic knowledge), IMN outperforms modular baselines from the 'Are We Ready' framework, and analysis showing that the holistic loss gradient leads to memory representations that balance storage stability and retrieval discriminability. (3) Design principles for non-modular memory systems, including the necessity of high dimensionality and gradient-based maintenance to capture cross-functional dependencies.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Multi-turn dialogue + QA (HotpotQA, MultiWOZ) with a long-horizon subset (≥5 turns) and a high-interference subset (≥3 similar entities) | Tests retrieval over time and state-dependent queries, with controlled difficulty. |
| Primary metric | Task success (F1 score) | Direct measure of answer accuracy. |
| Baseline 1 | No memory | Agent uses only current context. |
| Baseline 2 | Static RAG | Retrieves from fixed external knowledge base. |
| Baseline 3 | Traditional memory (write/LRU) | Standard write-and-forget with fixed capacity. |
| Baseline 4 (new) | Differentiable memory with modular losses (separate retrieval loss and storage loss, summed without holistic gradient) | Isolates the effect of joint optimization vs. modular losses. |
| Ablation of ours | IMN without gradient update (fixed memory, write/overwrite) | Isolates effect of holistic optimization. |
| Ablation 2 (new) | IMN with smaller memory dimension (d=256) | Verifies scalability and resource usage. |

**Resource estimates:** Each IMN step requires ~10 ms on a single V100 GPU for N=1024, d=1024; total memory ≈ 8 MB for memory matrix. Full experiment (5 seeds, 1000 episodes each) ≈ 10 GPU hours.

### Why this setup validates the claim

This design tests the central claim that holistic gradient-based memory update produces emergent synergy between retrieval and storage. The dataset (multi-turn dialogue + QA) requires both retaining facts across turns and retrieving the correct fact based on current state. The no-memory baseline isolates the benefit of any memory. Static RAG tests retrieval from a fixed store, lacking dynamic update. Traditional memory tests modular write/forget but without joint optimization. The new modular loss baseline uses the same architecture but with separate loss terms for retrieval and storage (without shared gradient), isolating the effect of holistic optimization. Our ablation removes the gradient update, testing whether the optimization itself drives synergy. The dimension ablation tests sensitivity to capacity. Task success (F1) directly measures whether the retrieved memory leads to correct answers, making it sensitive to retrieval-storage balance. If the claim holds, our full IMN should outperform all baselines, especially on tasks requiring long-term retention and state-dependent retrieval, while the ablation should degrade.

### Expected outcome and causal chain

**vs. No memory** — On a multi-turn task where the agent needs a fact from turn 1 to answer turn 3, no-memory baseline must rely on immediate context and fails because it lacks persistent storage. Our method retrieves the relevant fact via attention over stored vectors, enabling correct answer. Expect significant gap on long-horizon subsets (e.g., >5 turns), parity on single-turn.

**vs. Static RAG** — On a task where facts get updated (e.g., tool use with changing availability), static RAG retrieves outdated information because its knowledge base is fixed. Our method updates memory via gradient steps, reflecting dynamic changes, so it retrieves current facts. Expect large gap on dynamic knowledge subsets, small gap on static.

**vs. Traditional memory** — On a task with competing memories (e.g., multiple similar entities), traditional memory with LRU may evict a needed memory or store interfering copies because write/forget are decoupled. Our holistic loss jointly optimizes retrieval and storage, reducing interference via L2 regularization. Expect our method to maintain higher accuracy on high-interference subsets (e.g., many similar queries), while traditional memory declines.

**vs. Modular losses** — The modular baseline uses separate retrieval and storage losses (optimized independently via different gradient streams). This decoupling prevents the representation from balancing both goals, leading to either overfitted retrieval or unstable storage. Our holistic loss explicitly forces joint optimization, so we expect our method to outperform on the long-horizon and high-interference subsets, where modular losses cause conflict. Expect similar performance on short, low-interference tasks.

**vs. Ablation (no gradient update)** — On a task requiring adaptive memory update (e.g., learning to prioritize recent relevant facts), the ablation with fixed memory (write/overwrite) cannot adjust representations to reduce interference, so retrieval accuracy degrades over time. Our full method adapts memory vectors via gradient, maintaining retrieval quality. Expect diverging performance over episode length, with full method stable and ablation dropping.

**vs. Smaller dimension (d=256)** — The lower dimension reduces capacity, likely increasing interference and reducing retrieval accuracy. Our full method (d=1024) should outperform, especially on high-interference subsets, but the gap may shrink on simpler tasks. This ablation quantifies the trade-off between resource usage and synergy.

### What would falsify this idea
If our full IMN does not outperform the modular loss baseline on long-horizon or high-interference tasks, or if the dimension ablation shows no benefit (d=256 matching d=1024), then the central claim of emergent synergy via holistic optimization is not supported. Additionally, if the verification step detects catastrophic forgetting on the calibration set even after lambda tuning (variance drops >20%), the approach fails in practice.

## References

1. Are We Ready For An Agent-Native Memory System?
