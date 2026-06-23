# Hebbian Plasticity for Self-Adaptive Memory in Embodied Agents

## Motivation

Existing memory modules in modular agent frameworks (e.g., OAgents) are static: they use fixed retrieval or update rules that require manual tuning per task. This structural limitation arises because memory dynamics are hard-coded rather than adapting to task statistics. Without automatic adaptation, agents must be re-configured for each new environment, limiting scalability.

## Key Insight

Local Hebbian plasticity allows memory weights to self-organize based on co-occurrence statistics of queries and successful retrievals, enabling automatic configuration without gradient-based meta-learning.

## Method

**HebbMem** is a key-value memory module that adapts its read and write dynamics via local Hebbian and anti-Hebbian plasticity rules. Inputs: query vector q, reward signal r (binary or scalar). Outputs: retrieved value v or None.

**Key assumption:** The reward signal (or its eligibility trace) correctly indicates the utility of a memory for the current query. If reward is noisy or delayed, a decaying eligibility trace is used to smooth credit assignment (see `write` method).

```python
import numpy as np

class HebbMem:
    def __init__(self, key_dim=64, value_dim=64, lr_hebb=0.01, lr_anti=0.001, threshold=0.5, gamma=0.9):
        self.keys = []  # list of key vectors of dimension key_dim
        self.values = []  # list of value vectors of dimension value_dim
        self.weights = []  # scalar plasticity weight per memory slot
        self.eligibility = []  # scalar eligibility trace per slot (for delayed reward)
        self.lr_hebb = lr_hebb
        self.lr_anti = lr_anti
        self.threshold = threshold
        self.gamma = gamma  # decay factor for eligibility trace

    def retrieve(self, query):
        if not self.keys:
            return None, None, None
        scores = [np.dot(query, k) * w for k, w in zip(self.keys, self.weights)]
        idx = np.argmax(scores)
        if scores[idx] > self.threshold:
            return self.values[idx], idx, scores[idx]
        return None, None, None

    def write(self, query, value, reward, retrieved_idx=None):
        # Decay all eligibility traces
        self.eligibility = [e * self.gamma for e in self.eligibility]
        
        # If a memory was retrieved, add reward to its trace
        if retrieved_idx is not None:
            self.eligibility[retrieved_idx] += reward
            r_trace = self.eligibility[retrieved_idx]
            # Hebbian update if trace positive
            if r_trace > 0:
                self.weights[retrieved_idx] += self.lr_hebb * (1 - self.weights[retrieved_idx])
                self.keys[retrieved_idx] += self.lr_hebb * (query - self.keys[retrieved_idx])
            # Anti-Hebbian update if trace negative
            if r_trace < 0:
                self.weights[retrieved_idx] -= self.lr_anti * self.weights[retrieved_idx]
        
        # Add new memory if no retrieval or if reward > 0 (using trace?)
        # Using instantaneous reward for deciding to add new memory
        if retrieved_idx is None or reward > 0:
            self.keys.append(query.copy())
            self.values.append(value)
            self.weights.append(0.5)
            self.eligibility.append(0.0)
```

## Contribution

(1) A memory module for embodied agents that self-adapts its retrieval and storage dynamics via local Hebbian plasticity rules, without backpropagation or meta-training. (2) A design principle showing that weighted retrieval combined with Hebbian key and weight updates enables automatic memory configuration across tasks. (3) An open-source implementation integrated into the OAgents framework.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|----------------------|
| Dataset | Mind2Web + Synthetic sequential recall | Mind2Web tests real web navigation; synthetic task isolates Hebbian dynamics |
| Primary metric | Task success rate (Mind2Web) + Weight correlation (synthetic) | Success measures end-to-end; correlation measures plasticity directly |
| Baseline 1 | LLM-only (no memory) | Isolates need for explicit memory |
| Baseline 2 | FIFO buffer (last 10) | Simple fixed-size memory baseline |
| Baseline 3 | Existing Hebbian memory (HebbianSynapticMemory) | Tests against prior Hebbian approach |
| Ablation-of-ours | HebbMem w/o weight update (fixed weight=0.5) | Tests importance of plasticity |
| Ablation-of-ours | HebbMem w/o eligibility trace (instant reward) | Tests benefit of smoothing credit |

### Why this setup validates the claim

This setup provides a falsifiable test of HebbMem's central claim that adaptive Hebbian plasticity improves memory retrieval in embodied agents. Mind2Web includes tasks requiring retrieval of distant context (e.g., recalling a password from earlier steps). The LLM-only baseline tests whether explicit memory is necessary: if success rates are low on long-horizon tasks, memory is indeed needed. The FIFO buffer tests whether a simple fixed-size memory suffices; if it fails on tasks where relevant information falls outside its window or is crowded by irrelevant entries, we confirm that adaptive read-write dynamics are necessary. The existing Hebbian memory baseline tests whether our specific anti-Hebbian decay and eligibility trace provide an advantage over pure Hebbian models. The ablations isolate the plasticity and credit-smoothing components. The synthetic sequential recall task directly measures weight evolution: we generate a sequence of correlated queries and known target keys, and track whether HebbMem's weights converge to 1 for relevant memories and 0 for irrelevant ones. Task success rate captures whether the agent completes the task, which hinges on proper memory retrieval. Weight correlation on the synthetic task directly quantifies how well the Hebbian updates extract task structure. Thus, observing a clear performance gap on the predicted failure modes would validate the claim, while no gap would falsify it.

### Expected outcome and causal chain

**vs. LLM-only (no memory)** — On a case where the agent must recall a username from a login page visited 10 steps earlier, the LLM-only baseline produces a wrong guess because it has no mechanism to store and retrieve past observations. Our method instead retrieves the stored key-value pair activated by a similar query (e.g., "username field"), because its Hebbian weights emphasize slots that previously yielded positive reward (after smoothing via eligibility). We expect a gap of >15% success rate on tasks with context dependencies beyond the LLM's native context window, but parity on short tasks where all information fits in the prompt.

**vs. FIFO buffer** — On a case where the agent needs a password entered 20 steps ago, but the FIFO buffer (last 10) has already overwritten it with irrelevant steps, the baseline produces a memory miss because it discards old information. Our method instead retains the password memory with high weight (due to positive reward on login) while the anti-Hebbian decay suppresses irrelevant memories, so the password is still retrievable via similarity threshold. We expect a gap of >20% success rate on tasks with long-range dependencies exceeding the buffer size, but parity on tasks where the buffer is sufficient.

**vs. Existing Hebbian memory (HebbianSynapticMemory)** — On the synthetic sequential recall task, HebbianSynapticMemory uses only Hebbian strengthening without anti-Hebbian decay and without eligibility trace. Our method should achieve faster weight convergence (within 50 episodes vs. 80 episodes) and higher final weight correlation (0.95 vs. 0.85) because anti-Hebbian decay prevents over-association and eligibility trace smooths out reward noise.

**What would falsify this idea**
If the gain of HebbMem over the FIFO baseline is uniform across all task lengths rather than concentrated on tasks requiring retrieval beyond the buffer size, then the plasticity is not addressing the predicted failure mode and the central claim would be wrong. Similarly, if on the synthetic task the weight correlation of full HebbMem is not significantly higher than that of the ablation without weight updates, then plasticity is not the cause of improvement.

## References

1. OAgents: An Empirical Study of Building Effective Agents
2. OS-Copilot: Towards Generalist Computer Agents with Self-Improvement
3. CogAgent: A Visual Language Model for GUI Agents
4. OpenAgents: An Open Platform for Language Agents in the Wild
5. Pix2Struct: Screenshot Parsing as Pretraining for Visual Language Understanding
6. WebShop: Towards Scalable Real-World Web Interaction with Grounded Language Agents
