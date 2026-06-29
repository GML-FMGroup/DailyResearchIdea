# Unified Memory Module: A Differentiable Architecture for Agent Memory Systems

## Motivation

The analytical framework in 'Are We Ready For An Agent-Native Memory System?' evaluates modular memory systems but never challenges the assumption that storage, retrieval, and maintenance should be separate components. This modular separation forces each module to be designed independently, missing opportunities for joint optimization where interactions between these functions could improve coordination. A unified architecture that learns all three operations end-to-end can potentially achieve better holistic emergence.

## Key Insight

Because memory operations are fundamentally interdependent—storage affects retrieval, retrieval influences maintenance—a monolithic neural network that jointly learns them can discover coordination patterns that are inaccessible to separately optimized modules.

## Method

(A) **What it is**: The Unified Memory Module (UMM) is a differentiable neural network that maps the current agent observation to a retrieved memory vector and an updated memory matrix. It merges storage, retrieval, and maintenance into a single trainable component.

(B) **How it works**: ```python
import torch
import torch.nn as nn

class UMM(nn.Module):
    def __init__(self, N=256, d=64):
        super().__init__()
        self.memory = nn.Parameter(torch.randn(N, d))  # storage
        self.query_net = nn.Linear(d, d)               # retrieval
        self.write_net = nn.Linear(d, d)               # write content
        self.gate_net = nn.Linear(d, 1)                # write gate

    def forward(self, x_t):
        # x_t: observation embedding (d-dim)
        q = self.query_net(x_t)                         # query
        attn = torch.softmax(q @ self.memory.T / (d**0.5), dim=-1)  # (N,)
        r_t = attn @ self.memory                        # retrieved memory (d,)
        w = self.write_net(x_t)                         # write content
        g = torch.sigmoid(self.gate_net(x_t))            # gate
        # Write to the same locations that were read, weighted by attention
        self.memory = self.memory + g * (w - attn @ self.memory) * attn.unsqueeze(1)
        return r_t
```
Hyperparameters: N=256 slots, d=64, learning rate 1e-3.

(C) **Why this design**: We chose a single memory matrix with shared attention for read and write over separate read/write heads as in the Differentiable Neural Computer (DNC). This forces the module to use the same addressing scheme for both operations, promoting coherent memory usage where the location of a memory also determines where updates are applied. The trade-off is that separate heads could allow independent read and write positions, which may be beneficial when updates should not affect future reads. However, in agent memory, recent memories are often revisited; shared addressing aligns with the intuition that location binds to content and simplifies gradient flow. We used gated writing (g) rather than unconditional update to allow the module to decide when to overwrite, preventing catastrophic forgetting. We chose MLPs over recurrent networks for the query/write/gate networks to keep the module stateless except for M, making training more stable via straightforward backpropagation.

(D) **Why it measures what we claim**: The memory matrix M directly operationalizes *storage*; the attention weights a computed by the query network operationalize *retrieval*; the gate g and write content w operationalize *maintenance*. Because all these quantities are learned jointly via the same task loss, gradients flow through every operation, measuring the concept of *holistic optimization*. The assumption that joint training yields better coordination is that the task loss provides a single coherent signal that aligns storage, retrieval, and maintenance; this fails if the memory capacity N is too small, in which case performance degrades due to inadequate storage. The soft attention mechanism assumes that each retrieval is a weighted combination of memories; this fails when exact recall (e.g., a specific fact) is needed, in which case the retrieved vector may be a diffuse blend that loses precision.

## Experiment

### Evaluation Setup
| Role | Choice | Rationale (≤12 words) |
|------|--------|----------------------|
| Dataset | MultiWOZ 2.1 | Multi-turn dialogue with memory needs. |
| Primary metric | F1 score | Measures task success and answer accuracy. |
| Baseline 1 | No-memory | Context-only without retrieval or storage. |
| Baseline 2 | Static RAG | Fixed retrieval from external knowledge base. |
| Ablation | UMM with separate heads | Isolates benefit of shared read/write addressing. |

### Why this setup validates the claim
MultiWOZ requires tracking dialogue state over multiple turns, directly testing memory capabilities. The no-memory baseline isolates the contribution of any memory. Static RAG tests retrieval without adaptive maintenance, while the separate-heads ablation tests the shared addressing hypothesis. F1 captures precise slot-value retrieval, reflecting both storage and retrieval quality. If UMM outperforms on slots requiring coherent updates (e.g., corrections or evolving preferences), the claim of holistic optimization is supported. Conversely, uniform gains would contradict the predicted mechanism.

### Expected outcome and causal chain

**vs. No-memory** — On a turn where the user corrects a car model from "sedan" to "SUV", the no-memory agent loses the update because it has no state; it either repeats the old value or omits it. Our method retrieves the previous memory via attention, then writes the update with gating, preserving the corrected value. We expect a noticeable F1 gap on slot-update instances (e.g., >15% improvement) but parity on simple single-turn queries.

**vs. Static RAG** — When user dietary preferences change mid-dialogue, static RAG retrieves an outdated knowledge base entry and cannot write the new constraint. UMM writes the new preference into memory with a high gate, so subsequent queries retrieve the updated value. We expect UMM to outperform on dynamic slots (e.g., 10% higher F1) but similar performance on static facts.

**vs. Ablation (separate heads)** — In a long episode (>5 turns) where consistent location binding helps, separate heads may read from one slot and write to another, fragmenting memory. UMM's shared attention aligns read and write locations, maintaining coherence. We expect UMM to have higher F1 on long dialogues (e.g., >5% gap) but comparable performance on short ones.

### What would falsify this idea
If UMM's gain over the ablation is uniform across dialogue lengths rather than concentrated on long episodes, or if its advantage over static RAG is absent on dynamic slots, the central claim of coherent holistic optimization would be unsupported.

## References

1. Are We Ready For An Agent-Native Memory System?
