# StreamConsolidate: A Benchmark for Online Adaptive Consolidation in Agent Memory Systems

## Motivation

Existing memory benchmarks evaluate static retrieval or offline consolidation, but real-world agents must adapt memory from continuous interactions without re-processing entire histories. 'Are We Ready For An Agent-Native Memory System?' decomposes memory into modules but does not test online adaptive consolidation—the ability to update memory representations on-the-fly. This gap means current evaluations can miss critical failures like catastrophic interference or slow incorporation of streaming feedback.

## Key Insight

By requiring agents to concurrently manage interleaved memory writes and reads under controlled semantic interference, the benchmark isolates the core trade-offs of online consolidation that are hidden in static or offline evaluations.

## Method

### (A) What it is
StreamConsolidate is a multi-task benchmark that evaluates agent memory systems on their ability to continuously consolidate information from a stream of interactions. It comprises a set of tasks where each task requires the agent to first encounter new information, then later retrieve or reason with it, while other tasks impose interference.

### (B) How it works
Pseudocode for the benchmark procedure:

```pseudocode
# StreamConsolidate Benchmark Protocol
Input: Agent A, Memory System M (with read/write operations)
Output: Metrics: consolidation_speed, accuracy, interference

1. Initialize virtual world with K=10 entities (each entity is a key-value store with attributes: name, color, location, rule).
2. For each episode e in 1..E:
   a. Generate streaming interaction sequence S_e of length T=50, where each step t provides:
      - new_info: (entity_id, attribute, value) e.g., ('Alice', 'color', 'blue')
      - later_query: a question about that info (e.g., 'What is Alice\'s favorite color?') presented after delay d (varied 1-10 steps)
      - interference_task: a concurrent memory write to a semantically similar entity (embedding cosine distance < τ=0.3 using Sentence-BERT all-MiniLM-L6-v2) or a retrieval of a related item.
   b. For each step t in S_e:
      i. Present new_info to Agent A through M.write(info, timestamp)
      ii. Later, present a query related to that info; record whether M.retrieve(query) yields correct answer and latency (number of steps from query to correct retrieval, clipped to [0,100] and averaged).
      iii. Concurrently, introduce interference: update a related entity (same attribute, different value) or retrieve a related item.
   c. After episode, compute:
      - consolidation_speed = average latency to correctly retrieve newly learned info over episodes (lower is faster)
      - accuracy = proportion of correct retrievals
      - interference = drop in retrieval accuracy on old info after new info is added (compared to baseline accuracy on a control episode with unrelated updates, embedding distance >0.8)
3. Report average over episodes.

Hyperparameters: K=10, T=50, d ∈ {1,3,5,10}, interference threshold τ=0.3 (cosine distance), embedding model: Sentence-BERT all-MiniLM-L6-v2.

Load-bearing assumption: Semantic similarity measured by embedding distance (cosine distance < 0.3) reliably induces interference in online consolidation. This assumption is verified via a calibration pilot:
- Before main benchmark, generate 100 candidate interference pairs (pairs of (entity, attribute, old_value, new_value)).
- For each pair, collect annotations from 3 human judges: "Does learning the new information likely interfere with memory of the old information?" (Yes/No).
- Keep only pairs with >80% agreement (all three say Yes). Adjust the embedding threshold τ so that at least 90% of human-validated pairs fall below τ, and less than 10% of non-interfering pairs (human said No) fall below τ. This calibrated τ is used in the benchmark.

Precomputed embeddings for all entity-attribute-value triples are provided as a CSV file (embedding_dim=384) to standardize interference control.
```

### (C) Why this design
We chose a multi-task streaming environment over a static dataset because online consolidation requires temporal dependencies that offline evaluation cannot capture. The design includes three key decisions: (1) We separate consolidation speed, accuracy, and interference as distinct metrics rather than a single utility score, because these dimensions trade off; a system that quickly overwrites old memories may have high speed but high interference. (2) We control interference via semantic similarity between concurrent updates (using embedding distance) rather than using random associations, because real-world interference is semantic; the cost is that benchmark generation requires pre-computed embeddings, increasing setup complexity. (3) We use a virtual world with explicit entities rather than free-form text because entity-based memory is easier to ground and evaluate; the trade-off is reduced realism compared to open-domain dialogues. These choices ensure the benchmark is reproducible and diagnostic while still reflecting agent interactions. Load-bearing assumption: Semantic similarity measured by embedding distance (cosine distance < 0.3) reliably induces interference; this is verified via human annotation pilot study described above.

### (D) Why it measures what we claim
**Assumption A:** The metric `consolidation_speed` (latency to correct retrieval after learning) measures **memory update efficiency** because it captures the time a system needs to transition from not knowing to stably storing new information; this assumption fails when a system uses lazy storage (e.g., appends to a list without integration) and retrieval latency is artificially low, in which case the metric reflects append speed rather than consolidation. To mitigate, we explicitly require memory systems to implement an update operation (not only append); if a system only appends, its consolidation speed metric is flagged as invalid. **Assumption B:** The metric `accuracy` measures **retention fidelity** because it operationalizes the proportion of correctly recalled information under zero-shot retrieval; this assumption fails when the benchmark queries are too simple or ambiguous, in which case accuracy reflects retrieval trick rather than actual memory. **Assumption C:** The metric `interference` (drop in old-memory accuracy after new updates) measures **memory stability** because it quantifies how much new learning degrades prior knowledge; this assumption fails when the interference task is too dissimilar from old memories, in which case interference becomes negligible and uninformative. Each metric is designed to separate the components of online consolidation, but the failure modes must be controlled by calibrating task difficulty via pilot studies.

## Contribution

(1) A benchmark framework, StreamConsolidate, that systematically evaluates online adaptive consolidation in agent memory systems through streaming interactions. (2) A set of diagnostic metrics (consolidation speed, accuracy, interference) that decompose consolidation behavior into separable, interpretable dimensions. (3) A controlled generation procedure for interference tasks based on semantic similarity, enabling reproducible stress tests of memory interference.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | StreamConsolidate | evaluates online memory consolidation specifics |
| Primary metric | Consolidation Score | weighted sum of speed, accuracy, interference |
| Baseline 1 | No-memory (context-only) | shows need for memory persistence |
| Baseline 2 | Simple RAG (static KB) | lacks online consolidation |
| Baseline 3 | Continual Learning (EWC) | tests specific interference mitigation |
| Ablation-of-ours | Remove interference tasks | isolates stability measurement |

### Why this setup validates the claim
This combination tests the central claim that StreamConsolidate measures three distinct consolidation dimensions. The dataset's streaming structure with delayed queries and semantically controlled interference forces memory systems to balance speed, accuracy, and stability. The no-memory baseline reveals the necessity of any persistent storage; RAG shows inability to handle dynamic updates; EWC provides a continual learning baseline that explicitly mitigates interference. The ablation removes interference tasks, directly testing the interference metric's contribution. The composite Consolidation Score prevents gamesmanship on a single dimension, requiring balanced performance. If the claim holds, our method must outperform baselines specifically on interference-heavy episodes while maintaining competitive speed and accuracy, yielding a higher composite score.

### Expected outcome and causal chain

**vs. No-memory (context-only)** — On a case where a query appears 10 steps after the relevant information (delay d=10), the no-memory baseline cannot retrieve it because context is lost, producing a wrong answer or failure. Our method stores the information in a persistent memory system and consolidates it, so retrieval succeeds. We expect a large gap on long-delay episodes (accuracy near 0% for baseline vs >80% for ours), but parity on immediate queries (delay=0) where both can use context.

**vs. Simple RAG (static KB)** — On a case where new information updates a previously known fact (e.g., Alice's favorite color changes from blue to green), RAG retrieves the outdated version from static KB, producing wrong answer. Our method updates the memory entry and consolidates the new information, preventing interference from old data. We expect a noticeable gap on episodes with semantically similar updates (interference high) where RAG accuracy drops significantly (e.g., 40% lower) while ours remains stable.

**vs. Continual Learning (EWC)** — On a case where many items are learned sequentially with high semantic interference, EWC may suffer from quadratic computational overhead or inability to handle non-i.i.d. streams. Our method, using online consolidation, should achieve better speed and accuracy on long sequences (T=50) with interference. We expect a moderate gap on interference-heavy episodes (e.g., 15% higher accuracy for ours) but comparable performance on low-interference episodes.

### What would falsify this idea
If our method shows similar consolidation scores to the baselines across all delay and interference conditions, or if the advantage is uniform instead of concentrated on interference-heavy subsets, the central claim that StreamConsolidate isolates consolidation dimensions would be invalidated.

## References

1. Are We Ready For An Agent-Native Memory System?
