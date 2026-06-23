# Causal Cross-Audio Attention for Multi-Hop Reasoning

## Motivation

The SAKURA benchmark reveals that existing large audio-language models correctly extract information from multiple audio streams but fail to integrate it for multi-hop reasoning, due to the absence of explicit cross-source fusion mechanisms. These models treat each audio stream independently and rely on shallow pairwise attention that cannot enforce temporal or logical consistency across events, leading to integration failures even when individual facts are correctly recalled.

## Key Insight

Modeling the causal relationships between events across streams as a latent directed graph, with edges constrained by temporal order and logical implication, provides a structure that forces attention to respect the compositional dependencies required for multi-hop reasoning.

## Method

(A) **What it is**: Causal Cross-Audio Attention (CCAA) is a mechanism that encodes each audio stream independently, extracts event-level representations, infers a latent causal graph over those events, and computes graph-structured attention to integrate information for multi-hop reasoning. The input is a set of audio streams and a textual query; the output is a text answer.

(B) **How it works** (pseudocode):
```python
def CCAA(audio_streams: List[Audio], query: str) -> str:
    # Step 1: Independent event extraction
    events_per_stream = []
    for audio in audio_streams:
        features = WhisperEncoder(audio)          # frame-level features
        events = EventExtractor(features)          # outputs list of (timestamp, embedding) per event
        events_per_stream.append(events)
    all_events = concatenate(events_per_stream)   # flattened list of events
    N = len(all_events)
    
    # Step 2: Latent causal graph inference
    # For each pair of events, predict causal direction (forward/backward/none) using their embeddings
    # Enforce temporal ordering: if predicted forward and t_i < t_j, edge i->j; if predicted backward and t_j < t_i, edge j->i
    edge_prob_matrix = causal_classifier(all_events)  # NxN matrix, bilinear with temperature 0.5
    adjacency = zeros(N,N)
    for i in range(N):
        for j in range(N):
            if i==j: continue
            prob = edge_prob_matrix[i,j]
            if prob > 0.5:  # threshold
                if prob > 0.5 and all_events[i].timestamp < all_events[j].timestamp:
                    adjacency[i,j] = 1
                elif prob < -0.5 and all_events[j].timestamp < all_events[i].timestamp:
                    adjacency[j,i] = 1
    graph = Graph(nodes=all_events, adjacency=adjacency)
    
    # Step 3: Graph-structured cross-attention
    # Use a single-layer GAT (Veličković et al., 2018) with 4 attention heads
    event_embeds = GATLayer(graph, all_events.embeddings)  # updated embeddings with causal context
    
    # Step 4: Query-guided reasoning
    query_embed = TextEncoder(query)  # BERT
    # Attend over events using bilinear attention weighted by query
    attn_scores = softmax(event_embeds @ W @ query_embed)  # W learned, dimension 768x768
    fused = sum(attn_scores * event_embeds)
    answer = Decoder(fused)  # feedforward + argmax over answer vocabulary
    return answer
```

**Hyperparameters**: EventExtractor: 2-layer Transformer with 512 hidden dim; causal_classifier: bilinear with hidden dim 256; GAT: 1 layer, 4 heads, concat; Decoder: 2-layer MLP with ReLU.

(C) **Why this design**: We chose to separate event extraction and causal graph inference (rather than joint end-to-end modeling) because it allows independent supervision from timestamp labels and semantic event categories, which are easy to obtain synthetically, and avoids the combinatorial explosion of learning causal relations directly from raw audio. The directed causal graph (rather than undirected similarity) is essential because logical implication is inherently asymmetric; undirected attention would conflate "A implies B" with "B implies A", leading to incorrect reasoning chains. We enforce temporal ordering as a hard constraint on predicted edges because in real speech, causal effects cannot precede their causes; this prunes implausible edges and improves precision, at the cost of missing valid simultaneous causal relations (e.g., two speakers confirming each other). Compared to standard cross-attention (e.g., Flamingo's Perceiver Resampler), our method does not treat all tokens uniformly but respects the inferred causal structure, which reduces attention dilution over irrelevant events.

(D) **Why it measures what we claim**: The computational quantity `causal_classifier` measures the logical implication between events because we assume that event embeddings capture sufficient semantic content to predict implication direction; this assumption fails when two events are lexically similar but logically opposite (e.g., "I agree" vs "I disagree"), in which case the classifier may output spurious edges based on lexical overlap rather than true implication. The `temporal ordering constraint` on edges enforces temporal causation, because we assume that causes always precede effects in time; this assumption fails for simultaneous events (e.g., overlapping cheers), where the constraint blocks all edges between them, losing potential logical dependencies. The `GATLayer` computes attention weights that are proportional to the causal influence along inferred edges; this measures the relevance of each event for multi-hop reasoning because we assume that causal paths in the graph correspond to valid reasoning chains; this assumption fails when the inferred graph is incomplete (missing an edge between two dependent events), in which case attention may attend to causally unrelated events, degrading reasoning accuracy.

## Contribution

(1) A novel cross-source attention mechanism, Causal Cross-Audio Attention (CCAA), that explicitly models causal relationships between events across multiple audio streams via latent graph inference and graph-structured attention. (2) The design principle that enforcing temporal and logical constraints in the attention computation is necessary and sufficient for compositional multi-hop reasoning from multiple audio sources, as opposed to simplistic pairwise attention. (3) A synthetic dataset of multi-stream audio dialogues with annotated causal event graphs for training and evaluation (optional).

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|-----------------------|
| Dataset | SAKURA | Multi-hop reasoning from audio streams. |
| Primary metric | Accuracy | Measures correct multi-hop answer. |
| Baseline 1 | Standard cross-attention | Treats all events uniformly. |
| Baseline 2 | End-to-end joint model | Learns causal relations directly from raw audio. |
| Baseline 3 | Concatenation + attention | Simple baseline without causal structure. |
| Ablation | CCAA without causal graph | Removes causal graph, keeps event extraction and attention. |

### Why this setup validates the claim
SAKURA is designed to test multi-hop reasoning across multiple audio streams, directly matching our claim that CCAA improves such reasoning by inferring causal structure. The baselines isolate critical assumptions: standard cross-attention (no causality), end-to-end joint (no separate event extraction), and concatenation (no structure). The ablation removes the causal graph while preserving event extraction and attention, testing whether the graph itself drives improvements. Accuracy is appropriate because it directly reflects whether the model derives the correct logical answer from the audio events. If our method outperforms these on SAKURA, it confirms that explicit causal graph construction yields better multi-hop reasoning than alternatives.

### Expected outcome and causal chain

**vs. Standard cross-attention** — On a case where one event implies another (e.g., speaker A says "I'll turn left" then speaker B says "I'll follow"), standard cross-attention dilutes attention across all events, treating the logical implication as mere temporal co-occurrence, so it often misses the dependency. Our method explicitly infers the causal edge from A to B and strengthens the attention between them, yielding correct answer. We expect a noticeable gap on reasoning-intensive subsets (e.g., those with causal chains) but parity on simple factoid queries.

**vs. End-to-end joint model** — On a case with temporally separated but causally related events (e.g., a door slam after a knock), the joint model suffers from combinatorial explosion in learning causal relations from raw audio, often missing long-range dependencies. Our method first extracts discrete events and then infers causality, which is easier to supervise and generalizes better. We expect our method to outperform on long-horizon reasoning where events are far apart in time.

**vs. Concatenation + attention** — On a case with two contradictory statements (e.g., "I agree" then "I disagree"), concatenation treats them as a flat sequence and fails to identify the logical contradiction because it lacks causal direction. Our method's causal classifier can detect the backward edge (the second statement contradicts the first), leading to correct understanding. We expect our method to show a clear advantage on adversarial examples with logical negation or contradiction.

**Ablation (CCAA without graph)** — On a case where a causal chain involves multiple hops (e.g., A→B→C), removing the graph causes attention to spread across unrelated events, degrading multi-hop reasoning. We expect the full CCAA to outperform its ablation on multi-hop questions, demonstrating that the causal graph is essential for chaining.

### What would falsify this idea
If CCAA's accuracy gain over the ablation is uniform across all question types rather than concentrated on multi-hop or causal-reasoning subsets, then the improvement is not due to causal graph inference but to some other component (e.g., event extraction). Similarly, if standard cross-attention matches CCAA on causal reasoning questions, the central claim that causal structure is necessary would be falsified.

## References

1. SAKURA: On the Multi-hop Reasoning of Large Audio-Language Models Based on Speech and Audio Information
2. MMAU: A Massive Multi-Task Audio Understanding and Reasoning Benchmark
3. The Song Describer Dataset: a Corpus of Audio Captions for Music-and-Language Evaluation
4. MMMU: A Massive Multi-Discipline Multimodal Understanding and Reasoning Benchmark for Expert AGI
5. MuLan: A Joint Embedding of Music Audio and Natural Language
6. Contrastive Audio-Language Learning for Music
7. Large-Scale Contrastive Language-Audio Pretraining with Feature Fusion and Keyword-to-Caption Augmentation
8. Data-Efficient Playlist Captioning With Musical and Linguistic Knowledge
