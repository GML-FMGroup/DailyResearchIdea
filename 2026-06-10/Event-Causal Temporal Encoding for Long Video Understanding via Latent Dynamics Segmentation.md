# Event-Causal Temporal Encoding for Long Video Understanding via Latent Dynamics Segmentation

## Motivation

Existing methods for long video understanding, such as TimeScope, either ignore fine-grained temporal dependencies by uniformly aggregating frame features, or rely on frozen vision-language models that lack explicit temporal reasoning. This convergent gap across multiple approaches—from frame-level agnostic models (Tree 0) to static VLM-dependent methods (Tree 1) and uniform coherence assumptions (Tree 2)—stems from the absence of a learned temporal representation that captures event boundaries and causal structure. Without such representation, models cannot reason about the order and causation of events, leading to brittle performance on tasks requiring temporal grounding and causal inference.

## Key Insight

Event boundaries naturally correspond to shifts in the latent dynamics of video features, and causal relations between events can be inferred from the temporal asymmetry of these dynamics without requiring explicit annotations.

## Method

**Event-Causal Temporal Encoding for Long Video Understanding via Latent Dynamics Segmentation**

### (A) What it is  
ECTE (Event-Causal Temporal Encoding) is a self-supervised framework that takes a long video as input and outputs a temporally structured representation composed of discrete event segments and a causal graph over them. It learns to segment the video into events by detecting changes in a latent dynamical system, and to infer causal relations from the order and conditional dependencies of event dynamics.

### (B) How it works  
```python
# Pseudocode for ECTE training
# Input: video frame sequence V = [v1, v2, ..., vT] (T frames, possibly subsampled)
# Hyperparameters: window_size=8, causal_threshold=0.6, contrastive_temperature=0.07

# Step 1: Frame encoding
frames = encode_per_frame(V)  # each frame -> d-dim vector using pretrained ViT-B/16 (frozen)

# Step 2: Learn latent dynamics via a recurrent S4 layer (state space model, hidden_dim=512, 4 layers)
hidden_states = S4_encoder(frames)  # S4 captures long-range dependencies with linear complexity

# Step 3: Event boundary detection
# Compute change point score: for each time t, measure KL divergence between distributions of hidden states in windows [t-K, t] and [t, t+K]
scores = [KL_div(hidden_states[t-window_size:t], hidden_states[t:t+window_size]) for t in range(window_size, T-window_size)]
# Detect peaks above threshold (height=1.5, distance=8) as event boundaries (non-maximal suppression)
event_boundaries = peak_detection(scores, height=1.5, distance=window_size)

# Step 4: Event representation aggregation
# For each event segment between boundaries, compute mean hidden state and temporal position (start, end)
event_reps = [mean(hidden_states[start:end]) for (start,end) in event_intervals]
event_positions = [(start,end) for (start,end) in event_intervals]

# Step 5: Causal relation inference
# Load-bearing assumption: Linear Granger causality on the hidden states of event segments accurately infers causal relations between events in long videos. This assumption may be false for nonlinear dynamics (see Sugihara et al., 2012).
causal_scores = {}  # dictionary of (i,j) -> score
for i, j in permutations(range(len(event_reps)), 2):
    # Extract hidden states from both event intervals
    states_i = hidden_states[event_intervals[i]]
    states_j = hidden_states[event_intervals[j]]
    # Granger test with lag=1 (since events are ordered)
    err_without = mse_predict(states_j, only_states_j[:-1])  # predict last frame from previous
    err_with = mse_predict(states_j, concatenate(states_j[:-1], states_i[-1:]))  # add last frame of i
    score = err_without - err_with  # larger positive means i causes j
    if score > causal_threshold:
        causal_scores[(i,j)] = score

# Build causal graph: nodes = events, directed edges from i to j if i precedes j and causal score > threshold.
# Additionally, enforce temporal order constraint: edges only go forward in time.
causal_graph = directed_graph with edges from event i to event j where i.index < j.index and (i,j) in causal_scores with positive score.

# Output: list of event boundaries, event representations, and causal graph.
# The entire pipeline is trained end-to-end via a reconstruction loss on the S4 encoder and a self-supervised contrastive loss (InfoNCE) for event representations (to make events distinct).
```

### (C) Why this design  
We chose a state space model (S4) over a transformer for modeling latent dynamics because S4 provides linear scalability with sequence length, which is critical for long videos (e.g., 10K+ frames) while capturing long-range dependencies that transformers would require quadratic attention to handle. We use a peak-detection method on KL divergence of hidden states for event boundaries rather than training a separate segmentation network because it is unsupervised and avoids the need for expensive frame-level annotation; the trade-off is that boundaries may be noisy when dynamics are smooth, requiring a threshold hyperparameter that must be tuned per dataset. For causal inference, we adopt a Granger causality test on the learned hidden states rather than a more complex causal discovery algorithm (e.g., PC or LiNGAM) because it is computationally efficient and aligns with the temporal ordering assumption; the cost is that Granger causality captures only linear predictive dependencies, potentially missing nonlinear causal relationships that might be present in complex video events. We use a frozen ViT for frame encoding to leverage pretrained representations while keeping the framework lightweight, but this limits adaptation to video-specific features; we accept this to focus on learning temporal and causal structure rather than low-level adaptation.

### (D) Why it measures what we claim  
The KL divergence between hidden state distributions in adjacent windows measures **event boundary occurrence** because a sudden change in latent dynamics indicates a transition between distinct event types; this assumption relies on the S4 encoder producing locally stationary representations within events, which fails when the video contains gradual transitions (e.g., a slow zoom), in which case the score reflects rate of change rather than a clear boundary. The Granger causality score from event i to event j measures **causal influence** because if the past of event i reduces prediction error of event j beyond its own past, then i has predictive power over j that is not shared by j alone; the underlying assumption is that the hidden states contain sufficient statistics for the system's dynamics and that the linear VAR model approximates the true causal generative process, which fails when the true causation involves long time lags or nonlinear interactions that the linear model cannot capture—in that case the score reflects only linear predictive correlation. The combination of temporal ordering (i before j) and Granger score ensures that **causal direction** is identified from temporal precedence and predictive asymmetry, but this assumes no latent confounders affecting both events; if a third unseen event drives both i and j, the method may infer a spurious causal link. The contrastive loss for event representations ensures **event distinctness** by maximizing separation between event features in embedding space, which holds when events have different visual content; if events are visually similar but causally distinct (e.g., repeated actions), the metric reflects visual similarity rather than event identity. Additionally, KL divergence on S4 hidden states measures event boundary occurrence, assuming the S4 encoder produces locally stationary representations within events. If dynamics change gradually, false boundaries may appear.

## Contribution

(1) Introduces ECTE, a self-supervised framework that learns event boundaries and causal relations from long videos using a combination of state-space modeling and Granger causality, without requiring event or causal annotations. (2) Demonstrates that event-centric temporal representations can be extracted from latent dynamics changes, enabling explicit causal reasoning over learned event segments. (3) Provides a principled causal-bridge analysis linking each computational component (KL divergence, Granger score) to the theoretical concepts it operationalizes, with explicit failure assumptions.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset 1 | NExT-QA (causal questions) | Contains causal video QA requiring reasoning. |
| Dataset 2 | CATER (synthetic) | Explicit causal annotations for ground-truth evaluation. |
| Primary metric | Causal question accuracy (NExT-QA), precision/recall of boundaries & causal edges (CATER) | Directly tests causal understanding claim and validates assumptions. |
| Baseline 1 | TimeScope | No explicit causal reasoning. |
| Baseline 2 | TinyLLaVA-Video | Small LMM without event segmentation. |
| Baseline 3 | TimeSformer | Standard video transformer, no temporal structure. |
| Baseline 4 | Transformer with boundary tokens (end-to-end learned segmentation) | Isolates novelty of S4+KL boundary detection. |
| Ablation 1 | Ours w/o causal graph | Isolates benefit of causal inference. |
| Training | 1x A100 GPU for 48 hours, batch size 16, learning rate 1e-4, AdamW optimizer, cosine schedule | Feasibility confirmation. |

### Why this setup validates the claim

NExT-QA's causal questions require understanding event order and cause-effect relations, directly targeting the core claim of ECTE. By comparing against baselines that lack explicit event segmentation and causal inference (TimeScope, TinyLLaVA-Video, TimeSformer, end-to-end segmentation baseline), we isolate the contribution of our method. CATER provides synthetic videos with known ground-truth boundaries and causal graphs, allowing direct validation of boundary detection and causal edge precision/recall. The ablation without causal graph further tests whether observed gains stem from causal reasoning or event segmentation alone. Accuracy on causal questions is a sensitive probe: if ECTE improves causal reasoning, it should significantly outperform all baselines and its own ablation on this subset, while performing similarly on descriptive questions where causality is less critical. This setup thus provides a falsifiable test of whether event-causal representation enhances understanding of long videos.

### Expected outcome and causal chain

**vs. TimeScope** — On a video where two events appear in order A then B but the true causal relation is B→A (e.g., a person trips (B) after seeing a banana peel (A) but the trip causes the peel to be kicked), TimeScope grounds queries temporally but cannot infer reverse causality; it will answer causal questions incorrectly. Our method uses Granger causality on latent dynamics to detect that B's past predicts A's future, building the correct graph. We expect a noticeable gap (≥10%) on reversal cases but parity on simple temporal questions.

**vs. TinyLLaVA-Video** — On a long cooking video with multiple causally linked events (e.g., chopping onions causes tears), TinyLLaVA processes the full sequence without explicit event boundaries, so it may miss the long-range dependency between chopping and tearing. Our method segments events and infers causal edges, enabling reasoning over composed cause-effect pairs. We expect our method to outperform by ≥5% on multi-step causal questions that require chaining events.

**vs. TimeSformer** — On a video with subtle causal cues (e.g., a hand wave causes a faraway object to fall via a rope), TimeSformer's uniform attention may not capture the specific spatiotemporal relation because it lacks event-level abstraction. Our method detects the hand-waving event and the falling object event, then tests causality via Granger, directly capturing the relation. We expect our method to show higher accuracy (≥8%) on questions about indirect causes.

**vs. Transformer with boundary tokens** — On videos with gradual transitions, the end-to-end learned segmentation may more accurately detect boundaries, but our S4+KL method may produce false positives; we expect comparable boundary precision on clean cuts but lower recall on smooth transitions. Our method still benefits from the causal graph for downstream tasks.

### What would falsify this idea

If ECTE's performance on causal questions is not significantly higher than its ablation without causal graph, or if the gain is uniform across all question types (causal, temporal, descriptive), then the causal graph is not the source of improvement and the central claim would be falsified. On CATER, if boundary precision/recall is below 0.6 and causal edge precision/recall is below 0.5, the assumptions of the method are not validated.

## References

1. TimeScope: Towards Task-Oriented Temporal Grounding In Long Videos
2. TinyLLaVA-Video-R1: Towards Smaller LMMs for Video Reasoning
3. LLaVA-Video: Video Instruction Tuning With Synthetic Data
4. Video-ChatGPT: Towards Detailed Video Understanding via Large Vision and Language Models
5. Multimodal Foundation Models: From Specialists to General-Purpose Assistants
6. OPT: Open Pre-trained Transformer Language Models
7. Fine-tuned CLIP Models are Efficient Video Learners
8. Commonsense Video Question Answering through Video-Grounded Entailment Tree Reasoning
