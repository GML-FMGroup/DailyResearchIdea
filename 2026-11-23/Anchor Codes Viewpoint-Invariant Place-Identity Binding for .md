# Anchor Codes: Viewpoint-Invariant Place-Identity Binding for Episodic Memory in Embodied 3D LLMs

## Motivation

Across the embodied-spatial-memory literature, every memory architecture stores and retrieves episodic content by feature similarity to the current observation, so a revisited room is only recognized if its stored features happen to match the new view — a property that viewpoint change and same-type room distractors systematically violate. ReSpaceBench (Lost in the Hallway) diagnoses exactly this failure at scale: the strongest zero-shot model reaches 50.5% overall and 52.2% on its 450-question subset versus 90.1% for humans, and agentic video-understanding frameworks give no improvement over direct inference, with execution traces showing evidence omissions and incorrect cross-visit reasoning. The root cause is structural: 3DLLM-Mem, 3D-Mem, the persistent embodied world model, and Memo all retrieve by content similarity, so no stored representation carries an explicit, pose-invariant place identity that a revisit protocol could certify — the benchmark measures the absence of a binding mechanism, not a weakness of any single memory store.

## Key Insight

A place's identity is the orbit of its observations under camera pose, so a place code trained with a contrastive objective whose positives are pose-perturbed views of the same physical room and whose negatives are same-type rooms is invariant by construction, and binding that code to each episodic token gives retrieval a channel that cannot be confused by viewpoint change or room-type distractors.

## Method

**What it is.** AnchorCodes augments 3DLLM-Mem's episodic memory with a learned place-identity descriptor per memory entry. Inputs: the agent's working-memory tokens (current observation) and the episodic store of (feature token, place code, visit id) entries. Outputs: for each working-memory token, a retrieval distribution over episodic entries produced by two additive channels — a content-similarity score and a place-code score — fused into the attention weights used to read episodic memory. A separate place-code head maps any observation to a unit-norm descriptor; a visit-level aggregation step produces the stored place code for each room entry.

**How it works.**

```pseudocode
# Phase 1: place-code head (trained jointly with the memory-fusion model)
def place_code(obs_embedding):            # obs_embedding: pooled patch tokens of one view
    z = MLP(obs_embedding, hidden=512, out=128)   # 2-layer MLP, GeLU activation, LayerNorm after hidden
    return z / ||z||                       # unit sphere

# Phase 2: contrastive training of the place-code head on ReSpaceBench walkthroughs
def contrastive_loss(views, room_ids, room_types):
    # views: list of (obs_embedding, room_id, room_type) sampled from controlled walkthroughs
    # margin=0.2, cosine distance d(a,b)=1-cos(a,b)
    loss = 0
    for anchor in views:
        positives = [v for v in views if v.room_id == anchor.room_id and v.view_id != anchor.view_id]
        negatives = [v for v in views if v.room_id != anchor.room_id]      # includes same-type distractors
        for p in positives:
            for n in negatives:
                loss += max(0, margin + d(z_anchor, z_p) - d(z_anchor, z_n))
    return loss / (len(views) * len(positives) * len(negatives))

# Phase 3: episodic store with bound place codes
def store_episode(observation, visit_id, room_id):
    feat = memory_encoder(observation)                 # as in 3DLLM-Mem
    z = place_code(observation)
    episodic.append(Entry(feat=feat, place_code=z, visit_id=visit_id, room_id=room_id))
    # per-room code = running mean of z over all views of that room, re-normalized
    room_code[room_id] = normalize(room_code.get(room_id, 0) + z)

# Phase 4: dual-channel retrieval at query time
def retrieve(working_memory_tokens):
    q_feat = query_projection(working_memory_tokens)      # working-memory tokens as queries
    q_place = place_code(pool(working_memory_tokens))
    for e in episodic:
        s_content = cosine(q_feat, e.feat)               # 3DLLM-Mem's existing channel
        s_place   = cosine(q_place, e.place_code)        # new channel, weight 0.5
        s_total   = s_content + 0.5 * s_place
    weights = softmax(s_total / temperature)              # temperature=0.1
    read = sum(weights * e.feat for e in episodic)
    return read                                          # fed to the LLM decoder as in 3DLLM-Mem
```

**Why this design.** We chose a contrastive objective with pose-perturbed positives and same-type-room negatives over a supervised room-classification head because classification collapses distinct rooms of the same type into one class, destroying exactly the discrimination ReSpaceBench tests; the cost is that contrastive training requires paired multi-view walkthroughs, which ReSpaceBench's controlled visits supply. We chose a per-room running-mean code over storing only per-view codes because a mean over views is the empirical estimate of the orbit centroid and is more stable under a single noisy view, accepting that it cannot represent rooms whose appearance changes across sessions — a case we leave to the content channel. We chose additive fusion with a fixed 0.5 weight over a learned router because a learned router reintroduces the "two modules plus controller" failure mode and has no closed-form decision rule, whereas a fixed weight keeps the place channel's contribution auditable; the cost is that the weight is not adapted per task, which we ablate. Unlike standard retrieval-augmented memory (the RAG/Toolformer family) and unlike 3DLLM-Mem's purely content-similarity attention, our retrieval score is a sum of two quantities that are computed from different surfaces — one from content features, one from a pose-invariant descriptor — so a domain expert would not describe AnchorCodes as a variant of RAG: RAG has no notion of an orbit-invariant code, and its nearest-neighbor score is exactly the content channel we augment.

**Why it measures what we claim.** The contrastive loss measures viewpoint-invariant place identity because it enforces that cosine distance between codes of the same room under different poses is smaller than to any other room, including same-type distractors; this equivalence assumes that ReSpaceBench's controlled walkthroughs label physical room entry correctly and that pose perturbation is the only nuisance factor, and it fails when two distinct rooms are visually near-identical and never co-visited, in which case the loss reflects visual similarity rather than physical identity. The place-code retrieval score measures cross-visit binding because it is the cosine between the query's place code and the stored room code, which equals one only when the query's inferred place matches the stored place; this assumes the place-code head generalizes to unseen viewpoints of the same room, and it fails when a novel viewpoint is out of the training distribution, in which case the score reflects the head's extrapolation error rather than the binding. The fused attention weight measures the agent's use of place identity in memory readout because it is the softmax of content plus place scores; this assumes the LLM decoder consumes the read vector faithfully, and it fails when the decoder ignores the read vector and relies on its language prior, in which case the weight reflects decoder bias rather than memory use.

**Load-bearing assumption (explicit).** The central assumption is that a contrastive head trained on pose-perturbed positives and same-type-room negatives yields a place code that is invariant to viewpoint and discriminative between physically distinct same-type rooms, such that cosine similarity in that code space is a reliable proxy for physical place identity at query time. This is the classic invariance-vs-discriminability tradeoff: the objective explicitly optimizes away the very appearance variation that distinguishes two visually near-identical same-type rooms, so the learned code is pushed toward room-type/geometry statistics rather than instance identity. We do not assume this away; we measure it directly via the place-code separation diagnostic (see experiment) and report the invariance/discriminability curve. If the code proves appearance-only, the contribution is reframed as a retrieval re-ranking prior with a measured invariance/discriminability curve rather than a claimed orbit-invariant identity.

## Contribution

['AnchorCodes: a place-code binding layer for episodic memory in embodied 3D LLMs that stores a contrastively trained, pose-invariant descriptor per memory entry and fuses a place-code retrieval channel with the existing content-similarity channel.', "A contrastive training recipe that uses ReSpaceBench's controlled revisit protocol — pose-perturbed positives and same-type-room negatives — as the supervision signal for place-identity invariance, turning a diagnostic benchmark into a training signal.", 'An analysis of which failure modes of cross-visit reasoning (evidence omission, same-type confusion) are reduced by an explicit place-identity channel versus content-only retrieval, via execution-trace comparison.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | ReSpaceBench (Lost in the Hallway) | Controlled revisits with same-type distractors test place identity. |
| Primary metric | Accuracy on room-association and persistent-state tasks | Directly probes cross-visit binding the method targets. |
| Baseline | 3DLLM-Mem (content-only episodic attention) | Isolates contribution of the place-code channel. |
| Baseline | Strongest zero-shot video model (direct inference) | Tests whether memory architecture beats raw perception. |
| Baseline | Agentic video-understanding framework | Tests whether added place code beats agent scaffolding. |
| Baseline | NetVLAD-style place-recognition descriptor retrieval | Tests whether contrastive orbit objective beats standard place recognition. |
| Baseline | AnchorCodes with learned-router fusion | Tests whether fixed 0.5 weight suffices vs. learned fusion. |
| Ablation-of-ours | AnchorCodes with place weight 0.0 | Confirms gain comes from place channel, not retraining. |
| Diagnostic | Place-code separation vs. viewpoint change magnitude | Directly measures invariance/discriminability tradeoff. |
| Diagnostic | Read-vector perturbation probe | Tests decoder faithfulness to retrieved memory. |

### Why this setup validates the claim

The central claim is that augmenting 3DLLM-Mem's content-similarity retrieval with an orbit-invariant place code improves cross-visit spatial memory specifically where content matching fails: revisits under viewpoint change and same-type room distractors. ReSpaceBench is the right dataset because its controlled walkthroughs define visits by physical room entry and deliberately inject same-type distractors and cross-session object changes, so the failure mode the place channel addresses is directly instantiated rather than incidental. The content-only 3DLLM-Mem baseline tests the sub-claim that content similarity alone conflates same-type rooms; the zero-shot video model and agentic framework test whether the gain is architectural rather than perceptual or scaffolding-driven. The NetVLAD-style baseline tests whether the contrastive orbit objective with same-type negatives yields discrimination that a standard place-recognition descriptor cannot, isolating what is genuinely new. The learned-router baseline tests whether the fixed 0.5 weight is sufficient or whether a learned fusion is needed. The weight-0.0 ablation is the decisive control: it shares the place-code head's training and the same episodic store, so any gap between it and full AnchorCodes is attributable to the place channel's contribution rather than to extra parameters or walkthrough exposure. Accuracy on room-association and persistent-state tasks is the right metric because those tasks require binding a query to the correct physical place across separated visits, which is exactly what the fused attention weight is supposed to carry; a metric that averaged over all nine tasks would dilute the predicted effect. The place-code separation diagnostic measures same-room vs. same-type-distractor cosine separation as a function of viewpoint change magnitude, directly testing the load-bearing invariance/discriminability assumption. The read-vector perturbation probe measures whether the decoder consumes the read vector faithfully by perturbing it and measuring output change, testing the decoder-faithfulness assumption.

### Expected outcome and causal chain

**vs. 3DLLM-Mem (content-only)** — On a revisit where the agent re-enters a kitchen after passing a visually similar kitchen, the content-only attention assigns near-equal weights to both rooms' episodic entries because both match the query's appearance features, so the read vector blends two places and the decoder reports the wrong room's state. AnchorCodes instead computes a place-code score that is high only for the stored code of the physically entered room, so the fused softmax concentrates weight on the correct entries. We expect a noticeable accuracy gap on the same-type-distractor subset but near-parity on revisit tasks without distractors, where content similarity already suffices.

**vs. Strongest zero-shot video model** — On a cross-session change where an object moved between visits, the zero-shot model has no persistent store and must infer state from the current clip, so it reports the pre-change or default configuration because it cannot bind the current view to the prior visit. AnchorCodes retrieves the stored room code and its bound feature tokens, letting the decoder compare current observation against the recorded prior state. We expect the gap to concentrate on persistent-state-tracking questions and to shrink on room-association questions the zero-shot model can answer from a single clip.

**vs. Agentic video-understanding framework** — On a long walkthrough with many revisits, the agentic framework's evidence selection omits earlier visits because its retrieval is driven by textual or content cues that drift as the scene changes, producing evidence omissions in its execution trace. AnchorCodes' place channel re-identifies the revisited room from a pose-invariant descriptor, so earlier entries for that room are re-surfaced even when their appearance has shifted. We expect an improvement in cross-visit reasoning accuracy and a reduction in trace-level evidence omissions, with parity on single-visit questions.

**vs. NetVLAD-style place-recognition descriptor retrieval** — NetVLAD is trained on place recognition datasets with a different objective (often triplet loss with GPS-annotated positives), not on ReSpaceBench's controlled walkthroughs with same-type negatives. We expect AnchorCodes to outperform NetVLAD on same-type-distractor subsets because its contrastive objective explicitly uses same-type rooms as negatives, whereas NetVLAD's training may not. On tasks without distractors, we expect parity. This baseline isolates the contribution of the orbit-invariant contrastive objective over standard place recognition.

**vs. AnchorCodes with learned-router fusion** — A learned router may adapt the fusion weight per task, potentially improving performance on tasks where content or place channel is more reliable. However, it reintroduces the "two modules plus controller" failure mode and may overfit. We expect the fixed 0.5 weight to be competitive or slightly worse on some tasks but more stable and auditable. The gap, if any, will inform whether the fixed weight is a limitation.

**Diagnostic: Place-code separation vs. viewpoint change** — We will measure the cosine similarity between place codes of the same room under varying viewpoint changes (e.g., 0°, 30°, 60°, 90°, 180°) and between different same-type rooms. We expect same-room similarity to remain high and different-room similarity to be lower, but the margin may shrink as viewpoint change increases. We will report the invariance/discriminability curve and the distribution of within-room code variance to test whether the running mean is a valid orbit centroid or collapses distinct sub-regions.

**Diagnostic: Read-vector perturbation probe** — We will perturb the read vector (e.g., add Gaussian noise or swap with a random read vector) and measure the change in the decoder's output distribution. If the decoder ignores the read vector, the output change will be small, indicating that the fused attention weight does not measure the agent's use of place identity. We expect a significant output change, supporting decoder faithfulness.

### What would falsify this idea

If AnchorCodes' gain over the weight-0.0 ablation is uniform across all nine tasks rather than concentrated on the same-type-distractor and cross-visit subsets, the place channel is not doing the binding work we claim and the central mechanism is wrong. Equally fatal: if the ablation matches full AnchorCodes on distractor-heavy subsets, the place code adds nothing beyond retraining. Additionally, if the place-code separation diagnostic shows no advantage over NetVLAD or if the read-vector perturbation probe shows the decoder ignores the read vector, the load-bearing assumptions are violated and the method needs redesign.

## References

1. Lost in the Hallway: Benchmarking Persistent Spatial Memory Across Revisits
