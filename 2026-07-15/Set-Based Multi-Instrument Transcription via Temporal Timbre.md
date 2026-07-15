# Set-Based Multi-Instrument Transcription via Temporal Timbre Consistency

## Motivation

Existing multi-instrument transcription models like MuScriptor rely on instrument presence conditioning and autoregressive token decoding, which assume known instrument labels and a sequential ordering of note events. This structural dependency limits generalization to unseen instrument combinations and prevents fully parallel prediction. We identify a meta-gap: simultaneous polyphonic transcription with unknown instruments requires a mechanism that is invariant to instrument identity and note order, yet current approaches hard-code these assumptions.

## Key Insight

Temporal invariance of timbre across overlapping audio segments provides a self-supervisory signal that ties note events from the same instrument to a shared embedding, enabling instrument discovery without labels and set prediction without imposed ordering.

## Method

### (A) What it is
TTC-Set is a set-based transcription model that takes an audio spectrogram and outputs a set of note events, each with pitch, onset, offset, velocity, and a learned instrument embedding (no predefined instrument classes). Training uses a consistency loss on instrument embeddings from overlapping segments of the same audio, plus a Hungarian matching loss for note attributes.

### (B) How it works (pseudocode)
```python
# TTC-Set Training Loop (single audio clip)
# Input: spectrogram X (T x F) with 2 random crops: X1, X2, overlapping by at least 1 sec
# Output: two sets of predictions Y1, Y2 (each set size N_max, with tokens for pitch, onset, offset, velocity, embedding)

# Hyperparameters:
N_max = 500           # maximum number of notes per segment
overlap_min = 1.0     # minimum overlap duration in seconds
lambda_consistency = 0.1  # weight for consistency loss
lambda_spread = 0.01      # weight for embedding spread regularization
embedding_dim = 128

# Encoder: hierarchical frequency-time transformer (e.g., 4-layer Swin-like, dim=256)
H1 = encoder(X1)  # (seq_len1, d_model=256)
H2 = encoder(X2)  # (seq_len2, d_model=256)

# Decoder: transformer decoder with 6 learned queries (dim=256) and 4 cross-attention layers
Q = learned_queries(N_max, dim=256)
Y1 = decoder(Q, H1)  # list of note predictions: each element is dict of pitch, onset, offset, vel, embed (embed L2-normalized)
Y2 = decoder(Q, H2)

# Note attribute loss via Hungarian matching
loss_notes1 = hungarian_loss(Y1, G1, attributes=['pitch','onset','offset','velocity'])
loss_notes2 = hungarian_loss(Y2, G2, attributes=['pitch','onset','offset','velocity'])

# Identify matched notes between segments (based on ground truth in overlap region)
matched_indices = []  # list of (idx1, idx2) pairs
for gt_note in overlap_notes(G1, G2):
    idx1 = hungarian_match(gt_note, Y1)  # index in Y1
    idx2 = hungarian_match(gt_note, Y2)
    matched_indices.append((idx1, idx2))

# Consistency loss: cosine distance on L2-normalized embeddings
loss_consistency = 0
for idx1, idx2 in matched_indices:
    e1 = Y1[idx1]['embed']  # already L2-normalized
    e2 = Y2[idx2]['embed']
    loss_consistency += 1 - torch.dot(e1, e2)  # cosine distance = 1 - cosine similarity
loss_consistency /= max(len(matched_indices), 1)

# Embedding spread regularization: encourage embeddings to be distinct
# Compute pairwise cosine similarity among all embeddings in Y1 and Y2 (only negative pairs from different notes)
# For simplicity, compute only on Y1: penalize high similarity between different predictions
def spread_loss(Y):
    embs = torch.stack([n['embed'] for n in Y])  # (N_max, embedding_dim)
    sim = embs @ embs.T  # (N_max, N_max)
    # Exclude diagonal (self-similarity)
    mask = ~torch.eye(N_max, dtype=bool)
    return sim[mask].mean()  # mean cosine similarity across different notes
loss_spread = 0.5 * (spread_loss(Y1) + spread_loss(Y2))

# Total loss
loss = loss_notes1 + loss_notes2 + lambda_consistency * loss_consistency + lambda_spread * loss_spread
```

### (C) Why this design
We chose set prediction over autoregressive token decoding to avoid imposing an arbitrary note order that is not present in the task; the trade-off is increased decoder complexity and the need for Hungarian matching, which scales quadratically with N_max. We used temporal consistency across overlapping segments instead of a classification head for instrument labels because instrument annotations are scarce and timbre invariance is a natural property of the same instrument; the cost is that the loss becomes noisy for notes that rarely co-occur in overlapping segments. We chose cosine distance for embedding similarity because it is bounded and scale-invariant, but it assumes that embeddings are unit-normalized, which may collapse if all embeddings are pushed to a single point; we avoid this by using a large embedding dimension (128) and a regularization term (spread loss) that pulls different embeddings apart. Finally, we used random overlapping crops rather than fixed aligned frames because it provides diverse timbre variations from the same instrument; however, it increases training time due to forward passes of two crops.

### (D) Why it measures what we claim
The consistency loss (cosine distance between embeddings of matched notes across segments) measures **timbre invariance** under the assumption that the same instrument's timbre is stable across short time intervals. This invariance operationalizes the concept of **instrument identity**, because if two notes share the same instrument, their timbre embeddings should be similar, and if they are from different instruments, they should differ. The assumption fails when an instrument changes timbre dramatically due to effects (e.g., guitar wah-wah) or when multiple instruments share identical timbre (e.g., two identical pianos); in these cases, the consistency loss would incorrectly force embeddings to be similar for different instruments or allow divergence for the same instrument. The Hungarian matching for note attributes measures **accurate transcription** via attribute loss (pitch, onset, offset, velocity) under the standard assumption that the ground truth is correct and the matching is optimal; this assumption fails if the ground truth contains spurious notes or if the matching algorithm mismatches due to dense polyphony, in which case the loss reflects alignment quality rather than transcription accuracy. The embedding spread loss ensures that distinct notes have separated embeddings, mitigating collapse; it assumes that notes from different instruments are sufficiently different in timbre, which may not hold for instruments with very similar timbres (e.g., two different string instruments playing in the same register).

## Contribution

(1) A set-based transcription model that jointly predicts note events and instrument embeddings without autoregressive sequencing or instrument labels. (2) A self-supervised temporal timbre consistency loss that leverages overlapping audio segments to learn consistent instrument representations. (3) Demonstration that the approach can generalize to unseen instrument combinations by not relying on predefined instrument classes.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|-----------------------|
| Dataset | Slakh2100 | Large multi-instrument MIDI-synth dataset. |
| Primary metric | Note F1 (onset+offset+pitch) | Standard transcription accuracy metric. |
| Baseline 1 | MT3 (T5-based) | Strong multi-instrument baseline. |
| Baseline 2 | SpecTNT (Transformer) | Prior state-of-the-art set-based method? |
| Baseline 3 | Perceiver TF (Transformer) | Strong performer on multi-instrument. |
| Baseline 4 | Contrastive + set prediction (no embedding consistency) | Isolates contrastive pretraining effect. |
| Baseline 5 | Instrument classification + set prediction | Highlights advantage of label-free discovery. |
| Ablation | TTC-Set w/o consistency loss | Isolates embedding learning contribution. |

Compute estimate: Training on Slakh2100 (75 GB, 210k audio clips) using 4×A100 (40 GB) takes ~120 GPU hours (batch size 8, 200k steps). Memory footprint: ~12 GB per GPU with N_max=500.

### Why this setup validates the claim
Slakh2100 provides diverse multi-instrument mixtures with exact ground truth, enabling precise evaluation of transcription accuracy. The primary metric (Note F1) directly measures the core transcription task. Comparing against MT3, SpecTNT, and Perceiver TF—strong baselines lacking our consistency mechanism—tests whether learned timbre embeddings improve accuracy. The ablation (removing consistency loss) isolates its contribution, creating a falsifiable test: if gains appear only on multi-instrument mixtures where instrument confusion is high, the claim that consistency fosters timbre invariance is supported; uniform gains would suggest other factors. Adding contrastive pretraining (Baseline 4) tests if consistency is redundant with simpler contrastive learning, while adding instrument classification (Baseline 5) tests if labeled timbre discovery is superior.

### Expected outcome and causal chain

**vs. MT3** — On a dense polyphonic mixture with two saxophones playing similar lines, MT3 often merges notes into a single track or confuses pitch identities because its T5 decoder treats all notes sequentially without instrument context. Our method learns separate embedding clusters for each saxophone via temporal consistency, so its set decoder assigns distinct instrument embeddings, reducing confusion. We expect a >5% Note F1 improvement on such mixtures, with parity on single-instrument clips.

**vs. SpecTNT** — On a recording where the same instrument (e.g., piano) undergoes a sudden timbre shift due to pedal or attack variation, SpecTNT’s frame-based attention may break note continuity and produce spurious instrument splits. Our consistency loss forces embeddings from overlapping crops to be similar, smoothing out timbre perturbations and maintaining stable instrument identity. We expect fewer insertion errors (lower false positives) on such passages, improving F1 by ~3-5%.

**vs. Perceiver TF** — On a multi-instrument passage where a violin and flute play in non-overlapping registers but both have similar spectral envelopes, Perceiver TF’s latent tokens may fail to disambiguate their timbres, leading to cross-assignment errors. Our explicit embedding consistency across overlapping crops encourages distinct clusters for different instruments, even when spectral features overlap. Expect a reduction in instrument confusion errors (e.g., false onsets from substituting one instrument for another) by ~10%.

**vs. Contrastive + set prediction (no embedding consistency)** — On mixtures where instruments have overlapping timbres, contrastive pretraining on non-overlapping segments may produce less instrument-specific embeddings than our consistency loss on overlapping segments. Expected: ~2-3% F1 gain for TTC-Set on such mixtures.

**vs. Instrument classification + set prediction** — On mixtures with unseen instrument combinations during training, the classification head fails, while TTC-Set (label-free) generalizes better. Expected: +5-7% F1 on unseen instrument sets.

### What would falsify this idea
If the full TTC-Set shows no significant Note F1 gain over the ablation on multi-instrument mixtures (i.e., the gain is uniform across all subsets), the central claim that consistency loss learns useful timbre invariance would be falsified, because the predicted benefit from instrument-specific embeddings would be absent. Additionally, if the consistency loss does not yield measurable embedding clustering (e.g., silhouette score < 0.3 on validation set), the underlying assumption of timbre invariance would be unsupported.

## References

1. MuScriptor: An Open Model for Multi-Instrument Music Transcription
2. Unaligned Supervision For Automatic Music Transcription in The Wild
3. YourMT3+: Multi-Instrument Music Transcription with Enhanced Transformer Architectures and Cross-Dataset STEM Augmentation
4. Multitrack Music Transcription with a Time-Frequency Perceiver
5. Automatic Piano Transcription with Hierarchical Frequency-Time Transformer
6. HPPNet: Modeling the Harmonic Structure and Pitch Invariance in Piano Transcription
7. Towards Automatic Transcription of Polyphonic Electric Guitar Music: A New Dataset and a Multi-Loss Transformer Model
8. Jointist: Joint Learning for Multi-instrument Transcription and Its Applications
