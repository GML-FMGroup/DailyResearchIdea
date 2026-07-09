# Generative Latent for Query-Guided Video Feature Reconstruction in Long-Form Reasoning

## Motivation

Existing long-video methods compress the entire video into a fixed-size state (e.g., Light-Omni's multimodal script) that loses fine-grained details, or require segment-level re-encoding for each query, sacrificing efficiency. The finite capacity of summary-based approaches precludes iterative refinement over multiple temporal segments. WorldMM attempts multi-memory but still relies on retrieval from stored features, not on-demand generation, limiting the ability to dynamically access arbitrary segments without re-encoding.

## Key Insight

By conditioning a generative decoder on both a query and a desired temporal window, the fixed-size latent from a single-pass encoder can reconstruct discriminative features for any segment on demand, enabling iterative reasoning without linear cost increase.

## Method

**(A) What it is**  
We propose the Query-Conditioned Generative Latent (QGL) method. It takes a long video and a stream of queries, and outputs reconstructed feature vectors for arbitrary temporal segments specified by the queries, without re-encoding the video. The central load-bearing assumption is that a fixed-size set of discrete latent tokens (K = 4096) from a hierarchical VQ-VAE encoder captures all fine-grained details necessary for reconstructing discriminative features for any queried segment. This assumption is tested by varying K in ablation and by the diversity loss that encourages full codebook utilization.

**(B) How it works (pseudocode)**  
```python
# Encoding (single pass)
latent_tokens = HierarchicalVQVAEEncoder(video_frames)  # shape [K, d], K=4096, d=512
# Query processing
def decode_segment(query_text, t_start, t_end):
    query_emb = TextEncoder(query_text)  # e.g., 768-dim
    pos_emb = SinusoidalPositionEncoding(t_start, t_end, num_segment_tokens=16)  # shape [16, d]
    condition = Concat([query_emb.expand(16, -1), pos_emb], dim=-1)  # shape [16, 2d]
    after_layer_norm = LayerNorm(condition)  # normalized to prevent position dominance
    segment_tokens = TransformerDecoder(
        tokens=LearnedQueryTokens(16),  # initial learnable tokens
        context=latent_tokens,  # key/values from encoder
        cross_attn_condition=after_layer_norm,  # inject condition in cross-attention layers
        num_layers=4
    )  # shape [16, d_out]
    feature_map = LinearProject(segment_tokens)  # reshape to 1D temporal map, e.g., [16, D']
    return feature_map
# Training: end-to-end on QA pairs
for (video, queries, answers) in dataloader:
    latents = encoder(video)
    for q, a in zip(queries, answers):
        seg = decode_segment(q.text, q.start, q.end)
        pred = TaskHead(seg)
        loss = CrossEntropy(pred, a) + λ * DiversityLoss(latents)  # diversity loss to encourage codebook use
    optimizer.step()
```
**Hyperparameters:** K=4096, d=512, d_out=256, number of segment tokens=16, λ=0.1. The encoder uses a hierarchical VQ-VAE with two levels (first level 1024 codes, second level 4096 codes) to form the final 4096 latent tokens. The diversity loss is defined as L_div = -H(p), where H is entropy of codebook usage over a batch, encouraging uniform selection.

**(C) Why this design**  
We chose a hierarchical VQ-VAE encoder over a continuous latent (e.g., NeRF embedding) because discrete tokens prevent blurring across temporal boundaries and allow efficient cross-attention; the cost is potential codebook collapse, mitigated by a diversity loss. We chose a transformer decoder with cross-attention over a deterministic sampling grid (e.g., bilinear interpolation) because it dynamically allocates representational capacity to the segment's most relevant latent tokens; the trade-off is increased per-query compute (∼2 GFLOPs) but it remains constant regardless of video length. We chose to concatenate query embedding with position embedding rather than using a separate cross-attention head for each because it forces the decoder to jointly condition on both signals in a single pass, simplifying training; however, it may underutilize the query if the position dominates, requiring careful normalization (layer norm after concatenation). The increased K from 256 to 4096 raises encoder compute by 16× but single-pass encoding is amortized over all queries; per-query decoder compute remains unchanged.

**(D) Why it measures what we claim**  
The computational quantity `latent_tokens` (shape [4096, 512]) measures the compressed video information that is available for reconstruction; this assumes the hierarchical VQ-VAE captures all necessary details for any downstream segment, which fails when the codebook size is insufficient and task-critical fine motion is quantized away, in which case `latent_tokens` reflects only coarse scene structure. The `segment_tokens` from the decoder measure the generated features for the queried segment; this assumes the cross-attention can correctly attend to the latent subset relevant to the query and temporal bounds, which fails when the query is ambiguous or the segment boundaries are misaligned with latent temporal granularity, causing the decoder to produce averaged or mis-localized features. The `pred` from the task head measures the discriminative quality of reconstructed features; this assumes the end-to-end training aligns reconstruction with task-relevant information, which fails if the task head exploits spurious correlations in the small codebook, leading to high accuracy on biased predictions rather than genuine understanding. These assumptions are explicitly addressed by our diversity loss (to prevent codebook collapse) and by training on multiple segment queries per video (to encourage local feature fidelity). The choice of K=4096 is a direct mitigation of the load-bearing assumption; if ablation shows accuracy saturates beyond 4096, the assumption is supported; otherwise, a continuous latent or larger K is needed.

## Contribution

(1) A novel framework (QGL) that uses a hierarchical VQ-VAE encoder to compress long videos into fixed-size discrete latents, and a query-conditioned transformer decoder to reconstruct feature maps for arbitrary temporal segments on demand. (2) The design principle that generative latent reconstruction enables iterative multi-step reasoning without re-encoding, as the decoder's cost per query is constant. (3) Empirical demonstration on long-video QA benchmarks (e.g., LongTVQA) that QGL outperforms fixed-summary baselines (e.g., Light-Omni) and retrieval-augmented methods (e.g., WorldMM) in both accuracy and inference speed.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | M3-Bench-web (920 videos) | Multimodal QA, long video, diverse queries |
| Primary metric | QA accuracy | Directly measures understanding of queried segments |
| Baseline 1 | Gemini-1.5-pro (prompting) | Strong prior on reasoning but no latent compression |
| Baseline 2 | GPT-4o (prompting) | Similar to Gemini but larger context window |
| Baseline 3 | OmniAgent | Omnimodal baseline, lacks query-conditioned decoding |
| Baseline 4 (new) | QGL-interp | Replaces transformer decoder with bilinear interpolation of latent tokens at query-specified timestamps, then conditions a small MLP on query embedding (2-layer MLP, hidden=256). Tests benefit of learned cross-attention. |
| Ablation 1 | QGL (w/o diversity loss) | Tests mitigation of codebook collapse |
| Ablation 2 | QGL with codebook size 128, 256, 512, 1024, 4096 | Tests load-bearing assumption: reconstruction fidelity on fine-grained temporal queries vs. K |
| Resource estimate | Training on 8×A100 GPUs, ∼7 days for full model (each video ∼1 hour for encoding + training loops); inference per query ∼2 GFLOPs, 15ms on A100. |

### Why this setup validates the claim
M3-Bench-web provides long videos with hundreds of QA pairs requiring temporal localization and fine-grained reasoning. By comparing against Gemini-1.5-pro and GPT-4o—which rely on full-context processing and lack our latent compression—we test whether discrete token reconstruction improves segment-specific accuracy. OmniAgent, an omni-modal baseline, tests whether query-conditioned decoding is crucial over general multimodal fusion. The new QGL-interp baseline isolates the effect of learned cross-attention over simple interpolation. The ablation varying codebook size directly tests the load-bearing assumption that 4096 tokens retain fine-grained details; if accuracy drops sharply below 4096, the assumption is validated. The resource estimate allows readers to assess feasibility. Accuracy on the QA metric directly reflects the quality of our reconstructed features for downstream tasks, making it a direct falsification test: if our method does not outperform baselines on queries requiring detailed temporal reasoning, the central claim (efficient and accurate segment reconstruction) fails.

### Expected outcome and causal chain

**vs. Gemini-1.5-pro** — On a query like 'What color was the jacket in the third scene?' where the video is 10 minutes long, Gemini processes the entire video as text tokens, likely losing spatial-temporal resolution and producing a vague or hallucinated answer. Our method encodes the video once into discrete latent tokens that preserve fine details, and the cross-attention decoder dynamically queries the relevant temporal region, outputting precise features. Thus we expect a noticeable gap (e.g., 15-20% accuracy difference) on queries requiring fine-grained temporal localization, but parity on global questions like 'What is the main activity?'.

**vs. GPT-4o** — On a query involving multiple modalities (e.g., audio and visual) within a short segment, GPT-4o's larger context may capture more detail but still treats every frame equally, causing performance to degrade as video length increases. Our method's fixed-size latent tokens (4096) maintain constant compute per query, avoiding degradation from long context. We expect our method to match GPT-4o on medium-length videos but significantly outperform (e.g., 10-15% higher accuracy) on the longest videos (e.g., >30 minutes).

**vs. OmniAgent** — On a query requiring temporal alignment of audio and visual cues (e.g., 'When did the speaker mention the key number?'), OmniAgent's multimodal fusion processes audio and video jointly but without temporal query conditioning, leading to blurred boundaries. Our method explicitly conditions the decoder on the temporal segment and query text, focusing attention on the relevant latent tokens. We expect our method to achieve higher accuracy (e.g., 10-12% better) on cross-modal temporal queries, while performance on single-modality queries may be similar.

**vs. QGL-interp** — On a query that requires attending to spatially localized fine details within a segment (e.g., a small object), the interpolation baseline will blur features across the segment, whereas our transformer decoder can selectively weight latent tokens. We expect a 5-8% accuracy gap on such queries, with parity on coarse segment-level queries. This quantifies the benefit of learned cross-attention.

**Scaling analysis** — We plot accuracy vs. video length (in minutes) for all methods. Our method's accuracy should remain flat (constant per-query compute), while Gemini/GPT-4o accuracy decays linearly with length (log scale). This demonstrates the claimed constant-cost advantage.

### What would falsify this idea
If our method shows a uniform gain across all query types (e.g., both global and detailed) compared to baselines, or if the ablation without diversity loss performs comparably, it would indicate that the proposed mechanism (hierarchical VQ-VAE with diversity loss and query-conditioned decoding) is not the cause of improvement. Furthermore, if accuracy does not change significantly when K is increased from 256 to 4096, or if it still underperforms on fine-grained tasks, the load-bearing assumption that discrete tokens preserve fine details is falsified and a continuous latent would be required.

## References

1. Light-Omni: Reflex over Reasoning in Agentic Video Understanding with Long-Term Memory
2. Active Perception Agent for Omnimodal Audio-Video Understanding
3. Seeing, Listening, Remembering, and Reasoning: A Multimodal Agent with Long-Term Memory
4. Memory in the Age of AI Agents
5. LongVideoAgent: Multi-Agent Reasoning with Long Videos
6. Qwen3-VL Technical Report
7. Qwen3-Omni Technical Report
8. WorldMM: Dynamic Multimodal Memory Agent for Long Video Reasoning
