# StreamEval: Online Evidence Boundary Discovery for Audio-Visual Reasoning

## Motivation

EvaOmni evaluates evidence grounding only on static, pre-segmented videos with human-annotated spans, so it never tests whether a model can discover evidence boundaries online from a continuous stream — a capability required in every real streaming application. This is a structural blind spot shared by evidence-grounded benchmarks: because evidence is handed to the model as a discrete span, the benchmark cannot distinguish a model that genuinely localizes evidence from one that merely consumes a pre-localized span, and the resulting scores overstate grounding ability by an unknown margin.

## Key Insight

The streaming penalty — the drop in temporal IoU between model-discovered evidence boundaries and human spans when the model must emit boundaries online from block-wise input rather than consume a pre-segmented span — is a stable, model-ranking-preserving measure of genuine evidence identifiability, because pre-segmented evaluation leaks the boundary decision that streaming forces the model to make.

## Method

**What it is.** StreamEval is a streaming evaluation protocol for EvaOmni-style audio-visual reasoning benchmarks. It takes as input a video V delivered as consecutive 1.5-second audio-visual blocks (b_1, ..., b_T) and a question Q, and requires the model to emit, online after each block, a set of evidence boundary events (open/close for audio and visual channels) and a final answer A. It outputs per-question evidence spans that are scored by temporal IoU against EvaOmni's human-annotated spans, plus the answer.

**How it works.**

```pseudocode
Input: video V, block size s=1.5s, question Q, model M, patience k=2
Output: answer A, evidence spans E_audio, E_visual, streaming IoU

# Phase 1: online boundary emission
state = M.init(Q)
open_audio = open_visual = False
for t = 1..T:
    b_t = V[(t-1)*s : t*s]              # audio + visual frames
    out_t = M.step(state, b_t)           # returns boundary events + optional answer
    if out_t.event == OPEN_AUDIO and not open_audio:
        a_start = t*s; open_audio = True
    if out_t.event == CLOSE_AUDIO and open_audio:
        a_end = t*s; open_audio = False; emit_audio_span(a_start, a_end)
    if out_t.event == OPEN_VISUAL and not open_visual:
        v_start = t*s; open_visual = True
    if out_t.event == CLOSE_VISUAL and open_visual:
        v_end = t*s; open_visual = False; emit_visual_span(v_start, v_end)
    if out_t.answer is not None:
        # patience: wait k blocks to confirm answer stability
        if M.confirm_answer(out_t.answer, next k blocks) == out_t.answer:
            A = out_t.answer; break
# close any open span at video end

# Phase 2: scoring
IoU_audio = mean over gold audio spans of max IoU(pred, gold)
IoU_visual = mean over gold visual spans of max IoU(pred, gold)
streaming_penalty = IoU_static_with_events - IoU_streaming
```

**Why this design.** We chose a fixed 1.5-second block size over adaptive segmentation because adaptive segmentation would itself require a boundary detector, confounding the measurement of the model's boundary-discovery ability with the segmentation policy; the cost is that some human spans shorter than 1.5s cannot be matched exactly, which we accept because EvaOmni spans are predominantly multi-second. We chose an explicit open/close event protocol over asking the model to emit a span after seeing the whole video because the latter collapses back to static evaluation and leaks the boundary decision; the cost is that models must be prompted into an event-emission interface, which we standardize with a fixed prompt template across all models. We chose a patience-k=2 answer confirmation over accepting the first answer because streaming models often answer prematurely before evidence is complete; the cost is up to 3s of extra latency, which we report separately. We chose temporal IoU against human spans over a learned boundary scorer because IoU is the same metric already used in EvaOmni's static setting, making the streaming penalty a directly comparable delta; the cost is that IoU is sensitive to small offsets, which we mitigate by reporting both strict IoU and a 0.5s-tolerance IoU. Unlike prior retrieval-style boundary detectors that score candidate segments against the question, StreamEval does not retrieve a segment from a library — it forces the model to commit to boundary events causally, block by block, which no retrieval baseline can produce.

**Why it measures what we claim.** The streaming IoU between model-discovered boundaries and human spans measures evidence identifiability because the model must commit to a boundary before seeing future blocks, so a correct boundary can only arise from the model having actually localized the evidence in the audio-visual content rather than from post-hoc span selection; this assumption fails when a model learns a positional prior (e.g., always opening evidence at block 2), in which case the metric reflects prior-matching rather than content-based localization. The streaming penalty (static-with-events IoU minus streaming IoU) measures the degree to which static evaluation overstates grounding because static-with-events evaluation hands the model the boundary decision while keeping the output format identical, so the penalty isolates exactly the boundary-discovery component; this assumption fails when the static-with-events and streaming prompts differ in ways beyond boundary availability (e.g., different output formats), in which case the penalty reflects prompt-sensitivity rather than boundary discovery. The patience-k answer confirmation measures answer stability under streaming because an answer that survives k additional blocks is one the model does not retract upon seeing more evidence; this assumption fails when the model's answer is insensitive to evidence order, in which case patience reflects answer prior rather than evidence-driven confidence.

## Contribution

['A streaming evaluation protocol (StreamEval) for evidence-grounded audio-visual benchmarks that replaces pre-segmented evidence spans with online block-wise boundary emission and scores evidence identifiability by temporal IoU.', 'A design principle: the streaming penalty — the static-to-streaming IoU gap — is a model-ranking-preserving measure that separates genuine evidence localization from pre-segmented span consumption, which static evaluation cannot do.', 'An open-source harness that converts any EvaOmni-style benchmark into a streaming variant with configurable block size, patience, and tolerance, enabling reproducible streaming evaluation of audio-visual reasoning models.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | EvaOmni audio-visual reasoning benchmark | Provides human-annotated evidence spans for IoU scoring |
| Primary metric | Streaming temporal IoU (strict + 0.5s tolerance) | Directly comparable to EvaOmni static IoU delta |
| Baseline | Static EvaOmni evaluation (span consumption) | Upper bound that hands model the boundary decision |
| Baseline | Static-with-events evaluation (format-matched control) | Isolates causality from prompt-format effects |
| Baseline | Retrieval-style segment scorer | Tests whether causal commitment beats candidate retrieval |
| Baseline | Adaptive-segmentation boundary detector | Tests whether fixed blocks confound boundary discovery |
| Baseline | Positional-prior model (fixed block index) | Measures non-content boundary matching floor |
| Ablation-of-ours | StreamEval without patience-k confirmation | Isolates contribution of answer-stability gating |
| Ablation-of-ours | Boundary-shuffled condition (±1 block shift) | Tests whether static IoU stays high while streaming collapses |
| Ablation-of-ours | Content-shuffle condition (evidence moved randomly) | Tests whether streaming IoU tracks content not position |

### Why this setup validates the claim

The central claim is that forcing causal, block-by-block boundary commitment measures evidence identifiability that static evaluation overstates. Static EvaOmni evaluation tests the upper bound: it supplies the boundary, so any gap between it and StreamEval isolates exactly the boundary-discovery component, which is the quantity the streaming penalty is defined to capture. The format-matched static-with-events control tests the load-bearing assumption that the penalty is not merely prompt-format sensitivity: it uses the same open/close event grammar and output format as StreamEval but provides the whole video, so the difference between static-with-events IoU and streaming IoU isolates online commitment from format effects. The retrieval-style segment scorer tests the sub-claim that causal commitment matters: it can score candidate segments against the question but cannot emit an open event before future blocks arrive, so if StreamEval beats it, the gain is attributable to causality rather than to segment scoring quality. The adaptive-segmentation baseline tests the confound we explicitly designed against: if adaptive segmentation matches or beats fixed blocks, our block-size choice is not the source of any penalty, and the measurement is clean. The positional-prior model and content-shuffle ablation test whether streaming IoU reflects content-based localization rather than positional priors: the positional-prior model opens evidence at a fixed block index regardless of content, and the content-shuffle condition moves evidence to a random location; if streaming IoU remains high under content shuffle or is matched by the positional-prior model, the metric is tracking position rather than content. The boundary-shuffled condition tests whether static IoU stays high while streaming IoU collapses when the gold span is shifted by ±1 block, demonstrating that the protocol measures boundary discovery rather than span confirmation. The patience-k ablation isolates the answer-stability sub-claim by removing confirmation while holding boundary emission fixed, so any drop in answer correctness is attributable to premature answering rather than to boundary quality. Temporal IoU is the right metric because it is EvaOmni's native static metric, making the streaming penalty a directly interpretable delta rather than a cross-metric comparison.

### Expected outcome and causal chain

**vs. Static EvaOmni evaluation (span consumption)** — On a case where the gold visual evidence span is a 4-second interval in the middle of a 60-second video, the static baseline produces near-ceiling IoU because the prompt effectively supplies the boundary decision and the model only has to confirm it. Our method instead must emit an OPEN_VISUAL event at the correct block before seeing subsequent blocks, so errors in localization surface as span offsets. We expect a clear streaming penalty concentrated on questions whose evidence is short or late in the video, with near-parity on questions whose evidence occupies most of the clip.

**vs. Static-with-events evaluation (format-matched control)** — On the same 4-second evidence case, the static-with-events baseline sees the whole video but must still emit open/close events in the same grammar as StreamEval, so it does not benefit from the boundary decision being supplied. We expect the static-with-events IoU to be lower than static span-consumption IoU but higher than streaming IoU, with the difference between static-with-events and streaming isolating the cost of online commitment from prompt-format effects. If the streaming penalty computed against static-with-events is near zero, the original penalty was driven by format sensitivity rather than boundary discovery.

**vs. Retrieval-style segment scorer** — On a case where the question's evidence is only identifiable after integrating an audio cue that precedes the visual event, the retrieval baseline produces a high-scoring but wrong segment because it scores each candidate against the question independently and cannot carry causal state across blocks. Our method instead maintains streaming state and opens the audio channel at the cue block, then the visual channel at the dependent event, so the joint span reflects the cross-modal dependency. We expect a noticeable gap on cross-modal-dependency questions and parity on single-modality questions.

**vs. Adaptive-segmentation boundary detector** — On a case where a human span is 0.9 seconds, shorter than our 1.5-second block, the adaptive baseline can in principle match it exactly because its segmentation granularity follows content, while our fixed blocks cannot resolve below 1.5 seconds. Our method instead accepts this quantization cost but avoids the confound of a learned segmentation policy deciding boundaries on the model's behalf. We expect our strict IoU to trail slightly on sub-1.5-second spans while matching or exceeding adaptive segmentation on multi-second spans, and the 0.5s-tolerance IoU to close most of that gap.

**vs. Positional-prior model (fixed block index)** — On a case where the model ignores content and always opens evidence at block 2, the positional-prior model produces a fixed span regardless of where the true evidence lies. We expect its streaming IoU to be near zero on average but non-trivial on questions whose evidence happens to align with the fixed index. If the positional-prior model achieves non-trivial streaming IoU, we will report it as a floor and condition the streaming penalty on content-shuffle results to show that our method's IoU exceeds this floor.

**vs. Boundary-shuffled condition (±1 block shift)** — On a case where the gold span is supplied but shifted by ±1 block, the static span-consumption baseline still produces high IoU because it only has to confirm the supplied span, while the streaming IoU collapses because the model must discover the true boundary. We expect static IoU to remain high (near ceiling) while streaming IoU drops sharply, demonstrating that the protocol measures boundary discovery rather than span confirmation.

**vs. Content-shuffle condition (evidence moved randomly)** — On a case where the evidence is moved to a random location in the video, a content-based model's streaming IoU should track the new location, while a positional-prior model's IoU should remain unchanged. We expect streaming IoU to follow the shuffled evidence location, confirming that the metric reflects content-based localization rather than positional bias.

**vs. StreamEval without patience-k confirmation** — On a case where the model answers prematurely before evidence is complete, removing patience-k confirmation leads to lower answer correctness because the model does not wait for additional blocks. We expect a drop in answer correctness when patience-k is removed, isolating the contribution of answer-stability gating.

### What would falsify this idea

If the streaming penalty computed against the format-matched static-with-events control is uniform across questions regardless of whether evidence is short, late, or cross-modal, rather than concentrated on the subsets where causal commitment is predicted to hurt, then the penalty reflects prompt-format sensitivity rather than boundary discovery and the central claim is wrong. Likewise, if the positional-prior model achieves non-trivial streaming IoU or the content-shuffle condition does not reduce streaming IoU, the metric is tracking position rather than content, and the claim that streaming IoU measures evidence identifiability is unsupported.

## References

1. EvaOmni: An Evidence-Grounded Benchmark for Audio-Visual Reasoning
