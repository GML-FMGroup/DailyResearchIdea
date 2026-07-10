# Self-Supervised Tactile-Visual Alignment via Stick-Slip Temporal Synchronization

## Motivation

Existing tactile alignment methods like Splash (from 'Wake up for Touch!') rely on supervised paired visuo-tactile data from specific sensors, requiring costly synchronized high-resolution recordings and limiting generalization across sensor types. This sensor-specific dependence is a structural limitation because both technical approaches assume that training requires dense correspondences between visual and tactile modalities, which is not scalable. We propose to exploit stick-slip events—a universal physical phenomenon during tactile interaction—as a natural, unsupervised temporal synchronization signal, enabling alignment without paired data.

## Key Insight

Stick-slip events produce characteristic high-frequency vibrations in tactile signals and concurrent discontinuities in visual motion, providing a sensor-invariant temporal correspondence that can be detected without supervised data.

## Method

### (A) What it is
**TV-STS** (Tactile-Visual Synchronization via Stick-Slip) is a self-supervised method that aligns tactile and visual representations by detecting and matching stick-slip events from raw sensor streams. Inputs: unpaired tactile time series (from any sensor) and video frames. Outputs: aligned embeddings from a tactile encoder and a visual encoder.

### (B) How it works
```python
# Pseudocode for training loop
for each unlabeled pair of tactile stream T and video V (roughly synchronized within ±5s):
    # 1. Detect tactile stick-slip events
    tactile_energy = bandpass_filter(T, [0.5, 5.0] kHz) ** 2
    tactile_events = argmax(tactile_energy > threshold_τ)  # τ = 3x median energy
    
    # 2. Detect visual stick-slip events via optical flow
    flow = compute_optical_flow(V)  # e.g., RAFT
    flow_mag = magnitude(flow)
    flow_change = temporal_derivative(flow_mag)  # diff between consecutive frames
    visual_events = argmax(flow_change > threshold_ν)  # ν = 2x std of flow_change
    
    # 3. Match events across modalities (nearest neighbor within window w=10s)
    matches = []
    for t_e in tactile_events:
        candidates = [v_e for v_e in visual_events if abs(t_e - v_e) < w]
        if candidates:
            matches.append((t_e, argmin(abs(t_e - candidates))))
    
    # 4. Contrastive learning on event-matched pairs
    for (t, v) in matches:
        z_t = tactile_encoder(T_window_around_t)  # window length 100ms
        z_v = visual_encoder(V_frame_around_v)
        loss = InfoNCE(z_t, z_v, negatives=pairs from other events in batch)
        update encoders via momentum contrast (MoCo-style)
```
Hyperparameters: threshold τ (3x median), ν (2x std), window w (10s), window length (100ms), batch size 64.

### (C) Why this design
We chose a simple energy threshold over a learned event classifier (e.g., a CNN) because our goal is sensor-agnostic generalization: a learned classifier would require labeled stick-slip events from each sensor, reintroducing the data-dependence we aim to eliminate. We accept the cost that a fixed threshold may be suboptimal for sensors with very different noise floors, so we normalize by median energy to adapt automatically. For visual event detection, we use optical flow discontinuities rather than object recognition or motion detectors based on bounding boxes, because flow is texture-independent: it works on any surface visible in the video, making it compatible with diverse tactile interactions. The trade-off is that flow is more computationally expensive and can be noisy under poor lighting. We adopt a nearest-neighbor matching strategy within a conservative temporal window w=10s rather than dynamic time warping (DTW) because DTW assumes continuous alignment and is sensitive to event density; nearest-neighbor is more robust to false positives and runs in O(n log n). The cost is that very dense events may be matched incorrectly, but with w=10s and typical event rates (~1-5 events/s), this is manageable. We use InfoNCE contrastive loss with momentum encoders (following MoCo) because it scales to many negative pairs without large batch sizes, which is important for small batch scenarios with limited event counts. The trade-off is the memory overhead of the momentum queue, but this is acceptable given modern GPU hardware.

### (D) Why it measures what we claim
The energy threshold on bandpassed tactile signal (step 1) measures stick-slip events because stick-slip produces impulsive high-frequency vibrations that exceed the median energy (assumption: events are sparse and energetic); this assumption fails when the tactile sensor has low bandwidth or high noise, in which case the threshold reflects noise rather than events. The temporal derivative of optical flow magnitude (step 2) measures stick-slip events because slip causes sudden velocity changes in the tactile region (assumption: visual motion is dominated by slip rather than camera or object movement); this assumption fails in scenes with rapid camera motion, where flow change reflects egomotion. The nearest-neighbor matching with window w (step 3) measures temporal coincidence because events co-occur within a short time window (assumption: tactile and visual streams are roughly synchronized within ±5s); this assumption fails when the sensor clocks are misaligned by more than w, leading to false pairings. The contrastive loss (step 4) measures alignment by pulling matched event embeddings together (assumption: the temporal proximity implies semantic correspondence); this assumption fails if the same event type (e.g., finger sliding) produces different cross-modal patterns due to variation in materials, in which case the loss enforces false correspondence.

## Contribution

['(1) A self-supervised method (TV-STS) that aligns tactile and visual representations by detecting and matching naturally occurring stick-slip events, eliminating the need for paired supervised data.', '(2) The insight that high-frequency vibrations and optical flow discontinuities provide a universal, sensor-agnostic temporal synchronization signal across diverse tactile sensors and visual conditions.', '(3) An empirical demonstration that TV-STS achieves competitive alignment quality on multiple tactile sensors (e.g., BioTac, GelSight) without any sensor-specific calibration, validated through downstream tasks like tactile-visual reasoning.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|-----------------------|
| Dataset | SSVTP | Standard visuo-tactile benchmark with events. |
| Primary metric | Recall@1 | Measures cross-modal retrieval accuracy. |
| Baseline 1 | Supervised Event Classifier | Requires labeled events, sensor-specific. |
| Baseline 2 | Raw Signal Cross-Correlation | No learned representation, naive temporal. |
| Ablation | TV-STS w/o Contrastive | Isolates effect of contrastive learning. |

### Why this setup validates the claim
This combination tests TV-STS’s central claim: self-supervised stick-slip event alignment can produce effective cross-modal embeddings without labeled data. The supervised baseline tests whether our method can approach performance when labels are available; if TV-STS matches it, then self-supervision is sufficient. The raw correlation baseline tests the necessity of learned encoders over simple temporal matching; a gap would show that learned representations capture deeper semantics. The ablation isolates the contrastive loss contribution—if TV-STS w/o contrastive underperforms, then the event-matched InfoNCE is critical. Recall@1 directly measures whether the aligned embeddings can retrieve the corresponding tactile from visual (or vice versa), a task that requires true semantic correspondence. This setup yields a clear falsifiable pattern: gains must be specific to events, not due to spurious correlations.

### Expected outcome and causal chain

**vs. Supervised Event Classifier** — On a case where tactile sensor type changes (e.g., from BioTac to GelSight), the supervised classifier fails because its event detector is sensor-specific and lacks labeled data for the new sensor. TV-STS instead adapts via energy threshold normalized by median, so it generalizes without retraining. We expect TV-STS to achieve comparable Recall@1 on the original sensor but significantly higher on a cross-sensor evaluation (e.g., gap >10%).

**vs. Raw Signal Cross-Correlation** — On a case where tactile and visual streams have different temporal offsets or noise, the raw correlation fails because it cannot account for variable signal shapes or non-linearities. TV-STS overcomes this by learning invariant embeddings through contrastive learning on detected events. We expect TV-STS to outperform raw correlation by a large margin (e.g., >20% Recall@1) especially under moderate temporal misalignment.

**vs. TV-STS w/o Contrastive** — On a case where multiple events occur close together, the ablation (which only uses temporal matching) pairs events incorrectly because nearest-neighbor can mismatch dense events. TV-STS with contrastive learning refines embeddings to be distinct per event, reducing false matches. We expect a moderate but consistent gap (e.g., 5-10% Recall@1) in high-density event scenarios.

### What would falsify this idea
If TV-STS shows no improvement over random alignment on subsets with clear stick-slip events (e.g., high-energy tactile bursts with concurrent optical flow spikes), then the central claim that self-supervised stick-slip alignment yields meaningful embeddings is false.

## References

1. Wake up for Touch! Mask-isolated Tactile Alignment Learning in MLLMs
2. Mitigating Catastrophic Forgetting in Large Language Models with Forgetting-aware Pruning
3. Qwen2.5-VL Technical Report
4. Lottery Ticket Adaptation: Mitigating Destructive Interference in LLMs
