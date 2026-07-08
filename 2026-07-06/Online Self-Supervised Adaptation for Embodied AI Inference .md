# Online Self-Supervised Adaptation for Embodied AI Inference Runtimes

## Motivation

Existing embodied AI inference runtimes like Embodied.cpp assume a fully trained model and do not address out-of-distribution generalization at runtime. The root cause is that the model's representations are fixed, but deployment environments can shift due to lighting, geometry, or dynamics. By incorporating test-time adaptation using self-supervised objectives on the robot's continuous video stream, we can adapt model representations without labeled data.

## Key Insight

The temporal consistency of consecutive video frames in embodied agents provides a natural supervisory signal for contrastive learning, enabling online adaptation that improves representation robustness without requiring ground truth labels.

## Method

```python
# OSSA Module integrated into Embodied.cpp execution path
# Hyperparameters: lr=0.001, momentum=0.9, update_interval=10 frames, queue_size=128

class OSSA:
    def __init__(self, backbone, adapter_rank=16):
        self.backbone = backbone  # pre-trained backbone
        self.adapter = LowRankAdapter(backbone.hidden_dim, rank=adapter_rank)
        self.queue = deque(maxlen=queue_size)
        self.optimizer = SGD(self.adapter.parameters(), lr=lr, momentum=momentum)
        self.frame_counter = 0

    def process_frame(self, frame):
        self.queue.append(frame)
        self.frame_counter += 1
        if self.frame_counter % update_interval == 0 and len(self.queue) >= 2 * update_interval:
            # Sample two batches: recent frames (positives) and older frames (negatives)
            batch1 = random.sample(list(self.queue)[-update_interval:], min(update_interval, len(self.queue)))
            batch2 = random.sample(list(self.queue)[:-update_interval], min(update_interval, len(self.queue)))
            # Compute contrastive loss between representations of temporally close vs distant frames
            z1 = self.backbone(batch1, adapter=self.adapter)
            z2 = self.backbone(batch2, adapter=self.adapter)
            loss = NTXentLoss(z1, z2, temperature=0.1)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        # Run standard inference with updated adapter
        output = self.backbone(frame, adapter=self.adapter)
        return output
```

**(A) What it is:** OSSA (Online Self-Supervised Adaptation) is a lightweight module that integrates into Embodied.cpp's runtime execution path. It takes the robot's camera stream as input and outputs updated adapter weights for the backbone model, enhancing out-of-distribution generalization at test time.

**(B) How it works:** Pseudocode above details the core operation: a queue stores recent frames; every `update_interval` frames, two batches are sampled (recent vs. older) and a contrastive loss updates a low-rank adapter via SGD with momentum. The adapted backbone then processes the next frame for task inference.

**(C) Why this design:** We chose a lightweight low-rank adapter over full finetuning to minimize computational overhead during runtime, accepting the trade-off that limited adaptation capacity may not cover large distribution shifts. We update the adapter every 10 frames to balance adaptation speed with inference latency—updating too frequently would spike latency, while too infrequently delays adaptation. We use a memory queue with past frames to construct contrastive pairs, enabling temporal consistency learning without storing all frames, but this introduces a staleness bias where older frames may be less representative of current distribution. We use SGD with momentum rather than Adam to reduce memory and compute overhead, at the cost of slower convergence and sensitivity to learning rate tuning. Finally, we chose NTXent loss over other contrastive losses because it is well-studied for unsupervised representation learning, but it assumes temporally close frames are positive pairs—this fails under rapid viewpoint changes or occlusions.

**(D) Why it measures what we claim:** The contrastive loss between recent and older frame representations measures feature consistency because it assumes that consecutive frames from a robot's camera are semantically similar under invariant representations; this assumption fails during abrupt motion or scene transitions, in which case the loss may enforce incorrect invariances. The low-rank adapter update operationalizes online adaptation by modifying only a small subspace of features, assuming the domain shift is low-rank; this fails when the shift requires global feature changes (e.g., full lighting change). The update interval hyperparameter controls the adaptation rate, measuring the trade-off between stability and plasticity; it assumes the environment changes slowly relative to the interval, which fails in dynamic environments with rapid perturbations. Each component is linked to a motivation-level concept (consistency, adaptability, reactivity) under explicit assumptions that define its validity.

## Contribution

(1) A novel online self-supervised adaptation module (OSSA) for embodied AI runtimes that fine-tunes model representations on the fly using contrastive learning on video streams. (2) A design principle for integrating lightweight adaptation hooks into modular inference runtimes without breaking the execution path, demonstrated on Embodied.cpp's architecture. (3) An analysis of trade-offs in temporal consistency learning for online adaptation, including bound conditions on update frequency and adapter capacity.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|-----------------------|
| Dataset | Habitat ObjectNav (dynamic lighting) | Realistic embodied task with domain shifts |
| Primary metric | Success Rate (SR) | Measures task completion under adaptation |
| Baseline 1 | Frozen backbone (no adaptation) | Shows necessity of online adaptation |
| Baseline 2 | Online MDA | Non-contrastive online feature alignment |
| Baseline 3 | Test-time training (TTT) | Full fine-tuning upper bound (at high cost) |
| Ablation | OSSA w/o contrastive loss | Isolates effect of contrastive objective |

### Why this setup validates the claim

This combination of dataset, baselines, and metric forms a falsifiable test of the central claim that OSSA’s temporal contrastive adaptation improves out-of-distribution generalization. The Habitat ObjectNav benchmark under dynamic lighting introduces realistic continuous distribution shifts (illumination changes), which are exactly the scenarios where online adaptation should help. Frozen backbone serves as the null hypothesis: if OSSA fails to improve, its adaptation mechanism is ineffective. Online MDA tests whether a non-contrastive alignment method suffices; if OSSA outperforms it, temporal contrastive learning adds value beyond simple feature distribution matching. TTT provides an upper bound on achievable performance through expensive full-model adaptation, highlighting the trade-off between adaptation quality and runtime efficiency. The ablation (removing contrastive loss) directly tests whether the temporal self-supervision is the critical component. The metric Success Rate directly quantifies task-level generalization, which is the ultimate goal of embodied AI. If OSSA’s gains are concentrated in segments with significant lighting changes (where temporal coherence matters most) rather than uniformly, the causal mechanism is confirmed.

### Expected outcome and causal chain

**vs. Frozen backbone** — On a case where the robot enters a suddenly darker corridor, the frozen backbone produces poor features because it was trained under uniform lighting and cannot adapt. Our method updates the low-rank adapter via contrastive loss on recent frames (the bright frames before transition) to maintain feature consistency, so the adapted representation generalizes better. We expect a noticeable gap (e.g., +20% SR) on scenes with abrupt luminance change, but parity on well-lit static scenes.

**vs. Online MDA** — Online MDA aligns feature distributions by minimizing margin discrepancy but does not exploit temporal structure. When the robot rotates rapidly, MDA may misalign distributions due to frame-by-frame shifts, causing oscillation. Our method uses contrastive pairs from a memory queue, encouraging temporal smoothness, so adaptation is more stable under motion. We expect OSSA to show higher SR (e.g., +10%) on sequences with significant ego-motion, with similar performance on slow motion.

**vs. Test-time training (TTT)** — TTT updates the entire backbone on each test sample via self-supervised loss, but this is slow and can overfit to a single frame. On a fast-moving robot, TTT fails to keep up, causing latency spikes and poor adaptation. Our method’s lightweight adapter update every 10 frames avoids overfitting and maintains real-time throughput, so we expect OSSA to achieve comparable SR (within 5%) but with much lower inference latency (not measured in SR). On static scenes, TTT may slightly outperform due to full capacity, but dynamic scenes narrow the gap.

### What would falsify this idea

If OSSA’s success rate is no better than frozen backbone across all dynamic conditions, or its gains are uniform across static and dynamic scenarios (rather than concentrated where temporal consistency is challenged), the central claim would be falsified; the contrastive adaptation would be ineffective or the assumptions about distribution shift slow change would be violated.

## References

1. Embodied.cpp: A Portable Inference Runtime of Embodied AI Models on Heterogeneous Robots
