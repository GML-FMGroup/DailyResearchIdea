# StreamVLA: Bounded-Latency Action Decoding from Streaming Visual Features for Real-Time Robot Control

## Motivation

Existing vision-language-action (VLA) models, such as the one in 'From Foundation to Application: Improving VLA Models in Practice', rely on heavy video representation and depth estimation models for predictive dynamics, causing unpredictable and often high inference latency. This violates the real-time control requirement in dynamic environments because the system cannot guarantee a reaction within a fixed time budget. We identify the root cause as the use of full-sequence predictive models that lack architectural constraints on inference time.

## Key Insight

By compressing the visual history into a fixed-size latent representation via a learned query token and a temporal memory buffer, the action decoder can compute the next action in constant time independent of the sequence length, enabling a worst-case latency guarantee.

## Method

We propose StreamVLA, a lightweight action decoder that operates on streaming visual features with bounded inference latency.

(A) **What it is**: StreamVLA takes as input a stream of visual features (output from a frozen vision encoder) and outputs an action command with a latency bound τ. The core is a transformer decoder with a learned query token that attends to a fixed-size FIFO memory buffer of recent visual features. **Explicit assumption**: A fixed-size FIFO memory of K=16 recent visual features captures all task-relevant temporal context for real-time robot control. This assumption will be verified by measuring the distribution of context lengths required in the evaluation dataset (see calibration below).

(B) **How it works** (pseudocode):
```python
def stream_vla_action(visual_stream, memory_buffer, query_token):
    # visual_stream: generator of features from frozen vision encoder
    # memory_buffer: deque of max_len=K, stores past K visual features
    # query_token: learned vector (dim=d=256)
    # Return: action vector and updated memory_buffer
    
    # Step 1: Read next visual feature f from stream
    f = next(visual_stream)
    
    # Step 2: Update memory buffer (FIFO)
    memory_buffer.append(f)
    if len(memory_buffer) > K:  # K=16
        memory_buffer.popleft()
    
    # Step 3: Form input sequence [query_token; memory_buffer tokens]
    # memory_buffer tokens are positional-encoded by recency (sinusoidal positions 0..K-1)
    inputs = [query_token] + memory_buffer  # shape: (K+1) x d
    
    # Step 4: Pass through lightweight transformer decoder (L=2 layers, heads=4, d_ff=512)
    hidden = transformer_decoder(inputs)  # (K+1) x d
    # Step 5: Action head from query token's hidden state
    action = MLP(hidden[0])  # 2-layer MLP: hidden=256, GeLU, output=action_dim
    
    # Latency constraint enforced during training: network must finish within τ=10ms
    # Use a latency-aware loss: L = task_loss + λ * max(0, time_elapsed - τ)
    # where λ=0.1, time_elapsed measured from input ready to output computed (includes any batched processing)
    return action, memory_buffer
```
Hyperparameters: K=16, L=2, λ=0.1, τ=10ms. Vision encoder is frozen pretrained DINOv2+SigLIP (from OpenVLA). Model size: ~8M parameters (decoder+query+MLP), total with encoder frozen: ~0.6B parameters (encoder only). Training on a single A100: ~200 GPU hours for GM-100 (10 epochs).

(C) **Why this design**: We chose a fixed-size FIFO memory over full video history because it bounds the input size and thus the computational cost of attention, accepting the loss of long-term temporal context. We use a learned query token instead of a recurrent state because it allows parallelizable inference and constant-time decoding, whereas RNNs have sequential dependency. We incorporate explicit latency regularization during training (λ * max(0, time_elapsed - τ)) rather than relying on model capacity to naturally achieve fast inference, because prior work (e.g., OpenVLA's 7B model) shows that scaling alone does not guarantee bounded latency; the cost is that the model may sacrifice performance under severe latency constraints.

(D) **Why it measures what we claim**: The component `time_elapsed` measured during each forward pass measures `inference latency` because we assume no non-deterministic hardware or software overhead (e.g., GPU scheduling, memory transfer); this assumption fails when such overhead dominates, in which case `time_elapsed` reflects wall-clock time that may exceed algorithmic complexity. The component `memory_buffer size K` operationalizes `temporal context bound` because K limits the maximum number of past features considered, enforcing a constant-time attention computation; this assumption fails when the effective temporal horizon needed exceeds K, in which case the model acts without sufficient context. The `query_token` operationalizes `fixed-index action decoding` because it always occupies the same position in the input, enabling a consistent attention pattern; this assumption fails if the query token's embedding becomes entangled with the memory content (e.g., through shared residual connections), in which case it no longer serves as a fixed anchor. To mitigate, we add a separate positional embedding for the query token and clip gradients to prevent drift. We will also measure the empirical distribution of required context lengths in the dataset (by analyzing ground-truth dependencies) and report the fraction of steps where K=16 is sufficient.

**Calibration/Verification**: Before training, we analyze the dataset to determine the typical temporal context length needed per task (e.g., number of frames between relevant events). We then verify that for at least 90% of training steps, the required context is ≤K. If not, we increase K accordingly (e.g., K=32) and re-evaluate budget. This ensures the assumption is grounded.

## Contribution

(1) A lightweight action decoder architecture for VLA models that guarantees bounded inference latency by design, using a fixed-size memory and learned query token. (2) A latency-aware training procedure that penalizes actual inference time exceeding a target budget, enabling trade-off between speed and task performance. (3) Empirical demonstration that the proposed method maintains task success comparable to heavy predictive models while reducing worst-case latency by an order of magnitude.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|------------------------|
| Dataset | GM-100 | Tests generalist manipulation with long horizon |
| Primary metric | Task success rate | Direct measure of action correctness |
| Baseline | OpenVLA | Largest VLA model, strong performance |
| Baseline | RT-2 | Prior state-of-the-art VLA baseline |
| Baseline | Tiny-VLA (distilled OpenVLA) | Latency-reduced baseline via distillation |
| Baseline | Pruned OpenVLA (50% parameters) | Latency-reduced baseline via pruning |
| Ablation-of-ours | StreamVLA w/o latency loss | Isolates effect of latency regularization |
| Ablation-of-ours | StreamVLA with K=8, K=32 | Tests sensitivity to buffer size |

### Why this setup validates the claim
This combination forms a falsifiable test of StreamVLA's central claim: a lightweight action decoder with bounded latency can achieve competitive performance on generalist manipulation. GM-100 evaluates diverse long-horizon tasks requiring temporal reasoning, so if bounded memory (K=16) sacrifices performance, success rate will drop. Comparing against large models OpenVLA and RT-2 (no latency constraints) tests whether the performance gap is acceptable given latency benefits. Adding Tiny-VLA and pruned OpenVLA provides baselines from other latency-reduction methods, showing relative advantage of our approach. The ablation without latency loss isolates whether explicit regularization is needed. The K=8 and K=32 ablations test sensitivity to buffer size and help calibrate the assumption. The primary metric (success rate) directly captures action quality, the key trade-off against latency. We also measure inference latency (mean and 99th percentile) on the same hardware (one A100, no batching) to verify bounded timing.

### Expected outcome and causal chain

**vs. OpenVLA** — On a case where a task requires recalling an object location from 30 frames ago (e.g., sequential pick-and-place), OpenVLA can attend to that distant frame because its full-history attention spans the entire video, but this incurs high latency (hundreds of ms) due to large model size. Our StreamVLA, with K=16, loses that distant context, leading to incorrect action (e.g., reaching empty space). However, on tasks where relevant context is within 16 frames (most manipulation steps), our method reacts faster (10ms vs >50ms) and performs similarly. We expect a noticeable gap (e.g., 10% lower success) on long-horizon tasks but near parity (within 2%) on short-horizon ones.

**vs. RT-2** — On a case where a rapid sequence of actions (e.g., wiping a table) requires low-latency feedback, RT-2's full-history processing causes delayed reactions (e.g., overshooting due to stale visual features). Our method, with fixed-size memory and 10ms latency, responds quickly to new observations, enabling precise closed-loop control. RT-2 may still have higher overall success on tasks with long temporal dependencies (e.g., 5% better), but its higher latency (e.g., 30ms) leads to more failures on dynamic tasks. We expect StreamVLA to match or exceed RT-2 on latency-sensitive subsets, while trailing slightly on long-horizon ones.

**vs. Tiny-VLA** — Tiny-VLA is a distilled version of OpenVLA with ~1B parameters and latency ~20ms. StreamVLA (8M decoder) will be faster (10ms). On short-horizon tasks, we expect similar success; on long-horizon tasks, Tiny-VLA may retain more context via distillation from full model, giving it a 3-5% edge. However, our latency guarantee (τ=10ms) is stricter, making StreamVLA preferable for hard real-time systems.

**vs. Pruned OpenVLA** — Pruned OpenVLA (50% parameters) has latency ~30ms and may suffer performance drops on all tasks. StreamVLA should outperform it on latency and match or exceed on success, especially on dynamic tasks.

### What would falsify this idea
If StreamVLA's success rate is uniformly lower than all baselines across all task subsets (especially short-horizon), or if the ablation without latency loss performs equally or better than the full method on latency-constrained tasks, then the central claim that bounded latency can be achieved without major performance loss is wrong. Additionally, if the calibration analysis shows that >10% of steps require context beyond K=16, then the fixed-size assumption fails, and the method must be redesigned (e.g., larger K or learned compression).

## References

1. From Foundation to Application: Improving VLA Models in Practice
2. Igniting VLMs toward the Embodied Space
3. FastUMI: A Scalable and Hardware-Independent Universal Manipulation Interface with Dataset
4. Robotic Control via Embodied Chain-of-Thought Reasoning
5. OpenVLA: An Open-Source Vision-Language-Action Model
6. GELLO: A General, Low-Cost, and Intuitive Teleoperation Framework for Robot Manipulators
7. Scalable. Intuitive Human to Robot Skill Transfer with Wearable Human Machine Interfaces: On Complex, Dexterous Tasks
8. VILA: On Pre-training for Visual Language Models
