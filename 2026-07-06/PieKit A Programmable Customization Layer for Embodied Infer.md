# PieKit: A Programmable Customization Layer for Embodied Inference Runtimes

## Motivation

Embodied.cpp introduces a five-layer fixed architecture for deploying embodied AI models, but this design lacks programmability — users cannot insert custom pre/post-processing steps or modify control flow without forking the runtime. This limitation restricts adoption by researchers who need to experiment with pipeline variants, the very flexibility that prior work like Pie's inferlet decomposition provided but was removed in the name of portability. The root cause is that the execution path abstraction is treated as a rigid template rather than an extensible interface.

## Key Insight

Because the runtime's layer interfaces are already abstracted from hardware specifics, inserting a handler layer at those interfaces preserves portability while enabling arbitrary user-defined transformations.

## Method

PieKit: A Programmable Customization Layer for Embodied Inference Runtimes

(A) **What it is**: PieKit is a hook-based customization layer that intercepts data and control tokens at each of Embodied.cpp's five layers via user-registered handler functions, allowing arbitrary pre/post-processing, control flow modification, and backbone override without modifying the runtime core. Input: a set of handler definitions (function pointer + hook point). Output: the same runtime execution but with user-defined transformations applied at specified stages. Implementation roadmap: from unmodified Embodied.cpp to PieKit in 4 weeks on a single GPU with moderate RAM (e.g., 16 GB). Hardware requirements: single GPU (e.g., NVIDIA RTX 3090), 16 GB RAM, and 100 GB storage for Open X-Embodiment dataset.

(B) **How it works** (Pseudocode):
```python
class PieKitRuntime:
    # Layer hooks: dict mapping layer name to list of (pre_hook, post_hook) pairs
    hooks = {}
    # Mutex for thread-safe handler execution
    mutex = threading.Lock()
    
    def register_hook(self, layer, phase, handler):
        # phase is 'pre' or 'post'
        with self.mutex:
            self.hooks.setdefault(layer, []).append((phase, handler))
    
    def execute_model(self, input_data):
        # Input adapters
        x = self.run_input_adapters(input_data)
        x = self.apply_hooks('input', 'post', x)
        
        # Sequence builders
        seq = self.run_sequence_builders(x)
        seq = self.apply_hooks('seq', 'post', seq)
        
        # Backbone execution
        features = self.run_backbone(seq)
        features = self.apply_hooks('backbone', 'post', features)
        
        # Head plugins
        heads = self.run_heads(features)
        heads = self.apply_hooks('head', 'post', heads)
        
        # Deployment adapters
        output = self.run_deployment_adapters(heads)
        output = self.apply_hooks('deploy', 'post', output)
        return output
    
    def apply_hooks(self, layer, phase, data):
        with self.mutex:
            for (p, handler) in self.hooks.get(layer, []):
                if p == phase:
                    data = handler(data)
        return data
```
Handlers can return special tokens (e.g., `SkipLayer`, `ReorderLayer`, `ConditionalBranch`) to modify control flow. For instance, `SkipLayer` bypasses subsequent layers, `ReorderLayer` swaps the order of the next two layers, and `ConditionalBranch` selects an alternative layer based on runtime state.

(C) **Why this design**: We chose a hook-based architecture over a plugin system because hooks have minimal overhead (one callback per layer) and are easier to integrate with existing C++ pipelines, at the cost that hooks cannot easily modify cross-layer interactions. We chose to expose the data tokens (raw tensors, sequences, features) directly to handlers rather than a higher-level API, accepting the risk that users may break abstraction but gaining maximum flexibility. We chose to support both synchronous (blocking) and asynchronous (deferred) handlers, allowing latency-sensitive users to avoid stalls while asynchronous handlers add complexity in thread-safety. To ensure thread-safety, we use a per-hook mutex lock as shown in pseudocode. Finally, we decided to implement hooks at the exact boundary of each of the five layers to mirror Embodied.cpp's original decomposition, trading potential redundancy for clarity over alternative approaches like single global hooks. **Load-bearing assumption**: The set of three control tokens (`SkipLayer`, `ReorderLayer`, `ConditionalBranch`) is sufficient to implement all desired control-flow modifications in embodied AI pipelines. We will verify this by testing on a benchmark covering 10 common control-flow patterns (e.g., skip, reorder, conditional, loop).

(D) **Why it measures what we claim**: The computational quantity `handler execution count` and `data transformation throughput` measure programmability — the ability for users to inject custom logic — because the hook interface ensures every transformation is applied at a point where the data has the same semantic meaning as the layer's output; this assumption fails when a user's intended modification requires access to internal state not exposed by the data token (e.g., per-device memory layout), in which case the hook count overestimates programmability as not all customizations are expressible. The `SkipLayer` token measures control-flow modification because the runtime's layer sequencing is entirely defined by the set of tokens returned; this assumption fails when a user wants to reorder layers rather than skip them, in which case the token captures only binary control flow. With extended tokens (`ReorderLayer`, `ConditionalBranch`), we assume all common patterns are covered.

## Contribution

(1) A hook-based customization layer (PieKit) for Embodied.cpp that exposes each of the five original layers to user-defined handlers, enabling arbitrary pre/post-processing and control flow modification without modifying the runtime core. (2) The design principle that runtime programmability can be restored by inserting handlers at the existing abstraction boundaries, preserving portability because the handlers operate on abstract data tokens that are hardware-independent. (3) A set of reference handlers demonstrating common VLA pipeline modifications (e.g., dynamic head swapping, multi-modal fusion point alteration).

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|-----------------------|
| Dataset | Open X-Embodiment dataset | Widely used in embodied AI evaluation. |
| Primary metric | Customization success rate | Directly measures programmability. |
| Secondary metric | Diverse control-flow patterns implemented | Measures expressiveness of control tokens. |
| Microbenchmark metric | Hook overhead (mean latency per call) | Quantifies runtime cost of customization. |
| Baseline 1 | Embodied.cpp (no hooks) | Standard runtime without customization. |
| Baseline 2 | Plugin-based extension | Higher overhead but similar flexibility. |
| Baseline 3 | TensorFlow hooks (tf.function overlays) | Common ML runtime hook mechanism. |
| Baseline 4 | ROS2 lifecycle nodes | Robotics middleware with customization. |
| Ablation-of-ours | PieKit no skip token | Tests control-flow contribution. |

### Why this setup validates the claim

The claim is that PieKit's hook-based design improves programmability and control-flow modification over existing runtimes. Open X-Embodiment includes tasks requiring diverse data transformations and control-flow changes, providing a realistic testbed. The primary metric (customization success rate) directly measures whether users can implement desired modifications on a set of 20 predefined customization tasks (e.g., custom normalization, layer skipping, reordering, conditional branching). The secondary metric counts how many of the 10 common control-flow patterns (from the load-bearing assumption verification) are expressible. The microbenchmark measures latency per hook call to ensure practicality. Embodied.cpp without hooks shows the deficiency of rigid pipelines; plugin-based extension isolates hook benefits; TensorFlow hooks and ROS2 lifecycle provide external context. The ablation removes the skip token, isolating its effect. Together, these choices form a controlled experiment: if PieKit yields higher success rates and broader pattern coverage with acceptable overhead, the claim is supported; if not, falsified.

### Expected outcome and causal chain

**vs. Embodied.cpp (no hooks)** — On a case requiring custom normalization before backbone, the baseline fails to inject any modification because its pipeline is fixed. Our method instead allows registering a pre-backbone hook that transforms tensors, so we expect a clear success gap on such tasks (e.g., 0% vs. 90% success rate).

**vs. Plugin-based extension** — On a case needing to skip layers for efficiency, the plugin system incurs high overhead due to message passing. Our method's hook interface is minimal, so we expect lower latency and higher throughput for control-flow modifications (e.g., 5x latency reduction on skip tasks).

**vs. TensorFlow hooks** — TensorFlow hooks only operate on graph-level execution (e.g., before/after tf.function), not on layer internals. Embodied.cpp's five-layer decomposition is not exposed in TensorFlow, so TF hooks cannot replicate the fine-grained control. We expect PieKit to succeed on tasks requiring per-layer intervention (e.g., 80% vs. 20% success rate).

**vs. ROS2 lifecycle** — ROS2 lifecycle nodes allow state transitions but not on-the-fly data interception within a single node. For tasks like modifying features after backbone, ROS2 requires additional nodes and IPC overhead. PieKit offers lower overhead and finer granularity (e.g., 10x lower latency for inter-layer modifications).

**vs. PieKit no skip token** — On a case where skipping a layer is essential, the ablation cannot alter control flow, so it fails. Our full method handles it via SkipLayer token, so we expect a gap (e.g., 0% vs. 80% success rate). For reordering or conditional branching, the ablation also fails, while our full method with extended tokens succeeds (e.g., 0% vs. 70% success rate).

**Microbenchmark** — Expected mean latency per hook call ≤ 5 μs (C++ implementation with mutex), making total overhead < 25 μs per full pipeline pass. This is acceptable for real-time control loops. If latency exceeds 50 μs, the approach may be too slow for real-time applications.

### What would falsify this idea

If customization success rate is similar between PieKit and the no-hooks baseline, then the hook interface fails to improve programmability. Alternatively, if the skip token ablation performs equally to the full method on tasks requiring skipping, then the control-flow claim is unsupported. If the microbenchmark overhead exceeds 50 μs per hook, the approach may be impractical.

## References

1. Embodied.cpp: A Portable Inference Runtime of Embodied AI Models on Heterogeneous Robots
