# Modular Adapter Interface for General Embodied AI Inference Pipelines

## Motivation

Embodied.cpp's five-layer architecture assumes a common execution path shared by VLA models, but this assumption fails for non-VLA architectures (e.g., world models, diffusion-based policies) that have fundamentally different computation structures. This forces practitioners to reimplement inference runtimes for each new model type, hindering portability and reuse.

## Key Insight

By decoupling the runtime's control flow from model-specific pipeline logic via a composable adapter interface, any model can be expressed as a sequence of custom stages that the runtime orchestrates, preserving generality without sacrificing performance.

## Method

(A) **What it is**: The 'GenAdapter' interface is a set of C++17 abstract classes that define how a model's inference pipeline is divided into stages (e.g., preprocess, backbone, postprocess) and how data flows between them. Users implement subclasses for their model, and the runtime uses these to construct an execution graph. The interface assumes that any embodied AI model's inference pipeline can be decomposed into a static directed acyclic graph (DAG) of stages with fixed input/output keys. Models with dynamic control flow (e.g., loops, early-exit) must be implemented as a single monolithic stage (reducing generality) or use the optional dynamic edge extension described below. We extend the adapter interface to support conditional edges (enabled/disabled based on runtime tensor values) and dynamic node creation (via a special stage that can add new stages to the graph). This extension is optional; models that do not use it maintain static graph guarantees.

(B) **How it works**:
```python
# Define abstract base class for a pipeline stage
class Stage:
    def configure(config: dict) -> None: ...
    def execute(input: TensorDict, context: dict) -> TensorDict: ...
    def input_keys() -> list[str]: ...
    def output_keys() -> list[str]: ...

# Adapter: user specifies list of stages and connections
class ModelAdapter:
    stages: list[Stage]
    edges: list[tuple[int, str, int, str]]  # (from_stage_idx, from_key, to_stage_idx, to_key)
    
    def build_graph(self) -> DAG:
        graph = DAG()
        for i, stage in enumerate(stages):
            graph.add_node(i, stage)
        for src_stage, src_key, dst_stage, dst_key in edges:
            graph.add_edge(src_stage, dst_stage, src_key, dst_key)
        return graph

# Runtime execution
runtime = Runtime(backend="CUDA")
adapter = UserModelAdapter()  # user implements
graph = adapter.build_graph()
runtime.execute(graph, inputs)
```
Tensor dictionaries use key-value pairs where keys are strings and tensors are contiguous float arrays in row-major order. The DAG nodes are index-based and edges store source stage index, output key, destination stage index, input key. The runtime uses a topological order for execution.

(C) **Why this design**: We chose a stage-based adapter over a monolithic plugin interface because it forces explicit data flow specification, enabling the runtime to optimize memory transfers between stages (e.g., fusing consecutive GPU ops). We chose abstract base classes rather than a YAML configuration file because compile-time type checking catches mismatched tensor shapes earlier. We chose to expose input/output keys at each stage so the runtime can verify connectivity at load time, sacrificing some flexibility for safety. We also chose not to include a built-in scheduler to keep the interface minimal; users who need custom scheduling can implement a single stage that performs internal orchestration, accepting potential inefficiency for generality. We limit the interface to static DAGs to simplify optimization, acknowledging that dynamic models must either be implemented as a single monolithic stage or use our optional extension for conditional edges and dynamic node creation.

(D) **Why it measures what we claim**: The set of stages and edges in the adapter measures the 'generality' of the inference pipeline because a model that can be decomposed into stages with clear data dependencies can be executed by any runtime supporting this interface; this assumption fails when a model requires dynamic control flow (e.g., loops over unknown iterations), in which case the adapter must either be implemented as a single monolithic stage (reducing generality) or extended with conditional edges. The number of user-implemented stages versus existing reusable stages measures 'effort of porting', because fewer custom stages imply more reuse; this assumption fails if existing stages have subtle model-specific assumptions (e.g., input normalization parameters), in which case the metric underestimates adaptation effort. To calibrate, we record the time to implement custom stages from scratch versus reuse. The DAG structure measures 'pipeline parallelism opportunity' because stages with no dependencies can be executed on different devices; this assumption fails if stages share GPU memory state (e.g., a diffusion model's latent buffer), in which case parallel execution would cause race conditions. Therefore, we include a runtime check that detects shared state by tracking tensor lifetimes, and if conflict is detected, we serialize execution.

## Contribution

(1) Introduces GenAdapter, a modular adapter interface that allows users to define custom inference pipelines for any embodied AI model, decoupling the runtime from model-specific execution paths. (2) Provides a taxonomy of typical pipeline stages for VLA, world models, and diffusion-based policies, along with reusable stage implementations. (3) Demonstrates that the interface can express the pipelines of Embodied.cpp's VLA models and two additional non-VLA models (a simple world model and a diffusion policy) without modifying the runtime core.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | VLA Model Suite (HY-VLA, pi0, custom world model) | Covers diverse model pipelines and tasks, including those with dynamic control flow |
| Primary metric | Inference latency (ms) | Detects latency improvements from interface optimizations |
| Secondary metrics | Runtime failure rate (%), Porting time (hours), GPU memory utilization (MB) | Captures safety, adaptation effort, and resource usage |
| Baseline | Monolithic Plugin Interface | Tests necessity of stage decomposition |
| Baseline | YAML-Configured Pipeline | Tests value of compile-time type checking |
| Baseline | Python Eager Execution | Tests gain from structured execution graph |
| Baseline | Single-Stage Adapter | Ensures implementation effort for multiple stages is justified |
| Ablation-of-ours | GenAdapter w/o DAG Fusion | Isolates effect of DAG-based optimizations |

### Why this setup validates the claim
This combination of dataset, baselines, and metric forms a falsifiable test of the central claim that GenAdapter's stage-based interface enables generality, safety, and performance improvements. The VLA Model Suite includes models with varying pipeline structures (e.g., independent stages vs. data-dependent loops), allowing evaluation across different decomposition scenarios. Monolithic baseline tests whether stage decomposition alone reduces latency, while YAML baseline tests if compile-time checking prevents runtime errors, Python baseline tests if the DAG execution graph yields efficiency gains, and the single-stage baseline tests if splitting into multiple stages is worthwhile. The ablation removes DAG fusion to isolate the performance contribution of the graph structure. Inference latency is the right metric because it directly reflects the runtime optimizations (memory fusion, parallelism) that the interface enables. If GenAdapter improves latency across diverse models over all baselines, while the ablation shows a smaller gain, the claim is supported. Conversely, if latency is no better than monolithic on decomposable models, the interface fails.

### Expected outcome and causal chain

**vs. Monolithic Plugin Interface** — On a model with two independent post-processing stages (e.g., object detection and depth estimation), the monolithic baseline must execute them sequentially within a single stage, missing parallelism. Our method splits them into separate stages, allowing the runtime to execute them on different CUDA streams. We expect a latency reduction of 20–30% on such models, observable as a noticeable gap on the subset with clear stage separation but parity on sequential models.

**vs. YAML-Configured Pipeline** — On a model where a stage outputs a tensor of shape (1, 3, 224, 224) but the next stage expects (1, 4, 224, 224), the YAML pipeline would silently crash at runtime or produce wrong results. Our adapter catches this mismatch at graph build time via type checking over input/output keys, raising a clear error. We expect zero runtime shape mismatches with GenAdapter, versus multiple crashes with YAML, measured by runtime failure rate across 100 runs.

**vs. Python Eager Execution** — On a model with repeated identical operations (e.g., a convolutional layer applied to multiple inputs), Python eager executes each operation independently with Python overhead. Our runtime constructs a DAG and fuses consecutive ops on the same device, reducing kernel launch overhead. We expect 15–20% lower latency on repetitive pipelines, observable as a consistent latency gap across all models with repeated operations.

**vs. Single-Stage Adapter** — On a model with clearly decomposable stages (e.g., a VLM with separate vision encoder and LLM), the single-stage adapter executes all operations sequentially within one stage, missing parallelism and fusion opportunities. Our method splits them into multiple stages, enabling pipeline parallelism and operator fusion. We expect a latency reduction of 10–15% on such models, observable as a gap for decomposable models but parity for inherently sequential pipelines.

**Ablation (GenAdapter w/o DAG Fusion)** — By disabling DAG-based fusion (e.g., fusing consecutive GPU kernels), we expect latency to increase by 5–10% compared to full GenAdapter, isolating the performance contribution of the DAG structure.

### What would falsify this idea
If GenAdapter shows no latency improvement over the monolithic baseline on models with clearly decomposable stages, or incurs higher latency on simple models, the claim that the interface enables effective optimizations would be falsified. Specifically, if the ablation (w/o DAG fusion) performs similarly to full GenAdapter, then the DAG structure is not contributing to performance.

## References

1. Embodied.cpp: A Portable Inference Runtime of Embodied AI Models on Heterogeneous Robots
