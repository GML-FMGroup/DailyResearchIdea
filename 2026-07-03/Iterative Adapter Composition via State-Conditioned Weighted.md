# Iterative Adapter Composition via State-Conditioned Weighted Averaging for Neural Program Synthesis

## Motivation

Existing single-pass adapter generation methods like Program-as-Weights assume a natural language specification can be compiled into a fixed adapter, failing to capture sub-tasks that emerge during generation. This leads to incomplete coverage on complex tasks because the compiler does not iteratively refine based on partial outputs. Text-to-LoRA and GenerativeAdapter also generate adapters in one shot, lacking the ability to decompose tasks into iterative sub-steps. The root cause is that the generation process is treated as monolithic rather than compositional, and adapter weights are static after compilation.

## Key Insight

By interleaving adapter generation and generation state, the compiler can focus each adapter on a sub-task that is currently unsolved, and weighted averaging naturally combines multiple sub-task expertises without explicit task decomposition supervision.

## Method

### (A) What it is
**Iterative Refinement Adapter Composition (IRAC)** is a framework that extends the PAW compiler with an iterative loop: at each step, the compiler outputs a set of adapters and corresponding scalar weights, which are combined via weighted averaging and loaded into the interpreter to generate a continuation of the program. The process repeats until a stop criterion is met.

**Load-bearing assumption:** The compiler can learn to output adapters and importance weights conditioned on the current generation state that effectively target the remaining sub-tasks, and weighted averaging of these adapters yields a composition that improves generation. This assumption is verified via the correlation ablation in the experiment.

### (B) How it works

```python
# Pseudocode for IRAC inference
def irac_compile(specification, interpreter, max_iter=5):
    state = []  # generated tokens so far
    adapters = []  # list of (adapter_weights, weight_scalar)
    for step in range(max_iter):
        # Encode current state into a context vector using a 2-layer transformer encoder (hidden=256, 4 heads, 2 layers, GeLU activation)
        context = encode(state)  # output dim=256
        # Use compiler to produce K=3 adapters and their importance weights
        # Compiler is a 4-layer transformer decoder (hidden=512, 8 heads) taking (specification, context) and outputting
        # K LoRA adapter weight matrices (rank=16) and K scalar weights per adapter (raw logits)
        new_adapters, new_weights_logits = compiler(specification, context, K=3)
        # Append new adapters and their weights (as logits) to global lists
        adapters.extend(new_adapters)
        weights_logits.extend(new_weights_logits)
        # Softmax over all accumulated weight logits to get normalized weights
        weights = softmax(weights_logits)
        # Compute weighted average of all adapters (each is a LoRA-like matrix of rank 16, i.e., shape (d_in, rank) and (rank, d_out))
        combined_adapter = weighted_average(adapters, weights)
        # Load combined adapter into interpreter and generate next tokens (max_new_tokens=64)
        continuation = interpreter.generate(specification + state, adapter=combined_adapter, max_new_tokens=64)
        state.append(continuation)
        if complete(state):  # e.g., program ends with proper return or error
            break
    return state
```

**Hyperparameters**: max_iter=5, K=3, adapter rank=16, context encoder dim=256 (transformer with 2 layers, 4 heads), compiler decoder dim=512 (4 layers, 8 heads), weight normalization = softmax over all accumulated logits, max_new_tokens=64.

### (C) Why this design
We chose iterative refinement over single-pass generation because complex synthesis tasks often require decomposing into sub-tasks that are not known a priori; iterating on the current state allows the compiler to adaptively select which sub-task to address next, similar to how humans debug code. We chose weighted averaging over learned fusion mechanisms (e.g., HyperLoader's hypernetwork) because weighted averaging requires no additional training for the composition step, making it computationally cheap and compatible with many adapters; the cost is that linear combination may not capture non-linear interactions between adapters. We chose to accumulate all adapters from previous iterations (rather than replace) because retaining earlier sub-task expertise prevents catastrophic forgetting; the trade-off is linear growth in memory, which is acceptable since adapters are low-rank (typically <1% of base model parameters). We chose to condition the compiler output on a context vector encoding the current generation state (rather than just the specification) because this provides the compiler with feedback on which parts of the task are already solved, enabling it to generate adapters that target remaining deficiencies; the cost is increased computational overhead for context encoding, but we use a lightweight encoder (100M parameters) to keep overhead low.

### (D) Why it measures what we claim
The iterative refinement loop operationalizes the concept of **compositional coverage** by producing multiple adapters over steps, each intended to cover a sub-task. The **continuation length** (max_new_tokens=64) measures **partial solution completeness** because it assumes that if the adapter is correctly focused, the interpreter will generate at least a part of the missing functionality; this assumption fails when the adapter is not specific enough, in which case the continuation may be irrelevant or redundant. The **weighted average of adapters** measures **task decomposition accuracy** because the compiler's weight scalar for each adapter represents its relevance to the current state; this assumes that the compiler can correctly assign high weight to adapters that address unsolved parts, which fails when the context encoding is insufficiently detailed, in which case weights may be uniform or random. The **accumulation of adapters** across steps measures **progressive coverage** because each new adapter adds knowledge not present in previous ones; this assumes that the compiler does not repeatedly generate redundant adapters, which fails if the context does not change enough between steps, in which case memory grows without benefit.

## Contribution

(1) An iterative refinement framework for adapter composition in neural program synthesis, where a compiler generates sub-task adapters conditioned on generation state. (2) A weighted averaging mechanism that combines multiple adapters adaptively based on scalar weights output by the compiler, enabling smooth integration of sub-task expertise. (3) A design principle that accumulation of adapters across iterations preserves earlier sub-task knowledge without requiring explicit task decomposition supervision.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|-----------------------|
| Dataset | Fuzzy Program Synthesis Benchmark | From PAW; tests compositional program generation. |
| Primary metric | Task success rate | Directly measures complete program correctness. |
| Baseline 1 | Direct Prompting (Qwen3-32B) | Tests base model without adaptation. |
| Baseline 2 | PAW (single-pass) | Tests single-shot adapter composition. |
| Baseline 3 | Text-to-LoRA (pre-trained adapters) | Tests non-iterative adapter fusion. |
| Baseline 4 (added) | IRAC with random state encoding | Isolates novelty of state conditioning by using uniform random vector (256-dim) as context. |
| Ablation-of-ours | IRAC (single-pass, max_iter=1) | Removes iterative refinement loop. |
| Compositional coverage metric | Number of distinct functional operations per program | Measures diversity of sub-tasks covered; computed by parsing generated code into operation types (e.g., sort, filter, map) and counting unique ones. |

### Why this setup validates the claim

This setup isolates the central claim that iterative refinement improves compositional coverage. The fuzzy program synthesis benchmark requires generating programs with multiple sub-tasks, where single-pass methods often miss or conflate sub-tasks. Direct prompting tests whether base model alone can handle composition; PAW tests whether a single adapter composition suffices; Text-to-LoRA tests whether static fusion of pre-trained adapters can match iterative adaptation; IRAC with random state encoding tests whether the state-conditioning mechanism is responsible for gains. The ablation (IRAC without iteration) tests whether the iterative loop itself is responsible for gains. The primary metric, task success rate, directly captures end-to-end correctness, which is the ultimate goal. The compositional coverage metric provides a quantitative proxy for sub-task decomposition, allowing us to verify that higher success rate correlates with higher coverage. If IRAC outperforms all baselines and the coverage metric also increases, the claim is supported; if not, the claim is falsified.

### Expected outcome and causal chain

**vs. Direct Prompting (Qwen3-32B)** — On a multi-step fuzzy function (e.g., "compute fuzzy average and then apply threshold"), the base model often generates a program that does one step but ignores the second because it lacks a mechanism for task decomposition. Our method instead iterates: the compiler generates an adapter for the first sub-task, the interpreter produces a partial program, then the next adapter targets the remaining sub-task. Thus, we expect a noticeable gap on complex tasks (e.g., success rate >30% vs. <10%), but near parity on simple single-step tasks. We will specifically report results on a task "sort then filter" (from the benchmark) where iterative refinement is crucial; we expect IRAC to achieve >40% success while all baselines fall below 5%, demonstrating a failure mode of existing methods.

**vs. PAW (single-pass)** — PAW compiles all adapters in one forward pass, but on a task with an ordering dependency (e.g., sort then filter), PAW may assign equal weight to both sub-tasks, producing a muddled program. IRAC’s iteration allows sequential focus: the first iteration generates an adapter for sorting, the next for filtering, guided by the current state. We expect IRAC to achieve higher accuracy on tasks with visible sequential structure (e.g., 15% absolute improvement), and minimal difference on tasks where sub-tasks are independent.

**vs. Text-to-LoRA (pre-trained adapters)** — T2L fuses fixed adapters statically, so on a novel task combining known sub-tasks in an unforeseen way (e.g., combine two adapters for different base functions), the fixed combination may mis-weight them. IRAC dynamically generates new adapters per state, so it can adapt to the exact combination needed. We expect IRAC to outperform on novel compound tasks by a larger margin (e.g., 20%) but similar performance on tasks that exactly match pre-trained adapters.

**vs. IRAC with random state encoding** — If state conditioning is critical, the random variant will underperform (e.g., 10% lower success rate) because the compiler cannot focus on unsolved sub-tasks. This isolates the novelty of our conditioning mechanism.

### What would falsify this idea

If IRAC’s improvement over PAW (single-pass) is uniform across all task types rather than concentrated on those with sequential sub-task dependencies, then the iterative refinement loop is not the cause—instead, some other factor (e.g., more parameters) would be driving gains, falsifying the claim. Additionally, if the correlation between assigned adapter weights and actual contribution to sub-task completion (measured via leave-one-out ablation on a held-out set of 200 examples) is not statistically significant (Spearman ρ < 0.2, p > 0.05), then the load-bearing assumption that weights reflect sub-task relevance is rejected.

## References

1. Program-as-Weights: A Programming Paradigm for Fuzzy Functions
2. Text-to-LoRA: Instant Transformer Adaption
3. Generative Adapter: Contextualizing Language Models in Parameters with A Single Forward Pass
4. Compress then Serve: Serving Thousands of LoRA Adapters with Little Overhead
5. HyperLoader: Integrating Hypernetwork-Based LoRA and Adapter Layers into Multi-Task Transformers for Sequence Labelling
6. Meta-Learning Online Adaptation of Language Models
7. Compressed Context Memory For Online Language Model Interaction
8. ZipLoRA: Any Subject in Any Style by Effectively Merging LoRAs
