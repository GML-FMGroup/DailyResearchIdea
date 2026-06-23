# Execution-Grounded Iterative Refinement for Small Language Model Code Generation

## Motivation

Existing code generation methods for games often rely on large language models that are expensive and slow, while small language models (SLMs) struggle with correctness. Iterative refinement has been applied to bug fixing (e.g., LangGraph-based agents from [Empirical Research on Utilizing LLM-based Agents for Automated Bug Fixing via LangGraph]), but these methods assume a multi-agent architecture and do not adapt to generating code from scratch. Furthermore, retrieval-augmented generation (RAG) typically retrieves static examples without leveraging execution feedback, failing to guide targeted corrections. The root cause is the lack of a unified state that couples execution outcomes with retrieval to drive effective refinement in a single-loop process.

## Key Insight

Execution traces provide a structured, task-specific signal that aligns retrieval and generation, enabling iterative refinement with a small model to converge faster than static RAG or simple reprompting.

## Method

### (A) What it is
EGIR (Execution-Grounded Iterative Refinement) is a single-loop framework that takes a natural language description and a test suite as input, and outputs a corrected program. It uses an SLM (2.7B-parameter Transformer, based on PolyCoder) with a learned edit head (2-layer MLP, hidden=512, GeLU activation) fine-tuned on execution feedback, and a dynamic memory of (code embedding, execution trace features, test result string) vectors. The code embedding is a mean pooling of the SLM's last hidden layer (dimension 768); execution trace features are a 128-dimensional vector from a small MLP over trace attributes (e.g., number of passed/failed tests, error types, line numbers).

### (B) How it works (pseudocode)
```python
def EGIR(description, tests, max_iters=5):
    state = {"code": None, "test_results": None, "memory": []}
    for i in range(max_iters):
        if i == 0:
            state["code"] = slm.generate(description)
        else:
            # Encode current code + failure summary into query
            code_emb = mean_pool(slm.encode(state["code"]))
            trace_features = extract_trace_features(state["test_results"])  # 128-dim
            query = concatenate([code_emb, trace_features])  # 896-dim
            retrieved = memory.search(query, k=3)  # cosine similarity, returns (code_emb, trace, code) tuples
            # Refine code using edit head: input = (current code, retrieved examples, test failures)
            new_code = slm.edit(state["code"], retrieved, state["test_results"])
            # Verification step: roll back if new_code passes fewer tests
            new_results = execute(new_code, tests)
            if new_results.passed_count < state["test_results"].passed_count:
                state["code"] = state["code"]  # keep previous
            else:
                state["code"] = new_code
                state["test_results"] = new_results
        
        # Execute against augmented test suite (EvalPlus-style)
        state["test_results"] = execute(state["code"], tests)
        
        # Store successful execution trace in memory (only if all pass)
        if all(state["test_results"].passed):
            trace_features = extract_trace_features(state["test_results"])
            memory.store(code_emb, trace_features, state["code"])
        
        if all(state["test_results"].passed):
            break
    return state["code"]
```
Hyperparameters: `max_iters=5`, `k=3`, `embed_dim=768`, `trace_feature_dim=128`, learning rate 1e-5, batch size 16, trained on GameCodeTrain (4000 examples) for 20 epochs on 4 A100 GPUs (~48 hours). The edit head is trained with cross-entropy loss on edit tokens, supervised by pairs of (buggy code, corrected code) derived from execution feedback.

### (C) Why this design
We chose a **unified state object** over separate agent modules (as in LangGraph) because game code generation requires tight coupling between generation and execution feedback; decoupling would introduce latency and potential state inconsistency. We chose to **store execution traces** (not just code) in memory because traces provide fine-grained information about test failures, enabling more relevant retrieval (trade-off: increased storage and encoding cost). We used an **SLM with a learned edit head** (trained on execution feedback) instead of simple reprompting because reprompting without structural guidance often leads to superficial changes (trade-off: requires additional training data and may overfit to certain error patterns). We **augmented the test suite** using EvalPlus-style generation to improve fault detection, accepting the cost of higher evaluation time. The verification step (rollback on decreased pass count) ensures monotonic improvement, mitigating the risk of the learned edit head making harmful changes. **Load-bearing assumption:** The learned edit head, fine-tuned on execution feedback, can reliably produce minimal and correct code edits from the combination of current code, retrieved examples, and test failures; this assumption is verified by the rollback step.

### (D) Why it measures what we claim
The computational quantity `execute(state["code"], tests).passed_count` measures **functional correctness** because it directly counts passing tests. The quantity `memory.search(query, k=3)` measures **execution-grounded similarity**: we assume the concatenated query embedding captures failure-relevant features; this assumption fails when irrelevant code structure dominates the code embedding, leading to retrieval of unhelpful examples (then similarity reflects surface code similarity rather than error relevance). The quantity `slm.edit(...)` measures **targeted refinement capability**: we assume the edit head has learned to apply minimal changes to fix specific failures; this assumption fails when the edit head learns spurious correlations between retrieved examples and output edits (e.g., always adding a try-catch), leading to unnecessary changes. The number of iterations to convergence measures **efficiency of refinement**: we assume monotonic improvement (enforced by verification); this assumption fails if the model oscillates due to over-optimization or if the verification step is too conservative.

## Contribution

(1) EGIR, an execution-grounded iterative refinement framework that unifies generation, execution, and retrieval in a single state loop for SLM code generation. (2) The finding that encoding test failures into the retrieval query significantly improves the relevance of retrieved exemplars compared to using code-only embeddings. (3) A methodology for augmenting game-specific test suites using EvalPlus-like mutation to increase evaluation rigor for SLMs.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | GameCodeTests (500 problems), HumanEval (164), MBPP (500) | Game-specific + general code generation tests |
| Primary metric | Functional correctness (Pass@1, Pass@5) | Measures final program correctness; Pass@5 allows multiple attempts |
| Baseline 1 | Direct SLM generation | Single pass, no iteration |
| Baseline 2 | LangGraph-based agent (GPT-3.5) | Separate modules for generation/fix |
| Baseline 3 | SWE-Agent+GPT-4 | Strong reference with large model |
| Baseline 4 | CodeRL (with same SLM) | Existing iterative refinement method |
| Baseline 5 | LEAP (with same SLM) | Existing iterative refinement method |
| Ablation 1 | EGIR w/o memory retrieval | Tests contribution of dynamic memory |
| Ablation 2 | EGIR w/ code-only retrieval | Replaces trace features with code-only query |
| Ablation 3 | EGIR w/ error-only retrieval | Replaces trace features with error-only query |

### Why this setup validates the claim
This setup directly tests EGIR’s central claim: that execution-grounded iterative refinement with learned editing and trace memory improves code generation for games and general programming. GameCodeTests provides diverse game tasks with failing tests, creating a need for targeted fixes. HumanEval and MBPP test generalizability. Primary metrics Pass@1 and Pass@5 capture correctness with and without multiple attempts. Baselines like Direct SLM and LangGraph isolate the benefit of unified iteration vs. modular agents. CodeRL and LEAP compare against existing iterative methods. SWE-Agent+GPT-4 tests whether SLM+EGIR can compete with large models. Ablations with code-only or error-only retrieval isolate the role of execution traces. Together, they form a minimal falsifiable test: if EGIR outperforms all baselines and ablations, the claim is supported; if not, the design fails.

### Expected outcome and causal chain

**vs. Direct SLM generation** — On a case where the first generated code has a subtle type error, Direct SLM outputs a single incorrect program. Our method executes, detects failure, retrieves similar past fixes, and applies a minimal edit. Thus EGIR achieves higher Pass@1 on problems requiring iteration, while performing similarly on simple tasks.

**vs. LangGraph-based agent** — On a state-dependent bug, LangGraph may suffer state inconsistency between modules. EGIR’s unified state keeps code and failure trace coupled, and the edit head applies precise changes. We expect EGIR to converge in fewer iterations and higher final correctness, especially on bugs requiring tight feedback.

**vs. SWE-Agent+GPT-4** — On domain-specific game logic, SWE-Agent may generate plausible but incorrect code. EGIR’s SLM, fine-tuned on execution feedback, learns game-specific patterns, and memory stores successful traces. We expect EGIR to match or exceed on game tasks, but lag on general tasks where GPT-4 excels.

**vs. CodeRL** — CodeRL uses actor-critic with execution reward but no retrieval. On recurring bug patterns, EGIR retrieves past fixes, converging faster. We expect EGIR to achieve higher Pass@1 especially on problems with repeated error types.

**vs. LEAP** — LEAP bootstraps with program synthesis but lacks execution-grounded retrieval. On tasks requiring iterative debugging, EGIR’s trace memory provides more targeted signals. Expect EGIR to outperform on complex multi-step tasks.

**vs. EGIR w/o memory retrieval** — On a case with recurring off-by-one errors, ablation regenerates edit blindly. Full EGIR retrieves the successful trace and avoids the mistake. Expect clear gain on problems where bug patterns recur across tasks.

**vs. EGIR w/ code-only retrieval** — On a case with mismatched variable names but similar error type, code-only retrieval may miss the error context. Trace-enhanced retrieval includes failure location, leading to better fixes. Expect EGIR full to outperform on bugs requiring error-context understanding.

**vs. EGIR w/ error-only retrieval** — On a case where error message is ambiguous (e.g., 'TypeError'), error-only retrieval may retrieve irrelevant examples. Full EGIR uses both code structure and trace, providing more precise matches. Expect full EGIR to achieve higher correctness.

### What would falsify this idea
If EGIR’s advantage over baselines is uniform across all problem types rather than concentrated on tasks requiring iterative refinement or trace memory, it would indicate the method’s success stems from unrelated factors (e.g., better prompt engineering) and the central claim is wrong. If the ablation with code-only retrieval matches full EGIR, the importance of execution traces is falsified.

## References

1. Empirical Research on Utilizing LLM-based Agents for Automated Bug Fixing via LangGraph
2. Assessing Small Language Models for Code Generation: An Empirical Study with Benchmarks
3. SWE-Bench+: Enhanced Coding Benchmark for LLMs
4. Is Your Code Generated by ChatGPT Really Correct? Rigorous Evaluation of Large Language Models for Code Generation
5. CoditT5: Pretraining for Source Code and Natural Language Editing
6. A systematic evaluation of large language models of code
