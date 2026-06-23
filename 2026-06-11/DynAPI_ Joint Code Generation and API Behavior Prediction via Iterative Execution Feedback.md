# DynAPI: Joint Code Generation and API Behavior Prediction via Iterative Execution Feedback

## Motivation

Existing code generation benchmarks and models assume a fixed set of known APIs and rely on static test cases, limiting generalization to open-world environments where APIs are unknown and dynamic. For instance, the study on small language models (SLMs) by [Assessing Small Language Models for Code Generation] evaluates only on static benchmarks like HumanEval, leaving a structural gap: models cannot handle novel APIs that require dynamic discovery and adaptation. This limitation stems from the decoupling of code generation from API behavior modeling, causing hallucinated or incorrect API usage when encountering unseen interfaces.

## Key Insight

The joint consistency between generated code and a probabilistic API behavior model, enforced through iterative execution feedback, enables generalization to unknown APIs without a fixed library, because the feedback loop forces both to be mutually predictive of observed execution outcomes.

## Method

### (A) What it is
DynAPI is a framework that jointly generates code calling unknown APIs and a probabilistic model of those APIs' behavior, then refines both via iterative sandbox execution feedback. Input: a natural language description and a set of API signatures (but no implementation). Output: executable code and a learned API model that can predict outputs for new inputs.

### (B) How it works
```pseudocode
1. Initialize: prompt LLM to generate initial code c_0 and initial API models M_0 (one per unknown API, as a lightweight neural network or Gaussian process).
2. For t = 1 to T (e.g., T=3):
   a. Execute c_{t-1} in sandbox, collecting execution traces: for each API call, record input x, output y, and any runtime errors.
   b. Update API models: train M_t on all traces so far (e.g., minimize MSE for continuous outputs, cross-entropy for discrete).
   c. Compute consistency loss L_cons = sum over API calls of negative log-likelihood of observed y under M_t(x) + L_exec (e.g., pass@1 success rate).
   d. Prompt LLM with previous code and trace summary to generate refined code c_t that better exploits API models (e.g., adding error handling, using predicted outputs).
   e. Optionally, update API model architecture if uncertainty high.
3. Return final code c_T and API models M_T.
```
Hyperparameters: T=3, API model = 2-layer MLP with 64 hidden units for continuous, softmax for discrete; L_exec weight = 0.5.

### (C) Why this design
We chose to jointly generate code and API models rather than first retrieving API documentation (anti-pattern 2) because the APIs are truly unknown—no library exists. This rules out standard retrieval-augmented generation. We opted for a lightweight MLP for API modeling over a full LLM (trade-off: less expressive but faster to update iteratively) accepting that complex APIs may require more capacity. The consistency loss uses negative log-likelihood rather than simple error count because it provides gradient signal for API model training; the cost is that it assumes output distributions are modeled correctly. Finally, we use iterative refinement with T=3 rather than one-shot because the first speculation may be poor; the increase in inference cost is acceptable given the open-world setting. This design ensures that the method does not collapse onto a simple cooperation of separate modules (anti-pattern 4): the code and API model are jointly updated through a single loss, not a controller.

### (D) Why it measures what we claim
The negative log-likelihood (NLL) of observed execution outputs under the API model measures the consistency between code and API predictions because it quantifies how well the API model explains actual behavior; this assumes that execution traces are sampled from the true API distribution. This assumption fails when the sandbox is not representative (e.g., non-deterministic APIs), in which case NLL reflects only partial consistency. The execution success rate L_exec measures code correctness under the predicted API model because it directly tests executable functionality; this assumes the sandbox can simulate API behavior, which fails when APIs require network or hardware access, in which case L_exec reflects synthetic correctness. Together, the joint objective ensures that neither the code nor the API model can drift independently, as the consistency gradient couples them.

## Contribution

(1) DynAPI, a framework that jointly generates code and a probabilistic API behavior model, enabling generalization to unknown APIs without a fixed library. (2) A consistency objective that uses iterative execution feedback to align code and API predictions, demonstrated to reduce hallucinated API usage. (3) A new benchmark for open-world code generation comprising tasks with simulated unknown APIs, enabling evaluation of dynamic API discovery.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | DynAPI-Bench (synthetic unknown APIs) | Controlled, unknown API behavior |
| Primary metric | Joint success rate (pass@1 + API consistency) | Measures code & model alignment |
| Baseline | Standard LLM (no API model) | Tests necessity of API modeling |
| Baseline | LLM + static API model (no refinement) | Tests benefit of iterative joint update |
| Baseline | Iterative refinement without API model | Tests necessity of learned API model |
| Ablation | DynAPI w/o consistency loss (only exec feedback) | Isolates effect of joint objective |

### Why this setup validates the claim

This experimental design tests the central claim that jointly generating code and a probabilistic API model, then refining both via execution feedback, yields better code for unknown APIs than any approach that treats code and API modeling separately. The synthetic dataset DynAPI-Bench provides ground-truth API implementations, enabling precise measurement of code correctness and API model fidelity. The primary metric combines pass@1 (code functionality) with API model consistency (negative log-likelihood), directly reflecting the joint objective. Each baseline isolates a key design choice: standard LLM tests whether any API modeling is needed; LLM + static API model tests whether iterative refinement is beneficial; iterative refinement without API model tests whether modeling APIs explicitly matters. The ablation removes the consistency loss to see if the joint objective couples code and model beyond execution feedback alone. Together, these comparisons ensure that any performance gain can be attributed to the specific mechanism proposed, not to incidental factors. If DynAPI outperforms baselines on tasks where API behavior is complex and non-obvious, but not on trivial tasks, the claim is supported; if gains are uniform, it suggests a simpler explanation.

### Expected outcome and causal chain

**vs. Standard LLM (no API model)** — On a case where the unknown API returns values that depend nonlinearly on inputs (e.g., a hash function), the baseline guesses random outputs, leading to incorrect control flow or error handling because it assumes default patterns. Our method instead learns an API model from early execution traces, predicting outputs accurately, so the refined code uses those predictions to branch correctly. We expect a noticeable gap on tasks with complex API behavior (e.g., 20-30% higher joint success) but near parity on tasks with trivial APIs.

**vs. LLM + static API model (no refinement)** — On a case where the initial API model is misspecified (e.g., Gaussian process chosen but API is discrete), the static model gives poor predictions, causing code to make bad decisions. Our method iteratively updates the model architecture and retrains on new traces, correcting the misspecification. We expect our approach to outperform on tasks where the API's true distribution is hard to guess a priori, with a 15-25% improvement in joint success.

**vs. Iterative refinement without API model** — On a case where execution errors arise from logical dependencies between API calls (e.g., second call's input is output of first), the baseline only sees execution success/failure, not why. It may randomly modify code without fixing root cause. Our method uses the API model to predict outputs and compute consistency loss, pinpointing which API call caused the failure, leading to targeted fixes. We expect our method to converge faster (fewer iterations) and achieve higher final success rates (10-20% better).

### What would falsify this idea
If DynAPI's joint success rate is not significantly higher than the 'iterative refinement without API model' baseline on tasks where API output prediction is critical (e.g., non-deterministic or highly nonlinear APIs), then the claimed benefit of explicit API modeling is unsupported.

## References

1. Assessing Small Language Models for Code Generation: An Empirical Study with Benchmarks
2. Hallucination by Code Generation LLMs: Taxonomy, Benchmarks, Mitigation, and Challenges
3. LLM Hallucinations in Practical Code Generation: Phenomena, Mechanism, and Mitigation
4. Collu-Bench: A Benchmark for Predicting Language Model Hallucinations in Code
5. FELM: Benchmarking Factuality Evaluation of Large Language Models
6. A Survey on Hallucination in Large Language Models: Principles, Taxonomy, Challenges, and Open Questions
7. A Survey of Large Language Models for Code: Evolution, Benchmarking, and Future Trends
8. Understanding Factual Errors in Summarization: Errors, Summarizers, Datasets, Error Detectors
