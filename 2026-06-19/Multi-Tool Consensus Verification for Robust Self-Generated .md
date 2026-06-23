# Multi-Tool Consensus Verification for Robust Self-Generated Task Selection

## Motivation

Existing self-challenging agents, such as the Self-Challenging Language Model Agent, rely solely on the agent's own verification to filter self-generated tasks. This verification is fallible and domain-limited, causing distribution drift over iterative training because there is no external validator; task quality is judged by a single, potentially biased system. This structural gap means that tasks that pass internal verification may still be misaligned with human intent or tool-specific artifacts.

## Key Insight

Consistency across diverse independent tool implementations is a structural test of task validity because tool diversity introduces uncorrelated failure modes, making consensus a reliable indicator that the task is well-posed and not tool-specific.

## Method

(A) **What it is**: Multi-Tool Consensus Verification (MTCV) filters self-generated tasks by executing the task's solution through K diverse tool implementations and accepting the task only if all outputs are equivalent. Input: a task specification (including a solution program) and K tool implementations. Output: accept/reject decision.

(B) **How it works**:

```python
# Hyperparameters: K=3, tolerance=1e-6 for numeric outputs
def mtcv_filter(task_solution, tools, tolerance=1e-6):
    outputs = []
    for tool in tools:
        output = tool.execute(task_solution)  # each tool runs the same solution
        outputs.append(output)
    # Compare all outputs pairwise
    for i in range(K):
        for j in range(i+1, K):
            if not equivalent(outputs[i], outputs[j], tolerance):
                return False  # reject task
    return True  # accept task

def equivalent(a, b, tol):
    if type(a) != type(b):
        return False
    if isinstance(a, float):
        return abs(a - b) < tol
    if isinstance(a, str):
        return a == b
    # extend to other types as needed
    return a == b
```

(C) **Why this design**: We chose tool diversity over model ensembling (e.g., multiple LLM verifiers) because tools have independent, systematic errors (e.g., different rounding, different parsing) while LLMs share correlated biases from training data; this structural independence makes consensus a stronger signal. We required consensus among all K tools (rather than majority voting) because a single disagreement may indicate an ill-posed or tool-specific task, and accepting partial consensus would allow distribution drift. We opted for exact comparison (with tolerance for numerics) rather than semantic equivalence checks via another LLM, because semantic checks reintroduce the same fallible verification we aim to replace. The cost is that some valid tasks that are implementation-sensitive (e.g., relying on tool-specific behaviors) may be incorrectly rejected, but this aligns with our goal of filtering out tasks that are not robustly defined.

(D) **Why it measures what we claim**: The computational quantity `consensus (all outputs equivalent)` measures task validity because we assume that if a task is well-posed (its specification uniquely determines the correct output under any correct implementation), then all correct tool implementations must produce the same output; this assumption fails when tools themselves have correlated errors (e.g., both use the same flawed library), in which case consensus reflects shared bias rather than validity. The computational quantity `tool diversity` measures independence of error sources because we assume that different tool implementations have no common failure modes; this assumption fails when tools share underlying dependencies (e.g., both call the same external API), in which case consensus may be artificially high or low. By explicitly tracking these assumptions, MTCV provides a grounded external validity signal that is interpretable and auditable.

## Contribution

(1) Introduces a multi-tool consensus verification mechanism (MTCV) that provides an external validity signal for self-generated tasks, replacing reliance on the agent's own fallible verification. (2) Establishes a design principle that tool diversity can serve as an independent validation source, leveraging uncorrelated failure modes to filter tasks. (3) Demonstrates that consensus filtering prevents distribution drift in iterative self-training by ensuring tasks are robust and implementation-independent.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | M3ToolEval | Multi-turn tool-use benchmark |
| Primary metric | Task success rate | Measures end-to-end task completion |
| Baseline 1 | Base model (no self-training) | Lower bound; shows need for training |
| Baseline 2 | Self-training + no filter | Tests benefit without task filtering |
| Baseline 3 | Self-training + LLM verifier | Compares to LLM-based task filtering |
| Ablation-of-ours | MTCV with majority voting | Ablates consensus strictness |

### Why this setup validates the claim

This setup tests whether MTCV's consensus filtering improves training data quality over alternatives. The dataset M3ToolEval requires multi-step tool use, so task validity directly impacts success. Comparing to no filtering isolates the effect of filtering; comparing to an LLM verifier tests the claim that tool diversity provides stronger error independence than model correlation. The ablation with majority voting tests whether full consensus is necessary. Task success rate is the right metric because it captures whether the agent learns to complete tasks robustly, reflecting the quality of training tasks. If MTCV outperforms all baselines, it supports the claim that consensus among diverse tools filters ill-posed tasks effectively.

### Expected outcome and causal chain

**vs. Base model (no self-training)** — On a case where the task expects precise output (e.g., "compute the average of [3,5,7]" expecting 5), the base model may fail due to lack of tool-use training. Our method trains on filtered tasks, so the agent learns to use tools consistently. We expect MTCV to achieve a noticeably higher success rate on such tasks, with a gap of at least 10%.

**vs. Self-training + no filter** — On a case where the agent generates an ill-posed task (e.g., "find latest news" without date), unfiltered training includes such ambiguous tasks, causing distribution drift. Our method rejects them because different tools (e.g., date-sensitive APIs) would disagree. We expect MTCV to show a significant advantage on tasks requiring specific outputs (e.g., numeric calculations), with a gap of 5-10%, and parity on simple tasks.

**vs. Self-training + LLM verifier** — On a case where the LLM verifier accepts a biased task due to shared training data (e.g., both proposer and verifier are from Llama-3.1), it may pass ill-posed tasks. Our tool consensus uses independent tools (e.g., calculator vs. math library), so diverging errors are caught. We expect MTCV to outperform on tasks with exact string/numeric outputs, with a moderate gap of 3-5%.

### What would falsify this idea

If MTCV's task success rate is no better than the no-filter baseline, or if its gains are uniform across all task types rather than concentrated on tasks requiring precise output consistency (where tool errors diverge), then the central claim that diverse tool consensus provides a stronger validity signal is falsified.

## References

1. Self-Challenging Language Model Agents
2. Proposer-Agent-Evaluator(PAE): Autonomous Skill Discovery For Foundation Model Internet Agents
3. Building Math Agents with Multi-Turn Iterative Preference Learning
4. You Only Look at Screens: Multimodal Chain-of-Action Agents
5. Bootstrap Your Own Skills: Learning to Solve New Tasks with Large Language Model Guidance
6. ProgPrompt: Generating Situated Robot Task Plans using Large Language Models
7. OPT: Open Pre-trained Transformer Language Models
