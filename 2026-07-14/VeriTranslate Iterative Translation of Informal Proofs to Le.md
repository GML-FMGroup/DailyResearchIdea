# VeriTranslate: Iterative Translation of Informal Proofs to Lean via Kernel Error Feedback

## Motivation

AdvancedMathBench's verifier judges informal proofs but lacks machine-checkable guarantees because it does not interact with a formal kernel; this leaves a gap between natural-language reasoning and formal verification that cannot be bridged by learned judgment alone. Existing translation approaches either require large parallel corpora (scarce for advanced math) or produce one-shot formalizations that fail silently, requiring costly manual correction.

## Key Insight

The deterministic error taxonomy of the Lean kernel partitions the failure space into a finite set of corrective contexts, enabling targeted iterative refinement without requiring human annotation.

## Method

(A) **What it is**: VeriTranslate is a system that translates informal mathematical proofs into formal Lean proofs by iterative refinement using kernel error feedback. Input: natural language proof text (e.g., LaTeX-style). Output: a Lean proof that passes the `#check` command in Lean's kernel.

(B) **How it works**:
```python
# Pseudocode for translation with kernel feedback
import lean_kernel

def translate(informal_proof: str, model, max_iters=10) -> str:
    candidate = model.generate(informal_proof)  # step 1: initial translation
    for i in range(max_iters):
        result = lean_kernel.check(candidate)    # step 2: submit to kernel
        if result.success:
            return candidate
        # step 3: parse error category (e.g., "type_mismatch", "unknown_identifier")
        error_cat = parse_error(result.error_msg)
        # step 4: prepend error context to input
        context = f"{informal_proof} [Error: {error_cat}]"
        candidate = model.generate(context)      # step 5: regenerate
    return candidate  # may still fail
```
Training uses REINFORCE with reward = +1 if final proof passes kernel, -0.1 per iteration. Pretraining on synthetic parallel corpus (informal renderings of Mathlib proofs via an LLM). Hyperparameters: temperature=0.8 for generation, learning rate=1e-5, batch size=64.

(C) **Why this design**: We chose iterative refinement with error messages over direct beam search because the kernel's error feedback provides structured, deterministic signals that beam search cannot exploit; the cost is increased inference latency (multiple iterations). We used a synthetic pretraining corpus from Mathlib instead of relying on existing informal-formal pairs because such pairs are scarce, accepting that synthetic data may introduce noise. We used REINFORCE instead of supervised learning on successful trajectories because the search space is large and we want to maximize success probability rather than mimic fixed outputs, at the cost of high variance gradient estimates. We parse error categories rather than raw error strings to reduce sparsity and improve generalization; this loses fine-grained error details but standardizes feedback. This design contrasts with prior work like GPT-f which generates formal proofs directly without an informal-to-formal translation step, thereby lacking the ability to leverage user-provided natural-language reasoning.

(D) **Why it measures what we claim**: The success probability of translation (fraction of informal proofs that compile into Lean) measures the quality of formalization because the Lean kernel is a ground-truth correctness checker; the assumption is that a correct formal proof exactly captures the informal reasoning. This assumption fails when the informal proof is itself incorrect or ambiguous, in which case the metric reflects inability to formalize flawed reasoning rather than translation capability. The error category conditioning (e.g., "type mismatch") measures the specific gap in the translation because it directly corresponds to the kernel's static analysis; this assumption fails when the error is misleading (e.g., a type mismatch caused by a missing lemma), in which case the error category reflects a surface symptom rather than root cause. The iterative refinement count measures translation efficiency because it quantifies the number of correction steps needed; this assumption fails when early errors cascade, making later corrections harder, so the count may conflate translation quality with error propagation dynamics.

## Contribution

(1) A method for translating informal proofs to formal Lean proofs using iterative kernel feedback, with a REINFORCE training procedure that leverages error categories. (2) The finding that error message categories from the Lean kernel provide actionable signals that significantly improve translation success over one-shot generation. (3) A synthetic parallel corpus of informal-formal proof pairs derived from Mathlib, enabling supervised pretraining.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | AdvancedMathBench | tests translation on advanced math. |
| Primary metric | Success probability | measures kernel acceptance. |
| Baseline | GPT-f (direct generation) | lacks informal-to-formal. |
| Baseline | Beam search (no feedback) | no iterative error correction. |
| Ablation | Ours w/o error categorization | raw error instead of categories. |

### Why this setup validates the claim

This design isolates the contributions of each component. AdvancedMathBench provides a diverse set of informal proofs with known formal counterparts, ensuring the metric directly assesses the core claim: improving formalization success through iterative refinement with kernel feedback. The GPT-f baseline tests the value of the informal-to-formal translation step; if our method succeeds where GPT-f fails, it demonstrates that leveraging natural language reasoning is beneficial. The beam search baseline tests the importance of using error feedback for iterative correction; outperforming it shows that structured signals from the kernel are more effective than blind search. The ablation tests whether parsing error categories (rather than raw strings) improves generalization; parity would indicate that raw errors suffice. Together, these comparisons form a falsifiable test: if our method does not outperform the baselines on cases where informal reasoning is critical or error feedback is needed, the central claim is undermined.

### Expected outcome and causal chain

**vs. GPT-f (direct generation)** — On a proof involving a complex case split described informally (e.g., "consider the two possibilities"), GPT-f generates a formal proof without the informal input, likely missing the logical structure or producing a wrong type, causing kernel rejection. Our method translates the informal reasoning directly, uses kernel errors (e.g., "type_mismatch") to refine, and typically succeeds after a few iterations. We expect a noticeable gap on problems with substantial informal commentary, with our method achieving >60% success versus GPT-f <30%, especially on those with explicit reasoning steps.

**vs. Beam search (no feedback)** — On a translation with a subtle type error (e.g., mismatched operator in a sum), beam search generates many candidates but cannot focus on the error; its success is limited by the number of beams, and it often fails after exhausting beams. Our method receives the error category "type_mismatch" and re-generates conditioned on that context, usually fixing the error in 1-2 iterations. We expect our method to have higher success (>70%) and fewer iterations (mean <3) compared to beam search with comparable beam width (e.g., 10 beams, success <40%, many failures).

**vs. Ours w/o error categorization** — On an error message with a long, cluttered string (e.g., multiple hints), the ablation uses the raw string directly, which may confuse the model or cause overfitting to specific phrasing. Our full method parses it into standardized categories (e.g., "unknown_identifier"), leading to more robust refinement. We expect the full method to outperform the ablation on diverse error types, with a gap of at least 10 percentage points on rare or verbose errors.

### What would falsify this idea
If our method does not significantly outperform GPT-f on problems with rich informal reasoning, or if the ablation matches our full method on error-rich cases, the central claim that iterative refinement with structured kernel feedback improves translation success is false.

## References

1. AdvancedMathBench: A Benchmark Suite for Advanced Mathematical Proof Generation and Verification
