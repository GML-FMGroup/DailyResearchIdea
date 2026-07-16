# Recursive Self-Verification for Mathematical Proofs via LLM-Generated Decompositions

## Motivation

Existing verification pipelines, such as the one in AdvancedMathBench, require large-scale human-annotated proof trajectories to train a verifier. This manual decomposition step is expensive and domain-specific, limiting scalability to new proof settings. Without such annotations, the verifier cannot assess correctness at the sub-problem level, stalling progress in automated proof generation.

## Key Insight

The correctness of a proof decomposition is equivalent to the logical consistency of its sub-proofs with the original statement, which can be verified by an LLM through a recursive cycle of decomposition and implication checking, eliminating the need for external annotations.

## Method

### (A) What it is
**Recursive Self-Verification (RSV)** is a framework that takes a mathematical statement and a candidate proof as input, and outputs a binary correctness verdict. It operates by having an LLM propose a hierarchical decomposition of the proof into smaller sub-proofs, recursively verifying each sub-proof, and finally checking that the conjunction of sub-proofs logically implies the original statement. **This method relies on the assumption that an LLM can propose valid and complete proof decompositions (i.e., sets of sub-proofs that together constitute a correct proof of the original statement) without any training on proof decomposition data.** To mitigate the risk of invalid decompositions, we add a lightweight refinement loop as described below.

### (B) How it works
```python
def RSV(statement, proof, max_depth=5, max_attempts=3):
    if max_depth == 0:
        # Base case: direct LLM verification
        return LLM_verify_direct(statement, proof)
    
    # Step 1: LLM proposes decomposition
    attempts = 0
    while attempts < max_attempts:
        decomposition = LLM_propose_decomposition(statement, proof)
        if decomposition is None:
            break
        # Lightweight validation: check each sub-proof for self-consistency
        valid = True
        for sub_stmt, sub_proof in decomposition:
            # Ask LLM to rate correctness of sub-proof for sub_stmt (1-5 scale)
            rating = LLM_rate_correctness(sub_stmt, sub_proof)
            if rating < 4:  # threshold for accepting sub-proof
                valid = False
                break
        if valid:
            break
        attempts += 1
    if decomposition is None or not valid:
        return LLM_verify_direct(statement, proof)
    # decomposition = [(sub_stmt1, sub_proof1), ...]
    
    # Step 2: recursively verify each sub-proof
    for sub_stmt, sub_proof in decomposition:
        if not RSV(sub_stmt, sub_proof, max_depth-1):
            return False
    
    # Step 3: verify implication
    sub_stmts = [s for s, _ in decomposition]
    if not LLM_check_implication(sub_stmts, statement):
        return False
    return True
```
Hyperparameters: `max_depth=5` limits recursion depth; `max_attempts=3` limits refinement attempts; LLM calls use temperature=0 for deterministic output. **LLM call cost estimate:** Assuming a binary tree decomposition (branch factor ~2), worst-case calls ≤ 2 * (2^5) = 64; average calls with early stopping ~10-15 per proof.

### (C) Why this design
We chose a recursive decomposition over a monolithic one-pass verification because it mirrors the hierarchical structure of mathematical proofs, allowing the verifier to focus on smaller, more tractable sub-problems. This comes at the cost of increased LLM calls (each recursion step invokes up to two LLM queries per decomposition: one for proposal, one for implication; plus additional refinement attempts). We chose to let the LLM propose decompositions automatically rather than using hand-crafted rules or a trained decomposition model; this avoids the need for annotated decompositions but sacrifices decomposition quality control—the LLM may propose incorrect or non-minimal splits, which the lightweight validation step attempts to catch. The implication check (`LLM_check_implication`) uses the same LLM to verify logical entailment; this relies on the LLM's reasoning ability, which can be brittle. The recursion depth bound prevents infinite loops and limits computational cost.

### (D) Why it measures what we claim
The recursive verification loop operationalizes the concept of decomposition correctness. **Formal condition:** Decomposition is valid iff the conjunction of sub-proofs (as derivations) logically implies the original statement. The `LLM_propose_decomposition` call measures the model's ability to identify self-contained sub-proofs because it depends on the LLM's internal understanding of proof structure; this assumption fails when the model generates decompositions that are logically unrelated or omit necessary steps—in such cases, the lightweight validation may catch inconsistent sub-proofs, but if the decomposition is structurally flawed beyond repair, the verdict may be unreliable. The `LLM_check_implication` call measures the logical consistency between the sub-proofs and the original statement because it tests whether the conjunction of sub-proof claims entails the target; this assumes the LLM can reason about logical implication accurately—a failure mode occurs when the LLM accepts a false implication due to hallucination or missing counterexamples, leading to a false-positive verdict. The recursion provides redundancy: errors in one sub-proof can be caught by deeper recursion, and the implication check at each level ensures that the decomposition collectively supports the original claim. The lightweight validation adds a sanity check, but it itself relies on the LLM's self-assessment, which can be overconfident.

## Contribution

(1) A novel recursive self-verification framework for mathematical proofs that eliminates the need for human-annotated decompositions by leveraging LLM-generated decompositions and consistency checking. (2) The finding that LLM-based implication checking between sub-proofs and the original statement can serve as an effective verifier without supervised training on expert annotations.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|-----------------------|
| Dataset | AdvancedMathBench | Expert-annotated proof correctness labels. |
| Primary metric | Verification accuracy | Directly measures correct verdict rate. |
| Baseline 1 | Direct LLM verification | Single-pass, no decomposition. |
| Baseline 2 | Chain-of-thought verification | Step-by-step reasoning without hierarchy. |
| Baseline 3 | Fixed decomposition verifier | Splits by rule types, not adaptive. |
| Ablation-of-ours | RSV w/o implication check | Tests necessity of the final implication step. |
| Ablation-of-assumption | RSV with randomized sub-proof splits | Tests dependence on LLM decomposition quality. |

### Why this setup validates the claim
This experimental design forms a falsifiable test of RSV's central claim—that recursive decomposition improves verification accuracy over monolithic approaches. AdvancedMathBench provides ground-truth labels to compute accuracy. Comparing to direct and CoT baselines isolates the contribution of decomposition; the fixed-decomposition baseline tests whether adaptive splitting matters. The ablation without implication check pinpoints the role of final logical consistency verification. The added ablation with randomized splits (where sub-proof boundaries are chosen randomly, mimicking a broken decomposition) directly tests the load-bearing assumption that LLM-proposed decompositions are logically valid. If RSV with random splits still performs well, the decomposition step is unnecessary; if it degrades significantly, the assumption is crucial. Verification accuracy is the appropriate metric because it directly reflects the method's ability to output correct verdicts.

### Expected outcome and causal chain

**vs. Direct LLM verification** — On a long proof with a subtle error in an early sub-proof, the direct LLM overlooks the error due to cognitive overload; it outputs a false positive. Our method recursively verifies each sub-proof, catching the error at the responsible sub-proof, leading to a correct rejection. We expect a noticeable accuracy gap on proofs longer than 100 logical steps, with parity on short proofs.

**vs. Chain-of-thought verification** — On a proof with a circular reasoning chain, CoT may accumulate reasoning drift and miss the flaw; it produces a false positive. Our method decomposes into independent sub-proofs, breaking circularity, and the implication check verifies logical dependencies. We expect higher accuracy on proofs with multiple interdependent lemmas, with similar performance on linear proofs.

**vs. Fixed decomposition verifier** — On a proof where the logical structure does not match common inference rules (e.g., a non-standard induction), fixed decomposition splits incorrectly, causing sub-proofs to be non-separate or incomplete, leading to false negatives. Our method adaptively proposes decompositions via the LLM, capturing the true hierarchical structure. We expect a significant accuracy advantage on proofs with unusual or complex structure.

**vs. RSV with randomized sub-proof splits** — On any proof, random splits will break logical dependencies, causing sub-proofs to be non-sequitur or incomplete. The recursive verifier will likely fail to verify them correctly, leading to high false-negative rates. We expect RSV with true LLM decompositions to significantly outperform the random-split version, confirming the importance of decomposition quality.

### What would falsify this idea
If RSV shows no accuracy improvement over direct LLM verification on long or complex proofs, or if the gain is uniform across all proof lengths rather than concentrated on hierarchical or error-prone proofs, then the central claim of recursive decomposition being beneficial is false. Additionally, if RSV with randomized splits performs similarly to RSV with LLM decompositions, the load-bearing assumption that LLM decomposition quality matters is falsified.

## References

1. AdvancedMathBench: A Benchmark Suite for Advanced Mathematical Proof Generation and Verification
