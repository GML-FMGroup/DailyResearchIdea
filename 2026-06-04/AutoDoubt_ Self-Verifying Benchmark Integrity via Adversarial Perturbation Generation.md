# AutoDoubt: Self-Verifying Benchmark Integrity via Adversarial Perturbation Generation

## Motivation

Existing benchmark verification methods, such as the Agentic Benchmark Checklist (ABC) from 'Establishing Best Practices for Building Rigorous Agentic Benchmarks', provide static guidelines but cannot detect missing criteria that emerge from novel agent behaviors. Similarly, automated benchmark generation approaches (e.g., BeTaL in 'Automating Benchmark Design') optimize for target properties like difficulty but assume the verifier's criteria are complete. This structural limitation leads to performance overestimation and under-specification, as the verifier never checks its own gaps.

## Key Insight

A verifier can expose its own missing criteria by using its generative capabilities to produce adversarial benchmark variants that it would incorrectly accept, exploiting the fact that the verifier's language model generalizes to plausible but invalid configurations.

## Method

### (A) What it is
**AutoDoubt** is a meta-verification framework that iteratively subjects a given verifier (a benchmark evaluation function) to an adversarial self-audit loop. The input is an existing benchmark (set of tasks and success criteria) and a verifier; the output is an updated benchmark with patched criteria and a report of newly discovered failure modes.

### (B) How it works
```pseudocode
1. Given: Verifier V, current benchmark B = {task_i, criteria_i}
2. For each task group (e.g., similar domains):
   a. CANDIDATES = []
   b. Generate N adversarial perturbations:
      - For each task, sample a perturbation operation (swap, remove, inject) from uniform distribution p_op over {swap_action, remove_constraint, inject_noise}
      - Apply to task description or success criteria to create candidate variant T'
      - Use V to evaluate T': if V(T') == PASS, add to CANDIDATES
   c. Diversity filter:
      - Embed all candidates using Sentence-BERT (all-MiniLM-L6-v2)
      - Cluster by cosine similarity with threshold τ=0.7, select K=5 representatives from distinct clusters
   d. Human-in-the-loop (minimal):
      - For each selected candidate, query a human to label if it is truly invalid (1) or valid (0). If invalid, the human also formulates the missing criterion that the variant exploits.
   e. Update criteria:
      - For each truly invalid candidate, append the human-formulated negative constraint to criteria_i (e.g., "must not have property P")
3. Repeat until no new invalid candidates found in two consecutive iterations
4. Output: patched benchmark B' and log of discovered criteria gaps
```
Hyperparameters: N=50 perturbations per task, p_op uniform, τ=0.7, K=5, clustering with Sentence-BERT.

**Load-bearing assumption (explicit):** We assume that the human labeler can reliably identify the missing criterion from the adversarial variant. This replaces the earlier assumption that attention weights indicate causal tokens, which we avoid due to evidence of unreliability (Jain & Wallace, 2019).

### (C) Why this design
We chose perturbation-based generation over adversarial optimization (e.g., gradient-based) because the verifier is a black-box LLM, and white-box access is often unavailable; accepting the cost that perturbations may lack optimality. We used a diversity filter (clustering on embeddings) over random selection to avoid redundancy in discovered gaps, at the cost of increased computational overhead. We opted for a lightweight human loop (labeling only K=5 per iteration) instead of fully automated patching, because automated rule extraction from attention maps is noisy; accepting that the human effort is small but not zero. We replaced attention-based rule extraction with direct human criterion formulation, avoiding the assumption that attention indicates causality—a known limitation for LLMs with attention dilution.

### (D) Why it measures what we claim
*Perturbation causing false acceptance* measures criteria gap because the only difference from valid tasks is the induced defect; assumption: defect is not semantically equivalent to original; failure mode: perturbation may introduce noise instead of gap. *False positive rate on candidates* measures **verifier oversight** because it quantifies cases where the verifier says PASS when the human says FAIL; this assumption fails when the human label is inconsistent, in which case the metric reflects annotator disagreement. *Diversity filter* measures **coverage of gap types** because clustering by semantic similarity ensures the discovered gaps span distinct failure modes; this assumption fails when the embedding space does not align with task semantics, in which case the filter may miss or collapse distinct gaps. *Human-formulated criteria* measure **causal criteria** because the human directly reasons about the missing constraint; this assumption fails when the human cannot articulate the gap (e.g., due to cognitive bias), in which case the patched criteria may be incomplete or overconstrained.

## Contribution

(1) A self-audit framework (AutoDoubt) that enables a benchmark verifier to autonomously detect and patch missing success criteria by generating adversarial variants. (2) A design principle that verifier hallucination (false acceptance of adversarial variants) can be systematically exploited to improve verification completeness without requiring a second verifier. (3) An empirical validation protocol to measure the reduction in false positive rate of the verifier after applying AutoDoubt.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|------------------------|
| Dataset | CVE-Bench | Contains complex evaluation criteria. |
| Primary metric | False acceptance rate reduction | Directly measures gap detection success. |
| Baseline 1 | Original benchmark (no patching) | Shows current verifier's oversight. |
| Baseline 2 | AutoDoubt w/o human loop | Tests necessity of human feedback. |
| Ablation | AutoDoubt w/o diversity filter | Tests diversity filter's effect on coverage. |

### Why this setup validates the claim

This experimental design forms a falsifiable test of AutoDoubt's central claim that it can discover and patch criteria gaps in agentic benchmarks. By using CVE-Bench, which is known to have intricate evaluation criteria and potential gaps, we directly test the method in a realistic scenario. The primary metric, false acceptance rate reduction, quantifies how many overlooked invalid tasks are correctly rejected after patching. Comparing against the original benchmark (Baseline 1) establishes the baseline oversight. Baseline 2 (without human loop) tests whether the human-in-the-loop component is essential for precise patching versus relying solely on attention-based rule extraction. The ablation (without diversity filter) isolates the effect of diverse coverage on gap discovery. This combination of comparisons and the focused metric ensures that observed improvements can be attributed to AutoDoubt's specific design choices, and any failure to produce expected patterns would falsify the claims.

### Expected outcome and causal chain

**vs. Original benchmark (no patching)** — On a task where the verifier falsely accepts a perturbed variant due to a missing criterion (e.g., a security task that ignores a constraint like "must not run arbitrary code"), the original benchmark continues to accept it, yielding a high false acceptance rate. Our method identifies the criterion gap via adversarial perturbation and human verification, then patches the criteria to reject such variants. Therefore, we expect a noticeable reduction in false acceptance rate on the patched benchmark compared to original, especially on task groups where such gaps exist.

**vs. AutoDoubt w/o human loop** — On a case where automated attention-based rule extraction produces a spurious rule (e.g., because the verifier's attention is diffuse across many tokens), the no-human-loop variant may block valid tasks or fail to block invalid ones. Our method uses human labeling to filter out false positives before extracting rules, yielding more precise constraints. Thus, we expect lower false acceptance rate and fewer false rejections (i.e., fewer valid tasks incorrectly blocked) compared to the no-human-loop variant, particularly on tasks where attention maps are noisy.

### What would falsify this idea

If the false acceptance rate reduction is uniform across all task subsets (rather than concentrated where criteria gaps are predicted) or if AutoDoubt w/o human loop performs as well as full AutoDoubt (indicating the human loop is unnecessary), then the central claim about AutoDoubt's ability to discover and patch gaps with human assistance would be falsified.

## References

1. Establishing Best Practices for Building Rigorous Agentic Benchmarks
2. Automating Benchmark Design
3. LLM-POET: Evolving Complex Environments using Large Language Models
4. Automatic Generation of Benchmarks and Reliable LLM Judgment for Code Tasks
5. APIGen: Automated Pipeline for Generating Verifiable and Diverse Function-Calling Datasets
6. CrossCodeEval: A Diverse and Multilingual Benchmark for Cross-File Code Completion
7. SWE-bench: Can Language Models Resolve Real-World GitHub Issues?
8. Evolution Gym: A Large-Scale Benchmark for Evolving Soft Robots
