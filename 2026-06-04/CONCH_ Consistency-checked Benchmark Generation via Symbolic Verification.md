# CONCH: Consistency-checked Benchmark Generation via Symbolic Verification

## Motivation

Existing LLM-based benchmark generation frameworks, such as BeTaL, assume that the LLM can reliably reason about design spaces to meet target properties like difficulty. However, this assumption fails in unfamiliar domains where the LLM lacks domain knowledge, leading to benchmarks that violate intended constraints or are structurally flawed. The root cause is the absence of a verification mechanism that checks whether the generated low-level instances logically follow from the high-level specification—a verification that does not require domain-specific training because it relies on structural consistency across abstraction levels.

## Key Insight

Logical consistency across abstraction levels is a necessary condition for correct benchmark design, and violations of this consistency can be detected using domain-independent symbolic reasoning, thereby catching LLM reasoning failures without needing domain-specific knowledge.

## Method

(A) **What it is**: CONCH (Consistency-checked Benchmark Generation) is a framework that takes a high-level benchmark specification and generates a set of concrete benchmark instances via an LLM, then iteratively verifies logical consistency across abstraction levels using a symbolic verifier (e.g., SAT solver) and a learned consistency classifier, and issues repair directives to the LLM until all constraints are satisfied. Its inputs are a high-level specification H (task description, target difficulty, constraints) and an abstraction library A (mapping H to low-level constraints); its output is a verified low-level benchmark instance L.

(B) **How it works**:
```pseudocode
Procedure CONCH(H, A, LLM):
    // Phase 1: Generate low-level candidate
    L = LLM(H, A)  // e.g., parameterized template from BeTaL filled by LLM
    // Phase 2: Extract constraints
    Φ_H = extract_high_level_constraints(H, A)  // e.g., "difficulty ∈ [0.5,0.8]"
    Φ_L = extract_low_level_propositions(L, A)  // e.g., "agent reward after step 5 = 0.6"
    // Phase 3: Consistency check
    iter = 0
    while iter < max_iterations (3):
        symbolic_result = SymbolicVerifier.check(Φ_H ∧ Φ_L)  // SAT solver, timeout=5s
        if symbolic_result == SATISFIABLE:
            learned_result = LearnedVerifier.predict(Φ_H, Φ_L)  // 2-layer MLP, hidden=256, ReLU, trained on 1000 verified examples; threshold=0.5
            if learned_result == CONSISTENT:
                return L  // verified
            else:
                conflict = LearnedVerifier.get_explanation()  // e.g., features contributing to inconsistency
                repair = generate_repair_directive(conflict)  // e.g., "Adjust parameter P to increase consistency score"
        else:
            conflict = SymbolicVerifier.get_conflict()  // e.g., "difficulty constraint violated"
            repair = generate_repair_directive(conflict)  // e.g., "Adjust parameter P to increase difficulty"
        L = LLM(H, A, L, repair)  // LLM revises L with repair context
        Φ_L = extract_low_level_propositions(L, A)
        iter++
    // Phase 4: Fallback
    return L  // best effort after max iterations
```
Hyperparameters: max_iterations = 3, verifier_timeout = 5s, learned_verifier_threshold = 0.5, learned_verifier_architecture: 2-layer MLP with hidden units 256, ReLU activation, trained on 1000 verified examples from diverse domains.

(C) **Why this design**: We chose a symbolic verifier (SAT solver) over a learned consistency classifier because verifiability guarantees soundness—any inconsistency detected is real—whereas a learned model would require domain-specific training and risk false positives/negatives. The trade-off is that we must define a formal abstraction library A for each new domain, which requires upfront effort but ensures the verification is structurally domain-independent. We chose an iterative generate-and-repair loop rather than a single-pass generate-and-verify because the LLM’s generative strength is best leveraged when it receives targeted feedback, analogous to how human designers refine specifications. The cost is increased runtime from multiple LLM calls, acceptable for offline benchmark generation. The 3-iteration limit balances quality and runtime, acknowledging that some complex constraints may remain unresolved. By using propositional logic via SAT solvers, we keep verification efficient; for richer constraints, SMT solvers could be used at the cost of scalability. To address the limitation that symbolic verification relies on complete abstraction, we add a learned consistency classifier as an orthogonal check that captures patterns not formalized in A, trained on previously verified examples across domains. This hybrid approach reduces reliance on the completeness of A.

(D) **Why it measures what we claim**: The satisfiability of Φ_H ∧ Φ_L under the abstraction mapping A is equivalent to logical consistency between high-level specification H and low-level implementation L only if the abstraction library A is complete for the intended semantics. This assumption fails when A omits critical properties (e.g., difficulty depends on unmodeled agent behavior), in which case satisfiability may hold despite actual inconsistency (false positive). Conversely, if A includes overly strict constraints, verifier may reject correct benchmarks (false negative). To mitigate this risk, we add a learned consistency classifier as an orthogonal check; however, the learned classifier itself may have generalization errors. Thus, CONCH reduces but does not eliminate the risk of undetected errors. The computational quantity `SymbolicVerifier.check(Φ_H ∧ Φ_L)` measures logical consistency under the completeness assumption; the addition of `LearnedVerifier.predict(Φ_H, Φ_L)` provides a second measure that captures consistency patterns not encoded in A.

## Contribution

(1) A novel framework (CONCH) that integrates LLM-based benchmark generation with symbolic verification to ensure cross-abstraction consistency, enabling error detection without domain-specific fine-tuning. (2) A design principle showing that structural consistency verification (e.g., SAT-based constraint checking) can catch the majority of reasoning failures in LLM-based design tasks, as demonstrated by the iterative repair loop. (3) An analysis of the trade-offs between verification coverage and generality, formalized through the abstraction library design.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | 50 high-level specs from BeTaL domains | Diverse, representative of real benchmark design |
| Primary metric | Consistency verification pass rate | Directly measures logical correctness claim |
| Secondary metric | Abstraction coverage (fraction of high-level constraints expressible in A) | Assesses completeness assumption |
| Baseline 1 | Random parameter generation | No consistency enforcement, shows necessity of constraints |
| Baseline 2 | Single-pass LLM generation (no repair) | Tests benefit of iterative repair over one-shot |
| Baseline 3 | Human-designed benchmarks | Gold standard for benchmark quality assessment |
| Ablation of ours | CONCH without symbolic verifier (just LLM) | Isolates impact of symbolic consistency checking |
| Ablation of ours | CONCH without learned verifier (just symbolic) | Isolates impact of learned consistency checking |

### Why this setup validates the claim
The dataset spans diverse domains to test generality of CONCH across different abstractions. The primary pass rate metric directly quantifies logical consistency between high-level intent and low-level implementation. The secondary abstraction coverage metric evaluates whether the abstraction library is sufficiently complete to capture relevant constraints. Random baseline establishes the difficulty of meeting constraints without guidance; single-pass baseline isolates the value of iterative repair; human baseline sets an upper bound on achievable consistency. The dual ablation separates the contributions of symbolic and learned verifiers. If CONCH achieves significantly higher pass rates than random and single-pass, and approaches human-level consistency, the claim that hybrid verification with repair enforces benchmark correctness is supported. Conversely, if CONCH fails to outperform baselines or if the learned verifier ablation shows no drop, the added value is unclear.

### Expected outcome and causal chain

**vs. Random parameter generation** — On a case where target difficulty is 0.7, random sampling produces a benchmark with actual difficulty 0.2 because it ignores constraints. Our method enforces constraints via the verifier, so we expect a pass rate of >90% for CONCH versus <10% for random, a large gap.

**vs. Single-pass LLM generation** — On a case requiring complex interactions (e.g., agent speed and obstacle layout must be compatible), the LLM often generates conflicting parameters because it lacks feedback. Our method iteratively repairs based on verifier conflict reports, so we expect CONCH pass rate ~80% vs single-pass ~50%, a moderate but clear advantage.

**vs. Human-designed benchmarks** — Humans may overlook subtle inconsistencies (e.g., hidden state conflicts) due to cognitive load. Our verifier catches all formalizable contradictions. We expect CONCH pass rate ~90% vs human ~95%, a small gap suggesting near-human quality.

**vs. CONCH without symbolic verifier (ablation)** — On a case where constraints are purely symbolic (e.g., strict inequality), removing the symbolic verifier will miss violations that the learned verifier cannot capture. Expected pass rate drop from ~90% to ~70%, demonstrating the necessity of symbolic verification.

**vs. CONCH without learned verifier (ablation)** — On a case where symbolic verifier passes due to incomplete abstraction but actual inconsistency exists (e.g., difficulty relies on unmodeled agent behavior), removing the learned verifier will let such errors through. Expected pass rate drop from ~90% to ~80%, showing the benefit of the learned check.

### What would falsify this idea
If CONCH's pass rate is not significantly higher than single-pass generation, or if performance gain is uniform across simple and complex constraints (indicating verifier adds no targeted benefit), the central claim that hybrid symbolic and learned verification with iterative repair improves benchmark consistency would be wrong. Additionally, if the learned verifier ablation shows no performance drop, the added complexity is not justified.

## References

1. Automating Benchmark Design
2. LLM-POET: Evolving Complex Environments using Large Language Models
3. Automatic Generation of Benchmarks and Reliable LLM Judgment for Code Tasks
4. APIGen: Automated Pipeline for Generating Verifiable and Diverse Function-Calling Datasets
5. CrossCodeEval: A Diverse and Multilingual Benchmark for Cross-File Code Completion
6. SWE-bench: Can Language Models Resolve Real-World GitHub Issues?
7. Evolution Gym: A Large-Scale Benchmark for Evolving Soft Robots
8. MarioGPT: Open-Ended Text2Level Generation through Large Language Models
