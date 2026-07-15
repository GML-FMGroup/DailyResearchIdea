# VeriQA: Static-Verified Knowledge Acquisition for Software Repositories

## Motivation

Existing QA-driven knowledge acquisition methods (e.g., ACQUIRE from 'Know Before Fix') assume that generated answers are correct without validation, leading to propagation of errors in downstream tasks. This stems from the lack of a closed-loop verification mechanism that cross-checks acquired knowledge against code facts. We address this by integrating static program analysis as a verifier for atomic code claims extracted from QA interactions.

## Key Insight

Static analysis provides a sound, decidable ground truth for structurally typed and dependency-based code claims, enabling automated verification and targeted correction without runtime instrumentation.

## Method

### VeriQA: Static-Verified Knowledge Acquisition for Software Repositories

(A) **What it is**: VeriQA is a verification loop that wraps QA-driven knowledge acquisition. Input: codebase, issue description, static analysis tools (type checker: MyPy for Python/TypeScript compiler for JS/TS, call graph analyzer: pyan3). Output: verified set of atomic code claims.

(B) **How it works** (pseudocode):
```
1. Initialize knowledge base K = empty set.
   Hyperparameters: max_iterations=5, confidence_threshold=0.7
2. Generate initial question Q based on issue description.
3. For each QA iteration (max_iterations=5):
   a. Answer A = Answerer(Q, codebase)  // from ACQUIRE
   b. Claims C = extract_atomic_claims(A):
      - parse NL to subject-predicate-object triples
      - Heuristic: use SpaCy dependency parsing with 20 hand-crafted rules:
        subject: code entity (function, variable, class)
        verb: one of {returns, calls, contains, has_type, is_nullable}
        object: property (type, function name, boolean)
   c. For each claim c in C:
        result = static_verify(c):
          - if verb='returns': call type checker on function's inferred return type vs claimed type → True/False/Unknown
          - if verb='calls': check call graph (pyan3) for edge existence
          - if verb='has_type': check variable type annotation in AST
          - if verb='is_nullable': check if variable type is Optional or has None default
          - if verb='contains': check file existence in repository
          - if result == True: add c to K with confidence=1.0
          - if result == False: generate re-query Q' using template: 'For claim c, static analysis shows conflict. Please provide the correct <subject> <predicate> <object> with evidence (code snippet and file location).' 
                          trigger new QA iteration with Q'
          - if result == Unknown: add c to K with low-confidence flag (confidence=0.3)
   d. After each re-query, re-verify the new claim using static_verify; if still False, discard claim after 3 attempts.
4. Return K with confidence scores
```
Load-bearing assumption: The LLM-based answerer can reliably interpret static analysis feedback (e.g., type errors, call graph conflicts) to produce corrected atomic claims in subsequent re-queries. We calibrate this assumption by measuring re-query success rate on a development set (50 SWE-bench Lite issues); if success rate < 0.6, we adjust the re-query template or add fallback to discard low-confidence claims.

(C) **Why this design**: We chose static analysis over dynamic analysis because static analysis provides exhaustive coverage of compile-time properties without requiring test execution or runtime environments, accepting the cost of conservatism (potential false negatives for undefined behavior). We chose to decompose answers into atomic claims rather than verify entire answers holistically, because atomic claims enable precise localization of errors and targeted re-queries; the trade-off is increased parsing overhead and the risk of losing contextual relations between claims. We chose re-queries over discarding inconsistent claims because re-queries allow the QA system to correct its understanding, leveraging the LLM's ability to produce evidence-grounded answers when given specific feedback; this design assumes that the QA system can interpret the static analysis feedback, which may fail if the feedback is too technical or ambiguous—we mitigate this with a predefined template and measure re-query success rate.

(D) **Why it measures what we claim**: The verification status output (True/False/Unknown) for each claim measures the correctness of that claim relative to the codebase's static properties. The assumption is that static analysis captures the ground truth for the claim's truth value; this assumption fails when the claim concerns runtime behavior (e.g., 'variable x is always non-null'), in which case verification reflects a conservative over-approximation (Unknown). The re-query count measures the completeness of the initial QA knowledge: each re-query indicates a missing or incorrect piece of knowledge that requires correction. This measure relies on the assumption that the re-query leads to a correct claim; when the re-query cannot resolve the conflict (e.g., due to ambiguous feedback or a claim that is inherently undecidable by static analysis), the re-query count reflects attempts rather than actual progress—we therefore track re-query success rate (fraction of re-queries that yield a True claim after re-verification) as a separate metric.

## Contribution

(1) Introduction of a closed-loop verification mechanism for QA-driven knowledge acquisition using static program analysis. (2) A method for decomposing natural language answers into atomic code claims and mapping them to static analysis queries. (3) A targeted re-query strategy that improves knowledge correctness without additional model training.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|-----------------------|
| Dataset | SWE-bench Lite | Real-world repo issues with codebases. |
| Primary metric | Claim verification F1 | Measures correctness of extracted claims. |
| Secondary metric | Re-query success rate | Fraction of re-queries yielding True claim after re-verify. |
| Coverage metric | Static-verifiable claim coverage | Proportion of claims that are decidable by static analysis (True/False vs Unknown). |
| Baseline 1 | Know Before Fix | Direct QA without verification. |
| Baseline 2 | RepoQA-Agent (iterative) | Iterative retrieval without static check. |
| Baseline 3 | Live-SWE-agent | Online learning without explicit verification. |
| Ablation | VeriQA no static verify | Removes static verification step; still uses re-query on uncertain claims. |
| Calibration set | 50 random SWE-bench Lite issues | For tuning extraction heuristics and measuring re-query success. |

### Why this setup validates the claim
This design isolates the effect of static verification on knowledge acquisition correctness. SWE-bench Lite provides realistic codebases with known ground truth for claim correctness (via manual annotation of atomic claims on a 100-issue subset). The primary metric, claim verification F1, directly captures whether the system's claims match static analysis results (or ground truth for Unknown cases). Baselines contrast different approaches: Know Before Fix tests the necessity of any verification; RepoQA-Agent tests whether iterative retrieval alone suffices; Live-SWE-agent tests whether online learning can replace explicit static checks. The ablation removes static verification while keeping the re-query mechanism, attributing any gains specifically to static feedback. The secondary metric (re-query success rate) validates the load-bearing assumption that LLMs can correct claims given static feedback. Coverage metric quantifies how many claims the system can verify. Together, these comparisons form a falsifiable test: if static verification is critical, VeriQA should outperform all baselines and its ablation on claims requiring compile-time validation, and re-query success rate should be above a baseline (e.g., >0.5 on the calibration set).

### Expected outcome and causal chain

**vs. Know Before Fix** — On a case where the issue involves a type mismatch (e.g., passing a string to an int parameter), the baseline accepts the LLM's claim that the call is correct because it lacks static verification. Our method detects the type error via static analysis, marks the claim false, and triggers a re-query that corrects the claim (with re-query success rate expected ~0.7). We expect a noticeable gap on claims involving static properties (e.g., type errors, signature mismatches), with VeriQA achieving higher F1 on that subset while performing similarly on purely documentation claims.

**vs. RepoQA-Agent (iterative)** — On a case where the relevant code snippet is scattered across multiple files (e.g., a bug spanning a base class method and a derived override), the iterative retrieval may keep fetching the same wrong context due to misleading similarity scores, producing persistent incorrect claims. Our method, after extracting an atomic claim ‘method X returns type Y’, runs static call graph analysis to verify the return type, finds a conflict, and issues a re-query focused on that specific discrepancy. This yields a targeted correction. We expect VeriQA to show larger gains on multi-file, cross-module claims.

**vs. Live-SWE-agent** — On a case where the agent previously succeeded on similar issues but the current issue has a subtle static constraint (e.g., a nullable annotation that was introduced in a recent commit), the online learning baseline may over-rely on its past policy and neglect to re-check the code. Our static verification immediately flags the nullness mismatch and forces a re-query. We expect VeriQA to maintain consistent accuracy across evolving codebases, while Live-SWE-agent degrades on out-of-distribution static patterns.

### What would falsify this idea
If VeriQA’s claim F1 is not significantly higher than its ablation (no static verification) on the static-property subset, or if the gain is uniform across all claim types, then the central causal role of static verification is invalid. Specifically, if the re-query mechanism alone (without static feedback) suffices to correct errors, the idea fails. Additionally, if the re-query success rate on the calibration set is below 0.5, the load-bearing assumption is falsified and the method would need redesign (e.g., fallback to discarding False claims).

## References

1. Know Before Fix: QA-Driven Repository Knowledge Acquisition for Software Issue Resolution
2. Empowering RepoQA-Agent based on Reinforcement Learning Driven by Monte-carlo Tree Search
3. Live-SWE-agent: Can Software Engineering Agents Self-Evolve on the Fly?
4. Training Software Engineering Agents and Verifiers with SWE-Gym
5. SWE-Search: Enhancing Software Agents with Monte Carlo Tree Search and Iterative Refinement
6. SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering
7. OpenHands: An Open Platform for AI Software Developers as Generalist Agents
8. AgentTuning: Enabling Generalized Agent Abilities for LLMs
