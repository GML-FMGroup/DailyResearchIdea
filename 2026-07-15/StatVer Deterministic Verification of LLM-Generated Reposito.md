# StatVer: Deterministic Verification of LLM-Generated Repository Knowledge via Static Program Semantics

## Motivation

Across recent QA-driven repository knowledge acquisition systems such as Know Before Fix, LLM-generated (question, answer) pairs are used without external verification, leading to propagation of incorrect or unsupported answers into downstream patch generation. The structural flaw is that all these systems treat LLM outputs as authoritative despite lacking a correctness guarantee from the codebase's static properties, because the LLM has no inherent grounding in call graph reachability, type constraints, or documented invariants.

## Key Insight

Static program semantics—call graph reachability, type constraints, and documented invariants—provide a decidable ground truth that can verify whether a claimed answer is logically entailed by the codebase, eliminating the need for probabilistic validation.

## Method

(A) **What it is**: StatVer is a deterministic verification module that takes an (question, answer) pair and a repository codebase, and returns a boolean indicating whether the answer is logically entailed by the static semantics of the codebase. Input is a natural language question and answer (from an LLM); output is a verification decision.

(B) **How it works**:
```python
def stat_ver(question: str, answer: str, repo: Repository) -> bool:
    # 1. Parse answer to extract claims (call, type, invariant, data_flow).
    claims = extract_claims(answer)  # list of (claim_type, subject, predicate, object)

    # 2. For each claim, generate static analysis query.
    for claim in claims:
        if claim.type == "call_graph":
            # assert that function subject calls function object
            query = CallReachability(subject, object, max_depth=3)
        elif claim.type == "type":
            # assert that variable subject has type object
            query = TypeConstraint(subject, object, strictness=0.9)
        elif claim.type == "invariant":
            # assert that invariant subject holds at location object
            query = InvariantCheck(subject, location=object, source="documentation")
        elif claim.type == "data_flow":
            # assert that value flows from subject to object
            query = DataFlowReachability(subject, object, max_depth=5)
        else:
            return False  # unsupported claim type

        # 3. Execute static analysis and check entailment.
        result = repo.static_analysis(query)
        if not result:
            return False  # claim not entailed
    return True  # all claims entailed
```
Hyperparameters: max_depth=3 for call reachability, strictness=0.9 for type matching, source="documentation" for invariants, max_depth=5 for data flow reachability.

(C) **Why this design**: We chose a claim-extraction step over direct end-to-end verification because static analysis operates on structured predicates, not free text; this requires parsing natural language into formal claims, accepting the cost that imperfect parsing may miss or misrepresent claims. We chose a per-claim verification approach (rather than a holistic one) to provide granular feedback and allow partial verification, trading off computational overhead for modularity. We defined four claim types (call graph, type, invariant, data flow) based on the most commonly verifiable static properties in repository code; this limits coverage but ensures each verification is tractable and deterministic. **This design assumes that all knowledge relevant to verifying the answer is encoded in these static properties (call graph reachability, type constraints, documented invariants, data flow reachability) of the codebase.** Hyperparameters like max_depth and strictness are set to balance precision and recall, acknowledging that some entailments may require deeper analysis or fuzzy matching. This design deliberately avoids the anti-pattern of a learned discriminator because static properties are inherently decidable; unlike prior work like RepoSearch-R1 which relies on environment feedback for training, StatVer uses first-order logic over code structure.

(D) **Why it measures what we claim**: The computational quantity `result` from static analysis measures logical entailment of the answer by the codebase because we assume that all knowledge relevant to the answer is encoded in the static properties we check (call graph reachability, type constraints, documented invariants, data flow reachability). Specifically, the answer is considered logically entailed by the codebase iff each extracted claim is entailed by the static analysis. This decomposition assumes that the set of extracted claims perfectly captures the meaning of the answer; failure in claim extraction (missing or incorrect claims) directly leads to false negatives or false positives respectively. This assumption fails when the answer depends on runtime behavior (e.g., dynamic dispatch, external libraries) or undocumented invariants, in which case the metric reflects false negatives (incorrectly rejecting a correct answer). The `max_depth` hyperparameter measures the scope of call graph or data flow exploration; it operationalizes "reachability" under the assumption that indirect calls/paths beyond the depth limit are rare; this assumption fails in highly modular code where deep chains are common, causing false negatives. The `strictness` hyperparameter measures type compatibility; it operationalizes "type correctness" under the assumption that similar types (e.g., subclass) are acceptable; this assumption fails when strict typing is required, leading to false positives if too lenient, or false negatives if too strict.

## Contribution

['(1) The StatVer algorithm for verifying LLM-generated (question, answer) pairs against static program semantics using call graph reachability, type constraints, and documented invariants.', '(2) A design principle showing that deterministic static analysis can serve as a ground-truth oracle for repository knowledge acquisition, reducing reliance on LLM self-consistency or environment feedback.', '(3) A mapping from natural language claims to formal static analysis queries, enabling integration into existing QA-driven pipelines like Know Before Fix.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|-----------------------|
| Dataset | RepoQA-static | focuses on static properties |
| Primary metric | F1 score | balances precision/recall |
| Baseline-1 | Direct LLM answer | shows need for verification |
| Baseline-2 | RepoSearch-R1 | strong agent with feedback |
| Baseline-3 | Naive AST matching | simple static baseline |
| Ablation-of-ours | StatVer w/o extraction | tests claim parsing importance |
| Claim extraction accuracy | 1000 labeled claims from RepoQA-static | validates parsing step |
| Coverage analysis | Random sample of 500 RepoQA-static questions | estimates fraction statically verifiable |

### Why this setup validates the claim
This experimental design tests the central claim that StatVer can determine logical entailment of answers using repository static semantics. RepoQA-static contains questions whose answers are verifiable via static properties (call graphs, types, invariants, data flow), providing ground truth entailment labels. The F1 metric captures both precision and recall of the binary verification decision. Direct LLM (no verification) tests whether verification is necessary; RepoSearch-R1 tests against a learned, feedback-driven method; Naive AST matching tests the added value of structured claim extraction. The ablation (StatVer without claim extraction) isolates the contribution of parsing natural language into formal claims. Additionally, claim extraction accuracy is measured on 1000 labeled claims to validate the parsing step, and coverage analysis estimates the proportion of questions that are statically verifiable on real-world repositories. If StatVer outperforms all baselines and the ablation degrades, it validates that deterministic static analysis with claim extraction is effective for entailment checking on static properties.

### Expected outcome and causal chain

**vs. Direct LLM answer** — On a case where the answer states "function `A` calls `B`" and `B` is reachable via a depth-2 call chain, the Direct LLM may hallucinate or miss context, producing a false acceptance or rejection because it lacks code analysis. Our method traverses the call graph with max_depth=3, correctly verifying the entailment. We expect a noticeable gap: StatVer achieves F1 > 0.8 on static questions, while Direct LLM achieves F1 < 0.5.

**vs. RepoSearch-R1** — On a type assertion like "variable `x` is type `int`", RepoSearch-R1 may have learned from environment rewards and could guess correctly for common patterns, but it may produce false positives on rare type hierarchies due to reliance on statistical correlation rather than deterministic analysis. Our method uses type constraint analysis with strictness=0.9, precisely checking subtyping. We expect StatVer to show higher precision (fewer false positives) on type claims, while recall remains comparable.

**vs. Naive AST matching** — On a case where the answer says "function `A` calls `C`" but `A` contains a lambda that passes `C` to a higher-order function, naive AST matching fails to detect the call because it only looks for direct call syntax. Our method uses call reachability analysis to follow indirect calls, correctly verifying despite syntactic indirection. We expect StatVer to have recall >0.9 on call graph claims, while Naive AST matching achieves recall <0.6.

**vs. StatVer w/o extraction** — On a data flow claim like "value `v` flows to `w`", the ablation (which uses raw text matching) may miss the connection, while full StatVer uses data flow reachability. We expect StatVer to achieve recall >0.85 on data flow claims, while the ablation achieves recall <0.4.

### What would falsify this idea
If StatVer's F1 score is not significantly higher than Naive AST matching on the static subset, or if the ablation without claim extraction performs equally well, then the claim extraction and structured analysis are not providing added value, contradicting the central idea. Also, if StatVer fails on simple static properties due to hyperparameter misconfiguration (e.g., many false negatives from max_depth too low), that would indicate the approach is not robust.

## References

1. Know Before Fix: QA-Driven Repository Knowledge Acquisition for Software Issue Resolution
2. Empowering RepoQA-Agent based on Reinforcement Learning Driven by Monte-carlo Tree Search
3. Live-SWE-agent: Can Software Engineering Agents Self-Evolve on the Fly?
4. Training Software Engineering Agents and Verifiers with SWE-Gym
5. SWE-Search: Enhancing Software Agents with Monte Carlo Tree Search and Iterative Refinement
6. SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering
7. OpenHands: An Open Platform for AI Software Developers as Generalist Agents
8. AgentTuning: Enabling Generalized Agent Abilities for LLMs
