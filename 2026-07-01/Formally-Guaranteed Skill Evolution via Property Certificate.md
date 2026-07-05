# Formally-Guaranteed Skill Evolution via Property Certificate Verification

## Motivation

SkillHone relies on the LLM's implicit reliability to generate improvements, lacking any formal mechanism to ensure that revisions are actually beneficial. This trust assumption can lead to regressions and wasted effort, as the LLM may propose changes that degrade performance without detection. We address this limitation by coupling persistent history with a formal property lattice and a symbolic verifier, ensuring that each revision preserves all previously verified properties and adds at least one new property, thereby guaranteeing monotonic skill improvement.

## Key Insight

Monotonic improvement is guaranteed by requiring every revision to come with a verifiable certificate that expands the skill's property set, not by trusting the LLM's judgment.

## Method

### (A) What it is
FGSE (Formally-Guaranteed Skill Evolution) is a framework that maintains a property lattice for each agent skill. Given a persistent decision history (from SkillHone), it extracts a set of behavioral properties (LTL formulas on finite traces) that the current skill satisfies, using mining via a bounded model checker (trace length bound 10). When proposing a revision, the LLM must also provide a certificate: a formal proof that the new version preserves all existing properties and adds at least one new property. A symbolic verifier (CBMC on translated C code) checks the certificate, and only if it passes does the revision replace the skill. Additionally, after verification, the revision is evaluated on a held-out set of tasks (size 100) from the same distribution; the revision is accepted only if task completion rate does not decrease. The input is the current skill code + history + property lattice; the output is an updated skill code and an updated property lattice. This framework relies on the load-bearing assumption that the extracted properties are causally relevant to task performance; if a property is spurious, the monotonic guarantee may not translate to empirical gains.

### (B) How it works
```python
def evolve_skill(skill_program, history, property_lattice):
    # 1. Propose revision and certificate
    revision, certificate = LLM_propose(skill_program, history, property_lattice)
    # certificate is a formal proof that:
    #   - revision preserves all properties in property_lattice
    #   - there exists a new property p_new not in property_lattice such that revision satisfies p_new
    # The certificate may include intermediate lemmas and calls to the verifier.
    
    # 2. Verify certificate
    if verifier.verify(certificate, revision, property_lattice):
        # 3. Compute new property lattice: add p_new to property_lattice
        p_new = certificate.new_property()
        new_lattice = property_lattice + {p_new}
        # 4. Evaluate on held-out tasks (100 tasks) to check for regression
        held_out_performance = evaluate(revision, held_out_tasks)
        if held_out_performance >= current_performance:
            # 5. Return new skill and lattice
            return revision, new_lattice
        else:
            return skill_program, property_lattice
    else:
        # Revision fails verification; keep current skill
        return skill_program, property_lattice
```
**Details**: The LLM is prompted with the current skill code, history summaries, and the list of current properties. It generates a revision and a structured certificate. The verifier is CBMC (C Bounded Model Checker) for the skill language subset (Python with no recursion, loops with fixed bounds ≤10 iterations, integer and boolean variables). Properties are restricted to LTL over finite traces (LTLf) with maximum trace length 20, ensuring decidability via bounded model checking.

### (C) Why this design
We chose a symbolic verifier over LLM self-verification because the LLM's own output is unreliable for correctness; using an external verifier provides a ground truth that is not subject to the same failure modes. The trade-off is that the verifier introduces computational overhead (approx. 1 second per certificate verification) and requires properties to be in a formal language, limiting expressiveness. We chose a property lattice over a scalar quality score because a lattice enables fine-grained, composable guarantees: the set of properties explicitly captures what the skill has been proven to achieve, and the monotonic increase guarantees that no prior capability is lost. The cost is that extracting properties from history requires an additional abstraction step (mining LTLf invariants via bounded model checking). We chose to require a certificate from the LLM rather than relying on the verifier to search for properties directly because the LLM can leverage the history to suggest plausible new properties, while the verifier only checks; this division of labor reduces the search space for the verifier. The trade-off is that the LLM may propose trivial properties, but the verifier ensures they are genuine. The held-out evaluation step was added to mitigate the risk that properties are not causally relevant; it ensures that each revision does not cause empirical regression, complementing the formal guarantee.

### (D) Why it measures what we claim
The computational quantity `certificate.new_property()` measures the concept of "skill improvement" because it identifies a behavioral property that the revision satisfies and previous versions did not, under the assumption that property satisfaction is the correct notion of progress for the task; this assumption fails when the property is not causally relevant to the task (e.g., a spurious invariant), in which case the method records progress that does not actually help the agent. The verification step `verifier.verify(certificate, revision, property_lattice)` measures the concept of "monotonic preservation" because it checks that each property in the lattice still holds for the revision, under the assumption that the verifier is sound for the property language; if the verifier is incomplete (e.g., due to abstraction), it may accept a revision that actually violates a property, compromising the guarantee. The persistent history provides the raw evidence from which properties are extracted; the extraction process must identify properties that are both true of the current skill and discriminative of performance—if extraction misses critical properties, the lattice may be too weak to ensure convergence. We assume a positive correlation between property satisfaction and task completion rate. To test this, we will measure the Pearson correlation between the number of verified properties in the lattice and the success rate on a held-out evaluation set after each revision. If the correlation is weak (|r|<0.3), the monotonic guarantee may not translate to performance gains, and we would consider revising the property extraction method.

## Contribution

(1) A framework for formally-guaranteed skill evolution that integrates LLM-based revision proposals with a symbolic verifier and property lattice, ensuring monotonic improvement. (2) The design principle of separating generation from verification, enabling formal convergence guarantees without requiring the LLM to be reliable. (3) A technique for extracting behavioral properties from persistent decision history using diagnostic records and outcome evidence.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|------------------------|
| Dataset | WebWalkerQA-EN | Tests web navigation skill evolution |
| Primary metric | Task completion rate | Measures practical skill improvement |
| Baseline | SkillHone | Prior skill evolution without formal guarantees |
| Baseline | Automated Skill Discovery | Alternative skill discovery approach |
| Baseline | Static agent with fixed skills | No evolution baseline |
| Baseline | Random mutation + verifier | Isolates LLM contribution |
| Ablation | FGSE w/o verifier (LLM only) | Isolates effect of formal verification |
| Ablation | FGSE w/o held-out check | Isolates effect of empirical regression check |

### Why this setup validates the claim

This combination of dataset, baselines, metrics, and ablations directly tests FGSE's central claim: that formal verification combined with property preservation ensures monotonic skill improvement that translates to task performance. WebWalkerQA-EN requires agents to sequentially navigate websites, where skill regression is detrimental. Task completion rate captures whether evolved skills actually help. Comparing against SkillHone tests whether property preservation matters; against Automated Skill Discovery tests whether selective addition via certificates outperforms undirected discovery; against static agent tests necessity of evolution; against random mutation+verifier isolates the LLM's role in proposing useful properties. The ablations (no verifier, no held-out check) isolate the contributions of verification and empirical regression checking. Additionally, we will compute the Pearson correlation between the number of verified properties and task success rate across revisions; a weak correlation would indicate that property extraction needs improvement, validating the assumption underlying the monotonic guarantee.

### Expected outcome and causal chain

**vs. SkillHone** — On a case where a skill revision inadvertently loses the ability to click a previously learned link pattern, SkillHone accepts the revision because it lacks property preservation. Our method requires a certificate proving all previous properties hold, so the revision is rejected, preserving that ability. Thus we expect FGSE to have a noticeably higher success rate on tasks that reuse such patterns, with the gap widening as task complexity grows.

**vs. Automated Skill Discovery** — On a case where the agent explores and proposes a skill that is actually irrelevant (e.g., scrolling down before clicking a button), that method adds it anyway since it only checks novelty. FGSE requires the new skill to satisfy a new property derived from history, which will be rejected if the skill does not actually improve behavior. Hence we expect FGSE to accumulate fewer useless skills and converge faster to high success rates on tasks requiring only useful skills.

**vs. Static agent** — On tasks requiring adaptation (e.g., a new website layout), the static agent cannot change its skills, so it fails on all such tasks. FGSE evolves by adding properties that handle the new layout, so we expect FGSE to succeed on a growing fraction of these tasks, while the static agent remains at zero. The gap will be large and increasing over time.

**vs. Random mutation + verifier** — On a case where the LLM proposes a certificate that precisely targets a missing property (e.g., handling a pop-up), FGSE will add it; random mutation is unlikely to generate such a targeted property because mutations are blind. Thus we expect FGSE to achieve higher success rates on tasks requiring specific new capabilities, and to require fewer revisions overall, supporting the value of the LLM's guided proposal.

### What would falsify this idea

If FGSE's task completion rate is not significantly higher than its ablation (FGSE without verifier) on tasks where property preservation is critical, then the central role of formal verification is false. A uniform improvement across all tasks would also suggest the source is not the causal chain we hypothesize. Additionally, if the correlation between property count and task success rate is weak (|r|<0.3), the load-bearing assumption of causal relevance would be undermined, invalidating the motivation for the approach.

## References

1. SkillHone: A Harness for Continual Agent Skill Evolution Through Persistent Decision History
2. Multi-Agent Collaboration: Harnessing the Power of Intelligent LLM Agents
3. Automated Skill Discovery for Language Agents through Exploration and Iterative Feedback
4. Live-SWE-agent: Can Software Engineering Agents Self-Evolve on the Fly?
5. Chain of Thought Prompting Elicits Reasoning in Large Language Models
6. OpenWebVoyager: Building Multimodal Web Agents via Iterative Real-World Exploration, Feedback and Optimization
7. DataEnvGym: Data Generation Agents in Teacher Environments with Student Feedback
8. Proposer-Agent-Evaluator(PAE): Autonomous Skill Discovery For Foundation Model Internet Agents
