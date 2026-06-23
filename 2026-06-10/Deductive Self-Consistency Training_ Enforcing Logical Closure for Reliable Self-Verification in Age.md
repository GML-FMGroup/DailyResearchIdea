# Deductive Self-Consistency Training: Enforcing Logical Closure for Reliable Self-Verification in Agents

## Motivation

Existing self-verification methods, such as the self-validation and self-consistency checks in RHO (Evolving Agents in the Dark), rely on heuristic or independent verification signals that may not reflect true task quality. The root cause is that self-verifications are generated without enforcing logical consistency among themselves, allowing contradictory signals to persist undetected. This structural limitation means the agent lacks a self-supervised way to internally detect unreliable verifications, because any single verification can be incorrect without creating an observable contradiction.

## Key Insight

Logical consistency of self-verifications is a self-verifiable property because violations of deductive closure produce detectable contradictions that require no external ground truth, enabling a fully self-supervised training signal for verification reliability.

## Method

```python
from typing import List, Dict, Any
import itertools

def dsct_loss(trajectory: List[Dict], agent) -> float:
    """
    trajectory: list of steps, each step dict with 'action', 'verification', 'state'
    agent: provides method 'implies(premises, conclusion) -> bool' (internal implication generator)
    returns: scalar loss = number of closure violations / total possible implications
    """
    # Step 1: Extract all verifications from trajectory
    verifications = [step['verification'] for step in trajectory if step.get('verification') is not None]
    if len(verifications) < 2:
        return 0.0
    
    # Step 2: Generate all ordered pairs (v_i, v_j) where v_i could imply v_j
    implications = []
    for v_i, v_j in itertools.product(verifications, repeat=2):
        if v_i is not v_j and agent.implies(v_i, v_j):  # self-implication from agent's knowledge
            implications.append((v_i, v_j))
    
    # Step 3: Count violations: v_i is verified (True) but v_j is not verified (False or missing)
    verification_set = set(verifications)
    violations = 0
    for v_i, v_j in implications:
        if v_i in verification_set and v_j not in verification_set:
            violations += 1
    
    # Normalize by total implications (or max possible)
    total_possible = max(1, len(implications))
    closure_loss = violations / total_possible
    
    # Step 4: Also penalize if v_j is verified but v_i is not (optional symmetric check)
    # (omitted for brevity; in full version we use directed closure only)
    
    return closure_loss
```

**Implementation specifics**: `agent.implies` is implemented as a hand-crafted rule-based system consisting of 30 manually defined logical implication rules specific to the SWE-Bench domain (e.g., "verified solution matches expected output" implies "step is correct"). These rules are defined as predicate templates over verification strings and are fixed during training (not learned). The rules are designed to be sound (high precision) but not complete. The agent is a transformer-based policy (e.g., CodeLlama-7B) fine-tuned via PPO with the DSCT loss added as a regularization term (weight λ=0.1). Training uses a batch size of 8 trajectories, learning rate 1e-5, and runs for 10,000 steps. The closure loss is computed per trajectory and averaged over the batch.

(C) **Why this design**: We chose explicit logical closure checking over implicit consistency via contrastive learning (e.g., SimCLR-style) because closure provides a deterministic ground-truth for inconsistency, avoiding the need for negative sampling heuristics. We used the agent's own `implies` function rather than an external knowledge base to maintain end-to-end self-supervision, accepting the risk that the implication generator may be incomplete (missing implications) or unsound (false implications) — a tradeoff we mitigate by initially using a hand-crafted `implies` that is sound but incomplete, and later plan to train it jointly. We opted for a per-trajectory normalized loss rather than a global regularizer (e.g., adding consistency as a KL penalty) because it allows gradient updates on each sample, which is more sample efficient; the cost is increased variance due to trajectory length variation. Finally, we used directed implications (if premise then conclusion) rather than bidirectional equivalence because agent verifications are often asymmetric (e.g., 'answer is correct' implies 'step 3 is correct' but not vice versa), and forcing symmetry would introduce false violations.

(D) **Why it measures what we claim**: Computational quantity `closure violation count` measures `self-verification reliability` because we assume that any logical implication derivable from the agent's own knowledge (via `implies`) should be reflected in the set of verifications; this assumption fails when the implication generator is incomplete (missing valid implications) or the agent's knowledge contains contradictions itself, in which case closure violations may indicate missing or incorrect implications rather than unreliable verifications — we handle this by initially using a hand-crafted `implies` that is sound but incomplete, so violations indicate true contradictions while low violations may still hide unreliability. The normalized loss `closure_loss` measures the fraction of missed implications, which we claim is a proxy for the degree of logical inconsistency; this holds if the set of implications considered is representative of all relevant logical relationships, which is approximate until the implication generator is fully trained. With our hand-crafted rules, we only detect a subset of possible contradictions, so the loss is a lower bound on actual inconsistency.

## Contribution

(1) Deductive Self-Consistency Training (DSCT), a self-supervised loss that enforces logical closure in an agent's self-verifications using its own internal implication generator. (2) The principle that logical consistency provides a self-verifiable signal for improving verification reliability without external labels, applicable to any agent architecture with a symbolic or neural implication module. (3) A lightweight implication generation module that can be trained jointly with the agent to produce task-relevant logical rules.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | SWE-Bench Pro | Standard agent benchmark, used in related work |
| Primary metric | Task success rate (pass@1) | Direct measure of agent utility |
| Baseline | RHO | Self-supervised harness optimization |
| Baseline | GEPA | Reflective prompt evolution, sample-efficient |
| Baseline | ADAS | Automated agent design via meta-evolution |
| Ablation-of-ours | DSCT w/o closure loss | Isolates effect of closure objective |
| Additional diagnostic | Instances with known complete implication coverage | Tests sensitivity to implication completeness |

### Why this setup validates the claim

The chosen dataset (SWE-Bench Pro) provides a realistic multi-step agent task where trajectory verifications are naturally assessable. The primary metric, task success rate, directly reflects whether improvements in verification reliability translate to better outcomes. Comparing against RHO, GEPA, and ADAS tests different sub-claims: RHO validation via self-preference, GEPA's sample efficiency, and ADAS's ability to discover novel agent programs. The ablation of removing closure loss pinpoints whether DSCT's gains stem from logical consistency rather than raw self-verification. Using pass@1 avoids conflating reliability with search budget, making it a strict test: if DSCT improves logical closure but not success rate, the central claim is unsupported. The diagnostic set (instances where our 30 hand-crafted implication rules are known to be complete) directly tests the linking assumption: if closure loss improvements only help when coverage is high, that confirms the sensitivity to implication quality.

### Expected outcome and causal chain

**vs. RHO** — On a case where the agent's initial verifications are logically inconsistent (e.g., verifying a step as correct but later contradicting it), RHO's self-preference may reinforce these contradictions because it optimizes based on agent's own preferences without explicit consistency checks. Our DSCT directly penalizes such violations via closure loss, so we expect a noticeable gap on trajectories with many verifications, where RHO's success rate plateaus but DSCT steadily improves, especially on instances requiring multi-step reasoning. For example, if the agent verifies "code compiles" and later "test passes" but a rule states "test passes implies code compiles", DSCT will penalize if both are verified yet the implied condition (code compiles) is not present (which would be contradictory in the agent's state). RHO may not detect this inconsistency.

**vs. GEPA** — On a case where a single verification error cascades (e.g., a misverified intermediate step leads to wrong final answer), GEPA's prompt evolution might correct it given enough rollouts, but its sample efficiency is limited because it relies on sparse reward signals. DSCT exploits the dense closure signal from logical implications, thus requiring fewer trajectories to identify and fix inconsistency. We expect DSCT to match or exceed GEPA's final performance with roughly 35x fewer samples, as seen in GEPA's own claims.

**vs. ADAS** — On a case where the agent architecture itself encourages contradictory verifications (e.g., a memory-less model that asserts inconsistent facts across steps), ADAS might evolve a new architecture to mitigate this, but it requires many generations. DSCT operates within a fixed architecture by enforcing logical closure, offering a complementary improvement. We expect DSCT to show stronger gains on the same architecture, while ADAS may find orthogonal gains. The observed signal: DSCT's improvement over ADAS's best architecture is moderate but consistent on verification-heavy tasks.

**vs. DSCT w/o closure loss** — On a case where the agent's self-verification is already decent but not logically closed (e.g., it verifies each step independently but misses cross-step consistency), the ablation will still produce some verification signals but fail to penalize contradictions. DSCT full shows higher success rates on tasks requiring step interdependence, while ablations may even degrade due to overconfidence. Expect a clear gap on multi-step logical tasks.

### What would falsify this idea
If DSCT's improvements are uniform across all task types (including single-step ones where closure is trivial) rather than concentrated on multi-step tasks with high verification counts, then the central claim—that closure violation loss drives gains—would be falsified, suggesting the loss is merely acting as a regularizer rather than targeting logical inconsistency. Furthermore, if on the diagnostic set (with complete implication coverage) DSCT shows no additional gain over the incomplete set, then the linking assumption that closure loss measures verification reliability is unsupported.

## References

1. Evolving Agents in the Dark: Retrospective Harness Optimization via Self-Preference
2. DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines
3. GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning
4. Automated Design of Agentic Systems
5. Demonstrate-Search-Predict: Composing retrieval and language models for knowledge-intensive NLP
6. Prompting Is Programming: A Query Language for Large Language Models
7. Large Language Models Can Self-Improve
8. Mathematical discoveries from program search with large language models
