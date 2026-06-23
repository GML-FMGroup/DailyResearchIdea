# Decentralized Verification Ensemble via Cooperative Multi-Agent Learning for Adaptive Formal Verification

## Motivation

Existing multi-agent discovery systems (e.g., the concept discovery system with decentralized conjecture generation) rely on a fixed, centralized verifier that cannot generalize beyond a single formal system. This fixed-verifier assumption prevents autonomous exploration of arbitrary mathematical domains. To enable full autonomy, the verification mechanism itself must adapt to new formal systems without human intervention.

## Key Insight

Cooperative multi-agent learning enables a diverse set of verifier agents to collectively explore and adapt their verification strategies by sharing feedback, analogous to how a scientific community reviews each other's work across different formalisms.

## Method

# Cooperative Decentralized Verification Ensemble (CoDVE)

(A) **What it is**: CoDVE is a multi-agent reinforcement learning framework that replaces the fixed verifier with a set of decentralized verifier agents. Each agent learns to verify conjectures in a given formal system, and they cooperate to adapt their policies across different systems. Input: a formal system description (axioms and inference rules) and a pool of candidate conjectures. Output: verified theorems with associated proofs.

**Key assumption**: Cooperative reward sharing and shared knowledge base with consensus mechanism enable agents to learn reusable proof strategies that generalize across different formal systems. To mitigate risks of misaligned rewards and spurious consensus, we replace the cooperative reward with difference rewards (Wolpert & Tumer, 2002) and validate consensus lemmas via a lightweight independent verifier (e.g., Lean 4 kernel) before adding to the knowledge base.

(B) **How it works**:
```python
# CoDVE: Cooperative Decentralized Verification Ensemble
# Input: formal system S = (Axioms, InferenceRules), pool of conjectures C
# Initialize: N=5 verifier agents, each with a differentiable verification policy π_i (3-layer Transformer,
#             4 heads, hidden_size=128, GELU activation) trained via PPO with clip ratio 0.2
# Each agent i maintains a state s_i = (proof_state, context) as token sequence
# Also maintain a shared knowledge base K (set of verified theorems and proof patterns)
# Training loop:
for epoch in range(1000):
    # Phase 1: Decentralized verification attempt
    for each conjecture c in C:
        for each agent i (in parallel):
            # Agent i tries to prove c using current policy, guided by K
            proof_i = π_i(c, S, K)
            # Each proof step is a move in a verification game; agent receives reward +1 if step is valid, -1 if invalid
            # At end, agent i gets overall reward R_i = 1 if proof is complete and valid, else 0
            # Also collect trajectory (states, actions, rewards)
    # Phase 2: Cooperative reward sharing using difference rewards
    # Agents share their proof trajectories and outcomes
    # Compute cooperative reward for each agent:
    #   R_diff_i = R_i - baseline_i, where baseline_i = mean of R_i over last 100 episodes (per agent)
    #   Then R_coop_i = R_diff_i + 0.5 * (avg_proof_length_improvement_over_other_agents - moving_average)
    # Update policies via PPO using R_coop_i as advantage
    # Phase 3: Knowledge base update
    # Verified theorems are added to K
    # Agents distill shared proof patterns via consensus: if multiple agents use the same proof step sequence (exact match)
    #   and it is verified by a lightweight independent theorem prover (Lean 4 kernel), it is promoted as a lemma and added to K
    # Periodically (every 50 epochs), agents can propose modifications to the inference rules (if allowed) to adapt to new systems;
    #   proposals require majority vote among agents
# Output: set of verified theorems with proofs
```

(C) **Why this design**: We chose cooperative reward sharing (instead of purely individual rewards) to encourage agents to produce proofs that are reusable by others, accepting the cost that some agents may sacrifice their own reward to produce more general proofs. We use shared knowledge base K (instead of independent memory) to leverage collective experience across formal systems, but this introduces communication overhead and potential for stale knowledge. We include periodic rule-modification proposals (instead of fixed rules) to enable adaptation to arbitrary formal systems, but this risks destabilizing the verification process if agents propose inconsistent rule changes. The combination of decentralized verification attempts with cooperative credit assignment allows agents to specialize in different proof strategies while still benefiting from each other's successes.

(D) **Why it measures what we claim**: The reward R_i = 1 for a complete valid proof measures verification success because it directly corresponds to the agent's ability to produce a correct derivation in the formal system; this assumption fails when the proof is invalid but the agent's policy accidentally produces a superficially correct sequence (e.g., due to lucky random actions), in which case R_i measures spurious correctness. The cooperative reward component (difference reward) measures an agent's marginal contribution to cross-agent proof utility via proof length improvement; this assumption fails when other agents cannot actually reuse the proof pattern due to different internal representations, in which case the metric measures only superficial similarity. The shared knowledge base K is updated with verified theorems and consensus proof patterns (which are independently verified), measuring accumulated community knowledge because it aggregates validated results across agents; this assumption fails when agents have different verification standards or the independent verifier is fallible, in which case K measures noise. We test transferability by swapping agents' proof trajectories during evaluation and measuring the success rate of using another agent's proof.

## Contribution

(1) A novel decentralized verification ensemble (CoDVE) that adapts to arbitrary formal systems via cooperative multi-agent learning, replacing the fixed verifier assumed in prior work. (2) A cooperative reward sharing mechanism that encourages reusable proof generation across agents, enabling the ensemble to generalize to new formal systems. (3) A framework that integrates multi-agent reinforcement learning with shared knowledge bases for adaptive theorem proving.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Formal system benchmark set (group theory, ring axioms, non-associative magmas, fragment of ZFC) | Diverse axiom sets challenge adaptability. |
| Primary metric | Verification success rate, avg proof length, swap success rate (proof reuse between agents) | Measures correctness, efficiency, and reusability. |
| Baseline 1 | Static single-agent verifier (same architecture but single agent, no sharing) | Proves need for multi-agent adaptation. |
| Baseline 2 | Rule-based prover (Aesop) | Tests against non-LM automation. |
| Ablation | CoDVE without cooperative reward (individual rewards only) | Isolates benefit of cooperation. |

### Why this setup validates the claim
Using a benchmark of diverse formal systems (group theory axioms, ring axioms, non-associative algebras, and a fragment of ZFC) tests the method's claim of adapting to arbitrary systems. The primary metrics—verification success rate, avg proof length, and swap success rate—capture correctness, efficiency, and reusability, the key dimensions of the proposed improvement. Comparing against a static single-agent verifier tests the necessity of multiple agents; a rule-based prover tests the advantage of learned adaptive policies over fixed heuristics. The ablation removes cooperative reward sharing to isolate its causal contribution. If the method works, we expect CoDVE to outperform both baselines on systems requiring cross-system lemma reuse or non-standard inference rules, while the ablation reveals the cooperative mechanism's role.

### Expected outcome and causal chain

**vs. Static single-agent verifier** — On a system where a theorem requires a non-obvious lemma not in the initial knowledge, a single agent fails because it cannot leverage insights from other agents. Our method uses the shared knowledge base where agents pool lemmas discovered by others, allowing it to find the lemma quickly. We expect CoDVE to show a large gap on such multi-step theorems, while performing comparably on simple ones where a single agent suffices. Additionally, the swap success rate should be high, indicating reusability.

**vs. Rule-based prover (Aesop)** — On a system with non-standard inference rules (e.g., non-associative magma laws), Aesop's fixed heuristics are misaligned, causing failure. Our method adapts via periodic rule-modification proposals, allowing agents to discover effective proof strategies for the new rules. We expect CoDVE to succeed on such systems while Aesop fails, but on standard systems (e.g., standard group theory) both perform similarly. The swap success rate on non-standard systems will indicate whether agents' adapted strategies are transferable.

### What would falsify this idea
If CoDVE's improvement is uniform across all systems (even those where no cross-system insight or rule adaptation is needed), or if the ablation without cooperative reward performs equally well, then the central claim that cooperative sharing and adaptation to new rules drives performance is wrong. Additionally, if the swap success rate is low (e.g., <30%), the assumption that cooperative rewards yield reusable proofs is falsified.

## References

1. Discovering mathematical concepts through a multi-agent system
2. Lean Copilot: Large Language Models as Copilots for Theorem Proving in Lean
3. The Equational Theories Project: Advancing Collaborative Mathematical Research at Scale
4. Machine learning and information theory concepts towards an AI Mathematician
5. Symbols and mental programs: a hypothesis about human singularity.
6. LeanDojo: Theorem Proving with Retrieval-Augmented Language Models
7. LLMSTEP: LLM proofstep suggestions in Lean
