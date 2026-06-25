# Adversarial Distillation and Verification for Robust Agent Experience Learning

## Motivation

The Execute-Distill-Verify (EDV) paradigm reduces executor-centric bias by using a dedicated third-party distillation agent, but this agent introduces a new single-point-of-failure bias: the distilled experiences reflect the distiller's own subgoal generation tendencies, as observed in HiAgent. This persistent bias limits the diversity and robustness of learned experiences, as the distillation agent can carry forward its own blind spots without structural correction. We replace the passive distillation agent with an adversarial generator-critic pair, structurally eliminating reliance on any single agent's perspective.

## Key Insight

Adversarial verification forces the distilled experience to survive directed criticism, ensuring that stored experiences are robust to the specific biases of any single agent.

## Method

### (A) What it is
ADV (Adversarial Distillation and Verification) is an extension of EDV that replaces the passive distillation agent with a Generator (G) that proposes experiences and a Critic (C) that attempts to refute them, using the original execution group as arbiter to adjudicate the adversarial exchange.

### (B) How it works
```
Algorithm: ADV
Input: Task space T, Generator G (LLM, e.g., GPT-3.5-turbo), Critic C (LLM, e.g., GPT-4), execution group E (heterogeneous agents, with at least one agent using a fundamentally different reasoning paradigm, e.g., neural-symbolic vs. pure neural, to reduce correlated blind spots), iteration limit K=5, consensus threshold θ=0.5 (majority of E)
Output: Verified experience library L

Phase 1: Execute
  for each task in T:
    E explores in parallel, generating trajectories T_set

Phase 2: Adversarial Distill-Verify
  for each trajectory t in T_set:
    repeat up to K times:
      e = G(t)                // proposed experience (e.g., subgoal decomposition)
      c = C(t, e)             // counterexample trajectory contradicting e
      // Evaluate via E consensus:
      valid_refutation = consensus(votes from E on "does c achieve goal ≥ t while violating e?") > θ
      if valid_refutation:
        reject e; penalize G (add c to context for future proposals); continue loop
      else:
        accept e; store in L; break
  return L
```

**Load-bearing assumption**: The execution group's consensus vote on refutation validity is a reliable ground-truth proxy for whether a counterexample genuinely refutes an experience. To mitigate correlated blind spots, we guarantee diversity among executors: at least one executor uses a fundamentally different reasoning paradigm (e.g., neural-symbolic vs. pure neural) or is trained on a disjoint dataset (e.g., distinct fine-tuning tasks). This reduces the risk of systematic errors being amplified by majority voting.

### (C) Why this design
We chose an adversarial generator-critic pair over a single distillation agent because the latter suffers from a single-point-of-failure bias (e.g., HiAgent's subgoal tendencies). The adversarial loop actively searches for failure modes, forcing robustness. We use the execution group as arbiter rather than a third party because they have firsthand knowledge, reducing new bias. We limit iterations K to cap cost (approximately K × (|E|+2) API calls per task; |E|=3, so ~15 calls per task), accepting that some valid experiences may be discarded if G fails within budget. We update G with in-context learning from refutations (no gradients), avoiding training overhead while leveraging the LLM's ability to incorporate examples.

### (D) Why it measures what we claim
The computational quantity "consensus vote on refutation validity" measures robustness against single-agent biases because the critic's counterexample represents a failure case a biased agent might miss; if consensus validates the refutation, the experience is flawed. This measures robustness under the assumption that the critic's search space covers all possible failure modes. To verify this, we calibrate the critic by measuring its refutation rate on a set of 512 intentionally flawed experiences (hand-crafted by injecting common biases). We require the critic to refute at least 90% of these to ensure adequate coverage. If the critic fails this calibration, we replace it with a stronger model (e.g., GPT-4). The generator's proposal e is a candidate distilled experience; its survival after critic attacks indicates it is not easily refuted, measuring resistance to the critic's specific bias. The critic's counterexample c is a directed attempt to violate e; its generation measures the completeness of the critic's coverage; the assumption that a valid refutation exists if e is flawed fails if the critic is too weak to find it, in which case acceptance may occur despite underlying biases. The calibration step mitigates this risk.

## Contribution

(1) A novel adversarial adaptation of the EDV paradigm that replaces passive distillation with a generator-critic game, structurally mitigating single-agent biases. (2) A verification protocol that uses the execution group to adjudicate adversarial refutations, ensuring that accepted experiences are robust to directed counterexamples. (3) An in-context learning mechanism for the generator that leverages successful refutations to improve future proposals without gradient updates.

## Experiment

### Evaluation Setup
| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | 50 problems from AIME (2020-2023, subset) | Tests robust reasoning under varied solution paths |
| Primary metric | Pass rate@1 (average over 5 random seeds) | Measures correct final answer proportion |
| Baseline 1 | Base LLM (zero-shot) | No experience learning; naive performance |
| Baseline 2 | EDV (prior work) | Prior distillation; single-agent bias |
| Baseline 3 | FLEX (continual learning) | Continuous learning from experience |
| Ablation 1 | ADV w/o Critic | Generator only; no adversarial check |
| Ablation 2 | ADV with Weak Critic (GPT-3.5) | Tests impact of critic strength |

### Why this setup validates the claim
AIME problems require multi-step reasoning where solution paths vary, making them ideal to test robustness against biased experiences. Comparing to Base LLM isolates the effect of experience learning. EDV tests whether adversarial refutation improves over passive distillation. FLEX provides a strong baseline from another experience learning paradigm. The ablation w/o Critic directly measures the adversarial component's contribution; the weak critic ablation tests sensitivity to critic strength. Pass rate@1 is a clear, objective metric for solution correctness, directly reflecting the quality of learned experiences.

### Expected outcome and causal chain
**vs. Base LLM** — On a problem requiring a subgoal decomposition not present in the base model's priors, Base LLM fails because it lacks relevant experiences. Our method learns from past tasks and distills robust subgoals. We expect ADV to achieve 20-30% higher pass rate than Base LLM overall.

**vs. EDV** — On a problem where the distillation agent's bias favors a suboptimal solution approach (e.g., always algebraic), EDV's experience may be flawed, causing failures on problems needing geometric insight. ADV's critic actively searches for counterexamples; if a biased experience is proposed, the critic finds a case where it fails, forcing refinement. We expect ADV to outperform EDV by 10-15% on problems requiring diverse strategies.

**vs. FLEX** — FLEX learns from its own trajectories but lacks external validation of experiences. On a problem where a subtle pitfall is encountered (e.g., division by zero), FLEX might incorporate a flawed experience if it avoids the pitfall in its own trials. ADV's adversarial loop exposes such pitfalls via the critic, preventing acceptance. We expect ADV to have higher consistency (lower variance) across problems, with a 5-10% advantage on edge-case problems.

**Ablation: ADV w/o Critic vs. ADV** — Without the critic, the generator may propose biased experiences that are accepted without challenge. We expect ADV (full) to outperform w/o Critic by at least 10%, especially on problems where the generator's bias is strong.

**Ablation: Weak Critic vs. Strong Critic** — A weak critic may miss refutations, leading to acceptance of flawed experiences. We expect ADV with strong critic (GPT-4) to outperform weak critic (GPT-3.5) by 5-10%, validating the importance of critic coverage.

### What would falsify this idea
If ADV w/o Critic matches ADV's performance, the adversarial loop provides no benefit. Alternatively, if ADV underperforms EDV on the majority of problems, the adversarial process introduces harmful noise rather than robust verification. If the weak critic ablation achieves similar performance to the strong critic, the critic's calibration is unnecessary.

## References

1. Escaping the Self-Confirmation Trap: An Execute-Distill-Verify Paradigm for Agentic Experience Learning
2. FLEX: Continuous Agent Evolution via Forward Learning from Experience
3. Multi-Agent Evolve: LLM Self-Improve through Co-evolution
4. Scalable Reinforcement Post-Training Beyond Static Human Prompts: Evolving Alignment via Asymmetric Self-Play
5. From Generation to Judgment: Opportunities and Challenges of LLM-as-a-judge
6. AutoDefense: Multi-Agent LLM Defense against Jailbreak Attacks
7. A Survey on LLM-as-a-Judge
8. Revolve: Optimizing AI Systems by Tracking Response Evolution in Textual Optimization
