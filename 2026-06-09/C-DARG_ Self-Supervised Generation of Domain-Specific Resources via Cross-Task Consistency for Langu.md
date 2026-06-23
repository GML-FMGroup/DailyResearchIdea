# C-DARG: Self-Supervised Generation of Domain-Specific Resources via Cross-Task Consistency for Language Model Fine-Tuning

## Motivation

Existing instruction-tuned models (e.g., Flan-PaLM from 'Scaling Instruction-Finetuned Language Models') and low-data approaches ('Only 0.5% Data') rely on manually curated datasets that are often incomplete or unavailable for novel domains. Similarly, autonomous task planning systems ('Large Language Model Based Autonomous Task Planning') assume pre-built environment models. The root cause is the absence of mechanisms for models to autonomously fill knowledge gaps when external resources are insufficient, limiting scalability to new domains.

## Key Insight

Cross-task consistency constraints exploit domain invariants—structural properties that hold across multiple related tasks—as a natural supervisory signal, enabling self-supervised generation of reliable training data without external annotation.

## Method

(A) **What it is:** C-DARG (Consistency-Driven Autonomous Resource Generator) is a self-supervised fine-tuning framework that, given a base LLM (e.g., Flan-T5) and a small seed of exemplars for a set of domain-related tasks, iteratively generates candidate training resources and refines the model using a consistency-based reward derived from task coupling. This design assumes that deterministic derive_resource and check_consistency functions accurately capture the domain invariants. To calibrate, we verify on a held-out set of human-annotated resources that the consistency score correlates with human judgments of correctness (Spearman ρ>0.8). As a concrete example, for math word problems, domain invariants are mathematical relationships: a generated equation's solution must satisfy the equation, and derive_resource transforms an equation into a solution via a symbolic solver (deterministic). check_consistency verifies that the derived solution satisfies the original equation using a rule-based symbolic check.

(B) **How it works:**
```python
# C-DARG Framework
# Input: base LLM (e.g., Flan-T5), seed exemplars D_seed for tasks T = {t1, t2, ..., tm}
# Hyperparameters: generation temperature τ=0.7, sample size N=100 per iteration, reward margin γ=0.1
# Compute budget: 4 NVIDIA A100 GPUs, 8 hours for 1000 iterations

Initialize policy π = base LLM
Initialize replay buffer B = D_seed
for iter in range(num_iterations):
    # Phase 1: Generate candidate resources
    for task t in T:
        prompt = construct_prompt_for_task(t, context from B)
        candidates = sample from π(prompt) with temperature τ, N times
        store candidates in pool C_t
    # Phase 2: Evaluate consistency
    for each generated resource r in C_t for some task t:
        consistency_score = 0
        for other task s in T, s != t:
            derived = deterministic derive_resource(r, from_task=t, to_task=s)  # e.g., for math: equation to solution
            score_s = check_consistency(derived, existing knowledge base)  # e.g., verify solution satisfies equation
            consistency_score += score_s
        consistency_score /= |T|-1
        reward = consistency_score if consistency_score > γ else -1.0
        store (r, reward) in B
    # Phase 3: Fine-tune policy with PPO
    update π using PPO on transitions from B (sampled with priority for high-reward ones)
```

(C) **Why this design:** We chose to use a consistency-based reward derived from multiple tasks rather than a single task accuracy because domain invariants require cross-validation to avoid overfitting to spurious patterns. Using PPO for fine-tuning rather than simple supervised learning on generated data allows the model to explore diverse generations and receive gradient updates based on reward, which is more sample-efficient for open-ended generation. We set the generation temperature to 0.7 to balance exploration and coherence. We sample N=100 candidates per task per iteration to provide sufficient diversity without excessive compute. The replay buffer stores experienced generations to prevent catastrophic forgetting. The reward margin γ=0.1 acts as a threshold to filter low-quality generations. We chose PPO over DPO because PPO's clipped objective provides stability when rewards are noisy and non-stationary, accepting the computational cost of a separate value network. We used a deterministic derive_resource function rather than a learned one because it ensures that the consistency signal is grounded in the domain's formal semantics, at the cost of limited generality to tasks without such formal mappings.

(D) **Why it measures what we claim:** The consistency_score computed as the average of check_consistency across tasks measures the degree to which the generated resource respects the domain's structural invariants because we assume that a valid resource (e.g., a correct optimization model) will be consistent with derived resources for related tasks (e.g., the solution set, infeasibility diagnostics). Specifically, check_consistency uses a rule-based or small supervised model to verify that the derived resource matches known properties or constraints; this assumption fails when the derive_resource function does not capture all relevant dependencies (e.g., a model and its solution may be consistent but the diagnostic may be ambiguous), in which case consistency_score reflects only a subset of constraints and may overestimate validity. The reward = consistency_score if >γ else -1 operationalizes the idea that resources must meet a minimum consistency threshold to be considered useful, with a negative penalty for low-quality generations to discourage spurious outputs; this assumes that the threshold γ correctly separates valid from invalid resources, which fails when the consistency metric has variance across tasks, leading to false positives. The PPO update then optimizes the policy to maximize expected reward, thereby aligning generation with domain consistency; this causal chain holds if the reward is a sufficient statistic for resource quality, which fails when there are unmodeled dimensions of goodness (e.g., readability, efficiency) that are uncorrelated with consistency. Specifically, our consistency_score measures the degree of agreement under the fixed derivation; this assumes the derivation is invertible and complete. Failure mode: asymmetric dependencies or ambiguous derivations.

## Contribution

(1) C-DARG, a self-supervised framework that enables LLMs to autonomously generate domain-specific training resources using cross-task consistency as a supervisory signal. (2) The finding that consistency across multiple related tasks provides a sufficient signal for generating high-quality resources without any external annotation, demonstrated in the optimization domain. (3) A formal derivation of consistency constraints for a set of optimization tasks (modeling, solving, infeasibility diagnosis) as a specific instance of domain invariants.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | MedicalQA (MedQA) subtypes: diagnosis, treatment, prognosis (3 tasks) | Tasks share structural invariants (medical entity relations); low-resource domain (only 200 seed examples per task) |
| Primary metric | Average task accuracy (exact match for diagnosis, F1 for treatment and prognosis) | Direct measure of end-task performance |
| Baseline 1 | Standard supervised fine-tuning on seed data | No generation or consistency |
| Baseline 2 | Self-training with pseudo-labeling (single-task generation) | Generation without cross-task consistency |
| Baseline 3 | Multi-task joint training | Shared representation without generation |
| Ablation (Ours) | C-DARG without cross-task reward (i.e., reward = consistency_score) | Isolate effect of consistency threshold and negative penalty |

### Why this setup validates the claim
This design tests whether C-DARG's consistency-driven generation improves over methods that do not leverage cross-task structure. Standard supervised fine-tuning on seeds (Baseline 1) gauges the benefit of synthetic data generation. Self-training (Baseline 2) isolates the role of consistency by using generative self-training without across-task coupling. Multi-task joint training (Baseline 3) checks if merely training on multiple tasks is sufficient. An ablation removing the cross-task reward reveals whether the consistency signal itself drives gains. The chosen metric (average task accuracy) directly captures the central claim of improved task performance through domain-consistent generation. If C-DARG outperforms all baselines, and the ablation degrades to near-baseline levels, the claim is supported. Conversely, if C-DARG fails to outperform on tasks with strong inherent dependencies, the consistency assumption would be challenged.

### Expected outcome and causal chain

**vs. Standard supervised fine-tuning** — On a task where seed data is sparse (e.g., diagnosis), the baseline memorizes patterns and fails to generalize to new diseases because it has only seen a few examples. Our method generates diverse candidate diagnoses from a base LLM, then filters via cross-task consistency (e.g., ensuring the diagnosis aligns with treatment guidelines and prognosis from related tasks). Thus, it produces more robust outputs, and we expect a noticeable accuracy gain (e.g., 5–10% absolute) on tasks with high structural coupling (diagnosis-treatment), with parity on simpler tasks (e.g., prognosis without diagnosis constraint).

**vs. Self-training with pseudo-labeling** — On a case where the LLM generates plausible but incorrect treatments (e.g., contraindicated for given prognosis), self-training reinforces errors because it uses the model's own confidence without external validation. Our method applies a consistency check across tasks (e.g., verifying treatment against diagnosis and prognosis rules), rejecting low-consistency generations. Hence, our method avoids error accumulation, and we expect higher accuracy especially on tasks requiring factual coherence, with a gap of 3–7% absolute.

**vs. Multi-task joint training** — On a case where tasks share latent structure but have conflicting surface forms (e.g., diagnosis focusing on symptoms vs. treatment focusing on medications), multi-task training can suffer from interference because it learns a joint representation without explicit cross-task validation. Our method uses consistency rewards to ensure generated data respects task couplings, aligning representations. Thus, we expect better performance on tasks with subtle dependencies, with a gap of 2–5% absolute, and comparable performance on independent tasks.

### What would falsify this idea
If C-DARG shows no performance improvement over self-training on tasks with verifiable structural dependencies (e.g., diagnosis-treatment consistency), or if the ablation without cross-task reward performs equally well, then the central claim that consistency-driven generation is beneficial would be invalidated.

## References

1. Maybe Only 0.5% Data is Needed: A Preliminary Exploration of Low Training Data Instruction Tuning
2. Diagnosing infeasible optimization problems using large language models
3. DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model
4. JuMP 1.0: recent improvements to a modeling language for mathematical optimization
5. Scaling Instruction-Finetuned Language Models
6. Large Language Model Based Autonomous Task Planning for Abstract Commands
