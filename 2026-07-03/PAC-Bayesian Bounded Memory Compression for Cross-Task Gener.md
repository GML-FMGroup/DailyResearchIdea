# PAC-Bayesian Bounded Memory Compression for Cross-Task Generalization in LLM Agents

## Motivation

Existing bounded memory testbeds such as AgenticSTS evaluate memory architectures only within a single game (Slay the Spire 2), assuming that performance generalizes across domains without empirical verification. This assumption is structurally unvalidated because no theoretical link exists between memory size and cross-task performance. We address this gap by deriving a PAC-Bayesian bound that explicitly ties compression rate to expected loss across multiple tasks, providing a principled optimization target for learning task-invariant memories.

## Key Insight

The PAC-Bayesian bound decomposes cross-task generalization error into a compression penalty and a task-identity mutual information term, so minimizing the bound naturally encourages memory representations that are both compact and task-agnostic.

## Method

(A) **What it is**: Cross-Task Invariant Memory Compression (CTIMC) is a framework that trains a bounded memory agent by minimizing a PAC-Bayesian upper bound on the expected loss across a family of training tasks. It takes as input a set of tasks with trajectories, and outputs a compressed memory encoder and a policy decoder.

(B) **How it works**:
```pseudocode
Algorithm CTIMC
Input: Task set T = {τ₁,...,τₘ}, each with interaction history H_τ
Hyperparameters: codebook size K, bound tightness λ, number of training tasks m
Output: Encoder E, decoder D, codebook C

1. Initialize VQ-VAE encoder E (maps history to code) and decoder D (maps code + current state to action).
2. For each task τ, collect trajectories by interacting with the environment using current policy (E and D). Store histories.
3. Encode each history h into a latent code z = E(h). Quantize z to nearest code in C.
4. Compute PAC-Bayesian bound:
   L_bound = (1/m) Σ_τ L_τ + λ * KL(Q(z|h) || P(z)) + (1/λ) * (1/2m) * (K log 2 + log(2m/δ))
   where L_τ is average loss on task τ, Q(z|h) is encoder posterior, P(z) is prior over codes (uniform).
5. Update E, D, C to minimize L_bound via gradient descent.
6. Repeat steps 2-5 until convergence.
```

(C) **Why this design**: We chose VQ-VAE over continuous latent variables because discrete codes enforce a hard memory budget (K codes), directly linking compression to performance; the trade-off is that quantization may lose some task-specific details, but the bound compensates by encouraging task-invariant features. We use a uniform prior P(z) to maximize entropy of the code distribution, which promotes diversity and prevents collapse; accepting that a non-informative prior may not capture true task statistics, but the KL penalty still regularizes the encoder. We adopt a PAC-Bayesian bound over an empirical risk minimization objective because it provides a finite-sample generalization guarantee without requiring separate validation sets; the cost is that the bound may be loose for small m, but it remains a valid upper bound that can be minimized.

(D) **Why it measures what we claim**: The KL divergence KL(Q(z|h) || P(z)) measures task-invariance of the memory representation because it quantifies how much the code distribution varies across histories; under the assumption that a task-invariant representation should have similar code distributions across tasks, a small KL indicates that the encoder maps histories from different tasks to similar codes; this assumption fails when tasks share surface features but differ in underlying structure (e.g., same environment with different reward functions), in which case the KL might be low even though the representation is not truly task-invariant. The empirical loss L_τ measures within-task performance, and the bound combines both to ensure that compression does not sacrifice task-specific utility. The codebook size K measures memory capacity, and the bound's dependency on K reflects the trade-off between expressiveness and generalization; under the assumption that larger K allows more task-specific details, the bound penalizes large K to avoid overfitting, but this assumption fails when tasks require high-capacity memories (e.g., long-term dependencies), in which case the bound might unduly restrict performance.

## Contribution

(1) A PAC-Bayesian bound that explicitly connects bounded memory compression to cross-task generalization error, providing a theoretical justification for single-domain evaluation assumptions. (2) CTIMC, a practical algorithm that optimizes this bound to learn task-invariant memory representations without requiring explicit task labels during deployment. (3) An analysis of the trade-off between memory size and cross-task transfer, formalized via the bound's dependency on codebook size.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | DSGBench | Diverse strategic games for generalization |
| Primary metric | Average win rate | Measures task performance across tasks |
| Baseline 1 | No memory | Baseline with zero memory |
| Baseline 2 | Full history | Baseline with unbounded memory |
| Baseline 3 | Continuous VAE | Continuous latent variable baseline |
| Ablation-of-ours | CTIMC large codebook | Test effect of memory capacity |

### Why this setup validates the claim
DSGBench includes multiple strategic games with distinct dynamics, enabling a test of cross-task generalization. The no-memory baseline establishes a lower bound for tasks requiring memory, while the full-history baseline shows the risk of overfitting to irrelevant details. Continuous VAE isolates the effect of discrete codes, and the large-codebook ablation tests the impact of memory capacity. Average win rate directly reflects both within-task performance and generalization, making it a suitable metric to detect whether task-invariant compression improves overall ability. This combination creates a falsifiable test: if our method outperforms all baselines consistently, the central claim is supported; if not, the breakdown by condition reveals the failure mode.

### Expected outcome and causal chain

**vs. No memory** — On a case where the agent must recall a past opponent move to counter a strategy, the no-memory baseline repeats the same mistake because it cannot store information across steps. Our method compresses that history into a discrete code, enabling recall without storing the full trajectory. We expect a noticeable gap on such memory-dependent tasks, but parity on tasks solvable without memory.

**vs. Full history** — On a case where the environment has many irrelevant details (e.g., random background events), full history overfits to specific sequences, leading to poor generalization on new instances. Our method discards task-specific details via compression, focusing on invariant patterns. We expect our method to outperform on tasks with high variability across episodes, but possibly equal on deterministic tasks.

**vs. Continuous VAE** — On a case where the agent needs to categorize opponents into discrete types (e.g., aggressive vs. passive), the continuous VAE may blur category boundaries, causing suboptimal actions. Our discrete codes form clean clusters, allowing crisp policy selection. We expect our method to have higher win rates on tasks requiring categorization, but similar performance on tasks where continuous features suffice.

### What would falsify this idea
If our method performs similarly to no memory on tasks that clearly demand memory, or if its advantage over continuous VAE is absent on categorization-heavy tasks, the central claim that discrete task-invariant compression is key would be falsified.

## References

1. AgenticSTS: A Bounded-Memory Testbed for Long-Horizon LLM Agents
2. DYSTIL: Dynamic Strategy Induction with Large Language Models for Reinforcement Learning
3. DSGBench: A Diverse Strategic Game Benchmark for Evaluating LLM-based Agents in Complex Decision-Making Environments
4. UrzaGPT: LoRA-Tuned Large Language Models for Card Selection in Collectible Card Games
5. The Illusion of Diminishing Returns: Measuring Long Horizon Execution in LLMs
6. ExpeL: LLM Agents Are Experiential Learners
7. Large Language Models can Learn Rules
8. GameBench: Evaluating Strategic Reasoning Abilities of LLM Agents
