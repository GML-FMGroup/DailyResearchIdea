# ConsistRL: Self-Supervised Consistency Rewards for Robust Reinforcement Learning on Reasoning Traces

## Motivation

Existing RL-for-reasoning methods, such as the chess-based study in Understanding Reasoning from Pretraining to Post-Training, rely on verifiable outcomes (e.g., checkmate status) that are unavailable in real-world noisy reasoning traces. This structural reliance on clean, verifiable domains prevents application to ambiguous or unverifiable reasoning tasks. The root cause is that the reward signal requires an external oracle, rather than being derived from the reasoning process itself.

## Key Insight

Reasoning correctness is structurally invariant under input perturbations that preserve logical content, so aligning model reasoning across multiple perturbed versions of the same problem yields a self-supervised reward that does not require verifiable outcomes.

## Method

**(A) What it is**
ConsistRL is a reinforcement learning algorithm that replaces the standard verifiable-outcome reward with a self-supervised consistency reward. The reward is computed by generating a set of perturbed versions of the input problem, running the current policy to obtain reasoning traces and final answers for each perturbation, and rewarding the policy when all perturbed traces produce the same final answer.

**(B) How it works**
```pseudocode
Input: policy π_θ, perturbation function P (preserves logical content), number of perturbations K, RL algorithm (PPO)
for each iteration do
  Sample input x from environment
  Generate K perturbed versions: {x_i = P(x) for i=1..K}
  For each x_i, sample reasoning trace t_i ~ π_θ(·|x_i) and final answer a_i
  Compute consistency reward: r = 1 if all a_i are identical else 0
  // Optionally use fine-grained trace similarity; we use hard agreement for simplicity
  Update policy using PPO with reward r on the original input x (or on average of perturbed states) to maximize expected consistency
end for
```
Hyperparameters: K=5, PPO clip epsilon=0.2, learning rate=1e-5.

**(C) Why this design**
We chose hard binary reward over soft agreement (e.g., fraction of matching answers) because soft rewards can be gamed by the policy producing plausible but incorrect reasoning that happens to agree on a subset of perturbations; binary reward forces all perturbations to align, creating a stronger signal for true invariance. We selected PPO over REINFORCE for its stability and sample efficiency, accepting the cost of higher per-step computation. We set K=5 as a trade-off between perturbation diversity and computational budget: smaller K risks spurious agreement on wrong answers, larger K increases generation cost. The perturbation function P is implemented as a lightweight paraphrasing model (e.g., T5-small) to alter surface form while preserving answer semantics; we chose a learned model over rule-based heuristics because rule-based methods may fail on complex natural language, accepting the risk of occasional semantic drift in the perturbation.

**(D) Why it measures what we claim**
The computational quantity `all a_i identical` measures `reasoning correctness` under the assumption that the perturbation function P preserves the logical content of the problem while varying surface form; this assumption fails when P accidentally changes the underlying reasoning required (e.g., adding a negation), in which case agreement reflects spurious invariance to the perturbed logic rather than correct reasoning. The quantity `number of perturbations K` measures `robustness to input variation` under the assumption that for a correct reasoning trace, the final answer should be invariant to any surface-form change that does not alter logical content; this assumption fails when the problem itself is ambiguous or multiple correct answers exist, in which case consistency may penalize legitimate alternative solutions.

## Contribution

(1) ConsistRL, a self-supervised RL framework that generates reward signals from consistency across perturbed inputs, eliminating the need for verifiable outcomes or clean reasoning traces. (2) A demonstration that reasoning correctness can be proxied by input invariance, enabling RL for reasoning in noisy or ambiguous domains where prior methods fail.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | GSM8K (math word problems) | Tests multi-step reasoning invariance |
| Primary metric | Accuracy on test set | Measures final answer correctness |
| Baseline 1 | Standard RL (outcome reward) | No consistency regularization |
| Baseline 2 | Supervised fine-tuning only | Baseline without RL |
| Ablation | ConsistRL with soft agreement reward | Tests hard vs soft reward signal |

### Why this setup validates the claim

GSM8K requires multi-step arithmetic where problems admit many surface-form paraphrases that do not alter the underlying logic. Standard RL baseline isolates the effect of the consistency reward over conventional outcome-based RL; SFT baseline tests whether the RL phase is even necessary. The ablation with soft agreement separates the role of binary hard reward from a graded consistency signal. Accuracy directly measures the final answer, which is the target of reasoning correctness. If ConsistRL outperforms both baselines on paraphrased or compositionally novel problems, it supports the claim that enforcing answer invariance across perturbations yields more robust reasoning. Conversely, if gains are uniform or absent, the claim is falsified.

### Expected outcome and causal chain

**vs. Standard RL** — On a case where the problem is paraphrased (e.g., "John has 5 apples, gives 2" vs "John gives away 2 of his 5 apples"), standard RL may overfit to specific token sequences, producing correct answer on one phrasing but wrong on another. Our method forces agreement across 5 perturbations, so the policy learns invariant reasoning. We expect a noticeable gap on paraphrased subsets (e.g., 10–15% higher accuracy) while similar on canonical forms.

**vs. Supervised fine-tuning** — On a case requiring a novel composition of operations unseen in training (e.g., multi-step addition with subtraction in a new order), SFT may fail because it memorizes patterns rather than learning generalizable reasoning. Our RL with consistency reward encourages stable traces across perturbations, leading to better generalization. We expect a larger gap on out-of-distribution harder problems (e.g., 20% higher accuracy) but smaller on simple ones.

**vs. ConsistRL with soft agreement** — On a case where a wrong answer appears on 3 out of 5 perturbations, soft reward (e.g., 0.4) still provides mild positive signal, allowing the policy to converge to a plausible wrong answer that agrees on a majority. Our hard reward (0 only if all match) prevents this. We expect our method to have higher overall accuracy due to fewer consistent wrong answers, while soft agreement may show higher variance and occasional collapse to majority-wrong states.

### What would falsify this idea

If ConsistRL shows uniform accuracy gains across all problem types rather than concentrated gains on problems with high surface-form variation or compositional complexity, the central claim that consistency drives invariance is unsupported. Similarly, if the soft agreement ablation performs comparably to hard reward, the necessity of binary signal is falsified.

## References

1. Understanding Reasoning from Pretraining to Post-Training
2. On the Interplay of Pre-Training, Mid-Training, and RL on Reasoning Language Models
3. Front-Loading Reasoning: The Synergy between Pretraining and Post-Training Data
4. The Art of Scaling Reinforcement Learning Compute for LLMs
5. Instruction Pre-Training: Language Models are Supervised Multitask Learners
6. Resolving Discrepancies in Compute-Optimal Scaling of Language Models
7. Observational Scaling Laws and the Predictability of Language Model Performance
8. Asynchronous RLHF: Faster and More Efficient Off-Policy RL for Language Models
