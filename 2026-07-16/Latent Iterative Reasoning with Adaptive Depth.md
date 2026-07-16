# Latent Iterative Reasoning with Adaptive Depth

## Motivation

Existing RL reasoning methods like Ring-Zero generate fixed-length token chains, wasting compute on easy problems and truncating hard ones because they lack adaptive depth. This structural limitation stems from autoregressive decoding that ties reasoning depth to output length, preventing dynamic allocation of compute.

## Key Insight

By performing iterative refinement in a continuous latent space, we can decouple reasoning steps from token generation and use a stopping criterion based on latent state stability, enabling compute-efficient adaptive depth.

## Method

(A) **What it is**: LIRAD (Latent Iterative Reasoning with Adaptive Depth) trains a model to iteratively update a latent state vector using a learned transition function, halting when the state changes below a threshold, then decoding the final answer from the stable latent state.

(B) **How it works**:
```python
# Input: problem prompt p, encoder E, transition f, decoder g, max_iter T_max, threshold tau
z = E(p)  # initial latent state
for t in range(1, T_max+1):
    z_new = f(z, p)  # iterative refinement
    delta = ||z_new - z||  # stability measure
    z = z_new
    if delta < tau:
        break
answer = g(z)  # decode from stable latent
# Training: RL with verifiable reward R, plus per-step cost c
# Loss = -E[R] + lambda * (number of steps) * c
```

(C) **Why this design**: We chose iterative latent updates over autoregressive token generation because the continuous latent space allows fine-grained control over reasoning depth without generating intermediate tokens, which reduces sequence length and compute waste. We chose a Euclidean distance-based halting criterion over a learned halting network because it is parameter-free and avoids additional training complexity, accepting the cost that it may not capture semantic convergence if the latent space is poorly structured. We chose a fixed maximum iterations T_max to bound worst-case compute, a trade-off that ensures predictable resource usage at the risk of truncating very hard problems. The per-step cost c in the RL objective directly penalizes unnecessary iterations, encouraging the model to converge quickly without sacrificing correctness—this is a design choice that aligns the reward with compute efficiency, rather than relying solely on accuracy.

(D) **Why it measures what we claim**: The stability delta ||z_t - z_{t-1}|| measures reasoning convergence because we assume that latent states encoding similar answer content are close in Euclidean space due to the continuity of the latent representation; this assumption fails when the latent space is under-trained or collapses, in which case delta may reflect representation drift rather than answer convergence. The iteration count measures computational cost because each f call is a constant-complexity forward pass, making the total cost proportional to steps; this assumption fails if f has dynamic computation per step (e.g., early layers are skipped). The halting policy operationalizes adaptive depth by stopping when further updates yield minimal change, assuming diminishing returns—this assumption fails when the latent space has plateaus that are not indicative of answer convergence (e.g., local minima), in which case the model may halt prematurely.

## Contribution

(1) A novel training framework for latent iterative reasoning with adaptive depth using a stability-based halting criterion. (2) A design principle showing that decoupling reasoning depth from output length improves compute efficiency without sacrificing accuracy. (3) An empirical finding that latent iterative refinement enables variable-depth reasoning across problem difficulties.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | GSM8K | Tests multi-step math reasoning |
| Primary metric | Accuracy per compute | Measures efficiency of reasoning |
| Baseline 1 | Ring-Zero model | Naive scaling baseline |
| Baseline 2 | ProRL model | Extended training baseline |
| Baseline 3 | GSPO model | Group policy baseline |
| Ablation-of-ours | LIRAD without cost | Isolates cost penalty effect |

### Why this setup validates the claim

This combination directly tests the central claim that iterative latent updates with adaptive depth improve reasoning efficiency. The GSM8K dataset requires multiple reasoning steps, making it ideal to measure both accuracy and compute cost. Ring-Zero serves as a baseline to test whether scaling alone can match our method's efficiency; if it cannot, it supports our adaptive depth mechanism. ProRL tests whether prolonged token-level generation is necessary; if our method outperforms it, then latent iterations are more compute-efficient. GSPO tests group-level policy optimization; if our method surpasses it, then per-step cost penalty is critical. The ablation (no cost penalty) isolates the effect of explicit compute regularization. Accuracy per compute captures the trade-off directly, providing a falsifiable test: if our method does not achieve higher accuracy per compute than all baselines, the claim is unsupported.

### Expected outcome and causal chain

**vs. Ring-Zero model** — On a problem requiring many reasoning steps (e.g., multi-step equation), the Ring-Zero model uses a fixed, large number of forward passes per token, wasting compute on irrelevant intermediate tokens. Its accuracy may be high but at high compute cost. Our method iteratively refines a latent representation, halting when stable, thus using fewer total passes. We expect LIRAD to achieve comparable accuracy with significantly fewer compute units (e.g., 30% fewer steps) on hard problems, while matching on easy ones.

**vs. ProRL model** — On a problem with a long chain of reasoning, ProRL generates many tokens, each requiring a forward pass, leading to high compute. Prolonged training may improve accuracy but does not address per-step inefficiency. Our method condenses reasoning into latent iterations, which are more compact. We expect LIRAD to show higher accuracy per compute, especially on problems where token generation is verbose (e.g., arithmetic with large numbers). The gap should widen with problem complexity.

**vs. GSPO model** — On a problem where early correct reasoning steps are followed by later mistakes, GSPO's group-level policy may not penalize excessive steps, leading to wasted compute. Our method's per-step cost penalty directly discourages unnecessary iterations, so it should converge faster. We expect LIRAD to have lower variance in number of steps across problems, and higher accuracy per compute on problems where the baseline would over-reason.

### What would falsify this idea

If LIRAD's accuracy per compute is not consistently higher than all baselines across difficulty levels, or if its advantage is uniform rather than concentrated on problems requiring deep reasoning, then the central claim of improved efficiency via adaptive depth is falsified.

## References

1. Ring-Zero: Scaling Zero RL to a Trillion Parameters for Emergent Reasoning
2. ProRL: Prolonged Reinforcement Learning Expands Reasoning Boundaries in Large Language Models
3. Group Sequence Policy Optimization
4. Every Step Evolves: Scaling Reinforcement Learning for Trillion-Scale Thinking Model
5. Pass@k Training for Adaptively Balancing Exploration and Exploitation of Large Reasoning Models
6. Rethinking Sample Polarity in Reinforcement Learning with Verifiable Rewards
7. AReaL: A Large-Scale Asynchronous Reinforcement Learning System for Language Reasoning
8. DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models
