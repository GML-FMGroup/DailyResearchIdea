# Self-Consistency Driven Autotelic Agent: Learning Step-Level Credit from Internal Predictive Consistency

## Motivation

Both SiriuS and Darwin Godel Machine rely on a single external scalar signal (success/failure or reward) to drive self-improvement, which limits open-endedness and prevents credit assignment at the step level. Without step-level credit, the agent cannot learn from intermediate actions, leading to inefficient exploration and an inability to autonomously define internal objectives.

## Key Insight

By using the agent's own predictive model to evaluate the consistency of its trajectory with self-generated goals, the agent can derive an intrinsic reward signal that is both step-level and avoids the need for external evaluation.

## Method

**SCDA: Self-Consistency Driven Autotelic Agent**

(A) SCDA is a self-improving agent that generates candidate goals from its own experience and uses a learned predictive model to assign step-level consistency scores, which serve as intrinsic rewards. Inputs are environment observations and past trajectories; outputs are actions and updated models.

(B) The algorithm:
```pseudocode
Initialize policy π (2-layer MLP, hidden=256, tanh output for action mean, log std separate), predictive model M (probabilistic: outputs latent mean and log variance, 2-layer MLP hidden=256, GeLU), goal generator G (evolution: elite fraction 0.1, mutation noise σ=0.1), encoder φ (2-layer MLP hidden=256, output=64, GeLU). Experience buffer E (size 1e5).
for episode = 1 to N:
  reset environment, observe initial state s0
  latent state z0 = φ(s0)
  sample a goal g from G conditioned on z0 (g is a latent vector of dimension 64)
  trajectory = []
  for t = 1 to T:
    sample action a_t ~ π(z_t, g)  # π outputs Gaussian, action sampled via reparameterization
    execute a_t, observe s_{t+1}
    latent state z_{t+1} = φ(s_{t+1})
    compute predicted distribution: (μ_{t+1}, logσ²_{t+1}) = M(z_t, a_t)
    compute negative log-likelihood: nll = 0.5 * sum( ( (z_{t+1} - μ_{t+1})^2 / exp(logσ²_{t+1}) + logσ²_{t+1} + log(2π) ) )
    compute consistency c_t = exp(-nll / 64)  # scaled to [0,1] approximately; higher means more consistent
    store (z_t, a_t, g, c_t, z_{t+1}, μ_{t+1}, logσ²_{t+1}) in E
    trajectory.append((z_t, a_t, z_{t+1}, c_t))
  end
  # Update:
  # Update M and φ jointly by minimizing NLL on all (z_t, a_t, z_{t+1}) pairs (Adam, lr 1e-3)
  # Update π using REINFORCE with returns computed from cumulative c_t (discount 0.95), value baseline (2-layer MLP hidden=256) trained with MSE loss (Adam, lr 3e-4)
  # Update G via evolution: maintain population of 100 goals. After episode, compute sum(c_t) as fitness. Select top 10% (10 goals). Generate new goals by adding Gaussian noise (σ=0.1) to elites, also keep elites. Sample goals for next episode proportionally to fitness.
end
```

(C) Why this design: We chose to use a probabilistic predictive model M rather than a deterministic one because it allows us to compute aleatoric uncertainty and downweight consistency in stochastic states, mitigating the stochasticity trap. We selected NLL-based consistency in a learned latent space over raw state similarity because the NLL naturally accounts for prediction uncertainty, and the latent space captures task-relevant features while ignoring irrelevant variations, but this introduces a dependency on the quality of the encoder. We opted to update the goal generator G via evolution instead of gradient-based methods because the space of goals is discrete and high-dimensional, and evolution can explore diverse goals without requiring a differentiable reward, at the cost of sample efficiency. We use intrinsic rewards c_t directly as step-level credit rather than a sparse end-of-episode reward because it provides denser feedback, but may risk overfitting to consistency rather than actual task performance if not balanced.

(D) Why it measures what we claim: The step consistency c_t = exp(-NLL/64) measures the agent's ability to accurately predict the next latent state while accounting for aleatoric uncertainty. This serves as a proxy for the agent's understanding of its own dynamics; this assumption holds when the predictive model is well-calibrated on the agent's policy distribution, but fails when the model is poorly trained or the state space is inherently unpredictable (e.g., noisy TV), in which case c_t may be low even for competent actions, or high for trivial actions that produce low-variance predictions (e.g., no movement). To mitigate this, we use a probabilistic model that downweights stochastic states, but degeneracy can still occur if the goal generator selects trivial goals that lead to high consistency without task progress. Therefore, c_t measures step-level credit only under the assumption that predictive consistency correlates with task-relevant progress; this assumption is load-bearing and we test it via ablations.

## Contribution

(1) A novel framework, Self-Consistency Driven Autotelic Agent (SCDA), that replaces external rewards with an intrinsic consistency signal derived from a learned predictive model and self-generated goals. (2) A step-level credit assignment mechanism that uses predictive consistency to assign per-timestep intrinsic rewards, enabling fine-grained learning without external supervision. (3) Design principles for integrating goal generation, predictive modeling, and policy improvement in a closed-loop self-improving system.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | MiniGrid multi-room navigation | Tests goal-directed exploration and self-consistency. |
| Primary metric | Average task success rate | Measures task completion, directly relevant to self-improvement. |
| Baseline 1 | Standard soft actor-critic (no intrinsic reward) | Isolates effect of consistency intrinsic reward. |
| Baseline 2 | SCDA with random goals (no evolution) | Tests importance of goal evolution. |
| Baseline 3 | SCDA with raw state cosine similarity | Tests necessity of learned latent space. |
| Ablation-of-ours | SCDA with sparse consistency (end-of-episode only) | Tests density of consistency feedback. |

### Why this setup validates the claim

The central claim of SCDA is that self-improvement emerges from step-level consistency rewards and evolutionary goal generation. This experimental design forms a falsifiable test: comparing against standard RL (no intrinsic reward) tests whether consistency alone provides a learning signal beyond external rewards. Comparing against random goals tests whether evolutionary selection drives meaningful exploration. Comparing against raw state similarity tests whether the learned latent representation filters irrelevant features. The ablation with sparse consistency tests whether dense step-level feedback is crucial. The primary metric, success rate, directly reflects task performance and improvement over time. On a complex multi-room navigation task requiring goal-directed exploration, if SCDA outperforms all baselines, it validates that self-consistency plus evolution enables autonomous improvement. Conversely, if baselines match SCDA, the core assumptions are falsified.

### Expected outcome and causal chain

**vs. Standard RL** — On a long-corridor task where external rewards are sparse, the baseline agent fails to learn because it receives no signal until the goal. Our method provides a dense consistency reward at each step, allowing the agent to build a predictive model and explore effectively. Thus we expect a large gap in success rate (e.g., 20% vs 60%) on long-horizon tasks, but parity on short tasks where external rewards are sufficient.

**vs. SCDA with random goals** — On a task requiring discovery of a novel tool-use sequence, random goals sample many irrelevant objectives, wasting experience. Our evolution selects goals with high cumulative consistency, focusing on promising subspaces. Hence we expect faster convergence and higher final performance (e.g., 70% vs 40% success) because evolution drives the agent toward self-consistent challenges.

**vs. SCDA with raw state similarity** — On a task with visual distractions (e.g., changing wall colors), raw similarity penalizes harmless variations, confusing the agent. Our latent similarity ignores irrelevant features, so the agent learns more robust policies. We expect a 10-15% higher success rate on noisy variants, showing the benefit of learned latent representations.

**vs. SCDA with sparse consistency** — On a task requiring precise control (e.g., picking up an object), sparse end-of-episode consistency leads to high-variance gradients and slow learning. Our dense reward provides low-variance feedback, enabling smoother updates. We expect a 15% higher peak success and faster convergence (e.g., 65% vs 50%).

### What would falsify this idea
If SCDA's performance is not significantly better than standard RL on long-horizon tasks, or if random goals achieve comparable success to evolved goals on open-ended tasks, the central claims about consistency and evolution are falsified. Specifically, if the gap between dense and sparse consistency is negligible, the step-level reward hypothesis fails.

## References

1. SiriuS: Self-improving Multi-agent Systems via Bootstrapped Reasoning
2. Darwin Godel Machine: Open-Ended Evolution of Self-Improving Agents
3. Constitutional AI: Harmlessness from AI Feedback
4. LLM-POET: Evolving Complex Environments using Large Language Models
5. Evolution Gym: A Large-Scale Benchmark for Evolving Soft Robots
6. Quality-Diversity through AI Feedback
7. MarioGPT: Open-Ended Text2Level Generation through Large Language Models
8. Procedural content generation using neuroevolution and novelty search for diverse video game levels
