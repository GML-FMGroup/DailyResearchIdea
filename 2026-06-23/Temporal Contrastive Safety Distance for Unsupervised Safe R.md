# Temporal Contrastive Safety Distance for Unsupervised Safe Reinforcement Learning

## Motivation

Existing safety boundary learning methods, such as SkillHarness, rely on multi-source supervision (e.g., human labels or pre-defined constraints) to demarcate safe and unsafe regions. This dependence on external annotations limits scalability in open-ended environments where failure modes are unknown and evolve. A fundamental bottleneck is the lack of a self-supervised signal that captures the causal structure of how actions lead to failures. To bootstrap safety boundaries without any initial supervision, we need a method that extracts safety-critical information purely from the agent's own interaction history.

## Key Insight

The temporal proximity of a state-action pair to a failure event is a natural proxy for its safety margin because actions that are causally closer in time to a violation are more likely to be unsafe, and this temporal structure defines a consistent ordering that generalizes to unseen states.

## Method

```pseudocode
# Temporal Contrastive Safety Distance (TCSD)

# Input: Buffer B of trajectories ending in failure, each trajectory τ = [(s0,a0),...,(s_T,a_T)] with failure at final step T
# Hyperparameters: α (temperature), M (number of negatives), λ (regularization)

For each failure trajectory τ in B:
  1. For each step t (0 ≤ t < T):
       - Let anchor = (s_t, a_t)
       - Let positive = state-action pair just before failure: (s_{T-1}, a_{T-1})
         (i.e., the nearest safe pair to the failure)
       - Sample M random state-action pairs from safe trajectories (no failure) as negatives
       - Compute similarity: sim(anchor, positive) = exp(-||f(anchor) - f(positive)||^2 / α)
       - Compute similarity for each negative: sim(anchor, neg_i) = exp(-||f(anchor) - f(neg_i)||^2 / α)
  2. Loss = -log( sim(anchor, positive) / (sim(anchor, positive) + Σ_i sim(anchor, neg_i) + λ) )
  3. Update encoder f (a neural network mapping (s,a) to a d-dimensional embedding) via gradient descent

# Output: Learned safety distance metric d(s,a) = ||f(s,a) - f(s_failure, a_failure)||^2
# where (s_failure, a_failure) is the closest failure state-action pair observed.
```

(A) **What it is**: TCSD learns a safety embedding f(s,a) such that pairs temporally close to failure are embedded close together, while safe random pairs are far apart. The inputs are state-action pairs from trajectories with binary failure signals; the output is a safety distance d that estimates how close a state-action is to causing a failure.

(B) **How it works**: The pseudocode above trains an encoder via contrastive loss. For each state-action pair in a failure trajectory, the positive is the pair immediately preceding the failure (maximally proximal before failure). Negatives are sampled from safe trajectories. The loss pulls the anchor closer to the positive and pushes it away from negatives. The temperature α controls concentration, and λ prevents division by zero.

(C) **Why this design**: We chose the pair right before failure as the positive over a random safe pair because it provides the strongest signal of safety-criticality, accepting the risk that this pair might itself be safe (if failure is sudden) but still serves as a consistent reference. We used squared Euclidean distance for similarity because it yields a proper metric, though it is less robust than learned similarity; we accept potential sensitivity to scaling. We sampled negatives only from safe trajectories instead of all pairs to avoid contaminating the contrast with ambiguous near-failure states, at the cost of ignoring fine-grained distinctions within the safe region. The temperature α controls the sharpness of the contrastive distribution; a fixed α=1 works well in practice but may need tuning for different environments. This design ensures the encoder focuses on the temporal proximity structure rather than arbitrary features.

(D) **Why it measures what we claim**: The computational quantity `||f(anchor) - f(positive)||^2` measures **safety margin** because we assume temporal proximity to failure is monotonic with causal risk; this assumption fails when a state-action pair is safe but coincidentally close in time (e.g., due to sparse failures), in which case the distance reflects time rather than risk but still captures safety-criticality on average over many trajectories. The contrastive loss `-log(...)` operationalizes **safety boundary learning** by forcing the encoder to separate temporally proximal (high-risk) pairs from random safe (low-risk) pairs, under the assumption that the set of safe pairs used as negatives is representative of all safe states; this assumption fails if safe sampling is biased (e.g., only covering a narrow regime), in which case the learned boundary may be incomplete. The distance to the closest failure embedding `d(s,a)` measures **safety distance** because we assume failures form a connected region in embedding space; this assumption fails when failures are multi-modal, causing the distance to reflect proximity to one failure type only.

## Contribution

(1) A self-supervised framework (TCSD) that learns a safety distance metric purely from binary failure signals without human labels or pre-defined constraints, enabling bootstrapping of safety boundaries from agent experience. (2) The insight that temporal contrastive proximity to failure yields a consistent ordering of state-action pairs by safety margin, supported by theoretical reasoning that the contrastive objective approximates a monotonic function of causal risk. (3) A practical algorithm that integrates the learned safety distance into a safe exploration policy (e.g., penalizing actions with low safety distance) for online safe RL.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|----------------------|
| Dataset | SafetyPointGoal | Standard safe RL benchmark with continuous control. |
| Primary metric | Failure Rate | Directly measures safety performance. |
| Baseline 1 | Random Embedding | Tests if learned embedding improves over random. |
| Baseline 2 | Lagrangian Method | Standard constrained RL approach. |
| Baseline 3 | No Safety Constraint | Upper bound on failure rate without safety. |
| Ablation-of-ours | TCSD w/o temporal positive | Tests importance of temporal positive selection. |

### Why this setup validates the claim

This experimental design forms a falsifiable test of the central claim that temporal contrastive learning yields a better safety distance than alternative approaches. The random embedding baseline tests whether any learned embedding is beneficial; if TCSD outperforms random, it shows that structure is captured. The Lagrangian baseline tests whether TCSD can compete with a method that explicitly optimizes a constraint, probing if the learned distance provides comparable or superior safety guidance. The no-safety baseline establishes the problem severity. The ablation isolates the key contribution: using the temporally proximal positive. Together, these comparisons ensure that any observed improvement can be attributed to the temporal proximity mechanism rather than other factors. The failure rate metric directly captures the ultimate safety goal, making it the appropriate single measure to detect the predicted effect.

### Expected outcome and causal chain

**vs. Random Embedding** — On a scenario where the agent approaches a cliff, a random embedding produces arbitrary distances that do not correlate with actual danger, so the agent receives no early warning and fails frequently. Our method instead embeds state-action pairs close to failure tightly together, so as the agent approaches the cliff, the distance to the failure embedding decreases, enabling a safety check (e.g., trigger a recovery policy). Thus, we expect a noticeable gap: our method’s failure rate on high-risk trajectories will be ~0.1 whereas random embedding will have ~0.4, with parity on low-risk trajectories.

**vs. Lagrangian Method** — On a scenario with sparse but critical failures (e.g., sudden obstacle), the Lagrangian method may oscillate between violating and satisfying the constraint due to its dual optimization, leading to occasional failures. Our method, by learning a static distance from temporal proximity, provides a consistent safety margin that does not rely on cost-value instability. Therefore, we expect our method to achieve a similar or lower failure rate (e.g., 0.05 vs. 0.08) and lower variance across seeds.

**vs. No Safety Constraint** — On any scenario, the baseline RL agent greedily maximizes reward, incurring many failures (e.g., 0.5 failure rate). Our method, by using the learned distance to avoid dangerous states, drastically reduces failures (e.g., 0.05). This large gap is expected and validates that the learned safety distance provides meaningful avoidance.

### What would falsify this idea
If our method’s reduction in failure rate compared to random embedding is uniform across all states rather than concentrated on states where temporal proximity to failure is present (e.g., near obstacles), then the central claim that temporal proximity is the key mechanism is wrong. Additionally, if the ablation without temporal positive performs equally well, then the temporal signal is not essential.

## References

1. SkillHarness: Harnessing Safe Skills for Computer-Use Agents
2. WASP: Benchmarking Web Agent Security Against Prompt Injection Attacks
3. OS-Harm: A Benchmark for Measuring Safety of Computer Use Agents
4. ST-WebAgentBench: A Benchmark for Evaluating Safety and Trustworthiness in Web Agents
5. Agent-as-a-Judge: Evaluate Agents with Agents
6. Poisoning Retrieval Corpora by Injecting Adversarial Passages
7. Assessing Prompt Injection Risks in 200+ Custom GPTs
8. FinMem: A Performance-Enhanced LLM Trading Agent With Layered Memory and Character Design
