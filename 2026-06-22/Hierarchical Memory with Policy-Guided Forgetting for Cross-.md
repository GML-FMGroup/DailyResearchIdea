# Hierarchical Memory with Policy-Guided Forgetting for Cross-Session Agent Persistence

## Motivation

Current agent benchmarks like GAIA and evaluation frameworks such as OAgents only test memory within short episodes, ignoring the need to retain task-relevant information across multiple sessions. OAgents’ modular memory design, while flexible, lacks a mechanism for long-term persistence because it assumes tasks are solvable within a few steps. This leads to memory overflow or irrelevant storage when agents operate over extended timelines. The root cause is the absence of a learned forget mechanism that compresses trajectories into hierarchical abstractions and selectively discards details based on future utility.

## Key Insight

Hierarchical compression with policy-gradient-trained forget gates allows an agent to automatically determine which level of detail to retain for maximizing cumulative task success across sessions, because the policy gradient directly optimizes the trade-off between memory capacity and future reward.

## Method

### (A) What it is
**Hierarchical Memory with Policy-Guided Forgetting (HMPF)** is a memory mechanism that compresses agent trajectories into a multi-level latent representation and learns, via REINFORCE with a learned state-dependent baseline (critic), a per-level forget signal that decides which abstraction level to keep for long-term storage.

**Inputs:** Trajectory of (state, action) pairs over one session; episode reward at session end.
**Outputs:** Compressed memory tensor; forget decisions sampled during training.

### (B) How it works
```python
class HMPF:
    def __init__(self, L=3, d=256):
        # Encoders for each level: 2-layer MLP (d -> 256 -> d) with GeLU
        self.encoders = [MLP(d, 256, d, activation='GeLU') for _ in range(L)]
        # Forget gate network per level: 2-layer MLP (d -> 64 -> 1) with sigmoid
        self.forget_gates = [MLP(d, 64, 1, activation='sigmoid') for _ in range(L)]
        # Critic network: 2-layer MLP (concatenated level representations -> 1) with no activation on output
        self.critic = MLP(L*d, 256, 1)  # input: mean over time of each level, flattened
        # Optimizer: Adam, lr=1e-3
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

    def encode_trajectory(self, trajectory):
        # Assume trajectory is list of feature vectors (e.g., from observation encoder) of dimension d
        # Level 0: raw features (list of vectors)
        h0 = [traj_feat for traj_feat in trajectory]
        # Level 1: mean pooling over chunks of size 4, then encoded
        h1 = [self.encoders[1](torch.mean(torch.stack(h0[i:i+4]), dim=0)) for i in range(0, len(h0), 4)]
        # Level 2: mean pooling over chunks of size 8, then encoded
        h2 = [self.encoders[2](torch.mean(torch.stack(h1[i:i+2]), dim=0)) for i in range(0, len(h1), 2)]
        return [h0, h1, h2]

    def sample_forget_decisions(self, levels):
        log_probs = []
        decisions = []
        state = torch.cat([torch.mean(torch.stack(lv), dim=0) for lv in levels], dim=0)
        for l, level in enumerate(levels):
            level_rep = torch.mean(torch.stack(level), dim=0)
            p = self.forget_gates[l](level_rep)  # scalar probability
            decision = torch.bernoulli(p)
            log_prob = decision * torch.log(p + 1e-8) + (1 - decision) * torch.log(1 - p + 1e-8)
            log_probs.append(log_prob)
            decisions.append(decision)
        return log_probs, decisions, state

    def compress(self, levels, decisions):
        # If decision=1 (forget), drop that level entirely and rely on higher level
        # Preserve the highest level always
        kept_levels = [levels[-1]]
        for l in reversed(range(len(levels)-1)):
            if decisions[l] == 0:  # remember: keep this level's details
                kept_levels.insert(0, levels[l])
        # Return concatenated summary of kept levels
        return torch.cat([torch.mean(torch.stack(lv), dim=0) for lv in kept_levels], dim=0)

    def update(self, reward, log_probs, state):
        # Learned baseline (critic)
        baseline = self.critic(state)
        advantage = reward - baseline.detach()
        # Policy loss
        policy_loss = -torch.sum(torch.stack(log_probs)) * advantage
        # Critic loss (MSE)
        critic_loss = (reward - baseline).pow(2).mean()
        loss = policy_loss + critic_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss
```

### (C) Why this design
We chose hierarchical compression over flat vector storage because hierarchical abstraction allows the agent to retain coarse structure (e.g., overall task progress) while forgetting fleeting pixel-level details, at the cost of additional computational overhead for multi-level encoding. We use REINFORCE instead of backpropagation-through-time for the forget gate because the optimal forget decision is a discrete choice that depends on future rewards, making it a reinforcement learning problem; the trade-off is higher variance in updates, which we mitigate with a learned state-dependent baseline (critic). We use separate forget gates per level rather than a single global gate because different abstraction levels capture different temporal scales (e.g., level 0: single-step features; level 2: multi-step patterns), and forgetting at one level should not force forgetting at another—this increases parameter count but prevents over-forgetting of fine-grained details that are occasionally useful. The top level is always retained to prevent catastrophic loss of high-level context.

### (D) Why it measures what we claim
The computational quantity `forget_prob` at level l, sampled as a Bernoulli decision, measures the *task-relevance* of that level’s features for future success because the policy gradient algorithm maximizes expected cumulative reward, and the forget decision determines whether those features are stored; this equivalence holds under the assumption that the reward signal is a sufficient statistic for feature utility—an assumption that fails when reward is sparse and temporally misaligned, in which case the forget gate may learn a noisy proxy or converge to a suboptimal policy. However, we explicitly verify this assumption by comparing learned forget decisions against oracle task-relevance labels (e.g., which objects are relevant) in the BabyAI environment. The hierarchical compression itself operationalizes *abstraction*: the lower the level, the more fine-grained the temporal resolution; thus, forgetting a lower level while retaining a higher one captures the concept of “selective forgetting”—retaining what matters at coarser granularity. The critic baseline ensures that the advantage signal is centered, making the gradient estimate consistent with the true long-term utility, but this consistency relies on the baseline being an unbiased estimate of the expected reward given the policy, which may not hold if the critic is poorly approximated—we mitigate this by using a simple MLP and training with MSE loss, and we monitor the correlation between critic predictions and true rewards.

**Load-bearing assumption (explicit):** The session-end reward provides a sufficient and unbiased signal for learning which hierarchical levels to forget via REINFORCE, enabling optimal memory for cross-session persistence. To verify, we measure the correlation between forget decisions and oracle task-relevance labels in BabyAI; if correlation <0.3, we flag possibility of reward insufficiency.

**Smallest method repair applied:** Replaced scalar moving-average baseline with a learned state-dependent baseline (critic network) to reduce variance; eligibility traces or off-policy corrections are not added to preserve simplicity.

**Number of parameters:** ~500K (encoders: 3*(256*256+256) ≈ 200K, forget gates: 3*(256*64+64) ≈ 50K, critic: 3*256*256+256 ≈ 200K). Training GPU hours: ~10 hours on a single RTX 3090 for BabyAI 50K episodes.

## Contribution

(1) A hierarchical memory compression mechanism with per-level policy-gradient-trained forget gates that enable selective forgetting of irrelevant details across sessions. (2) A demonstration that policy gradient can be used to learn which abstraction levels to retain for maximizing cumulative task success, providing a principled alternative to heuristic memory pruning. (3) A protocol for evaluating long-term memory persistence in agent benchmarks by extending short-horizon tasks to multi-session setups.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| ---- | ------ | --------------------- |
| Dataset | BabyAI (including cross-session variant: tasks split across episodes) | Tests hierarchical memory and cross-session retention. |
| Primary metric | Success rate (per session and cumulative across sessions) | Direct measure of task completion and persistence. |
| Baseline 1 | Flat Memory | No hierarchical compression. |
| Baseline 2 | Uniform Forgetting | Random forget decisions (same architecture). |
| Baseline 3 | No Forgetting | Stores all levels every time. |
| Baseline 4 | Differentiable Neural Computer (DNC) | State-of-the-art external memory with learned read/write. |
| Ablation 1 | HMPF w/o hierarchy (only top level kept) | Tests necessity of multi-level abstraction. |
| Ablation 2 | HMPF with oracle forget decisions (from env labels) | Upper bound on forget policy learning. |

### Why this setup validates the claim
This setup tests the central claim that hierarchical memory with policy-guided forgetting improves task success by selectively retaining task-relevant abstractions, especially across sessions where reward signals may be delayed. BabyAI provides tasks of varying lengths and complexity, requiring the agent to remember instructions over many steps. The cross-session variant explicitly tests whether learned forget decisions transfer compressed representations across episodes, enabling the agent to retain useful context from previous sessions. The Flat Memory baseline tests whether compression is beneficial; Uniform Forgetting tests whether learning the forget policy is crucial; No Forgetting tests whether forgetting is necessary at all; DNC benchmarks against a learned memory controller. The ablation removes the hierarchy to evaluate multi-level abstraction; an oracle forget condition provides an upper bound and validation of the assumption that forget decisions should correlate with task-relevance labels. Success rate directly measures whether the agent completes tasks correctly. If HMPF outperforms baselines specifically on long-horizon, distractor-rich, or cross-session tasks, it confirms the advantage of learned, hierarchical forgetting.

### Expected outcome and causal chain

**vs. Flat Memory** — On a long trajectory (e.g., 50 steps) where irrelevant details accumulate, Flat Memory stores all raw features, causing memory overload and interfering with retrieval of crucial early instructions. Our method compresses sequences into hierarchical abstractions and forgets low-level details when they become useless, retaining only coarse task context. Thus, we expect a noticeable gap on tasks longer than 20 steps, but parity on very short tasks where memory is not stressed. On cross-session tasks, Flat Memory cannot compress across episodes, leading to memory overflow; HMPF's hierarchical compression allows retention across sessions, yielding higher cumulative success.

**vs. Uniform Forgetting** — On a task with variable relevance (e.g., a red key is important in one episode but not another), Uniform Forgetting randomly drops either crucial or irrelevant details with equal probability, leading to inconsistent performance. Our method learns via REINFORCE to forget non-rewarding features, so it adapts per episode. We expect lower variance and higher average success rate, especially on tasks where feature relevance shifts across sessions.

**vs. No Forgetting** — On a task with many distractor objects (e.g., multiple colored keys, only one matters), No Forgetting retains all levels, causing interference between irrelevant and relevant memories when the agent needs to recall the target. Our method forgets the detailed features of distractors after confirming they are irrelevant, keeping only high-level patterns. We expect HMPF to succeed more often on distractor-rich episodes, with no significant difference on minimal-distractor episodes. For cross-session tasks, No Forgetting accumulates irrelevant cross-session data, degrading performance.

**vs. DNC** — DNC's learned read/write mechanisms may adapt to cross-session memory but require many parameters and can overfit training tasks. HMPF's explicit hierarchical abstraction may provide more interpretable and generalizable forgetting. We expect HMPF to match or exceed DNC on BabyAI while being more parameter-efficient.

### What would falsify this idea
If HMPF shows uniform improvement across all task lengths and distractor conditions rather than concentrated gains on long/distractor/cross-session episodes, then the hierarchical forgetting mechanism is not the cause; alternatively, if the ablation without hierarchy matches or exceeds HMPF, then hierarchy is detrimental. Additionally, if the correlation between learn forget decisions and oracle task-relevance labels is low (<0.3), it suggests the reward signal is insufficient for learning useful forget policies, contradicting the load-bearing assumption.

## References

1. OAgents: An Empirical Study of Building Effective Agents
2. OS-Copilot: Towards Generalist Computer Agents with Self-Improvement
3. CogAgent: A Visual Language Model for GUI Agents
4. OpenAgents: An Open Platform for Language Agents in the Wild
5. Pix2Struct: Screenshot Parsing as Pretraining for Visual Language Understanding
6. WebShop: Towards Scalable Real-World Web Interaction with Grounded Language Agents
