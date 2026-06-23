# Decentralized Adaptive Trust-based Aggregation for Robust Multi-Agent Coordination

## Motivation

AgentsNet and multiagent debate methods (e.g., Du et al.) assume all agents are cooperative and trustworthy, but real-world deployments include adversarial or untrusted agents. These approaches fail structurally because they lack a mechanism to detect and counteract untrusted behavior, leading to vulnerability to misinformation. For example, AgentsNet evaluates coordination under cooperative assumptions without robustness to adversarial agents, and multiagent debate equally weights all agents, allowing a single adversary to corrupt the consensus.

## Key Insight

Trust derived from communication consistency provides a decentralized, assumption-free signal of agent reliability because consistent agents are statistically less likely to be adversarial.

## Method

### Decentralized Adaptive Trust-based Aggregation (DATA) Framework

(A) **What it is**: We propose the Decentralized Adaptive Trust-based Aggregation (DATA) framework, where each agent maintains a trust score for every other agent based on observed consistency between their statements and actions, and a periodic correctness signal from a calibration set, and uses these scores to weight contributions for robust consensus. The load-bearing assumption is that an agent's communication consistency over time is a reliable indicator of its trustworthiness; to mitigate the 'sleeper agent' failure mode where a consistent liar maintains trust, we augment with a correctness verification step using a small calibration set of known ground-truth inputs. Inputs: multiple rounds of agent responses; Outputs: aggregated decision with adversarial resilience.

(B) **How it works**:
```pseudocode
Initialize trust matrix T[i][j] = 1 for all i, j
Initialize calibration set C of 100 queries with known ground-truth answers
for each round r:
  for each agent i:
    receive messages m_j from all j (including own)
    compute consistency c_ij = cosine_similarity(embed(m_j^(r-1)), embed(m_j^(r)))
    update trust: T[i][j] = alpha * T[i][j] + (1-alpha) * c_ij  # alpha=0.9
  if r % 5 == 0:  # every 5 rounds
    for each agent j:
      correctness corr_j = accuracy of agent j on calibration set C
      for each agent i:
        T[i][j] = beta * T[i][j] + (1-beta) * corr_j  # beta=0.8
  for each agent i:
    # weighted aggregation (for numerical responses)
    r_i = sum_j( T[i][j] * m_j ) / sum_j( T[i][j] )
    # optional threshold theta=0.3 to ignore agents with T[i][j] < theta
```
Hyperparameters: alpha=0.9 (trust decay factor), theta=0.3 (ignored trust threshold), beta=0.8 (calibration update weight), calibration set size |C|=100, calibration frequency K=5 rounds. Embedding model: SBERT (all-MiniLM-L6-v2) by default.

(C) **Why this design**: We chose consistency-based trust over reputation from external sources because it is fully decentralized and does not require a central authority or predefined trust anchors, accepting the cost that initial trust values are uniform and may be exploited by gradually inconsistent adversarial agents. We selected exponential moving average for trust updates over a sliding window because it gives more weight to recent observations, enabling faster adaptation to adversarial behavior, at the cost of some sensitivity to noise. We opted for weighted aggregation rather than hard filtering of low-trust agents to avoid discarding potentially useful contributions from agents that are occasionally inconsistent; this trade-off increases computational overhead but improves robustness in mixed settings. Unlike prior work such as SMoA which sparsifies communication to reduce cost, our method adaptively weights agents based on trust, specifically targeting adversarial scenarios. The calibration step using a small set of known ground-truth queries addresses the 'sleeper agent' failure mode by periodically injecting a correctness signal independent of consistency.

(D) **Why it measures what we claim**: The consistency score c_ij measures trustworthiness because it quantifies the reliability of an agent's communication under the assumption that adversarial agents are more likely to produce inconsistent statements over time; this assumption fails when an adversary deliberately maintains consistency to gain trust (sleeper agent), in which case c_ij reflects compliance rather than honesty, and the system may be slower to detect such an adversary. The correctness signal corr_j from calibration set C directly measures trustworthiness on known ground truth, mitigating the sleeper agent failure mode under the assumption that the calibration set is representative of the overall task; this assumption fails if the calibration set distribution shifts from the main task, in which case corr_j may be misleading. The weighted aggregation with trust scores operationalizes robust coordination because trust-weighted contributions reduce the influence of low-trust agents under the assumption that trust correlates with correctness; this assumption fails when a low-trust agent is actually correct due to random chance, in which case the aggregation may undervalue a correct minority opinion. The trust update rule operationalizes decentralized learning because each agent independently updates pairwise trust using local observations and shared calibration results, assuming that agents can observe the same communication history and calibration outputs; this assumption fails if communication is private or delayed, in which case trust estimates may diverge.

## Contribution

(1) A decentralized trust learning mechanism that uses communication consistency to detect untrusted agents without external validation or central authority. (2) A robust multi-agent coordination protocol that weights contributions by learned trust, enabling resilient consensus under adversarial conditions. (3) Empirical demonstration that the framework significantly outperforms equal-weight consensus and standard multiagent debate when a subset of agents is adversarial.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Simulated Consensus Task with Adversaries | Synthetic multi-agent QA with injected adversarial agents |
| Primary metric | Adversarial Robustness (accuracy under attack) | Measures resilience to inconsistent or malicious agents |
| Baseline 1 | Single-agent LLM | No multi-agent collaboration, baseline performance |
| Baseline 2 | Dense Mixture-of-Agents (full communication) | Prior work; all agents equally weighted |
| Ablation-of-ours | DATA w/o trust updates (uniform weights) | Isolates effect of adaptive trust mechanism |
| Ablation-of-ours | DATA with BERT embeddings (vs. SBERT) | Tests generalizability to different embedding models |

### Why this setup validates the claim

This setup tests the central claim that trust-weighted aggregation improves adversarial resilience in decentralized multi-agent systems. The synthetic adversarial dataset creates controlled scenarios where some agents produce inconsistent statements, forcing the method to rely on trust. The single-agent baseline shows the upper bound of individual reasoning without collaboration, while the dense MoA baseline represents the naive collaborative approach that treats all agents equally, vulnerable to adversaries. The ablation removes trust adaptation, isolating its contribution. The primary metric—robustness under varying attack intensity—directly measures the method's ability to maintain accuracy when trust is critical. If our method outperforms dense MoA specifically under high attack rates, the trust mechanism is validated. If gains are uniform, the claim is weakened.

### Expected outcome and causal chain

**vs. Single-agent LLM** — On a case requiring diverse perspectives (e.g., complex reasoning with missing information in one agent), the single-agent produces incomplete or incorrect output because it lacks collaborative aggregation. Our method instead weighs contributions from multiple agents based on their consistency and calibration correctness, so we expect a noticeable gap on reasoning-intensive tasks where single-agent struggles, but parity on simple tasks.

**vs. Dense Mixture-of-Agents (full communication)** — On a case where 30% of agents are adversarial and intentionally flip their answers each round, the dense MoA aggregates equally, so the adversarial votes dominate and overall accuracy drops. Our method downweights these agents after observing inconsistency and periodic calibration, leading to higher accuracy under strong attacks. We expect a clear gap in the high-attack subset, with similar performance when few adversaries exist.

### What would falsify this idea

If our method shows no accuracy advantage over dense MoA specifically on subsets with high adversarial inconsistency (e.g., >20% adversaries), the trust mechanism fails to detect or mitigate adversarial behavior, falsifying the claim that consistency-based trust combined with periodic calibration improves robustness.

## References

1. AgentsNet: Coordination and Collaborative Reasoning in Multi-Agent LLMs
2. Towards Multi-Agent Reasoning Systems for Collaborative Expertise Delegation: An Exploratory Design Study
3. SMoA: Improving Multi-agent Large Language Models with Sparse Mixture-of-Agents
4. Merging Experts into One: Improving Computational Efficiency of Mixture of Experts
5. MedAgents: Large Language Models as Collaborators for Zero-shot Medical Reasoning
6. Problem-Solving in Language Model Networks
7. MMAU: A Holistic Benchmark of Agent Capabilities Across Diverse Domains
8. Improving Factuality and Reasoning in Language Models through Multiagent Debate
