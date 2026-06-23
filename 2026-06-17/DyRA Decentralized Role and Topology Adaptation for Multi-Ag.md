# DyRA: Decentralized Role and Topology Adaptation for Multi-Agent Educational Systems

## Motivation

Existing multi-agent educational systems (e.g., Towards Multi-Agent Reasoning Systems, MedAgents) rely on static role assignments and fixed communication topologies, failing to adapt to evolving task demands and student feedback. This structural rigidity stems from assuming roles and connections are independent of the learning context, leading to suboptimal collaboration as tasks vary in complexity. The root cause is the lack of a decentralized mechanism for runtime interdependence between role plasticity and network evolution.

## Key Insight

Role and topology decisions are coupled via a shared additive utility function over agents that measures each agent's marginal contribution to overall collaboration quality, enabling decentralized optimization through local message passing.

## Method

# DyRA: Decentralized Role and Topology Adaptation

**Assumption:** The linear functions f_r and info_gain accurately approximate the true utility from student state to role appropriateness and collaboration benefit. f_r is trained via linear regression on a calibration set of 512 examples from the RLTeacher simulator with L2 regularization (λ_reg=0.01). info_gain is defined as the absolute difference in role relevance scores minus a communication cost term.

### (A) What it is
DyRA is a lightweight, online algorithm where each pedagogical agent iteratively refines its role (e.g., tutor, motivator, assessor) and communication partners based on local task-context signals and neighbor agreements, without any central controller. Input: current student state (e.g., knowledge level, engagement); output: role assignment and communication graph for the next interaction round.

### (B) How it works (pseudocode)
```python
# Per round, for each agent i in parallel:
# Hyperparameters: consensus_steps=5, max_neighbors=3, learning_rate=0.2, communication_cost_lambda=0.1
# f_r: linear function with learned weights (trained via linear regression on 512 calibration examples)

# Step 1: Compute local task-relevance scores for each role r
scores_i = [f_r(student_state) for r in roles]  # f_r is a learned linear function with L2 regularization (reg=0.01)

# Step 2: Decentralized consensus to smooth scores across agents
for step in range(consensus_steps):
    for each neighbor j:
        send scores_i to j
        receive scores_j from j
    scores_i = (1 - learning_rate) * scores_i + learning_rate * average(scores_j for all j)

# Step 3: Each agent selects role with highest smoothed score
role_i = argmax(scores_i)

# Step 4: Compute information gain from connecting to each other agent
# based on current roles and student state, with communication cost
# info_gain = |scores_i[role_i] - scores_j[role_j]| - communication_cost_lambda * (msg_sent + msg_recv)
gains = [abs(scores_i[role_i] - scores_j[role_j]) - communication_cost_lambda * (1 + 1) for j in agents if j != i]
# bid for connections: each agent ranks potential partners by gains and bids for top k
bids = sorted(zip(agents, gains), key=lambda x: -x[1])[:max_neighbors]
# send bids to corresponding agents

# Step 5: Accept bids greedily: if multiple agents bid for you, accept the one with highest gain
# Agents update their neighbor set based on mutual acceptances
neighbors_i = {j for j in neighbors_i if bid accepted} union new accepted
```

### (C) Why this design
We chose a decentralized consensus mechanism (Step 2) over centralized aggregation because it preserves agent autonomy and scales without a single point of failure, accepting the cost that consensus may require multiple rounds and may not converge perfectly under severe network asynchrony. We used additive utility decomposition (scores and gains) rather than learned end-to-end policies because it is interpretable and requires no training data, but it assumes that the linear functions f_r and info_gain are good proxies for task demands—a trade-off that limits expressivity. To mitigate this, we calibrate f_r on a small dataset (512 examples) using linear regression. We adopted greedy bid acceptance (Step 5) instead of optimal matching to keep computation local and avoid exponential complexity, at the cost of possible suboptimal global topologies. These three design choices together ensure that the algorithm runs online with minimal overhead, at the expense of optimality guarantees.

### (D) Why it measures what we claim
The task-relevance score (Step 1) measures adaptation to task demands because it assumes that the student state is a sufficient statistic for the current learning context; this assumption fails when the student state does not capture unobserved factors (e.g., fatigue), in which case the score reflects only observable features. The consensus-smoothed score (Step 2) measures collective role plasticity because it assumes that peer agreement improves role appropriateness under varying contexts; this assumption fails when agents have identical biases, in which case smoothing amplifies errors. The information gain estimate (Step 4) measures topology adaptation because it assumes that exchanging information between specific role pairs is beneficial for collaboration quality; this assumption fails when the gain metric ignores message cost or latency, in which case it overestimates the value of dense connections. We add a communication cost term to mitigate this. The bid-acceptance step measures decentralized coordination because it assumes that mutual selection increases task-relevant information flow; this assumption fails when agents have conflicting bid rankings, in which case the final topology may not satisfy global efficiency.

## Contribution

(1) A decentralized algorithm (DyRA) for simultaneous role and topology adaptation in multi-agent systems, requiring no central controller or training. (2) An information-theoretic formulation that decomposes collaboration quality into additive local utility functions based on task-relevance and information gain. (3) A design principle that runtime role plasticity and network evolution are interdependent and can be jointly optimized through local message passing.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|----------------------|
| Dataset | Simulated student tutoring dataset using RLTeacher simulator (50 virtual students, 10 sessions each) | Tests adaptation to diverse student states with controlled variation |
| Primary metric | Student learning gain | Direct measure of pedagogical effectiveness |
| Baseline 1 | Single-agent tutor | No multi-agent coordination baseline |
| Baseline 2 | Multi-agent fixed roles | Tests need for role adaptation |
| Baseline 3 | Dense multi-agent full communication | Tests benefit of selective communication |
| Ablation 1 | DyRA without consensus step | Isolates role smoothing effect |
| Ablation 2 | DyRA with communication cost (λ=0.1) | Tests sensitivity to information overload |
| Ablation 3 | DyRA with communication cost (λ=0.5) | Tests sensitivity to high cost |

### Why this setup validates the claim

This setup tests the central claim that decentralized role and topology adaptation improves pedagogical outcomes in multi-agent tutoring. The RLTeacher simulator provides controlled variation in student states (knowledge, engagement, difficulty) which triggers the need for role and topology changes. The single-agent baseline isolates the benefit of having multiple agents at all. Fixed roles baseline tests the necessity of role adaptation, while full communication baseline tests whether selective communication (via bids) reduces noise or overhead. Ablation without consensus measures the contribution of the consensus mechanism to role stability and coordination. Ablations with communication cost test the sensitivity of the info_gain metric to information overload, verifying that the greedy bid acceptance can handle realistic constraints. The learning gain metric directly captures end-to-end teaching effectiveness, which our method aims to maximize. If our method outperforms all baselines on learning gain, especially under diverse student profiles, it validates that the mechanisms work as intended. Conversely, failure patterns on specific baselines would pinpoint whether the adaptation or the consensus is crucial.

### Expected outcome and causal chain

**vs. Single-agent tutor** — On a case where a student shows fluctuating engagement, the single agent cannot switch roles (e.g., from tutor to motivator) and persists with a fixed teaching style, causing disengagement and lower learning gain. Our method allows agents to assume motivational roles dynamically, re-engaging the student. Thus we expect a noticeable gap on low-engagement subgroups but parity on highly motivated students.

**vs. Multi-agent fixed roles** — On a case where the student's knowledge level shifts across topics (e.g., weak in algebra but strong in geometry), fixed role agents cannot reassign who tutors which topic, leading to mismatched expertise and suboptimal instruction. Our method adapts roles per round via consensus, so the best tutor for the current topic emerges. We expect a clear advantage on mixed-knowledge student profiles.

**vs. Dense multi-agent full communication** — On a case where a student has high prior knowledge, sending all opinions from many agents creates information overload and confusion, harming learning. Our method selectively connects only high-info-gain agent pairs, reducing redundancy. Thus we expect better learning gain in high-knowledge scenarios, but similar performance in simple tasks where communication load is low.

**vs. DyRA without cost** — On scenarios with limited student attention, the communication cost penalty prunes low-gain connections, reducing cognitive load and improving learning gain. We expect DyRA with cost (λ=0.1) to outperform the version without cost on high-difficulty tasks where information overload is detrimental, while λ=0.5 may over-penalize and degrade performance.

### What would falsify this idea
If DyRA performs no better than fixed roles on diverse student states, or if the ablation without consensus matches or outperforms full DyRA, then the central claims about role adaptation and consensus are unsupported. Additionally, if DyRA with communication cost performs worse than DyRA without cost across all scenarios, the hypothesized benefit of selective communication is not realized.

## References

1. Towards Multi-Agent Reasoning Systems for Collaborative Expertise Delegation: An Exploratory Design Study
2. SMoA: Improving Multi-agent Large Language Models with Sparse Mixture-of-Agents
3. Merging Experts into One: Improving Computational Efficiency of Mixture of Experts
4. MedAgents: Large Language Models as Collaborators for Zero-shot Medical Reasoning
5. AgentsNet: Coordination and Collaborative Reasoning in Multi-Agent LLMs
6. Problem-Solving in Language Model Networks
7. MMAU: A Holistic Benchmark of Agent Capabilities Across Diverse Domains
8. Improving Factuality and Reasoning in Language Models through Multiagent Debate
