# CPS-OCI: A Framework for Evaluating Long-Horizon Agents with Online Constraint Inference

## Motivation

Existing benchmarks like DeepPlanning evaluate agents under static, fully known constraints, failing to capture the dynamic and partially observable nature of real-world constraints. This structural limitation means agents are not tested on their ability to adapt plans when constraints change unexpectedly or when they must infer hidden constraints through interaction. The meta-problem is that all prior evaluation methods assume a priori constraint specification, which is unrealistic for deployment in open-world environments.

## Key Insight

Because constraint violation is the only feedback signal, the optimal plan must trade off between goal achievement and information gathering about the current constraint set, a coupling that prior evaluation methods ignore.

## Method

### (A) What it is
CPS-OCI is a unified evaluation framework where an agent maintains a belief distribution over a dynamically evolving set of partially observable constraints and plans actions to maximize expected reward while actively gathering information to disambiguate the constraints.

### (B) How it works
```python
# CPS-OCI evaluation loop
# Input: goal G, prior P_0 over constraint sets C, constraint dynamics P(C'|C), observation model P(o|a,C), reward R(s,a)
# Agent policy π: belief -> action

Initialize belief b_0 = set of N particles sampled from P_0
for t = 0,1,...,T:
    # Agent selects action using planner that considers belief
    a_t = plan(b_t, G)  # uses POMCP with generative model
    
    # Execute action in environment
    o_t = observe(a_t)  # binary feedback: success/failure or noisy indicator
    
    # Environment updates true constraint set (unobserved to agent)
    C_{t+1} = sample from P(C'|C_t)
    
    # Agent updates belief via particle filter
    for each particle c in b_t:
        # Predict step: c' ~ P(C'|c)
        # Update step: weight *= P(o_t | a_t, c')
        # Resample to form b_{t+1}
    
    # Evaluate agent performance: cumulative reward (goal achieved + constraint satisfaction)
```

**Key components:**
- **Constraint representation:** A set of Boolean predicates over plan steps (e.g., `total_cost < 100`, `max_time_per_day < 8h`). The true set is hidden and evolves via a Markov chain.
- **Observation model:** `P(o|a,C) = 1` if `a` satisfies all constraints in `C`, and `0` otherwise (or with noise for realistic scenarios).
- **Belief update:** Particle filter with systematic resampling and transition kernel from the constraint dynamics.
- **Planner:** POMCP (Silver & Veness, 2010) adapted for constraint-adaptive planning. Each search node represents a belief state; rollout uses particles sampled from the belief and simulates constraint transitions. UCB1 selects actions balancing exploitation (goal progress) and exploration (information gain).
- **Hyperparameters:** N=1000 particles, POMCP simulations=100 per step, UCB1 exploration constant c=1.4, discount factor γ=0.99.

### (C) Why this design
We chose a particle filter over exact Bayesian inference because the constraint space is combinatorial (exponential in the number of constraints), accepting that particle depletion may cause belief collapse when N is too small. We chose POMCP over exact POMDP solvers because the state space includes both continuous and categorical variables and the transition dynamics are complex; the trade-off is that POMCP requires many simulations to achieve stable policies, increasing per-step compute cost. We chose binary observation feedback (success/failure) over richer signals like budget remaining because it mirrors many real-world scenarios (e.g., an action either works or not), even though it slows belief convergence and forces the agent to actively probe. The planning horizon T is set to the total number of steps allowed; we fixed it to 50 steps per task, trading off task complexity against computational feasibility. The UCB1 exploration constant c=1.4 was chosen based on default POMCP settings, balancing the risk of over-exploring (wasting steps) against under-exploring (missing constraint inference).

### (D) Why it measures what we claim
The particle belief update measures the agent's uncertainty about constraints; the assumption is that the particle set approximates the true posterior under the specified transition and observation models—this fails when the number of particles is too small, causing overconfidence and missing constraint sets. The POMCP planner measures the agent's ability to optimize long-term expected reward under uncertainty; the assumption is that the generative model (transition and observation) matches the real environment—this fails when constraint dynamics are misspecified, leading to plans that are optimal under the wrong model. The binary observation o_t measures the agent's ability to infer constraints from indirect feedback; the assumption is that o_t is conditionally independent of irrelevant factors—this fails when noise or confounding observations make multiple constraint sets equally likely, causing ambiguity. The cumulative reward (goal achievement + constraint satisfaction) directly operationalizes the evaluation of adaptive planning under dynamic constraints: a higher reward implies the agent successfully inferred and adapted to constraints over the horizon. This causal chain links the algorithm's computational quantities (belief over constraints, MCTS rollouts, observation likelihood) to the meta-problem of evaluating long-horizon planning under realism.

## Contribution

(1) A unified evaluation framework, CPS-OCI, that models planning tasks with dynamic, partially observable constraints via a POMDP formulation, enabling systematic testing of agents' ability to infer and adapt to evolving constraints. (2) A concrete instantiation of the framework using particle filtering for belief tracking and POMCP for constraint-adaptive planning, demonstrating the joint optimization of goal achievement and information gathering. (3) The identification of a new failure mode in current benchmarks—static constraint assumption—and a quantitative metric (cumulative reward with constraint adaptation) that reveals agent performance in dynamic settings.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|-----------------------|
| Dataset | DeepPlanning benchmark | Verifiable dynamic constraints; binary feedback. |
| Primary metric | Constraint satisfaction rate | Directly measures adaptation to hidden constraints. |
| Baseline 1 | GPT-4 | Strong LLM without explicit belief tracking. |
| Baseline 2 | Claude | Another frontier LLM lacking constraint inference. |
| Baseline 3 | Optimal (oracle) | Full constraint knowledge; upper bound. |
| Ablation-of-ours | Ours w/o belief update | No particle filter; static prior only. |

### Why this setup validates the claim
This design creates a falsifiable test of the central claim that explicit belief tracking over dynamic constraints improves long-horizon planning. DeepPlanning provides tasks where constraints evolve and only binary success/failure feedback is available, mirroring real-world ambiguity. The primary metric—constraint satisfaction rate—directly operationalizes the agent's ability to infer and adapt to hidden constraints. GPT-4 and Claude serve as strong baselines that lack our belief-update mechanism; if our method outperforms them specifically on tasks with high constraint dynamics, it supports that belief tracking is beneficial. The optimal oracle establishes an upper bound, showing the maximum achievable satisfaction. Our ablation (removing the particle filter) isolates the contribution of belief updates: if it performs worse than the full method, the belief component is essential. Thus, the combination of these choices allows us to detect whether the predicted causal chain—belief updates → better constraint inference → higher satisfaction—holds empirically.

### Expected outcome and causal chain

**vs. GPT-4** — On a multi-step travel planning task where the daily cost limit drops unexpectedly at step 10, GPT-4's fixed prompt cannot revise its prior plan, causing cost violations after the change. Our method's particle filter updates belief from binary failure feedback, inferring the new limit and replanning accordingly. We expect a noticeable gap (e.g., 20–30% higher constraint satisfaction) on tasks with frequent or abrupt constraint changes, but parity on static constraints.

**vs. Claude** — On a scenario where an action's success depends on a hidden constraint that only activates after a certain step (e.g., "no flights after 6pm" on day 3), Claude's reactive behavior may continue violating until explicit failure occurs repeatedly. Our method proactively explores actions to disambiguate constraints via POMCP, reducing violations. We expect Claude to show a steep decline in satisfaction on such temporally gated constraints, while our method maintains high satisfaction (e.g., 15–25% gap on affected tasks).

**vs. Optimal (oracle)** — On a task where constraints are initially unknown to all agents, the oracle instantly knows the true set and never violates. Our method requires a few steps to infer constraints via probing, leading to occasional early violations. Thus, we expect our average satisfaction to be slightly lower than optimal (e.g., 85% vs 100%) but significantly higher than LLM baselines (e.g., 50–60%). The gap should concentrate on early steps and diminish over time.

### What would falsify this idea
If our method's constraint satisfaction rate is not significantly higher than GPT-4 on tasks with dynamic constraints, or if the ablation without belief update achieves comparable performance, then the central claim—that explicit belief tracking of evolving constraints drives improvement—would be falsified.

## References

1. DeepPlanning: Benchmarking Long-Horizon Agentic Planning with Verifiable Constraints
2. TripScore: Benchmarking and rewarding real-world travel planning with fine-grained evaluation
3. TripTailor: A Real-World Benchmark for Personalized Travel Planning
4. NATURAL PLAN: Benchmarking LLMs on Natural Language Planning
5. TravelPlanner: A Benchmark for Real-World Planning with Language Agents
6. ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs
7. Reasoning with Language Model is Planning with World Model
8. Large Language Models Cannot Self-Correct Reasoning Yet
