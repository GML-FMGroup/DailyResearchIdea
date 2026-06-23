# OPAS: Online Poset Adaptation for Search via Bayesian Nonparametric Priors

## Motivation

Existing search agents (e.g., those in AI Research Agents for ML) fix the operator set and search policy at initialization, preventing adaptation to task variations. This static precondition, as exemplified by the fixed operator sets and policies in MLE-bench agents, forces a trade-off between coverage and efficiency that cannot be resolved without online adaptation. The structural problem is that operators and policies are defined once and never updated from the agent's own search trajectory, blocking learning from experience.

## Key Insight

A partially ordered set (poset) of operators, combined with a Bayesian nonparametric prior, allows the agent to dynamically add valid composite operators from successful search trajectories while guaranteeing that the policy's random walk never selects malformed operators, because the poset encodes the composition rules.

## Method

### (A) What it is
**OPAS** (Online Poset Adaptation for Search) is an algorithm that maintains a probabilistic poset of operators and a search policy defined as a random walk over that poset. Input: a task, base operators, and a maximum number of iterations. Output: a solution (candidate) and the updated poset and policy.

### (B) How it works
```pseudocode
Initialize poset P with base operators (nodes) and edges defining valid compositions.
Initialize transition count matrix T (|P| x |P|) with pseudocounts from a Dirichlet process prior (concentration α).
Initialize policy π as a random walk: from node u, pick next node v with prob ∝ (T[u,v] + α/|P|).

for each episode (task instance):
    candidate = initial solution
    trajectory = []
    for step in 1..max_steps:
        sample operator o by walking from current node (e.g., root) using π for L steps.
        candidate = apply(o, candidate)
        append (o, intermediate_reward) to trajectory
    final_reward = evaluate(candidate)
    # Compute discounted returns for each step (γ = 0.9)
    returns = []
    cumulative = 0
    for t from step_max down to 1:
        cumulative = trajectory[t].intermediate_reward + γ * cumulative
        returns[t] = cumulative
    # Update policy transitions using discounted returns (load-bearing assumption: locally beneficial transitions weighted by delayed reward correlate with global utility)
    for each transition (u->v) in trajectory:
        T[u,v] += returns[t]    # discounted cumulative reward from that step
    # Optionally add composite operators
    best_sequence = argmax over trajectory of cumulative reward
    if best_sequence not in P:
        add composite node = sequence of operators as a new node
        add edges from components to composite, and from composite to others as per composition rules
        expand T with new rows/columns initialized with Dirichlet process pseudocounts
    # Recompute π from T (normalize rows)
```
Hyperparameters: L (walk length, e.g., 3), α (Dirichlet concentration, e.g., 0.1), γ (discount factor, e.g., 0.9), composite addition triggered every K episodes (e.g., 5).

### (C) Why this design
We chose a poset over a flat set (e.g., a dictionary) because the partial order enforces composition constraints—every new composite operator is guaranteed to be a valid sequence of existing operators, preventing the policy from selecting malformed operators (a known failure in manual operator engineering). We chose a random walk policy over a learned policy (e.g., Q-learning) because the walk's Markov property aligns with the poset's local structure, enabling online Bayesian updates without requiring value function approximation; the cost is that the policy may explore suboptimally compared to a global planner. We chose a Dirichlet process prior over a fixed parametric model (e.g., a multinomial) because it allows the number of operators and transitions to grow unboundedly as new composites are discovered, matching the open-ended nature of search; the trade-off is added computational overhead for posterior sampling. Finally, we update the transition counts by discounted cumulative reward weighting rather than simple increments to better capture global utility, accepting that the discount factor introduces a bias-variance trade-off.

### (D) Why it measures what we claim
The transition count T[u,v] measures the empirical probability of transitioning from operator u to v after successful steps, which operationalizes **adaptivity** because it encodes how the agent's search history shapes future policy; the underlying assumption is that discounted cumulative reward weighting ensures high transition counts correspond to beneficial operator sequences (the **experience-bridge assumption**). This assumption fails when rewards are sparse or deceptive (e.g., locally frequent transitions yield high immediate reward but lead to dead ends later), in which case the count reflects only local myopic success rather than global utility. The addition of composite operators from best sequences measures **learning of new building blocks** because it compresses successful trajectories into new primitives; the assumption is that the best sequence is a reusable sub-routine (the **reusability assumption**). This assumption fails when the best sequence is overfitted to a single task instance, in which case the new operator adds noise rather than generalizable knowledge. The Dirichlet process prior over the number of operators measures **capacity unboundedness** because it assigns positive probability to new operators not yet observed; the assumption is that the base measure covers all plausible composites (the **infinite-composition assumption**). This assumption fails when novel successful operators are structurally different from any combination of existing ones, in which case the prior cannot generate them and capacity remains limited.

## Contribution

(1) A novel framework (OPAS) that combines a partially ordered set of operators with a Bayesian nonparametric prior to enable online adaptation of both the operator set and the search policy. (2) The design principle that poset structure guarantees operator validity during random-walk policy execution, eliminating the need for explicit validity checks. (3) An empirical demonstration (to be done) that OPAS outperforms static-policy baselines on a suite of ML engineering tasks from MLE-bench, with the adaptation mechanism responsible for the improvement.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|---------|-----------|
| Dataset | MLE-bench lite + synthetic tasks (10 tasks with known optimal operator sequences and ground-truth utility) | MLE-bench lite provides realistic task diversity; synthetic tasks enable controlled verification of local-global correlation |
| Primary metric | Medal success rate (MLE-bench); correlation between transition counts and ground-truth utility (synthetic); average cumulative reward | Direct measure of task completion on realistic tasks; controlled metric for assumption validation |
| Baseline 1 | AIDE (SOTA) | Strongest existing search agent |
| Baseline 2 | Human median (Kaggle) | Realistic performance ceiling |
| Baseline 3 | Population-Based Training (PBT) for online operator selection | State-of-the-art online adaptation method, quantifies practical significance of OPAS’s specific adaptation mechanism |
| Ablation 1 | OPAS w/o composite add (no new operators) | Tests learning of new building blocks |
| Ablation 2 | OPAS w/ flat set (poset replaced by flat operator set, same count-based update) | Isolates novelty of poset structure |
| Ablation 3 | OPAS w/ simple count increment (instead of discounted reward weighting) | Tests the effect of the smallest repair from the adversarial alert |

### Why this setup validates the claim
This combination of datasets, baselines, and ablations forms a falsifiable test of OPAS's claim to adaptivity, composable learning, and unbounded capacity. MLE-bench lite's diverse tasks require flexible operator sequences, so a successful agent must adapt its policy per task. AIDE represents a static-policy baseline; outperforming it would show adaptivity benefits. The human median provides a ceiling for meaningful improvement. PBT represents a strong online adaptation method; comparing to it reveals whether OPAS's poset-based random walk offers unique advantages. The controlled synthetic tasks directly measure the local-global correlation assumption by comparing learned transition counts to ground-truth operator utilities. The ablations isolate the effects of composite addition (reusability), poset structure (composition constraints), and the discounted reward weighting (local-global correlation). The medal success rate is a holistic metric that captures both final solution quality and efficiency, directly reflecting the agent's ability to navigate the operator poset effectively.

### Expected outcome and causal chain

**vs. AIDE (SOTA)** — On a case where the optimal operator sequence is long and context-dependent, AIDE relies on a fixed set of operators and a learned policy from offline training, which may fail to adapt to unseen composition patterns. AIDE's static policy cannot dynamically reweight operator transitions; it may get stuck in suboptimal loops. OPAS, by contrast, updates its transition counts online via Bayesian posterior weighted by discounted returns, so after a few episodes it biases toward sequences that yielded high long-term rewards in similar tasks. We expect OPAS to achieve a significantly higher medal rate (e.g., +15% absolute) on tasks requiring novel compositions, while performing similarly on simple tasks.

**vs. Human median (Kaggle)** — On a case requiring creative recombination of basic ML operations (e.g., feature engineering steps), the human median represents typical Kaggle competitor performance. Humans can reason about long-term utility, but they are limited by manual exploration. OPAS, with its random walk over a growing poset, may discover unusual but effective sequences through stochastic exploration and composite operator reuse. We expect OPAS to match or exceed human median on half the tasks, particularly those with well-defined building blocks, but fall short on tasks requiring deep domain insight (where human priors dominate).

**vs. PBT** — On tasks with varying optimal operator sequences, PBT hyperparameter optimization adapts the operator set by exploring and exploiting across population members. However, PBT does not exploit compositional structure (poset) and uses a fixed number of operators. OPAS’s poset allows it to grow the operator set systematically and reuse subroutines. We expect OPAS to achieve similar or better medal rates on tasks where composition is key, but may have higher computational cost due to poset maintenance.

**vs. OPAS w/o composite add (Ablation 1)** — On tasks with recurring multi-step operator sequences, the ablation cannot compress them into composite operators, leading to slower convergence. We expect full OPAS to show clear improvement in learning speed and final medal rate.

**vs. OPAS w/ flat set (Ablation 2)** — On tasks requiring strict operator ordering (e.g., preprocessing must come before modeling), the flat set allows malformed sequences (e.g., applying model before features), wasting steps. The poset’s partial order prevents such errors, leading to higher efficiency. We expect full OPAS to outperform the flat set version on tasks with strong ordering constraints.

**vs. OPAS w/ simple count increment (Ablation 3)** — On tasks with delayed rewards (e.g., final evaluation after many steps), simple count increments may favor myopic sequences. Discounted reward weighting better captures global utility. On such tasks, we expect the full OPAS to achieve higher medal rate.

### What would falsify this idea
If OPAS fails to outperform AIDE on tasks requiring adaptivity (i.e., performance gap uniform across all task types), the adaptive policy mechanism is not beneficial. If full OPAS performs similarly to the flat set ablation on tasks with ordering constraints, the poset structure is not providing composition guarantees. If full OPAS performs no better than the simple count increment ablation on tasks with delayed rewards, the discounted reward weighting is ineffective.

## References

1. AI Research Agents for Machine Learning: Search, Exploration, and Generalization in MLE-bench
2. MLE-bench: Evaluating Machine Learning Agents on Machine Learning Engineering
3. SWE-Search: Enhancing Software Agents with Monte Carlo Tree Search and Iterative Refinement
4. Eureka: Human-Level Reward Design via Coding Large Language Models
5. Improving Factuality and Reasoning in Language Models through Multiagent Debate
6. Tree of Thoughts: Deliberate Problem Solving with Large Language Models
7. ML-Bench: Evaluating Large Language Models and Agents for Machine Learning Tasks on Repository-Level Code
8. MLAgentBench: Evaluating Language Agents on Machine Learning Experimentation
