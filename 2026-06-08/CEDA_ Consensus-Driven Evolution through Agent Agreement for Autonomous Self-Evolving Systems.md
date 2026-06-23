# CEDA: Consensus-Driven Evolution through Agent Agreement for Autonomous Self-Evolving Systems

## Motivation

Existing self-evolving agent systems, such as Yunjue Agent (Yunjue Agent Tech Report), rely on external binary feedback from task execution to guide tool evolution, limiting autonomy because task-specific success signals are unavailable in open-ended settings. More broadly, both research trees (e.g., Alita-G and AFlow) share the assumption that evolution must be guided by an external oracle rather than emerging from intrinsic interaction, blocking full autonomy. This dependency on external evaluation is a structural bottleneck because it requires a predefined evaluation task or human-designed reward, which cannot scale to arbitrary novel environments.

## Key Insight

Mutual agreement among multiple independently-developed predictive agent models provides a self-contained evaluation signal that correlates with solution quality without requiring an external oracle, because diverse inductive biases ensure that high consensus is unlikely for poor proposals.

## Method

### A) What it is
CEDA (Consensus-Driven Evolution through Agent Agreement) is a multi-agent self-evolution framework that uses the degree of consensus among agents' predictive models as the sole evolution criterion, eliminating the need for an external evaluator. The inputs are a population of agents, each with a learned predictive model of the task environment, and a set of candidate modifications (tools, plans). The output is an evolved population with improved capabilities, where modifications are accepted based on consensus scores.

### B) How it works
```pseudocode
Initialize population P = {agent_i with predictive model M_i} (size N=20)
// Diversity maintenance: train each M_i on a disjoint subset of an initial task corpus (100 random tasks per agent) to encourage diversity.
for generation in 1..G (G=100):
    for each agent i in P:
        candidate = propose_modification(agent_i)  // new tool or strategy
        for each agent j in P (j != i):
            simulate outcome o_j = M_j(candidate, current_state)
        compute consensus C(candidate) = 1 - std({o_j}) / mean({o_j})   // normalized inverse variance
        if C(candidate) > threshold θ (θ=0.75):
            accept candidate into agent_i's repertoire
            update M_i with new experience (online learning)
        else:
            reject
    // optional: agent selection and reproduction (e.g., keep top 20 by cumulative consensus over gen)
    // Periodic calibration: every 10 generations, use a validation set of 50 unseen tasks to compute correlation between consensus and actual success; if correlation < 0.3, recalibrate θ by grid search in [0.5,0.9].
```
Hyperparameters: population size N=20, threshold θ=0.75, generations G=100, and for online learning of M_i we use a small Bayesian neural network updated via stochastic gradient descent with learning rate 0.001. Diversity is maintained by initializing each M_i on disjoint task subsets.

### C) Why this design
We chose consensus based on inverse variance of predicted outcomes over std (scaled by mean) instead of using majority vote or cross-entropy because variance captures agreement in continuous spaces without requiring a categorical decision boundary, accepting the cost that it assumes outcomes are real-valued and roughly Gaussian. We used a fixed population of 20 agents rather than a dynamically sized population because it provides sufficient diversity for robust consensus while keeping computational cost manageable; the trade-off is that smaller populations may yield unreliable consensus. We set a fixed threshold θ=0.75 instead of a learned or adaptive threshold because a static threshold is simpler and avoids additional hyperparameter tuning; however, it may not be optimal across all tasks and may need adjustment, which we address by periodic recalibration with a validation set. We update each agent's predictive model using online learning with new experience rather than retraining from scratch because it allows continuous adaptation without storing all past data; the cost is potential catastrophic forgetting, which we mitigate by using a Bayesian neural network with weight uncertainty. We chose to train each predictive model on disjoint data subsets to enforce diversity (see Zhou, 2012), which reduces the risk of correlated biases. The trade-off is that each agent sees less data, potentially reducing individual model accuracy; we mitigate this by using a Bayesian neural network that can learn from limited data. These design choices prioritize simplicity and computational efficiency over theoretical optimality, making CEDA practical for real-world deployment without an external evaluator.

### D) Why it measures what we claim
The inverse variance of predicted outcomes across agents (consensus C) measures solution quality because we assume that agents' predictive models have diverse and independent inductive biases, so that only proposals that are correct (i.e., lead to high true success) would be consistently predicted across agents; this assumption fails when the models share systematic biases (e.g., all trained on similar flawed data), in which case high consensus may reflect collective blind spots rather than true quality. The online update of each M_i using accepted proposals measures the degree to which agents align with the intrinsic evaluation signal, ensuring that the population co-evolves; however, if rejection decisions are too frequent, the models may stagnate because they receive little diverse data. We empirically measure agent diversity via pairwise mutual information between model embeddings on a held-out set; low mutual information indicates high diversity, supporting the assumption that agents' predictive models are sufficiently independent.

## Contribution

['(1) A novel multi-agent self-evolution framework, CEDA, that replaces external evaluators with an intrinsic consensus signal derived from agent interactions, enabling fully autonomous evolution.', '(2) The design principle that mutual agreement among diverse predictive models serves as a sufficient and self-consistent evaluation criterion for tool and strategy evolution, demonstrated through the inverse-variance consensus metric.', '(3) An open-source implementation of the CEDA architecture, including the predictive model update mechanism and consensus computation, to facilitate reproducibility and further research.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Open-ended task suite (GAIA, SWE-bench, toy domain) | Tests generalization across diverse tasks |
| Primary metric | Task success rate | Direct measure of agent capability |
| Baseline 1 | Static agent (ReAct) | No self-evolution baseline |
| Baseline 2 | Offline self-improvement (DGM-style) | Tests advantage of online consensus |
| Baseline 3 | CEDA-MajorityVote | Isolates effect of continuous consensus measure |
| Ablation 1 | CEDA – online model update | Isolates effect of continual learning |
| Ablation 2 | CEDA with population size N=10 | Tests sensitivity to population size; speeds up initial experiments |
| Controlled experiment | CEDA with agents sharing correlated biases (all pre-trained on identical data) | Tests failure mode of high consensus on low-quality proposals |
| Additional metric | Agent diversity (pairwise mutual information between model embeddings) | Validates assumption of diverse inductive biases |

### Why this setup validates the claim
This combination forms a falsifiable test of CEDA’s central claim: that consensus without external evaluator drives effective self-evolution. Using diverse open-ended tasks ensures the result is not task-specific. The static baseline measures whether any evolution helps at all; the offline baseline tests whether online, consensus-driven updates outperform batch retraining; the majority vote baseline tests whether the continuous nature of consensus (inverse variance) is crucial. The ablation removing online model updates isolates whether continual adaptation is necessary. The controlled experiment with correlated biases directly tests the load-bearing assumption about diversity. Adding an explicit diversity metric provides an empirical check on the assumption. If CEDA outperforms baselines, the ablation degrades performance, and diversity correlates with success, the claim is supported; if diversity is low or CEDA fails with correlated biases, the claim is falsified.

### Expected outcome and causal chain

**vs. Static agent (ReAct)** — On a case where a beneficial new tool exists (e.g., a calculator for math tasks), the static agent never discovers it because it cannot modify its repertoire. Our method instead proposes the tool, achieves high consensus (most predictive models agree it helps), and accepts it. Thus we expect a noticeable gap on tasks requiring tool innovation (e.g., 20+ point success rate difference), but parity on tasks solvable by base repertoire.

**vs. Offline self-improvement (DGM-style)** — On a case where the environment shifts (e.g., task distribution changes after 50 generations), the offline baseline must retrain from scratch, wasting data and failing to adapt quickly. Our method updates models incrementally via online learning, so it smoothly tracks the shift. We expect CEDA to maintain or improve performance over time, while offline baseline shows a dip and slow recovery. The observable signal: CEDA’s success rate slope remains positive, offline’s shows a temporary decline.

**vs. CEDA-MajorityVote** — On a case where candidate modifications yield continuous-valued outcomes, inverse variance captures fine-grained agreement, whereas majority vote discards magnitude information. For example, if one agent predicts a large positive outcome and nine predict small negative, inverse variance yields low consensus (variance high) while majority vote yields high consensus (9/10 agree on negative). We expect CEDA to reject such candidates because low consensus correctly indicates uncertainty; majority vote might erroneously accept them. Thus CEDA should achieve higher success rate on tasks requiring nuanced evaluation.

**Controlled experiment (correlated biases)** — When all agents are pre-trained on identical data, we expect their predictions to be highly correlated. In this scenario, consensus may be high even for poor proposals. CEDA’s success rate should drop compared to the diverse setting, and diversity metric (mutual information) will be high. This would validate the assumption and show a clear failure mode.

### What would falsify this idea
If CEDA’s performance improvement is uniform across all subsets and tasks (rather than concentrated where consensus is expected to distinguish good vs. bad modifications), or if the ablation without online updates performs equally well, indicating that consensus alone suffices without adaptation, then the central claim (that online, consensus-driven evolution is key) is falsified. Additionally, if CEDA with correlated biases performs as well as with diverse agents, the load-bearing assumption about diversity is false.

## References

1. Yunjue Agent Tech Report: A Fully Reproducible, Zero-Start In-Situ Self-Evolving Agent System for Open-Ended Tasks
2. Live-SWE-agent: Can Software Engineering Agents Self-Evolve on the Fly?
3. Alita-G: Self-Evolving Generative Agent for Agent Generation
4. AFlow: Automating Agentic Workflow Generation
5. AgentSquare: Automatic LLM Agent Search in Modular Design Space
6. AutoGenesisAgent: Self-Generating Multi-Agent Systems for Complex Tasks
7. EvoAgent: Towards Automatic Multi-Agent Generation via Evolutionary Algorithms
8. MarsCode Agent: AI-native Automated Bug Fixing
