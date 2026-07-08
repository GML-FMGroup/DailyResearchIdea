# Self-Play Workflow Induction via Counterfactual Regret Minimization for Scientific Literature Search

## Motivation

Existing methods for multi-turn agentic search, such as PaperPilot and PaSa, rely on costly human demonstrations or synthetic data for training, which limits scalability and generalization. The root cause is the difficulty of long-horizon credit assignment over DAG-structured action sequences without per-step rewards. Counterfactual regret minimization (CFR) theoretically addresses this by computing regret from terminal outcomes in extensive-form games, but it has not been applied to workflow induction due to the combinatorial state space.

## Key Insight

By modeling workflow generation as a perfect-information extensive-form game, CFR’s no-regret property enables self-play to converge to an optimal policy using only terminal retrieval quality, without requiring any human demonstrations or per-step reward engineering.

## Method

### DAG-CFR: Self-Play Workflow Induction via Counterfactual Regret Minimization

(A) **What it is**: DAG-CFR is a self-play algorithm that induces a DAG workflow for scientific literature search by iteratively updating a neural policy network through counterfactual regret minimization, using only the final retrieval quality (e.g., nDCG@10) as feedback. Input: initial query and anchor paper. Output: a distribution over DAG workflows (each a sequence of operators).

(B) **How it works** (pseudocode):
```python
# Hyperparameters: num_iterations=5000, num_rollouts=20, gamma=1.0, lr=0.001, hidden_dim=256, max_steps=10, heldout_size=512
# State encoding: concatenate embeddings of executed operators, current results, and query (each embedded via a 128-dim MLP)
# Action space: {KeywordSearch(terms), CitationExpand(), FilterByYear(year), ScoreLLM(), ReRank(), ExtractEvidence()}

initialize policy network π_θ(a|s) as 2-layer MLP with hidden_dim=256, ReLU, and softmax output
initialize value network V_φ(s) as 2-layer MLP with hidden_dim=256, ReLU, and scalar output
initialize heldout set of 512 random states (collected from initial random rollouts)

for iteration in range(num_iterations):
    # Self-play: generate one complete workflow from current policy
    workflow = []
    s = initial_state(query, anchor_paper)
    while not terminal(s):  # terminal after max_steps=10 or operator limit
        a ~ π_θ(·|s)
        s' = execute_operator(s, a)  # deterministic transition
        workflow.append((s, a, s'))
        s = s'
    R = evaluate_retrieval(s)  # e.g., nDCG@10 of final retrieved papers
    
    # Compute counterfactual values for each state-action via Monte Carlo rollouts
    for (s, a, s') in workflow:
        # Estimate Q_π(s,a) = expected return if action a taken then follow π
        Q = 0.0
        for _ in range(num_rollouts=20):
            s_rollout = s'
            t = 0
            while not terminal(s_rollout):
                a_rollout ~ π_θ(·|s_rollout)
                s_rollout = execute_operator(s_rollout, a_rollout)
                t += 1
            Q += evaluate_retrieval(s_rollout) / 20
        # Counterfactual value V_π(s) = V_φ(s)  (from value network)
        V = V_φ(s)
        regret[a] = Q - V  # regret for action a
    
    # Update policy network: train to predict regret-weighted action distribution
    target_prob = softmax(relu(regret_vector))  # positive regret only, else zero
    loss_policy = cross_entropy(π_θ(·|s), target_prob)
    update θ via gradient descent (Adam, lr=0.001)
    # Update value network: regression to observed returns
    loss_value = MSE(V_φ(s), R)
    update φ via gradient descent (Adam, lr=0.001)
    
    # Calibration check every 500 iterations
    if iteration % 500 == 0:
        compute MSE between V_φ(s) and Monte Carlo returns on heldout set
        if MSE > 0.1:
            log warning: value network may be diverging
```

(C) **Why this design**: (1) We chose CFR over policy gradient because CFR provides no-regret guarantees and naturally handles credit assignment via regret computation from terminal rewards, whereas policy gradient requires per-step reward shaping (expensive and brittle). The trade-off is that CFR requires estimating counterfactual values, which adds variance via Monte Carlo rollouts. (2) We use neural function approximation (MLPs) rather than tabular CFR because the state space of partial DAGs is combinatorial; this introduces approximation error but enables generalization across similar workflow states. This design relies on the assumption that the neural value network V_φ(s) provides sufficiently accurate estimates for regret computation; to verify this, we periodically compute the mean squared error between V_φ and Monte Carlo returns on a held-out set of 512 states and log a warning if MSE exceeds 0.1. (3) We use self-play without human demonstrations because self-play can explore the space of workflows comprehensively; the trade-off is that the initial policy is random and may visit low-reward regions, but CFR's regret minimization ensures convergence to a Nash equilibrium, which corresponds to optimal workflow distribution given the reward function. (4) We separate policy and value networks to enable off-policy learning and reduce variance, accepting the cost of maintaining two separate architectures and additional training overhead.

(D) **Why it measures what we claim**: The counterfactual regret for each action measures the opportunity loss of not taking that action, operationalizing the concept of "action optimality". The assumption is that the counterfactual value Q (estimated via Monte Carlo) accurately reflects the expected reward if that action were chosen; this holds when the rollouts are representative and the environment is deterministic. When rollouts are insufficient (due to variance or limited count), regret estimates become noisy, and the policy may converge to a suboptimal equilibrium. The value network V_φ(s) approximates the state value, which measures the expected return from that state under the current policy. The assumption is that V_φ(s) generalizes across unseen states via the learned representation; this fails when the state embedding lacks discriminatory features, causing biased value estimates and inaccurate regret. The terminal reward R measures the final retrieval quality, which is assumed to be a sufficient proxy for search success; this fails if the user's goal is not fully captured by one-shot retrieval metrics (e.g., diversity or novelty), in which case the induced workflow optimizes an incomplete objective.

## Contribution

(1) A self-play algorithm for unsupervised workflow induction that eliminates the need for human demonstrations or synthetic data by leveraging counterfactual regret minimization. (2) A demonstration that neural CFR can be applied to DAG-structured action spaces in scientific literature search, providing a principled credit assignment mechanism without per-step rewards. (3) A framework for benchmarking induced workflows using existing retrieval metrics, facilitating comparison with supervised methods.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | RealScholarQuery (200 queries), BioASQ (50 biomedical queries), LegalCase (50 legal queries) | Standard benchmarks covering diverse domains |
| Primary metric | nDCG@10 | Reflects ranking quality at top |
| Baseline 1 | PaSa | Strongest published agent |
| Baseline 2 | Google Scholar | Traditional search baseline |
| Baseline 3 | GPT-4o agent | LLM baseline without workflow |
| Ablation | DAG-CFR without value network (using Monte Carlo V estimate with 100 rollouts) | Isolates effect of value estimation |

### Why this setup validates the claim

RealScholarQuery contains diverse, realistic queries that require multi-step reasoning and tool use. nDCG@10 directly measures the quality of the final ranked list, which is the objective our method optimizes. Comparing against PaSa (a strong learned agent) and Google Scholar (a static but effective baseline) tests whether our self-play induced workflow surpasses both learned and heuristic approaches. The GPT-4o baseline controls for the effect of LLM reasoning without explicit workflow induction. The ablation (removing the value network) tests whether the value head provides meaningful advantage in regret estimation. Including biomedical (BioASQ) and legal (LegalCase) datasets tests generalization across domains. This combination ensures that if our method succeeds, it must be due to the CFR-based workflow search, not simply due to neural function approximation or LLM capabilities.

### Expected outcome and causal chain

**vs. PaSa** — On a query like "methods for few-shot learning in NLP", PaSa may get stuck in a local exploration because its policy is trained on a fixed set of trajectories. For example, after an initial keyword search, PaSa might repeatedly cite-expand from a narrow set of papers, missing diversity. Our DAG-CFR method, by exploring random workflows and minimizing regret, would discover that an early filtering operator followed by a citation expansion from a different cluster yields higher diversity and relevance. We expect DAG-CFR to achieve a noticeable gap (e.g., 0.05-0.10 nDCG@10) on queries requiring broad exploration, with parity on simple queries. This gap is expected to hold across domains.

**vs. Google Scholar** — On a query like "recent advances in transformer architectures", Google Scholar returns a generic list sorted by citation count, missing recent impactful papers that have low citation counts. Our method, through iterative operator selection, can combine citation expansion from a known recent paper and a keyword filter for confidences. The DAG workflow may choose to first retrieve from a specialized embedding model then re-rank, leading to inclusion of recent work. We expect DAG-CFR to outperform Google Scholar by a large margin (0.15-0.20 nDCG@10) on queries where temporal relevance matters, but less on established topics.

**vs. GPT-4o agent** — On a query like "explainable AI for medical diagnosis", a GPT-4o agent given a toolset may try to generate a query directly from its knowledge, potentially missing niche papers. Because it has no exploration mechanism, it may converge to a suboptimal plan. Our DAG-CFR, through regret-guided exploration, will try multiple tool sequences (e.g., first filter by year, then search with specific terms) and find one that retrieves specialized medical AI papers. We expect DAG-CFR to show a consistent advantage (e.g., 0.08-0.12 nDCG@10) on domain-specific queries, with smaller differences on general topics.

**Calibration check**: If the value network MSE on the heldout set exceeds 0.1, we expect performance to degrade; we will report results both with and without the calibration warning to validate the load-bearing assumption.

### What would falsify this idea
If DAG-CFR performs no better than the ablation without the value network on queries requiring long-term credit assignment, or if it is outperformed by PaSa on queries that require adaptive planning (the very scenario our method targets), then the central claim that CFR with neural function approximation improves workflow induction would be falsified.

## References

1. Multi-Turn Agentic Scientific Literature Search via Workflow Induction
2. PaSa: An LLM Agent for Comprehensive Academic Paper Search
3. ChatCite: LLM Agent with Human Workflow Guidance for Comparative Literature Summary
4. SPAR: Scholar Paper Retrieval with LLM-based Agents for Enhanced Academic Search
5. Language agents achieve superhuman synthesis of scientific knowledge
6. Target-aware Abstractive Related Work Generation with Contrastive Learning
7. G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment
8. LitLLM: A Toolkit for Scientific Literature Review
