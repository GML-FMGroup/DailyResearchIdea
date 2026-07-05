# MUBA: Marginal Utility-Based Adaptive Tool Selection and Budget Allocation for Multimodal Agents

## Motivation

Existing iterative multimodal agents, such as ProMSA, rely on fixed tool sets and per-dataset manually tuned budgets, which are suboptimal for heterogeneous queries where the marginal benefit of each tool varies. This structural limitation arises because the agent has no mechanism to reason about when to use which tool and how many calls to make, treating tool selection and budget as static hyperparameters. Without adaptive allocation, the agent either wastes resources on low-utility calls or misses critical information, leading to inefficient and costly reasoning.

## Key Insight

By framing tool selection and budget allocation as a resource allocation problem with diminishing returns, the optimal per-instance allocation satisfies an equi-marginal utility condition that can be enforced through learned utility predictors and a greedy selection rule.

## Method

**(A) What it is**: MUBA (Marginal Utility-Based Allocation) is an iterative multimodal agent that dynamically selects tools and allocates call budgets per instance by learning a utility predictor for each tool and applying an equi-marginal utility condition for budget allocation. Input: image and question; Output: answer.

**(B) How it works**:
```python
def MUBA(question, image, budget_frac):
    state = encode(question, image)  # ViT encoder, hidden=768, 12 layers
    max_cost = sum(cost[t] for t in tools)  # cost per call: 1 unit each
    budget = budget_frac * max_cost
    usage = {t:0 for t in tools}
    while budget > 0 and not stop_signal(state):
        marg_utils = {}
        for t in tools:
            if usage[t] >= max_calls[t]: continue  # max_calls: image_search=3, text_search=5
            mu = utility_predictor[t](state, usage[t])  # 2-layer MLP hidden=256, GeLU, trained offline
            marg_utils[t] = mu / cost[t]  # utility per unit cost
        if not marg_utils: break
        best_t = argmax(marg_utils)
        if marg_utils[best_t] < threshold: break  # threshold=0.01, calibrated via Platt scaling on held-out set
        result = execute_tool(best_t, state)  # tool outputs: image search returns top-5 images, text search returns top-5 passages
        state = update_state(state, result)  # concatenate embeddings, projection to 768
        budget -= cost[best_t]
        usage[best_t] += 1
    answer = generate_answer(state)  # GPT-2 decoder, 12 layers, vocab size 30522
    return answer
```
**Load-bearing assumption**: Utility predictors provide well-calibrated estimates of expected accuracy gain from one additional tool call, i.e., E[Δacc | state, usage[t]] ≈ predictor output. This assumption is critical for equi-marginal allocation to be optimal. To ensure calibration, utility predictors are trained offline on 10,000 trajectories (collected by a random policy) with a calibration set of 512 held-out trajectories used to tune a temperature τ (Platt scaling) for each predictor, target ECE<0.05. The stop threshold is cross-validated on this calibration set.

**(C) Why this design**: We chose a greedy marginal utility allocation over a learned policy (e.g., PPO) because the equi-marginal condition provides a closed-form optimality principle that generalizes across query types without requiring extensive RL on long trajectories; this accepts the cost that the utility predictor must be well-calibrated for the condition to hold. We use per-tool utility predictors rather than a joint selector because it enables modular training and reuse across different tool sets; the trade-off is potential covariance neglect that a joint model could capture but at the cost of reduced transferability. We set the stop threshold as a learned parameter via RL (or zero) rather than hardcoded because optimal stopping depends on the noise in utility estimates; this adds training complexity but avoids systematic under- or over-calling. This design contrasts with ProMSA, which uses fixed budgets and a learned action policy without explicit utility modeling; a reviewer would not describe MUBA as a variant of ProMSA because MUBA's core mechanism is resource allocation via marginal utilities, not learned action selection.

**(D) Why it measures what we claim**: The marginal utility estimate U_t'(x_t) measures the expected improvement in answer correctness from one additional call to tool t given current state, because we define the utility predictor's training target as the observed accuracy gain when making that additional call; this assumption fails when the tool has non-local effects (e.g., a search result changes future utilities of other tools), in which case U_t' captures only immediate improvement but not cross-tool synergies, reflecting instead a local greedy bias. The usage count x_t is used as input to model diminishing returns, measuring saturation; it assumes that past calls to the same tool reduce marginal benefit, which holds empirically for retrieval tools but fails when later calls are complementary to earlier ones (e.g., diverse search terms), in which case the model may underestimate utility. The cost ratio mu/cost operationalizes the concept of budget efficiency: it measures utility per unit cost, assuming costs are additive and independent; this fails when tools share resources or have synergy costs, in which case the metric reflects only isolated efficiency rather than joint cost effects.

Specifically, the marginal utility predictor U_t'(x_t) measures the expected immediate accuracy gain from an additional call to tool t given current state. **Assumption A**: the predictor is unbiased and conditionally independent of future state changes. **Failure mode F**: when tool calls have synergistic or non-local effects (e.g., one search enables another), the immediate gain underestimates true value, breaking the equi-marginal allocation optimality.

## Contribution

(1) A framework (MUBA) that jointly learns adaptive tool selection and budget allocation via per-tool utility predictors and an equi-marginal allocation rule, enabling per-instance optimization. (2) The identification and exploitation of diminishing returns in multimodal tool use for efficient allocation, supported by empirical characterization of utility curves. (3) A training procedure combining supervised utility learning with reinforcement learning fine-tuning for budget-sensitive rewards.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | E-VQA, Slake-VQA (medical) | Tests multimodal search and cost sensitivity in medical domain. |
| Primary metric | Accuracy, ECE (Expected Calibration Error) | Accuracy measures correctness; ECE validates utility predictor calibration. |
| Baseline 1 | ProMSA | Learned action policy baseline. |
| Baseline 2 | SenseNova-MARS | RL-based agent with fixed budget. |
| Baseline 3 | Strong RAG | No iterative tool use. |
| Ablation | MUBA-random | Random tool allocation. |

### Why this setup validates the claim
This combination forms a falsifiable test because it isolates the core claim: marginal utility allocation with equi-marginal stopping improves over fixed-budget learned policies and non-iterative retrieval. Comparing against ProMSA tests whether our utility-driven allocation beats a learned action policy on average. SenseNova-MARS tests whether our dynamic budget allocation (via threshold) outperforms RL-based fixed budgets. Strong RAG tests the necessity of iterative tool use. The ablation (random allocation) directly measures the contribution of the utility predictor. Accuracy is a direct metric for answer correctness; ECE ensures that the utility predictor is well-calibrated, a prerequisite for the equi-marginal condition to be optimal. The addition of a medical dataset (Slake-VQA) tests cost sensitivity in a high-stakes domain where wasted calls are costly.

### Expected outcome and causal chain

**vs. ProMSA** — On a case requiring many search calls (e.g., multi-step entity disambiguation), ProMSA's learned policy may under-allocate due to sparse training examples; it produces incomplete retrieval and wrong answer. Our method instead allocates more calls because marginal utility remains high, so we expect a noticeable gap on queries needing many retrievals but parity on simple ones.

**vs. SenseNova-MARS** — On a case with easy first search but diminishing returns (e.g., identifying a common object), MARS' fixed budget wastes calls on redundant retrievals; its RL training overfits to specific budget lengths. Our method stops early via threshold when utility drops, so we expect higher accuracy per cost, visible as better accuracy at the same average call count.

**vs. Strong RAG** — On a complex visual question requiring multiple reasoning steps (e.g., 'What is the relation between the two animals?'), RAG's single retrieval fails to capture the context; it produces wrong answer. Our iterative process refines state with successive tool calls, so we expect a significant gap on compositional questions but parity on simple ones.

**vs. MUBA-random** — On any instance, random allocation wastes budget on low-utility tools or over-calls, leading to lower accuracy. Our method outperforms consistently, confirming that the utility predictor guides efficient allocation.

### What would falsify this idea
If our gain is uniform across question difficulty or budget levels (rather than concentrated on queries needing many calls or early stopping), it would indicate the marginal utility model isn't driving improvement. Additionally, if ECE on a held-out set exceeds 0.1, the utility predictors are miscalibrated, undermining the equi-marginal condition and contradicting the central claim.

## References

1. ProMSA:Progressive Multimodal Search Agents for Knowledge-Based Visual Question Answering
2. SenseNova-MARS: Empowering Multimodal Agentic Reasoning and Search via Reinforcement Learning
3. Group Sequence Policy Optimization
4. VisRAG: Vision-based Retrieval-augmented Generation on Multi-modality Documents
