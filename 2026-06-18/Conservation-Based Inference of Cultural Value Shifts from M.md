# Conservation-Based Inference of Cultural Value Shifts from Multi-Agent Negotiation Agreements

## Motivation

Existing approaches such as 'Beyond Alignment' rely on static World Values Survey (WVS) benchmarks to measure cultural diversity in multi-agent systems, but they cannot capture real-time value dynamics during interactions. Social interaction erodes diversity and alignment reduces conceptual diversity, yet we lack methods to track how values evolve without repeatedly administering full surveys—an expensive and impractical process that assumes stationarity. The root cause is that surveys are static, but multi-agent negotiations produce observable agreement patterns that encode hidden value shifts under structural conservation laws (e.g., zero-sum trade-offs), which remain unexploited.

## Key Insight

In multi-agent negotiations over culturally charged issues, the total of each value dimension across agents is conserved (e.g., a gain in one agent's 'hedonism' exactly offsets a loss in another's 'security'), enabling closed-form inference of individual value shifts from a few probe responses without full retesting.

## Method

(A) **What it is**: CoValIn (Conservation-based Value Inference) is a method that takes as input initial value vectors (from WVS) for N agents and a set of observed negotiation outcomes (issues and concessions), and outputs inferred final value vectors per agent.

(B) **How it works** (pseudocode):

```python
# CoValIn
# Input: V0 (N x K initial value vectors), outcomes (list of (agent, issue, delta))
# Output: V (N x K inferred value vectors)

1. For each dimension k: impose conservation constraint sum_i V[i,k] = sum_i V0[i,k]
2. Define mapping matrix W (K x M) from Schwartz's circumplex (fixed, from psychological literature)
3. For each outcome (agent i, issue j, concession delta):
   - Assume linear model: delta = sum_k W[k,j] * (V[i,k] - V0[i,k])
4. Collect all equations (including conservation constraints) into linear system A * x = b, x size N*K
5. Solve via least squares with Tikhonov regularization (lambda=0.1): x = argmin ||A x - b||^2 + lambda * ||x||^2
6. Reshape x to (N,K) and output V = V0 + x
```

(C) **Why this design**: We chose a linear mapping between concessions and value shifts (step 3) because it provides a tractable closed-form solution and aligns with the conservation law assumption; the cost is that real human value trade-offs may be nonlinear, but this linearity is a first approximation consistent with the frontier's focus on structural invariants. We use least squares with Tikhonov regularization (step 5) rather than a Bayesian approach to avoid specifying prior distributions, accepting that regularization does not quantify uncertainty. We fix the mapping matrix W from Schwartz's circumplex rather than learning it, because we need interpretability and to avoid overfitting to negotiation-specific domains; the trade-off is that W may not perfectly capture issue-value associations in all cultures. These choices prioritize analytical tractability and reproducibility over expressiveness.

(D) **Why it measures what we claim**: The computational quantity `delta = sum_k W[k,j] * (V[i,k] - V0[i,k])` measures the value shift of agent i on dimension k because we assume that concessions during negotiation are linear functions of value changes and that issues load onto dimensions via a known psychological mapping. This assumption fails when agents' preferences are not linear (e.g., threshold effects, lexicographic preferences), in which case the measured `delta` reflects only the linear projection of the shift onto the issue-space, potentially missing abrupt changes. The conservation constraint `sum_i V[i,k] = constant` measures that the total sum of a value dimension is conserved because we assume the multi-agent system is closed and zero-sum in that dimension; this assumption fails when value dimensions are not zero-sum (e.g., both agents can gain in benevolence through cooperation), in which case the conservation constraint imposes a false invariance and the inferred shifts will be distorted.

## Contribution

(1) A novel method, CoValIn, that leverages conservation laws in multi-agent negotiations to infer hidden cultural value shifts from observed agreement patterns, enabling online value tracking without resurveying. (2) An operationalization of value dynamics through linear systems, showing that under zero-sum conditions, value shifts are identifiable from a small number of negotiation probes. (3) A framework for integrating Schwartz's circumplex model into negotiation outcome analysis, providing a principled mapping between concrete negotiation issues and abstract value dimensions.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | Simulated multi-agent negotiations | Known ground truth value shifts |
| Primary metric | RMSE of inferred value shifts | Direct accuracy measure |
| Baseline 1 | No inference (initial vectors) | Measures need for inference |
| Baseline 2 | Simple average of observed shifts | Naive baseline |
| Ablation | CoValIn w/o conservation constraint | Tests conservation importance |

### Why this setup validates the claim

The central claim is that CoValIn accurately infers value shifts from negotiation outcomes using conservation and linear mapping. Using simulated data with known ground truth allows direct measurement of inference accuracy. The RMSE metric captures deviation, and baseline comparisons isolate the contribution of the conservation constraint and linear mapping. The ablation tests whether conservation is necessary; if CoValIn outperforms both baselines and the ablation, the claim is supported. This design ensures falsifiability by targeting conditions where the method's mechanisms should matter.

### Expected outcome and causal chain

**vs. No inference (initial vectors)** — On a case where agents' values shift significantly during negotiation (e.g., both sides concede on conservation vs. openness issues), the baseline predicts no change, yielding high RMSE because it ignores outcome information. Our method uses observed concessions to update values via the linear mapping and conservation, so we expect RMSE close to the noise floor on such cases, with a >50% reduction compared to baseline on high-shift subsets.

**vs. Simple average of observed shifts** — On a case where shifts are heterogeneous (e.g., some agents shift pro-socially while others shift pro-self), the baseline assigns a uniform shift to all agents, producing poor per-agent accuracy. Our method infers per-agent shifts using the mapping matrix and regularization, so we expect lower RMSE, especially a >30% gap on high-variance subsets.

### What would falsify this idea

If CoValIn's RMSE is not significantly lower than both baselines on high-shift and high-variance subsets, or if the ablation (without conservation) performs similarly to the full method on cases where conservation is expected to bind (e.g., zero-sum dimensions), then the central claim that conservation and linear mapping are key is wrong.

## References

1. Beyond Alignment: Value Diversity as a Collective Property in Multicultural Agent Systems
2. One fish, two fish, but not the whole sea: Alignment reduces language models' conceptual diversity
3. On the Dynamics of Multi-Agent LLM Communities Driven by Value Diversity
4. Evolving AI Collectives to Enhance Human Diversity and Enable Self-Regulation
5. Value FULCRA: Mapping Large Language Models to the Multidimensional Spectrum of Basic Human Value
6. Turning large language models into cognitive models
7. Llama 2: Open Foundation and Fine-Tuned Chat Models
8. Efficiently Scaling Transformer Inference
