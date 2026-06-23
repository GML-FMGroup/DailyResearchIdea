# Meta-Pairwise Calibration for Domain-Adaptable LLM Action Scoring

## Motivation

Existing LLM-based action evaluators, such as Monte Carlo Planning with Large Language Model for Text-Based Game Agents (MC-DML), rely on static LLM outputs and do not update scoring based on in-domain interactions, leading to miscalibrated scores in specific domains. The root cause is the absence of a mechanism to incorporate domain-specific pairwise preferences without costly fine-tuning. MC-DML uses memory but not a structured adaptation of the scoring function itself.

## Key Insight

Pairwise preferences from a target domain impose a partial order that can be used to learn a monotonic transformation of LLM scores, with a meta-learned prior preventing overfitting from few comparisons.

## Method

Meta-Pairwise Calibration (MPC) is a lightweight module that takes an LLM's raw action scores and a small set of pairwise comparisons from a target domain, and outputs domain-adapted scores via a meta-learned monotonic calibration network with ranking loss.

**How it works (pseudocode):**
```python
# Define monotonic calibration network:
#   f(s; θ) = W2 * softplus( softplus(W1) * s + b1 ) + b2
#   where W1, W2 are scalar weights, b1,b2 are biases, all parameters θ = (W1, W2, b1, b2)
#   Monotonicity: softplus ensures W1>0, W2>0, and activation is non-decreasing → f is monotonic increasing in s.

# Meta-training (offline, across domains)
For each domain in meta-training set (e.g., 8 Jericho games):
    Collect LLM scores s_i and pairwise comparisons C = {(a,b,pref)}
    For each domain, find optimal parameters θ* by minimizing:
        L(θ) = Σ_{(a,b,pref) in C} max(0, 1 - pref*(f(s(a)) - f(s(b)))) + λ_reg * (||θ||^2)   # λ_reg = 0.01
    Store θ* across domains
Learn prior parameters θ_prior and diagonal precision τ from collected θ* (e.g., mean and inverse variance)

# Meta-test (adaptation to new domain)
Input: LLM scores s_new, few comparisons C_new (e.g., 20 pairs), θ_prior, τ, λ_meta (set λ_meta = 1.0)
Optimize over θ:
    L(θ) = Σ_{(a,b,pref) in C_new} max(0, 1 - pref*(f(s_new(a)) - f(s_new(b)))) + λ_meta * (θ - θ_prior)^T diag(τ) (θ - θ_prior)
Output: Calibrated scores s'(a) = f(s_new(a))
```

**Why this design:** We chose a learned monotonic neural network (2-layer with hidden size 1? Actually scalar input, but we use a simple 2-layer with 1 hidden neuron? To keep low capacity, we use a network with 2 hidden units and softplus activations; total 6 parameters) over a simple affine transformation because it can capture systematic non-linear but monotonic biases (e.g., diminishing returns). This design is computationally efficient and regularized by the meta-prior to prevent overfitting from few comparisons. We accept that some non-monotonic errors may be missed, but the monotonicity assumption is verified empirically across training domains. We selected the hinge ranking loss over cross-entropy because it is robust to label noise and does not require a probability model, at the cost of ignoring preference intensity. We incorporated a meta-learned prior instead of a fixed L2 regularizer because it transfers knowledge from similar domains, making adaptation more sample-efficient, but requires a meta-training corpus with diverse domains.

**Why it measures what we claim:** The adapted score s'(a) = f(s(a)) measures the target-domain action quality under the assumption that pairwise preferences are consistent with a latent monotonic transformation of the LLM score; the hinge loss enforces that s' respects observed preferences, so s' approximates the true quality order. This assumption fails when preferences are noisy or intransitive, in which case s' reflects a compromise that may not correspond to any consistent quality. The regularization term λ_meta (θ-θ_prior)^T diag(τ) (θ-θ_prior) measures domain similarity to the training distribution; it prevents overfitting but assumes the target domain is drawn from the same meta-distribution, failing if the domain is entirely novel, causing the adapted scores to be biased towards the prior.

**D. Assumption (proxy metric grounding):** The monotonic transformation learned from a few pairs extrapolates to all pairs; failure occurs when held-out pairs involve actions outside the order of training pairs (F). We verify the monotonicity assumption by computing Spearman rank correlation between raw LLM scores and true quality (from held-out pairs) across meta-training domains; we discard domains with correlation <0.5 to ensure reasonable monotonicity.

## Contribution

(1) A novel framework, Meta-Pairwise Calibration, for domain-adaptable LLM action scoring that uses few pairwise comparisons to calibrate scores via a meta-learned prior. (2) A design principle that a simple affine transformation with ranking loss and meta-regularization enables efficient adaptation without fine-tuning, addressing the structural gap of static LLM evaluation.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | Jericho text games (10 domains) | Diverse action spaces, LLM scores available |
| Primary metric | Pairwise preference accuracy | Directly measures calibration quality |
| Baseline | Raw LLM scores | No calibration baseline |
| Baseline | Platt scaling (few-shot) | Standard calibration baseline |
| Baseline | Prior-only (no adaptation) | Tests necessity of adaptation |
| Ablation | MPC without meta-prior (L2 to zero) | Isolates benefit of meta-learning |

### Why this setup validates the claim
Using Jericho text games provides multiple distinct domains (different game environments) with varying decision spaces, enabling meta-training across domains and testing on an unseen domain with few comparisons. Pairwise preference accuracy on held-out pairs directly reflects whether calibrated scores respect observed preferences and generalize. Baselines isolate key claims: Raw scores test if calibration is needed; Platt scaling tests data efficiency; Prior-only tests whether adaptation is necessary. The ablation tests whether the meta-learned prior provides additional benefit over generic regularization. This combined setup forms a falsifiable test: if MPC outperforms all baselines and the ablation on held-out pairs, it confirms that meta-learning an affine transformation with a prior effectively transfers knowledge across domains, using few target comparisons.

### Expected outcome and causal chain

**vs. Raw LLM scores** — On a domain where LLM scores are systematically biased (e.g., overconfident in a frequent but suboptimal action), raw scores produce incorrect rankings because they are uncalibrated. Our method uses a few pairwise comparisons to learn an affine shift and scale that corrects this bias, so we expect that MPC achieves significantly higher pairwise accuracy on that domain, especially on pairs involving the biased action.

**vs. Platt scaling (few-shot)** — On a domain with very few comparisons (e.g., only 10 pairs), Platt scaling overfits to those examples because it has no regularization, leading to poor generalization. Our method uses a meta-learned prior to constrain the affine parameters, preventing overfitting. Thus, we expect MPC to maintain high accuracy on held-out pairs, while Platt scaling performance drops to near random.

**vs. Prior-only (no adaptation)** — On a novel domain that differs from the meta-training distribution (e.g., a game with unique mechanics), the prior-only method outputs a transformation averaged across training domains, which may be mismatched. Our method adapts using the few target comparisons, adjusting the transformation to the domain. Therefore, we expect MPC to outperform prior-only on such domains, especially when the raw scores are miscalibrated in a direction opposite to the prior.

### What would falsify this idea
If the improvement of MPC over raw scores is uniform across all domains rather than concentrated in domains where raw scores are miscalibrated (e.g., high bias), then the method is not addressing the intended failure mode of score calibration.

## References

1. Monte Carlo Planning with Large Language Model for Text-Based Game Agents
2. Large Language Models as Commonsense Knowledge for Large-Scale Task Planning
3. Ghost in the Minecraft: Generally Capable Agents for Open-World Environments via Large Language Models with Text-based Knowledge and Memory
4. Inner Monologue: Embodied Reasoning through Planning with Language Models
5. Interactive Language: Talking to Robots in Real Time
6. Perceiver-Actor: A Multi-Task Transformer for Robotic Manipulation
7. ProgPrompt: Generating Situated Robot Task Plans using Large Language Models
8. Chain of Thought Prompting Elicits Reasoning in Large Language Models
