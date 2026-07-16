# Invariant Causal Reward Transfer for Unverifiable Reasoning Tasks

## Motivation

Existing RL methods for reasoning (e.g., Ring-Zero, ProRL, GSPO) all require a verifiable reward function, limiting their application to tasks where correctness cannot be automatically checked. The root cause is that these methods treat reward as a domain-specific signal, ignoring the fact that the causal effect of reasoning steps on answer correctness is invariant across domains that share the same underlying structure. We address this by learning a reward function that captures this invariant causal effect from a source domain with verifiable rewards and transferring it to unverifiable domains.

## Key Insight

The causal effect of a reasoning chain on answer correctness is invariant across domains that share the same causal structure, enabling reward transfer via invariant risk minimization.

## Method

ICRT (Invariant Causal Reward Transfer) learns a domain-invariant reward function for reasoning by modeling the causal effect of reasoning steps on answer correctness and applying Invariant Risk Minimization (IRM) to isolate the invariant causal component. Input: a set of reasoning traces from a source domain with binary correct/incorrect labels. Output: a reward model r(c) that outputs a scalar reward for a reasoning chain c.

Assumption (load-bearing): The causal effect of reasoning steps on answer correctness is invariant across domains that share the same underlying causal structure. The perturbation-based environments (Phase 1) capture variation in spurious correlations without altering the causal effect. This assumption may fail if perturbations accidentally corrupt causal features (e.g., shuffling sentence order breaks logical coherence) or if the target domain has a fundamentally different causal structure.

(B) How it works:
```python
# Phase 1: Construct environments from source domain
# Env1: original traces
# Env2: traces with perturbed spurious features (e.g., add irrelevant tokens with probability 0.2)
# Env3: traces with shuffled sentence order (shuffle probability 0.2)
# Phase 2: Learn representation phi (2-layer MLP, hidden=256, GeLU) and linear predictor w via IRM
# For each environment e, compute risk R_e(w, phi) = E[ -log sigma(w(phi(X))) for correct Y, -log(1-sigma(...)) for incorrect ]
# Solve: min_{phi, w} sum_e R_e(w, phi) + lambda * sum_e ||grad_{w} R_e(w, phi)||^2
# Hyperparameters: lambda=0.1, environments created by random perturbation probability 0.2
# Phase 3: At target domain, reward = w(phi(X_test))
# Phase 4: Calibration: On a held-out calibration set of 512 source domain examples, compute Pearson correlation r between reward and correctness. If r < 0.5, increase lambda to 0.5 and re-run Phase 2 (up to 3 attempts). If no calibration set available, proceed without calibration.
```

(C) Why this design: We chose IRM over domain-adversarial methods because IRM explicitly enforces that the optimal predictor is invariant across environments, while adversarial methods only align feature distributions without guaranteeing causal invariance. We constructed environments by perturbing spurious features (adding irrelevant tokens, shuffling order) rather than splitting by domain labels, because source domain may lack natural domain splits; this perturbation forces the representation to ignore spurious correlations. We chose a linear predictor on top of phi for interpretability and to avoid overfitting, accepting that the reward function may be less expressive than a deep network; however, the nonlinear representation phi can capture complex relationships, and the linear head prevents the model from relying on non-causal patterns that differ across environments. The trade-off of using fixed perturbation probabilities is that if the perturbation is too strong, it may destroy causal signal; we set probability 0.2 to balance invariance learning and preservation of causal content.

(D) Why it measures what we claim: The computational quantity `phi(X)` measures the invariant causal effect of reasoning steps on correctness because IRM forces the predictor to be optimal across environments that differ in spurious associations; this relies on the assumption that the causal effect of X on Y is stable across environments while spurious correlations change. This assumption fails when the causal effect itself changes across domains (e.g., different reasoning strategies have different causal effects), in which case `phi(X)` may reflect a confound across domains rather than a pure causal effect. However, for reasoning tasks sharing a common underlying causal structure (e.g., mathematical deduction), the invariance assumption holds. The regularizer `||grad_{w} R_e||^2` ensures that the representation is not tailored to any particular spurious pattern, directly operationalizing the concept of causal invariance. **Failure mode**: Perturbations (e.g., shuffling sentence order) may accidentally corrupt causal features (e.g., logical coherence), causing the invariant predictor to miss causal signals. This is mitigated by the calibration step (Phase 4), which detects poor alignment and adapts the regularization strength.

## Contribution

(1) A novel method ICRT that learns a domain-invariant reward function for reasoning by combining causal modeling with invariant risk minimization, enabling RL training on unverifiable reasoning tasks. (2) A practical procedure for constructing training environments via input perturbation (adding irrelevant tokens, reordering) to enforce causal invariance without requiring multiple natural domains. (3) Identification of the invariant causal effect of reasoning steps on answer correctness as a transferable signal.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|----------|
| Dataset | GSM8K | Standard math reasoning benchmark |
| Primary metric | Solution accuracy | Direct measure of correct reasoning |
| Baseline 1 | Ring-Zero | Standard zero RL without transfer |
| Baseline 2 | ProRL | Prolonged RL training |
| Baseline 3 | GSPO | Group-based policy optimization |
| Ablation | ICRT w/o IRM | Direct reward learning on source |

### Why this setup validates the claim

This setup tests the central claim that invariant causal reward transfer improves reasoning generalization. Comparing against Ring-Zero evaluates whether reward transfer offers any advantage over training from scratch. Against ProRL, we test if our method's sample efficiency stems from invariance rather than extended training. GSPO checks if our reward function outperforms an alternative policy optimization that may amplify spurious correlations. The ablation isolates the role of IRM by removing the invariance constraint. Solution accuracy directly measures the end-task performance, providing a clear signal of whether learning the true causal reward leads to better reasoning. The combination of baselines and ablation creates a falsifiable test: if ICRT outperforms only when spurious correlations shift across domains, the invariance mechanism is validated.

### Expected outcome and causal chain

**vs. Ring-Zero** — On a target domain instance where a spurious feature (e.g., keyword "therefore") correlates with correctness in source but not target, Ring-Zero's policy learns to rely on that feature, causing incorrect reward assignment and low accuracy. Our method, via IRM, ignores such spurious correlations by enforcing optimal prediction across perturbed environments, so it correctly rewards causal steps. We expect ICRT to achieve ~15% higher accuracy on instances with shifted spurious patterns, but similar accuracy on simple instances.

**vs. ProRL** — On a complex multi-step reasoning instance, ProRL's prolonged training may eventually overfit to spurious cues in the source domain, plateauing at moderate accuracy. Our method's invariant reward provides consistent signal, enabling the policy to improve further by focusing on causal structure. We expect ICRT to show faster convergence and a final accuracy ~10% higher than ProRL on hard problems, with comparable performance on easy ones.

**vs. GSPO** — When groups in GSPO are formed based on surface-level features (e.g., problem length) that are spurious, the group-level optimization can reinforce those spurious correlations, causing poor generalization. Our method avoids this by learning a reward that is invariant to such group definitions. We expect ICRT to outperform GSPO by ~12% on cross-group evaluation where group boundaries do not align with causal structure.

**vs. ICRT w/o IRM** — On target instances where the source domain has a strong spurious correlation (e.g., all correct answers contain a certain word), the ablation learns to rely on that word, failing on target where it is absent. ICRT with IRM discards that spurious cue, maintaining high accuracy. We expect ICRT to be ~20% more accurate than the ablation on such distribution-shifted instances.

### What would falsify this idea

If the accuracy gains of ICRT over baselines are uniform across all problem subsets (e.g., equally large on both invariant and spurious-correlation-shifted instances), rather than concentrated where the predicted failure mode occurs, then the invariance mechanism is not the cause of improvement and the central claim is wrong. Alternatively, if the ablation (ICRT w/o IRM) performs comparably to the full method, then IRM is unnecessary, falsifying the necessity of causal invariance.

## References

1. Ring-Zero: Scaling Zero RL to a Trillion Parameters for Emergent Reasoning
2. ProRL: Prolonged Reinforcement Learning Expands Reasoning Boundaries in Large Language Models
3. Group Sequence Policy Optimization
4. Every Step Evolves: Scaling Reinforcement Learning for Trillion-Scale Thinking Model
5. Pass@k Training for Adaptively Balancing Exploration and Exploitation of Large Reasoning Models
6. Rethinking Sample Polarity in Reinforcement Learning with Verifiable Rewards
7. AReaL: A Large-Scale Asynchronous Reinforcement Learning System for Language Reasoning
8. DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models
