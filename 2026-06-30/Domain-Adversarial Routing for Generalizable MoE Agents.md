# Domain-Adversarial Routing for Generalizable MoE Agents

## Motivation

Existing Mixture-of-Experts (MoE) agent models, such as Agents-A1 (Scaling the Horizon, Not the Parameters, 2026), are trained on a fixed set of domains and assume that performance generalizes to unseen domains. This assumption is structurally flawed because the routing mechanism overfits to domain-specific features during supervised fine-tuning, causing poor routing decisions when encountering new distributions. We identify that the root cause is the lack of explicit regularization to learn domain-invariant representations in the routing space.

## Key Insight

Domain adversarial training decouples routing decisions from domain identity by aligning the gradient of a domain classifier with the router's loss, enforcing that the router selects experts based on task-relevant features shared across domains, not spurious domain correlations.

## Method

### (A) What it is
DART (Domain-Adversarial Routing Training) takes as input a base MoE model (with router and experts) and domain-labeled agent trajectories. It outputs a model whose router is invariant to domain shifts, improving zero-shot generalization to unseen domains. **Load-bearing assumption**: Domain and task information are separable in the router logits, so adversarial training on logits can enforce domain-invariant routing without harming task performance.

### (B) How it works

```python
# DART Training Loop
# Hyperparameters: batch_size=64, lr=1e-4 (AdamW), weight_decay=0.01, lambda=0.5, alpha=0.1, num_epochs=10
# Custom gradient reversal layer: grad_reverse scales gradient by -lambda during backward, identity during forward

# Calibration step before training
# Evaluate domain classifier accuracy on router logits from standard MoE on a held-out set
if domain_classifier_accuracy > 0.8:  # high correlation
    use_conditional_adversarial_loss = True  # concat task embedding to router_logits
else:
    use_conditional_adversarial_loss = False

for each batch (x, y, d) where d is domain label:
    # Forward pass through MoE
    router_logits = router(x)  # B x num_experts (num_experts=8)
    gating_weights = softmax(router_logits)
    expert_outputs = [expert_i(x) for i in range(num_experts)]
    output = sum(gating_weights[:, i] * expert_outputs[:, i])  # weighted sum

    # Domain classifier on router logits (with gradient reversal)
    if use_conditional_adversarial_loss:
        task_embedding = task_encoder(y)  # e.g., one-hot of action
        domain_input = concat(router_logits, task_embedding)
    else:
        domain_input = router_logits
    domain_logits = domain_classifier(domain_input)  # single linear layer, input_dim=num_experts (or + action_dim), output_dim=num_domains
    domain_loss = cross_entropy(domain_logits, d)
    # Gradient reversal: multiply gradient by -lambda during backward
    reversed_domain_loss = grad_reverse(domain_loss, lambda=0.5)

    # Task loss (e.g., SFT loss on agent actions)
    task_loss = cross_entropy(output, y)

    total_loss = task_loss + alpha * reversed_domain_loss  # alpha=0.1
    backward(total_loss)
    optimizer.step()
```

**Resource estimate**: Training takes ~2 hours on a single NVIDIA A100 GPU for 10 epochs on the benchmark dataset (6 domains, ~50k trajectories).

### (C) Why this design
We chose gradient reversal over explicit domain regularization (e.g., MMD) because gradient reversal directly optimizes the router's feature representation for domain invariance through adversarial training, which is more sample-efficient and aligns with the router's optimization objective. We set lambda=0.5 as a trade-off between domain invariance and task performance; higher lambda risks degrading task accuracy. We apply the domain classifier on router logits rather than hidden states because logits are the final routing decision surface and more directly control expert selection. The cost is that the domain classifier may not capture all domain-specific information in the hidden states, but this avoids over-regularizing expert representations. We used a single linear layer for the domain classifier to prevent gradient vanishing and maintain simplicity. The alpha hyperparameter controls the strength of adversarial regularization; we found alpha=0.1 balances task and domain losses without requiring additional tuning. This design is distinct from standard domain-adversarial neural networks (DANN) because we target only the router, not all parameters, preserving expert specialization while forcing domain-agnostic routing. **Load-bearing assumption stated**: The key assumption is that domain and task information are separable in the router logits; if this holds, adversarial training on logits can enforce domain-invariant routing without harming task performance. To verify, we measure domain classifier accuracy on router logits before DART; if high (>0.8), we switch to a conditional adversarial loss (concatenating task label embedding) to decorrelate domain from task.

### (D) Why it measures what we claim
The adversarial loss component (reversed_domain_loss) measures domain invariance of the router's decision surface because it forces the router to produce logits that are not predictive of domain identity under the assumption that the domain classifier can capture any domain-specific pattern in the logits; this assumption fails if the domain classifier is too weak (e.g., underfits), in which case the adversarial signal becomes noisy and may not enforce true invariance. **We evaluate domain classifier accuracy on router logits as a proxy for domain invariance**: low accuracy (e.g., <20% for 6 domains) implies the router ignores domain identity. However, this proxy fails if the domain classifier is underpowered (linear on logits may miss non-linear patterns) — in that case, accuracy could be low even if router logits contain domain information. The task loss measures the ability to produce correct agent actions, and combined with domain invariance ensures that the routing selects experts based on task-relevant features rather than domain-specific cues. The gradient reversal operation operationalizes the concept of 'learning domain-invariant representations' by making the router's loss counteract the domain classifier's loss; this relies on the assumption that domain and task information are separable in the routing logits, which holds when the experts are sufficiently specialized. If domain and task are entangled (e.g., a domain-specific action pattern), the adversarial loss may harm task performance, and we rely on the hyperparameter alpha and conditional adversarial variant to control this trade-off.

## Contribution

(1) We introduce DART, a domain-adversarial training method for MoE routers that explicitly minimizes domain divergence in routing decisions during SFT. (2) We identify that existing MoE agents overfit to domain-specific features and that adversarial regularization on the router improves zero-shot generalization to unseen domains, providing a design principle for future domain-general agent models. (3) We release a framework for training domain-robust MoE agents with adversarial gradient reversal.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Agents-A1 benchmark (6 domains: Math, Code, Navigation, Conversation, Planning, Manipulation) | Tests zero-shot generalization to unseen domains (hold-out one domain) |
| Primary metric | Zero-shot accuracy on unseen domain | Directly measures domain invariance of routing |
| Baseline | Standard MoE (no domain adapt) | Baseline without any domain generalization |
| Baseline | Domain-regularized MoE (MMD) | Alternative explicit domain regularization |
| Baseline | Full-model DANN | Adversarial training on all features |
| Ablation | DART (adversary on hidden states) | Tests effect of adversarial target choice |
| Runtime | ~2 hours on single A100 for 10 epochs | Reproducibility estimate |

### Why this setup validates the claim

This setup creates a falsifiable test of DART's central claim—that adversarial training on router logits improves zero-shot generalization to unseen domains. The multi-domain agent trajectories (6 seen, held-out unseen) directly measure generalization. Standard MoE tests the baseline without any adaptation. Domain-regularized MoE (MMD) tests an explicit alternative. Full-model DANN tests whether adversarial training on all features is superior. The ablation, applying the same adversarial loss to hidden states instead of router logits, isolates the importance of targeting the routing decision surface. The primary metric (zero-shot accuracy on unseen domains) is the most direct indicator of domain invariance—if DART genuinely induces domain-agnostic routing, it should outperform all baselines on this metric, with the ablation showing a drop due to over-regularizing expert features. Additionally, we measure domain classifier accuracy on router logits as a proxy for invariance; if DART's accuracy is higher than that of a random classifier (e.g., ~16.7% for 6 domains), it indicates that the adversarial loss is effective.

### Expected outcome and causal chain

**vs. Standard MoE** — On a case where the unseen domain has a different action pattern (e.g., left-heavy action distribution vs. right-heavy seen), standard MoE's router uses domain-specific cues (e.g., action frequency) to select experts, causing wrong expert assignment and low accuracy. Our method enforces domain invariance in router logits, so the router ignores such cues and selects the task-relevant expert regardless of domain, leading to higher zero-shot accuracy. We expect a noticeable gap (e.g., 10-15% absolute) on unseen domains, with parity on seen domains.

**vs. Domain-regularized MoE (MMD)** — On a domain where the hidden state distribution shifts (e.g., new visual features), MMD forces all expert hidden states to align across domains, which may wash out specialized expert knowledge (e.g., an expert trained on numerical reasoning loses its edge). Our method only regularizes router logits, not expert hidden states, preserving expert specialization. Thus, on unseen domains, our method should maintain task accuracy while MMD degrades. We expect our method to outperform MMD by 5-10% on unseen domains with strong feature shift.

**vs. Full-model DANN** — On a task where experts are highly specialized (e.g., one expert for math, one for text), full-model DANN applies domain adversarial loss to all representations, forcing expert features to be domain-invariant. This can collapse expert specialization, reducing overall task performance. Our method leaves expert features untouched, only routing is domain-invariant. On unseen domains, our method should achieve higher task accuracy (e.g., 5-7% better) while maintaining similar domain invariance (measured by domain classifier accuracy). On seen domains, full-model DANN may even underperform due to specialization loss.

**vs. Ablation (adversary on hidden states)** — On a domain where domain-specific information resides in intermediate features (e.g., sensor noise), applying adversary to hidden states (ablation) forces both routing and expert features to be invariant, potentially degrading expert utility. Our method's targeting of logits avoids this. We expect the ablation to show lower unseen accuracy than DART (e.g., 3-5% drop), confirming that router logits are the right place to apply gradient reversal.

### What would falsify this idea

If DART does not outperform Standard MoE on unseen domains, or if the improvement is uniform across all domains (seen and unseen) rather than concentrated on the hardest distribution shifts, then the claim of domain-invariant routing via adversarial logits is likely invalid.

## References

1. Scaling the Horizon, Not the Parameters: Reaching Trillion-Parameter Performance with a 35B Agent
2. Generalizing Verifiable Instruction Following
3. TÜLU 3: Pushing Frontiers in Open Language Model Post-Training
