# ADAPT: Adversarial Domain-Adaptive On-Policy Distillation for Cross-Domain Language Modeling

## Motivation

Existing on-policy distillation methods, such as Demystifying On-Policy Distillation, assume homogeneous training data and ignore domain shifts, leading student models to overfit domain-specific patterns (e.g., math problems vs. code generation). The root cause is the lack of explicit invariance to domain-specific biases; the student learns shortcuts tied to domain features rather than generalizable reasoning, degrading performance when domains mix.

## Key Insight

Adversarially forcing the student's hidden representations to be indistinguishable across domains compels reliance on cross-domain predictive features, preserving distillation dynamics while eliminating spurious domain correlations.

## Method

**ADAPT Training Loop (PyTorch-style pseudocode)**

```python
# Input: Teacher T (frozen, e.g., LLaMA2-7B), Student S (trainable, e.g., LLaMA2-1B), mixed-domain dataloader D
# Domain classifier C with gradient reversal layer (GRL) at input; C is a 2-layer MLP (hidden=256, GeLU)
# Conditional discriminator: C receives concatenation of last hidden state and student's softmax prediction (probabilities of the generated token)
# Hyperparams: α=0.1 (adversarial weight), λ=0.5 (GRL scale), τ=0.8 (temperature for distillation)
for each batch (batch_size=32):
    # (A) Student on-policy generation
    student_outputs = S.generate(prompts)  # autoregressive, temperature=1.0
    # (B) Teacher guiding signal (token-level KL divergence as in OPD)
    with torch.no_grad():
        teacher_logits = T(student_outputs)
        teacher_probs = F.softmax(teacher_logits / τ, dim=-1)
        student_logits = S(student_outputs)  # forward pass on generated tokens
        student_probs = F.softmax(student_logits / τ, dim=-1)
        guidance = F.kl_div(F.log_softmax(student_logits, dim=-1), teacher_probs, reduction='none').sum(-1)
    loss_distill = guidance.mean()  # minimize KL divergence (note: original sign may be reversed; we minimize it)
    # (C) Domain invariance via conditional adversarial classifier
    h = S.get_hidden(student_outputs)[-1][:, -1, :]  # last layer, final token; shape (batch, hidden_dim)
    p = student_probs[:, -1, :]  # probabilities of the last generated token (teacher's probability distribution? Actually student's own probabilities)
    # Condition on student's own softmax prediction for that token (to avoid discarding task-relevant features)
    h_cond = torch.cat([h, p], dim=-1)  # shape (batch, hidden_dim + vocab_size)
    domain_logits = C(GRL(h_cond, λ))  # GRL reverses gradients
    loss_adv = F.cross_entropy(domain_logits, domain_labels)
    # (D) Total loss
    loss = loss_distill + α * loss_adv
    loss.backward()
    optimizer.step(parameters=[S, C])
```

**Load-bearing assumption stated explicitly:** The conditional gradient reversal layer forces student representations (conditioned on the student's own predictions) to be domain-invariant while preserving task-relevant features necessary for on-policy distillation. This assumes that by conditioning on the student's predicted output, the discriminator cannot use task-predictive information to infer domain, thus the adversarial loss only removes domain-specific, non-task-relevant features. **Calibration/verification story:** Before full evaluation, we run a controlled synthetic experiment where we create two domains: one where surface-level tokens (e.g., "math" vs "code") are perfectly correlated with domain labels, but the underlying reasoning task is identical. We measure domain classification accuracy after GRL (should be near chance) and transfer accuracy (should be high). If domain accuracy is high or transfer accuracy degrades, we tune λ and adjust the conditioning mechanism (e.g., use a smaller conditioning vector).

**(C) Why this design:** We chose a gradient reversal layer (GRL) over two-step adversarial training (e.g., alternating min-max) because GRL provides a single forward-backward pass, reducing training instability from competing objectives; the trade-off is sensitivity to the scaling hyperparameter λ, which controls the degree of domain confusion without destabilizing the distillation loss. The domain classifier is a 2-layer MLP (hidden=256, GeLU) that takes the concatenation of the last hidden state (final token) and the student's softmax probabilities of that same token (to condition on task prediction). This conditional design (as in Long et al., 2018) prevents discarding features that are both task-predictive and domain-specific but necessary for reasoning—a failure mode of standard adversarial training. The adversarial weight α=0.1 is intentionally small to avoid overwhelming the distillation signal, as overly aggressive domain invariance could discard domain-specific but reasoning-relevant features (e.g., math notation). Sampling student outputs on-policy (rather than using teacher-generated sequences) preserves the exploration dynamics essential for effective knowledge distillation, as shown in the OPD work, but introduces variance; we mitigate with batch size 32 and gradient clipping (max_norm=1.0). Unlike standard domain-adversarial training (Ganin et al., 2016), which is applied to supervised tasks, our method must maintain delicate on-policy policy gradients, hence the conservative α and single-layer classifier. Resource estimates: Training on 4×A100 GPUs with mixed precision takes ~48 hours for LLaMA2-1B student distilled from LLaMA2-7B teacher on MATH (7.5k training examples). Student parameters: ~1B, teacher frozen: 7B, domain classifier: ~0.5M.

**(D) Why it measures what we claim:** The distillation loss (token-level KL divergence) operationalizes "teacher guiding signal quality" because it directly measures alignment with the teacher's distribution, under the assumption that the teacher provides correct and domain-general reasoning; this assumption fails when the teacher itself is biased toward a specific domain (e.g., trained only on math), causing the loss to reinforce domain-specific shortcuts instead of generalizability. The adversarial loss (cross-entropy on domain classifier after GRL) operationalizes "domain invariance" because the classifier's inability to predict domain identity implies that hidden representations (conditioned on student predictions) contain no domain-specific discriminative information, under the assumption that domain labels are exhaustive and non-overlapping; this assumption fails when domains share latent structure (e.g., code and math both require logical steps) such that the classifier cannot separate them even with domain-specific features, in which case invariance may be achieved at the cost of losing shared semantics. The combination ensures that only features serving both objectives (high distillation alignment and low domain predictability) survive, forcing the student to rely on cross-domain predictive patterns.

## Contribution

(1) A novel training framework, ADAPT, that integrates adversarial domain confusion into on-policy distillation to learn cross-domain invariant representations for language models. (2) The design of a gradient-reversal-based mechanism that balances distillation fidelity and domain invariance without requiring domain-specific teacher models, with analysis of hyperparameter trade-offs.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|------------------------|
| Dataset | MATH benchmark (all categories: Prealgebra, Algebra, Number Theory, Counting & Probability, Geometry, Intermediate Algebra, Precalculus) | Diverse domains: algebra, geometry, etc. |
| Primary metric | Per-domain average accuracy (macro average over categories) | Captures cross-domain consistency. |
| Secondary metric | Standard deviation across domains | Measures overfitting to specific domains. |
| Baseline 1 | Standard KD (off-policy, teacher-generated sequences) | Baseline without on-policy sampling. |
| Baseline 2 | On-policy distillation (OPD) | Baseline without domain invariance. |
| Baseline 3 | Domain-adversarial training (DANN) with teacher-generated sequences | Baseline without on-policy generation. |
| Ablation-of-ours | Ours without GRL (α=0) | Isolates effect of domain adversary. |

### Why this setup validates the claim

The central claim is that combining on-policy distillation with conditional domain-adversarial training yields a student that generalizes across domains better than either component alone. By comparing against Standard KD (off-policy), we test the importance of on-policy sampling. Against OPD (on-policy but no adversarial), we test the added value of conditional domain invariance. Against DANN (adversarial but off-policy), we test the necessity of on-policy generation. The ablation (α=0) isolates the adversarial component. The per-domain average accuracy is the right metric because it forces the model to perform well on all domains equally; a uniform improvement across domains would support the claim, while a gain only in some domains would suggest domain-specific overfitting. Additionally, we include a controlled synthetic experiment (see below) to directly validate the link between domain invariance (measured by domain classifier accuracy after GRL) and cross-domain generalization (measured by accuracy on held-out combined-domain test set). This synthetic setup uses two artificially created domains from a subset of MATH problems where surface tokens (e.g., "factor" vs. "solve") are swapped to create spurious correlations, isolating the effect of invariance. If our method achieves near-chance domain accuracy and high transfer accuracy while OPD fails, the causal chain is confirmed.

### Expected outcome and causal chain

**vs. Standard KD** — On a case where the student generates an unconventional proof path (e.g., using a geometric lemma in an algebra problem), the standard KD baseline uses teacher-generated sequences that may not cover such paths, leading to poor guidance. Our method instead samples from the student's own distribution, so the teacher provides targeted feedback on student mistakes, and the adversarial classifier prevents overfitting to domain-specific language (e.g., geometry notation). We expect our method to outperform on harder problems (e.g., from the "hard" subset of each category) by at least 5% absolute, with parity on easy problems (variance <3%).

**vs. On-policy distillation (OPD)** — On a case where certain domains (e.g., Counting & Probability vs. Geometry) have distinct surface patterns (like "draw" vs. "solve"), the OPD baseline may latch onto these patterns as shortcuts instead of learning the underlying reasoning, causing accuracy drops when the pattern is misleading. Our method's conditional adversarial component forces the hidden representation (conditioned on student prediction) to be uninformative of domain identity, so it discards such surface cues. We expect our method to show lower variance across MATH categories (standard deviation <5% vs. >10% for OPD) and higher accuracy on domains where shortcuts are prevalent (e.g., Counting & Probability with formulaic phrases), expecting a 3-7% improvement.

**vs. Domain-adversarial training (DANN)** — On a case where the teacher's generation style is very formal (e.g., always using first-order logic) but student's on-policy outputs are more natural language, DANN trains on teacher-generated sequences that are domain specific in style, so the student learns to mimic its teacher's style rather than generalize. Our method combines on-policy sampling with adversarial training, so the student explores its own outputs while being regularized to be domain-agnostic. We expect our method to outperform on domains that require informal reasoning (e.g., geometry with diagrams), showing a 5-10% absolute improvement in those categories.

### What would falsify this idea

If our method's per-domain accuracy variance is comparable to OPD's (e.g., standard deviation >8%), or if improvements are uniformly distributed across all domains (rather than concentrated on domains where domain-specific patterns are known to occur), then the central claim fails. Specifically, seeing no reduction in domain-specific overfitting (e.g., accuracy on Counting & Probability not significantly higher than OPD) would indicate that the adversarial component does not achieve cross-domain invariance. Additionally, if the synthetic controlled experiment shows that domain classifier accuracy after GRL is not near chance (e.g., >60%) despite our method having higher transfer accuracy, then the link between invariance and generalization is unproven.

## References

1. Demystifying On-Policy Distillation: Roles, Pathologies, and Regulations
2. Solving Quantitative Reasoning Problems with Language Models
3. DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models
4. Llemma: An Open Language Model For Mathematics
5. The Stack: 3 TB of permissively licensed source code
