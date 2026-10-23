# Constraint-Sufficient Latent Prediction for Language-Conditioned World Models

## Motivation

Across both language-conditioned world models and instruction-following benchmarks, the training objective is structurally decoupled from the human-grounded acceptance criterion: LaLa-World optimizes next-step latent prediction while constraint satisfaction is never part of the learning signal, and AgentIF/CFBench treat constraint satisfaction as a post-hoc annotation or scorer. This decoupling means that improving latent prediction provides no guarantee that the predicted futures satisfy the constraints that human evaluators actually care about, because the latent predictor is never forced to encode the information that determines constraint satisfaction.

## Key Insight

Requiring the predicted latent to be a sufficient statistic for a constraint-satisfaction classifier trained against instruction-perturbed contrastive negatives forces the latent predictor to encode exactly the directions that causally determine constraint satisfaction, making the prediction objective and the acceptance criterion structurally identical rather than merely correlated.

## Method

**What it is.** We introduce COST-LWM (Constraint-Sufficient Latent World Model), a training procedure for language-conditioned latent future predictors. Given a context latent $z_t$, a language instruction $\ell$, and a proposed numerical action $a$, the predictor $f_\theta$ outputs a predicted next latent $\hat{z}_{t+1}$. A constraint classifier $C_\phi$ maps $(\hat{z}_{t+1}, \ell)$ to a binary satisfaction logit $s$. The method jointly trains $f_\theta$ to minimize latent prediction error and to maximize the mutual information between $\hat{z}_{t+1}$ and the true constraint satisfaction label $y$, while $C_\phi$ is trained adversarially on instruction perturbations that flip $y$. The load-bearing assumption is stated explicitly: $C_\phi$, trained on $(\hat{z}_{t+1}, \ell)$ pairs with grammar-based constraint flips as negatives, learns a calibrated estimate of $P(y \mid \hat{z}_{t+1}, \ell)$ that reflects genuine constraint semantics rather than perturbation artifacts, so that gradients from its logits into $\hat{z}_{t+1}$ shape the latent along directions causally determining constraint satisfaction. We operationalize this assumption with three concrete controls inside the existing training loop: (i) a multi-flip negative distribution with explicit no-op controls, (ii) a paraphrase-invariance calibration check on $C_\phi$, and (iii) a held-out calibration gate on the Phase 4 gradient.

**How it works.**

```pseudocode
Input: dataset D = {(z_t, l, a, z_{t+1}, y, L_pert)} where y = 1 if constraint set is satisfied by the true future,
       L_pert = list of M=4 perturbed instructions per example, each with one constraint-violating flip
                (negated or swapped constraint) and label y_pert = 0, plus M_noop=2 no-op flips
                (redundant constraint rephrased) with label y_pert = 1,
       l_para = paraphrase of l with identical constraint satisfaction (y unchanged).
Hyperparameters: lambda_mi = 0.1, lambda_adv = 0.5, K = 4 classifier gradient steps per predictor step,
                 temperature tau = 0.07, perturbation budget B = 1 constraint flip per instruction,
                 M = 4 violating flips, M_noop = 2 no-op flips, calibration set size N_cal = 512 held-out examples,
                 calibration gate threshold epsilon = 0.05 (gradient scale = max(0, 1 - ECE/epsilon)).

# Pre-training calibration check (run once before main loop)
Compute flip-validity rate: fraction of L_pert entries where the flipped constraint actually changes y
  (verified by programmatic constraint checker on ground-truth futures).
Compute paraphrase-invariance: |C_phi(z_hat, l) - C_phi(z_hat, l_para)| averaged over N_cal examples.
If flip-validity < 0.9 or paraphrase-invariance > 0.1, regenerate perturbations / re-train C_phi before proceeding.

for each training step:
  # Phase 1: predictor forward
  z_hat = f_theta(z_t, l, a)
  z_true = z_{t+1}
  L_pred = || z_hat - z_true ||^2

  # Phase 2: constraint classifier forward (no gradient to f_theta through this path)
  s_pos = C_phi(z_hat.detach(), l)          # satisfaction logit for original instruction
  s_neg_list = [C_phi(z_hat.detach(), l_pert) for l_pert in L_pert]   # M+M_noop logits
  L_cls = BCE(s_pos, y) + (1/(M+M_noop)) * sum(BCE(s_neg, y_pert) for s_neg, y_pert in zip(s_neg_list, L_pert_labels))

  # Phase 3: adversarial classifier update (K steps)
  for k in range(K):
    update phi by gradient descent on L_cls

  # Phase 4: predictor update with mutual-information term
  s_pos_g = C_phi(z_hat, l)                 # gradient flows through z_hat to f_theta
  s_neg_g_list = [C_phi(z_hat, l_pert) for l_pert in L_pert]
  # Contrastive InfoNCE over all negatives: z_hat should be more predictive of y than of any y_pert
  logits = concat([s_pos_g] + s_neg_g_list) / tau
  labels = concat([y] + L_pert_labels)
  L_mi = CrossEntropy(logits, labels)
  # Adversarial term: classifier is frozen here; predictor tries to make each s_neg_g close to its y_pert
  L_adv = (1/(M+M_noop)) * sum(BCE(s_neg_g, y_pert) for s_neg_g, y_pert in zip(s_neg_g_list, L_pert_labels))
  # Calibration gate: scale gradient through C_phi by held-out calibration quality
  ECE = expected_calibration_error(C_phi, calibration_set)   # computed every 100 steps on N_cal=512 examples
  gate = max(0.0, 1.0 - ECE / epsilon)
  L_total = L_pred + gate * (lambda_mi * L_mi + lambda_adv * L_adv)
  update theta by gradient descent on L_total
```

**Why this design.** We chose a contrastive InfoNCE mutual-information term over a direct binary cross-entropy on $s_{\text{pos}}$ because the contrastive form explicitly requires $\hat{z}_{t+1}$ to discriminate between the original and the constraint-violating instruction, which is the exact structural property that forces the latent to encode constraint-relevant directions; the cost is that the classifier must be trained well enough to provide a meaningful contrastive signal, which we address with $K=4$ inner steps. We chose an adversarial perturbation of the instruction over random noise or dropout because a constraint-violating perturbation is semantically meaningful and targets exactly the human-grounded criterion, accepting the cost that we must generate valid perturbations (we use a grammar-based constraint flipper that negates or swaps one annotated constraint, and we add $M_{\text{noop}}=2$ no-op flips as explicit $y_{\text{pert}}=1$ controls to prevent the classifier from fitting perturbation surface form). We chose to give the predictor gradients through the classifier's latent bottleneck only (not through the classifier's own weights in Phase 4) because this prevents the predictor from gaming the classifier by exploiting classifier parameters, accepting the cost that the classifier must be periodically refreshed (we update $\phi$ every step with $K$ inner steps). We chose to keep the latent prediction loss $L_{\text{pred}}$ as the dominant term ($\lambda_{\text{mi}}=0.1$) because the world model must still predict accurately; the MI term acts as a regularizer that shapes the latent geometry rather than replacing the prediction objective. We added a calibration gate $\text{gate} = \max(0, 1 - \text{ECE}/\epsilon)$ on the Phase 4 gradient because the load-bearing assumption requires $C_\phi$ to be calibrated; if ECE exceeds $\epsilon=0.05$, the gate down-weights the constraint terms so that an artifact-fitting classifier contributes near-zero gradient to $f_\theta$. A domain expert would not describe this as a variant of post-hoc constraint scoring (AgentIF, CFBench) or of LaLa-World because the constraint classifier is not a scorer applied after training: it is trained adversarially against instruction perturbations and its gradients flow into the latent predictor, creating a training-time coupling that neither prior method has.

**Why it measures what we claim.** The contrastive InfoNCE loss $L_{\text{mi}}$ measures the mutual information between the predicted latent $\hat{z}_{t+1}$ and the true constraint satisfaction $y$ because the classifier's logits $s_{\text{pos}}, s_{\text{neg}}$ are trained to be calibrated estimates of $P(y \mid \hat{z}_{t+1}, \ell)$ and $P(y_{\text{pert}} \mid \hat{z}_{t+1}, \ell_{\text{pert}})$; this assumption fails when the classifier overfits to the perturbation pattern rather than the constraint semantics, in which case $L_{\text{mi}}$ reflects the classifier's ability to detect perturbation artifacts rather than genuine constraint sufficiency. We guard against this failure with the paraphrase-invariance check (logit unchanged under constraint-preserving paraphrase) and the calibration gate (ECE $\le 0.05$). The adversarial term $L_{\text{adv}}$ measures the causal necessity of constraint-relevant information in $\hat{z}_{t+1}$ because forcing $s_{\text{neg}}$ toward $y_{\text{pert}}$ under the perturbed instruction requires the latent to contain the information that distinguishes satisfied from violated constraints; this assumption fails when the perturbation does not actually change the constraint satisfaction (e.g., a redundant constraint), in which case $L_{\text{adv}}$ becomes a no-op. We guard against this with the flip-validity rate check (fraction of flips that actually change $y$, verified programmatically) and the inclusion of no-op flips as explicit $y_{\text{pert}}=1$ controls. The latent prediction loss $L_{\text{pred}}$ measures perceptual fidelity to the true future because the L2 distance in latent space is assumed to correlate with perceptual similarity; this assumption fails when the latent space is anisotropic, in which case $L_{\text{pred}}$ reflects distance along high-variance directions rather than perceptual fidelity. Together, these three quantities operationalize the motivation-level concept of coupling the training objective to the human-grounded acceptance criterion: $L_{\text{mi}}$ and $L_{\text{adv}}$ force the latent predictor to be a sufficient statistic for constraint satisfaction, while $L_{\text{pred}}$ preserves prediction accuracy.

## Contribution

['A training-time coupling mechanism (COST-LWM) that makes the language-conditioned latent predictor a sufficient statistic for constraint satisfaction via an adversarially trained constraint classifier with gradients flowing through the latent bottleneck.', 'A contrastive InfoNCE objective over instruction-perturbed satisfaction labels that identifies which latent directions are causally responsible for constraint satisfaction, providing a design principle for tying world-model training to human-grounded acceptance criteria.', 'An empirical demonstration that the coupled training objective improves constraint satisfaction of imagined trajectories without degrading latent prediction accuracy, compared to post-hoc scoring and static annotation baselines.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | LaLa-World mixed sim+robot trajectories with annotated constraints | Matched domain and constraint labels already exist |
| Primary metric | Constraint-satisfaction rate of MPC plans under held-out instructions | Directly measures the human-grounded criterion we target |
| Secondary metric | Held-out probe accuracy: linear probe trained on $\hat{z}_{t+1}$ to predict $y$ | Direct sufficiency test of the latent |
| Diagnostic metric | Classifier ECE and flip-validity rate on held-out calibration set (N=512) | Verifies load-bearing calibration assumption |
| Baseline | LaLa-World (per-domain LWM) | Tests whether coupling beats accurate-but-uncoupled prediction |
| Baseline | Post-hoc constraint scorer on frozen latent | Tests whether training-time coupling beats after-the-fact scoring |
| Baseline | RAEv2 video-generation planner | Tests whether latent sufficiency beats pixel-level generation |
| Ablation-of-ours | COST-LWM without L_mi/L_adv (L_pred only) | Isolates the contribution of the constraint-coupling terms |
| Ablation-of-ours | COST-LWM with single-flip negatives (original design) | Tests whether multi-flip + no-op controls matter |
| Ablation-of-ours | COST-LWM without calibration gate | Tests whether gating prevents artifact gradients |

### Why this setup validates the claim

The central claim is that coupling the latent predictor to constraint satisfaction during training makes the predicted latent a sufficient statistic for the human-grounded acceptance criterion, and that this sufficiency translates into better constraint-following plans. LaLa-World is the right accuracy-matched baseline because it already achieves strong latent prediction on this exact data; if our method beats it on constraint satisfaction while matching it on prediction error, the gain must come from the coupling rather than from better dynamics modeling. The post-hoc scorer baseline isolates the training-time-coupling sub-claim: a scorer applied to a frozen LaLa-World latent can only succeed if the latent already encodes constraint-relevant directions, so any gap between it and COST-LWM is attributable to the coupling. The RAEv2 baseline tests the pixel-fidelity alternative — if generating explicit futures beat latent sufficiency, our premise would be wrong. The ablation is the sharpest test: removing L_mi and L_adv while keeping identical architecture and data tells us whether the constraint terms, not capacity, produce the effect. The held-out probe accuracy directly measures sufficiency: if $\hat{z}_{t+1}$ is a sufficient statistic for $y$, a linear probe on $\hat{z}_{t+1}$ should achieve high accuracy, and this metric is independent of the classifier $C_\phi$ used in training. The diagnostic metrics (ECE and flip-validity rate) verify the load-bearing assumption: if ECE is low and flip-validity is high, the classifier's logits are calibrated and the perturbations genuinely flip $y$, so $L_{\text{mi}}$ and $L_{\text{adv}}$ measure constraint sufficiency rather than perturbation-artifact detection. Constraint-satisfaction rate on held-out instructions is the right primary metric because it is exactly the quantity the motivation names, and it is measured on plans the model did not train on, so it cannot be gamed by memorizing training constraints.

### Expected outcome and causal chain

**vs. LaLa-World (per-domain LWM)** — On a case where an instruction contains a spatially restrictive constraint (e.g., "place the cup left of the bowl"), LaLa-World produces a latent that predicts plausible dynamics but is agnostic to the left/right distinction, because its L2 objective never pressures the latent to encode constraint-relevant directions. Our method instead forces the latent to discriminate original from constraint-flipped instructions via L_mi and L_adv, so the predicted latent carries the left/right information the planner needs. We expect a noticeable gap on constraint-dense instruction subsets but rough parity on unconstrained free-form instructions, where the coupling term has nothing to shape. We also expect our held-out probe accuracy to be higher on constraint-dense subsets, confirming sufficiency.

**vs. Post-hoc constraint scorer on frozen latent** — On a case where the constraint is only weakly correlated with the dominant latent variance (e.g., a subtle orientation constraint on a small object), the post-hoc scorer fails because the frozen LaLa-World latent simply does not contain the distinguishing direction, so no scorer can recover it. Our method instead shapes the latent geometry during training so that constraint-relevant directions are explicitly represented, because the classifier gradients flow into the predictor. We expect the post-hoc scorer to match us on constraints aligned with high-variance latent directions but to fall clearly behind on low-variance constraint subsets, and we expect our probe accuracy to be higher on those low-variance subsets.

**vs. RAEv2 video-generation planner** — On a case where the constraint concerns a fine-grained relational property that pixel generation renders ambiguously (e.g., "stack the block on the red one"), the RAEv2 planner produces a visually plausible future whose constraint status is hard to read off, because pixel fidelity and constraint satisfaction are not the same objective. Our method instead keeps a compact latent that is trained to be a sufficient statistic for the constraint, so the satisfaction signal is directly available. We expect RAEv2 to be competitive on perceptual realism but to trail us on constraint-satisfaction rate, with the gap widening on relational and counting constraints.

**vs. COST-LWM without L_mi/L_adv (L_pred only)** — This ablation removes the constraint-coupling terms while keeping architecture and data identical. If the full method beats this ablation on constraint-satisfaction rate and probe accuracy, the gain is attributable to the coupling terms rather than to capacity or data. We expect the ablation to match LaLa-World, confirming that the coupling terms are the source of improvement.

**vs. COST-LWM with single-flip negatives (original design)** — This ablation uses only one grammar-based flip per instruction (no no-op controls, no multi-flip). If the full method with multi-flip and no-op controls outperforms this ablation, it confirms that the negative distribution matters and that the original single-flip design was vulnerable to shortcut learning. We expect the single-flip variant to show higher ECE and lower probe accuracy, indicating artifact-fitting.

**vs. COST-LWM without calibration gate** — This ablation removes the calibration gate on the Phase 4 gradient. If the full method with the gate outperforms this ablation, it confirms that gating prevents artifact-sensitive gradients from corrupting the predictor. We expect the no-gate variant to have higher variance across seeds and worse performance on low-variance constraint subsets.

### What would falsify this idea

If COST-LWM's gain over LaLa-World were uniform across all instruction subsets rather than concentrated on constraint-dense instructions and on low-variance constraint directions, then the coupling terms would not be doing what we claim and the effect would be attributable to generic regularization. Likewise, if the ablation without L_mi and L_adv matched full COST-LWM on constraint-satisfaction rate, the central coupling claim would be false. Additionally, if the diagnostic metrics show high ECE (>0.1) or low flip-validity (<0.8) yet the method still appears to work, the sufficiency claim would be unsupported because the classifier would not be measuring constraint semantics.

## References

1. LaLa-World: Scaling Latent World Models with Natural Language Actions
2. AGENTIF: Benchmarking Instruction Following of Large Language Models in Agentic Scenarios
3. SysBench: Can Large Language Models Follow System Messages?
4. CFBench: A Comprehensive Constraints-Following Benchmark for LLMs
5. Instruction-Following Evaluation for Large Language Models
6. FollowBench: A Multi-level Fine-grained Constraints Following Benchmark for Large Language Models
7. Can LLMs Follow Simple Rules?
8. COLLIE: Systematic Construction of Constrained Text Generation Tasks
