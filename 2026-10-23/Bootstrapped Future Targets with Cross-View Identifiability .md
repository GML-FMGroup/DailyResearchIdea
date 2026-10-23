# Bootstrapped Future Targets with Cross-View Identifiability for World Action Models

## Motivation

FRED's self-distillation loss anchors the target to ground-truth future observations, so target identifiability under self-generated futures is never established; this forces reliance on scarce future observations and a fixed external feature extractor. More broadly, feature-prediction methods such as V-JEPA and I-JEPA inherit the semantics of their chosen target feature space, meaning the learned representation cannot be invariant to the extractor that defines the targets. Closing this gap requires a mechanism that bootstraps distillation targets from the model's own latent predictions while enforcing that those targets remain identifiable across different feature extractors.

## Key Insight

A shared latent space with a cross-view consistency constraint between teacher pseudo-futures and student predicted futures makes the learned representation invariant to the external feature extractor, because the only stable solution is one that maps different extractor views of the same future to the same latent point.

## Method

**What it is** — BFT-WAM (Bootstrapped Future Targets for World Action Models) is a self-distillation training procedure for a video world action model that removes the need for ground-truth future observations and for a fixed external feature extractor. Its inputs are a video diffusion model backbone, a student predictor, and two independently pre-trained feature extractors (e.g., V-JEPA and DINOv2); its outputs are a future-latent predictor and an action decoder conditioned on the predicted future representation.

**How it works** —

```pseudocode
# Given: video diffusion model D, student predictor S, action decoder A
# Two frozen external extractors E1, E2 (e.g., V-JEPA, DINOv2)
# Shared latent space Z (dim 768), EMA teacher T (copy of D)
# Shared projection P (768-d output) + per-extractor adapters A1, A2 (2-layer MLP, hidden=512, GeLU)
# Anti-collapse regularizer: VICReg-style variance + covariance on Z
# Reliability gate g in [0,1] from per-extractor feature-space agreement score

for each video clip x = (o_{1:t}, o_{t+1:t+H}) with actions a_{t+1:t+H}:
    # 1. Ground-truth future features from two views
    z1_gt = E1(o_{t+1:t+H})          # shape [H, d1]
    z2_gt = E2(o_{t+1:t+H})          # shape [H, d2]

    # 2. Student predicts future latent from noise + history context
    z_hat = S(o_{1:t}, eps)          # single denoising step, eps ~ N(0,I)

    # 3. Teacher pseudo-future: EMA teacher predicts from same noise, no grad
    z_tilde = T(o_{1:t}, eps)        # EMA of S, stop-gradient

    # 4. Cross-view consistency: project both extractor views into shared Z
    #    via shared projection P + per-extractor adapters A1, A2 (trained jointly)
    u1 = P(A1(z1_gt));  u2 = P(A2(z2_gt))
    #    Reliability gate: g = sigmoid(alpha * (agreement(z1_gt, z2_gt) - beta))
    #    where agreement is cosine similarity of centered features; alpha=10, beta=0.5
    #    Consistency loss forces g * ||u1 - u2||^2 + g * (||u1 - z_tilde||^2 + ||u2 - z_tilde||^2)
    L_cons = g * (||u1 - u2||^2 + ||u1 - z_tilde||^2 + ||u2 - z_tilde||^2)

    # 5. Student distillation loss: predict teacher pseudo-future
    L_distill = ||z_hat - sg(z_tilde)||^2

    # 6. Action loss: action decoder conditioned on z_hat
    L_action = ||A(z_hat, o_{1:t}) - a_{t+1:t+H}||^2

    # 7. Anti-collapse regularizer on shared latent Z (VICReg-style)
    #    variance term: mean over dims of max(0, 1 - std(u1)) + max(0, 1 - std(u2))
    #    covariance term: sum of squared off-diagonal entries of cov(u1) + cov(u2)
    L_reg = lambda_var * (var_term(u1) + var_term(u2)) + lambda_cov * (cov_term(u1) + cov_term(u2))

    # 8. Total loss; update S, P, A1, A2, A; EMA update T
    L = L_distill + lambda_cons * L_cons + lambda_act * L_action + lambda_reg * L_reg
    update(S, P, A1, A2, A);  T = tau*T + (1-tau)*S   # tau=0.996
```

Hyperparameters: lambda_cons=1.0, lambda_act=1.0, lambda_var=1.0, lambda_cov=0.04, tau=0.996, latent dim 768, single denoising step, batch size 64, learning rate 1e-4, alpha=10, beta=0.5.

**Why this design** — We chose to bootstrap the distillation target from the EMA teacher's own latent prediction rather than from ground-truth future observations because the motivation requires learning from unlabeled videos without future observations; the cost is that the target is initially noisy and may drift, which we accept because the cross-view consistency constraint anchored by the anti-collapse regularizer anchors it to real feature geometry. We chose two independent external extractors (V-JEPA and DINOv2) rather than a single fixed extractor because invariance to the extractor is only meaningful if the model sees multiple views of the same future; the cost is doubled feature extraction compute during training, which we accept because it is offloaded to frozen models and does not affect inference. We chose a shared latent space with a shared projection plus per-extractor adapters rather than separate per-extractor spaces because the consistency constraint must operate on a common representation to enforce identifiability; the cost is that the adapters add parameters, which we accept because they are small and trained jointly with the student. We added a VICReg-style variance/covariance regularizer on the shared latent because the consistency loss alone admits a trivial constant-output solution (as in BYOL/SimSiam without asymmetry); the cost is two extra hyperparameters, which we accept because without it the extractor-invariance claim is underdetermined. We added a reliability gate on the consistency loss because the two extractors may disagree when one is degraded; the cost is a scalar gate computation, which we accept because it prevents the consistency term from fitting the weaker view. This design is not a variant of data2vec or BYOL: those methods use a single fixed target view (masked input or EMA of the same encoder) and do not enforce cross-extractor consistency, so they cannot achieve extractor invariance; our method explicitly couples two different feature extractors through the shared latent space with an anti-collapse regularizer, which is the structural property that prior self-distillation methods lack.

**Why it measures what we claim** — The cross-view consistency loss `L_cons` measures extractor invariance because it forces the projected features from two independent extractors of the same future to coincide with each other and with the teacher pseudo-future; this assumes that the two extractors' features are linearly alignable to a shared latent geometry (i.e., a common linear map exists) and that the anti-collapse regularizer prevents the trivial constant-output solution; this assumption fails when one extractor is severely degraded (e.g., motion-blurred video where DINOv2 fails) or when the two are linearly non-alignable, in which case the consistency loss reflects the noise of the weaker extractor rather than true invariance (mitigated by the reliability gate g). The distillation loss `L_distill` measures target identifiability because it forces the student's prediction to match the teacher's pseudo-future, which is itself anchored by the consistency constraint; this assumes that the EMA teacher provides a stable target that does not collapse, an assumption that fails when the EMA decay is too low (tau < 0.99), in which case the target becomes noisy and the student learns to predict noise rather than future semantics. The action loss `L_action` measures the usefulness of the learned representation for control because it trains the action decoder on the student's predicted future; this assumes that the predicted future latent contains sufficient information for action prediction, an assumption that fails when the future is highly stochastic (e.g., multi-modal demonstrations), in which case the action loss reflects the ambiguity of the future rather than representation quality.

## Contribution

["A self-distillation framework for world action models that bootstraps future targets from the model's own latent predictions via an EMA teacher, eliminating the need for ground-truth future observations during training.", 'A cross-view consistency constraint in a shared latent space that enforces target identifiability across two independent feature extractors, making the learned representation invariant to the choice of external extractor.', 'An empirical design principle: extractor invariance in future-prediction self-distillation requires at least two independent views of the future, and a single-view EMA target (as in data2vec) is insufficient to prevent extractor-specific collapse.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | Real-world manipulation demos (multi-task, unlabeled futures) | Tests learning without ground-truth future observations |
| Primary metric | Action prediction error on held-out tasks | Directly reflects representation usefulness for control |
| Secondary metric | Downstream policy success rate on two real robot tasks (e.g., pick-and-place, push) | Ties significance to control-relevant outcome |
| Baseline | Competing World Action Model (FRED-style) | Tests whether bootstrapped targets beat observation-based targets |
| Baseline | Single-extractor self-distillation (data2vec-style) | Tests whether cross-view coupling adds extractor invariance |
| Baseline | Pixel-reconstruction world model | Tests whether latent prediction beats generative futures |
| Baseline | Cross-view BYOL/SwAV variant (closest multi-view self-supervised method) | Tests whether our EMA target anchored jointly by two frozen extractors through a shared projection has a structural advantage |
| Ablation-of-ours | BFT-WAM without cross-view consistency loss | Isolates the consistency term's contribution |
| Ablation-of-ours | BFT-WAM without anti-collapse regularizer | Tests whether the variance/covariance term is necessary to avoid collapse |
| Ablation-of-ours | BFT-WAM without reliability gate | Tests whether gating by extractor agreement improves robustness |

### Why this setup validates the claim

The central claim is that bootstrapping future targets from an EMA teacher, anchored by cross-extractor consistency with an anti-collapse regularizer, yields a future representation that is both identifiable and useful for control without ground-truth future observations. The FRED-style WAM baseline tests the first sub-claim: it uses observation-grounded future targets, so if our unlabeled bootstrap matches or beats it on action error and policy success rate, the need for ground-truth futures is genuinely removed rather than merely traded away. The single-extractor self-distillation baseline tests the second sub-claim: it has an EMA target but no cross-view coupling, so any gain we show over it isolates extractor invariance as the causal ingredient. The cross-view BYOL/SwAV baseline tests whether our specific structural property (EMA target anchored jointly by two frozen extractors through a shared projection) is necessary, rather than multi-view consistency alone. The pixel-reconstruction baseline tests whether latent-space prediction is doing the work versus generative futures. The no-consistency and no-regularizer ablations are the sharpest tests: if removing `L_cons` collapses performance toward the single-extractor baseline, the consistency term is confirmed as the mechanism; if removing `L_reg` causes representation collapse (measured by rank/variance of Z), the anti-collapse term is confirmed as necessary. Action prediction error and policy success rate on held-out tasks are the right metrics because they are the downstream quantities the representation is supposed to serve; a representation that is merely consistent but control-useless would not move these numbers.

### Expected outcome and causal chain

**vs. FRED-style WAM** — On a case where demonstrations are plentiful but future observations are sparse or unlabeled, the FRED-style baseline produces degraded future predictions because its targets depend on observed futures that are missing or noisy. Our method instead bootstraps targets from the EMA teacher anchored by consistency and anti-collapse regularization, so it learns a usable future representation from the same unlabeled clips. We expect a noticeable action-error gap and policy success rate gap on the unlabeled-heavy subset but rough parity on the fully-labeled subset, confirming the gain is concentrated where ground-truth futures are absent. We will report the gap as a function of the fraction of labeled futures (0%, 25%, 50%, 100%).

**vs. Single-extractor self-distillation** — On a case where the single fixed extractor is weak for the domain (e.g., motion-heavy manipulation where one view underrepresents dynamics), the baseline produces a drifted, extractor-specific target because its EMA target has no second view to correct it. Our method instead couples two independent extractors through the shared latent space with a reliability gate, so the consistency term pulls the target toward a view-invariant geometry. We expect a moderate action-error advantage that grows with the degree of extractor disagreement, and near-parity when the two extractors already agree.

**vs. Cross-view BYOL/SwAV variant** — On a case where the two extractors are linearly non-alignable or one is degraded, the cross-view BYOL/SwAV baseline may collapse or fit the weaker view because it lacks an explicit anti-collapse regularizer and reliability gate. Our method adds a VICReg-style variance/covariance term and gates the consistency loss by extractor agreement, so it preserves rank and avoids fitting noise. We expect a clear advantage on the degraded-extractor subset and on the synthetic uniqueness check (rank of Z), with near-parity on clean, alignable data.

**vs. Pixel-reconstruction world model** — On a case where the future is visually ambiguous but action-relevant (e.g., contact-rich manipulation with small but decisive motion), the baseline produces futures that spend capacity on pixels and blur the action-relevant detail, because its loss is pixel-space. Our method instead predicts in latent space anchored to semantic features, so it preserves action-relevant structure. We expect a clear action-error gap on contact-rich tasks but smaller differences on visually simple tasks.

### What would falsify this idea

If the no-consistency ablation matches full BFT-WAM across all subsets, or if our advantage over the single-extractor baseline is uniform rather than concentrated where extractor disagreement is high, then cross-view consistency is not the causal mechanism and the central claim is wrong. Likewise, if we fail to beat the FRED-style baseline specifically on the unlabeled-future subset, the claim that ground-truth futures are unnecessary is falsified. If removing the anti-collapse regularizer does not cause collapse (measured by rank/variance of Z), the regularizer is unnecessary; if the reliability gate does not improve robustness on degraded-extractor subsets, the gating mechanism is not needed.

## References

1. Self-Distilling Future Video Representations for World Action Models
2. Revisiting Feature Prediction for Learning Visual Representations from Video
3. Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture
4. VideoMAE V2: Scaling Video Masked Autoencoders with Dual Masking
5. data2vec: A General Framework for Self-supervised Learning in Speech, Vision and Language
6. Context Autoencoder for Self-supervised Representation Learning
