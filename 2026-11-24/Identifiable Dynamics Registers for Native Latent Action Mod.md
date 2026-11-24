# Identifiable Dynamics Registers for Native Latent Action Models

## Motivation

Native LAM (World Models as Native Latent Action Models) distills video dynamics into learnable registers via a motion-only counterfactual objective, but that objective only enforces motion-correlated variation; it never constrains the register state to be a sufficient, disentangled statistic of the true dynamics. Consequently the register-to-dynamics mapping is underconstrained, and PAWlicy's VLA pretraining inherits a dynamics teacher whose latent actions are not identifiable — two motion-different segments can map to the same register state, or one segment's registers can entangle appearance with motion. This is the recurrent meta-problem across video-world-model work (e.g., Video World Models with Long-term Spatial Memory, Learning 3D Persistent Embodied World Models): latent state that is merely correlated with dynamics, not pinned to it, silently degrades downstream planning and pretraining.

## Key Insight

Motion-only counterfactual pairs are invariant to any register reparameterization that preserves motion-correlation, so identifiability requires an external anchor — action-labeled segments provide a conditional sufficient statistic that fixes the register-to-dynamics map up to the action-equivalence class, making the learned latent action recoverable rather than merely correlated.

## Method

**What it is.** IDR-LAM (Identifiable Dynamics Registers for Native LAM) augments the Native LAM training loop with an auxiliary inverse-dynamics head and a register-minimality penalty, producing registers whose state is an identifiable function of the motion between a counterfactual pair. Inputs: counterfactual temporal pairs (frames differing in motion only) and a smaller set of action-labeled video segments; outputs: dynamics registers z_t and a trained inverse-dynamics head used only at train time.

**Load-bearing assumption (stated explicitly).** The method assumes that adding an auxiliary inverse-dynamics cross-entropy head on action-labeled segments, plus a Gaussian total-correlation penalty and L1 sparsity on register differences, makes the register-to-dynamics map identifiable up to action-equivalence — i.e., that a discriminative action-prediction constraint plus marginal-independence regularization pins the latent geometry rather than merely adding another predictive-but-reparameterization-invariant readout. This assumption is load-bearing and is tested directly by the reparameterization-invariance probe in the experiment section; if the probe shows register differences are not stable under invertible transforms that preserve action prediction, the identifiability claim is false.

**How it works.**

```pseudocode
# Native LAM backbone: pretrained video diffusion transformer D
# with learnable dynamics registers R (K=8 registers, dim d=64)
# IDM_head: 2-layer MLP, hidden=256, GeLU activation
# Optimizer: AdamW, lr=1e-4 for D and R, lr=1e-3 for IDM_head, batch size 32
for each training step:
    # Phase 1: motion-only counterfactual objective (inherited)
    (x_t, x_t_plus, x_t_minus) = sample_counterfactual_pair()   # same scene, motion differs
    z_t       = R.encode(D, x_t)
    z_t_plus  = R.encode(D, x_t_plus)
    z_t_minus = R.encode(D, x_t_minus)
    L_cf = || D.forward(z_t, x_t) - x_t_plus ||^2
         + || D.forward(z_t, x_t) - x_t_minus ||^2      # FDM must reproduce both

    # Phase 2: auxiliary inverse-dynamics head on action-labeled segments
    (x_a, x_b, a) = sample_action_labeled_segment()       # a = true action, 12 classes
    z_a = R.encode(D, x_a); z_b = R.encode(D, x_b)
    a_hat = IDM_head(z_a, z_b)                            # MLP, 2-layer, hidden 256, GeLU
    L_inv = CE(a_hat, a)                                  # cross-entropy over 12 action classes

    # Phase 3: register minimality / disentanglement penalty
    # mutual-information proxy: total correlation across registers
    Z = stack([z_t, z_t_plus, z_t_minus, z_a, z_b])       # (B, K, d)
    L_tc = sum_k KL( q(z_k) || prod_j q(z_j) )            # Gaussian q, closed-form, K=8
    # sparsity: only registers that change between counterfactual pair may be active
    delta = |z_t_plus - z_t_minus|                        # (B, K)
    L_sparse = || delta ||_1 / K

    L = L_cf + lambda_inv * L_inv + lambda_tc * L_tc + lambda_sp * L_sparse
    # lambda_inv=0.5, lambda_tc=0.01, lambda_sp=0.001 (tuned on held-out)
    update(D, R, IDM_head)
```

**Why this design.** We chose an auxiliary inverse-dynamics head over a purely unsupervised disentanglement penalty because action labels provide the external anchor that motion-only counterfactuals lack — the head forces the register difference z_a - z_b to be predictive of the true action, breaking the reparameterization invariance that makes Native LAM's registers unidentifiable; the cost is that action-labeled segments are scarcer than raw video, so we weight L_inv at 0.5 and rely on the counterfactual term to carry most of the gradient. We chose a total-correlation penalty over an adversarial disentanglement loss because TC has a closed-form Gaussian estimator that is stable at the small register dimension (d=64) and does not require a discriminator that could collapse; the cost is that TC only penalizes linear dependence, so nonlinear entanglement can survive. We chose an L1 sparsity penalty on the counterfactual register difference over a hard top-k selection because L1 is differentiable and lets the model learn which registers carry motion; the cost is that lambda_sp must be tuned to avoid killing all registers. A domain expert would not describe this as 'Native LAM + an IDM head' because the IDM head is not a readout — it is a constraint that, combined with the TC penalty, changes the geometry of the register space itself, and we verify this by showing the register space is identifiable up to action-equivalence, which a plain auxiliary head does not establish.

**Why it measures what we claim.** L_inv (cross-entropy of the inverse-dynamics head) measures action-identifiability of the register state because the assumption that action-labeled segments share the same motion-generating process as the counterfactual pairs makes a_hat a sufficient statistic for the true action given z_a - z_b; this assumption fails when action labels are coarse (e.g., 'move left' conflates different speeds), in which case L_inv reflects label granularity rather than register identifiability. L_tc (total correlation across registers) measures disentanglement because the assumption that independent registers should have independent marginals under a Gaussian q makes TC zero iff the registers are marginally independent; this assumption fails when the true dynamics require correlated registers (e.g., rotation couples x and y), in which case TC penalizes a legitimate coupling. L_sparse (L1 on counterfactual register difference) measures motion-locality because the assumption that only motion-carrying registers should change between a counterfactual pair makes the L1 norm a proxy for the number of active motion registers; this assumption fails when appearance also changes slightly between pairs, in which case L_sparse reflects nuisance variation rather than motion locality.

## Contribution

['IDR-LAM, a training framework that augments Native LAM with an auxiliary inverse-dynamics head and a total-correlation plus motion-sparsity penalty, converting the underconstrained register-to-dynamics mapping into an empirically identifiable latent action.', 'A concrete empirical finding that action-labeled segments, even when scarce, are sufficient to anchor register identifiability, and that the resulting dynamics teacher improves PAWlicy VLA pretraining compared to the unconstrained Native LAM teacher.', "A diagnostic protocol (register-probe accuracy and counterfactual consistency gap) for measuring whether a video world model's latent action is identifiable, applicable to other world-model teachers."]

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Counterfactual motion pairs + action-labeled egocentric video (Ego4D, 12 action classes, 512 labeled segments) | Motion-only pairs isolate dynamics; labels anchor identifiability. |
| Primary metric | Downstream action-prediction accuracy from register differences (linear probe, 5-fold cross-validation) | Directly tests whether registers are action-identifiable. |
| Baseline | Native LAM (pretrained video diffusion transformer registers) | Tests whether our constraints add identifiability beyond inherited objective. |
| Baseline | Native LAM + plain auxiliary IDM head (no TC/sparsity) | Isolates whether geometry constraints, not just the head, drive gains. |
| Baseline | Supervised IDM trained from scratch on action-labeled video | Tests whether world-model registers beat direct dynamics learning. |
| Ablation-of-ours | IDR-LAM minus total-correlation penalty | Tests whether disentanglement constraint is necessary for identifiability. |
| Diagnostic | Reparameterization-invariance probe: fit two linear maps from registers to actions that both achieve >90% training accuracy but disagree on held-out fine-grained actions | Directly measures reparameterization ambiguity in Native LAM registers. |

### Why this setup validates the claim

The central claim is that IDR-LAM's registers become an identifiable function of motion, not merely a predictive but reparameterization-ambiguous encoding. The plain Native LAM baseline tests the first sub-claim: without an external action anchor, register differences should be predictive but not consistently aligned to action semantics, so held-out action decoding from registers should be weak. The plain-IDM-head baseline tests the crucial second sub-claim: an auxiliary head alone can make registers predictive without changing register geometry, so if our gains came only from adding a head, this baseline would match us; the gap between it and IDR-LAM is the causal signature of the TC and sparsity constraints reshaping the register space. The from-scratch supervised IDM tests whether the pretrained world model's dynamics are actually doing the work, or whether any action-labeled training suffices. The ablation isolates the TC term's contribution to disentanglement. The reparameterization-invariance probe directly measures the failure mode: if Native LAM registers admit two linear maps that both fit training actions but disagree on held-out fine-grained actions, the ambiguity exists; if IDR-LAM registers do not admit such disagreeing maps, the constraints have pinned the geometry. Action-prediction accuracy from register differences is the right primary metric because identifiability is precisely the property that register differences form a sufficient statistic for the action; a predictive-but-unidentifiable model can score well on reconstruction yet fail this decoding test under distribution shift in action granularity.

### Expected outcome and causal chain

**vs. Native LAM** — On a case where two segments share coarse action labels but differ in speed or direction nuance, plain Native LAM's registers encode motion densely and entangled across layers, so a linear or MLP probe on register differences confuses these segments and action decoding collapses on the fine-grained subset. Our method instead forces register differences to predict the true action through the IDM head, and the TC penalty spreads distinct motion factors across registers, so we expect a large accuracy gap on fine-grained action subsets and near-parity on coarse subsets. The reparameterization-invariance probe should show Native LAM registers admit two linear maps with >90% training accuracy but <60% agreement on held-out fine-grained actions, while IDR-LAM registers show <10% disagreement.

**vs. Native LAM + plain auxiliary IDM head** — On a case where the register space is free to rotate or rescale while still predicting actions, the plain-head baseline achieves high training accuracy but its register geometry remains ambiguous, so decoding degrades when the probe is trained on one action distribution and tested on another. Our method instead constrains the geometry via total correlation and motion-locality sparsity, making the register state identifiable up to action-equivalence, so we expect the plain-head baseline to match us in-distribution but fall noticeably behind under action-granularity shift. The reparameterization-invariance probe should show the plain-head baseline still admits disagreeing linear maps (disagreement >30% on held-out fine-grained actions), while IDR-LAM does not.

**vs. Supervised IDM from scratch** — On a case with scarce action labels but abundant unlabeled video, the from-scratch IDM overfits the labeled segments and fails to capture dynamics present only in unlabeled motion pairs, producing erratic action predictions on rare motions. Our method instead inherits the pretrained video diffusion transformer's dynamics through the counterfactual objective and uses labels only as an anchor, so we expect the from-scratch IDM to trail us most sharply on rare-motion subsets while approaching parity on frequent actions.

### What would falsify this idea

If the plain-IDM-head baseline matches IDR-LAM on fine-grained action decoding and under action-granularity shift, then the TC and sparsity constraints add nothing and the central claim that they reshape register geometry is wrong. Likewise, if the reparameterization-invariance probe shows IDR-LAM registers still admit disagreeing linear maps that fit training actions but diverge on held-out fine-grained actions, the identifiability mechanism is not the cause.

## References

1. World Models as Native Latent Action Models
