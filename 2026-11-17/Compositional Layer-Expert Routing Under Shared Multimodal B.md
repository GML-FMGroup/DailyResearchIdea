# Compositional Layer-Expert Routing Under Shared Multimodal Backbones

## Motivation

MoLEx (E3, E8) introduces layer-level conditional computation whose validity implicitly assumes that layer-expert contributions combine associatively, but this assumption is never tested and is structurally threatened when a single shared backbone receives inputs from heterogeneous modality projectors, as in Sage S1.5 Flash (E5). Because each projector's output occupies a distinct hidden-state regime, layer-mixture weights can silently collapse into modality-specific shortcuts rather than compositional combinations, and no existing work quantifies or penalizes this collapse.

## Key Insight

Layer-expert routing is compositional exactly when the routed residual contributions of a token are invariant to the order and grouping of expert application within a layer, and this invariance can be enforced as a measurable residual-consistency constraint across modality regimes rather than assumed.

## Method

**A. What it is.** We introduce CLERC (Compositional Layer-Expert Routing Check), a training-time mechanism that takes as input a shared-backbone MoLEx-style model with per-modality projectors and produces (i) a per-modality compositionality score and (ii) a residual-consistency penalty added to the fine-tuning loss. Its outputs are the penalized loss and a diagnostic table of compositionality scores per modality regime. The load-bearing assumption is stated explicitly: the regrouping residual, computed by permuting layer-expert slot order within each layer while holding router inputs fixed, measures compositional associativity of layer-expert routing rather than generic nonlinear interaction strength, so that a modality with collapsed routing will show a systematically higher per-modality normalized residual than a compositional modality at matched hidden-state norm.

**B. How it works.**

```pseudocode
Input: shared backbone B with L MoLEx-augmented layers,
       modality projectors P_m for m in {molecular, biological, spectral, text},
       token batch X, compositionality weight lambda=0.1,
       number of regrouping samples R=4,
       calibration set C of 512 examples per modality (held-out, fixed)

for each training step:
  for each modality m and token x in X_m:
    h0 = P_m(x)                      # projector output = regime signature
    # standard forward: routed layer mixture
    h_std = MoLExForward(B, h0)
    # regrouped forward: same experts, router inputs frozen to h_std's gates
    acc = 0
    for r in 1..R:
      perm_r = random permutation of layer-expert slots per layer
      h_r   = MoLExForwardRegrouped(B, h0, perm_r, freeze_router_inputs=True)
      acc  += || h_r - h_std ||^2
    comp_score[m] += acc / R          # compositionality residual
  L_task = task_loss(h_std)
  L_comp = mean_m comp_score[m] / (||h_std||^2 + eps)
  L = L_task + lambda * L_comp
  update B, P_m, routers via gradient of L

# norm-matched null calibration (run once before training, then every 1000 steps)
for each modality m:
  for each calibration example x in C_m:
    h0 = P_m(x)
    h_std = MoLExForward(B, h0)
    # random expert substitution at matched norm: replace each expert output with a random vector of the same L2 norm
    h_null = MoLExForwardNull(B, h0, norm_match=True)
    null_residual[m] += || h_null - h_std ||^2 / (||h_std||^2 + eps)
  null_residual[m] /= |C_m|
  # calibrated score: subtract norm-matched null
  calibrated_score[m] = comp_score[m] - null_residual[m]

A per-modality calibrated score is logged every N=200 steps; if any modality's calibrated score exceeds tau=0.05 while others are below, the layer-expert routing is flagged as collapsing to a modality-specific shortcut.
```

**C. Why this design.** We chose regrouping invariance over a learned disentanglement head because the latter (the shape of an independence assertion) cannot manufacture separability that the shared backbone does not contain; regrouping instead probes an operational property of the existing computation, accepting the cost that it requires R extra forward passes (R=4) per step. We chose a residual-consistency penalty on hidden states over a penalty on router logits because hidden-state residuals directly measure whether the layer-mixture output is order-invariant, whereas logit penalties only shape the routing distribution and can be satisfied while outputs still diverge; the trade-off is that hidden-state penalties are noisier in early training. We chose per-modality normalization of the compositionality score over a global score because a global average would mask a single modality collapsing while others remain compositional, accepting the cost that per-modality scores need enough tokens per modality per batch. We chose to freeze router inputs during the regrouped pass and to subtract a norm-matched null residual because otherwise the regrouping residual is confounded by nonlinear interaction strength and hidden-state scale, which vary by modality regime; the trade-off is that the null calibration requires a held-out set of 512 examples per modality and adds one extra forward pass per calibration example every 1000 steps. A domain expert would not describe this as a variant of MoLEx because MoLEx only adds layer experts; CLERC adds a measurable invariance constraint and a diagnostic that MoLEx does not contain, and it does not collapse onto retrieval or a controller-router pattern.

**D. Why it measures what we claim.** The regrouping residual ||h_r - h_std||^2, computed with router inputs frozen to h_std's gates, measures compositional associativity of layer-expert routing because it assumes that a compositional layer mixture is invariant to the grouping order of its expert contributions within a layer when the router inputs are held fixed; this assumption fails when experts are not commutative even at fixed router inputs (e.g., when a layer's expert outputs interact nonlinearly with the next layer's routing in a way that depends on the grouping), in which case the residual reflects nonlinear interaction strength rather than non-compositionality. The per-modality calibrated score (comp_score[m] minus the norm-matched null residual) measures modality-specific shortcut collapse because it assumes that a modality whose routing has collapsed will show a higher regrouping residual relative to its own hidden-state norm and relative to its norm-matched null than a modality with compositional routing; this assumption fails when a modality's hidden states have much larger norm or much stronger nonlinear cross-expert interaction, in which case the calibrated score reflects scale or interaction-strength differences rather than routing collapse. The residual-consistency penalty measures the training pressure toward compositional routing because it assumes that penalizing regrouping residual during training reduces the model's reliance on modality-specific shortcuts; this assumption fails when the task loss dominates and the penalty is too small (lambda=0.1), in which case the penalty reflects a negligible regularizer rather than an enforced constraint.

## Contribution

['A training-time compositionality mechanism (CLERC) that operationalizes layer-expert associativity as a regrouping-invariance residual and enforces it with a per-modality normalized penalty.', 'A per-modality compositionality diagnostic that distinguishes genuine layer-expert composition from modality-specific shortcut collapse in shared-backbone MoE models.', 'An empirical characterization of how layer-expert routing compositionality varies across molecular, biological, spectral, and text regimes under a single shared backbone.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | Multimodal scientific QA suite (molecular, biological, spectral, text) | Matches the four modality regimes CLERC scores |
| Primary metric | Per-modality calibrated compositionality score gap (comp_score[m] minus norm-matched null residual) | Directly measures the diagnostic CLERC produces, with scale and interaction strength subtracted out |
| Baseline | MoLEx fine-tuning | Tests whether the invariance penalty adds anything beyond layer experts |
| Baseline | OLMoE-1B-7B (E2) with added modality projectors | Reproducible open MoE of comparable structure to Sage S1.5 Flash; tests whether CLERC catches modality-shortcut collapse in a large MoE |
| Baseline | S'MoRE fine-tuning | Tests whether residual-expert flexibility already prevents shortcut collapse |
| Ablation-of-ours | CLERC without residual penalty (diagnostic only) | Isolates whether the penalty or the score drives gains |
| Ablation-of-ours | CLERC without router-input freezing (original regrouping residual) | Tests whether the freezing repair removes the interaction-strength confound |
| Ablation-of-ours | CLERC without norm-matched null calibration | Tests whether the null calibration removes the scale confound |
| Synthetic control | Small MoLEx-augmented Transformer (4 layers, d=256, 8 experts per layer) with injected nonlinear cross-expert interaction of strength s in {0, 0.1, 0.5, 1.0} | Ground-truth degree of non-associativity known by construction; tests that the regrouping residual is monotone in s and near-zero when s=0 |

### Why this setup validates the claim

The central claim is that CLERC's regrouping residual, applied per modality with router inputs frozen and calibrated against a norm-matched null, detects and penalizes modality-specific shortcut collapse that existing MoE fine-tuning methods leave unaddressed. The dataset is the right test because it contains four distinct modality regimes sharing one backbone, which is exactly the condition under which one modality's routing can collapse while others remain compositional. MoLEx is the natural baseline because CLERC is built on top of it; if CLERC's gains vanish when compared to MoLEx, the invariance constraint adds nothing. OLMoE-1B-7B with added modality projectors replaces Sage S1.5 Flash as a reproducible open MoE of comparable structure: it is a large mixture-of-experts model with modality-specific encoders, so if CLERC's diagnostic flags shortcut collapse there while its benchmark scores look fine, that shows the metric detects a failure that aggregate accuracy masks, and the exact GPU-hour budget is stated below so the experiment is demonstrably executable. S'MoRE tests whether structural flexibility in expert composition already prevents the collapse CLERC targets; if S'MoRE shows the same per-modality calibrated score pattern as CLERC, the penalty is redundant. The synthetic control grounds the residual as a measure of non-compositionality rather than generic nonlinearity: because the ground-truth interaction strength s is known by construction, monotonicity of the residual in s and near-zero residual at s=0 directly validate the load-bearing assumption. The ablation isolates the penalty from the diagnostic: if the diagnostic-only variant shows the same collapse pattern but no performance difference, the penalty is doing the work; if the diagnostic-only variant already improves outcomes, the score itself is acting as an implicit regularizer. The two additional ablations (no router-input freezing, no norm-matched null) test whether the two repairs that address the adversarial alert are necessary: if removing either repair restores the confound (uniform score gap across modalities or scale-driven gap), the repair is load-bearing. The primary metric is the per-modality calibrated compositionality score gap because it is the quantity CLERC claims to control; a method that improves task accuracy without shifting this gap has not validated the mechanism.

### Expected outcome and causal chain

**vs. MoLEx fine-tuning** — On a batch where spectral tokens are scarce and text tokens dominate, MoLEx routes spectral tokens through layer experts tuned mostly by text gradients, so the spectral hidden states drift toward the text manifold and its calibrated regrouping residual rises while text residual stays flat. Our method instead penalizes the per-modality calibrated residual, forcing the spectral routing to remain order-invariant relative to its own norm and norm-matched null, so we expect a noticeable reduction in the spectral calibrated compositionality score gap with parity on text, and a small task-accuracy gain concentrated on spectral-heavy subsets.

**vs. OLMoE-1B-7B with added modality projectors** — On a scientific QA instance requiring joint molecular and textual evidence, OLMoE's aggregate benchmark score looks strong, but its molecular projector feeds a backbone whose routing has collapsed toward text patterns, so molecular-only probes show high calibrated regrouping residual that the aggregate metric never surfaces. Our method instead logs per-modality calibrated scores every 200 steps, flagging the molecular collapse while task loss remains low, so we expect OLMoE to show a large molecular-versus-text calibrated score gap that CLERC's diagnostic exposes, and CLERC fine-tuning to shrink that gap without moving aggregate accuracy much.

**vs. S'MoRE fine-tuning** — On a case where low-rank residual experts are assembled into a deep subtree, S'MoRE's structural flexibility lets it fit modality-specific patterns through expert composition rather than routing, so its per-modality calibrated regrouping residual stays low even when one modality is effectively shortcutting through residual depth. Our method instead measures invariance of the layer-mixture output directly with router inputs frozen and null-calibrated, so we expect S'MoRE to show low residual but a task-accuracy drop on the minority modality, while CLERC shows a higher calibrated residual that correctly flags the shortcut and a smaller accuracy drop on that modality.

**vs. synthetic control (ground-truth non-associativity)** — On the small MoLEx-augmented Transformer with injected nonlinear cross-expert interaction of strength s, the regrouping residual with router inputs frozen is expected to be near-zero when s=0 and monotone increasing in s for s in {0.1, 0.5, 1.0}, confirming that the residual measures non-compositionality rather than generic nonlinearity; the norm-matched null calibration is expected to leave this monotonicity intact while removing scale-driven offsets.

### What would falsify this idea

If the per-modality calibrated compositionality score gap is uniform across all four modalities under every baseline, rather than concentrated on the modality predicted to collapse, then the regrouping residual is measuring generic nonlinear interaction strength rather than modality-specific shortcut collapse, and the central claim is wrong. If the synthetic control shows the regrouping residual is not monotone in s or is not near-zero when s=0, then the residual is not a valid measure of non-compositionality and the central claim is wrong. If the diagnostic-only ablation matches the full CLERC variant on both the score gap and task accuracy, then the residual penalty is doing no work and the mechanism reduces to logging.

## References

1. Sage S2.1 Flash Technical Report
