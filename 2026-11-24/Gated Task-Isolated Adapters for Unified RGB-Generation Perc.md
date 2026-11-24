# Gated Task-Isolated Adapters for Unified RGB-Generation Perception

## Motivation

Unified perception frameworks that adapt a single generative backbone to many tasks face a structural parameter-competition bottleneck: Open Vision Banana (OVB) shows that depth and 2D segmentation compete for one shared low-rank adapter, so adding a task contract perturbs the representation used by sibling tasks and unaligned depth accuracy stays far below aligned accuracy. This is a meta-level failure of the prevailing benign-unification assumption — a single shared capacity budget forces gradient updates from one task to overwrite the subspace another task depends on, and no amount of extra training data removes the conflict because the conflict lives in the parameter sharing itself.

## Key Insight

Task-conditioned adapter isolation turns the shared-parameter competition into a compositional sum whose cross-task interference term is structurally zero by construction, so each task's representation is invariant to sibling-task gradient updates regardless of how imbalanced the task losses are.

## Method

**What it is.** We introduce GATED-TIA (Gated Task-Isolated Adapters), a drop-in replacement for OVB's single shared low-rank adapter. It takes the same inputs as OVB — a frozen 7B Qwen-Image-2.1 diffusion backbone, an RGB-space task contract (encoder, instruction, decoder), and an image — and produces a task-conditioned residual update to the backbone's attention projections, composed from per-task low-rank adapters by a learned gate.

**Load-bearing assumption (stated explicitly).** GATED-TIA's scale-commitment claim rests on the assumption that OVB's unaligned depth δ_1 instability (0.11–0.50) is caused by parameter-subspace competition between depth and segmentation gradients in the shared low-rank adapter, so that splitting capacity into disjoint per-task adapters will recover scale commitment. This assumption is not verified by OVB's reported evidence: median-aligned δ_1 stays at 0.89–0.93 while unaligned δ_1 wanders, and a ten-seed ensemble lifts unaligned δ_1 from 0.348 to 0.516 without retraining, which points to a frozen-prior scale under-determination rather than gradient overwriting. We therefore treat the competition diagnosis as a hypothesis to be tested by the calibration runs below, not as an established fact, and we frame the primary contribution as: task-conditioned adapter composition improves per-task fidelity over a shared adapter under matched parameter budget. The scale-commitment recovery is a secondary, conditional claim that holds only if the depth-only OVB control shows the same unaligned δ_1 instability as the mixed-task OVB run.

**How it works.**

```pseudocode
# Backbone frozen. Attention projection W0 in R^{d x d}.
# Four task contracts: depth, semantic, instance, referring.
# Each task t has a dedicated LoRA pair (A_t in R^{r x d}, B_t in R^{d x r}), r = 16.
# Gate g_phi: 2-layer MLP over [task_embedding e_t (d=64); pooled image feature f (d=64)] -> softmax over 4 tasks.
# Gate MLP: hidden width 128, GeLU activation, output dimension 4, temperature 1.0.

for each training step:
    (image, task_t, instruction) -> contract_t
    f = backbone.pooled_feature(image)                # frozen, d=64
    g = softmax(MLP_phi([e_t ; f]))                   # gate over 4 tasks, temperature 1.0
    delta = sum_k g[k] * (B_k @ A_k)                  # composed residual, d x d
    W_eff = W0 + alpha * delta                        # alpha = 1.0 fixed
    y = RGB_decode(backbone_forward(x; W_eff), contract_t)
    L = L_task(y, target) + lambda * KL(g || onehot(t))   # lambda = 0.1, annealed to 0 by step 20k

# Inference: g is computed once per (image, task) pair; task identity is known from the contract.
```

Hyperparameters: rank r=16 per adapter; gate MLP hidden width 128 with GeLU; gate input is concatenation of a 64-d learned task embedding and a 64-d pooled image feature; KL-to-onehot regularizer weight lambda=0.1 linearly annealed to zero over the first 20k steps so the gate starts task-aligned and then is free to learn input-dependent composition. Training uses a single 100k-update run per configuration with a pre-registered checkpoint-evaluation schedule (every 10k steps, 10 checkpoints per run).

**Calibration and verification runs (to test the load-bearing assumption).** Before interpreting any GATED-TIA result as scale-commitment recovery, we run three calibration controls under identical data and step budget: (C1) OVB shared rank-64 adapter trained on depth only; (C2) OVB shared rank-64 adapter trained on all four tasks (the original OVB regime); (C3) GATED-TIA with four rank-16 adapters (total rank 64, matched parameter budget). If C1 shows the same unaligned δ_1 instability as C2, the competition diagnosis is falsified and the paper repositions GATED-TIA as a fidelity/transfer mechanism rather than a scale-commitment fix. If C1 is stable and C2 unstable, the competition diagnosis is supported and the isolation signature is tested by comparing C3 to C2.

**Mechanism diagnostic (gradient conflict and subspace overlap).** To directly measure the proposed parameter-competition mechanism, we compute at each checkpoint: (i) per-step gradient cosine similarity between depth and segmentation gradients on the shared adapter in OVB (C2) versus on the disjoint adapters in GATED-TIA (C3); (ii) principal-angle overlap between the top-8 singular subspaces of the depth-gradient and segmentation-gradient matrices accumulated over a 500-step window. We report the correlation between conflict magnitude (mean cosine similarity and mean principal angle) and unaligned δ_1 drift across the 10 checkpoints of each run. A positive correlation in OVB and near-zero conflict in GATED-TIA is the direct evidence that parameter isolation removes the interference; absence of correlation falsifies the causal path.

**Why this design.** We chose four dedicated rank-16 adapters over a single shared rank-64 adapter because OVB's shared-adapter competition is a parameter-subspace conflict, and splitting capacity removes the conflict at its source rather than regularizing it away — accepting the cost that total adapter parameters grow 4x (still <0.5% of the 7B backbone) and that cross-task transfer must now be recovered by the gate rather than inherited for free. We chose a learned gate over hard task routing because the task contract already supplies task identity, so a hard route would collapse GATED-TIA to the trivial "one adapter per task" baseline and forfeit any input-dependent sharing; the gate lets a depth input borrow segmentation-learned features when the pooled feature indicates shared structure, accepting the cost of a small amount of re-introduced coupling that we bound with the KL-to-onehot warmup. We chose to anneal the KL term to zero rather than keep it fixed because a permanently task-aligned gate would make the gate a no-op at convergence, accepting the cost that late-training gate drift must be monitored via the gate-entropy ablation. A domain expert would not describe this as the standard mixture-of-experts composition, because the gate is conditioned on the *known task contract* as well as the input, making the composition a task-conditioned correction to an identity prior rather than a free-form router — the KL warmup is what distinguishes it from a generic MoE router.

**Why it measures what we claim.** The cross-task interference term — the change in task t's output when sibling task s receives a gradient step — measures *compositional non-interference* because the parameter-isolation assumption makes sibling gradients orthogonal in parameter space (they touch disjoint A_s, B_s), so any measured output change must flow through the gate weights phi; this assumption fails when the gate assigns nonzero mass to a sibling adapter on a task-t input, in which case the measured interference reflects gate-mediated coupling rather than parameter competition. The gate mass g[k] on the correct task measures *task-conditional capacity allocation* because the KL-to-onehot warmup assumption makes g approximately one-hot early in training, so deviations from one-hot at convergence indicate genuine input-dependent sharing; this assumption fails when the pooled feature f is uninformative (e.g., out-of-distribution inputs), in which case g reflects the task-embedding prior alone rather than input-conditioned allocation. The unaligned NYUv2 delta_1 gap between GATED-TIA and OVB measures *scale-commitment recovery from non-interference* because the isolation assumption implies depth's adapter is never overwritten by segmentation gradients, so any residual scale gap must be attributed to the backbone's frozen generative prior rather than to adapter competition; this assumption fails when the backbone itself is the binding constraint, in which case the gap reflects backbone capacity and not adapter design. The gradient cosine similarity and principal-angle overlap between depth and segmentation gradients measure *parameter-subspace competition* because the competition hypothesis predicts that conflicting gradients on the shared adapter align in the same subspace, so high cosine similarity and small principal angles should co-occur with δ_1 drift; this assumption fails when the gradients are orthogonal despite shared parameters (e.g., due to batch ordering), in which case the diagnostic reflects optimization noise rather than structural competition. The depth-only OVB control (C1) measures *whether the scale instability is caused by task competition or by the frozen prior* because if the instability persists without any segmentation gradients, the competition diagnosis is falsified; this assumption fails when the depth-only run is stable but the mixed run is unstable, in which case the competition diagnosis is supported and the isolation signature is the relevant test.

## Contribution

["GATED-TIA, a task-conditioned mixture-of-low-rank-adapters with a contract-conditioned gate and KL-to-onehot warmup, that replaces OVB's single shared adapter while preserving the unified RGB-generation interface.", 'A design principle: task-contract-conditioned gating with annealed task alignment recovers cross-task transfer without reintroducing the parameter-subspace competition that a single shared adapter suffers.', 'A diagnostic protocol — the sibling-gradient interference probe and gate-entropy ablation — that separates parameter-competition interference from gate-mediated coupling in unified perception models.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | NYUv2 depth + ADE20K segmentation + referring splits | Matches OVB's four-task contract and scale metric |
| Primary metric | Unaligned depth delta_1 (plus cross-task interference) | Directly probes scale commitment and non-interference |
| Secondary metric | Gradient cosine similarity and principal-angle overlap between depth and segmentation gradients | Direct diagnostic of the parameter-competition mechanism |
| Baseline | OVB shared low-rank adapter (rank-64, all four tasks) | The exact mechanism we replace |
| Baseline | OVB shared low-rank adapter, depth-only training (C1) | Causal anchor: isolates task competition from frozen-prior effects |
| Baseline | Hard per-task adapter routing (four rank-16 adapters, no gate) | Tests whether the gate adds anything over isolation |
| Baseline | Fixed uniform gate over four rank-16 adapters | Tests whether learned task-conditioned composition beats equal mixing |
| Baseline | Closed Vision Banana | Upper reference for the scale gap |
| Ablation-of-ours | GATED-TIA with KL warmup removed | Isolates the task-aligned prior's contribution |
| Ablation-of-ours | GATED-TIA with gate-entropy monitoring | Detects late-training gate drift |
| Compute budget | Single 100k-update run per configuration; checkpoint evaluation every 10k steps (10 checkpoints per run); smaller open diffusion backbone (e.g., 2B-parameter open generator) for ablation-of-ours and hard-routing baselines | Fits compute budget while preserving the causal comparisons |

### Why this setup validates the claim

The central claim is that OVB's shared adapter causes parameter-subspace competition between depth and segmentation, and that isolating capacity per task recovers scale commitment. OVB's shared adapter is the direct counterfactual: same backbone, same contracts, same data, only the adapter composition differs, so any delta_1 change is attributable to the isolation mechanism rather than to backbone or data. The depth-only OVB control (C1) is the causal anchor that tests the load-bearing assumption: if C1 shows the same unaligned δ_1 instability as the mixed-task OVB run, the competition diagnosis is falsified and the scale-commitment claim must be withdrawn; if C1 is stable, the competition diagnosis is supported and the isolation signature becomes the relevant test. Hard per-task routing tests the second sub-claim — that input-conditioned gating beats pure isolation — because if hard routing matches GATED-TIA, the gate is doing nothing and the design collapses to the trivial baseline. The fixed uniform gate tests whether learned task-conditioned composition beats equal mixing, separating the gate's contribution from the 4x capacity increase. The KL-removed ablation tests the third sub-claim, that the task-aligned warmup prevents early gate drift from re-introducing the coupling we removed. Unaligned delta_1 is the right primary metric because it is exactly the quantity OVB reports as unstable (wandering 0.11–0.50), so it is sensitive to the predicted effect; the gradient-conflict and subspace-overlap diagnostics are the secondary measures that confirm the causal path is parameter isolation, not incidental regularization.

### Expected outcome and causal chain

**vs. OVB shared adapter (mixed-task)** — On a depth input whose gradient step coincides with a segmentation batch, the shared rank-64 adapter receives conflicting updates and the depth scale estimate drifts, because one parameter subspace must serve both tasks and segmentation's gradient overwrites the scale-sensitive directions. Our method instead routes the depth gradient only into the depth adapter, because sibling adapters are disjoint in parameter space, so the scale-bearing weights are never touched by segmentation. We expect a noticeably larger unaligned delta_1 on mixed-task training runs than on depth-only runs for OVB, and near-parity between those two regimes for GATED-TIA — the isolation signature. The gradient-conflict diagnostic should show high cosine similarity and small principal angles between depth and segmentation gradients in OVB, and near-zero conflict in GATED-TIA, with conflict magnitude correlating with δ_1 drift across checkpoints.

**vs. OVB depth-only (C1)** — If C1 already shows the same unaligned δ_1 instability as the mixed-task OVB run, the competition diagnosis is falsified and the scale-commitment claim is withdrawn; GATED-TIA is then repositioned as a fidelity/transfer mechanism, and the expected outcome is that GATED-TIA matches or exceeds OVB on per-task fidelity metrics (e.g., segmentation mIoU, depth RMSE) under matched parameter budget, without claiming scale-commitment recovery. If C1 is stable, the competition diagnosis is supported and the isolation signature (mixed-task vs depth-only parity for GATED-TIA) is the predicted outcome.

**vs. Hard per-task routing** — On a referring-segmentation input that shares low-level structure with semantic segmentation, hard routing sends the input only to the referring adapter, which was trained on far less data, so it produces weaker boundaries than a model that could borrow semantic features. Our method instead lets the gate assign partial mass to the semantic adapter when the pooled feature indicates shared structure, recovering that transfer. We expect GATED-TIA to match hard routing on depth (where isolation suffices) but to show a modest gain on referring and instance segmentation, with gate mass on sibling adapters rising specifically on those inputs.

**vs. Fixed uniform gate** — A fixed uniform gate averages all four adapters equally, which dilutes task-specific capacity and cannot exploit input-dependent sharing. Our method learns to concentrate mass on the correct task early (via KL warmup) and then to borrow selectively. We expect GATED-TIA to outperform the fixed uniform gate on all four tasks, with the largest margin on depth (where task-specific scale-sensitive directions are most vulnerable to dilution), demonstrating that the learned task-conditioned composition — not merely the 4x capacity increase — drives the gain.

**vs. Closed Vision Banana** — On an out-of-distribution scene, the closed system commits absolute scale because its instruction tuning saw broad scale variation, whereas OVB's open generator leaves unaligned delta_1 unstable across checkpoints. Our method does not fix the backbone's prior, but by removing adapter competition it should let the depth adapter converge to whatever scale signal the frozen prior supports, so we expect GATED-TIA to close part of the OVB-to-closed gap on unaligned delta_1 while leaving a residual gap that we attribute to the backbone — the diagnostic that separates adapter design from backbone capacity. If the depth-only OVB control shows the same instability, this expected outcome is revised to attribute the entire gap to the backbone prior.

### What would falsify this idea

If GATED-TIA's gain over OVB is uniform across depth-only and mixed-task runs rather than concentrated in the mixed-task regime where the interference mechanism predicts it, the parameter-competition explanation is wrong. Likewise, if the depth-only OVB control (C1) shows the same unaligned δ_1 instability as the mixed-task run, the load-bearing assumption is falsified and the scale-commitment claim must be withdrawn; if removing the KL warmup produces no gate drift and no interference increase, the task-aligned prior is doing no work and the gate is a no-op.

## References

1. Open Vision Banana: An Open Framework for Unified Visual Perception through Generation in RGB Space
