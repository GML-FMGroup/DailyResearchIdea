# Factorial Component Ablation of Hybrid Attention and Blockwise Token Separation in Unified Multimodal Models

## Motivation

Unified multimodal models such as UniSpark report only a single aggregate score (2470) and sparse quantitative comparisons, so the field cannot attribute performance gains to specific architectural choices like full-attention versus linear-attention branch allocation or blockwise token separation. This is a meta-level problem: across unified multimodal architectures, aggregate scores conflate multiple design decisions, making it impossible to identify which components expand performance boundaries and which are inert. Without a controlled, per-component ablation framework, architectural progress remains ungrounded and non-reproducible.

## Key Insight

Performance-boundary expansion in hybrid-attention unified models is attributable only when architectural factors are varied factorially and their interaction effects are measured, because the aggregate score is a sum of main effects and interactions that cannot be decomposed post hoc.

## Method

**What it is** — We introduce FACTOR-ABLATE, a controlled factorial ablation framework that takes a UniSpark-style hybrid-attention unified model and outputs per-component performance deltas for each combination of (i) full-attention vs linear-attention branch allocation and (ii) blockwise token-separation on/off. Inputs: a trained UniSpark-9B checkpoint and a fixed evaluation suite covering VQA, text-to-image, editing, and interleaved prediction. Outputs: a 2×2×K table of task-level scores and interaction effects, plus a capacity-confound bound and a convergence-verified adaptation check.

**Load-bearing assumption (stated explicitly)** — The framework assumes that reallocating full vs. linear attention branches while holding parameter count fixed via width adjustment, followed by finetuning to convergence, yields cells whose differences isolate the causal effect of branch allocation rather than reflecting capacity/optimization mismatch or insufficient adaptation. This assumption is not taken on faith: it is bounded by two control cells (width-matched vs. FLOP-matched) and a per-cell convergence criterion, and the interaction term is reported only after both controls show the confound is smaller than the effect.

**How it works** —
```pseudocode
# Factors:
# F1: branch allocation ∈ {full-heavy (75% full), linear-heavy (25% full), balanced (50% full)}
# F2: blockwise separation ∈ {on, off}
# F3 (control): seed ∈ {1,2,3}
# Capacity controls (new, explicit):
#   C_width: width-adjusted to match parameter count (original)
#   C_flops: width-adjusted to match forward FLOPs of the balanced cell (reported as capacity-confound bound)
#   C_null:  width-only sweep with no attention change (null control)
# Adaptation control:
#   finetune until validation loss plateaus within tolerance (max 15000 steps), not fixed 5000

for each (f1, f2) in F1 × F2:
    for seed in F3:
        model = load_UniSpark_checkpoint()
        model = reallocate_attention_branches(model, f1, capacity_mode=C_width)   # replace linear layers with full layers or vice versa, keeping parameter count fixed via width adjustment
        model = set_blockwise_separation(model, f2)          # toggle attention masks and recurrent-state retention rules
        finetune_until_convergence(model, curriculum_stages_1_to_4,
                                   val_tolerance=0.005, patience=3, max_steps=15000) # same data mixture and schedule across all cells
        scores[task][f1][f2][seed] = evaluate(model, suite)

# Capacity-confound bound (run for the two extreme F1 cells only, seed=1):
for f1 in {full-heavy, linear-heavy}:
    model_flops = reallocate_attention_branches(load_UniSpark_checkpoint(), f1, capacity_mode=C_flops)
    model_flops = set_blockwise_separation(model_flops, on)
    finetune_until_convergence(model_flops, ...)
    scores_flops[task][f1] = evaluate(model_flops, suite)
    delta_capacity[task][f1] = scores[task][f1][on][1] - scores_flops[task][f1]

# Null control (width-only, no attention change, seed=1):
model_null = width_sweep_only(load_UniSpark_checkpoint(), width=balanced_width)
finetune_until_convergence(model_null, ...)
scores_null[task] = evaluate(model_null, suite)

# Compute per-component deltas and interactions:
for task in suite:
    main_F1 = mean_over(f2,seed) scores[task][full-heavy] - scores[task][linear-heavy]
    main_F2 = mean_over(f1,seed) scores[task][on] - scores[task][off]
    interaction = (scores[full-heavy,on] - scores[full-heavy,off]) - (scores[linear-heavy,on] - scores[linear-heavy,off])
    report(task, main_F1, main_F2, interaction, std_over_seeds,
           capacity_bound=delta_capacity[task], null_control=scores_null[task])

# Full-attention-only separation diagnostic (tests assumption A):
model_fa_only = set_all_branches_full(load_UniSpark_checkpoint())
for f2 in {on, off}:
    model_fa_only_f2 = set_blockwise_separation(model_fa_only, f2)
    finetune_until_convergence(model_fa_only_f2, ...)
    scores_fa_only[task][f2] = evaluate(model_fa_only_f2, suite)
separation_effect_full_only[task] = scores_fa_only[task][on] - scores_fa_only[task][off]
# If separation_effect_full_only is non-zero, assumption A fails and interaction reflects mixed mechanisms.
```

**Why this design** — We chose a full factorial design over one-factor-at-a-time ablation because interaction effects between branch allocation and blockwise separation are the quantity of interest: a linear-attention-heavy model may depend more on blockwise separation to prevent noisy visual targets from contaminating recurrent states, so marginal deltas would misattribute gains. We chose to hold parameter count fixed via width adjustment rather than let it vary because otherwise branch allocation is confounded with capacity, accepting the cost that width-adjusted models are not identical to the original UniSpark architecture. We chose to finetune each cell to convergence (validation-loss plateau within 0.005, patience 3, max 15000 steps) rather than a fixed 5000 steps because the recurrent-state retention rules and attention masks are learned behaviors and a fixed short budget would penalize the off-condition for slower adaptation; this accepts the cost of variable compute per cell (estimated 12 runs × up to 15000 steps, ~2× the original budget). We added a FLOP-matched control and a width-only null control to bound the capacity confound explicitly, and a full-attention-only separation diagnostic to test whether separation affects full-attention layers through an independent pathway. A domain expert would not describe this as a simple hyperparameter sweep because the interaction term directly tests whether blockwise separation is a compensating mechanism for linear-attention state contamination, a structural hypothesis that UniSpark's aggregate score cannot address, and the controls make the capacity and adaptation confounds measurable rather than assumed.

**Why it measures what we claim** — The main effect of F1 (full-heavy minus linear-heavy) measures the causal contribution of full-attention capacity to task performance because the only difference between cells is the branch allocation, assuming that width adjustment preserves the functional role of each branch type; this assumption fails when width adjustment changes the effective receptive field or gradient flow, in which case the delta reflects capacity mismatch rather than attention mechanism — we bound this by reporting the FLOP-matched vs. width-matched delta as an explicit capacity-confound bound and the width-only null control. The main effect of F2 (separation on minus off) measures the causal contribution of blockwise token separation because the only difference is the attention mask and recurrent-state retention rule, assuming that the finetuning budget is sufficient for the model to adapt to the absence of separation; this assumption fails when the budget is too short for the off condition to recover, in which case the delta reflects adaptation speed rather than the necessity of separation — we address this by finetuning to convergence (validation-loss plateau within 0.005, patience 3, max 15000 steps) and reporting the step count per cell. The interaction term measures whether separation is a compensating mechanism specifically for linear-attention state contamination because it contrasts the separation effect across branch allocations, assuming that any other interaction between branch type and separation is negligible; this assumption fails when full-attention layers also benefit from separation through a different pathway (e.g., mask-induced regularization), in which case the interaction reflects a mix of mechanisms — we test this directly with the full-attention-only separation diagnostic and report the interaction only if that diagnostic is near zero.

## Contribution

['A factorial ablation framework (FACTOR-ABLATE) that decomposes UniSpark-style hybrid-attention unified models into branch-allocation and blockwise-separation factors, reporting per-component main effects and interaction effects across four task families.', 'An empirical design principle: the interaction term between branch allocation and blockwise separation quantifies whether token separation is a compensating mechanism for linear-attention state contamination, providing a reusable diagnostic for hybrid-attention architectures.', 'A reproducible protocol with fixed parameter count, identical data curriculum, and multi-seed evaluation that enables attribution of performance-boundary expansion to specific architectural choices rather than aggregate scores.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | UniSpark four-task suite (VQA, T2I, editing, interleaved) | Spans both understanding and generation where attention branches matter |
| Primary metric | Per-task score with factorial interaction term | Interaction directly tests the compensating-mechanism hypothesis |
| Baseline | UniSpark-9B original checkpoint | Anchors deltas to the unmodified hybrid architecture |
| Baseline | Linear-attention-only UniSpark | Tests whether full-attention capacity is the load-bearing factor |
| Baseline | One-factor-at-a-time ablation | Tests whether marginal deltas misattribute the interaction |
| Ablation-of-ours | Blockwise separation off at linear-heavy allocation | Isolates the predicted compensating-mechanism cell |
| Capacity control | FLOP-matched reallocation (C_flops) vs. width-matched (C_width) | Bounds the capacity confound: delta between them is reported as an explicit capacity-confound bound |
| Null control | Width-only sweep with no attention change (C_null) | Ensures width adjustment alone does not produce the observed deltas |
| Adaptation control | Finetune to convergence (val-loss plateau within 0.005, patience 3, max 15000 steps) | Prevents the off-condition from being penalized for slower adaptation |
| Diagnostic | Full-attention-only separation on/off | Tests assumption A: separation affects full-attention layers only through the same state-contamination channel |

### Why this setup validates the claim

The central claim is that blockwise token separation compensates specifically for linear-attention state contamination, which is an interaction claim, not a main-effect claim. That makes the design choice of the 2×2 factorial the falsifiable core: if separation helps equally under full-heavy and linear-heavy allocation, the interaction term collapses toward zero and the compensating-mechanism story dies, even if both main effects look healthy. The original UniSpark checkpoint anchors the deltas so we know the reallocated cells are not simply degraded versions of the reference model. The linear-attention-only baseline tests the sub-claim that full-attention capacity carries a real, separable contribution on generation-heavy tasks. The one-factor-at-a-time baseline is the diagnostic that motivates the whole framework: it should reproduce the same aggregate gain as the factorial design while misassigning it to the wrong factor, which is exactly the misattribution we claim to fix. The per-task score rather than an aggregate is the right metric because the predicted contamination should be concentrated on tasks with dense visual targets (T2I, editing, interleaved) and near-parity on VQA, where recurrent states carry less noisy visual load — an aggregate would wash out precisely the signature we predict. The capacity controls (FLOP-matched vs. width-matched, and width-only null) make the load-bearing assumption measurable: the interaction term is reported only if the capacity-confound bound is smaller than the interaction effect. The convergence criterion (validation-loss plateau within 0.005, patience 3, max 15000 steps) ensures the off-condition is not penalized for slower adaptation, and the full-attention-only separation diagnostic tests whether separation has an independent regularization pathway that would confound the interaction.

### Expected outcome and causal chain

**vs. UniSpark-9B original checkpoint** — On an interleaved-prediction case where long visual context must be carried across turns, the original checkpoint produces drift in later-turn image predictions because its fixed 50/50 branch allocation gives linear-attention layers no explicit rule for evicting stale visual tokens from recurrent state. Our linear-heavy-with-separation cell instead shows that separation restores later-turn fidelity because the mask and retention rules stop noisy visual targets from overwriting state, so we expect the original to sit between our full-heavy and our linear-heavy-with-separation cells, with the gap concentrated on later interleaved turns and near-zero on single-turn VQA. The capacity-confound bound (FLOP-matched vs. width-matched delta) is expected to be small (< 0.5 points) relative to the interleaved-turn gap, and the width-only null control is expected to show no significant change, confirming the gap is not a width artifact.

**vs. Linear-attention-only UniSpark** — On a text-to-image case requiring fine spatial binding (e.g., counting or attribute binding across many objects), the linear-only baseline produces blurred or miscounted structure because linear attention compresses the visual target into a fixed-size state that cannot resolve fine spatial detail. Our full-heavy cell instead recovers the binding because full-attention layers retain per-token resolution over the visual target, so we expect a clear full-heavy advantage on T2I and editing, parity on VQA, and — critically — a separation effect that is larger in the linear-heavy cells than in the full-heavy cells, which is the interaction signature. The full-attention-only separation diagnostic is expected to show near-zero separation effect, supporting assumption A; if it is non-zero, the interaction is interpreted as a mix of mechanisms.

**vs. One-factor-at-a-time ablation** — On a case where branch allocation and separation are varied separately, the OFAT baseline reports a separation gain that looks like a general improvement because it averages over branch allocations, attributing to separation what is actually a linear-attention-specific rescue. Our factorial design instead decomposes the same data into a small full-heavy separation effect and a large linear-heavy separation effect because the interaction term explicitly contrasts the two, so we expect the OFAT marginal delta to roughly equal the average of our two conditional deltas while the interaction term itself is clearly non-zero and same-signed across seeds. The convergence criterion ensures the OFAT baseline is not advantaged by a fixed short budget that would truncate the off-condition's adaptation.

### What would falsify this idea

If the interaction term is near zero across all four tasks — i.e., the separation effect is statistically indistinguishable between full-heavy and linear-heavy allocation — then separation is not a compensating mechanism for linear-attention state contamination and the central structural hypothesis is wrong. A second falsifier: if the OFAT baseline's marginal deltas match our conditional deltas cell-for-cell, then the factorial decomposition adds nothing and the misattribution claim is unsupported. A third falsifier: if the capacity-confound bound (FLOP-matched vs. width-matched delta) is larger than the interaction effect, or if the full-attention-only separation diagnostic shows a non-zero effect comparable to the linear-heavy separation effect, then the interaction is confounded by capacity or an independent regularization pathway and the compensating-mechanism claim is not established.

## References

1. UniSpark: Exploring Hybrid Attention for Multimodal Understanding and Generation
