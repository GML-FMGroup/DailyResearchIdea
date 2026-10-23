# Information-Bottleneck Budgets for Policy-Driven Feature Selection in Diffusion Transfer

## Motivation

Policy-Mask prescribes a channel budget (e.g., 128 dimensions) as a hyperparameter rather than deriving it from a task-specific information criterion, so its selection policy cannot certify that the retained subset preserves task-relevant information; the budget is tuned by downstream validation, not by an explicit objective. Marigold and PixArt-Σ similarly assume that their respective approximations (full fine-tuning / fixed compression) are adequate without any preservation guarantee. This structural gap—a budget chosen exogenously rather than as the solution to an information-preservation problem—recurs across transfer-learning paradigms and is what leaves budget-constrained information preservation unestablished.

## Key Insight

The minimal sufficient statistic of the frozen diffusion features for the downstream label is the solution to a rate–relevance trade-off, so the channel budget must be an output of that trade-off (a Lagrange multiplier on the information-bottleneck Lagrangian) rather than an input, and a Gumbel-relaxed channel gate trained with a Lagrangian dual step makes the selected subset's relevance-to-rate ratio the training signal itself.

## Method

**What it is.** IB-Budget is a drop-in replacement for Policy-Mask's prescribed channel budget: it learns a per-channel gate over frozen diffusion features (across timesteps and blocks) by solving an information-bottleneck Lagrangian whose multiplier λ is updated by dual ascent, so the channel budget emerges as the cardinality at which marginal task-relevant information equals λ times marginal rate. Inputs: frozen diffusion features for a batch, downstream labels; outputs: a binary channel mask, its realized budget, and the converged λ.

**How it works.**
```pseudocode
# Frozen backbone: features F[t,b,c] for timestep t, block b, channel c
# Task head h_phi ; gate logits a[t,b,c] (learned) ; temperature tau annealed 1.0 -> 0.3
# IB Lagrangian: L = CE(h_phi(F * m), y) + lambda * R(m)
#   R(m) = E[ sum_{t,b,c} m[t,b,c] ] / C_total     (rate = expected mask density)
#   m[t,b,c] ~ Bernoulli(sigmoid(a[t,b,c]/tau))    (Gumbel-relaxed at train time)
#   Assumption A (stated explicitly): channel dimensions are exchangeable and uncorrelated,
#     so expected cardinality equals the Shannon rate of the selected subset.
#   Failure mode F: correlated channels (e.g., adjacent early U-Net blocks) make density
#     overcount rate; the dual then undershoots the information-theoretic optimum.
#   Calibration check: report residual gap between density and a correlation-aware rate
#     estimate (log-det covariance of retained channels) as a diagnostic.

for each minibatch (x, y):
    F = backbone(x)                                  # frozen
    m = GumbelSigmoid(a / tau)                       # relaxed mask, straight-through
    logits = h_phi( F * m )
    ce = CrossEntropy(logits, y)
    rate = m.mean()                                  # realized channel density
    loss = ce + lambda * rate
    update a, phi by Adam(loss)
    # dual ascent on lambda (projected to [0, lambda_max=10]):
    lambda <- clamp( lambda + eta * (rate - B_target_soft), 0, lambda_max )
    # B_target_soft is a soft ceiling (e.g., 0.5 * C_total); lambda adapts
    # eta=0.01, lambda_max=10; monitor dual residual |rate - B_target_soft|

# At convergence: budget B* = round(rate) ; hard mask = (a > 0)
# Report (B*, lambda*) as the task-specific information criterion.
```

**Why this design.** We chose a Lagrangian dual on a continuous rate penalty over Policy-Mask's hard cardinality constraint because a hard budget provides no gradient signal about *which* channels are worth their cost, whereas the multiplier λ directly prices each channel's marginal rate against its marginal cross-entropy reduction, accepting the cost that λ must be tuned by dual ascent and can oscillate if η is too large (we use η=0.01 with projection). We chose a Gumbel-sigmoid straight-through gate over a top-k selector because top-k makes the budget a discrete input that cannot be differentiated with respect to cardinality, accepting the cost of gradient bias from the straight-through estimator, which we mitigate by annealing τ from 1.0 to 0.3. We chose to keep the diffusion backbone frozen (as in Policy-Mask) rather than fine-tune (as in Marigold) because the information-bottleneck criterion is defined on the frozen feature distribution, accepting the cost that if the frozen features are task-irrelevant the criterion will correctly select near-zero channels and the method will report a small budget rather than rescue the task. A domain expert would not describe this as a variant of Policy-Mask because Policy-Mask's budget is an exogenous hyperparameter tuned on validation mIoU, whereas here the budget is the converged dual variable of an explicit rate–relevance objective and is reported as the task's information criterion; nor is it a variant of Marigold, which fine-tunes all parameters without any rate constraint.

**Why this measures what we claim.** The realized mask density `rate` measures the *rate* term of the information bottleneck (the expected number of retained channels per feature) because we assume channel-wise feature dimensions are exchangeable in their marginal contribution to the downstream label, so the expected cardinality is a valid proxy for the Shannon rate of the selected subset under a uniform coding assumption; this assumption fails when channels are strongly correlated (e.g., adjacent spatial channels in early U-Net blocks), in which case `rate` overcounts the true information rate and the dual will select a smaller budget than the information-theoretic optimum. The cross-entropy term `ce` measures *task-relevant information* (relevance) because we assume the downstream head h_phi is expressive enough that CE reduction is monotone in the mutual information I(selected features; label) up to a constant, an assumption that fails when the head underfits (small data regime), in which case `ce` reflects head capacity rather than feature relevance and λ will be driven too high. The converged multiplier `lambda*` measures the *marginal price of rate* at the task-specific operating point because the KKT condition of the Lagrangian equates it to the derivative of relevance with respect to rate; this assumption fails when the dual has not converged (oscillating λ), in which case `lambda*` reflects the last dual step rather than the equilibrium price, and we detect this by monitoring the dual residual |rate - B_target_soft| and rejecting runs with residual > 0.02.

## Contribution

['An information-bottleneck formulation of channel-budget selection for frozen diffusion features, in which the channel budget is the converged Lagrange multiplier of a rate–relevance Lagrangian rather than a prescribed hyperparameter, yielding a task-specific information criterion (B*, λ*) per downstream task.', 'A Gumbel-sigmoid straight-through gate with dual ascent on the rate multiplier that produces a differentiable cardinality signal, enabling the selection policy to be trained end-to-end on the downstream objective while the budget adapts.', 'An empirical characterization across segmentation, classification, and semantic correspondence of how the converged budget varies with task difficulty and feature layer, providing a diagnostic for when frozen diffusion features are information-sufficient for a task.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | ADE20K semantic segmentation | Dense labels expose per-channel relevance across blocks |
| Primary metric | Masked-channel mIoU at realized budget | Directly reads relevance-per-retained-channel tradeoff |
| Baseline | Policy-Mask (fixed budget sweep) | Exogenous budget; tests whether emergent budget matches tuning |
| Baseline | Top-k channel selector | Discrete cardinality; tests gradient-based pricing vs hard cutoff |
| Baseline | Frozen-feature linear probe | No selection; tests whether gating beats using all channels |
| Ablation-of-ours | IB-Budget with λ frozen at init | Isolates dual ascent from static rate penalty |

### Why this setup validates the claim

The central claim is that the budget should be an *emergent* dual variable pricing marginal rate against marginal relevance, not an exogenous hyperparameter. Policy-Mask at matched budgets tests the sub-claim that a learned price finds a better relevance-per-channel operating point than a swept constant: if the dual converges to a budget that Policy-Mask only reaches by validation search, and does so with higher mIoU at the same realized density, the pricing mechanism is doing real work. The top-k selector tests the sub-claim that differentiable rate pricing beats hard cardinality: top-k cannot express *which* channels are worth their cost, so on cases where relevance is unevenly distributed across blocks it should misallocate. The frozen linear probe tests whether gating adds anything over using all frozen channels, guarding against the trivial explanation that the backbone alone carries the signal. The λ-frozen ablation is the key falsifier: if emergent λ gives no benefit over a static penalty, dual ascent is decorative. Masked-channel mIoU at the realized budget is the right metric because it couples relevance and rate in a single number, so a method that merely shrinks the mask cannot win by default.

### Expected outcome and causal chain

**vs. Policy-Mask (fixed budget sweep)** — On a case where the optimal channel budget differs sharply between early U-Net blocks (high spatial redundancy) and late blocks (task-specific semantics), Policy-Mask produces a uniform-density mask tuned to a single validation budget because its cardinality is exogenous and cannot vary the price across blocks. Our method instead lets λ price each channel's marginal CE reduction against its marginal rate, so redundant early-block channels get gated off while late-block channels survive, and we expect a noticeable mIoU gap at *matched realized budgets* concentrated on images where block-wise relevance is most uneven, with parity on images where relevance is flat.

**vs. Top-k channel selector** — On a case where two channels have near-identical marginal relevance but very different marginal rate (one cheap, one expensive in the rate proxy), top-k produces a mask that keeps whichever channels rank highest by a relevance-only score because its cardinality is a discrete input with no gradient through the budget. Our method instead keeps the cheap channel because the Lagrangian prices rate explicitly, so we expect a small but consistent mIoU gain at the same realized budget, largest on subsets where the relevance–rate correlation is weakest.

**vs. Frozen-feature linear probe** — On a case where most frozen channels are task-irrelevant noise, the linear probe produces predictions diluted by noise dimensions because it has no selection mechanism and must fit weights over all channels. Our method instead gates those channels off via the rate penalty, so we expect a clear gain on noisy-feature subsets and near-parity on subsets where frozen features are already clean, which is the diagnostic signature that gating, not the backbone, is responsible.

### What would falsify this idea

If the converged λ* produces a budget indistinguishable from Policy-Mask's best validation-swept budget *and* the λ-frozen ablation matches IB-Budget's mIoU at every realized budget, then the dual-ascent pricing is decorative and the central claim that the budget is an information criterion collapses. Equally damning: if IB-Budget's gain over top-k is uniform across subsets rather than concentrated where relevance–rate correlation is weakest, the rate-pricing mechanism is not the cause of any improvement.

## References

1. Policy-Mask: Budget-Aware Adaptation of Diffusion Features to Discriminative Tasks
2. Repurposing Diffusion-Based Image Generators for Monocular Depth Estimation
