# Block-Size-Invariant Memory Calibration for Recurrent-Memory Block Diffusion

## Motivation

Block-diffusion models such as Clover fuse a Gated Delta Network recurrent-memory readout with current-block tokens inside a single shared softmax, and the only knob controlling that competition is one learnable block-size calibration scalar. A scalar with no normalization or identifiability constraint cannot guarantee that the historical-memory contribution stays calibrated as block size and denoising step vary, so the memory-vs-current competition is empirically tuned rather than a validated conditioning property; the same structural gap recurs wherever a persistent-state readout is mixed with local bidirectional attention (e.g., OLMoE-style sparse backbones give no calibration account either). The root cause is that the readout logits and current-block logits live on incomparable scales that shift with block size and noise level, and a single free scalar absorbs that shift only at the block sizes seen in training.

## Key Insight

Calibrated historical-memory contribution is not a free scalar to be learned but an invariant of the attention distribution: if the memory readout is normalized by its own step- and block-size-conditioned partition function, the memory mass assigned to each current-block token becomes block-size-invariant and monotone in denoising step, a property that can be tested directly rather than assumed.

## Method

**What it is.** We introduce BIC-Mem (Block-Invariant Calibrated Memory), a drop-in replacement for Clover's single learnable block-size calibration scalar. Its inputs are the Gated Delta Network memory state $M$, the current-block token states $X_{1:B}$, the block size $B$, and the denoising step index $t$; its output is the memory-vs-current attention logit matrix and the resulting readout used by the block-diffusion attention.

**Load-bearing assumption (stated explicitly).** The invariance claim is conditional on the memory state $M$ being block-size-independent given the prior context, so that the memory logits $s_mem$ are fixed across block sizes and any observed drift in memory mass $\rho(B,t)$ is attributable solely to the softmax competition rather than to the recurrence itself. This assumption is not assumed away: we measure it with a control probe (step 11) that recomputes $s_mem$ from a block-size-matched reference state $M_ref$ obtained by replaying the same prefix at a fixed canonical block size $B_ref=1$, and we report the drift of $\rho$ computed from $M_ref$ alongside the deployed $\rho$. The calibration effect is the difference between the two; the recurrence-induced state drift is the residual.

**How it works.**

```pseudocode
# per attention head, block size B, denoising step t in [0, T)
# learned parameters: tau_0 (scalar, init 1.0), alpha (scalar, init 0.0),
#   {gamma_b} for b in candidate block sizes {1,2,4,8,16}, tied via
#   gamma_b = exp(u_b) with u_b = W2 @ GeLU(W1 @ onehot(b)), W1: [5,4], W2: [4,1]

1. memory logits:      s_mem = (M @ W_m) / sqrt(d)              # [B] per query
2. current logits:     s_cur = (X_{1:B} @ W_q) @ (X_{1:B} @ W_k).T / sqrt(d)   # [B, B]
3. step temperature:   tau(t) = tau_0 * (1 + alpha * (1 - t/T))   # monotone in t
4. block normalizer:   z(B) = gamma_B                             # gamma_B tied across B in {1,2,4,8,16}
5. memory partition:   logZ_mem = logsumexp_b( s_mem_b / (tau(t) * z(B)) )
6. current partition:  logZ_cur = logsumexp_j( s_cur_j / tau(t) )
7. calibrated logits:  a_mem = s_mem / (tau(t)*z(B)) - logZ_mem
                       a_cur = s_cur / tau(t)          - logZ_cur
8. shared softmax:     p = softmax( concat(a_mem, a_cur) )
9. readout:            y = p_mem * V_mem + p_cur @ V_cur
10. invariance probe:  record rho(B,t) = sum(p_mem)   # memory mass, logged each forward
11. control probe:     replay prefix at canonical block size B_ref=1 to obtain M_ref;
                       recompute s_mem_ref = (M_ref @ W_m) / sqrt(d);
                       record rho_ref(B,t) = sum(softmax(concat(
                         s_mem_ref/(tau(t)*z(B)) - logZ_mem_ref,
                         s_cur/tau(t) - logZ_cur))_mem)
                       and state drift delta_M(B) = ||M(B) - M_ref||_2 / ||M_ref||_2
```

Step 3 and step 4 together replace Clover's free scalar; step 5–7 give the memory readout its own partition function so that the memory branch is normalized independently of the current block. $\gamma_B$ is tied across the candidate block sizes $\{1,2,4,8,16\}$ via a shared low-rank parameterization (a 2-layer MLP with hidden width 4 and GeLU activation over the one-hot block-size index, initialized so that $\gamma_b=1$ for all $b$), so the model cannot fit a separate scalar per block size without penalty. The invariance probe $\rho(B,t)$ is logged at every forward pass and forms the test statistic; the control probe $\rho_ref(B,t)$ and state drift $\delta_M(B)$ are logged at every forward pass on the same batch and form the confound-separation statistic.

**Why this design.** We chose a separately normalized memory partition (steps 5–7) over Clover's single learnable scalar because a shared softmax lets the memory branch's total mass drift with block size, and a scalar can only absorb that drift at the block sizes seen during training, accepting the cost that the memory branch now has its own temperature to tune. We chose a step-conditioned temperature $\tau(t)$ over a constant temperature because the current-block logits are noisier at high $t$ (early denoising) and the memory branch should dominate then, accepting the cost of one extra scalar $\alpha$ and a monotonicity assumption on $t$. We chose block-size-tied $\gamma_B$ over independent per-block constants because independent constants would reintroduce the identifiability problem Clover's scalar already has, accepting the cost that the tie may underfit genuinely different block regimes. We chose a logged invariance probe $\rho(B,t)$ over a post-hoc evaluation because the property must be checkable during training to be a conditioning guarantee rather than a benchmark. We added the control probe $\rho_ref(B,t)$ and state drift $\delta_M(B)$ because the invariance claim is conditional on $M$ being block-size-independent, and the control probe measures that condition rather than assuming it. A domain expert would not describe BIC-Mem as a variant of the Clover calibration scalar: Clover's scalar is a free multiplicative factor on a shared softmax, whereas BIC-Mem replaces the shared softmax with two separately normalized partitions whose ratio is constrained to be invariant, which is a different object (a normalized distribution ratio, not a scalar).

**Why it measures what we claim.** The memory mass $\rho(B,t)=sum(p_mem)$ measures calibrated historical-memory contribution because, under the assumption that the memory logits $s_mem$ are block-size-independent given $M$ (true since $M$ summarizes only prior context), the ratio $\rho(B,t)/\rho(B',t)$ equals the ratio of partition functions $Z_mem(B)/Z_mem(B')$, which the tied $\gamma_B$ and step temperature drive to one; this assumption fails when the Gated Delta Network state $M$ itself depends on $B$ through the recurrence, in which case $\rho$ reflects the recurrence's block-size sensitivity rather than the calibration mechanism. The control probe $\rho_ref(B,t)$ measures the calibration mechanism alone because it recomputes $s_mem$ from a block-size-matched reference state $M_ref$ (replayed at $B_ref=1$), so any residual drift in $\rho_ref$ is attributable to the softmax partition ratio; this fails when the replay itself is infeasible (e.g., prefix longer than the replay budget), in which case $\rho_ref$ is undefined for that prompt and we report the fraction of prompts with defined $\rho_ref$. The state drift $\delta_M(B)$ measures the recurrence's own block-size sensitivity because it compares the deployed state to the reference state at the same prefix; this fails when the recurrence is exactly block-size-invariant, in which case $\delta_M=0$ and $\rho$ and $\rho_ref$ coincide. The step monotonicity of $\tau(t)$ measures denoising-step conditioning because, under the assumption that current-block logit variance decreases monotonically with $t$, $\tau(t)$ tracks that variance; this fails when the noise schedule is non-monotone, in which case $\tau(t)$ reflects the schedule rather than the logit scale. The tied $\gamma_B$ measures cross-block-size consistency because, under the assumption that the candidate block sizes share a logit scale, a single low-rank $\gamma$ suffices; this fails when block sizes genuinely differ in scale, in which case the tie reflects scale mismatch rather than consistency.

## Contribution

(1) BIC-Mem, a separately normalized recurrent-memory readout with a step- and block-size-conditioned temperature and a block-size-tied normalizer, replacing Clover's single learnable calibration scalar. (2) An invariance test — the logged memory-mass ratio $\rho(B,t)$ across block sizes and denoising steps — that turns calibrated historical-memory contribution from an empirically tuned scalar into a checkable conditioning property. (3) A design principle: tying the block-size normalizer across candidate block sizes is what makes the calibration identifiable, rather than fitting a free constant per block size.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Clover internal agent workload + SWE-bench Verified | Matches source's own evaluation regime |
| Dataset (surrogate) | Open block-diffusion pretraining corpus (e.g., OpenWebText + code), 100M–1B parameter block-diffusion MoE with the same memory-vs-current shared-softmax structure | Tractable surrogate for block-size sweep |
| Primary metric | Memory-mass drift \|ρ(B,t) − ρ(B',t)\| across block sizes, computed both from deployed ρ and from control-probe ρ_ref | Separates calibration from recurrence drift |
| Secondary metric | State drift δ_M(B) = \|\|M(B) − M_ref\|\|₂ / \|\|M_ref\|\|₂ at fixed t | Measures the load-bearing assumption directly |
| Secondary metric | Mediation: per-prompt SWE-bench delta regressed on per-prompt ρ-drift (OLS with prompt fixed effects) | Tests whether ρ-drift causally tracks accuracy |
| Baseline | Clover single learnable scalar | Tests whether separate partitions beat a free scalar |
| Baseline | Constant-temperature shared softmax | Isolates value of step-conditioned τ(t) |
| Baseline | Independent per-block γ_B (untied) | Tests whether tying γ prevents identifiability drift |
| Ablation-of-ours | BIC-Mem without invariance probe logging | Tests whether in-training conditioning matters |
| Ablation-of-ours | BIC-Mem with recurrence state detached across the block-size sweep | Isolates softmax partition ratio from recurrence |

### Why this setup validates the claim

The central claim is that a separately normalized memory partition with a tied block normalizer and step-conditioned temperature keeps the memory branch's contribution invariant across block sizes, where Clover's single free scalar cannot. The primary metric — drift in memory mass ρ across block sizes at fixed t — is the direct observable of that invariance, so a null result (drift comparable to baselines) falsifies the mechanism, not merely the performance. Because the invariance claim is conditional on M being block-size-independent, we report ρ_ref and δ_M alongside ρ: if δ_M is large, the deployed ρ conflates calibration with recurrence drift, and the calibration effect is read from ρ_ref; if δ_M is near zero, ρ and ρ_ref coincide and the control probe is a no-op. Each baseline isolates one design choice: the Clover scalar tests the partition split, the constant-temperature variant tests the step conditioning, and the untied γ tests the tying. The ablation without probe logging tests whether the property is actually enforced during training or only appears post hoc. The detached-recurrence ablation freezes M across the block-size sweep so the test statistic measures only the softmax partition ratio it claims to measure. The surrogate block-diffusion MoE (100M–1B parameters) is trained on open data with the same memory-vs-current shared-softmax structure, and its block-size sweep is reported as primary evidence that the invariance probe and baseline ordering reproduce at tractable scale before arguing transfer to Clover-scale. The mediation analysis regresses per-prompt SWE-bench delta on per-prompt ρ-drift to test whether lower drift causally improves token-level denoising accuracy rather than merely correlating with it. Running on Clover's own agent workload and SWE-bench Verified keeps the target task fixed, so any gain is attributable to calibration rather than a domain shift.

### Expected outcome and causal chain

**vs. Clover single learnable scalar** — On a case where the same prompt is denoised at B=1 versus B=16, the baseline's shared softmax lets the memory branch's total mass shift with the number of current-block competitors, because its one scalar absorbs drift only at trained block sizes. Our method instead normalizes the memory branch with its own partition and tied γ_B, so the ratio ρ(1,t)/ρ(16,t) is pinned near one. We expect a clear drift gap on the B=1 vs B=16 subset but near-parity on the trained block size, with downstream SWE-bench gains concentrated on prompts whose block size differs from training. The control probe ρ_ref is expected to show the same ordering with a smaller magnitude if δ_M is non-negligible, and the mediation regression is expected to yield a negative coefficient on ρ-drift (lower drift → higher accuracy) with a 95% bootstrap CI excluding zero.

**vs. Constant-temperature shared softmax** — On a case with high denoising noise at early t, the baseline's fixed temperature lets noisy current-block logits steal mass from memory, producing premature commitment to corrupted current tokens. Our method's τ(t) rises at high t, suppressing current logits so memory dominates early. We expect fewer early-step commitment errors and a visible ρ(t) curve that starts memory-heavy and decays, with the accuracy gain concentrated on the earliest denoising steps and vanishing late.

**vs. Independent per-block γ_B (untied)** — On a case spanning block sizes 2 and 8, the untied baseline fits a separate scalar per size and can match ρ at each size while leaving the underlying ratio unconstrained, so it drifts under distribution shift to an unseen block size. Our tied low-rank γ shares scale across {1,2,4,8,16}, forcing a single consistent ratio. We expect the untied variant to match on seen block sizes but show a sharp ρ drift and accuracy drop at held-out block sizes, where ours stays flat.

**vs. BIC-Mem with recurrence state detached across the block-size sweep** — On the same B=1 vs B=16 subset, the detached variant removes the recurrence's block-size sensitivity from the test statistic, so its ρ_ref should match ours while its deployed ρ may differ if δ_M is large. We expect the detached variant to confirm that the calibration effect is carried by the softmax partition ratio, and that any residual gap between deployed ρ and ρ_ref is explained by δ_M.

### What would falsify this idea

If memory-mass drift is uniform across all block sizes and denoising steps — i.e., no concentration of the gain at the B=1 vs B=16 subset or early-t steps — then the calibration mechanism is not what produces any improvement, and the central claim is wrong. Similarly, if the untied-γ variant matches ours even at held-out block sizes, the tying argument fails; and if the mediation regression shows no association between ρ-drift and per-prompt SWE-bench delta, the assumption that lower drift causally improves denoising accuracy is falsified.

## References

1. Clover-35B-A3B-Preview Technical Report
