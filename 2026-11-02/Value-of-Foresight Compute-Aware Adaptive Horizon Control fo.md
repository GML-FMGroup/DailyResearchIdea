# Value-of-Foresight: Compute-Aware Adaptive Horizon Control for Hierarchical World-Model VLAs

## Motivation

Multi-horizon world-model VLAs such as HiFOR, Mantis, and DUST all assume that predicting more future horizons and retrieving more future latents is monotonically beneficial, and none establishes a resource-scaling property: HiFOR in particular adds hierarchical foresight parameters and retrieval complexity without ever testing whether an additional horizon earns back its inference cost. The root cause is structural — foresight depth is a fixed architectural hyperparameter baked into the prediction head and retrieval factor, so the model has no surface on which to express that, for a given state, the marginal future horizon carries almost no action-relevant information (e.g., a near-static grasp phase), while for another state it carries decisive information (e.g., an imminent contact transition). Without a per-state, per-horizon estimate of decision value, the field can only scale foresight uniformly, which is exactly the compute-inefficiency that RynnVLA-002, DUST, and Mantis leave unaddressed.

## Key Insight

The marginal action-decision value of an additional foresight horizon is identifiable from a single forward pass as the disagreement between the action distribution conditioned on foresight depth k and the one conditioned on depth k−1, so a horizon can be truncated precisely when adding it would not have changed the chosen action — turning foresight depth into a per-state, per-horizon computable quantity rather than a fixed hyperparameter.

## Method

**What it is.** VOF (Value-of-Foresight) augments HiFOR's hierarchical foresight head with a *truncation head* that, for each horizon level k, estimates the marginal action-decision value Δ_k = the expected change in the action distribution if the entire suffix of horizons {k, ..., K_max} were removed (the actual intervention performed at inference), and stops hierarchical prediction and retrieval at the first k where Δ_k falls below a learned threshold τ. Inputs: current observation embedding, action query, and the prefix of already-computed foresight latents z_1..z_{k−1}. Outputs: a truncated foresight depth K*, the retained latents, and the final action.

**How it works.**

```pseudocode
# Offline: build the truncation head's training signal from the frozen HiFOR policy
for each training step (o, a_gt):
    # Teacher: run HiFOR at full depth K_max, record action logits per depth
    pi_full[1..K_max] = HiFOR.forward(o, depth=K_max)   # action dist at each prefix depth
    for k in 1..K_max:
        # Counterfactual target: action dist with the entire suffix {k,...,K_max} REMOVED
        pi_wo_suffix_k = HiFOR.forward(o, horizons=1..k-1)   # one extra forward per k
        delta_target[k] = D_KL( pi_full[k] || pi_wo_suffix_k )  # marginal decision value of suffix
    train truncation_head(o, k) -> delta_hat[k]  with MSE to delta_target[k]
    # threshold tau: smallest tau s.t. mean success drop < 0.5% on held-out probe set

# Inference: adaptive truncation
z = []; K_star = 1
for k in 1..K_max:
    z_k = HDM.predict_horizon(o, z_{1:k-1})       # hierarchical foresight step
    delta_hat[k] = truncation_head(o, k, z_{1:k})
    if delta_hat[k] < tau and k > 1: break          # stop prediction AND retrieval
    z.append(z_k); K_star = k
retrieved = ActionGuidedRetrieval(o, z_{1:K_star})  # retrieval factorized over retained horizons
pi = policy_head(o, retrieved); return pi, K_star
```

Hyperparameters: K_max = 3 (HiFOR's short-term / milestone / goal levels); truncation head = 2-layer MLP over [o; k; mean-pooled z_{1:k}], 256 hidden, GeLU activation; τ calibrated on a 500-episode held-out probe set as the largest value keeping success-rate drop under 0.5%; counterfactual targets are recomputed every 5k steps against the current HiFOR checkpoint.

**Why this design.** We chose a *counterfactual* marginal-value target (action-distribution change when the entire suffix of horizons from k onward is removed) over a next-horizon prediction-error target because prediction error measures how *surprising* the future is, not how *action-relevant* it is — a volatile but action-irrelevant background would inflate error and defeat truncation, accepting the cost that each training step needs K_max extra teacher forwards. We chose to truncate *jointly* over prediction and retrieval rather than truncating retrieval only, because HiFOR's action-guided retrieval factorizes over horizons, so a retained-but-unpredicted horizon would be retrieved from a stale latent — accepting the cost that a truncation error also removes retrieval evidence. We chose a fixed learned threshold τ calibrated on a probe set rather than a per-state adaptive threshold, because a per-state threshold reintroduces the calibration problem that log-prob gating suffers from (Kadavath et al.); accepting the cost that τ is a global scalar and may be suboptimal for rare states. A domain expert would not describe VOF as a variant of standard early-exit networks (BranchyNet-style), because early-exit gates on *confidence in the current prediction*, whereas VOF gates on a *counterfactual action-distribution change* that requires intervening on the horizon set — a signal no confidence head can produce.

**Why this measures what we claim.** The quantity `delta_hat[k]` measures *marginal action-decision value of the suffix {k,...,K_max}* because we assume that the KL divergence between the full-depth action distribution and the action distribution with the entire suffix removed is a sufficient statistic for whether retaining horizons k..K_max would change the committed action; this assumption fails when two horizons are redundant (removing either alone leaves the action unchanged, but removing both changes it), in which case delta_hat reflects single-horizon redundancy rather than joint necessity — we therefore use the suffix-removal target, which matches the joint truncation performed at inference and avoids the single-horizon leave-one-out failure. The quantity `tau` measures the *acceptable decision-value floor* because we assume the probe-set success rate is monotone in the retained decision value; this assumption fails when a rare high-value state is under-represented in the probe set, in which case τ reflects the probe distribution's decision-value scale rather than the deployment distribution's. The quantity `K_star` measures *compute allocated to foresight* because we assume each retained horizon costs one HDM forward plus one retrieval factor; this assumption fails when retrieval dominates cost, in which case K_star under-counts the compute actually saved. The truncation head's input `mean-pooled z_{1:k}` measures the *state's foresight-demand* because we assume the prefix latents summarize all task-relevant context for horizon k; this assumption fails when the demand depends on an observation feature not encoded in the latents (e.g., a task instruction introduced after z is computed), in which case the head reflects latent-prefix demand rather than full-state demand.

## Contribution

(1) VOF, a compute-aware adaptive horizon controller for hierarchical world-model VLAs that estimates per-horizon marginal action-decision value via counterfactual leave-one-horizon-out KL targets and truncates both hierarchical prediction and action-guided retrieval at a calibrated threshold. (2) The empirical finding that marginal decision value is highly non-uniform across horizons and states, so uniform foresight depth wastes the majority of its compute on horizons whose removal would not change the action. (3) A resource-scaling characterization of hierarchical foresight relative to single-horizon world models: success rate as a function of retained foresight depth, establishing where the multi-horizon advantage actually lives.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | RoboCasa manipulation suite | Multi-stage tasks with variable foresight demand |
| Primary metric | Success rate vs. compute (HDM forward count) | Jointly captures accuracy and truncation savings |
| Baseline | HiFOR (full depth, no truncation) | Upper-bound accuracy, no compute savings |
| Baseline | Confidence early-exit gate (BranchyNet-style) | Tests whether counterfactual signal beats confidence |
| Baseline | Random horizon dropout at matched compute | Controls for any compute-accuracy tradeoff |
| Ablation-of-ours | VOF with prediction-error truncation target | Isolates value of counterfactual target |
| Diagnostic | Joint leave-two-out KL for all horizon pairs | Measures redundancy failure mode |
| Diagnostic | Decorrelated confidence vs. Δ states | Tests mechanism not reducible to confidence |
| Ablation | VOF with K_max ∈ {2,3,5,8} | Tests generalization beyond fixed three levels |

### Why this setup validates the claim

The central claim is that a counterfactual marginal-decision-value signal truncates foresight where it is action-irrelevant, preserving success while cutting compute. HiFOR full-depth establishes the accuracy ceiling and the compute cost we must reduce; if VOF cannot approach its success rate, the truncation is destroying useful evidence. The confidence early-exit baseline is the sharpest test of the core mechanism: it gates on prediction confidence, so if VOF's gain over it is concentrated on volatile-but-action-irrelevant states, that confirms the counterfactual target — not confidence — is what carries the signal. Random dropout at matched compute controls for the trivial possibility that any compute reduction costs nothing; if random dropout matches VOF, the head is doing no work. The prediction-error ablation isolates the target choice, and the success-vs-compute curve is the right metric because it detects both accuracy loss and savings, letting us see whether VOF dominates the frontier rather than trading one for the other. The joint leave-two-out diagnostic directly measures the redundancy failure mode: we compute the KL divergence between the full-depth action distribution and the action distribution with each pair of horizons removed, and report the fraction of states where single-horizon Δ is near-zero but joint removal changes the action; VOF's success rate on that subset bounds the damage from redundancy. The decorrelated confidence vs. Δ diagnostic constructs states where confidence and counterfactual Δ are explicitly decorrelated (e.g., confidently-wrong shallow states with high Δ_k) and quantifies the divergence rate between VOF's gate decisions and the confidence gate on exactly those states, demonstrating the mechanism is not reducible to confidence gating. The K_max ablation varies K_max ∈ {2,3,5,8} to show the value-of-foresight signal generalizes beyond HiFOR's fixed three levels, reporting the success-vs-compute frontier across K_max.

### Expected outcome and causal chain

**vs. HiFOR (full depth)** — On a case where a milestone-level foresight latent is action-irrelevant (e.g., a static background region dominating the milestone prediction), HiFOR computes and retrieves all three horizons, spending full HDM and retrieval compute for an action distribution identical to a shallower run. Our method instead estimates a near-zero Δ at that horizon and stops, so we expect near-parity success rate with a clear reduction in mean K_star and HDM forwards, with the gap widening on tasks whose later horizons are inert.

**vs. Confidence early-exit gate** — On a case where the action distribution is confidently wrong at a shallow depth but a later horizon would correct it (volatile but action-relevant future), the confidence gate exits early because its current prediction is sharp, committing the wrong action. Our method instead sees a large Δ_k from the counterfactual target and retains the horizon, so we expect a noticeable success gap favoring VOF specifically on the volatile-action-relevant subset, with parity on the easy confident subset. The decorrelated diagnostic will show that VOF's gate decisions diverge from the confidence gate on exactly these states, with a divergence rate significantly above zero.

**vs. Random horizon dropout at matched compute** — On a case where the dropped horizon happens to be the decisive one, random dropout removes it blindly and fails, while on cases where it is inert it succeeds by luck. Our method instead drops only low-Δ horizons, so we expect VOF to sit above the random-dropout success-vs-compute frontier, with the separation largest at aggressive compute budgets where random dropout frequently discards decisive horizons.

**vs. Joint leave-two-out diagnostic** — We expect the fraction of states where single-horizon Δ is near-zero but joint removal changes the action to be non-negligible (e.g., >5%), confirming the redundancy failure mode. VOF's success rate on that subset will be lower than on the non-redundant subset, but the suffix-removal target should mitigate the damage compared to a single-horizon leave-one-out target.

**vs. K_max ablation** — We expect the success-vs-compute frontier to shift outward as K_max increases, with VOF consistently dominating random dropout and confidence gating across all K_max values, demonstrating that the value-of-foresight signal generalizes beyond the fixed three-level hierarchy.

### What would falsify this idea

If VOF's success gain over the confidence gate is uniform across all subsets rather than concentrated on volatile-but-action-relevant states, the counterfactual target is not the source of the benefit. If VOF's success-vs-compute curve is indistinguishable from random dropout at matched compute, the truncation head carries no decision-relevant signal.

## References

1. HiFOR: Hierarchical Foresight with Action-Guided World Grounding for Vision-Language-Action Model
2. Mantis: A Versatile Vision-Language-Action Model with Disentangled Visual Foresight
3. Dual-Stream Diffusion for World-Model Augmented Vision-Language-Action Model
