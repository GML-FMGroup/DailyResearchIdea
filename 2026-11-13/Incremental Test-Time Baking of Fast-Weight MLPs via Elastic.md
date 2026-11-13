# Incremental Test-Time Baking of Fast-Weight MLPs via Elastic Weight Consolidation

## Motivation

Existing test-time baking methods such as TTB (Test-time Baking for Efficient Rendering of Decoder-only View Synthesis Models) require retraining the baked MLP from scratch for each new scene or view set, because the bake is treated as a one-shot function of a fixed set of observed views. This structural limitation—no closure rule for incremental updates—means that as new views arrive, the baked MLP cannot be updated without full retraining, leading to temporal instability and prohibitive computational cost. The problem is meta-level: any method that computes a scene representation once from a fixed input set (including Long-LRM's long-sequence reconstruction) lacks a principled mechanism to incorporate new observations without forgetting or retraining.

## Key Insight

The baked MLP's parameters can be updated incrementally by treating the bake as a recursive Bayesian update where the Fisher information from previously observed views acts as a quadratic penalty, preserving the mapping's fidelity to old views while incorporating new ones—a property that follows from the local quadratic approximation of the MLP's loss landscape around its current parameters.

## Method

**What it is:** We introduce *Incremental TTB* (iTTB), a method that updates a test-time-baked MLP's parameters when new views arrive without full retraining. Inputs: current baked MLP parameters θ_t, new view set V_{t+1}, and a diagonal Fisher information matrix F_t accumulated from previous views. Output: updated parameters θ_{t+1} and updated F_{t+1}.

**Load-bearing assumption (stated explicitly):** iTTB assumes that the diagonal Fisher information accumulated from old views, treated as a fixed quadratic penalty, is a sufficient and transferable summary of old-view parameter importance for the baked MLP, such that penalizing movement along high-Fisher directions preserves old-view rendering fidelity without storing old views. This assumption rests on the local quadratic approximation of the old-view loss around θ_t and on the neglect of off-diagonal Hessian terms. It is expected to hold when old and new views are similar and to fail when they are highly dissimilar, in which case the quadratic term is trusted only within a validated displacement regime (see step 3b).

**How it works:**
```pseudocode
# Given: baked MLP f_θ (2-layer MLP, hidden=256, GeLU, token input dim=768, token output dim=768),
#        current params θ_t, Fisher diag F_t (float16, per-layer), new views V_{t+1}
# Hyperparameters: λ (penalty strength, default 1.0), η (learning rate, default 1e-4),
#                 K (inner steps, default 10), δ (trust-region cap on ||θ-θ_t||_F, default 0.01),
#                 τ (dissimilarity gate threshold, default 0.5)

1. Compute per-view loss L(θ) = Σ_{v in V_{t+1}} || f_θ(camera_tokens_v) - gt_tokens_v ||^2
2. Compute Fisher diagonal F_{t+1} = F_t + diag(∇_θ L(θ_t)^2)  # accumulate, store float16 per layer
3. Compute dissimilarity d = mean cosine distance between camera_tokens_v (v in V_{t+1}) and stored anchor tokens A_t (a fixed set of 64 old-view camera tokens kept from the initial bake; constant memory).
   3a. If d < τ: set λ_eff = λ (trust quadratic penalty).
   3b. If d ≥ τ: set λ_eff = λ * (τ / d) (down-weight penalty) and enforce trust-region cap: after each step, if ||θ - θ_t||_F > δ, rescale the update Δθ ← Δθ * (δ / ||Δθ||_F).
4. For k = 1..K:
     θ ← θ - η * ( ∇_θ L(θ) + λ_eff * F_t * (θ - θ_t) )
     if ||θ - θ_t||_F > δ: θ ← θ_t + (θ - θ_t) * (δ / ||θ - θ_t||_F)
5. Return θ_{t+1} = θ, F_{t+1}
```

**Why this design:** We chose elastic weight consolidation (EWC) over naive fine-tuning because the baked MLP's parameters are highly sensitive to the original view set, and without a penalty term, fine-tuning on new views catastrophically forgets the mapping learned from old views—a problem well-documented in continual learning (Kirkpatrick et al., 2017). The Fisher information matrix F_t serves as a local quadratic approximation of the loss landscape around θ_t, penalizing changes to parameters that were important for old views; this is structurally different from replay-based methods (which store old views) because it requires no additional memory beyond the diagonal Fisher and a fixed set of 64 anchor camera tokens. We chose a diagonal Fisher approximation over full Fisher because the MLP has millions of parameters and storing a full matrix is infeasible, accepting the cost that off-diagonal parameter correlations are ignored—a trade-off justified by the empirical success of diagonal EWC in continual learning. We chose a fixed number of inner steps K over convergence-based stopping because test-time baking must be fast, and K=10 provides a good balance between update quality and latency. The penalty strength λ is set via a held-out set of old views to prevent over-regularization. The dissimilarity gate and trust-region cap are included because the quadratic approximation is only valid locally; when new views are far from old ones (d ≥ τ), the penalty is down-weighted and parameter displacement is capped to keep the update within the regime where the Fisher summary is trusted. Unlike TTB, which retrains from scratch per scene, iTTB's update is O(K * |V_{t+1}|) and preserves old-view fidelity by construction within the validated displacement regime.

**Why it measures what we claim:** The Fisher diagonal F_t measures parameter importance for old views because the assumption that the loss landscape is locally quadratic around θ_t makes the Fisher information equivalent to the Hessian's diagonal, and this assumption fails when the MLP's loss is highly non-convex or when old and new views are highly dissimilar, in which case F_t reflects only local curvature and may under-penalize changes that affect old views globally. The penalty term λ_eff * F_t * (θ - θ_t) measures the deviation from old-view-optimal parameters weighted by importance, because the EWC objective is a second-order Taylor expansion of the old-view loss, and this assumption fails when the new-view loss has a large gradient that pushes parameters far from θ_t, in which case the quadratic approximation breaks and the update may still forget; the trust-region cap δ bounds this failure by limiting displacement. The number of inner steps K measures the trade-off between update fidelity and speed, because K steps of gradient descent on the EWC objective approximates the proximal update, and this assumption fails when the loss is ill-conditioned, in which case K steps may not reach the minimum of the penalized objective. The dissimilarity d measures the validity of the quadratic penalty because it quantifies how far new views are from the anchor set, and this assumption fails when d is large, in which case the gate reduces λ_eff and the trust-region cap enforces local validity.

## Contribution

['A recursive update rule for test-time-baked MLPs that uses a diagonal Fisher information matrix to penalize changes to parameters important for previously observed views, enabling incremental scene updates without full retraining.', 'An empirical demonstration that incremental baking maintains rendering quality on old views while incorporating new views, with a computational cost that scales with the number of new views rather than the total number of views.', 'A practical algorithm for temporal stability in decoder-only view synthesis models, showing that the baked MLP can be updated online as new views arrive, with a held-out validation procedure for setting the penalty strength λ.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Sequential novel-view splits of LLFF and NeRF-360 (8 scenes each, 5 sequential splits per scene, 20 views per split) | Tests forgetting across arriving views in realistic order |
| Primary metric | Old-view PSNR after each update (mean over 20 old views) | Directly measures retention of previously baked views |
| Secondary metric | New-view PSNR after each update | Measures incorporation of new views |
| Secondary metric | Update wall-clock time per step (ms) and peak GPU memory (MB) | Verifies constant-memory and speed claims |
| Baseline | Naive fine-tuning of baked MLP (λ=0, no gate, no cap) | Tests whether EWC penalty is necessary |
| Baseline | Replay-based fine-tuning (store all old views, replay buffer size = all old views) | Tests whether Fisher suffices without extra memory |
| Baseline | TTB retrain-from-scratch per scene | Tests whether incremental update matches full rebake |
| Ablation-of-ours | iTTB without Fisher penalty (λ=0, gate and cap active) | Isolates the effect of the EWC term |
| Ablation-of-ours | iTTB with full Fisher (dense Hessian diagonal computed via full Fisher on a subset of 1k parameters) vs diagonal Fisher on dissimilar-view scenes (d ≥ τ) | Tests the diagonal approximation assumption A |
| Ablation-of-ours | iTTB without dissimilarity gate and trust-region cap (λ_eff=λ, δ=∞) | Tests whether the gate/cap are necessary for the quadratic regime |

### Why this setup validates the claim

The central claim is that iTTB updates a baked MLP on new views without catastrophic forgetting of old views, using only a diagonal Fisher and no stored views (beyond 64 anchor tokens). The sequential novel-view split creates exactly the condition where forgetting can occur: each step's new views are disjoint from old ones, so any method that overwrites old-view-optimal parameters will show a drop in old-view PSNR. Naive fine-tuning tests the necessity of the EWC penalty; if it forgets while iTTB does not, the penalty is doing causal work. Replay-based fine-tuning tests whether the Fisher approximation is sufficient without extra memory: if iTTB matches replay, the claim that no stored views are needed is validated. TTB retrain-from-scratch is the upper bound on quality; if iTTB approaches it with far less compute, the incremental claim is supported. The λ=0 ablation is the sharpest test: it removes only the Fisher term, so any gap between it and full iTTB isolates the penalty's contribution. The full-vs-diagonal Fisher ablation directly tests the load-bearing assumption A: if the old-view PSNR gap between full and diagonal Fisher is small on dissimilar-view scenes, the diagonal approximation is sufficient; if large, the assumption fails and the gate/cap are necessary. The no-gate/no-cap ablation tests whether the trust-region mechanism is needed to keep the update within the validated quadratic regime. Old-view PSNR is the right metric because forgetting is a drop in fidelity on previously seen views, and PSNR is sensitive to exactly that degradation. The runtime and memory metrics verify the constant-memory and speed claims empirically.

### Expected outcome and causal chain

**vs. Naive fine-tuning of baked MLP** — On a case where the new view set contains a camera angle far from the old views (d ≥ τ), naive fine-tuning produces a large drop in old-view PSNR because its gradient step moves parameters freely toward the new-view loss, overwriting the mapping learned for old views. Our method instead adds the Fisher-weighted penalty (down-weighted by the gate) and the trust-region cap that resists changes to parameters important for old views, so we expect old-view PSNR to stay close to the pre-update level while new-view PSNR improves, yielding a noticeable gap on old views but parity on new views.

**vs. Replay-based fine-tuning (store old views)** — On a case where old views are numerous, replay-based fine-tuning produces high old-view PSNR but requires storing all old views, and its update cost grows with the replay buffer. Our method instead uses only the diagonal Fisher as a summary of old-view importance plus 64 anchor tokens, so we expect comparable old-view PSNR to replay (within a small margin) while using constant memory, with the gap widening only when old and new views are highly dissimilar and off-diagonal correlations matter (d ≥ τ).

**vs. TTB retrain-from-scratch per scene** — On a case where the scene has many views arriving in sequence, TTB retrain-from-scratch produces the highest old-view PSNR but at a cost that grows with total views, making it slow per update. Our method instead performs K=10 gradient steps on the EWC objective, so we expect old-view PSNR close to TTB's (small gap) with a large reduction in update time, and the gap should shrink as K increases, confirming the trade-off.

**vs. iTTB without Fisher penalty (λ=0, gate and cap active)** — On a case where new views conflict with old views, the λ=0 ablation produces old-view PSNR degradation similar to naive fine-tuning because it lacks any retention term. Our method instead includes the Fisher penalty, so we expect a clear separation in old-view PSNR between λ=0 and full iTTB, with the gap concentrated on scenes where new views are dissimilar to old ones, confirming the penalty is the causal mechanism.

**vs. iTTB with full Fisher (dense) on dissimilar-view scenes** — On scenes where d ≥ τ, the full Fisher captures off-diagonal curvature that the diagonal ignores. If the old-view PSNR gap between full and diagonal Fisher is small (< 0.5 dB), the diagonal approximation is sufficient; if large (> 1.0 dB), the load-bearing assumption A fails and the gate/cap are necessary to keep the update within the validated regime.

**vs. iTTB without dissimilarity gate and trust-region cap** — On scenes where d ≥ τ, removing the gate and cap allows the quadratic penalty to be trusted outside its valid regime, leading to either over-regularization (new-view PSNR drops) or under-regularization (old-view PSNR drops). We expect the gated/capped version to outperform on both old and new PSNR on dissimilar-view scenes, confirming the gate/cap are necessary for the quadratic approximation to hold.

### What would falsify this idea

If the old-view PSNR gap between iTTB and naive fine-tuning is uniform across all scenes rather than concentrated where new views are dissimilar to old views, then the Fisher penalty is not acting as an importance-weighted retention mechanism and the central claim is wrong. Similarly, if iTTB with λ=0 matches full iTTB on old-view PSNR, the EWC term is not doing the claimed work. If the full-vs-diagonal Fisher gap is large (> 1.0 dB) on dissimilar-view scenes, the diagonal approximation assumption A fails and the method's retention claim is not supported without the gate/cap.

## References

1. TTB: Test-time Baking for Efficient Rendering of Decoder-only View Synthesis Models
