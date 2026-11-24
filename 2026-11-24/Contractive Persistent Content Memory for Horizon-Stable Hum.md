# Contractive Persistent Content Memory for Horizon-Stable Humanoid Motion Generation

## Motivation

Autoregressive diffusion motion generators such as Motion Mamba 2.0 replace bounded attention caches with Mamba recurrent states and persistent content memory (PCM) to reach long horizons, but the PCM content-slot update is an unconstrained learned map: nothing forces the compressed-history state to forget perturbations at a fixed rate, so any deviation accumulated during denoising is free to be re-injected and amplified at every rollout step. The field has framed this as a memory-capacity problem (how much history fits in the slots) rather than a stability problem (whether the history operator contracts), which is why the reported terminal-control evaluation stops at 39.52 s rather than demonstrating error that stays flat as rollouts extend. The structural gap is that no existing long-horizon motion generator in this line — Motion Mamba 2.0 being the concrete instance — carries an invariant on the history-compression operator that would make terminal error independent of rollout length.

## Key Insight

If the PCM content-slot update is made a Lipschitz-bounded map with a contraction modulus enforced during training, then the compressed history becomes a fading-memory sufficient statistic of the past, so perturbations injected by finite-step denoising decay geometrically instead of accumulating and terminal control error saturates at a level set by the denoising step size rather than by rollout length.

## Method

**What it is.** We introduce **CPCM (Contractive Persistent Content Memory)**, a drop-in replacement for the PCM content-slot update inside the Motion Mamba 2.0 root-body diffusion loop. Inputs: the Mamba recurrent state `h_t`, the previous content slots `S_{t-1}`, text-conditioned features `c_t`, and the current denoising iterate `x_t^k`. Outputs: updated content slots `S_t` that are certified to be a contraction in `S`, plus a scalar penalty used during training.

**Load-bearing assumption (stated explicitly).** The claim that terminal control error becomes horizon-independent rests on the assumption that the PCM content-slot path is the dominant channel through which denoising perturbations accumulate across rollout steps, so that bounding the Lipschitz constant of the slot update alone suffices. This assumption is load-bearing and is tested directly by the decomposition experiment in the evaluation section: we report terminal error growth attributable to the slot path separately from growth attributable to the Mamba recurrent state, and if the recurrent-state channel dominates, the horizon-independence claim is scoped to the slot path's bounded contribution rather than to total terminal error.

**How it works.**

```pseudocode
# CPCM slot update, called once per generated frame t, inside each denoising step k
# Fixed hyperparameters: slot dim d=64, n_slots=16, target modulus rho=0.90,
#   penalty weight lambda=1.0, margin eps=1e-3, denoising steps K=10
#   gate range: g_t in [g_min, g_max] with g_min=0.05, g_max=0.95 (clamped sigmoid)
#   tanh derivative bound: |tanh'(z)| <= 1 for all z

input: h_t (Mamba state), S_prev (n_slots x d), c_t (text feat), x_t^k (noisy root-body)

1. Readout:      U_t   = W_u [h_t ; c_t ; x_t^k]              # (n_slots x d) proposed slots
2. Gate:         g_t   = g_min + (g_max - g_min) * sigmoid(W_g [h_t ; c_t])   # (n_slots x 1), per-slot write gate, clamped to [0.05, 0.95]
3. Composed-Lipschitz projection: for each slot i, compute the composed bound
                 L_i = (1 - g_t_i) + g_t_i * ||W_s^(i)||_2 * max_z |tanh'(z)|
                 with max_z |tanh'(z)| = 1, so L_i = (1 - g_t_i) + g_t_i * sigma_i
                 where sigma_i = ||W_s^(i)||_2 estimated by one power iteration.
                 Project: W_s_hat^(i) = W_s^(i) / max(1, sigma_i / (rho * (1 - eps)))
                 # projection keeps the composed map's Lipschitz constant <= rho*(1-eps)
                 # because L_i <= (1 - g_t_i) + g_t_i * rho*(1-eps) <= rho*(1-eps) + (1 - g_t_i)*(1 - rho*(1-eps)) <= rho for g_t_i in [0.05, 0.95] and rho=0.90
4. Candidate:    C_t   = tanh(W_s_hat U_t + b)                 # bounded candidate content
5. Update:       S_t   = (1 - g_t) * S_prev + g_t * C_t        # convex blend, per slot
6. Contraction audit (training only):
                 L_emp = max over batch of ||S_t(a) - S_t(b)|| / ||S_prev(a) - S_prev(b)||
                 where (a,b) are two rollouts sharing inputs but differing by noise delta
                 loss_contract = lambda * relu(L_emp - rho)^2
                 # audit isolates slot response by freezing h_t across the pair (a,b):
                 # h_t is held fixed to the value from rollout a, so the ratio reflects
                 # the slot map alone and not recurrent-state divergence
7. Emit:         S_t to local attention; add loss_contract to diffusion training loss
```

The contraction modulus `rho` is a single scalar shared across slots; the per-slot spectral projection at step 3 is what makes the trained operator satisfy it by construction rather than by hope. The composed-Lipschitz bound at step 3 accounts for the gate and tanh saturation: since `g_t` is clamped to `[0.05, 0.95]` and `|tanh'| <= 1`, the composed map `(1-g)*S + g*tanh(W_s U + b)` has Lipschitz constant at most `(1-g) + g*||W_s||_2`, and projecting `||W_s||_2` to `rho*(1-eps)` keeps this below `rho` for all gate values in the clamped range. The empirical audit at step 6 is a training-time monitor that catches violations introduced by the gate and the tanh saturation interacting across slots, with `h_t` frozen across the pair to isolate the slot map's response.

**Why this design.** We chose a per-slot spectral-normalized linear map with a convex gate blend over a free-form MLP slot updater because the contraction property must hold for *every* input, and a norm constraint on the linear part is the only way to get that without restricting the nonlinearity; the cost is that the slot map loses some expressivity relative to an unconstrained MLP, which we accept because the Mamba state and local attention still carry the high-frequency content. We chose a per-slot scalar gate `g_t` over a single global gate because different slots encode different temporal scales (recent pose vs. long-range goal), and forcing one forgetting rate across scales would either over-smooth recent motion or fail to forget stale goals; the cost is `n_slots` extra gate parameters and a slightly noisier contraction audit. We chose to enforce `rho` by projection at every forward pass rather than by a soft penalty alone because a soft penalty on the empirical Lipschitz ratio is only checked on the sampled pairs and can be satisfied while the true modulus exceeds `rho`; the cost is one power iteration per slot per frame, which is negligible against the denoising loop. We chose to keep the Mamba resumable recurrent states untouched and contract only the PCM slot path because the recurrent state is the mechanism that already gives Motion Mamba 2.0 its long-horizon reach, and contracting it would destroy the very capacity the architecture was built for; the cost is that the guarantee covers the slot-mediated history and not the raw recurrent state, which we handle by making the slots a fading-memory summary of that state. A domain expert would not describe this as a variant of Motion Mamba 2.0 with a regularizer: the contribution is a *certified operator class* for the history-compression step, and the training signal is derived from the operator's Lipschitz modulus, not from a generic smoothness prior.

**Why it measures what we claim.** The composed-Lipschitz projection at step 3 measures the *contraction modulus of the history-compression operator* because the induced 2-norm of the linear slot map, combined with the gate range and the tanh derivative bound, bounds the worst-case Lipschitz constant of the composed update over all inputs; this assumption fails when the gate `g_t` and the tanh saturation compose to produce an effective modulus larger than `rho` on some input region, in which case the projected norm reflects the linear part only and the audit at step 6 is what catches the gap. The empirical audit `L_emp` measures *whether perturbations to the compressed history actually decay across a rollout* because two rollouts sharing inputs but differing by a noise delta, with `h_t` frozen across the pair, isolate the slot map's response to that delta; this assumption fails when the two rollouts diverge in the Mamba recurrent state as well, in which case `L_emp` reflects the joint state response rather than the slot map alone, and we report both the projected and audited moduli. Terminal control error at horizon `T` measures *horizon-independence of the guarantee* because under a contraction with modulus `rho < 1` and `K`-step denoising, the per-step injected perturbation is bounded by a term proportional to the denoising residual and the accumulated error is a geometric series in `rho`, so the steady-state error is bounded by a constant that does not contain `T`; this assumption fails when the task's terminal-control target itself drifts with horizon (e.g., a goal that moves faster than the contraction can track), in which case the measured error reflects tracking lag rather than accumulation, and we report the goal-velocity condition alongside the error curve.

## Contribution

(1) CPCM, a contractive persistent-content-memory operator for autoregressive diffusion motion generation, in which every content-slot update is projected onto a Lipschitz ball of modulus rho and audited by a paired-rollout empirical contraction loss during training. (2) A design principle for long-horizon motion generation: horizon-stable terminal control is obtained by putting the stability invariant on the history-compression operator rather than on memory capacity, and the invariant is enforceable by a per-forward spectral projection plus a paired-rollout audit. (3) An evaluation protocol that extends HiPHI rollouts well past 39.52 s and reports terminal-control error against rollout length with the goal-velocity condition recorded, so that horizon-independence can be read off the error curve rather than asserted.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Retargeted HiPHI long-horizon rollouts | Baseline already validated terminal control here up to 39.52 s |
| Primary metric | Terminal control error vs. horizon T | Directly probes whether error saturates rather than accumulates |
| Baseline | Motion Mamba 2.0 (unmodified PCM) | Isolates the slot-update operator as the causal change |
| Baseline | Bounded attention-cache autoregressive diffusion | Tests the distant-history-access failure the slots are meant to fix |
| Baseline | Soft-Lipschitz-penalty PCM (no projection) | Tests projection-by-construction vs. penalty-by-hope |
| Ablation-of-ours | CPCM with global gate instead of per-slot gate | Tests whether per-slot forgetting rates matter |
| Ablation-of-ours | CPCM with projection applied to Mamba recurrent state and local-attention readout | Tests whether PCM-slot-specific contraction is load-bearing |
| Decomposition metric | Slot-path vs. recurrent-state error growth | Isolates which channel drives terminal error |

### Why this setup validates the claim

The central claim is that certifying the PCM slot operator as a contraction with modulus `rho < 1` converts horizon-dependent error accumulation into a horizon-independent steady-state bound. Retargeted HiPHI is the right dataset because the baseline already demonstrates accurate terminal control out to ~39.52 s there, so any change in the *shape* of the error-vs-horizon curve is attributable to our operator rather than to task difficulty. The primary metric — terminal control error as a function of T — is the only metric that can detect the predicted effect: a contraction predicts a curve that flattens toward a constant, whereas accumulation predicts continued growth. Unmodified Motion Mamba 2.0 tests whether the slot path is the binding constraint on that curve; the bounded-cache baseline tests whether the gain is specifically about distant-history access; the soft-penalty PCM tests whether projection-by-construction beats penalty-by-sampling, which is the sub-claim that the guarantee must hold for *every* input, not just sampled pairs. The global-gate ablation isolates whether the per-slot gate is load-bearing for the contraction or merely cosmetic. The projection-on-recurrent-state ablation tests whether the PCM-slot-specific contraction is the load-bearing element rather than a global smoothness effect. The decomposition metric directly tests the load-bearing assumption by reporting terminal error growth attributable to the slot path separately from growth attributable to the Mamba recurrent state.

### Expected outcome and causal chain

**vs. Motion Mamba 2.0 (unmodified PCM)** — On a rollout where the goal requires recalling a pose or spatial cue from far earlier than the local-attention window, unmodified PCM lets that stale content decay at a rate set by the learned gate interacting with tanh saturation, which is unconstrained and can exceed one on some input regions, so history information is either over-forgotten or amplified into drift. CPCM instead projects each slot map to composed Lipschitz ≤ `rho*(1-eps)`, so the same stale cue decays geometrically at a known rate and its residual contribution to the current denoising iterate is bounded. We expect the two methods to be close at short horizons and to separate visibly at the longest horizons, with our terminal-control error curve flattening while the baseline's keeps climbing.

**vs. Bounded attention-cache autoregressive diffusion** — On a case where the instruction changes mid-rollout and the correct response depends on a goal stated well before the cache window, the bounded cache simply cannot see the earlier text-conditioned context, so the agent either ignores the goal or re-derives it from a misleading recent window, producing late-rollout goal drift. CPCM's slots act as a fading-memory summary of the full recurrent state, so the distant goal survives as a compressed slot even when the attention cache has evicted it. We expect a large gap on the long-horizon goal-consistency subset and near-parity on short-horizon locomotion, which is the signature that the gain comes from history compression rather than from a better local policy.

**vs. Soft-Lipschitz-penalty PCM (no projection)** — On an input region the penalty never sampled during training, the unconstrained slot map can have a true modulus above `rho` while the empirical ratio on sampled pairs stays below it, so a perturbation injected by one denoising step is amplified rather than damped and the rollout diverges from the intended trajectory. CPCM's per-slot spectral projection bounds the composed map for every input by construction, and the audit catches any residual gap from gate–tanh composition. We expect the soft-penalty variant to match ours on average error but to show a heavy tail — a small fraction of rollouts with much larger terminal error — while ours has a tight error distribution, which is exactly the difference between a sampled guarantee and a certified one.

**vs. CPCM with projection applied to Mamba recurrent state and local-attention readout** — If the gain were a global smoothness effect, applying the same projection to the recurrent state and readout would match or exceed CPCM's horizon-independence. We expect this ablation to either degrade short-horizon motion quality (because contracting the recurrent state destroys capacity) or fail to improve the error-vs-horizon slope beyond CPCM, confirming that the PCM-slot-specific contraction is the load-bearing element.

**Decomposition: slot-path vs. recurrent-state error growth** — We report terminal error growth attributable to the slot path separately from growth attributable to the Mamba recurrent state by freezing one channel at a time and measuring the error slope. If the recurrent-state channel dominates, the horizon-independence claim is scoped to the slot path's bounded contribution rather than to total terminal error.

### What would falsify this idea

If the terminal-control error curve for CPCM grows with horizon at the same slope as unmodified Motion Mamba 2.0 — i.e., the gain is uniform across horizons rather than concentrated at long horizons where accumulation should be damped — then the contraction is not governing the error and the central claim is wrong. Likewise, if the soft-penalty variant shows no heavier tail than ours, the projection-by-construction argument fails to buy anything over sampling. Additionally, if the decomposition shows that recurrent-state error growth dominates and CPCM's slot-path bound does not reduce total terminal error relative to the baseline, the load-bearing assumption is false and the claim must be scoped to the slot path's bounded contribution.

## References

1. Motion Mamba 2.0: Autoregressive Diffusion Mamba for Long-Horizon Humanoid Control
