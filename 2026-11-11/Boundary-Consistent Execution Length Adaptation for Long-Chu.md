# Boundary-Consistent Execution Length Adaptation for Long-Chunk Policies

## Motivation

ExRL adapts execution length via reward-driven RL but lacks any constraint tying the adapted length to the base policy's chunk-boundary dynamics, so the adapter can select lengths that shift the distribution of next-chunk start states away from what the base policy was trained on. This schedule-inconsistency is structural: reward maximization alone provides no signal that a length change breaks the temporal alignment between consecutive chunks, causing the base policy to encounter out-of-distribution boundary states and degrade. The field's trajectory toward low-dimensional online adaptation (ExRL) makes this gap binding because the adapter's only degree of freedom is precisely the quantity that controls boundary alignment.

## Key Insight

The base policy's chunk-boundary start-state distribution is an invariant of the policy-chunking pair, so penalizing deviations of the adapted next-chunk start state from a learned boundary model yields reward-driven length changes that cannot silently violate the temporal contract the base policy was trained under.

## Method

```pseudocode
Input: base long-chunk policy π_base, chunk size H, adapter A_θ(s, chunk) → ℓ ∈ {1..H}, boundary reference set R = {(s_i, chunk_i, s'_i)} collected from N_ref = 5000 offline rollouts of π_base with fixed length H (each rollout contributes ⌊T/H⌋ boundary transitions), boundary model B_φ (diagonal Gaussian, 2×64 hidden), replay buffer D, reward r
Output: adapted execution lengths ℓ_t per chunk

1. Initialize A_θ (small MLP, 2×64 hidden, GeLU), B_φ (diagonal Gaussian, 2×64 hidden, GeLU), D empty. Freeze R (never updated).
2. Pretrain B_φ on R: minimize NLL of observed next-chunk start state s' given (s, chunk), for 5k gradient steps (Adam, lr=1e-3, batch=256).
3. For each online episode:
   a. Observe state s_t; query π_base → chunk a_{t:t+H}.
   b. Sample ℓ_t ~ A_θ(s_t, chunk) (Gaussian policy, clipped to [1,H]).
   c. Execute a_{t:t+ℓ_t}; record (s_t, chunk, ℓ_t, r_t, s_{t+ℓ_t}).
   d. Compute boundary-consistency loss: L_bc = -log B_φ(s_{t+ℓ_t} | s_t, chunk)  [penalizes start states unlikely under base boundary distribution].
   e. Store transition (s_t, chunk, ℓ_t, r_t, s_{t+ℓ_t}, L_bc) in D.
   f. Every 1 environment step: sample batch from D; update A_θ with SAC loss:
        L_A = -E[Q(s,chunk,ℓ)] + α·E[log A_θ(ℓ|s,chunk)] + λ·E[L_bc]   (λ=0.1, α auto-tuned)
      Update Q with standard SAC critic loss.
   g. Every 10 steps: fine-tune B_φ on recent D (next-chunk start states under adapted lengths) with a small learning rate (1e-4) to prevent stale boundary model.
4. Return adapted lengths for deployment.
```

**Calibration and verification of the load-bearing assumption** — The method assumes that B_φ accurately captures the base policy's chunk-boundary dynamics and that L_bc therefore measures true temporal-contract violation rather than model uncertainty. We make this assumption explicit and test it as follows. (i) Calibration test: on a held-out set of 1000 base-policy boundary transitions (disjoint from R), compute the correlation (Pearson and Spearman) between L_bc and an independent ground-truth boundary-violation signal, defined as the drop in base-policy action log-likelihood under fixed H, i.e., log π_base(a_{t:t+H} | s_{t+ℓ_t}) − log π_base(a_{t:t+H} | s_t). Report the correlation; if |r| < 0.3, the mechanism claim is unsupported. (ii) Off-manifold stress test: on adapted rollouts, bin transitions by L_bc quintile and report the mean ground-truth violation signal per bin; a monotone increase supports that L_bc tracks violation, while a flat or inverted trend indicates L_bc reflects novelty/uncertainty. (iii) Non-parametric boundary baseline: replace B_φ with a kNN density estimate over R (k=10, Euclidean distance in state space) and report the same calibration and stress-test diagnostics, so the paper isolates whether the learned parametric boundary model is the novel ingredient or merely a convenient regularizer. The periodic fine-tuning of B_φ (step 3g) is retained as specified, but the calibration test is run both before and after fine-tuning to quantify any drift in the L_bc–violation correlation.

**Why this design** — We chose a learned diagonal-Gaussian boundary model B_φ over a fixed replay buffer of base-policy transitions because the base policy's boundary distribution is state- and chunk-dependent, and a non-parametric buffer would require prohibitive coverage; this accepts the cost that B_φ may misestimate rare boundary states, which we mitigate by fine-tuning every 10 steps. We chose to add the boundary-consistency loss as a soft penalty (λ=0.1) to the SAC actor loss rather than a hard constraint on ℓ because a hard constraint would require solving a constrained MDP online, which is sample-inefficient and brittle with a learned boundary model; the soft penalty trades off reward maximization against consistency, and λ controls this trade-off. We chose to pretrain B_φ offline on base-policy rollouts with fixed H rather than learning it jointly from scratch because the boundary distribution is a property of the base policy and chunking, not of the adapter, and joint learning would confound adapter-induced distribution shift with base-policy boundary dynamics; the cost is that B_φ may drift if the adapter systematically shifts boundaries, which the periodic fine-tuning addresses. Unlike ExRL, which trains the adapter purely on task reward and thus has no mechanism to detect boundary violations, our design explicitly couples length selection to the base policy's boundary model, so a domain expert would not describe this as a variant of ExRL but as a schedule-consistency regularizer that ExRL lacks.

**Why it measures what we claim** — The boundary-consistency loss L_bc = -log B_φ(s_{t+ℓ_t} | s_t, chunk) measures schedule-consistency because it quantifies the likelihood of the adapted next-chunk start state under the base policy's boundary distribution; this assumes that B_φ accurately captures the base policy's chunk-boundary dynamics, an assumption that fails when the adapter visits states far outside the base policy's training distribution, in which case L_bc reflects model uncertainty rather than true boundary violation. The adapter's sampled length ℓ_t measures the execution-length decision because it is the sole degree of freedom, assuming that length is the only quantity varied while the chunk remains fixed; this assumption fails when the base policy's chunk quality varies with state, in which case ℓ_t conflates length adaptation with chunk suitability. The reward r_t measures task performance because it is the environment's objective, assuming the reward is stationary and well-shaped; this assumption fails under reward hacking or sparse rewards, in which case the adapter may exploit reward while violating boundaries, which is exactly why L_bc is needed. The periodic fine-tuning of B_φ measures boundary-model freshness because it tracks the adapted state distribution, assuming the adapter's boundary shifts are gradual; this assumption fails under abrupt policy changes, in which case B_φ lags and L_bc is temporarily miscalibrated.

## Contribution

(1) A boundary-consistency regularizer for execution-length adapters that penalizes length changes violating the base policy's chunk-boundary distribution, implemented via a learned boundary model and integrated into off-policy RL training. (2) A schedule-consistency constraint that couples the adapter's length decisions to the base policy's temporal contract, yielding reward-driven adaptation that preserves boundary alignment. (3) An empirical demonstration that boundary-consistent length adaptation improves success rates and sample efficiency over unregularized ExRL on simulated manipulation benchmarks.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|------------------------|
| Dataset | Meta-World long-horizon manipulation tasks (6 tasks: door-open, drawer-close, pick-place, peg-insert-side, hammer, push) | Standard benchmark for chunked policy adaptation |
| Primary metric | Success rate vs. adaptation environment steps | Directly measures sample efficiency of adaptation |
| Secondary metric | Calibration correlation (Pearson/Spearman) between L_bc and ground-truth boundary-violation signal (base-policy action log-likelihood drop) on 1000 held-out adapted transitions | Tests whether L_bc tracks true violation |
| Baseline | ExRL (reward-only length adapter) | Tests whether boundary-consistency adds value over pure reward |
| Baseline | Fixed execution length H | Tests whether any adaptation beats no adaptation |
| Baseline | Full-chunk RL fine-tuning | Tests whether length-only control is more sample-efficient |
| Baseline | kNN boundary density (k=10) replacing B_φ | Isolates whether learned parametric model is the novel ingredient |
| Baseline | Hard trust-region on ℓ derived from B_φ (reject ℓ if L_bc > τ, τ set to 95th percentile of base-policy L_bc) | Tests soft penalty vs. hard constraint |
| Ablation-of-ours | Ours without L_bc (λ=0) | Isolates the causal contribution of boundary consistency |
| Ablation-of-ours | λ ∈ {0.01, 0.1, 1.0} | Sensitivity to penalty strength |
| Ablation-of-ours | B_φ fine-tune interval ∈ {10, 100, never} | Sensitivity to boundary-model freshness |
| Compute budget | 3 seeds × 6 tasks × 7 methods (ExRL, fixed H, full-chunk RL, kNN, hard trust-region, ours, λ=0) = 126 runs; each run ≤ 200k environment steps; estimated 2 GPU-days per run on a single V100 | Bounded, reproducible grid |

### Why this setup validates the claim

The central claim is that coupling execution-length selection to a base-policy boundary model yields more sample-efficient and safer adaptation than reward-only length control. Meta-World tasks with long action chunks provide exactly the regime where the base policy's boundary distribution is state- and chunk-dependent, so a reward-only adapter (ExRL) can drift into boundary-violating lengths that look locally rewarding but compound into failure. The primary metric — success rate as a function of environment steps — is the right detector because the predicted effect is a sample-efficiency gap, not a final-performance gap; if our mechanism works, the gap should appear early in adaptation and shrink as all methods converge. The fixed-length baseline tests whether adaptation matters at all; full-chunk RL tests whether length-only control is the right degree of freedom; ExRL tests the specific sub-claim that boundary consistency is the missing ingredient beyond reward; the kNN and hard trust-region baselines test whether the learned parametric boundary model is the novel ingredient or merely a convenient regularizer; and the λ=0 ablation directly isolates L_bc's contribution, making the causal attribution falsifiable rather than correlational. The secondary calibration metric directly tests the load-bearing assumption that L_bc tracks true boundary violation: if the correlation between L_bc and the base-policy action log-likelihood drop is weak (|r| < 0.3), then the mechanism claim is unsupported regardless of success-rate gains.

### Expected outcome and causal chain

**vs. ExRL (reward-only length adapter)** — On a case where the base policy's chunk boundary distribution is narrow but the task reward temporarily favors a longer chunk (e.g., a reaching phase where extending the chunk overshoots into a contact-rich phase), ExRL produces overlong execution lengths because its actor sees only Q-value gradient and has no signal that the resulting next-chunk start state is off the base-policy manifold. Our method instead penalizes low-likelihood boundary states via L_bc, so the actor is pushed back toward lengths whose start states B_φ assigns high probability, and we expect a noticeable success-rate gap concentrated on tasks with sharp phase transitions (door-open, peg-insert-side), with parity on smooth single-phase tasks (push).

**vs. Fixed execution length H** — On a case where the optimal execution length varies across task phases (short during precision alignment, long during free-space transit), the fixed-length baseline produces either premature chunk termination during transit (wasting the base policy's long-horizon plan) or overshoot during alignment (accumulating error before feedback), because it cannot condition length on state. Our method instead samples ℓ_t from A_θ(s_t, chunk), which learns to shorten during precision phases and lengthen during transit, so we expect a moderate but consistent success-rate advantage that grows with task horizon and phase heterogeneity.

**vs. Full-chunk RL fine-tuning** — On a case where the base policy's chunk is already near-optimal but slightly misaligned with the task's timing, full-chunk RL fine-tuning produces high-variance updates and slow adaptation because it must perturb all H action dimensions to correct a timing error, requiring many more samples to find the right adjustment. Our method instead varies only the scalar execution length, so the effective search space is one-dimensional and the credit assignment is direct, and we expect full-chunk RL to require substantially more environment steps to reach the same success rate, with the gap largest early in adaptation.

**vs. kNN boundary density (k=10)** — On a case where the base-policy boundary distribution is multi-modal (e.g., contact-rich transitions), the kNN density estimate may be more accurate than the diagonal-Gaussian B_φ, so we expect the kNN baseline to match or slightly exceed our method's success rate on those tasks. If the kNN baseline matches our method, the learned parametric boundary model is not the novel ingredient; if our method outperforms kNN, the parametric model's generalization is beneficial.

**vs. Hard trust-region on ℓ** — On a case where the boundary model is well-calibrated, the hard trust-region baseline should achieve similar success rates but may be more sample-inefficient early in adaptation because rejected lengths provide no gradient signal. We expect our soft penalty to adapt faster initially, with the hard trust-region catching up later.

### What would falsify this idea

If the λ=0 ablation matches our method's success-rate-vs-steps curve across all task subsets — i.e., the gain is uniform rather than concentrated on tasks with sharp phase transitions and heterogeneous optimal lengths — then boundary consistency is not the causal ingredient and the central claim is wrong. Additionally, if the calibration correlation between L_bc and the ground-truth boundary-violation signal is weak (|r| < 0.3) on held-out adapted transitions, then L_bc does not measure what we claim and the mechanism is unsupported.

## References

1. ExRL: Sample-Efficient Online Adaptation of Long-Chunk Policies via Execution Length Control
