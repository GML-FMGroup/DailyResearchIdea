# Length-Conditioned Residual Heads for Frozen Flow Policies via Off-Policy RL

## Motivation

ExRL shows that execution-length control is a remarkably sample-efficient adaptation space for long-chunk policies, but it structurally confines adaptation to a 1-D scalar per chunk: the flow-based chunk sampler stays frozen, so any action-level suboptimality in the base chunk is inherited verbatim. The result is that when the length decision merely extends or shortens a bad chunk, task reward cannot flow back into the action distribution — the adapter is optimizing a timing knob while the underlying trajectory remains uncorrectable. This is the recurring meta-level bottleneck of frozen-generator adaptation: low-dimensional adapters are cheap to train but cannot refine the object they condition on.

## Key Insight

Conditioning a bounded residual correction on the chosen execution length makes residual magnitude and timing jointly identifiable under a single task reward, because the residual is only ever queried at the horizon it will actually be executed — so credit for a lengthened chunk is assigned to a correction that is consistent with that length rather than to an action that was optimized for a different horizon.

## Method

**What it is.** LCR-ExRL (Length-Conditioned Residual ExRL) augments a frozen flow-based VLA with a single small adapter network that, given the base action chunk and proprioceptive state, jointly emits an execution length ℓ and a bounded residual correction Δa conditioned on ℓ; the executed action sequence is the base chunk plus Δa over the first ℓ steps. The adapter is trained online with off-policy SAC; the base flow policy is never unfrozen.

**How it works.**

```pseudocode
# Frozen: flow-based VLA pi_base, chunk length H, action dim d
# Trainable: adapter f_theta (2-layer MLP, hidden 256, tanh output heads)
# Residual scale: per-dimension learnable log-std s_j initialized to log(0.1), so delta_max_j = exp(s_j)
#   (replaces fixed delta_max=0.1; s_j is updated by SAC like any policy parameter)
# Calibration check (run once before training): on 512 states from the replay buffer,
#   compute the residual magnitude needed to move the base chunk's first action to the
#   nearest expert action (or to the action that maximizes the critic's Q among 100 random
#   perturbations). If the 90th percentile of this required magnitude exceeds exp(s_j)
#   for any dimension, increase the initial s_j for that dimension to match the 90th percentile.

for each rollout step t:
    obs = get_observation()
    A_base = pi_base.sample_chunk(obs)            # [H, d], frozen

    # --- adapter forward ---
    z       = MLP_trunk(concat(obs, flatten(A_base)))     # 256-d
    logits_l = Linear(z -> H)                              # length head
    ell     = sample_or_argmax(logits_l)                   # in {1..H}
    delta_pre = Linear(concat(z, onehot(ell)) -> H*d)      # pre-tanh residual
    delta   = tanh(delta_pre) * exp(s)                     # s is [d], broadcast over H steps
    A_exec  = A_base + reshape(delta, [H, d])

    # --- execute first ell steps, then re-query ---
    for k in 1..ell:
        env.step(A_exec[k])
    store transition (obs, A_base, ell, delta, sum_reward_over_ell, next_obs) in replay buffer

# --- off-policy update (SAC) ---
for each gradient step:
    sample minibatch from replay buffer
    # length head: categorical policy over {1..H}, entropy-regularized
    # residual head: Gaussian over delta_pre, squashed by tanh*exp(s), entropy-regularized
    # joint log-prob = log pi(ell|obs,A_base) + log pi(delta|obs,A_base,ell)
    update f_theta to maximize  Q(obs, A_base, ell, delta) + alpha * H(joint policy)
    update Q-networks (twin critics, target networks, tau=0.005) by TD on the
        length-summed reward with discount gamma^ell
    update s (log-std) by SAC policy gradient (same as other policy parameters)

# --- diagnostic for length-residual confounding (logged every 1000 gradient steps) ---
#   Compute Pearson correlation between ell and ||delta||_2 in the replay buffer.
#   If |corr| > 0.7, log a warning and freeze the length head for 1000 steps
#   to allow the residual head to adapt without length confounding.
```

Hyperparameters: SAC with twin critics, learning rate 3e-4, replay buffer 1e5 transitions, batch 256, target update τ=0.005, entropy temperature auto-tuned, residual scale s initialized to log(0.1) per dimension and updated by SAC, chunk length H inherited from the base policy (e.g., 8–16).

**Why this design.** We chose to condition the residual head on the sampled length via a one-hot concatenation rather than emitting a single length-agnostic residual because the optimal correction to a chunk is horizon-dependent — a chunk executed for 2 steps should be corrected differently than the same chunk executed for 12 — accepting the cost that the residual head must be queried once per length sample and cannot be cached across lengths. We chose to bound the residual with tanh·exp(s) (learnable per-dimension scale) rather than a fixed δ_max=0.1 because a fixed small bound cannot correct directionally wrong base chunks, and a learnable scale lets the adapter grow the correction when the critic's advantage of the residual over the base chunk is positive, while still preventing the adapter from silently becoming a full policy; we accept the cost that the scale must be calibrated (via the 512-state check above) and that large base-policy errors may still exceed the learned bound. We chose a single shared adapter trunk with two heads over two separate networks because length and residual must share the chunk representation to remain mutually consistent, accepting the cost of entangled gradients between the two heads. We chose SAC over PPO because the length action is discrete and the residual is continuous, and SAC's off-policy replay is what gives ExRL its sample efficiency. Unlike ExRL, which treats adaptation as a pure 1-D timing problem, and unlike standard RL fine-tuning of the full flow policy, LCR-ExRL keeps the base sampler frozen while restoring action-level refinement — a domain expert would not describe this as 'ExRL plus a residual' because the residual is conditioned on the length sample, making the two heads a single coupled action rather than two stacked adapters.

**Why it measures what we claim.** The length-conditioned residual Δa(ℓ) measures action-level refinement under task reward because the residual is only ever executed over exactly the ℓ steps it was conditioned on, so the reward attributed to it is the reward of the corrected trajectory at that horizon; this assumption — that the length sample is the execution horizon — fails only if the environment terminates early, in which case the residual reflects a correction for a horizon that was not fully realized. The joint log-probability log π(ℓ) + log π(Δa|ℓ) measures the coupling between timing and correction because SAC's entropy-regularized objective treats the pair as one action, so the critic's value is a function of both; this assumption fails when the critic cannot distinguish length-driven from residual-driven reward, in which case the gradient reflects whichever factor dominates the TD error. The residual bound tanh·exp(s) measures 'bounded refinement of a frozen generator' because tanh saturation guarantees the executed action stays within exp(s) of the base chunk; this assumption fails when the base action is near the action-space boundary and the residual is clipped by the environment, in which case the bound reflects environment saturation rather than the adapter's intended correction. The calibration check (512 states, 90th percentile of required residual magnitude) measures whether the learned bound is large enough to correct the targeted base-chunk errors because it directly estimates the residual magnitude needed to reach a higher-reward action; this assumption fails when the critic's Q is inaccurate, in which case the calibration reflects critic error rather than true required correction.

## Contribution

(1) LCR-ExRL, an adapter architecture that couples a categorical execution-length head with a length-conditioned bounded residual head on top of a frozen flow-based VLA, trained jointly by off-policy SAC as a single composite action. (2) The design principle that length-conditioned residuals make timing and action correction jointly identifiable under one task reward, whereas length-only adapters (ExRL) and action-only adapters each leave one factor unrefined. (3) A concrete recipe for online adaptation of long-chunk policies that preserves the base generator's sample efficiency while restoring action-level refinement.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Meta-World long-horizon tasks (e.g., door-open, drawer-open, pick-place) | Long chunks expose horizon-dependent correction |
| Dataset | RoboSuite long-horizon manipulation (e.g., kitchen, lift) | High-fidelity simulated benchmark for significance |
| Primary metric | Success rate vs. adaptation steps | Directly tests sample-efficient refinement claim |
| Secondary metric | Wall-clock adaptation cost (GPU hours to reach 80% success) | Practical frozen-VLA refinement cost |
| Baseline | ExRL (length-only adaptation) | Isolates value of action-level residual |
| Baseline | Full-policy RL fine-tuning | Tests whether frozen-generator prior helps |
| Baseline | Fixed-length residual adaptation | Tests whether length-conditioning matters |
| Ablation-of-ours | Length-agnostic residual (no ℓ conditioning) | Isolates coupling of length and correction |
| Ablation-of-ours | Fixed residual bound δ_max=0.1 (no learnable scale) | Tests calibration of residual magnitude |
| Diagnostic | Correlation between ℓ and ||Δa|| in replay buffer | Detects length-residual confounding |

### Why this setup validates the claim

The central claim is that a frozen flow-VLA can be sample-efficiently refined by a coupled (length, residual) adapter, and that the coupling matters. Each baseline targets one link in that chain. ExRL tests the sub-claim that length control alone is insufficient: if residual refinement adds nothing, ExRL should match us. Full-policy RL fine-tuning tests the sub-claim that freezing the base generator preserves sample efficiency: if unfreezing is harmless, our adapter's advantage vanishes. Fixed-length residual adaptation tests the sub-claim that the residual must be horizon-conditioned: if a length-agnostic residual performs equally, the one-hot conditioning is wasted capacity. The ablation is the sharpest test, because it removes exactly the coupling mechanism we argue is essential while keeping every other component identical. The fixed-bound ablation tests whether the learnable residual scale is necessary to correct directionally wrong base chunks. Success rate as a function of adaptation steps is the right metric because the claim is about sample efficiency under task reward, not asymptotic performance; a method that reaches the same success rate with far fewer environment steps but the same final plateau is exactly what the mechanism predicts, and a metric like final return would be blind to it. Wall-clock adaptation cost on RoboSuite substantiates the practical significance claim beyond Meta-World. The ℓ–Δa correlation diagnostic directly tests the credit-assignment assumption: if the correlation is high, the critic may be confounding length and residual, and the claimed coupling is not doing its job.

### Expected outcome and causal chain

**vs. ExRL (length-only adaptation)** — On a task where the base chunk is directionally wrong at the start of execution but correct near the end (e.g., a reach that overshoots early), ExRL can only shorten the executed horizon, so it either commits to the wrong early actions or truncates before the useful tail, producing slow or failed adaptation. Our method instead applies a bounded residual over exactly the ℓ steps it was conditioned on, correcting the early segment while preserving the tail, so we expect a clear success-rate gap on horizon-misaligned tasks and near-parity on tasks where the base chunk is already well-timed. The learnable residual scale (calibrated to the 90th percentile of required correction on 512 states) ensures the residual is large enough to reverse directionally wrong early actions; the fixed-bound ablation (δ_max=0.1) is expected to fail on these tasks, confirming the calibration is load-bearing.

**vs. Full-policy RL fine-tuning** — On a task with a sparse success signal and a high-dimensional action space, full fine-tuning must relearn the flow sampler from scratch under weak reward, so it burns many environment steps before any success appears and often collapses the pretrained prior. Our method instead only searches a bounded residual plus a discrete length, keeping the base distribution intact as a prior, so we expect our curve to rise within a small fraction of the steps full fine-tuning needs, with the gap largest early and narrowing only if full fine-tuning is given an order of magnitude more data. Wall-clock adaptation cost on RoboSuite is expected to be lower for our method by at least 3× to reach 80% success.

**vs. Fixed-length residual adaptation** — On a task where the optimal execution horizon varies across states (e.g., contact-rich manipulation where some states need a short corrective burst and others a long sweep), fixed-length residual adaptation must compromise on one horizon, so its residual is systematically mis-scaled for the states that need the other extreme. Our method instead samples ℓ per state and conditions the residual on that ℓ, so we expect a bimodal pattern: parity on states whose optimal horizon matches the fixed choice, and a marked gap on states whose optimal horizon is far from it. The ℓ–Δa correlation diagnostic is expected to remain below 0.7, indicating that length and residual are not confounded and the coupling is effective.

### What would falsify this idea

If the length-agnostic ablation matches the full method on every task subset — and especially if the gain over ExRL and fixed-length adaptation is uniform rather than concentrated on horizon-misaligned states — then the claimed coupling of timing and correction is doing no work, and the central mechanism is wrong. Likewise, if full-policy fine-tuning matches our sample efficiency, the frozen-generator premise fails. If the fixed-bound ablation (δ_max=0.1) matches the full method, the learnable residual scale is unnecessary and the calibration story is falsified.

## References

1. ExRL: Sample-Efficient Online Adaptation of Long-Chunk Policies via Execution Length Control
