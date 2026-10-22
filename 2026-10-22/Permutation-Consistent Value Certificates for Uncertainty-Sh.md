# Permutation-Consistent Value Certificates for Uncertainty-Shrunk Policy Conditioning

## Motivation

Value-as-Memory (VaM) and GR-RL both replace external reward or intervention with a learned value function and then condition the policy on that value as if it were ground-truth task progress. Neither provides any structural guarantee that the cached value is identifiable from sparse reward, nor does either propagate its uncertainty into the conditioning interface. This is the meta-gap: across both trees, the value signal is the sole source of policy-improvement conditioning, yet no testable invariant certifies that the cached estimate is the identifiable progress signal rather than an arbitrary extrapolation.

## Key Insight

Reward-consistency under action-permutation is a structural invariant that certifies whether a cached value estimate is identifiable from sparse reward, because any value function that is not invariant to permuting the action-labeling of equivalent state-action pairs cannot be the unique sparse-reward-consistent value, and this certificate — unlike a learned confidence head — is falsifiable by construction.

## Method

**What it is.** We introduce PCV (Permutation-Certified Value), a value-memory interface that outputs a certified, uncertainty-shrunk value signal for policy conditioning. Inputs are the current observation, the reward buffer, and a candidate value estimate from an online value network; outputs are a scalar conditioning value `v_cert` and a binary certificate `c ∈ {0,1}` that gates whether the policy is conditioned on the value at all.

**Load-bearing assumption (stated explicitly).** The certificate is valid only if the following holds: for the true sparse-reward-consistent value, the action-value `Q(s,a)` is invariant across the task symmetry orbit `A_equiv(a)` — i.e. `Q(s,a) = Q(s,a_perm)` for every `a_perm` that is reward-equivalent to `a` — and the spread of `Q` across that orbit is monotone in value-estimation error. The certificate tests this invariance directly on an action-conditioned `Q_theta(s,a)` so that the permuted action enters the certified quantity and the reward term is non-trivial; a state-only `V_theta(s)` cannot carry this test because the permutation group never enters its computation. A non-degeneracy null check guards against trivially small residuals from degenerate (near-constant) value functions.

**How it works.**

```pseudocode
# Phase 1: Action-permutation probe construction
# Symmetry group G per task is obtained concretely as:
#   - manipulation: object-centric relabeling from segmentation masks
#     (permute which object is designated the target; reward-equivalent
#      when the mask class and contact geometry match)
#   - gripper tasks: action-label equivalence from known gripper symmetry
#     (left/right finger swap, 180-degree wrist-roll about the approach axis)
#   - fallback: approximate G from a learned classifier over (s,a) pairs
#     with reward-preservation score > 0.9 (used only in the sensitivity run)
def build_permutation_probes(D_reward, K=8):
    # D_reward: sparse-reward transitions (s, a, r, s')
    probes = []
    for each transition (s, a, r, s') in D_reward:
        A_equiv = {a' : a' ~ a under task symmetry group G}  # object-centric relabeling or gripper symmetry
        for k in 1..K:
            a_perm = sample_permutation(A_equiv)
            probes.append((s, a, a_perm, r, s'))  # reward must be preserved
    return probes

# Phase 2: Action-conditioned permutation-invariance residual + non-degeneracy null
def certificate(Q_theta, probes, delta=0.05, tau_var=0.05 * reward_scale):
    residuals = []
    for (s, a, a_perm, r, s') in probes:
        # Permutation-invariance residual: permuted action enters directly
        residuals.append(abs(Q_theta(s, a) - Q_theta(s, a_perm)))
    # One-sided test: certificate holds if residual quantile below delta
    q = quantile(residuals, 1 - delta)
    # Non-degeneracy null check: reject near-constant Q over probe states
    var_Q = variance({Q_theta(s, a) for (s, a, a_perm, r, s') in probes})
    nondegenerate = (var_Q >= tau_var)
    c = 1 if (q <= tau_cert and nondegenerate) else 0   # tau_cert = 0.1 * reward_scale
    return c, q, var_Q

# Phase 3: Uncertainty-shrunk conditioning
def condition_policy(obs, Q_theta, c, q, beta=0.5):
    v_raw = max_a Q_theta(obs, a)
    if c == 1:
        # shrink toward certified mean by residual magnitude
        v_cert = v_raw / (1 + beta * q)
    else:
        v_cert = 0.0   # no value conditioning when uncertified
    return pi(a | obs, v_cert)

# Online loop (per gradient step)
for step in 1..N:
    collect rollout with pi(· | obs, v_cert)
    update Q_theta on mixed offline+online buffer
    probes = build_permutation_probes(D_reward)
    c, q, var_Q = certificate(Q_theta, probes)
    update pi with PPO surrogate using v_cert as conditioning input
```

Hyperparameters: `K=8` permutation samples per transition; `delta=0.05` residual quantile; `tau_cert=0.1·reward_scale`; `tau_var=0.05·reward_scale` non-degeneracy threshold; `beta=0.5` shrinkage coefficient; `gamma=0.99`.

**Why this design.** We chose a reward-consistency residual under action-permutation over a learned confidence head because a learned head can only reproduce the value network's own biases, whereas the permutation residual is computed against the sparse reward buffer and therefore falsifies non-identifiable value estimates directly — accepting the cost that probe construction requires a known task symmetry group (object-centric relabeling or action-label equivalence), which we assume is available in manipulation domains. We chose a one-sided quantile test over a mean residual because sparse-reward tasks produce heavy-tailed Bellman residuals where the mean is dominated by a few transition outliers, accepting the cost that the quantile threshold `tau_cert` must be tuned per reward scale. We chose multiplicative shrinkage `v_raw/(1+beta·q)` over hard clipping because shrinkage preserves the relative ordering of value estimates across states while attenuating uncertified ones, accepting the cost that the policy still receives a (small) signal when `q` is moderately large. We chose to zero the value conditioning entirely when `c=0` rather than fall back to a learned value, because a fallback would reintroduce exactly the uncalibrated signal the certificate is meant to exclude — this is the design decision that distinguishes PCV from VaM's unconditional value caching, and a domain expert would not describe PCV as a variant of VaM because VaM has no certificate gate and no permutation probe. We chose an action-conditioned `Q_theta(s,a)` over a state-only `V_theta(s)` because the permutation group must enter the certified quantity for the invariance test to be non-vacuous, accepting the cost of a larger network and a max over actions at conditioning time.

**Why this measures what we claim.** The permutation-invariance residual `|Q(s,a) − Q(s,a_perm)|` measures *identifiability of the cached value from sparse reward* because the assumption that permuting the action within the task symmetry group preserves the reward makes the residual zero exactly when `Q` is invariant across the reward-equivalent orbit, which is the unique sparse-reward-consistent action-value on the probe set; this assumption fails when the symmetry group is misspecified (e.g., a permutation that is not actually reward-preserving), in which case the residual reflects symmetry-group error rather than value non-identifiability. The quantile `q` measures *propagated uncertainty* because the assumption that residual quantiles are monotone in value-estimation error makes `q` a conservative proxy for the error magnitude; this assumption fails when the value network is systematically biased in a direction that cancels across the orbit, in which case `q` reflects residual cancellation rather than true uncertainty. The non-degeneracy check `var_Q ≥ tau_var` measures *whether the value signal carries progress information at all* because the assumption that a certified value must vary across probe states excludes the degenerate constant solution `Q(s,a)=c` (which yields zero invariance residual regardless of identifiability); this assumption fails when the probe states are too homogeneous to exhibit variance, in which case `var_Q` reflects probe coverage rather than value informativeness. The certificate `c` measures *whether the value signal is safe to condition on* because the assumption that thresholding `q` at `tau_cert` and passing the non-degeneracy check separates identifiable from non-identifiable estimates makes `c=1` equivalent to certification; this assumption fails when the reward buffer is too sparse to populate probes, in which case `c` reflects probe coverage rather than identifiability.

## Contribution

(1) A calibrated value-memory interface, PCV, that augments a cached value estimate with a task-conditional identifiability certificate derived from reward-consistency under action-permutation and a propagated uncertainty field, gating policy conditioning on the certified, shrunk value. (2) A design principle: the certificate is falsifiable by construction against the sparse reward buffer, so it cannot be gamed by a learned confidence head, and this falsifiability is what makes the value conditioning safe for online RL. (3) An online training procedure that interleaves permutation-probe construction, certificate evaluation, and PPO updates on the certified value signal, compatible with the mixed offline/online data schedule of VaM.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Long-horizon deformable + contact-rich manipulation suite | Matches VaM's regime where value caching is claimed to help |
| Primary metric | Success rate on sparse-reward long-horizon tasks | Directly measures whether certified conditioning improves policy behavior |
| Secondary metric | Certificate-failure rate vs. ground-truth value error (on tabular control task with known optimal Q) | Validates that `q` tracks true value error before attributing gains |
| Baseline (VaM) | Unconditional online value-guided post-training | Tests whether the certificate gate adds value over raw value caching |
| Baseline (GR-RL) | Filtered+augmented VLA with online RL | Tests whether certification replaces expensive data filtering |
| Baseline (matched offline) | Offline training without value conditioning | Tests whether any value conditioning is necessary at all |
| Ablation-of-ours | PCV with certificate forced to c=1 always | Isolates the contribution of the certificate gate from shrinkage |
| Ablation-of-ours | PCV with Bellman-residual proxy instead of permutation-invariance residual | Tests whether the two residuals are equivalent (reviewer request) |
| Ablation-of-ours | PCV with non-degeneracy check removed | Isolates the contribution of the null check |
| Diagnostic | Apply certificate to VaM, GR-RL, and vanilla value-guided baseline; report failure rate vs. actual value error | Tests generality as an audit tool |
| Sensitivity | PCV with approximate symmetry group G (learned classifier, reward-preservation > 0.9) | Quantifies sensitivity to symmetry misspecification |

### Why this setup validates the claim

The central claim is that a permutation-certified, uncertainty-shrunk value signal improves policy post-training over unconditioned value caching precisely because it suppresses non-identifiable value estimates on sparse-reward transitions. The matched-offline baseline establishes the floor: if value conditioning of any kind matters, both VaM and PCV should beat it, so a PCV-vs-offline gap alone would not be evidence for the certificate. The decisive comparison is PCV vs. VaM: VaM conditions on every cached value regardless of identifiability, so if the certificate is doing real work, PCV should gain specifically on the subset of tasks where sparse rewards make cached values least identifiable (contact-rich, long-horizon, low-reward-density transitions) while matching VaM on reward-dense subsets. The forced-c=1 ablation is the sharpest test: it is PCV's mechanism with the gate removed, so any PCV advantage over it must come from the gating decision itself, not from shrinkage or the probe machinery in general. The Bellman-residual-proxy ablation directly tests the reviewer's concern that the permutation residual is a no-op: if the two residuals produce different certificate decisions and different success rates, the permutation-invariance test is doing work the Bellman residual cannot. The non-degeneracy ablation tests whether the null check prevents certification of degenerate constant values. The diagnostic run applies the certificate as a standalone audit to VaM, GR-RL, and a vanilla value-guided baseline, reporting certificate-failure rate against actual value error on a tabular control task with known optimal Q, so the contribution is a general audit tool rather than a single pipeline. The sensitivity run with approximate G quantifies how much symmetry misspecification degrades the certificate. Success rate is the right primary metric because the predicted effect is behavioral—fewer confidently-wrong value-conditioned actions on hard transitions—and success rate aggregates exactly those action-level failures over long horizons where value errors compound.

### Expected outcome and causal chain

**vs. VaM (unconditional value-guided post-training)** — On a contact-rich long-horizon task where sparse rewards leave many transitions with near-zero Bellman signal, VaM conditions the policy on cached values that are unidentifiable from reward and therefore confidently wrong on the rare decisive transitions, producing premature or oscillating manipulation behavior. Our method instead detects the elevated permutation-invariance residual quantile on those transitions, sets c=0 so the policy receives no value conditioning there, and shrinks v_cert where the certificate holds, so we expect a noticeable success-rate gap concentrated on low-reward-density, contact-rich tasks and rough parity on reward-dense tasks.

**vs. GR-RL (filtered and augmented VLA with online RL)** — On a long-horizon dexterous task where the demonstrations are noisy and the reward is sparse, GR-RL must filter and augment data to remove the transitions that would corrupt online RL, and any transition the filter misses still injects a bad value target, producing a slower and less stable learning curve on the hardest stages. Our method instead certifies per-step at training time using the reward buffer itself, so it does not need a clean dataset to avoid conditioning on bad values, and we expect PCV to reach comparable final success with a visibly steeper early learning curve on the multi-stage tasks.

**vs. Matched offline training (no value conditioning)** — On a task where the policy must distinguish near-identical observations that differ only in task progress, matched offline training has no progress signal and defaults to the marginal behavior in the demonstration distribution, producing stalls at stage boundaries. Our method instead supplies a certified scalar progress signal that is shrunk toward zero exactly when it cannot be trusted, so the policy gets a usable conditioning signal on identifiable states without being misled elsewhere, and we expect PCV to beat matched offline training by a clear margin on stage-boundary success while the forced-c=1 ablation loses part of that margin.

**vs. Bellman-residual-proxy ablation** — If the permutation-invariance residual is equivalent to the Bellman residual, the two ablations should produce identical certificate decisions and identical success rates. We expect the Bellman-residual proxy to certify degenerate constant values (which yield small residuals regardless of identifiability) and to miss non-identifiable extrapolations that happen to satisfy `Q(s,a)=γQ(s',a')` on observed transitions, so the proxy should show higher certificate-failure rate against ground-truth value error on the tabular control task and lower success rate on sparse-reward manipulation tasks.

**vs. Non-degeneracy ablation** — Removing the null check should allow certification of near-constant Q values that pass the invariance residual trivially, producing value conditioning that carries no progress signal; we expect lower success on stage-boundary tasks where progress information is required.

### What would falsify this idea

If PCV's gain over VaM is uniform across reward-density subsets rather than concentrated on the sparse-reward, contact-rich transitions where the permutation-invariance residual should be high, then the certificate is not the causal mechanism and the central claim is wrong. Likewise, if the forced-c=1 ablation matches PCV, the gate contributes nothing and the shrinkage/probe machinery is doing all the work; and if the Bellman-residual-proxy ablation matches PCV on both certificate decisions and success rate, the permutation-invariance test is not adding information beyond a standard Bellman residual.

## References

1. Value as Memory: Online Value-Guided Post-training for General Robot Policies
2. GR-RL: Going Dexterous and Precise for Long-Horizon Robotic Manipulation
3. RL-100: Performant Robotic Manipulation with Real-World Reinforcement Learning
4. One-Step Diffusion Policy: Fast Visuomotor Policies via Diffusion Distillation
5. Diffusion Policy Policy Optimization
6. UFOGen: You Forward Once Large Scale Text-to-Image Generation via Diffusion GANs
7. Diff-Instruct: A Universal Approach for Transferring Knowledge From Pre-trained Diffusion Models
8. One-Step Diffusion with Distribution Matching Distillation
