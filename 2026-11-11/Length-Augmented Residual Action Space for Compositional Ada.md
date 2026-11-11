# Length-Augmented Residual Action Space for Compositional Adaptation of Long-Chunk Policies

## Motivation

Execution-length adapters such as ExRL train a scalar length predictor in a separate, non-composable space, so the length decision cannot be corrected jointly with the action content it governs; the adapter can only stretch or shrink a fixed chunk and cannot repair a suboptimal chunk. This is one instance of a meta-level gap across residual RL for assembly, residual off-policy RL, and open-world VLA branches: residual adaptation is applied to heterogeneous quantities (torques, actions, execution horizons) that live in disjoint spaces, so residuals cannot be composed, reused, or credited together. The structural cause is that the length dimension is excluded from the residual action space, forcing any length correction to be a standalone policy rather than a residual that shares representation and credit assignment with action residuals.

## Key Insight

If the execution length is embedded as one coordinate of the residual action space, then length residuals and action residuals become elements of the same vector space and the same off-policy residual critic assigns credit to both, so length adaptation and action correction are composed by a single residual policy rather than coordinated by an external scheduler.

## Method

**What it is.** LARX (Length-Augmented Residual eXecution) is a residual off-policy RL method that augments the residual action space with a length coordinate. Its inputs are the current state and the base long-chunk policy's proposed chunk; its outputs are a residual action vector and a residual length, both produced by one residual policy and both scored by one residual Q-function. The load-bearing assumption is stated explicitly: a single residual Q-function over the concatenated vector [r_len, r_a] can assign correct credit to the length coordinate and the action coordinates simultaneously, i.e., the value of the joint residual decomposes so that the length decision's causal effect on the executed action sequence is captured by the same scalar return the critic regresses. This assumption is made explicit in block (D) and tested by the counterfactual residual-ablation diagnostic in the experiment.

**How it works.**
```pseudocode
def LARX_step(state, base_policy, residual_policy, residual_Q, replay):
    # 1. Base chunk (frozen, long-horizon)
    chunk = base_policy(state)                      # shape [H, d]
    H = chunk.shape[0]

    # 2. Residual policy emits joint residual in augmented space
    #    r = [r_len, r_a_0, ..., r_a_{H-1}]
    r = residual_policy(state, chunk)               # shape [1 + H*d]
    r_len = r[0]                                    # scalar, unconstrained
    r_a   = r[1:].reshape(H, d)                     # action residual

    # 3. Compose: length and actions share one residual representation
    L = clip(round(L_base * exp(r_len)), 1, H)      # L_base = H/2 default
    a_t = chunk[t] + r_a[t]  for t in 0..L-1        # execute composed chunk
    # accumulate discounted return over the L executed steps
    R = 0; disc = 1.0
    for t in 0..L-1:
        R += disc * reward_t
        disc *= gamma
    execute(a_t for t in 0..L-1)  # environment returns per-step rewards

    # 4. Off-policy update on the SAME augmented residual vector
    #    duration-aware (semi-MDP) target: sum_{t=0}^{L-1} gamma^t r_t + gamma^L * Q(next_state, r')
    transition = (state, r, R, L, next_state)
    replay.add(transition)
    batch = replay.sample(256)
    # critic is conditioned on L explicitly via a length embedding
    target = R + (gamma ** L) * residual_Q(next_state, residual_policy(next_state, base_policy(next_state)), L_next)
    loss_Q = MSE(residual_Q(state, r, L), target)
    loss_pi = -residual_Q(state, residual_policy(state, base_policy(state)), L)
    update(residual_Q, loss_Q); update(residual_policy, loss_pi)
```
Hyperparameters: replay buffer 1e5, batch 256, gamma 0.99, target update tau 0.005, residual policy a 2-layer MLP (256 units, GeLU activation), r_len initialized to zero-mean small variance (std 0.01) so early behavior matches the base chunk length. The residual Q-function is a 2-layer MLP (256 units, GeLU) with a shared trunk that concatenates state, r, and a length embedding (L encoded as [L, log L] then projected to 16 dims via a linear layer) before the final value head; total parameters ~ (state_dim + H*d + 1 + 16) * 256 + 256 * 256 + 256 ≈ 1.2M for H=10, d=4. Training uses one GPU (e.g., NVIDIA V100) with ~2 GPU-hours per 1e5 environment steps per task.

**Why this design.** We chose to embed length as a coordinate of the residual vector over a separate length adapter because a separate adapter (as in ExRL) cannot share credit assignment with action corrections, accepting the cost that the residual policy now outputs H*d+1 dimensions and needs more samples per update. We chose a multiplicative exponential parameterization for the length residual over an additive one because length is a positive, scale-sensitive quantity and exp(r_len) keeps the composition well-defined for any real residual, accepting that the critic must learn a log-scale length value. We chose a single residual Q-function over two critics because a shared critic is what forces the length and action residuals to be credited by the same return signal, accepting that the critic's input dimension grows with H and must be compressed by a shared trunk. A domain expert would not describe this as ExRL plus a residual policy, because ExRL's adapter and the action policy are trained as separate objects with separate objectives, whereas LARX's length coordinate is literally one entry of the residual action vector that the residual critic scores jointly; the closest prior method, ExRL, has no residual action space at all.

**Why this measures what we claim.** The augmented residual vector r = [r_len, r_a] measures compositional residual adaptation because the shared residual critic Q(s, r, L) is a single value function over the joint residual, which is equivalent to compositional credit only under the assumption that the return decomposes additively over the length and action coordinates given the state; this assumption fails when length changes the effective horizon of the return (e.g., executing a chunk for 3 vs 8 steps changes which future states are reachable), in which case Q(s, r, L) reflects a horizon-confounded value rather than a clean per-coordinate credit. The exponential length composition L = clip(round(L_base * exp(r_len))) measures the length residual in a scale-invariant space because log-length is the natural additive coordinate for a positive multiplicative quantity; this assumption fails when L_base is near the clip boundary, in which case the gradient through r_len is truncated and the length residual reflects a saturated rather than a free adaptation. The single replay buffer over (state, r, R, L, next_state) measures shared residual representation because both coordinates are stored and sampled as one transition; this assumption fails when the length coordinate's variance dominates the action coordinates' variance in the replay distribution, in which case the critic's shared trunk is effectively fit to length alone and the action residual receives diluted credit. The duration-aware target R + gamma^L * Q(next_state, r', L_next) measures the return over the variable-length execution because it explicitly discounts by the number of executed steps L; this assumption fails when L is misestimated or when the environment's reward is not per-step (e.g., sparse terminal reward), in which case the target reflects a mis-scaled return and the critic's credit to r_len is biased.

## Contribution

(1) A residual off-policy RL formulation in which execution length is a coordinate of the residual action space, yielding a single residual policy and single residual critic that jointly adapt length and action content. (2) A design principle: compositional residual adaptation across heterogeneous quantities is achievable when those quantities are embedded in one residual vector space and scored by one value function, rather than coordinated by separate adapters. (3) An open-source implementation and evaluation protocol for length-augmented residual adaptation on long-chunk manipulation policies.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | Meta-World long-horizon tasks (e.g., pick-place, push, shelf-place) | Long chunks expose length/action coupling |
| Primary metric | Adaptation success rate vs. samples (measured every 1e4 environment steps up to 1e5) | Directly measures sample-efficient residual adaptation |
| Baseline | ExRL (length adapter) | Tests whether joint credit beats separate adapter |
| Baseline | Standard RL chunk fine-tuning | Tests whether residual beats full-chunk adaptation |
| Baseline | Fixed execution length (L = H/2) | Tests whether length adaptation matters at all |
| Ablation-of-ours | LARX with additive length residual (L = clip(round(L_base + r_len))) | Tests multiplicative log-length parameterization |
| Diagnostic | Counterfactual residual ablation: for a fixed state, perturb r_len by ±0.5 and measure change in Q(s, r, L) vs. perturbing a single action residual coordinate by ±0.5; report ratio of Q-sensitivity to length vs. action | Tests the additive-credit assumption (A) directly |

### Why this setup validates the claim

The central claim is that embedding length as one coordinate of a shared residual vector yields better sample-efficient adaptation than treating length and actions as separate objects. ExRL isolates this sub-claim: it also adapts execution length, but via a separate adapter with its own objective, so any gap between LARX and ExRL is attributable specifically to joint credit assignment rather than to length adaptation existing at all. Standard RL chunk fine-tuning tests the complementary sub-claim that a residual over a frozen long-chunk base beats adapting the full chunk, isolating the value of the residual formulation. The fixed-length baseline is the floor: if LARX does not beat it, length adaptation contributes nothing and the joint-credit story is moot. The additive-length ablation is the sharpest test of the exponential parameterization: it changes only how r_len composes, so a gap there specifically implicates the log-scale length coordinate rather than the shared critic. The counterfactual residual-ablation diagnostic directly probes the load-bearing assumption: if Q(s, r, L) is sensitive to r_len in a way that is not explained by the action residuals (i.e., the ratio of Q-sensitivity to length vs. action is near zero or unstable), then the shared critic is not assigning credit to length, and the compositional-credit claim is unsupported. Success rate as a function of adaptation samples is the right metric because the claim is about sample efficiency of residual adaptation, not asymptotic capability; a method that reaches the same ceiling with fewer samples is exactly what the shared-credit mechanism predicts, and a metric like final return would not distinguish sample efficiency from ceiling.

### Expected outcome and causal chain

**vs. ExRL (length adapter)** — On a task where the optimal execution length must shrink mid-episode while action corrections also change (e.g., a reaching-then-grasping phase transition), ExRL's separate adapter receives length credit from a length-only signal and cannot propagate the action-correction consequence back into the length decision, so it oscillates between long chunks that overshoot and short chunks that under-reach. Our method instead scores the joint residual with one critic, so the return that penalizes overshoot also credits the length coordinate that caused it, and we expect a visible success-rate gap concentrated on phase-transition tasks with parity on single-phase tasks.

**vs. Standard RL chunk fine-tuning** — On a long-horizon task with a strong pretrained base chunk, full-chunk fine-tuning must relearn H*d action dimensions from scratch, so early updates destroy the base behavior and success rate dips before recovering, requiring many more samples. Our method instead keeps the base frozen and learns only a residual, so early behavior stays near the base chunk (r_len initialized near zero) and updates perturb rather than overwrite, and we expect LARX to match the base's initial success rate while full-chunk fine-tuning shows a pronounced early dip and a slower climb.

**vs. Fixed execution length** — On a task where the required chunk length genuinely varies across states (e.g., free-space transit vs. contact-rich manipulation), a fixed length either commits too long in contact phases or too short in transit, so the baseline plateaus below the achievable success rate regardless of samples. Our method instead adapts L through r_len, so the composed chunk shortens in contact and lengthens in transit, and we expect a success-rate gap that grows with the heterogeneity of optimal length across the task's state distribution, near-zero on homogeneous-length tasks.

**vs. LARX with additive length residual** — On a task where L_base is far from the optimal length, an additive residual must traverse a large absolute distance in length units, and gradient magnitude in that coordinate is poorly scaled relative to action residuals, so adaptation is slow or clips. Our method instead parameterizes length multiplicatively in log space, so a fixed r_len step produces a proportional length change regardless of L_base, and we expect the additive ablation to lag specifically on tasks whose optimal length deviates strongly from L_base, with parity when optimal length is near L_base.

**vs. Counterfactual residual-ablation diagnostic** — If the shared critic assigns credit to length, perturbing r_len should produce a Q-change comparable to perturbing an action residual coordinate of similar magnitude, and the ratio should be stable across states. If the ratio is near zero or highly variable, the critic is not crediting length, and the compositional-credit claim is falsified.

### What would falsify this idea

If LARX's advantage over ExRL and over the additive ablation is uniform across all task subsets rather than concentrated on phase-transition and large-length-deviation tasks where the shared-credit and log-scale mechanisms are predicted to matter, then the joint-coordinate story is wrong and the gain comes from something else. Likewise, if the fixed-length baseline matches LARX everywhere, length adaptation is not the source of any benefit and the central claim collapses. Additionally, if the counterfactual residual-ablation diagnostic shows that Q(s, r, L) is insensitive to r_len or that the sensitivity ratio is unstable, the load-bearing assumption of joint credit is directly falsified.

## References

1. ExRL: Sample-Efficient Online Adaptation of Long-Chunk Policies via Execution Length Control
