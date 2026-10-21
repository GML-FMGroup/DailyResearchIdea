# Capacity-Matched Verifiable Latent States for Looped Transformers

## Motivation

Looped/recurrent transformers such as PAR and Coconut feed a single fixed-dimensional continuous hidden state back through iterations, assuming that one vector can encode the entire reasoning prefix for arbitrarily many operations. Neither method measures when this sufficiency assumption breaks: PAR supervises iteration-wise prefix states but never tests whether the latent can actually be decoded back to the discrete prefix state, and Coconut feeds the last hidden state forward without any reconstruction check. The structural gap across these paper-trees is that reasoning-state sufficiency is assumed rather than made a capacity-matched, verifiable property, so the exact prefix length at which a fixed-width latent silently compresses (and thus fails to length-generalize) is never exposed.

## Key Insight

If the latent state at iteration k is required to be losslessly decodable to the discrete prefix state through an invertible codebook whose dimension grows with the number of distinguishable prefix states, then latent sufficiency becomes a measurable invariant and capacity expansion is triggered exactly at the prefix length where the single-vector assumption breaks, rather than being silently violated.

## Method

**What it is.** CAPS (Capacity-Aware Prefix-State coding) is a training procedure for a looped/recurrent transformer in which each recurrent latent s_k at iteration k is passed through a learned invertible codebook decoder D_θ that must reconstruct the discrete prefix state p_k (the state after the first k operations). A prefix-state reconstruction probe continuously measures the reconstruction error; when it exceeds a threshold τ=1e-3, the latent width is expanded on a fixed capacity-expansion schedule, so latent dimension tracks the number of distinct prefix states rather than sequence length.

**How it works.**

```pseudocode
Input: backbone B (Qwen2.5-0.5B, looped), training seqs of <=32 ops, discrete prefix labels p_1..p_T
Init: latent width d_0 = 256, codebook C_0 = {d_0-dim vectors}, invertible decoder D_theta (normalizing-flow head with 4 coupling layers, each a 2-layer MLP with hidden=512, tanh activation), capacity schedule d_{j+1} = d_j * 2, threshold tau = 1e-3

for each training step:
    s_0 = B.encode(x)
    for k = 1..T:
        s_k = B.recurrent_step(s_{k-1})            # norm-anchored to first-iteration norm
        p_hat_k = D_theta(s_k)                      # invertible decode to discrete prefix state
        L_rec  += CE(p_hat_k, p_k)                  # reconstruction loss (PAR-style prefix label)
        L_task += CE(B.readout(s_k), y_k)           # progress-aligned task loss
    # capacity probe every N=200 steps:
    if mean_recon_error(probe_set) > tau:
        expand latent width d -> 2d via D_theta re-parameterization (schedule step j+1)
        freeze old codebook rows, initialize new rows orthogonally
    update (B, D_theta) with L = L_rec + L_task

# sufficiency certificate: report the smallest k* where reconstruction error first exceeds tau
# under the *final* width d_J; this k* is the measured failure prefix length.
```

**Why this design.** We chose an *invertible* codebook decoder over a discriminative probe because a discriminative head can score well while discarding information, so only invertibility (a normalizing-flow head with exact inverse) certifies that the latent contains the prefix state rather than merely correlates with it; the cost is that flow heads are harder to train and add parameters. We chose a *capacity-expansion schedule tied to reconstruction error* over a fixed large latent because a fixed width either wastes capacity or silently caps the representable prefix set; the cost is a non-stationary architecture that requires re-parameterizing the decoder at expansion. We chose *discrete prefix-state labels* (available from PAR's progress-aligned supervision) over self-supervised reconstruction because only discrete labels make the target set countable, allowing dimension to be matched to the number of distinct prefix states; the cost is dependence on PAR-style label availability, mitigated by prefix self-distillation. This is not a variant of Coconut (which feeds a latent forward with no reconstruction check) nor of PAR (which supervises progress but never tests latent sufficiency): CAPS adds the measurable invariant and the capacity coupling that neither has.

**Why it measures what we claim.** The reconstruction error of D_θ(s_k) against p_k measures *latent sufficiency* because the invertibility of D_θ makes reconstruction error zero iff s_k contains all information in p_k; this assumption fails when the flow head is under-trained (error stays high even for a sufficient latent), in which case the metric reflects decoder optimization rather than representation capacity. The first prefix length k* at which error exceeds τ measures the *break point of the single-vector assumption* because the capacity schedule guarantees the latent width is matched to the count of distinguishable prefix states up to k*; this assumption fails when two distinct prefix states collide in the discrete label space (aliasing), in which case k* reflects label ambiguity rather than capacity. The capacity-expansion trigger measures *capacity need* because expansion is invoked only when reconstruction fails at the current width; this fails when the probe set is unrepresentative, in which case expansion reflects probe distribution rather than true representational demand.

**Decoder-capacity control (to separate optimization from capacity).** To verify that reconstruction error reflects representational capacity rather than flow-head under-training, we train a separate copy of D_θ (same architecture) to convergence on a held-out set of latents that are *known to be sufficient* (obtained by running the backbone on sequences of length ≤32, where the latent width d_0=256 is provably sufficient for the number of distinct prefix states). We then measure the reconstruction error of this converged decoder on the same probe set. If the converged decoder achieves error < τ on the probe set, the original decoder's error above τ is attributed to capacity limitation; if the converged decoder also fails, the probe is deemed unreliable and expansion is not triggered. This control is run before each capacity-expansion decision.

## Contribution

(1) A capacity-matched, verifiable latent-state framework (CAPS) that couples recurrent latent width to the number of distinguishable discrete prefix states via an invertible codebook decoder and a reconstruction-triggered capacity-expansion schedule. (2) A prefix-state reconstruction probe that yields a measurable sufficiency certificate — the first prefix length k* at which a fixed-width latent fails — turning the single-vector sufficiency assumption into an enforced invariant. (3) An empirical demonstration that capacity-matched latents expose and repair the exact prefix length at which looped transformers silently compress, providing a diagnostic absent from PAR and Coconut.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | swap-S_5 state tracking, train ≤32 moves, test to 512 | Directly probes iterative state composition and length extrapolation |
| Dataset | parity (sequence of bits, prefix state = running parity) and modular arithmetic (mod 7 addition, prefix state = running sum mod 7) | Non-synthetic algorithmic tasks with countable prefix states to test transfer of k* measurement |
| Primary metric | Full-state accuracy vs. prefix length | Detects the predicted break point k* and post-break degradation |
| Secondary metric | Reconstruction error of D_θ(s_k) vs. p_k, and linear/MLP probe accuracy and mutual-information estimate between s_k and p_k | Independent measurement of latent sufficiency to validate the flow-head error |
| Baseline | PAR (progress-aligned recurrence) | Tests whether latent sufficiency, not just progress supervision, drives length generalization |
| Baseline | Coconut (continuous latent, no reconstruction) | Tests whether an invertible reconstruction check is necessary for sufficiency |
| Baseline | Looped Transformer (adaptive steps) | Tests whether capacity coupling beats fixed-width looping on unseen lengths |
| Ablation-of-ours | CAPS without capacity expansion (fixed d=256) | Isolates the effect of capacity tracking prefix-state count |
| Ablation-of-ours | CAPS with decoder-capacity control disabled (expansion triggered by raw reconstruction error without the converged-decoder check) | Tests whether the control prevents spurious expansions due to decoder under-training |

### Why this setup validates the claim

The central claim is that length generalization fails when the recurrent latent's capacity is not matched to the number of distinct prefix states, and that an invertible reconstruction probe can detect and trigger capacity expansion at the right point. swap-S_5 with training capped at 32 moves and evaluation stretched to 512 moves creates a long tail of prefix states whose count grows with sequence length, so any capacity ceiling should manifest as a measurable break point. The parity and modular arithmetic tasks provide additional testbeds where the prefix state is a simple countable variable (parity bit, sum mod 7), allowing us to check whether the measured break point k* transfers beyond the permutation-state setting. The secondary metrics—linear/MLP probe accuracy and mutual-information estimate between s_k and p_k—provide an independent measurement of latent sufficiency that does not rely on the flow head, so we can check whether reconstruction error and information content diverge (which would indicate decoder optimization artifacts). PAR tests whether progress supervision alone is sufficient: if PAR's latent already contains prefix state, adding invertibility and capacity coupling should add little; if PAR's latent is merely correlated with progress, CAPS should pull ahead precisely where the prefix-state count exceeds a fixed width. Coconut tests whether the reconstruction check is doing work: without it, latents can be sufficient-looking yet lossy, so we expect Coconut to degrade earlier and more steeply than CAPS. The looped-transformer baseline with adaptive steps tests whether simply adding compute steps at test time can compensate for a fixed latent width; if capacity, not step count, is the binding constraint, adaptive looping should plateau while CAPS continues to improve. The fixed-width ablation is the critical internal control: if CAPS's gains come from capacity expansion, removing expansion should reproduce the fixed-width failure mode even though the reconstruction loss and invertible decoder are still present. The decoder-capacity-control ablation tests whether the expansion trigger is confounded by decoder under-training: if the control is necessary, disabling it should cause spurious expansions (and thus no improvement in k*) when the flow head is not yet converged. Full-state accuracy is the right primary metric because it is a strict, all-or-nothing measure of whether the latent carries the complete prefix state; a partial-credit metric would obscure the predicted break point. Together these conditions make the claim falsifiable: the method must show a break point k* that tracks capacity expansion, its advantage over PAR must be concentrated beyond the prefix lengths where PAR's fixed latent is predictably saturated, and the independent probe/MI measurements must agree with the flow-head reconstruction error.

### Expected outcome and causal chain

**vs. PAR (progress-aligned recurrence)** — On a swap-S_5 sequence of, say, 200 moves, PAR's recurrent latent is supervised only to encode progress toward the final answer, so it can compress a long prefix into a single vector that tracks progress but discards the exact permutation state needed to continue swapping correctly. Our method instead forces each latent s_k through an invertible decoder to reconstruct the discrete prefix state p_k, so the latent cannot drop state information without incurring reconstruction error, and the capacity probe expands the latent width whenever the number of distinct prefix states outgrows the current width. We expect PAR and CAPS to be close on sequences up to roughly the training length and modest extrapolations, then a clear divergence at the prefix lengths where PAR's fixed latent saturates—visible as a sharp drop in full-state accuracy for PAR while CAPS's accuracy declines more gradually and its measured k* sits beyond PAR's saturation point. On parity and modular arithmetic, we expect a similar pattern: PAR's latent will fail to track the running parity or sum beyond a certain length, while CAPS's k* will be measurable and will shift with capacity expansion.

**vs. Coconut (continuous latent, no reconstruction)** — On a long swap-S_5 instance, Coconut feeds a latent vector forward through recurrent steps with only final-answer task loss, so the latent can drift into a representation that is predictive of the answer but not invertible back to the exact prefix state; once two distinct prefix states map to nearby latents, the recurrent dynamics cannot distinguish them and the trajectory commits to the wrong permutation. Our method instead applies an invertible flow decoder with exact inverse at every step and a reconstruction loss against the discrete prefix label, so any collapse of two prefix states into one latent is penalized directly, and the capacity schedule widens the latent before such collisions become unavoidable. We expect Coconut to fail earlier and more abruptly than CAPS on the longest test lengths, with CAPS showing a measurable gap that widens with sequence length rather than a uniform advantage across all lengths. The independent probe/MI measurements should show that Coconut's latents have lower mutual information with the prefix state even when task accuracy is high, confirming that the reconstruction check is necessary.

**vs. Looped Transformer (adaptive steps)** — On a swap-S_5 sequence well beyond training length, a looped transformer with adaptive test-time steps can apply more recurrent iterations, but each iteration operates in a latent space whose width was fixed at training time; once the number of distinct prefix states exceeds what that width can represent, extra steps only cycle through a lossy encoding and cannot recover the missing state, so accuracy plateaus or degrades with additional steps. Our method instead couples latent width to the reconstruction-error probe, expanding capacity when the probe shows the current width cannot reconstruct the prefix state, so the representable prefix set grows with the task's demand rather than with step count. We expect adaptive-step looping to show diminishing returns and a flat or falling accuracy curve at large lengths, while CAPS continues to improve as capacity expands, with the gap concentrated at the lengths where the fixed width is provably insufficient.

**vs. CAPS without capacity expansion (fixed d=256)** — On a long swap-S_5 sequence, the fixed-width variant has the same invertible decoder and reconstruction loss as full CAPS, but its latent width never grows, so once the number of distinct prefix states exceeds 256 dimensions' worth of distinguishable codes, the reconstruction error rises above τ and cannot be reduced by further training. Full CAPS instead triggers a capacity expansion at that point, freezes the old codebook rows, and initializes new rows orthogonally, restoring the reconstruction error below τ and allowing correct prefix tracking to continue. We expect the fixed-width variant to reproduce the PAR-like failure pattern despite having the reconstruction probe, while full CAPS shows a break point k* that shifts rightward as capacity expands—demonstrating that the gain comes from capacity coupling, not merely from the invertible probe. The independent probe/MI measurements should show that the fixed-width variant's latents lose information about the prefix state at the same point where reconstruction error rises, confirming that the failure is due to capacity, not decoder optimization.

**vs. CAPS with decoder-capacity control disabled** — On a long swap-S_5 sequence, the variant without the converged-decoder check may trigger expansions prematurely if the flow head is under-trained, leading to unnecessary capacity increases and a k* that does not reflect true representational demand. We expect this variant to show a noisier expansion pattern and a k* that is less stable across runs, while full CAPS (with the control) shows expansions only when the converged decoder also fails, resulting in a k* that reliably tracks the prefix-state count. This ablation tests whether the control is necessary to make the sufficiency metric identifiable.

### What would falsify this idea

If CAPS's advantage over PAR, Coconut, and adaptive-step looping is uniform across all prefix lengths rather than concentrated beyond the lengths where the fixed-width ablation and the baselines' latents are predicted to saturate, or if the measured break point k* does not shift with capacity expansion, or if the independent probe/MI measurements diverge from the flow-head reconstruction error (indicating that the latter reflects decoder optimization rather than capacity), then the claim that capacity matching to prefix-state count causes the length generalization gain is wrong.

## References

1. PAR: Progress-Aligned Recurrence for Length Generalization in State Tracking
2. Training Large Language Models to Reason in a Continuous Latent Space
3. Looped Transformers for Length Generalization
4. Guiding Language Model Reasoning with Planning Tokens
5. Think before you speak: Training Language Models With Pause Tokens
6. Looped Transformers as Programmable Computers
7. Teaching Algorithmic Reasoning via In-context Learning
8. Transformers Learn Shortcuts to Automata
