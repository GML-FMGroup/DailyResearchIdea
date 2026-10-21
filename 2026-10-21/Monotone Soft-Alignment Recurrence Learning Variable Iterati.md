# Monotone Soft-Alignment Recurrence: Learning Variable Iteration-to-Operation Correspondence for Length Generalization

## Motivation

Progress-aligned recurrence (PAR) supervises recurrent iteration k to match the prefix state after exactly k discrete operations, and every branch of this line—looped Transformers with adaptive steps, RASP-L iterative decompositions, and PAR itself—assumes a fixed one-to-one correspondence between computation steps and task operations. This assumption is structurally false for real state-tracking tasks where a single move may apply multiple swaps, a step may be a no-op, or operations merge, so the supervision signal becomes misaligned and the model is forced to learn a decomposition it was never given. The meta-level gap is that iteration-to-operation identifiability is assumed rather than established; without a mechanism that can recover a variable, monotone correspondence, progress alignment is undefined outside the narrow uniform-decomposition regime PAR was demonstrated on.

## Key Insight

Constraining the iteration-to-operation map to be monotone and surjective makes the alignment identifiable up to a global time reparameterization, so the model can discover how many discrete operations each recurrent iteration advances instead of being told.

## Method

**What it is.** MASR (Monotone-Aligned Soft Recurrence) is a training procedure for a looped/recurrent Transformer that jointly learns (i) a recurrent state trajectory over K iterations and (ii) a monotone, surjective soft alignment between iterations and a variable-length sequence of discrete operations. Inputs: an operation sequence of length T (variable per example) and its prefix states; outputs: per-iteration predictions supervised through a soft-DTW alignment cost, plus a learned completion readout.

**How it works.**

```pseudocode
# Given: operation sequence o_1..o_T, prefix states s_0..s_T, backbone f_theta, max iterations K
# Latent operation-count variable: a_k = number of ops consumed by iteration k, a_k >= 1, sum_k a_k = T
# Alignment cost uses a frozen target embedding phi (e.g., a fixed random projection or pretrained encoder) to break circularity:
#   d(p_k, s_t) = || phi(p_k) - phi(s_t) ||_2^2, with phi frozen (no gradients)
# Soft-DTW hyperparameters: gamma = 0.1, band width w = max(1, ceil(T/K)) (adaptive, not fixed)
# Warm-up: for first 1000 steps, train only L_align (no L_pred, no L_completion)

for each training example:
    h_0 = embed(initial_state)
    for k in 1..K:
        h_k = f_theta(h_k-1, o_prefix_so_far)          # recurrent step
        h_k = h_k * (||h_1|| / ||h_k||)                  # norm anchoring (as in PAR)
        p_k = readout(h_k)                               # predicted state after iteration k

    # Soft-DTW alignment over the cumulative operation-count variable
    # Cost matrix C[k, t] = d(p_k, s_t) with d computed in frozen phi space (p_k detached in cost term)
    # Monotone path from (1,0) to (K,T) with steps (1,a_k), a_k >= 1  -> surjective onto ops
    A = softdtw(C, gamma=0.1, band=w)                   # differentiable, monotone, surjective
    L_align = A(1, T)                                    # soft-DTW cost
    L_pred  = sum_k  sum_t  A_grad[k,t] * d(p_k, s_t)   # expected prediction loss under alignment posterior
    L = L_align + L_pred + L_completion
    update theta by backprop through L

# Inference: run K iterations; completion readout selects stopping iteration
```

The alignment posterior A_grad[k,t] (the soft-DTW gradient) is the learned distribution over which operation index iteration k corresponds to; because the path is monotone and surjective, each operation is assigned to exactly one iteration and iteration order matches operation order. The latent operation-count variable a_k is recovered as the expected step size of the optimal soft path.

**Why this design.** We chose a soft-DTW alignment over a hard one-to-one assignment because the hard assignment (PAR's supervision) is only defined when T = K; soft-DTW admits variable a_k while remaining differentiable, accepting the cost that alignment is approximate and requires a band width w to keep the DP tractable. We chose a monotone-and-surjective path constraint over an unconstrained alignment because monotonicity plus surjectivity is exactly the condition that makes the map identifiable up to global time reparameterization (a classical DTW property), accepting the cost that the model cannot represent non-monotone operation reorderings. We chose to train the alignment jointly with the recurrent state rather than in a separate EM-style phase because the state predictions and the alignment are mutually informative—better states sharpen the alignment and vice versa—accepting the cost that joint optimization can stall in a degenerate alignment early in training, which we mitigate with the norm anchoring and a warm-up on the alignment cost alone. A domain expert would not describe MASR as a variant of PAR because PAR fixes the alignment by construction; MASR makes the alignment a learned latent variable with an identifiability constraint, which is a different object.

**Why it measures what we claim.** The soft-DTW alignment cost measures iteration-to-operation correspondence because monotone-surjective path constraints make the optimal path equivalent to a valid variable-rate alignment under the assumption that operation order is preserved in iteration order; this assumption fails when the task interleaves operations that the recurrent state must process out of order, in which case the alignment reflects the best monotone approximation rather than the true correspondence. The alignment posterior A_grad[k,t] measures how many discrete operations iteration k advances because the expected step size of the optimal soft path equals the latent operation-count a_k under the assumption that prefix-state distances are minimized at the true correspondence; this assumption fails when two distinct prefix states are close in representation space, in which case the posterior smears across operation indices and the recovered a_k is an average over ambiguous alignments. The norm-anchored state p_k measures progress because rescaling to first-iteration norm keeps the distance metric comparable across iterations under the assumption that state scale is not itself informative; this fails when the task encodes progress in state magnitude, in which case anchoring removes a genuine signal.

**Identifiability and calibration (explicit assumption and verification).** The load-bearing assumption is that the soft-DTW posterior A_grad[k,t] is a faithful, identifiable estimate of the true iteration-to-operation correspondence, i.e., the prefix-state distance d(p_k, s_t) is minimized at the true correspondence so that the expected step size of the optimal soft path recovers the latent operation count a_k. This assumption is not automatic: soft-DTW is a smoothed min over paths, and its gradient is a Boltzmann-weighted posterior over all near-optimal monotone paths; when the cost landscape is flat or multimodal, the posterior smears and the expected step size is an average over ambiguous alignments, not the true a_k. To break the circularity that the cost matrix C[k,t] = d(p_k, s_t) is itself a function of the learned backbone f_theta, we compute d in a frozen target embedding phi (e.g., a fixed random projection of dimension 64) and detach p_k in the cost term, so the alignment cost depends on a quantity not co-trained to satisfy it. We add an explicit entropy regularizer on the soft-DTW posterior: L_ent = -lambda * H(A_grad), with lambda = 0.01, to encourage peaked alignments. During training we monitor per-example posterior entropy and the gap between best and second-best path cost; we trust the recovered a_k only where the alignment is peaked (entropy below a threshold of 0.5 nats).

## Contribution

(1) A training framework, MASR, that replaces PAR's fixed one-iteration-equals-one-swap supervision with a differentiable monotone-surjective soft-DTW alignment over a latent operation-count variable, trained jointly with the recurrent state. (2) A design principle: monotone-surjective alignment constraints are what make iteration-to-operation correspondence identifiable under variable operation counts, yielding a learned invariant rather than a task assumption. (3) A procedure for recovering per-iteration operation counts from the alignment posterior, enabling diagnosis of how a looped model allocates computation across a variable-length operation sequence.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Swap-S_5 state tracking, train ≤32 moves, test to 512 | Tests length generalization 16x beyond training length |
| Dataset | No-op and multi-swap variant of Swap-S_5 (train ≤32, test to 512) | Tests variable operation counts per iteration |
| Primary metric | Full-state accuracy at each test length | Detects degradation profile, not just endpoint |
| Secondary metric | Alignment NLL and expected calibration error (ECE) of A_grad vs. true alignment on synthetic data with known correspondence | Quantifies posterior calibration |
| Secondary metric | Mean absolute error of recovered a_k vs. ground-truth operation counts | Tests latent operation-count recovery |
| Baseline | PAR (hard progress-aligned recurrence) | Strong fixed-alignment reference for the same backbone |
| Baseline | RLT (recurrent latent transformer) | Tests whether recurrence alone explains any gain |
| Baseline | Final-answer-supervised looped Transformer | Isolates value of intermediate alignment supervision |
| Baseline | Unconstrained soft-DTW alignment (no monotone-surjective constraint) | Tests whether the constraint is doing non-trivial work |
| Baseline | CTC-style learned alignment (HMM) | Alternative latent-alignment scheme |
| Ablation-of-ours | MASR with norm anchoring removed | Tests whether norm anchoring carries the alignment |
| Ablation-of-ours | MASR with frozen target embedding phi replaced by co-trained embedding | Tests circularity mitigation |
| Ablation-of-ours | MASR with entropy regularizer removed | Tests whether peakedness regularizer is necessary |

### Why this setup validates the claim

The central claim is that making the iteration-to-operation alignment a learned, monotone-surjective latent variable—rather than a fixed one-to-one assignment—lets a looped Transformer track state at lengths far beyond training. Swap-S_5 is the right testbed because it has a known iterative solution whose operation order is genuinely sequential, so a monotone alignment is a valid inductive bias and any failure of that bias is visible. The no-op and multi-swap variant directly violates the uniform-decomposition assumption, testing whether MASR recovers variable operation counts. Full-state accuracy as a function of length, rather than a single endpoint number, is the right metric because the claim predicts a specific *shape*: near-parity at short lengths and a widening gap only once the fixed-alignment baselines run out of iterations. PAR tests the sub-claim that a learned variable-rate alignment beats a hard fixed assignment; RLT tests whether recurrence alone, without alignment supervision, suffices; final-answer supervision tests whether the gain comes from intermediate prefix-state targets rather than from looping. The unconstrained soft-DTW and CTC baselines test whether the monotone-surjective constraint is necessary. The norm-anchoring ablation is the diagnostic: if removing it collapses the gain, the mechanism is the comparability of distances across iterations, not merely extra supervision. The alignment NLL and ECE metrics directly test the load-bearing assumption that A_grad is a calibrated posterior; the a_k recovery error tests whether the expected step size recovers the latent operation count.

### Expected outcome and causal chain

**vs. PAR (hard progress-aligned recurrence)** — On a 512-move sequence where the number of operations per iteration must grow beyond the training-time rate, PAR produces wrong final states because its one-to-one alignment is only defined when iterations equal operations, so it must either fix K = T or mis-assign operations to iterations. Our method instead lets the soft-DTW path absorb the extra operations into larger expected step sizes a_k, keeping each iteration's prediction matched to the correct prefix state. We expect near-parity below ~64 moves and a growing accuracy gap at 128–512 moves, concentrated on the longest sequences.

**vs. RLT (recurrent latent transformer)** — On a case where the recurrent state must advance by a variable number of operations before the next useful update, RLT produces drift because its iterations have no supervision tying them to operation indices, so the latent trajectory is free to encode whatever minimizes the final loss. Our method instead anchors each iteration to a prefix state through the alignment posterior, so the trajectory is pinned to operation order. We expect RLT to degrade smoothly with length while MASR holds a plateau, with the gap largest on mid-length inputs where alignment ambiguity first appears.

**vs. Final-answer-supervised looped Transformer** — On a case where an intermediate state is ambiguous but the final answer is still reachable by several paths, the final-answer baseline produces a state trajectory that is correct at the endpoint but wrong at intermediate iterations, because it receives no signal about which operation each iteration should have consumed. Our method instead receives per-iteration prefix-state supervision through the soft-DTW cost, so intermediate predictions are pinned. We expect the gap to appear in full-state accuracy at intermediate lengths even when final-answer accuracy is comparable, i.e. a diagnostic dissociation between endpoint and trajectory accuracy.

**vs. Unconstrained soft-DTW alignment** — Without the monotone-surjective constraint, the alignment can assign operations out of order or skip operations, so the recovered a_k is not a valid operation count and the state predictions are supervised against wrong prefix states. We expect unconstrained soft-DTW to underperform MASR, with the gap increasing on the no-op and multi-swap variant where the constraint is most informative.

**vs. CTC-style learned alignment (HMM)** — CTC assumes a hidden Markov model with a fixed transition structure and marginalizes over alignments; it does not enforce surjectivity onto operations and may collapse multiple operations to the same iteration. We expect MASR to recover a_k more accurately (lower MAE) and to achieve higher full-state accuracy on the no-op variant, where CTC's blank symbol handling may misalign.

### What would falsify this idea

If MASR's gain over PAR is uniform across all lengths rather than concentrated at lengths where the fixed alignment breaks, or if removing norm anchoring leaves accuracy unchanged while the alignment posterior still smears, or if the alignment NLL and ECE metrics show that A_grad is poorly calibrated (e.g., ECE > 0.1) and the a_k recovery error is high even on synthetic data with known correspondence, then the claimed causal mechanism—learned variable-rate alignment plus comparable cross-iteration distances—is not what produces the result and the central claim is wrong.

## References

1. PAR: Progress-Aligned Recurrence for Length Generalization in State Tracking
2. Training Large Language Models to Reason in a Continuous Latent Space
3. Looped Transformers for Length Generalization
4. Guiding Language Model Reasoning with Planning Tokens
5. Think before you speak: Training Language Models With Pause Tokens
6. Looped Transformers as Programmable Computers
7. Teaching Algorithmic Reasoning via In-context Learning
8. Transformers Learn Shortcuts to Automata
