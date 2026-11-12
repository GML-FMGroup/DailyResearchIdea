# Runtime Factorization Reconfiguration for GPU-Accelerated QP in Model Predictive Control

## Motivation

FAME-QP hard-codes its operator bank at configuration time and relies on lazy updates, so it cannot adapt the factorization when the active-set structure or sparsity of the QP changes online; the solver therefore either stalls on stale operators or falls back to host-side recompilation. This is a concrete instance of a broader structural gap: existing GPU QP solvers treat the factorization pattern as an offline design choice rather than a runtime-adaptive object, which becomes binding in MPC settings where the active set evolves across time steps.

## Key Insight

The active-set transition graph of a QP is a finite, data-independent state space whose edges can be pre-synthesized as GPU kernel variants, so runtime factorization reconfiguration reduces to a graph walk over pre-compiled templates rather than a compilation problem.

## Method

**What it is.** FARA (Factorization-Aware Runtime Adaptation) is a runtime factorization reconfiguration layer for GPU QP solvers. It takes as input the current QP data (Hessian, constraint Jacobian, active-set estimate) and a stream of active-set transitions from the solver's iteration, and outputs a re-bound operator bank whose factorization kernels match the current sparsity and active-set structure without host-side recompilation.

**Load-bearing assumption (stated explicitly).** FARA assumes that the active-set transition graph of an MPC QP is a finite, data-independent state space dominated by a small set of high-frequency patterns, so a runtime-growing, signature-keyed template cache populated on first occurrence will cover most online transitions and amortize JIT cost to near zero. This assumption is not verified a priori; it is measured online by the cache hit rate and the cold-transition fraction, and the method is designed to degrade gracefully (via JIT fallback) if the transition distribution is heavy-tailed.

**How it works.**

```pseudocode
# Phase 1: Offline template synthesis (at install time, not per-QP)
for each canonical active-set pattern P in {P_1,...,P_K} induced by constraint graph G:
    emit parameterized CUDA kernel template T_P with slots for
        (sparsity mask, block sizes, pivot order, conditioning threshold)
    compile T_P once with nvcc -cubin; store in template bank B

# Phase 2: Online runtime loop (per MPC step, per solver iteration)
inputs: QP data (H, A, b), current active set S_t, template bank B,
        runtime-growing signature-keyed cache C (initially empty)

1. compute active-set transition signature sigma_t = (S_t \ S_{t-1}, S_{t-1} \ S_t)
2. look up candidate template T_P = C[hash(sigma_t)]   # O(1) hash into runtime cache
   if miss: T_P = B[hash(sigma_t)]  # fallback to pre-synthesized bank
3. if T_P is resident and its sparsity mask matches current H,A
   AND its pivot order satisfies the conditioning threshold (see step 5):
       rebind operator bank in place via cudaGraphExecKernelNodeSetParams
   else:
       trigger just-in-time specialization: patch T_P's parameter slots
       (sparsity mask, block sizes, pivot order) and instantiate a new cubin via NVRTC
       (JIT cost amortized over subsequent iterations); insert new cubin into C
4. launch fused factorization kernel with rebound operators
5. compute pivot-order conditioning estimate kappa_t = ||H_S||_2 * ||H_S^{-1}||_2
   (or a cheaper proxy: ratio of max/min diagonal of the factorized block)
   if kappa_t > kappa_max (1e8) OR residual ||r||_2 > tol_reconfig (1e-6):
       fall back to generic dense kernel from B and mark sigma_t as cold

output: updated operator bank O_t, factorization F_t, updated cache C
```

**Why this design.** We chose a runtime-growing, signature-keyed cache over a fixed top-K pre-synthesized bank because the active-set transition distribution in contact-rich MPC is heavy-tailed and not known a priori; the cache is populated on first occurrence and persists across MPC steps, so the amortization claim is tested empirically rather than assumed. The trade-off is that rare transitions must fall back to NVRTC JIT, accepting a one-time ~10ms cost that is amortized over subsequent iterations. We chose in-place cudaGraphExecKernelNodeSetParams over rebuilding CUDA graphs because FAME-QP's fused kernel iteration is replayed via CUDA graphs and graph rebuild costs ~100us, whereas node parameter patching costs ~5us; the trade-off is that only kernels with identical topology can be patched, so we constrain template slots to preserve launch geometry. We chose a hash-based O(1) lookup over a learned classifier because the transition signature is a deterministic function of the active-set difference and does not benefit from learned generalization; the trade-off is that hash collisions on rare signatures trigger unnecessary JIT, which we mitigate by a small collision-resolution table. A domain expert would not describe FARA as a variant of FAME-QP's cached operator bank because FAME-QP's bank is indexed by constraint set at configuration time and never updated online, whereas FARA's bank is indexed by active-set transition signatures and rebound in place at runtime.

**Why it measures what we claim.** The active-set transition signature sigma_t measures the structural change in the QP's factorization pattern because the assumption that the factorization pattern is determined by the active-set partition (a standard result in parametric QP sensitivity) makes sigma_t a sufficient statistic for the required operator rebind; this assumption fails when the Hessian's sparsity changes independently of the active set (e.g., a constraint becomes numerically inactive but not structurally removed), in which case sigma_t reflects only the combinatorial transition and the sparsity-mask check in step 3 catches the mismatch. The template-bank hit rate measures the fraction of transitions that avoid JIT compilation because the assumption that high-frequency transitions dominate the MPC trajectory makes the top-K templates cover most steps; this assumption fails in MPC problems with rapidly switching contact modes (e.g., legged locomotion), in which case the hit rate drops and the JIT fallback dominates latency. The residual check ||r||_2 > tol_reconfig measures whether the rebound factorization remains numerically faithful to the current QP because the assumption that in-place parameter patching preserves the factorization's numerical equivalence (up to rounding) makes the residual a direct proxy for reconfiguration correctness; this assumption fails when the patched kernel's pivot order becomes ill-conditioned, in which case the residual reflects numerical instability rather than structural mismatch and triggers an unnecessary fallback. The pivot-order conditioning estimate kappa_t measures whether the template's pivot order is numerically valid for the current Hessian values because the assumption that the active-set partition alone determines a well-conditioned factorization fails when the Hessian's conditioning varies within the same active-set pattern; in that case kappa_t catches the ill-conditioning before the residual check and triggers a pivot-order re-specialization rather than a generic fallback.

## Contribution

['A runtime factorization reconfiguration layer (FARA) that maintains a structure-aware operator bank indexed by active-set transition signatures and rebinds GPU factorization kernels in place via CUDA graph node parameter patching, eliminating host-side recompilation for online QP structure changes.', 'A design principle for GPU QP solvers: the active-set transition graph is a finite state space whose edges can be pre-synthesized as parameterized kernel templates, making runtime factorization adaptation a graph-walk problem rather than a compilation problem.', 'An open-source implementation of FARA integrated with FAME-QP, including the template synthesis pipeline and the NVRTC-based JIT fallback for cold transitions.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | Humanoid + quadrotor MPC traces (FAME-QP) | Contact-mode switches stress active-set transitions |
| Primary metric | Per-step solver latency (median + tail) | Reconfiguration cost is the claimed bottleneck |
| Secondary metric | Cache hit rate (fraction of transitions served from runtime cache) | Directly measures the load-bearing assumption |
| Secondary metric | Cold-transition fraction (fraction requiring NVRTC JIT) | Quantifies heavy-tail behavior |
| Baseline | FAME-QP cached operator bank | Static bank, no online rebind |
| Baseline | GPU interior-point / operator-splitting QP | Generic dense factorization path |
| Baseline | FARA without template bank (pure NVRTC JIT) | Isolates pre-synthesis amortization |
| Ablation-of-ours | FARA with residual check disabled | Tests whether fallback guards correctness |
| Ablation-of-ours | FARA with pivot-order conditioning check disabled | Tests whether conditioning check is load-bearing |

### Why this setup validates the claim

The central claim is that indexing a factorization operator bank by active-set transition signature and rebinding in place cuts per-iteration reconfiguration latency without sacrificing numerical fidelity, specifically on MPC trajectories where active sets switch frequently. FAME-QP's cached bank tests the sub-claim that configuration-time indexing is insufficient when the active set moves: if FARA does not beat it on switching-heavy segments, the online-rebind premise is dead. The generic GPU QP solver tests the sub-claim that template matching is worth the machinery at all — FARA must show a latency advantage that is not merely parity with a well-tuned dense path. The pure-JIT baseline isolates the amortization claim: it should be competitive on steady segments but blow up on the first occurrence of each transition, so the gap between it and full FARA is the value of pre-synthesis. The residual-disabled ablation tests whether the fallback is load-bearing for accuracy, not just latency. The pivot-order conditioning ablation tests whether the conditioning check prevents numerically invalid rebinds that the residual check alone might miss. Median and tail latency are the right metrics because the claim is about worst-case reconfiguration stalls in a control loop, where a single 10ms JIT spike can violate the control deadline even if the mean is fine. The cache hit rate and cold-transition fraction directly measure the load-bearing assumption: if the hit rate is low and the cold-transition fraction is high, the assumption that a small set of high-frequency patterns dominates is false for these traces, and the method's amortization claim is not supported.

### Expected outcome and causal chain

**vs. FAME-QP cached operator bank** — On a humanoid segment where the contact set switches (e.g., foot strike to double support), FAME-QP's bank is indexed at configuration time and has no entry matching the new active-set partition, so it falls back to a slower generic factorization or a stale operator, producing a latency spike at exactly the transition step. Our method looks up the transition signature, finds or patches a matching template, and rebinds in place, so we expect a visible latency gap concentrated on transition steps with near-parity on steady-contact steps.

**vs. GPU interior-point / operator-splitting QP** — On a quadrotor segment with a tight active-set structure, the generic solver re-forms and refactorizes a dense operator every iteration because it has no notion of structural reuse across active-set changes. Our method reuses the template's launch geometry and patches only the sparsity mask and block sizes, so we expect FARA to show lower per-iteration latency throughout, with the gap widening as the active-set churn rate increases.

**vs. FARA without template bank (pure NVRTC JIT)** — On a trajectory where a rare transition appears for the first time, the pure-JIT variant pays the full ~10ms specialization cost at that step, producing a single severe tail-latency outlier. Our method serves the common transitions from the runtime cache and only JITs cold signatures, so we expect the two to match on median latency but FARA to show a markedly lower tail, with the gap tracking the fraction of cold transitions.

**vs. FARA with residual check disabled** — Without the residual check, the method may accept a rebound factorization that is structurally correct but numerically inaccurate due to an ill-conditioned pivot order, leading to solver divergence or inaccurate MPC solutions. We expect the residual-disabled variant to show occasional large solution errors (measured as KKT residual or control performance degradation) on segments where the pivot-order conditioning is poor, while full FARA maintains accuracy.

**vs. FARA with pivot-order conditioning check disabled** — Without the conditioning check, the method may rebind a template whose pivot order is numerically invalid for the current Hessian values, even if the sparsity mask matches. We expect the conditioning-disabled variant to show increased solution errors and more frequent residual-check fallbacks, confirming that the conditioning check is load-bearing for numerical validity.

### What would falsify this idea

If FARA's latency advantage over FAME-QP is uniform across steady and switching segments rather than concentrated on active-set transition steps, the transition-signature mechanism is not the cause of the gain and the central claim is wrong. Likewise, if disabling the residual check or the pivot-order conditioning check produces no accuracy degradation on any segment, the corresponding fallback is not load-bearing and the correctness argument collapses.

## References

1. FAME-QP: A GPU-Accelerated Quadratic-Programming Solver for Real-Time Model Predictive Control
