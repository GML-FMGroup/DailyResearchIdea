# Dual-State-Preserving CUDA Graph Replay for Warm-Started Active-Set QP in MPC

## Motivation

FAME-QP caches operators and replays fixed CUDA graphs for dense QPs, but its graph replay cannot encode the branching active-set recursion that warm-starting requires, so the dual active-set state from the embedded solver is discarded between MPC iterations. This structural limitation—CUDA graphs capture a static kernel sequence while active-set transitions are data-dependent branches—means FAME-QP cannot preserve warm-start validity when the active set changes, forcing re-solves from scratch or invalid warm starts.

## Key Insight

The dual active-set recursion is a finite-state automaton whose transitions depend only on the current dual state and the QP data; encoding this automaton as a persistent device-resident state buffer with a branch-selection kernel yields a GPU-replayable warm start that is equivalent to the recursive solver's state evolution.

## Method

**What it is:** DS-Graph is a GPU execution framework for dual active-set QP solvers that preserves the dual-state recursion across CUDA graph replays. Its inputs are the QP data (H, g, A, b, C, d) and the previous iteration's dual state (working set W, multipliers λ, ν); its output is the updated dual state and primal solution, with the same active-set transitions as the recursive warm-start solver.

**Load-bearing assumption (explicit):** The dual active-set recursion is encoded as a single persistent cooperative kernel that runs the entire active-set loop on-device (with grid.sync() between iterations) and is launched once per MPC step via graph replay. The graph encodes one solve, not one transition. The fused branch-selection and active-set-step logic is inside this persistent kernel; the number of add/drop/solve iterations is data-dependent and resolved entirely on-device without host intervention. The persistent dual-state buffer is updated in-place and is race-free because grid.sync() separates the branch-selection phase from the active-set-step phase within each iteration.

**How it works:**
```pseudocode
# Persistent device buffers allocated once:
#   dual_state: {W (active set bitmask), λ, ν, x, residuals}
#   branch_flag: int (0 = no transition, 1 = add constraint, 2 = drop constraint)
#   next_kernel_id: int (0 = SOLVE, 1 = ADD, 2 = DROP)
#   done_flag: int
#   max_active_set_size: int (pre-registered, e.g., 512)

function DS_Graph_Replay(QP_data, persistent_dual_state):
    # Single persistent cooperative kernel launched once per MPC step via graph replay
    launch active_set_loop_kernel<<<blocks, threads, cooperative=True>>>(QP_data, persistent_dual_state)
        # Inside kernel: loop until done_flag:
        #   Phase 1: check_warmstart_kernel logic (inlined): compute branch_flag from KKT conditions
        #   grid.sync()
        #   Phase 2: branch_select_kernel logic (inlined): map branch_flag -> next_kernel_id
        #   grid.sync()
        #   Phase 3: active_set_step_kernel logic (inlined): switch(next_kernel_id) { ADD, DROP, SOLVE }
        #   grid.sync()
        #   Phase 4: convergence_kernel logic (inlined): set done_flag
        #   grid.sync()
        #   Phase 5: copy_dual_state (inlined): update persistent_dual_state in-place
        #   grid.sync()
    return persistent_dual_state
```
The graph is instantiated once with this single persistent kernel; the kernel contains the full active-set loop with grid.sync() between phases. The persistent dual-state buffer is updated in-place, so the next graph replay reads the warm-start state directly. The maximum active-set size is pre-registered (e.g., 512) and the buffer is allocated for that size; if the active set exceeds this, the kernel falls back to a cold-start path that reinitializes the working set.

**Why this design:** We chose a persistent device-resident dual-state buffer over host-side state storage because host-device transfers would break the graph replay's zero-copy assumption and add latency, accepting the cost that the buffer must be allocated for the maximum active-set size. We chose a single persistent cooperative kernel over separate kernels with conditional graph nodes because CUDA graphs do not support data-dependent branching between nodes without host intervention, and a persistent kernel with grid.sync() keeps the entire active-set loop on-device, accepting the cost that the kernel must use cooperative launch and may have lower occupancy due to the switch statement. We chose device-side dispatch (next_kernel_id) over host-side graph re-instantiation because re-instantiation would negate the launch-overhead benefit of graph replay, accepting the cost that the kernel's control flow is less transparent for profiling. Unlike FAME-QP's fixed operator bank, which assumes problem structure changes infrequently, DS-Graph explicitly handles active-set transitions at every iteration, which is the structural property that FAME-QP's static graph cannot provide.

**Why it measures what we claim:** The persistent dual-state buffer measures warm-start validity because the dual state (working set and multipliers) is the sufficient statistic for the active-set recursion—by the recursive warm-start theorem, the next active set depends only on the current dual state and QP data; this assumption fails when the QP data changes so drastically that the previous working set is infeasible, in which case the buffer reflects a stale warm start that triggers a fallback to cold start. The branch_flag measures active-set transition necessity because it is computed from the KKT conditions of the current dual state; this assumption fails when numerical tolerances cause false positives, in which case the flag reflects a conservative transition that may add an unnecessary iteration. The next_kernel_id measures the correct active-set operation because it is derived from the sign of the constraint violation; this assumption fails when multiple constraints violate simultaneously, in which case the kernel selects the most violated one, reflecting a greedy transition rather than a simultaneous update. The in-place device update is race-free because grid.sync() separates the branch-selection phase from the active-set-step phase within each iteration; this assumption fails if grid.sync() is omitted or if the kernel is launched without cooperative launch, in which case the persistent dual-state buffer may reflect a corrupted state due to concurrent writes. The fused switch kernel's greedy single-constraint transition yields the same active-set sequence as the reference recursive solver when at most one constraint violates the KKT conditions per iteration; this assumption fails when multiple constraints violate simultaneously, in which case the kernel selects the most violated one, reflecting a greedy transition rather than a simultaneous update, and the iterate sequence may diverge from the reference solver.

## Contribution

['A dual-state-preserving CUDA graph replay framework (DS-Graph) that encodes the active-set recursion as a persistent device-resident state buffer with a fused branch-selection kernel, enabling warm-start-valid graph replay for QP-based MPC.', 'A design principle for GPU-replayable warm starting: the dual state must be stored on-device and updated in-place, and active-set transitions must be dispatched via device-side branching within a single kernel to avoid host intervention.', 'An open-source implementation and integration recipe for embedding DS-Graph into existing MPC pipelines, with device calibration for buffer sizing and kernel fusion.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | Dense QP suite from MPC control benchmarks (humanoid, quadrotor trajectories) | Matches FAME-QP benchmark distribution for direct comparison |
| Primary metric | Per-iteration wall-clock latency and total solve time | Isolates graph-replay overhead reduction vs. algorithmic work |
| Secondary metric | Iterate-sequence divergence rate vs. reference recursive active-set solver (bitwise or within tolerance 1e-6) | Verifies the load-bearing assumption that device-side dispatch reproduces the reference solver's state evolution |
| Baseline 1 | FAME-QP | Static-graph GPU QP solver without active-set recursion |
| Baseline 2 | GPU operator-splitting QP solver | Fixed-iteration GPU baseline, no branching |
| Baseline 3 | GPU interior-point QP solver | Iteration-count-sensitive GPU baseline |
| Baseline 4 | Reference recursive active-set solver (CPU, double precision) | Ground truth for iterate-sequence equivalence |
| Ablation-of-ours | DS-Graph with host-side dual-state storage (no persistent buffer) | Tests whether device-resident state drives the gain |
| Ablation-of-ours | DS-Graph with non-cooperative launch (grid.sync() replaced by host-loop replay) | Tests whether the persistent cooperative kernel is necessary for the latency gain |

### Why this setup validates the claim

The central claim is that preserving the dual-state recursion across CUDA graph replays removes per-iteration host-device synchronization and launch overhead while retaining active-set convergence behavior. FAME-QP tests the sub-claim that a static operator bank cannot exploit warm-start active-set transitions at every iteration; the operator-splitting and interior-point baselines test the sub-claim that fixed-iteration GPU solvers cannot skip work when the warm start is already near the solution. The reference recursive active-set solver tests the load-bearing assumption that the persistent cooperative kernel with fused branch-selection and active-set-step logic reproduces the exact iterate sequence of the recursive solver; divergence rate quantifies how often the greedy single-constraint transition differs from the reference. The ablation with host-side state storage isolates the mechanism: if the persistent device-resident dual-state buffer is what matters, replacing it with host-side storage should erode the gain specifically on problems with many active-set transitions. The ablation with non-cooperative launch tests whether the persistent cooperative kernel is necessary for the latency gain, or whether a host-loop replay of the same fused kernel achieves similar performance. Latency is the right metric because the predicted effect is a per-iteration overhead reduction, not a change in solution accuracy; accuracy is held as a sanity parity check rather than a headline result.

### Expected outcome and causal chain

**vs. FAME-QP** — On a receding-horizon MPC instance where consecutive QPs differ only slightly, FAME-QP replays its static graph and re-solves from a fixed operator bank without carrying the previous working set, so it repeats work that a warm start would skip. Our method instead reads the persistent dual state, detects that the working set is nearly valid, and dispatches a SOLVE step rather than re-deriving the active set, because the persistent cooperative kernel consumes next_kernel_id directly on device. We expect a visible latency gap that grows with horizon length and shrinks to parity on cold-start or heavily perturbed QPs where the warm start is stale.

**vs. GPU operator-splitting QP solver** — On a QP whose active set stabilizes after a few iterations, the operator-splitting baseline must run its full fixed iteration budget regardless of convergence, because its iteration count is not data-dependent. Our method instead terminates early via the convergence check inside the persistent kernel and skips active-set transitions once the working set is stable, because the dual-state buffer records that no transition is needed. We expect a noticeable per-solve time reduction concentrated on instances with few active-set transitions, and near-parity on ill-conditioned instances requiring many transitions.

**vs. GPU interior-point QP solver** — On a QP where the primal solution lies on a constraint boundary, the interior-point baseline takes many iterations to approach the active set and cannot reuse the previous solution's boundary identification. Our method instead starts from the previous working set and only adds or drops constraints that the KKT check flags, because the branch_flag logic inside the persistent kernel tests KKT conditions on the warm-started dual state. We expect lower total solve time on sequentially related QPs and a smaller iteration count, with the gap widening as the QP sequence becomes more temporally correlated.

**vs. Reference recursive active-set solver** — On a randomized QP suite, the reference solver provides the ground-truth iterate sequence. Our method should match the reference's active-set sequence within tolerance 1e-6 on instances where at most one constraint violates the KKT conditions per iteration; divergence is expected on instances with simultaneous multi-constraint violations, where the greedy single-constraint transition may differ. We expect a divergence rate below 5% on the benchmark suite, with divergence concentrated on ill-conditioned instances.

### What would falsify this idea

If the latency gain over FAME-QP and the ablation is uniform across all QP subsets rather than concentrated on instances with many active-set transitions and temporally correlated warm starts, then the persistent dual-state mechanism is not what produces the speedup and the central claim is wrong. A second falsifier: if the ablation with host-side state storage matches DS-Graph's latency on transition-heavy instances, the device-resident buffer is not the causal driver. A third falsifier: if the iterate-sequence divergence rate vs. the reference recursive solver exceeds 5% on the benchmark suite, the load-bearing assumption that device-side dispatch reproduces the reference solver's state evolution is false.

## References

1. FAME-QP: A GPU-Accelerated Quadratic-Programming Solver for Real-Time Model Predictive Control
