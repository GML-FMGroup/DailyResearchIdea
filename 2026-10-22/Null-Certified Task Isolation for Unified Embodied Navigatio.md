# Null-Certified Task Isolation for Unified Embodied Navigation Agents

## Motivation

Unified multi-task embodied models such as Uni-NaVid and SAME assume that harmonizing I/O or adding state-adaptive MoE routing produces cross-task synergy, yet neither provides any structural certificate that gradients from one navigation task leave the shared representation of another uncorrupted; SAME's own limitation section concedes the model may degrade on unseen task combinations precisely because negative interference is assumed away rather than excluded. The root cause is that interference is only ever diagnosed post-hoc (gradient surgery, PCGrad) or sidestepped by capacity partitioning, so the shared parameters are never required to admit a task-conditional decomposition whose cross-task second-moment structure is controlled at training time.

## Key Insight

If each task's shared-parameter Jacobian block is projected onto a certified null-space of every other task's gradient subspace at every step, then the shared representation's cross-task second-moment matrix is block-diagonal under the model's own forward map, so any composition of tasks preserves per-task loss monotonicity rather than merely hoping for it.

## Method

**(A) What it is.** Null-Certified Task Isolation (NCTI) is a training-time invariant enforcer for a single shared-parameter embodied navigation model (frozen Qwen-class LLM backbone + a low-dimensional policy head shared across tasks). Inputs: per-task batches from the VLNVerse task toolkit (each exposing a task ID and per-task loss). Outputs: a shared policy-head parameter update whose per-task Jacobian blocks are orthogonal to every other task's gradient subspace, plus a per-step null-space certificate matrix logged for auditing.

**(B) How it works.**

```pseudocode
# Shared surface: policy-head params theta (d dims, d=256). Backbone frozen.
# Tasks T = {1..K} from VLNVerse toolkit, K=6.
# Load-bearing assumption (stated explicitly): the Jacobian column space of the
# shared policy head is the complete and only channel through which one task's
# update perturbs another task's features/loss. This holds under (i) frozen
# backbone, (ii) small effective step size (lr=1e-4, Adam), and (iii) locally
# linear head dynamics over the step. If any fails, the certificate is first-order only.
for step in 1..N:
    for each task k in batch:
        g_k = grad(L_k(theta))            # task-k gradient, d-dim
        J_k = jacobian(shared_features_k, theta)  # d x m block, m=128
        U_k = orthonormal_basis(J_k)      # task-k Jacobian column space (SVD, tol=1e-3)
        # Conditioning guard: detect near-collinear Jacobian spaces
        cond_k = cond(J_k)                # 2-norm condition number
        rank_k = rank(J_k, tol=1e-3)
    for each task k:
        # Null-space certificate: subspace orthogonal to ALL other tasks' Jacobian spaces
        N_k = null_space( concat_{j!=k} U_j )   # via QR of [U_j]_{j!=k}
        g_k_iso = Project_{N_k}(g_k)      # remove cross-task-interfering component
        # residual r_k = ||g_k - g_k_iso|| logged as interference magnitude
        # Rank/conditioning guard: if dim(N_k) < 0.1*d or cond_k > 1e4, fall back to
        # per-task adapter head (capacity partition) rather than emit vacuous certificate
        if dim(N_k) < 0.1*d or cond_k > 1e4:
            use per-task adapter head for task k (d_adapter=64, frozen shared head)
            log fallback event
        else:
            theta <- theta - lr * sum_k g_k_iso          # lr=1e-4, Adam on projected grads
    certify: assert  max_{k!=j} || U_k^T g_j_iso ||_F < eps   # eps=1e-5, per-step check
    if certificate fails: shrink lr by 0.5 and re-project (max 3 retries)
    # Log per-step: certificate matrix, residual r_k, cond_k, rank_k, fallback flag
```

Each step (i) estimates per-task Jacobian column spaces on the shared policy head, (ii) builds a null-space certificate orthogonal to all other tasks' spaces, (iii) projects each task gradient into its certified null-space, and (iv) asserts the cross-task orthogonality residual stays below eps before applying the update. The certificate matrix and residual r_k are logged per task per step. The conditioning guard (cond_k > 1e4 or dim(N_k) < 0.1*d) triggers a per-task adapter fallback to avoid vacuous certificates under near-collinear Jacobians.

**(C) Why this design.** We chose a Jacobian-null-space projection over PCGrad-style pairwise gradient surgery because PCGrad only removes pairwise gradient conflict after the fact and gives no per-step certificate on the shared representation, accepting the cost that we must materialize and SVD each task's Jacobian block (feasible only because the shared policy head is low-dimensional, which is why we freeze the Qwen backbone rather than fine-tune it end-to-end). We chose an explicit orthogonality certificate with a retry/shrink loop over a soft penalty term because a penalty can be satisfied on average while individual steps violate isolation, accepting the cost of occasional learning-rate shrinkage that slows convergence in exchange for a checkable per-step invariant. We chose to build the null-space from the other tasks' Jacobian column spaces rather than from their raw gradients because the Jacobian space is the subspace that actually transmits interference through the forward map, accepting the higher per-step compute of SVD over a plain gradient dot-product. A domain expert would not describe this as PCGrad or gradient surgery applied to navigation: those methods resolve conflicts between gradients and offer no certificate, whereas NCTI certifies that each task's update lies in a null-space of the others' forward-map Jacobians, which is the structural property PCGrad does not produce.

**(D) Why it measures what we claim.** The cross-task orthogonality residual ||U_k^T g_j_iso||_F measures task-interference isolation because the assumption that the shared policy head's forward map is locally linear in theta over the step (small lr, frozen backbone) makes the Jacobian column space exactly the subspace through which task j's update perturbs task k's features; this assumption fails when the policy head is highly nonlinear within a step or the backbone is unfrozen, in which case the residual reflects only first-order isolation and higher-order cross-talk can persist. The per-task residual r_k measures realized interference magnitude because the assumption that g_k decomposes into an isolatable and an interfering component along N_k and its complement holds when the Jacobian spaces are full-rank and well-conditioned; this fails under near-collinear task Jacobians, in which case r_k reflects conditioning noise rather than genuine interference. The block-diagonality of the cross-task second-moment matrix measures the motivation-level composition-closure property because the assumption that per-step orthogonality accumulates to block-diagonal second moments holds when task gradients are stationary across the null-space projection; this fails under rapidly shifting task distributions, in which case the second-moment matrix drifts from block-diagonal between certification checks. The conditioning guard (cond_k > 1e4 or dim(N_k) < 0.1*d) measures the validity of the null-space certificate because the assumption that N_k is non-vacuous requires well-conditioned Jacobian spaces; this fails when tasks share near-collinear features, in which case the guard triggers a per-task adapter fallback rather than emitting a vacuous certificate.

## Contribution

(1) NCTI, a training-time invariant enforcer that projects each task's shared-parameter gradient into a certified null-space of every other task's Jacobian block and emits a per-step orthogonality certificate for a unified embodied navigation model. (2) The design principle that freezing the LLM backbone and confining multi-task learning to a low-dimensional shared policy head makes exact per-step Jacobian-orthogonality certification tractable, whereas end-to-end fine-tuning makes it only approximate. (3) A per-task interference-residual diagnostic logged alongside the certificate, exposing which VLNVerse task pairs induce the largest cross-task Jacobian overlap.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | VLNVerse multi-task suite (6 tasks: instruction-following, object-goal, room-to-room, vision-language navigation, embodied QA, and long-horizon switch) | Shared-surface tasks with per-task losses exposed |
| Primary metric | Cross-task interference residual r_k (logged per step) | Directly measures certified isolation, not just accuracy |
| Secondary metric | Per-task feature drift ||Δfeatures_k|| after other-task updates (held-out, not optimized) | Independent check that r_k tracks realized interference |
| Baseline | PCGrad gradient surgery | Tests whether pairwise gradient conflict removal suffices |
| Baseline | Uni-NaVid unified model | Tests whether a strong generalist already composes tasks |
| Baseline | Task-specific fine-tuning | Tests whether isolation beats dedicated per-task heads |
| Ablation-of-ours | NCTI minus null-space certificate (projection onto raw-gradient complements) | Tests whether the certificate itself drives the effect |
| Ablation-of-ours | NCTI with conditioning guard disabled | Tests whether the guard prevents vacuous certificates |

### Why this setup validates the claim

The central claim is that per-step Jacobian-null-space certification produces isolation that gradient-surgery and unified-training baselines cannot, and that this isolation is causally responsible for reduced cross-task interference. VLNVerse is the right dataset because it exposes per-task losses on a shared surface, which is exactly the condition under which NCTI's certificate is defined; without per-task loss decomposition the certificate cannot even be computed. PCGrad tests the specific sub-claim that pairwise post-hoc conflict removal is insufficient — if PCGrad matches NCTI, the certificate is doing nothing. Uni-NaVid tests whether a strong existing unified model already composes tasks implicitly, which would undercut the motivation for explicit isolation. Task-specific fine-tuning tests the upper bound on per-task performance and reveals whether isolation costs task-specific specialization. The ablation (NCTI without the certificate, i.e. projection onto the other tasks' raw-gradient complements rather than Jacobian null-spaces) is the critical control: it isolates whether the Jacobian-space construction, not merely any projection, produces the gain. The primary metric is the interference residual r_k rather than aggregate success rate because the claim is about isolation as a mechanism; success rate is a downstream consequence that could improve for unrelated reasons, whereas r_k is the quantity the method explicitly certifies and logs, so a failure to reduce it directly falsifies the mechanism. The secondary metric (feature drift) provides an independent, non-optimized measure to break the self-fulfilling loop: if r_k decreases but feature drift does not, the certificate is certifying only its own definition.

### Expected outcome and causal chain

**vs. PCGrad gradient surgery** — On a case where two VLNVerse tasks (e.g. instruction-following and object-goal navigation) push the shared policy head in conflicting directions on the same feature coordinates, PCGrad projects only the pairwise conflicting gradient component after the fact, leaving the shared Jacobian column space of each task perturbed by the other's forward-map contribution. Our method instead removes each task's gradient component that lies in any other task's Jacobian column space before the update, so the forward map of task k is untouched by task j's step. We expect a noticeable gap in r_k on the high-conflict task pairs (instruction vs. object-goal) but near-parity on low-conflict pairs, with PCGrad's residual fluctuating step-to-step while NCTI's stays pinned below eps. The independent feature-drift metric should mirror this gap, confirming that r_k tracks realized interference.

**vs. Uni-NaVid unified model** — On a case where a long-horizon instruction requires switching between navigation subtasks mid-episode, Uni-NaVid's shared representation drifts because its multi-task training optimizes aggregate loss and lets dominant tasks overwrite the shared features that minority tasks rely on. Our method instead certifies per step that no task's update enters another task's forward-map subspace, so the shared features retain task-specific structure across the episode. We expect Uni-NaVid to show a larger r_k concentrated on the minority-task subsets and longer-horizon episodes, while NCTI shows a flatter residual profile and a visible advantage on the long-horizon switch subset specifically. Feature drift should be correspondingly lower for NCTI on the switch subset.

**vs. Task-specific fine-tuning** — On a case where a single task is data-scarce, task-specific fine-tuning overfits its narrow distribution and cannot leverage shared navigation priors, producing brittle behavior on held-out scenes. Our method instead shares the frozen backbone and policy head across tasks while certifying isolation, so scarce tasks borrow representation without being overwritten. We expect task-specific fine-tuning to win on data-rich tasks but lose on scarce tasks, while NCTI approaches task-specific parity on rich tasks and exceeds it on scarce tasks — a crossover pattern that confirms isolation preserves rather than erases shared structure. The conditioning guard should trigger more often on scarce tasks, where Jacobians are noisier, and the fallback adapter should prevent vacuous certificates.

### What would falsify this idea

If NCTI's interference residual r_k is not lower than PCGrad's on the predicted high-conflict task pairs, or if the certificate-with-ablation (raw-gradient complement) matches NCTI's residual, then the Jacobian-null-space construction is not the causal driver and the central claim is wrong. A uniform gain across all task pairs rather than a gap concentrated where the predicted forward-map interference occurs would likewise indicate the effect is not mechanism-specific. Additionally, if the independent feature-drift metric does not correlate with r_k, the certificate is self-fulfilling and the mechanism is falsified.

## References

1. VLNVerse: A Benchmark for Vision-Language Navigation with Versatile, Embodied, Realistic Simulation and Evaluation
2. Uni-NaVid: A Video-based Vision-Language-Action Model for Unifying Embodied Navigation Tasks
3. GRUtopia: Dream General Robots in a City at Scale
4. Navigation World Models
5. Towards Long-Horizon Vision-Language Navigation: Platform, Benchmark and Method
6. SAME: Learning Generic Language-Guided Visual Navigation with State-Adaptive Mixture of Experts
7. Habitat 3.0: A Co-Habitat for Humans, Avatars and Robots
8. Qwen Technical Report
