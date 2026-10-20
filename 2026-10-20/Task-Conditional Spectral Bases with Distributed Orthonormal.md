# Task-Conditional Spectral Bases with Distributed Orthonormalized Updates for Matryoshka Representations

## Motivation

SpecMRL reorganizes the feature space with a single parameter-derived spectral basis ordered by global feature importance, so its ordering is fixed across tasks and its prefix readout offers no task-conditional identifiability: leading dimensions that are informative for one downstream task may be uninformative for another, and the basis is never constrained to remain orthonormal under the truncation that Matryoshka training actually applies. The root cause is structural — SpecMRL inherits MRL's multi-prefix objective unchanged and adds the basis as a post-hoc reorganization, so nothing in the training signal couples the basis's orthonormality to the nested-prefix truncations, and nothing conditions the basis on the task. A joint framework is needed that learns a task-conditional basis whose orthonormality is enforced exactly at every truncation depth the model is evaluated at.

## Key Insight

If the spectral basis is updated only by skew-symmetric (Cayley) steps, orthonormality is an exact invariant of the update rather than a penalty, so a single shared constraint can simultaneously guarantee truncation-consistent orthonormality at every prefix depth and task-conditional ordering via per-task rotation costs — a joint guarantee that neither a fixed global basis nor a soft orthogonality penalty can provide.

## Method

**What it is.** TCS-MRL (Task-Conditional Spectral Matryoshka Representation Learning) learns a shared orthonormal spectral basis $U \in \mathbb{R}^{d \times d}$ together with a small set of task-conditional rotation generators $\{A_t\}_{t=1}^{T}$ ($A_t$ skew-symmetric, $d \times d$) and per-task diagonal scaling matrices $\{D_t\}_{t=1}^{T}$ ($D_t$ diagonal with positive entries). Inputs: a batch of (example, task-id) pairs and nested prefix dimensions $\mathcal{D}=\{d_1,\dots,d_K\}$. Outputs: task-conditional bases $U_t = U \exp(A_t) D_t$ and nested prefix readouts in each $U_t$'s leading coordinates, trained under MRL's multi-prefix objective. The load-bearing assumption made explicit: task-specific informative directions are related to a common orthonormal frame by a pure rotation followed by per-coordinate scaling, i.e., $U_t = U \exp(A_t) D_t$ with $U$ orthonormal, $A_t$ skew-symmetric, and $D_t$ diagonal positive. This assumption is verified empirically via the diagnostic in step 6.

**How it works.**
```pseudocode
initialize U = I_d, A_t = 0, D_t = I_d for all tasks t
for each minibatch B = {(x_i, t_i)}:
    # 1. task-conditional basis via exact orthogonal exponential and diagonal scaling
    for each distinct task t in B:
        U_t = U @ expm(A_t) @ D_t        # expm of skew-symmetric => orthogonal; D_t diagonal positive
    # 2. nested prefix readout in spectral space
    z_i = encoder(x_i)                    # d-dim backbone embedding
    for each d_k in D:
        h_i^k = (U_{t_i}[:, :d_k])^T z_i  # project onto leading d_k columns
        loss_k += CE(head_k(h_i^k), y_i)
    # 3. distributed orthonormalized update of the SHARED basis
    G_U = dL/dU                           # accumulated over all tasks in B
    U <- cayley_update(U, G_U, eta)       # U <- U (I - eta/2 G_U^skew)^{-1}(I + eta/2 G_U^skew)
    # 4. per-task rotation and scaling updates
    for each task t in B:
        G_A = dL/dA_t
        A_t <- A_t - eta * (G_A - G_A^T)/2  # project gradient to skew-symmetric subspace
        G_D = dL/dD_t
        D_t <- D_t - eta * diag(G_D)        # diagonal update, positivity enforced by clamping to [1e-3, 1e3]
    # 5. shared identifiability-orthonormality constraint (single scalar, no penalty weight)
    L_id = sum_t || diag(U_t^T Sigma_t U_t) - sort(diag(U_t^T Sigma_t U_t), desc) ||_1
    # enforced as a projection: re-orthonormalize U via polar decomposition after step 3
    U <- polar(U)                         # exact orthonormality, no soft weight
    # 6. diagnostic (every 100 steps): compute Spearman correlation between variance order and per-prefix accuracy order across tasks
```
Hyperparameters: Cayley step size $\eta$ (default 0.05, cosine-decayed), $|\mathcal{D}|$ prefix depths (MRL default grid), $T$ task heads, $D_t$ initialized to identity, $D_t$ clamp range $[10^{-3}, 10^{3}]$. The Cayley update and polar projection are the only places $U$ changes; $A_t$ changes only through skew-symmetric gradient projection; $D_t$ changes only through diagonal gradient updates. Compute budget: for $d=1024$, $T=8$, batch size 256, training 100 epochs on MTEB retrieval+classification subsets, estimated 120 GPU-hours on a single NVIDIA A100 40GB. Numerical stability: $\expm(A_t)$ computed via scaling-and-squaring with Padé approximation (13 terms) using `scipy.linalg.expm`; fallback to truncated Taylor series (10 terms) if $\|A_t\|_F > 10$; polar decomposition via SVD of $U$ (cost $O(d^3)$).

**Why this design.** We chose the Cayley transform over a Gram-Schmidt or soft orthogonality penalty because Cayley updates preserve orthonormality as an exact algebraic invariant at every step, accepting the cost of a matrix inverse per update ($O(d^3)$, negligible for $d \le 1024$); a soft penalty (as in standard orthogonal-regularized MRL variants) leaves residual non-orthonormality that grows with truncation depth, which is precisely the failure SpecMRL exhibits. We chose a shared basis $U$ with per-task rotations $\exp(A_t)$ and diagonal scalings $D_t$ over $T$ independent bases because independent bases destroy cross-task transfer and multiply parameters by $T$, accepting the cost that tasks with genuinely orthogonal informative subspaces must be expressed as rotations and scalings of a common frame — a restriction we mitigate by letting $A_t$ be full-rank and $D_t$ be full-rank diagonal. The diagonal scaling $D_t$ is included because the ordering criterion $\mathrm{diag}(U_t^\top \Sigma_t U_t)$ is variance-based and anisotropic per task; a pure rotation cannot adjust the scale of coordinates, so $D_t$ carries the per-task anisotropy while $U$ remains orthonormal. We chose the $\ell_1$ ordering constraint on $\mathrm{diag}(U_t^\top \Sigma_t U_t)$ over a learned importance head because the diagonal of the task covariance in the spectral basis is exactly the variance captured by each leading coordinate, accepting the cost that this requires per-task covariance estimates (running estimates over the batch). This differs from SpecMRL (the closest prior, which fixes one global basis ordered by parameter magnitude and never conditions on task) and from orthogonal-MRL penalty variants (which do not make orthonormality an invariant); a domain expert would not call TCS-MRL a variant of either because the update rule itself, not a loss term, carries the guarantee.

**Why it measures what we claim.** The Cayley/polar update of $U$ measures truncation-consistent orthonormality because $U^\top U = I$ holds exactly after every step under the assumption that the encoder gradients do not require $U$ to leave the orthogonal manifold; this assumption fails when the optimal basis for a task is genuinely non-orthogonal (e.g., highly correlated informative directions), in which case the constraint reflects projected orthonormality rather than unconstrained optimality. The per-task rotation $\exp(A_t)$ and scaling $D_t$ measure task-conditional identifiability because $\mathrm{diag}(U_t^\top \Sigma_t U_t)$ equals the task-$t$ variance captured by each spectral coordinate under the assumption that $\Sigma_t$ is a sufficient statistic for task-$t$ label information and that the readout head is variance-monotone; this assumption fails when label information lives in higher moments or cross-covariances, in which case the ordering reflects second-moment importance only. The $\ell_1$ ordering loss measures task-specific prefix informativeness because it penalizes any inversion of the variance ordering under the assumption that variance ordering equals information ordering for the readout head; this assumption fails when the head is nonlinear and can recover low-variance directions, in which case the constraint enforces a conservative but not necessary ordering. The diagnostic in step 6 directly measures the Spearman correlation between variance order and per-prefix accuracy order; a high correlation supports the variance-order-equals-information-order assumption, while a low correlation indicates the assumption fails for that task.

## Contribution

(1) A joint training framework in which a shared orthonormal spectral basis and per-task rotation generators are updated exclusively through skew-symmetric/Cayley steps, making truncation-consistent orthonormality an exact invariant rather than a penalty. (2) A design principle — the shared identifiability-orthonormality constraint — that couples task-conditional ordering and prefix-depth orthonormality in a single constraint, resolving the residual gap left by fixed-basis SpecMRL and soft-penalty orthogonal MRL. (3) A per-task covariance-diagonal ordering criterion that replaces global feature-importance ordering with task-conditional spectral ordering while keeping parameter count linear in the number of tasks.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | MTEB retrieval + classification subsets (e.g., 8 tasks: 4 retrieval, 4 classification) | Matches SpecMRL benchmark; tests truncation across tasks |
| Primary metric | Mean performance across reduced dimensions $\mathcal{D}=\{8,16,32,64,128,256,512,1024\}$ | Directly measures truncation-consistency, the core claim |
| Baseline | SpecMRL | Closest prior: single global spectral basis, no task condition |
| Baseline | Standard MRL | Tests whether orthonormal invariant beats unconstrained nesting |
| Baseline | Orthogonal-penalty MRL (weight 0.1) | Tests exact invariant vs soft penalty under truncation |
| Ablation-of-ours | TCS-MRL without per-task rotation ($A_t=0$) | Isolates contribution of task-conditional rotations |
| Ablation-of-ours | TCS-MRL without per-task scaling ($D_t=I$) | Isolates contribution of per-task anisotropy |
| Diagnostic | Spearman correlation between variance order and per-prefix accuracy order | Tests variance-order-equals-information-order assumption |

### Why this setup validates the claim

The central claim is that exact orthonormality as an algebraic invariant, combined with task-conditional rotations and scalings of a shared frame, yields truncation-consistent task-conditional representations that neither a fixed global basis (SpecMRL) nor a soft penalty can match. The dataset choice matters because MTEB spans tasks with genuinely different informative subspaces, so a task-blind basis should degrade precisely where tasks diverge. Standard MRL tests the value of any spectral structure; orthogonal-penalty MRL isolates whether exactness (invariant) versus approximation (penalty) drives gains; SpecMRL isolates task-conditioning. The ablation without rotations removes $A_t$, so any gain it loses is attributable to task conditionality rather than orthonormality alone; the ablation without scaling removes $D_t$, testing whether per-task anisotropy is necessary for the variance-based ordering. The diagnostic directly measures the correlation between variance order and information order, providing empirical support for the assumption that variance ordering equals information ordering. Mean performance across reduced dimensions is the right metric because the claim is specifically about behavior under truncation — a method that only wins at full dimension does not validate the truncation-consistency argument. Together these form a falsifiable grid: each baseline pins one sub-claim, and the metric reads out exactly the predicted failure axis.

### Expected outcome and causal chain

**vs. SpecMRL** — On a case where two tasks share a backbone but have informative directions that are rotated and scaled relative to each other, SpecMRL produces a single global ordering that is optimal for the average task and misorders prefixes for the minority task, because its basis is fixed and task-blind. Our method instead applies per-task rotations $\exp(A_t)$ and scalings $D_t$ so each task reads its own leading coordinates, because the shared frame plus rotations and scalings preserves cross-task transfer while allowing task-specific ordering. We expect a noticeable gap concentrated on tasks whose informative subspace diverges most from the global average, with parity on tasks aligned with the dominant direction.

**vs. Standard MRL** — On a case with aggressive compression (small prefix depth), standard MRL produces prefixes whose coordinates are correlated and redundant, because unconstrained nesting lets later dimensions leak information into earlier ones and truncation is not orthonormal. Our method instead enforces $U^\top U = I$ exactly via Cayley/polar updates, so each prefix captures a disjoint variance budget, because orthonormality is an algebraic invariant rather than a soft preference. We expect the largest gap at the most aggressive dimensions, shrinking toward full dimension where the constraint is inactive.

**vs. Orthogonal-penalty MRL** — On a case requiring deep truncation, orthogonal-penalty MRL produces residual non-orthonormality that accumulates with depth, because the penalty is traded against the task loss and never exactly satisfied. Our method instead re-orthonormalizes $U$ exactly after every update, because the polar projection and Cayley step leave the manifold only through the invariant-preserving map. We expect a gap that grows monotonically with truncation depth, with near-parity at shallow truncation where residual penalty error is negligible.

**vs. TCS-MRL without per-task rotation ($A_t=0$)** — On a case where tasks have genuinely orthogonal informative subspaces, the ablation produces a single shared ordering that cannot satisfy both tasks simultaneously, because with $A_t=0$ every task reads the same leading coordinates. Our method instead expresses task-specific orderings as rotations of the common frame, because full-rank $A_t$ spans the needed rotations. We expect the ablation to match ours on single-task or aligned-task subsets but fall behind on multi-task subsets with divergent spectra, isolating the rotation contribution.

**vs. TCS-MRL without per-task scaling ($D_t=I$)** — On a case where tasks have informative directions with different variances (anisotropic), the ablation cannot adjust the scale of coordinates, because $D_t=I$ forces all tasks to share the same scaling. Our method instead uses diagonal $D_t$ to carry per-task anisotropy, because the variance-based ordering requires scaling to align variance magnitudes. We expect the ablation to underperform on tasks with high variance disparity, isolating the scaling contribution.

**Diagnostic** — We expect Spearman correlation between variance order and per-prefix accuracy order to be high (e.g., >0.7) for most tasks, supporting the variance-order-equals-information-order assumption; lower correlation (e.g., <0.5) would indicate the assumption fails for that task, explaining any performance gap.

### What would falsify this idea

If our gain over SpecMRL and orthogonal-penalty MRL is uniform across all truncation depths and task subsets rather than concentrated at aggressive truncation and on tasks with divergent informative subspaces, the claimed causal mechanism (exact orthonormality plus task-conditional rotation and scaling) is wrong. Equivalently, if the no-rotation or no-scaling ablations match the full method everywhere, task conditionality contributes nothing and the central claim collapses; additionally, if the diagnostic shows low correlation between variance order and information order, the variance-based ordering assumption is invalid.

## References

1. SpecMRL: Redesign Matryoshka Representation Learning within Parametric Spectrum
