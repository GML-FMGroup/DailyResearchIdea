# Task-Conditional Spectral Ordering with Contrastive Identifiability for Matryoshka Embeddings

## Motivation

SpecMRL learns a single global ordering of spectral basis components by feature importance, so every task reads out the same prefix dimensions. This structurally prevents task-conditional identifiability: two tasks with different informative directions are forced to share one ordering, and nothing in the objective penalizes the orderings from collapsing to the same permutation. The gap is not a lack of capacity but a missing constraint — SpecMRL's spectral MRL structure has no mechanism that makes the ordering task-dependent or that enforces different tasks to expose different spectral importance orderings.

## Key Insight

A task-conditional permutation of a shared spectral basis is identifiable up to task identity exactly when the contrastive ordering loss forces the induced importance orderings of distinct tasks to disagree, because the shared basis plus per-task gating makes the ordering a function of the task embedding rather than of the parameters alone.

## Method

**What it is.** We introduce **TC-SpecMRL** (Task-Conditional Spectral MRL). Inputs: a backbone embedding $h\in\mathbb{R}^d$ (d=768 for BERT-base), a task identifier mapped to a lightweight task embedding $e_\tau\in\mathbb{R}^k$ (k=16), and a parameter-derived spectral basis $U=[u_1,\dots,u_d]$ shared across tasks. Outputs: a task-conditional ordering $\pi_\tau$ of spectral components and a nested set of prefix readouts in the reordered spectral space.

**Load-bearing assumption (stated explicitly).** The method assumes that the task embedding $e_\tau$, through a low-capacity 2-layer MLP gate, can recover each task's true task-conditional spectral importance ordering well enough that the induced permutation $\pi_\tau$ is identifiable and meaningfully different from the global ordering. This is a task-level, input-independent reweighting: the ordering is a function of the task embedding alone, not of the actual input $h$. The assumption fails when two tasks share an embedding (coarse task labels) or when task identity alone does not determine which spectral directions are informative for a given input; in that case $m$ reflects shared cluster importance rather than the individual task's, and the ordering remains near-global. We make this assumption explicit and add a calibration/verification story below.

**How it works.**
```pseudocode
# Shared spectral basis from SpecMRL parameters
U = orthonormalize(W_basis)                      # d x d, shared
s = U^T h                                        # spectral coefficients
# Task-conditional gating over spectral components
g = sigmoid(MLP(e_tau))                          # d-dim gate, k=16 task emb dim; MLP: 2-layer, hidden=64, GeLU
m = softmax(g * |s| / T)                         # T=0.1 temperature; importance mask
# Task-conditional ordering: rank by gated importance
pi_tau = argsort_desc(m)                         # permutation of {1..d}
s_ord = s[pi_tau]; U_ord = U[:, pi_tau]
# Nested prefix readouts in reordered spectral space
for j in prefixes P = {8,16,32,64,...,d}:
    z_j = concat(s_ord[:j])                      # prefix readout
    L_task += CE(head_j(z_j), y)                 # standard MRL multi-prefix loss
# Contrastive ordering loss across tasks in the batch
for each pair (tau, tau') with tau != tau':
    L_ord += max(0, margin - d_H(pi_tau, pi_tau'))   # margin=0.3, d_H = normalized Kendall tau distance over top-K=32 components
L = L_task + lambda * L_ord                      # lambda=0.5
```
The task embedding $e_\tau$ is a 16-dim learned vector per task (or per task-cluster when task labels are coarse); the MLP is a 2-layer 64-unit network with GeLU activation. The gate $g$ modulates the spectral coefficients before ranking, so the ordering $\pi_\tau$ depends on both the shared basis and the task. The contrastive ordering loss uses normalized Kendall tau distance between the top-$K$ ($K=32$) components of each task's ordering, encouraging divergence when tasks differ.

**Calibration / verification story (inside the existing structure).** To verify the load-bearing assumption, we add two diagnostic checks that are computed during training and logged, without changing the training procedure: (i) we measure the Spearman correlation between the pre-argsort gate-score ordering and the realized argsort permutation on a held-out set of 512 examples per task; if this correlation is below 0.8, the gate scores are not controlling the ordering and the assumption is violated; (ii) we measure the correlation between task-embedding distance $\|e_\tau - e_{\tau'}\|$ and the true spectral divergence (computed as the normalized Kendall tau distance between the top-32 components of the per-task empirical importance $|s|$ averaged over 512 examples); if this correlation is below 0.3, the task embedding does not encode spectral divergence and the gate cannot recover task-conditional orderings. These checks are reported as diagnostics, not used as losses.

**Why this design.** We chose a shared orthonormal spectral basis with per-task gating over a fully task-specific basis because sharing preserves the SpecMRL structural guarantee that leading prefix dimensions capture informative directions, while gating makes the ordering task-dependent, accepting the cost that tasks with very different spectra must still share the same basis vectors. We chose a contrastive ordering loss based on Kendall tau distance over the top-$K$ components rather than a direct diversity penalty on the gate vectors because Kendall tau operates on the discrete ordering that is actually used at readout, accepting the cost that the loss is non-differentiable through the argsort and is applied as a pairwise margin on the resulting permutations (computed from the differentiable gate scores before argsort). We chose a lightweight 2-layer MLP over a full task-conditioned linear layer because the task embedding is low-dimensional and the gate must remain cheap at inference, accepting the cost that the gate cannot express high-order interactions between task identity and spectral coefficients. A domain expert would not describe this as a variant of SpecMRL because SpecMRL has a single global ordering derived from parameter importance, whereas TC-SpecMRL makes the ordering an explicit function of the task embedding and adds a contrastive identifiability constraint that SpecMRL lacks entirely.

**Why it measures what we claim.** The gated importance score $m_i = \text{softmax}(g_i |s_i| / T)$ measures task-conditional spectral importance because the assumption is that the task embedding $e_\tau$ captures the sufficient statistics of the task's informative directions; this assumption fails when two tasks share an embedding (coarse task labels) and $m$ then reflects the shared cluster importance rather than the individual task's. The Kendall tau distance $d_H(\pi_\tau, \pi_{\tau'})$ measures ordering divergence between tasks because the assumption is that a task's ordering is identifiable from its top-$K$ spectral components; this assumption fails when $K$ is too small relative to $d$, in which case $d_H$ reflects noise in the tail rankings rather than genuine task difference. The contrastive margin loss $\max(0, \text{margin} - d_H)$ measures identifiability pressure because the assumption is that pushing orderings apart when tasks differ forces the gate to encode task-specific structure; this assumption fails when tasks are genuinely similar, in which case the loss is inactive and the ordering remains near-global, which is the correct behavior but means the method provides no benefit over SpecMRL for that task pair.

## Contribution

['A task-conditional spectral gating module that modulates a shared parameter-derived spectral basis with a lightweight task embedding, producing a task-dependent permutation of spectral components while preserving the SpecMRL nested-prefix readout structure.', 'A contrastive ordering loss based on normalized Kendall tau distance over top-$K$ spectral components that enforces task-specific spectral importance orderings to diverge when tasks differ, providing an explicit identifiability constraint absent from SpecMRL.', 'An empirical design principle: task-conditional ordering yields larger gains over global ordering precisely when tasks have heterogeneous informative directions, and the contrastive loss margin controls the trade-off between task specialization and shared-basis stability.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | MTEB task suite (multi-task) | Matches SpecMRL baseline; spans diverse tasks |
| Primary metric | nDCG@10 at reduced dims (8, 16, 32, 64, 128, 256, 768) | Detects per-dimension retrieval quality where ordering matters |
| Baseline | SpecMRL | Tests whether task-conditional ordering beats global ordering |
| Baseline | Standard MRL | Tests whether spectral structure itself is needed |
| Baseline | Task-specific full-dim heads (no sharing) | Tests whether shared basis costs accuracy |
| Ablation-of-ours | TC-SpecMRL without contrastive ordering loss | Isolates contribution of identifiability constraint |
| Diagnostic | Spearman correlation between pre-argsort gate-score ordering and realized argsort permutation (per task, 512 examples) | Verifies that gate scores control the ordering (load-bearing assumption check) |
| Diagnostic | Correlation between task-embedding distance and true spectral divergence (normalized Kendall tau between top-32 empirical importance vectors, 512 examples per task) | Verifies that task embeddings encode spectral divergence |

### Why this setup validates the claim

The central claim is that making the spectral ordering an explicit, task-conditional function—plus a contrastive identifiability pressure—improves low-dimension readouts over a single global ordering, without sacrificing shared-basis efficiency. SpecMRL is the exact baseline that isolates this: it has the same spectral basis and nested-prefix machinery but a fixed global ordering, so any gain concentrated at aggressive compression dimensions is attributable to task-conditional reordering. Standard MRL tests the weaker sub-claim that spectral structure matters at all; if TC-SpecMRL only matches MRL, the spectral machinery is doing nothing. The task-specific full-dim baseline tests the cost of sharing: if we match it while sharing a basis, sharing is free. The no-ordering-loss ablation is the decisive control—it holds the gate and basis fixed and removes only the identifiability pressure, so it tells us whether the contrastive term, rather than the gate alone, drives the effect. nDCG@10 at reduced dimensions is the right metric because the predicted effect is a reordering of which directions survive prefix truncation; retrieval quality at small prefixes is precisely where a better ordering should show up, and it is insensitive to tail dimensions that the ordering does not touch. The two diagnostic checks directly test the load-bearing assumption: if the Spearman correlation between gate scores and realized permutation is below 0.8, the gate is not controlling the ordering; if the correlation between task-embedding distance and true spectral divergence is below 0.3, the task embedding does not encode spectral divergence. These diagnostics are reported alongside the main results to ground the linking assumptions empirically.

### Expected outcome and causal chain

**vs. SpecMRL** — On tasks whose informative directions are concentrated in spectral components that a global importance ranking pushes past the small-prefix cutoff, SpecMRL's fixed ordering truncates those directions and its low-dimension nDCG@10 drops sharply. Our method instead up-weights task-relevant components via the gated mask and reorders them into the prefix, because the gate uses the task embedding to rank by task-conditional importance. We expect a noticeable gap at aggressive prefixes (8–32) that shrinks toward full dimension, with parity or near-parity at the largest prefixes where ordering no longer matters. The diagnostic Spearman correlation between gate scores and realized permutation is expected to be above 0.8, confirming that the gate controls the ordering; the correlation between task-embedding distance and true spectral divergence is expected to be above 0.3, confirming that task embeddings encode spectral divergence.

**vs. Standard MRL** — On tasks where the learned representation's energy is spread across correlated dimensions, standard MRL's nested prefixes are not aligned to informative directions, so truncation discards signal and low-dimension nDCG@10 degrades. Our method instead projects onto an orthonormal spectral basis and reads out reordered coefficients, because the shared basis concentrates informative directions into leading components. We expect TC-SpecMRL to beat MRL by a moderate margin across reduced dimensions, with the gap largest where the basis alignment matters most.

**vs. Task-specific full-dim heads (no sharing)** — On a task with few training examples, a task-specific full-dim readout overfits and its reduced-dimension performance is unstable, because it has no shared structural prior. Our method instead shares the spectral basis across tasks and only learns a cheap gate, because the task embedding conditions a low-capacity MLP. We expect parity with the task-specific baseline at full dimension but a clear advantage at reduced dimensions and on low-resource tasks, indicating the shared basis regularizes rather than constrains.

**vs. TC-SpecMRL without contrastive ordering loss** — On a batch containing tasks with genuinely different spectra, the gated orderings collapse toward similar top-$K$ permutations because nothing penalizes overlap, so the gate learns a near-global ordering and low-dimension gains over SpecMRL vanish. Our method instead pushes task orderings apart via the Kendall-tau margin, because the contrastive term makes orderings identifiable across tasks. We expect the ablation to match ours on homogeneous task pairs but trail on heterogeneous pairs, with the gap concentrated at small prefixes. The diagnostic correlation between task-embedding distance and true spectral divergence is expected to be higher for our method than for the ablation, indicating that the contrastive loss encourages task embeddings to encode spectral divergence.

### What would falsify this idea

If TC-SpecMRL's gain over SpecMRL is uniform across all tasks and dimensions rather than concentrated at aggressive prefixes on tasks whose spectra diverge from the global ordering, the task-conditional ordering is not the causal mechanism and the central claim is wrong. Likewise, if removing the contrastive ordering loss leaves performance unchanged on heterogeneous task pairs, the identifiability constraint is inert and the design rationale fails. Additionally, if the diagnostic Spearman correlation between gate scores and realized permutation is below 0.8, or the correlation between task-embedding distance and true spectral divergence is below 0.3, the load-bearing assumption is violated and the method's ordering is not controlled by the task embedding as claimed.

## References

1. SpecMRL: Redesign Matryoshka Representation Learning within Parametric Spectrum
