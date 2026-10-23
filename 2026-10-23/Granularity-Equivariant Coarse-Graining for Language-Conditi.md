# Granularity-Equivariant Coarse-Graining for Language-Conditioned Latent World Models

## Motivation

Language-conditioned latent world models such as LaLa-World (and the broader trajectory toward free-form language as a universal conditioning interface) inherit a structural defect: the same physical transition admits arbitrarily many natural-language descriptions at different granularities ('move arm left' vs. 'translate end-effector 5cm along -x while keeping orientation'), yet nothing in the architecture forces the latent next-state prediction to depend on the underlying transition rather than on the phrasing lattice. Prior instruction-following work (AgentIF, FollowBench, CFBench) documents that models are highly phrasing-sensitive, and LaLa-World's limitation notes that robustness to phrasing is assumed rather than enforced; soft paraphrase-pair regularizers cannot cover the granularity lattice because the lattice is combinatorial in the number of describable attributes, so invariance must be a structural identity on the latent transition map rather than a statistical penalty.

## Key Insight

If a coarse-graining operator C exists on latent space that commutes with the language-conditioned transition map (C∘T = T∘C for all re-descriptions of the same transition), then phrasing granularity becomes a gauge degree of freedom of the latent dynamics rather than a nuisance the predictor must memorize, and this commutativity is a testable algebraic identity trainable without exhaustive paraphrase pairs.

## Method

**What it is.** GECO (Granularity-Equivariant Coarse-graining Operator) augments a LaLa-World-style language-conditioned latent predictor with a learned coarse-graining map C: Z→Z and enforces that the predicted latent transition commutes with C up to a re-description-invariant residual, using a granularity-lattice training signal built from descriptions at three canonical levels rather than from paraphrase pairs. The load-bearing assumption is stated explicitly: the L0/L1/L2 descriptions of a transition form a genuine coarsening lattice — i.e., the L2 description is exactly the composition of the L0 primitives and L1 subgoals, so that the true language-conditioned transition map T satisfies C∘T = T∘C exactly across levels. This assumption is not assumed silently; it is verified per-sample by a nesting-verification head and a held-out nesting check, and the commutativity constraint is gated by that verification so the method degrades gracefully to 'no constraint' where the lattice fails.

**How it works.**

```pseudocode
Inputs: encoder E (frozen RAE-style), LM-based transition predictor T_theta(z_t, u), coarse-graining C_phi: Z->Z (2-layer MLP, hidden=256, GeLU, linear output), nesting-verification head H_psi: (z_t, u_Ll, u_L2) -> g_l in [0,1] (2-layer MLP, hidden=128, GeLU, sigmoid output)
Canonical granularity levels: L0 (atomic: single degree of freedom), L1 (subgoal: 2-3 DoF), L2 (task: full transition)
Hyperparameters: beta=1.0 (equivariance weight), gamma=0.1 (coarse-graining regularizer), lambda_rank=0.01 (rank penalty), k=2-3 (L1 chunk count), tau_nest=0.5 (nesting gate threshold), N_cal=512 (calibration set size for rank target)

for each transition tau in batch:
    z_t   = E(o_t)
    z_t+1 = E(o_{t+1})                     # target latent next-state
    # 1. Build granularity lattice for tau (NOT paraphrases)
    u_L2  = describe(tau, level=L2)        # full description
    u_L1  = describe(tau, level=L1)        # subgoal decomposition (k=2-3 chunks)
    u_L0  = describe(tau, level=L0)        # atomic primitives
    # 2. Predict at each level
    z_hat_L2 = T_theta(z_t, u_L2)
    z_hat_L1 = C_phi(T_theta(z_t, u_L1))   # coarse-grain the subgoal prediction
    z_hat_L0 = C_phi(C_phi(T_theta(z_t, u_L0)))  # coarse-grain twice
    # 3. Nesting verification: predict whether L_l is a strict coarsening of L2 for this sample
    g_L1 = H_psi(z_t, u_L1, u_L2)          # in [0,1]
    g_L0 = H_psi(z_t, u_L0, u_L2)          # in [0,1]
    # 4. Gated equivariance loss: enforce commutativity only where nesting is verified
    L_eq = g_L1 * || z_hat_L2 - z_hat_L1 ||^2 + g_L0 * || z_hat_L2 - z_hat_L0 ||^2
    # 5. Reconstruction loss at each level (prevents C_phi collapsing to constant)
    L_rec = sum_{l in {L0,L1,L2}} || z_hat_l - z_t+1 ||^2
    # 6. Idempotence / non-triviality regularizer on C_phi
    L_C = || C_phi(C_phi(z)) - C_phi(z) ||^2  -  lambda_rank * rank_estimate(C_phi)
    # 7. Nesting-verification supervision (binary cross-entropy on held-out nesting labels)
    L_nest = BCE(g_L1, y_L1) + BCE(g_L0, y_L0)   # y_l = 1 if L_l is a strict coarsening of L2, else 0
    loss = L_rec + beta * L_eq + gamma * L_C + delta * L_nest   # delta=0.5
```

Hyperparameters: beta=1.0 (equivariance weight), gamma=0.1 (coarse-graining regularizer), lambda_rank=0.01 tuned so C_phi retains ≥80% effective rank on a calibration set of N_cal=512 held-out latents (rank target set by a small sweep over {0.001, 0.01, 0.1}); L1 chunk count k=2–3; L0 decomposition uses the simulator's primitive action vocabulary when available, otherwise the finest annotation level in the dataset; nesting gate threshold tau_nest=0.5 for reporting binary nesting decisions; delta=0.5 for nesting-verification loss.

**Why this design.** We chose a learned coarse-graining operator C_phi over a fixed pooling or projection because the latent space of a LaLa-World-style predictor is not axis-aligned with physical degrees of freedom, so a hand-specified pooling would destroy the transition information the predictor needs, accepting the cost that C_phi must be regularized against collapse (hence the idempotence and rank terms). We chose the granularity lattice (L0/L1/L2) over paraphrase pairs because paraphrase pairs sample the surface-form axis while the failure is on the granularity axis — the lattice is the actual structure of the re-description group, accepting the cost that lattice construction requires structured annotation or a simulator with primitive actions. We chose a commutativity residual (C∘T = T∘C) over a contrastive InfoNCE objective because commutativity is an algebraic identity that holds exactly for the true transition and is falsifiable per-sample, whereas contrastive objectives require negative sampling that reintroduces the coverage problem, accepting the cost that the residual is only as tight as C_phi's approximation. A domain expert would not describe GECO as a variant of paraphrase-robustness fine-tuning (e.g., the soft regularizers implied by LaLa-World's phrasing assumptions) because the training signal is a commutativity residual over a lattice of granularities, not a similarity penalty over surface paraphrases — the object being constrained is the transition map's algebraic relation to a coarse-graining operator, not the predictor's output distribution over rephrasings. The nesting-verification head H_psi and the gated equivariance loss are added to make the load-bearing lattice assumption explicit and testable: where the lattice fails (a coarser level introduces an attribute absent from the finer level), the gate g_l goes to zero and the method does not force a false identity, converting the failure mode from a silent confound into a measured quantity.

**Why it measures what we claim.** The gated commutativity residual `g_L1 * ||z_hat_L2 - C_phi(z_hat_L1)||^2 + g_L0 * ||z_hat_L2 - C_phi(C_phi(z_hat_L0))||^2` measures description-granularity invariance of the latent transition because the assumption that the L0/L1/L2 descriptions denote the same underlying transition and that the nesting-verification head correctly identifies strict coarsenings makes the true transition map satisfy C∘T = T∘C exactly on the gated subspace; this assumption fails when a coarser description is not a strict coarsening of the finer one (e.g., the L2 description introduces an attribute absent from the L0 decomposition), in which case the gate g_l should be low and the residual reflects genuine transition disagreement rather than a granularity artifact. The coarse-graining operator C_phi measures the re-description quotient map because the assumption that latent states differing only by described-vs-undescribed attributes are identified by C_phi makes C_phi the projection onto the description-invariant subspace; this assumption fails when C_phi collapses attributes that are physically relevant but happen to be omitted at a given level, in which case C_phi reflects an information-destroying projection rather than a quotient. The idempotence/rank regularizer measures non-triviality of the quotient because the assumption that a valid coarse-graining is a projection (C²=C) with non-degenerate image makes rank a proxy for retained transition information; this assumption fails when the latent manifold is curved such that a linear rank estimate underestimates effective dimension, in which case the regularizer reflects local curvature rather than collapse. The nesting-verification head H_psi measures the per-sample validity of the lattice assumption because the assumption that the binary nesting labels y_l are available on a calibration set and that H_psi generalizes to held-out transitions makes g_l a calibrated estimate of whether L_l is a strict coarsening of L2; this assumption fails when the nesting labels are noisy or the head overfits, in which case g_l reflects annotation noise rather than lattice validity.

## Contribution

(1) GECO, a coarse-graining operator with a commutativity-based training objective that enforces description-granularity invariance of a language-conditioned latent transition map as an algebraic identity rather than a paraphrase-pair regularizer. (2) A granularity-lattice construction procedure that generates L0/L1/L2 descriptions of the same transition from simulator primitive vocabularies or structured annotations, yielding a training signal that scales with the number of describable attributes rather than with the number of paraphrase pairs. (3) An empirical characterization of the trade-off between coarse-graining rank and equivariance residual, identifying the regime in which commutativity holds tightly enough to support MPC planning under re-described instructions.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | LaLa-World simulation + robot trajectories with L0/L1/L2 annotations | Provides granularity lattice and held-out transitions. |
| Primary metric | Granularity-consistency error: mean g_L1 * ||z_hat_L2 - C(z_hat_L1)||² + g_L0 * ||z_hat_L2 - C(C(z_hat_L0))||² on held-out transitions | Directly measures gated commutativity residual, the claimed invariant. |
| Co-primary metric | Shuffled-transition residual: same metric but with z_t+1 replaced by a random transition's target latent | Tests whether low residual is informative vs. trivially satisfiable. |
| Baseline | LaLa-World (paraphrase-augmented) | Tests whether granularity lattice adds beyond surface robustness. |
| Baseline | Fixed-pooling coarse-graining (PCA/mean-pool) | Tests whether learned C_phi is necessary vs. hand-specified pooling. |
| Baseline | No-coarse-graining (direct multi-level prediction) | Tests whether explicit quotient structure is needed. |
| Ablation-of-ours | GECO without idempotence/rank regularizer | Isolates the role of non-triviality regularization in preventing collapse. |
| Ablation-of-ours | GECO without nesting-verification head (g_l=1 always) | Isolates the role of gating in handling lattice failures. |

### Why this setup validates the claim

The central claim is that a learned coarse-graining operator C_phi, trained via a gated commutativity residual over a granularity lattice, yields a latent transition predictor that is invariant to description granularity — an algebraic property that paraphrase-based robustness cannot enforce. The dataset provides the lattice (L0/L1/L2 annotations) that makes the commutativity residual computable per-sample; the primary metric directly quantifies that residual on held-out transitions, so a low value is exactly the predicted effect. The co-primary shuffled-transition residual is critical: it measures whether the residual is meaningful (low on true transitions, high on shuffled ones) or trivially satisfiable (low on both), directly addressing the load-bearing assumption that the lattice is exact and C_phi is the exact quotient. The three baselines each test a distinct sub-claim: LaLa-World with paraphrase augmentation tests whether surface-form robustness already captures granularity invariance (if it does, our lattice is redundant); fixed-pooling coarse-graining tests whether the learned C_phi is necessary (if fixed pooling matches, the learning is unnecessary); no-coarse-graining tests whether explicit quotient structure matters at all (if direct multi-level prediction matches, the commutativity constraint is inert). The ablation without idempotence/rank regularizer isolates the regularizer's role: without it, C_phi may collapse to a constant, which would trivially satisfy commutativity but destroy reconstruction — the ablation should show exactly that failure, confirming the regularizer is load-bearing. The ablation without nesting-verification head isolates the gating's role: if the lattice is exact, gating should not hurt; if the lattice fails on some transitions, gating should improve the primary metric by not forcing a false identity. Together, these form a falsifiable test: the claim predicts a specific ordering (GECO < fixed-pooling < paraphrase-augmented ≈ no-coarse-graining on granularity-consistency error) concentrated on transitions where L0/L1/L2 descriptions genuinely coarsen, and a large gap between true and shuffled residuals for GECO but not for the unregularized ablation.

### Expected outcome and causal chain

**vs. LaLa-World (paraphrase-augmented)** — On a case where the L2 description says "move the cup to the shelf" while the L0 decomposition lists "grasp cup, lift, translate, release," the paraphrase-augmented baseline produces inconsistent latent predictions across these descriptions because its training signal only penalizes output differences among surface rephrasings of the same level, never across granularity levels. Our method instead enforces gated C∘T = T∘C across L0/L1/L2, so the coarse-grained subgoal prediction is pulled toward the full-transition prediction where the nesting head verifies a strict coarsening. We expect a noticeable gap on transitions with rich L0 decompositions but parity on transitions where L0 and L2 are nearly identical in content or where the nesting head correctly gates off the constraint.

**vs. Fixed-pooling coarse-graining (PCA/mean-pool)** — On a case where the transition involves a degree of freedom that is axis-aligned in the latent space but not aligned with the pooling direction, the fixed-pooling baseline produces a coarse-grained prediction that destroys the transition-relevant component because its projection is hand-specified and blind to the predictor's learned geometry. Our method instead learns C_phi jointly with the gated commutativity residual, so C_phi adapts to the latent manifold and preserves the subspace the predictor uses. We expect a gap that grows with the misalignment between the fixed pooling basis and the transition-relevant subspace, and near-parity on transitions whose relevant subspace happens to align with the pooling basis.

**vs. No-coarse-graining (direct multi-level prediction)** — On a case where the L0 and L1 descriptions are consistent coarsenings of L2 but the predictor is trained only to match z_{t+1} at each level independently, the no-coarse-graining baseline produces level-specific latent predictions that disagree after any reasonable coarse-graining because nothing ties the levels together algebraically. Our method instead makes the levels agree by construction through the gated commutativity residual, so the coarse-grained L1/L0 predictions align with L2 where the nesting head verifies a strict coarsening. We expect a large gap on transitions where the levels are informationally nested, and a smaller gap on transitions where the L0 decomposition introduces attributes absent from L2 (where the assumption genuinely fails, the gate should be low, and the residual should reflect real disagreement).

**vs. GECO without idempotence/rank regularizer (ablation)** — On a case where the training signal is dominated by the reconstruction loss, the unregularized ablation produces a C_phi that collapses toward a low-rank constant because collapsing trivially satisfies the commutativity residual while the reconstruction loss is carried by T_theta alone. Our method instead keeps C_phi non-degenerate via the idempotence and rank terms, so the commutativity residual remains informative. We expect the ablation to show a much higher rank deficiency in C_phi and a commutativity residual that is low but meaningless (near-zero on both true and shuffled transitions), whereas GECO's residual should be low on true transitions and high on shuffled transitions.

**vs. GECO without nesting-verification head (ablation)** — On a case where the L2 description introduces an attribute absent from the L0 decomposition (e.g., "shelf" in L2 but not in L0), the ungated ablation forces the commutativity residual to be small even though the true transition map does not commute, leading to a biased C_phi that absorbs the description gap into the latent projection. Our method instead uses the nesting head to gate off the constraint on such samples, so C_phi is not distorted by non-nested descriptions. We expect the ungated ablation to show a higher primary metric on transitions with known nesting failures (as identified by held-out nesting labels) and a lower gap between true and shuffled residuals, whereas GECO should maintain a clear separation.

### What would falsify this idea

If GECO's gain over the paraphrase-augmented baseline is uniform across all transitions rather than concentrated on those with rich L0/L1/L2 coarsening structure, or if the unregularized ablation achieves the same granularity-consistency error without rank collapse, or if the shuffled-transition residual is as low as the true-transition residual for GECO (indicating the residual is trivially satisfiable), then the central claim — that the lattice commutativity residual is the causal mechanism — is wrong.

## References

1. LaLa-World: Scaling Latent World Models with Natural Language Actions
2. AGENTIF: Benchmarking Instruction Following of Large Language Models in Agentic Scenarios
3. SysBench: Can Large Language Models Follow System Messages?
4. CFBench: A Comprehensive Constraints-Following Benchmark for LLMs
5. Instruction-Following Evaluation for Large Language Models
6. FollowBench: A Multi-level Fine-grained Constraints Following Benchmark for Large Language Models
7. Can LLMs Follow Simple Rules?
8. COLLIE: Systematic Construction of Constrained Text Generation Tasks
