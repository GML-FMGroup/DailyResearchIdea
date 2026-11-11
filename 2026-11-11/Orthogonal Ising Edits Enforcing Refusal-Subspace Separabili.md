# Orthogonal Ising Edits: Enforcing Refusal-Subspace Separability in Bit-Level Weight Repair

## Motivation

Bit-level edit methods such as Tree-Ising repair targeted loop/conclude directions by solving per-row QUBO problems over admissible BF16 spin flips, but they assume—without enforcing—that the edit subspace is separable from other behavior-mediating directions. Because the same residual stream carries both reasoning directions and the refusal direction, a spin-flip configuration that minimizes the loop/conclude objective can acquire nonzero projection onto the refusal direction, silently perturbing safety behavior. This is the meta-level structural gap: across knowledge-neuron and refusal-direction branches, no edit method establishes functional isolation of the edited subspace from co-resident behavior directions, so separability remains a hope rather than a constraint.

## Key Insight

Adding a projection penalty on the refusal direction to the Ising objective makes refusal-orthogonality a hard constraint satisfied by construction of the QUBO solution, so edited weights cannot acquire refusal-direction components that the unconstrained objective would freely admit.

## Method

**What it is.** Refusal-Orthogonal Ising Editing (ROIE) augments Tree-Ising's per-row QUBO with a projection penalty that keeps the bit-level spin-flip perturbation's *activation-space effect* orthogonal to a refusal subspace estimated once from the model, returning edited BF16 weights whose reasoning repair does not project onto the refusal subspace.

**How it works.**
```pseudocode
INPUT: model M (BF16), target layers L, calibration prompts P_cal (n=512), refusal probes P_ref (n=512)
OUTPUT: edited weights W' with reasoning repair and preserved refusal

1. Estimate refusal subspace R_l per target layer l in L:
     for each p in P_ref: h_l(p) = residual-stream activation at layer l (last token)
     for each p in P_cal: h_l(p) = residual-stream activation at layer l (last token)
     Compute PCA on the concatenated set {h_l(p) : p in P_ref ∪ P_cal}.
     Select top-k=3 principal components with largest absolute mean difference
     between P_ref and P_cal; orthonormalize to form R_l ∈ ℝ^{d×k}.
     (subspace frozen for all subsequent steps)

2. For each output row i in layer l, build Tree-Ising objective:
     E_i(s) = lambda_loop * <loop_dir, W_i + delta(s)>^2
             + lambda_conc * <conc_dir, W_i + delta(s)>^2
             + lambda_pres * || (W_i + delta(s)) x_i - y_i^prod ||^2
             + lambda_orth * (1/|P_cal|) * Σ_{x in P_cal} || R_l^T (delta(s) @ x) ||^2   # NEW activation-space projection penalty
   where delta(s) enumerates admissible sign flips / exponent shifts
   from the state tree; x is the input activation to row i from calibration prompts;
   hyperparameters lambda_loop=1.0, lambda_conc=1.0, lambda_pres=0.5, lambda_orth=10.0 (orthogonality prioritized).

3. Solve each per-row QUBO with simulated annealing on GPU
   (1000 sweeps, geometric cooling 10.0 -> 0.01).

4. Apply edits: W'_i = W_i + delta(s*_i).

5. Validation gate: for each layer, compute activation-space projection
     pi_l = (1/|P_held|) * Σ_{x in P_held} || R_l^T ( (W'_l - W_l) @ x ) || / || (W'_l - W_l) @ x ||
   on held-out probes P_held (n=256, disjoint from P_cal and P_ref),
   and reject the layer edit if pi_l > tau (tau = 0.05).
```

**Why this design.** We chose a refusal subspace estimated from contrastive calibration/refusal probes via PCA over a per-token learned refusal classifier because refusal is known to be multi-directional (Wollschlager et al., 2025) and a single mean-difference vector may not capture the full refusal cone; we accept the cost that k=3 may still under-represent the true refusal geometry. We chose an activation-space quadratic projection penalty inside the existing per-row QUBO over a weight-space penalty or a post-hoc projection of edited weights because the QUBO solver then avoids refusal-aligned *output* effects during search rather than requiring a second corrective pass, and because weight-space orthogonality does not imply activation-space orthogonality for inputs with large row-space components; we accept the cost that lambda_orth must be tuned to avoid over-constraining the reasoning objective. We chose a hard validation gate (tau=0.05) over trusting the penalty alone because the penalty is soft and the state tree is discrete, accepting the cost that some rows may be left unedited when no orthogonal admissible edit exists. A domain expert would not describe ROIE as a variant of Tree-Ising: Tree-Ising optimizes only reasoning directions and leaves the refusal subspace unconstrained, whereas ROIE introduces a new constraint surface (the refusal subspace) that changes which spin-flip configurations are admissible and adds a rejection criterion absent from Tree-Ising.

**Why it measures what we claim.** The activation-space projection ratio pi_l measures refusal-subspace leakage of the edit because the assumption that the refusal subspace R_l (top-k PCA directions) linearly mediates refusal behavior makes the projection of the edit's output effect onto R_l equivalent to the edit's refusal-relevant component; this assumption fails when refusal is mediated by nonlinear or multi-directional features outside the top-k subspace, in which case pi_l reflects only linear leakage and may under-report true refusal perturbation. The penalty term lambda_orth * (1/|P_cal|) Σ || R_l^T (delta @ x) ||^2 measures separability of the edit subspace from the refusal subspace because the assumption that admissible bit-level perturbations span the edit subspace makes penalizing their refusal projection equivalent to enforcing orthogonality; this assumption fails when the state tree cannot realize an orthogonal edit for a given row, in which case the term reflects an unattainable constraint rather than achieved separability. The validation gate tau measures preservation of refusal behavior because the assumption that small activation-space projection implies small behavioral change makes the threshold a proxy for stability; this assumption fails when refusal behavior is threshold-like, in which case a small projection may still cross a behavioral boundary.

## Contribution

(1) A refusal-orthogonal Ising editing algorithm (ROIE) that embeds a projection penalty on an estimated refusal direction directly into the per-row QUBO, making refusal-orthogonality a search-time constraint rather than a post-hoc check. (2) A design principle—frozen contrastive refusal direction plus hard projection gate—that operationalizes edit-subspace separability for bit-level weight repair and can be layered onto any QUBO-based editing method. (3) A projection-leakage diagnostic (pi_l) that quantifies how much a reasoning repair perturbs the refusal subspace, enabling layer-level accept/reject decisions.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | MATH-500 | Same benchmark as Tree-Ising and Astra edit |
| Primary metric | MATH-500 accuracy with refusal-preservation check | Jointly detects reasoning repair and refusal leakage |
| Baseline | Astra agent edit | Reference edit with known 51.7% reasoning repair |
| Baseline | Tree-Ising (unconstrained) | Tests whether orthogonality penalty is needed |
| Baseline | ROIE without validation gate | Tests whether hard rejection adds anything |
| Baseline | Post-hoc projection of Tree-Ising delta onto orthogonal complement of R_l | Tests whether in-search penalty beats post-hoc correction |
| Ablation-of-ours | ROIE with lambda_orth = 0 | Isolates contribution of projection penalty |

### Why this setup validates the claim

The central claim is that adding a refusal-subspace orthogonality constraint to Tree-Ising's per-row QUBO lets reasoning repair proceed without leaking into the refusal subspace. MATH-500 is the right dataset because it is the exact benchmark on which the Astra agent and Tree-Ising were compared, so any accuracy movement is directly attributable to the editing mechanism rather than to task shift. The Astra agent edit anchors the upper end of achievable reasoning repair, while unconstrained Tree-Ising tests whether reasoning-only optimization already leaks refusal-relevant weight mass. The lambda_orth = 0 ablation is the critical control: it removes only the projection penalty while keeping the solver, hyperparameters, and validation gate identical, so any difference between it and full ROIE is caused by the orthogonality constraint itself. The post-hoc projection baseline tests whether the in-search penalty yields strictly better reasoning-repair/refusal-preservation Pareto points than simply projecting the unconstrained Tree-Ising delta onto the orthogonal complement of R_l after solving. The validation gate variant tests whether the soft penalty suffices or whether discrete state-tree limits require hard rejection. Together these arms form a falsifiable test: if ROIE's reasoning gain is real and its refusal preservation is caused by the projection penalty, then full ROIE should match unconstrained Tree-Ising on MATH-500 while showing markedly lower projection ratios and preserved refusal on probe prompts, and the lambda_orth = 0 ablation should recover the leakage pattern.

### Expected outcome and causal chain

**vs. Astra agent edit** — On a case where the agent's edit improves MATH-500 reasoning but shifts residual-stream activations along the refusal subspace, the baseline produces repaired reasoning with degraded refusal behavior because its edit was selected without any refusal-subspace constraint. Our method instead searches over the same admissible bit flips but penalizes refusal-aligned activation-space components during annealing, so the selected edit reaches comparable reasoning repair while keeping its output effect orthogonal to R_l. We expect ROIE to land near the Astra agent's MATH-500 accuracy (within a few points) while showing substantially lower projection ratios and intact refusal on held-out refusal probes.

**vs. Tree-Ising (unconstrained)** — On a case where the per-row QUBO finds a spin-flip configuration that strongly reduces loop/concordance terms but happens to align with the refusal subspace, the baseline produces high MATH-500 accuracy with measurable refusal leakage because its objective has no term penalizing R_l^T (delta @ x). Our method instead adds the activation-space projection penalty with lambda_orth = 10.0, which raises the energy of those refusal-aligned configurations so the annealer selects a different, orthogonalizing edit. We expect a small accuracy gap on MATH-500 (ROIE slightly below or at parity with unconstrained Tree-Ising) but a clear separation on projection ratio and refusal-probe preservation, concentrated in layers where unconstrained edits show high pi_l.

**vs. ROIE without validation gate** — On a case where the state tree for a row contains no admissible edit with projection below tau, the ungated variant produces an edit that passes the soft penalty but still leaks because the penalty is quadratic and finite. Our method instead rejects that layer edit when pi_l > 0.05, leaving those rows unedited. We expect the gated variant to show lower worst-case projection ratios and better refusal preservation on adversarial probes, at the cost of slightly weaker MATH-500 repair on the small subset of rows where no orthogonal edit exists — a gap that should be localized rather than uniform.

**vs. Post-hoc projection of Tree-Ising delta** — On a case where the unconstrained Tree-Ising delta has a large component along R_l, the post-hoc baseline projects it onto the orthogonal complement after solving, which may distort the reasoning-repair directions and yield a worse accuracy/refusal trade-off because the projection is not optimized jointly with the reasoning objective. Our method instead penalizes refusal-aligned components during annealing, so the solver directly searches for edits that are both reasoning-effective and orthogonal. We expect ROIE to achieve strictly better Pareto points (higher MATH-500 accuracy at matched projection ratio, or lower projection ratio at matched accuracy) than the post-hoc baseline.

### What would falsify this idea

If ROIE's projection ratios and refusal-preservation gains over unconstrained Tree-Ising are uniform across all layers rather than concentrated in the layers where unconstrained edits exhibit high pi_l, then the projection penalty is not actually selecting orthogonal edits and the central causal claim fails. Similarly, if the lambda_orth = 0 ablation matches full ROIE on both MATH-500 accuracy and refusal preservation, the orthogonality constraint contributes nothing and the idea is falsified.

## References

1. Tree-Ising: Repairing Language Models with Bit-Level Ising Edits
