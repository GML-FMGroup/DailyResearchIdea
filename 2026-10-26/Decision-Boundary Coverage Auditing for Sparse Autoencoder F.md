# Decision-Boundary Coverage Auditing for Sparse Autoencoder Feature Selection in Auditable Detectors

## Motivation

Existing auditable-detector pipelines, such as the one in 'Building Auditable AI Text Detectors with Sparse Autoencoders', select SAE features by a global per-feature reconstruction-fidelity threshold, which certifies only that each feature reconstructs its own activation direction well—not that the retained feature set spans the directions the detector's decision surface actually depends on. Because fidelity is a per-feature, decision-agnostic criterion, the selected concept basis can omit or under-weight directions that are individually low-fidelity yet jointly load-bearing for the logit margin, so the interpretable surrogate may agree with the detector on average while systematically failing on the subpopulation whose classification hinges on those uncovered directions. The structural root cause is that feature selection is decoupled from the downstream decision surface: nothing in the fidelity threshold audits whether the chosen features cover the detector's decision-relevant subspace.

## Key Insight

A feature set is adequate for an interpretable surrogate exactly when its span contains the detector's decision-relevant subspace, so coverage of that subspace—measured as residual projection of decision-boundary directions onto the feature span—is the criterion that fidelity silently substitutes for and fails to guarantee.

## Method

# Input: fine-tuned detector f, SAE (encoder E, decoder D) trained on f's residual stream,
#        validation set V = {(x_i, y_i)} with n=2000 examples, fidelity threshold tau_fid=0.7,
#        coverage threshold tau_cov=0.1 (Frobenius norm of residual basis), max greedy iterations=50

Phase 1 - Extract decision-relevant directions (distribution-robust subspace):
  # Load-bearing assumption (stated explicitly): The top-k principal directions of the |margin|-weighted
  # gradient Gram matrix G^T W G estimated on the validation set constitute a stable, detector-intrinsic
  # decision-relevant subspace S that the SAE decoder directions can meaningfully be projected onto and
  # that transfers to the OOD generator split. This assumption is verified (not assumed) via the
  # subspace stability audit in Phase 1c.
  for each x_i in V:
      h_i = residual stream activation at detector's readout layer
      g_i = grad_{h} [ f_logit(x_i) ]  # gradient of detector logit wrt h
      m_i = f_logit(x_i)               # margin, signed
  Build direction matrix G = [g_i] (n x d), weighted by |m_i| to emphasize confident boundary points
  # Compute per-cluster subspaces to avoid dominance by high-margin outliers:
  Cluster V into C=5 clusters via k-means on h_i (standardized, k=5 chosen by silhouette score on V)
  For each cluster c, compute weighted Gram G_c^T W_c G_c, take top-k_c principal directions
      where k_c chosen so that components explain >= 0.90 of weighted variance within cluster c
      (k_c <= 10; if variance criterion not met, use k_c=10)
  Concatenate all cluster bases into a candidate matrix U_cat (d x sum_c k_c)
  Re-orthonormalize U_cat via QR decomposition to obtain orthonormal basis U in R^{d x k}
      where k = rank(U_cat) (k <= sum_c k_c)
  # Phase 1c: Subspace stability audit (explicit verification):
  Compute principal angles between U (in-distribution) and U_ood (computed identically on a held-out
      generator split of size 500) using SVD of U^T U_ood; report the top-5 angles in degrees.
  If the maximum principal angle > 30 degrees, flag the subspace as distribution-sensitive and
      report this as an audit artifact; the coverage claim is then conditioned on this measured stability.

Phase 2 - Per-feature fidelity (baseline filter, retained as a necessary gate):
  for each SAE feature j:
      fid_j = 1 - || h - D_j E_j h ||^2 / ||h||^2 averaged over V  # standard reconstruction fidelity
  F_fid = { j : fid_j >= tau_fid }

Phase 3 - Decision-boundary coverage audit:
  for each feature j in F_fid:
      d_j = D_j (unit decoder direction in activation space)
      # coverage contribution: how much of S this feature's direction reconstructs
      proj_j = || U^T d_j ||^2 / ||d_j||^2      # fraction of decision subspace aligned with feature
      cov_j = proj_j * fid_j                    # fidelity-weighted coverage
  Greedy selection (subspace coverage maximization):
      Selected = empty; Resid = U  # residual basis of S not yet covered
      iter = 0
      while ||Resid||_F > tau_cov and iter < 50:
          # Tie-breaking: if multiple j achieve within 1e-6 of max score, pick the one with highest fid_j
          j* = argmax_{j in F_fid \ Selected} || Resid^T d_j ||^2 / ||d_j||^2
          Selected = Selected U {j*}
          Resid = Resid - Resid (d_{j*} d_{j*}^T)/||d_{j*}||^2   # project out covered direction
          iter += 1
      # Report final ||Resid||_F and iteration count as audit artifacts
  Output: Selected feature set (coverage-audited concept basis)

Phase 4 - Interpretable surrogate:
  Train logistic regression (L2 penalty, C=1.0, lbfgs solver, max_iter=1000) on activations of Selected
      features over V
  Report per-feature cov_j and residual coverage ||Resid||_F as audit artifacts
  # Diagnostic for linearity/additivity assumption (see block D):
  Compute joint least-squares span-coverage estimate: for each feature j, solve min_{alpha} || U - D_j alpha ||_F^2
      and record the residual norm; compare this joint estimate to the greedy per-feature coverage to detect
      when decoder directions jointly span S through cancellation (if joint residual < 0.5 * greedy residual,
      flag potential cancellation).

## Contribution

(1) A decision-boundary coverage audit that replaces per-feature fidelity ranking with a set-level subspace-coverage selection criterion, operationalized as greedy residual projection of gradient-derived decision directions onto SAE decoder directions. (2) The finding that fidelity and coverage are empirically dissociable—feature sets matched on fidelity can differ substantially in decision-subspace coverage, and coverage-audited sets yield interpretable surrogates whose agreement with the detector is more uniform across boundary-proximal subpopulations. (3) An audit artifact (per-feature coverage scores and residual coverage norm) that makes explicit which decision directions the interpretable concept basis fails to span.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | AI-text detection with held-out OOD generator split (n=2000 validation, 500 OOD) | Tests whether concept basis transfers across generators |
| Primary metric | Surrogate-vs-detector decision agreement (AUC gap) | Directly measures boundary fidelity, not raw accuracy |
| Baseline | Original fine-tuned neural detector | Opaque upper bound on boundary behavior |
| Baseline | Fidelity-threshold feature selection (prior pipeline) | Isolates value of coverage audit over fidelity ranking |
| Baseline | Stylometric classifier | Handcrafted-feature reference for interpretability tradeoff |
| Baseline | Logit-attribution-ranked top-N selection | Natural alternative selector using per-feature attribution |
| Baseline | Orthogonal matching pursuit (OMP) over decoder directions | Natural alternative selector using greedy span coverage |
| Baseline | Supervised sparse probe (L1 logistic regression on SAE features) | Natural alternative selector using supervised feature selection |
| Ablation-of-ours | Coverage audit without fidelity gate | Tests whether the fidelity gate is load-bearing |
| Ablation-of-ours | Single in-distribution subspace (no per-cluster aggregation) | Tests whether distribution-robust subspace is necessary |

### Why this setup validates the claim

The central claim is that selecting SAE features by *set-level coverage of the gradient-derived decision subspace* yields a surrogate that agrees with the detector's decisions better than per-feature fidelity ranking, especially where the boundary is spanned by redundant or individually-weak features. The fidelity-threshold baseline is the critical comparator: it shares our SAE, our detector, and our surrogate training, differing only in selection signal, so any agreement gap is attributable to the coverage criterion rather than confounds. The opaque detector establishes the ceiling of achievable agreement, and the stylometric baseline shows whether the gain is merely "interpretable enough" or genuinely boundary-faithful. The additional baselines (logit-attribution top-N, OMP, sparse probe) test whether the coverage criterion is merely a re-derivation of a known selector: if our method matches or exceeds these on near-boundary OOD examples, the coverage audit is not redundant. The OOD generator split is where the predicted failure mode of fidelity ranking should concentrate: fidelity is estimated on validation activations and does not certify boundary span, so a fidelity-ranked set can look reliable in-distribution yet miss a decision direction that only dominates on shifted inputs. Decision agreement, not classification accuracy, is the right metric because it isolates whether the surrogate reproduces the detector's *boundary geometry* rather than coincidentally matching labels; accuracy can stay high while the surrogate keys on the wrong directions. The no-fidelity-gate ablation tests the secondary claim that the gate is necessary rather than merely harmless. The single-subspace ablation tests whether the distribution-robust subspace (per-cluster aggregation) is necessary for OOD transfer, directly addressing the load-bearing assumption.

### Expected outcome and causal chain

**vs. Fidelity-threshold feature selection (prior pipeline)** — On a case where the detector's boundary on OOD inputs is carried by a direction that several individually-mediocre-fidelity features jointly span, the fidelity-ranked baseline selects high-fidelity features that reconstruct activations well but point off-boundary, so its surrogate misclassifies near-boundary OOD examples because its selection signal never checked whether the chosen set spans the decision subspace. Our method instead greedily projects out covered directions and keeps adding features until residual coverage falls below tau_cov, so it retains the jointly-spanning set even at lower individual fidelity, and we expect a noticeable agreement gap concentrated on near-boundary OOD examples with parity on easy in-distribution examples.

**vs. Original fine-tuned neural detector** — On a case where the detector is strongly nonlinear near the boundary, the opaque detector remains the reference, but our surrogate is a linear readout over selected features, so it will diverge on examples where the detector's decision flips under small perturbations the linear surrogate cannot represent; this is the accepted cost of a linear surrogate, not a selection failure. We expect our method to track the detector closely on the bulk of the distribution and to show its largest disagreement precisely in the high-curvature band, which we read as the surrogate's expressiveness limit rather than a coverage defect — distinguishing the two is why we report residual coverage alongside agreement.

**vs. Stylometric classifier** — On a case where stylistic cues (e.g., uniform sentence length) correlate with labels in-distribution but are absent or inverted in a shifted generator, the stylometric baseline degrades sharply because its handcrafted features were never tied to this detector's boundary at all. Our method derives directions from the detector's own gradients, so its selected features are anchored to that specific decision surface, and we expect a clear OOD advantage over stylometry with roughly comparable in-distribution behavior, confirming the gain comes from boundary-anchoring rather than from feature richness.

**vs. Logit-attribution-ranked top-N selection** — This baseline ranks features by the magnitude of their average logit attribution (gradient times activation) and selects the top-N. It is local and per-feature, so it may pick features that are individually high-attribution but redundant in direction. Our coverage audit explicitly projects out covered directions, so it should select a more diverse set that spans the decision subspace. We expect our method to outperform this baseline on near-boundary OOD examples where the boundary is spanned by multiple directions, with parity on easy examples.

**vs. Orthogonal matching pursuit (OMP) over decoder directions** — OMP greedily selects features to minimize the residual of representing the decision subspace U, but it uses the full least-squares residual rather than the fidelity-weighted coverage and does not include the fidelity gate. We expect OMP to perform similarly to our method in-distribution but to be less robust on OOD examples because it may select low-fidelity features that reconstruct U well but are noisy. Our fidelity gate should provide an advantage on near-boundary OOD examples where reliability matters.

**vs. Supervised sparse probe (L1 logistic regression on SAE features)** — This baseline directly trains a sparse linear classifier on all SAE features with L1 regularization, selecting features that are predictive of the label. It is supervised and may overfit to spurious correlations in the training distribution. Our method selects features based on geometric coverage of the decision subspace, which is label-agnostic and should be more robust to distribution shift. We expect our method to show better OOD agreement, especially on near-boundary examples where the sparse probe may rely on non-boundary directions.

### What would falsify this idea

If the coverage-audited selection's agreement advantage over fidelity-threshold selection is uniform across near-boundary and easy subsets rather than concentrated on the near-boundary OOD cases where fidelity ranking is predicted to miss spanning directions, the claim that coverage captures something fidelity cannot is wrong. Likewise, if removing the fidelity gate leaves agreement unchanged, the gate is not load-bearing and the two-hyperparameter design is unjustified. Additionally, if the subspace stability audit shows large principal angles (>30 degrees) between in-distribution and OOD subspaces and the agreement advantage disappears, the load-bearing assumption of subspace transfer is falsified.

## References

1. Building Auditable AI Text Detectors with Sparse Autoencoders
