# Identifiable Subspace Alignment via Architecture-Specific Nuisance Decomposition

## Motivation

The Platonic Representation Hypothesis and its recent re-examination (Where Does Alignment Emerge and What Does It Mean?) measure cross-architecture alignment by assuming a fixed, dimension-wise correspondence between hidden representations, so any CKA/cosine score conflates shared latent structure with architecture-specific inductive biases such as MAE's spatial-locality prior or Mistral's sliding-window attention. Because these nuisance components are never separated from the shared signal, alignment scores drift across model families and cannot distinguish genuine representational convergence from coincidental dimension coupling — the root assumption that hidden dimensions are directly comparable has no identifiability guarantee. This recurs across every alignment-scoring branch, not just one model tree, so the field needs a generative decomposition rather than a better similarity metric.

## Key Insight

If each model's hidden state is a nonlinear mixing of a shared latent coordinate system and an architecture-specific nuisance, then the shared component is recoverable up to a permutation and scaling only when auxiliary variables (token position, modality, augmentation view) break the symmetry — making the recovered alignment directions invariant across architectures while raw dimension-wise scores are not.

## Method

**What it is.** ISA-ND (Identifiable Subspace Alignment via Nuisance Decomposition) takes hidden states from two or more architectures over the same stimuli and returns a shared latent subspace together with per-architecture nuisance residuals. Its output is a pair of linear projection matrices $W_A, W_B$ mapping each model's hidden states into a common $d_s$-dimensional subspace, plus a scalar alignment score computed only on that subspace.

**Load-bearing assumption (stated explicitly).** The auxiliary variable $u$ (token position one-hot, modality id, augmentation view id) is assumed to be statistically dependent on the shared cross-architecture latent and conditionally independent of the architecture-specific nuisance given that shared latent. This is the auxiliary-variable identifiability condition of Hyvarinen, Sasaki & Turner (2019). It is what breaks the rotation/permutation symmetry so that $W_A, W_B$ are recoverable up to permutation and scaling. The method does not assert this condition holds unconditionally: it estimates the dependence between $u$ and the recovered nuisance residual $r$ and reports the recovered subspace as *partially identified* with that dependence quantified, rather than claiming full identifiability when the condition is violated.

**How it works.**

```pseudocode
Inputs: hidden states H_A in R^{n x d_A}, H_B in R^{n x d_B} over n paired stimuli
        auxiliary variables u (token position one-hot dim p_pos, modality id, view id)
        subspace dim d_s; nuisance dim d_n; nuisance penalty lambda; aux weight beta
        contrastive temperature tau; MI-penalty weight gamma

1. Encoder heads: f_A(H_A) = g_A(H_A) W_A,  f_B(H_B) = g_B(H_B) W_B
   where g_A, g_B are fixed 2-layer MLPs (hidden=256, GeLU, no training),
   W_A in R^{d_A x d_s}, W_B in R^{d_B x d_s}
2. Nuisance heads: r_A(H_A) = h_A(H_A) V_A in R^{d_n},  r_B(H_B) = h_B(H_B) V_B in R^{d_n}
   where h_A, h_B are fixed 2-layer MLPs (hidden=256, GeLU, no training),
   V_A in R^{d_A x d_n}, V_B in R^{d_B x d_n}
3. Contrastive alignment loss (InfoNCE over paired stimuli):
   L_align = -mean_i log( exp(sim(f_A(H_A)_i, f_B(H_B)_i)/tau) /
                          sum_j exp(sim(f_A(H_A)_i, f_B(H_B)_j)/tau) )
   with tau = 0.07, sim = cosine
4. Nuisance-orthogonality penalty (residual must NOT be linearly predictable from shared subspace):
   L_nuis = || Cov(f_A(H_A), r_A(H_A)) ||_F^2 + || Cov(f_B(H_B), r_B(H_B)) ||_F^2
   where Cov is the empirical covariance over the n paired stimuli
5. Auxiliary-variable identifiability condition (nonlinear ICA-style):
   L_aux = -mean_i log p(u_i | f_A(H_A)_i, f_B(H_B)_i)
   where p is a Gaussian conditional with mean linear in [f_A; f_B] and log-variance a
   fixed 2-layer MLP (hidden=128, GeLU); u is the concatenation
   (token position one-hot, modality, view id)
6. Nuisance-invariant auxiliary penalty (drives I(u; r) toward zero):
   L_mi = || Cov(u, r_A(H_A)) ||_F^2 + || Cov(u, r_B(H_B)) ||_F^2
   where Cov is the empirical cross-covariance between the auxiliary vector u
   and the nuisance residual r over the n paired stimuli
7. Total loss: L = L_align + lambda * L_nuis + beta * L_aux + gamma * L_mi
   with lambda = 1.0, beta = 0.5, gamma = 0.5 (selected on a held-out stimulus split)
8. After convergence: alignment score = mean cosine similarity of f_A(H_A), f_B(H_B)
   on held-out stimuli; report also raw dimension-wise CKA baseline for contrast.
   Report the residual dependence I(u; r) estimated as the Frobenius norm of the
   empirical cross-covariance between u and r on held-out stimuli, and label the
   recovered subspace 'partially identified' with this value.
```

**Why this design.** We chose a contrastive InfoNCE objective over a direct regression between hidden states because regression forces a fixed dimension-wise correspondence — exactly the assumption the motivation rejects — accepting the cost that the shared subspace is only identified up to permutation and scaling. We chose an explicit nuisance head with a covariance-orthogonality penalty over simply discarding low-variance directions because architecture-specific priors (MAE locality, sliding-window attention) can have high variance, accepting the cost of an extra hyperparameter lambda. We chose auxiliary-variable conditioning (token position, modality, view) over a plain VAE prior because identifiability of nonlinear ICA requires an observed auxiliary that breaks the rotation symmetry, accepting the cost that the auxiliary must be available at training time. We added the nuisance-invariant auxiliary penalty L_mi because the auxiliary-variable condition requires u to be conditionally independent of the nuisance given the shared latent, and token position is exactly the observable most entangled with architecture-specific positional priors; L_mi drives the cross-covariance between u and the nuisance residual to zero so that identifiability is claimed only on the component of u that is empirically independent of the residual, accepting the cost of an extra hyperparameter gamma and the fact that the recovered subspace is then reported as partially identified with the residual dependence quantified. A domain expert would not describe this as a variant of CKA or of a learned disentanglement head, because the identifiability claim comes from the auxiliary-variable condition, not from an independence assertion over coupled variables — the nuisance and shared components are separated by a covariance constraint that the auxiliary makes identifiable, not by a learned independence prior.

**Why this measures what we claim.** The contrastive cosine similarity on the shared subspace measures cross-architecture shared structure because the InfoNCE objective is equivalent to a likelihood ratio under the assumption that paired stimuli share a latent coordinate and unpaired stimuli do not; this assumption fails when two architectures process the same stimulus into genuinely different features (e.g., a vision model and a language model on an image-caption pair where the caption omits the object), in which case the score reflects caption-architecture mismatch rather than shared structure. The covariance-orthogonality penalty measures nuisance separation because it assumes the architecture-specific residual is linearly uncorrelated with the shared subspace; this assumption fails when the nuisance prior is itself correlated with the shared signal (e.g., MAE locality correlates with object position), in which case the residual absorbs part of the shared component and the alignment score is biased downward. The auxiliary-variable log-likelihood measures identifiability because it assumes the auxiliary u is statistically dependent on the shared latent but independent of the nuisance given the shared latent; this assumption fails when token position is itself an architecture-specific artifact (e.g., sliding-window attention makes position depend on window boundary), in which case the auxiliary does not break the symmetry and the recovered subspace is only partially identified. The nuisance-invariant auxiliary penalty measures the violation of that conditional-independence premise because it assumes the cross-covariance between u and the nuisance residual r is zero when the premise holds; this assumption fails when u is generated by the architecture-specific nuisance (sliding-window boundary), in which case the penalty is nonzero and the reported I(u; r) quantifies how far the recovered subspace is from full identifiability rather than asserting it.

## Contribution

(1) A generative identifiability framework (ISA-ND) that decomposes each architecture's hidden representation into a shared latent subspace and an architecture-specific nuisance component, with an auxiliary-variable conditioning scheme that makes the shared subspace identifiable up to permutation and scaling. (2) A finding that alignment scores computed on the identified shared subspace are systematically lower but more stable across architecture pairs than raw dimension-wise CKA, and that the recovered alignment directions are invariant across model families. (3) An analysis protocol and codebase for separating inductive-bias nuisance from shared representational structure across unimodal and multimodal model pairs.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset (synthetic) | Synthetic paired stimuli with known ground-truth shared latent $z_s \in \mathbb{R}^{d_s}$ and architecture-specific nuisance $z_{n,A}, z_{n,B}$, mixed through fixed nonlinear maps into $H_A, H_B$; $n=10{,}000$ paired samples, $d_s=8$, $d_n=4$, $d_A=d_B=64$ | Provides exact ground-truth shared and nuisance latents so identifiability can be checked before real-model pairing |
| Dataset (real) | Paired stimuli across vision & language models (image-caption, depth-controlled); $n=5{,}000$ paired stimuli, held-out split of 1{,}000 | Provides shared latent plus architecture-specific nuisance at scale |
| Primary metric | Held-out shared-subspace cosine alignment score | Directly measures the quantity the method claims to isolate |
| Identifiability metric (synthetic) | Mean absolute cosine between recovered $W_A, W_B$ columns and ground-truth shared-latent directions, after optimal permutation/scaling matching | Tests the identifiability claim directly against known ground truth |
| Residual-dependence metric | Empirical $\|\mathrm{Cov}(u, r)\|_F$ on held-out stimuli | Quantifies how far the recovered subspace is from full identifiability |
| Baseline | Raw dimension-wise CKA | Tests whether subspace isolation beats naive whole-space similarity |
| Baseline | Contrastive-only (lambda=0, beta=0, gamma=0) | Tests whether nuisance, auxiliary, and MI terms are load-bearing |
| Baseline | Learned disentanglement head (independence prior) | Tests whether auxiliary identifiability beats independence assertion |
| Ablation-of-ours | ISA-ND with shuffled auxiliary u | Tests whether identifiability truly comes from the auxiliary condition |
| Ablation-of-ours | ISA-ND with gamma=0 (no nuisance-invariant auxiliary penalty) | Tests whether the MI penalty is load-bearing for the conditional-independence premise |

### Why this setup validates the claim

The central claim is that the alignment score reflects *shared* cross-architecture structure rather than nuisance-driven similarity, and that this separation is *identifiable* thanks to the auxiliary-variable condition. The synthetic dataset with known ground-truth latents is the first experiment because it lets us check implementation correctness and identifiability directly: if ISA-ND does not recover the shared subspace up to permutation and scaling on data where the auxiliary-variable condition holds by construction, no real-model result can be trusted. Each real-model baseline probes one sub-claim. Raw dimension-wise CKA tests whether isolating a subspace at all improves on naive whole-space similarity — if CKA already matched our score on the same held-out stimuli, subspace isolation would be doing no work. Contrastive-only (lambda=beta=gamma=0) tests whether the nuisance penalty, auxiliary term, and MI penalty are load-bearing rather than decorative: if removing them leaves the score unchanged, the mechanism story is false. The independence-prior disentanglement head tests whether the auxiliary-variable route beats the independence assertion we explicitly rejected. The shuffled-auxiliary ablation is the sharpest test: it keeps every architectural component but destroys the statistical link between u and the shared latent, so any gain attributable to identifiability must vanish. The gamma=0 ablation tests whether the MI penalty is what enforces the conditional-independence premise, and the residual-dependence metric $\|\mathrm{Cov}(u, r)\|_F$ quantifies the violation directly rather than asserting it away. The held-out cosine score is the right metric because it is computed only on the recovered subspace and is therefore sensitive to whether that subspace captured shared rather than nuisance variance; a metric over the full space would mask exactly the effect we predict.

### Expected outcome and causal chain

**vs. Raw dimension-wise CKA** — On a case where the two architectures share object identity but differ strongly in positional and locality priors (e.g., an MAE and a sliding-window transformer on the same image), CKA produces a high similarity driven partly by shared low-level texture and position statistics, because it averages over all dimensions including nuisance-dominated ones. Our method instead projects onto a subspace orthogonalized against the nuisance heads, so shared identity coordinates dominate the cosine. We expect a visible gap concentrated on stimuli where nuisance priors diverge most, with near-parity on stimuli where priors coincide. On the synthetic dataset, we expect ISA-ND to recover the ground-truth shared directions with mean absolute cosine above 0.9 after permutation matching, while CKA and the independence-prior baseline do not.

**vs. Contrastive-only (lambda=0, beta=0, gamma=0)** — On a case where the nuisance prior is high-variance and correlated with the shared signal (MAE locality tracking object position), the plain contrastive model lets the shared subspace absorb positional variance, because nothing penalizes covariance between shared and residual. The resulting score is inflated by nuisance-driven similarity. Our method adds the covariance-orthogonality penalty, so the shared subspace is pushed away from the residual and the score reflects identity structure. We expect the contrastive-only model to over-score on position-varying stimuli and our method to show a modest but consistent downward correction there, with parity elsewhere.

**vs. Learned disentanglement head (independence prior)** — On a case where shared and nuisance latents are statistically coupled (token position depends on content), an independence-prior head cannot separate them because it asserts independence the data violate, so it splits variance arbitrarily and the score drifts. Our method conditions on the auxiliary variable, which breaks the rotation symmetry only where u is informative, and the MI penalty drives the component of u that is entangled with the residual to zero. We expect our recovered subspace to be stable across seeds while the independence head's score varies noticeably, and a clear advantage on the subset where the auxiliary is strongly predictive. We expect the residual-dependence metric $\|\mathrm{Cov}(u, r)\|_F$ to be lower for full ISA-ND than for the gamma=0 ablation, and to be nonzero on real data where the conditional-independence premise is only partially satisfied.

**vs. ISA-ND with shuffled auxiliary u** — Shuffling u destroys its dependence on the shared latent while leaving the architecture unchanged. If the identifiability gain comes from the auxiliary condition, the shuffled version should lose the synthetic ground-truth recovery (mean absolute cosine drops toward the independence-prior baseline) and show a larger residual-dependence metric, because the MI penalty can no longer align u with the shared component.

**vs. ISA-ND with gamma=0** — Removing the MI penalty leaves the auxiliary term free to use the component of u that is entangled with the nuisance residual. We expect the gamma=0 ablation to show a higher residual-dependence metric and a less stable recovered subspace across seeds on the real data, with the gap concentrated on stimuli where token position is most strongly tied to the architecture-specific positional prior.

### What would falsify this idea

If the gain over contrastive-only and CKA were uniform across all stimulus subsets rather than concentrated where the predicted nuisance-correlation failure occurs, the mechanism story would be wrong. Likewise, if the shuffled-auxiliary ablation matched full ISA-ND, or if the synthetic ground-truth recovery did not exceed the independence-prior baseline, the identifiability claim — the core of the idea — would be falsified.

## References

1. Where Does Alignment Emerge and What Does It Mean? A Closer Look at the Platonic Representation Hypothesis
