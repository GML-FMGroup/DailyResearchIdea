# Injective Binding Regularization for Identifiable Latent Chain-of-Thought in Unified Multimodal Models

## Motivation

Latent-reasoning unified multimodal models such as LARC reason over continuous hidden states, but the binding between a latent reasoning state and the observed multimodal evidence (text tokens and image patches) is realized only through cross-attention maps that are modality-specific observables. Because the latent-to-binding map is never constrained to be invertible, distinct latent reasoning states can produce identical or near-identical attention bindings under modality-specific encoding bias, so the latent state is not identifiable from what is observed and RL reward on bindings cannot reliably credit the latent reasoning step that caused a generation outcome. This identifiability gap recurs structurally across linguistic-binding and image-generation branches rather than being a local defect of one model.

## Key Insight

If the cross-attention binding is constrained to be an injective, Lipschitz-invertible function of the latent reasoning state, then the latent state is recoverable from observed bindings up to the Lipschitz constant, which converts binding-level RL reward into a well-posed credit signal for the latent reasoning step rather than a modality-biased proxy.

## Method

**What it is.** IBRL (Injective Binding Regularization for Latent reasoning) augments a LARC-style unified multimodal model with a bijective cross-attention binding head and a contrastive identifiability loss, so that during information-gain RL the latent reasoning state z_t is recoverable from its binding b_t = f_theta(z_t, E) where E is the observed multimodal evidence. The load-bearing assumption, stated plainly: the binding head f_theta can be made injective and Lipschitz-invertible in z_t while remaining a faithful, reward-informative readout of E — i.e., f_theta has enough capacity and the latent trajectory has enough intrinsic dimensionality for a bijective map to exist without destroying the information-gain signal the RL objective depends on. This assumption is not taken for granted: it is made verifiable by (i) architectural invertibility of f_theta, (ii) computing the reward on the recovered latent z_hat_t rather than on b_t, and (iii) an explicit Jacobian-rank injectivity certificate on held-out latents (see Phase 2 step (e)).

**How it works.**

```pseudocode
Input: unified model M with latent reasoning states z_t (t=1..T), dim d_z=256,
       cross-attention binding head f_theta (invertible-by-construction),
       evidence E=(text tokens, image patches)
Hyperparams: lambda_id=0.1, tau=0.07, L_lip=1.0, K=8 negatives, rho=0.5,
             N_cert=512 held-out latents, rank_tol=1e-3

Binding head architecture (concrete):
  f_theta = CouplingFlow(d_z=256, n_layers=4, hidden=512, act=GeLU)
  # each layer: split z into (z_a, z_b); z_b' = z_b * exp(s(z_a,E)) + t(z_a,E)
  # s,t are 2-layer MLPs (hidden=512, GeLU); log-det-Jacobian accumulated per layer
  # bijectivity is architectural (coupling layers are invertible by construction)
  g_phi = f_theta^{-1}  # exact analytic inverse of the coupling flow, no separate params

Phase 1  Curriculum SFT (as in LARC): token CoT -> latent CoT
Phase 2  Identifiability-regularized RL:
  for each rollout:
    z_1..z_T   = M.latent_reason(E)              # latent reasoning trajectory
    b_t        = f_theta(z_t, E)                 # invertible coupling-flow binding
    # (a) invertibility probe: recover z_hat_t from b_t via analytic inverse g_phi=f_theta^{-1}
    z_hat_t    = g_phi(b_t, E)                   # exact inverse, no co-trained params
    L_inv      = ||z_hat_t - z_t||^2             # reconstruction of latent from binding
    # (b) contrastive identifiability: bindings of distinct latents must be distinct
    L_con      = -log( exp(sim(b_t, b_t^+)/tau) /
                       sum_{k=1..K} exp(sim(b_t, b_t^{-,k})/tau) )
                 # b_t^+ = binding of same z_t under augmented evidence E'
                 # b_t^{-,k} = bindings of K other latent states in batch
                 # negatives drawn by coverage-certified sampler (see (e))
    # (c) Lipschitz-invertibility penalty: bound binding sensitivity
    L_lip      = max(0, ||f_theta(z_t,E) - f_theta(z_t',E)|| / ||z_t - z_t'|| - L_lip)
    L_id       = lambda_id*(L_inv + rho*L_con) + L_lip
    # (d) information-gain RL objective (LARC) + identifiability regularizer
    #     reward computed on RECOVERED latent z_hat_t, not on b_t, so a contracting
    #     binding head cannot shrink the reward signal
    reward     = info_gain(z_hat_t, E)
    L_rl       = -reward * log pi(z_t | E)
    L_total    = L_rl + L_id
  # (e) injectivity certificate (every 500 steps, on N_cert=512 held-out latents):
  #     compute Jacobian J = d b / d z at each held-out z; check rank(J) == d_z
  #     (singular values > rank_tol); if any rank-deficient, increase coupling layers
  #     and log the certificate; negatives for L_con are sampled from this held-out set
  #     to cover the latent support rather than only the current batch
  update M, f_theta by gradient descent on L_total  # g_phi is analytic inverse, no update
```

**Why this design.** We chose a *fixed inverse head* g_phi over a learned discriminator because a discriminator only tests distinguishability of bindings, not recoverability of the latent, accepting the cost that g_phi adds parameters and must be co-trained; here g_phi is the exact analytic inverse of the coupling flow, so it adds no parameters and cannot drift from f_theta. We chose an *injectivity penalty via a Lipschitz bound* over a soft independence regularizer because independence between latent and modality bias is not structurally guaranteed (the two are coupled through the shared evidence encoder), accepting the cost that the Lipschitz constant must be tuned per modality. We chose *contrastive positives from evidence augmentation* over random negatives because the positive must preserve z_t while perturbing modality-specific encoding bias, accepting the cost that augmentation quality bounds the contrastive signal. A domain expert would not describe this as a variant of LARC: LARC optimizes information gain on bindings without any invertibility constraint, whereas IBRL adds a recoverability condition that makes the binding a sufficient statistic for the latent state, which LARC's reward cannot exploit.

**Why it measures what we claim.** The reconstruction loss L_inv measures *latent identifiability* because it assumes the inverse head g_phi is expressive enough to approximate f_theta^{-1} on the support of the latent distribution; this assumption fails when the binding head collapses distinct latents to the same binding (non-injective f_theta), in which case L_inv reflects only the residual of a many-to-one map and not true invertibility. The contrastive loss L_con measures *modality-bias invariance of the binding* because it assumes evidence augmentation changes only modality-specific encoding bias and not the latent reasoning state; this assumption fails when augmentation alters task-relevant evidence, in which case L_con rewards invariance to genuine content changes. The Lipschitz penalty L_lip measures *bounded sensitivity of binding to latent perturbation* because it assumes the latent space is locally Euclidean with meaningful distances; this assumption fails when the latent manifold is highly curved, in which case L_lip constrains only the ambient metric and not the intrinsic one. The injectivity certificate (e) measures *global injectivity over the latent support* because it assumes the held-out N_cert=512 latents cover the support; this assumption fails when the latent distribution has modes absent from the held-out set, in which case the certificate reflects local rank only on sampled modes and not global bijectivity.

## Contribution

(1) IBRL, a training scheme that enforces an injective, Lipschitz-invertible cross-attention binding map between latent reasoning states and multimodal evidence via a reconstruction probe, contrastive identifiability loss, and Lipschitz penalty during information-gain RL. (2) The design principle that binding-level RL reward becomes a well-posed credit signal for latent reasoning only when the latent-to-binding map is constrained to be recoverable, which we operationalize through a co-trained inverse head. (3) A diagnostic protocol that separates identifiability failures (non-injective binding) from reward-design failures in latent-reasoning unified multimodal models.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | Multimodal CoT benchmarks needing evidence-grounded latent reasoning (e.g., ScienceQA multimodal split, A-OKVQA CoT subset) | Exercises binding under genuine modality coupling |
| Synthetic collapse-prone set | 2,000 examples with known ground-truth latents z* (d_z=256), constructed so distinct z* map to near-identical bindings under a frozen evidence encoder | Validates invertibility before scaling to full multimodal RL loop |
| Primary metric | Information-gain reward on held-out reasoning trajectories, computed on recovered z_hat_t | Directly measures what IBRL claims to improve |
| Secondary metric | Downstream task accuracy (multimodal CoT answer accuracy) | Tests whether identifiability translates into capability, not only self-referential reward |
| Secondary metric | Latent reconstruction fidelity ||z_hat_t - z*||^2 on synthetic set | Directly tests the invertibility claim |
| Secondary metric | Jacobian-rank injectivity certificate pass rate on N_cert=512 held-out latents | Verifies global injectivity rather than assuming it |
| Baseline | LARC (info-gain RL, no invertibility) | Tests whether recoverability adds value beyond LARC |
| Baseline | Soft-independence regularized LARC | Tests injectivity penalty vs. independence heuristic |
| Baseline | Discriminator-based binding LARC | Tests fixed-inverse recoverability vs. distinguishability |
| Ablation-of-ours | IBRL minus L_lip (injectivity penalty) | Isolates the Lipschitz-bound contribution |
| Ablation-of-ours | IBRL with reward on b_t instead of z_hat_t | Isolates the reward-on-sufficient-statistic contribution |

### Why this setup validates the claim

The central claim is that making the binding a sufficient statistic for the latent state — via invertibility, contrastive identifiability, and a Lipschitz injectivity bound — lets information-gain RL exploit recoverable latents that LARC cannot. The design is a falsifiable test because each baseline removes exactly one sub-claim. Plain LARC tests whether recoverability matters at all: if IBRL's gain over it is null, the invertibility condition is inert. The soft-independence baseline tests whether structural injectivity beats a heuristic that does not guarantee it through the shared evidence encoder. The discriminator baseline tests whether recoverability (fixed inverse) beats mere distinguishability. The L_lip ablation isolates whether bounded sensitivity contributes beyond reconstruction and contrastive terms. The reward-on-b_t ablation isolates whether computing the reward on the recovered latent (rather than the binding) is what makes the credit signal well-posed. The synthetic collapse-prone set with known z* provides a controlled setting where invertibility can be validated directly via reconstruction fidelity and the Jacobian-rank certificate before any claim of gain over LARC is made. The primary metric — held-out information-gain reward on recovered latents — is the right detector because it is precisely the quantity the reward optimizes and the quantity a recoverable, bias-invariant binding should raise; the downstream accuracy metric guards against the reward being self-referential, and a metric like generation FID would confound reasoning quality with decoder capacity and could not isolate the binding's sufficiency.

### Expected outcome and causal chain

**vs. LARC** — On a case where two distinct reasoning latents z_t and z_t' induce near-identical bindings under the shared evidence encoder (a many-to-one collapse), LARC's info-gain reward rises but the binding cannot disambiguate which latent produced it, so downstream reasoning stalls on ambiguous evidence. Our method instead forces distinct latents to distinct bindings via L_con and recovers z_t from b_t via g_phi, so the reward is computed on a sufficient statistic. We expect a noticeable reward gap concentrated on the collapse-prone subset and near-parity where bindings are already separable; on the synthetic set we expect reconstruction fidelity ||z_hat_t - z*||^2 to be near-zero for IBRL and non-zero for LARC, and downstream accuracy to track this gap.

**vs. Soft-independence LARC** — On a case where modality-specific encoding bias correlates with the latent through the shared encoder, the soft-independence penalty leaves residual coupling because independence is not structurally enforced, so the binding leaks bias and the reward over-credits modality cues. Our method instead bounds binding sensitivity with L_lip and enforces injectivity, so bias cannot masquerade as latent signal. We expect the gap to appear specifically on bias-correlated instances, with parity on bias-neutral ones, and the Jacobian-rank certificate to pass at a higher rate for IBRL than for the soft-independence baseline.

**vs. Discriminator-based LARC** — On a case where the discriminator accepts bindings as distinguishable but the map is still non-invertible (distinct bindings, unrecoverable latents), the discriminator baseline passes its test yet z_t cannot be reconstructed, so reward gradients misattribute credit. Our method instead uses the fixed inverse g_phi to demand actual reconstruction, so credit flows to the true latent. We expect higher reconstruction fidelity and a reward gap that widens as latent dimensionality grows (tested at d_z in {64, 128, 256}).

**vs. IBRL minus L_lip** — On a case where the latent manifold is locally curved, the L_lip penalty constrains binding sensitivity in the ambient metric; removing it should leave residual sensitivity to latent perturbations that L_con alone does not suppress. We expect the full IBRL to show lower reconstruction residual and higher certificate pass rate than the ablation, with the gap concentrated on curved-manifold instances.

**vs. IBRL with reward on b_t** — On a case where the coupling flow contracts the binding, computing the reward on b_t shrinks the reward signal and lets the binding head game the objective; computing it on z_hat_t should restore the reward magnitude. We expect the reward-on-z_hat_t variant to show higher held-out reward and downstream accuracy, with the gap widening as the flow's contraction (measured by log-det-Jacobian) increases.

### What would falsify this idea

If IBRL's gain over LARC is uniform across all subsets rather than concentrated where binding collapse and modality-bias coupling are predicted to occur, the recoverability mechanism is not the cause of improvement. Equally, if the L_lip ablation matches full IBRL, the injectivity penalty contributes nothing and the central structural claim fails; and if the Jacobian-rank certificate fails to pass on held-out latents for IBRL, the architectural invertibility does not extend to the latent support and the load-bearing assumption is false.

## References

1. Latent Reasoning in Continuous Space for Unified Multimodal Models
