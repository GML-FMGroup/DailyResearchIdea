# Contraction-Certified Language-Conditioned Latent Dynamics via Identifiable Effect Bases

## Motivation

Latent world models now accept free-form natural language as a universal action interface (LaLa-World), and agentic benchmarks score constraint adherence statically (AgentIF), but neither imposes a structural condition on the language→transition map: distinct phrasings need not produce distinct, recoverable effects, and iterated latent rollouts carry no compounding-error account. The root cause is that language conditioning is treated as an unconstrained high-capacity predictor input rather than as coordinates in a shared, embodiment-invariant effect basis, so identifiability and rollout stability are never tied to the same quantity. This is the meta-gap recurring across the language-as-action and language-as-constraint paper-trees: a verifiable conditioning signal was replaced by free-form text and only downstream success was retained as the check.

## Key Insight

If the language→effect map is constrained to a shared effect basis whose coordinates are recoverable across embodiments, then the same identifiable coordinates induce a measurable contraction modulus of the latent dynamics, making the compounding-error certificate a consequence of identifiability rather than an added-on bound.

## Method

**What it is.** CCLD (Contraction-Certified Language Dynamics) is a language-conditioned latent dynamics model whose language→transition map is factored through a learned, embodiment-invariant effect basis `E = {e_1..e_K}` (K=16). Inputs: latent state `z_t` (dim 256), free-form language instruction `ℓ`, embodiment tag `m`. Outputs: predicted latent `z_{t+1}`, an effect coordinate `α(ℓ) ∈ R^K`, and a contraction modulus `κ` estimated from the same coordinates that certifies the multi-step rollout error.

**Load-bearing assumption (stated explicitly).** The contrastive loss `L_id` over LaLa-World paraphrase pairs yields an effect coordinate `α(ℓ)` that is identifiable up to the true physical effect partition, i.e., paraphrase-equivalence labels in the dataset coincide with physical-effect equivalence across embodiments, so that `κ̂` computed from `α(ℓ)` is a valid contraction modulus for the true dynamics rather than for a coarsened, label-induced partition. This assumption is not taken on faith: it is verified from observed transitions by the inverse-dynamics consistency check in Phase 1b, and the certificate is reported with an explicit partition-refinement term (Phase 3) so that `κ̂` is certified against observed dynamics rather than annotation labels.

**How it works.**

```pseudocode
# Phase 1: effect-basis factorization
encode:  α(ℓ) = f_θ(ℓ)            # K=16 dim effect coordinates, L2-normalized; f_θ = 2-layer MLP (hidden=512, GeLU)
          β(m) = g_φ(m)            # embodiment embedding (dim 16); g_φ = 1-layer MLP (hidden=256, GeLU)
          z_{t+1} = h_ψ(z_t, α(ℓ) ⊙ β(m))   # ⊙ = per-coordinate gating; h_ψ = 4-layer MLP (hidden=512, GeLU)
train:    L_pred = ||z_{t+1} - z_{t+1}^*||^2
          L_id   = InfoNCE(α(ℓ_i), α(ℓ_j); τ=0.07) with positives = paraphrase pairs from LaLa-World action descriptions
                   (pulled to α(ℓ_j) iff effect(ℓ_i)=effect(ℓ_j)), negatives = distinct-effect pairs
          L_inv  = ||α(ℓ) - α(ℓ)|| across embodiment swaps (β absorbs m, not α)

# Phase 1b: inverse-dynamics consistency check (verifies L_id positives)
learn inverse map:  α̂(z_t, z_{t+1}) = q_ω(z_t, z_{t+1})   # q_ω = 2-layer MLP (hidden=512, GeLU), output dim K
residual:           r(ℓ, z_t, z_{t+1}) = ||α(ℓ) - α̂(z_t, z_{t+1})||_2
reweight:           w(ℓ_i,ℓ_j) = exp(-r_ij^2 / (2σ^2)), σ = median residual over training set
                    L_id uses w(ℓ_i,ℓ_j) as pair weight; pairs with w < 0.1 are dropped
partition-refinement term: δ = quantile_{0.95}(r) over held-out transitions

# Phase 2: contraction modulus from effect coordinates
for each transition:  J_t = ∂h_ψ/∂z_t  evaluated at (z_t, α(ℓ)⊙β(m))
          κ̂ = sup_t ||J_t||_2   estimated by power iteration on J_t^T J_t (5 iters)
          L_con = relu(κ̂ - (1 - λ·min_k |α_k(ℓ)|))^2   # λ=0.1, penalize κ̂ above 1-λ·effect-strength

# Phase 3: certified multi-step rollout
rollout H steps:  z_{t+H} = h_ψ^H(z_t, α(ℓ))
bound:            ||z_{t+H} - ẑ_{t+H}|| ≤ κ̂^H/(1-κ̂) · (ε + δ),  ε = one-step residual, δ = partition-refinement term from Phase 1b
report:           certificate = (κ̂, ε, δ, H) emitted with every plan
```

**Why this design.** We chose a shared effect basis `E` over per-embodiment action heads because the identifiability claim requires that the same coordinate recover the same physical effect across embodiments, accepting the cost that embodiments whose dynamics are not span-able by `E` will be poorly modeled. We chose to estimate `κ̂` from the Jacobian of the same `h_ψ` that consumes `α(ℓ)` rather than from a separate stability head, because a separate head would decouple the certificate from the map and reintroduce the unverified-rollout problem LaLa-World already has, accepting the cost of a power-iteration pass per transition. We chose a contrastive identifiability loss over an explicit inverse-dynamics reconstruction because contrastive pairs are available from LaLa-World's paraphrase data without extra labels, accepting the cost that effect-equivalence labels are only as good as the paraphrase set; the inverse-dynamics consistency check in Phase 1b is added precisely to verify those labels against observed transitions, so the certificate does not rest on an unverified annotation partition. A domain expert would not describe this as LaLa-World plus a stability regularizer: LaLa-World never ties its language interface to a recoverable coordinate system, so its contraction is not a function of the conditioning signal — here `κ̂` is literally computed from `α(ℓ)`, which is the structural coupling that prior latent-world-model work (V-JEPA 2-AC, LaLa-World) does not have.

**Why it measures what we claim.** `α(ℓ)` measures *effect identifiability* because the contrastive loss assumes paraphrases with equal annotated effects are valid positive pairs and distinct effects are negatives; this assumption fails when the paraphrase set conflates two physically distinct effects under one description, in which case `α` collapses those effects and the modulus is estimated on a coarser partition than the true dynamics — the inverse-dynamics residual `r` and its 95th-percentile `δ` detect exactly this failure by flagging pairs whose annotated equivalence is inconsistent with observed transitions. `κ̂` measures *rollout stability* because the power-iteration estimate of the spectral norm of `∂h_ψ/∂z_t` equals the local Lipschitz constant of the latent map under the assumption that `h_ψ` is differentiable and the supremum over visited states is representative; this assumption fails at state-distribution shift (out-of-distribution `z_t`), in which case `κ̂` reflects in-distribution contraction and the certificate is optimistic. The certified bound `κ̂^H/(1-κ̂)·(ε+δ)` measures *compounding error* because the standard contraction argument assumes the one-step residual `ε` is uniformly bounded over the rollout horizon; this assumption fails when the language instruction drives the state into a region where `ε` grows, in which case the bound is valid only for the prefix where `ε` holds.

## Contribution

(1) CCLD, a language-conditioned latent dynamics model that factors the language→transition map through a learned embodiment-invariant effect basis and estimates a contraction modulus from the same effect coordinates, emitting a rollout certificate with every plan. (2) A design principle: tying the identifiability contrastive loss and the contraction penalty to a shared coordinate `α(ℓ)` makes the compounding-error certificate a function of the conditioning signal, rather than an independent stability regularizer. (3) An effect-paraphrase protocol derived from LaLa-World action descriptions and a certificate-calibration analysis showing when `κ̂` tracks realized rollout error.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | LaLa-World multi-embodiment sim+robot trajectories (paraphrase action descriptions; cross-embodiment coverage) | Provides paraphrase action descriptions and cross-embodiment coverage; also supplies observed (z_t, z_{t+1}) transitions for the Phase 1b inverse-dynamics check |
| Primary metric | Planning success under MPC with certificate validity; certificate violation rate (actual error > bound) | Directly tests whether certified rollouts are trustworthy and whether the bound is empirically sound |
| Baseline | LaLa-World (no effect basis, no certificate) | Tests claim that structural coupling beats plain language conditioning |
| Baseline | V-JEPA 2-AC | Tests claim against latent action model without certified contraction |
| Baseline | Per-domain LWM (separate heads) | Tests whether shared effect basis beats per-embodiment specialization |
| Ablation-of-ours | CCLD without contraction loss (κ̂ uncoupled) | Isolates whether κ̂-from-α drives the gain |
| Ablation-of-ours | CCLD without inverse-dynamics reweighting (L_id on raw paraphrase pairs) | Tests whether the Phase 1b consistency check is necessary for certificate validity |

### Why this setup validates the claim

The central claim is that tying the contraction modulus to the language-derived effect coordinates `α(ℓ)` produces rollouts whose error can be certified and therefore trusted at plan time. LaLa-World is the right substrate because it already supplies paraphrase action descriptions and multiple embodiments, so the effect-equivalence partition needed by `L_id` exists without new labels; using it also makes the baseline comparison fair, since both models see identical data. The primary metric — planning success under MPC where the planner is allowed to consume the certificate — is the right detector because it is sensitive precisely to the predicted effect: if `κ̂` is a valid contraction estimate, the planner can prune or shorten rollouts whose bound is loose, converting stability into success. The certificate violation rate is a first-class metric because it directly measures whether the emitted bound `κ̂^H/(1-κ̂)·(ε+δ)` holds on held-out rollouts; a low violation rate is the empirical signature that `κ̂` is sound, not merely emitted. The LaLa-World baseline tests whether the effect basis matters at all; V-JEPA 2-AC tests whether the certificate beats an uncertified latent action model; the per-domain baseline tests whether the shared basis is better than specialization; the uncoupled-κ̂ ablation isolates the causal contribution of computing `κ̂` from `α(ℓ)` rather than from a free stability head; and the no-reweighting ablation isolates the contribution of the Phase 1b inverse-dynamics consistency check to certificate validity. A gain that appears only when the certificate is consumed, disappears under the uncoupled-κ̂ ablation, and degrades under the no-reweighting ablation is exactly the signature the claim predicts.

### Expected outcome and causal chain

**vs. LaLa-World** — On a case where a paraphrased instruction ("push the block left" vs. "slide the block leftward") must produce the same physical effect across two embodiments, LaLa-World's language interface has no recoverable coordinate system, so it maps the two phrasings to nearby but distinct transition directions and the planner's rollout drifts. Our method instead collapses both phrasings onto the same `α(ℓ)` via `L_id` (with Phase 1b reweighting ensuring the collapse is consistent with observed transitions), so the transition and its `κ̂` are shared, and we expect a visible success gap concentrated on paraphrase-pair subsets, with near-parity on unique-instruction subsets. We also expect a lower certificate violation rate than LaLa-World's (which emits no certificate, so violation rate is undefined; we report the fraction of rollouts whose actual error exceeds the nominal one-step residual bound as a proxy).

**vs. V-JEPA 2-AC** — On a case where a multi-step plan must be committed before execution, V-JEPA 2-AC emits a latent action without any bound on compounding error, so the planner cannot distinguish a rollout that will stay on-manifold from one that will drift, and it over-commits on the drifting one. Our method emits `(κ̂, ε, δ, H)` from the same map, so the planner can shorten or reject the high-κ̂ rollout, and we expect a larger success margin on long-horizon plans than on short-horizon ones, with the gap widening as H grows. The certificate violation rate for CCLD is expected to be low (below 5% on held-out rollouts) and to increase with H, reflecting the known limitation that `ε` may grow under language-driven state shift.

**vs. Per-domain LWM** — On a case where a robot embodiment is data-poor but shares its physical effect vocabulary with a data-rich simulation embodiment, the per-domain model overfits its narrow transition head and mispredicts the effect direction. Our method instead routes the effect through the shared basis `E` with embodiment gating `β(m)`, so the data-rich embodiment's effect coordinate transfers, and we expect the gain to be largest on the low-data embodiments and to shrink toward zero on the data-rich ones.

**vs. CCLD without inverse-dynamics reweighting** — On paraphrase pairs whose annotated equivalence is inconsistent with observed transitions (e.g., "move the block left" vs. "push the block left" differing in contact regime), the raw-contrastive model collapses them onto the same `α(ℓ)` and estimates `κ̂` on a coarser partition, leading to optimistic certificates and higher violation rates. Our method down-weights these pairs via `w(ℓ_i,ℓ_j)` and reports `δ`, so we expect lower violation rates and more conservative (but still useful) certificates on the affected subsets.

### What would falsify this idea

If our gain over LaLa-World is uniform across paraphrase-pair and unique-instruction subsets rather than concentrated on the paraphrase subsets, then the effect-basis coupling is not what drives the improvement. Similarly, if removing the contraction loss (free `κ̂` head) leaves planning success unchanged while the certificate is consumed, or if the certificate violation rate is high (e.g., >20%) on held-out rollouts, then the structural coupling of `κ̂` to `α(ℓ)` is not doing the work the claim assigns it.

## References

1. LaLa-World: Scaling Latent World Models with Natural Language Actions
2. AGENTIF: Benchmarking Instruction Following of Large Language Models in Agentic Scenarios
3. SysBench: Can Large Language Models Follow System Messages?
4. CFBench: A Comprehensive Constraints-Following Benchmark for LLMs
5. Instruction-Following Evaluation for Large Language Models
6. FollowBench: A Multi-level Fine-grained Constraints Following Benchmark for Large Language Models
7. Can LLMs Follow Simple Rules?
8. COLLIE: Systematic Construction of Constrained Text Generation Tasks
