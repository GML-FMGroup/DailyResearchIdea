# Certified Minibatch OT Couplings for Flow-Matching Music Source Separation

## Motivation

FLOMSS and the diffusion branch of multi-source diffusion models both rely on a minibatch optimal-transport coupling as a stochastic proxy for the global OT plan, yet neither quantifies the deviation between the two; the learned vector field is therefore trained against a target whose global transport consistency is unverified. This is a structural gap across OT-CFM, flow-matching MSS, and diffusion-based separation: the minibatch coupling is treated as faithful by assumption, not by measurement, so a batch of paired stems can silently induce a coupling far from the global plan without any training-time signal.

## Key Insight

The empirical OT plan's deviation from the global plan is controlled by a batch-computable concentration radius, and penalizing the flow-matching vector field's induced coupling against that radius turns an unverified stochastic approximation into a per-batch transport-consistency certificate.

## Method

**What it is.** We introduce **COT-FM** (Certified Optimal-Transport Flow Matching), which augments the OT-CFM regression objective with a batch-computable transport-consistency regularizer. Inputs: a minibatch of paired stems $(x_0^i, x_1^i)$, the minibatch OT coupling $\pi_B$ from OT-CFM, and the flow network $v_\theta$. Outputs: the standard flow-matching loss plus a certified penalty $\mathcal{L}_{\text{cert}}$ and a scalar certificate $r_B$ reported per batch.

**How it works.**
```pseudocode
Input: batch B = {(x0_i, x1_i)}_{i=1..n}, flow net v_theta, cost c(x,y)=||x-y||^2, reg. weight lambda=0.1, confidence delta=0.05, Sinkhorn reg. eps=0.01 (fallback)
1. Compute cost matrix C_ij = c(x0_i, x1_j); solve exact minibatch OT -> coupling pi_B (n x n) via network simplex (POT library)
2. Compute batch radius  r_B = C_ref * sqrt( (1/n) * ( log(2/delta) + d * log(n) ) )   # d = ambient dim (audio: 2*44100*2? use flattened stem dim), C_ref = max_{i,j} C_ij in batch
3. For each pair (i,j) with pi_B(i,j) > 0:
      sample t ~ U[0,1];  x_t = (1-t) x0_i + t x1_j
      target u = x1_j - x0_i
      l_FM += pi_B(i,j) * || v_theta(x_t, t) - u ||^2
4. Induced coupling check: run a second exact minibatch OT solve on the field's predicted endpoints:
      For each pair (i,j), predict endpoint y_hat_ij = x_t + v_theta(x_t, t) with t=0.5 fixed (or average over K=4 t samples)
      Compute cost matrix D_ij = || y_hat_ij - x1_j ||^2
      Solve exact OT between uniform source weights (1/n) and uniform target weights (1/n) with cost D -> coupling q (n x n)
      l_cert += max(0, W2_hat(pi_B, q) - r_B)^2         # hinge at the certified radius
5. Total loss = l_FM + lambda * l_cert ; report certificate r_B and margin W2_hat - r_B
```
Here $W2_hat$ is the entropic-regularized 2-Wasserstein distance between the minibatch plan $\pi_B$ and the coupling $q$ induced by the learned vector field's predicted endpoints (computed via a second exact OT solve on endpoint residuals), and $r_B$ is the Fournier–Guillin/Weed–Bach-style concentration radius computed from batch statistics alone, with $\delta=0.05$ a user-set confidence and $C_ref$ the maximal transport cost observed in the batch.

**Why this design.** We chose an exact minibatch OT solve over an entropic approximation because the certificate radius is stated for the exact empirical plan, and an entropic plan would make the radius uncomputable without a second approximation; the cost is $O(n^3)$ per batch, acceptable for the batch sizes used in FLOMSS ($n=16$). We chose a hinge penalty at $r_B$ rather than a plain MSE between $\pi_B$ and $q$ because the hinge certifies only violations of the stated radius, avoiding over-constraining batches that are already within tolerance and preventing the regularizer from fighting the flow-matching loss; the cost is that small violations below $r_B$ are not penalized, which is intentional. We chose a second exact OT solve on the field's predicted endpoints to define $q$ rather than a residual-softmax proxy because the residual-softmax construction is not a transport plan (it is a per-pair regression error normalized by temperature, not by coupling mass), whereas an OT solve on predicted endpoints yields a genuine coupling whose Wasserstein distance to $\pi_B$ is a true plan-to-plan distance; the cost is an extra $O(n^3)$ solve per batch, which we accept for $n=16$. This differs from OT-CFM (Tong et al.), which uses the minibatch plan as a fixed target with no deviation term; a domain expert would not describe COT-FM as OT-CFM with an extra loss, because the certificate radius $r_B$ is computed from the same batch statistics that define the plan and is reported as a falsifiable scalar, not tuned as a generic regularizer.

**Why it measures what we claim.** The batch radius $r_B$ measures the deviation between the minibatch OT plan and the global OT plan because the Fournier–Guillin concentration inequality bounds the empirical plan's Wasserstein deviation by a term scaling as $\sqrt{(\log(2/\delta) + d\log n)/n}$ under the assumption that the cost is bounded by $C_ref$ and the source/target measures have finite second moment; this assumption fails when the batch contains outlier stems with cost exceeding $C_ref$, in which case $r_B$ reflects the clipped cost rather than the true deviation and the certificate becomes conservative. The hinge $\max(0, W2_hat - r_B)^2$ measures transport inconsistency because $W2_hat$ between $\pi_B$ and the endpoint-OT coupling $q$ equals the Wasserstein distance between the empirical plan and the plan implied by the learned vector field's predicted endpoints under the assumption that the endpoint prediction $y_hat_ij = x_t + v_\theta(x_t,t)$ is a faithful estimate of the transport destination for pair $(i,j)$; this assumption fails when the vector field is far from convergence, in which case $q$ reflects the field's current endpoint errors rather than a stable coupling. The reported margin $W2_hat - r_B$ measures certified consistency because a non-positive margin means the learned coupling is within the stated radius of the minibatch plan under the assumption that $\pi_B$ is itself within $r_B$ of the global plan—the exact chain the concentration bound supplies; this assumption fails when the global plan is not unique (e.g., degenerate cost), in which case the certificate is relative to the chosen minibatch plan rather than a unique global plan.

## Contribution

['A batch-computable certified transport-consistency regularizer (COT-FM) that couples the flow-matching regression target to a concentration-radius certificate, producing a per-batch scalar margin that is falsifiable on MUSDB18-HQ.', 'A design principle: the minibatch OT coupling in flow-matching MSS should be treated as an approximation with a reported radius rather than a fixed target, and the hinge-at-radius form prevents the regularizer from over-constraining already-consistent batches.', 'An open implementation of the certificate computation (exact minibatch OT, batch radius, residual-induced soft coupling) integrated into the FLOMSS training loop.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | MUSDB18-HQ | Standard multi-stem benchmark with paired mixtures and stems |
| Primary metric | SDR gain vs. baseline | Detects separation quality directly tied to coupling accuracy |
| Secondary metric | Margin-SDR correlation (Pearson r) | Tests whether certificate margin predicts per-batch SDR degradation |
| Baseline | OT-CFM (Tong et al.) | Tests whether certification beats fixed-plan regression |
| Baseline | Standard CFM | Tests whether OT coupling itself is necessary |
| Baseline | Demucs | Discriminative reference for achievable separation ceiling |
| Ablation-of-ours | COT-FM without hinge penalty | Isolates whether the certified penalty drives gains |
| Ablation-of-ours | COT-FM with residual-softmax q (original proxy) | Tests whether endpoint-OT q is necessary |
| Synthetic validation | 2D Gaussian mixture with known global OT plan | Validates that endpoint-OT q recovers true coupling |

### Why this setup validates the claim

COT-FM claims that a batch-computable certificate plus hinge penalty makes the learned coupling stay near the minibatch OT plan, and that this yields better separation than an uncertified plan. MUSDB18-HQ is the right testbed because it provides paired stems, so the minibatch OT plan is well-defined and the certificate radius $r_B$ is computable from batch statistics alone. OT-CFM isolates the sub-claim that certification adds value over a fixed OT target; standard CFM isolates whether OT coupling matters at all; Demucs serves as a discriminative ceiling so we can tell whether any generative gain is meaningful. The ablation (COT-FM minus hinge) is the critical control: if the certificate is doing real work, removing the hinge should reproduce OT-CFM-like behavior rather than COT-FM behavior. The residual-softmax ablation tests whether the endpoint-OT coupling $q$ is necessary versus the original proxy. The synthetic validation with known global OT plan directly tests the load-bearing assumption that endpoint-OT $q$ recovers true coupling mass. SDR is the right primary metric because it is sensitive to the exact failure mode we predict — misassigned source mass — and because it is the metric on which OT-CFM's coupling advantage should manifest. The margin-SDR correlation tests the diagnostic claim: if the margin $W2_hat - r_B$ is a meaningful certificate, it should negatively correlate with per-batch SDR degradation. Together these form a falsifiable test: the claim predicts a specific ordering (COT-FM > OT-CFM > CFM) concentrated where the coupling is ambiguous, not uniform across all stems.

### Expected outcome and causal chain

**vs. OT-CFM (Tong et al.)** — On a case where two stems in a batch have similar energy profiles (e.g., co-occurring guitar and piano), OT-CFM's fixed minibatch plan can assign transport mass to the wrong target because the plan is computed once and never corrected by the network's own regression behavior, so the flow learns a slightly misaligned pairing. Our method instead computes the endpoint-OT coupling $q$ and hinges against $r_B$, so when the network's predicted endpoints reveal ambiguity the penalty nudges the plan back toward the certified region. We expect a modest but consistent SDR gain over OT-CFM on harmonically overlapping stems, with parity on well-separated stems like drums.

**vs. Standard CFM** — On a case where the source and target marginals are multi-modal (e.g., a mixture containing both a sustained pad and a transient), standard CFM's independent coupling produces crossing flow paths because each pair is sampled independently with no global transport structure. Our method inherits OT-CFM's minibatch plan and additionally certifies that the learned field stays within $r_B$ of it, so paths remain non-crossing. We expect a larger SDR gap versus CFM than versus OT-CFM, concentrated on multi-modal mixtures, confirming that the coupling structure — not just the regularizer — is the dominant contributor.

**vs. Demucs** — On a case where the mixture is heavily reverberant, Demucs produces over-smoothed estimates because its discriminative objective regresses to the conditional mean, collapsing fine source detail. Our method instead generates via a flow, so it can represent the full conditional distribution, and the certificate keeps that generation anchored to a valid transport plan. We do not expect to beat Demucs uniformly; rather we expect parity on clean stems and a visible advantage on reverberant or overlapping stems where the conditional mean is a poor estimate.

**vs. COT-FM without hinge penalty** — On a case where the minibatch OT plan is ambiguous (e.g., two stems with similar energy), removing the hinge penalty leaves the learned coupling free to drift from $\pi_B$ because no term penalizes deviation. We expect the ablation to reproduce OT-CFM-like SDR, confirming the hinge is causal.

**vs. COT-FM with residual-softmax q** — On a case where two distinct targets yield similar residuals (harmonically overlapping stems), the residual-softmax proxy assigns mass based on residual scale rather than transport, so the certificate becomes uninformative. We expect the endpoint-OT variant to outperform the residual-softmax variant on these ambiguous cases, validating the load-bearing assumption.

**Synthetic validation** — On a 2D Gaussian mixture with known global OT plan, we compute the endpoint-OT coupling $q$ and measure its Wasserstein distance to the true global plan. We expect this distance to be small (within $2 r_B$) and to decrease with training, validating that $q$ recovers true coupling mass.

### What would falsify this idea

If COT-FM's SDR gain over OT-CFM is uniform across all stem types and mixture conditions rather than concentrated on the ambiguous, harmonically overlapping cases where the certificate should bind, then the hinge penalty is not doing the work we claim and the central mechanism is wrong. Equally falsifying: if removing the hinge (the ablation) leaves SDR unchanged, the certificate is decorative rather than causal; or if the margin $W2_hat - r_B$ does not negatively correlate with per-batch SDR degradation (Pearson r > -0.1), the certificate is not a useful diagnostic.

## References

1. FLOMSS: Generative Music Source Separation by Flow Matching
2. Multi-Source Diffusion Models for Simultaneous Music Generation and Separation
3. Improving and generalizing flow-based generative models with minibatch optimal transport
4. Flow Matching for Generative Modeling
