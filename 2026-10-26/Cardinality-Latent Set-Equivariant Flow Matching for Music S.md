# Cardinality-Latent Set-Equivariant Flow Matching for Music Source Separation

## Motivation

Existing generative separation models—FLOMSS, multi-source diffusion models, and their ancestors—condition the learned vector field on a fixed, known source inventory with isolated ground-truth stems, so the conditional field has no defined behavior when the number or timbre of sources at inference differs from training. This structural limitation is shared across the paper-tree: source cardinality and identity are treated as a training-time constant rather than a latent variable, making the source posterior unidentifiable from the mixture alone. As a result, these models cannot represent a posterior over variable-length source sets, and their separation degrades or is undefined off the fixed-inventory support.

## Key Insight

Making the flow-matching vector field exactly equivariant to source permutation and invariant to source count—via a cardinality-conditioned coupling estimated from the mixture—turns source cardinality into an identifiable latent variable, so the mixture-only likelihood defines a proper posterior over variable-length source sets rather than a fixed-inventory conditional.

## Method

**What it is.** We introduce CLSE-FM (Cardinality-Latent Set-Equivariant Flow Matching), a generative separation framework that learns a posterior over variable-length sets of music sources. Its inputs are a stereo mixture waveform and a latent cardinality variable K; its outputs are K separated source waveforms whose ordering is arbitrary and whose count is inferred, not fixed.

**How it works.**

```pseudocode
# Training (simulation-free, per minibatch)
Input: mixture x, ground-truth stem set S = {s_1,...,s_K} (K varies per example, K_max=6)
1. Sample time t ~ U[0,1] and noise eps ~ N(0, I) for each stem.
2. Form conditional flow path: s_i(t) = (1-t) eps_i + t s_i  (OT path).
3. Cardinality posterior q_phi(K | x):
     a. Compute mixture embedding h = Enc(x), where Enc is a 6-layer 1D-convolutional encoder (channels 32,64,128,256,256,256; kernel size 5; stride 2; GeLU) followed by global mean-pooling to a 256-dim vector.
     b. Predict a categorical distribution over K in {1,...,K_max} via a 2-layer MLP head (hidden=256, GeLU, softmax over K_max=6 classes).
     c. During training, q_phi is trained with cross-entropy against true K.
4. Set-equivariant vector field v_theta(s(t), t, x, K):
     a. Encode each noisy stem s_i(t) with a shared stem encoder E: 4-layer 1D-convolutional network (channels 32,64,128,256; kernel size 5; stride 2; GeLU) followed by global mean-pooling to a 256-dim vector.
     b. Apply permutation-equivariant attention across stems AND cross-attention to mixture encoding h: 4 blocks of multi-head attention (8 heads, dim=256, feedforward dim=1024, GeLU).
     c. Condition every layer on K via FiLM modulation: scale and shift parameters are produced by a 2-layer MLP (hidden=256, GeLU) from a 64-dim learned embedding of K (one embedding per integer 1..K_max).
     d. Decode to per-stem velocity predictions v_i via a 4-layer 1D-convolutional decoder (channels 256,128,64,32; kernel size 5; stride 2; GeLU) with a final linear projection to waveform dimension.
5. Loss = E_{t,eps,S} [ (1/K) * sum_i || v_theta(s_i(t), t, x, K) - (s_i - eps_i) ||^2 ]
         + lambda_ce * CE(q_phi(K|x), K_true)   (lambda_ce = 0.1)
6. Enforce permutation invariance: the loss is invariant to reordering S because the
   attention is symmetric and the sum over i is order-independent.

# Inference (mixture-only)
Input: mixture x
1. Compute cardinality posterior q_phi(K|x) over K in {1,...,K_max}.
2. For each K in {1,...,K_max} with q_phi(K|x) > 0.01 (at most 6 values):
     a. Sample K noise vectors {eps_i} ~ N(0,I).
     b. Integrate ODE: ds_i/dt = v_theta(s_i(t), t, x, K) from t=0 to 1
        using an RK4 solver with 20 steps.
     c. Obtain candidate set S_K = {s_i}.
3. Form the final output as a weighted union over K: for each K, include the set S_K with weight q_phi(K|x). (In practice, we output the set from the K with highest q_phi(K|x) as the primary estimate, and report the full posterior-weighted set for evaluation of calibration.)
4. Mixture consistency is enforced by a cardinality-aware projection: compute residual r = x - sum_{K} q_phi(K|x) * sum_{i in S_K} s_i. Add r equally to all stems in the MAP set, and also report the residual energy as an 'unexplained source' term in diagnostics.
```

**Why this design.** We chose a set-equivariant attention architecture over a fixed-slot U-Net because the latter cannot represent a variable number of sources without padding and masking, which breaks exact permutation equivariance; the cost is that attention over stems scales quadratically with K, but K is small (≤ 6 in music). We chose to condition the vector field on K via FiLM modulation rather than concatenating a one-hot vector, because FiLM injects cardinality at every layer without inflating the input dimension, accepting a slightly larger parameter count. We chose a separate cardinality posterior q_phi(K|x) trained with cross-entropy rather than inferring K from the flow itself, because the flow's likelihood is intractable and a discriminative head gives a direct, simulation-free training signal; the trade-off is that q_phi may be miscalibrated for out-of-distribution mixtures, which we mitigate by training on variable-K mixtures. We chose OT paths over diffusion paths because they yield straighter flows and fewer ODE steps, accepting that minibatch OT coupling introduces approximation error. A domain expert would not describe this as a variant of FLOMSS because FLOMSS conditions on a fixed inventory and has no cardinality latent; nor is it a simple combination of flow matching and a classifier—the key is that the vector field is made exactly equivariant and cardinality-invariant, which no prior flow-based separator does.

**Why it measures what we claim.** The cardinality posterior q_phi(K|x) measures the latent source count because we assume that the mixture embedding h is a sufficient statistic for K; this assumption fails when two sources are perfectly correlated in the mixture (e.g., identical timbre and phase), in which case q_phi reflects the effective rank of the mixture rather than the true count. The set-equivariant vector field v_theta measures the conditional velocity for a source set of size K because we assume that permutation equivariance and cardinality conditioning make the field's behavior well-defined for any K; this assumption fails when K exceeds K_max or when the mixture contains instruments never seen during training, in which case the field extrapolates from the nearest cardinality and may produce artifacts. The mixture consistency projection measures fidelity to the observed mixture because we assume that the residual x - sum_i s_i is due to model error rather than a missing source; this assumption fails when K_hat underestimates the true K, in which case the residual absorbs the unmodeled source and the projection smears it across all stems. The cross-entropy loss on q_phi measures cardinality identification because we assume that the training set covers the range of K and instrument combinations; this assumption fails for open-set mixtures, in which case q_phi reflects the closest training cardinality rather than the true one.

## Contribution

['A novel generative separation framework, CLSE-FM, that defines a flow-matching vector field exactly equivariant to source permutation and invariant to source count via cardinality-conditioned FiLM modulation, enabling a posterior over variable-length source sets.', 'A mixture-only cardinality posterior q_phi(K|x) trained jointly with the flow, which makes source cardinality an identifiable latent variable without requiring isolated stems at inference.', 'A set-equivariant attention architecture with mixture-consistency projection that produces separated stems for an inferred number of sources, along with a training procedure that re-weights the flow-matching regression target by the cardinality posterior without nested sampling.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Slakh2100 (variable-K mixtures) | Natural variable stem counts per track |
| Primary metric | SDR vs. true K, stratified by K | Isolates cardinality-correctness effect |
| Baseline | FLOMSS | Fixed-inventory flow separator, no cardinality latent |
| Baseline | Demucs | Discriminative fixed-slot separator |
| Baseline | Multi-Source Diffusion Model | Generative separator with fixed source inventory |
| Ablation-of-ours | CLSE-FM with K fixed to training-mean | Removes cardinality inference, keeps equivariant field |
| Diagnostic | q_phi calibration (ECE) and sufficiency probe | Tests if h is sufficient for K |
| Diagnostic | Synthetic near-identical sources | Quantifies effective-rank failure |
| Open-set split | Hold out 2 instrument classes from training | Tests off-support generalization |

### Why this setup validates the claim

The central claim is that making the vector field exactly set-equivariant and cardinality-conditioned lets a single model separate variable-length source sets, and that a discriminative cardinality posterior q_phi is what supplies the correct K at inference. Slakh2100 is the right dataset because its tracks genuinely vary in stem count, so stratification by true K directly probes the sub-claim that performance should hold across K rather than degrade as K departs from a fixed training mode. FLOMSS tests the fixed-inventory sub-claim: it conditions on a known inventory, so any gap concentrated on non-modal K is attributable to the missing cardinality latent. Demucs tests the fixed-slot sub-claim: its padded slots cannot express arbitrary cardinality, so its errors should cluster where K differs from its slot count. The diffusion baseline tests whether the equivariance-plus-FiLM conditioning, not generativity alone, drives the effect. The ablation is the decisive control: it keeps the equivariant field but removes q_phi, so if the gain survives without cardinality inference, the claim that q_phi matters is falsified. SDR stratified by K is the right metric because the predicted effect is a K-dependent pattern, not a uniform average improvement; an aggregate SDR would average away exactly the signal the theory predicts. The q_phi calibration (ECE) and sufficiency probe directly test the load-bearing assumption that h is a sufficient statistic for K: if the probe accuracy is near chance or ECE is high, the cardinality latent is not identifiable from the mixture. The synthetic near-identical sources experiment quantifies the failure mode where q_phi reflects effective rank rather than true count. The open-set split tests whether the method generalizes to unseen instruments, as claimed for off-support mixtures.

### Expected outcome and causal chain

**vs. FLOMSS** — On a track with five stems where the training mode is four, FLOMSS conditions on a fixed inventory and must either pad or drop a source, so it leaks one instrument into another's estimate. Our method instead reads K_hat from q_phi and conditions every attention layer on it via FiLM, so the field is defined for the true cardinality. We expect near-parity on modal-K tracks and a clear SDR gap on off-modal-K tracks, growing with |K − mode|.

**vs. Demucs** — On a three-stem track, Demucs's fixed slots force a bijection between slots and sources, so with mismatched cardinality it assigns two slots to one source or splits a source across slots, producing characteristic bleed and phantom energy. Our method instead produces an unordered set whose size is K_hat, so no slot assignment is forced and permutation symmetry is exact. We expect Demucs to show high variance across K and our method a flatter SDR-vs-K curve, with the largest gap at low K.

**vs. Multi-Source Diffusion Model** — On a mixture with an instrument absent from its fixed inventory, the diffusion baseline must map that energy onto a known source, smearing the unseen instrument into neighbors. Our method instead lets q_phi raise K_hat and the equivariant field allocate a fresh set element, so the extra source is modeled rather than absorbed. We expect a noticeable gap on open-set or high-K subsets and parity on in-inventory subsets.

**vs. CLSE-FM with K fixed to training-mean** — This ablation removes cardinality inference but keeps the equivariant field. If the full model's gain over this ablation is uniform across K strata, then q_phi contributes nothing and the cardinality-latent premise is wrong. We expect the full model to outperform the ablation primarily on off-modal-K tracks, with parity on modal-K tracks.

**Diagnostic: q_phi calibration and sufficiency** — We expect the sufficiency probe (a logistic regression on h to predict K) to achieve accuracy well above chance but below oracle, and ECE to be low on in-distribution K but higher on open-set mixtures. On synthetic near-identical sources, we expect q_phi to underestimate K, reflecting effective rank.

**Open-set split** — On held-out instrument classes, we expect our method to maintain reasonable SDR by increasing K_hat, while baselines degrade more sharply.

### What would falsify this idea

If our SDR gain over FLOMSS and Demucs is uniform across K strata rather than concentrated on off-modal and high-K tracks, then cardinality conditioning is not the causal driver and the central claim fails. Likewise, if the fixed-K ablation matches full CLSE-FM across all strata, then q_phi contributes nothing and the cardinality-latent premise is wrong. Additionally, if the sufficiency probe shows that h is not predictive of K (accuracy near chance) or if q_phi calibration is poor (high ECE) across all K, then the load-bearing assumption that h is a sufficient statistic for K is false, and the method's cardinality inference is not identifiable from the mixture.

## References

1. FLOMSS: Generative Music Source Separation by Flow Matching
2. Multi-Source Diffusion Models for Simultaneous Music Generation and Separation
3. Improving and generalizing flow-based generative models with minibatch optimal transport
4. Flow Matching for Generative Modeling
