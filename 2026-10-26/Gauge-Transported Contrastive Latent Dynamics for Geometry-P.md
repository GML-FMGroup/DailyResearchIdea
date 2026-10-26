# Gauge-Transported Contrastive Latent Dynamics for Geometry-Preserving World Models

## Motivation

Decoder-free latent world models such as Contrastive World Models (CWM) maximize a Deep InfoMax-style mutual information between predicted latents and future patch features, but MI is invariant to any reparameterization of the latent that preserves the estimate, so the model is free to discard fine-grained geometry and physical-law structure that a pixel decoder or simulator would otherwise force it to retain. This is the new bottleneck implied by the field's trajectory away from reconstruction: as DreamGen and HunyuanVideo-scale video generators and Moto's motion-token pretraining make pixel-space fidelity the expensive part, the convergent move is to drop the decoder, and CWM shows this is viable — yet without a decoder nothing pins down the *relative* geometry between latent states, so the latent can be MI-accurate and physically unfaithful at the same time. The root cause is structural: the contrastive target is a global MI estimate, and global MI is exactly the quantity that is blind to the group of latent reparameterizations under which geometric and physical detail lives.

## Key Insight

Conditioning the contrastive target on a local frame (a gauge) that is itself transported through time by the predicted dynamics breaks the reparameterization invariance of the MI objective at the level of *relative* latent geometry, because the gauge fixes the residual freedom to a single global transform that the transported-frame consistency term then constrains — so detail that global MI cannot see becomes the very quantity the loss is computed on.

## Method

**What it is.** Gauge-Contrastive Latent Dynamics (GCLD) is a decoder-free world model that learns latent dynamics from action-observation sequences by maximizing a contrastive objective whose positives and negatives are expressed in an action-relative local frame. Inputs: sequences of observations $o_t$, actions $a_t$. Outputs: a latent state $z_t$, a learned local frame $G_t$ (an orthonormal $k\times k$ matrix acting on a $k$-dim subspace of $z$), and predicted dynamics $f_\theta$. No pixel decoder, no simulator.

**How it works.**

```pseudocode
# Encoders
z_t      = E_obs(o_t)                      # global latent, dim d=256
G_t      = orth(E_frame(o_t))              # k x k orthonormal local frame, k=16 (default), k in {8,16,32} for sweep

# Dynamics prediction (recurrent, RSSM-style deterministic path)
h_t      = f_theta(h_{t-1}, z_{t-1}, a_{t-1})   # GRU, hidden dim 256
z_hat_t  = g_theta(h_t)                    # predicted next latent, 2-layer MLP (256->256->256, GeLU)
G_hat_t  = Transport(G_{t-1}, h_t)         # predicted gauge: G_{t-1} parallel-transported

# Gauge-relative coordinates (the only place contrastive loss is computed)
u_t      = G_t^T  @ P(z_t)                 # project z_t onto frame subspace, express in frame
v_hat_t  = G_hat_t^T @ P(z_hat_t)          # predicted latent in *transported* frame

# Contrastive objective (InfoNCE), computed in gauge-relative space
pos      = sim(v_hat_t, u_t)               # predicted vs. realized, same frame; sim = cosine similarity
neg      = { sim(v_hat_t, u_{t'}) : t' != t, sampled from batch + future patches }  # batch size 64, 64 negatives per positive
L_contrast = -log( exp(pos/tau) / sum_{neg} exp(sim/tau) )      # tau = 0.1

# Gauge-transport consistency (the term that pins relative geometry)
L_gauge  = || G_hat_t^T G_t - I ||_F^2 + || Transport(G_t, h_{t+1})^T G_{t+1} - I ||_F^2

# Total
L = L_contrast + lambda_g * L_gauge          # lambda_g = 0.5, annealed 0 -> 0.5 over 50k steps
```

$P(\cdot)$ is a fixed linear projection onto the $k$-dim subspace (implemented as a fixed random orthonormal matrix of shape $d\times k$, seed 0); $\text{Transport}(G,h)$ is a 2-layer MLP (input dim $k^2 + 256$, hidden 256, GeLU, output $k^2$) mapping $(G,h)\mapsto G'$ constrained to the orthogonal group via a Cayley parameterization: given skew-symmetric $A = (M - M^T)/2$ where $M$ is the MLP output reshaped to $k\times k$, $G' = (I - A)(I + A)^{-1} G$. Negatives are drawn both from other timesteps in the batch and from local patch features of future frames (as in CWM), but *all* similarities are computed after projection into the respective frames, so the contrastive signal is a function of relative coordinates, not raw latents.

**Why this design.** We chose to compute the contrastive loss in gauge-relative coordinates rather than on raw latents (as CWM does) because raw-latent MI is invariant to the orthogonal group acting on $z$, which is exactly the group under which relative geometry is encoded; the cost is that we must learn and transport a frame, adding a second encoder head and a transport MLP. We chose an explicit orthonormality constraint on $G$ (Cayley parameterization) over an unconstrained learned frame because an unconstrained frame can absorb arbitrary linear transforms and re-introduce the invariance we are trying to break; the cost is a constrained optimization that can be slower to converge early in training. We chose to *transport* the frame through the predicted dynamics rather than re-estimate it independently at each step because independent re-estimation would let the frame drift and make the consistency term vacuous; the cost is that transport errors compound over long rollouts, which is why $\lambda_g$ is annealed in rather than applied from step 0. A domain expert would not describe this as a variant of CWM: CWM's objective is a global Deep InfoMax bound on raw latents, whereas GCLD's objective is a bound on a *gauge-fixed* relative quantity, and the gauge-transport consistency term has no analogue in CWM or in Dreamer's KL-based latent alignment.

**Why it measures what we claim.** The quantity $\text{sim}(v\_hat_t, u_t)$ measures *relative geometric fidelity* between predicted and realized latent states because the assumption is that the learned frame $G_t$ is a sufficient statistic for the local geometry of $z_t$ on the $k$-subspace — i.e., that projecting onto $G_t$ and expressing in $G_t$'s coordinates preserves all geometric information relevant to the dynamics; this assumption fails when the relevant geometry lives outside the $k$-dim subspace (e.g., high-frequency texture that the frame encoder down-weights), in which case the contrastive loss reflects only subspace geometry and is blind to the discarded complement. The quantity $L\_gauge = \|G\_hat_t^T G_t - I\|_F^2$ measures *transport faithfulness* because the assumption is that the predicted dynamics $f_\theta$ induce a parallel transport on the frame bundle — i.e., that the frame should return to itself after a round trip through the predicted dynamics; this assumption fails when the dynamics are non-integrable (the transport is path-dependent and no consistent global frame exists), in which case $L\_gauge$ penalizes curvature that is physically real rather than an error, and the model is pushed toward artificially flat latent dynamics. The annealing of $\lambda_g$ from 0 to 0.5 measures *progressive commitment to geometric consistency* because the assumption is that early training should be free to find a coarse latent before the frame is pinned; this assumption fails when the coarse latent found in the unconstrained phase is geometrically incompatible with any consistent frame, in which case the annealed term fights the contrastive term and training stalls — which is why we monitor the contrastive loss during annealing and abort the schedule if it rises (pre-registered abort criterion: abort if $L_\text{contrast}$ rises more than 5% over any 5k-step window during annealing; on abort, freeze $\lambda_g$ at its current value and log the event).

**Assumptions and failure modes (D).** The method relies on two assumptions that we state explicitly and bound empirically. (A1) *Subspace sufficiency*: the $k$-dim frame subspace captures all geometry relevant to the dynamics. This fails when relevant geometry lies outside the subspace (e.g., high-frequency texture), in which case $L_\text{contrast}$ reflects only subspace geometry and is blind to the complement. (A2) *Integrability of transport*: the predicted dynamics induce a path-independent parallel transport, so $G_\text{hat}_t$ can be driven to $G_t$ by penalizing $\|G_\text{hat}_t^T G_t - I\|_F^2$ without distorting the latent dynamics. This fails when the latent dynamics have nonzero curvature (transport is path-dependent), in which case $L_\text{gauge}$ penalizes physically real curvature and flattens the dynamics. We do not assume A2 holds; we instrument it: during training we compute the round-trip transport error $\epsilon_t = \|\text{Transport}(\text{Transport}(G_t, h_{t+1}), h_{t+2})^T G_{t+2} - I\|_F$ and log the fraction of timesteps with $\epsilon_t > 0.1$ (path-dependence rate). We also log the residual energy $r_t = \|z_t - P(z_t)\|_2^2 / \|z_t\|_2^2$ (fraction of latent energy outside the $k$-subspace) to bound A1. These diagnostics are reported alongside retrieval accuracy so the assumptions are empirically bounded rather than asserted.

## Contribution

(1) A decoder-free latent dynamics framework, GCLD, that replaces the global MI objective of Contrastive World Models with a gauge-relative contrastive objective computed in a learned, dynamics-transported local frame, together with a gauge-transport consistency term that constrains the residual reparameterization freedom. (2) The design principle that geometric and physical detail in decoder-free world models can be pinned down not by adding a decoder but by making the contrastive target depend on *relative* latent coordinates, which is a structural fix to MI's reparameterization invariance rather than an auxiliary loss. (3) An analysis of when the gauge-transport assumption (integrability of the predicted dynamics) holds and a diagnostic — the contrastive-loss rise during $\lambda_g$ annealing — that flags non-integrable latent dynamics.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | CWM small-scale suite: default, distractors, natural backgrounds | Matches the invariance regime where gauge matters |
| Primary metric | Future-latent retrieval accuracy in gauge-relative space (top-1 over 64 negatives) | Directly probes relative geometry, not raw MI |
| Secondary metric | Round-trip transport error $\epsilon_t$ (path-dependence rate at threshold 0.1) and residual energy $r_t$ outside the $k$-subspace | Bounds assumptions A1 (subspace sufficiency) and A2 (integrability) |
| Baseline | CWM (raw-latent Deep InfoMax) | Tests whether gauge-fixing beats invariant MI |
| Baseline | Dreamer (pixel reconstruction) | Tests whether decoder-free avoids distractor absorption |
| Baseline | Momentum prediction | Tests whether dynamics, not just smoothness, are captured |
| Ablation-of-ours | GCLD without $L_\text{gauge}$ ($\lambda_g=0$) | Isolates the transport-consistency contribution |
| Ablation-of-ours | GCLD with $k \in \{8,16,32\}$ (default $k=16$) | Tests whether subspace choice is binding |
| Synthetic control | Ground-truth latent generated with known SO(k) nuisance rotation on a k-dim subspace | Demonstrates structural distinction: CWM's raw-latent MI cannot distinguish two relative poses while GCLD can |

### Why this setup validates the claim

The central claim is that computing the contrastive objective in gauge-relative coordinates, anchored by a transported orthonormal frame, yields latent dynamics that track relative geometry better than raw-latent MI (CWM) or reconstruction (Dreamer). The dataset choice is load-bearing: the default, distractor, and natural-background settings form a graded sequence where nuisance variation is exactly the orthogonal-group ambiguity that raw-latent MI is blind to. CWM tests the sub-claim that gauge-fixing beats invariance-preserving MI; if CWM matches us on the distractor subset, the gauge story is unnecessary. Dreamer tests the sub-claim that a decoder-free objective avoids absorbing distractor pixels into the latent; its reconstruction loss gives it a mechanism to model nuisance. Momentum prediction tests whether we capture action-conditioned dynamics rather than mere temporal smoothness, which would trivially score well on retrieval. The ablation ($\lambda_g=0$) is the decisive control: it keeps the frame-relative contrastive loss but removes transport consistency, so any gain attributable to the frame being *pinned* rather than merely *present* shows up as the ablation-vs-full gap. Gauge-relative retrieval accuracy is the right metric because it is computed in the same coordinate system the method optimizes in; a raw-latent metric would reintroduce the invariance we are trying to break and could not distinguish a correct frame from an arbitrary rotation of it. The synthetic control with a known SO(k) nuisance rotation provides a direct structural test: in that setting the ground-truth relative pose is known, so we can check whether CWM's raw-latent MI assigns equal similarity to two distinct relative poses (as invariance predicts) while GCLD's gauge-relative similarity separates them. The diagnostics $\epsilon_t$ and $r_t$ bound the two assumptions (A1, A2) that the losses silently depend on: if $r_t$ is large, subspace sufficiency is violated and the contrastive loss is blind to the complement; if the path-dependence rate is high, the integrability assumption fails and $L_\text{gauge}$ is penalizing real curvature. Reporting these alongside retrieval accuracy makes the causal chain from gauge-fixing to gain empirically checkable rather than asserted.

### Expected outcome and causal chain

**vs. CWM (raw-latent Deep InfoMax)** — On a case where the distractor set introduces an orthogonal nuisance transform (e.g., a rotating background patch that rotates the local latent geometry), CWM produces high raw-latent similarity between predicted and realized states regardless of whether the *relative* configuration is correct, because its InfoNCE bound is invariant to that rotation and cannot penalize a wrong relative pose. Our method instead expresses both predicted and realized latents in their transported frames, so a rotation that preserves raw similarity but breaks relative geometry becomes a visible negative, because the frame coordinates change under it. We expect a noticeable retrieval-accuracy gap concentrated on the distractor and natural-background subsets, with near-parity on the clean default subset. In the synthetic control with a known SO(k) rotation, we expect CWM's similarity to be identical (within noise) for two relative poses that differ by an orthogonal transform, while GCLD's gauge-relative similarity separates them by a margin that grows with the rotation angle.

**vs. Dreamer (pixel reconstruction)** — On a case where natural video backgrounds contain high-frequency texture uncorrelated with dynamics, Dreamer produces a latent that allocates capacity to reconstructing that texture, because its pixel decoder rewards modeling nuisance pixels, and its predicted latents drift toward background-dominated features. Our method instead never decodes pixels, and the contrastive signal in gauge-relative space only rewards coordinates that predict the realized frame-relative state, so texture outside the $k$-subspace is down-weighted rather than fit. We expect the gap to widen with visual complexity — small on default, moderate with distractors, largest on natural backgrounds — and to appear as Dreamer's retrieval degrading faster than ours as complexity rises. The residual-energy diagnostic $r_t$ should be larger for Dreamer than for GCLD on natural backgrounds, consistent with Dreamer allocating latent capacity to out-of-subspace texture.

**vs. Momentum prediction** — On a case where the action sequence changes the future trajectory (e.g., a turn that a constant-velocity predictor would miss), momentum prediction produces a predicted latent that continues the prior direction, because it has no action-conditioned dynamics and scores well only on smooth, low-action-variation segments. Our method instead conditions the recurrent dynamics on $a_{t-1}$ and transports the frame through $h_t$, so the predicted frame-relative state reflects the action-induced change. We expect a clear gap on high-action-variation segments and near-parity on quasi-static segments, which is the signature that the gain comes from action-conditioned dynamics rather than temporal smoothness.

### What would falsify this idea

If the full GCLD model's gain over CWM is uniform across the default, distractor, and natural-background subsets rather than concentrated where orthogonal nuisance is present, the gauge-fixing claim is wrong and the gain is attributable to something else. Likewise, if removing $L_\text{gauge}$ ($\lambda_g=0$) leaves retrieval accuracy unchanged, the transport-consistency mechanism is doing no work and the central geometric argument collapses. Additionally, if the path-dependence rate is high (e.g., >30% of timesteps with $\epsilon_t > 0.1$) and the $\lambda_g=0$ ablation matches the full model, the integrability assumption A2 is violated and the consistency term is not contributing as claimed.

## References

1. Contrastive World Models
2. DreamGen: Unlocking Generalization in Robot Learning through Video World Models
3. HunyuanVideo: A Systematic Framework For Large Video Generative Models
4. Moto: Latent Motion Token as the Bridging Language for Learning Robot Manipulation from Videos
5. Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture
6. Mastering diverse control tasks through world models
7. Context Autoencoder for Self-supervised Representation Learning
8. Masked Siamese Networks for Label-Efficient Learning
