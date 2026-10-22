# Motion-Adaptive Temporal Gap and Uncertainty-Weighted FCP for Autoregressive Video Generation

## Motivation

Autoregressive video diffusion methods such as FCP (Looking Ahead) rely on a fixed temporal gap and point prediction for future-evolution supervision, which structurally cannot calibrate supervision to the actual dynamics: rare or complex motion produces high-variance, multimodal futures that a fixed gap and an unweighted L2 loss treat identically to smooth motion, leaving evolution modeling unidentifiable exactly where it matters. This is not a tuning issue but a supervision-allocation failure shared across branches: the training signal is neither timed nor scaled by motion complexity, so the model learns an average evolution that washes out rare dynamics.

## Key Insight

The temporal horizon over which a future chunk is predictable, and the confidence with which it should be supervised, are both monotone functions of a motion-complexity statistic that can be estimated from the context chunk's optical-flow energy and latent variance, so conditioning the gap and the loss weight on that statistic makes the FCP objective identifiable for complex motion rather than averaging it away.

## Method

**What it is.** MATCH-FCP (Motion-Adaptive Temporal Calibration for Future-Chunk Prediction) replaces FCP's fixed-gap point prediction with a per-sample dynamic gap predictor and an uncertainty-weighted FCP loss, both driven by a motion-complexity score computed from the context chunk. Inputs: context chunk x_c, future chunk x_f, and a motion-complexity score s. Outputs: a predicted gap g(s) and a per-sample FCP loss weight w(s) applied to a Gaussian NLL head.

**Load-bearing assumption (stated explicitly).** The method assumes that the motion-complexity score s computed from the context chunk (flow energy + latent variance) is a monotone sufficient statistic for the predictability horizon and the appropriate supervision weight of the future chunk, so that a single closed-form monotone map g(s) and w(s) correctly calibrates supervision per sample. This assumption is load-bearing: if s is not monotone in the true predictability horizon (e.g., because flow is dominated by ego-motion or because latent variance tracks representation collapse), then both g(s) and w(s) are miscalibrated simultaneously. To make the calibration identifiable even if E_flow and V_lat are poor proxies, we (i) validate and calibrate g(s) and w(s) against an empirical predictability measure computed on held-out clips, and (ii) decouple the gap and weight into two separately calibrated monotone maps rather than one shared sigmoid.

**How it works.**

```pseudocode
# Per training sample, given context chunk x_c and a future chunk x_f
# 1. Motion-complexity score (no extra network; computed from context only)
flow = RAFT(x_c)                          # optical flow between frames of x_c
E_flow = mean(||flow||^2)                 # flow-energy term
V_lat  = mean(var(z_c, dim=time))         # latent variance of context tokens z_c
s = log1p(E_flow) + lambda_v * V_lat      # lambda_v = 0.5 (initial; calibrated below)

# 2. Dynamic gap predictor (closed-form, not a learned head)
#    Predictability horizon scales inversely with motion energy.
g(s) = clip( round( g_min + (g_max - g_min) * exp(-kappa * s) ), g_min, g_max )
#    g_min=1, g_max=4 chunks, kappa=1.0 (initial; calibrated below)

# 3. Select the supervised future chunk at gap g(s)
x_f = chunk_at_offset(x_c, g(s))

# 4. Uncertainty-weighted FCP loss with a Gaussian NLL head
mu, log_sigma2 = model_fcp(x_c, x_f)      # predicts mean and log-variance
L_fcp = 0.5 * ( exp(-log_sigma2) * ||x_f - mu||^2 + log_sigma2 )
w(s)  = sigmoid(alpha * s + beta)         # alpha=1.0, beta=0.0 (initial; calibrated below)
L_total = L_denoise + gamma * w(s) * L_fcp  # gamma = 0.1 (FCP weight)

# 5. Calibration of g(s) and w(s) on held-out clips (done once before main training)
#    Empirical predictability measure: for each held-out clip, compute the FCP NLL
#    of the future chunk at offset d under a frozen probe model (the same FCP backbone
#    trained for 10k steps with fixed gap=2 and constant weight=1).
#    For each motion-score bin (deciles of s), record NLL(d) for d in {1,2,3,4}.
#    Fit g(s) by finding the largest d such that NLL(d) <= NLL(1) + tau, tau=0.1 nats.
#    Fit w(s) by regressing the inverse of the empirical NLL variance across clips in the bin
#    onto s with a logistic link, yielding alpha and beta.
#    Report the empirical g(s) curve against the assumed exponential to validate or refute the prior.
```

**Why this design.** We chose a closed-form, monotone gap predictor over a learned gap head because a learned head would be trained on the same FCP loss it is meant to calibrate, creating a circular objective in which the gap collapses to whatever minimizes loss rather than tracking motion; the exponential-decay form encodes the structural prior that predictability horizon shrinks with motion energy, accepting the cost that the decay constant kappa must be tuned per dataset (we replace hand-tuning with the held-out calibration procedure in step 5). We chose a Gaussian NLL head over a point-prediction L2 loss because L2 forces the model to regress the mean of a multimodal future, washing out rare dynamics, whereas NLL lets the model express high aleatoric variance for complex motion and down-weight those samples automatically; the cost is an extra output channel and a slightly noisier gradient early in training. We chose a context-only motion score (flow energy plus latent variance) over a future-aware score because the latter would leak information from the target chunk and make the gap trivially predictive; the cost is that the score is a proxy for future complexity and can underestimate abrupt motion onsets (we mitigate this by calibrating g(s) and w(s) against an empirical predictability measure on held-out clips, as in step 5). We chose to fold the motion score into the loss weight w(s) rather than use a separate curriculum because a single monotone weight couples gap and supervision strength to the same statistic, avoiding a two-controller design; the cost is less flexibility if gap and weight should diverge (we decouple them into two separately calibrated monotone maps g(s) and w(s), each fitted to its own empirical target, so a misestimated s does not corrupt both simultaneously). A domain expert would not describe this as a variant of Diffusion Forcing or MDT: those methods fix the noise schedule and masking pattern independently of motion, whereas MATCH-FCP's gap and weight are both deterministic functions of a measured motion statistic, so the supervision surface itself changes per sample.

**Why it measures what we claim.** The flow-energy term E_flow measures motion complexity because the assumption that optical-flow magnitude is monotone in the number of independently moving surfaces and their speed makes it a sufficient statistic for how multimodal the next chunk is; this assumption fails when flow is dominated by camera motion or textureless regions, in which case E_flow reflects ego-motion rather than object complexity. The latent-variance term V_lat measures the model's own uncertainty about context evolution because the assumption that token-level variance across time tracks dynamical instability makes it a proxy for future multimodality; this assumption fails when the latent space is anisotropic or poorly calibrated early in training, in which case V_lat reflects representation collapse rather than dynamics. The dynamic gap g(s) measures the predictability horizon because the exponential-decay assumption that horizon shrinks with motion energy is the structural prior; this assumption fails for periodic or oscillatory motion where high flow energy is predictable, in which case g(s) underestimates the useful horizon. The NLL weight w(s) measures the appropriate supervision strength because the assumption that high-complexity samples should be down-weighted to avoid mean-regression makes sigmoid(s) a calibrated confidence; this assumption fails when complex motion is the target capability rather than noise, in which case w(s) suppresses exactly the samples we want to learn from. (We name this missing equivalence explicitly: w(s) measures appropriate supervision strength under assumption A that high-complexity samples are aleatoric noise rather than the target capability; the failure mode F under which A breaks is when complex motion is the capability of interest, and we add a 2x2 ablation crossing {dynamic vs fixed gap} x {uncertainty weight vs constant} to disentangle the two confounded controllers.)

## Contribution

['A motion-adaptive FCP training objective (MATCH-FCP) that replaces the fixed temporal gap and point prediction with a closed-form motion-complexity-conditioned gap predictor and a Gaussian-NLL uncertainty-weighted FCP loss, computed entirely from the context chunk without an auxiliary network.', 'A design principle: coupling the temporal gap and the supervision weight to a single motion-complexity statistic makes the FCP objective identifiable for complex motion, whereas decoupling them (learned gap head or separate curriculum) reintroduces the mean-regression failure.', 'An analysis of when the context-only motion score is a valid proxy for future complexity (object motion vs. ego-motion, periodic motion) that delineates the operating regime of adaptive FCP supervision.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Something-Something V2 video clips | Motion-diverse clips with varied object dynamics |
| Primary metric | Long-horizon rollout FVD | Directly captures future-evolution fidelity |
| Baseline | Standard teacher-forced AR (no FCP) | Tests whether motion-calibrated supervision matters at all |
| Baseline | Fixed-gap FCP (original) | Tests whether dynamic gap beats static gap |
| Baseline | Diffusion Forcing | Tests claim against motion-agnostic noise/mask schedules |
| Baseline | Diffusion Forcing with motion-conditioned noise/mask schedule (same s as input) | Tests whether motion-conditioning alone explains gains |
| Baseline | MDT with motion-conditioned masking (same s as input) | Tests against motion-conditioned variant of MDT |
| Ablation-of-ours | MATCH-FCP with constant w(s)=1 | Isolates contribution of uncertainty weighting |
| Ablation-of-ours | MATCH-FCP with fixed gap g=2 | Isolates contribution of dynamic gap |
| Ablation-of-ours | 2x2 crossing {dynamic vs fixed gap} x {uncertainty weight vs constant} | Disentangles the two confounded controllers |

### Why this setup validates the claim

The central claim is that supervision should adapt to measured motion complexity, so the test must separate *whether FCP helps* from *whether motion-adaptivity helps over a fixed FCP schedule* and from *whether motion-conditioning alone (without our specific gap/weight maps) explains the gain*. Standard teacher-forced AR tests the first sub-claim: if FCP's future-chunk supervision is doing real work, removing it entirely should degrade long-horizon stability. Fixed-gap FCP tests the second and most important sub-claim: it has the same FCP machinery but a static gap, so any gain of MATCH-FCP over it is attributable specifically to the dynamic gap g(s), not to FCP per se. Diffusion Forcing tests the third sub-claim — that making the supervision surface a function of a measured motion statistic beats motion-agnostic schedules — by holding the general "train on multiple horizons" idea constant while removing motion conditioning. The motion-conditioned Diffusion Forcing and MDT baselines test whether simply feeding the same motion statistic s as an input conditioning signal (to modulate noise/mask schedules) is sufficient, so novelty rests on an empirical separation from motion-conditioned variants rather than on a definitional claim. The constant-weight and fixed-gap ablations, together with the 2x2 crossing, isolate the two mechanisms (dynamic gap and uncertainty weighting) and disentangle their contributions: a gain that survives the constant-weight ablation is attributable to the gap, a gain that vanishes is attributable to the weight, and the 2x2 reveals any interaction. FVD on long rollouts is the right metric because the predicted effect is about future-evolution fidelity over horizons where mean-regression and gap mismatch accumulate; a per-frame short-horizon metric would be insensitive to exactly the failure mode we predict. We also report the empirical g(s) curve from held-out calibration against the assumed exponential to validate or refute the structural prior, and we report wall-clock/compute overhead relative to fixed-gap FCP to quantify the cost of the added pipeline.

### Expected outcome and causal chain

**vs. Standard teacher-forced AR (no FCP)** — On a clip with fast, multi-object motion, the baseline supervises only the immediately next chunk, so it learns a sharp one-step predictor that drifts once rollout exceeds a few steps and averages over divergent futures. Our method instead supervises a motion-chosen future chunk with a Gaussian NLL head, so it is trained to represent the spread of plausible futures rather than a single mean, and the weight keeps high-complexity samples from dominating. We expect a clear long-horizon FVD advantage concentrated on high-motion clips, with near-parity on low-motion clips.

**vs. Fixed-gap FCP (original)** — On a clip where motion energy is high and the predictability horizon is short, the fixed-gap baseline still supervises a distant chunk whose content is essentially unpredictable, injecting noise into the FCP gradient and pulling the model toward a blurred average. Our method instead shrinks g(s) toward g_min under high s, so it supervises a chunk that is actually predictable, and down-weights the residual hard samples via w(s). We expect MATCH-FCP to beat fixed-gap FCP specifically on the high-motion subset, with the gap between them shrinking to near zero on low-motion clips — a motion-conditional advantage, not a uniform one.

**vs. Diffusion Forcing** — On a clip with abrupt motion onset, the baseline's fixed noise schedule and masking pattern were tuned for average dynamics, so it allocates supervision uniformly across timesteps and under-trains the transition region where motion changes fastest. Our method instead reads the motion statistic from context and both shifts the supervised gap and scales the loss, concentrating capacity on the transition. We expect the largest FVD margin on clips containing motion onsets or direction changes, and comparable performance on smooth, stationary clips.

**vs. Diffusion Forcing with motion-conditioned noise/mask schedule** — This baseline receives the same motion statistic s as an input conditioning signal to modulate its noise/mask schedule, so it tests whether motion-conditioning alone is sufficient. If MATCH-FCP's advantage over this baseline is uniform across motion-complexity bins rather than concentrated on high-motion or motion-onset clips, then the specific closed-form gap and weight maps are not doing what we claim and the gain is a generic motion-conditioning effect. We expect MATCH-FCP to retain a margin on high-motion clips because its gap and weight are calibrated to an empirical predictability measure, whereas the baseline's schedule is only conditioned on s without that calibration.

**vs. MDT with motion-conditioned masking** — Similar to the Diffusion Forcing motion-conditioned baseline, this tests whether a motion-conditioned masking pattern in MDT can match MATCH-FCP. We expect MATCH-FCP to outperform on clips with abrupt motion onsets because its gap shrinks and weight down-weights unpredictable chunks, whereas the baseline's masking is only conditioned on s without a calibrated horizon.

### What would falsify this idea

If MATCH-FCP's advantage over fixed-gap FCP is uniform across motion-complexity bins rather than concentrated on high-motion or motion-onset clips, the dynamic gap is not doing what we claim and the gain is a generic FCP effect. Likewise, if the constant-weight ablation matches full MATCH-FCP everywhere, the uncertainty-weighting mechanism is inert and the central coupling claim is wrong; and if the empirical g(s) curve from held-out calibration deviates substantially from the assumed exponential, the structural prior is refuted.

## References

1. Looking Ahead: Modeling Future Evolution in Autoregressive Video Generation
2. Diffusion Forcing: Next-token Prediction Meets Full-Sequence Diffusion
3. Masked Diffusion Transformer is a Strong Image Synthesizer
4. AR-Diffusion: Auto-Regressive Diffusion Model for Text Generation
5. Diffusion-LM Improves Controllable Text Generation
6. Continuous diffusion for categorical data
7. DiffuSeq: Sequence to Sequence Text Generation with Diffusion Models
8. Text Generation with Diffusion Language Models: A Pre-training Approach with Continuous Paragraph Denoise
