# Environment-Invariant Temporal Kernels for Causal Brain Encoding

## Motivation

BrainLoRA's single-pass multi-timepoint head learns one temporal kernel mapping backbone features to time-resolved fMRI responses, and it is trained by average prediction accuracy alone — so nothing forces that kernel to represent a stationary causal mapping rather than a subject- and scanner-specific correlational shortcut. The ancestor work only correlated predicted and observed timecourses post hoc, and the frontier optimizes mean accuracy, so the field's structural property — time-resolved causal identifiability of the encoding map under distribution shift — is never tested or enforced across the entire line of work that BrainLoRA exemplifies. This is a meta-level gap: every model in this tree assumes stationarity of the feature-to-response kernel and none can distinguish a kernel that generalizes causally from one that memorizes environment-specific lag structure.

## Key Insight

If the true feature-to-response temporal kernel is a causal mechanism shared across subjects, stimuli, and scanners, then it must be simultaneously predictive and environment-invariant, so a kernel trained to be invariant across environment partitions is forced toward the shared causal component while environment-specific lag idiosyncrasies become uninformative.

## Method

**What it is.** EITK (Environment-Invariant Temporal Kernel) augments the BrainLoRA readout with an environment-conditional invariance objective on the temporal kernel. Inputs: frozen backbone features $h_t$ (from LoRA-adapted backbone), an environment label $e$ (subject / stimulus / scanning-parameter bin), and multi-timepoint fMRI targets $y_{t+1..t+K}$. Output: a single temporal kernel $\kappa(\tau)$ and time-resolved predictions $\hat{y}_{t+\tau}$ that are predictive and environment-invariant.

**How it works.**

```pseudocode
# Per training batch: features h_t, targets y_{t+1..t+K}, env label e
# Temporal kernel head (LoRA-adapted readout):
kappa = TemporalKernelHead(h_t)          # vector of length K, shared across envs
# Environment-specific hemodynamic response (HRF) parameters:
#   delay d_e in [-2, 2] TRs, dispersion s_e in [0.5, 2.0] (scalar per env)
#   HRF_e(tau) = exp(-(tau - d_e)^2 / (2 * s_e^2))  # Gaussian HRF, normalized to sum 1
# Effective kernel for env e:
kappa_e = conv(kappa, HRF_e)             # length K + K_hrf - 1, truncated to K
yhat  = sum_tau kappa_e[tau] * h_{t}     # single-pass multi-timepoint prediction

# L1: prediction loss (per-environment, so no env can dominate)
L_pred = mean_e( MSE( yhat_e, y_e ) )

# L2: environment-conditional contrastive invariance on kappa_shared
#   positives: same kappa_shared under different envs for same (subject-agnostic) stimulus;
#   negatives: kappa_shared from different stimuli in same env
z = Projector(kappa)                     # 64-d, L2-normalized
L_inv = InfoNCE( z, positives=kappa_across_envs_same_stim,
                     negatives=kappa_same_env_diff_stim, temperature=0.1 )

# L3: gradient-reversal domain classifier on kappa_shared (adversarial env removal)
L_dom = CE( DomainClassifier( GRL_lambda * kappa ), e )   # GRL flips gradient, lambda ramps 0->1

L_total = L_pred + alpha * L_inv + beta * L_dom    # alpha=0.5, beta=0.1
```

Environment partitions are constructed by binning (subject, stimulus family, scanning-parameter) into disjoint folds; held-out environments are never seen during training. Causal identifiability is evaluated by (i) held-out-environment generalization of $\kappa$ and (ii) counterfactual stimulus perturbations — swapping stimulus while holding subject/scanner fixed and checking that $\hat{y}$ responds in the direction predicted by $\kappa$ alone.

**Why this design.** We chose an environment-conditional contrastive term *plus* a gradient-reversal domain classifier over a single invariance penalty because the contrastive term shapes the *relative geometry* of kernels across environments (positives across envs, negatives within env) while the GRL term removes residual environment information that contrastive pairs cannot see; accepting the cost that two invariance signals can conflict and require the $\alpha,\beta$ trade-off. We chose to apply invariance on the *temporal kernel output* rather than on backbone features because BrainLoRA’s contribution is precisely the kernel-mediated single-pass prediction, and regularizing features would perturb the frozen backbone’s multimodal representation; the cost is that invariance is only enforced where the kernel is used. We chose per-environment prediction loss averaging over a pooled loss because pooled MSE lets the largest environment dominate and would make “invariance” trivially satisfiable by ignoring small environments; the cost is noisier gradients from small environments. A domain expert would not describe this as a variant of BrainLoRA: BrainLoRA has no environment variable and no invariance objective at all, whereas EITK’s training signal is defined by environment partitions and its target quantity is the invariant kernel, not average accuracy.

**Why it measures what we claim.** The contrastive InfoNCE on $\kappa$ measures *causal invariance of the temporal kernel* because it assumes that two kernels producing the same stimulus response under different subjects/scanners share the same underlying causal lag structure; this assumption fails when a subject-specific hemodynamic delay mimics a stimulus-driven lag, in which case the contrastive term reflects shared scanner/subject dynamics rather than causal stimulus-response structure. The GRL domain-classifier loss measures *absence of environment-specific information in $\kappa$* because it assumes that environment labels are the only channel through which scanner/subject idiosyncrasy enters the kernel; this fails when environment information is entangled with stimulus category (e.g., one subject saw only one stimulus family), in which case the classifier removes stimulus-relevant signal along with environment signal. Held-out-environment generalization of $\kappa$ measures *causal identifiability* because it assumes that a kernel which predicts unseen environments is the one supported by the shared causal mechanism rather than by environment-specific fitting; this fails when held-out environments are distributionally too close to training environments, in which case the metric reflects interpolation rather than causal transfer. Counterfactual stimulus perturbation of $\hat{y}$ measures *causal necessity of the stimulus for the predicted response* because it assumes that swapping the stimulus while holding subject/scanner fixed is a valid intervention on the causal variable; this fails when the backbone’s features are themselves correlated with subject identity, in which case the perturbation changes both stimulus and subject-proxy and the response shift is not attributable to stimulus alone.

## Contribution

(1) EITK, an environment-invariant temporal-kernel training framework that adds an environment-conditional contrastive loss and a gradient-reversal domain classifier to the BrainLoRA single-pass multi-timepoint head, yielding a kernel trained to be invariant across subject, stimulus, and scanning-parameter partitions. (2) A design principle: invariance must be enforced on the temporal kernel output rather than on backbone features, because the kernel is the only quantity in the single-pass head that carries the feature-to-response mapping and is therefore the correct locus for causal-identifiability constraints. (3) A held-out-environment and counterfactual-perturbation evaluation protocol that operationalizes time-resolved causal identifiability of brain encoding models, applicable across multimodal backbones and the Algonauts 2025 setting.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Algonauts 2025 fMRI benchmark | Multi-subject, multi-stimulus, held-out env folds available |
| Primary metric | Held-out-environment kernel transfer R² | Directly tests invariant-kernel generalization claim |
| Baseline | Frozen-backbone readout (no LoRA) | Tests whether invariance needs adapted backbone |
| Baseline | BrainLoRA pooled-MSE readout | Tests whether env-partitioned loss + invariance matters |
| Baseline | BrainLoRA + feature-level invariance | Tests kernel-level vs feature-level invariance placement |
| Ablation-of-ours | EITK minus GRL term | Isolates contrastive from adversarial invariance signal |
| Ablation-of-ours | EITK minus HRF decomposition (fixed HRF_e = delta) | Tests whether explicit HRF channel is load-bearing |

### Why this setup validates the claim

The central claim is that EITK's environment-conditional invariance on the *temporal kernel* yields a kernel that transfers to unseen environments and responds causally to stimulus perturbations. The dataset choice matters because Algonauts 2025 supplies genuinely disjoint environment folds (subject, stimulus family, scanning-parameter bins), so held-out-environment kernel transfer R² is a real transfer measurement rather than interpolation. The frozen-backbone baseline tests the sub-claim that invariance must be learned rather than inherited from the backbone; BrainLoRA pooled-MSE tests whether environment-partitioned prediction loss is doing work independent of invariance; BrainLoRA with feature-level invariance tests whether applying the penalty to the kernel (not features) is the load-bearing choice. The ablation isolates the two invariance signals so we can attribute any gain to contrastive geometry versus adversarial removal. The HRF-decomposition ablation tests whether the explicit environment-specific hemodynamic response channel is necessary to absorb subject/scanner lag, as the method now assumes. Held-out-environment kernel R² is the right primary metric because it is the quantity the theory names as identifiable: if the kernel is causal, it should predict new environments; if it is environment-fitted, it should collapse exactly there. Counterfactual perturbation is a secondary diagnostic that distinguishes causal from correlational kernels.

### Expected outcome and causal chain

**vs. Frozen-backbone readout (no LoRA)** — On a held-out subject whose hemodynamic delay shifts the effective lag by a few TRs, the frozen readout produces a kernel whose peak lag tracks that subject's delay because nothing in its training signal distinguishes subject dynamics from stimulus dynamics. Our method instead pulls kernels for the same stimulus across subjects together via InfoNCE and strips residual subject information via GRL, while the explicit HRF_e channel absorbs the subject-specific delay, so the shared kernel peak stays anchored to the stimulus lag. We expect a clear gap on held-out-subject folds and near-parity on held-in subjects.

**vs. BrainLoRA pooled-MSE readout** — On a small stimulus family that contributes few samples, pooled MSE lets the dominant environment drive the kernel, so on that small family the baseline's kernel is essentially the majority-environment kernel and its predictions are systematically mis-lagged. Our per-environment averaged prediction loss prevents any environment from dominating, so the kernel retains the small family's lag structure. We expect a visible gain concentrated on the minority-environment subset with little change on the majority subset.

**vs. BrainLoRA + feature-level invariance** — On a case where subject identity is entangled with stimulus category, feature-level invariance perturbs the frozen backbone's multimodal representation and degrades the shared signal for all subjects, so the baseline loses prediction accuracy on the entangled subset without gaining transfer. Our method applies invariance only at the kernel output, leaving the backbone representation intact, so we expect the baseline to show a transfer gain offset by a prediction-loss regression, while ours shows transfer gain with prediction parity.

**vs. EITK minus GRL term** — On a held-out environment whose idiosyncrasy is not represented by any contrastive positive pair (e.g., a scanner parameter bin with no cross-environment stimulus match), the contrastive-only variant leaves residual environment information in the kernel, so its held-out kernel R² degrades relative to full EITK. Our GRL term removes that residual, so we expect the ablation to match full EITK on environments with good positive coverage but to fall behind on poorly covered held-out environments.

**vs. EITK minus HRF decomposition** — On a held-out subject with a large hemodynamic delay, the variant without explicit HRF_e must encode that delay inside the shared kernel, so the invariance terms (InfoNCE and GRL) will strip the delay as environment information, degrading prediction for that subject. Our method allows HRF_e to absorb the delay, so the shared kernel remains stimulus-driven and the delay is captured by the environment-specific parameters. We expect the ablation to show lower prediction accuracy on subjects with atypical HRF delays and lower held-out kernel transfer R², with the gap concentrated on folds where HRF delay variability is largest.

### What would falsify this idea

If EITK's held-out-environment kernel transfer gain over BrainLoRA pooled-MSE were uniform across all environment folds rather than concentrated on folds where subject/stimulus/scanner idiosyncrasy is largest, the invariance mechanism would not be doing the work the theory claims. Equally, if counterfactual stimulus perturbations produced response shifts uncorrelated with the direction predicted by the learned kernel, the kernel would not be causally identifiable in the claimed sense.

## References

1. Scaling Parameter-Efficient Fine-Tuning for Brain Encoding
