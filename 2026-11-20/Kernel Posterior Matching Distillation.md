# Kernel Posterior Matching Distillation

## Motivation

Distribution-based diffusion distillation methods such as PMD (Posterior Matching Distillation) match only a low-order statistic—the tilted posterior mean—of the student's conditional distribution to the teacher's, leaving the full conditional posterior unconstrained; DisBack similarly matches only endpoint score differences along a recorded degradation path. Because these objectives never enforce equality of the entire conditional distribution at each noise level, the distilled student can match the mean while differing in higher-order structure, mode coverage, and diversity—precisely the properties that distribution-based distillation claims to preserve. The root cause is structural: mean matching is a necessary but not sufficient condition for distributional equality, and no existing distillation objective in this family supplies a sufficient condition.

## Key Insight

Matching the teacher's full conditional posterior at each noise level is equivalent to matching the gradient of the student's log-density to the teacher's score, so a conditional score-matching objective with a kernel-smoothed target enforces distributional equality rather than merely mean equality.

## Method

**What it is.** KPMD (Kernel Posterior Matching Distillation) is a distribution-based distillation objective for one-step and any-step stochastic diffusion students. Its inputs are a pretrained teacher score function $s_T(x_t,t)$, a student generator $G_\theta(x_t,t,\xi)$ with noise $\xi$, and a batch of noisy states $x_t$; its output is a trained student whose conditional output distribution at every trained noise level equals the teacher's posterior $p_T(x_0\mid x_t)$.

**How it works.**

```pseudocode
# Per training step, for each noise level t in the trained grid:
# 1. Sample noisy state and student outputs
x_t ~ q_t(x_t)                      # forward-noised data
{x_0^+[i]}_{i=1..K} ~ G_theta(x_t, t, xi)   # K student samples, K=8

# 2. Build kernel-smoothed conditional score targets from student samples
#    Kernel: Gaussian with bandwidth h_t (set per noise level as h_t = c * sigma_t, c=0.1)
#    Target score at each student sample:
#    s_hat(x_0^+[i]) = sum_j k_h(x_0^+[i], x_0^+[j]) * s_T(x_0^+[j], t_shift) / sum_j k_h(...)
#    where t_shift is PMD's Gaussian-corruption tilt: evaluate teacher at shifted input
#    corresponding to the exponentially tilted posterior mean.

# 3. Conditional score-matching loss:
#    L = E_{x_t} E_{x_0^+ ~ G_theta} || s_theta(x_0^+, t) - s_hat(x_0^+) ||^2
#    where s_theta is the student's implicit score, obtained by differentiating
#    the log-density of G_theta via the reparameterization trick:
#    s_theta(x_0^+, t) = - (x_0^+ - mu_theta(x_t,t)) / sigma_theta(x_t,t)^2
#    (student parameterizes a Gaussian posterior per noise level).

# 4. Update theta by gradient descent on L.
```

**Why this design.** We chose a kernel-smoothed conditional score-matching objective over PMD's direct tilted-mean matching because mean matching is a necessary but not sufficient condition for posterior equality, accepting the cost that the kernel bandwidth $h_t$ introduces a smoothing bias that must be annealed. We chose a Gaussian student posterior parameterization over an implicit generator because it yields a closed-form student score via the reparameterization trick, accepting the cost that the student's within-noise-level distribution is restricted to a Gaussian family (a restriction we relax by conditioning the Gaussian parameters on $x_t$ through a high-capacity network, so the marginal over $x_t$ remains flexible). We chose to reuse PMD's Gaussian-corruption tilt to evaluate the teacher at shifted inputs rather than requiring direct teacher posterior samples, accepting the cost that the target score is only exact under Gaussian corruption; this is the same assumption PMD makes, so the comparison isolates the effect of full-distribution matching versus mean matching. A domain expert would not describe KPMD as a variant of PMD because PMD's loss is a first-moment matching loss (a squared error between student sample averages and a teacher mean), whereas KPMD's loss is a score-matching loss over the full conditional density; the two coincide only in the limit of zero kernel bandwidth and Gaussian student, which is not the operating regime.

**Why it measures what we claim.** The computational quantity $\|s_\theta(x_0^+,t) - \hat{s}(x_0^+,t)\|^2$ measures conditional posterior distributional equality because, under the assumption that the kernel-smoothed target $\hat{s}$ converges to the true teacher posterior score as $h_t \to 0$ and $K \to \infty$, score matching is equivalent to matching the full conditional density (Hyvärinen's identity); this assumption fails when $K$ is small and $h_t$ is not annealed, in which case the loss reflects a smoothed mean-like statistic rather than full distributional equality. The kernel bandwidth $h_t$ measures the trade-off between variance of the target estimate and bias toward the mean; this assumption fails when the teacher posterior is multimodal with modes separated by more than $h_t$, in which case the kernel average collapses modes and the loss again reflects only a local mean. The Gaussian student parameterization measures the student's conditional density because the reparameterization score is exact for Gaussian posteriors; this assumption fails when the true student posterior is non-Gaussian, in which case the score is a Gaussian approximation and the loss reflects a projected score mismatch.

**Load-bearing assumption (explicit).** The kernel-smoothed, student-sample-based target score $\hat{s}(x_0^+[i])$ computed with $K=8$ samples and bandwidth $h_t = 0.1\sigma_t$ is a sufficiently accurate estimator of the teacher's true conditional posterior score that minimizing $\|s_\theta - \hat{s}\|^2$ enforces full conditional posterior equality rather than merely a smoothed mean-like statistic. This assumption is load-bearing because the entire claimed advantage over PMD rests on the target being a full-posterior score rather than a mean-like statistic. It can fail when $K$ is small and $h_t$ is not annealed (the estimator degenerates to a local mean) or when the teacher posterior is multimodal with mode separation exceeding $h_t$ (the kernel average collapses modes). We make this assumption explicit and test it via the synthetic multimodality experiment and the $K$-sweep and $h_t$-annealing ablations described in the experiment section.

## Contribution

["A kernel-smoothed conditional score-matching distillation objective (KPMD) that replaces PMD's tilted posterior-mean matching with full conditional posterior matching, using a Gaussian-kernel density estimator over student samples to construct teacher score targets without requiring teacher posterior samples.", 'A design principle: full conditional posterior equality in diffusion distillation can be enforced by matching conditional scores rather than conditional means, and the kernel bandwidth provides an explicit bias-variance knob that interpolates between PMD-like mean matching and full distribution matching.', 'An empirical characterization of when full posterior matching improves mode coverage and diversity relative to mean-matching distillation, and when the kernel smoothing bias dominates.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | ImageNet 64×64 | Standard distillation benchmark with multimodal posteriors |
| Primary metric | FID at 1 step | Directly reflects distributional match quality |
| Baseline | PMD (tilted-mean matching) | Isolates mean-matching vs full-distribution matching |
| Baseline | CED (contrastive energy distillation) | Tests against NCE-based one-step distillation |
| Baseline | DisBack (distribution backtracking) | Tests against trajectory-based distillation |
| Ablation-of-ours | KPMD with fixed large bandwidth h_t | Probes kernel-smoothing bias contribution |
| Ablation-of-ours | KPMD with K ∈ {2, 8, 32} | Tests sensitivity to sample count K |
| Ablation-of-ours | KPMD with h_t-annealing schedule (h_t = c * sigma_t * (1 - epoch/T)) | Tests the effect of annealing bandwidth |
| Synthetic dataset | 2D Gaussian mixture with known posterior score | Validates kernel-smoothing bias-variance and mode-collapse threshold |

### Why this setup validates the claim

The central claim is that matching the full conditional posterior score, rather than only its mean, yields a better one-step student when the teacher posterior is multimodal. PMD is the sharpest baseline because it shares our Gaussian-corruption tilt and teacher evaluation but matches only the tilted posterior mean; any gain over PMD therefore isolates the effect of full-distribution matching, which is exactly our sub-claim. CED and DisBack test that the gain is not an artifact of a single comparison family: they use different objectives (NCE and trajectory backtracking), so if KPMD wins across all three the advantage is attributable to the objective class rather than to a shared implementation detail. The fixed-bandwidth ablation is the key diagnostic: it holds the score-matching loss but removes bandwidth annealing, so if our predicted bias mechanism is real, its FID should degrade relative to full KPMD specifically on multimodal-posterior noise levels. The K-sweep and h_t-annealing ablations directly test the load-bearing assumption by varying K and the bandwidth schedule, allowing us to empirically locate the failure mode where the kernel-smoothed target collapses to a mean-like statistic. The synthetic 2D Gaussian mixture experiment provides a controlled setting where the true posterior score is known in closed form, enabling direct measurement of the bias and variance of the kernel-smoothed target as a function of h_t and K, and validation of the mode-collapse threshold (mode separation vs. h_t). FID is the right metric because it is sensitive to mode collapse and mode dropping, which is precisely the failure mode mean matching predicts; a metric like precision alone would miss the diversity loss.

### Expected outcome and causal chain

**vs. PMD** — On a case where the teacher posterior at an intermediate noise level is multimodal (e.g., ambiguous coarse structure with several plausible completions), PMD's squared error between student sample averages and the teacher tilted mean pulls all student samples toward a single conditional mean, producing blurry or averaged outputs because its first-moment loss has no term penalizing within-mode variance collapse. Our method instead matches the kernel-smoothed conditional score at each student sample, which retains gradient toward each mode because the kernel-weighted target preserves local density structure, so we expect a noticeable FID gap concentrated at noise levels with high posterior multimodality and near-parity at low-noise levels where posteriors are unimodal.

**vs. CED** — On a case where the multistep teacher's score is sharp but the one-step student must cover a broad conditional distribution, CED's noise-contrastive objective optimizes an implicit KL that can underweight rare modes because the NCE classifier is dominated by high-density regions, yielding a student that matches common modes but drops tail modes. Our method instead evaluates the teacher score at shifted inputs derived from each student sample, so each sample receives a target reflecting its own local posterior neighborhood, and we expect higher sample diversity (visible as better FID despite similar or slightly worse per-sample sharpness) particularly on classes with heterogeneous intra-class appearance.

**vs. DisBack** — On a case where the degradation path from teacher to untrained student passes through intermediate distributions that are not well approximated by a single linear backtracking trajectory, DisBack's recorded path can misalign with the student's actual convergence, producing a student that converges fast but to a slightly offset distribution because its trajectory supervision is a global path rather than a per-noise-level conditional target. Our method instead imposes a conditional score-matching constraint independently at each trained noise level, so errors at one level do not propagate through a shared trajectory, and we expect DisBack to show a modest FID deficit that grows with the number of trained noise levels while KPMD stays roughly flat.

### What would falsify this idea

If KPMD's FID gain over PMD is uniform across noise levels and class-multimodality subsets rather than concentrated where teacher posteriors are multimodal, then the full-distribution-matching mechanism is not the cause and the central claim is wrong. Likewise, if the fixed-bandwidth ablation matches full KPMD, the kernel-annealing bias story is falsified and the method reduces to a mean-matching variant.

## References

1. Posterior Matching Distillation
