# Noise-Modeled Reward Denoising for Weak-Teacher GRPO

## Motivation

Weak-teacher RL rewards are treated as ground truth by GRPO, yet the teacher in prior work — e.g., the JEV per-action helpfulness scorer of 'Actor, Judge, or Teacher?' that lifts Qwen2.5-1.5B from 72.8% to 82.9% on ALFWorld — is itself a decision model whose per-action judgments are noisy and uncalibrated, and the paper explicitly notes the teacher's judgments 'may be imperfect' without any correction. Structurally, this fails because GRPO's group-relative advantage normalizes each reward against its own trajectory group, so a teacher that is systematically biased toward easy subgoals injects a correlated error that normalization cannot cancel, and the strong policy inherits a ceiling set by the teacher's noise rather than its own capability.

## Key Insight

If teacher noise is modeled as a per-state latent reliability variable multiplying a shared action-quality signal, then the true quality ordering is identifiable from repeated teacher queries up to a monotone transform, so denoising the reward before GRPO recovers the strong policy's ranking capability that raw noisy rewards destroy.

## Method

**What it is.** NRD (Noise-Robust Denoising) takes as input a trajectory's per-action teacher scores `s_t` and the state features `x_t` (the teacher's per-option probability vector at step t), and outputs a denoised per-action quality `q_t` plus a scalar reliability weight `w_t ∈ (0,1]`. These replace the raw teacher scores as the reward fed to GRPO.

**Load-bearing assumption (stated explicitly).** The method assumes that repeated queries to the JEV teacher at the same state differ only by independent, zero-mean judgment noise, so that empirical variance across K=5 repeats is an unbiased estimate of noise magnitude and the shared latent factor recovered by the inverse-variance-weighted DAE is the true action-quality ordering. This assumption fails when the teacher is deterministically biased (repeats agree but are all wrong), in which case `w_t` reflects consistency rather than correctness and the DAE faithfully reconstructs the bias.

**How it works.**

```pseudocode
# Phase 1: Noise model fitting (offline, on a held-out probe set)
for each state x in probe_set (K=5 repeated teacher queries each):
    collect scores s_1..s_K
    sigma2(x) = empirical variance across repeats        # per-state noise variance
    fit heteroscedastic model: sigma2(x) = g_phi(x)      # 2-layer MLP, 64 hidden, GeLU
    fit reliability: rho(x) = 1 / (1 + g_phi(x)/tau),  tau = 0.1

# Phase 2: Denoising autoencoder over action-quality signal
train DAE encoder E_psi / decoder D_psi on (x_t, s_t) pairs:
    z_t = E_psi(x_t, s_t)                  # latent quality, dim 32
    s_hat_t = D_psi(x_t, z_t)
    loss = ||s_hat_t - s_t||^2 / (2*sigma2(x_t)) + lambda*||z_t||_1,  lambda = 1e-3
    # inverse-variance weighting: noisy states contribute less to latent fit

# Phase 3: Reward reconstruction for GRPO
for each action t in trajectory:
    z_t = E_psi(x_t, s_t)
    q_t = D_psi(x_t, z_t)                  # denoised quality
    w_t = rho(x_t)                         # reliability in (0,1]
    r_t = w_t * q_t + (1 - w_t) * q_bar    # q_bar = trajectory mean quality
    feed r_t to GRPO group-relative advantage
```

**Why this design.** We chose a heteroscedastic Gaussian noise model `sigma2(x)=g_phi(x)` over a single global noise scalar because teacher unreliability is state-dependent (the JEV teacher is distracted by easy subgoals, so noise is higher near subgoal boundaries), accepting the cost that `g_phi` needs K repeated queries per probe state and can overfit with few probes. We chose inverse-variance weighting inside the DAE loss over uniform reconstruction because states with high `sigma2(x)` carry less signal about true quality, accepting that a mis-estimated `sigma2` will mis-weight the latent fit. We chose to shrink low-reliability rewards toward the trajectory mean `q_bar` rather than zero them out because GRPO's group-relative baseline already centers rewards, so zeroing would discard the residual ordering signal, accepting that `w_t` near zero makes `r_t` nearly uninformative for that step. This is not a variant of VinePPO or a Process Reward Model: those estimate step values from Monte-Carlo rollouts or a separately trained reward model, whereas here the reward model is the teacher and the contribution is an identifiability-driven noise correction whose reliability weights are estimated from repeated teacher queries, not from value estimation.

**Why it measures what we claim.** The reliability weight `w_t = rho(x_t)` measures teacher trustworthiness at state `x_t` because the assumption that repeated teacher queries differ only by independent judgment noise makes empirical variance `sigma2(x_t)` an unbiased estimate of noise magnitude; this assumption fails when the teacher is deterministically biased (repeats agree but are all wrong), in which case `w_t` reflects consistency rather than correctness. The latent `z_t` measures true action quality because the inverse-variance-weighted DAE loss assumes the shared latent is the maximum-likelihood common factor across noisy observations; this fails when noise is correlated across steps within a trajectory, in which case `z_t` absorbs trajectory-level bias. The denoised `q_t` measures recovered quality because the decoder maps the latent back through the same noise model; this fails when the DAE over-smooths and collapses distinct quality levels, in which case `q_t` reflects a shrunken ranking.

## Contribution

(1) A noise-modeled reward aggregation mechanism that treats teacher judgment noise as a per-state latent reliability variable and produces denoised, reliability-weighted per-action rewards for GRPO. (2) A design principle — inverse-variance-weighted denoising autoencoding over teacher scores — that recovers action-quality ordering when the teacher is noisy, with reliability estimated from repeated queries rather than from log-probs. (3) A calibration procedure that estimates per-state teacher reliability and shrinks low-reliability rewards toward the trajectory mean, isolating the effect of teacher noise from policy capability.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | ALFWorld + WebShop | Long-horizon tasks where JEV teacher noise is state-dependent |
| Primary metric | Success rate | Directly reflects reward quality fed to GRPO |
| Baseline | GRPO with raw JEV scores | Isolates whether noise correction itself helps |
| Baseline | GRPO without teacher (Qwen2.5-1.5B) | Lower bound; checks teacher signal is being used |
| Baseline | Global-scalar noise correction | Tests whether state-dependent variance matters |
| Ablation-of-ours | NRD with uniform DAE weighting (no inverse-variance) | Tests whether heteroscedastic weighting drives gains |
| Ablation-of-ours | NRD with K=2,3,5 repeated queries | Tests sensitivity to number of teacher queries |
| Ablation-of-ours | NRD with probe-set sizes 256, 512, 1024 | Tests sensitivity to probe-set size |
| Diagnostic | Synthetic bias injection (known deterministic bias) | Tests where w_t misleads under correlated error |

### Why this setup validates the claim

The central claim is that teacher-score noise is state-dependent and that correcting it with a per-state reliability weight plus inverse-variance-weighted denoising yields better GRPO rewards than raw teacher scores. ALFWorld and WebShop are the right testbeds because the JEV teacher is documented to be distracted near subgoal boundaries, so noise concentrates in identifiable state regions rather than being uniform. The raw-JEV baseline tests the core sub-claim that denoising beats using noisy scores directly; the no-teacher baseline guards against the trivial explanation that any teacher signal helps; the global-scalar baseline tests the specific sub-claim that per-state heteroscedasticity is necessary rather than a single noise scale. The uniform-weighting ablation isolates the inverse-variance mechanism inside the DAE, so if gains persist without it, the credit belongs to the reliability shrinkage rather than the weighted latent fit. The K and probe-set ablations test feasibility under normal compute by showing the method works with fewer repeated queries and smaller probe sets, and report total teacher-query overhead versus the raw-JEV baseline. The synthetic bias-injection diagnostic tests the load-bearing assumption: by injecting a known deterministic bias into teacher scores at a subset of states, we can measure whether `w_t` incorrectly assigns high reliability to biased states (low variance but high error), directly probing the failure mode where consistency does not equal correctness. Success rate is the right metric because GRPO's advantage is group-relative, so a reward correction that preserves ordering but reduces variance should show up as higher task completion, not as a reward-magnitude change. Together these conditions form a falsifiable test: the method must beat raw JEV specifically where predicted noise is high.

### Expected outcome and causal chain

**vs. GRPO with raw JEV scores** — On a case where the trajectory passes near a subgoal boundary, the raw-JEV baseline receives a noisy score that the distracted teacher inflates or deflates, so GRPO's group-relative advantage misranks that step against cleaner siblings and reinforces the wrong action. Our method instead estimates high `sigma2(x_t)` at that state, assigns low `w_t`, and shrinks `r_t` toward the trajectory mean, so the misranked step contributes little to the advantage. We expect a noticeable success-rate gap concentrated on episodes with many subgoal-boundary states, with near-parity on short, boundary-sparse episodes.

**vs. GRPO without teacher** — On a case where the policy is stuck in a plausible but wrong action loop, the no-teacher baseline has no external signal to break the loop and keeps sampling the same low-value action because its own reward is uninformative. Our method instead supplies a denoised teacher quality `q_t` that, even after shrinkage, carries residual ordering signal distinguishing the stuck action from alternatives, so GRPO's advantage nudges the policy off the loop. We expect the no-teacher baseline to plateau early and our method to show a clear late-training success-rate advantage, largest on the harder WebShop subset.

**vs. Global-scalar noise correction** — On a case where noise is high at boundaries but low on routine steps, the global-scalar baseline applies one average noise level everywhere, so it over-smooths clean steps and under-corrects noisy ones, flattening the quality ranking that GRPO needs. Our method instead uses per-state `sigma2(x_t)` from repeated queries, so clean steps keep sharp `q_t` and noisy steps get shrunk, preserving within-trajectory ordering. We expect our method to outperform global-scalar correction specifically on episodes whose state-noise variance is high, with the two methods converging on low-variance episodes.

**vs. NRD with uniform DAE weighting** — On a case where high-noise states dominate the trajectory, the uniform-weighting ablation fits the latent to noisy observations equally, so the recovered `q_t` is pulled toward the noise and the advantage ranking degrades. Our method instead down-weights high-`sigma2` states in the DAE loss, so the latent is driven by cleaner observations and `q_t` better preserves true ordering. We expect full NRD to outperform the uniform-weighting ablation on episodes with high state-noise variance, with parity on low-variance episodes.

**vs. NRD with K=2,3,5 repeated queries** — On a case where K is small, the empirical variance estimate is noisy, so `w_t` is less accurate and the DAE inverse-variance weighting is less reliable. We expect success rate to improve monotonically with K up to 5, with diminishing returns beyond K=3, and we report total teacher-query overhead (K times the raw-JEV query count) to quantify the cost.

**vs. NRD with probe-set sizes 256, 512, 1024** — On a case where the probe set is small, `g_phi` may overfit, leading to mis-estimated `sigma2(x)` and suboptimal `w_t`. We expect performance to stabilize at 512 probe states, with 256 showing a small drop and 1024 offering marginal gains, demonstrating feasibility with modest probe sets.

**vs. Synthetic bias injection** — On a case where a known deterministic bias is added to teacher scores at a subset of states, repeated queries will agree (low variance) but be systematically wrong, so `w_t` will incorrectly assign high reliability and the DAE will reconstruct the bias. We expect success rate to degrade on biased states, confirming the failure mode where consistency does not equal correctness, and we will report the magnitude of this degradation to bound the method's applicability.

### What would falsify this idea

If our gain over raw-JEV GRPO is uniform across high-noise and low-noise state subsets rather than concentrated where the estimated `sigma2(x_t)` is high, then the reliability-weighting mechanism is not what produces the improvement and the central claim is wrong. Likewise, if the uniform-weighting ablation matches full NRD, the inverse-variance denoising sub-claim is falsified. Additionally, if the synthetic bias-injection diagnostic shows that `w_t` assigns high reliability to biased states and the method fails to correct them, the load-bearing assumption is violated and the method's scope is limited to independent noise.

## References

1. Actor, Judge, or Teacher? Understanding Decision Models in Long-Horizon Agents
