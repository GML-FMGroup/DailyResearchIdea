# Distribution-Shift-Aware Reward Calibration for Groupwise Design Reward in Prompt Enhancement

## Motivation

Qwen-Image-PE's Groupwise Design Reward is trained on a fixed reward model whose validity under policy exploration is unestablished; the RL stage does not guarantee calibrated design rewards when the policy generates prompts outside the SFT caption distribution. This structural flaw recurs across prompt-enhancement branches (BeautifulPrompt, HPSv3): a stale evaluator silently rewards distributional artifacts rather than genuine design improvement, so gains measured against it are not guaranteed to be real. The root cause is that reward calibration is treated as a one-time offline property rather than a quantity that must be monitored and enforced as the policy's own output distribution evolves.

## Key Insight

Reward calibration under policy shift is a measurable, enforceable invariant: by maintaining a held-out prompt set drawn from the policy's evolving distribution and tracking rank correlation between the groupwise design reward and fresh human/meta-reward judgments, calibration degradation becomes an observable signal that can be penalized in the RL objective before it manifests as reward hacking.

## Method

**What it is** — We introduce **DRACO** (Distribution-shift Reward cAlibration for Groupwise Design Rewards), a wrapper around Qwen-Image-PE's RL fine-tuning that (i) maintains a rolling held-out probe set sampled from the current policy, (ii) periodically re-estimates the calibration of the Groupwise Design Reward against a held-out meta-reward, (iii) injects a calibration penalty / ensemble reweighting into the RL reward whenever calibration degrades, and (iv) verifies the meta-reward anchor itself against a small, periodically refreshed human-labeled subset before allowing the penalty to fire. Inputs: policy π_θ, base design reward R_design, meta-reward R_meta, probe buffer B, human-labeled anchor set H. Outputs: calibrated RL reward R_cal and updated policy.

**How it works** —

```pseudocode
# Initialization
B = sample_probe_set(π_SFT, n=256)            # held-out prompts from SFT policy
H = fixed_human_labeled_set(n=128)            # pairwise human preferences on prompts/images from π_SFT
cal_history = []
meta_cal_history = []

for RL_iter in 1..N:
    # 1. Standard groupwise rollout
    for request q in batch:
        prompts p_i = π_θ(q)  for i=1..G       # G=8 group size
        images  I_i = T2I(p_i)
        r_design_i = R_design(q, p_i, I_i)    # groupwise design reward

    # 2. Refresh probe set every K=20 iters from current policy
    if RL_iter % K == 0:
        B_new = sample_probe_set(π_θ, n=256)  # prompts sampled from evolving policy
        B = B ∪ B_new (keep most recent 512)   # rolling buffer

    # 3. Re-estimate calibration on probe set every M=5 iters
    if RL_iter % M == 0:
        for q in B:
            p = π_θ(q); I = T2I(p)
            s_design = R_design(q, p, I)
            s_meta   = R_meta(q, p, I)         # held-out meta-reward (human proxy)
        ρ = spearman_rank_corr({s_design}, {s_meta})
        cal_history.append(ρ)
        ρ_ref = median(cal_history[:-1]) if len(cal_history)>1 else ρ

        # 3b. Verify meta-reward anchor on human-labeled set H (every M iters)
        for (q, p_a, p_b, human_pref) in H:
            I_a = T2I(p_a); I_b = T2I(p_b)
            s_meta_a = R_meta(q, p_a, I_a)
            s_meta_b = R_meta(q, p_b, I_b)
            # record whether meta-reward agrees with human preference
        ρ_meta_human = spearman_rank_corr({s_meta_a - s_meta_b}, {human_pref})
        meta_cal_history.append(ρ_meta_human)
        ρ_meta_ref = median(meta_cal_history[:-1]) if len(meta_cal_history)>1 else ρ_meta_human

        # 4. Calibration penalty / reweighting (gated by meta-reward validity)
        meta_valid = (ρ_meta_human >= ρ_meta_ref - δ_meta)  # δ_meta=0.05
        if (ρ < ρ_ref - δ) and meta_valid:      # δ=0.05 degradation threshold
            λ = clip( (ρ_ref - ρ) / δ, 0, 1 )   # penalty strength
        else:
            λ = 0
        # 5. Ensemble reweighting: blend design and meta reward
        α = 1 - λ
        R_cal(q,p,I) = α * zscore_within_group(r_design_i) + (1-α) * zscore_within_group(r_meta_i)

    # 6. Policy update with calibrated reward
    loss = - E[ R_cal * log π_θ(p|q) ] + β * KL(π_θ || π_SFT)
    θ ← θ - ∇loss
```

**Why this design** — We chose a *rolling held-out probe set sampled from the current policy* over a fixed validation set because the failure mode is precisely distribution shift: a fixed probe set would itself become stale and fail to detect the shift we care about, accepting the cost that probe sampling adds T2I generation overhead every K iterations. We chose *Spearman rank correlation against a held-out meta-reward* over direct reward-model retraining because rank correlation is invariant to monotone rescaling of the design reward and cheap to estimate, accepting the cost that it only detects ordinal miscalibration, not affine bias. We chose *ensemble reweighting with a degradation-triggered λ* over a hard reward switch because a hard switch would destabilize RL and discard the design reward's dense signal, accepting the cost that λ must be tuned (δ=0.05, K=20, M=5) and that partial blending may slow recovery. We chose *meta-reward as the calibration anchor* over human labels in the loop because human labels are prohibitively expensive at RL frequency, accepting the cost that the meta-reward must itself be a trustworthy proxy — a limitation we make explicit and mitigate by gating λ on a periodic human-label verification of the meta-reward (step 3b). A domain expert would not describe DRACO as a variant of standard reward-model retraining (e.g., iterative reward modeling) because DRACO does not retrain the reward; it *monitors the rank-calibration of the existing reward on the policy's own evolving distribution* and modulates the RL objective accordingly, which is a different mechanism than fitting a new reward to new data.

**Why this measures what we claim** — The computational quantity `ρ = Spearman(design, meta)` on the rolling probe set measures *reward calibration under policy shift* because the named assumption is that the meta-reward is a monotone-in-quality proxy for human design judgment on the probe distribution; this assumption fails when the meta-reward itself is systematically biased toward a particular visual style, in which case ρ reflects agreement between two biased evaluators rather than true calibration. The quantity `λ = clip((ρ_ref - ρ)/δ, 0, 1)` measures *severity of calibration degradation* because the assumption is that ρ_ref (historical median) is a valid reference for calibrated behavior; this fails when the policy undergoes a legitimate distribution expansion that temporarily lowers ρ but improves true quality, in which case λ penalizes genuine exploration. The ensemble weight `α = 1 - λ` measures *trust in the design reward* because the assumption is that the meta-reward is better calibrated precisely when the design reward degrades; this fails when both rewards degrade together (e.g., adversarial T2I outputs), in which case the ensemble provides no protection and the KL term is the only safeguard. The gating quantity `ρ_meta_human = Spearman(meta, human)` on the fixed human-labeled set H measures *meta-reward validity as a calibration anchor* because the assumption is that human pairwise preferences on H are a trustworthy ground truth for design quality; this fails when H is too small or unrepresentative of the policy's evolving distribution, in which case the gate may falsely certify or falsely reject the meta-reward.

## Contribution

(1) DRACO, a distribution-shift-aware reward calibration wrapper for groupwise design rewards that maintains a rolling probe set sampled from the evolving policy and modulates the RL reward via rank-correlation-triggered ensemble reweighting. (2) The empirical finding that reward calibration under policy shift is detectable via Spearman rank correlation on policy-sampled probes, and that degradation-triggered reweighting reduces reward hacking relative to a fixed design reward. (3) A reusable probe-buffer and calibration-monitoring protocol for RL fine-tuning of prompt enhancers, applicable to BeautifulPrompt and HPSv3 branches.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Qwen-Image-Bench | Standard T2I design benchmark with human preference labels. |
| Primary metric | Spearman ρ(design, meta) on held-out probe set | Directly measures the calibration under policy shift. |
| Secondary metric | Spearman ρ(meta, human) on fixed human-labeled set H (n=128) | Quantifies trustworthiness of the calibration anchor. |
| Baseline | Qwen-Image-PE (uncalibrated RL) | Tests whether calibration wrapper adds value over base RL. |
| Baseline | Reward-model retraining (iterative RM) | Tests whether monitoring beats refitting the reward. |
| Baseline | Fixed-validation-set calibration | Tests whether rolling probe set is necessary. |
| Baseline | Fixed-weight ensemble (α=0.5, no drift detection) | Tests whether degradation-triggered λ does real work. |
| Baseline | KL-adaptive (β tuned per iteration, no calibration) | Tests whether calibration signal beats generic KL tuning. |
| Ablation-of-ours | DRACO with λ=0 (no reweighting) | Isolates the effect of the calibration penalty itself. |
| Ablation-of-ours | DRACO without meta-reward gating (always trust R_meta) | Isolates the effect of the human-label verification gate. |

### Why this setup validates the claim

This setup forms a falsifiable test of the central claim—that DRACO’s rolling-probe calibration detects and corrects reward miscalibration under policy shift—by pitting it against five distinct mechanism gaps. The uncalibrated RL baseline tests the sub-claim that any calibration is needed: if DRACO does not beat it on the primary metric, the wrapper is inert. Iterative reward-model retraining tests the sub-claim that monitoring the existing reward is preferable to refitting: if retraining matches or exceeds DRACO, the core design choice is wrong. Fixed-validation-set calibration tests the sub-claim that the probe set must roll with the policy: if a stale probe set performs equally well, the distribution-shift motivation collapses. The fixed-weight ensemble baseline tests whether the degradation-triggered λ is doing real work beyond a static blend: if a constant α=0.5 matches DRACO, the dynamic penalty is unnecessary. The KL-adaptive baseline tests whether generic KL tuning can substitute for calibration monitoring: if it matches DRACO, the calibration signal is redundant. The ablation (λ=0) isolates the calibration penalty from the probe-set machinery, so any gain can be attributed to the reweighting step rather than to auxiliary overhead. The ablation without meta-reward gating isolates the effect of the human-label verification gate, so any gain from gating can be attributed to anchor validation rather than to the probe-set refresh. Spearman ρ on the held-out probe set is the right primary metric because it directly operationalizes the quantity DRACO claims to protect—rank agreement between design and meta rewards on the policy’s own evolving distribution—and it is invariant to monotone rescaling, matching the mechanism’s ordinal assumption. The secondary metric ρ(meta, human) on H directly measures the load-bearing assumption that the meta-reward is a trustworthy anchor; if ρ(meta, human) is low or degrades, the primary metric’s interpretation is undermined. A gain concentrated where the predicted failure mode occurs (late RL iterations, high-shift prompts) would confirm the causal story; a uniform gain would not.

### Expected outcome and causal chain

**vs. Qwen-Image-PE (uncalibrated RL)** — On a case where late-stage RL pushes the policy toward a degenerate prompt style (e.g., overusing a few high-reward templates), the baseline’s design reward saturates and stops distinguishing quality, so the policy drifts toward reward-hacking outputs because it has no signal that calibration has degraded. Our method instead detects the drop in ρ on the rolling probe set and blends in the meta-reward, so we expect a visible ρ gap emerging after ~20–40 RL iterations, with DRACO maintaining ρ near its historical median while the baseline’s ρ declines steadily, and a corresponding human-preference gap concentrated on late-iteration outputs.

**vs. Reward-model retraining (iterative RM)** — On a case where the design reward is miscalibrated only in ordinal rank (monotone rescaling of scores), the retraining baseline refits a new reward head to fresh labels and inherits the same rank distortion because it optimizes fit to the same biased signal, so it produces a reward that is numerically different but rank-equivalent, leaving RL behavior unchanged. Our method instead measures rank correlation directly and reweights toward the meta-reward, so we expect DRACO to show a larger ρ recovery than retraining on rank-distorted subsets, with retraining showing gains only on affine-bias subsets where refitting can help—a diagnostic split that confirms the mechanism distinction.

**vs. Fixed-validation-set calibration** — On a case where the policy shifts toward a new prompt distribution (e.g., a new domain of user requests) that is absent from the initial validation set, the fixed-set baseline’s ρ stays high because it measures calibration on stale prompts, so it never triggers λ and the policy drifts uncalibrated on the new distribution, producing a growing gap between probe-set ρ and true held-out ρ. Our method instead refreshes the probe set every K iterations, so we expect DRACO’s ρ on freshly sampled probes to track the true calibration while the fixed-set baseline’s ρ becomes increasingly optimistic—observable as a divergence between the two ρ estimates that widens with iteration count.

**vs. Fixed-weight ensemble (α=0.5, no drift detection)** — On a case where calibration degrades only in late iterations, the fixed-weight ensemble always blends in the meta-reward at α=0.5, so it cannot adapt to the timing of degradation and may over-regularize early iterations (slowing genuine exploration) while under-regularizing late iterations (when λ should be high). Our method instead sets λ=0 when ρ is healthy and ramps λ up only after degradation, so we expect DRACO to show better early-iteration quality (higher human preference on early outputs) and comparable or better late-iteration ρ recovery, with the fixed-weight ensemble showing a flatter ρ trajectory that fails to concentrate gains where the failure mode occurs.

**vs. KL-adaptive (β tuned per iteration, no calibration)** — On a case where the design reward is miscalibrated but the KL term is well-tuned, the KL-adaptive baseline controls policy drift but has no signal that the reward itself is invalid, so it can stabilize the policy around a reward-hacking optimum. Our method instead detects calibration degradation and reweights the reward, so we expect DRACO to show a larger ρ recovery and higher human preference on late-iteration outputs than KL-adaptive, with KL-adaptive showing stable but low ρ because it never corrects the reward.

**vs. DRACO with λ=0 (ablation)** — On a case where calibration genuinely degrades (ρ falls below ρ_ref−δ), the ablation computes the same ρ but never applies the penalty, so the RL objective remains the uncalibrated design reward and the policy follows the same drift path as the uncalibrated baseline. Our full method instead activates λ>0 and blends in the meta-reward, so we expect the ablation to match the uncalibrated baseline’s ρ trajectory and the full method to show a recovery gap that appears only after the first degradation event—confirming that the penalty, not the probe-set monitoring alone, drives the improvement.

**vs. DRACO without meta-reward gating (always trust R_meta)** — On a case where the meta-reward itself degrades (ρ(meta, human) falls below ρ_meta_ref−δ_meta), the ungated variant still fires λ based on ρ(design, meta), so it may blend toward a degraded anchor and fail to recover true quality. Our method instead gates λ on meta_valid, so we expect the ungated variant to show a larger gap between ρ(design, meta) and ρ(design, human) on late-iteration outputs, while the gated DRACO maintains higher ρ(design, human) and human preference—confirming that the human-label verification gate protects against anchor degradation.

### What would falsify this idea

If DRACO’s gain over the uncalibrated baseline is uniform across all subsets and iterations rather than concentrated on late-iteration, high-shift prompts where ρ is predicted to degrade, then the calibration mechanism is not what produces the improvement and the central claim is wrong. Similarly, if the fixed-validation-set baseline matches DRACO’s ρ trajectory, the distribution-shift motivation—the core reason for the rolling probe set—is falsified. If the fixed-weight ensemble or KL-adaptive baseline matches DRACO’s ρ recovery and human preference, the degradation-triggered λ and calibration signal are not doing real work. If ρ(meta, human) on H is low or degrades alongside ρ(design, meta), the meta-reward anchor is not trustworthy and the primary metric’s interpretation is undermined.

## References

1. Qwen-Image-PE: Learning Visual Design for Text-to-Image Prompt Enhancement
