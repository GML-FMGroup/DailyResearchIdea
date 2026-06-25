# Consistency-Enforced Corrective Feedback for Hierarchical RL Agents

## Motivation

MobileForge's HiFPO assumes process rewards sum to the final success reward, but under distribution shift this property breaks, causing inaccurate hints that degrade policy learning. Prior work like SEAgent and UI-R1 rely on external reward models or rule-based rewards that do not self-diagnose such inaccuracies. The root cause is the absence of a self-supervised calibration mechanism for process rewards, leading to unchecked feedback errors.

## Key Insight

The violation of cumulative reward decomposition under distribution shift provides a self-supervised signal—the mismatch between cumulative process reward and final success—that directly indicates inaccurate feedback, enabling correction without human labels.

## Method

### Consistency-Enforced Process Reward Calibration (CEPRC)

**Input:** trajectory τ with steps t=1..T, initial process rewards r_t from HiFPO, final binary success indicator R ∈ {0,1}.
**Output:** calibrated process rewards r'_t that satisfy cumulative consistency.

**Assumption (explicit):** Under distribution shift, the variance of repeated process reward evaluations across rollouts at a given step is proportional to the error in those rewards. This assumption holds when environment noise is low relative to model uncertainty; when environment is highly stochastic (e.g., ads change appearance), variance may reflect noise rather than misalignment. To mitigate, we use a fallback: if the variance of a step exceeds a threshold relative to a baseline variance estimate from in-distribution data (computed from the initial HiFPO training), we revert to uniform weighting for that step.

**How it works:**
```pseudocode
for each trajectory τ with steps t=1..T:
  R ← final_success(τ) ∈ {0,1}
  r_t ← initial_process_rewards(τ) from HiFPO
  S ← sum(r_t for t=1..T)
  δ ← 0.1  # threshold hyperparameter; chosen to balance noise tolerance (low=0.05) and sensitivity (high=0.15)
  if |S - R| > δ:
    # Detect inconsistency
    # Estimate step importance via variance across K rollouts (K=5, chosen as trade-off between sample cost and variance estimate reliability)
    for each t: sample K rewards r_{t,k} from HiFPO (re-evaluating hint for same step under same observation)
    w_t ← variance(r_{t,1..K}) + ε  # ε=1e-6 to avoid zero
    # Fallback: if w_t > 0.2 * baseline_variance (baseline_variance computed on in-distribution validation set, using same K), then set w_t = 1 (uniform)
    if w_t > 0.2 * baseline_variance:
      w_t ← 1
    # Correct rewards to match R while preserving relative ordering
    r'_t ← r_t + (R - S) * (w_t / sum(w_t))
  else:
    r'_t ← r_t
  use r'_t in GRPO update (as in HiFPO).
```

**(C) Why this design:** We chose an additive correction over re-sampling new hints because it is computationally efficient and does not require additional generation passes, critical for online training. We used a threshold δ to avoid over-correcting small mismatches caused by stochasticity; setting δ=0.1 balances noise tolerance and sensitivity (trade-off: too low δ causes jitter, too high ignores genuine errors). We used variance-based importance weights to allocate corrections to steps where feedback is most uncertain, under the rationale that high-variance steps are likely misaligned; an alternative uniform weighting would treat all steps equally but dilute correction effectiveness (trade-off: variance weights may be noisy with few rollouts; we accept increased sample cost for better targeting). We added a fallback to uniform weighting when variance exceeds 0.2×baseline_variance to handle highly stochastic environments, ensuring robustness. We avoided learning a correction model because it would require labeled data under distribution shift, which is unavailable; our heuristic is domain-agnostic and generalizes without retraining (trade-off: less flexible but robust to unseen shifts).

**(D) Why it measures what we claim:** The cumulative sum S measures predicted success probability under the original feedback; its deviation from R measures feedback inaccuracy under the assumption that HiFPO's process rewards are calibrated to the environment; this assumption fails under distribution shift (e.g., new app layouts), in which case deviation reflects misaligned feedback. The variance w_t measures step-level feedback uncertainty because we sample multiple evaluations of the same hint at the same step; this assumes that consistent feedback yields low variance and inconsistent feedback yields high variance; this assumption fails when the environment itself is stochastic (e.g., ads change appearance), in which case w_t reflects environment noise rather than feedback inaccuracy. The correction term (R - S) * (w_t / sum(w_t)) measures the proportion of total error attributable to each step, under the assumption that uncertainty is a proxy for error; this assumption fails when high-variance steps are not the source of error (e.g., low-variance but systematically wrong feedback), in which case correction may misallocate. The fallback condition (w_t > 0.2*baseline_variance) mitigates the failure mode by reverting to uniform weighting when environment noise dominates, thus preserving robustness.

## Contribution

(1) A self-supervised consistency enforcement mechanism (CEPRC) that detects and corrects inaccurate process rewards in hierarchical RL without human labels. (2) The design principle that cumulative reward decomposition violation under distribution shift can be used as a diagnostic signal for feedback quality. (3) An importance-weight-based correction scheme that preserves relative ordering of feedback steps, enabling targeted adjustment.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | AndroidWorld + New App Layouts + Synthetic Controlled Shift | Covers in-distribution, out-of-distribution, and controlled ground-truth error scenarios |
| Primary metric | Pass@3 | Standard metric for agents |
| Baseline 1 | HiFPO | Base RL without calibration |
| Baseline 2 | UI-R1 | Alternative RL approach |
| Baseline 3 | No correction | Raw process rewards |
| Ablation | CEPRC-uniform | Uniform weight correction (no variance weighting) |
| Additional | Synthetic environment with known true rewards | Validates the variance-error correlation by comparing correction accuracy |

### Why this setup validates the claim

The combination of AndroidWorld (in-domain) and New App Layouts (out-of-domain) directly tests the central claim that CEPRC improves robustness under distribution shift. HiFPO serves as the direct predecessor, isolating the effect of reward calibration. UI-R1 provides a stronger baseline based on a different RL method, showing whether calibration benefits general RL for mobile agents. The no-correction baseline isolates the effect of the correction step itself. Pass@3 is appropriate because it reflects success rate over three attempts, measuring practical usefulness. The ablation CEPRC-uniform tests the necessity of variance-based weighting. To further validate the core assumption (variance correlates with misalignment), we introduce a synthetic environment where true per-step error is known: we take a base task and artificially corrupt a subset of process rewards by adding Gaussian noise (σ=0.2), with controlled distribution shift (e.g., altering observation features). In this environment, we can compute the ground-truth correction needed and compare CEPRC's allocation to the ideal, thereby verifying that variance weighting indeed targets misaligned steps. If CEPRC outperforms HiFPO and no-correction on the new layouts but not on the original dataset, that supports the claim. If it also outperforms UI-R1, it shows generality.

### Expected outcome and causal chain

**vs. HiFPO** — On a case where the agent encounters an unfamiliar app layout (distractor), HiFPO's process rewards are miscalibrated due to distribution shift, leading to overconfidence in suboptimal steps and ultimately a failed task. Our method detects the cumulative mismatch between process reward sum and final success, and redistributes correction to high-variance steps (which are precisely those misaligned by the shift). Thus, on such out-of-distribution instances, we expect a noticeable improvement in Pass@3 (e.g., 10-20% absolute gain) while performance on in-domain tasks remains similar.

**vs. UI-R1** — On a complex multi-step task requiring precise action selection, UI-R1, which uses standard RL with binary rewards, may struggle with sparse rewards and delayed credit assignment, leading to random exploration. Our method's calibrated process rewards provide dense, step-level guidance even under shifted conditions, enabling more efficient credit assignment. Therefore, we expect CEPRC to achieve higher Pass@3 on long-horizon tasks, especially those with significant distribution shift, but possibly parity on simple tasks.

**vs. No correction** — On a trajectory where the initial process rewards are slightly off (small inconsistency), the no-correction baseline continues with those rewards, potentially reinforcing wrong behavior. Our method applies a small correction to meet the terminal success, preventing error accumulation. Thus, even on in-domain tasks, we expect a modest but consistent improvement in Pass@3 (e.g., 2-5% absolute gain), demonstrating the benefit of consistency enforcement even when shift is not severe.

**vs. CEPRC-uniform** — On a step where the process reward is systematically wrong but the variance is low (e.g., a consistently misjudged hint), uniform weighting would spread the correction across all steps, diluting the fix at the problematic step. Our variance weighting concentrates correction where uncertainty is high, which may not always align with error; however, in practice, we expect variance weighting to outperform uniform because high-variance steps are more likely to be misaligned. Furthermore, our fallback (reverting to uniform when variance exceeds environment noise threshold) prevents over-correction in highly stochastic settings. Therefore, we expect CEPRC to outperform CEPRC-uniform by a small margin (2-5%) on overall Pass@3, with the gap more pronounced on steps with high feedback uncertainty.

### What would falsify this idea

If CEPRC shows uniform improvement across all subsets (including simple in-domain tasks) rather than concentrated gains on challenging out-of-distribution instances, or if CEPRC-uniform matches CEPRC performance, then the central claim that variance-based calibration addresses distribution shift would be falsified. Additionally, if in the synthetic environment the correction allocation is uncorrelated with true error (i.e., variance weighting does not target misaligned steps), the core assumption is invalidated.

## References

1. MobileForge: Annotation-Free Adaptation for Mobile GUI Agents with Hierarchical Feedback-Guided Policy Optimization
2. UI-R1: Enhancing Efficient Action Prediction of GUI Agents by Reinforcement Learning
3. SEAgent: Self-Evolving Computer Use Agent with Autonomous Learning from Experience
4. Navigating the Digital World as Humans Do: Universal Visual Grounding for GUI Agents
5. Aguvis: Unified Pure Vision Agents for Autonomous GUI Interaction
6. WebRL: Training LLM Web Agents via Self-Evolving Online Curriculum Reinforcement Learning
7. A Zero-Shot Language Agent for Computer Control with Structured Reflection
8. Agent Lumos: Unified and Modular Training for Open-Source Language Agents
