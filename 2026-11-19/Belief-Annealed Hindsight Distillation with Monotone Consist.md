# Belief-Annealed Hindsight Distillation with Monotone Consistency for Robot Manipulation

## Motivation

HINT-WAM's on-policy self-distillation supervises the flow-matching action expert with the full future demonstration video as a fixed, binary privileged teacher: the student never sees a degraded version of that privilege, so there is no pressure for its action distribution to become identifiable from observations that are actually available at deployment. DexTrack similarly curates demonstrations and mixes RL with imitation but never constrains the controller's sampling trajectory to be consistent as task-relevant state becomes progressively observable. Structurally, both lineages treat privileged information as an all-or-nothing oracle rather than a graded signal, leaving latent task-relevant state identifiability under partial observability untested and unestablished.

## Key Insight

If the hindsight privilege is annealed from full to absent along a schedule while a distributional consistency loss forces the degraded-privilege action distribution toward the full-privilege one, the action expert is driven to recover task-relevant state from observable history alone, because any residual gap between the two distributions is exactly the state information the student has not yet made identifiable.

## Method

**What it is.** BAHID (Belief-Annealed Hindsight Distillation) trains a flow-matching action expert inside a world action model by self-distilling on its own denoising trajectories, but replaces HINT-WAM's fixed full-video privilege with a curriculum-controlled degraded privilege and adds a monotone belief-consistency loss. Inputs: demonstration observations, future video, a privilege-degradation level schedule, and the student's own denoising samples. Outputs: an action expert whose action distribution under degraded privilege converges to the full-privilege distribution as observable history accumulates.

**Load-bearing assumption (stated explicitly).** The full-video privileged reference distribution q_ref(a | o_{1:t}, v_{t+1:T}) is treated as a sufficient statistic for task-relevant latent state, so that any 2-Wasserstein gap between it and the degraded-privilege student distribution q_theta(a | o_{1:t}, v_deg) is exactly recoverable missing identifiability that the consistency loss can drive to zero from observable history alone. This assumption fails when the future video contains information not recoverable from any observable history (e.g., stochastic object dynamics), in which case the gap contains irreducible aleatoric entropy rather than missing identifiability. To keep the loss from chasing that unattainable target, we subtract an estimated aleatoric floor from L_consist and gate w_consist on a held-out reducibility check, as specified below.

**How it works.**

```pseudocode
# Fixed hyperparameters (pre-registered, no schedule search):
#   K = 4 privilege levels; annealing: k_t = max(0, K - floor(t / T_anneal)) with T_anneal = 0.6 * total_steps
#   lambda = 1.0; w_consist = lambda * (1 - k_t/K)
#   M = 8 future-video samples per o_{1:t} for aleatoric-floor estimation
#   N = 8 action samples per distribution for sliced-Wasserstein approximation
#   teacher update: EMA with decay 0.999 (frozen copy, not live stop-grad)
#   probe: 2-layer MLP, hidden=256, GeLU, trained on held-out 10% of demos

for each training step t:
    sample demo (o_{1:T}, a*_{1:T}, v_{t+1:T})   # obs, expert actions, future video
    k = AnnealSchedule(t)          # privilege level in {0..K}, K=4 (full video)
    v_deg = Degrade(v_{t+1:T}, k)  # k=K: full; k=0: withheld
    # student samples its own denoising trajectory
    a_s = FlowSample(pi_theta, o_{1:t}, v_deg)      # on-policy action
    # full-privilege reference (EMA-anchored teacher, full video)
    a_ref = FlowSample(pi_theta_ema, o_{1:t}, v_{t+1:T})
    L_distill = || a_s - a*_t ||^2                  # dense hindsight supervision
    # aleatoric floor: sample M future rollouts per o_{1:t}, measure spread of q_ref
    {a_ref^(m)}_{m=1..M} = FlowSample(pi_theta_ema, o_{1:t}, v_{t+1:T}^{(m)})
    H_aleatoric = mean_pairwise_W2({a_ref^(m)})     # irreducible entropy estimate
    # sliced-Wasserstein approximation with N=8 samples
    L_consist_raw = SlicedW2( q_theta(a | o_{1:t}, v_deg), q_ref(a | o_{1:t}, v_{t+1:T}) )
    L_consist = max(0, L_consist_raw - H_aleatoric)  # penalize only epistemic gap
    # reducibility gate: held-out probe checks if residual gap is reducible
    gate = 1 if ProbeReducibleGap() > 0 else 0
    w_consist = lambda * (1 - k_t/K) * gate
    L = L_distill + w_consist * L_consist
    update theta
    update pi_theta_ema via EMA(theta, decay=0.999)
```

Degradation operators: temporal masking of future frames beyond a horizon, Gaussian noise injection on retained frames, and rectangular occlusion — all applied to the hindsight video only, never to o_{1:t}. The annealing schedule withholds privilege monotonically over training (k decreases from K to 0), and the consistency weight lambda grows as privilege is withdrawn. The aleatoric floor H_aleatoric is estimated by sampling M=8 future-video rollouts per o_{1:t} and measuring the mean pairwise 2-Wasserstein distance among the resulting reference action samples; this estimates the conditional action entropy attributable to future-video stochasticity. The reducibility gate is a held-out probe (2-layer MLP, hidden=256, GeLU) trained to predict task-relevant latent state from observable history and from full video; if the mutual-information gap between the two is nonzero, the probe reports the irreducible aleatoric floor, and the gate is set to 1 only when the residual gap after subtracting H_aleatoric is positive on held-out data. The teacher is an EMA-anchored copy of the student (decay 0.999), not a live stop-grad copy, to prevent teacher collapse from driving L_consist to zero without belief refinement.

**Why this design.** We chose a graded privilege schedule over HINT-WAM's fixed full-video teacher because a binary teacher never exposes the student to the observation regime it faces at deployment, accepting the cost that early training with degraded privilege is noisier and slower to converge. We chose a distributional consistency loss (2-Wasserstein between sampled action distributions) over a pointwise action-matching penalty because pointwise matching cannot express the claim that the *entire* action distribution should be invariant to privilege withdrawal, accepting the cost of extra sampling compute per step. We chose to keep the teacher as an EMA-anchored copy of the student with full video rather than a separately pretrained teacher, because a fixed external teacher would reintroduce the binary-privilege assumption, accepting the cost that the reference distribution drifts with the student (mitigated by EMA anchoring). A domain expert would not describe this as a variant of HINT-WAM because HINT-WAM's contribution is on-policy self-distillation with a static privileged signal, whereas BAHID's load-bearing object is the monotone consistency constraint between two privilege-conditioned distributions — a property HINT-WAM's objective does not contain.

**Why it measures what we claim.** The Wasserstein consistency loss `L_consist` measures *belief refinement* — the claim that task-relevant state becomes identifiable from observable history — because the assumption is that the full-video reference distribution is a sufficient statistic for task-relevant state, so any gap to the degraded distribution is exactly un-identified state; this assumption fails when the future video contains information not recoverable from any observable history (e.g., stochastic object dynamics), in which case the loss reflects irreducible aleatoric uncertainty rather than missing identifiability. To make this measurable, we subtract the estimated aleatoric floor H_aleatoric from L_consist_raw, so the reported L_consist reflects only the epistemic gap; the probe-based reducibility gate further ensures that w_consist is zero when the residual gap is not reducible on held-out data. The annealing level k measures *degree of partial observability* because the assumption is that masked/noised/occluded frames remove precisely the task-relevant content, which fails when degradation removes redundant rather than informative frames, in which case k understates the true information loss; we control for this by using a redundancy-controlled degradation operator that masks frames with highest probe-predicted task-relevant information first. The consistency weight `w_consist` measures *pressure toward identifiability* because it is tied to privilege withdrawal, which fails if the schedule withdraws privilege faster than the student can absorb it, in which case the weight reflects optimization difficulty rather than identifiability pressure; the pre-registered T_anneal = 0.6 * total_steps and fixed lambda = 1.0 remove schedule search as a confound.

## Contribution

['A training framework, BAHID, that converts a fixed hindsight privileged teacher into a monotonically refining belief teacher via an annealed privilege-degradation curriculum coupled with a Wasserstein distributional consistency regularizer on the flow-matching action expert.', 'A design principle: distributional consistency between full- and degraded-privilege action distributions is a usable training signal for establishing latent task-relevant state identifiability under partial observability, whereas pointwise action matching is not.', 'A privilege-degradation operator suite (temporal masking, noise injection, occlusion) with a monotone annealing schedule that can be layered onto on-policy self-distillation for manipulation.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Simulated manipulation suites with partial-observability variants (ManiSkill-HAB, RoTO) | Enables controlled privilege degradation and clean belief tests |
| Primary metric | Success-rate gap: degraded-privilege vs full-privilege eval | Directly measures belief-refinement claim |
| Secondary metric | Probe-based sufficiency gap (mutual information between observable history and task-relevant latent state vs full video) | Tests the load-bearing sufficiency assumption directly |
| Baseline | HINT-WAM (fixed full-video privilege) | Tests whether graded privilege beats binary privilege |
| Baseline | On-policy RL with sparse task reward | Tests whether dense hindsight distillation is necessary |
| Baseline | Pointwise action-matching distillation | Tests whether distributional consistency is load-bearing |
| Ablation-of-ours | BAHID with constant k (no annealing) | Isolates the monotone schedule's contribution |
| Ablation-of-ours | BAHID without aleatoric-floor subtraction | Tests whether the floor is necessary to avoid chasing unattainable targets |
| Ablation-of-ours | BAHID with live stop-grad teacher (no EMA) | Tests whether teacher collapse drives L_consist to zero spuriously |

### Why this setup validates the claim

The central claim is that a monotone, curriculum-controlled withdrawal of hindsight privilege plus a distributional consistency loss produces an action expert whose degraded-privilege distribution converges to the full-privilege distribution as observable history accumulates. Each baseline isolates one load-bearing choice. HINT-WAM tests whether graded privilege beats a static binary teacher; sparse-reward on-policy RL tests whether dense hindsight supervision matters at all; pointwise action-matching tests whether the 2-Wasserstein term is doing real work beyond a mean-matching penalty. The constant-k ablation removes only the schedule, so any gap between it and full BAHID is attributable to the monotone annealing rather than to the consistency loss or the degradation operators. The aleatoric-floor ablation tests whether subtracting the estimated irreducible entropy is necessary to prevent the student from collapsing toward an unattainable target; the EMA-teacher ablation tests whether the teacher collapse failure mode is real. The probe-based sufficiency gap directly tests the load-bearing assumption: if the mutual-information gap between observable history and full video is nonzero even after subtracting the aleatoric floor, then the consistency loss cannot drive the degraded distribution to the full-privilege one from observable history alone, and the belief-refinement claim is falsified. The primary metric — the success-rate gap between degraded-privilege and full-privilege evaluation on the same policy — is the right detector because the claim is precisely that this gap should shrink as observable history accumulates; a method that improves absolute success uniformly without closing the privilege gap has not demonstrated belief refinement, only better fitting. This design therefore converts an abstract identifiability claim into a measurable invariance property.

### Expected outcome and causal chain

**vs. HINT-WAM** — On a case where deployment observations are partially occluded and the missing content is task-relevant (e.g., an object behind a manipulator arm), HINT-WAM produces brittle actions because its teacher always sees full future video, so the student never learns to act under the observation regime it faces at test time. Our method instead anneals privilege down to zero and enforces distributional consistency between degraded and full-privilege action distributions, so the student is trained in the deployment regime. We expect a noticeable success-rate gap on heavily occluded subsets and parity on fully observable subsets, with the gap widening as occlusion severity increases.

**vs. On-policy RL with sparse task reward** — On a long-horizon task where success is rare (e.g., multi-stage rearrangement), sparse-reward RL produces high-variance exploration and slow credit assignment because scalar rewards give no direct action targets. Our method instead supervises the action expert densely on hindsight video at every step, so each denoising sample receives a direct target. We expect our method to reach usable success rates in far fewer environment interactions, with the largest advantage on the rarest-success tasks and convergence on the easiest ones.

**vs. Pointwise action-matching distillation** — On a stochastic-contact task where multiple action sequences are valid (e.g., grasping a deformable or rolling object), pointwise matching collapses the action distribution toward a single mode because it penalizes deviations from one sampled reference action. Our method instead matches the full 2-Wasserstein distance between action distributions, preserving multimodal validity. We expect a gap concentrated on stochastic-dynamics subsets, with near-parity on deterministic subsets, and a visibly higher action-distribution entropy for our method on those stochastic tasks.

**vs. BAHID with constant k (no annealing)** — The constant-k ablation never withdraws privilege, so the student never faces the deployment observation regime and the degraded-vs-full privilege gap remains large. Our method anneals k from K to 0, so the student is progressively forced to recover task-relevant state from observable history. We expect full BAHID to show a significantly smaller degraded-vs-full privilege gap than the constant-k ablation, with the gap difference concentrated on heavily degraded subsets.

**vs. BAHID without aleatoric-floor subtraction** — Without subtracting H_aleatoric, the consistency loss penalizes the student for not matching irreducible stochasticity in the future video, collapsing the action distribution toward a mode of an unattainable target. Our method subtracts the estimated floor, so L_consist penalizes only the epistemic gap. We expect the no-floor ablation to show lower action-distribution entropy and worse success on stochastic-dynamics subsets, while full BAHID preserves multimodal validity and achieves higher success on those subsets.

**vs. BAHID with live stop-grad teacher (no EMA)** — A live stop-grad teacher co-adapts with the student and can collapse, driving L_consist to zero without belief refinement. Our method uses an EMA-anchored teacher (decay 0.999) to prevent collapse. We expect the live-teacher ablation to show a smaller L_consist value that does not correspond to a smaller degraded-vs-full privilege gap, while full BAHID shows a larger L_consist value that does correspond to a smaller gap, confirming that the EMA anchor is necessary for the loss to measure identifiability.

### What would falsify this idea

If BAHID's gain over HINT-WAM is uniform across occlusion severities rather than concentrated on the heavily degraded subsets where the binary-privilege failure mode is predicted, the belief-refinement claim is wrong. Likewise, if the constant-k ablation matches full BAHID on the degraded-vs-full privilege gap, the monotone annealing schedule is not load-bearing and the central mechanism claim fails. Additionally, if the probe-based sufficiency gap remains nonzero after subtracting the aleatoric floor, the load-bearing sufficiency assumption is false and the consistency loss cannot drive the degraded distribution to the full-privilege one from observable history alone.

## References

1. HINT-WAM: Hindsight-Informed Training for World Action Models
