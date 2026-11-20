# Pose-Fact Visual Consistency Verification for Egocentric Caption Reward Shaping

## Motivation

Egocentric captioning pipelines such as EgoCaptioner (E2, E3) and egocentric QA generators such as MMEgo (E1) treat auxiliary signals — pose-derived motion facts and auto-generated QA — as ground truth without checking them against video pixels, so sensor noise, calibration drift, and annotation-generation errors propagate directly into rewards and supervision. This is a shared structural gap across both branches: the reliability of an auxiliary signal is assumed rather than measured, and no existing component in either pipeline can detect the case where a motion fact contradicts what the camera actually shows in the corresponding interval.

## Key Insight

A pose-derived motion fact is only usable as a reward or supervision target to the extent that it is visually identifiable in the interval it describes, and this identifiability can be measured by whether the fact's predicted visual signature is discriminable from perturbed counterfactual signatures in the same interval — a per-fact, per-interval reliability weight rather than a global trust assumption.

## Method

**What it is.** We introduce PAVV (Pose-Agreement Visual Verifier), a per-fact, per-interval consistency scorer that sits between the pose-derived motion facts and EgoCaptioner's GRPO reward. Input: a motion fact f (e.g., "wearer turns left ~90° between t=12s and t=15s") and the video interval [t_s, t_e]. Output: a scalar reliability weight w(f, [t_s,t_e]) ∈ [0,1] and a binary exclusion flag, which are multiplied into the factual-accuracy reward term before GRPO updates.

**Load-bearing assumption (stated explicitly).** PAVV assumes that the cosine margin between a true pose-derived motion fact and its four structured perturbations (invert, scale, shift, swap) is monotone in that fact's true visual discriminability in the video interval, so that a low margin reliably indicates sensor noise rather than verifier blindness. This assumption is breakable: the margin is computed against a fixed, enumerated perturbation set, so it measures discriminability only along those four directions; a calibration offset that preserves direction and magnitude passes undetected, and a 90° vs 180° turn can produce similar optical flow in a narrow FOV. To partially close this gap, we add a fifth perturbation (constant-velocity drift / calibration-offset simulation) and a per-domain null-distribution calibration step, described below.

**How it works.**

```pseudocode
# Stage 0: per-domain null calibration (new, closes part of the load-bearing gap)
# Held-out clean-sensor intervals: 512 facts from Ego4D videos with pose-confidence > 0.9
# and no injected noise. Compute raw margins m_clean for these facts using Stages 1-3 below.
# Fit a per-domain (indoor/outdoor, lighting bucket) empirical CDF F_domain(m).
# w is then defined as the percentile rank: w = F_domain(m_raw), clipped to [0,1].
# This replaces the absolute sigmoid; the sigmoid is retained only as a fallback
# when fewer than 64 clean facts exist for a domain.

# Stage 1: signature extraction (per fact, per interval)
for each motion fact f with interval [t_s, t_e]:
    clip = video[t_s - 0.5s : t_e + 0.5s]        # 0.5s context pad
    F_vis = VisualEncoder(clip)                    # frozen video backbone, 768-d
    F_pose = PoseSignatureEncoder(f)               # 2-layer MLP, hidden=256, GeLU, over (Δyaw, Δpitch, Δtranslation, duration)
    S_true = ContrastiveHead(F_vis, F_pose)        # 2-layer MLP, hidden=256, GeLU, output 256-d joint embedding

# Stage 2: counterfactual perturbation set (K=5 perturbations per fact)
perturbations = [
    invert_direction(f),        # left -> right
    scale_magnitude(f, 2.0),    # 90° -> 180°
    shift_interval(f, +2s),     # temporal misalignment
    swap_with_neighbor_fact(f), # substitute a different fact from same video
    drift_calibration_offset(f) # NEW: constant-velocity drift preserving direction/magnitude
]
S_pert_k = ContrastiveHead(F_vis, PoseSignatureEncoder(perturbations[k]))

# Stage 3: agreement score
# cosine similarity margin against each perturbation
d_k = cos(S_true, S_pert_k)          # lower = more discriminable
m_raw = mean_k[ 1 - max(0, d_k - d_min) / (d_max - d_min + eps) ]
# d_min and d_max are computed per-fact over the K=5 perturbations (not global);
# we verify in the calibration study that w is invariant to this choice.

# Stage 4: calibration + exclusion + reward integration
w = F_domain(m_raw)                  # percentile against per-domain null (Stage 0)
if w < tau_exclude (=0.25):
    exclude f from reward supervision entirely
else:
    factual_reward_term *= w        # down-weight, do not drop
```

Training the verifier: we build contrastive pairs by taking each of the 50,000 EgoCaptioner training examples, extracting its pose-derived facts, and generating the K=5 perturbations above as negatives; the unperturbed fact is the positive. We train only the PoseSignatureEncoder and ContrastiveHead (VisualEncoder frozen) with InfoNCE loss, temperature 0.07, batch 256, for 3 epochs. Crucially, the verifier is trained on the *same* pose facts EgoCaptioner uses, so it learns the visual signature of a correct fact rather than a generic motion classifier. The per-domain null calibration (Stage 0) uses 512 held-out clean-sensor facts and is fit once after verifier training; it adds no trainable parameters.

**Why this design.** We chose a contrastive margin against structured perturbations over a binary consistency classifier because a classifier would need labeled inconsistent facts that do not exist, while perturbations give us a controlled negative distribution whose difficulty we can tune per fact; the cost is that the verifier can only detect inconsistency types we enumerate, so an unmodeled sensor failure (e.g., a calibration offset that preserves direction and magnitude) will pass. We chose to down-weight rather than hard-drop facts above τ_exclude because hard filtering removes supervision signal from exactly the noisy-sensor videos where the model most needs to learn robustness; the cost is that a moderately wrong fact still contributes a small gradient. We chose to freeze the visual encoder rather than fine-tune it because fine-tuning on perturbed pairs would let the encoder co-adapt to the perturbation set and inflate margins without improving real consistency; the cost is that the verifier inherits the frozen backbone's blind spots (e.g., fast motion blur). We added the per-domain null calibration (Stage 0) and the fifth drift perturbation to partially close the admitted blind spot: the null calibration makes w a percentile against facts known to be correct in the same domain, so domain shift cannot flip the score, and the drift perturbation gives the negative set coverage over the calibration-offset failure mode. This differs from the obvious composition of "add a classifier + threshold" (the anti-pattern closest to our pitch): a classifier trained to predict consistency is a discriminative model over facts, whereas PAVV computes a *relative* margin against counterfactuals of the same fact, so its score is invariant to fact difficulty and video-domain shifts that would flip a classifier's absolute output. EgoCaptioner's own reward (E3) uses factual accuracy against references with no such per-fact calibration; PAVV is not a variant of that reward but a precondition on which facts enter it.

**Why this measures what we claim.** The quantity w(f,[t_s,t_e]) measures *visual identifiability of the motion fact* because the named assumption is that the contrastive head's cosine margin between the true fact and its structured perturbations is monotone in the fact's visual discriminability; this assumption fails when the perturbation is visually indistinguishable from the truth (e.g., a 90° turn and a 180° turn produce similar optical flow in a narrow FOV), in which case w reflects perturbation-set coverage rather than true reliability. The per-domain null calibration (Stage 0) partially mitigates this by ranking the margin against clean-sensor facts in the same domain, but it cannot recover discriminability along directions absent from the perturbation set. The exclusion threshold τ_exclude measures *fact unreliability* because the assumption is that facts with w below 0.25 are dominated by sensor noise rather than by verifier blindness; this fails when the verifier is systematically blind to a whole video domain, in which case τ_exclude reflects domain shift rather than noise. The factual-reward multiplication w·r_factual measures *noise-attenuated factual supervision* because the assumption is that w is independent of the caption's own content (the verifier sees only pose and pixels, not the generated caption); this fails when a fact is visually ambiguous but the caption happens to be correct, in which case the down-weighting penalizes a correct caption.

## Contribution

['A per-fact, per-interval pose-visual consistency verifier (PAVV) that uses structured counterfactual perturbations of pose-derived motion facts to compute a reliability weight, providing the first mechanism in the egocentric captioning pipeline that checks auxiliary signals against video pixels before they enter a reward.', "A reward-shaping scheme that multiplies EgoCaptioner's factual-accuracy reward term by the verifier weight with a hard exclusion floor, establishing the design principle that auxiliary-signal supervision should be reliability-weighted rather than globally trusted.", 'A diagnostic protocol — injected pose noise at controlled levels (yaw drift, magnitude scaling, temporal offset) — for measuring caption factual accuracy degradation as a function of auxiliary-signal corruption, which isolates the propagation path that prior egocentric captioning work leaves unmeasured.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Ego4D-derived EgoCaptioner examples with pose facts | Same facts the verifier is trained on; tests real pipeline |
| Primary metric | Factual-accuracy reward on low-pose-reliability subset | Isolates where noise attenuation should pay off |
| Baseline | EgoCaptioner (GRPO, unweighted factual reward) | Tests whether per-fact weighting beats uniform supervision |
| Baseline | Hard-drop thresholding on pose-fact confidence | Tests down-weighting against the obvious filtering variant |
| Baseline | Binary consistency classifier + threshold | Tests relative margin against absolute discriminative scoring |
| Ablation-of-ours | PAVV with fixed w=1 (no verifier) | Isolates the verifier's contribution from the EgoCaptioner base |
| Calibration study | Rank correlation of w against injected synthetic pose noise (σ ∈ {0.5°, 1°, 2°, 5°} yaw/translation) | Tests monotonicity of w in true fact reliability |

### Why this setup validates the claim

The central claim is that a per-fact, per-interval reliability weight derived from counterfactual contrastive margins improves factual caption supervision precisely where pose-derived facts are noisy, without harming clean cases. The dataset choice keeps the verifier and the captioner on the same pose-fact distribution, so any gain cannot be attributed to a distribution shift in supervision. The primary metric — factual-accuracy reward restricted to the low-pose-reliability subset — is the right detector because the predicted effect is a *redistribution* of gains toward noisy facts, not a uniform lift; a uniform lift would be indistinguishable from generic regularization. EgoCaptioner-without-verifier tests whether per-fact calibration beats uniform reward at all. The hard-drop baseline tests the down-weighting-vs-filtering sub-claim: if hard dropping matches PAVV, the down-weighting argument is unnecessary. The binary classifier baseline tests the relative-margin-vs-absolute-score sub-claim: if an absolute classifier matches PAVV, the counterfactual construction adds nothing. The w=1 ablation anchors the scale of the verifier's effect, ensuring any observed gain is attributable to the weighting rather than to incidental training dynamics. The calibration study directly tests the load-bearing monotonicity assumption: we inject controlled Gaussian yaw/translation noise at known σ into Ego4D pose tracks to create a graded ground-truth noise axis, then measure Spearman rank correlation between w and the known noise magnitude; a high negative correlation (w decreases as σ increases) supports the assumption, while a flat or non-monotone curve falsifies it. The low/moderate/high pose-reliability subsets are pre-registered as the tertiles of injected σ (low: σ ≤ 1°, moderate: 1° < σ ≤ 2°, high: σ > 2°), ensuring reproducible band comparisons before running GRPO.

### Expected outcome and causal chain

**vs. EgoCaptioner (unweighted factual reward)** — On a case where a pose-derived fact reports a left turn that the wearer did not make (sensor drift during a stationary pause), the baseline assigns full factual reward to a caption that faithfully narrates the spurious turn, because its reward has no per-fact calibration and trusts every reference fact equally. Our method instead computes a low cosine margin between the true fact and its inverted-direction perturbation, yielding w below τ_exclude and excluding the fact from supervision. We expect a noticeable factual-accuracy gap concentrated on the low-pose-reliability subset, with near-parity on clean-sensor videos where no facts are down-weighted.

**vs. Hard-drop thresholding on pose-fact confidence** — On a case where a fact is moderately noisy (a 90° turn narrated with slightly wrong timing), the baseline drops the fact entirely because its confidence falls below threshold, removing supervision from exactly the video where the captioner most needs to learn robustness. Our method instead down-weights the fact with a mid-range w, retaining a small gradient. We expect PAVV to outperform hard dropping on the moderate-noise band specifically, with the two converging on the extreme-noise band where both effectively exclude.

**vs. Binary consistency classifier + threshold** — On a case where a fact is genuinely correct but the video domain is visually unusual (low light, fast motion), the baseline classifier's absolute consistency score drops and it wrongly excludes a correct fact, because its output is not calibrated against fact difficulty. Our method instead compares the true fact against perturbations of the same fact and ranks the margin against a per-domain null of clean-sensor facts, so its score stays high when the fact is discriminable relative to its counterfactuals and its domain peers. We expect PAVV to show fewer false exclusions on domain-shifted videos, visible as a smaller drop in factual accuracy on that subset.

### What would falsify this idea

If PAVV's gain over unweighted EgoCaptioner is uniform across the low- and high-pose-reliability subsets rather than concentrated on the low-reliability subset, the claim that per-fact calibration targets noise is wrong. If the calibration study shows a flat or non-monotone rank correlation between w and injected synthetic pose noise, the load-bearing monotonicity assumption is false and w reflects perturbation-set coverage rather than fact reliability.

## References

1. EgoCaptioner: Towards Comprehensive and Fine-Grained Egocentric Video Captioning
