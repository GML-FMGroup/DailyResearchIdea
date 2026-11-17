# Goal-Consistent Action DiT: Coupling Visual Chain-of-Thought to Action via Learned Forward Dynamics

## Motivation

Across the vision-language-action model (VLA) literature, visual chain-of-thought and action generation remain structurally decoupled: CoT-VLA (E1) generates future frames as visual goals but never constrains the action sequence to causally produce them, and ScaleVLA (E5) pairs a 552B VLM with an action DiT without any goal-consistency term. This decoupling means a VLA can achieve high action success while its generated visual goal is causally irrelevant to the executed trajectory—the reasoning channel is decorative rather than load-bearing. The field's scale/unify trajectory (E5) inherits this gap at larger scale: bigger backbones improve sample efficiency and generalization but do not add the causal-consistency constraint that would make visual CoT mechanistically necessary for action.

## Key Insight

A learned forward dynamics model trained on the robot's own transition data defines a measurable causal path from action chunks to visual outcomes, and a monotone consistency margin on that path forces the VLM's generated goal to be a sufficient statistic for action selection rather than a parallel decorative output.

## Method

## FARA: Forward-dynamics Anchored Reasoning for Actions

**(A) What it is.** FARA is a training objective and architecture wrapper for ScaleVLA-style VLAs that adds a goal-conditioned consistency loss. Inputs: a language instruction, current observation, the 552B VLM backbone, and a trainable action DiT. Outputs: a visual goal latent and an action chunk that, when rolled through a learned forward dynamics model, lands within a monotone margin of that goal latent. The load-bearing assumption is stated explicitly: **the learned forward dynamics model FDM(obs, A) is an accurate local forward model on the robot's transition distribution for the action chunks the DiT actually emits during training, so that 1 - cos(z_pred, z_goal) measures true causal goal-consistency rather than FDM extrapolation error.** FARA makes this assumption per-sample verifiable via an uncertainty gate (step 4b) and a curriculum on λ_cons (step 6).

**(B) How it works.**

```pseudocode
# Per training step, given (instruction, obs, future_frames, action_chunk_gt):

# 1. VLM generates goal latent
z_goal = VLM.encode_goal(instruction, obs)          # d=4096 latent, from frozen 552B backbone

# 2. Action DiT produces chunk
A = ActionDiT(z_goal, obs, instruction)              # A in R^{T x d_action}, T=16, d_action=7

# 3. Forward dynamics model predicts visual outcome
#    FDM is an ensemble of K=5 small trainable ViT-style predictors (12 layers, 256-dim, 8 heads)
#    each trained on (obs, A) -> future latent; output projection initialized from VLM.encode_future
z_pred_k = FDM_k(obs, A) for k=1..5                # each z_pred_k in R^4096
z_pred = mean_k(z_pred_k)                           # ensemble mean
sigma_pred = std_k(z_pred_k)                        # ensemble disagreement, scalar per sample

# 4. Goal-consistency loss with monotone margin and uncertainty gate
#    margin m = 0.1 (fixed); cosine distance d_cos between latents
d_goal = 1 - cos(z_pred, z_goal.detach())            # detach goal: no gradient into VLM
# 4b. Uncertainty gate: mask loss where FDM is not calibrated
gate = 1[sigma_pred <= tau]                          # tau = 0.05, set as 90th percentile of sigma_pred on held-out transitions
L_cons = gate * max(0, d_goal - m)                  # hinge: only penalize when goal is far AND FDM is confident

# 5. Standard action loss (behavior cloning)
L_act = ||A - action_chunk_gt||^2

# 6. FDM is trained jointly with a reconstruction loss on true transitions
L_fdm = mean_k ||FDM_k(obs, action_chunk_gt) - VLM.encode_future(future_frames)||^2

# Total loss
L = L_act + lambda_cons(t) * L_cons + lambda_fdm * L_fdm
# lambda_fdm = 0.1, tuned on held-out sim
# lambda_cons(t) annealed linearly from 0 to 0.5 over first 20% of training steps
# (curriculum: consistency signal enters only as DiT outputs enter FDM training support)
```

**Diagram of gradient flow.**

```
instruction, obs ──► 552B VLM (frozen) ──► z_goal ──(detach)──┐
                                                              ▼
instruction, obs ──► Action DiT ──► A ──► FDM ensemble (K=5) ──► z_pred, sigma_pred ──► gate ──► L_cons (hinge, margin m)
                                                              ▲
future_frames ──► VLM.encode_future ──► z_future ──► L_fdm ──┘
```

**(C) Why this design.** We chose a learned forward dynamics model (FDM) over a frozen pretrained video predictor because the FDM must be trained on the same robot transition distribution as the action DiT; a frozen web-scale video model would introduce a domain gap that makes the consistency signal measure visual plausibility rather than robot-specific causal outcome, accepting the cost of extra trainable parameters and a joint-training schedule. We chose a hinge loss with a fixed margin m=0.1 over an unconstrained L2 penalty because an unconstrained penalty would force z_pred to collapse onto z_goal even when the goal is unreachable given the current observation, destroying the VLM's ability to express aspirational goals; the hinge permits a tolerance band that preserves goal diversity, accepting the cost that the margin must be tuned per latent scale. We detached z_goal from the consistency gradient because otherwise the VLM would be incentivized to generate trivially predictable goals rather than informative ones—the same failure mode that makes naive CoT-VLA (E1) goals decorative in the opposite direction; detaching keeps the VLM's goal generation shaped only by its own pretraining and the action loss's indirect pressure, accepting slower VLM adaptation. We chose cosine distance over Euclidean because latent visual representations from VLMs are directionally meaningful but not norm-calibrated, so cosine isolates semantic alignment from scale drift, accepting the cost that magnitude information about goal uncertainty is discarded. We added an ensemble of K=5 FDMs with an uncertainty gate (sigma_pred <= tau) and a λ_cons curriculum because the FDM is queried on out-of-distribution action chunks early in training; the gate ensures L_cons fires only on action chunks the FDM is calibrated on, and the curriculum delays consistency pressure until the DiT's output distribution enters the FDM's training support, accepting the cost of 5x FDM parameters and a per-sample gate computation. A domain expert would not describe FARA as a variant of CoT-VLA (E1): CoT-VLA generates future frames but never rolls actions through a dynamics model to check causal consistency, so its visual goal is a parallel output; FARA's hinge loss operates on the composition FDM(A(z_goal)), making the goal a constraint on the action manifold rather than an auxiliary prediction.

**(D) Why it measures what we claim.** The quantity d_goal = 1 - cos(z_pred, z_goal) measures causal goal-consistency because the assumption that the FDM is an accurate local forward model on the robot's transition distribution makes z_pred equivalent to the visual outcome the action chunk would actually produce; this assumption fails when the FDM is queried on action chunks far outside the training distribution (e.g., early in training when the DiT outputs random actions), in which case d_goal reflects FDM extrapolation error rather than true causal inconsistency, and the uncertainty gate (sigma_pred <= tau) masks these samples while the hinge margin m absorbs residual error. The detach of z_goal measures goal informativeness preservation because the assumption that the VLM's pretrained goal encoder produces semantically meaningful latents without consistency-gradient pressure makes z_goal a stable reference; this assumption fails when the VLM's goal latents are themselves noisy or degenerate for a given instruction, in which case the consistency loss measures alignment to a meaningless target rather than causal coupling. The joint FDM reconstruction loss L_fdm measures forward-model fidelity because the assumption that future-frame latents from the VLM encoder capture task-relevant visual state makes reconstruction error a proxy for dynamics accuracy; this assumption fails when future frames contain task-irrelevant visual variation (lighting, background), in which case L_fdm measures reconstruction of nuisance factors rather than dynamics. The ensemble disagreement sigma_pred measures FDM calibration because the assumption that ensemble members diverge on out-of-distribution inputs makes disagreement a proxy for prediction error; this assumption fails when ensemble members share a correlated failure mode (e.g., all trained on the same biased subset), in which case sigma_pred underestimates true error and the gate admits corrupted consistency gradients.

## Contribution

["A goal-conditioned consistency training objective (FARA) that couples a VLM's visual chain-of-thought goal latent to an action DiT's output through a learned forward dynamics model, using a detached-goal hinge loss with a monotone margin.", 'A design principle for VLA training: the forward dynamics model must be trained on the same robot transition distribution as the action policy and the goal latent must be detached from the consistency gradient, otherwise the consistency signal either measures visual plausibility or collapses goal diversity.', 'An open-source implementation of the FDM and consistency loss as a drop-in wrapper for ScaleVLA-style architectures, with hyperparameter guidance (margin, loss weights, detach policy) for reproduction.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | LIBERO-PRO + real-world charger insertion | Generalization and contact-sensitive precision stress causal consistency |
| Primary metric | Success rate on contact-sensitive manipulation subset | Directly reflects causal goal-consistency, not visual plausibility |
| Baseline | ScaleVLA (no consistency loss) | Tests whether FDM anchoring adds anything beyond scale |
| Baseline | CoT-VLA-style visual goal prediction | Tests whether parallel goals suffice versus causal anchoring |
| Baseline | Unconstrained L2 consistency (no hinge, no detach) | Tests whether hinge and detach are load-bearing |
| Baseline | Frozen pretrained video predictor as FDM (web-scale video diffusion) | Tests whether robot-distribution-trained FDM yields different consistency signal |
| Ablation-of-ours | FARA with frozen random FDM | Tests whether learned robot-distribution dynamics matter |
| Ablation-of-ours | FARA with FDM output projection frozen to VLM encoder | Tests whether cross-encoder latent alignment is load-bearing |
| Ablation-of-ours | FARA without uncertainty gate (gate=1 always) | Tests whether calibration gate prevents FDM-error corruption |
| Ablation-of-ours | FARA with λ_cons fixed at 0.5 from step 0 | Tests whether curriculum on FDM validity matters |
| Feasibility study | FARA with 2B and 7B VLM backbones (CoT-VLA 7B setup) | Tests core claim without frontier-scale compute |
| Scaling study | FARA vs. ScaleVLA at 2B, 34B, 552B backbones on LIBERO-PRO | Tests whether consistency term changes scaling trajectory |
| Calibration analysis | Plot FDM prediction error vs. d_goal on held-out transitions; report d_goal distribution for consistent vs. inconsistent chunks | Justifies m=0.1 separates regimes |

### Why this setup validates the claim

The central claim is that anchoring actions to a VLM goal through a learned forward dynamics model produces causally consistent behavior that pure behavior cloning and parallel visual-goal prediction cannot. LIBERO-PRO generalization plus real-world charger insertion is the right testbed because both stress the exact regime where the FDM's local accuracy assumption holds and where contact-sensitive alignment exposes causal inconsistency: a policy that merely imitates demonstrations or emits decorative goals will drift on novel objects and miss precise insertions. The ScaleVLA baseline isolates whether the consistency term adds value beyond frontier-scale backbone capacity, since ScaleVLA already reaches 45.1% LIBERO-PRO and 9/10 insertions; any FARA gain must appear on top of that. The CoT-VLA-style baseline tests the sub-claim that a goal must constrain the action manifold rather than sit as a parallel output. The unconstrained-L2 baseline tests whether the hinge and detach are doing real work or whether any consistency pressure suffices. The frozen-video-predictor baseline tests whether the robot-distribution-trained FDM yields a measurably different (not just better) consistency signal, establishing that the learned-FDM choice is a substantive mechanism rather than a hyperparameter. The random-FDM ablation tests whether the dynamics model must be trained on robot transitions rather than acting as a generic latent regularizer. The frozen-output-projection ablation tests whether the cross-encoder latent-alignment assumption (that FDM output space and VLM goal space share a metric) is load-bearing. The no-gate and fixed-λ ablations test whether the uncertainty gate and curriculum are necessary to prevent FDM-error corruption. The 2B/7B feasibility study tests whether the consistency gain replicates without frontier-scale compute. The 2B/34B/552B scaling study tests whether the consistency term changes the scaling relationship itself. The calibration analysis plots FDM prediction error against d_goal on held-out transitions and reports the empirical distribution of d_goal for causally consistent vs. inconsistent action chunks, justifying that m=0.1 separates the two regimes rather than being an arbitrary tolerance. Success rate on the contact-sensitive subset is the right primary metric because it is precisely where causal goal-consistency should manifest; aggregate success would dilute the predicted effect.

### Expected outcome and causal chain

**vs. ScaleVLA (no consistency loss)** — On a charger-insertion instance where the demonstration distribution covers approach but not the final contact-sensitive alignment, ScaleVLA produces a plausible-looking approach trajectory that misses the port because behavior cloning has no signal distinguishing causally correct from merely visually similar actions. Our method instead rolls the candidate chunk through the FDM and hinges the predicted outcome toward the VLM's goal latent, so chunks that fail to land near the goal are penalized, and we expect the gain to concentrate on contact-sensitive insertions (roughly 4/10 to 8-9/10) with near-parity on coarse approach tasks.

**vs. CoT-VLA-style visual goal prediction** — On a novel-object LIBERO-PRO case, the CoT-VLA-style baseline emits a future-frame goal that looks reasonable but is never checked against the action it conditions, so the executed chunk can be causally inconsistent with its own goal and the policy fails on the novel instance. Our method instead makes the goal a constraint on the action manifold via the hinge on FDM(A), so inconsistency is directly penalized; we expect a noticeable LIBERO-PRO gap (roughly 45% versus low-30s) with parity on in-distribution tasks where decorative goals suffice.

**vs. Unconstrained L2 consistency (no hinge, no detach)** — On an instruction whose goal is aspirational given the current observation (e.g., a target out of reach this step), the unconstrained baseline forces z_pred onto z_goal, collapsing the VLM's goal diversity and producing degenerate, over-committed actions that fail when the goal is temporarily unreachable. Our method instead lets the hinge absorb unreachable distances and detaches the goal, preserving informative goals; we expect the unconstrained variant to match FARA on easy subsets but degrade sharply on multi-stage or recovery tasks, showing a gap concentrated where goal diversity matters.

**vs. Frozen pretrained video predictor as FDM** — On a contact-sensitive insertion, the frozen video predictor's consistency signal reflects web-scale visual plausibility rather than robot-specific causal outcome, so it may reward visually plausible but dynamically infeasible chunks. Our method's robot-distribution-trained FDM yields a different consistency signal (we expect a measurable difference in d_goal ranking of chunks, not just lower error), and we expect FARA to outperform this baseline on contact-sensitive insertions, establishing that the learned-FDM choice is a substantive mechanism.

**vs. FARA with frozen random FDM** — If the random FDM matches FARA, the learned robot-distribution dynamics model is not the source of the effect; we expect FARA to significantly outperform the random-FDM ablation on contact-sensitive and novel-object subsets, confirming that dynamics trained on robot transitions are necessary.

**vs. FARA with FDM output projection frozen to VLM encoder** — If freezing the output projection (enforcing cross-encoder latent alignment) matches or improves FARA, the alignment assumption is not load-bearing; we expect the frozen-projection variant to underperform FARA on contact-sensitive tasks, confirming that joint training of the FDM output projection is necessary for the cosine distance to measure causal goal-consistency.

**vs. FARA without uncertainty gate (gate=1 always)** — On early-training steps where the DiT emits random actions, the no-gate variant receives corrupted consistency gradients from FDM extrapolation error, leading to worse final performance; we expect FARA with the gate to outperform the no-gate variant on contact-sensitive and novel-object subsets, confirming that the calibration gate prevents FDM-error corruption.

**vs. FARA with λ_cons fixed at 0.5 from step 0** — The fixed-λ variant applies consistency pressure before the DiT's outputs enter the FDM's training support, causing early training instability; we expect FARA with the curriculum to converge faster and achieve higher final success on contact-sensitive tasks, confirming that the curriculum on FDM validity matters.

**vs. FARA at 2B and 7B backbones (feasibility)** — We expect the consistency gain over ScaleVLA to replicate at 2B and 7B, with smaller absolute gains than at 552B but the same qualitative pattern (concentration on contact-sensitive and novel-object subsets), confirming the core claim is testable without frontier-scale compute.

**vs. ScaleVLA at 2B, 34B, 552B (scaling study)** — We expect FARA's gain over ScaleVLA to be roughly constant or increasing with backbone scale on LIBERO-PRO, indicating that the consistency term changes the scaling trajectory rather than being a fixed offset; if the gain shrinks with scale, the significance claim is weakened.

**Calibration analysis** — We expect FDM prediction error to correlate positively with d_goal on held-out transitions, and the empirical distribution of d_goal for causally consistent chunks to concentrate below m=0.1 while inconsistent chunks spread above, justifying the margin choice; if the distributions overlap heavily, m=0.1 does not separate the regimes and the hinge is arbitrary.

### What would falsify this idea

If FARA's gain over ScaleVLA is uniform across coarse approach and contact-sensitive subsets rather than concentrated on the contact-sensitive and novel-object cases where the causal-consistency mechanism is predicted to bite, the central claim that forward-dynamics anchoring produces causally consistent behavior is wrong. Likewise, if the random-FDM ablation matches FARA, the learned robot-distribution dynamics model is not the source of the effect; if the frozen-projection ablation matches FARA, the cross-encoder latent-alignment assumption is not load-bearing; and if the no-gate ablation matches FARA, the uncertainty gate does not prevent FDM-error corruption.

## References

1. ScaleVLA: Scaling Vision-Language-Action Models to 552 Billion Parameters
