# Action-Coupled Low-Light VQA with Stochastic Rollout Consistency

## Motivation

DarkQA benchmarks vision-language models on low-light visual-primitive QA but decouples perception from action, so action-conditioned robustness under sensor noise remains unestablished; this mirrors a structural gap across embodied VLM ancestors (RT-2, PaLM-E) that assume perception can be evaluated separately from action. The root cause is that existing evaluation protocols treat QA correctness and action execution as independent surfaces, so a model can score well on primitive questions while its action policy collapses under the same noisy observations.

## Key Insight

If a VLM's primitive answers and its action head are trained on the same noisy latent state, then the residual between the action head's predicted next latent and the VLM's answer-conditioned latent prediction is a direct, observable signal of action-conditioned perceptual robustness — a quantity that decoupled QA benchmarks structurally cannot expose.

## Method

**What it is.** ACRO (Action-Coupled Rollout Objective) couples a VLM's primitive QA head with a lightweight action head in a simulated indoor environment. Inputs: low-light egocentric RGB frames synthesized by DarkQA's ISP-based pipeline; outputs: (i) primitive answers to DarkQA-style questions and (ii) discrete navigation actions (forward/left/right/stop) executed in a grid-world simulator with stochastic transitions.

**How it works.**

```pseudocode
# Phase 1: Coupled forward pass on a noisy frame x_t
z_t   = VLM.encode(x_t)                      # shared latent, dim 4096
q_t   = VLM.qa_head(z_t)                     # primitive answer logits
a_t   = ActionHead(z_t)                      # 4-way action logits

# Phase 2: Rollout with stochastic transitions
x_{t+1} = Sim.step(x_t, a_t; slip_prob=0.15)  # 15% action slip
a_t^*, q_t^* = Oracle(x_t)                    # ground-truth action & answer

# Phase 3: Coupled loss (single backward)
L_qa   = CE(q_t, q_t^*)
L_act  = CE(a_t, a_t^*)
# Rollout consistency is computed ONLY on answer-preserving transitions
# (q_t^* == q_{t+1}^*), and includes a negative term to prevent latent collapse.
if q_t^* == q_{t+1}^*:
    L_roll = || VLM.qa_latent(x_{t+1}) - StopGrad(VLM.qa_latent(x_t)) ||_2^2
else:
    L_roll = 0
# Negative term: push apart latents of answer-different frames
L_neg = max(0, margin - || VLM.qa_latent(x_{t+1}) - StopGrad(VLM.qa_latent(x_t)) ||_2^2)  # margin=1.0
L      = L_qa + 0.5*L_act + 0.1*L_roll + 0.1*L_neg
```

At evaluation, we report (a) QA accuracy on DarkQA question families, (b) success rate on navigation goals, and (c) the **action-conditioned robustness gap** = QA accuracy on frames reached by successful rollouts minus QA accuracy on frames reached after an action slip. A model that merely answers primitive questions well but cannot execute actions will show a large gap; ACRO is designed to minimize it. To isolate action-conditioned noise from illumination variation, we compute the gap on **illumination-matched** frame sets: for each slipped frame, we match a successful frame with the same mean luminance (within ±5% of the slipped frame's mean pixel value) and compute the gap only on these matched pairs. We also report the **scene-change control**: for each slipped frame, we compute the L2 distance between the frame and its pre-slip counterpart (scene-change magnitude) and regress it out of the gap via linear residualization; the residual gap is the action-conditioned robustness gap after controlling for visual change.

**Why this design.** We chose a shared latent `z_t` feeding both heads over two separate encoders because the motivation requires that QA and action see the *same* noisy observation, accepting that the action head cannot specialize its features. We chose a rollout-consistency loss on the VLM's answer-conditioned latent over a pure action-imitation loss because imitation alone does not penalize the VLM for producing answers that are inconsistent with the state it will actually reach after acting, accepting the cost of a stop-gradient target that makes the loss asymmetric. We chose a stochastic simulator with 15% slip probability over deterministic transitions because the bridge_add mechanism requires stochastic rollouts to expose action-conditioned robustness, accepting higher variance in training that we absorb with a fixed 0.1 weight. A domain expert would not describe ACRO as a variant of RT-2 or PaLM-E because those models train action heads on *clean* observations and evaluate QA separately; ACRO's load-bearing signal is the rollout-consistency residual on the *same* low-light latent, which those methods never compute.

**Why it measures what we claim.** The rollout-consistency residual `L_roll` measures action-conditioned perceptual robustness because it is the squared distance between the VLM's answer-conditioned latent at the *next* state (reached by acting) and the latent at the current state, computed only on transitions where the oracle answer is unchanged (q_t^* == q_{t+1}^*), so it targets noise-robustness conditional on semantic equivalence rather than raw visual change; this assumption fails when the slip moves the agent to a visually distinct region that still preserves the answer, in which case the residual reflects visual change rather than perceptual noise. The negative term `L_neg` prevents the trivial constant-latent solution by penalizing small distances between answer-different frames. The action-conditioned robustness gap measures the motivation-level concept of coupled perception-action robustness because it compares QA accuracy on successful vs. slipped rollout frames after illumination matching and scene-change residualization; this assumes that the only remaining difference between the two frame sets is the action slip, and this assumption fails when the slip correlates with lighting changes, in which case the gap reflects illumination variation rather than action-conditioned noise. The shared latent `z_t` measures the coupling between QA and action because both heads read from the same representation; this assumes the heads do not learn to ignore `z_t` for their respective tasks, and this assumption fails when one head dominates training, in which case `z_t` reflects only the dominant task.

## Contribution

['ACRO: a coupled training and evaluation framework that shares a VLM latent between a primitive-QA head and an action head, with a rollout-consistency loss on the answer-conditioned latent under stochastic simulator transitions.', "The action-conditioned robustness gap metric, which quantifies how much a VLM's primitive QA accuracy degrades when its own actions lead to slipped states, exposing a failure mode that decoupled QA benchmarks structurally cannot measure.", "An open-source simulated indoor environment with DarkQA's ISP-based low-light synthesis integrated into the rollout loop, enabling reproducible action-conditioned low-light evaluation."]

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | DarkQA low-light QA pairs | Provides primitive questions on noisy egocentric frames |
| Primary metric | Action-conditioned robustness gap (illumination-matched, scene-change residualized) | Directly quantifies QA degradation after action slips while controlling for illumination and visual change |
| Baseline | DarkQA VLMs without LLIE | Tests whether illumination preprocessing alone closes the gap |
| Baseline | RT-2-style action head on clean frames | Tests whether decoupled action training suffices |
| Baseline | LLaVA visual instruction tuning | Tests whether general multimodal QA transfers to coupled robustness |
| Ablation-of-ours | ACRO without rollout-consistency loss | Isolates the contribution of the coupled residual |
| Ablation-of-ours | ACRO with two separate encoders + cross-head consistency term | Tests whether shared-latent coupling is the novel ingredient |

### Why this setup validates the claim

The central claim is that coupling QA and action through a shared latent with a rollout-consistency residual reduces the action-conditioned robustness gap beyond what decoupled or preprocessing-based approaches achieve. The DarkQA dataset supplies the low-light primitive QA pairs and the grid-world simulator supplies stochastic rollouts, so the gap metric can be computed on frames reached by successful versus slipped actions. The DarkQA-without-LLIE baseline tests whether the gap is merely an illumination artifact: if it closes the gap as well as ACRO, the coupling claim is unsupported. The RT-2-style baseline tests whether a clean-frame action head is sufficient, isolating the effect of training on noisy, rollout-reached frames. The LLaVA baseline tests whether generic multimodal instruction tuning transfers to this coupled setting. The ablation removes the rollout-consistency loss while keeping the shared latent and action head, so any remaining gap reduction must come from the consistency residual rather than from multi-task training alone. The two-encoder ablation replaces the shared latent with separate encoders plus a cross-head consistency term, so if it matches ACRO's gap reduction, the shared-latent coupling is not the novel ingredient. The gap metric is the right detector because it is computed on the exact frame distribution the mechanism is designed to stabilize, after illumination matching and scene-change residualization to remove confounds, making a uniform improvement across all subsets a red flag rather than a success.

### Expected outcome and causal chain

**vs. DarkQA VLMs without LLIE** — On a frame reached after a 15% action slip moves the agent toward a dim corner, the baseline produces a wrong primitive answer because its features are dominated by sensor noise and low illumination without any action-conditioned correction. Our method instead assigns a similar answer-conditioned latent to the slipped frame as to the pre-slip frame because the rollout-consistency residual explicitly penalizes latent drift under stochastic transitions, computed only on answer-preserving slips. We expect a noticeable gap reduction concentrated on slip-heavy question families, with parity on frames reached without slips, and this reduction persists after illumination matching and scene-change residualization.

**vs. RT-2-style action head on clean frames** — On a rollout where the action head was trained only on clean observations, the baseline executes the correct action but its QA head produces an answer inconsistent with the noisy next state because the two heads never share a rollout-consistency signal. Our method instead couples the heads through the shared latent and the residual, so the QA latent is pulled toward the state the agent will actually reach. We expect the baseline to show a larger gap on slipped frames and near-parity on successful frames, while ACRO shows a compressed gap across both, with the compression surviving the scene-change control.

**vs. LLaVA visual instruction tuning** — On a low-light frame with a primitive question about a shadowed object, the baseline answers from generic visual priors because instruction tuning never exposed it to action-conditioned noise distributions. Our method instead reshapes the shared latent through the action-coupled loss, so the answer-conditioned latent is stable across slip-induced visual changes. We expect LLaVA to show a wide gap that is roughly uniform across question families, while ACRO shows a narrower gap concentrated in the families where slips actually change the visual scene, and the negative term prevents latent collapse so QA accuracy on visually distinct slipped frames does not degrade.

**vs. ACRO without rollout-consistency loss** — The ablation keeps the shared latent and action head but removes L_roll and L_neg, so it can only benefit from multi-task training. We expect it to show a gap similar to the RT-2-style baseline on slipped frames, confirming that the consistency residual is load-bearing.

**vs. ACRO with two separate encoders + cross-head consistency term** — This ablation replaces the shared latent with separate encoders and a cross-head consistency term. If it matches ACRO's gap reduction, the shared-latent coupling is not the novel ingredient; we expect it to underperform ACRO because the heads do not see the same noisy observation.

### What would falsify this idea

If ACRO's gap reduction is uniform across all question families and frame subsets rather than concentrated on slip-reached frames, or if the ablation without the rollout-consistency loss matches ACRO's gap, the residual is not measuring action-conditioned robustness. If the two-encoder ablation matches ACRO's gap, the shared-latent coupling is not the novel ingredient.

## References

1. DarkQA: Benchmarking Vision-Language Models on Visual-Primitive Question Answering in Low-Light Indoor Scenes
2. Improved Baselines with Visual Instruction Tuning
3. Thinking in Space: How Multimodal Large Language Models See, Remember, and Recall Spaces
