# Causal Sufficiency Tokens for Vision-Language-Action Models

## Motivation

Existing VLA models like PDRVLA learn physical tokens by predicting future RGB/depth/flow features, which conflates predictive adequacy with causal adequacy: the tokenizer is free to encode nuisance variation (lighting, camera pose, distractors) that helps future-feature prediction but is irrelevant to action outcomes, while underweighting action-relevant variation (contact, object pose, gripper state) that future prediction can partially infer from context. This failure is structural across the field—every branch treats a learned conditioning signal (task embedding in BC-Z, raw trajectory, simulated eval, frozen world-model distillation in PDRVLA) as a valid control substrate without any formal condition linking the representation to the causal structure of the transition it summarizes. The root cause is that no existing objective enforces the token to be a minimal sufficient statistic of the latent transition kernel that determines action outcomes.

## Key Insight

Constraining the token space to be a minimal sufficient statistic of the latent transition kernel via interventional identifiability—invariance to action-irrelevant nuisance interventions and maximal sensitivity to action-relevant interventions—yields a representation that is causally sufficient for control, a property that future-feature prediction cannot guarantee because it rewards encoding any variation that helps predict the future, including nuisance.

## Method

# Causal Sufficiency Tokens (CST) for VLA
Input: VLA backbone f_theta (frozen or fine-tuned), physical dynamics module g_phi (2-layer MLP, hidden=256, GeLU), action head pi_psi (2-layer MLP, hidden=256, GeLU)
Data: demonstrations D = {(o_t, a_t, o_{t+1})}, simulation rollouts with intervention API
Hyperparameters: lambda_inv=1.0, lambda_sens=1.0, lambda_min=0.1, beta=0.1, K=4 interventions per sample (K=2 during first 10k steps, then K=4), warmup_steps=10k, cache_size=50k transitions

1. Token extraction:
   z_t = g_phi(f_theta(o_t))   # physical token, dim d=64

2. Interventional data generation (per minibatch):
   For each transition (o_t, a_t, o_{t+1}):
     - Sample K action-irrelevant interventions I_nuis (camera pose, lighting, distractor objects) via simulator API
     - Sample K action-relevant interventions I_act (contact, object pose, gripper state) via simulator API
     - Produce counterfactual observations: o_t^{nuis,k}, o_t^{act,k} and their outcomes o_{t+1}^{nuis,k}, o_{t+1}^{act,k}
   - Pre-collect a fixed intervention dataset of 50k transitions with K=4 labeled counterfactuals each once and cache it; during training, sample from cache for first 10k steps, then switch to on-the-fly generation with K=4.

3. Interventional identifiability objective:
   # (i) Invariance to nuisance interventions
   L_inv = mean_k || z_t^{nuis,k} - z_t ||^2
   # (ii) Sensitivity to action-relevant interventions
   L_sens = - mean_k || z_t^{act,k} - z_t ||^2 / (||o_t^{act,k} - o_t||^2 + eps)
   # (iii) Minimality: token should be minimally sufficient for predicting action outcome
   #     Use a variational bottleneck: z_t ~ N(mu, sigma^2), reparameterize, KL to N(0,I)
   L_min = beta * KL(q(z_t|o_t) || N(0,I))
   # (iv) Action outcome prediction (to ensure sufficiency)
   L_pred = || pi_psi(z_t, a_t) - o_{t+1} ||^2   # or negative log-likelihood of action

4. Total loss:
   L = L_pred + lambda_inv * L_inv + lambda_sens * L_sens + lambda_min * L_min

5. Training: update g_phi, pi_psi (and optionally f_theta) via Adam, lr=1e-4, batch size 256

**Why this design** — We chose to enforce invariance via a squared-difference penalty on token embeddings under nuisance interventions rather than an adversarial discriminator because the simulator provides exact intervention labels, making the invariance constraint a direct regression target; this avoids the instability of adversarial training, accepting the cost that the penalty may be dominated by large nuisance effects and require careful scaling (we normalize by the observation difference for the sensitivity term). We chose a variational information bottleneck for minimality rather than a discrete quantization scheme because the bottleneck provides a differentiable, soft constraint on the information content of z_t with respect to the observation, which is necessary for end-to-end training with the VLA backbone; this accepts the cost that the bottleneck may not achieve true minimal sufficiency if the KL weight is poorly tuned. We chose to generate interventional data on-the-fly via the simulator API rather than pre-collecting a fixed intervention dataset because it allows the tokenizer to be trained on interventions that are dynamically selected to challenge the current token space (e.g., focusing on nuisance factors that currently cause large token changes), accepting the cost of increased simulation compute per training step. Unlike PDRVLA, which trains tokens only to predict future features and thus has no mechanism to separate nuisance from action-relevant variation, our method explicitly optimizes for the causal sufficiency property by construction.

**Why it measures what we claim** — The computational quantity L_inv measures invariance to action-irrelevant nuisance because we assume that the simulator's nuisance intervention API changes only variables that are not ancestors of the action outcome in the causal graph; this assumption fails when a nuisance intervention (e.g., camera pose change) actually alters the observation in a way that correlates with action-relevant state (e.g., occluding the gripper), in which case L_inv would penalize the token for encoding genuinely action-relevant information, leading to under-sensitivity. The quantity L_sens measures sensitivity to action-relevant interventions because we assume that the action-relevant intervention API changes only variables that are ancestors of the action outcome; this assumption fails when an action-relevant intervention (e.g., object pose change) also changes nuisance factors (e.g., shadows), in which case L_sens would reward encoding nuisance variation, leading to over-sensitivity. The quantity L_min measures minimality because we assume that the variational bottleneck's KL term upper-bounds the mutual information between z_t and the observation given the action, and that minimal sufficiency is achieved when this information is minimized subject to sufficiency for outcome prediction; this assumption fails when the outcome prediction loss L_pred is not a sufficient statistic of the transition kernel (e.g., due to limited capacity of pi_psi), in which case L_min would compress away information that is actually needed for control, leading to a token that is minimal but not sufficient.

## Contribution

['A novel training objective for physical tokens in VLA models that enforces interventional identifiability: invariance to action-irrelevant nuisance interventions and sensitivity to action-relevant interventions, implemented as a differentiable loss over token embeddings using simulator-provided intervention APIs.', "A minimal-sufficient-statistic constraint on the token space via a variational information bottleneck, which provably (under the assumption that the bottleneck's KL term upper-bounds mutual information) yields tokens that discard nuisance variation while retaining action-relevant variation, unlike future-feature prediction which conflates the two.", "An empirical demonstration that causal sufficiency tokens improve action success rates on simulation benchmarks compared to PDRVLA's future-feature-predictive tokens, particularly under distribution shifts involving nuisance factors (lighting, camera pose, distractors)."]

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | LIBERO-Plus with intervention API | Simulator exposes labeled nuisance/action interventions for causal tests |
| Primary metric | Success rate under nuisance shift | Directly probes invariance claim, not raw in-distribution accuracy |
| Baseline | PDRVLA | Future-feature tokens lack nuisance/action separation mechanism |
| Baseline | Standard VLA (no world model) | Tests whether CST gains exceed plain end-to-end VLA |
| Baseline | PDRVLA + nuisance augmentation | Tests whether data augmentation alone replicates CST |
| Ablation-of-ours | CST without L_inv | Isolates whether invariance term drives the predicted gain |
| Ablation-of-ours | CST with raw sensitivity (no normalization) | Tests whether normalization in L_sens is load-bearing |
| Diagnostic | Leakage probe: regress action-relevant state from nuisance-intervened observations | Measures violation of intervention-label completeness assumption |

### Why this setup validates the claim

The central claim is that explicitly optimizing a token for causal sufficiency — invariance to nuisance interventions plus sensitivity to action-relevant ones — yields control-relevant representations that survive distribution shift better than future-prediction tokens. LIBERO-Plus with its intervention API is the right testbed because it lets us construct evaluation episodes where nuisance factors (camera pose, lighting, distractors) are shifted while action-relevant dynamics are held fixed, so success rate under nuisance shift isolates exactly the property we claim to improve. PDRVLA tests the sub-claim that future-feature prediction alone does not separate nuisance from action-relevant variation; the standard VLA tests whether any world-model tokenization is needed; the nuisance-augmentation baseline tests whether our gain is merely from seeing shifted data rather than from the causal objective. The ablation (dropping L_inv) is the load-bearing test: if invariance is the mechanism, removing it should collapse the gain specifically on nuisance-shifted episodes while leaving clean episodes intact. The raw-sensitivity ablation tests whether the observation-difference normalization in L_sens is necessary to prevent nuisance co-variation from dominating the sensitivity term. The leakage probe directly measures the failure mode of the intervention-label completeness assumption: if action-relevant state can be predicted from nuisance-intervened observations, then L_inv will penalize genuinely action-relevant information. Success rate, rather than token-level metrics, is the right metric because the claim is about control outcomes, not representation geometry — a token can be invariant yet useless, and only downstream success reveals that.

### Expected outcome and causal chain

**vs. PDRVLA** — On an episode where the camera pose shifts mid-task, PDRVLA's future-feature token drifts because its supervision only rewards predicting the next observation feature, and camera-induced pixel change is a large fraction of that prediction target, so the policy misreaches. Our method instead suppresses this variation via L_inv because the simulator labels camera pose as a nuisance intervention and the squared-difference penalty directly pulls z_t^{nuis} toward z_t, so we expect a noticeable success gap on nuisance-shifted episodes but near-parity on clean ones. The leakage probe will quantify how much action-relevant state (e.g., gripper position) is predictable from nuisance-intervened observations; if leakage is high, we expect L_inv to also suppress gripper encoding, and the gap may shrink or reverse on episodes requiring precise gripper state.

**vs. Standard VLA (no world model)** — On a contact-rich manipulation where gripper-object contact state must be inferred from subtle visual cues, the plain VLA overfits spurious correlations between background texture and action because nothing forces its features to track action-relevant dynamics. Our method instead amplifies contact-relevant variation via L_sens because contact interventions are labeled action-relevant and the normalized sensitivity term rewards token change per unit observation change there, so we expect our advantage to concentrate on contact-sensitive subtasks and shrink on simple reach-and-grasp. If the raw-sensitivity ablation performs similarly, the normalization is not load-bearing; if it performs worse, normalization is necessary to prevent nuisance co-variation from dominating.

**vs. PDRVLA + nuisance augmentation** — On a distractor-object episode where an irrelevant object moves, the augmented baseline sees varied distractors during training but its token still encodes them because augmentation supplies no gradient that says "ignore this," only more data. Our method instead receives an explicit invariance gradient on exactly the distractor intervention, so we expect a modest but consistent gap on distractor-heavy episodes that persists even when augmentation matches our data volume, indicating the objective rather than the data is doing the work. The leakage probe will show whether distractor interventions also affect action-relevant state (e.g., by occluding the gripper); if so, the gap may be attenuated.

### What would falsify this idea

If the ablation dropping L_inv shows no degradation specifically on nuisance-shifted episodes — i.e., if our gain over PDRVLA is uniform across clean and shifted subsets rather than concentrated where the predicted failure mode occurs — then invariance is not the mechanism and the causal-sufficiency framing is wrong. Likewise, if L_sens removal costs nothing on contact-sensitive subtasks, the sensitivity claim is unsupported. Additionally, if the leakage probe reveals high predictability of action-relevant state from nuisance-intervened observations, the intervention-label completeness assumption is violated, and the method's gains may be due to confounding rather than causal sufficiency.

## References

1. PDRVLA: A Vision-Language-Action Model with Physical Dynamics Representations
2. BC-Z: Zero-Shot Task Generalization with Robotic Imitation Learning
3. Wan: Open and Advanced Large-Scale Video Generative Models
