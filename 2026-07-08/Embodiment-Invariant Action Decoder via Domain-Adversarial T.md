# Embodiment-Invariant Action Decoder via Domain-Adversarial Training for Zero-Shot Robot Transfer

## Motivation

Current vision-language-action (VLA) models, such as those trained on large-scale multi-embodiment data (From Foundation to Application, 2026), fail to generalize to unseen robot morphologies because their learned representations are implicitly tied to the training embodiments. Existing invariance mechanisms (e.g., HiMoE-VLA's structural invariance) are architectural but do not enforce representation-level invariance, leaving zero-shot transfer unsolved.

## Key Insight

Domain-adversarial training forces the shared representation to be uninformative of the robot identity, thereby distilling only aspects of the task that are common across embodiments, which enables generalization to new robots without finetuning.

## Method

We propose Embodiment-Invariant VLA (EI-VLA). It takes an image and language instruction, encodes them into a shared representation via a vision-language encoder, then decodes actions conditioned on a learnable embodiment embedding. Training jointly optimizes imitation learning and a domain-adversarial loss that encourages the shared representation to be invariant to robot identity.

```pseudocode
Input: Batch (image, instruction, action, robot_id) from K training embodiments.
Output: Updated model parameters.

Encoder f_enc: CNN+Transformer (pretrained, frozen except last 2 layers) -> shared representation z (dim=512).
Decoder f_dec: 4-layer MLP (hidden=1024, ReLU) taking (z, e) -> action (dim=action_space).
Domain classifier f_dom: 2-layer MLP (hidden=256, ReLU) -> softmax over K robot_ids.
Embodiment embedding table E: each training robot_id -> learnable vector e (dim=128).

# Gradient reversal layer (GRL): forward pass identity; backward multiply gradients by -lambda.
def backward(self, grad_output):
    return -lambda * grad_output

for each batch:
    z = f_enc(image, instruction)
    a_pred = f_dec(z, E[robot_id])
    L_act = MSE(a_pred, action)
    L_dom = CrossEntropy(GRL(f_dom(z)), robot_id)
    L_total = L_act + L_dom  # lambda is inside GRL
    backward and update all parameters (f_enc, f_dec, f_dom, E)

At test time for new robot R_new:
    Predict e_new = MLP_pred(kinematic_params) where MLP_pred is a 2-layer MLP (hidden=256, ReLU) trained on training robot's kinematic features (joint count, link lengths) to predict their learned embeddings.
    a_pred = f_dec(f_enc(image, instruction), e_new)
    (No fine-tuning)
```
Hyperparameters: lambda = 0.1 (tuned via grid search on a held-out validation embodiment), embedding dimension = 128.
**Load-bearing assumption:** The kinematic parameters alone are sufficient to predict the embodiment embedding. To verify, we compute the correlation between embedding prediction error (L2 distance between true and predicted embedding for training robots) and zero-shot success rate on unseen robots; a high negative correlation supports the assumption.

**(C) Why this design:** We chose domain-adversarial training (DANN) over contrastive methods because DANN directly maximizes the classifier's inability to predict the domain, providing a stronger invariance guarantee—the classifier is trained adversarially, so the encoder must suppress all domain-specific information to fool it. The trade-off is sensitivity to lambda; we fixed lambda=0.1 after a small sweep, accepting the risk of suboptimal balance for some embodiments. Second, we used learnable embodiment embeddings rather than handcrafted kinematic features because embeddings can capture unmodeled aspects (e.g., control latency, stiffness) that affect action generation. The cost is that for unseen robots, we must predict embeddings; we train a small MLP regressor on training robot's kinematic parameters to predict their learned embeddings, adding minor overhead but enabling zero-shot. Third, we used a single shared encoder for all embodiments rather than per-embodiment encoders to force representation sharing; this saves parameters but may lose embodiment-specific features. We compensate with the conditioned decoder, which consumes the embedding. This design differs from prior VLA works (OpenVLA, From Foundation to Application) that lack any invariance mechanism, and from sim-to-real DANN applications that focus on visual domain shift rather than embodiment morphology shift.

**(D) Why it measures what we claim:** The domain-adversarial loss L_dom measures embodiment invariance of z because it quantifies the classifier's ability to predict robot_id from z; when L_dom is high after gradient reversal (classifier performs poorly), z contains less robot-specific information. This relies on the assumption that a well-trained classifier's accuracy is a direct proxy for the amount of domain information in z; this assumption fails if the classifier is under-capacitated or not trained to convergence, in which case low accuracy may reflect classifier weakness rather than invariance. To validate this, we also directly measure mutual information I(z; robot_id) using a Kraskov k-nearest neighbor estimator (k=5) on held-out batches. The imitation loss L_act ensures that z retains task-relevant information; the assumption is that MSE on actions filters out irrelevant but domain-varying features, which holds if actions are determined only by the task and embodiment—if the task itself varies across embodiments, then invariance may discard useful signals. The embodiment embedding e captures remaining domain-specific action patterns; the assumption that e is necessary and sufficient for action adaptation is supported by the decoder structure, but fails if the embedding dimension is too small to encode complex morphology, leading to residual domain bias. Together, these components ensure that z is both task-relevant and domain-invariant, as evidenced by the ability to zero-shot transfer to new robots.

## Contribution

(1) We introduce domain-adversarial training to VLA models for learning embodiment-invariant representations, enabling zero-shot transfer to new robot morphologies. (2) We propose a novel embodiment-conditioned action decoder with learnable embeddings that can be predicted from kinematic parameters, allowing deployment without fine-tuning. (3) We provide a method for estimating embodiment embeddings for unseen robots from their kinematic properties, making the approach practically viable.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | GM-100 multi-robot benchmark (100 embodiments, 50 tasks each) | Tests zero-shot transfer across embodiments |
| Primary metric | Zero-shot success rate on unseen robot tasks (average over 5 runs) | Direct measure of embodiment invariance |
| Baseline 1 | OpenVLA (no invariance) | No adaptation mechanism |
| Baseline 2 | LingBot-VLA (fine-tuned per robot with 100 episodes) | Requires per-robot data |
| Baseline 3 | Sim-to-Real DANN (visual domain adaptation) | Addresses visual shift, not morphology |
| Ablation-of-ours | EI-VLA w/o domain-adversarial loss (only L_act) | No invariance constraint |
| Additional validation | Mutual information I(z; robot_id), centroid distance between z of different robots, correlation with classifier accuracy | Validates invariance proxy directly |

### Why this setup validates the claim

GM-100 provides multiple robot embodiments sharing task instructions, enabling controlled tests of zero-shot transfer. The primary metric—success rate on unseen robots—directly quantifies embodiment invariance because it measures whether shared representations generalize. OpenVLA tests the necessity of any invariance mechanism; its failure on novel embodiments would confirm the problem. LingBot-VLA tests whether fine-tuning (which implicitly captures embodiment-specific features) is sufficient; our method should match it on seen robots but surpass it on unseen when fine-tuning is impossible. Sim-to-Real DANN tests whether visual domain adaptation suffices; its failure highlights the unique challenge of morphology shift. Our ablation isolates the contribution of the adversarial loss: if it underperforms the full model, the invariance constraint is essential. Additionally, we validate the load-bearing assumption that kinematic parameters predict embodiment embeddings by measuring the correlation (Pearson's r) between embedding prediction error and zero-shot success across unseen robots; a strong negative correlation (r<-0.7) would confirm the assumption. Together, these components form a falsifiable test: if our method does not outperform all baselines on unseen embodiments, the central claim—that adversarial invariance enables zero-shot transfer—is unsupported.

### Expected outcome and causal chain

**vs. OpenVLA** — On a case where the training robot has 6-DOF arm and the unseen robot has 7-DOF, OpenVLA produces actions that treat the extra joint as fixed or irrelevant because its shared representation is contaminated by training embodiment details. Our method instead learns a representation that ignores embodiment-specific joint counts because the domain classifier forces it to suppress robot identity. So we expect a large gap (e.g., 30-40% absolute success rate) on unseen robots, but parity on seen robots.

**vs. LingBot-VLA** — On a case where the unseen robot has different control latency (e.g., slower response), LingBot-VLA would require fine-tuning to adapt its per-robot parameters; without fine-tuning, it fails catastrophically (near 0% success) because its actions are tuned to training dynamics. Our method uses a learned embodiment embedding predicted from kinematics; even if latency is not directly captured, the embedding can approximate it, yielding moderate success (e.g., 40-50%) without any adaptation data.

**vs. Sim-to-Real DANN** — On a case where the robot morphology is identical but the environment lighting changes (a visual shift), Sim-to-Real DANN succeeds (high invariance) because its domain classifier targets visual appearance. But on a case where the morphology differs (e.g., from Panda to Franka arm), Sim-to-Real DANN fails because its invariance is visual, not kinematic. Our method handles both shifts due to embodiment embeddings, so we expect our method to outperform on morphology-varying subsets (30% gain) while matching on visual-only subsets.

### What would falsify this idea

If our method shows no significant improvement over the "w/o adversarial loss" ablation on unseen embodiments, or if the gain on unseen robots is uniform across all test cases rather than concentrated on those with large morphology differences, then the invariance mechanism is not driving the improvement and the central claim is invalid. Additionally, if mutual information I(z; robot_id) does not correlate strongly (r<0.5) with domain classifier accuracy across embodiments, then the proxy measure is unreliable and the invariance claim weakens.

## References

1. From Foundation to Application: Improving VLA Models in Practice
2. Igniting VLMs toward the Embodied Space
3. FastUMI: A Scalable and Hardware-Independent Universal Manipulation Interface with Dataset
4. Robotic Control via Embodied Chain-of-Thought Reasoning
5. OpenVLA: An Open-Source Vision-Language-Action Model
6. GELLO: A General, Low-Cost, and Intuitive Teleoperation Framework for Robot Manipulators
7. Scalable. Intuitive Human to Robot Skill Transfer with Wearable Human Machine Interfaces: On Complex, Dexterous Tasks
8. VILA: On Pre-training for Visual Language Models
