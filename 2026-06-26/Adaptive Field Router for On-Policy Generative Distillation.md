# Adaptive Field Router for On-Policy Generative Distillation

## Motivation

DanceOPD composes multiple pre-defined capability fields via on-policy distillation but suffers from conflicting gradients when fields impose incompatible velocity updates on the same state. This conflict arises because field selection is fixed a priori, ignoring the student's current state and the interference between expert outputs. A learned router that dynamically selects fields per state can resolve these conflicts by steering the student toward updates that maximize alignment with the dominant expert's direction.

## Key Insight

Conflicting gradients between expert fields can be resolved by learning a per-state routing policy that selects fields whose velocity fields are most aligned with the student's current gradient direction, thereby minimizing a global measure of interference.

## Method

AFROPD (Adaptive Field Router for On-Policy Distillation) is a training framework that augments DanceOPD with a neural router that, given a meta-feature embedding of a state, outputs a categorical distribution over available expert fields. The student is trained by sampling one field per state according to the router, then performing on-policy velocity matching. The router is simultaneously trained to maximize the student's learning efficiency measured by the reduction in a gradient-conflict surrogate.

**(B) How it works**
```pseudocode
Input: Expert fields {f_k}, student initial parameters θ, router network R_φ (2-layer MLP, hidden=256, ReLU, output dim=num_fields), buffer B
Hyperparameters: β=1e-4 (router learning rate), γ=0.1 (entropy regularization coefficient), τ=1.0 (softmax temperature), batch size=64, optimizer=Adam for both θ and φ

for each training step:
   Sample batch of states s = (x, t, c) from student's on-policy distribution  # x: noisy sample, t: time, c: condition
   For each s, compute meta-feature φ(s) = [noise_level (from t), text_embedding (CLIP), student_gradient_norm (L2 norm of ∇_θ L_selected)]
   Get router logits a_k = R_φ(s)_k, then probabilities p_k = softmax(a_k / τ)
   Sample expert field k ~ Categorical(p_k)
   Compute target velocity v_target = f_k(s)
   Compute student prediction v_θ(s)
   Update θ via ∇_θ ||v_θ(s) - v_target||^2  (standard on-policy MSE)
   
   # Router update: minimize gradient conflict
   For all k, compute surrogate conflict C_k = ||∇_θ L_k(s) - ∇_θ L_selected(s)||^2, where L_k(s) = ||v_θ(s) - f_k(s)||^2
   Router loss L_R = Σ_k p_k * C_k + γ * H(p)   # H is entropy
   Update φ via ∇_φ L_R
   
   Optionally: store (s, p_k, v_target) in B for offline replay (buffer size=10000)
   
   # Verification: every 500 steps, compute average cosine similarity between ∇_θ L_k and ∇_θ L_selected on a held-out calibration set of 100 states. If similarity < 0.3, multiply τ by 0.9 to encourage exploration.
```

**(C) Why this design**
We chose a neural router with a categorical output over a deterministic rule because the optimal field depends on complex interactions between state and expert behaviors that cannot be hand-specified. The router is trained to minimize a gradient-conflict surrogate C_k, which directly quantifies how much selecting a field would misalign the student's update with the other fields. This contrasts with simply minimizing the prediction error on each field, which would not penalize interference. We use on-policy sampling from the router rather than the true multi-field mixture (as in DanceOPD) because it allows the router to specialize: it learns to select fields where the student currently makes most progress. The entropy regularization encourages exploration, accepting a short-term cost of non-optimal selections to avoid premature convergence. We set τ as a temperature to control exploration sharpness; lower τ encourages deterministic routing. The meta-feature includes student gradient norm because high-gradient regions signal where the student is uncertain, which should influence routing. This design trades off complexity (training an additional network) for improved multi-capability composition without manual tuning. The approach does not assume any dependence between expert fields beyond their shared state space, so it generalizes to arbitrary field sets.  

**(D) Why it measures what we claim**
The router's selection probability p_k measures the expected reduction in gradient conflict because p_k is proportional to exp(-C_k/τ), where C_k = ||∇L_k - ∇L_selected||^2, assuming the gradient directions of different fields are conditionally independent given the state; this assumption fails when fields are strongly correlated (e.g., text-to-image and editing on the same concept), in which case p_k reflects cooperation rather than conflict reduction. The router loss L_R minimizes the expected conflict across fields, which directly quantifies the interference we aim to reduce, under the assumption that the student's gradient is a sufficient statistic for learning progress; this assumption fails when gradient norm is dominated by noise, in which case the router may chase spurious patterns. The entropy term H(p) measures the explorer's uncertainty, preventing overcommitment to a single field; it operationalizes the need to maintain diverse field exposure, but fails when the optimal policy is deterministic, in which case entropy regularization distorts the trade-off. The periodic verification on a held-out calibration set (cosine similarity threshold) ensures that when the conditional independence assumption breaks, the router's temperature is adjusted to maintain exploration, thus mitigating the failure mode.

## Contribution

(1) A learned adaptive router that selects expert fields per state during on-policy distillation, replacing manual field scheduling. (2) A training objective that minimizes gradient conflict between multiple expert fields by optimizing a surrogate of interference. (3) A demonstration that the router improves multi-capability composition compared to fixed field selection in a flow-matching generative model.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | MultiCapBench (50k text-to-image from COCO, 50k local editing from OpenImages, 50k inpainting from Places2) | Tests multi-capability composition across diverse domains |
| Primary metric | Composite perceptual score = 0.5*FID + 0.3*LPIPS + 0.2*CLIP score | Captures quality across tasks; weights calibrated on a validation set |
| Baseline 1 | DanceOPD (fixed mixture of 1/3 per field) | No adaptive routing; fixed mixture |
| Baseline 2 | Off-policy distillation (static fields on teacher samples) | Ignores student state dynamics |
| Baseline 3 | Single specialist per field (one model per task) | Upper bound for isolated tasks |
| Baseline 4 | MoE-based distillation (soft gating over fields via 2-layer MLP) | Compares to other learned routing |
| Ablation-of-ours | AFROPD without conflict loss (L_R = H(p) only) | Tests necessity of gradient conflict minimization |
| Ablation-of-ours | AFROPD with meta-feature ablation (remove one component: noise_level, text_embedding, gradient_norm) | Tests contribution of each meta-feature |

### Why this setup validates the claim
The chosen dataset, MultiCapBench, combines text-to-image, local editing, and inpainting tasks that require distinct expert behaviors, creating natural conflicts that test the router's ability to select the most beneficial field per state. The composite perceptual metric (weighted FID, LPIPS, CLIP) ensures we measure overall capability composition rather than single-task performance. Baseline DanceOPD tests whether adaptive routing improves over a fixed multi-field mixture; Off-policy distillation tests the necessity of on-policy state dependence; Single specialists provide per-task upper bounds and highlight the need for composition; MoE-based distillation compares to an alternative learned routing method. The ablation omits the gradient-conflict surrogate to isolate its contribution, and the meta-feature ablation tests the necessity of each input. If our central claim holds—that reducing gradient conflict via adaptive routing enhances multi-capability learning—then our method should outperform baselines particularly on states where fields diverge, and the ablation should degrade to near-baseline levels. This design thus provides a direct falsifiable test of the mechanism.

### Expected outcome and causal chain

**vs. DanceOPD** — On a case where the student is initially poor at editing but proficient at text-to-image, DanceOPD's fixed mixture allocates equal gradient updates to both fields, causing the editing errors to dominate and slow overall progress. Our method instead uses the router to select the text-to-image field more often, because its gradient is less conflicting with the student's current state, accelerating editing learning through indirect reuse. We expect a 5-10% improvement on editing tasks but near parity (±1%) on text-to-image tasks.

**vs. Off-policy distillation** — On a case where the student's on-policy generation deviates from the teacher's data distribution, off-policy distillation applies fields based on static teacher samples that are irrelevant to the student's current mistakes, leading to harmful gradient updates. Our method's on-policy router selects fields that align with the student's actual trajectory, mitigating such mismatches. We expect lower training variance (30% reduction in std of composite score across seeds) and a 3-5% higher composite score on all tasks, especially during early training (first 25% of steps).

**vs. Single specialist per field** — On a combined task like editing a text-generated image, a single specialist model cannot jointly handle both capabilities, resulting in poor quality on either aspect. Our method routes between text-to-image and editing fields per state, enabling coherent composition. We expect our method to match the single specialist on isolated tasks (within 1%) but outperform by 10-15% on composite tasks.

**vs. MoE-based distillation** — The MoE baseline uses soft gating that mixes all fields per state, which can still cause gradient conflict. Our method's discrete selection and conflict loss should yield 2-4% higher composite score and better task separation, especially on conflicting states.

**Ablation without conflict loss** — Without C_k, the router minimizes only entropy, degenerating to near-uniform selection, thus performing similarly to DanceOPD (within 1%).

**Meta-feature ablation** — Removing gradient_norm degrades performance by 2-3%; removing text_embedding or noise_level degrades by 1-2%, indicating gradient_norm is most informative.

### What would falsify this idea
If our method's advantage over DanceOPD is uniform across all tasks rather than concentrated on conflicting states (e.g., no difference in editing improvement vs. text improvement), or if the ablation without gradient-conflict loss performs comparably (within 1%), then the claim that adaptive routing via conflict reduction is the driver is falsified.

## References

1. DanceOPD: On-Policy Generative Field Distillation
2. On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes
3. MultiDiffusion: Fusing Diffusion Paths for Controlled Image Generation
4. Stochastic Interpolants: A Unifying Framework for Flows and Diffusions
5. Large Language Models Are Reasoning Teachers
6. SpaText: Spatio-Textual Representation for Controllable Image Generation
7. Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow
8. Building Normalizing Flows with Stochastic Interpolants
