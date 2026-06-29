# MoDE: Modular Dynamics Decomposition for Compositional In-Context Robotic Control

## Motivation

Current in-context learning methods for robotics, such as ICWM, rely on pre-training over diverse dynamics to generalize, but fail to extrapolate to novel combinations of dynamics components (e.g., new object physics or robot morphologies). This is because they treat the full system dynamics as a monolithic function, ignoring the underlying compositional structure where each physical entity (robot arm, object, surface) has independent internal dynamics that interact only through localized forces.

## Key Insight

The dynamics of robotic manipulation are inherently compositional due to localized physical interactions; by learning a separate latent module for each entity that captures its internal dynamics, and recombining them through a learnable interaction function, we enable zero-shot generalization to unseen entity combinations.

## Method

(A) **What it is**: MoDE (Modular Dynamics Decomposition) is a framework that learns to decompose the full system dynamics into independent modules, each corresponding to a physical entity, and then recombines them via a compositional bilinear interaction operator for in-context prediction. Input: a short history of observation-action pairs (context) and a set of entity masks (either given or learned via slot attention). Output: predicted next observation and action conditioned on recombined modules.

(B) **How it works**:
```python
# Pseudocode for MoDE training and inference
class MoDE:
    def __init__(self, num_slots, latent_dim):
        self.slot_attention = SlotAttention(num_slots, latent_dim)  # extracts per-entity latent states
        self.module_net = nn.ModuleList([MLP(latent_dim, latent_dim) for _ in range(num_slots)])  # each module is a small MLP
        self.interaction_fn = Bilinear(latent_dim, latent_dim)  # learned bilinear interaction
        self.prediction_head = nn.Linear(latent_dim, action_dim+obs_dim)

    def forward(self, context_obs, context_actions, entity_masks=None):
        # context_obs: (T, num_obs_features), context_actions: (T, num_action_features)
        # Step 1: Encode context into slot states using slot attention
        slots = self.slot_attention(context_obs.unsqueeze(0))  # (1, num_slots, latent_dim)
        # Step 2: Apply each module to its slot (temporal dynamics)
        per_module_dynamics = [self.module_net[i](slots[:, i]) for i in range(num_slots)]  # list of (1, latent_dim)
        # Step 3: Compute pairwise interactions via bilinear
        interaction_terms = []
        for i in range(num_slots):
            for j in range(num_slots):
                if i != j:
                    interaction_terms.append(self.interaction_fn(per_module_dynamics[i], per_module_dynamics[j]))
        # Step 4: Sum module dynamics and interactions, then predict
        aggregate = torch.sum(torch.stack(per_module_dynamics), dim=0) + torch.sum(torch.stack(interaction_terms), dim=0)
        prediction = self.prediction_head(aggregate)
        return prediction
```
Hyperparameters: num_slots typically equals number of distinct physical entities; latent_dim=256.

(C) **Why this design**: We chose slot attention over object-centric backbones (e.g., MONet) because it provides differentiable entity segmentation without ground-truth masks, accepting the cost that slot assignment may be noisy and require careful training. We chose mutual information minimization (implemented via contrastive learning on slot dynamics) over a simple reconstruction loss for module independence because mutual information gives a stronger statistical guarantee of disentanglement, though it increases computational cost and requires negative sampling. We chose bilinear interaction over graph neural networks because it is lightweight and supports variable number of entities without a fixed adjacency structure, albeit with less expressive interactions that may fail for complex non-local forces. We kept the prediction head as a single linear layer to avoid overfitting and maintain interpretability, but this limits expressivity for highly nonlinear dynamics.

(D) **Why it measures what we claim**: The mutual information between slot dynamics (approximated via contrastive loss) measures the degree of modular independence because we assume that physical entities have negligible influence on each other's internal dynamics except through sparse contacts; this assumption fails when entities are tightly coupled (e.g., two objects glued together), in which case the contrastive loss would be high even though modules are not independent, leading the model to over-regularize and degrade prediction. The entity reconstruction error (how well each module predicts future observations of its own slot) measures whether the module captures the entity's dynamics faithfully because we train each module to predict future states of its own entity; this fails if slot attention produces inaccurate masks, causing cross-contamination where a module receives information from other entities, making reconstruction error an unreliable indicator of module quality. The bilinear interaction term measures pairwise coupling strength because it captures the effect of one module's state on another's; this assumption is valid only if interactions are linear in the latent space, which may not hold for highly nonlinear physical forces.

## Contribution

(1) A novel framework, MoDE, that learns modular dynamics decomposition using slot attention and mutual information minimization to achieve disentangled per-entity dynamics modules. (2) A compositional bilinear interaction operator that enables zero-shot recombination of modules for novel dynamics combinations. (3) Empirical demonstration on simulated robotic tasks with unseen object physics and robot morphologies, showing significant improvement over monolithic in-context learning baselines.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Robosuite simulation tasks | Widely used, varied physical interactions |
| Primary metric | Task success rate | Directly measures downstream task performance |
| Baseline 1 | ICWM (world model in-context) | Tests need for explicit module decomposition |
| Baseline 2 | RICL (VLA in-context adaptation) | Tests benefit of modular vs. holistic adaptation |
| Baseline 3 | Vanilla VLA (no in-context) | Upper bound without adaptation |
| Ablation | MoDE w/o mutual info loss | Isolates effect of module independence |

### Why this setup validates the claim

This setup directly tests whether MoDE’s modular decomposition improves in-context adaptation over holistic world models (ICWM) and holistic VLA policies (RICL). The dataset of simulated robotic tasks with multiple interacting objects provides a natural testbed where modular independence should hold for distinct physical entities. Task success rate is chosen because it reflects both prediction accuracy and policy utility, and is the standard metric in the baselines. The ablation (removing mutual information loss) isolates the contribution of enforcing module independence, showing that the modular structure alone is not sufficient without disentanglement. If MoDE outperforms baselines primarily on scenarios with object contacts, it confirms that the bilinear interaction captures sparse coupling better than monolithic dynamics. Conversely, if gains are uniform, it suggests the modular design is not the driver. This experimental design forms a falsifiable test: the central claim predicts that MoDE’s advantage is largest when object interactions are sparse and modularity holds.

### Expected outcome and causal chain

**vs. ICWM** — On a case with two objects that interact only when they collide (e.g., pushing a block with a puck), ICWM tries to model the joint dynamics as a single high-dimensional system, requiring many context examples to capture all possible contact states. Its latent representation entangles the objects, so it must learn the interaction from data. Our method instead separates each object into its own module and only couples them via a bilinear interaction when contact occurs, so with few examples the modules already capture each object’s independent dynamics and the interaction is small. Thus we expect a noticeable gap on tasks with rare but important collisions, but parity on tasks with mostly independent objects.

**vs. RICL** — On a case where the robot must adapt to a new tool (e.g., a stick with different length) after only 5 demonstrations, RICL attempts to adapt the whole VLA policy via in-context examples, but the high-dimensional policy struggles to generalize the stick’s specific dynamics from few examples. Our method splits the stick as its own module, so the policy can reuse the pre-trained arm module and only the stick module needs adaptation. Since the stick module is low-dimensional, even 5 examples suffice. We expect MoDE to succeed where RICL fails on tasks requiring precise tool dynamics, but perform similarly on tasks with no new objects.

**vs. Vanilla VLA** — On any novel task, the vanilla VLA has zero-shot performance on unseen objects or dynamics because it lacks adaptation. Our method uses in-context history to infer the current objects’ behavior, so it should dramatically outperform vanilla VLA on any task that differs from training distribution. However, on in-distribution tasks where the VLA was heavily trained, vanilla may still be competitive. So we expect MoDE to show a large gap on out-of-distribution tasks but parity on in-distribution ones.

### What would falsify this idea
If MoDE’s gain over ICWM and RICL is uniform across all tasks, not concentrated on those with sparse interactions and distinct objects, that would indicate the modular decomposition and bilinear interaction are not capturing the promised mechanism, and the claim that modular independence is key would be wrong.

## References

1. In-Context World Modeling for Robotic Control
2. RICL: Adding In-Context Adaptability to Pre-Trained Vision-Language-Action Models
3. OpenVLA: An Open-Source Vision-Language-Action Model
4. π0: A Vision-Language-Action Flow Model for General Robot Control
5. Action Tokenizer Matters in In-Context Imitation Learning
6. RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control
7. VILA: On Pre-training for Visual Language Models
8. Vision-Language Foundation Models as Effective Robot Imitators
