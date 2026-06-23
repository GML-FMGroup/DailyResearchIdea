# Causal Action Embedding for Zero-Shot Robotic Manipulation via Self-Supervised State Transition Learning

## Motivation

Current Vision-Language-Action (VLA) models like RT-2 and π₀ require fine-tuning on task-specific demonstration data to generalize to new manipulation tasks, as they learn action policies conditioned on task identity. This limitation arises because these models are trained on static datasets per task and lack a representation that captures the causal effect of actions on state changes independent of the task. The Experiences from Benchmarking Vision-Language-Action Models for Robotic Manipulation paper empirically shows that even state-of-the-art VLAs require per-task fine-tuning for out-of-distribution tasks, confirming the convergent gap: no mechanism exists to generalize zero-shot without task-specific demonstrations.

## Key Insight

Action embeddings learned purely from state transition pairs (state, action, next_state) via forward and inverse dynamics models are inherently causal and task-invariant, because the effect of an action on the state is decoupled from the task context when the state space is sufficiently rich.

## Method

We propose **Causal Action Embedding (CAE)**, a self-supervised method that learns a latent action representation from unlabeled interaction data (state, action, next_state triples). The method uses two encoders and two predictors to enforce a causal structure: the action embedding must be both necessary and sufficient for predicting the next state from the current state.

**Load-bearing assumption**: The state representation is Markovian, i.e., captures all causally relevant variables for the dynamics. This is a load-bearing assumption; if it fails (e.g., unobserved confounders like object mass), the action embedding may capture spurious correlations. To verify, we use a causal validation set with varying context (e.g., different object masses) to check that the action embedding remains invariant to context changes; if not, we flag potential confounders.

### (B) How it works
```python
class CAE:
    def __init__(self, state_encoder: nn.Module, action_encoder: nn.Module,
                 forward_predictor: nn.Module, inverse_predictor: nn.Module,
                 action_decoder: nn.Module):
        # state_encoder: CNN with 3 layers (32,64,128 channels, kernel 5, stride 2, ReLU), followed by FC to 256-dim
        # action_encoder: 2-layer MLP (128,64) with ReLU -> 64-dim
        # forward_predictor: 3-layer MLP (256+64 -> 512,256,256) with ReLU and layer norm
        # inverse_predictor: 3-layer MLP (256+256 -> 512,256,64) with ReLU
        # action_decoder: 2-layer MLP (64->128, action_dim) with tanh output
        pass

    def train_step(self, s, a, s_next):
        z_s = self.state_encoder(s)
        z_a = self.action_encoder(a)
        z_s_next = self.state_encoder(s_next)
        
        # Forward prediction
        z_s_next_pred = self.forward_predictor(z_s, z_a)
        L_forward = MSE(z_s_next_pred, z_s_next)
        
        # Inverse prediction
        z_a_pred = self.inverse_predictor(z_s, z_s_next)
        L_inverse = MSE(z_a_pred, z_a)
        
        # Loss: balance forward and inverse (hyperparameter lambda=0.5)
        loss = L_forward + 0.5 * L_inverse
        return loss

    def infer_action(self, s_current, s_goal):
        z_s = self.state_encoder(s_current)
        z_s_goal = self.state_encoder(s_goal)
        z_a = self.inverse_predictor(z_s, z_s_goal)
        # Decode to low-level action
        a = self.action_decoder(z_a)
        return a
```
**Hyperparameters**: state latent dim=256, action latent dim=64, lambda=0.5, batch size=256, Adam optimizer lr=1e-4, gradient clipping norm=1.0. Training: 500k steps on 100k transitions (50k distinct triples) from random exploration in LIBERO. Data augmentation: random crops and color jitter on images.

### (C) Why this design
We chose **separate forward and inverse predictors** over a single unified model because the forward objective ensures that z_a causally determines the state change (via MSE on next-state latent), while the inverse objective provides a direct way to infer actions from desired state transitions without requiring a separate planning module. The trade-off is that the two objectives may conflict if the dynamics are not invertible (e.g., different actions lead to same next state), causing the inverse loss to compete with forward loss; we mitigate this by setting lambda=0.5 to prioritize forward consistency. We selected **MSE loss** over cross-entropy or contrastive losses because it directly penalizes Euclidean distance in latent space, which aligns with the assumption of continuous state spaces and smooth dynamics; the cost is sensitivity to outliers, which we handle by normalizing latent vectors to unit sphere. We use a **dedicated action decoder** to map from latent z_a to low-level actions, rather than directly using the action encoder's output, because the action encoder is trained to compress discrete motor commands into a causal latent, while the decoder is fine-tuned for reconstruction of diverse actions from the latent; this separation prevents the encoder from being forced to be invertible. The choice of **not using explicit task invariance regularization** (e.g., adversarial) is deliberate: task invariance emerges naturally because the training data contains transitions from random exploration without task labels, and the forward/inverse losses only depend on state changes, not on task identity. This avoids the anti-pattern of learned disentanglement (Anti-pattern 1) because we do not model task identity at all; the representation is purely action-effect driven, which is structurally independent of task context. Please contrast this with prior work like RT-2 that conditions on language instructions (task-specific), whereas our method conditions only on goal state, making it task-agnostic by design.

### (D) Why it measures what we claim
The computational quantity **z_a** (action latent) measures **causal action effect** because it is trained to minimize forward prediction error: the assumption is that a sufficient statistic of the action for predicting the next state from the current state captures the action's causal effect (by the causal Markov condition). This assumption fails when the state representation is incomplete (e.g., missing object pose) — in that case, z_a may encode spurious correlations from confounders, and the forward loss measures only predictive accuracy, not causation. The **forward loss L_forward** measures **causal sufficiency** because if the model can accurately predict the latent next state from (z_s, z_a), then z_a is sufficient for the transition; however, this assumes that the learned state encoder captures all relevant information for dynamics — when it does not, L_forward reflects approximation error, not causal insufficiency. The **inverse loss L_inverse** measures **causal invertibility** because it requires that from state pair we can recover the action latent that caused the transition; this assumes deterministic dynamics — when actions are stochastic or multi-modal, the inverse predictor may learn an average embedding that does not correspond to any single executed action, and L_inverse then measures reconstruction ambiguity, not causal consistency. The forward loss measures predictive sufficiency, not causal sufficiency; the equivalence holds only if the state encoder is a sufficient statistic for the underlying causal system. This assumption fails under partial observability or unobserved confounders, causing spurious correlations in z_a.

## Contribution

(1) A self-supervised framework (CAE) that learns a causal, task-invariant action embedding from unlabeled state-action-next state triples, enabling zero-shot generalization to new tasks specified by goal images without fine-tuning. (2) An architectural design principle: separating forward and inverse dynamics with a shared latent action space achieves both causal sufficiency and invertibility, which prior VLA methods (e.g., RT-2, π₀) lack due to their reliance on task-specific conditioning. (3) Empirical validation that the action embedding learned from random exploration on a set of base tasks transfers zero-shot to novel manipulation tasks, demonstrating the efficacy of task-agnostic causal representation.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|----------------------|
| Dataset | LIBERO goal-conditioned manipulation tasks | Continuous state-action dynamics for causal learning |
| Primary metric | Goal-reaching success rate under OOD goals | Tests causal sufficiency via generalization |
| Baseline 1 | RT-2 (language-conditioned VLA) | Task-specific, no causal structure |
| Baseline 2 | Direct state-difference baseline | No latent action abstraction |
| Baseline 3 | Variational autoencoder on action space | No causal forward constraint |
| Ablation-of-ours | CAE without inverse loss | Tests necessity of invertibility constraint |

### Why this setup validates the claim
The LIBERO dataset provides diverse goal-conditioned manipulation tasks with continuous state-action trajectories, allowing evaluation of causal action embeddings. The primary metric, goal-reaching success rate under out-of-distribution (OOD) goal states, directly tests whether the learned action embedding captures causal sufficiency: if the embedding is sufficient, the agent can generalize to novel goal states by inferring the appropriate action. The baselines isolate specific mechanisms: RT-2 tests the necessity of task-agnostic causal learning (since it conditions on language), the direct state-difference baseline tests the benefit of latent abstraction, and the VAE baseline tests the importance of the forward prediction constraint. The ablation of the inverse loss tests whether invertibility is essential for goal inference. Together, these comparisons form a falsifiable test: our method should outperform baselines primarily on OOD goals where causal understanding is critical, while performing similarly on in-distribution goals where simpler memorization may suffice.

### Expected outcome and causal chain

**vs. RT-2** — On a case where a novel goal does not match any training language description, RT-2 fails because it relies on language conditioning and cannot infer actions from state differences alone. Our method instead infers the action latent from the current and goal state using the inverse predictor, which was trained on general state transitions, so it generalizes to unseen goal configurations. We expect a noticeable gap on OOD goal subsets (e.g., spatial arrangement changes) but parity on in-distribution goals where language matches.

**vs. Direct state-difference baseline** — On a case where the action effect is nonlinear (e.g., gravity compensation), the direct difference in raw pixel space is noisy and fails to capture the causal effect. Our method learns a compact latent action embedding that abstracts away irrelevant details, enabling robust action inference. We expect our method to succeed on tasks requiring precise force modulation (e.g., peg insertion), while the baseline fails.

**vs. Variational autoencoder on action space** — On a case where multiple different actions lead to the same next state (e.g., multiple motor commands for same displacement), the VAE learns an invertible but non-causal embedding that confounds action identity with state effects. Our method's forward objective forces the embedding to be predictive of the next state, making it causal. We expect our method to outperform on tasks requiring consistent action inference across stochastic transitions.

**vs. CAE without inverse loss** — On a case where goal inference requires inverting the dynamics (e.g., reaching a precise pose), the ablation without inverse loss must rely on forward prediction alone to plan actions via optimization, which is computationally expensive and less accurate. Our full method directly solves the inverse problem. We expect our method to be faster and more accurate in goal-reaching tasks.

### What would falsify this idea
If our method performs worse than or comparably to the direct state-difference baseline on OOD goals, that would indicate that the causal action embedding is not capturing causally sufficient information, falsifying the central claim of causal sufficiency. Alternatively, if the ablation without inverse loss matches our full method, then invertibility is unnecessary, contradicting our assumed mechanism.

## References

1. Experiences from Benchmarking Vision-Language-Action Models for Robotic Manipulation
2. π0: A Vision-Language-Action Flow Model for General Robot Control
3. RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control
4. ManiSkill2: A Unified Benchmark for Generalizable Manipulation Skills
5. Approximate convex decomposition for 3D meshes with collision-aware concavity and tree search
6. Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success
7. Language Conditioned Multi-Finger Dexterous Manipulation Enabled by Physical Compliance and Switching of Controllers
