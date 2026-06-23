# SELF-ROLE: Self-Supervised Role Identity Inference for Consistent Role-Playing Agents

## Motivation

Existing role-playing agents rely on explicit character profiles as static inputs, which are costly to create and limit adaptability (e.g., Thinking in Character uses profile-based reasoning; Character-LLM fine-tunes on curated profiles). This reliance stems from the assumption that role identity must be provided externally. We argue that role identity can be inferred from the agent's own interaction history, enabling self-supervised role acquisition without profiles.

## Key Insight

Temporal consistency of agent behavior under a fixed role provides a natural supervisory signal: the latent role representation should be invariant across different interaction episodes of the same role.

## Method

### SELF-ROLE: Self-Supervised Role Identity Inference for Consistent Role-Playing Agents

(A) **What it is**: SELF-ROLE is a self-supervised learning framework that trains a role encoder to extract a consistent role embedding from interaction history, and a policy network that uses this embedding to generate role-consistent responses. Input: a sequence of dialogue turns (the history of a single agent). Output: a role embedding vector and generated response. The critical load-bearing assumption is that within a single role-playing episode, role identity is the only invariant factor across different segments of the same character; all other contextual factors vary sufficiently.

(B) **How it works** (pseudocode):
```python
# Training loop
f_θ = RoleEncoder()          # role encoder parameters (Transformer, 12 layers, hidden=768)
f_θ_target = copy(f_θ)       # target network (momentum updated)
π_φ = PolicyNetwork()        # response generator (autoregressive LM, e.g., GPT-2 small)

for batch in dataloader:     # each episode is a dialogue from one role
    # Apply dropout augmentation to input history (retain only role identity)
    # Use SimCSE-style: two forward passes with different dropout masks (dropout rate p=0.1)
    h1 = f_θ(dropout(episode))  # dropout mask 1
    h2 = f_θ_target(dropout(episode))  # dropout mask 2, target network
    z1 = mean_pool(h1)  # pool token embeddings to fixed-size vector
    z2 = mean_pool(h2)
    # Contrastive loss (InfoNCE)
    sim = cosine_similarity(z1, z2) / τ   # τ=0.07
    logits = concat(sim, sim_all_negatives_in_batch)  # negatives from other roles in batch
    loss_contra = cross_entropy(logits, positive_idx=0)
    # Update encoder (target network via momentum)
    loss_contra.backward()
    update f_θ (Adam, lr=1e-5)
    f_θ_target = α * f_θ_target + (1-α) * f_θ   # α=0.999

# Inference: given history h, compute z = f_θ(h), then y = π_φ(z, h)
```
Hyperparameters: τ=0.07, α=0.999, d=256 (embedding dimension), dropout rate p=0.1, batch size=64, encoder learning rate=1e-5, policy learning rate=3e-5.

(C) **Why this design**: We chose a contrastive objective over autoencoding because role identity is defined by invariance across contexts rather than reconstruction; autoencoding could collapse to low-level surface patterns. We use a target network with momentum updates to prevent representation collapse, a known issue in self-supervised learning (Chen et al., 2020), accepting the cost of slightly slower convergence. We replace the original within-episode segmentation with dropout augmentation (as in SimCSE) because simple splitting may lead to trivial shortcuts: the encoder could exploit surface-level overlap rather than role identity. Dropout introduces stochastic noise that varies context while preserving the latent role signal. This design avoids the need for multi-episode data per role and ensures positive pairs differ in context by construction. The contrastive loss uses negatives from other roles in the batch, which encourages the embedding to be discriminative across roles. The trade-off is that batch size must be large enough to have diverse negatives (here 64).

(D) **Why it measures what we claim**: The contrastive loss L_contra encourages the role embedding z to be invariant across different augmented views of the same role's history. Specifically, the cosine similarity between embeddings from two dropout views of the same role is maximized, while similarity to other roles is minimized. This operationalizes 'role consistency' as the temporal stability of the representation under controlled augmentation; the underlying assumption is that role identity is the only shared factor across views while context is perturbed by dropout. This assumption fails when the dropout noise inadvertently removes role-relevant information (e.g., key personality markers), in which case the embedding may capture spurious correlations instead. The contrastive loss measures role identity only if the only shared factor across views is role. If context shifts systematically (e.g., topic changes within the episode), the augmentation may not sufficiently vary context, and the embedding could capture extraneous features. To verify, we add an intervention test: for a fixed role, we generate two histories by swapping context segments between different episodes of that role; the embedding should remain consistent. Additionally, we use a synthetic dataset with known role vectors to directly measure embedding fidelity.

## Contribution

(1) A self-supervised learning framework, SELF-ROLE, that infers role identity from interaction history without explicit profiles, using temporal consistency as the supervisory signal. (2) A demonstration that role consistency can be achieved through representation invariance across contexts, eliminating the need for costly profile engineering.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|-----------------------|
| Dataset | LIGHT (Urbanek et al., 2019) + Synthetic (controlled role vectors) | LIGHT: multi-turn role-playing; Synthetic: known ground truth role IDs |
| Primary metric | Role consistency (ROCO) score + Embedding-to-role correlation (for synthetic) | ROCO: embedding coherence across views; correlation: direct measure of identity encoding |
| Baseline 1 | Standard RPA (no reasoning) | Generates responses without role memory. |
| Baseline 2 | CoT RPA (chain-of-thought) | Adds reasoning but no learned role representation. |
| Baseline 3 | Direct LRM character sim | Models character as a fixed persona vector. |
| Baseline 4 | SimCLR-style (instance discrimination) | Same augmentation but contrastive between whole episodes (no role-specific objective) |
| Ablation | SELF-ROLE w/o target net | Uses same encoder without momentum update. |

### Why this setup validates the claim
We choose LIGHT because it provides long-term dialogues with stable character identities, ideal for testing role consistency. The synthetic dataset (512 roles, each with 20 episodes, role vectors drawn from a 256-dim Gaussian) allows us to compute the Pearson correlation between learned embeddings and ground-truth role vectors, directly verifying that the embedding captures role identity rather than context. The ROCO metric measures the invariance of the learned role embedding across dropout views of the same character, which is the central claim. The baselines isolate different failure modes: Standard RPA lacks a persistent role representation, CoT RPA relies on explicit reasoning but no learned embedding, Direct LRM uses a static persona that may not adapt to context, and SimCLR-style tests whether role-level contrast is superior to instance-level discrimination. Our ablation tests whether the target network is necessary for preventing collapse. This combination ensures that if our method outperforms, it must be due to the self-supervised learning of dynamic role embeddings with role-level contrastive objective, because the only extra component is the contrastive loss with momentum updates and role-aware negative sampling. We also include an intervention test: for each role, we swap the roles of two dialogue segments (i.e., take context from role A but assign role B's embedding) and measure a drop in ROCO, confirming that the embedding responds to role identity, not context.

### Expected outcome and causal chain
**vs. Standard RPA** — On a case where the dialogue history contains a long context with multiple topics, the standard RPA generates responses that drift away from the initial role because it has no persistent identity. Our method extracts a role embedding from the entire history, ensuring consistent behavior. We expect a significant gap on long conversations (e.g., >20 turns) but parity on very short ones. On the synthetic dataset, our method should achieve correlation >0.8 with the true role vector, while Standard RPA has no embedding and thus 0 correlation.

**vs. CoT RPA** — On a case where the role's personality is subtly conveyed (e.g., sarcasm), CoT may misinterpret because its reasoning is prompted but not learned from data. Our method learns the representation from within-episode consistency across augmentations, so it captures implicit patterns. We expect higher ROCO on sarcastic or polite roles, with a gap of at least 0.15 in similarity on LIGHT, and higher correlation on synthetic roles with subtle traits.

**vs. Direct LRM** — On a case where the same role acts differently across episodes due to plot progression, a fixed persona fails; our method adapts by learning episode-specific embeddings that are invariant to augmentation. We expect our method to maintain high ROCO across episodes, while Direct LRM's static vector causes lower scores. On synthetic, the fixed-persona baseline cannot model role changes, achieving lower correlation.

**vs. SimCLR-style** — On a case where distinguishing roles requires noticing subtle differences between similar roles, instance-level contrast (treating each episode as a class) may confuse episodes from the same role. Our role-level contrast groups episodes by role, leading to higher ROCO and better role separation. We expect a gap of 0.1 in ROCO on LIGHT and a higher correlation on synthetic.

### What would falsify this idea
If our method's ROCO scores are not significantly higher than the ablation (without target network) on all subsets, or if the gains are uniform across baselines rather than concentrated on long contexts or implicit traits, the central claim that contrastive learning with target network enables context-invariant role embeddings is not supported. Additionally, if on synthetic data the embedding correlation is below 0.6, it would indicate the method fails to capture the intended role identity.

## References

1. Thinking in Character: Advancing Role-Playing Agents with Role-Aware Reasoning
2. LLMs + Persona-Plug = Personalized LLMs
3. IBSEN: Director-Actor Agent Collaboration for Controllable and Interactive Drama Script Generation
4. CAMEL: Communicative Agents for "Mind" Exploration of Large Language Model Society
5. Character-LLM: A Trainable Agent for Role-Playing
6. Generative Agents: Interactive Simulacra of Human Behavior
7. LARP: Language-Agent Role Play for Open-World Games
8. MetaAgents: Large Language Model Based Agents for Decision-Making on Teaming
