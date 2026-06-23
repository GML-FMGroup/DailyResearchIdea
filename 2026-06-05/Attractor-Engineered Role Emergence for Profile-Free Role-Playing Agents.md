# Attractor-Engineered Role Emergence for Profile-Free Role-Playing Agents

## Motivation

Current role-playing agents (e.g., RAR, IBSEN) rely on explicit character profiles prepended to every input, creating a bottleneck in scenarios where profiles are unavailable or costly to construct. This external dependency prevents the model from autonomously internalizing role identity, limiting scalability and naturalness. We address this by eliminating profile input entirely, requiring the model to self-activate role identity from context alone.

## Key Insight

Role-consistent behavior emerges from the model's intrinsic dynamics when its hidden state attractors are shaped during training to correspond to low-dimensional role-specific subspaces, making profile input unnecessary.

## Method

### (A) What it is:
AERE (Attractor-Engineered Role Emergence) is a training method that shapes the hidden state space of a language model to contain role-specific attractors. Input: a base language model (e.g., GPT-2 small with 12 layers, 124M parameters) and a set of role demonstration dialogues (without explicit profiles) from Character-LLM (10 roles, each with 1000 dialogues). Output: a model that, given a situation context, autonomously enters a hidden state attractor corresponding to the intended role, producing role-consistent responses without any profile prefix.

### (B) How it works:
```python
# Pseudocode for AERE training on one role
for each role r:
    # Collect role-specific dialogues (situation, response) without profiles
    dialogues = load_role_dialogues(r)  # 1000 dialogues per role
    
    # Step 1: Compute role attractor subspace via PCA on hidden states
    hidden_states = []
    for (situation, response) in dialogues:
        # Run model on situation + response, get last-layer hidden state of last token
        h = model.forward(situation + response).hidden_states[-1][-1, :]  # shape (768,)
        hidden_states.append(h)
    hidden_states = stack(hidden_states)   # shape (1000, 768)
    P = PCA(n_components=64).fit(hidden_states).components_   # subspace projection matrix (64, 768)
    
    # Step 2: Fine-tune with dual objective
    optimizer = Adam(model.parameters(), lr=1e-5)
    for batch in dialogues (batch_size=16):
        situations, responses = batch
        logits, hidden = model(situations + responses, output_hidden_states=True)
        h_last = hidden[-1][-1, :]   # last token hidden state (768,)
        
        # Language modeling loss (next-token prediction on responses)
        lm_loss = cross_entropy(logits, targets)
        
        # Attractor loss: penalize component orthogonal to subspace
        h_proj = h_last @ P.T @ P   # project onto subspace
        attractor_loss = ((h_last - h_proj) ** 2).sum(dim=-1).mean()  # L2 distance
        
        # Separation loss (only during multi-role joint training): maximize inter-subspace distance
        # For simplicity, assume we train all roles jointly; separation loss uses pairwise cosine similarity of subspace centers
        # But we pretrain individual subspaces first, then joint fine-tuning
        # separation_loss = -sum_{i<j} cos_sim(center_i, center_j) where center_i = mean of projected h_last for role i
        # We use beta=0.01
        
        total_loss = lm_loss + alpha * attractor_loss + beta * separation_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

# Hyperparameters: k=64 (subspace dimension), alpha=0.1, beta=0.01, lr=1e-5, batch_size=16, epochs=5
# Calibration: after training, evaluate attractor loss on held-out 512 dialogues per role to verify subspace alignment.
# Verification: compute average attractor loss on calibration set; expect low loss (<0.1) for trained roles.
```

### (C) Why this design:
We chose low-dimensional subspaces over fixed-point attractors because natural language generation requires flexibility within a role—a subspace allows variation while maintaining consistency; a fixed point would be too restrictive, causing identical responses. We chose PCA-initialized subspaces from role data rather than random initialization to ensure the subspace captures actual role-specific patterns; random initialization would require longer convergence and risk degenerate attractors. We applied the attractor loss only at the last token rather than per-token to avoid over-constraining the dynamics of intermediate tokens—this trades off some fine-grained control for training stability and preserves the model's ability to adapt to context mid-sequence. Separating attractors via a divergence loss (maximizing inter-subspace distance) prevents role confusion and encourages distinct attractors, but adds a hyperparameter that must be tuned to avoid pushing attractors too far apart (which could degrade language modeling). The dual loss (LM + attractor) is crucial: without LM loss, the model could satisfy the attractor at the expense of fluent generation; without attractor loss, role consistency is not enforced.

### (D) Why it measures what we claim:
The attractor loss L_att = ||(I-P)h_last||^2 measures the *degree of role identity activation* because it quantifies how much the model's final hidden state deviates from the characteristic subspace of the role; the underlying assumption is that role-consistent behavior corresponds to hidden states lying on that subspace. **Explicit linking assumption:** L_att measures role identity activation assuming subspace P captures all role-consistent states. Failure case: subspace may encode topic instead of role (e.g., medical vocabulary) when dialogues are homogeneous. This assumption fails when the subspace is learned from insufficient or noisy data, in which case L_att may measure overfitting to noise rather than true role identity. The subspace dimensionality k controls the *specificity* of role activation: smaller k enforces stricter adherence but risks collapsing variation, while larger k allows flexibility but may dilute the signal. The projection onto P is equivalent to a soft role activation signal—the model is incentivized to produce hidden states that align with the role's subspace; this operationalizes the concept of *self-activating role identity* by making the attractor an implicit, learned prior rather than an external prompt. During inference, the model's dynamics naturally converge to the nearest trained attractor, enabling profile-free role activation. The separation loss ensures attractors are distinct, but if roles are similar, the assumption of linear separability may fail, causing the model to blend roles—a limitation of the subspace model.

## Contribution

(1) A novel training framework, AERE, that eliminates the need for explicit character profiles by shaping hidden-state attractors as role-specific subspaces. (2) Demonstration that role consistency can be achieved through attractor loss without profile injection, using only role-labeled dialogue data. (3) Analysis of subspace dimensionality and attractor strength trade-offs for role fidelity.

## Experiment

### Evaluation Setup

| Dataset | Character-LLM benchmark dialogues (10 roles, 1000 dialogues each, held-out 200 for testing) |
| --- | --- |
| Primary metric | Role consistency score (BERTScore F1) on test responses |
| Secondary metrics | Perplexity, human evaluation (1-5 Likert on consistency and naturalness, N=50 raters) |
| Baseline 1 | Standard fine-tuning with profile prefix (same model, same data with profile appended) |
| Baseline 2 | Prompting with role description (no tuning, e.g., "You are a pirate. Respond accordingly.") |
| Baseline 3 | Learned role embedding vector (trainable vector per role concatenated to input) – isolates benefit of geometric attractor structure |
| Ablation A | AERE without separation loss (β=0) |
| Ablation B | AERE without attractor loss (α=0) – equivalent to LM fine-tuning only |
| Additional analysis | Hidden state trajectory convergence: for each test context, track h_last at each token, compute distance to each role subspace center; plot convergence over sequence length. Verify that distance to correct role subspace decreases. |

### Why this setup validates the claim
This combination forms a falsifiable test of whether AERE enables profile-free role activation via hidden-state attractors. The Character-LLM dataset provides dialogues without explicit profiles, requiring the model to infer role from context. The primary metric (BERTScore) captures semantic role consistency without exact match. Baseline 1 (profile prefix fine-tuning) tests whether explicit profile context is necessary—if AERE matches or exceeds it without profile, the attractor mechanism succeeds. Baseline 2 (prompting) tests if attractor training provides an advantage over zero-shot role adoption. Baseline 3 (learned embedding) isolates whether the subspace geometry (vs. a single vector) is beneficial. The ablation without separation loss tests whether attractor distinctness is crucial; without attractor loss tests whether any fine-tuning helps. Hidden state trajectory analysis provides direct evidence that hidden states converge to role subspaces during generation.

### Expected outcome and causal chain

**vs. Standard fine-tuning with profile prefix** — On a case where the context is ambiguous (e.g., a hospital scene with multiple roles), the baseline with profile prefix may rely on explicit role mention (like "Doctor") to guide behavior, but if the profile is missing, it fails (producing generic responses). Our method instead uses the attractor subspace to autonomously converge to the doctor role from contextual cues (e.g., medical terms), because the hidden state is pulled toward the role-specific subspace. We expect a noticeable gap (e.g., +0.05 BERTScore) on ambiguous contexts but parity on clear, prefix-supported contexts.

**vs. Prompting with role description (no tuning)** — On a case requiring nuanced role knowledge (e.g., a pirate character with specific jargon), prompting only gives a high-level description, so the model may generate stereotyped or inconsistent pirate talk because it lacks fine-grained role experience. Our method's attractor subspace encapsulates the full distribution of hidden states from actual pirate dialogues, so it produces more nuanced and consistent pirate responses. We expect a clear gap (e.g., +0.08 BERTScore) on roles with distinctive interaction patterns, especially in longer dialogues where prompting fades.

**vs. Learned role embedding vector** — On a case where the same role appears in diverse contexts, a single vector may not capture the full diversity of role expressions. Our subspace allows multiple modes, so we expect better consistency across contexts (e.g., +0.03 BERTScore). If the vector is sufficient, the gap will be small, but our method's geometric structure may still help in ambiguous contexts.

### What would falsify this idea
If AERE's role consistency scores are uniformly worse than the prompting baseline across all contexts, or if the ablation without separation loss performs similarly to the full method, then the attractor mechanism is not creating distinct, effective role subspaces—the central claim fails. Specifically, if the metric shows no significant improvement on ambiguous contexts over prompting, the subspace may be capturing irrelevant noise rather than role identity. Also, if hidden state trajectory analysis shows no convergence to role subspaces (distance to correct subspace does not decrease), the attractor mechanism is not operative.

## References

1. Thinking in Character: Advancing Role-Playing Agents with Role-Aware Reasoning
2. LLMs + Persona-Plug = Personalized LLMs
3. IBSEN: Director-Actor Agent Collaboration for Controllable and Interactive Drama Script Generation
4. CAMEL: Communicative Agents for "Mind" Exploration of Large Language Model Society
5. Character-LLM: A Trainable Agent for Role-Playing
6. Generative Agents: Interactive Simulacra of Human Behavior
7. LARP: Language-Agent Role Play for Open-World Games
8. MetaAgents: Large Language Model Based Agents for Decision-Making on Teaming
