# Dynamic Drift Budget Constraint for Long-Term Persona Consistency in Role-Playing Agents

## Motivation

Existing role-playing agents, such as those evaluated by RPEval (RPEval: A Role-Playing Evaluation Benchmark for Large Language Models), treat persona as static and fail to maintain consistency over long multi-turn interactions due to temporal drift. The root cause is that both training and evaluation assume fixed identity profiles, ignoring that persona can evolve naturally but must remain anchored to a core identity. Structural property missing: a shared, adaptive measure of allowable drift that couples how agents are trained and how they are evaluated.

## Key Insight

Core identity entropy provides a principled, self-calibrating measure of identity strength that naturally bounds the allowable persona drift, transforming the drift budget from a hyperparameter into a function of the agent's own internal state.

## Method

[Current method]
(A) **What it is**: We propose **Dynamic Drift Budget Constraint (DDBC)**, a training regularization and evaluation metric that constrains persona drift relative to a core identity vector by adapting a drift budget based on core identity entropy. Inputs: core identity profile C (a vector of key trait embeddings, dimension 768), dialogue history H = [r1, r2, ..., rn] (agent responses, each truncated to 512 tokens). Outputs: a drift score and a binary constraint violation flag.

(B) **How it works** (pseudocode):
```python
# Parameters: entropy threshold τ = 0.5, scaling factor α = 0.1, sliding window length L = 512 tokens
# Assumption: Cross-attention weights reliably reflect identity activation. Calibration: correlation with variance-based entropy on validation set (512 examples); if r<0.5, fall back to variance entropy.
# Step 1: Compute core identity entropy
p = softmax(attention_weights(responses, C))  # normalized attention over responses; responses are last L tokens
H_entropy = -sum(p * log(p))

# Step 2: Compute drift budget
budget = α * (1 - sigmoid(H_entropy - τ))  # budget in [0, α]

# Step 3: Compute drift score for each response
drift_scores = [1 - cosine_similarity(embed(r_i), C) for r_i in H]
avg_drift = mean(drift_scores)

# Step 4: Apply constraint
violation = avg_drift > budget
# Training loss: L_reg = max(0, avg_drift - budget)
# Total loss: L_total = L_gen + λ * L_reg, λ=0.1
# Optimizer: AdamW, lr=1e-5, batch_size=16
# Evaluation: return violation flag and ratio avg_drift/budget
```

(C) **Why this design**: We chose to compute core identity entropy via cross-attention weights rather than using a fixed profile embedding because it captures how much the agent's responses rely on core traits at each turn, dynamically reflecting identity strength. The trade-off is increased computational cost during training (cross-attention over all responses) but avoids assuming identity is uniformly expressed. We selected a sigmoid mapping from entropy to budget because it provides a smooth transition between strict and loose constraints, with hyperparameters τ and α controlling the activation point and maximum budget. An alternative of a hard threshold would create discontinuity during optimization. We picked cosine similarity for drift measurement because it is scale-invariant and widely used in representation learning; however, it assumes that core identity and responses are mapped to the same embedding space, which may fail if the embedding model is not trained for this task. Finally, we use mean aggregation over turns rather than max or sum to avoid outlier responses dominating the constraint; this choice assumes that drift is a cumulative property, not an instantaneous one. To address the assumption that attention weights are reliable (Jain & Wallace, 2019), we perform a calibration: on a validation set of 512 examples, we compute the Spearman correlation between attention-based entropy and the variance of cosine similarities between response embeddings and C. If correlation < 0.5, we fall back to the variance-based entropy measure; we reuse the same hyperparameters τ and α.

(D) **Why it measures what we claim**: The core identity entropy H_entropy measures the "strength" of identity consistency because it quantifies how uniformly the agent attends to core traits across responses; this assumption fails when attention weights are noisy due to model randomness, in which case entropy may be artifactually high. The drift budget derived from entropy then measures the allowable deviation from core identity given the identity's current strength; this assumes a monotonic relationship between entropy and needed flexibility, which may break if identity strength is not simply low entropy but also requires diversity (e.g., a multifaceted character). The average drift score measures actual drift from core identity because cosine similarity to core embedding captures semantic deviation; this fails when the embedding space is not well-calibrated for role-playing nuances, so drift may reflect general topic shift rather than persona inconsistency. To validate the drift score's validity, we conduct a human evaluation: 200 responses are rated for persona consistency on a 1-5 Likert scale by 3 experts; we compute Spearman correlation between drift scores and average ratings. A correlation >0.6 confirms the drift score captures consistency. The violation flag measures whether the agent has exceeded the acceptable drift for its identity strength, operationalizing the meta-gap's requirement of a shared, adaptive standard; the assumption is that the adaptive budget correctly calibrates strictness, which fails if the entropy-budget mapping is poorly tuned to the task.

## Contribution

(1) A dynamic drift budget mechanism that adapts based on core identity entropy, unifying training regularization and evaluation metric for persona consistency. (2) Empirical demonstration that DDBC reduces persona drift in long multi-turn dialogues compared to static profile baselines. (3) An analysis showing that core identity entropy correlates with perceived personality consistency in human evaluation.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | RoleBench | Diverse roles and dialogues (5000 dialogues, 100 roles) |
| Primary metric | Expert-rated persona consistency (1-5 Likert, 3 raters, average) | Measures actual drift; reported as Spearman correlation with drift score |
| Baseline 1 | Standard RP agent (no constraint) | Isolates effect of any constraint |
| Baseline 2 | RAR agent | Role-aware reasoning; tests if explicit reasoning alone suffices |
| Baseline 3 | Persona-Plug agent | Personalized generation (fine-tuned on persona data); tests complementarity |
| Ablation | DDBC (fixed budget = 0.05) | Tests necessity of adaptive mechanism |
| Additional ablation | Attention noise ablation (Gaussian noise σ=0.1, 0.2 added to attention weights) | Tests causal role of entropy on budget effectiveness |

### Why this setup validates the claim
This experimental design falsifiably tests the central claim that DDBC reduces persona drift through adaptive constraint based on identity entropy. RoleBench exposes agents to varied roles with known personalities, allowing direct measurement of consistency via expert ratings. The Standard RP baseline isolates the effect of any constraint. The RAR baseline tests whether explicit reasoning suffices; if DDBC outperforms it, the adaptive budget provides additional value. Persona-Plug tests whether fine-tuning on persona data alone matches the constraint; if not, DDBC offers a complementary approach. The fixed-budget ablation pinpoints the necessity of adaptivity. The attention noise ablation directly tests the causal chain: if adding noise to attention weights degrades DDBC's performance, it confirms that entropy (not unrelated factors) drives the adaptive budget. Together, these contrasts form a layered test: success requires DDBC to beat all baselines, especially on roles where identity strength varies, confirming the entropy-driven flexibility.

### Expected outcome and causal chain

**vs. Standard RP agent** — On a case where the character has weak identity (e.g., vague traits), the baseline produces inconsistent responses because it has no drift constraint. Our method computes high entropy, grants a larger budget, allowing natural variation without violating consistency. Thus we expect a noticeable gap on weak-identity roles and parity on strong-identity roles, with overall higher consistency scores.

**vs. RAR agent** — On a case requiring a multi-faceted character (e.g., polite yet sarcastic), RAR produces rigid responses because its reasoning focuses on explicit rules. Our method uses entropy to detect nuanced identity expression and adjusts budget accordingly, enabling flexibility. We expect our method to outperform RAR specifically on complex roles, with similar performance on straightforward ones.

**vs. Persona-Plug agent** — On a case with limited persona data (few training utterances), Persona-Plug overfits or becomes generic because it learns fixed patterns. Our method uses a core identity embedding and dynamic budget, so it adapts to dialogue context. We expect higher consistency on rare or unseen roles, while performance on common roles remains comparable.

**vs. DDBC (fixed budget) and attention noise ablation** — On roles with high entropy variation, the adaptive budget should outperform fixed budget, and adding attention noise should reduce performance, confirming the causal role of entropy.

### What would falsify this idea
If our method's gain over the fixed-budget ablation is uniform across all roles rather than concentrated on roles with high entropy variation, then the adaptive mechanism is not responsible for the improvement. Similarly, if adding attention noise does not degrade performance (or degrades performance of fixed-budget equally), the entropy mechanism is not causal.

## References

1. Role-Playing Evaluation for Large Language Models
2. Thinking in Character: Advancing Role-Playing Agents with Role-Aware Reasoning
3. LLMs + Persona-Plug = Personalized LLMs
4. Identity-Driven Hierarchical Role-Playing Agents
5. Large Language Models for User Interest Journeys
6. RETA-LLM: A Retrieval-Augmented Large Language Model Toolkit
7. ExpertPrompting: Instructing Large Language Models to be Distinguished Experts
8. Character-LLM: A Trainable Agent for Role-Playing
