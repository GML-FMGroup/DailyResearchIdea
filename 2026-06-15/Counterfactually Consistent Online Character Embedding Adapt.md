# Counterfactually Consistent Online Character Embedding Adaptation for Role-Playing Agents

## Motivation

Existing role-playing agents, such as those using the Role-Aware Reasoning (RAR) framework from 'Thinking in Character', rely on pre-defined static profiles that fail to adapt to evolving character traits or sparse, streaming observations. This structural bottleneck prevents agents from maintaining coherent personas when character information is initially incomplete or changes over time. Similarly, methods like Persona-Plug require offline access to full user history and cannot update incrementally. The root cause is the lack of an online learning mechanism that reconciles new observations with existing character knowledge without catastrophic forgetting or requiring full retraining.

## Key Insight

Counterfactual self-consistency provides a natural, unsupervised objective for online embedding updates because it enforces that the latent character representation must yield consistent reasoning across both observed and hypothetical scenarios, preventing drift while incorporating new evidence.

## Method

### (A) What it is
We introduce **Counterfactual Character Embedding Update (CCEU)**, a method that maintains a continuous latent character embedding vector $\mathbf{e}_t$ and updates it online using a counterfactual self-consistency loss. Input: streaming dialogue turn $x_t$, current embedding $\mathbf{e}_t$, and a buffer $\mathcal{B}$ of recent interactions. Output: updated embedding $\mathbf{e}_{t+1}$.

### (B) How it works
```python
# Hyperparameters: α (learning rate, 0.01), K (number of counterfactuals, 3), η (buffer size, 5)
# Pretrained components: Encoder E (maps text to embedding), LM (LLM for response generation)

# Initialize e_0 randomly or from initial profile description

for each new turn x_t:
    # 1. Encode the turn
    h_t = E(x_t)
    
    # 2. Generate K counterfactual alternatives x_t^k by replacing entities or sentiment
    #    (e.g., using a small perturbation model or rule-based substitutions)
    counterfactuals = generate_counterfactuals(x_t, K)
    
    # 3. Compute consistency loss for current embedding
    #    For each counterfactual, condition LM on e_t and the counterfactual context,
    #    then compare the predicted response distribution (logits) to the original response.
    #    Use KL divergence between the conditional distributions.
    loss = 0
    for k in 1..K:
        # Get conditional distribution over next tokens given e_t and counterfactual context
        p_k = LM(prefix=concat(e_t, counterfactual_context))
        # Get distribution for original context (no perturbation)
        p_orig = LM(prefix=concat(e_t, x_t))
        loss += KL(p_orig || p_k)   # encourage similar behavior despite perturbation
    
    # 4. Compute loss for earlier buffer turns (to prevent forgetting)
    for each (x_i, e_i_old) in buffer B:
        # Re-encode with current embedding
        p_i = LM(prefix=concat(e_t, x_i))
        # Compute consistency with stored embedding's prediction (or a fixed target)
        # Using stored response y_i from the original interaction
        loss -= log p_i(y_i)   # maximize likelihood of observed responses
    
    # 5. Gradient update on embedding (e_t is a trainable parameter)
    e_{t+1} = e_t - α * ∇_e loss
    
    # 6. Update buffer: add (x_t, response) and remove oldest if full
    append_to_buffer((x_t, response))
```

### (C) Why this design
Three key design decisions under trade-offs. *First*, we use counterfactual self-consistency rather than direct supervised fine-tuning on new turns because supervised learning on observed interactions alone would overfit to surface patterns and cause embedding drift away from the core character. By requiring consistency across perturbed versions, we enforce that the embedding encodes invariant character traits rather than spurious correlations. The cost is computational overhead for generating and evaluating counterfactuals. *Second*, we maintain a small buffer of recent turns to mitigate catastrophic forgetting, instead of storing all history or using a recurrent memory. This balances memory efficiency (fixed buffer size) against the risk of forgetting older traits; we accept that very long-term consistency may degrade. *Third*, we update the embedding via gradient descent on a consistency loss rather than using a Bayesian update or a learned update network. Gradient descent is simple, stable, and directly optimizes the objective, but it requires backpropagation through the LM, which is expensive. We chose it over a learned update rule to avoid the meta-learning complexity and because the LM is frozen, making the gradient computable. This design is not a simple combination of Persona-Plug (which uses a frozen embedder) and Character-LLM (which fine-tunes offline) because it introduces an online, unsupervised consistency objective that prevents both drift (unlike fine-tuning) and static representation (unlike plug-in).

### (D) Why it measures what we claim
The computational quantity $\text{KL}(p_{\text{orig}} \mid\mid p_{\text{counterfactual}})$ measures **persona consistency under perturbation** because it assumes that a well-calibrated character embedding should yield similar response distributions when the surface form of the observation changes (entity swap, sentiment flip) but the underlying character intent remains the same. This assumption fails when the perturbation alters the pragmatic meaning (e.g., swapping a neutral entity for one that triggers a different personality facet), in which case the KL divergence reflects not inconsistency but legitimate role-appropriate variation. The buffer likelihood term $\log p_i(y_i)$ measures **temporal coherence** because it assumes that the current embedding should still explain past observations; this assumption fails when the true character has evolved, causing the embedding to artificially resist legitimate change. In that case, the loss reflects a trade-off as the embedding must balance old and new evidence, which we accept as a feature for preventing catastrophic forgetting. The gradient update $\nabla_e \text{loss}$ measures **online adaptation** because it directly modifies the embedding to minimize inconsistency; this assumption fails if the loss landscape is non-convex and the gradient step converges to a poor local optimum, in which case the update reflects a suboptimal embedding. We mitigate this by using a small learning rate and multiple counterfactuals.

## Contribution

(1) A novel online algorithm, CCEU, that updates a latent character embedding from streaming observations using a counterfactual self-consistency loss, without requiring offline retraining or static profiles. (2) A design principle that counterfactual consistency can serve as an unsupervised signal for persona adaptation, decoupling character learning from explicit supervisory labels. (3) An empirical finding (to be verified) that such online updates maintain character coherence over evolving interactions while avoiding catastrophic forgetting, as demonstrated through role-playing consistency metrics.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|----------------------|
| Dataset | CharacterDial | Multi-facet character dialogues with perturbations |
| Primary metric | Persona consistency (human rating) | Directly measures character coherence under shift |
| Baseline 1 | Standard RPA (no persona) | Tests importance of any persona encoding |
| Baseline 2 | Persona-Plug (static embedding) | Tests benefit of dynamic update |
| Baseline 3 | Fine-tuned Character-LLM (offline FT) | Tests robustness to drift from fine-tuning |
| Ablation | CCEU w/o counterfactual loss | Isolates effect of counterfactual consistency |

### Why this setup validates the claim

CharacterDial contains dialogues for each character that vary in context (e.g., formal vs. casual) and include controlled entity/sentiment perturbations. The persona consistency metric captures whether the agent maintains a coherent character despite surface variations. Standard RPA tests the necessity of any character representation; failure would show that persona embedding itself is valuable. Persona-Plug tests whether a static embedding suffices; if CCEU outperforms, it confirms online adaptation is beneficial. Fine-tuned Character-LLM tests whether offline fine-tuning causes drift; CCEU’s counterfactual loss should prevent this. The ablation directly tests the core claim: removing counterfactual consistency should degrade performance on perturbed examples, while the buffer loss alone handles forgetting. The metric is human-rated consistency, which directly reflects the intended goal. A falsification would occur if CCEU’s advantage over Persona-Plug is uniform across all dialogue turns, rather than concentrated on perturbed turns where counterfactual loss is expected to matter.

### Expected outcome and causal chain

**vs. Standard RPA** — On a case where the character must respond politely to a request but the surface form changes (e.g., "Can you help me?" vs. "I need assistance"), Standard RPA produces inconsistent responses because it has no persona grounding. Our CCEU retains the polite persona via the embedding, so responses remain consistent regardless of phrasing. We expect CCEU to show a noticeable gap on perturbed subsets (e.g., 20-30% higher consistency) but parity on simple utterances.

**vs. Persona-Plug** — On a case where the character acquires a new trait (e.g., becomes more cautious after a negative experience), Persona-Plug’s static embedding fails to adapt, producing old behavior. Our CCEU updates the embedding via buffer and counterfactual losses, capturing the shift. We expect CCEU to outperform on evolving characters (e.g., 15-25% higher consistency in later turns) while performing similarly on static characters.

**vs. Fine-tuned Character-LLM** — On a case where the character has multiple facets (e.g., humorous at a party, serious at work), Fine-tuned Character-LLM overfits to recent examples, causing facet “spillover” (e.g., telling jokes at work). Our CCEU’s counterfactual loss enforces stability across contexts, preserving facet separation. We expect CCEU to show a 20-30% improvement on cross-facet generalization, while Fine-tuned Character-LLM degrades on dissimilar contexts.

### What would falsify this idea
If CCEU’s gain over the ablation (w/o counterfactual loss) is uniform across all dialogue turns rather than concentrated on turns with entity/sentiment perturbations, then the counterfactual mechanism is not driving the improvement as hypothesized.

## References

1. Thinking in Character: Advancing Role-Playing Agents with Role-Aware Reasoning
2. LLMs + Persona-Plug = Personalized LLMs
3. Identity-Driven Hierarchical Role-Playing Agents
4. Large Language Models for User Interest Journeys
5. RETA-LLM: A Retrieval-Augmented Large Language Model Toolkit
6. ExpertPrompting: Instructing Large Language Models to be Distinguished Experts
7. Character-LLM: A Trainable Agent for Role-Playing
8. Scaling Instruction-Finetuned Language Models
