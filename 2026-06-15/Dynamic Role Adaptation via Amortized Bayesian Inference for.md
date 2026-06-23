# Dynamic Role Adaptation via Amortized Bayesian Inference for Conversational Agents

## Motivation

Existing role-playing agents, such as those built on static character profiles (e.g., Thinking in Character), assume that role representations are fixed prior to interaction. This fails to capture real-time engagement dynamics—like turn length and user feedback—that are critical for maintaining naturalistic interaction and valid evaluation. The root cause is that both evaluation benchmarks and methods treat role-playing dimensions as immutable, ignoring the conversational evidence that should update them.

## Key Insight

Treating the role representation as a latent variable that evolves via amortized Bayesian inference from conversation history enables dynamic adaptation without manual labels, because the inference network learns to extract role-relevant signals from the interaction stream.

## Method

**DRA-ABI: Dynamic Role Adaptation via Amortized Bayesian Inference**

(A) **What it is:** DRA-ABI is a latent variable model where the agent's role representation is a continuous latent state that updates after every turn via amortized variational inference. Inputs: conversation history (user utterances and agent's own previous actions). Outputs: next agent utterance conditioned on the inferred role posterior.

(B) **How it works (pseudocode):**

```python
# Hyperparameters:
# latent_dim = 64          # dimensionality of the role latent space
# prior_mean = 0, prior_std = 1  # standard Gaussian prior
# inference_net: Transformer encoder (4 layers, 8 heads, 256 hidden)
# decoder: Pre-trained LLM (e.g., LLaMA-7B) with cross-attention to latent z

def dra_abi_step(history, z_prev=None):
    # history: list of alternating user and agent utterances
    # z_prev: previous latent role (if None, sample from prior)
    
    # 1. Encode history into sufficient statistics
    h = inference_net(history)  # (seq_len, d_model) -> pooled context vector
    
    # 2. Compute posterior parameters via amortized inference
    posterior_mean = MLP_mean(h)     # 2-layer MLP (256->128->64)
    posterior_logvar = MLP_logvar(h) # same architecture as mean
    
    # 3. Reparameterized sample from posterior
    z = posterior_mean + exp(0.5 * posterior_logvar) * eps    # eps ~ N(0,1)
    
    # 4. Generate response conditioned on z and context
    response = decoder.generate(history, latent=z)   # cross-attention to z
    
    # 5. Compute ELBO loss (for training)
    kl_loss = KL_divergence(N(posterior_mean, posterior_logvar), N(0,1))
    nll_loss = -log P(response | history, z)
    loss = nll_loss + beta * kl_loss   # beta=0.1 (annealed)
    
    return response, z, loss
```

(C) **Why this design:**
We chose amortized inference (a transformer encoder) over sequential Monte Carlo (SMC) because SMC requires resampling steps that are expensive for long conversations; the trade-off is that the inference network may produce biased posterior estimates when the encoder capacity is limited, but end-to-end training compensates by jointly optimizing encoder and decoder. We chose a continuous latent space (Gaussian) over a discrete one (e.g., categorical roles) because continuous representations allow smooth interpolation and gradient-based optimization, but they sacrifice interpretability—we cannot directly inspect which role facet changed. We condition the decoder on the entire history (via attention) rather than only the last user turn to capture long-term engagement patterns (e.g., shifts in user enthusiasm); the cost is increased memory and compute for long contexts. Finally, we use a KL annealing schedule (beta increases from 0 to 1) to mitigate posterior collapse, a common issue in VAEs; this reduces the risk of the latent variable being ignored early in training, but it may slow convergence.

(D) **Why it measures what we claim:**
The posterior mean vector computed by the inference network measures the agent's adapted role state because it aggregates evidence from the entire conversation history under the Bayesian assumption that the latent role is a sufficient statistic for predicting the next appropriate utterance under evolving character traits. This assumption fails when the inference network overly relies on surface-level cues (e.g., keyword frequency) rather than true role-consistent signals; in that case, the inferred posterior reflects spurious correlations (e.g., topic drift) instead of genuine role adaptation. The KL divergence term measures the deviation from the prior (a neutral role assumption), ensuring the posterior does not wander arbitrarily; it operationalizes the concept of role stability—the latent role should not change without conversational evidence. This assumption fails if the prior is misspecified (e.g., too broad), allowing role drift even when no new evidence is present. Finally, the negative log-likelihood (NLL) of the response measures role consistency under the model, assuming that a well-adapted role should make the agent's next turn more predictable; this holds only if the decoder learns to use the latent role appropriately, and fails when the decoder ignores the latent variable (posterior collapse), making NLL reflect only language model fluency rather than role fidelity.

## Contribution

(1) A principled framework (DRA-ABI) that treats role adaptation as amortized Bayesian inference over a latent role state, enabling real-time dynamic behavior without manual labels. (2) A demonstration that integrating turn-level evidence (e.g., user engagement signals) into the role posterior improves both naturalness and evaluation validity, as the latent role captures evolving character traits. (3) A training objective combining ELBO with KL annealing that balances predictive accuracy and role stability, providing a prototype for adaptive role-playing agents.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Role-playing dialogue transcripts | Real interactions with character constraints |
| Primary metric | Role consistency score | Measures adherence to role traits |
| Baseline 1 | Vanilla LLM (no role conditioning) | Tests need for any role representation |
| Baseline 2 | Fixed role embedding (static latent) | Tests value of dynamic adaptation |
| Baseline 3 | Discrete role classifier (VQ-VAE) | Tests benefit of continuous latent space |
| Ablation | DRA-ABI with prior-only (no inference) | Isolates effect of posterior inference |

### Why this setup validates the claim
This setup forms a falsifiable test of DRA-ABI's central claim—that dynamic role adaptation via amortized Bayesian inference improves role-consistent generation. The dataset of real role-playing dialogues provides a natural testbed where character traits must be inferred and maintained over turns. The vanilla LLM baseline establishes the floor: without role conditioning, the agent cannot maintain consistency. The fixed embedding baseline tests whether a static role representation suffices; if it performs on par with DRA-ABI, then dynamic adaptation is unnecessary. The discrete baseline compares continuous vs. discrete latent spaces, testing whether smooth interpolation matters. The ablation (prior-only) removes history-driven inference, isolating its contribution. The role consistency score directly measures the key property of interest—whether the agent's utterances match the intended character. If DRA-ABI outperforms these baselines specifically on conversation segments where the role shifts (e.g., user introduces new context that should trigger adaptation), the claim is supported; if gains are uniform, the mechanism may be spurious.

### Expected outcome and causal chain

**vs. Vanilla LLM** — On a case where the user asks about a topic contradicting the agent's role (e.g., a knight recommending cowardice), the vanilla LLM produces a neutral or contradictory response because it has no role representation. Our method infers the role posterior from history, so it recognizes the knight's bravery trait and produces a consistent refusal. We expect DRA-ABI to have a large gap on role-contradictory queries (∆ > 20%) but parity on role-neutral queries.

**vs. Fixed role embedding** — On a case where the role should gradually shift (e.g., the agent starts as aloof but becomes warmer over turns), the fixed embedding cannot adapt, producing cold responses throughout. Our method updates the latent after each turn, so it captures the warming trend. We expect DRA-ABI to outperform significantly on long conversations (>15 turns) where role dynamics accumulate (∆ > 30% in consistency scores on late turns).

**vs. Discrete role classifier** — On a case where the role requires fine-grained nuance (e.g., a sarcastic but helpful character), the discrete classifier forces hard assignments, missing the mixture. Our continuous latent interpolates between sarcasm and helpfulness smoothly. We expect DRA-ABI to excel on ambiguous utterances (∆ > 15% on those segments), while both methods perform similarly on clear-cut role labels.

### What would falsify this idea
If DRA-ABI's gain over fixed embedding is uniform across all conversation lengths (not concentrated in late turns) and equally large on short dialogues where role drift is negligible, then the inference mechanism is not capturing real adaptation—the claim that dynamic inference matters would be falsified.

## References

1. Thinking in Character: Advancing Role-Playing Agents with Role-Aware Reasoning
2. Role-Playing Evaluation for Large Language Models
3. Unveiling the Secrets of Engaging Conversations: Factors that Keep Users Hooked on Role-Playing Dialog Agents
4. Character-LLM: A Trainable Agent for Role-Playing
5. Dungeons and Dragons as a Dialog Challenge for Artificial Intelligence
6. IBSEN: Director-Actor Agent Collaboration for Controllable and Interactive Drama Script Generation
7. LLMs + Persona-Plug = Personalized LLMs
8. Identity-Driven Hierarchical Role-Playing Agents
