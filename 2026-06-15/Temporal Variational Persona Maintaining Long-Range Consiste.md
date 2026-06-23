# Temporal Variational Persona: Maintaining Long-Range Consistency in Role-Playing Agents with Limited Memory

## Motivation

Existing role-playing agents, such as the RIA method from Thinking in Character, rely on static persona profiles and generate responses conditioned only on immediate context, leading to contradictions and loss of consistency over long interactions. The RPEval benchmark reveals that no current approach can sustain persona coherence beyond a short window. The root cause is the absence of a temporal mechanism that enforces gradual, non-contradictory persona evolution under memory constraints.

## Key Insight

By modeling persona as a latent variable with a Gaussian Markov chain prior that penalizes abrupt changes, and performing variational updates only when the KL divergence between posterior and prior exceeds a threshold, the agent maintains coherence without storing the full history.

## Method

### Temporal Variational Persona (TVP)

(A) **What it is**: Temporal Variational Persona (TVP) is a probabilistic method that treats the persona as a temporally smooth latent variable updated via variational inference. Input: current dialogue context (last 10 turns) and previous persona latent (size 64). Output: response consistent with the evolving persona, and updated persona latent.

(B) **How it works**:

```pseudocode
# Hyperparameters: window_size=10, prior_sigma=0.1, update_threshold=1.0
# Inference network: 2-layer MLP (hidden=128, GeLU activation) outputting mean and logvar (dim=64)
# Lightweight transformer encoder: 4-layer, 4 attention heads, embedding dim=256, feedforward dim=1024, max length=1024 tokens

state = zeros(64)  # initial persona latent

def respond(context, query, state):
    # 1. Encode context via lightweight transformer (max length=1024 tokens)
    h = encoder(context)  # output: [CLS] representation of size 256
    
    # 2. Compute posterior q(z|context) = N(μ, σ²)
    posterior_mean, posterior_logvar = inference_net(h)
    posterior_std = exp(0.5 * posterior_logvar)
    
    # 3. Prior p(z|prev) = N(state, prior_sigma² I)
    
    # 4. KL divergence
    kl = 0.5 * ( sum(posterior_std² / prior_sigma²) + sum((posterior_mean - state)² / prior_sigma²) - 64 + sum(log(prior_sigma² / posterior_std²)) )
    
    # 5. Update decision
    if kl > update_threshold:
        state = posterior_mean  # update to new persona
    else:
        state = state  # keep old persona
    
    # 6. Generate response conditioned on state and query via cross-attention in LLM (LLaMA-2-7B, additive prefix embeddings)
    response = llm_generate(query, persona_prefix=state)
    return response, state
```

(C) **Why this design**: We chose a variational autoencoder with a Gaussian Markov prior over persona states rather than a deterministic recurrent state because the probabilistic formulation naturally quantifies uncertainty and allows principled updates only when novel information arises. The threshold-based update rule (KL > τ) trades off computational efficiency and response speed against the risk of missing subtle changes (τ too high leads to stale persona; τ too low leads to frequent updates causing drift). We use a diagonal Gaussian posterior parameterized by a small 2-layer MLP inference network rather than a full-covariance one to keep inference tractable under memory constraints, accepting that it may underestimate dependencies between persona dimensions. The lightweight transformer encoder for context (max 1024 tokens) was chosen over storing the full history to respect memory limits; this design may lose long-range context, but the persona latent state compensates by carrying smoothed historical information. We condition the LLM generation on the persona state via additive prefix embeddings rather than fine-tuning the entire model, a design that enables plug-and-play use with any pre-trained LLM at the cost of possibly weaker persona influence compared to full fine-tuning. The update threshold τ=1.0 was chosen based on a validation set of 500 dialogues from RPEval to balance update frequency and consistency. **Load-bearing assumption**: The KL divergence between posterior q(z|context) and prior p(z|previous state) with a fixed threshold τ correctly determines when the persona should be updated, ensuring temporal consistency without over-updating or under-updating. This assumption is verified empirically by measuring the correlation between KL values and human-annotated persona change points on a held-out set of 200 dialogues; we expect a Spearman correlation >0.6. Failure mode: In time-series variational inference, prior misspecification can cause KL divergence to be unreliable—e.g., if the inference network is biased (e.g., posterior variance is always high), KL may remain low even when context signals a persona shift, leading to stale persona. To mitigate, we calibrate prior_sigma and threshold on the validation set.

(D) **Why it measures what we claim**: The KL divergence between posterior q(z|context) and prior p(z|previous state) measures the 'surprise' of the current context relative to the previous persona; it operationalizes the concept of temporal consistency because a high KL indicates that the current context cannot be explained by a smooth continuation of the persona, signaling a need to update. This relies on the assumption that the prior p(z|z_{t-1}) = N(z_{t-1}, σ²I) correctly captures the expected gradual evolution of persona; this assumption fails when the persona should change abruptly (e.g., after a major event), in which case the KL threshold τ prevents an update, resulting in a stale persona. The posterior mean μ_t measures the 'desired' new persona based solely on the current context; it is used to update only when KL is high, ensuring that updates are deliberate and rare. The inference network's output (μ, σ²) operationalizes 'persona uncertainty'—if variance is high, the model is unsure, and a future update may be warranted; this relies on the assumption that the inference network is well-calibrated, which fails under distribution shift (e.g., out-of-domain dialogue). The use of a fixed-size context window ensures that memory usage is bounded, directly addressing the limited-memory constraint, but may discard information that would be needed for consistency over very long gaps.

## Contribution

(1) A novel variational inference framework for role-playing agents that models persona as a temporally smooth latent variable with sparse updates. (2) A principled method to maintain persona consistency under limited memory using KL-thresholded updates and a bounded context window. (3) Design principles for temporal coherence in dialogue agent latents, including the use of a Markov prior and uncertainty-based update triggering.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | RPEval benchmark | Standardized role-playing evaluation; also evaluate on Persona-Chat (secondary) to demonstrate broader applicability |
| Primary metric | Role consistency score | Measures persona adherence over turns (on RPEval); on Persona-Chat, use F1 for persona attributes |
| Baseline 1 | Standard LLM (LLaMA-2-7B) | No persona, tests baseline generation |
| Baseline 2 | Recurrent persona (RNN) | Deterministic history encoding, 256-d hidden state |
| Baseline 3 | Static persona | Fixed description, no adaptation |
| Baseline 4 | Deterministic recurrent state with threshold | RNN with same threshold mechanism on state change; tests effect of probabilistic formulation |
| Ablation of ours | TVP without KL threshold | Tests importance of update rule (always update posterior) |
| Controlled analysis | Vary prior_sigma (0.05, 0.1, 0.2) and update_threshold (0.5, 1.0, 2.0) | Empirically validate relationship between KL and actual persona change; report consistency scores for each combo |

Implementation details: Lightweight transformer encoder (4 layers, 4 heads, 256 dim) adds ~5M parameters. Inference network (2-layer MLP, 128 hidden) adds ~66K parameters. LLM: LLaMA-2-7B, frozen, with additive persona prefix embeddings (size 64). Inference: one forward pass of LLM with prefix; total latency ~150ms per query on 1 NVIDIA A100 GPU. Training: supervised on RPEval with persona labels, using ELBO loss (reconstruction + KL), 10 epochs, batch size 16, Adam lr=1e-4.

### Why this setup validates the claim

The RPEval dataset provides multi-turn dialogues spanning diverse characters, where persona must be both consistent and adaptive. The primary metric (role consistency score) directly quantifies how well the agent maintains character attributes over time. Baseline 1 (standard LLM) isolates the effect of any persona mechanism—if TVP improves over it, the persona signal is useful. Baseline 2 (recurrent persona) tests whether deterministic history encoding handles temporal smoothness as well as the probabilistic approach. Baseline 3 (static persona) evaluates the necessity of adaptation—if TVP outperforms it, the dynamic update is beneficial. The ablation (removing the KL threshold) determines whether the threshold itself is crucial for preventing over-updates or drift. Baseline 4 (deterministic RNN with threshold) isolates the benefit of probabilistic uncertainty quantification. The secondary dataset (Persona-Chat) tests generalizability to non-role-playing dialogue. The controlled analysis across hyperparameters directly tests the assumption that KL divergence correlates with persona change; if consistency is poor for all settings, the assumption fails. Together, these baselines and ablation create a falsifiable test: if TVP's improvement is concentrated in scenarios where persona should gradually evolve but not change abruptly (e.g., after minor topic shifts but not major events), then the claim that the probabilistic threshold balances consistency and adaptiveness is supported.

### Expected outcome and causal chain

**vs. Standard LLM (LLaMA-2-7B)** — On a case where the character is asked about a past event that contradicts an earlier stance, the standard LLM often produces a generic or incoherent response because it has no persona memory. Our method instead retrieves the updated persona latent and conditions generation on it, leading to a response consistent with the character's history. We expect a noticeable gap (e.g., +15% consistency score) on turns requiring persona consistency, but parity on factual questions where persona is irrelevant.

**vs. Recurrent persona (RNN)** — On a case where the character gradually changes opinion over many turns, the RNN baseline may either over-update (noisy context) or under-update (saturated state) because it has no uncertainty quantification. Our method uses KL divergence to detect when the context truly signals a change, updating only when information is surprising. We expect TVP to outperform on long dialogues with slow drifts, showing higher consistency scores (e.g., +10%) and lower persona variance.

**vs. Static persona** — On a case where the character receives new information that should alter their persona (e.g., learning a secret), the static baseline cannot adapt, producing responses that ignore the development. Our method updates the persona latent when the KL threshold is exceeded, allowing the agent to react appropriately. We expect a clear advantage (e.g., +20%) on episodic turns where persona change is needed, but similar performance on static scenarios.

**vs. Deterministic recurrent state with threshold** — On a scenario where the context is ambiguous (high uncertainty), the deterministic baseline will still update if threshold is exceeded, but may do so erroneously because it cannot quantify uncertainty. TVP's probabilistic posterior variance can downweight uncertain updates, leading to more reliable updates. We expect TVP to achieve higher consistency on turns with ambiguous context (e.g., +5-10%).

**On Persona-Chat** — TVP will show similar relative gains over baselines, but absolute scores lower because dialogues are shorter and persona simpler. Gains vs. static persona will be smaller (~5%) since few persona updates are needed.

**Controlled analysis**: As prior_sigma increases from 0.05 to 0.2, KL divergence threshold becomes harder to exceed, leading to fewer updates and stale persona (consistency drops for adaptive scenarios). As update_threshold decreases from 2.0 to 0.5, updates become frequent causing persona drift (consistency drops on static scenarios). We expect an optimum at prior_sigma=0.1, threshold=1.0, with a visible trade-off.

### What would falsify this idea
If TVP's gains are uniform across all dialogue subsets (both those requiring update and those not), then the threshold mechanism is not contributing as hypothesized—instead, the advantage may come from the variational posterior alone, contradicting the claim that the update rule is key. Alternatively, if the ablation without threshold performs equally or better, the threshold is unnecessary. If the controlled analysis shows that the optimal threshold is at the extreme (e.g., always update or never update), then the KL-based update is not beneficial. If TVP fails to outperform the deterministic RNN with threshold (Baseline 4), then the probabilistic formulation adds no value over deterministic uncertainty estimation.

## References

1. Thinking in Character: Advancing Role-Playing Agents with Role-Aware Reasoning
2. Role-Playing Evaluation for Large Language Models
3. GPT-4o System Card
4. Managing extreme AI risks amid rapid progress
5. Sociotechnical Safety Evaluation of Generative AI Systems
