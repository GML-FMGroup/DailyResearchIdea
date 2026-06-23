# Contextual Persona Gating: Zero-Shot Persona Acquisition in Multi-Agent Language Models from Raw Conversation

## Motivation

Existing role-playing agents like RIA (Thinking in Character) require predefined character profiles to anchor role identity, but in open-domain multi-agent conversations, such profiles are often unavailable or costly to obtain. Other methods rely on fine-tuning on character-specific data (Character-LLM) or using external personality tests (CharacterChat), limiting generality. The structural challenge is how to extract persona-relevant information from raw conversational history without any labeled profiles, enabling zero-shot adaptation to any speaker.

## Key Insight

Persona-relevant spans naturally exhibit higher cross-utterance consistency within the same speaker than across different speakers, and this contrastive signal can be exploited via a learned gating mechanism that weights token contributions to a differentiable persona conditioning vector.

## Method

We propose Contextual Persona Gating (CPG). (A) **What it is**: CPG is a module that, given a multi-turn conversation history \( H = \{u_1, ..., u_n\} \) with speaker labels, produces a persona conditioning vector \( p_s \) for each speaker \( s \) by aggregating token representations weighted by their estimated persona relevance, trained via a contrastive objective over utterance-level persona consistency. (B) **How it works**: The method proceeds as follows.
```
1. Encode each utterance \( u_i \) using a frozen LLM encoder to get token representations \( \{h_{i,t}\} \).
2. For each token, compute a gate score \( g_{i,t} = \sigma( W_g \cdot h_{i,t} + b_g ) \), where \( \sigma \) is sigmoid.
3. For each utterance, compute an utterance-level persona embedding \( e_i = \frac{1}{T_i} \sum_t g_{i,t} h_{i,t} \). (Weighted average)
4. For each speaker \( s \), collect all their utterances \( \{u_i: speaker = s\} \). Positive pairs: (e_i, e_j) for same speaker. Negative pairs: (e_i, e_k) for different speakers.
5. Train with contrastive loss: \( L = -\mathbb{E}[\log \frac{\exp(sim(e_i, e_j))}{\sum_{k} \exp(sim(e_i, e_k))}] \), where \( sim \) is cosine similarity.
6. At inference, for target speaker \( s \), aggregate persona vectors from their utterances using a weighted sum: \( p_s = \frac{1}{N_s} \sum_i \alpha_i e_i \), where \( \alpha_i = \text{softmax}(W_\alpha e_i) \) to give more weight to informative utterances.
7. Condition the LLM's generation by cross-attending to \( p_s \) at each decoding step, or by prepending \( p_s \) as a soft prefix to the prompt.
```
Hyperparameters: gate dimension \( d_g=128 \), contrastive temperature \( \tau=0.07 \), \( \alpha \) weight dimension \( W_\alpha \in \mathbb{R}^{d \times 1} \). (C) **Why this design**: We chose token-level gating over utterance-level pooling because it allows the model to selectively focus on persona-relevant words (e.g., opinions, preferences) while ignoring filler, accepting the cost of increased parameter count and training instability from sparse gradients. We used cosine similarity in the contrastive loss instead of a learned metric to avoid overfitting to the training data, trading off discriminative power for robustness. We conditioned generation via cross-attention on the aggregated persona vector rather than concatenation because cross-attention can dynamically weigh different aspects of the persona vector at each step, but this increases inference latency due to additional attention operations. We opted for a single-layer gating network (linear+sigmoid) instead of a deeper network to maintain training efficiency, at the cost of reduced expressiveness. (D) **Why it measures what we claim**: The gate score \( g_{i,t} \) measures persona relevance under the assumption that tokens consistently associated with a speaker across multiple utterances carry persona information; this assumption fails when a speaker uses formulaic or generic language (e.g., "yes" or "I don't know") that yields high gate scores but low persona distinctiveness, in which case the gate score reflects utterance salience rather than persona identity. The contrastive loss measures utterance-level persona consistency under the assumption that utterances from the same speaker share latent persona features; this assumption fails when the speaker's language varies across topics (e.g., switching from professional to casual tone), leading to high variance in \( e_i \) within the same speaker, in which case the loss pushes apart embeddings that should be similar. The aggregated vector \( p_s \) measures the central persona tendency of the speaker under the assumption that weighted averaging of utterance embeddings removes idiosyncratic noise; this assumption fails when the speaker's persona is multi-faceted and context-dependent, in which case a single vector cannot capture the full persona.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|------------------------|
| Dataset | Multi-turn role-playing dialogue (e.g., MRstyle) | Tests multi-speaker persona grounding. |
| Primary metric | Persona consistency F1 | Measures if speaker identity is maintained. |
| Baseline 1 | Standard LLM (no persona conditioning) | Shows need for explicit persona representation. |
| Baseline 2 | Prompt-based persona conditioning | Tests effect of static vs dynamic gating. |
| Baseline 3 | Fine-tuned role-specific model | Compares learned persona from scratch vs ours. |
| Ablation of ours | CPG w/o contrastive pretraining | Isolates effect of contrastive persona learning. |

### Why this setup validates the claim

The combination of a multi-speaker dialogue dataset, persona consistency metric, and three baselines directly tests the central claim that Contextual Persona Gating (CPG) improves persona grounding. The dataset provides naturalistic scenarios where speakers exhibit latent persona traits across utterances. Using persona consistency F1 as the metric directly captures whether generated responses maintain the speaker's identity, which is the claimed benefit. The baselines isolate different aspects: standard LLM tests the necessity of any persona conditioning; prompt-based conditioning tests the advantage of learned dynamic gating over static descriptions; fine-tuned model tests whether CPG's contrastive pretraining yields better persona representations than end-to-end fine-tuning. The ablation (CPG w/o contrastive pretraining) identifies whether the contrastive objective is responsible for the gains, or if the gating mechanism alone suffices. If our method outperforms all baselines on consistency, especially in subtle persona shifts, the claim is supported. Conversely, if baseline 3 matches our performance, the contrastive pretraining may be unnecessary.

### Expected outcome and causal chain

**vs. Standard LLM (no persona conditioning)** — On a case where a speaker has distinct opinions (e.g., political stance), the baseline produces inconsistent responses across turns because it lacks any memory of persona. Our method instead aggregates persona-relevant tokens across utterances via gating and contrastive loss, so we expect a noticeable gap on multi-turn consistency (e.g., 20% higher F1) but parity on single-turn tasks where persona is less critical.

**vs. Prompt-based persona conditioning** — On a case where a speaker's persona is implicitly encoded in word choice (e.g., formal vs casual), the prompt baseline relies on a static description that misses nuanced cues. Our method dynamically gates tokens, capturing subtle lexical preferences, so we expect a noticeable gain on style-related consistency (e.g., 15% higher F1) but smaller gains on explicit trait adherence (e.g., name recall).

**vs. Fine-tuned role-specific model** — On a case where the target speaker has limited training data, the fine-tuned model overfits and fails to generalize persona to new contexts. Our CPG learns a robust persona embedding via contrastive pretraining across speakers, so we expect a significant advantage on low-resource speakers (e.g., 25% higher F1) and parity on high-resource ones.

**vs. CPG w/o contrastive pretraining** — On a case where two speakers share similar vocabulary, the ablation (random gates) yields noisy persona vectors that confuse identities. Our full method uses contrastive loss to pull apart speaker embeddings, so we expect a clear advantage on distinguishing similar speakers (e.g., 10% higher F1) but similar performance on distinct speakers.

### What would falsify this idea

If our method shows no improvement over the ablation without contrastive pretraining, or if the gains are uniform across all baselines (e.g., consistent 5% improvement) rather than concentrated on the predicted failure modes (e.g., multi-turn consistency or subtle style), then the central claim that CPG provides selective persona grounding would be falsified.

## References

1. Thinking in Character: Advancing Role-Playing Agents with Role-Aware Reasoning
2. A Multi-Task Role-Playing Agent Capable of Imitating Character Linguistic Styles
3. Identity-Driven Hierarchical Role-Playing Agents
4. Generative Agents: Interactive Simulacra of Human Behavior
5. CharacterChat: Learning towards Conversational AI with Personalized Social Support
6. Character-LLM: A Trainable Agent for Role-Playing
7. ExpertPrompting: Instructing Large Language Models to be Distinguished Experts
8. Exploring Collaboration Mechanisms for LLM Agents: A Social Psychology View
