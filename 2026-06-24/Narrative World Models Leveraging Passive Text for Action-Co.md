# Narrative World Models: Leveraging Passive Text for Action-Conditioned Prediction via Masked Language Modeling

## Motivation

Existing language world models like Qwen-AgentWorld require large-scale supervised interaction trajectories from real environments, which are expensive to collect. In contrast, abundant passive text (e.g., internet articles, stories) contains rich descriptions of actions and their outcomes but lacks explicit action labels. The root cause is that current methods treat action-conditioned prediction as a supervised learning problem, ignoring the self-supervised signal in narrative discourse. Specifically, Qwen-AgentWorld assumes action labels are necessary for learning state transitions, yet narratives implicitly encode causal relationships between actions and state changes.

## Key Insight

The causal structure of language in narratives naturally pairs actions with resulting state changes, enabling a masked language model to learn action-conditioned transitions by predicting tokens that represent state differences, without requiring explicit action supervision.

## Method

### Narrative World Model (NWM)

**A) What it is**: Narrative World Model (NWM) is a two-stage training framework that first pre-trains a transformer on a large corpus of narrative texts using a self-supervised masked prediction task designed to capture action-conditioned state transitions, then fine-tunes on a small set of real interaction trajectories for action-conditioned next-state prediction.

**B) How it works**:
```python
# Pseudocode for NWM training

def extract_training_examples(text):
    for paragraph in text:
        sentences = split_into_sentences(paragraph)
        for i in range(1, len(sentences)):
            # Action extraction: trained BERT classifier (bert-base-uncased fine-tuned on 500 expert-annotated narrative sentence pairs; achieves 85% F1 on validation)
            action = extract_action_bert(sentences[i-1], sentences[i])
            state_before = sentences[i-1]
            state_after = sentences[i]
            # Create input with masking on state_after
            masked_after = mask_tokens(state_after, p=0.15)  # masking probability 0.15
            input_tokens = tokenize(state_before + " [ACTION] " + action + " [NEXT] " + masked_after)
            yield input_tokens, state_after

# Pre-training hyperparameters:
# Model: 125M parameter transformer (RoBERTa-style); also 60M variant for resource comparison
# Masking probability: 0.15
# Training data: BookCorpus, Wikipedia, and filtered narrative subsets from C4
# Optimizer: AdamW, lr=1e-4, linear warmup 10k steps
# Compute: 8 TPUv3 pods for 2 weeks (125M); for 60M variant: 4 TPUv3 pods for 1 week

# Stage 2: Fine-tuning on interaction trajectories
# Format: (state_t, action_t, state_{t+1}) as text
# Fine-tune with causal LM loss to generate state_{t+1} given state_t and action_t
# Input: [CLS] state_t [SEP] action_t [SEP] state_{t+1}
# Hyperparameters: lr=5e-5, batch_size=32, 10k steps

# Inference: given state and action, generate next state auto-regressively.
```

**C) Why this design**: We chose to pre-train on passive text using a masked prediction task over consecutive sentences rather than explicit action-conditioned prediction because narratives naturally contain implicit action-effect pairs without a formal action label. The trade-off is that heuristically extracted actions may be noisy, but the abundance of data compensates. We used a 125M parameter transformer as a balance between expressivity and training cost; larger models could capture more nuances but would require more data and compute, which may be prohibitive for many labs. We designed the pre-training to only see pairs of sentences where an action verb is identified, avoiding cases where no action occurs; this reduces false positives but may miss subtle actions. Finally, we fine-tune on full interaction trajectories using next-token prediction rather than masked loss to align with the sequential nature of real environment steps. This choice accepts that fine-tuning may need more data than masked fine-tuning, but it directly optimizes for the inference task.

**D) Why it measures what we claim**: The pre-training task uses masked prediction on state_after tokens given state_before and extracted action; this measures the model's ability to infer state changes from action descriptions because the masking forces the model to rely on the causal relationship encoded in the narrative. The assumption is that the trained BERT action extractor (85% F1) captures the true action causing the state change; this assumption fails when the extractor misses subtle actions (e.g., passive constructions or implicit causality), in which case the model may learn spurious correlations between state_before and state_after independently of the action. However, the pre-training's diversity of narrative styles mitigates this, as many sentences do explicitly state actions. The fine-tuning stage measures action-conditioned prediction directly, and the pre-training provides a strong prior that reduces the amount of interaction data needed. We additionally validate the action extraction precision/recall on a held-out sample of 100 narrative excerpts via human evaluation to quantify the noise level.

## Contribution

(1) A novel self-supervised pre-training objective for world models that leverages passive narrative text to learn action-conditioned state transitions without explicit action labels. (2) The demonstration that pre-training on generic text corpora with implicit action information significantly reduces the required amount of supervised interaction data for fine-tuning a language world model, as evidenced by performance on AgentWorldBench. (3) A curated narrative subset of existing text corpora annotated with heuristic action extraction, enabling reproducible pre-training for future world model research.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | WebArena (text states) | Real web interaction data |
| Primary metric | Next-state prediction accuracy | Direct world model quality |
| Baseline1 | GPT-4o (prompted) | No world model pre-training |
| Baseline2 | Transformer (interaction only) | No narrative pre-training |
| Baseline3 | NWM pre-trained on random sentence pairs (non-narrative) | Isolates narrative structure effect |
| Baseline4 | NWM (random actions) | Tests action extraction role |
| Ablation of ours | NWM (60M parameters) | Tests resource requirements |

### Why this setup validates the claim
The combination of baselines and ablation isolates the contribution of narrative pre-training and action extraction. Comparing against GPT-4o tests whether any form of world model pre-training improves over a purely prompted LLM. Comparing against a transformer trained solely on interaction data tests whether narrative pre-training provides a data-efficient advantage. The ablation with random action labels tests whether the heuristic action extraction is crucial or whether the model learns despite noise. The baseline with random sentence pairs (shuffled sentences during pre-training) tests whether the narrative structure (ordered pairs) is essential. The 60M parameter variant tests whether performance can be maintained with lower compute. The metric, next-state prediction accuracy, directly measures the core capability of predicting state changes given actions, making it a clear falsifiable test: if NWM does not outperform the interaction-only transformer on low-frequency actions, the narrative pre-training is ineffective for generalization. The uniform gain pattern across all actions would also falsify the claim that narrative pre-training helps specifically via action-effect reasoning.

### Expected outcome and causal chain

**vs. GPT-4o (prompted)** — On a case where a subtle state change follows a click (e.g., a hidden element becomes visible), GPT-4o often fails to update because it lacks explicit action-conditioned next-state training; it may hallucinate or repeat the previous state. Our method, pre-trained on narratives where such action-effect pairs are common, predicts the exact update step-by-step. We expect a noticeable gap on tasks requiring precise state transitions (e.g., form filling) but near parity on trivial changes where GPT-4o can hallucinate correctly.

**vs. Transformer (interaction only)** — On a rare action like "undo", the interaction-only transformer has few examples, leading to generic or incorrect predictions. Our method leverages narratives where "undo" appears in diverse contexts (e.g., "he undid his previous action, returning to the blank page"), providing a robust prior. We expect larger gains on actions with low frequency in the interaction data (bottom 10%) compared to frequent actions, as measured by per-action accuracy.

**vs. NWM pre-trained on random sentence pairs** — On all actions, especially rare ones, the random-pair variant should perform worse, demonstrating that narrative structure (temporal order) is crucial for learning action-effect relations.

**vs. NWM (random actions)** — If the action extractor is noisy, the random action ablation may still perform reasonably, but we expect a significant drop in per-action accuracy, confirming that accurate action extraction is important.

**Ablation: NWM (60M)** — We expect only a small degradation (<2%) in next-state accuracy, showing that the method remains competitive at lower resource.

Additionally, we will conduct a human evaluation on a sample of 100 narrative excerpts to measure the precision/recall of the trained BERT action extractor. We expect precision ≥0.80 and recall ≥0.75. If these metrics are lower, the causal chain from narrative to action-conditioned learning is weakened, and we will report the correlation with downstream world model performance.

### What would falsify this idea
If NWM shows similar gains on all actions (no concentration on rare or complex ones) or fails to outperform the interaction-only transformer on rare actions, the narrative pre-training does not effectively transfer action-effect knowledge. Also, if the ablation with random actions performs as well as the full NWM, then the action extraction is not critical. Furthermore, if the random sentence pair baseline matches full NWM, then narrative structure is not essential, contradicting the core insight.

## References

1. Qwen-AgentWorld: Language World Models for General Agents
2. Web Agents with World Models: Learning and Leveraging Environment Dynamics in Web Navigation
3. WebArena: A Realistic Web Environment for Building Autonomous Agents
4. Mind2Web: Towards a Generalist Agent for the Web
5. Learning Interactive Real-World Simulators
6. Do As I Can, Not As I Say: Grounding Language in Robotic Affordances
7. Don’t Generate, Discriminate: A Proposal for Grounding Language Models to Real-World Environments
8. MineDojo: Building Open-Ended Embodied Agents with Internet-Scale Knowledge
