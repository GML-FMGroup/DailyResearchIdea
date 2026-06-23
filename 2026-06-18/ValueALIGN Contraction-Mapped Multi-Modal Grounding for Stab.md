# ValueALIGN: Contraction-Mapped Multi-Modal Grounding for Stable Cultural Value Embeddings in Multi-Agent Systems

## Motivation

Existing methods for cultural value embedding in LLMs rely on text-only conditioning (prompts, fine-tuning, survey items) that fails to capture the embodied, multi-sensory nature of cultural learning, leading to unstable and low-fidelity embeddings. For instance, the Beyond Alignment paper uses World Values Survey text items to evaluate value diversity, assuming textual transmission suffices, but this ignores the interactive, multi-modal contexts where cultural values are actually expressed and learned. This structural limitation—text-only conditioning without grounding in multi-modal experience—recurs across multiple branches (e.g., FULCRA, Value Diversity paper), leaving value embeddings that may not stably reflect true cultural dynamics.

## Key Insight

Contraction mapping in the iterative embedding update guarantees convergence to a unique fixed point, ensuring that multi-modal task grounding yields stable and bounded value embeddings regardless of initial random initialization.

## Method

### Multi-Value Contraction Grounding (MVCG) – Specific Version

**(A) What it is:** MVCG is an iterative algorithm that updates each agent's cultural value embedding by combining a task-grounded self-update (from multi-modal observations) and an inter-agent consensus term (uniform average) via a contraction mixture, ensuring stable convergence to a unique fixed point. Inputs: initial value embeddings (random vectors in R^10), multi-modal task observations (text, image, sound). Outputs: converged value embeddings.

**(B) How it works (pseudocode):**
```python
# Hyperparameters: β=0.1 (contraction rate, 0<β<1), α=0.05 (self-update step), K=100 (max iterations), d=10 (embedding dimension)

Initialize value embeddings v_i ∈ R^d for each agent i (random unit vectors)
Encoder: CLIP ViT-L/14 (frozen), outputs 512-dim vector; project to d via learned linear layer
For iteration k in 1..K:
    # Multi-modal task experience
    For each agent i:
        observe task context O_i (a tuple: text, image, sound)  # each modality from the dataset
        t_i = encoder(O_i)  # task embedding (d-dim after projection)
        self_update = v_i + α * (t_i - v_i)  # adaptive step toward task
    # Inter-agent consensus: uniform average (ensures contraction)
    v_avg = mean(self_update for all i)
    For each agent i:
        v_i_new = (1 - β) * self_update + β * v_avg  # contraction mixture
    # Convergence check: if max_i ||v_i_new - v_i|| < 1e-5: break
    v_i = v_i_new
# Verify contraction condition on a calibration set of 512 samples: compute empirical Lipschitz constant L of the combined update operator (self-update + consensus); assert L < 1. If L >= 1, reduce β until condition holds.
Return final v_i
```

**(C) Why this design:** We chose a contraction mapping (explicit β<1 mixture with uniform average) over simple averaging or gradient-based updates because contraction guarantees a unique fixed point and stability regardless of initialization, directly addressing the meta-gap of embedding stability. We chose a multi-modal encoder (CLIP ViT-L/14) over text-only features because cultural values are often conveyed through visual and auditory cues (e.g., rituals, trade customs); the trade-off is increased computational cost (~0.5 GPU hours for 10 agents, 500 iterations) but it provides richer grounding. We introduced an inter-agent uniform average (instead of weighted average) to ensure the overall operator is a strict contraction (Lipschitz constant < 1) – verified on calibration set – avoiding instability from state-dependent weights like softmax. This design balances stability (contraction) with adaptability (task grounding), ensuring that the embedding does not drift due to individual anomalous experiences.

**(D) Why it measures what we claim:** The core update `v_i_new = (1-β)*self_update + β*v_avg` measures **stability** of value embedding because the contraction property (β<1 and uniform averaging is non-expansive) ensures iteration reduces distance between successive embeddings, bounding variance; this assumption fails when β≥1 or when the self-update operator has Lipschitz constant >1 (e.g., if α is too large), in which case embeddings may diverge. The `self_update = v_i + α*(t_i - v_i)` measures **task grounding** because it moves the embedding toward the task context vector; this assumes the encoder (CLIP) maps task observations to a space where semantic similarity reflects cultural relevance, failing when the encoder is biased (e.g., trained on Western-centric data), in which case self_update pulls toward a skewed cultural perspective. The uniform average consensus measures **community alignment** because it treats all agents equally; this assumes the mean represents the shared cultural norm, failing under strongly polarized distributions (e.g., two distinct clusters), where the mean may not represent either cluster well, potentially reducing alignment for all agents.

## Contribution

(1) Introduces MVCG, a contraction-mapping-based algorithm that iteratively grounds cultural value embeddings via multi-modal task experience and inter-agent consensus, providing provable stability and convergence. (2) Establishes a design principle: value learning in multi-agent systems requires both individual task grounding (multi-modal adaptation) and community consensus (contraction mixing) for stable, high-fidelity embeddings. (3) Provides a framework that operationalizes cultural values beyond text-only conditioning, enabling more robust multicultural agent systems.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | Multi-Cultural Scenario Dataset (MCSD) – 500 scenarios, each with text, image, sound; sourced from Cultural Atlas and M3ED; human-annotated value profiles (10 Schwartz values) by 5 expert raters per scenario | Covers diverse cultural contexts and provides multi-modal grounding; profiles derived from multi-modal scenarios to preserve modality advantage |
| Primary metric | Average cosine similarity between final agent embedding and human-annotated value profile for the scenario it experienced; also variance of final embeddings across 10 random initializations (lower = more stable) | Directly measures cultural alignment and stability |
| Baseline 1 | Per-agent value alignment (no interaction) – each agent executes self-update only, no consensus | Isolates effect of consensus |
| Baseline 2 | Homogeneous community (all same initial embedding) – all agents start with identical embedding | Tests necessity of initial diversity |
| Baseline 3 | CAV (Cultural Alignment Vector) – text-only survey-based embedding (no multi-modal) | Baseline to show advantage of multi-modal grounding |
| Baseline 4 | Probe tuning – fine-tune CLIP text encoder on cultural probes (text version of scenarios) | Baseline for comparison to a common fine-tuning approach |
| Ablation | MVCG without consensus term (i.e., α=0 after first step or self-update only) | Quantifies contribution of inter-agent step |

### Why this setup validates the claim
This combination of dataset, baselines, and metric forms a falsifiable test of whether MVCG produces stable, culturally-aligned value embeddings. The Multi-Cultural Scenario Dataset provides multi-modal observations that ground values in diverse contexts; human-annotated profiles are derived from the same multi-modal scenarios, ensuring the metric captures alignment with culturally appropriate responses. The primary metric (cosine similarity to profiles) directly measures cultural alignment, while variance across initializations measures stability – both key claims. Comparing against per-agent alignment tests whether consensus improves alignment over pure self-update. The homogeneous community baseline tests whether initial diversity is necessary or if convergence to a shared embedding suffices. CAV and probe tuning baselines compare against state-of-the-art text-only and fine-tuning methods. The ablation isolates the consensus term, revealing its role in stability and alignment. By also measuring variance, we directly test the stability claim.

### Expected outcome and causal chain

**vs. Per-agent value alignment (no interaction)** – On a case where an agent observes a task context that deviates from its initial value (e.g., a Western agent encountering a collectivist ritual), the baseline self-update pulls the embedding toward the task, causing drift and reducing alignment with its culture. Our method instead uses consensus (uniform average) to correct such drift: the contraction operator mixes self-update with the community average, damping extreme deviations, so the embedding remains close to the shared cultural norm. We expect a noticeable gap on culturally ambiguous tasks (e.g., gift-giving norms) but parity on universal tasks (e.g., safety rules). Also, variance across initializations will be lower for our method due to contraction.

**vs. Homogeneous community (all same initial)** – On a case where all agents start with identical embeddings (e.g., all assigned individualistic values), the baseline cannot capture cross-cultural diversity; agents cannot learn new cultural perspectives even if tasks require them. Our method, through task-grounded self-updates, allows each agent to diverge appropriately based on multi-modal observations, while consensus via uniform average prevents fragmentation. Thus, we expect our method to achieve higher alignment on tasks that require adapting to local cultural cues, whereas the baseline remains stuck with the initial homogeneous embedding.

**vs. CAV (text-only)** – On a scenario with strong visual or auditory cultural cues (e.g., a ritual involving specific objects or sounds), the text-only baseline has no access to these modalities, so its embedding will miss cultural nuances. Our method, using CLIP to encode all modalities, should achieve higher similarity to the human profile, which was derived from multi-modal observation.

**vs. Probe tuning** – Probe tuning fine-tunes the encoder on textual descriptions, losing cross-modal synchronization. Our method keeps the encoder frozen and updates embeddings iteratively, preserving multi-modal transfer. We expect higher alignment on scenarios where visual/sound cues provide complementary information not captured in text alone.

**Ablation (no consensus)** – Without consensus, agents drift independently based on their own task experiences; variance across agents will be high, and alignment may suffer if an agent encounters a task misaligned with its initial culture. Consensus reduces variance and improves alignment by pooling information.

### What would falsify this idea
If MVCG yields no higher alignment than the per-agent baseline on tasks with strong multi-modal cultural cues, or if the consensus term causes convergence to a bland average that underperforms on all cultures (especially when compared to CAV), then the central claim that contraction grounded in multi-modal data ensures stability and alignment is false. Additionally, if variance of final embeddings remains high across initializations (comparable to per-agent baseline), then the stability claim is contradicted.

## References

1. Beyond Alignment: Value Diversity as a Collective Property in Multicultural Agent Systems
2. One fish, two fish, but not the whole sea: Alignment reduces language models' conceptual diversity
3. On the Dynamics of Multi-Agent LLM Communities Driven by Value Diversity
4. Evolving AI Collectives to Enhance Human Diversity and Enable Self-Regulation
5. Value FULCRA: Mapping Large Language Models to the Multidimensional Spectrum of Basic Human Value
6. Turning large language models into cognitive models
7. Llama 2: Open Foundation and Fine-Tuned Chat Models
8. Efficiently Scaling Transformer Inference
