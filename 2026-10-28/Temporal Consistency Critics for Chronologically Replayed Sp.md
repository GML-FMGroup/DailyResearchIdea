# Temporal Consistency Critics for Chronologically Replayed Spatial VLM Supervision

## Motivation

ChronoReplay reorders heterogeneous spatial supervision into chronological episodes but trains with per-target SFT, so successive answers within an episode remain statistically independent — the model can predict 'left of' at turn t and 'right of' at turn t+1 about the same scene without any training signal penalizing the contradiction. This is a shared-residual gap: the offline per-target SFT assumption (also implicit in SpatialVLM and SpatialBot's independent QA training) structurally blocks streaming spatial intelligence because no gradient ever couples adjacent predictions, so temporal coherence is left to emerge by accident from data ordering rather than being enforced as an invariant.

## Key Insight

Within a ChronoReplay episode the observation stream is fixed while only the queried spatial target changes, so a temporally adjacent pair of answers is a paired counterfactual over the same scene — any disagreement between them is attributable to the model rather than the world, making contradiction a scene-invariant, label-free training signal that per-target SFT cannot access.

## Method

**What it is.** We introduce TCC (Temporal Consistency Critic), an auxiliary sequence-level regularizer trained jointly with ChronoReplay's family-balanced SFT loss. Inputs are a replayed episode's successive (query_t, answer_t, query_{t+1}, answer_{t+1}) pairs with their frozen observation stream; output is a scalar consistency penalty added to the SFT gradient.

**Load-bearing assumption (stated explicitly).** TCC assumes that within a ChronoReplay episode the observation stream is fixed and only the query changes, so any disagreement between two adjacent answers is attributable to the model rather than the world, making the critic's consistency score a valid scene-invariant, label-free contradiction signal. This assumption is enforced at pair-construction time by an observation-identity gate (see Phase 2), not by scene-id equality alone, because streaming benchmarks (OVO-S-Bench, OST-Bench) move the agent so o_t != o_{t+1} in general. When the gate cannot verify observation identity, the pair is dropped from L_cons and L_critic rather than treated as a positive.

**How it works.**

```pseudocode
for each episode E = [(o_1,q_1,a_1), ..., (o_T,q_T,a_T)]:
    # Phase 1: SFT (unchanged ChronoReplay)
    L_sft = sum_t  -log p_theta(a_t | o_t, q_t)

    # Phase 2: build paired counterfactual adjacent pairs
    for t in 1..T-1:
        # observation-identity gate (replaces scene-id-only guard)
        if obs_identical(o_t, o_{t+1}):          # exact hash match OR known camera transform within tol=1e-3
            pair = (o_t, q_t, a_t, q_{t+1}, a_{t+1})
            # Phase 3: critic scores contradiction
            # h = frozen hidden state at answer's last token; stop-gradient on h for L_cons
            c = Critic_phi( sg(h_theta(o_t,q_t,a_t)), sg(h_theta(o_t,q_{t+1},a_{t+1})) )
            L_cons += max(0, margin - c) * w_pair
            # margin=0.5, w_pair=0.3, critic = 2-layer MLP (256-d, GeLU)

    # Phase 4: critic trained with contrastive signal
    # positive: (a_t, a_{t+1}) from same episode AND obs_identical(o_t,o_{t+1})==True
    # negative: (a_t, a_{t+1}') where a_{t+1}' = answer from a DIFFERENT scene
    #           filtered by scene-layout similarity: drop negative if cos_sim(emb(scene_t), emb(scene_{t+1}')) > 0.9
    L_critic = -log sigma(c_pos) - log(1 - sigma(c_neg))

    L = L_sft + lambda * L_cons + mu * L_critic   # lambda=0.2, mu=0.1
```

**Why this design.** We chose a learned critic over a hand-coded rule (e.g., string-match 'left'/'right') because spatial answer families in ChronoReplay are heterogeneous and source-native, so a fixed lexical rule would fire on only a subset, accepting the cost that the critic must itself be trained and can drift. We chose to condition the critic on frozen hidden states h_theta rather than on decoded text because it lets the consistency gradient flow back into the language model through h_theta, accepting the cost that the critic and the LM co-adapt and can collapse to a constant; we mitigate collapse with a stop-gradient on h_theta for L_cons and an EMA target (decay=0.999) on the critic features. We chose contrastive pairs drawn from *different scenes* as negatives rather than random answer permutations, because only cross-scene pairs guarantee that the negative is genuinely contradictory rather than merely a different valid answer to the same query, accepting the cost that negative mining requires scene-id metadata; we additionally filter negatives whose scene-layout embedding is within cosine similarity 0.9 of the positive scene to avoid false contradictions. A reviewer familiar with ChronoReplay would not describe this as a variant of it: ChronoReplay's contribution is *episode ordering and family balancing*, whereas TCC adds a gradient path *between* adjacent answers that ChronoReplay's per-target SFT explicitly lacks — the same-data controls in ChronoReplay isolate ordering effects, while TCC isolates coupling effects, which no ChronoReplay ablation can produce.

**Why it measures what we claim.** The critic score `c` measures *temporal answer consistency* because we assume that within an episode the observation stream is fixed and only the query changes (verified by the obs_identical gate), so a high `c` is equivalent to the two answers being jointly satisfiable under one scene; this assumption fails when ChronoReplay concatenates supervision from different scenes into one episode, in which case `c` reflects cross-scene topic similarity rather than within-scene coherence. The contrastive negative `(a_t, a_{t+1}')` measures *contradiction* because we assume answers drawn from different scenes are inconsistent by construction; this assumption fails when two scenes share the same spatial layout (e.g., two identical rooms), in which case the negative is a false contradiction and the critic learns to penalize valid agreement; we filter such negatives by scene-layout embedding cosine similarity > 0.9. The margin term `max(0, margin - c)` measures *enforced coherence* because we assume that pushing `c` above `margin` for positive pairs is sufficient to suppress contradictions at inference; this assumption fails when the critic overfits to the frozen h_theta distribution of training episodes and does not transfer to unseen query orderings, in which case the penalty reflects episode memorization rather than a general coherence invariant.

**(D) Equivalence and failure mode.** The critic score `c` operationalizes *joint answer satisfiability under a fixed scene*. This equivalence holds under assumption A: frozen hidden states h_theta preserve answer semantics across queries (i.e., the last-token hidden state is a sufficient statistic for the answer's spatial content). A breaks under failure mode F: scene concatenation (o_t != o_{t+1} despite scene-id match) or critic overfitting to training episode orderings. We test A directly by comparing TCC against a ground-truth scene-consistency oracle critic (a deterministic relation-composition table over left/right/front/behind and viewpoint transforms) that bounds the achievable gain, and we monitor collapse via the variance of `c` across positive pairs during training (collapse detector: variance < 1e-3 over 100 steps triggers early stop).

## Contribution

(1) TCC, a sequence-level consistency critic that supplies a gradient path between temporally adjacent spatial answers in chronologically replayed episodes, using same-scene paired counterfactuals as a label-free contradiction signal. (2) A contrastive critic-training scheme with cross-scene negatives that yields a consistency score whose margin threshold is calibrated by construction rather than by post-hoc confidence. (3) An analysis of when per-target SFT and sequence-level coupling are complementary versus redundant, characterized by the fraction of episodes whose adjacent queries share a scene.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | OVO-S-Bench, OST-Bench, UCS-Bench | Streaming spatial benchmarks with adjacent-query structure |
| Primary metric | Three-benchmark mean accuracy | Aggregate signal across spatial answer families |
| Secondary metric | Multi-view contradiction subset accuracy; family-level breakdown (left/right/front/behind) | Isolates the predicted coherence effect |
| Baseline | ChronoReplay SFT checkpoint | Tests whether coupling adds beyond ordering |
| Baseline | Ordinary joint SFT | Tests whether coupling adds beyond plain supervision |
| Baseline | Shifted-order control | Tests whether coupling survives reordering |
| Baseline | Lexical/entailment rule consistency penalty (non-learned) | Tests whether the learned critic is necessary vs. a generic regularizer |
| Baseline | KL penalty between adjacent answer distributions | Tests whether a simple distributional coupling suffices |
| Ablation-of-ours | TCC with critic frozen at init | Tests whether learned critic is doing the work |
| Ablation-of-ours | TCC with ground-truth scene-consistency oracle critic | Bounds achievable gain and tests assumption A |
| Ablation-of-ours | TCC without obs_identical gate (scene-id only) | Tests whether observation-identity verification matters |

### Why this setup validates the claim

The central claim is that TCC's cross-answer gradient path produces temporal coherence that per-target SFT cannot, and the design must isolate coupling from ordering and from data. ChronoReplay's checkpoint is the right primary baseline because it already controls episode ordering and family balancing; any gain over it therefore attributes to the coupling path rather than to data composition. Ordinary joint SFT tests the weaker sub-claim that any joint supervision suffices, and the shifted-order control tests whether the effect is an artifact of adjacency order rather than of the consistency signal itself. The lexical/entailment rule and KL-penalty baselines are load-bearing for novelty: if either matches TCC, the learned critic is not necessary and the contribution reduces to a generic regularizer. The frozen-critic ablation is the load-bearing one for mechanism: if TCC's gain persists with an untrained critic, the mechanism is regularization-by-architecture, not learned contradiction detection, and the paper's causal story collapses. The oracle-critic ablation bounds the achievable gain and directly tests assumption A (frozen hidden states preserve answer semantics). The obs_identical-gate ablation tests whether the load-bearing assumption (fixed observation stream) is actually enforced. The three-benchmark mean is the right primary metric because the predicted effect is a cross-family coherence improvement, not a single-family lexical win; family-level breakdowns and the multi-view contradiction subset are reported to show the effect is not uniform overfitting.

### Expected outcome and causal chain

**vs. ChronoReplay SFT checkpoint** — On a case where an episode asks about a landmark from two viewpoints and the second query's answer contradicts the first, ChronoReplay produces the contradiction because its per-target SFT never couples the two answer tokens. Our method instead pushes the critic score above margin for that adjacent pair because the frozen-hidden-state penalty flows back through both answers, so we expect a noticeable gap on the multi-view contradiction subset but parity on single-query episodes.

**vs. Ordinary joint SFT** — On a case where a long episode contains a late query whose answer drifts toward a generic prior, ordinary joint SFT produces the drift because natural turn weighting lets early targets dominate and no term opposes inconsistency. Our method instead regularizes the late answer against its predecessor because the critic scores the pair jointly, so we expect a modest gain concentrated in late-episode turns with near-parity on early turns.

**vs. Shifted-order control** — On a case where queries are shuffled so adjacent pairs no longer share a scene, the shifted-order control produces answers that are individually correct but mutually inconsistent because the coupling signal has nothing to bind. Our method instead retains coherence only where obs_identical(o_t, o_{t+1}) holds, so we expect our gain to shrink toward the shifted-order control on shuffled episodes — the pattern that distinguishes coupling from ordering.

**vs. Lexical/entailment rule consistency penalty** — On a case where adjacent answers use heterogeneous source-native spatial families (e.g., 'to the left of' vs. 'west of'), the lexical rule fires only on a subset and leaves other contradictions unpenalized. Our method instead scores all pairs through the learned critic, so we expect TCC to outperform the lexical baseline on the multi-view contradiction subset; if the lexical baseline matches TCC, the learned critic is not necessary.

**vs. KL penalty between adjacent answer distributions** — On a case where adjacent answers are both valid but differently phrased, the KL penalty pushes them toward identical distributions and can penalize valid paraphrase. Our method instead scores joint satisfiability via the critic, so we expect TCC to outperform KL on family-level breakdowns while KL may show a spurious gain on lexical overlap; if KL matches TCC, the learned critic is not necessary.

**vs. TCC with ground-truth scene-consistency oracle critic** — The oracle critic defines consistency by geometry (relation-composition table) rather than by model agreement, so it bounds the achievable gain. We expect TCC to approach but not exceed the oracle on the multi-view contradiction subset; a large gap indicates the learned critic is underperforming the verified signal, and a small gap supports assumption A (frozen hidden states preserve answer semantics).

**vs. TCC without obs_identical gate** — On a case where the agent moves between turns (o_t != o_{t+1}) but scene-id matches, the ungated variant treats a genuine world change as a model contradiction and penalizes valid answers. We expect the ungated variant to underperform TCC on streaming episodes with agent motion, confirming that observation-identity verification is load-bearing.

### What would falsify this idea

If TCC's gain over ChronoReplay is uniform across all subsets rather than concentrated on adjacent-query contradiction cases, or if the frozen-critic ablation matches the trained-critic result, or if the lexical/entailment or KL-penalty baseline matches TCC, the claim that a learned cross-answer coupling path drives the improvement is wrong. Additionally, if the oracle-critic ablation shows no headroom over TCC, assumption A is unsupported and the critic score does not measure joint answer satisfiability as claimed.

## References

1. ChronoReplay: One Answer Interface Is Not Enough for Streaming Spatial Intelligence
2. SpatialVLM: Endowing Vision-Language Models with Spatial Reasoning Capabilities
3. VideoLLM-online: Online Video Large Language Model for Streaming Video
4. SpatialBot: Precise Spatial Understanding with Vision Language Models
5. RoboVQA: Multimodal Long-Horizon Reasoning for Robotics
6. RT-1: Robotics Transformer for Real-World Control at Scale
7. Interactive Language: Talking to Robots in Real Time
8. VideoCoCa: Video-Text Modeling with Zero-Shot Transfer from Contrastive Captioners
