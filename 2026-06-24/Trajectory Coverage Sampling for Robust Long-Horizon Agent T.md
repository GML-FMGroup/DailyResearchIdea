# Trajectory Coverage Sampling for Robust Long-Horizon Agent Training

## Motivation

Existing data curation pipelines for agentic models, such as the OT-Agent pipeline (OpenThoughts-Agent, 2026), focus on task source diversity but do not explicitly ensure coverage of the state-action space, especially for long-horizon tasks where trajectories are sparse. This leads to models that perform well on benchmark tasks but fail to generalize to novel scenarios because underrepresented branches of the state-action tree are not reinforced during training. A concrete limitation is that OT-Agent's 100K training set, while diverse in task sources, may contain dense homogeneous sequences within each task, missing rare but critical state transitions, as evidenced by its evaluation on only seven benchmarks without coverage analysis.

## Key Insight

By modeling visitation density in a learned state-action embedding space and upweighting trajectories from low-density regions, we enforce exploration of the data pool's latent topology, ensuring that training data spans the full variety of possible agent experiences.

## Method

**Trajectory Coverage Sampling (TCS)**

**(A) What it is:** TCS is a data curation step that takes a pool of candidate agent trajectories (each a sequence of state-action pairs) and outputs a weighted subset that maximizes coverage of the state-action space. It uses a kernel density estimator (KDE) over embeddings of (state, action) pairs to compute per-trajectory coverage weights.

**(B) How it works:**
```python
# Input: pool of trajectories T = {τ_i}, each τ_i = [(s_t, a_t)]
# Output: importance weight w_i for each trajectory

# 1. Embed each (s_t, a_t) using a frozen pre-trained encoder
#    (e.g., a 2-layer MLP with hidden dimension 256, output dimension 128, ReLU activation)
embeddings = [encoder(s_t, a_t) for τ_i in T for (s_t, a_t) in τ_i]

# 2. Fit a Kernel Density Estimator with Gaussian kernel over all embeddings
#    (hyperparameter: bandwidth h=0.1, chosen by Silverman's rule of thumb)
kde = KernelDensity(kernel='gaussian', bandwidth=h).fit(embeddings)

# 3. For each trajectory τ_i, compute its average visitation density:
#    log-density of each step, then mean over steps
density_i = mean([kde.score_samples(encoder(s_t, a_t).reshape(1,-1)) for (s_t, a_t) in τ_i])

# 4. Convert density to weight inversely, with smoothing to avoid extreme values:
#    (hyperparameter: temperature τ=1.0)
w_i = exp(-density_i / τ)

# 5. Normalize weights to sum to N (pool size)
w_i = (w_i / sum(w_i)) * len(T)

# Optional: Sample subset with replacement proportional to w_i to form final training set
```

**(C) Why this design:** We chose KDE over parametric density models (e.g., Gaussian mixture) because KDE is non-parametric and can represent arbitrary density landscapes without assuming a fixed number of modes, which is critical when the state-action space is highly irregular; the cost is higher computational overhead during curation (O(n^2) if naive, but we use KD-tree acceleration). We used a frozen encoder rather than end-to-end learning with the agent to avoid overfitting to the density objective; this decouples representation quality from coverage optimization, but risks that the encoder may not capture task-relevant state-action distinctions—if the encoder is poorly chosen (e.g., too generic), density estimates may reflect spurious similarities. The inverse density weighting with exponential temperature allows smooth prioritization of low-density regions; an alternative like threshold-based rejection (only keep trajectories below a density cutoff) would be more aggressive but could discard too many trajectories, especially if the pool is already sparse. Finally, we average step-level densities per trajectory rather than using per-step weights individually, because the goal is to bias the dataset toward whole trajectories containing rare events, not to oversample individual steps that might disrupt temporal coherence.

**(D) Why it measures what we claim:** The density estimate `kde.score_samples` measures the local crowdedness of the state-action embedding space: a low score indicates that the (state, action) pair is unlike others in the pool, i.e., it is underrepresented. By assigning higher weights to trajectories with low average density, we prioritize trajectories that cover rare regions, operationalizing the motivation-level concept of *state-action space coverage*. The assumption underlying this equivalence is that the embedding space is *sufficiently task-relevant*: if the encoder maps distinct state-action pairs that are task-equivalent (e.g., different ways to reach the same outcome) into the same region, the density measure would fail to distinguish them, and the weight would not reflect true coverage of diverse behaviors. This assumption fails when the encoder is too coarse (e.g., using only state-level features without action information) or when the task has multiple reward-equivalent strategies—in that case, the metric reflects *embedding-space coverage*, not necessarily *behavioral coverage*. Additionally, the method assumes the pool already contains at least one representative from each region of interest; if a whole region is missing, the KDE cannot detect it, and the method cannot create coverage from nothing—it only amplifies existing underrepresented paths.

**Load-bearing assumption (explicit statement):** The frozen pre-trained encoder (2-layer MLP with hidden 256, output 128, ReLU) is trained on a self-supervised inverse dynamics task to ensure task-relevant state-action distinctions are captured. Specifically, we pre-train the encoder to predict the action a_t given consecutive states (s_t, s_{t+1}) from a held-out corpus of agent trajectories. This calibration ensures that the embedding space reflects state-action transitions relevant to the agent's decision-making, rather than arbitrary perceptual differences. We validate this by measuring the encoder's accuracy on a held-out set of 512 diverse transitions; if accuracy is below 85%, we fine-tune the encoder with additional data.

## Contribution

(1) A novel trajectory coverage sampling algorithm that uses kernel density estimation over state-action embeddings to prioritize underrepresented trajectories during data curation. (2) A design principle: explicit coverage-aware weighting during data curation improves generalization to novel long-horizon agent tasks compared to uniform or task-diversity-only sampling. (3) An analysis framework linking density-based coverage metrics to downstream task performance, enabling principled dataset construction.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|-----------------------|
| Dataset | Agentic task suite (WebShop, ScienceWorld, ALFWorld, and 4 withheld benchmarks for generalization) | Covers diverse agentic domains; held-out sets test generalization |
| Primary metric | Average task success rate | Direct measure of agent performance |
| Baseline 1 | Random sampling from pool | Tests if TCS improves over no curation |
| Baseline 2 | Uniform weighting (no TCS) | Isolates effect of coverage weighting |
| Baseline 3 | Gaussian Mixture Model density (5 components) | Tests non-parametric vs parametric |
| Baseline 4 | MOPO-style density sampling (prediction error-based) | Prior coverage-based method for comparison |
| Ablation | TCS with per-step weights (instead of trajectory average) | Evaluates trajectory-level vs step-level |

### Why this setup validates the claim
This experimental design forms a falsifiable test of the claim that TCS improves agent training by prioritizing trajectories covering rare state-action pairs. Using a multi-domain agentic task suite ensures varied state-action spaces, testing generalization. Random sampling establishes whether any curation helps; uniform weighting tests if weighting itself is beneficial beyond just selecting a subset. The GMM baseline probes whether KDE's non-parametric flexibility is necessary, and the MOPO baseline compares against an existing coverage-based method. The ablation examines if trajectory-level averaging preserves temporal coherence. The primary metric, average success rate, directly captures overall agent competence, and any improvement must be attributed to better coverage of the state-action space rather than alternative factors. Additionally, we perform a quantitative analysis: on a held-out set of 1000 state-action pairs labeled by human annotators as 'rare' or 'common', we compute the correlation between TCS weight and rarity label (expect ρ > 0.5).

### Expected outcome and causal chain

**vs. Random sampling** — On a case where rare but critical actions exist (e.g., a specific tool use in ScienceWorld), random sampling may include few trajectories with that action due to low frequency, leading to under-learning. Our method assigns higher weight to those trajectories because they reside in low-density regions, so the model trains on them more often. Thus we expect a noticeable gap on tasks requiring rare actions, but parity on common tasks. Quantitative correlation analysis confirms that low-density trajectories correspond to rare events.

**vs. Uniform weighting** — Uniform weighting treats all trajectories equally even if some are highly repetitive. For example, in WebShop, many trajectories follow a common path, causing the model to overfit to common patterns. Our method down-weights dense regions, reducing repetition. Thus we expect our method to improve on tasks where diversity matters, e.g., tasks with multiple valid strategies.

**vs. GMM density** — GMM assumes a fixed number of Gaussian components (5); if the state-action space has irregular or multimodal structure (e.g., different behaviors for different object types in ALFWorld), GMM may merge distinct regions into one component, misestimating density. KDE can capture arbitrary shapes. Thus we expect our method to outperform on benchmarks with highly varied state-action embeddings, particularly where the number of modes is unknown.

**vs. MOPO density** — MOPO uses prediction error of a dynamics model as a proxy for novelty; this can be noisy for long-horizon tasks. Our KDE-based method directly measures embedding density, which may be more stable. We expect TCS to show better performance on tasks where dynamics are stochastic, because prediction error becomes unreliable.

### What would falsify this idea
If our method's improvement over uniform weighting is uniform across all task types rather than concentrated on tasks with sparse trajectories, then the coverage mechanism is not driving gains. Additionally, if the quantitative correlation between TCS weight and rarity is low (ρ < 0.3), the embedding space is not capturing task-relevant rarity.

## References

1. OpenThoughts-Agent: Data Recipes for Agentic Models
