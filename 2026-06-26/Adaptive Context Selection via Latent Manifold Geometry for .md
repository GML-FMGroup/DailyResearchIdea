# Adaptive Context Selection via Latent Manifold Geometry for In-Context Learning in Robotics

## Motivation

Existing methods for in-context learning in robotics, such as ICWM and RICL, rely on a fixed-length context window of recent interactions or retrieved demonstrations. This assumption fails when dynamics are non-stationary: a fixed window may include irrelevant past states after a regime change (e.g., a terrain shift) or miss causally relevant states from earlier in the episode if dynamics cycle. The root cause is the use of temporal contiguity as a proxy for causal relevance, which breaks under non-stationarity.

## Key Insight

The latent manifold of robot states is locally piecewise smooth, and geometric consistency in this manifold—measured by proximity weighted by local curvature—is a more reliable indicator of causal relevance than temporal proximity, because dynamics transitions create high-curvature regions where the manifold folds.

## Method

**(A) What it is:** The method, called **ProCR** (Projective Context Reconciliation), takes as input a current observation and a memory buffer of past observation–action pairs with their latent codes, and outputs a dynamically sized set of past interactions to serve as the context window for in-context learning.

**(B) How it works (pseudocode):**
```python
# Encoder E: observation -> latent vector (pre-trained on diverse robot experiences)
# Memory: list of tuples (obs_i, act_i, z_i) from past steps
# Current observation obs_t

z_t = E(obs_t)  # project current state into latent manifold

# Step 1: Compute distances to all past latent codes
distances = [||z_t - z_i||_2 for each z_i in memory]

# Step 2: Estimate local manifold curvature at z_t
k = min(K, len(memory))  # hyperparameter K=20
knn_indices = argsort(distances)[:k]
# Curvature approximation: mean squared distance from knn centroid
centroid = mean([z_i for i in knn_indices])
curvature = mean([||z_i - centroid||_2**2 for i in knn_indices])

# Step 3: Compute adaptive radius
r0 = 0.5  # base radius (hyperparameter)
alpha = 0.1  # curvature scaling factor
radius = r0 * (1 + alpha * curvature)

# Step 4: Select past states within radius
selected_indices = {i : distances[i] < radius}
if len(selected_indices) < 1:  # fallback: nearest neighbor if empty
    selected_indices = [argmin(distances)]

# Step 5: Return corresponding observation-action pairs as context
context = [(memory[i].obs, memory[i].act) for i in selected_indices]
```

**(C) Why this design (≥80 words):** We chose a curvature-adaptive radius over a fixed threshold (e.g., k-nearest neighbors) because curvature directly captures how strongly the dynamics compress or expand the latent space. In low-curvature regions (smooth dynamics), a larger radius includes more past states, enriching the context without overfitting; in high-curvature regions (sharp transitions), a smaller radius excludes irrelevant states that are geometrically far, even if temporally close. We used the mean squared distance to the local centroid as a curvature proxy rather than a Hessian-based method, accepting that it is a rougher estimator but computationally cheaper and robust to small sample sizes at the start of an episode. We set a fallback to the nearest neighbor to avoid an empty context window when the manifold is sparse, trading off the risk of including an irrelevant state against the failure of having no context at all. Finally, we pre-train the encoder on diverse system configurations (following ICWM's approach) to ensure the latent manifold captures dynamic similarities rather than visual appearance, accepting the cost of requiring a large offline dataset.

**(D) Why it measures what we claim (≥60 words):** The distance `||z_t - z_i||_2` measures **causal relevance** because, under the assumption that the encoder has learned a manifold where nearby states share similar transition dynamics (a property induced by training on a world-modeling objective), a smaller distance implies the past interaction is likely to be predictive of future outcomes in the current regime. This assumption fails when the encoder generalizes poorly to an unseen dynamic regime (e.g., a novel contact event not in the training data), in which case distance reflects dissimilarity in the training distribution rather than causal relevance. The curvature `c` measures **degree of non-stationarity** because, under the assumption that high curvature indicates a fold in the manifold corresponding to a regime change, a larger curvature implies the local region is unstable and should be sampled from more conservatively. This assumption fails when the curvature is inflated by noise or sparse data, causing over-constriction of the context window.

## Contribution

(1) A novel algorithm, ProCR, that dynamically selects the context window for in-context learning in robotics by adapting to local manifold curvature, overcoming the limitation of fixed-length windows. (2) A concrete design principle: geometric consistency in a learned latent space, weighted by curvature, is more robust to non-stationary dynamics than temporal contiguity. (3) An analysis connecting manifold curvature to the degree of dynamic change, providing a theoretical grounding for the adaptive radius.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | MetaWorld with varying object dynamics | Covers diverse dynamics with ground truth success. |
| Primary metric | Task success rate | Directly measures adaptation effectiveness. |
| Baseline 1 | Fixed k-nearest neighbors | Tests necessity of adaptive radius. |
| Baseline 2 | Fixed radius (no curvature) | Tests curvature-based radius adaptation. |
| Baseline 3 | Uniform random context | Tests relevance of distance-based selection. |
| Ablation-of-ours | Ours w/o curvature (fixed radius) | Isolates curvature adaptation impact. |

### Why this setup validates the claim

The chosen dataset, MetaWorld with tasks that vary object dynamics (e.g., different masses, friction coefficients), directly challenges the method's ability to detect and adapt to non-stationary dynamics. The primary metric, task success rate, captures the end goal of in-context learning: choosing the right past interactions to inform the current policy. The baselines form a logical decomposition: fixed kNN tests whether a fixed number of neighbors suffices; fixed radius tests whether our curvature-adaptive radius improves upon a static threshold; random context tests the importance of distance-based relevance. The ablation removes only the curvature scaling, leaving the rest of the pipeline intact, isolating the specific contribution of curvature. Together, they allow a falsifiable test: if our method outperforms all fixed variants, it confirms that both distance-based selection and curvature adaptation are beneficial.

### Expected outcome and causal chain

**vs. Fixed k-nearest neighbors** — On a case where the robot transitions from pushing a light block to a heavy block (high curvature region), fixed kNN always includes k=20 past states regardless of local dynamics. Among those, many are from the previous regime (light block) and are causally irrelevant for the new regime, leading to policy confusion and low success. Our method, detecting high curvature via spread of nearby latents, shrinks the radius to exclude those outdated states, selecting only the few relevant to the heavy block. Thus, we expect a noticeable success gap (e.g., 40% vs. 70%) on high-curvature tasks, but similar performance on low-curvature tasks where fixed kNN is adequate.

**vs. Fixed radius** — On a case where dynamics are smooth and low-curvature (e.g., continuous pushing over a long trajectory), the fixed radius of 0.5 may be too conservative, excluding many useful past states that are geometrically slightly farther but still relevant. Our method expands the radius proportional to low curvature, including those states and enriching the context. Consequently, we expect our method to achieve higher success (e.g., 80% vs. 65%) on low-curvature tasks, while on high-curvature tasks both methods perform similarly (since both shrink).

**vs. Uniform random context** — On any task, random context includes many irrelevant past interactions (e.g., different objects or actions) that mislead the policy. Our method selects only causally relevant ones via distance metric in a dynamics-aware latent space. Hence, we expect our method to consistently outperform random context by a large margin (e.g., 75% vs. 30% overall), regardless of curvature.

### What would falsify this idea
If our method's success gain over fixed radius is uniform across low and high curvature subsets, rather than being concentrated where the predicted failure modes occur (e.g., improvement only on high-curvature tasks for vs. kNN, and on low-curvature for vs. radius), then the central claim that curvature-adaptive radius is necessary would be falsified.

## References

1. In-Context World Modeling for Robotic Control
2. RICL: Adding In-Context Adaptability to Pre-Trained Vision-Language-Action Models
3. OpenVLA: An Open-Source Vision-Language-Action Model
4. π0: A Vision-Language-Action Flow Model for General Robot Control
5. Action Tokenizer Matters in In-Context Imitation Learning
6. RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control
7. VILA: On Pre-training for Visual Language Models
8. Vision-Language Foundation Models as Effective Robot Imitators
