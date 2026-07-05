# BADE: Bias-Aware Data Evolution for Balanced Text-Rich Image Generation

## Motivation

DataEvolver's iterative expansion (Liu et al.) does not correct initial biases in the candidate pool; it may amplify existing skews because the Retriever and Generator are blind to the distribution of already-generated data. This leads to persistent underrepresentation of rare text styles, layouts, or visual contexts. A structural solution is to explicitly model the distribution of generated samples and reweight the candidate pool to target low-density regions.

## Key Insight

By treating the evolving data distribution as an unknown density and using a moving-window kernel density estimate to compute per-sample reweighting factors, the system provably reduces the maximum mean discrepancy (MMD) between the generated and target distribution at each iteration, independent of the initial bias.

## Method

(A) **What it is**: BADE (Bias-Aware Data Evolution) extends the DataEvolver framework with a bias detection module that uses kernel density estimation (KDE) to identify underrepresented feature regions and reweights the candidate pool via importance sampling, ensuring distributional bias closure under iterative sampling. Input: initial text prompts P0, target distribution Q (implicitly uniform over features). Output: curated dataset C with improved coverage.

(B) **How it works**:
```python
# BADE Algorithm
Input: initial prompts P0, hyperparameters (h, lambda, R, K)
Initialize: memory M = empty, candidate pool C = empty, density model = None
FeatureExtractor: maps image+text to d-dimensional vector (e.g., text length, layout complexity, font style embedding)
For round r = 1..R:
    # Phase 1: Weighted retrieval
    if r == 1:
        C_r = Retriever(P0)   # uniform sampling
    else:
        # Compute prompt weights: w_p = 1/(density(feature(p)) + epsilon)
        w_p = [1/(density_model.score(feature(p)) + 1e-8) for p in P0]
        # Normalize to probabilities
        prob_p = softmax(log(w_p) * lambda)
        sampled_prompts = sample(P0, weights=prob_p, replace=True)
        C_r = Retriever(sampled_prompts)
    
    # Phase 2: Verification and feedback
    scores, reasons = Verifier(C_r)   # DataEvolver's Verifier
    Update memory M with (C_r, scores, reasons)
    
    # Phase 3: Critic summary
    summary = Critic(M, r)   # DataEvolver's Critic
    
    # Phase 4: Generate for low-density regions
    # Fit KDE on all samples so far (last K rounds for efficiency)
    recent_features = [FeatureExtractor(img) for img in C[-K:]]
    density_model = KernelDensity(bandwidth=h).fit(recent_features)
    # Identify low-density regions via random sampling of feature space
    candidate_features = sample uniformly from feature space (N_candidates)
    low_density_mask = density_model.score_samples(candidate_features) < tau
    target_features = candidate_features[low_density_mask][:max_samples]
    G_r = Generator(target_features, summary)
    
    # Update pool
    C = C ∪ C_r ∪ G_r

Return final C
```

(C) **Why this design**: We chose kernel density estimation (KDE) over parametric mixture models because KDE makes no distributional assumptions—crucial for unknown biases—at the cost of O(n) per query and storing all samples; we mitigate by using a moving window of the last K rounds. We reweight prompts rather than directly thresholding the Verifier scores because prompt reweighting is a global correction that preserves the Verifier's original calibration, while thresholding would add an extra hyperparameter with brittle sensitivity. The strength hyperparameter λ controls how aggressively we oversample low-density prompts; high λ accelerates coverage but risks overfitting to noise in early rounds (trade-off between speed and stability). We sample target features uniformly from the feature space to propose new generation targets, rather than using the Gradient from the density (e.g., gradient ascent on low-density), because uniform sampling is simpler and decouples generation from density estimation, avoiding the need for differentiable features—the cost is that some sampled features may be unrealistic, but the Generator's conditioning (via summary from Critic) filters implausible cases.

(D) **Why it measures what we claim**: The KDE density estimate d(x) at feature x measures the coverage of that feature region in the current pool because we assume the feature space fully captures task-relevant diversity (e.g., text length, layout); this assumption fails when latent dimensions like cultural context are unobserved, so d(x) reflects coverage only on the selected axes. The reweighting weight w_p = 1/(d(feature(p))+ε) measures the undersampling degree of prompt p's feature class because the density is proportional to the frequency of that feature in the pool; this equivalence holds under the assumption that all prompts with the same feature vector are equally well-represented, which fails if prompts differ in other attributes (e.g., image style) not captured by features—causing oversampling of redundant variations. The maximum mean discrepancy (MMD) between pool distribution and target is reduced because importance sampling aligns the empirical distribution toward the target (here uniform over features); this holds asymptotically when the feature space is finite and the density estimate converges, but fails if the target distribution is misspecified (e.g., non-uniform) or if the density estimate has high bias due to insufficient data or too large a bandwidth.

## Contribution

(1) We introduce BADE, a bias-aware extension of the DataEvolver framework that incorporates iterative distributional reweighting via kernel density estimation to correct initial biases in the candidate pool. (2) We establish that explicitly modeling the evolving data distribution and reweighting the candidate pool can prevent amplification of initial skews, achieving more balanced coverage over feature regions. (3) We provide a concrete realization using moving-window KDE and prompt-level importance sampling, with explicit design trade-offs.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | Text-rich image generation benchmark (e.g., AnyText test set) | Standard evaluation for text rendering quality and diversity |
| Primary metric | Feature Coverage MMD between pool and uniform target | Directly measures bias closure claim |
| Baseline 1 | Static data pipeline (DataEvolver without evolution) | Tests necessity of iterative evolution |
| Baseline 2 | AgentInstruct (generative teaching) | Tests alternative adaptive data generation |
| Baseline 3 | Random sampling from prompt pool | Naive baseline for coverage |
| Ablation | BADE without KDE reweighting (uniform resampling) | Isolates effect of density-guided reweighting |

### Why this setup validates the claim

This setup tests the central claim that BADE achieves distributional bias closure by iteratively detecting underrepresented feature regions via KDE and reweighting prompts. The dataset measures both image quality and text rendering, but the primary metric—Feature Coverage MMD—directly quantifies how well the final data pool covers the feature space uniformly, which is the explicit goal. The static pipeline baseline isolates the benefit of iterative refinement; AgentInstruct tests whether a different adaptive strategy (generating new data based on teacher models) achieves similar coverage; random sampling shows the severity of naive bias. The ablation (no KDE reweighting) isolates the importance of the density-guided reweighting mechanism. If BADE outperforms these baselines specifically on coverage while maintaining quality, the claim is supported. Conversely, if coverage gains come at the cost of quality, or if the ablation matches BADE, the density estimation step is not the driver.

### Expected outcome and causal chain

**vs. Static data pipeline** — On a case where initial prompts over-represent short text and simple layouts, the static pipeline produces a pool with dense clusters in those regions and sparse coverage of long, complex text because it never re-evaluates distributional gaps. Our method instead identifies the low-density long-text region via KDE and reweights prompts toward it, generating targeted images. We expect a noticeable gap in MMD (e.g., 0.2 vs. 0.4) and a visible improvement in rare feature bins, while quality metrics remain comparable.

**vs. AgentInstruct** — On a case where the feature space has a rare combination (e.g., vertical text in right-to-left script), AgentInstruct's teacher-driven generation may overlook it because its summarization is task-agnostic (focused on instruction following). Our method explicitly targets low-density features via uniform sampling of the feature space and condition generation on those features, leading to better coverage of such rare pairs. We expect BADE to achieve lower MMD on tail features (e.g., 0.1 vs. 0.3) while maintaining similar head performance.

**vs. Random sampling** — On a case where the prompt pool itself is biased (e.g., 90% prompts are short text), random sampling simply replicates the bias. Our method reweights prompts based on KDE densities, so even if the pool is skewed, it draws more from the underrepresented prompts (long text). We expect random sampling to have high MMD (e.g., 0.5) while BADE reduces it to ~0.15, demonstrating that reweighting is effective even with a fixed prompt set.

### What would falsify this idea
If the MMD improvement of BADE over the ablation (no KDE reweighting) is negligible or inconsistent across seeds, and the gap is concentrated only on easy features rather than the predicted low-density regions, then the core claim of effective bias detection via KDE is invalid.

## References

1. DataEvolver: Self-Evolving Multi-Agent Data Construction for Text-Rich Image Generation
2. AnyText: Multilingual Visual Text Generation And Editing
3. AgentInstruct: Toward Generative Teaching with Agentic Flows
4. AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation
5. MetaMath: Bootstrap Your Own Mathematical Questions for Large Language Models
6. OCR-VQGAN: Taming Text-within-Image Generation
7. Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding
8. Character-Aware Models Improve Visual Text Rendering
