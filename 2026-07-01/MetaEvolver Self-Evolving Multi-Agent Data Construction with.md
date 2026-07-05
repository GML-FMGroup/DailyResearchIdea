# MetaEvolver: Self-Evolving Multi-Agent Data Construction with Adaptive Quality Assurance

## Motivation

DataEvolver achieves self-evolving data construction through multi-agent collaboration, but its performance remains sensitive to manually designed Verifier thresholds and Critic prompt templates. These hand-crafted components require domain expertise and repeated tuning for each new task, limiting automation and scalability. Without automatic adjustment, the system cannot adapt to shifts in data distribution across rounds, leading to suboptimal sample quality and coverage.

## Key Insight

By treating the Verifier threshold and Critic prompt template as learnable hyperparameters and optimizing them via Bayesian optimization on a composite quality metric that balances sample diversity and fidelity, the system can automatically adapt to the evolving data distribution without human intervention.

## Method

MetaEvolver extends the DataEvolver multi-agent framework with a Meta-Learning Agent (MLA) that automatically adjusts the Verifier's rejection threshold and the Critic's prompt template after each round based on the statistics of generated samples. The MLA uses Bayesian optimization to select the next hyperparameter configuration, maximizing a composite objective function that combines sample quality and diversity.

**Core loop of MetaEvolver**
```python
# Hyperparameters: M (number of rounds), K (candidates per round), eta (initial threshold=0.5),
#   T_set (set of 5 candidate prompt templates for Critic, e.g., ["correct any errors", "improve clarity", "add details", "fix grammar", "make concise"]),
#   calibration_set (512 random held-out samples with ground-truth OCR accuracy)
# Phase 1: Initialize
thresh = eta
prompt_template = T_set[0]
memory = []
calibrate_every = 5  # round interval for recalibration
for round in range(M):
    # Phase 2: Generate candidates using Retriever and Generator as in DataEvolver
    candidates = retrieve_and_generate(prompt_template, memory)
    # Phase 3: Evaluate with Verifier using current thresh
    scores, failures = verifier(candidates, thresh)
    # Phase 4: Aggregate feedback via Critic
    feedback = critic(scores, failures, prompt_template)
    memory.append(feedback)
    # Phase 5: Update hyperparameters via Bayesian optimization
    # MLA observes the distribution of scores and the aggregate quality of generated samples
    # Compute composite objective: O = (mean(adjusted_scores) + lambda * diversity(samples)) / (1 - reject_rate)
    # where adjusted_scores = scores * (1 - |thresh - 0.5|)  # penalize extreme thresholds for stability
    # diversity is measured by 1 - average pairwise cosine similarity of CLIP embeddings of generated text regions
    # lambda = 0.3 (fixed), reject_rate = fraction of candidates that scored below thresh
    O = compute_objective(scores, candidates, thresh)
    # Update surrogate model (Gaussian Process with Matern kernel, length_scale=1.0, variance=1.0) with (thresh, prompt_template) -> O
    gp.update(thresh, prompt_template, O)
    # Select next hyperparameters by maximizing Expected Improvement (use 10 random restarts + local optimization)
    thresh_new, prompt_new = gp.optimize_acquisition()
    # Apply constraints: thresh in [0.2, 0.8], prompt in T_set
    thresh = clip(thresh_new, 0.2, 0.8)
    prompt_template = T_set[argmin(dist(prompt_new, T_set))]
    # Phase 6: Periodic calibration using held-out validation set (every calibrate_every rounds)
    if round % calibrate_every == 0 and round > 0:
        # Compute Verifier scores on calibration_set with current thresh
        val_scores = verifier(calibration_set, thresh)
        # Compute OCR accuracy on accepted samples (using ground truth)
        accepted_idx = val_scores > thresh
        ocr_acc = compute_ocr_accuracy(calibration_set[accepted_idx])
        # Adjust thresh to match target acceptance rate (e.g., 70%) if ocr_acc is low
        if ocr_acc < 0.9:
            thresh = min(thresh + 0.05, 0.8)  # raise threshold to increase precision
```

**(C) Why this design**: We chose Bayesian optimization (BO) over gradient-based meta-learning because BO naturally handles discrete prompt template selection and requires no differentiable pipeline, accepting a higher per-round computational cost (~1 sec for GP update) but offering sample-efficient exploration. We used a Gaussian Process surrogate with a Matern kernel to model the objective function, balancing flexibility and smoothness; this avoids the need for a large training set of prior rounds. The composite objective combines mean Verifier score (adjusted for threshold extremeness) and diversity (computed as 1 - average cosine similarity between CLIP embeddings of generated text regions) to prevent the optimizer from converging to narrow, high-scoring regions; the reject_rate normalization penalizes overly strict thresholds that discard too many samples. We chose Expected Improvement acquisition over Upper Confidence Bound because it naturally trades off exploration and exploitation without requiring a tunable exploration parameter, reducing human priors. The threshold bounds [0.2, 0.8] were set to avoid extreme values that could collapse the generation; this is the only hand-crafted component, but it is a safe range that can be automatically determined via a warm-up phase in future work. The periodic calibration step (every 5 rounds) uses a held-out validation set with ground-truth OCR accuracy to prevent the Verifier from drifting, ensuring that the objective remains aligned with true quality.

**(D) Why it measures what we claim**: The composite objective `O` (mean adjusted_scores + 0.3 * diversity) / (1 - reject_rate) measures automated quality assurance because the **mean adjusted_scores** term quantifies sample quality as perceived by the Verifier, under the assumption that the Verifier's scoring function aligns with true data quality (e.g., OCR accuracy). This assumption fails when the Verifier overfits to spurious features (e.g., font style) and misranks samples; in that case, `adjusted_scores` reflects Verifier confidence rather than actual quality. **diversity(samples)** (1 - average cosine similarity of CLIP embeddings) measures semantic diversity of generated text regions, assuming CLIP embedding similarity reflects content similarity; this fails when generated samples differ in non-semantic ways (e.g., word order shuffling that preserves meaning), reducing the diversity score artificially. **reject_rate** captures the fraction of candidates the Verifier discards, intended to measure data efficiency; it assumes that rejected samples are genuinely low-quality, but when the threshold is misaligned, reject_rate reflects Verifier strictness rather than true quality. The composite `O` balances these three quantities, operationalizing the concept of "automated quality assurance without human priors" by using Bayesian optimization to find hyperparameters that maximize this surrogate; the underlying assumption is that optimizing `O` on a small set of rounds generalizes to future rounds, which fails when the data distribution changes non-stationarily (e.g., a new font appears), requiring re-adaptation. The calibration step mitigates this by explicitly checking against ground-truth OCR accuracy periodically, adding a safety net.

## Contribution

(1) MetaEvolver, a self-evolving multi-agent data construction framework that automatically adjusts Verifier thresholds and Critic prompt templates via Bayesian optimization, eliminating the need for manual hyperparameter tuning. (2) A composite objective function combining Verifier scores, sample diversity, and reject rate that effectively guides the meta-learner toward configurations producing high-quality, diverse data without human priors. (3) Demonstration of the meta-learning approach on the text-rich image generation task, showing that automatic adjustment improves data quality and coverage compared to fixed hyperparameters in DataEvolver.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | AnyText benchmark (5,000 images across 10 font types, including rare fonts like script and blackletter) | Standard for text-rich image generation; includes rare fonts to test adaptation |
| Primary metric | Word accuracy (OCR) measured by Tesseract OCR | Directly measures text correctness; use exact match ignoring case |
| Secondary metric | Human evaluation (Likert scale 1-5 for quality and diversity) on a random subset of 200 samples per method | Tests correlation between Verifier scores and human judgment |
| Baseline 1 | DataEvolver (fixed threshold=0.5, single prompt template="correct any errors") | Original framework without meta-adaptation |
| Baseline 2 | Fixed threshold & prompt (threshold=0.5, template="correct any errors") | Isolates effect of meta-learning |
| Baseline 3 | Random hyperparameter search (sample 10 random (thresh, prompt) pairs, each run for M=10 rounds) | Tests Bayesian optimization's advantage |
| Baseline 4 | Meta-RL adaptive policy (a small 2-layer MLP trained via REINFORCE to adjust thresh and prompt selection based on past round statistics) | Tests whether Bayesian optimization is uniquely suitable |
| Ablation | MetaEvolver-no-diversity (lambda=0 in objective) | Removes diversity term from objective |

### Why this setup validates the claim
This combination of dataset, baselines, and metrics forms a falsifiable test of the central claim that MetaEvolver's Bayesian optimization over threshold and prompt template improves sample quality and diversity over static DataEvolver. The AnyText benchmark provides a diverse set of text-rich images, including rare font types that challenge the Verifier. Word accuracy (OCR) directly measures the primary goal: correct text generation. DataEvolver baseline tests the overall improvement; fixed threshold & prompt isolates the effect of meta-adaptation; random search tests if Bayesian optimization is superior to random exploration; meta-RL tests if a learned policy could be more efficient; the ablation tests the necessity of the diversity term. Human evaluation on a held-out subset directly tests the load-bearing assumption that Verifier scores correlate with human judgment. If MetaEvolver outperforms all baselines on word accuracy and human evaluation, especially on rare font subsets, the claim is supported. Conversely, if it does not significantly beat DataEvolver or if human evaluation shows misalignment, the claim is falsified.

### Expected outcome and causal chain

**vs. DataEvolver** — On a case where the Verifier's rejection threshold is too strict (e.g., generating rare font styles like blackletter), DataEvolver discards many valid samples because its fixed threshold rejects anything below a high score, leading to low diversity and missed high-quality rare samples. Our method adjusts the threshold dynamically via Bayesian optimization, setting it lower in rounds where scores are low to retain more samples, thus increasing diversity and eventually yielding more good samples. Additionally, the periodic calibration corrects for Verifier misalignment. We expect a noticeable gap in word accuracy on rare font subsets (e.g., script fonts) but parity on common fonts. For example, on blackletter fonts, we expect MetaEvolver to achieve 20% higher word accuracy than DataEvolver.

**vs. Fixed threshold & prompt** — On a case where the initial prompt template is suboptimal (e.g., it asks for "correct any errors" but generates overly simple images), the fixed version cannot adapt, causing the Critic to consistently give unhelpful feedback, limiting improvement across rounds. Our method's MLA selects prompt templates from a set based on past performance, enabling it to switch to a better template (e.g., "add details") that guides generation toward higher quality. We expect a clear divergence in word accuracy over rounds, with our method improving by +15% from round 1 to round 10 while the fixed one plateaus at round 5.

**vs. Random hyperparameter search** — On a case where the objective landscape is smooth (e.g., threshold values around 0.5 yield similar results), random search may waste rounds on poor configurations, achieving suboptimal performance within a limited number of rounds (M=10). Bayesian optimization with Expected Improvement efficiently focuses on promising regions, converging to good hyperparameters in fewer rounds. Thus we expect our method to achieve higher final word accuracy (e.g., 5% higher) and lower variance across runs (std dev <0.02 vs random search >0.05).

**vs. Meta-RL adaptive policy** — On a case where the optimal hyperparameters shift smoothly across rounds (e.g., threshold decreasing monotonically), meta-RL may learn a good policy quickly but struggles with discrete prompt selection due to the high-variance REINFORCE gradient. Bayesian optimization's sample-efficient acquisition handles discrete choices naturally. We expect our method to match or exceed meta-RL in final word accuracy (e.g., within 1%) but require fewer rounds to converge (e.g., 8 rounds vs 15 for meta-RL).

### What would falsify this idea
If the observed improvement of MetaEvolver over DataEvolver is uniform across all font types and text complexities, rather than concentrated on rare or challenging subsets predicted to be affected by threshold adaptation, then the central claim of automated quality assurance via hyperparameter tuning is unsupported. Alternatively, if MetaEvolver performs worse than the fixed threshold baseline, it would indicate that meta-adaptation harms performance. Additionally, if human evaluation shows that Verifier scores are inversely correlated with human judgment (e.g., Verifier prefers wrong text), then the entire framework collapses.

## References

1. DataEvolver: Self-Evolving Multi-Agent Data Construction for Text-Rich Image Generation
2. AnyText: Multilingual Visual Text Generation And Editing
3. AgentInstruct: Toward Generative Teaching with Agentic Flows
4. AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation
5. MetaMath: Bootstrap Your Own Mathematical Questions for Large Language Models
6. OCR-VQGAN: Taming Text-within-Image Generation
7. Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding
8. Character-Aware Models Improve Visual Text Rendering
