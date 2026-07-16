# Diverse Proxy Exploration with Uncertainty-Aware Intrinsic Rewards for Scalable LLM Post-Training

## Motivation

In the PUST framework (Proxy Exploration and Reusable Guidance), a lightweight proxy model explores to discover high-reward behaviors, but its limited capacity may cause it to miss critical high-reward regions, leading to suboptimal guidance signals for the primary model. Existing exploration bonuses (e.g., count-based) are ineffective for language model action spaces due to sparsity, and random exploration insufficiently covers the rich reward landscape. The root cause is that the proxy's exploration strategy does not explicitly account for its own epistemic uncertainty about which behaviors are high-reward, resulting in coverage gaps that propagate to the primary model's optimization.

## Key Insight

Ensemble disagreement among proxy copies directly quantifies epistemic uncertainty about the reward landscape, and intrinsically rewarding this uncertainty ensures exploration of regions that are both uncertain and likely to contain high-reward behaviors, naturally compensating for the proxy's capacity limitation.

## Method

### Method: DPEU (Diverse Proxy Exploration via Uncertainty)

**(A) What it is:** DPEU augments the proxy model's reward signal with an intrinsic exploration bonus calculated from an ensemble of proxy copies. The ensemble's diversity on a given query serves as a proxy for epistemic uncertainty, guiding the proxy to explore behaviors where its predictions are most uncertain.

**Load-bearing assumption:** We assume that ensemble disagreement (ROUGE-L diversity) among perturbed proxy copies directly quantifies epistemic uncertainty about which response is high-reward, and that this uncertainty signal is stable over training despite periodic parameter averaging. We verify this post-training by computing the Pearson correlation between d(s) and the reward prediction error (|R(s, a) - predicted reward|) on a held-out calibration set of 512 queries. If the correlation is below 0.3, we replace ROUGE-L diversity with query-by-committee entropy (disagreement measured as entropy over predicted reward scores).

**(B) How it works (pseudocode):**
```python
# Inputs: proxy policy π_θ (0.5B parameter LLM), reward model R, ensemble size K=5, coefficient β=1.0, training queries D

# Initialize K copies with independent parameter perturbations
θ_1, ..., θ_K = perturb(θ, noise_scale=0.01)

for step in range(1, T+1):
    sample batch B ~ D
    for s in B:
        # Each ensemble member generates a response
        a_1 ~ π_θ_1(·|s), ..., a_K ~ π_θ_K(·|s)
        
        # Compute diversity score d(s) = 1 - mean pairwise ROUGE-L F1 among {a_k}
        d(s) = 1 - (1 / (K*(K-1))) * sum_{i<j} ROUGE-F1(a_i, a_j)
        
        for k in 1..K:
            utility = R(s, a_k) + β * d(s)
            # Update ensemble member with PPO using utility as reward
            update π_θ_k with PPO on (s, a_k, utility)
    
    # Synchronize ensemble every 10 steps: set all θ_k to average of current parameters
    if step % 10 == 0:
        avg = mean(θ_1, ..., θ_K)
        for k in 1..K: θ_k = avg + small noise (e.g., N(0, 0.001))

# After training, select the ensemble member with highest average reward as the optimized proxy
# Calibration step (post-training):
# Compute pearson_r = correlation(d(s), |R(s,a) - predicted_reward|) on 512 held-out queries
# If pearson_r < 0.3, switch diversity metric to query-by-committee entropy (QBC)
```

**(C) Why this design:** We chose an ensemble over a single-model count-based bonus because language action spaces are sparse and combinatorial, making count-based methods (e.g., count of n-grams) unreliable and expensive. Ensemble disagreement provides a direct measure of epistemic uncertainty without requiring a separate dynamics model. We selected ROUGE-L diversity instead of prediction variance because the reward model is not a per-token scalar; ROUGE captures semantic diversity in open-ended generation. We introduce periodic averaging of ensemble parameters to prevent mode collapse from divergent solutions, which would render disagreement a measure of variance rather than uncertainty. The trade-off is a K-fold increase in forward passes during proxy exploration, but the proxy is already lightweight (0.5B parameters, ~500 GPU hours on A100-80GB with DeepSpeed ZeRO-3), and the resulting guidance signals are reusable across primary models. We deliberately avoid a separate intrinsic motivation module (e.g., ICM) because that would introduce additional learning targets and potential bias; using the proxy's own ensemble ensures the bonus aligns with the proxy's knowledge boundary. A post-training calibration checks whether d(s) correlates with reward prediction error; if not, we fall back to QBC entropy to preserve the uncertainty measure.

**(D) Why it measures what we claim:** The diversity score d(s) measures epistemic uncertainty about which response is optimal for query s because ensemble members, initialized differently, will agree when the proxy has learned a robust preference (low d) and disagree near decision boundaries where multiple responses appear plausible (high d); this assumption fails when early synchronization forces consensus despite high uncertainty, in which case d reflects ensemble correlation rather than true uncertainty. The reward R(s,a) measures the oracle's assessment of response quality, and the utility R + β*d operationalizes the trade-off between exploitation of known high-reward regions and exploration of uncertain ones. We assume that high-reward behaviors are initially uncertain for a capacity-limited proxy; this assumption fails when a high-reward region is already well-covered (low d) but the proxy still fails to generalize to the primary model's distribution, in which case the bonus provides no extra coverage. The coefficient β controls the strength of the coverage guarantee; a larger β forces more exploration but may distract from clear high-reward behaviors. The post-training calibration (Pearson correlation between d(s) and reward prediction error) provides a check on the core assumption: if correlation is low, we replace d(s) with a more reliable uncertainty score (QBC entropy).

## Contribution

(1) A diverse proxy exploration algorithm, DPEU, that uses ensemble disagreement as an intrinsic reward to improve coverage of high-reward behaviors in capacity-limited proxies. (2) An empirical finding that adding uncertainty-aware exploration to proxy exploration yields guidance signals that enable better primary model alignment on reasoning tasks compared to standard PUST. (3) Practical design principles for ensemble-based exploration with language models, including diversity metric selection and synchronization scheduling.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|----------------------|
| Dataset | GSM8K, MATH | Math reasoning requires diverse solution paths; MATH is harder |
| Primary metric | Accuracy on test set | Standard measure of reasoning ability |
| Secondary metric | Pearson correlation between d(s) and reward improvement on held-out set | Validate uncertainty quantification |
| Baseline | Vanilla PPO | Proxy trained only with reward signal |
| Baseline | PPO with n-gram count bonus | Count-based exploration baseline |
| Baseline | PPO with ensemble variance bonus | Alternative uncertainty-driven exploration |
| Baseline | PPO with Random Network Distillation (RND) | Prediction-error-based exploration without ensemble |
| Ablation | DPEU without ensemble sync | Tests necessity of periodic averaging |

Compute resource estimates: Training on A100-80GB GPUs with DeepSpeed ZeRO-3, proxy model has 0.5B parameters. Total compute: ~500 GPU hours for training on 100k query-sample (GSM8K+MATH). Memory: ~20GB for proxy ensemble (5 copies) with gradient checkpointing.

### Why this setup validates the claim
This setup tests whether the diversity bonus from ensemble disagreement improves proxy exploration over standard exploitation (Vanilla PPO), count-based exploration, prediction-variance-based exploration, and RND-based exploration. The GSM8K and MATH datasets provide queries with varying solution diversity, allowing us to measure if gains concentrate on high-uncertainty queries. The primary metric (accuracy) directly reflects the end task performance of the primary model trained with proxy-generated data. The secondary metric (correlation between d(s) and reward improvement) directly validates the load-bearing assumption that ensemble disagreement captures epistemic uncertainty. The ablation isolates the impact of the synchronization mechanism, which prevents mode collapse. By comparing against multiple exploration baselines, we can attribute improvements to the specific design choices of DPEU rather than any exploration bonus.

### Expected outcome and causal chain

**vs. Vanilla PPO** — On a query where the proxy's reward model is uncertain (e.g., multiple plausible reasoning paths), vanilla PPO repeatedly generates similar low-reward responses because it greedily exploits the highest reward seen, failing to explore. Our method instead uses ensemble disagreement as an exploration bonus to generate diverse responses, discovering higher-reward solutions. We expect a noticeable accuracy gap on queries with high ensemble diversity (e.g., >0.3 ROUGE-L diversity) but parity on low-diversity queries.

**vs. PPO with n-gram count bonus** — On a query with many possible surface forms (e.g., paraphrasing, different variable names), count-based bonus becomes unreliable and sparse, leading to insufficient exploration. Our method uses semantic diversity via ROUGE-L, which captures meaning-level variation, ensuring robust exploration. We expect DPEU to outperform on queries with high lexical variability, showing a larger gap (e.g., 5-10% accuracy difference) on such subsets.

**vs. PPO with ensemble variance bonus** — On a query where ensemble members collapse to similar outputs due to identical training, variance bonus is low even if true uncertainty is high (assuming collapse). Our ensemble periodically averages parameters and adds noise, maintaining diversity and preventing collapse. Thus, DPEU retains exploration drive on ambiguous queries. We expect DPEU to maintain accuracy on ambiguous queries while variance-based method degrades (e.g., 3-5% drop on high-diversity queries).

**vs. PPO with RND** — On a query where prediction error is low but multiple high-reward responses exist (e.g., multiple valid solution paths), RND exploration stalls because the prediction error of a random network is low for already-seen state-action pairs. DPEU's ensemble disagreement captures semantic diversity even when prediction error is low, leading to better exploration. We expect DPEU to outperform RND on queries with multiple distinct high-reward outputs, with a 2-4% accuracy gap.

**Correlation metric:** We expect a Pearson correlation of at least 0.5 between d(s) and reward improvement (relative to initial proxy) on the calibration set. If correlation is below 0.3, our method will switch to QBC entropy, which should restore the expected improvements.

### What would falsify this idea
If DPEU's accuracy improvement over vanilla PPO is uniform across all queries rather than concentrated on subsets with high ensemble diversity, or if the ablation without sync performs equally well, or if the Pearson correlation between d(s) and reward improvement is below 0.3 (even after fallback to QBC), then the central claim that the diversity bonus and synchronization drive exploration is invalid.

## References

1. Proxy Exploration and Reusable Guidance: A Modular LLM Post-Training Paradigm via Proxy-Guided Update Signals
2. Incentivizing Strong Reasoning from Weak Supervision
3. Weak-to-Strong Generalization beyond Accuracy: a Pilot Study in Safety, Toxicity, and Legal Reasoning
4. LegalBench: A Collaboratively Built Benchmark for Measuring Legal Reasoning in Large Language Models
5. Weak-to-Strong Generalization: Eliciting Strong Capabilities With Weak Supervision
6. Discovering Language Model Behaviors with Model-Written Evaluations
7. Constitutional AI: Harmlessness from AI Feedback
8. Measuring Progress on Scalable Oversight for Large Language Models
