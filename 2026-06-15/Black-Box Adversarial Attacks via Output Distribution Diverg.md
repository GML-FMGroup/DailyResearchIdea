# Black-Box Adversarial Attacks via Output Distribution Divergence Maximization

## Motivation

Existing black-box attacks either require internal model information (e.g., logits in REINFORCE Adversarial Attacks) or rely on costly iterative refinement with an auxiliary LLM (e.g., TAP, PAIR). Directly optimizing a divergence measure between the output distributions of benign and adversarial inputs using a gradient-free optimizer provides a principled and efficient alternative that requires only output probabilities, but this approach has not been explored due to the assumption that gradient-free optimization is sample-inefficient on high-dimensional token spaces.

## Key Insight

KL divergence between output distributions provides a smooth, query-efficient objective for gradient-free optimization because it captures the model's internal probability shift towards harmful completions without requiring gradient information.

## Method

### (A) What it is
We propose **Black-box Divergence Maximization (BDM)**, an adversarial attack that optimizes an adversarial suffix by maximizing the KL divergence between the target model's output probability distribution for a benign query and that for the adversarial query (benign + suffix), using a gradient-free optimizer (CMA-ES). Input: benign query, initial suffix, target model providing output probabilities. Output: optimized adversarial suffix.

### (B) How it works
```python
import cma
import numpy as np

def kl_divergence(p, q):
    return np.sum(p * np.log(p / q))

def attack(model, benign_query, init_suffix, max_iter=100, pop_size=10, sigma=0.1, vocab_size=32000):
    # init_suffix is a list of token indices of length L
    L = len(init_suffix)
    # We optimize a continuous matrix of logits of shape (L, vocab_size)
    # Initial logits correspond to init_suffix (one-hot like, but with small noise)
    init_logits = np.random.randn(L, vocab_size) * 0.01
    # Set initial logits to high value for the given tokens
    for i, token in enumerate(init_suffix):
        init_logits[i, token] = 10.0
    # Flatten for CMA-ES
    dim = L * vocab_size
    
    def decode(logits_flat, tau=1.0):
        logits = logits_flat.reshape(L, vocab_size)
        # Use Gumbel-Softmax for differentiable sampling during optimization; at test time use argmax
        if tau == 'argmax':
            tokens = np.argmax(logits, axis=1)
        else:
            gumbel = np.random.gumbel(size=logits.shape)
            soft = np.argmax(logits + gumbel, axis=1)  # straight-through Gumbel-Softmax
            # For continuous relaxation, we pass probabilities
            tokens = soft  # detach for forward pass but gradient flows through logits
        return tokens.tolist()
    
    def f(logits_flat):
        suffix = decode(logits_flat, tau=1.0)
        # Get first-token output probabilities for benign and adversarial queries
        p_benign = model.get_next_token_probs(benign_query)  # shape (vocab_size,)
        p_adv = model.get_next_token_probs(benign_query + suffix)  # shape (vocab_size,)
        # Avoid log(0) by clipping
        p_benign = np.clip(p_benign, 1e-10, 1.0)
        p_adv = np.clip(p_adv, 1e-10, 1.0)
        kl = kl_divergence(p_benign, p_adv)
        return -kl  # minimize negative KL
    
    es = cma.CMAEvolutionStrategy(init_logits.flatten(), sigma, {'popsize': pop_size, 'maxiter': max_iter, 'verbose': -1})
    while not es.stop():
        solutions = es.ask()
        fitness = [f(x) for x in solutions]
        es.tell(solutions, fitness)
    best_logits = es.result.xbest
    suffix = decode(best_logits, tau='argmax')
    return suffix
```

### (C) Why this design
We chose KL divergence over other divergences (e.g., JS divergence) because KL is asymmetric and using the benign distribution as reference maximizes the new information in the adversarial distribution, which correlates with unlikely (often harmful) content. We chose CMA-ES over Bayesian optimization because CMA-ES adaptively scales perturbation magnitude, reducing manual tuning in high-dimensional token spaces; the cost is that CMA-ES requires more evaluations per iteration (pop_size ≈ 10), but this is acceptable given typical query budgets (hundreds). We chose to only use the first token distribution rather than a sequence of tokens to keep the objective computationally cheap and focus on early signals of harmfulness; this sacrifices potential attacks that only become harmful later, but empirical evidence (e.g., GCG) shows first-token optimization is effective. We encode suffix as a continuous matrix of logits to leverage CMA-ES's Gaussian sampling; this introduces a relaxation gap but avoids combinatorial explosion, and we mitigate by using a high temperature during conversion. We assume that KL divergence between first-token output distributions correlates with harmfulness; we validate this via a calibration study (see Experiment).

### (D) Why it measures what we claim
The computed KL divergence between p_benign and p_adv measures the shift in the model's first-token output distribution. We assume that this shift correlates with the probability of generating a harmful response. This assumption fails when the distribution shift is due to topic change rather than harmfulness, in which case the KL divergence reflects topic shift instead. To verify this assumption, we compute Spearman correlation between KL divergence and harmfulness scores from a classifier (e.g., Llama Guard) on a calibration set of 500 prompts; if ρ < 0.5, we reject the method.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | HarmBench (500 held-out for calibration, 1000 for testing) | Standard benchmark for jailbreak attacks |
| Primary metric | Attack Success Rate (ASR) | Direct measure of jailbreak effectiveness |
| Baseline 1 | GCG (greedy coordinate gradient, white-box) | Strong gradient-based baseline |
| Baseline 2 | AdvPrefix (prefix optimization) | Optimized prefix using RL objective |
| Baseline 3 | LLMStinger (RL fine-tuned attacker) | RL fine-tuned LLM attacker |
| Ablation-of-ours | BDM with JS divergence | Tests KL divergence advantage |

### Why this setup validates the claim
HarmBench provides a diverse set of harmful behaviors covering multiple categories, ensuring the evaluation is comprehensive. We split it into a calibration set of 500 prompts (used to validate the KL-harmfulness correlation via Spearman ρ with Llama Guard scores; we require ρ ≥ 0.5) and a test set of 1000 prompts for final ASR. ASR is the standard metric for jailbreak attacks, directly measuring how often the model produces harmful content. Including GCG tests whether our black-box method can compete with gradient-based attacks; AdvPrefix tests whether optimizing a suffix is better than optimizing a prefix; LLMStinger tests against adaptive RL-based attacks. The ablation with JS divergence isolates the effect of using KL divergence, verifying that our specific divergence choice drives performance. This combination creates a falsifiable test: if BDM outperforms GCG on black-box settings, surpasses AdvPrefix on nuanced requests, and beats LLMStinger on out-of-distribution prompts, while the JS ablation performs worse, then the central claim is supported. Additionally, the calibration study provides explicit evidence that the load-bearing assumption holds.

### Expected outcome and causal chain

**vs. GCG** — On a black-box scenario where the target model does not provide gradients (e.g., API-only access to Llama-2-7B-chat), GCG fails because it requires differentiable model weights. BDM uses only output probabilities and CMA-ES. We expect BDM to achieve ASR of 65% vs GCG's 45% (ASR measured on HarmBench test set with a trained judge LLM). On white-box settings (gradients available), we expect parity (~60% ASR for both). The causal chain: BDM's query-efficient KL divergence optimization discovers adversarial suffixes without gradient information.

**vs. AdvPrefix** — On nuanced jailbreak requests (e.g., asking for step-by-step instructions with ethical disclaimers), AdvPrefix generates a fixed-style prefix that may trigger safety filters. BDM's suffix blends naturally with the query. We expect BDM to achieve ASR of 70% vs AdvPrefix's 55% on nuanced categories (e.g., 'weapons' and 'hate' in HarmBench). On simple requests (e.g., 'how to pick a lock'), both should achieve >80% ASR. The causal chain: BDM's per-instance optimization adapts suffix to context.

**vs. LLMStinger** — On out-of-distribution prompts (e.g., novel harmful scenarios unseen during LLMStinger's RL training), LLMStinger's transferable patterns degrade. BDM adapts per instance using the target model's outputs. We expect BDM to achieve ASR of 68% vs LLMStinger's 58% on OOD categories (e.g., 'privacy violations' not in training). On seen categories, both around 75% ASR. The causal chain: BDM's black-box optimization does not rely on transferable patterns from a surrogate model.

### What would falsify this idea
If BDM's ASR is not significantly higher than GCG on black-box models (e.g., less than 10% gap), or if the JS divergence ablation matches BDM (gap < 5%), or if the calibration study yields ρ < 0.5, then the central claim that KL divergence is key for black-box attack success is falsified.

## References

1. REINFORCE Adversarial Attacks on Large Language Models: An Adaptive, Distributional, and Semantic Objective
2. AdvPrefix: An Objective for Nuanced LLM Jailbreaks
3. LLMStinger: Jailbreaking LLMs using RL fine-tuned LLMs
4. Tree of Attacks: Jailbreaking Black-Box LLMs Automatically
5. Universal and Transferable Adversarial Attacks on Aligned Language Models
6. Jailbreaking Black Box Large Language Models in Twenty Queries
7. Self-Instruct: Aligning Language Models with Self-Generated Instructions
