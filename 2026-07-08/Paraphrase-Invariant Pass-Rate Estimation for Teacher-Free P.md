# Paraphrase-Invariant Pass-Rate Estimation for Teacher-Free Prompt Selection in Reinforcement Learning

## Motivation

Existing prompt selection methods for reinforcement learning, such as TREK, rely on ground-truth answers to measure pass-rate, which is unavailable in open-ended tasks. This structural bottleneck arises because automatic correctness signals cannot be defined for tasks like creative writing or open-ended QA, forcing RL training to either omit prompt selection or assume a reliable reward signal that does not exist.

## Key Insight

Paraphrase invariance is a reliable teacher-free pass-rate signal because a well-defined prompt constrains the output space to a semantic region, while an ambiguous prompt allows multiple interpretations, leading to divergent outputs — a structural property of language tasks that prior work ignored.

## Method

(A) **What it is**: PIPE (Paraphrase-Invariant Pass-rate Estimator) estimates the pass-rate of a language model on a given prompt by measuring the semantic consistency of the model's outputs across multiple paraphrases of that prompt, enabling prompt selection for reinforcement learning without ground-truth answers. Input: a prompt p, a student model M, a paraphrase generation model G, a semantic similarity function S. Output: a scalar pass-rate estimate ρ(p).

(B) **How it works** (pseudocode):

```python
def estimate_pass_rate(p, M, G, S, K=5, N=3):
    # Step 1: Generate paraphrases
    paraphrases = G.generate(p, K)  # K = 5
    # Step 2: Sample responses for each paraphrase
    all_responses = []
    for p_i in paraphrases:
        responses = M.sample(p_i, N)  # N = 3
        all_responses.extend(responses)
    # Step 3: Compute pairwise semantic similarity
    similarities = []
    for i in range(len(all_responses)):
        for j in range(i+1, len(all_responses)):
            sim = S(all_responses[i], all_responses[j])  # e.g., BERTScore
            similarities.append(sim)
    # Step 4: Aggregate to consistency score
    ρ = mean(similarities)  # pass-rate estimate
    return ρ
```

(C) **Why this design**: We chose to generate multiple paraphrases (K=5) rather than a single one to reduce variance from a potentially poor paraphrase, accepting the computational cost of additional inference. We sample N=3 responses per paraphrase to capture the model's output distribution, trading off estimation accuracy for compute; a larger N would improve reliability but increase cost. We aggregate pairwise similarities via mean rather than min or max because mean smooths outliers and reflects overall consistency, whereas min would be too conservative and max too optimistic. We use BERTScore as the semantic similarity function S because it captures semantic equivalence better than n-gram overlap, though it requires a pretrained encoder, which is a minor resource overhead. This design balances computational efficiency with estimation accuracy for the pass-rate proxy, and diverges from TREK's answer-based pass-rate by substituting a teacher-free signal, accepting that the proxy may be noisy but enabling operation in open-ended domains.

(D) **Why it measures what we claim**: The computational quantity `mean(similarities)` measures *paraphrase invariance* which we claim estimates pass-rate because the assumption is that a prompt with a clear, unambiguous instruction will produce semantically consistent outputs regardless of wording variations, while an ambiguous prompt will lead to divergent interpretations. This assumption fails when the model has systematic biases that cause it to produce consistent but incorrect outputs across paraphrases — in that case, `mean(similarities)` reflects output consistency rather than correctness, and the pass-rate estimate may be falsely high. Conversely, if the model is highly stochastic, even a clear prompt may produce low similarity, leading to a falsely low estimate. Despite these failure modes, the proxy remains useful for prompt selection because low consistency reliably indicates problematic prompts, while high consistency at least indicates that the model interprets the prompt reliably, even if not correctly. The method operationalizes the motivation-level concept of *pass-rate* without needing ground truth, substituting semantic consistency as a necessary condition for correctness.

## Contribution

(1) Introduces PIPE, a method for estimating pass-rate in open-ended tasks using paraphrase invariance, enabling teacher-free prompt selection for reinforcement learning. (2) Demonstrates that semantic consistency across paraphrases serves as a valid proxy for pass-rate, providing a structural property that prior prompt selection methods (e.g., TREK) lack. (3) Proposes a practical algorithm for integrating PIPE into RL training loops to prioritize high-pass-rate prompts without ground-truth answers.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | GSM8K | Known answers enable ground-truth pass-rate |
| Primary metric | Spearman correlation | Measures rank consistency of estimate |
| Baseline 1 | TREK | Uses teacher to compute pass-rate |
| Baseline 2 | DAPO | On-policy RL without pass-rate estimation |
| Baseline 3 | SeqKD | Off-policy distillation baseline |
| Ablation | PIPE w/o paraphrase | Uses single prompt, not multiple paraphrases |

### Why this setup validates the claim

GSM8K provides ground-truth pass-rates per prompt, allowing direct comparison of our proxy against the true metric. Comparing against TREK tests whether our teacher-free proxy can match or exceed a teacher-dependent approach, isolating the value of paraphrase invariance. DAPO serves as a baseline that relies on outcome rewards, highlighting our ability to function without ground truth. SeqKD tests against off-policy distillation, underscoring the importance of on-policy consistency. The ablation removes the core mechanism—multiple paraphrases—to confirm that the gain stems from measuring consistency across rephrasings. Spearman correlation captures rank-order agreement, crucial for prompt selection, and avoids calibration bias.

### Expected outcome and causal chain

**vs. TREK** — On a prompt where the teacher model has a systematic bias (e.g., always outputs a correct-looking but wrong answer), TREK's teacher-derived pass-rate is falsely high because the teacher's answer appears correct. Our method instead produces a low estimate because bias-induced consistency across paraphrases drops due to subtle semantic drift, revealing ambiguity. We expect a noticeable gap on prompts with teacher bias (our estimate lower) but parity on clear prompts.

**vs. DAPO** — On an open-ended prompt without a single correct answer (e.g., "explain why the sky is blue"), DAPO cannot compute an outcome reward and thus fails entirely. Our method produces a valid consistency score regardless of answer existence. We expect our method to maintain meaningful prompt rankings on such prompts while DAPO yields undefined or uniform pass-rates.

**vs. SeqKD** — On a prompt where the student's own distribution diverges from the teacher's fixed outputs (model collapse), SeqKD ignores the student's variability and assigns a spuriously high pass-rate based on teacher confidence. Our method captures the student's actual consistency across paraphrases, revealing uncertainty. We expect our estimate to be lower on collapse-prone prompts, with a clear separation on those where student distribution is wide.

### What would falsify this idea

If Spearman correlation between our proxy and true pass-rate is low (≤0.2) even on clear prompts, or if the ablation (single paraphrase) performs equally well, then paraphrase invariance is not sufficient for pass-rate estimation. Also, if our proxy shows high correlation on prompts where the model is consistently wrong (systematic bias), the assumption that consistency implies correctness is invalid.

## References

1. TREK: Distill to Explore, Reinforce to Refine
2. Black-Box On-Policy Distillation of Large Language Models
3. DAPO: An Open-Source LLM Reinforcement Learning System at Scale
4. VinePPO: Refining Credit Assignment in RL Training of LLMs
5. On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes
6. Large Language Models Are Reasoning Teachers
7. GEAR: A GPU-Centric Experience Replay System for Large Reinforcement Learning Models
8. PowerInfer: Fast Large Language Model Serving with a Consumer-grade GPU
