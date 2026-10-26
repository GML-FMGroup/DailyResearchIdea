# Consistency-Certified Self-Training: Filtering Self-Generated Privileged Traces by Trace-Level Agreement

## Motivation

On-Policy Self-Training (OST) and GKD both use self-generated privileged reasoning traces as feedback without any verification, so the correctness of the training signal is unestablished — OST explicitly assumes foundation models generate useful traces from coarse topics, and GKD assumes the teacher provides useful feedback on student-generated sequences. This structural gap recurs across both branches: no existing method certifies that a self-generated trace is internally coherent before using it as supervision, meaning errors in the trace are amplified rather than filtered during self-distillation.

## Key Insight

Trace-level agreement across independently sampled privileged traces is a data-free sufficient statistic for feedback reliability: if multiple on-policy traces reach the same final answer through mutually consistent reasoning steps, the probability that the shared answer is correct is strictly higher than for any single trace, and this certified subset can be used as feedback without an external oracle.

## Method

**What it is.** Consistency-Certified Self-Training (CCST) is a data-free self-training loop that augments On-Policy Self-Training with a trace-level consistency filter. Given a coarse topic description, the model generates a query set Q, then for each query q ∈ Q samples K=8 privileged reasoning traces T_q = {t_1, ..., t_K} at temperature τ=0.9. CCST outputs a filtered feedback set F = {(q, t*)} where t* is a trace whose final answer is agreed upon by at least m=4 of the K traces and whose intermediate reasoning steps are mutually consistent with the other agreeing traces; only F is used for on-policy self-distillation.

**Load-bearing assumption (stated explicitly).** CCST assumes that majority agreement across K=8 independently sampled privileged traces from the same base model is a consistent estimator of answer correctness: that the model's sampling distribution concentrates on the correct answer whenever it has latent competence, so answer-level consensus certifies correctness without an external oracle. This assumption fails when the model has a systematic bias toward a wrong answer (shared misconception, correlated pretraining bias), in which case the K traces are draws from one biased generator and majority agreement measures modal bias rather than truth. CCST does not claim to detect or correct this failure; it is the condition under which the filter admits biased traces, and we measure it empirically via the filter precision/recall calibration on the known-answer probe set (see experiment).

**How it works.**
```pseudocode
Input: base model π, topic descriptions C, K=8 samples, m=4 agreement threshold, N=20 queries per topic, R=3 rounds
Output: self-trained model π'

# Step-consistency definition (used in step 3):
#   For traces t and t' in G_q, parse each into an ordered list of reasoning steps
#   by splitting on newline-delimited step boundaries (e.g., "Step 1:", "Step 2:", ...).
#   Align steps by index (step i of t with step i of t').
#   step_overlap(t, t') = (1 / min(len(t), len(t'))) * sum_{i=1}^{min(len(t),len(t'))} 1[normalized_token_jaccard(t_i, t'_i) >= 0.5]
#   where normalized_token_jaccard lowercases, strips punctuation, and computes |A∩B|/|A∪B| over token sets.
#   step_consistency(t, G_q) = mean_{t' in G_q, t' != t} step_overlap(t, t').

1. Query synthesis:
   for each topic c in C:
       Q_c ← π.sample_queries(c, N=20)          # on-policy query generation

2. Privileged trace sampling:
   for each q in Q_c:
       T_q ← [π.sample_trace(q, temperature=0.9) for _ in range(K)]

3. Trace-level consistency filter:
   for each q in Q_c:
       A_q ← majority_answer(T_q)                # answer agreed by ≥ m traces
       if count(A_q) < m: discard q
       else:
           G_q ← {t in T_q : answer(t) == A_q}
           s* ← argmax_{t in G_q} step_consistency(t, G_q)  # max mean pairwise step-overlap (defined above)
           F ← F ∪ {(q, s*)}

4. On-policy self-distillation:
   for each (q, s*) in F:
       y ~ π(q)                                  # on-policy student rollout
       loss ← KL( π(y|q) || π(y | q, s*) )        # privileged trace as teacher
       π ← π - ∇ loss

5. Repeat steps 1–4 for R=3 rounds
```

**Why this design.** We chose trace-level majority voting over token-level self-consistency because token-level agreement conflates surface fluency with reasoning correctness — a model can be consistently wrong at the token level while diverging at the answer level, so answer-level agreement is the minimal sufficient statistic for correctness certification; the cost is that queries with no majority answer are discarded, reducing effective training data. We chose step-consistency ranking within the agreeing set over picking the first agreeing trace because it selects the trace whose intermediate reasoning is most representative of the consensus, reducing the chance that a lucky-but-flawed trace becomes the privileged signal; the cost is extra pairwise similarity computation over reasoning steps. We chose to discard non-agreeing queries entirely rather than down-weighting them because down-weighting still propagates unverified feedback, violating the reliability invariant; the cost is lower yield per topic, which we offset by generating 20 queries per topic. This design is not a variant of GKD: GKD assumes a teacher provides correct feedback on student sequences and never certifies that feedback, whereas CCST's certification step is the load-bearing mechanism that makes the feedback trustworthy without any teacher.

**Why it measures what we claim.** The computational quantity `count(A_q) ≥ m` measures feedback reliability because the assumption that independent samples from a well-calibrated model concentrate on the correct answer when the model has latent competence makes majority agreement a consistent estimator of correctness; this assumption fails when the model has a systematic bias toward a wrong answer (e.g., a common misconception), in which case majority agreement reflects shared bias rather than correctness. The quantity `step_consistency(t, G_q)` measures trace-level coherence because the assumption that correct reasoning traces share intermediate conclusions (e.g., the same sub-result at each arithmetic step) makes step-overlap a proxy for logical validity; this assumption fails when correct traces legitimately take different solution paths, in which case the metric reflects stylistic similarity rather than correctness. Together these two quantities operationalize the motivation-level concept of 'certified feedback reliability' under the named assumptions, and the failure modes are exactly the conditions under which the filter would admit biased or stylistically selected traces — which we detect via a held-out probe set of known-answer queries.

## Contribution

['A data-free consistency-certification module for self-training that filters on-policy privileged traces by answer-level majority agreement and step-level coherence before using them as feedback, establishing a reliability invariant absent in OST and GKD.', 'A concrete design principle: trace-level agreement is a stronger certification signal than token-level self-consistency for reasoning feedback, because it operates on the answer surface where correctness is defined rather than on fluency.', "An empirical demonstration that certified self-training improves over uncertified OST on reasoning benchmarks while retaining performance on non-self-trained domains, with ablation showing the filter's yield-accuracy trade-off."]

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | GSM8K + held-out known-answer probe set (512 queries with verified final answers, disjoint from training topics) | Arithmetic reasoning exposes trace-level consistency failures; probe set enables filter precision/recall calibration |
| Primary metric | Accuracy on held-out probe split | Directly tests certified-feedback reliability, not fluency |
| Secondary metric | Filter precision (fraction of admitted traces whose final answer is correct) and recall (fraction of correct-answer queries admitted) on the probe set, as a function of m ∈ {2,3,4} and K ∈ {4,8,16} | Empirically grounds the reliability claim rather than assuming it |
| Baseline | On-Policy Self-Training (no filter) | Tests whether certification filter is load-bearing |
| Baseline | Supervised KD on teacher traces | Tests data-free certification vs fixed teacher signal |
| Baseline | GKD (off-policy, teacher feedback) | Tests whether teacher-free certification matches distillation |
| Baseline | Majority-vote-filtered rejection sampling fine-tuning (answer-only filter, no step ranking) | Direct head-to-head against self-consistency-filtered self-training; isolates whether step-consistency certification beats answer-only filtering |
| Ablation-of-ours | CCST with answer-majority only (no step ranking) | Isolates step-consistency contribution |
| Ablation-of-ours | CCST with K ∈ {4,8,16} and m ∈ {2,3,4} (cost-accuracy sweep) | Shows gains are not simply bought with more samples |

### Why this setup validates the claim

The central claim is that trace-level consistency certification converts unreliable self-generated feedback into trustworthy privileged signal without any teacher. The no-filter On-Policy Self-Training baseline isolates the certification step: if CCST beats it, the filter is doing real work rather than the loop merely benefiting from more self-generated data. The supervised KD baseline tests the data-free sub-claim — if CCST approaches it while using no ground-truth answers, the certification substitutes for external supervision. GKD tests the teacher-free sub-claim against a strong distillation reference. The majority-vote-filtered rejection sampling baseline is a direct head-to-head against self-consistency-filtered self-training: if CCST beats it by a statistically significant margin (paired bootstrap, p<0.05) on the probe set, the step-consistency mechanism is not just self-consistency rebranded. The answer-majority-only ablation isolates whether step-consistency ranking contributes beyond answer voting, which is the specific mechanism the design defends. The filter precision/recall calibration on the known-answer probe set measures the fraction of admitted traces that are actually correct as a function of m and K, grounding the reliability claim empirically rather than assuming it. The cost-accuracy sweep over K and m shows whether gains are simply bought with more samples. Accuracy on the held-out known-answer probe set is the right primary metric because it is exactly where the predicted failure modes (shared bias, lucky-but-flawed traces) manifest as wrong answers; a metric like perplexity or trace-level fluency would not distinguish certified from uncertified feedback.

### Expected outcome and causal chain

**vs. On-Policy Self-Training (no filter)** — On a case where the model samples a plausible but flawed trace that reaches a wrong answer, the unfiltered baseline distills that trace as privileged feedback because it accepts every sampled trace without certification. Our method instead discards that query or selects a trace whose answer is agreed by ≥4 samples and whose steps overlap the agreeing set, so the flawed trace never becomes teacher signal. We expect a noticeable accuracy gap on the probe subset where the base model's single-sample accuracy is low but its majority-vote accuracy is high, with near-parity where single-sample accuracy is already high.

**vs. Supervised KD on teacher traces** — On a case where the fixed teacher trace takes a solution path mismatched to the student's on-policy distribution, supervised KD forces the student toward an off-policy sequence it would not generate, producing brittle imitation. Our method instead certifies traces sampled from the student's own distribution and distills on-policy rollouts against a consensus-ranked trace, so the privileged signal stays in-distribution. We expect CCST to close most of the gap to supervised KD on probe accuracy despite using no ground-truth answers, with the residual gap concentrated on queries the base model cannot majority-solve at all.

**vs. GKD (off-policy, teacher feedback)** — On a case where the teacher's feedback is correct but the student's self-generated sequence diverges early, GKD's teacher signal assumes the sequence is valid and provides no certification, propagating feedback on a bad prefix. Our method instead certifies the privileged trace before using it and only trains on queries where consensus exists, so feedback is gated by reliability. We expect CCST to match or slightly trail GKD where a strong teacher exists but to outperform on probe subsets where teacher feedback is unavailable, and the answer-majority-only ablation to fall between the two, showing step-consistency adds a smaller but real margin.

**vs. Majority-vote-filtered rejection sampling fine-tuning (answer-only filter)** — On a case where multiple traces agree on the correct final answer but some reach it through flawed intermediate steps (lucky-but-flawed traces), the answer-only filter admits all agreeing traces indiscriminately, including the flawed ones, and fine-tunes on them. Our method ranks within the agreeing set by step-consistency and selects the trace whose intermediate reasoning is most representative of the consensus, so the privileged signal is the most coherent agreeing trace rather than an arbitrary one. We expect CCST to beat this baseline by a statistically significant margin (paired bootstrap, p<0.05) on the probe subset where answer agreement is high but step-level coherence varies, and to match it where all agreeing traces are step-consistent.

### What would falsify this idea

If CCST's gain over unfiltered On-Policy Self-Training is uniform across probe subsets rather than concentrated on the low-single-sample-accuracy / high-majority-accuracy subset where the certification mechanism is predicted to bite, the central claim that trace-level certification is the load-bearing mechanism is wrong. Equivalently, if the answer-majority-only ablation matches full CCST everywhere, or if the majority-vote-filtered rejection sampling baseline matches CCST on the step-coherence-variable subset, step-consistency ranking contributes nothing and the design's second mechanism is unsupported.

## References

1. On-Policy Self-Training: Data-Free Self-Improvement in Large Language Models
2. On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes
3. Explanations from Large Language Models Make Small Reasoners Better
