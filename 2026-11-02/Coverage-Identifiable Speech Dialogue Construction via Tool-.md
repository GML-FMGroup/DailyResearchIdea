# Coverage-Identifiable Speech Dialogue Construction via Tool-Action Support Maximization

## Motivation

Auto-constructed speech-conditioned dialogue pipelines such as the one in Apriel Voice 27B Preview generate conversational training data without any guarantee that the resulting distribution covers the tool-call action space the model will face at inference; rare tool-use sequences are structurally under-represented because the generator samples from whatever the teacher or seed dialogues happen to produce, so adaptation is under-constrained exactly where it matters most. This is a meta-level failure that recurs across every pipeline that treats data construction as sampling rather than as support coverage: the constructed distribution's support is never audited against the inference-time tool-use distribution, and no existing method (Apriel Voice's automatic dialogue construction being a concrete instance) enforces that rare action sequences are reachable with non-negligible probability. The root cause is that construction objectives optimize plausibility or fidelity of individual dialogues, not the coverage of the discrete action space those dialogues induce.

## Key Insight

Treating the tool-call action space as an explicit finite support and maximizing a divergence-based coverage regularizer over that support during dialogue construction yields a speech-conditioned dialogue distribution whose support provably contains the inference-time tool-use distribution, because coverage of a finite discrete space is a support-containment property that plausibility-based sampling cannot supply but a regularized construction objective can.

## Method

**What it is.** COIN (Coverage-Identifiable dialogue construction) is a data-construction procedure that takes as input a speech-conditioned seed corpus, a tool schema defining a finite action space A (all tool-call sequences up to length L), and a teacher speech-language model, and outputs a set of constructed dialogues D whose induced tool-call distribution p_D has support covering A, together with a per-sequence coverage certificate. **Load-bearing assumption (stated explicitly):** COIN assumes that conditioning the teacher on a forced tool-call action prefix yields a dialogue that (i) realizes that exact action sequence, (ii) executes successfully in the tool environment, and (iii) remains a valid speech-conditioned training example, so that p_D(a) > 0 in the constructed corpus implies genuine support coverage of a at inference. This assumption is verified per-sequence by an execution-grounded acceptance test (step 12) and reported as a per-sequence acceptance rate (step 17), not merely asserted.

**How it works.**

```pseudocode
Input: seeds S, tool schema T -> action space A (|A| finite, sequences up to L=4),
       teacher model M_T, coverage target epsilon=0.05, rounds R=6, temp tau=1.0,
       sandboxed tool environment E, acceptance-rate floor rho_min=0.10
Output: constructed dialogues D, coverage certificate C, per-sequence acceptance rates R_acc

1. Enumerate A from T (all valid tool-call sequences length 1..L).
2. D <- S; p_D <- empirical tool-call distribution over A induced by D
3. for r in 1..R:
4.     # Coverage deficit: divergence-based regularizer
5.     deficit(a) <- -log p_D(a) for each a in A          # per-sequence deficit
6.     # Sample target action sequences weighted by deficit (rare-first)
7.     A_r <- sample K sequences from A ~ softmax(deficit / tau)
8.     for each a in A_r:
9.         # Condition teacher on the action prefix to force support
10.        d <- M_T.generate(seed_context, forced_tool_call=a)
11.        # Execution-grounded acceptance test (replaces surface-form verifier)
12.        if execute(d, a, E) succeeds AND speech_conditioned(d) AND entails(d.context, a):
13.            D <- D + {d}; n_accept(a) += 1
14.        n_attempt(a) += 1
15.    p_D <- empirical tool-call distribution over A induced by D
16.    # Coverage regularizer objective (divergence-based)
17.    L_cov <- sum_{a in A} p_D(a) * log(p_D(a) / q_unif(a))
18.    # Per-sequence acceptance rate under forced conditioning
19.    R_acc(a) <- n_accept(a) / max(1, n_attempt(a)) for each a in A
20.    if max_a deficit(a) < epsilon AND min_{a: p_D(a)>0} R_acc(a) >= rho_min: break
21. return D, certificate C = {a: (p_D(a), R_acc(a)) for a in A}
```

The construction loop alternates between (i) identifying the support deficit of the current constructed distribution via the negative log-probability of each action sequence, (ii) forcing the teacher to realize the deficit-heavy sequences through action-prefix conditioning, and (iii) re-estimating the empirical distribution to recompute the divergence-based coverage regularizer L_cov against a uniform reference q_unif. The certificate C reports the realized p_D(a) and the per-sequence acceptance rate R_acc(a) for every a in A, making the coverage claim auditable and conditioned on the teacher actually being able to realize the sequence, not merely being prompted to.

**Why this design.** We chose to enumerate the action space explicitly over a learned generative model of the action space because the tool schema is finite and structured, accepting the cost that L is capped at 4 and the space grows combinatorially. We chose deficit-weighted rare-first sampling over uniform sampling because uniform sampling wastes construction budget on already-covered sequences, accepting the cost that the deficit weights must be recomputed each round. We chose a divergence-based regularizer (KL against uniform) over a simple count threshold because the divergence penalizes near-zero probabilities on any sequence, not just zero counts, accepting the cost that the regularizer is sensitive to the reference distribution choice. We chose teacher action-prefix conditioning over rejection sampling from unconstrained generation because rejection sampling cannot reach rare sequences at any practical budget, accepting the cost that forced generation may produce less natural dialogues. We chose an execution-grounded acceptance test over a surface-form parser because the parser certifies syntactic realization, not executability or semantic coherence, accepting the cost that execution requires a sandboxed tool environment and may reject syntactically valid but non-executable calls. A domain expert would not describe COIN as a variant of Toolformer, Gorilla, or standard RAG: those methods retrieve or select tools for a given query and never audit or enforce the support of the constructed training distribution over the action space; COIN's mechanism is a construction-time support-coverage objective with a certificate, not a retrieval or selection operation.

**Why it measures what we claim.** The empirical distribution p_D(a) over enumerated action sequences measures support coverage of the tool-call action space because the assumption that the action space is finite and enumerable from the tool schema makes p_D(a) > 0 for all a equivalent to support containment; this assumption fails when the inference-time tool-use distribution includes sequences longer than L or tools outside the schema, in which case p_D reflects coverage only of the enumerated subspace. The divergence L_cov measures the identifiability gap between the constructed distribution and the uniform reference because the assumption that uniform coverage is the maximally conservative target makes KL(p_D || q_unif) a monotone proxy for the worst-case deficit; this assumption fails when the inference distribution is highly non-uniform, in which case L_cov over-penalizes sequences that are rare at inference and the certificate should be reweighted by an inference-time prior. The execution-grounded acceptance test execute(d, a, E) measures whether a constructed dialogue actually realizes action sequence a because the assumption that the sandboxed tool environment E is a faithful executor makes successful execution equivalent to support realization; this assumption fails when the tool environment is incomplete or the dialogue's speech context does not semantically entail the call, in which case the dialogue counts as covered when it is not executable or not grounded. The per-sequence acceptance rate R_acc(a) measures the teacher's ability to realize a under forced conditioning because the assumption that forced generation is a faithful sample of the conditional makes R_acc(a) a direct estimate of realizability; this assumption fails when the teacher is off its training manifold, in which case R_acc(a) reflects exposure bias rather than genuine support.

## Contribution

(1) COIN, a construction-time coverage-identifiability objective that explicitly enumerates the tool-call action space and maximizes a divergence-based coverage regularizer over the constructed speech-conditioned dialogue distribution, producing a per-sequence coverage certificate. (2) The design principle that rare-first deficit-weighted teacher conditioning is necessary for support coverage of finite action spaces, since unconstrained sampling and rejection sampling cannot reach rare tool-use sequences at practical construction budgets. (3) An auditable coverage certificate that reports realized per-sequence probabilities, enabling downstream training to verify support containment before adaptation.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Speech-BFCL tool-call dialogues | Speech-conditioned, finite tool schema, matches construction input |
| Primary metric | Worst-case per-sequence coverage deficit (max_a -log p_D(a)) | Directly measures support gap, not average accuracy |
| Secondary metric | Per-sequence acceptance rate R_acc(a) under forced conditioning | Conditions coverage claim on teacher realizability |
| Secondary metric | Execution success rate on held-out tool calls | Tests whether constructed coverage transfers to executability |
| Baseline | Uniform rejection sampling from teacher | Tests whether deficit-weighting beats naive sampling |
| Baseline | Toolformer-style retrieval fine-tuning | Tests whether construction-time coverage beats selection |
| Baseline | Apriel Voice 27B Preview (unconstrained) | Tests whether prefix-forcing reaches rare sequences |
| Ablation-of-ours | COIN minus deficit weighting (uniform A_r) | Isolates the rare-first sampling contribution |
| Ablation-of-ours | COIN minus execution-grounded acceptance (surface-form parser) | Isolates the execution-grounded verifier contribution |

### Why this setup validates the claim

The central claim is that COIN's construction-time support-coverage objective produces a training distribution whose induced tool-call support provably contains the enumerated action space, and that this coverage—not retrieval or selection quality—is what drives robust tool use on rare sequences. Speech-BFCL is the right dataset because its tool schema is finite and enumerable, so max_a -log p_D(a) is a direct, auditable measurement of the support claim rather than a proxy. The uniform rejection baseline tests the sub-claim that deficit-weighting is necessary: if it matches COIN, the rare-first mechanism is doing no work. The Toolformer-style baseline tests the sub-claim that construction-time coverage beats inference-time selection: it can retrieve common tools but cannot manufacture support for rare sequences. The unconstrained Apriel baseline tests whether prefix-forcing is needed to reach rare sequences at practical budget. The ablation isolates the deficit-weighting mechanism specifically, so any gain can be attributed to rare-first sampling rather than to the teacher or the enumeration. The execution-grounded acceptance ablation isolates the verifier contribution, so any gain in execution success rate can be attributed to the execution-grounded test rather than to surface-form matching. The per-sequence acceptance rate R_acc(a) is reported alongside the coverage certificate so that p_D(a) > 0 is conditioned on the teacher actually being able to realize a, not merely being prompted to. Together these form a falsifiable test: the worst-case deficit metric will expose a coverage gap exactly where each baseline's mechanism fails, the acceptance rate will expose whether forced generation collapses on the rare tail, and the ablation will show whether that gap is closed by the claimed mechanism.

### Expected outcome and causal chain

**vs. Uniform rejection sampling from teacher** — On a case where a valid 4-step tool sequence (e.g., a rare multi-tool chain) has near-zero teacher probability, uniform rejection sampling produces almost no dialogues realizing that sequence because it draws from the teacher's unconstrained marginal and wastes budget on already-common sequences. Our method instead samples that sequence rare-first via deficit-weighted softmax and forces the teacher through action-prefix conditioning, so we expect the worst-case deficit to drop sharply on rare sequences while common-sequence coverage stays at parity—a large gap concentrated in the rare tail, not uniform across the space. We additionally expect the per-sequence acceptance rate R_acc(a) to be reported for each rare sequence, showing whether the teacher can realize it under forced conditioning; if R_acc(a) is near zero for a sequence, the coverage claim for that sequence is flagged as unrealizable rather than silently certified.

**vs. Toolformer-style retrieval fine-tuning** — On a case where the query requires a tool sequence that never appears in the retrieval pool, the retrieval baseline produces a plausible but unexecutable tool call because its mechanism selects from available demonstrations and never audits support, so p_D(a)=0 for that sequence. Our method instead enumerates the schema, identifies the zero-probability sequence as a deficit, and constructs a dialogue realizing it, so we expect near-zero worst-case deficit versus a persistent positive deficit on the rare subset for the baseline, with parity on high-frequency sequences. The execution-grounded acceptance test ensures that the constructed dialogue for the rare sequence is executable, so the coverage gain is not an artifact of surface-form matching.

**vs. Apriel Voice 27B Preview (unconstrained)** — On a case where a rare sequence has probability too low to surface in unconstrained generation at any practical budget, the unconstrained model produces common sequences and never realizes the rare one because its sampling is not steered toward the deficit. Our method instead conditions the teacher on the action prefix, forcing realization, so we expect the unconstrained baseline to leave a measurable worst-case deficit on the rare tail while COIN closes it, with the gap concentrated where teacher marginal probability is lowest. The per-sequence acceptance rate R_acc(a) will show whether the forced realization is genuine (executable and speech-conditioned) or merely syntactic, and the execution-grounded acceptance ablation will show whether replacing the surface-form parser with execution changes the measured coverage.

### What would falsify this idea

If COIN's worst-case coverage deficit gain over the uniform-rejection and retrieval baselines is uniform across all subsets rather than concentrated on the rare, low-teacher-probability sequences the mechanism targets, the central claim that deficit-weighted prefix-forcing drives coverage is wrong. Equally, if the deficit-weighting ablation matches full COIN, the rare-first sampling mechanism is doing no causal work and the support-coverage story collapses; and if the execution-grounded acceptance ablation matches full COIN, the execution-grounded verifier is doing no work and the certificate is measuring surface form rather than executability.

## References

1. Apriel Voice 27B Preview: An Open Model for Speech, Reasoning, and Tool Use
