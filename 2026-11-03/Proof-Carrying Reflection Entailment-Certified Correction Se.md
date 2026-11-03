# Proof-Carrying Reflection: Entailment-Certified Correction Segments for Multimodal Reasoning

## Motivation

Existing RL-trained multimodal reasoners such as VL-Rethinker generate reflection segments as free-form extra tokens whose correctness is rewarded only at the final answer, so there is no structural guarantee that a revised conclusion actually follows from the detected error and the original evidence; the reflection trace is an unverified append rather than a checked invariant. This meta-gap recurs across paper-trees: both frontier approaches treat post-hoc revision as additional autoregressive tokens, never constraining the revision to be entailed by intermediate evidence or the detected inconsistency. Consequently, long reflection traces can be syntactically fluent yet logically disconnected from the error they claim to fix, and RL post-training cannot distinguish a genuine correction from a plausible-sounding rewrite. The problem is structural, not a matter of scale or reward shaping: without an entailment relation enforced at the segment level, self-verification remains an unverified append.

## Key Insight

A correction segment is permitted to alter the final answer only when it carries a machine-checkable entailment certificate linking detected inconsistency, revised conclusion, and original evidence, making the reflection-to-answer transition a proof-carrying operation whose validity is decidable rather than an unconstrained token append.

## Method

# PROOF-CARRYING REFLECTION (PCR)
# Input: multimodal instance x = (image I, question q), base reasoner M_theta
# Output: final answer a*, or abstain if no valid certificate

# LOAD-BEARING ASSUMPTION (stated explicitly):
#   The entailment checks ENTAIL and the attention-grounded support-set check are
#   sound enough that certificate validity is a reliable proxy for genuine logical
#   entailment: a VALID certificate guarantees the revision truly follows from
#   evidence and error, and an INVALID one reliably indicates a non-entailed
#   revision. This assumption is load-bearing for the hard invariant, the
#   certificate reward, and the 'decidable rather than unconstrained' claim.
#   We do NOT assume it holds universally; we measure it (see step 5b) and
#   restrict the certificate claim to the decidable fragment where it is checked.

1. INITIAL ROLLOUT
   y0, a0 = M_theta.reason(I, q)          # free CoT + answer, temperature 0.7

2. ERROR SEGMENT DETECTION
   For each reasoning step s_t in y0:
       e_t = M_theta.entail_score(s_t | I, q)   # scalar in [0,1]
   Let D = {t : e_t < tau_d}                 # tau_d = 0.35, detected inconsistency set
   If D empty: return a0

3. CORRECTION GENERATION (segment-level)
   For each detected inconsistency d in D:
       c_d = M_theta.correct(d, y0, I, q)    # candidate revised conclusion
       e_d = M_theta.extract_evidence(d, y0, I, q)  # evidence span(s) from y0 and I

4. CERTIFICATE CONSTRUCTION
   Certificate C_d = triple (d, c_d, e_d)
   Attach a proof-carrying decoding trace: for each token of c_d, record the
   attention-grounded support set over e_d and I (top-k=5 visual/text spans).

5. CERTIFICATE VERIFICATION (proof-carrying decoding check)
   For each C_d:
       v1 = ENTAIL(e_d, d)        # evidence entails the detected inconsistency
       v2 = ENTAIL(e_d & d, c_d)  # evidence + inconsistency entails revision
       v3 = CONSISTENT(c_d, a0)   # revision either agrees or explicitly overrides a0
       If v1 and v2 and v3: mark C_d VALID
       Else: mark C_d INVALID and discard c_d

5b. VERIFIER SOUNDNESS MEASUREMENT (calibration of the load-bearing assumption)
   On a held-out calibration set of 512 instances drawn from the same distribution
   (MathVista + MathVerse error-localization subsets), we compute:
       - Soundness = (# VALID certificates that are truly entailed) / (# VALID certificates)
       - Completeness = (# truly entailed revisions marked VALID) / (# truly entailed revisions)
   Ground truth for 'truly entailed' is obtained by human annotation (two annotators,
   adjudicated by a third) on the 512 calibration instances.
   We report these two numbers as first-class metrics and use them to bound the
   certificate claim: the hard invariant is claimed only for the fraction of cases
   where soundness is measured to be high (target: soundness ≥ 0.90 on the
   decidable fragment; otherwise the claim is restricted to that fragment).

6. ANSWER REVISION (invariant enforcement)
   If any VALID certificate C_d exists and c_d != a0:
       a* = c_d with certificate C_d as justification
   Else:
       a* = a0   # no valid certificate -> original answer preserved

7. TRAINING SIGNAL (GRPO with certificate reward)
   Reward r = r_answer(a*, a_gold) + lambda_cert * 1[any VALID certificate]
              - lambda_invalid * 1[INVALID certificate present]
   lambda_cert = 0.2, lambda_invalid = 0.1
   Advantage computed over group of rollouts; backprop through M_theta.

8. INFERENCE
   Run steps 1-6; output a* and certificate C_d (if any) for auditability.

**Why this design**
We chose a segment-level entailment certificate over free-form reflection tokens because the structural requirement is that a correction be *entailed* by the detected error and evidence, not merely appended; accepting the cost that certificate construction requires an explicit evidence-extraction step and a verifier pass, which increases latency per correction. We chose a proof-carrying decoding trace (attention-grounded support sets) over a post-hoc NLI classifier because the trace makes the entailment relation checkable at the token level and ties each revised token to specific evidence spans, accepting the cost that attention grounding is an imperfect proxy for semantic support. We chose a hard invariant (revision only if certificate valid) over a soft reward penalty because the motivation demands a structural guarantee rather than a preference, accepting the cost that some genuinely correct corrections may be discarded when the verifier is conservative. We chose GRPO with a certificate reward over pure answer-only reward because the certificate term gives dense credit to valid reflection segments, accepting the cost that lambda_cert and lambda_invalid require tuning and may trade off answer accuracy. This design is not a variant of VL-Rethinker's Forced Rethinking: VL-Rethinker appends a rethinking trigger token and rewards only the final answer, whereas PCR makes the reflection-to-answer transition conditional on a decidable entailment certificate, so a domain expert would not describe PCR as a rethinking-token variant but as an invariant-enforcing decoding procedure. We explicitly state the load-bearing assumption that the verifier is sound enough for certificate validity to proxy genuine entailment, and we measure soundness and completeness on a 512-instance calibration set (step 5b) to bound the claim rather than assume it.

**Why it measures what we claim**
The detected inconsistency set D measures *error localization* because entail_score is assumed to be monotone in the degree to which a step is unsupported by the evidence; this assumption fails when the base reasoner is confidently wrong and assigns high entail_score to a flawed step, in which case D reflects the model's self-consistency rather than true error. The certificate triple (d, c_d, e_d) measures *entailment closure* because ENTAIL(e_d, d) and ENTAIL(e_d & d, c_d) are assumed to be sound checks that the revision follows from evidence and error; this assumption fails when the verifier is a learned NLI model with false positives, in which case a certificate may be valid yet the revision not truly entailed. The proof-carrying decoding trace measures *evidence grounding of the revision* because attention support sets are assumed to identify the evidence spans that causally determine each revised token; this assumption fails when attention is diffuse or spurious, in which case the trace reflects correlational salience rather than causal support. The hard invariant (answer changes only if certificate valid) measures *verified correction* because validity of the certificate is assumed to be necessary and sufficient for a legitimate revision; this assumption fails when the verifier is incomplete, in which case some correct revisions are blocked (false negatives) and the final answer reflects verifier conservatism rather than reasoning quality. The verifier soundness measurement (step 5b) directly quantifies the first two assumptions on a labeled calibration set, so the claim is bounded by measured soundness and completeness rather than asserted.

## Contribution

(1) A proof-carrying reflection framework that introduces segment-level entailment certificates as a hard invariant governing when a correction segment may alter the final answer, with a certificate-verification procedure that checks evidence-entails-error and evidence-plus-error-entails-revision relations. (2) A certificate-aware GRPO training signal that provides dense credit to valid reflection segments and penalizes invalid certificates, yielding a design principle that reflection traces can be trained as checked invariants rather than free-form appends. (3) A certificate auditability mechanism that outputs the entailment triple and proof-carrying decoding trace alongside the final answer, enabling inspection of which evidence supported each revision.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | MathVista + MathVerse (error-localization subsets) | Multimodal steps with verifiable intermediate errors |
| Primary metric | Answer accuracy conditioned on certificate validity | Directly tests verified-correction claim |
| Secondary metric | Verifier soundness and completeness on 512-instance calibration set | Quantifies load-bearing assumption |
| Baseline | VL-Rethinker (GRPO + Forced Rethinking) | Tests whether certificate invariant beats rethinking-token reward |
| Baseline | Base reasoner M_theta, no reflection | Tests whether any correction machinery is needed |
| Baseline | Post-hoc NLI self-correction (no proof trace) | Tests whether proof-carrying trace adds beyond NLI check |
| Ablation-of-ours | PCR with soft reward, no hard invariant | Tests whether invariant, not reward shaping, drives gains |

### Why this setup validates the claim

The central claim is that conditioning answer revision on a decidable entailment certificate produces more trustworthy corrections than free-form reflection or answer-only reward. MathVista and MathVerse are the right testbed because their reasoning chains contain checkable intermediate steps, so a genuine error-localization-and-repair event can be distinguished from a lucky answer change. The primary metric — accuracy conditioned on certificate validity — is chosen because it directly measures the claim: if valid certificates are necessary and sufficient for legitimate revisions, accuracy on certificate-valid cases should substantially exceed accuracy on certificate-invalid cases, and this gap is the predicted effect. The secondary metric — verifier soundness and completeness on a 512-instance calibration set — directly quantifies the load-bearing assumption that the learned entailment verifier is sound enough for certificate validity to proxy genuine entailment; without this measurement the claim would rest on an unverified assumption. VL-Rethinker tests whether the certificate invariant beats rethinking-token reward; the no-reflection baseline tests whether correction machinery helps at all; the post-hoc NLI baseline isolates whether the proof-carrying trace adds beyond a plain verifier; and the soft-reward ablation isolates whether the hard invariant, rather than reward shaping, is the causal ingredient. Together these form a falsifiable test: if PCR's advantage disappears once the invariant is removed, the claim is wrong; if measured soundness is low, the certificate claim is bounded accordingly.

### Expected outcome and causal chain

**vs. VL-Rethinker (GRPO + Forced Rethinking)** — On a case where the model's first pass contains a misread chart value, VL-Rethinker's rethinking token triggers a second free-form pass that often re-commits to the same misread because the reward only checks the final answer, so the error survives. Our method instead detects the low-entailment step, extracts the chart region as evidence, and only revises if the evidence entails the corrected value, so the misread is structurally blocked. We expect a noticeable accuracy gap concentrated on error-localization subsets, with near-parity on easy items where no certificate is needed.

**vs. Base reasoner M_theta, no reflection** — On a case where the model's chain contains a plausible but unsupported arithmetic leap, the base reasoner outputs the wrong answer with no mechanism to notice, because there is no entailment scoring or revision stage. Our method flags the leap as a detected inconsistency, generates a correction, and accepts it only when the certificate verifies, so the same instance is repaired. We expect a moderate but consistent accuracy gain on multi-step items, and near-zero change on single-step items where D is empty.

**vs. Post-hoc NLI self-correction (no proof trace)** — On a case where the model proposes a correct revision but the NLI verifier gives a false positive on a spurious evidence span, the post-hoc baseline accepts an ungrounded correction and can flip a correct answer to wrong, because it has no token-level grounding check. Our method's proof-carrying trace ties each revised token to attention-grounded evidence spans, so spurious-support corrections fail the trace check and are discarded, preserving the original answer. We expect fewer harmful answer flips on cases where the original answer was correct, visible as higher accuracy on the subset where a0 was already right. Additionally, the verifier soundness measurement on the 512-instance calibration set will report the false-positive rate of the NLI check, quantifying the risk that the post-hoc baseline (and our own method) inherits.

**vs. PCR with soft reward, no hard invariant** — On a case where the verifier is conservative and rejects a genuinely correct correction, the soft-reward ablation can still let the correction leak into the answer through reward pressure, because nothing structurally forbids it. Our method enforces the invariant and abstains to a0, so correct-but-unverified corrections are blocked. We expect the ablation to show more answer churn and slightly lower precision on certificate-invalid cases, confirming that the invariant rather than reward shaping produces the trustworthiness gain.

### What would falsify this idea

If PCR's accuracy gain over VL-Rethinker is uniform across certificate-valid and certificate-invalid cases rather than concentrated on certificate-valid cases, the claim that the certificate is the causal mechanism is wrong. Likewise, if removing the hard invariant (soft-reward ablation) leaves accuracy unchanged, the invariant is not doing the work the design claims. Additionally, if the measured verifier soundness on the 512-instance calibration set is low (e.g., < 0.80), the load-bearing assumption fails and the certificate claim must be restricted to the decidable fragment or abandoned.

## References

1. VL-Rethinker: Incentivizing Self-Reflection of Vision-Language Models with Reinforcement Learning
2. Distildoc: Deep Reinforcement Learning for Token-Efficient Multimodal Document Question Answering
