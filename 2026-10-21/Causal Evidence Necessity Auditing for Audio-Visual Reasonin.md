# Causal Evidence Necessity Auditing for Audio-Visual Reasoning Benchmarks

## Motivation

Evidence-grounding benchmarks such as EvaOmni score cited evidence by surface span-matching against a single human-annotated gold span, which structurally cannot separate genuine evidence use from post-hoc rationalization: a model can emit the correct span after answering from a textual or unimodal shortcut and still receive full grounding credit. This residual gap recurs across multiple benchmark branches (A, B, D, E, G) because every branch retains span-matching as the ground-truth criterion, so no branch resolves the fact that the benchmark's central claim — that correct answers are attributable to cited multimodal evidence — is never tested by an intervention on that evidence.

## Key Insight

The causal necessity of a cited evidence span is identified by the answer-flip rate under matched evidence-removal interventions, a quantity span-matching cannot express because it requires the model's output distribution to change only when the cited span was load-bearing.

## Method

**What it is.** CENA (Causal Evidence Necessity Audit) is an evaluation protocol that, for each EvaOmni QA pair, synthesizes ablated media variants — gold evidence span removed, gold span replaced by a matched distractor span drawn from a different video, and a matched neutral control — and scores the cited evidence's causal necessity as the answer-flip rate across these interventions, replacing span-matching with an intervention-based criterion.

**How it works.**

```pseudocode
INPUT: QA pair q=(video V, question, options, gold_span s*, modality m in {audio, visual, both}),
       model M, distractor pool D (spans from other EvaOmni videos), N_boot=1000
OUTPUT: necessity score eta(q), rationalization flag r(q)

1. BASELINE: a0 = argmax M(V, q); c0 = M.conf(V, q)[a0]
2. ELICIT CITED EVIDENCE: s_hat = M.cite(V, q)   # model's own emitted span
3. BUILD INTERVENTION SET (matched on span length L, modality, and
   speaker/scene identity where available):
   V_rem  = V with s_hat masked (audio: silence; visual: inpaint)
   V_rep  = V with s_hat replaced by d in D, |d|=L, same modality
   V_ctrl = V with a random non-evidence span of length L masked
   # NEW: redundancy probe — mask s_hat plus its top-k co-occurring candidate spans
   V_rem_red = V with s_hat and top-k co-occurring spans masked (k=3)
4. RUN INTERVENTIONS:
   a_rem  = argmax M(V_rem, q);  a_rep = argmax M(V_rep, q);  a_ctrl = argmax M(V_ctrl, q)
   a_rem_red = argmax M(V_rem_red, q)
   flip(x) = 1[a_x != a0]
5. NECESSITY: eta(q) = (flip(rem) + flip(rep)) / 2
   SPECIFICITY: sigma(q) = flip(rem) - flip(ctrl)   # evidence-specific vs generic degradation
   # NEW: redundancy-aware necessity (contrast against matched perturbation baseline)
   eta_red(q) = flip(rem_red) - flip(ctrl)   # necessity after controlling for redundancy
6. RATIONALIZATION FLAG: r(q) = 1[eta(q) < tau_eta AND sigma(q) <= 0 AND a0 correct]
   with tau_eta = 0.5
   # NEW: redundancy flag
   r_red(q) = 1[eta_red(q) < tau_eta AND a0 correct]
7. AGGREGATE: report eta, sigma, r, eta_red, r_red per capability/level; bootstrap 95% CI over N_boot resamples
```

**Why this design.** We chose a matched-distractor replacement arm in addition to pure removal because removal alone confounds necessity with generic model brittleness to corrupted input, accepting the cost of needing a curated cross-video distractor pool; the control arm (random non-evidence mask) is what lets sigma isolate evidence-specific from generic degradation. We chose the model's own emitted citation s_hat rather than the human gold span s* as the intervention target because the audit question is whether the model's own grounding claim is load-bearing, accepting that this makes eta model-relative rather than a property of the dataset alone. We chose a binary answer-flip criterion over a continuous confidence-drop threshold because answer flips are discrete and robust to the well-documented miscalibration of LLM log-probabilities (Kadavath et al.), accepting reduced sensitivity for near-tie cases which the bootstrap CI absorbs. We chose fixed tau_eta=0.5 rather than a learned threshold to keep the protocol reproducible across models. A domain expert would not describe CENA as a variant of span-matching benchmarks like EvaOmni because the scored quantity is an intervention response, not a string overlap; it is also not a variant of RAG faithfulness probes because the distractor is matched on length, modality, and speaker/scene identity, so the intervention does not change the input distribution in ways that could explain a flip.

**Why it measures what we claim.** The answer-flip rate flip(rem) measures causal necessity of the cited span because the masking intervention is assumed to remove only the cited information while leaving all other evidence intact; this assumption fails when the cited span is redundant with another span in the video, in which case flip(rem)=0 reflects redundancy rather than non-use. The replacement flip(rep) measures necessity under substitution because the matched distractor is assumed to be informationally equivalent in surface statistics but different in content; this assumption fails when the distractor introduces an out-of-distribution speaker or scene that itself drives the flip, in which case the metric reflects distribution shift rather than evidence substitution. The specificity sigma measures evidence-specific use because the control mask is assumed to be non-informative; this assumption fails when the random control span happens to overlap semantically with the question, in which case sigma under-estimates specificity. The rationalization flag r(q) measures post-hoc citation because eta<tau_eta with correct a0 is assumed to imply the answer was reachable without the cited evidence; this assumption fails when the model uses evidence but is robust to its removal via a redundant pathway, in which case r(q) over-flags. The redundancy-aware necessity eta_red measures necessity after controlling for redundancy because the redundancy probe masks s_hat plus its top-k co-occurring spans, so flip(rem_red)=0 indicates non-use rather than redundancy; this assumption fails when the top-k co-occurring spans do not capture all redundant evidence, in which case eta_red under-estimates necessity. Each component is thus tied to its motivation-level concept through a named assumption whose failure mode is stated, so the protocol's claims are falsifiable rather than definitional.

## Contribution

(1) CENA, an intervention-based audit protocol that replaces span-matching with an answer-flip-rate criterion for evidence necessity, including matched-distractor and neutral-control arms that separate evidence-specific use from generic input brittleness. (2) A rationalization flag derived from the joint necessity/specificity criterion, together with a per-capability and per-reasoning-level breakdown that identifies where EvaOmni's grounding scores over-credit plausible-but-unused citations. (3) An open-source intervention-generation pipeline (masking, inpainting, matched cross-video distractor selection) that can be applied to any evidence-annotated audio-visual benchmark without re-annotation.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | EvaOmni QA (15 capabilities, 3 levels) | Provides gold spans and cross-video distractor pool |
| Primary metric | Necessity eta and specificity sigma | Directly score intervention response, not string overlap |
| Baseline | EvaOmni answer accuracy | Tests whether correctness alone predicts evidence use |
| Baseline | Span-overlap citation score | Tests whether string match tracks causal necessity |
| Baseline | Confidence-drop probe | Tests whether log-prob sensitivity tracks answer flips |
| Ablation-of-ours | CENA without control arm (sigma removed) | Isolates whether specificity arm drives rationalization detection |
| Ablation-of-ours | CENA without redundancy probe (eta_red removed) | Isolates whether redundancy control changes rationalization flags |

### Why this setup validates the claim

The central claim is that the cited evidence's causal necessity, measured by intervention response, is what distinguishes genuine multimodal grounding from post-hoc rationalization, and that this quantity is not recoverable from answer accuracy or span overlap. The design is falsifiable because each baseline targets one competing explanation of the same construct: EvaOmni accuracy tests whether correct answers already imply evidence use, span-overlap tests whether citation string match already implies causal load-bearing, and the confidence-drop probe tests whether continuous confidence sensitivity already implies the discrete flip pattern. If any of these baselines already predicts the rationalization pattern, CENA's intervention machinery is redundant. The ablation removes the control arm so that sigma is undefined; if rationalization flags are unchanged without sigma, the specificity component is doing no work and the matched-control design is unjustified. The redundancy-probe ablation removes eta_red; if rationalization flags are unchanged without it, the redundancy control is doing no work. Necessity eta is the right primary metric because the claim is about whether removing the cited span changes the model's committed answer, which is a discrete behavioral event rather than a graded confidence score, and the replacement arm guards against reading generic brittleness as necessity.

### Expected outcome and causal chain

**vs. EvaOmni answer accuracy** — On a case where the model answers correctly but its cited span is a plausible-sounding caption unrelated to the audio cue that actually determines the answer, accuracy scores it as a success because it never inspects the citation. Our method masks that cited span and replaces it with a matched cross-video distractor, so a non-load-bearing citation yields flip(rem)=flip(rep)=0 and eta near zero, flagging rationalization. We expect a substantial subset of high-accuracy items to carry low eta, producing a visible decoupling between accuracy and necessity concentrated in the higher reasoning levels.

**vs. Span-overlap citation score** — On a case where the model cites a span that overlaps the gold span verbatim but the answer would survive its removal because a redundant visual cue carries the same information, overlap scores it as fully grounded because the string matches. Our method observes flip(rem)=0 under masking and flip(ctrl)=0 under the neutral control, so sigma is at or below zero and the item is flagged as non-necessary despite perfect overlap. We expect overlap and eta to diverge most on capabilities with redundant cross-modal evidence, so the correlation between the two should be weak on those subsets and stronger where evidence is unique.

**vs. Confidence-drop probe** — On a case where the model's log-probability for the gold answer drops sharply when the cited span is masked but the argmax answer is unchanged because the drop stays within a near-tie, a confidence-threshold probe declares the evidence necessary while the committed answer never moved. Our method uses the discrete flip criterion, so the same intervention yields flip(rem)=0 and eta near zero, correctly reading the citation as not decision-relevant. We expect the confidence probe to over-attribute necessity on near-tie items, producing a systematic gap between the two methods that widens as answer entropy increases.

### What would falsify this idea

If rationalization flags r(q) were distributed uniformly across capabilities and reasoning levels rather than concentrated where the predicted failure mode (redundant evidence, near-tie answers, high-entropy items) occurs, the intervention-based criterion would be measuring generic model noise rather than post-hoc citation, and the central claim would be wrong. Likewise, if removing the control arm left the flag pattern essentially unchanged, sigma would be doing no discriminative work and the matched-control design would be unjustified. If removing the redundancy probe left the flag pattern essentially unchanged, eta_red would be doing no discriminative work and the redundancy control would be unjustified.

## References

1. EvaOmni: An Evidence-Grounded Benchmark for Audio-Visual Reasoning
