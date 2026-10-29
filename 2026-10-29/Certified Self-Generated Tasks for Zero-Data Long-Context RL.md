# Certified Self-Generated Tasks for Zero-Data Long-Context RL

## Motivation

Zero-data long-context RL pipelines such as LongZero manufacture their own QA-evidence triples and then let GRPO reward improvement serve as the only signal that a triple was ever valid, so malformed, unanswerable, or trivial triples silently contaminate the training pool and their damage is only detectable after many rollouts have already been spent. This is a meta-level structural gap: every self-generation pipeline in this family (LongZero, R-Zero, and the synthesis stage of QwenLong-L1.5) certifies task validity implicitly through downstream reward, meaning the validity predicate is entangled with the very optimization it is supposed to gate, and no pipeline exposes validity as an independent, pre-training check.

## Key Insight

Validity of a self-generated QA-evidence triple is a property of the triple alone under a fixed evidence-expansion operator, so it can be decided by an independent predictor whose training labels are constructed by intervention on the expansion operator — making the gate orthogonal to the GRPO reward rather than a restatement of it.

## Method

**What it is.** CERT-GRPO (Certified Generation for GRPO) is a wrapper around LongZero's self-generation loop. It takes as input a candidate QA-evidence triple `t = (q, a, e)` and its expanded multi-document context `C = Expand(e)`, and outputs a binary admission decision. Only admitted triples enter the GRPO pool.

**How it works.**

```pseudocode
# Phase 1: build the validity predictor's training set by intervention
for each seed triple t = (q, a, e) from the generator:
    C      = Expand(e)                       # LongZero expansion operator
    # Multi-operator ablation: ablate each supporting span individually
    for each supporting span s in e_support:
        C_neg_s = Expand(e \ s)
        label(t, C_neg_s) = UNANSWERABLE if a is not derivable by the frozen base model from C_neg_s in 3/3 greedy samples
                             else REDUNDANT_SUPPORT (new fourth class)
    # Redundancy check: if any C_neg_s is answerable, label the triple REDUNDANT_SUPPORT
    if any C_neg_s is answerable: label(t, C) = REDUNDANT_SUPPORT
    else: label(t, C) = WELLFORMED if a is derivable by the frozen base model from C in 3/3 greedy samples
    C_far  = Expand(e) + DistractorDocs(k=8) # AUGMENT: add distractors
    label(t, C_far)  = WELLFORMED   (distractors must not change answer)
    # Memorization control: answerability from shuffled/paraphrased context
    C_shuf = ShuffleSentences(C)  # or paraphrase with T5-large
    label(t, C_shuf) = TRIVIAL if a is derivable from C_shuf in 3/3 greedy samples
                       else WELLFORMED
    # Surface-token triviality
    label(t, C_triv) = TRIVIAL      if q is answerable from q's own surface tokens alone (no C needed)

# Phase 2: train the predictor V (a 1.5B frozen-encoder + 2-layer MLP head)
V(x) -> {WELLFORMED, UNANSWERABLE, REDUNDANT_SUPPORT, TRIVIAL}    # 4-way, cross-entropy
# calibrated on a held-out 10% split of the intervention set (512 examples)
# calibration: temperature scaling on validation set, producing calibrated probabilities

# Phase 3: the gate, applied per candidate before GRPO
for each freshly generated triple t:
    C = Expand(e)
    p = V(t, C)  # calibrated probability vector
    if p[WELLFORMED] < threshold_high (e.g., 0.9): discard t
    elif p[WELLFORMED] < threshold_low (e.g., 0.7): down-weight t (weight = p[WELLFORMED])
    elif Answerable(C, q) is not in {a} for all 3/3 greedy decodes: discard t
    else: admit t to GRPO pool

# Phase 4: GRPO on the admitted pool (unchanged from LongZero)
```

The gate is a **calibrated soft score with a hard threshold only above a validated confidence band**: a triple is admitted iff V assigns WELLFORMED with calibrated probability ≥ 0.9 *and* the answer is recoverable by the frozen base model from the expanded context. Triples with probability between 0.7 and 0.9 are down-weighted (weight = p[WELLFORMED]) rather than discarded, preserving orthogonality to GRPO reward while not over-claiming definitional validity. V is trained once per generator checkpoint and re-fit every 500 GRPO steps.

**Why this design.** We chose an *intervention-constructed* label set (`C_neg`, `C_far`, `C_triv`, `C_shuf`) over human or LLM-judge labels because the ablation/augmentation operator gives a label whose ground truth is definitional rather than annotator-dependent, accepting the cost that the predictor only learns validity axes the intervention can express. We chose a *calibrated soft gate with a hard threshold only above a validated confidence band* over a pure hard gate or a soft reward-penalty term because the motivation requires validity to be orthogonal to the GRPO reward, and a soft penalty would re-entangle the two signals; the cost is that borderline triples are down-weighted rather than cleanly separated. We chose a *frozen* encoder for V over fine-tuning the generator itself because a fine-tuned generator's validity judgment is exactly the implicit signal LongZero already relies on, whereas a frozen encoder cannot co-adapt to the GRPO objective; the cost is that V may miss validity failures specific to the generator's latest checkpoint, mitigated by re-fitting every 500 steps. A reviewer familiar with R-Zero's Challenger-Solver split would not describe CERT-GRPO as a variant of it: R-Zero's Challenger *proposes* tasks and its reward is the Solver's success rate, i.e. a downstream-reward signal; CERT-GRPO's gate is a *pre-training* predicate trained on interventions on the expansion operator, and its label set exists independently of any solver.

**Why it measures what we claim.** The quantity `V(t,C) = WELLFORMED` measures *task validity* because the intervention construction makes the label set definitionally equivalent to validity under the assumption that the expansion operator is the only source of context and that removing `e_support` is a valid intervention on answerability; this assumption fails when `e` contains redundant supporting spans, in which case `C_neg` is still answerable and the UNANSWERABLE label is wrong, so V instead measures *redundancy-robust answerability*. To address this, we add a fourth class REDUNDANT_SUPPORT and a memorization control (C_shuf) so that the predictor can learn to distinguish redundant-support cases and memorization-driven triviality. The quantity `Answerable(C,q) in {a} over 3/3 greedy decodes` measures *non-triviality of the evidence requirement* because it assumes the frozen base model's greedy answer is a faithful read-out of what the context supports; this assumption fails when the base model has memorized `a` from pretraining, in which case the check reflects memorization rather than grounding. The memorization control (C_shuf) provides a direct probe: if the answer is derivable from shuffled context, the triple is labeled TRIVIAL, mitigating the confound. The quantity `TRIVIAL` label from surface-token answerability measures *non-triviality* because it assumes that a question answerable without `C` is a question whose GRPO reward would be uninformative; this fails for questions whose answer is short but whose *reasoning path* is long, in which case the metric reflects answer-surface triviality rather than reasoning triviality. The calibrated soft score with a hard threshold only above a validated confidence band ensures that triples the predictor cannot certify are down-weighted rather than silently discarded, preserving orthogonality to GRPO reward while not over-claiming definitional validity.

## Contribution

(1) CERT-GRPO, a pre-training validity gate for zero-data long-context RL in which the validity predictor's labels are constructed by intervention on the evidence-expansion operator rather than by human annotation or downstream reward. (2) A design principle: validity certification must be orthogonal to the GRPO reward to be informative, operationalized as a hard gate trained on an intervention set that the GRPO objective never touches. (3) A reusable intervention-constructed validity dataset schema (ablate / augment / trivialize) for QA-evidence triples in long-context self-generation.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Long-context multi-hop QA (LoongRL KeyChain, 16K–32K) | Stresses evidence grounding where validity failures bite. |
| Primary metric | Accuracy on admitted-triple distribution, split by gate outcome | Isolates whether gating improves pool quality. |
| Baseline 1 | LongZero (ungated self-generation) | Tests claim that unvetted self-generated triples are noisy. |
| Baseline 2 | Soft-penalty reward variant of CERT-GRPO | Tests claim that hard gating beats reward re-entanglement. |
| Baseline 3 | R-Zero Challenger-Solver | Tests claim that pre-training predicate differs from solver-reward signal. |
| Baseline 4 | LLM-judge validity labeler (GPT-4) on same candidate pool | Tests whether intervention labels are merely a re-derivation of judge-based filtering. |
| Ablation-of-ours | CERT-GRPO with V replaced by random-admit at matched rate | Tests whether V's content, not admission rate, drives gain. |
| Ablation-of-ours | CERT-GRPO with V trained on single-operator labels (no multi-operator, no C_shuf) | Tests whether multi-operator and memorization control are necessary. |
| Ablation-of-ours | CERT-GRPO with hard gate only (no soft down-weighting) | Tests whether calibrated soft score improves over pure hard gate. |

### Why this setup validates the claim

The central claim is that a definitionally-labeled, calibrated soft validity gate removes triples whose GRPO reward is uninformative, so the admitted pool yields better reasoning per unit of training. The dataset choice matters because long-context multi-hop QA is exactly where expansion introduces distractors and where ablating supporting evidence produces genuinely unanswerable contexts, so the intervention labels are meaningful rather than vacuous. The primary metric — accuracy broken out by gate outcome — is the right detector because it lets us see whether the gain concentrates on triples the gate would have discarded, which is the predicted signature; a uniform gain would not distinguish gating from generic regularization. LongZero tests whether unvetted self-generation is the noise source; the soft-penalty variant tests whether the hard/soft distinction matters; R-Zero tests whether our predicate is genuinely a different signal from solver success. The LLM-judge baseline tests whether the intervention construction is not merely a re-derivation of existing judge-based filtering, by reporting agreement and downstream accuracy. The random-admit ablation at matched rate is the load-bearing control: if admission rate alone explains the gain, the ablation matches us; if V's content explains it, the ablation falls short on exactly the subsets where V disagrees with randomness. The single-operator ablation tests whether the multi-operator and memorization control are necessary for the gain, and the hard-gate-only ablation tests whether the calibrated soft score adds value.

### Expected outcome and causal chain

**vs. LongZero** — On a case where the generator emits a triple whose expanded context contains redundant supporting spans, LongZero's implicit validity judgment admits it, and GRPO spends gradient on a triple whose answer is recoverable from surface cues, so reward is uninformative and the policy drifts toward short-answer shortcuts. Our method instead routes the triple through `C_neg` and `C_triv` interventions, labels it TRIVIAL or UNANSWERABLE, and discards it before it enters the pool. We expect a noticeable accuracy gap on the long-context multi-hop subset with parity on short-context subsets, since the gate removes exactly the triples that were teaching shortcut behavior.

**vs. Soft-penalty variant** — On a case where a triple is borderline (answerable but weakly grounded), the soft-penalty variant assigns a small negative reward term, which GRPO's advantage normalization partially cancels against the positive reasoning reward, so the policy still receives a mixed signal and cannot cleanly separate validity from correctness. Our method instead applies a calibrated soft score with a hard threshold only above a validated confidence band and drops or down-weights the triple entirely, keeping validity orthogonal to the reward. We expect the soft variant to sit between LongZero and CERT-GRPO on the multi-hop subset, with the gap widening as the fraction of borderline triples in the generator's output grows.

**vs. R-Zero Challenger-Solver** — On a case where the Challenger proposes a task the Solver happens to solve by memorized pretraining recall, R-Zero's reward is the Solver's success rate, so the task is reinforced despite requiring no grounding in the provided context. Our method instead checks answerability from `C` with 3/3 greedy decodes and labels surface-answerable questions TRIVIAL, so memorization-driven tasks are filtered before training. We expect R-Zero to show inflated gains on tasks with high pretraining overlap and our method to show a gap concentrated on the low-overlap, high-distractor subset.

**vs. LLM-judge validity labeler** — On a case where the judge labels a triple as valid but the intervention labels it as REDUNDANT_SUPPORT or TRIVIAL, the judge's decision is based on surface plausibility rather than definitional intervention. Our method's labels are constructed by ablation and augmentation, so they are definitional. We expect moderate agreement (e.g., Cohen's kappa ~0.6) between judge and intervention labels, but our method to yield higher downstream accuracy on the multi-hop subset, demonstrating that the intervention construction is not merely a re-derivation of judge-based filtering.

**vs. random-admit ablation** — On a case where V confidently labels a triple as WELLFORMED but random admission would also admit it, both methods agree. The difference arises on triples where V disagrees with random. We expect CERT-GRPO to outperform random-admit on the subset of triples where V's prediction differs from random, confirming that V's content, not admission rate, drives the gain.

**vs. single-operator ablation** — On a case where a triple has redundant supporting spans, the single-operator labeler would mislabel it as UNANSWERABLE, while our multi-operator labeler correctly labels it REDUNDANT_SUPPORT. We expect the single-operator ablation to perform worse on the multi-hop subset, especially on triples with redundant evidence, demonstrating the necessity of the multi-operator and memorization control.

**vs. hard-gate-only ablation** — On a case where a triple is borderline (calibrated probability between 0.7 and 0.9), the hard-gate-only variant discards it, while our calibrated soft score down-weights it. We expect the calibrated soft score to yield slightly higher accuracy by retaining useful signal from borderline triples, with the gap widening as the fraction of borderline triples increases.

### What would falsify this idea

If the gain over LongZero is uniform across gate-admitted and gate-rejected triples rather than concentrated on the rejected subset, the gate is acting as generic regularization rather than a validity filter. If the random-admit ablation at matched rate matches CERT-GRPO on the multi-hop subset, V's content — not admission rate — is not doing the work the claim requires.

## References

1. LongZero: Enhancing Long-Context Reasoning in LLMs from Zero Data
2. LoongRL: Reinforcement Learning for Advanced Reasoning over Long Contexts
3. R-Zero: Self-Evolving Reasoning LLM from Zero Data
4. QwenLong-L1.5: Post-Training Recipe for Long-Context Reasoning and Memory Management
