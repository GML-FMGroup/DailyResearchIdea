# Claim-Certified Rationale Distillation: Verifier-Gated Supervision for Multimodal Reasoning

## Motivation

Across the distillation lineage for multimodal reasoning, rationale supervision is admitted by proxies that never certify intermediate factual correctness: MMVistaReason selects trajectories by student answer likelihood, Distilling Step-by-Step admits any LLM-generated rationale that co-occurs with a correct label, and AMA aggregates noisy votes without checking the steps that produced them. The structural root cause is that selection operates on the joint (rationale, answer) surface, so a rationale with a correct answer but a false intermediate claim — a visually hallucinated object, a misread chart axis — is indistinguishable from a sound one, and the student is trained to reproduce the false step. This failure recurs across every branch of the distillation lineage because none of them introduces an oracle that scores claims independently of the answer.

## Key Insight

Parsing each rationale into atomic, individually checkable claims and admitting a rationale only when every claim is certified by an external oracle converts rationale supervision from a joint answer-likelihood proxy into a per-step factual constraint, so the false-intermediate/correct-answer rationales that likelihood selection systematically admits are structurally excluded from the training set rather than merely down-weighted.

## Method

**What it is.** Claim-Certified Rationale Distillation (CCRD) is a distillation pipeline that takes a pool of teacher-generated multimodal rationales and produces (i) a certified SFT set containing only rationales whose every atomic claim is verified, and (ii) a dense claim-level reward used in RL post-training. Inputs: image $I$, question $Q$, candidate rationales $\{r_j\}$ from $K{=}3$ teachers, and an external claim oracle. Outputs: certified rationales $\mathcal{C}$ and a verifier $V_\phi$ that scores claims at inference and post-training time.

**Load-bearing assumption (stated explicitly).** The pipeline assumes that the claim decomposition is *faithful and complete*: every premise a rationale actually relies on surfaces as an individually checkable claim, so that the conjunctive AND over parsed claims is equivalent to certifying the rationale's full reasoning chain. This assumption is load-bearing because the admission gate is a conjunction over a set the parser chooses; any omitted premise is vacuously certified. To make this assumption checkable rather than assumed, the decomposition is constrained to a closed, typed claim grammar (below) and admission requires that the rationale be fully reconstructible from its certified claims.

**How it works.**

```pseudocode
# Phase 1: Claim decomposition (closed typed grammar, not open-vocabulary)
# Grammar has exactly three claim types, each tied to an oracle predicate:
#   PERCEPTUAL: <region_ref, attribute, value>   e.g. (bar@[x1,y1,x2,y2], height, tallest)
#   ARITHMETIC: <expression, stated_value>       e.g. (7*8, 54)
#   LOGICAL:    <premise_ids, conclusion>        e.g. ([c1,c2], c3)
# Parser = LLM prompted with the grammar; output must parse to a list of typed claims
# plus a residual string R_j (any rationale text not covered by a claim).
for each candidate rationale r_j for (I, Q):
    (C_j, R_j) = Decompose(r_j)   # C_j: typed claims; R_j: uncovered text
    # Reconstructibility check: concatenation of C_j must entail r_j and the answer.
    # Implemented as: EntailCheck(C_j, r_j) AND EntailCheck(C_j, answer).
    # If R_j is non-empty OR either entailment fails -> rationale is rejected (see Phase 3).

# Phase 2: Oracle certification (per claim, not per rationale)
for each claim c in C_j:
    if c.type == PERCEPTUAL:  s(c) = VisualGround(I, c)      # crop-and-check: does the referenced
                                                             # region support the proposition?
    if c.type == ARITHMETIC:  s(c) = SymbolicEval(c)         # exact recomputation of the stated value
    if c.type == LOGICAL:     s(c) = EntailCheck(Premises(c), c)  # NLI over prior certified claims
    s(c) in {0, 1}                       # binary certificate, not a soft score

# Phase 3: Rationale admission (conjunctive AND + reconstructibility)
certified(r_j) = (R_j == empty) AND EntailCheck(C_j, r_j) AND EntailCheck(C_j, answer)
                 AND (AND_{c in C_j} s(c))   # one false claim or any residual text rejects the rationale
mathcal{C} = { r_j : certified(r_j) }    # certified SFT set

# Phase 4: Train verifier on certified + corrupted claim pairs
Train V_phi on {(c, s(c))} plus contrastive corruptions of certified claims
(swap a grounded object, perturb a number) -> negative pairs.
# V_phi is a 2-layer MLP (hidden=256, GeLU) over frozen CLIP-ViT-L/14 text+image features,
# trained with binary cross-entropy, AdamW lr=1e-4, batch=256, 3 epochs.

# Phase 5: Student SFT on mathcal{C}; RL post-training with claim reward
r_claim(r) = (1/|C|) * sum_{c in C} V_phi(c)     # dense, per-claim
RL objective: maximize E[ r_answer + lambda * r_claim ],  lambda = 0.3
```

**Why this design.** We chose conjunctive claim-level certification over likelihood-based trajectory selection (the MMVistaReason recipe) because a joint answer-likelihood score cannot separate a correct answer reached through a false step from a correct answer reached soundly, accepting the cost that conjunctive AND rejects many otherwise-useful rationales and shrinks the SFT set. We chose type-routed oracles (visual grounding, symbolic evaluation, NLI) over a single monolithic factuality judge because each claim type has a checkable ground truth, accepting the engineering cost of a router that must tag claims correctly and the coverage gap for claims that fit no route. We chose a binary certificate over a soft verifier score for the SFT admission gate because a threshold on a soft score reintroduces the calibration problem that likelihood selection already suffers from, accepting that binary gates are brittle to parser segmentation errors. We chose to reuse the certified claims as a dense RL reward rather than discarding them after SFT because the claim certificates already encode per-step correctness, accepting that $V_\phi$ inherits the oracle's coverage limits. A domain expert would not describe this as Distilling Step-by-Step with a filter: Distilling Step-by-Step never inspects intermediate claims, and MMVistaReason's selection signal is the student's own answer likelihood, whereas CCRD's admission signal is an oracle certificate that is independent of the student and of the final answer.

**Why this measures what we claim.** The claim-level certificate $s(c)$ measures *factual correctness of an intermediate reasoning step* because the named assumption is that the type-routed oracle (visual grounding crop, symbolic recomputation, NLI over certified premises) is a sound check for its claim type; this assumption fails when a perceptual claim refers to a region the grounding model localizes incorrectly, in which case $s(c)$ reflects the grounding model's localization accuracy rather than the proposition's truth. The conjunctive admission $\text{certified}(r_j)=\bigwedge_c s(c)$ measures *freedom from false intermediate steps* because the assumption is that the decomposition is faithful — every premise the rationale relies on surfaces as a claim; this fails when the parser drops an implicit premise, in which case the gate certifies a rationale that is still unsound. The verifier reward $r_\text{claim}$ measures *step-level correctness during RL* because the assumption is that $V_\phi$ trained on certified/corrupted pairs generalizes to student-generated claims; this fails on out-of-distribution claim phrasings, in which case $r_\text{claim}$ reflects stylistic similarity to certified claims rather than correctness.

## Contribution

(1) A verifier-in-the-loop distillation pipeline that replaces joint answer-likelihood trajectory selection with conjunctive atomic-claim certification, using type-routed oracles (visual grounding, symbolic evaluation, NLI) as the admission gate for rationale supervision. (2) A claim-level verifier trained on certified claims and contrastive corruptions that supplies a dense per-step reward for RL post-training, substituting for likelihood-based trajectory selection. (3) An empirical characterization of which claim types (perceptual vs. arithmetic vs. logical) account for the rationales that likelihood selection admits but claim certification rejects, and how that composition varies with model scale.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | MVR-SFT-528K + MVR-RL-63K with held-out step-annotated reasoning subset | Matches baseline recipe; enables step-level diagnosis |
| Primary metric | Step-faithfulness rate (fraction of rationales with zero false claims) | Directly measures freedom from false intermediate steps |
| Baseline | Standard SFT + RL on MVR (MMVistaReason recipe) | Tests whether likelihood selection admits unsound rationales |
| Baseline | Distilling Step-by-Step (unfiltered rationale distillation) | Tests whether claim inspection is necessary at all |
| Baseline | Soft-verifier-score admission at matched threshold | Tests whether binary certificate beats calibrated gating |
| Baseline | Single monolithic factuality judge at matched admission rate | Tests whether atomic decomposition (not just a verifier) drives gain |
| Ablation-of-ours | CCRD with conjunctive gate replaced by answer-likelihood selection | Isolates the certification signal from pipeline plumbing |
| Ablation-of-ours | CCRD with k-failure relaxation (k=1,2) | Shows certified-set size vs. faithfulness trade-off |
| Calibration study | Human-annotated perceptual claims (n=512) for grounding oracle | Bounds false-certification rate of s(c) |
| OOD-phrasing test | Held-out student-generated claims (n=512) for V_phi | Tests verifier generalization beyond training distribution |

### Why this setup validates the claim

The central claim is that an oracle certificate independent of the student and of the final answer yields rationales that are free of false intermediate steps, and that this in turn produces denser, better-aligned RL signal. The dataset choice matters because MVR-SFT-528K and MVR-RL-63K are exactly the pools the MMVistaReason recipe was built on, so any difference in step-faithfulness cannot be attributed to data distribution — only to the admission mechanism. The primary metric is step-faithfulness rather than final-answer accuracy because the whole argument is that answer accuracy conflates sound and unsound reasoning; a method that improves faithfulness while matching answer accuracy is a win, and a method that improves answer accuracy without improving faithfulness is evidence against us. The MMVistaReason baseline tests the sub-claim that likelihood-based selection cannot separate correct-answer-via-false-step from correct-answer-soundly; Distilling Step-by-Step tests whether inspection is needed at all versus simply distilling teachers; the soft-verifier baseline tests whether the binary certificate is doing work beyond calibration; the monolithic-judge baseline tests whether atomic decomposition (not merely the presence of a verifier) is the source of any gain; and the ablation replaces only the admission gate so that any gain can be attributed to certification rather than to decomposition or RL plumbing. The k-failure relaxation shows how certified-set size and step-faithfulness trade off as the conjunctive gate is relaxed, indicating practicality at realistic teacher-pool sizes. The calibration study measures the visual-grounding oracle's localization accuracy on human-annotated perceptual claims and reports the resulting false-certification rate as an explicit bound on s(c) validity. The OOD-phrasing test measures V_phi's accuracy on student-generated claims outside its training distribution, bounding the verifier reward's validity. Together these form a falsifiable test: the predicted pattern is a faithfulness gap concentrated where likelihood selection is known to be blind, with parity on cases where the answer is wrong anyway.

### Expected outcome and causal chain

**vs. MMVistaReason (SFT+RL on MVR)** — On a case where a teacher rationale reaches the correct answer through a hallucinated object attribute (e.g. counting a partially occluded item by asserting it is fully visible), the baseline admits the rationale because its selection signal is the student's answer likelihood, which is high whenever the final number matches; the false perceptual step is invisible to a trajectory-level score. Our method instead decomposes that rationale into an atomic perceptual claim, routes it to crop-and-check visual grounding, and gets s(c)=0, so the rationale is rejected at admission and the RL reward penalizes the student for reproducing the claim. We expect a noticeable step-faithfulness gap on the perceptual-claim subset, with near-parity on arithmetic-only items where both methods already get the final number right.

**vs. Distilling Step-by-Step** — On a case where a teacher produces a fluent multi-step rationale with a plausible but unverified intermediate arithmetic value (e.g. stating 7×8=54 en route to a wrong final answer that the student then imitates), the baseline distills the rationale wholesale because it never inspects intermediate claims, so the student inherits the corrupted step and the error compounds on related items. Our method routes the arithmetic claim to symbolic recomputation, gets s(c)=0, and excludes the rationale entirely; the corrupted step never enters the SFT set. We expect a large faithfulness gain on the arithmetic-claim subset and a smaller but real gain on multi-hop items, with the gap widening as rationale length grows.

**vs. Soft-verifier-score admission** — On a case where the parser segments a rationale slightly differently than during verifier training, a soft score drifts just above the admission threshold and a rationale with one marginal claim slips through, because a threshold on a calibrated score reintroduces exactly the calibration problem likelihood selection suffers from. Our method's conjunctive binary gate rejects the rationale the moment any claim fails its type-routed oracle, regardless of how confident the verifier is on the other claims. We expect the binary gate to produce a smaller but strictly cleaner SFT set, visible as higher step-faithfulness at the cost of lower SFT-set size, and we expect the soft-verifier baseline to show a characteristic tail of admitted-but-unsound rationales that our gate truncates.

**vs. Single monolithic factuality judge** — On a case where a rationale contains one false perceptual claim embedded in otherwise sound reasoning, a monolithic judge applied to the whole rationale at matched admission rate may assign a high overall factuality score because the majority of the text is correct, admitting the rationale. Our method decomposes into atomic claims and rejects the rationale on the single false claim, so the admitted set is cleaner. We expect CCRD to outperform the monolithic judge on step-faithfulness at matched admission rate, demonstrating that atomic decomposition—not merely the presence of a verifier—is the source of the gain.

**vs. CCRD with k-failure relaxation (k=1,2)** — As k increases, more rationales are admitted, increasing certified-set size but also admitting rationales with up to k false claims. We expect step-faithfulness to decrease monotonically with k, quantifying the trade-off between set size and cleanliness, and showing that the strict conjunctive gate (k=0) yields the highest faithfulness at the cost of smallest set.

**vs. Calibration study (grounding oracle)** — On a held-out set of 512 human-annotated perceptual claims, we measure the visual-grounding oracle's localization accuracy and the false-certification rate (claims certified as true but human-labeled false). We expect to report this as an explicit bound: e.g., if localization accuracy is 92%, the false-certification rate is ≤8%, meaning s(c) validity is bounded by the oracle's accuracy.

**vs. OOD-phrasing test (V_phi)** — On 512 student-generated claims with phrasings not seen in V_phi's training, we measure V_phi's accuracy against human labels. We expect accuracy to drop relative to in-distribution claims, quantifying the risk that r_claim tracks style rather than truth, and we will report this drop as a limitation.

### What would falsify this idea

If the step-faithfulness gain over MMVistaReason were uniform across perceptual, arithmetic, and logical claim subsets rather than concentrated on the subsets where likelihood selection is structurally blind, the certificate would not be doing the causal work we claim. Equally, if the ablation with answer-likelihood admission matched the full CCRD pipeline on step-faithfulness, the central claim that oracle certification (not pipeline plumbing) drives the improvement would be wrong.

## References

1. MMVistaReason: Toward Open-Data and Post-Training Recipes for Multimodal Reasoning
2. Distilling Step-by-Step! Outperforming Larger Language Models with Less Training Data and Smaller Model Sizes
3. Ask Me Anything: A simple strategy for prompting language models
