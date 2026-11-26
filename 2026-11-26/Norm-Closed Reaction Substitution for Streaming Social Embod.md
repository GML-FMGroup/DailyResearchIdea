# Norm-Closed Reaction Substitution for Streaming Social Embodied Decision Benchmarks

## Motivation

Streaming social-decision benchmarks such as AIRIS score a single committed reaction per time anchor, rewarding calibrated waiting but never requiring that a committed reaction be retractable or substitutable once a social-norm constraint is discovered to be breached; the SOLAMI single-response social VLA decoder likewise emits one multimodal response with no substitution operator. Structurally, the field's decision benchmarks and decoders both treat a reaction as a terminal label rather than an element of a closed action algebra, so norm-violation recovery — a core capability of agents operating around people — is never measured. This is a meta-level gap: across streaming-decision trees and social-VLA trees, reaction commitment is irreversible, and no existing branch models retraction or compliant substitution.

## Key Insight

If the reaction head is required to emit a norm-closure set whose members are pairwise distinguishable by an explicit violated-constraint signature, then substitution competence becomes a measurable algebraic property of the committed action rather than a latent behavior, because closure under the violation-substitution operator is checkable without ever observing the future stream.

## Method

**What it is.** NORC (Norm-closure Operator for Reaction Commitment) is a benchmark-side reaction algebra layered onto AIRIS time anchors. Input: at anchor t, the observation prefix, the model's committed reaction r_t, and a set of social-norm constraints N. Output: a closure set C_t = {r_t} ∪ S_t of candidate reactions, each tagged with the constraint signature it is intended to satisfy, plus a per-anchor closure score.

**How it works.**

```pseudocode
INPUT: anchor t, prefix obs_{<=t}, committed reaction r_t, norm set N={n_1..n_k}
OUTPUT: closure set C_t, closure score kappa_t

1. COMMIT:  r_t <- model.react(obs_{<=t})            # standard AIRIS reaction
2. AUDIT:   for each n_j in N:
               v_j <- norm_checker(r_t, n_j, obs_{<=t})   # returns BREACH / OK / UNKNOWN
3. SIGNATURE: sig(r_t) <- {n_j : v_j == BREACH}          # violated-constraint signature
4. IF sig(r_t) is empty:
       C_t <- {r_t};  kappa_t <- 1.0                    # trivially closed
   ELSE:
       S_t <- model.substitute(obs_{<=t}, sig(r_t), budget=3)
       # each s in S_t must carry its own declared target signature sig_hat(s)
5. VERIFY SUBSTITUTES: for each s in S_t:
       ok(s) <- (sig_hat(s) == sig(r_t)) AND norm_checker(s, sig(r_t), obs_{<=t}) == OK
6. CLOSURE: C_t <- {r_t} ∪ {s in S_t : ok(s)}
7. SCORE:
       kappa_t <- 1[closure nonempty] * (|{s in S_t : ok(s)}| / |S_t|) * 1[sig_hat correct]
8. RETRACTION TEST (decision-aware): replay anchor t+1 with the same prefix;
       a model passes if it can replace r_t by an ok(s) without changing
       its earlier Attention/Intention outputs (locality check).
```

Hyperparameters: substitute budget = 3, signature match required exactly (no partial credit), norm set N fixed per clip from AIRIS's 11 context categories. The norm_checker is an ensemble of 5 independently prompted LLM checkers (temperature 0.7, distinct system prompts) that each return BREACH/OK/UNKNOWN per constraint; the consensus label for n_j is BREACH if ≥3 of 5 return BREACH, OK if ≥3 return OK, else UNKNOWN. The violation signature sig(r_t) is the set of n_j with consensus BREACH. For substitute verification, ok(s) requires that the substitute's declared target signature sig_hat(s) equals sig(r_t) exactly (set equality) and that the consensus label of s on every n_j in sig(r_t) is OK. The per-anchor closure score kappa_t is computed as above, but we also report a reliability ceiling: the inter-checker agreement rate (mean pairwise Cohen's kappa across the 5 checkers on a held-out human-labeled set) is reported alongside kappa_t, and kappa_t is interpreted as bounded above by this reliability.

**Why this design.** We chose a benchmark-side operator over retraining SOLAMI's decoder because SOLAMI's single-response social VLA has no substitution head, so any decoder-side fix would conflate the measurement with the model under test, accepting the cost that NORC scores models rather than improving them. We chose an explicit violated-constraint signature over a free-form substitute list because a signature makes closure checkable by an external norm checker rather than by the model's own claim, accepting the cost that the norm checker must be curated per context category. We chose exact signature matching over soft similarity because partial credit lets a model game closure by emitting many loosely-related substitutes, accepting the cost of harsher scores on near-miss reactions. We chose a locality check on retraction (Attention/Intention must be preserved) over a free re-decision because otherwise a model can pass by discarding its whole reasoning chain, accepting the cost that some legitimately revisionary reactions are penalized. This is not a variant of retrieval-based tool selection (Toolformer/Gorilla) because the substitute must be verified against a violated-constraint signature the model did not observe at commitment time — a counterfactual the retrieval library cannot supply.

**Why it measures what we claim.** The closure score kappa_t measures norm-violation recoverability because we assume the per-clip norm set N is exhaustive for the context category and that the consensus of 5 independently prompted checkers yields a stable BREACH label, so a nonempty verified closure set is equivalent to the existence of a compliant substitute; this assumption fails when a clip contains a norm not in N or when the checkers disagree (inter-checker agreement below 0.6), in which case kappa_t reflects closure against an incomplete or noisy constraint set rather than true recoverability. The signature-match term measures whether the model localizes the breach because we assume the consensus BREACH labels are the ground-truth violation signature, so an exact match is equivalent to correct localization; this fails when the checker ensemble over-flags or is unreliable, in which case the term measures agreement with checker noise. The locality check measures retraction without collateral reasoning damage because we assume Attention/Intention outputs are the sufficient statistic of the earlier decision, so invariance is equivalent to surgical retraction; this fails when a legitimate substitute requires revising intention, in which case the check penalizes correct revision as non-locality.

## Contribution

(1) NORC, a reaction-algebra operator that turns AIRIS's single-label reaction decision into a closure-under-substitution benchmark with an explicit violated-constraint signature and a locality-checked retraction test. (2) A design principle: exact-signature closure scoring with a fixed substitute budget separates norm-violation localization from substitution generation, exposing which capability streaming social agents lack. (3) An extension of AIRIS's decision-aware metrics with a per-anchor closure score kappa_t and a retraction-locality diagnostic over its 11 context categories.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | AIRIS egocentric streaming clips (11 context categories) | Time-anchored prefixes enable per-anchor closure scoring |
| Primary metric | κ_t closure score at each anchor | Directly measures norm-violation recoverability, not accuracy |
| Baseline | AIRIS raw reaction accuracy | Tests whether closure adds signal beyond commitment |
| Baseline | SOLAMI social VLA (single-response) | Tests whether substitution head absence drives closure failure |
| Baseline | SOLAMI social VLA with a substitute head (retrained) | Tests whether a decoder-side substitute head can pass closure |
| Baseline | LLM self-critique substitute list | Tests whether external signature verification is necessary |
| Ablation-of-ours | NORC with soft signature matching | Tests whether exact matching prevents gaming |
| Ablation-of-ours | NORC with single-checker (no ensemble) | Tests whether ensemble consensus reduces checker noise |

### Why this setup validates the claim

The central claim is that norm-violation recoverability is a separable, measurable property of social embodied models, distinct from raw reaction accuracy, and that it requires an externally verified violated-constraint signature rather than the model's own substitute proposal. AIRIS is the right dataset because its time-anchored chains (Attention, Intention, Reaction, Interaction) give us a per-anchor observation prefix and a committed reaction to audit, which is exactly the input NORC needs; without time anchors, closure would be defined over completed videos and the retraction locality test would be meaningless. The raw AIRIS accuracy baseline tests sub-claim one: if κ_t correlates near-perfectly with accuracy, closure is redundant and the method adds nothing. SOLAMI tests sub-claim two: if a single-response social VLA that cannot emit substitutes scores at parity with NORC on κ_t, then the substitution mechanism is not what drives recoverability, and the design rationale collapses. The SOLAMI variant with a substitute head tests whether the baseline can in principle pass, isolating the effect of the substitution operator from the absence of a head. The self-critique baseline tests sub-claim three: if a model allowed to grade its own substitutes achieves the same verified-closure rate, then external signature checking is unnecessary overhead. The soft-signature ablation tests sub-claim four: if partial credit does not inflate closure scores, then exact matching is not doing the anti-gaming work we claim. The single-checker ablation tests whether the ensemble consensus is necessary to stabilize the BREACH signature. κ_t is the right primary metric because it is the only quantity in the design that is defined as recoverability-under-external-verification; accuracy and substitute count both fail to isolate that property, and a reader who sees κ_t gaps concentrated on the predicted failure subsets has direct evidence for the causal story. The retraction locality check is folded into κ_t as the signature-correctness term, so a model that passes by discarding its reasoning chain cannot inflate the score. We also report the inter-checker agreement rate as a reliability ceiling on κ_t, so the metric is bounded by checker reliability rather than assumed exact.

### Expected outcome and causal chain

**vs. AIRIS raw reaction accuracy** — On a case where the committed reaction breaches a context-category norm (e.g., a premature intervention in a dyadic conversation clip), the raw-accuracy baseline produces a correct-looking reaction label because accuracy only checks the committed action against the gold action, not whether a compliant substitute exists. Our method instead audits the committed reaction against N, extracts the violated-constraint signature, and scores whether a verified substitute closes it, because NORC measures recoverability rather than commitment correctness. We expect a noticeable κ_t gap on the subset of anchors where the committed reaction is norm-breaching but action-label-correct, and near-parity on anchors where the commitment is already compliant (trivially closed, κ_t = 1.0 for both).

**vs. SOLAMI social VLA (single-response)** — On a case where the character's generated speech-and-motion response violates a turn-taking norm, SOLAMI produces a single multimodal response with no substitution head, so it cannot emit an alternative and its closure set is always {r_t}; its κ_t collapses to 0 whenever sig(r_t) is nonempty because S_t is empty and the closure set is empty. Our method instead invokes model.substitute with the violated-constraint signature and verifies each candidate against an external norm checker, because the signature makes closure checkable without trusting the model's own claim. We expect SOLAMI to score near-zero on the breaching subset and parity on the compliant subset, producing a bimodal κ_t distribution that NORC flattens.

**vs. SOLAMI social VLA with a substitute head (retrained)** — On a case where the character's response violates a norm, this variant is given a substitute head that emits up to 3 alternative speech-and-motion responses, each with a declared target signature, and is trained on the same AIRIS training split with a cross-entropy loss on substitute selection. Because it has a substitution mechanism, it can in principle pass closure; however, its substitutes are verified by the same external norm checker ensemble. We expect this variant to achieve higher κ_t than the single-response SOLAMI but lower than NORC on multi-constraint anchors, because its substitute head is trained to maximize likelihood of human substitutes rather than to satisfy an external violated-constraint signature, and the gap isolates the value of external signature verification over a learned substitute head.

**vs. LLM self-critique substitute list** — On a case where the model proposes substitutes without knowing which constraint was violated, the self-critique baseline produces a free-form list of loosely related reactions and grades itself as compliant, because it has no external signature to match against and no independent checker. Our method instead requires each substitute to declare its target signature and pass an external norm_checker on exactly the violated constraints, because exact signature matching prevents a model from gaming closure by volume. We expect the self-critique baseline to show high self-reported closure but low verified closure, so the gap between its self-score and NORC's κ_t is the observable signal, concentrated on multi-constraint anchors where free-form listing is easiest to inflate.

### What would falsify this idea

If NORC's κ_t gain over raw accuracy and self-critique is uniform across all context categories and capability dimensions rather than concentrated on anchors where the committed reaction actually breaches a norm, then the score is tracking general model quality rather than recoverability, and the central claim that closure measures a separable property is wrong. A second falsifier: if the soft-signature ablation matches exact matching on κ_t, then the anti-gaming rationale for exact matching is unsupported and the signature design is not doing the work we attribute to it. A third falsifier: if the single-checker ablation matches the ensemble on κ_t and inter-checker agreement is low (<0.6), then the ensemble consensus is not stabilizing the signature and κ_t is dominated by checker noise.

## References

1. AIRIS: Benchmarking Social Embodied Intelligence in Egocentric Streaming Videos
