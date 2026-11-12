# Distractor-Identifiability Scoring for Rubric-Based Evaluation of Goal Inference in Shopping Dialogue

## Motivation

Rubric-based evaluators such as CARTBench score whether a recommendation reply satisfies expert criteria, but they implicitly assume the latent user goal is recoverable from the timestamped dialogue, so a reply that happens to satisfy the rubric while inferring the wrong goal still scores highly; the persistent understanding-to-execution gap (no model exceeding 45% on Excellence) is thus partly an artifact of unmeasured goal identifiability. The root cause is structural: rubrics are written against a single annotated goal, so nothing in the evaluation distinguishes a reply that is consistent with the true goal from one equally consistent with a plausible alternative that the context does not rule out. We add an identifiability probe that makes this latent ambiguity explicit and penalizes rubric scores when a reply fails to discriminate the true goal from context-consistent distractors.

## Key Insight

A reply's goal inference is only creditable to the extent that it selects the true goal among all goals still admissible under the observed dialogue and timeline, so scoring the reply by its discriminative margin against generated admissible distractors converts an unobservable latent-goal assumption into a directly checkable uniqueness condition.

## Method

**What it is.** DISCO (Distractor-Identifiability Scoring for Criteria) wraps any frozen rubric evaluator. Inputs: a CARTBench-style case (timestamped multi-turn dialogue D, product records P, expert rubric R with criteria c_i, and the annotated true goal g*). Output: a per-criterion adjusted score and an identifiability report. It never retrains the evaluator; it edits the evaluation input.

**Load-bearing assumption (stated explicitly).** DISCO assumes the frozen rubric evaluator is *goal-sensitive*: when its goal variable is substituted from g* to an admissible distractor g, the score s_g(r) reflects the reply's goal-specific satisfaction rather than the evaluator's insensitivity to the goal variable or its anchoring on reply surface form (fluency, hedging, listing options). If this assumption fails, s_g ≈ s_{g*} for all g and the margin m_i collapses to a vagueness/verbosity signal. We therefore make this assumption a *checkable precondition* via a calibration gate (Phase 0) and only apply the penalty on criteria/cases where the gate passes; otherwise the margin is reported as uninformative rather than penalized.

**How it works.**

```pseudocode
PHASE 0 — Goal-sensitivity calibration gate (precondition)
  # Synthetic calibration set C (n=40 replies, 20 per class), constructed once per rubric:
  #   C_disc: replies that name a product uniquely implied by g* and violate a
  #           constraint of every g in G (provably goal-discriminative)
  #   C_agn:  generic hedge replies ("it depends", "both are good") that satisfy
  #           no goal-specific constraint (provably goal-agnostic)
  for each criterion c_i in Core ∪ Excellence:
    m_disc_i = mean_{r in C_disc} [ s_{g*}(r) - max_{g in G} s_g(r) ]
    m_agn_i  = mean_{r in C_agn}  [ s_{g*}(r) - max_{g in G} s_g(r) ]
  gate_pass_i = (m_disc_i >= tau_disc) AND (m_agn_i <= tau_agn)
    # tau_disc = 0.3, tau_agn = 0.1 (fixed a priori, not tuned on test)
  # If gate_pass_i is False, criterion c_i is marked UNINFORMATIVE:
  #   no penalty is applied for c_i; margin reported but flagged.

PHASE 1 — Admissible distractor generation
  for k in 1..K (K=3):
    g_k ~ LLM_gen( D, P, instruction="propose a shopping goal that is
             fully consistent with every stated constraint and the timeline,
             but differs from the most obvious reading in >=1 hard constraint")
  filter g_k: keep only if a verifier LLM confirms every explicit constraint
             and timestamp in D is satisfied by g_k  (admissibility check)
  set G = {g_1..g_K} \ {g*}
  # If |G| < 1 after filtering, resample once (new K=3 draws); if still |G| < 1,
  # case is marked NON-IDENTIFIABLE-BY-CONSTRUCTION and excluded from penalty
  # (reported separately).

PHASE 2 — Reply-goal consistency probe (per candidate g in G ∪ {g*})
  reply r is scored by the frozen rubric evaluator under goal g:
    s_g = rubric_score(r, R, g)            # same criteria R, goal substituted
  identifiability margin per criterion c_i:
    m_i = s_{g*}(r) - max_{g in G} s_g(r)   # aggregated over criteria

PHASE 3 — Penalized rubric score (only for criteria with gate_pass_i = True)
  base_i = s_{g*}(r)                        # original CARTBench score
  pen_i  = lambda * max(0, tau - m_i)       # lambda=0.5, tau=0.2
  final_i = base_i - pen_i                  # if gate_pass_i = False, pen_i = 0
  report identifiability flag: identifiable iff m_i >= tau for all i in Core
                               AND gate_pass_i = True for all i in Core
```

**Why this design.** We chose counterfactual goal substitution over asking the model to self-report its inferred goal because self-reports are not verifiable against the rubric and would reintroduce the same latent-variable assumption; the cost is K+1 rubric passes per reply, roughly a 4x evaluation budget. We chose to generate distractors with an explicit admissibility verifier rather than free-form LLM sampling because unverified distractors can be trivially distinguishable (a strawman), which would make every reply look identifiable; the cost is that the verifier can reject many candidates, so we cap K=3 and resample once. We chose a margin penalty with a fixed floor tau over a multiplicative gate because a gate is discontinuous and would zero out partially correct replies, whereas a margin preserves the ordering of near-identifiable replies; the cost is that lambda and tau are set by held-out calibration rather than derived. We added the Phase 0 calibration gate because the margin is only meaningful if the frozen evaluator is goal-sensitive; the gate converts that assumption into a checkable precondition at the cost of a one-time 40-reply calibration per rubric. This differs from CARTBench's single-goal rubric scoring and from standard rubric-LLM-as-judge pipelines: an expert would not describe DISCO as a variant of either, because neither intervenes on the goal variable during scoring.

**Why it measures what we claim.** The margin m_i = s_{g*}(r) − max_{g∈G} s_g(r) measures goal identifiability because we assume that if a reply genuinely infers g*, its rubric satisfaction should be strictly higher under g* than under any admissible alternative; this assumption fails when the reply is generic enough that the rubric scores it identically under every goal (a hedging reply), in which case m_i reflects reply vagueness rather than goal ambiguity. The admissibility filter measures the plausibility of distractors because we assume a distractor satisfying every explicit constraint and timestamp is indistinguishable from the true goal on observable evidence; this fails when the dialogue contains an implicit cue (e.g., a product already in the cart) that the verifier does not model, in which case G contains a goal a careful reader would have ruled out and m_i understates identifiability. The penalty term lambda·max(0, tau − m_i) measures the score correction owed to non-identifiability because we assume rubric satisfaction under the wrong goal should not be credited; this fails when the rubric itself is goal-independent (Safety criteria), in which case the penalty should be zero and we restrict it to Core and Excellence criteria. The Phase 0 gate measures whether the frozen evaluator is goal-sensitive because we assume s_g reflects goal-specific satisfaction; this fails when the evaluator anchors on surface form, in which case m_i collapses to a vagueness signal and the gate marks the criterion UNINFORMATIVE rather than penalizing.

## Contribution

(1) DISCO, an identifiability-probe wrapper that converts any frozen rubric evaluator into a distractor-discriminative scorer by substituting admissible alternative goals into the rubric and penalizing the consistency margin. (2) A design principle for LLM-based evaluation: rubric scores are only creditable up to the identifiability margin, and a verified-admissibility distractor set is the minimal intervention needed to measure it. (3) An identifiability-annotated extension of CARTBench cases with generated and verified distractor goals, enabling diagnosis of whether the understanding-to-execution gap is goal-inference failure or execution failure.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | CARTBench ambiguous shopping cases | Timestamped, rubric-annotated, multi-constraint; matches method assumptions |
| Primary metric | Score drop on distractor-sensitive subset vs. parity elsewhere | Directly detects goal-identifiability correction, not general quality |
| Baseline | CARTBench single-goal rubric score | Tests whether goal substitution adds signal over standard scoring |
| Baseline | Self-reported-inferred-goal judge | Tests whether verifiable substitution beats unverifiable self-report |
| Baseline | Free-form distractor sampling (no admissibility filter) | Tests whether verified distractors matter vs. strawman distractors |
| Baseline | Pairwise rubric comparison under perturbed goals | Tests whether margin is just goal-substituted contrastive scoring |
| Ablation-of-ours | DISCO without margin penalty (base_i only) | Isolates the penalty term's contribution from goal substitution |
| Ablation-of-ours | DISCO without Phase 0 calibration gate (penalty on all criteria) | Isolates the gate's contribution from goal substitution |

### Why this setup validates the claim

The central claim is that a reply can satisfy a rubric under the true goal while failing to *identify* that goal, and that this gap is measurable by counterfactual goal substitution. CARTBench is the right dataset because it uniquely supplies all four ingredients the method needs — timestamped dialogue, product records, expert rubric, and an annotated true goal — so the margin is computable without inventing ground truth. The single-goal rubric baseline tests the sub-claim that standard scoring is blind to identifiability: if it already separates identifiable from ambiguous replies, DISCO's premise collapses. The self-report baseline tests the sub-claim that verifiable substitution beats asking the model what goal it inferred; if self-report matches DISCO, the extra K+1 rubric passes are unjustified. The free-form distractor baseline tests the admissibility sub-claim: unverified distractors should produce near-zero margins everywhere (strawmen are easy to beat), so if DISCO's gains vanish when distractors are unfiltered, the verifier is doing the work. The pairwise rubric comparison baseline tests whether the margin is merely goal-substituted contrastive scoring: if a pairwise judge under perturbed goals matches DISCO's discrimination, the substitution mechanism adds nothing over contrastive scoring. The ablation without margin penalty isolates the penalty term from substitution itself, separating "we found a margin" from "we corrected the score." The ablation without Phase 0 gate tests whether the calibration gate is necessary: if removing it does not change results, the gate is redundant; if it inflates penalties on goal-agnostic criteria, the gate is doing the work. The primary metric — score drop concentrated on the distractor-sensitive subset with parity elsewhere — is chosen because it is the only observable that distinguishes a real identifiability effect from a uniform quality shift; a method that merely lowers all scores is not measuring identifiability.

### Expected outcome and causal chain

**vs. CARTBench single-goal rubric score** — On a case where the user's timeline implies a hard constraint (e.g., "needs it before the trip") but the reply hedges across two plausible products, the baseline awards full Core credit because each sentence satisfies the rubric under g*; it never asks whether the reply would also satisfy the rubric under a different admissible goal. Our method substitutes each verified distractor and finds the reply scores comparably under both, yielding a small margin m_i and a penalty. We expect a noticeable score gap on the distractor-sensitive subset (replies that hedge or list multiple options) but near-parity on unambiguous replies, where margins are large and penalties vanish.

**vs. Self-reported-inferred-goal judge** — On a case where the reply is fluent and confident, the self-report baseline asks the model to state its inferred goal, gets a plausible-sounding g', and scores against it; but g' is unverifiable and often just restates the reply's own framing, so a hedging reply reports a confident goal and escapes penalty. Our method instead forces the reply to be scored under externally generated, admissibility-verified alternatives, so a hedge cannot talk its way out. We expect the self-report baseline to show inflated scores precisely on the hedging subset, with DISCO's penalty concentrated there.

**vs. Free-form distractor sampling** — On a case with several hard constraints, unfiltered sampling proposes goals that violate an explicit constraint (wrong budget, wrong date), so the reply trivially scores higher under g* than under these strawmen; every reply looks identifiable and margins are uniformly large. Our method's admissibility verifier rejects those candidates and keeps only goals consistent with all constraints and timestamps, producing genuinely competitive alternatives. We expect the free-form baseline to show near-zero penalties across the board, while DISCO shows penalties concentrated where the true goal is genuinely underdetermined by the dialogue.

**vs. Pairwise rubric comparison under perturbed goals** — On a case where the reply is goal-discriminative, a pairwise judge comparing the reply under g* vs. a distractor may simply prefer the more fluent or longer response, yielding a margin that tracks surface form rather than goal inference. Our method's Phase 0 gate explicitly checks that the evaluator assigns high margin to provably goal-discriminative replies and near-zero to goal-agnostic hedges; if the pairwise baseline matches DISCO on the hedging subset, the substitution mechanism is not adding unique signal. We expect DISCO to show larger discrimination on the distractor-sensitive subset than the pairwise baseline, because the gate filters out criteria where the evaluator is goal-insensitive.

**vs. DISCO without margin penalty (base_i only)** — This ablation removes the penalty term but keeps goal substitution and the Phase 0 gate. It should produce the same margins as full DISCO but no score correction. We expect it to match the single-goal baseline on the primary metric (no score drop), confirming that the penalty term is responsible for the correction, while the margins themselves are informative only when the gate passes.

**vs. DISCO without Phase 0 calibration gate** — This ablation applies the penalty on all criteria, including goal-agnostic ones like Safety. On Safety criteria, the margin should be near-zero regardless of reply quality, so the penalty will incorrectly lower scores for safe, goal-agnostic replies. We expect this ablation to show inflated penalties on the Safety subset and reduced overall correlation with human judgments, confirming that the gate is necessary to avoid penalizing goal-agnostic criteria.

### What would falsify this idea

If DISCO's score corrections are uniform across all subsets rather than concentrated on replies that hedge or on cases with genuinely underdetermined goals, then the margin is tracking reply vagueness or a global quality shift, not goal identifiability, and the central claim is wrong. Likewise, if the self-report baseline matches DISCO's discrimination on the hedging subset, or if the Phase 0 gate fails on most criteria (indicating the evaluator is not goal-sensitive), the verifiable-substitution mechanism adds nothing over the latent-variable assumption it was meant to replace.

## References

1. CARTBench: Rubric-Based Evaluation of Shopping Advice for Complex and Ambiguous Requests with Timelines
