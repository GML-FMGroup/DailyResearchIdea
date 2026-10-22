# Verifier-Gated Self-Bootstrapping Trace Distillation with Cross-Sample Trace Agreement

## Motivation

TraceForge's generator still requires a supervised seed-corpus stage on the student family's own reasoning traces before adversarial training can begin, so its dependence on external reasoning traces is shifted rather than removed; the same residual gap appears in the Orca branch, where rich explanation traces must be harvested from a stronger proprietary teacher, and in the Let's Verify branch, where step-level human labels anchor the process reward model. Structurally, all three methods assume an exogenous supply of trustworthy traces or step labels that the student family cannot produce for itself, which means families with sparse or unavailable trace data cannot enter the loop at all. The bottleneck is not trace generation but the absence of a self-contained criterion for deciding which self-generated traces are worth training on.

## Key Insight

A correctness verifier gives a per-problem, answer-level ground-truth signal that is independent of trace style, so agreement between independently sampled traces that all pass the verifier is a self-supervised certificate of reasoning validity that no seed corpus is needed to obtain.

## Method

**What it is.** VEST (Verifier-gated Elucidated Self-Training) replaces TraceForge's supervised seed stage with a self-supervised loop: a generator proposes reasoning traces from problem statements alone, a critic scores family-style realism, a symbolic correctness verifier checks final answers, and only traces that are both realistic and verifiably correct are used for iterative self-training. Inputs: a set of unlabeled problems P (no family traces, no teacher traces), a base student model M0, a verifier V. Outputs: a generator G and critic C trained without any seed corpus.

**Load-bearing assumption (stated explicitly).** The method assumes that independently sampled traces that pass the verifier and share the same final answer via step-consistent paths constitute a self-supervised certificate of reasoning validity, i.e., that correlated errors (shared systematic misconceptions) are rare enough that cross-sample agreement is a reliable proxy for correctness of the reasoning process rather than mere consensus. This assumption is the load-bearing one: if it fails, A_p reflects consensus rather than correctness, and the buffer weights w reflect answer popularity rather than reasoning quality. The method does not prove this assumption; it is empirically bounded by the diagnostic in Phase 1c below.

**How it works.**

```pseudocode
# Phase 0: initialization (no seed traces)
G <- M0 ; C <- M0 ; buffer B <- {}

# Phase 1: self-supervised trace proposal
for each problem p in P:
    traces T_p = sample(G, p, n=8, temperature=0.9)   # problem statement only, no answer given
    for t in T_p:
        a = extract_final_answer(t)
        v_ok = V(p, a)                                  # symbolic/executable verifier
        r = C(p, t)                                     # realism score in [0,1]
        keep t if v_ok and r > 0.5
    # Phase 1b: cross-sample agreement signal
    A_p = 1 if at least 2 kept traces in T_p share the same final answer AND
              their intermediate steps are pairwise consistent under a
              step-alignment check (edit-distance on normalized step strings < tau=0.3)
          else 0
    if A_p == 1: add all kept traces of p to B with weight w=1.0
    elif any kept trace: add the majority-answer kept trace with weight w=0.3

# Phase 1c: correlated-error diagnostic (estimates false-positive rate of A_p)
# Run once per round on a held-out diagnostic set D of 512 problems with known
# shared misconceptions (e.g., problems where a common sign/unit error yields a
# consistent wrong answer). For each p in D, compute A_p as above and record
# whether the agreed answer is verifier-correct. Report false-positive rate
# FPR = (# p with A_p=1 and agreed answer wrong) / (# p with A_p=1).
# This FPR bounds how often A_p reflects consensus rather than correctness.

# Phase 2: iterative self-training (replaces supervised seed stage)
for round in 1..R (R=3):
    G <- finetune(G, B, loss=weighted_cross_entropy)     # generator learns from its own verified traces
    # Phase 3: critic training via adversarial discrimination against a
    # synthetic "family-style" reference built from G's own high-agreement traces
    positives = traces in B with A_p == 1
    negatives = traces from G sampled at temperature 1.2 that fail V or fail step-consistency
    C <- finetune(C, positives vs negatives, loss=BCE)
    # Phase 4: refresh buffer with improved G
    B <- filter(B, keep only traces whose realism score under current C > 0.6)
    # Phase 5: re-run Phase 1c diagnostic with current G and C; log FPR

return G, C
```

Hyperparameters: n=8 samples per problem, temperature 0.9 for proposal and 1.2 for critic negatives, realism threshold 0.5 (buffer admission) and 0.6 (buffer refresh), step-consistency edit-distance threshold tau=0.3, R=3 rounds, agreement weight 1.0 vs 0.3, diagnostic set size |D|=512 problems, FPR computed per round.

**Why this design.** We chose a symbolic/executable verifier over a learned reward model because the verifier's correctness signal is independent of trace style, which is exactly what lets the loop bootstrap without a seed corpus; the cost is that the method only applies to problems with checkable final answers. We chose cross-sample agreement as the training-weight signal over per-trace realism alone because realism scores cannot distinguish a fluent wrong trace from a fluent right one, and agreement supplies the missing validity dimension; the cost is extra inference (8 samples per problem per round). We chose to train the critic adversarially against G's own high-agreement traces rather than against a fixed reference corpus because the reference must co-evolve with G or the critic saturates; the cost is that early rounds have a weak critic, which we absorb by weighting low-agreement traces at 0.3 rather than discarding them. We chose iterative rounds (R=3) over a single pass because the generator's proposal distribution must improve before agreement becomes informative; the cost is 3x training compute. A domain expert would not describe this as TraceForge plus a verifier: TraceForge's critic is trained to match a fixed family-style distribution supplied by the seed corpus, whereas here the critic's positive class is defined by verifier-passing cross-sample agreement, so the style reference is endogenous to the loop rather than exogenous.

**Why it measures what we claim.** The verifier output V(p,a) measures final-answer correctness because we assume the verifier is sound on the problem distribution (e.g., exact-match on MATH or execution on code); this assumption fails when problems admit multiple valid answer forms, in which case V reflects format-matching rather than correctness. The cross-sample agreement indicator A_p measures reasoning validity because we assume that independently sampled traces reaching the same verifiable answer via step-consistent paths are unlikely to do so by correlated error; this assumption fails on problems where a single systematic misconception is shared across samples (e.g., a common sign error), in which case A_p reflects consensus rather than correctness. The Phase 1c diagnostic directly estimates the rate of this failure by measuring the false-positive rate of A_p on a held-out set of known-misconception problems. The critic realism score r measures family-style fidelity because we assume the positive class (verifier-passing, high-agreement traces) is representative of the family's style; this assumption fails in early rounds when G's style is still far from the family, in which case r reflects self-consistency of G rather than family fidelity. The buffer weight w operationalizes trace trustworthiness because we assume agreement and verifier-passing are jointly sufficient for a trace to be worth training on; this assumption fails for problems with non-unique answers, in which case w reflects answer popularity rather than reasoning quality.

## Contribution

(1) VEST, a self-supervised bootstrapping loop that eliminates TraceForge's supervised seed-corpus stage by replacing it with verifier-gated cross-sample trace agreement as the training signal. (2) A design principle: cross-sample agreement under an independent correctness verifier is a usable substitute for family-trace supervision, and its reliability degrades predictably on problems with non-unique or systematically-misconceived answers. (3) An analysis of when endogenous critic training (critic positives drawn from the generator's own verified traces) matches or diverges from exogenous family-style supervision.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | MATH (unlabeled problems only) | Checkable final answers enable sound verifier signal |
| Dataset (secondary) | GSM8K (unlabeled problems only) | Second verifiable domain for robustness check |
| Primary metric | Accuracy on held-out test problems | Directly measures reasoning transfer, not style mimicry |
| Secondary metric | Agreement false-positive rate (FPR) on diagnostic set D (512 problems with known shared misconceptions) | Bounds how often A_p reflects consensus vs correctness |
| Baseline | Standard KD (teacher logits, no traces) | Tests whether trace-level structure is necessary at all |
| Baseline | TraceForge (supervised seed corpus) | Tests whether the seed corpus is the load-bearing ingredient |
| Baseline | Outcome-only self-training (verifier, no critic) | Isolates the critic's contribution to style fidelity |
| Baseline | Self-consistency majority-vote filtering (no step-alignment) | Tests whether agreement signal reduces to known self-consistency |
| Baseline | STaR-style verifier-only self-training (no critic, no agreement weighting) | Tests whether agreement weighting adds beyond verifier filtering |
| Ablation-of-ours | VEST minus agreement weighting (w=1.0 uniform) | Tests whether cross-sample agreement is causal |
| Ablation-of-ours | VEST minus step-alignment check (answer agreement only) | Isolates step-consistency from answer-level agreement |

### Why this setup validates the claim

The design is a three-way discrimination test of the central claim: that a verifier-gated, agreement-weighted self-training loop can replace a supervised seed corpus without losing reasoning quality. Standard KD tests the null hypothesis that trace-level supervision is unnecessary; if VEST merely matches it, the whole trace-generation machinery is unmotivated. TraceForge is the critical comparator: it has the seed corpus VEST lacks, so any gap between them attributes directly to the missing exogenous style reference, and any parity shows the endogenous critic substitutes for it. Outcome-only self-training removes the critic while keeping the verifier, so its gap isolates realism/style gating from correctness gating. The self-consistency majority-vote baseline and the STaR-style verifier-only baseline test whether the agreement signal is reducible to known self-consistency or verifier filtering alone; if VEST matches them, the step-alignment and weighting machinery is decorative. The agreement ablation removes the weighting signal while keeping everything else, isolating the validity dimension from the verifier alone. The step-alignment ablation removes the edit-distance check while keeping answer-level agreement, isolating the step-consistency component from answer consensus. The agreement FPR diagnostic on the 512-problem known-misconception set directly measures the load-bearing assumption: if FPR is high, A_p reflects consensus rather than correctness, and any accuracy gain cannot be attributed to reasoning validity. MATH is chosen because its verifiable final answers make the verifier sound on most problems, which is precisely the regime where VEST's assumptions should hold — a regime where they fail would confound the test. GSM8K provides a second verifiable domain to check that results are not MATH-specific. Accuracy on held-out problems is the right metric because it detects reasoning transfer rather than style imitation; a method that only learned family style would show flat accuracy despite high critic scores, which is exactly the failure the metric must catch.

### Expected outcome and causal chain

**vs. Standard KD (teacher logits, no traces)** — On a multi-step problem where the teacher's final answer is right but the intermediate reasoning is compressed into logits, standard KD transfers answer associations but not the step structure that generalizes to harder variants. Our method instead trains on verifier-passing, step-consistent traces, so the student sees explicit intermediate steps. We expect a noticeable accuracy gap on the harder half of the test set and near-parity on single-step problems.

**vs. TraceForge (supervised seed corpus)** — On a problem family where the seed corpus's style is idiosyncratic and mismatched to the student's own distribution, TraceForge's critic enforces an exogenous style the generator cannot natively produce, yielding fluent-but-stilted traces that drift from the student's reasoning. Our method's critic is trained against the generator's own high-agreement traces, so the style reference co-evolves. We expect rough parity overall, with VEST slightly behind in early rounds but closing or matching by round 3, and parity concentrated on families where seed style was most alien.

**vs. Outcome-only self-training (verifier, no critic)** — On a problem where the verifier accepts a correct answer reached by a degenerate shortcut (e.g., answer-guessing then backfilling), outcome-only training admits the trace because it never checks realism. Our method's critic rejects low-realism traces and the agreement check requires step-consistency, so the shortcut trace is filtered. We expect a visible gap on problems with high shortcut-availability, and parity on problems where the only path to a verified answer is genuine reasoning.

**vs. Self-consistency majority-vote filtering (no step-alignment)** — On a problem where the model systematically converges on the same wrong answer via a shared misconception, majority-vote filtering admits the wrong trace because it only checks answer agreement. Our method's step-alignment check requires pairwise step-consistency, which we expect to filter some correlated-error traces, but the diagnostic FPR will bound how often it fails. We expect VEST to outperform majority-vote filtering on problems where correlated errors produce step-inconsistent traces, and to match it where correlated errors are step-consistent (where FPR is high).

**vs. STaR-style verifier-only self-training (no critic, no agreement weighting)** — On a problem where the verifier accepts a correct answer but the trace is low-realism or step-inconsistent, STaR-style training admits it because it only checks the verifier. Our method adds critic realism and agreement weighting, so we expect a gap on problems with high shortcut-availability and on problems where the verifier is noisy. The agreement FPR diagnostic will show whether the gap is attributable to validity or to style.

**vs. VEST minus agreement weighting (w=1.0 uniform)** — On a problem where low-agreement traces are noisy, uniform weighting admits them at full weight, diluting the training signal. Our method downweights them to 0.3, so we expect a gap on problems with high trace variance. The agreement FPR diagnostic will show whether the weighting is justified by validity or is merely a variance-reduction heuristic.

**vs. VEST minus step-alignment check (answer agreement only)** — On a problem where the model converges on the same wrong answer via a shared misconception, answer-only agreement admits the wrong trace. Our method's step-alignment check requires pairwise step-consistency, so we expect a gap on problems where correlated errors produce step-inconsistent traces. The agreement FPR diagnostic will directly measure how often answer-only agreement fails.

### What would falsify this idea

If VEST's accuracy gain over outcome-only self-training is uniform across shortcut-available and shortcut-free problem subsets rather than concentrated where degenerate traces are plausible, then the critic is not doing the filtering work we claim and the agreement signal is decorative. Likewise, if VEST matches TraceForge only on style-similar families and collapses on families where seed style was most alien, the endogenous-critic claim is wrong. If the agreement FPR on the 512-problem known-misconception set is high (e.g., >0.3), then A_p reflects consensus rather than correctness and the load-bearing assumption is false.

## References

1. TraceForge: Distilling with Your Own Thinking
2. Let's Verify Step by Step
3. Orca: Progressive Learning from Complex Explanation Traces of GPT-4
4. Solving math word problems with process- and outcome-based feedback
5. Self-Consistency Improves Chain of Thought Reasoning in Language Models
6. Self-Instruct: Aligning Language Models with Self-Generated Instructions
7. Challenging BIG-Bench Tasks and Whether Chain-of-Thought Can Solve Them
