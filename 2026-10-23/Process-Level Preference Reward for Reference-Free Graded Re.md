# Process-Level Preference Reward for Reference-Free Graded Research Quality

## Motivation

Existing deep-research benchmarks verify outcomes only at the task level and binarily: OpenDS-World uses independent human verification of evidence validity and answer uniqueness, which certifies that a task is well-posed but yields a single pass/fail bit that cannot distinguish a trace that found two of five required sources and synthesized them cleanly from one that found all five but garbled the synthesis. DeepResearch Bench tried to recover graded quality by scoring against reference reports with adaptive criteria, but that reintroduces the reference-report dependency the field is trying to escape, and its cost scales with the number of tasks rather than the number of intermediate decisions. Structurally, both designs throw away the richest supervision signal available — the dense, comparative human judgments that annotators can make cheaply about two intermediate research traces — because neither has a mechanism that turns pairwise process preferences into a continuous, reference-free score.

## Key Insight

If human annotators only ever judge which of two intermediate research traces is better at a given step, then a reward model trained on those pairwise comparisons inherits a total order over trace prefixes that is anchored to observable evidence coverage and synthesis coherence rather than to any reference report, so its scalar output is a graded quality measure by construction.

## Method

**What it is.** We introduce GRADE (Graded Research-quality Estimator from process preferences), a Bradley–Terry reward model over intermediate research traces. Its input is a research trace prefix — the ordered sequence of (search query, selected evidence spans, synthesis step) that an agent has produced so far — and its output is a scalar quality score q ∈ ℝ for that prefix. GRADE is trained on process-level human preference pairs and, at inference, scores prefixes and whole traces without any reference report.

**Load-bearing assumption (stated explicitly).** GRADE assumes that human annotators judging two aligned intermediate decision points (query, evidence span, or synthesis sentence) in isolation, without seeing the final answer, produce preferences that are consistent enough across annotators and across tasks that a single Bradley–Terry latent score trained on them recovers a task-independent, reference-free graded quality order rather than a per-task or per-annotator idiosyncratic ordering. This assumption is load-bearing: if annotators disagree systematically by domain (e.g., a biologist weighting primary sources over news), q reflects the majority annotator's ordering rather than an annotator-independent quality. We therefore (i) train a hierarchical Bradley–Terry model with a per-task random intercept (shared prior N(0, σ²), σ=1.0) so that cross-task comparability is estimated rather than assumed, (ii) retain and model low-confidence pairs with an explicit annotator-noise term (Dawid–Skene-style latent-true-label model over the 5-point confidence), and (iii) gate the headline claim on agreement with an unfiltered, cross-annotator held-out set.

**How it works.**

```pseudocode
# Phase 1: trace segmentation (offline, over OpenDS-World rollouts)
for each task t in OpenDS-World:
    collect K=6 rollouts from diverse agents (temperature 0.7, max 40 steps)
    segment each rollout into decision points d_i = (query_i, evidence_i, synthesis_i)
    # a decision point is emitted whenever the agent issues a query,
    # commits an evidence span, or writes a synthesis sentence

# Phase 2: process-preference elicitation (human annotation)
for each task t, for each pair of rollouts (A, B):
    align A and B at the k-th decision point (k = 1..min(|A|,|B|))
    show annotators the shared context up to k-1 plus the two candidate
    decision points d_k^A, d_k^B; ask: which is better for completing t?
    collect preference label y ∈ {A, B, tie} with a 5-point confidence
    # ~1,800 tasks x ~15 aligned pairs => ~27k preference pairs
    # retain ALL pairs (including |confidence| < 2) for training with noise model
    # hold out 10% of tasks (not pairs) for cross-task transfer evaluation

# Phase 3: reward model training
encoder E = frozen long-context LLM (32k tokens), mean-pool last layer
scorer  s(prefix) = MLP(E(prefix)) -> scalar q        # 2-layer, hidden 1024
# hierarchical Bradley-Terry with per-task random intercept:
#   s(d_k^A | ctx, t) = f(E(d_k^A, ctx)) + b_t
#   s(d_k^B | ctx, t) = f(E(d_k^B, ctx)) + b_t
#   b_t ~ N(0, sigma^2), sigma=1.0 (shared prior)
# loss = -log sigmoid( s(d_k^A | ctx, t) - s(d_k^B | ctx, t) )
#        weighted by annotator confidence (1-5 mapped to [0.2, 1.0])
#        + Dawid-Skene noise term: latent true label z ~ Bernoulli(pi_t)
#          P(y | z, confidence) modeled as 5x2 confusion matrix per annotator
#        AdamW, lr 1e-4, batch 32, 3 epochs
calibration head: a held-out 10% of tasks fit isotonic regression from
       (score gap) -> P(preferred); used to report reliability, not to gate

# Phase 4: inference (reference-free scoring)
q_trace = mean over decision points of s(d_i | ctx_{<i})   # per-task intercept b_t set to 0 at inference for cross-task comparability
q_complete = s(final synthesis | full context)
GRADE(trace) = 0.6 * q_trace + 0.4 * q_complete
```

**Why this design.** We chose pairwise process preferences over absolute per-step Likert ratings because absolute scales drift across annotators while pairwise comparisons are internally consistent, accepting the cost that we must align traces at decision points and discard unalignable pairs. We chose a Bradley–Terry head over a regression head on absolute scores because BT's likelihood is invariant to monotone rescaling of the latent score, which means the learned q is comparable across tasks without per-task normalization, accepting the cost that BT only constrains score *differences* and so we must fix the intercept. We chose to drop low-confidence pairs rather than model annotator noise explicitly because a noise model would require repeated annotations we cannot afford at 27k pairs, accepting the cost of a smaller but cleaner training set. Crucially, a domain expert would not describe GRADE as a Process Reward Model (Lightman 2023) variant: PRMs score *correctness of a reasoning step toward a known answer* and require a verifier or ground-truth answer to define the step label, whereas GRADE's labels are comparative human judgments about evidence coverage and synthesis quality with no answer in the loop, and its target quantity is a graded trace-level quality rather than a step-correctness probability.

**Why this measures what we claim.** The scalar q_trace measures graded research quality because the BT likelihood makes q monotone in the annotator's latent preference order under the assumption that annotators share a single ordering over trace prefixes; this assumption fails when annotators disagree systematically by domain (e.g., a biologist weighting primary sources over news), in which case q reflects the majority annotator's ordering rather than an annotator-independent quality. The evidence-span decision points measure completeness because a preference between two evidence selections is driven by how many independent required sources each covers under the assumption that annotators can recognize source independence without knowing the final answer; this fails when a source is redundant but unfamiliar, in which case the score reflects perceived diversity rather than true coverage. The synthesis-step score measures coherence because pairwise synthesis preferences track whether claims are supported by the cited spans under the assumption that annotators read the cited spans when judging; this fails when annotators skim only the synthesis text, in which case the score reflects surface fluency rather than grounding.

## Contribution

(1) GRADE, a Bradley–Terry reward model over segmented research-trace prefixes trained on confidence-filtered process-level human preference pairs, which produces a continuous reference-free quality score for deep-research traces. (2) A trace-segmentation and alignment procedure that turns unaligned agent rollouts into comparable decision points, enabling pairwise annotation of intermediate research steps at roughly 15 pairs per task. (3) A calibrated reliability analysis showing how isotonic-mapped score gaps track annotator agreement, giving a usable confidence band on the graded score.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | OpenDS-World held-out tasks (10% of tasks held out by task, not by pair) | Multi-step live-web research with traceable decision points; task-level holdout enables cross-task transfer evaluation. |
| Primary metric | Agreement with held-out human preference pairs (Kendall τ) on unfiltered, cross-annotator held-out set | Directly tests whether q ranks traces as humans do, including the ambiguous region. |
| Secondary metric | Cross-task transfer: Kendall τ on held-out tasks (task-level split) | Tests whether the hierarchical BT with per-task intercept yields task-independent ranking. |
| Baseline | Frontier-model LLM-as-judge | Tests whether process preferences beat holistic judging. |
| Baseline | Process Reward Model (Lightman-style) | Tests whether answer-free quality scoring beats step-correctness. |
| Baseline | Final-report-only BT reward | Tests whether intermediate trace scoring adds signal. |
| Ablation-of-ours | GRADE w/o synthesis term (q_trace only) | Isolates contribution of the final-synthesis decision point. |
| Ablation-of-ours | GRADE w/o hierarchical intercept (single global BT) | Tests whether per-task random intercept improves cross-task transfer. |
| Ablation-of-ours | GRADE w/o noise model (drop |confidence|<2 pairs) | Tests whether retaining low-confidence pairs with Dawid-Skene noise term improves agreement on ambiguous region. |

### Why this setup validates the claim

The central claim is that a reference-free, process-level Bradley–Terry score over research trace prefixes ranks traces the way human annotators do, and that this ranking signal is unavailable to systems that judge only final reports or only step correctness. OpenDS-World is the right dataset because its traces contain the exact decision points (query, evidence span, synthesis sentence) that GRADE scores, and because it is hard enough that frontier agents diverge in trace quality, giving the metric room to separate methods. The primary metric — rank agreement with held-out human preference pairs drawn from an unfiltered, cross-annotator pool — is the right detector because GRADE's only supervised signal is precisely such pairwise preferences, so if the mechanism works, agreement on unseen pairs must rise; any other metric (e.g., answer accuracy) would confound ranking quality with retrieval luck. The task-level holdout and secondary cross-task transfer metric directly test the load-bearing assumption that the BT score is task-independent rather than per-task idiosyncratic. The LLM-as-judge baseline tests the sub-claim that scalar holistic judging of a whole trace is coarser than aligned decision-point comparison; the PRM baseline tests the sub-claim that answer-free comparative quality is not reducible to step-correctness; the report-only baseline tests the sub-claim that intermediate trace information matters at all. The ablations test whether the synthesis term, hierarchical intercept, and noise model each contribute independently. Together these form a falsifiable design: the claim predicts a specific ordering of methods concentrated on the trace subsets where each baseline's mechanism gap bites, not a uniform advantage.

### Expected outcome and causal chain

**vs. Frontier-model LLM-as-judge** — On a case where two traces reach similar final answers but one skipped an independent primary source mid-search, the holistic judge produces near-identical scores because it reads the finished report and cannot localize the omission. Our method instead compares the aligned evidence-selection decision point against human preferences because it scores prefixes, so we expect a clear separation on the evidence-coverage subset and near-parity on traces whose final reports differ obviously.

**vs. Process Reward Model (Lightman-style)** — On a case where an agent's intermediate step is locally reasonable but poorly grounded in the cited span, the PRM produces a high step score because its labels encode progress toward a known answer, which this task lacks, so it degrades to a fluency proxy. Our method instead scores the synthesis decision point against comparative human judgments about grounding because its labels never reference an answer, so we expect a noticeable gap on the grounding-sensitive subset and parity where steps are trivially correct.

**vs. Final-report-only BT reward** — On a case where a trace recovers the right sources early but drifts into redundant queries later, the report-only model produces a high score because the final text looks complete, discarding the wasted-search signal. Our method instead penalizes the redundant decision points via the averaged prefix score because it observes the whole trace, so we expect a gap concentrated on long traces with mid-trace redundancy and parity on short, clean traces.

**vs. GRADE w/o synthesis term** — On a case where evidence selection is strong but the synthesis sentence overstates a claim, the ablation produces a high score because it averages only query and evidence decisions, missing the grounding failure. Our method instead lowers the score through the q_complete term because the synthesis decision point is explicitly scored, so we expect the full model to outperform the ablation specifically on the grounding-failure subset.

**vs. GRADE w/o hierarchical intercept** — On a case where tasks differ in annotator population and evidence pool, the single global BT produces scores that are not comparable across tasks because it assumes a shared intercept. Our method instead estimates a per-task random intercept with a shared prior, so we expect higher cross-task transfer Kendall τ on held-out tasks, especially when task difficulty varies.

**vs. GRADE w/o noise model** — On a case where annotators are uncertain (|confidence| < 2), the filtered model discards these pairs and trains only on the easy, high-agreement subset, inflating held-out agreement on that same filtered distribution. Our method instead retains all pairs with a Dawid–Skene noise term, so we expect better agreement on the unfiltered, cross-annotator held-out set, particularly on the ambiguous middle where the graded signal lives.

### What would falsify this idea

If GRADE's advantage over the LLM-as-judge and report-only baselines is uniform across all trace subsets rather than concentrated on the evidence-coverage and grounding subsets where those baselines' mechanism gaps predict failure, the claim that process-level comparative scoring is the causal driver would be wrong. Likewise, if the synthesis-term ablation matches the full model everywhere, the claimed contribution of scoring the final synthesis decision point is falsified. If the hierarchical-intercept and noise-model ablations do not improve cross-task transfer and unfiltered agreement respectively, the load-bearing assumption that a single BT score recovers a task-independent graded quality order is falsified.

## References

1. Deep Search in Real-World Web Environments
2. DeepResearch Bench: A Comprehensive Benchmark for Deep Research Agents
3. BrowseComp-Plus: A More Fair and Transparent Evaluation Benchmark of Deep-Research Agent
4. HelloBench: Evaluating Long Text Generation Capabilities of Large Language Models
5. LAB-Bench: Measuring Capabilities of Language Models for Biology Research
6. ScienceAgentBench: Toward Rigorous Assessment of Language Agents for Data-Driven Scientific Discovery
7. BRIGHT: A Realistic and Challenging Benchmark for Reasoning-Intensive Retrieval
8. GPQA: A Graduate-Level Google-Proof Q&A Benchmark
