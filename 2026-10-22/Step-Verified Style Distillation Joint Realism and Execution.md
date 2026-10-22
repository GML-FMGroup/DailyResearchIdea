# Step-Verified Style Distillation: Joint Realism and Execution-Correctness Critics for Reasoning Traces

## Motivation

TraceForge and Orca both assume that a trace which looks stylistically native to the student family is a faithful carrier of the teacher's reasoning; TraceForge's critic judges only realism, and Orca's evaluation is explicitly flagged as potentially overestimating capability when style is imitated rather than reasoning. This assumption is structurally unsafe because style realism and step-level reasoning correctness are only loosely coupled in trace space — a trace can be perfectly in-family yet contain a fabricated arithmetic or logical step that the realism critic cannot see. The field therefore lacks a training signal that makes the causal link between style alignment and reasoning fidelity explicit and measurable rather than assumed.

## Key Insight

A realism critic's decision boundary and a step-execution oracle's correctness boundary are not independent when conditioned on the same intermediate state, so a single critic trained to score both from shared step representations yields a generator gradient that is monotone in reasoning fidelity rather than merely in stylistic plausibility.

## Method

**What it is.** We introduce **SVSD (Step-Verified Style Distillation)**, a two-head critic plus generator training procedure. Inputs: a context $c$, a teacher answer $a$, and a student-family style reference set $\mathcal{R}$. Outputs: a generator $G_\theta(c,a)\to$ trace $\tau=(s_1,\dots,s_T)$ in the family's style whose steps pass a step-execution oracle, and a critic $C_\phi$ that emits per-step realism scores $r_t\in[0,1]$ and per-step correctness scores $q_t\in[0,1]$ from a shared step encoder.

**How it works.**

```pseudocode
# Stage 0: build the execution oracle O (no human labels)
# O is globally conditioned: O(s_t) = 1 iff executing s_t on the symbolic/numeric state
# derived from s_<t preserves reachability of the teacher's final answer a under the executor.
# Concretely, for math: a Python arithmetic evaluator that, after executing s_t, checks whether
# the remaining steps of the teacher's reference derivation can still be applied to reach a
# (forward reachability) and whether the teacher's intermediate value at step t is reproduced
# (backward check). For logic: a forward-chaining rule engine that checks whether the goal
# literal is still derivable from the updated fact set.
for each (c, a) in teacher_answer_corpus:
    tau_ref = sample_native_trace(c)          # student family's own traces
    for t in 1..T:
        O(s_t | s_<t, c) = 1 if (executing s_t on the symbolic/numeric state derived from s_<t
            reproduces the teacher's intermediate value) AND (the teacher's final answer a remains
            reachable by applying the remaining steps of tau_ref to the updated state) else 0
    store (s_t, O(s_t)) pairs                         # cheap, automatic

# Stage 1: supervised warm-start of G (TraceForge-style)
min_theta  E_{(c,tau_ref)} [ -log G_theta(tau_ref | c, a) ]

# Stage 2: joint adversarial + correctness training
# Shared step encoder: 2-layer Transformer (hidden=768, 12 heads, GeLU), initialized from the
# generator's decoder; the two heads are linear projections (768→1) with sigmoid.
# Calibration/agreement term: we add a consistency loss L_agree = mean_t (r_t - q_t)^2 to the
# critic loss, and we gate the generator's correctness gradient by the realism score:
# L_G uses lambda_q * mean_t log( q_t * r_t ) instead of lambda_q * mean_t log q_t, so a step
# must be both realistic and correct to receive correctness credit.
for iteration in 1..N:
    # ---- critic update ----
    tau_real ~ R (family's own traces); tau_gen = G_theta(c,a)
    for each step t in tau_real and tau_gen:
        r_t = C_phi.realism(h_t)          # h_t = shared step encoder
        q_t = C_phi.correct(h_t)          # same encoder, 2nd head
    L_C = BCE(r_t, 1[tau_real]) + BCE(q_t, O(s_t)) + 0.1 * mean_t (r_t - q_t)^2
    phi <- phi - lr * grad L_C

    # ---- generator update ----
    tau_gen = G_theta(c,a)
    L_G = -lambda_r * mean_t log r_t          # realism (as in TraceForge)
          - lambda_q * mean_t log( q_t * r_t )  # step-correctness gated by realism
          - lambda_ans * 1[final(tau_gen)=a]   # answer consistency
    theta <- theta - lr * grad L_G
```

Hyperparameters: $\lambda_r=1.0$, $\lambda_q=1.0$, $\lambda_{ans}=0.5$; critic and generator alternate one update each; oracle $O$ is the globally-conditioned symbolic executor described in Stage 0 (for math: a Python arithmetic evaluator with forward/backward reachability; for logic: a forward-chaining rule engine with goal-reachability check). The shared step encoder is a 2-layer Transformer (hidden=768, 12 heads, GeLU) initialized from the generator's decoder; the two heads are linear projections (768→1) with sigmoid. The agreement term weight is 0.1. We freeze the encoder for the first 500 critic steps.

**Why this design.** We chose a *shared step encoder with two heads* over two independent critics because the realism and correctness signals must be forced to agree on the same intermediate state representation — otherwise the generator can satisfy realism with steps that the correctness head never sees, which is exactly the TraceForge failure mode; the cost is that the two heads can interfere during early training, which we mitigate by freezing the encoder for the first 500 critic steps and by adding a consistency loss $0.1\cdot\text{mean}_t(r_t-q_t)^2$ to the critic objective. We chose an *automatic execution oracle* over human step labels (as in Let's Verify Step by Step) because human annotation is the binding cost for distillation at scale; the cost is that the oracle is only available for domains with executable intermediate state, so we scope to math and symbolic logic. We chose to *add the correctness term to the generator loss directly* rather than rerank candidates at inference (as in process-reward reranking) because reranking cannot change the generator's step distribution and therefore cannot close the causal gap between style and fidelity; the cost is that the generator must be trained, not just decoded. This is not a Process Reward Model + TraceForge composition: a PRM scores a fixed trace, whereas here the same encoder's correctness head is back-propagated into the trace *writer*, and the realism head is trained adversarially against that writer, so the two signals are mutually coupled rather than stacked.

**Why it measures what we claim.** The per-step correctness score $q_t = C_\phi.\text{correct}(h_t)$ measures *step-level reasoning fidelity* because we assume the execution oracle $O(s_t)$ is a sufficient statistic for whether step $t$ advances the derivation toward the teacher's answer; this assumption fails when a step is locally correct but globally off-track (e.g., a valid arithmetic manipulation applied to the wrong subgoal), in which case $q_t$ reflects local executability rather than global derivation validity. The per-step realism score $r_t$ measures *style alignment with the student family* because we assume the family's own trace set $\mathcal{R}$ is an adequate sample of the family's style distribution; this assumption fails for families with sparse or heterogeneous traces, in which case $r_t$ reflects similarity to the available reference subset rather than the family's true style. The joint generator objective $\lambda_r\log r_t + \lambda_q\log(q_t r_t)$ measures the *causal fidelity of style-aligned traces* because we assume that maximizing both heads on the same step representation forces the generator to place probability mass on steps that are simultaneously in-style and execution-verified; this assumption fails if the two heads' gradients conflict on a step, in which case the generator optimizes a weighted compromise that may sacrifice correctness for realism — which is why we log per-step head disagreement during training as a diagnostic.

## Contribution

['SVSD, a joint realism-and-execution-correctness critic with a shared step encoder and two heads, trained adversarially against a style-conditioned trace generator, with an automatic execution oracle replacing human step labels.', 'A design principle: coupling style realism and step-correctness on a shared step representation makes the generator gradient monotone in reasoning fidelity, whereas decoupled critics admit style-only optima; we operationalize this via per-step head-disagreement logging as a training diagnostic.', 'A reproducible two-stage training recipe (supervised warm-start + alternating critic/generator updates) for distilling teacher answers into student-family-style traces that are step-verified, applicable to math and symbolic-logic domains.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | GSM8K + ProofWriter (math + symbolic logic) | Only domains with executable intermediate state for the oracle |
| Primary metric | Step-execution-oracle pass rate on generated traces | Directly measures fidelity the method claims to fix |
| Baseline | Outcome-supervised distillation (final-answer KD) | Tests whether step-level signal beats answer-only supervision |
| Baseline | TraceForge (realism-only adversarial traces) | Tests whether correctness head closes the style-fidelity gap |
| Baseline | PRM + TraceForge reranking | Tests whether backprop beats inference-time reranking |
| Ablation-of-ours | SVSD minus correctness head (realism-only, shared encoder) | Isolates the causal contribution of the coupled q-head |
| Ablation-of-ours | SVSD minus agreement term (no consistency loss, no gating) | Tests whether the calibration term is necessary for coupling |

### Why this setup validates the claim

The central claim is that coupling a realism head and an execution-oracle correctness head on a shared step encoder, and back-propagating both into the trace writer, produces traces that are simultaneously in-family-style and step-verified — a joint property no stacked baseline can achieve. Each baseline isolates one sub-claim. Outcome-supervised distillation tests whether step-level signal matters at all versus answer-only supervision. TraceForge tests whether realism alone suffices, i.e. whether style-aligned traces are already fidelity-aligned. PRM-plus-reranking tests whether the correctness signal must enter the generator's gradient or can be applied post hoc. The ablation removing the correctness head while keeping the shared encoder isolates whether the q-head's gradient is the causal driver rather than the encoder sharing. The additional ablation removing the agreement term tests whether the calibration mechanism (consistency loss and gating) is necessary for the two heads to cooperate rather than conflict. The primary metric — oracle pass rate on generated traces — is the right detector because it is exactly the quantity the correctness head is trained against; if the method's mechanism works, this metric must move, and it moves only on subsets where the predicted failure mode (locally-plausible but unverified steps) occurs. A uniform gain would indicate the metric is confounded by general trace quality rather than the specific coupling we claim.

### Expected outcome and causal chain

**vs. Outcome-supervised distillation** — On a multi-step GSM8K item where the teacher's answer is reachable by several derivations, answer-only distillation assigns equal credit to a fluent-but-wrong intermediate step and a correct one because its loss sees only the final token. Our method instead scores each step against the execution oracle and back-propagates that per-step signal, so the wrong intermediate step is directly penalized. We expect a large oracle-pass-rate gap on multi-step items and near-parity on single-step items, where the final-answer signal already suffices.

**vs. TraceForge (realism-only)** — On a logic proof where a locally valid inference is applied to the wrong subgoal, TraceForge's realism critic rates the step as in-style and rewards it, because realism is blind to execution state; the generator drifts toward stylish but off-track traces. Our method's correctness head, sharing the encoder, flags the step as oracle-failing and its gradient suppresses it. We expect TraceForge to show high style-similarity but depressed oracle pass rate on off-track-prone subsets, and our method to close that gap while holding style similarity roughly constant.

**vs. PRM + TraceForge reranking** — On a case where the generator's step distribution systematically favors a plausible wrong step, reranking over samples cannot recover a correct trace because no sample contains the right step; the PRM scores a fixed distribution it cannot reshape. Our method instead changes the generator's step distribution via the q-head gradient. We expect reranking to plateau at a pass rate bounded by sample coverage, while our method exceeds that ceiling on subsets where the correct step is rare but learnable, with the gap widening as sample budget at inference is held fixed.

**vs. SVSD minus correctness head** — This ablation keeps the shared encoder and realism head but removes the q-head gradient. If the correctness head is the causal driver, this ablation should match TraceForge's oracle pass rate and fall well below full SVSD on off-track-prone subsets, while style similarity remains high. A result where this ablation matches full SVSD would falsify the claim that the q-head's gradient, not encoder sharing, produces the fidelity gain.

**vs. SVSD minus agreement term** — This ablation removes the consistency loss and the gating of the correctness gradient by realism. If the calibration term is necessary for the two heads to cooperate, this ablation should show higher per-step head disagreement and lower oracle pass rate than full SVSD, especially on steps where realism and correctness would otherwise conflict. A result where this ablation matches full SVSD would indicate the agreement term is not needed for coupling.

### What would falsify this idea

If the oracle-pass-rate gain over TraceForge is uniform across subsets rather than concentrated on off-track-prone multi-step items, or if the ablation removing the correctness head matches full SVSD, then the claimed coupling is not the causal driver and the idea is wrong. A second falsifier: if per-step head-disagreement logging shows the generator systematically trading correctness for realism (q-head loss rising as r-head loss falls) without net oracle-pass-rate improvement, the two-head coupling fails rather than helps.

## References

1. TraceForge: Distilling with Your Own Thinking
2. Let's Verify Step by Step
3. Orca: Progressive Learning from Complex Explanation Traces of GPT-4
4. Solving math word problems with process- and outcome-based feedback
5. Self-Consistency Improves Chain of Thought Reasoning in Language Models
6. Self-Instruct: Aligning Language Models with Self-Generated Instructions
7. Challenging BIG-Bench Tasks and Whether Chain-of-Thought Can Solve Them
