# Task-Conditioned Abstention for Interleaved Visual Reasoning

## Motivation

Interleaved-reasoning recipes such as ThinkMorph-SSL assume that generating more visual thoughts is monotonically beneficial, training the model to produce restored or canonical images and interpret them on every input regardless of task type. Yet MME-CoT (E4) empirically documents that CoT prompting often degrades LMM performance on perception-heavy tasks, a harmful overthinking failure mode. ThinkMorph-SSL's pretext-derived supervision contains no structural signal distinguishing inputs where visual thoughts help from those where they hurt, so the field lacks a training-time criterion for when interleaved visual reasoning should be suppressed.

## Key Insight

A held-out, task-stratified counterfactual margin between the model's answer quality with and without an interleaved visual trace is a directly estimable, monotone-in-correctness signal that can be distilled into a pre-generation abstention head, making abstention a learned function of input-task structure rather than of the model's own next-token likelihood.

## Method

**What it is.** We introduce TIGER (Task-conditioned Interleaved-Gate for Efficient Reasoning), a training recipe that augments ThinkMorph-SSL's two-stage pipeline with a pre-generation abstention head g_φ and a counterfactual-margin supervision signal. Inputs: a multimodal question (image, text). Outputs: either a direct answer or an interleaved visual-thought trace (describe → generate restored/canonical image → interpret → answer), chosen by g_φ before any reasoning tokens are emitted.

**Load-bearing assumption (stated explicitly).** The counterfactual margin m(x) is treated as a causal measure of whether interleaved visual reasoning helps for input x. This requires assumption A: forcing the trace versus suppressing it is a valid intervention on the reasoning process while holding the input fixed, with negligible shared-computation and decoding-variance contamination between the two paths. A fails when the forced-trace and direct-answer paths share enough computation that the margin reflects decoding variance rather than a genuine reasoning-mode difference, or when per-domain disagreement cells are too small to distinguish the margin from Bernoulli sampling noise. We operationalize this with a seed-variance diagnostic (Phase 0b below) that estimates the contamination rate and a threshold below which m(x) is treated as noise.

**How it works.**

```pseudocode
# Phase 0: build counterfactual margin labels (held-out probe set, size N=2000)
#   stratified by task_type in {math, science, OCR, logic, spacetime, general}
#   each probe is drawn from MME-CoT's six-domain suite, held out from ThinkMorph-SSL stage-2 demos
for each probe (x, y*) stratified by task_type:
    # multi-sample, variance-reduced correctness estimate per condition
    # k=5 stochastic decodes per condition (temperature=0.7, top_p=0.9, distinct seeds)
    c_direct = (1/k) * sum_{i=1..k} 1[correct(M0(x; seed_i)) == 1]
    c_trace  = (1/k) * sum_{i=1..k} 1[correct(M0(x, force_interleaved; seed_i)) == 1]
    # 95% Wilson interval on the paired difference; keep only statistically distinguishable probes
    (lo, hi) = wilson_ci_paired(c_trace, c_direct, k=5)
    if lo > 0:   m_soft(x) = +1   # trace-helps
    elif hi < 0: m_soft(x) = -1   # trace-hurts
    else:        m_soft(x) = 0    # tie / indistinguishable -> discarded
    # continuous target for soft distillation (used in Phase 1 and Phase 2)
    m_cont(x) = c_trace - c_direct   # in [-1, +1]

# Phase 0b: seed-variance diagnostic (contamination check)
#   for each probe, recompute m_soft(x) under 3 independent seed blocks (k=5 each)
#   flip_rate[task_type] = fraction of probes whose m_soft sign changes across blocks
#   if flip_rate[task_type] > 0.20: mark task_type as NO-GATE (default to direct at inference)
#   report per-domain disagreement-set sizes and flip rates in the results table

# Phase 1: train abstention head g_phi (2-layer MLP, hidden=512, GeLU) on frozen M0
#   input: mean-pooled penultimate-layer hidden state h(x) (dim d_model) concatenated with
#          a 6-dim task-type one-hot (math/science/OCR/logic/spacetime/general)
#   loss: binary cross-entropy to soft target (m_cont(x)+1)/2, class-balanced by task_type,
#         lr=1e-4, 10 epochs, batch size 64
#   calibration: temperature scaling on a held-out 20% split of the disagreement set
#   NO-GATE domains (from Phase 0b) are excluded from g_phi training

# Phase 2: joint fine-tune of M0 + g_phi with task-conditioned abstention objective
for each training batch from ThinkMorph-SSL stage-2 demonstrations:
    p_help = sigmoid(g_phi(h(x), task_onehot) / T)   # T from Phase 1
    if task_type in NO-GATE: route = direct
    elif p_help >= tau (tau=0.5): route = interleaved
    else: route = direct
    if route == interleaved:
        L = L_answer(interleaved_trace) + lambda * L_pretext   # lambda=0.5
    else:
        L = L_answer(direct_answer)                            # no trace tokens
    L_total = L + beta * BCE(g_phi(...), (m_cont(x)+1)/2)      # beta=0.3, soft margin-distill
    update M0 and g_phi jointly

# Inference: compute p_help once from a single forward pass of the prompt;
#   if task_type in NO-GATE: emit direct answer
#   elif p_help >= tau: emit direct answer
#   else: emit interleaved trace
```

**Why this design.** We chose to train g_φ on a *counterfactual margin* from a frozen model rather than on the model's own confidence or log-probability, because log-prob calibration for task correctness is a known open problem (Kadavath et al.; Tian et al.) and would make the gate a repackaged confidence threshold; the cost is that Phase 0 requires 2·k·N forward passes (k=5, N=2000 → 20,000 forward passes) and a disagreement set that may be small on easy benchmarks. We chose a *pre-generation* gate over a post-hoc trace-truncation controller, because a controller that decides mid-trace is the two-module-router anti-pattern where the router itself is the hard problem; the cost is that g_φ sees only the input, not intermediate trace content, so it cannot exploit evidence discovered mid-reasoning. We chose *task-type one-hot concatenation* over a purely learned task embedding, because MME-CoT's finding is explicitly stratified by domain (perception-heavy vs reasoning-heavy) and we want the gate to be auditable per domain; the cost is that the six-domain taxonomy must be supplied at both train and test time, which is a mild annotation dependency. A domain expert would not describe TIGER as a variant of ThinkMorph-SSL + a router, because the gate is trained on a *held-out counterfactual margin* that ThinkMorph-SSL's pretext supervision never produces, and the abstention decision is baked into the training loss (Phase 2) rather than bolted on at inference.

**Why it measures what we claim.** The counterfactual margin m(x) measures *whether interleaved visual reasoning is causally helpful for this input* because we assume (A) that forcing the trace versus suppressing it is a valid intervention on the reasoning process while holding the input fixed, and that the multi-sample estimate with a 95% Wilson interval separates genuine reasoning-mode benefit from decoding variance; this assumption fails when the forced-trace and direct-answer paths share enough computation that the margin reflects decoding variance rather than a genuine reasoning-mode difference, or when per-domain disagreement cells are too small to distinguish the margin from Bernoulli sampling noise — in which case m(x) reflects noise in the forward passes instead of task-conditioned benefit, and the Phase 0b seed-variance diagnostic (flip_rate > 0.20) flags the affected task type as NO-GATE. The abstention head's calibrated probability p_help measures *the input's task-conditioned need for visual thoughts* because we assume the frozen model's penultimate hidden state h(x) plus task one-hot is a sufficient statistic for the trace-benefit label; this assumption fails when a task type is underrepresented in the disagreement set, in which case p_help reflects the head's prior over that task type rather than input-specific structure, and the NO-GATE fallback defaults to direct answering. The Phase-2 margin-distillation loss measures *consistency between the gate and the counterfactual ground truth* because we assume the disagreement-set distribution matches the fine-tuning distribution; this assumption fails under distribution shift between ThinkMorph-SSL's automatically constructed demonstrations and the probe set, in which case the loss reflects the gate's fit to the probe distribution rather than to deployment inputs.

**Hyperparameters.** | Name | Value | Where used |
|---|---|---|---|
| N (probe set) | 2000 | Phase 0 |
| k (decodes per condition) | 5 | Phase 0 |
| decode temperature / top_p | 0.7 / 0.9 | Phase 0 |
| Wilson CI level | 95% | Phase 0 |
| seed blocks (Phase 0b) | 3 | Phase 0b |
| flip_rate threshold (NO-GATE) | 0.20 | Phase 0b |
| hidden dim | 512 | g_φ |
| activation | GeLU | g_φ |
| lr (g_φ) | 1e-4 | Phase 1 |
| epochs (g_φ) | 10 | Phase 1 |
| batch size (g_φ) | 64 | Phase 1 |
| τ (abstention threshold) | 0.5 | Phase 2 / inference |
| λ (pretext weight) | 0.5 | Phase 2 |
| β (margin-distill weight) | 0.3 | Phase 2 |
| T (temperature) | fitted | Phase 1 calibration |

## Contribution

['TIGER, a training recipe that adds a pre-generation task-conditioned abstention head to ThinkMorph-SSL, supervised by a held-out counterfactual margin between forced-trace and direct-answer correctness rather than by model confidence.', 'A margin-distillation objective that jointly fine-tunes the interleaved-reasoning model and the abstention head, so the model learns to allocate visual-thought generation only to inputs where the counterfactual margin is positive.', "A task-stratified disagreement-set construction procedure that yields auditable per-domain abstention rates, enabling direct comparison against MME-CoT's perception-heavy degradation finding."]

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|---|---|---|
| Dataset | MME-CoT six-domain suite | Stratified perception vs reasoning domains match gate design |
| Primary metric | Accuracy on reasoning-heavy vs perception-heavy split | Directly tests whether gating recovers both regimes |
| Secondary metric | Fraction of inputs routed to trace per domain | Community-usable diagnostic of gate behavior |
| Secondary metric | Seed-flip rate of disagreement labels per domain | Bounds validity of counterfactual margin |
| Baseline | ThinkMorph-SSL (always-trace) | Tests cost of unconditional visual thoughts |
| Baseline | ThinkMorph-SSL + always-direct | Tests benefit lost by suppressing traces |
| Baseline | Confidence-threshold gate on M0 logits | Tests whether counterfactual margin beats confidence |
| Baseline | Text-only-CoT margin gate (non-interleaved) | Tests whether interleaved-trace margin carries unique signal |
| Ablation-of-ours | TIGER without task-type one-hot | Tests whether domain conditioning drives the gain |
| Ablation-of-ours | TIGER with single-sample margin (k=1) | Tests whether variance reduction is load-bearing |

### Why this setup validates the claim

The central claim is that a pre-generation gate trained on a counterfactual margin, rather than on model confidence, correctly routes inputs to either a direct answer or an interleaved visual-thought trace, recovering trace benefits on reasoning-heavy inputs while avoiding trace harm on perception-heavy ones. MME-CoT is the right dataset because it is explicitly stratified into perception-heavy and reasoning-heavy domains and already documents that CoT prompting can degrade perception tasks — exactly the heterogeneity our gate must exploit. The always-trace baseline tests the sub-claim that unconditional traces hurt perception inputs; the always-direct baseline tests the sub-claim that suppressing traces loses reasoning gains; the confidence-gate baseline tests the sub-claim that the counterfactual margin is a better routing signal than log-probability. The text-only-CoT margin gate isolates whether the interleaved-trace-specific margin carries routing signal that a text-CoT margin does not, ruling out reduction to a generic router. The task-one-hot ablation isolates whether the auditable domain conditioning, rather than generic input features, produces the routing. The single-sample-margin ablation isolates whether the variance-reduced multi-sample estimate (k=5 with Wilson CI) is load-bearing versus the original single-pass label. Accuracy on the two-way split is the correct metric because a uniform gain would not distinguish our mechanism from a stronger base model, whereas a crossed pattern — gain on reasoning, parity or slight gain on perception — is the signature the causal story predicts. The per-domain trace-routing fraction and seed-flip rate are reported as diagnostics that bound the validity of the margin as a causal intervention.

### Expected outcome and causal chain

**vs. ThinkMorph-SSL (always-trace)** — On a perception-heavy case such as an OCR or fine-grained scene question, always-trace produces an interleaved describe-generate-interpret trace that overwrites or distorts the original visual evidence, because the pretext-style restoration step is not answer-relevant and injects hallucinated canonical content. Our method instead computes p_help from the prompt and routes to a direct answer because the counterfactual margin labeled this input as trace-hurting during Phase 0. We expect a noticeable accuracy gain on the perception subset with parity on the reasoning subset, and shorter average output length on the perception subset.

**vs. ThinkMorph-SSL + always-direct** — On a reasoning-heavy case such as a spatial or multi-step math question, always-direct produces a single-shot answer that skips the intermediate visualization needed to track relations, because the base model's direct path lacks the visual scaffolding. Our method instead routes to the interleaved trace because the margin labeled this input as trace-helping, so we expect a clear gain on the reasoning subset with parity on the perception subset — the mirror image of the always-trace pattern.

**vs. Confidence-threshold gate on M0 logits** — On a case where the model is confidently wrong under direct answering (a known calibration failure), the confidence gate routes to direct answering because high log-probability is mistaken for correctness, so the trace that would have corrected the error is never emitted. Our method instead routes by the counterfactual margin, which is defined on correctness disagreement rather than probability, so we expect the margin gate to outperform the confidence gate specifically on the confidently-wrong slice, with the two gates converging on well-calibrated inputs.

**vs. Text-only-CoT margin gate (non-interleaved)** — On a reasoning-heavy case where the benefit comes specifically from visual restoration (e.g., a spatial question requiring a canonicalized view), the text-only-CoT margin is near zero because a text trace cannot supply the missing visual evidence, so a gate trained on it routes to direct answering and loses the gain. Our method instead routes by the interleaved-trace margin, which is positive on these inputs, so we expect the interleaved-margin gate to outperform the text-CoT-margin gate specifically on the visual-restoration slice, with the two gates converging on inputs where text traces suffice.

### What would falsify this idea

If TIGER's gain over always-trace and always-direct is uniform across the perception and reasoning subsets rather than crossed — or if the counterfactual-margin gate fails to beat the confidence gate on the confidently-wrong slice, or fails to beat the text-only-CoT margin gate on the visual-restoration slice — then the routing mechanism is not doing the work the causal story claims, and the central claim is wrong. Additionally, if the Phase 0b seed-flip rate exceeds 0.20 in the domains where TIGER claims gains, the margin is not a stable causal signal and the reported gains cannot be attributed to task-conditioned trace benefit.

## References

1. ThinkMorph-SSL: Scaling Interleaved Multimodal Reasoning with Pretext Tasks
