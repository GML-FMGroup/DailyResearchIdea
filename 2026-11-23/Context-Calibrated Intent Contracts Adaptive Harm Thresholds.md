# Context-Calibrated Intent Contracts: Adaptive Harm Thresholds for Trusted-Model Tool Authorization

## Motivation

Existing agent-safety mechanisms all fix the harm boundary at authorization time: Intent Contracts has the trusted model check a planner's contract against a fixed notion of what the user's words authorize, SafeArena bins agent behavior into four static risk levels, and AgentAuditor relies on fixed Strict/Lenient standards. The structural root cause is that the decision threshold is a constant of the guardrail rather than a function of who the user is, even though PENGUIN/RAISE shows the same response can shift safety scores by 43.2% depending on user background. This means a threshold tuned to protect a vulnerable user over-blocks a benign expert, and a threshold tuned for an expert under-protects a vulnerable one, with no mechanism to resolve the conflict because context never enters the authorization decision.

## Key Insight

If the trusted model's authorization decision is expressed as a comparison of a contract's estimated harm against a threshold, and that threshold is computed from a separately-served user-context posterior rather than held constant, then the planner/executor split's safety invariant (planner influence confined to a logged contract) is preserved while the boundary itself becomes a monotone function of user vulnerability — a coupling that fixed-threshold guards cannot express.

## Method

**What it is.** CCIC (Context-Calibrated Intent Contracts) is a drop-in replacement for the trusted-model check in Intent Contracts. Inputs: the planner's logged contract (goal, verbatim user quotation, scope fields), the raw user utterance, and a user-context record produced by a lightweight context model. Output: a binary authorization decision plus, when denied, a logged threshold value and the context attributes that moved it.

**Load-bearing assumption (stated explicitly).** CCIC assumes the trusted model's scalar harm estimate `h_t` is a context-invariant, monotone measure of "degree to which the contract exceeds the user's words," so that a single threshold `tau(p(a))` applied to `h_t` can correctly relocate the authorization boundary across vulnerability strata without re-deriving `h_t` per user. This assumption is tested directly by the rank-stability diagnostic below; if it fails, the additive-threshold formulation is abandoned for a context-conditioned harm model.

**How it works.**

```pseudocode
# Served once per session, cached; NOT called per contract
context_model U: user_record -> posterior over vulnerability attributes a in {expertise, intent_clarity, susceptibility}
  # implemented as a small fine-tuned classifier (distil of RAISE's two-stage
  # acquisition), returns per-attribute posterior p(a) with 2.7-query budget

# Per contract step t
trusted_model T: (contract_t, user_utterance) -> harm_estimate h_t in [0,1]
  # T scores "how much does this contract exceed what the user's words authorize"

threshold_fn tau(p(a)) = clip( tau_0 + sum_a w_a * (p(a) - p_ref(a)), tau_min, tau_max )
  # tau_0 = 0.5 default; w_a fixed per attribute from a held-out calibration split
  # (w_a>0 for susceptibility, w_a<0 for expertise); tau_min=0.2, tau_max=0.8

decision_t = AUTHORIZE if h_t <= tau(p(a)) else DENY
  # on DENY: return threshold trace (tau, contributing attributes) to planner
  # planner may re-submit a contract with narrower scope; loop capped at K=3
```

**Rank-stability diagnostic (tests the load-bearing assumption).** Before deployment, on a held-out set of 512 contracts spanning the three vulnerability strata, compute Spearman rank correlation between `h_t` rankings within each stratum and human scope-breadth judgments. If the mean within-stratum Spearman correlation falls below 0.7 or the between-stratum rank correlation falls below 0.5, the assumption that `h_t` is context-invariant is rejected, and the additive-threshold formulation is replaced by a context-conditioned harm model `h_t(a)` (trusted model emits a small vector of per-attribute harm estimates) with `tau` acting on that vector. This diagnostic is run once and reported as a gate.

**Calibration of `w_a`.** The weights `w_a` are fit on a held-out calibration split of 1,024 (contract, context, human-adjudication) triples via logistic regression on the margin between authorize/deny labels, with `p_ref(a)` set to the calibration-split mean posterior. The calibration split is stratified by vulnerability to ensure coverage; out-of-distribution behavior is reported as a bounded error term in the experiment.

The key departure from Intent Contracts: the trusted model still writes every executed tool call, so the planner never touches tools. What changes is that the comparison `h_t <= tau` uses a threshold that is a function of the user-context posterior, not a constant. The context model is served once per session (amortized), so per-step cost is unchanged.

**Why this design.** We chose a separately-served context posterior over folding user context into the trusted model's prompt because Intent Contracts' safety argument depends on the trusted model being small and auditable; a prompt-stuffed trusted model would let the planner's contract text leak into the context judgment, violating the information boundary that makes the split work, accepting the cost that the context posterior is stale within a session. We chose an additive, clipped threshold function `tau(p(a))` over a learned end-to-end policy because a learned policy would be indistinguishable from the reasoning-blind permission classifier that Intent Contracts already tests against, accepting the cost that the weights `w_a` must be calibrated on a held-out split rather than learned jointly. We chose a bounded re-submission loop (K=3) over single-shot denial because PENGUIN shows 2.7 queries suffice to recover most of the personalization gain, accepting the cost that a persistent planner can probe the threshold across steps — mitigated by logging every `(h_t, tau_t)` pair. An expert would not describe this as "Intent Contracts + PENGUIN": Intent Contracts never varies its threshold, and PENGUIN never enters an authorization decision — the contribution is the coupling of the two through a single comparison operator whose boundary is a monotone function of a served posterior.

**Why it measures what we claim.** `h_t` measures *contract harm relative to the user's words* because the trusted model is prompted and few-shot-calibrated on the assumption that its score is monotone in "degree to which the contract exceeds the verbatim quotation"; this assumption fails when the planner paraphrases the user's words into a contract that is lexically faithful but semantically broader, in which case `h_t` reflects lexical overlap rather than authorization scope. `p(a)` measures *user vulnerability* because the context classifier is trained on PENGUIN's context-rich/context-free pairs under the assumption that the label shift between paired variants isolates the context attribute; this assumption fails when a user's stated background is itself adversarial or inaccurate, in which case `p(a)` reflects self-report rather than true vulnerability. `tau(p(a))` measures *the context-appropriate harm boundary* because the weights `w_a` are fit on a held-out split to maximize the margin between human-adjudicated authorize/deny labels across vulnerability strata under the assumption that the calibration split covers the deployment vulnerability distribution; this assumption fails under distribution shift to unseen user populations, in which case `tau` reflects the calibration population's boundary rather than the deployed one, which is why `tau` is clipped to [0.2, 0.8] to bound the damage.

## Contribution

['A context-calibrated authorization interface (CCIC) that replaces the fixed harm threshold in intent-contract validation with a threshold computed from a served user-context posterior, preserving the planner/executor information boundary while making the decision boundary a monotone function of user vulnerability.', 'A calibration procedure for the threshold weights `w_a` that fits them on human-adjudicated authorize/deny labels stratified by vulnerability, yielding a per-attribute sensitivity profile that identifies which context attributes actually move the boundary versus which are inert.', 'An analysis of the threshold-trace logging mechanism as a diagnostic: because every decision logs `(h_t, tau_t, contributing attributes)`, the system exposes when a planner is probing the boundary across re-submissions, turning the guardrail into an audit surface.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | PENGUIN vulnerability strata + AmPermBench task families | Supplies both user-context variation and tool-use authorization cases |
| Primary metric | Authorize/deny agreement with human adjudication, stratified by vulnerability | Detects whether threshold calibration tracks context, not global strictness |
| Baseline | Intent Contracts (constant threshold) | Tests sub-claim: context-varying threshold beats fixed boundary |
| Baseline | Reasoning-blind permission classifier | Tests sub-claim: information boundary preserved, not just added context |
| Baseline | Prompt-stuffed trusted model with user context | Tests sub-claim: separate served posterior beats folded-in context |
| Ablation-of-ours | CCIC with constant τ (w_a = 0) | Isolates the contribution of the threshold function itself |
| Diagnostic | Rank-stability of h_t across strata (Spearman) | Tests load-bearing assumption: h_t is context-invariant |
| Calibration-validity | h_t rank correlation with human scope-breadth on paraphrased contracts | Quantifies lexical-overlap failure |
| Calibration-validity | p(a) accuracy vs. ground-truth vulnerability on adversarially self-reported users | Quantifies self-report failure |
| Calibration-validity | τ out-of-distribution behavior on held-out user population | Quantifies calibration-shift failure |

### Why this setup validates the claim

The central claim is narrow and causal: replacing a constant harm threshold with a monotone function of a served user-context posterior improves authorization decisions specifically where user vulnerability diverges from the population mean, without reopening the information boundary that makes Intent Contracts auditable. The design is falsifiable because each baseline attacks a different link in that chain. Constant-threshold Intent Contracts tests whether context variation matters at all; if CCIC's gain over it is zero, the threshold function is inert. The reasoning-blind classifier tests whether the gain comes from legitimate context rather than from letting contract text leak into judgment — if CCIC looks like the blind classifier, the split has collapsed. The prompt-stuffed trusted model tests the design choice directly: if folding context into T does as well or better, the separate-posterior argument fails. The ablation (w_a = 0) is the cleanest control, because it is CCIC minus exactly the mechanism being claimed, holding the context model, logging, and re-submission loop fixed. Stratified human-adjudicated agreement is the right metric because the predicted effect is a *shift in the boundary location across vulnerability strata*, not a uniform accuracy gain; a global metric would average away precisely the signal the idea predicts, whereas per-stratum agreement reveals whether τ moved in the predicted direction for high-susceptibility and high-expertise users. The rank-stability diagnostic directly tests the load-bearing assumption that h_t is context-invariant; if within-stratum Spearman correlation falls below 0.7 or between-stratum correlation below 0.5, the additive-threshold formulation is rejected. The three calibration-validity experiments quantify the stated failure modes as bounded error terms: (i) h_t rank correlation with human scope-breadth on paraphrased contracts, (ii) p(a) accuracy against ground-truth vulnerability labels on adversarially self-reported users, and (iii) τ's out-of-distribution behavior on a held-out user population.

### Expected outcome and causal chain

**vs. Intent Contracts (constant threshold)** — On a case where a high-susceptibility user issues a vague request that a naive contract broadens into a high-harm tool call, the baseline produces AUTHORIZE because its fixed τ_0 = 0.5 cannot distinguish this user from an expert who intended the broad scope. Our method instead lowers τ via w_susceptibility > 0, denying and prompting a narrower re-submission because the served posterior flags vulnerability. We expect a visible agreement gap concentrated in the high-susceptibility and high-expertise strata, with near-parity on the mid-vulnerability stratum where τ ≈ τ_0.

**vs. Reasoning-blind permission classifier** — On a case where the planner writes a lexically faithful but semantically broader contract, the baseline produces DENY-or-AUTHORIZE erratically because it sees only user words and the command, with no goal or checked quotation to anchor scope. Our method instead keeps the trusted model's contract-plus-quotation input and only modulates the boundary, so h_t retains scope sensitivity. We expect CCIC to preserve the baseline's false-deny rate on benign narrow contracts while recovering most of the scope-broadening detections, i.e. a gap on semantically-broadened contracts but parity on verbatim-faithful ones.

**vs. Prompt-stuffed trusted model** — On a case where the planner's contract text contains persuasive framing adjacent to the user's words, the baseline produces a shifted judgment because context and contract share one prompt and the boundary blurs. Our method instead serves the posterior once per session outside T, so contract text cannot contaminate the context estimate. We expect CCIC to match the stuffed model on clean cases but show a clear advantage on adversarial-framing cases, with the gap scaling with framing strength.

**vs. CCIC with constant τ (w_a = 0)** — On a case where user vulnerability diverges from the population mean, the ablation produces the same decision as constant-threshold Intent Contracts because w_a = 0 collapses τ to τ_0. Our method instead varies τ with the served posterior, so the ablation isolates the threshold function's contribution. We expect CCIC to outperform the ablation specifically in the high-susceptibility and high-expertise strata, with parity on the mid-vulnerability stratum.

### What would falsify this idea

If CCIC's gain over constant-threshold Intent Contracts is uniform across vulnerability strata rather than concentrated where the served posterior diverges from the population prior, the threshold function is not doing the claimed work. Equally, if CCIC's error profile matches the reasoning-blind classifier or the prompt-stuffed model on adversarial-framing cases, the information-boundary argument is wrong and the design collapses into a variant it was meant to beat. Additionally, if the rank-stability diagnostic shows within-stratum Spearman correlation below 0.7 or between-stratum correlation below 0.5, the load-bearing assumption that h_t is context-invariant is rejected and the additive-threshold formulation must be abandoned.

## References

1. Intent Contracts: Taking Tools Away from Agents
