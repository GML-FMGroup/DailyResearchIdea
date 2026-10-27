# Calibrated Human Interpreter Models for Verifiable Instruction Interpretability in Embodied Assistants

## Motivation

InteractiveWorld evaluates an MLLM assistant against a scripted human agent, so task success certifies only that the assistant's instructions are unambiguous to a fixed deterministic policy — not that a real human would map them to the intended action. DialFRED has the complementary flaw: its oracle answers are noise-free, so question-asking is validated against an idealized respondent rather than the ambiguous, variable interpretations real users produce. Across both branches the instruction–action mapping is never checked against real human interpretation variability, so 'interpretability' remains an untested assumption rather than a measured property.

## Key Insight

If the human's instruction-to-action mapping is estimated as a calibrated conditional distribution P(a | instruction, observation) from real interaction logs, then the mutual information between the assistant's instruction and the executed action under that distribution is a directly estimable quantity that certifies interpretability in information-theoretic terms, because a calibrated channel uniquely determines how much of the assistant's intent survives transmission to action.

## Method

**What it is.** CALIPER (CAlibrated Language-to-Interpreter Policy for Evaluating Requests) replaces InteractiveWorld's scripted human agent with a probabilistic interpreter. Inputs: the MLLM assistant's instruction $u_t$, the current egocentric observation $o_t$, and dialogue history $h_t$. Output: a distribution $P_\theta(a \mid u_t, o_t, h_t)$ over the discrete action vocabulary, which the environment samples to advance the scene. CALIPER is trained on real human logs and calibrated so its predicted action entropy tracks empirical disagreement among annotators.

**Load-bearing assumption (stated explicitly).** CALIPER's calibrated conditional entropy $H(A\mid U)$ is treated as an estimate of the true human conditional entropy, so that the plug-in mutual information $I(U;A)=H(A)-H(A\mid U)$ estimates the information actually transmitted from instruction to action. This assumption is load-bearing and is not assumed to hold exactly: with $K=5$ annotator actions per instruction, plug-in entropy is biased downward (Miller–Madow / Grassberger 2003), and temperature scaling fitted to per-instruction histograms estimated from the same $K=5$ samples absorbs sampling noise rather than removing it. We therefore (i) replace the plug-in MI with a bias-corrected, $K$-aware estimator, (ii) fit a hierarchical Dirichlet-multinomial interpreter whose posterior over $P(a\mid u)$ carries credible intervals, and (iii) report MI only where posterior concentration is sufficient, together with MI as a function of $K$ to show stability as $K$ grows. The contribution is a calibrated channel with quantified estimation uncertainty, not a point MI asserted to equal true transmitted information.

**How it works.**

```pseudocode
# Phase 1: Data assembly
D = DialFRED_humanQA  ∪  new_ambiguous_instruction_logs
# new logs: each instruction answered by K=5 humans, each producing an action
# plus a natural-language rationale; rationale is discarded, action kept.
# Fixed collection budget: 300 instructions × 5 annotators = 1500 (u,o,h,a) tuples.

# Phase 2: Train interpreter
backbone = frozen MLLM (vision-language, 7B)
head      = linear projection to |A| action logits
for each (u, o, h, a) in D:
    z = backbone(u, o, h)            # pooled instruction+scene embedding
    logits = head(z)                 # |A|-dim
    loss = CE(logits, a) + λ * KL( softmax(logits) || uniform )   # λ=0.05
# λ term prevents collapse to a single deterministic action per instruction

# Phase 3: Calibration (temperature + entropy matching)
# Fit T on held-out split (20% of D, stratified by instruction-ambiguity stratum)
# by minimizing ECE between softmax(logits/T) and empirical action-frequency
# histograms per instruction (T≈1.4 typical).
# Replace per-instruction ECE with a binned reliability diagram over
# instruction-ambiguity strata (3 strata: low/medium/high annotator disagreement,
# defined by empirical entropy terciles) so calibration is estimable without
# repeated identical instructions.

# Phase 3b: Hierarchical Dirichlet-multinomial posterior over P(a|u)
# For each instruction u with K=5 observed actions, place a Dirichlet prior
# Dir(α) with α = 1 (Laplace) over the |A| simplex; posterior is
# Dir(α + counts). Report posterior mean and 95% credible interval for
# P(a|u) and for H(A|U).

# Phase 4: Verifiable evaluation loop (inside InteractiveWorld)
for turn t:
    u_t = MLLM_assistant(o_t, h_t)          # assistant issues instruction
    P_t = CALIPER(u_t, o_t, h_t)            # calibrated distribution
    a_t ~ P_t                                # sample action
    o_{t+1} = env.step(a_t)
    record (u_t, a_t, P_t)

# Phase 5: Interpretability metric (bias-corrected, K-aware)
# Plug-in MI: MI_plugin = H(A) - H(A|U), where H(A) uses marginal action freq
# and H(A|U) uses CALIPER's calibrated entropy.
# Bias-corrected MI: MI_BC = H_CS(A) - H_CS(A|U), where H_CS is the
# Chao–Shen (2003) coverage-adjusted entropy estimator applied to the
# empirical action counts (marginal and per-instruction).
# Report MI_BC in nats with 95% bootstrap CI (1000 resamples over instructions).
# Report MI only for instructions where the Dirichlet posterior concentration
# (sum of posterior α) ≥ 5, i.e., sufficient evidence.
# Report MI as a function of K by subsampling K=1,2,3,4,5 annotators per
# instruction and recomputing MI_BC; stability is shown when the K=5 estimate
# lies within the K=4 CI.
# Also report task success, and MI-per-success to separate interpretability
# from mere task completion.
```

**Why this design.** We chose a calibrated probabilistic head over a deterministic scripted policy because a point estimate cannot express interpretation variability, accepting the cost that CALIPER's entropy is only as faithful as its training logs. We chose to train on DialFRED's human-annotated QA plus newly collected ambiguous-instruction logs rather than on synthetic paraphrases because the calibration target is real annotator disagreement, accepting the cost of a smaller and noisier dataset. We chose mutual information between instruction and action as the headline metric rather than task success alone because success conflates interpretability with assistant competence, accepting the cost that MI requires a well-estimated conditional entropy. We chose to discard annotator rationales and keep only actions because rationales introduce a spurious textual channel, accepting the cost that we lose explanatory signal. This design is not a variant of DialFRED's questioner-performer: DialFRED uses an oracle respondent and optimizes question selection for task success, whereas CALIPER makes the respondent itself a calibrated stochastic channel and measures the information transmitted through it — a quantity DialFRED's oracle makes identically maximal and therefore uninformative.

**Why it measures what we claim.** The mutual information $I(U;A) = H(A) - H(A\mid U)$ measures instruction interpretability because the assumption that CALIPER's calibrated conditional entropy $H(A\mid U)$ equals the true human conditional entropy makes MI equal the true information transmitted from instruction to action; this assumption fails when the training logs under-represent a class of instructions (e.g., spatial references absent from DialFRED), in which case $H(A\mid U)$ reflects CALIPER's prior rather than human uncertainty and MI is biased upward. The temperature $T$ measures calibration quality because the assumption that per-instruction action frequencies are stable across the held-out split makes ECE a valid calibration proxy; this fails when instructions are unique and frequencies cannot be estimated, in which case ECE reflects sampling noise. The sampled action $a_t$ measures the executed human behavior because the assumption that CALIPER's distribution covers the action vocabulary used at test time makes sampling equivalent to drawing a human; this fails for out-of-vocabulary actions, in which case the loop silently substitutes a nearest in-vocabulary action and success is inflated. The bias-corrected MI $MI_{BC}$ measures transmitted information because the assumption that the Chao–Shen estimator removes the finite-$K$ downward bias in $H(A\mid U)$ makes $MI_{BC}$ closer to the true MI than the plug-in; this fails when the per-instruction sample size $K=5$ is too small for the coverage adjustment to be effective (e.g., when the action alphabet is large relative to $K$), in which case $MI_{BC}$ still carries residual bias and must be reported with its bootstrap CI and only for instructions with posterior concentration ≥ 5.

## Contribution

(1) CALIPER, a calibrated probabilistic human-interpreter model for InteractiveWorld trained on DialFRED human QA plus a newly collected ambiguous-instruction action-disagreement dataset, replacing the scripted agent with a stochastic channel whose entropy is matched to real annotator disagreement. (2) An information-theoretic interpretability metric — mutual information between assistant instruction and executed action under the calibrated channel — that separates instruction interpretability from raw task success, exposing cases where high success coexists with low transmitted information. (3) A public ambiguous-instruction interaction log with per-instruction multi-annotator action distributions, enabling calibration and reproducibility of human-interpreter models.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | InteractiveBench (120 tasks) via InteractiveWorld | Matches target deployment and verifiable loop |
| Primary metric | Bias-corrected mutual information $MI_{BC}$ in nats (Chao–Shen) with 95% bootstrap CI | Directly operationalizes interpretability claim with quantified uncertainty |
| Secondary metric | MI-per-success; calibration reliability diagram over ambiguity strata | Separates channel from agent; checks calibration estimability |
| Baseline | Scripted deterministic interpreter | Tests whether stochasticity alone drives signal |
| Baseline | DialFRED oracle respondent | Tests whether calibration beats oracle ceiling |
| Baseline | Uncalibrated MLLM action head | Tests whether calibration, not capacity, matters |
| Ablation-of-ours | CALIPER without KL entropy regularizer | Tests whether anti-collapse term is load-bearing |
| Ablation-of-ours | Plug-in MI vs. bias-corrected MI | Tests whether estimator bias drives headline result |
| Validation set | Held-out 300 instructions × 5 annotators (human study) | Empirically anchors MI against real annotator entropy |

### Why this setup validates the claim

The central claim is that a calibrated stochastic interpreter transmits measurable, non-trivial information from instruction to action, and that this information is a distinct quantity from task success. InteractiveBench is the right dataset because it is the environment the assistant actually operates in, so the interpreter's action distribution is consumed by the same loop that scores success, making MI and success jointly observable on identical turns. The scripted-interpreter baseline tests the sub-claim that point estimates cannot express interpretation variability: if MI is identical between scripted and calibrated respondents, stochasticity is doing nothing. The DialFRED oracle baseline tests the sub-claim that an oracle respondent makes MI trivially maximal and therefore uninformative, so a calibrated respondent must show lower MI than the oracle while retaining success — the gap is the interpretability signal the oracle hides. The uncalibrated head isolates calibration from raw MLLM capacity: if temperature/entropy matching does not change the MI-vs-success relationship, the calibration phase is decorative. The no-KL ablation tests whether the anti-collapse regularizer is what keeps the conditional entropy from collapsing to zero, which would artificially inflate MI. The plug-in-vs-bias-corrected ablation tests whether the headline MI is an estimator artifact: if $MI_{BC}$ and plug-in MI diverge substantially, the original plug-in claim is quantitatively bounded and must be reported as such. The held-out human validation set (300 instructions × 5 annotators) empirically anchors the metric: we compute empirical conditional entropy $H_{emp}(A\mid U)$ from real annotator actions and regress CALIPER's predicted $H(A\mid U)$ against it, reporting correlation and calibration slope; a slope deviating from 1 bounds the MI claim. MI is the right primary metric because success conflates the assistant's competence with the interpreter's faithfulness; MI separates the channel from the agent, and MI-per-success normalizes for the trivial path where easy tasks inflate both.

### Expected outcome and causal chain

**vs. Scripted deterministic interpreter** — On a case where the assistant says "move it a bit closer to the window," the scripted interpreter emits one fixed displacement because it has no mechanism to represent that "a bit" admits a range of human readings, so the executed action is either right or wrong with no variance. Our method samples from a calibrated distribution whose entropy tracks annotator disagreement, so the same instruction yields a spread of displacements matching human readings. We expect a clear $MI_{BC}$ gap favoring CALIPER concentrated on ambiguous-instruction subsets, with parity on unambiguous commands where both respondents are near-deterministic; the gap should exceed the bootstrap CI width.

**vs. DialFRED oracle respondent** — On a case where the assistant asks a clarifying question, the oracle respondent returns the single ground-truth answer because it is constructed to be maximally informative, so H(A|U) is near zero by design and MI is saturated regardless of instruction quality. Our method instead returns a distribution reflecting real annotator disagreement, so H(A|U) is positive and MI varies with how well the instruction constrains interpretation. We expect the oracle to show uniformly maximal MI and near-identical success, while CALIPER shows lower but instruction-sensitive MI that tracks success variation — the oracle's MI is uninformative, ours is diagnostic.

**vs. Uncalibrated MLLM action head** — On a case where the instruction is underspecified ("put it over there"), the uncalibrated head produces a sharp, overconfident distribution because cross-entropy training collapses toward the majority annotator action, so H(A|U) is underestimated and MI is biased upward. Our method applies temperature and entropy matching so the predicted entropy tracks empirical disagreement, yielding a flatter distribution on exactly these cases. We expect the uncalibrated baseline to show inflated MI on ambiguous subsets and a larger MI-success divergence, while CALIPER's MI stays closer to the empirical annotator MI and its MI-per-success is higher.

**vs. CALIPER without KL entropy regularizer** — On repeated identical instructions, the no-KL ablation collapses to a single action because cross-entropy alone drives the head toward the modal annotator choice, so H(A|U) approaches zero and MI is spuriously maximized. Our method's KL-to-uniform term preserves residual entropy, keeping H(A|U) near the empirical annotator entropy. We expect the ablation to report higher MI but lower success and a worse calibration error, exposing that its MI gain is a collapse artifact rather than genuine transmitted information.

**vs. Plug-in MI (ablation of estimator)** — On the same recorded turns, plug-in MI uses the raw calibrated entropy without bias correction, so with K=5 annotator actions per instruction it inherits the downward finite-sample bias in H(A|U) and overstates transmitted information. Our bias-corrected $MI_{BC}$ applies the Chao–Shen coverage adjustment and reports 95% bootstrap CIs, and we restrict reporting to instructions with Dirichlet posterior concentration ≥ 5. We expect plug-in MI to exceed $MI_{BC}$ by an amount that grows with action-alphabet size and shrinks as K increases from 1 to 5; if the K=5 estimate lies within the K=4 CI, the bias-corrected estimate is stable and the headline claim is bounded rather than assumed.

### What would falsify this idea

If CALIPER's $MI_{BC}$ advantage over the scripted and uncalibrated baselines is uniform across ambiguous and unambiguous instruction subsets rather than concentrated where annotator disagreement is high, the calibration mechanism is not what produces the signal and the central claim fails. Likewise, if the oracle baseline's MI is not saturated and instruction-insensitive, the argument that oracle respondents make MI uninformative is wrong; and if the regression of CALIPER's predicted H(A|U) on empirical $H_{emp}(A\mid U)$ yields a slope far from 1 with low correlation, the MI claim is quantitatively bounded and must be reported as such rather than as an equality.

## References

1. InteractiveWorld: A Verifiable Human-Centric Environment for Building Better Interactive MLLM Assistants
2. DialFRED: Dialogue-Enabled Agents for Embodied Instruction Following
