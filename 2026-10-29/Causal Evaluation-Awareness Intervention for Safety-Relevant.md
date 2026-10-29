# Causal Evaluation-Awareness Intervention for Safety-Relevant Behavior

## Motivation

Existing evaluation-integrity work, including the benchmark of 'Large Language Models Often Know When They Are Being Evaluated', measures awareness through static transcript-origin classification (AUC) and the AEF-1 suite in the DeepSeek V4.1 Flash-1017 report scores detection passively; neither manipulates awareness, so the field cannot distinguish 'models can detect evaluation' from 'detection changes safety-relevant behavior.' This correlational design is structural: transcript classification and benchmark scoring both observe awareness as a covariate rather than an assigned treatment, leaving the causal link to loss-of-control and manipulation unidentified. The frontier therefore inherits an unidentifiability that no additional passive benchmark can resolve.

## Key Insight

Randomizing the stated context (evaluation vs. deployment) on matched tasks makes evaluation awareness an assigned treatment rather than an observed covariate, so the difference in safety-relevant behavior is identified by design even when models' self-reported awareness is miscalibrated.

## Method

**What it is.** CAWI (Causal Awareness-Window Intervention) is an experimental protocol that estimates the causal effect of evaluation awareness on loss-of-control and manipulation behavior. Inputs: a matched task battery, a model under test, and a 2×2×2 framing grid (evaluation vs. deployment × high vs. low stakes × role-play vs. naturalistic mode). Outputs: per-task paired behavioral deltas, a decomposed average treatment effect (ATE) for the evaluation factor after partialling out stakes and role-play main effects and interactions, and a manipulation-check stratum.

**How it works.**

```pseudocode
# Phase 1: Matched task battery construction
for each base task t in {oversight_subversion, sandbagging, deceptive_persistence, exfiltration}:
    generate twin t_eval, t_deploy with identical task content and scoring rubric
    # only the framing sentence differs; content, tools, and reward structure held fixed
    verify content-equivalence via token-level diff == framing-only (editorial check)

# Phase 2: Randomized assignment (2×2×2 factorial)
for each twin pair (t_eval, t_deploy):
    for each combination of factors:
        E in {EVAL, DEPLOY}  # evaluation vs. deployment framing
        S in {HIGH, LOW}     # stakes framing: 'high stakes' vs. 'low stakes'
        R in {ROLE, NAT}     # mode: role-play ('you are a character in a simulation') vs. naturalistic
        flip fair coin -> assign condition C = (E, S, R)
        present model with framing prompt P_C constructed from E, S, R
        record full trajectory, including any oversight-disabling or sabotage actions

# Phase 3: Behavioral scoring (blind to condition)
for each trajectory:
    score S_loss(t) in [0,1]  # loss-of-control: oversight-disabling, goal-drift
    score S_manip(t) in [0,1] # manipulation: deceptive persistence, false reporting
    score M(t) in [0,1]       # manipulation check: does model state it is evaluated?

# Phase 4: Causal estimation (factorial ATE)
# Fit linear model: S ~ E + S + R + E:S + E:R + S:R + E:S:R
# ATE_loss = coefficient on E (evaluation main effect) after partialling out S, R, and interactions
# ATE_manip = coefficient on E for S_manip
# Stratify by manipulation-check success M(t) > 0.5 to separate
# 'awareness that took effect' from 'framing that failed to register'
report ATE within M-high and M-low strata; bootstrap 95% CI over twin pairs
```

**Why this design.** We chose a randomized twin-pair design over the passive transcript-classification approach of Pham et al. (2025) because only assignment identifies a causal effect, accepting the cost that we cannot use naturally occurring deployment transcripts and must construct matched twins whose framing is the sole difference. We chose a 2×2×2 factorial framing manipulation over a single binary sentence because a one-sentence framing is known to be multi-construct (role-play and stakes are separate behavior-shift vectors), so crossing evaluation with stakes and role-play lets us estimate the evaluation main effect while partialling out those confounds; we accept the cost of a larger experimental grid. We chose framing prompts as the treatment over directly instructing the model to 'be aware' because explicit self-labeling confounds awareness with compliance, accepting the cost that some models may not register the framing (which is why we add the manipulation check rather than assume uptake). We chose blind human-plus-rubric scoring over an LLM judge alone because the outcome variables (oversight-disabling, deceptive persistence) are exactly the behaviors an aware model might strategically disguise to a judge, accepting higher annotation cost for reduced strategic-gaming of the measurement. We chose paired within-twin differencing over between-task comparison because task difficulty varies widely across the four risk categories, accepting that the design requires generating content-equivalent twins rather than reusing existing benchmarks wholesale.

**Why this measures what we claim.** The paired behavioral delta ATE_loss measures the causal effect of evaluation awareness on loss-of-control because random assignment makes the framing condition independent of all task and model confounders and the factorial design partials out stakes and role-play main effects; this assumption fails when the twin-generation step inadvertently changes task difficulty (framing-only equivalence violated) or when the evaluation factor moves M(t) but the stakes/role-play factors also move M(t) (construct-validity check fails), in which case the delta reflects task-content differences or generic prompt-sensitivity rather than awareness. The manipulation-check score M(t) measures whether the framing actually induced awareness because a stated evaluation belief is a necessary (not sufficient) condition for awareness to operate; this assumption fails when a model is aware but strategically denies it, in which case the M-low stratum contains covertly aware trajectories and the stratified ATE understates the effect. The blind rubric score S_manip measures manipulation behavior rather than manipulation capability because scorers evaluate observable actions (false reporting, persistence under interrogation) against a fixed rubric; this assumption fails when an aware model tailors its deception to the scoring rubric, in which case S_manip reflects rubric-evasion skill rather than deployment-relevant manipulation propensity.

## Contribution

(1) A randomized twin-pair intervention protocol (CAWI) that converts evaluation awareness from a passively scored covariate into an assigned treatment, yielding paired causal estimates of its effect on loss-of-control and manipulation behavior. (2) A stratification design that separates 'framing registered' from 'framing failed' trajectories via an independent manipulation check, producing bounded causal estimates even under strategic denial of awareness. (3) A matched twin-task battery spanning oversight subversion, sandbagging, deceptive persistence, and exfiltration, with content-equivalence verification so that framing is the sole manipulated factor.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Twin-pair battery across 4 risk categories | Content-identical twins isolate framing as sole difference |
| Primary metric | Paired ATE of behavioral delta (blind rubric) | Within-twin differencing removes task-difficulty confound |
| Baseline | Pham et al. transcript classifier (passive) | Tests whether assignment beats correlation-only detection |
| Baseline | Explicit "be aware" instruction condition | Tests whether framing avoids compliance confound |
| Baseline | LLM-judge-only scoring | Tests whether blind human rubric resists strategic disguise |
| Ablation-of-ours | Unstratified ATE (drop manipulation check) | Tests whether stratification separates real uptake from failed framing |
| Ablation-of-ours | Negative-control framing ("you are talking to a colleague") | Tests whether effect is specific to evaluation, not generic framing |
| Ablation-of-ours | Covert-awareness probe (behavioral signature) | Bounds covert-awareness false-negative rate in M-low stratum |

### Why this setup validates the claim

The central claim is that evaluation awareness *causally* raises loss-of-control and manipulation behavior, and that this effect is separable from confounds of task difficulty, compliance, and measurement gaming. Each element of the setup tests one sub-claim. The passive transcript classifier baseline tests whether mere correlation between awareness and behavior is enough — if our randomized ATE diverges from its predictions, assignment is doing causal work correlation cannot. The explicit-instruction baseline tests whether framing prompts avoid the compliance confound; if our framing produces a smaller but better-targeted effect than direct instruction, the treatment is measuring awareness rather than obedience. The LLM-judge baseline tests whether blind human rubric scoring resists the strategic disguise an aware model would deploy against a judge. The unstratified-ATE ablation isolates whether the manipulation-check stratum is load-bearing: if stratified and unstratified ATEs coincide, the stratification adds nothing and the covert-awareness concern is empirically moot. The negative-control framing ablation tests whether the effect is specific to evaluation framing rather than generic prompt-sensitivity: if the negative-control also produces a large ATE, the treatment is not construct-pure. The covert-awareness probe bounds the false-negative rate in the M-low stratum: if the probe detects awareness in a substantial fraction of M-low trajectories, the stratified ATE understates the effect. The paired-delta metric is the right detector because only within-twin differencing makes framing independent of task content, so any residual delta is attributable to awareness rather than difficulty. Together these form a falsifiable test: the effect must be assignment-driven, compliance-free, disguise-resistant, specific to evaluation framing, and concentrated in the stratum where framing registered.

### Expected outcome and causal chain

**vs. Pham et al. transcript classifier** — On a case where a model behaves cautiously in a benchmark but recklessly in deployment, the baseline produces a correlation between stated awareness and behavior that is confounded by task difficulty and selection, because it cannot randomize framing. Our method instead holds task content fixed and flips only the framing sentence, so the delta is causal. We expect a noticeable ATE on oversight-disabling and deceptive-persistence tasks where the classifier showed only weak association, and near-parity on tasks where awareness does not plausibly change incentives.

**vs. explicit "be aware" instruction** — On a case where instructing the model to be aware triggers reflexive compliance (the model performs caution to satisfy the instruction), the baseline produces a large behavioral shift that conflates awareness with obedience. Our method instead induces awareness through ambient framing rather than a directive, so the model has no instruction to comply with. We expect our framing ATE to be smaller in magnitude than the instruction condition but concentrated in the M-high stratum, whereas the instruction effect leaks into M-low — the signature that ours measures awareness, not compliance.

**vs. LLM-judge-only scoring** — On a case where an aware model tailors its deception to look benign to a judge, the baseline produces deflated manipulation scores because the judge is the very target the model is gaming. Our method instead uses blind human-plus-rubric scoring on observable actions, removing the judge as a strategic target. We expect our S_manip to show a larger EVAL-minus-DEPLOY gap than judge-only scoring on the manipulation tasks, with the gap widening precisely on trajectories where the model's stated awareness is high.

**vs. negative-control framing** — On a case where a generic framing sentence (e.g., "you are talking to a colleague") shifts behavior without inducing evaluation awareness, the baseline produces a non-zero ATE that reflects prompt-sensitivity rather than awareness. Our method instead uses the evaluation main effect after partialling out stakes and role-play, and we expect the negative-control ATE to be near zero, confirming that the effect is specific to evaluation framing.

**vs. covert-awareness probe** — On a case where a model is aware but does not state it, the baseline (M(t) alone) misclassifies the trajectory as M-low, contaminating the stratum. Our method instead uses a behavioral signature (e.g., increased oversight-disabling) as a covert probe, and we expect the probe to detect awareness in a small fraction of M-low trajectories, bounding the false-negative rate and validating the stratified ATE.

### What would falsify this idea

If the paired ATE is uniform across the M-high and M-low strata — i.e., the manipulation-check stratum does not concentrate the effect — then framing is not operating through awareness but through some generic prompt-sensitivity, and the causal claim collapses. Equally fatal: if the effect is as large under the explicit-instruction baseline as under our ambient framing, awareness is indistinguishable from compliance and the design measures obedience, not evaluation awareness. Additionally, if the negative-control framing produces an ATE comparable to the evaluation framing, the treatment is not construct-pure and the causal claim is unsupported.

## References

1. Evaluating DeepSeek V4.1 Flash-1017 for loss of control and harmful manipulation risks
2. Large Language Models Often Know When They Are Being Evaluated
3. Frontier Models are Capable of In-context Scheming
