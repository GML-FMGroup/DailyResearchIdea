# Bayes-Ego: A Belief-State Consistency Protocol for Online Egocentric Agent Evaluation

## Motivation

Existing egocentric-agent benchmarks evaluate context handling as discrete snapshots rather than as a continuous belief-refinement process: EgoUseBench (arxiv:2611.34466) defines Computer-Use, Offline Ego Use, and Online Ego Use as separate context levels but scores each observation independently, so an agent that contradicts its own earlier physical-context conclusions is never penalized; HASARD (venue:iclr:2025:57b5dbc758c2) similarly assumes egocentric observations suffice for a fixed policy without explicit belief maintenance. The root cause is structural: neither protocol carries a latent physical-context state across the observation stream, so there is no quantity against which 'the agent revised its understanding consistently' can be measured. This leaves the field unable to distinguish an agent that genuinely integrates evidence over time from one that happens to answer each snapshot correctly.

## Key Insight

Monotone consistency under belief refinement is a property of the agent's posterior trajectory, not of any single answer, and it becomes measurable exactly when the benchmark supplies a recursive Bayesian reference filter whose posterior is the unique sufficient statistic against which the agent's implied belief can be compared at every step.

## Method

**What it is.** Bayes-Ego is an evaluation protocol that augments the Online Ego Use setting of EgoUseBench with a recursive Bayesian reference belief over a fixed attribute vector (object presence, activity phase, actor identity, task relevance), and scores each agent by the consistency of its implied posterior trajectory against that reference. Inputs: the first-person observation sequence $o_{1:T}$ and the agent's per-step task outputs $a_{1:T}$ (which include any stated context claims or grounding decisions). Outputs: per-step reference posteriors $b_t^{\text{ref}}$, per-step agent-implied posteriors $\hat b_t$, a per-episode readout-validity rate $V$, and three consistency metrics (MCS, TCS, ACD) defined below.

**Load-bearing assumption (stated explicitly).** The protocol assumes that the agent's internal physical-context belief can be faithfully recovered from its output surface — either its explicit context claims (one-hot) or a linear probe on its hidden states — so that the extracted posterior trajectory $\hat b_t$ is a valid readout of the belief the agent actually used to act. This is the probe-validity / faithfulness assumption: linear probes recover decodable information, not necessarily the information the model causally uses (Belinkov, Computational Linguistics 2022; Ravichander et al., EMNLP 2021). The explicit-claim path fails under strategic output, and ECBench (arxiv:2501.05031) documents that LVLM embodied cognition is riddled with hallucination and weak self-cognition — precisely the regime where stated claims diverge from operational belief. The two readout paths are not assumed to agree; instead, we add an interventional faithfulness check (Phase 2b) that makes $\hat b_t$ a validated quantity rather than an assumed one, and we report a per-episode readout-validity rate $V$ alongside MCS/TCS/ACD so that low-validity episodes are excluded rather than silently mis-scored.

**How it works.**

```pseudocode
# Phase 1: Reference belief kernel (built once per EgoUseBench episode)
attributes A = {object_present_j, activity_phase_k, actor_id_l, task_relevant_m}
for each attribute a in A:
    prior p(a) ~ elicited from episode annotators (Dirichlet over 2-4 values;
        3 annotators per episode, inter-annotator agreement reported as Krippendorff's alpha)
    observation likelihood p(o_t | a) ~ calibrated logistic probe trained on
        a fixed public split of EgoUseBench Offline Ego Use clips (frozen before evaluation;
        calibration set of 512 held-out clips; reliability diagram and ECE reported)
for t = 1..T:
    b_t_ref(a) = p(o_t | a) * b_{t-1}_ref(a) / Z          # recursive Bayes, Z = normalizer
    # Z is computed exactly over the finite attribute grid (no sampling)

# Phase 2: Agent-implied posterior extraction
for t = 1..T:
    if agent emits an explicit context claim c_t (object/activity/task label):
        b_hat_t = one-hot smoothed with epsilon=0.01 over the claimed value
    else:
        b_hat_t = probe_readout(agent_hidden_state_t)   # linear probe trained
                                                        # on a pre-registered fixed public split
                                                        # of 200 held-out episodes; probe accuracy reported

# Phase 2b: Interventional faithfulness check (per scored step)
for t = 1..T:
    # Counterfactual observation edit: flip the attribute value in o_t (e.g., remove object)
    # and require the agent's downstream action/output to change in the direction predicted by b_hat_t
    delta_action = agent(o_t^{counterfactual}) - agent(o_t)
    valid_t = 1[ sign(delta_action) == sign(b_hat_t - b_hat_{t-1}) ]
    # Also require probe readout and explicit claim to agree when both are available
    if both available: valid_t = valid_t AND (argmax b_hat_t == argmax c_t)
V = mean_t valid_t   # per-episode readout-validity rate
# Only steps with valid_t = 1 are included in MCS/TCS/ACD; episodes with V < 0.5 are excluded

# Phase 3: Consistency scoring (on valid steps only)
MCS = mean_t  KL( b_t_ref || b_hat_t )                   # calibration vs reference
TCS = mean_t 1[ argmax b_hat_t != argmax b_hat_{t-1} and
                 b_t_ref supports prior argmax ]          # oscillation count
ACD = mean_t  KL( b_hat_t || b_{t-1}_ref ) - KL( b_hat_t || b_t_ref )
                                                          # >0 means agent used
                                                          # stale evidence
```

MCS (Mean Calibration Score) is the average forward KL between the reference posterior and the agent's implied posterior; TCS (Transition Consistency Score) counts the fraction of steps where the agent flips its argmax belief against the reference's evidence direction; ACD (Anchoring Cost Difference) is positive when the agent's belief is closer to yesterday's reference than to today's, i.e. it ignored the new observation. All three are reported jointly with $V$; no single scalar is reported.

**Why this design.** We chose a *recursive Bayesian reference filter with a frozen likelihood probe* over a learned end-to-end belief model because the benchmark must be a measuring instrument, not a competitor: a learned reference would make consistency scores hostage to the reference model's own errors, so we accept the cost that the reference is only as good as the elicited priors and probe calibration. We chose *explicit-claim extraction plus a linear probe on hidden states* over free-form LLM-judge scoring because a judge introduces its own belief trajectory and confounds agent consistency with judge consistency, accepting the cost that agents which do not expose hidden states cannot be scored on the probe path. We chose *three separate metrics (MCS, TCS, ACD)* over a single aggregate because a single scalar hides the distinction between miscalibration and staleness, accepting the cost of a three-way reporting burden. We chose *finite-grid exact normalization* over particle filtering because the attribute space in EgoUseBench is small (≤ 4 attributes × ≤ 4 values), so exact posteriors are tractable and eliminate Monte-Carlo variance from the evaluation. A domain expert would not describe this as a variant of HASARD's safe-RL benchmark or of EgoUseBench's context-level comparison, because neither prior protocol constructs a latent belief trajectory at all: HASARD scores reward-cost trade-offs per episode, and EgoUseBench scores per-setting success; Bayes-Ego's scored object is the *posterior path*, which neither prior work defines.

**Why it measures what we claim.** The reference posterior $b_t^{\text{ref}}$ measures *the normatively correct physical-context belief given the observation history* because the recursive Bayes update is the unique sufficient statistic for the attribute vector under the stated conditional-independence assumption (observations depend on attributes only through the calibrated likelihood); this assumption fails when the likelihood probe is miscalibrated on a distribution shift (e.g., a new wearable camera), in which case $b_t^{\text{ref}}$ reflects the probe's bias rather than the true posterior. The agent-implied posterior $\hat b_t$ measures *the agent's internal physical-context belief* because the explicit-claim one-hot encoding and the linear probe are both readouts of the agent's own output surface; this assumption fails when the agent's claim is strategic rather than sincere, in which case $\hat b_t$ reflects the stated belief rather than its operational belief — the interventional faithfulness check (Phase 2b) and the readout-validity rate $V$ are designed to detect exactly this failure. TCS measures *monotone consistency under belief refinement* because a flip against the reference evidence direction is exactly the event the motivation defines as an oscillation; this assumption fails when the reference itself flips due to a genuinely ambiguous observation, in which case TCS penalizes correct revision. ACD measures *whether the agent integrated new evidence* because the difference of KL divergences is positive iff the agent's belief is closer to the stale reference than to the current one; this assumption fails when the agent's belief is uniformly far from both, in which case ACD is near zero and reflects indifference rather than anchoring.

## Contribution

['A benchmark protocol, Bayes-Ego, that formalizes the physical-context belief of an egocentric agent as a latent variable and scores the posterior trajectory rather than per-step answers, operationalized as three complementary consistency metrics (MCS, TCS, ACD).', 'A design principle for evolving-context evaluation: consistency must be measured against a recursive reference filter with frozen likelihoods, because any learned or judge-based reference confounds evaluator belief with agent belief.', 'An open reference-belief kernel and probe-extraction toolkit that can be attached to any EgoUseBench-style observation stream without modifying the agent under test.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | EgoUseBench Online Ego Use episodes | Evolving first-person context is the target regime |
| Dataset (secondary) | Re-instrumented HASARD episode set (egocentric vision tasks with safe-RL scoring) | Tests whether MCS/TCS/ACD rankings transfer across benchmarks |
| Primary metric | Mean Calibration Score (MCS) | Directly measures belief-trajectory agreement with reference |
| Secondary metrics | TCS, ACD, and readout-validity rate $V$ | Joint reporting distinguishes miscalibration, staleness, and probe artifacts |
| Baseline | EgoUseBench Computer-Use setting | Tests whether explicit instructions alone suffice |
| Baseline | EgoUseBench Offline Ego Use setting | Tests whether prerecorded context suffices |
| Baseline | HASARD safe-RL agent scoring | Tests whether reward-cost scoring captures belief consistency |
| Ablation-of-ours | Bayes-Ego without probe path (explicit claims only) | Isolates contribution of hidden-state readout |
| Synthetic control | Synthetic agent with known ground-truth belief trajectory | Validates that MCS/TCS/ACD recover the true belief path |

### Why this setup validates the claim

The central claim is that a recursive Bayesian reference belief over a fixed attribute vector can diagnose *belief-trajectory* failures that per-setting success (EgoUseBench) and per-episode reward-cost (HASARD) cannot see. The three baselines each probe a different sub-claim. The Computer-Use baseline tests whether explicit instructions alone yield a stable posterior path; if MCS is already high there, the evolving-context regime adds nothing and our motivation collapses. The Offline Ego Use baseline tests whether prerecorded first-person context yields a reference-consistent trajectory; if it matches Online, the recursion over arriving observations is unnecessary. The HASARD scoring baseline tests whether existing safe-RL episode scoring already captures what MCS captures; if MCS and HASARD score rank agents identically, our metric is redundant. The ablation (explicit-claims-only) isolates whether the hidden-state probe path is doing real work or merely re-reading stated claims. The synthetic control validates that MCS/TCS/ACD recover a known ground-truth belief trajectory, bounding the probe-validity assumption. MCS is the right primary metric because the claim is about *calibration of the belief path against a normative reference*, not about task success; a method could succeed by luck while its posterior path is wrong, and MCS is exactly the quantity that would reveal that. Together these form a falsifiable test: the predicted pattern is baseline parity on stable-context subsets and a gap on evolving-context subsets.

### Expected outcome and causal chain

**vs. EgoUseBench Computer-Use setting** — On a case where the wearer's activity phase silently shifts (e.g., a task-relevant object is put away mid-episode) without a new explicit instruction, the Computer-Use baseline produces a stale claim because it has no mechanism to revise belief absent new instructions. Our method instead tracks the reference posterior through the recursive Bayes update, so the agent's implied posterior is scored against the current reference, not the instruction-time one. We expect a noticeable MCS gap concentrated on episodes with mid-episode context shifts, with near-parity on episodes where the instruction fully specifies the context.

**vs. EgoUseBench Offline Ego Use setting** — On a case where the first-person observation sequence contains an ambiguous early frame later disambiguated by a new viewpoint, the Offline baseline produces a fixed interpretation because it commits to prerecorded context before execution and cannot integrate arriving evidence. Our method instead scores the agent's implied posterior at each step against a reference that updates as new observations arrive, so the anchoring cost difference (ACD) becomes positive for the offline agent precisely when it ignores the disambiguating frame. We expect Offline to show elevated ACD on late-disambiguation episodes while Online-trained agents show near-zero ACD, with parity on episodes whose context is unambiguous from the first frame.

**vs. HASARD safe-RL agent scoring** — On a case where an agent achieves high task reward by exploiting a lucky guess about object presence while its internal belief oscillates between values, the HASARD-style reward-cost scoring produces a favorable episode score because it aggregates outcomes, not belief paths. Our method instead reads the agent's implied posterior trajectory and flags the oscillation through TCS, so the same episode that scores well on reward-cost scores poorly on transition consistency. We expect the two scoring schemes to agree on episodes with stable context but to diverge sharply on episodes where success was achieved despite an inconsistent belief path.

**vs. Synthetic agent with known ground-truth belief** — On a synthetic agent whose ground-truth belief trajectory is known, the probe readout and explicit-claim extraction should recover that trajectory with high fidelity, yielding MCS near zero and TCS/ACD matching the known path. If MCS/TCS/ACD fail to recover the known trajectory, the metrics are measuring probe artifacts rather than belief consistency, and the protocol's validity is bounded accordingly.

### What would falsify this idea

If the MCS gap between Bayes-Ego and the EgoUseBench baselines is uniform across stable-context and evolving-context subsets rather than concentrated where the predicted belief-revision failure occurs, the central claim that Bayes-Ego diagnoses belief-trajectory failures is wrong. Equally, if the explicit-claims-only ablation matches the full method on MCS, the hidden-state probe path is not measuring operational belief and the design rationale collapses. Additionally, if the synthetic control fails to recover the known ground-truth belief trajectory, the metrics are not valid readouts of belief consistency.

## References

1. EgoUseBench: Benchmarking Computer-Use Agents under Evolving First-Person Contexts
