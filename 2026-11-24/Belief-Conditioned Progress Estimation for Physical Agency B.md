# Belief-Conditioned Progress Estimation for Physical Agency Benchmarks

## Motivation

WorldLeague (E3) measures Interactive Score as trajectory-derived progress toward physical goals, treating progress as externally observable from the robot's path alone. This fixed, hand-specified progress operator cannot distinguish genuine capability gains from changes in how much evidence an agent happened to gather, because the same trajectory can correspond to different beliefs about the scene; consequently the benchmark's central claim — that Interactive Score is monotone under belief refinement — is never established. The root cause is structural: progress is defined on trajectories without an explicit belief state, so the estimator has no way to express the dependence of goal progress on what the agent has inferred.

## Key Insight

If the progress operator is defined as a function of the agent's belief state rather than its trajectory, then monotone consistency under belief refinement becomes a checkable property of that function — one that can be tested by intervening on evidence, not merely assumed.

## Method

**What it is.** We introduce BCPE (Belief-Conditioned Progress Estimator), a drop-in replacement for WorldLeague's trajectory-only Interactive Score. Its inputs are the agent's observation history and actions within a WorldLeague episode; its output is a scalar progress estimate p_t ∈ [0,1] at each timestep together with a belief-state summary b_t that the estimate is conditioned on.

**Load-bearing assumption (stated explicitly).** BCPE's monotone-consistency test is only meaningful if the belief summary b_t is a sufficient statistic for task-relevant scene belief — i.e., the 128-d GRU hidden state recovers the agent's posterior over scene state rather than an arbitrary label-predictive compression. A recurrent state trained end-to-end on a single scalar cross-entropy against a binary terminal label is optimized for label sufficiency, not belief fidelity, and is free to discard evidence-dependent structure. We therefore (i) train b_t with an auxiliary next-observation-prediction objective so it must retain evidence-dependent information, and (ii) run the monotone-consistency probe on an externally specified Bayesian filter over a declared per-suite state space, not on the learned encoder's own hidden state, so the probe tests the progress operator's belief-dependence independently of the encoder that would otherwise define both sides of the test.

**How it works.**

```pseudocode
Input: episode observations o_{1:t}, actions a_{1:t}, task goal g, suite ∈ {WorldSeek, WorldShape, WorldShift}
Output: progress p_t, belief summary b_t

1. Belief encoder: b_t = Enc(o_{1:t}, a_{1:t})          # GRU over per-step features; 128-d
   - trained with two heads: (a) goal-progress scorer (step 2), (b) next-observation
     predictor o_{t+1} = Dec(b_t, a_t) (2-layer MLP, 128→256→|o|, MSE loss, weight 0.5)
2. Goal-conditioned scorer: s_t = f(b_t, g)            # 2-layer MLP, 128→128→1
3. Progress: p_t = sigmoid(s_t)                         # monotone in s_t
4. Calibration: at episode end, fit temperature τ on a held-out split of
   WorldLeague episodes so that p_T matches the suite's binary success label
   (cross-entropy loss, 5 epochs, lr 1e-3)
5. External belief filter (probe reference): for each suite, declare a discrete
   state space S (WorldSeek: {unexplored, explored-no-target, target-located};
   WorldShape: {object-at-start, object-in-transit, object-at-goal};
   WorldShift: {pre-shift, post-shift-unadapted, post-shift-adapted});
   maintain a Bayesian posterior q_t(s) over S via a hand-specified transition
   and observation model, updated with each (o_t, a_t).
6. Belief-refinement probe: for each episode, construct K=5 nested evidence
   subsets o_{1:t}^{(k)} (k=1..5, increasing t) and record p_t^{(k)} and q_t^{(k)}.
7. Monotone-consistency check: report Spearman ρ between k and p_t^{(k)};
   flag violations where p_t^{(k+1)} < p_t^{(k)} - ε (ε=0.02).
8. Violation attribution: for each flagged violation, query simulator ground-truth
   state at t and t+1; label as 'genuine correction' if the true goal progress
   decreased (earlier actions were counterproductive), else 'estimator defect'.
   Report defect rate separately from raw violation rate.
```

**Why this design.** We chose a recurrent belief encoder over a transformer over the full history because WorldLeague episodes are long-horizon and the belief must be updated online as evidence arrives; the cost is that the encoder's hidden state is a lossy summary, which we accept because the monotone-consistency probe is designed to detect exactly this lossiness. We chose a goal-conditioned scalar scorer over a per-suite hand-specified progress function because WorldLeague's three suites (WorldSeek, WorldShape, WorldShift) have different notions of progress, and a single learned scorer can share structure across them; the cost is that the scorer must be trained, so we calibrate it against the suite's binary success label rather than trusting raw scores. We chose an explicit belief-refinement probe over simply reporting trajectory-based progress because the whole point is to test whether progress is invariant under evidence refinement — a property that trajectory-only estimators (such as WorldLeague's original Interactive Score) cannot even express, let alone test. A domain expert would not describe BCPE as a variant of WorldLeague's estimator: WorldLeague computes progress from the trajectory alone, whereas BCPE makes the belief state an explicit argument of the progress function, which is what enables the monotone-consistency test. The auxiliary next-observation-prediction head and the external Bayesian filter are added specifically to make the load-bearing assumption checkable: the prediction head forces b_t to retain evidence-dependent information, and the external filter provides a probe reference that is not defined by the encoder under test.

**Why it measures what we claim.** The belief summary b_t measures the agent's evolving scene-belief because the recurrent encoder is trained end-to-end to predict goal progress and to reconstruct the next observation, so b_t is a sufficient statistic for progress given the observation history under the assumption that the encoder's hidden state captures all task-relevant evidence; this assumption fails when the agent's observations contain task-irrelevant distractors that the encoder nonetheless compresses into b_t, in which case b_t reflects distractor-sensitive features rather than scene-belief. The progress score p_t measures goal progress because the goal-conditioned scorer is calibrated so that p_T matches the suite's binary success label, under the assumption that the success label is a faithful indicator of goal achievement; this assumption fails when the success label is noisy or when the task admits partial success not captured by a binary label, in which case p_t reflects the label's idiosyncrasies rather than true progress. The monotone-consistency violation rate measures the estimator's invariance under belief refinement because the nested evidence subsets o_{1:t}^{(k)} differ only in how much evidence the agent has gathered, under the assumption that more evidence cannot reduce true progress; this assumption fails when additional evidence reveals that the agent's earlier actions were counterproductive, in which case a violation reflects a genuine correction rather than an estimator defect — which is why we separately report the defect rate after simulator-ground-truth attribution.

## Contribution

["BCPE, a belief-conditioned progress estimator that replaces WorldLeague's trajectory-only Interactive Score with an explicit function of the agent's evolving scene-belief, making the belief state a first-class argument of the progress operator.", 'A monotone-consistency probe that operationalizes the previously untested claim that Interactive Score is invariant under belief refinement, by constructing nested evidence subsets and measuring the violation rate of progress estimates.', "An empirical characterization of how belief-conditioned progress estimates differ from trajectory-only estimates across WorldLeague's three suites, identifying which suites and capability dimensions are most sensitive to belief refinement."]

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | WorldLeague 60-task suite (WorldSeek/Shape/Shift); pre-registered 10-task-per-suite subset for full probe sweep, full 60-task suite for headline comparison | Long-horizon tasks with per-suite progress notions; subset controls probe compute |
| Primary metric | Monotone-consistency defect rate (after ground-truth attribution) + Spearman ρ | Directly tests belief-refinement invariance claim, separating defect from genuine correction |
| Secondary metric | Belief-probe accuracy: linear probe predicting true task-relevant scene state from b_t | Measures belief lossiness rather than assuming it away |
| Baseline | WorldLeague trajectory-only Interactive Score | The estimator we claim cannot express belief refinement |
| Baseline | Recurrent progress regressor without belief probe | Tests whether probe, not recurrence, drives gains |
| Baseline | Per-suite hand-specified progress function | Tests whether learned cross-suite scorer matters |
| Baseline | Recurrent success classifier (belief-conditioned latent-state progress model) | Direct comparison against an existing belief-conditioned estimator |
| Ablation-of-ours | BCPE without calibration (raw sigmoid) | Isolates calibration's role in label agreement |
| Ablation-of-ours | BCPE without next-observation-prediction head | Isolates the auxiliary objective's role in belief fidelity |
| Ablation-of-ours | Probe on learned encoder state vs. external Bayesian filter | Tests whether the probe's reference matters |

### Why this setup validates the claim

The central claim is that making the belief state an explicit argument of the progress function yields a progress estimate that is invariant under evidence refinement — a property trajectory-only estimators cannot even express. The WorldLeague suite is the right dataset because its three suites give different notions of progress over long horizons, so a single learned scorer is forced to share structure rather than memorize one suite's hand-specified function. The primary metric, monotone-consistency defect rate after simulator-ground-truth attribution, paired with Spearman ρ between evidence-subset index and progress, is the right detector because it is precisely the property the design is built to test: if BCPE's belief summary is genuinely a sufficient statistic for progress, refining evidence should not decrease progress except where genuine correction occurs, and the ground-truth oracle lets us label each violation as defect vs. genuine correction. The belief-probe accuracy metric directly tests the load-bearing assumption by measuring whether b_t retains task-relevant scene state. The trajectory-only Interactive Score baseline tests the expressiveness sub-claim (it cannot run the probe at all, so it cannot be invariant or non-invariant). The unprobed recurrent regressor tests whether the probe or the recurrence does the work. The hand-specified per-suite baseline tests whether sharing structure across suites helps or hurts. The recurrent success classifier provides a direct comparison against an existing belief-conditioned estimator. The no-calibration and no-prediction-head ablations isolate calibration and the auxiliary objective. The probe-reference ablation tests whether using the external Bayesian filter rather than the learned encoder state changes the conclusion. Together these form a falsifiable test: if the gain is uniform across all subsets and baselines rather than concentrated where the predicted belief-lossiness failure occurs, the mechanism story is wrong.

### Expected outcome and causal chain

**vs. WorldLeague trajectory-only Interactive Score** — On a WorldShift episode where the scene changes mid-episode and the agent must revise its target, the trajectory-only estimator produces a progress curve that depends only on the action sequence and cannot distinguish an agent that gathered diagnostic evidence from one that did not, because its progress function has no belief argument. Our method instead conditions on the recurrent belief summary, so the same action trajectory with richer evidence yields a different progress estimate, and the monotone-consistency probe can register whether refinement helped. We expect a noticeable gap in defect rate on WorldShift (where belief revision matters most) but near-parity on WorldShape (where object rearrangement is largely action-determined).

**vs. Recurrent progress regressor without belief probe** — On a WorldSeek episode where the agent explores early and acts late, the unprobed regressor produces a progress curve that tracks actions but is never tested for evidence-refinement invariance, so its hidden state's lossiness goes undetected and it can silently assign high progress to a distractor-driven belief. Our method instead runs the nested-evidence probe and flags violations, exposing exactly where the belief summary is lossy. We expect the unprobed regressor to show comparable calibration error but a much higher undetected violation rate, since it cannot report what it never measures.

**vs. Per-suite hand-specified progress function** — On a WorldSeek task where the hand-specified function defines progress as distance-to-goal, the baseline produces a monotone-looking curve that is blind to whether the agent has actually gathered the evidence needed to know the goal location, because the hand-specified function encodes one suite's notion and cannot share structure. Our method instead learns a goal-conditioned scorer across suites, so it can trade off evidence-gathering against motion. We expect the learned scorer to match the hand-specified baseline on pure-motion WorldShape tasks but show lower defect rates on WorldSeek and WorldShift, where evidence and progress are entangled.

**vs. Recurrent success classifier (belief-conditioned latent-state progress model)** — On a WorldShift episode where the scene changes, the success classifier's latent state is trained only to predict terminal success, so it may compress away the evidence distinctions that distinguish pre-shift from post-shift adaptation. Our method adds the next-observation-prediction head and the external Bayesian filter probe, so it can detect and attribute violations. We expect BCPE to show lower defect rate and higher belief-probe accuracy than the success classifier, with the gap concentrated on WorldShift.

### What would falsify this idea

If BCPE's monotone-consistency defect rate is essentially uniform across WorldSeek, WorldShape, and WorldShift — rather than concentrated on the suites where belief revision and evidence-gathering are predicted to matter — then the belief-conditioning mechanism is not doing the work the argument claims, and the central claim is wrong. Likewise, if the no-calibration ablation matches full BCPE on label agreement, calibration is not the operative ingredient; and if the no-prediction-head ablation matches full BCPE on belief-probe accuracy, the auxiliary objective is not doing the work.

## References

1. WorldLeague: Benchmarking Physical Agency
