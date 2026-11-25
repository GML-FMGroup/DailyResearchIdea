# Latent-Belief Shaping: Self-Supervised Posterior Estimation for Commitment-Timing Rewards Without Simulator Access

## Motivation

Reinforcement-learning methods for training language agents to gather evidence and commit at the right moment currently depend on privileged simulator state: RLVB (Lazy Agents Answer Against Their Own Beliefs) computes the exact per-step posterior over candidate answers inside the simulator to decide whether the evidence already suffices, and LMRL-Gym similarly assumes reproducible environment-side evaluation of multi-turn interaction. This creates a structural identifiability problem — the 'when to commit' reward kernel is not a function of the observable interaction trace (actions, observations, emitted answers), so the same training signal cannot be recovered in any deployment where no oracle posterior exists. Because the reward is defined on a hidden variable that only the simulator can read, the method cannot transfer to real-world tool-use or dialogue settings where the agent's own interaction history is the only available evidence.

## Key Insight

The simulator's per-step posterior over candidate answers is identifiable from observable traces up to a monotone reparameterization whenever commitment timing and outcome consistency are jointly predictive of it, so a contrastively trained latent-belief estimator can recover the evidence-sufficiency boundary that RLVB currently reads directly from privileged state.

## Method

**What it is.** We introduce LBS (Latent-Belief Shaping), a training procedure that replaces RLVB's simulator-computed posterior with a learned latent-belief estimator q_φ(z_t | h_t) over a discrete set of candidate answers, trained self-supervised on observable interaction traces and used to shape evidence-gathering versus commitment rewards. Inputs: a replay buffer of interaction traces (action a_t, observation o_t, emitted answer â_t, terminal correctness y) collected by the current policy. Outputs: per-step scalar shaping rewards r_t^gather and r_t^commit that depend only on q_φ and the trace, plus updated policy π_θ.

**Load-bearing assumption (stated explicitly).** LBS assumes that q_φ, trained only on observable traces with terminal outcome labels, is calibrated enough that its entropy H(q_φ(h_t)) and posterior mass p_t = q_φ(h_t)[â] track the true evidence-sufficiency boundary — i.e., that H(q_φ) is monotone in remaining evidence ambiguity and p_t is monotone in evidence sufficiency — so that the shaping rewards reproduce RLVB's privileged-posterior timing signal. This assumption is not proven; it is operationalized and tested via the calibration diagnostic in Phase 2b and the synthetic ground-truth-posterior experiment in the evaluation. It fails when (i) two candidates are observationally equivalent under the trace, so H(q_φ) reflects candidate-set overlap rather than true ambiguity, or (ii) the policy's commitments are systematically lucky (y=1 with low evidence), so q_φ over-predicts sufficiency and the commit reward fires early.

**How it works.**

```pseudocode
# Phase 1: trace collection (any on-policy rollout, no simulator posterior read)
for episode in range(N_episodes):
    h_0 = initial observation
    for t in range(T_max):
        a_t ~ π_θ(· | h_t)
        o_t = env.step(a_t)              # simulator returns observation only
        h_{t+1} = h_t ⊕ (a_t, o_t)
        if a_t is a commit action: â = a_t.answer; break
    y = verifier(â, ground_truth)        # binary outcome, observable
    store (h_0..h_T, â, y) in buffer B

# Phase 2: contrastive training of latent-belief estimator q_φ
# Candidate set C = {â} ∪ {answers from other episodes in same task family}
for minibatch M ⊂ B:
    for each trace τ = (h_0..h_T, â, y) in M:
        for t in 0..T:
            z_t = q_φ(h_t)               # categorical over C, |C|=8, temperature τ_c=0.1
            L_commit = -log q_φ(h_t)[â]  if t == T_commit else 0
            L_consist = -log q_φ(h_t)[â] if y == 1 else -log(1 - q_φ(h_t)[â])
            L_contrast = InfoNCE(z_t, z_{t+1}; negatives from other traces)
        L_φ = mean_t (L_commit + λ_c L_consist + λ_n L_contrast)   # λ_c=0.5, λ_n=0.1
    φ ← φ - η ∇L_φ                       # η=1e-4, Adam

# Phase 2b: calibration diagnostic (held-out, per task family)
# Split B into train (80%) and calibration (20%, ≥512 traces per family).
# On calibration set, compute ECE of q_φ(h_t)[â] against y (10 equal-frequency bins).
# If ECE > 0.10, re-train φ with λ_c increased by 0.1 (up to 1.0) and repeat.
# Also compute candidate-set overlap flag: for each calibration trace, if two candidates
# in C have q_φ mass within 0.1 at t=T_commit, mark trace as ambiguous.
# Report fraction ambiguous per family; if > 30%, suppress r_t^gather (set α=0) for that family.

# Phase 3: shaped policy optimization (PPO, no simulator posterior)
for update in range(N_updates):
    for each step t in sampled trace:
        p_t = q_φ(h_t)[â]                # estimated posterior mass on committed answer
        entropy_t = H(q_φ(h_t))          # estimated remaining ambiguity
        r_t^gather = α · entropy_t · 1[not ambiguous]   # α=0.3, reward for gathering while ambiguous
        r_t^commit = β · 1[p_t > κ] · 1[a_t is commit]  # β=1.0, κ=0.7
        r_t = r_t^gather + r_t^commit + γ · r_terminal  # γ=1.0
    π_θ ← PPO_update(π_θ, r_t, clip=0.2)
```

Hyperparameters: candidate-set size |C|=8 (drawn per task family), contrastive temperature 0.1, λ_c=0.5, λ_n=0.1, α=0.3, β=1.0, κ=0.7, PPO clip 0.2, T_max=20. Calibration diagnostic: ECE threshold 0.10, calibration set size ≥512 traces per family, ambiguity flag threshold 0.1 mass difference, gather suppression if >30% ambiguous.

**Why this design.** We chose a contrastive InfoNCE objective over a variational autoencoder for q_φ because the candidate set is discrete and small (|C|=8), so the contrastive signal directly shapes the categorical posterior without requiring a decoder, accepting the cost that q_φ is only defined over the enumerated candidate set rather than an open vocabulary. We chose to condition the shaping reward on estimated entropy H(q_φ) rather than on the model's own log-probability of the committed answer, because entropy over a contrastively trained posterior is monotone in remaining ambiguity by construction, accepting the cost that q_φ must be refreshed as the policy improves. We chose a two-phase training loop (estimator then policy) over joint end-to-end optimization, because joint training would let the policy collapse q_φ toward its own commitments and destroy the contrastive signal, accepting the cost of stale-estimator bias that we mitigate by re-training φ every 5 policy updates. Unlike RLVB, which reads the simulator's exact posterior at every step, LBS never accesses simulator state beyond the observation and the terminal verifier; unlike a process reward model (Lightman 2023), q_φ is not trained on step-level labels but on outcome consistency and commitment timing, which are the only signals available in deployment. A domain expert would not describe LBS as a variant of RLVB because the reward kernel is a function of a learned latent variable rather than a privileged simulator quantity, and would not describe it as a PRM because no step-level human or heuristic label is used.

**Why this measures what we claim.** The computational quantity H(q_φ(h_t)) measures remaining evidence ambiguity because the contrastive objective forces q_φ to concentrate mass on the correct answer exactly when the trace contains sufficient evidence to distinguish it from the negative candidates — this assumption fails when two candidates are observationally equivalent under the trace (e.g., the same tool output is consistent with two answers), in which case H(q_φ) reflects candidate-set overlap rather than true ambiguity and the gather reward is inflated. The quantity q_φ(h_t)[â] measures evidence sufficiency for commitment because the consistency loss ties posterior mass to terminal correctness, so a calibrated q_φ assigns high mass to the committed answer only when the trace supports it — this assumption fails when the policy's commitments are systematically lucky (y=1 with low evidence), in which case q_φ learns to over-predict sufficiency and the commit reward fires early. The commitment-timing indicator 1[p_t > κ] measures timely commitment because κ is set at the empirical 70th percentile of p_t on held-out successful traces, so the threshold tracks the evidence-sufficiency boundary rather than a fixed prior — this assumption fails when the held-out distribution shifts (new task family), in which case κ must be re-estimated per family. The calibration diagnostic in Phase 2b directly tests the first two assumptions by measuring ECE of q_φ(h_t)[â] against y on held-out traces and flagging candidate-set overlap; if ECE exceeds 0.10, λ_c is increased and φ re-trained, and if overlap exceeds 30% for a family, the gather reward is suppressed for that family.

## Contribution

["A latent-belief shaping framework (LBS) that replaces RLVB's simulator-computed posterior with a contrastively trained categorical estimator over candidate answers, making the evidence-gathering/commitment reward a function of observable interaction traces only.", 'A self-supervised training objective for the latent-belief estimator that combines commitment-timing prediction, outcome-consistency calibration, and temporal contrastive separation, requiring no step-level labels, judge, critic, or simulator posterior access.', 'An empirical design principle: entropy of a contrastively trained posterior over a small candidate set is a sufficient statistic for evidence ambiguity under the assumption that candidates are observationally distinguishable, with a named failure mode when they are not.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Pro Worlds visual-puzzle benchmark | Long-horizon evidence gathering with observable verifier |
| Primary metric | Success rate on unseen task families | Tests generalization, not memorized candidate sets |
| Baseline | Base VLM, no RL training | Isolates whether shaping adds anything |
| Baseline | RLVB (simulator-posterior reward) | Upper bound using privileged posterior access |
| Baseline | Outcome-only PPO | Tests whether timing signal, not just reward, matters |
| Ablation-of-ours | LBS without entropy gather term (commit-only) | Isolates gather reward's contribution to timing |
| Diagnostic | Synthetic task with computable true posterior | Measures calibration of q_φ against ground-truth posterior |

### Why this setup validates the claim

The central claim is that a learned latent-belief estimator, trained only on observable traces, can substitute for RLVB's privileged simulator posterior and still produce correct evidence-gathering-versus-commitment timing. The base VLM baseline tests whether any RL signal is needed at all; the outcome-only PPO baseline tests whether the *timing* shaping matters beyond terminal reward; RLVB tests the ceiling that privileged posterior access buys, so LBS closing most of that gap without simulator state is the load-bearing comparison. The commit-only ablation isolates the gather reward: if removing it collapses performance to outcome-only levels, the entropy term is doing the causal work. The synthetic diagnostic directly tests the load-bearing calibration assumption: on a task where the true posterior over candidates is computable at every step, we measure the rank correlation (Spearman ρ) between H(q_φ(h_t)) and the true posterior entropy, and the ECE of q_φ(h_t)[â] against the true posterior mass; a high ρ (≥0.7) and low ECE (≤0.10) support the assumption, while low ρ or high ECE falsify it. Success rate on unseen task families is the right primary metric because the predicted failure mode — inflated entropy from candidate-set overlap, or premature commitment from lucky-trace over-fitting — shows up as a generalization gap, not as in-distribution degradation.

### Expected outcome and causal chain

**vs. Base VLM, no RL training** — On a case where the agent must inspect two tool outputs before the answer is distinguishable, the base model commits after the first observation because prompting gives it no incentive to defer. Our method instead rewards high estimated entropy early and only fires the commit bonus once q_φ concentrates, so we expect a clear gap on multi-step puzzles and near-parity on single-observation puzzles.

**vs. Outcome-only PPO** — On a case where a lucky early guess happens to be correct, outcome-only PPO reinforces the early commit because terminal reward is identical to an evidence-backed answer. Our method instead credits gathering via entropy shaping and withholds the commit bonus until p_t exceeds κ, so we expect fewer premature commits and a larger gain on puzzles where luck is common early.

**vs. RLVB, simulator-posterior reward** — On a case where the simulator's exact posterior is sharp but the observation trace is ambiguous to any observer, RLVB commits decisively because it reads privileged state. Our method instead sees only the trace, so q_φ stays diffuse and it gathers longer, costing some efficiency; we expect LBS to trail RLVB modestly but stay well above the other baselines, with the gap concentrated on observationally ambiguous instances.

**vs. Synthetic diagnostic (calibration check)** — On a synthetic task where the true posterior over candidates is computable, we measure Spearman ρ between H(q_φ) and true posterior entropy and ECE of q_φ(h_t)[â] against true posterior mass. We expect ρ ≥ 0.7 and ECE ≤ 0.10 on in-distribution families, degrading on out-of-distribution families; if ρ < 0.5 or ECE > 0.20, the calibration assumption is falsified and the shaping rewards are not tracking evidence sufficiency.

### What would falsify this idea

If LBS's gain over outcome-only PPO is uniform across ambiguous and unambiguous subsets rather than concentrated where premature commitment is predicted, the entropy-shaping mechanism is not the cause. Equally, if the commit-only ablation matches full LBS, the gather reward is inert and the central claim fails. Additionally, if the synthetic diagnostic shows Spearman ρ < 0.5 or ECE > 0.20 between H(q_φ) and true posterior entropy, the load-bearing calibration assumption is falsified.

## References

1. Lazy Agents Answer Against Their Own Beliefs: Reinforcement Learning with Verifiable Beliefs Teaches Them to Act on What They Believe
