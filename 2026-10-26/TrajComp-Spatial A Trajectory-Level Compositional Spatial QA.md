# TrajComp-Spatial: A Trajectory-Level Compositional Spatial QA Benchmark and Consistency-Constrained RL for Video-Language Models

## Motivation

Existing embodied vision-language models, such as RynnUnify, are evaluated on spatial reasoning and visual planning using benchmarks like EmbSpatial-Bench and the newly introduced Rynn-VP/Rynn-GEU, but none of these provide trajectory-level supervision that tests whether a model's spatial predictions remain consistent when object relations are composed across time. The root cause is structural: current benchmarks score each frame or subtask in isolation, so a model can achieve high per-frame accuracy while making temporally contradictory transitive inferences (e.g., predicting A behind B at t1 and B left of C at t2, yet A left of C at t3). This missing evaluation and training signal allows the persistent gap—temporally extended visual predictions are not tested for consistency under novel spatial-relation compositions—to remain unresolved.

## Key Insight

Temporal-spatial consistency is a compositional invariant: if a model's predicted pairwise spatial relations at each timestep are assembled into a single relation graph whose edges must satisfy transitivity and temporal persistence, then any contradiction is detectable without ground-truth future frames, turning consistency into a directly optimizable and auditable signal.

## Method

**What it is.** We introduce **TrajComp-Spatial**, a benchmark and training protocol for video-language models (VLMs) that predicts spatial relations across novel temporal compositions. Given a video and a sequence of compositional queries (e.g., 'Where is A relative to C after A moves behind B and then B moves left of C?'), the model outputs a sequence of pairwise spatial relations; the protocol assembles these into a per-timestep relation graph and computes a consistency score by checking transitivity and temporal persistence. The same graph-based consistency signal is used as a reward in reinforcement learning (RL) fine-tuning.

**How it works.**
```pseudocode
# Phase 1: Benchmark construction (offline)
for each scene in 3D simulated environments (AI2-THOR, 120 scenes):
    sample trajectory of N=5 objects over T=8 timesteps (1 Hz)
    for each timestep t:
        record ground-truth pairwise relations R_gt(t) (left/right/behind/in-front/above/below)
        record per-object pixel displacement d_i(t) = ||center_i(t) - center_i(t-1)||_2 (normalized by image width)
    generate compositional queries Q = {(A,B,C, t1, t2, t3)} requiring transitive inference
    # e.g., A behind B at t1, B left of C at t2 -> query A vs C at t3
    store video, queries, R_gt(t), and d_i(t) for all t

# Phase 2: Consistency-constrained RL fine-tuning (online)
initialize VLM policy π (RynnUnify-style video-language model, 7B parameters)
for each training episode:
    sample video V and compositional query Q
    π generates a sequence of relation predictions: r_1, r_2, ..., r_K (one per sub-query)
    # Build relation graph G_t for each timestep t from predicted relations
    for each t:
        G_t = assemble_graph(r_1..r_K, objects mentioned up to t)
    # Consistency reward: penalize transitivity violations and temporal contradictions
    # Temporal contradiction is penalized only when no motion is detected:
    #   motion_detected_i(t) = 1 if d_i(t) > τ_motion (τ_motion = 0.02 of image width)
    #   temporal_contradictions(G_t, G_{t-1}) = Σ_{(i,j)} 1[ r_ij(t) ≠ r_ij(t-1) AND NOT (motion_detected_i(t) OR motion_detected_j(t)) ]
    R_cons = - Σ_t [ transitivity_violations(G_t) + temporal_contradictions(G_t, G_{t-1}) ]
    # Task reward: accuracy against ground-truth relations (only for held-out test, not training)
    R_task = - Σ_k 1[r_k ≠ r_gt_k]   # computed only on training queries with GT
    # Constrained formulation: consistency is a feasibility filter; task reward selects among feasible sequences
    #   feasible = (transitivity_violations(G_t) == 0 for all t) AND (temporal_contradictions(G_t, G_{t-1}) == 0 for all t)
    #   total_reward = R_task if feasible else -∞ (i.e., large negative penalty -10)
    total_reward = R_task if feasible else -10
    update π with PPO using total_reward (batch size 64 trajectories, λ=0.5 for R_cons in logging only)

# Phase 3: Evaluation
for each test query:
    π predicts relation sequence
    compute per-query accuracy, consistency score (fraction of graphs with zero violations)
    report compositional generalization: accuracy on unseen temporal compositions (e.g., 3-step chains)
    report degeneracy diagnostics: entropy of predicted relation distribution (6-way), fraction of constant-graph episodes (all relations identical across objects/timesteps)
```

**Why this design.** We chose a graph-based consistency reward over a per-frame accuracy reward because per-frame accuracy cannot detect transitive contradictions that arise only when relations are composed across time; the cost is that graph assembly requires object tracking, which we handle by using ground-truth object identities from the simulator during training and a simple IoU tracker at test time. We chose to penalize temporal contradictions (between consecutive timesteps) rather than only static transitivity violations because the motivation targets temporal extension; this accepts the cost that rapid object motion may cause legitimate relation changes to be penalized, which we mitigate by allowing a one-timestep grace period when the object's bounding box displacement exceeds a threshold (τ_motion = 0.02 of image width). We chose PPO over supervised fine-tuning because consistency is a property of the entire predicted sequence, not of individual labels, and PPO can optimize sequence-level rewards; the trade-off is higher variance, which we reduce by using a batch of 64 trajectories per update. Unlike VinePPO or Process Reward Models, which assign credit to individual reasoning steps, our reward is computed on the assembled relation graph, a global structure that no per-step reward can capture; a domain expert would not describe this as a variant of step-level RL because the reward is not decomposable over steps. To prevent the consistency reward from being gamed by degenerate constant assignments, we make consistency a feasibility constraint (hard filter) and use the task reward to select among feasible sequences; additionally, we report degeneracy diagnostics (entropy, constant-graph fraction) as primary metrics.

**Why it measures what we claim.** The computational quantity `transitivity_violations(G_t)` measures **compositional consistency** because we assume that the ground-truth spatial relations in the simulator satisfy transitivity (a property enforced by the 3D geometry), so any violation in the predicted graph is a genuine inconsistency; this assumption fails when the model's object tracking is imperfect, in which case a violation may reflect a tracking error rather than a reasoning error, and the metric reflects tracking robustness instead. The quantity `temporal_contradictions(G_t, G_{t-1})` measures **temporal persistence** because we assume that object relations change only when objects move, so a relation flip without motion is a contradiction; this assumption fails when two objects move simultaneously in a way that swaps their relation, in which case the metric reflects motion detection rather than reasoning consistency. The reward `R_cons` measures the **degree of spatial coherence** because we assume that a model with correct spatial reasoning will produce graphs that are both transitive and temporally persistent; this assumption fails when the model overfits to the consistency reward by predicting only trivial relations (e.g., always 'left'), in which case the metric reflects degeneracy rather than coherence—we detect this via the task reward `R_task` and report both. The feasibility filter (hard constraint) ensures that degenerate-but-consistent sequences are dominated by feasible sequences that also achieve high task reward, and the degeneracy diagnostics (entropy, constant-graph fraction) directly quantify the failure mode.

## Contribution

['A trajectory-level compositional spatial QA benchmark (TrajComp-Spatial) with procedurally generated 3D scenes and queries that require transitive spatial inference across novel temporal compositions, providing the first evaluation protocol that scores temporal-spatial consistency rather than per-frame accuracy.', 'A consistency-constrained RL fine-tuning method that uses a graph-based reward penalizing transitivity violations and temporal contradictions, enabling VLMs to learn spatially coherent predictions without requiring ground-truth future frames.', 'An empirical design principle: consistency rewards must be paired with task rewards to avoid degenerate solutions, and the graph-based reward is complementary to per-step rewards because it captures global structure that step-level credit assignment cannot.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | TrajComp-Spatial 3D sim benchmark (AI2-THOR, 120 scenes, 5 objects, 8 timesteps) | Procedural trajectories give exact GT relations |
| Primary metric | Compositional accuracy on unseen 3-step chains | Directly tests the central generalization claim |
| Secondary metrics | Consistency score (fraction of zero-violation graphs), entropy of predicted relation distribution (6-way), fraction of constant-graph episodes | Quantifies degeneracy and consistency separately |
| Baseline | RynnUnify (direct reasoning) | Tests whether consistency reward adds anything |
| Baseline | RynnUnify + per-frame SFT | Tests sequence-level vs label-level supervision |
| Baseline | VinePPO-style step reward | Tests graph-global vs step-decomposable credit |
| Ablation-of-ours | TrajComp-Spatial w/o R_cons (task reward only) | Isolates the consistency reward's contribution |
| Ablation-of-ours | TrajComp-Spatial with additive reward (R_task + 0.5*R_cons) instead of feasibility filter | Tests whether hard constraint prevents degeneracy |
| Ablation-of-ours | TrajComp-Spatial with tracker-error injection (5%, 15%, 30% ID-switch) at test time | Quantifies sensitivity to tracking assumption |

### Why this setup validates the claim

The claim is that a graph-assembled consistency reward teaches a VLM to compose spatial relations across time in ways per-frame supervision cannot. The held-out 3-step compositional chains are the right primary metric because they are exactly the queries where transitive composition is required and where a model that only fits individual labels has no mechanism to succeed. The direct-reasoning RynnUnify baseline tests whether the reward is necessary at all; the per-frame SFT baseline tests the specific sub-claim that sequence-level credit matters; the VinePPO-style step-reward baseline tests the sharper sub-claim that the reward must be graph-global rather than step-decomposable. The ablation is the load-bearing control: it holds model, data, and PPO fixed while removing only R_cons, so any gap between it and the full method is attributable to the consistency signal itself rather than to RL or the benchmark. The additive-reward ablation tests whether the feasibility filter is necessary to prevent degenerate solutions. The tracker-error injection ablation quantifies the failure mode where consistency violations arise from tracking errors rather than reasoning errors. If the mechanism is real, the advantage should concentrate on multi-step chains and on instances where naive predictions violate transitivity, not spread uniformly, and degeneracy diagnostics should remain low (entropy > 1.0, constant-graph fraction < 5%).

### Expected outcome and causal chain

**vs. RynnUnify (direct reasoning)** — On a 3-step chain where A moves behind B and then B moves left of C, the baseline emits each relation from local appearance cues independently, so its final A-vs-C answer is whatever the last frame's geometry suggests, even when that contradicts the earlier steps. Our method assembles a per-timestep graph and its PPO update penalizes the transitive violation, pushing the policy toward the composition-consistent answer. We expect a clear accuracy gap on 3-step chains (≥5 points) and near-parity on 1-step relations, with the gap growing as chain length increases.

**vs. RynnUnify + per-frame SFT** — On a case where the model must flip a relation only after observed motion, per-frame SFT rewards each label in isolation, so it can learn locally plausible but mutually contradictory predictions across timesteps and has no gradient signal against the contradiction. Our temporal-contradiction term in R_cons directly penalizes a relation flip absent motion, so the policy learns to keep relations persistent until motion justifies change. We expect our method to win on the temporal-persistence subset (relation flips with and without motion) while matching SFT on static single-frame accuracy.

**vs. VinePPO-style step reward** — On a query where two individually correct sub-relations jointly violate transitivity (e.g., A left of B and B left of C but predicted A right of C), a step-level reward sees every step as correct and assigns no penalty, so the contradiction survives training. Our reward is computed on the assembled graph, so the violation is visible regardless of per-step correctness and is penalized. We expect the gap to be concentrated on the transitive-violation subset, with our consistency score substantially higher while per-step accuracy is comparable.

**vs. Ablation: additive reward (R_task + 0.5*R_cons)** — The additive formulation allows degenerate-but-consistent sequences to receive high total reward if R_task is small. We expect the additive ablation to show higher constant-graph fraction (e.g., >20%) and lower entropy, while the feasibility-filter version maintains low degeneracy and higher compositional accuracy.

**vs. Ablation: tracker-error injection** — Injecting ID-switch errors at test time will create spurious transitivity violations. We expect consistency score to degrade roughly linearly with error rate, and accuracy to degrade similarly, quantifying the sensitivity to the tracking assumption.

### What would falsify this idea

If the full method's gain over the ablation is uniform across 1-step, 2-step, and 3-step subsets rather than concentrated on multi-step chains and transitivity-violating instances, the consistency reward is not doing what we claim and the improvement is attributable to RL or extra signal rather than graph-level composition. A second falsifier: if the ablation matches the full method on transitive-violation instances specifically, the graph-global reward provides no benefit over label-level supervision. A third falsifier: if the feasibility-filter version does not reduce constant-graph fraction relative to the additive version, the hard constraint does not prevent degeneracy.

## References

1. RynnUnify: Towards Unified Embodied Brain
