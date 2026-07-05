# Counterfactual Skill Discovery via Latent Behavioral Manifold Search

## Motivation

While ASPIRE successfully discovers and repairs known failures through evolutionary search over program space, it cannot hypothesize new skill structures for failures outside its library because its search is unbounded and not guided by outcome prediction. The robot must execute random mutations to discover a fix, which is sample-inefficient and may not produce novel compound skills. We address this by searching over a learned latent behavioral manifold where outcome predictions can be made quickly using a counterfactual model.

## Key Insight

The latent skill manifold is structured such that minimal perturbations correspond to semantically meaningful skill variations, and a learned outcome predictor can evaluate candidate skills without execution, enabling efficient counterfactual-guided search.

## Method

(A) **What it is**: We propose Counterfactual Skill Discovery via Latent Manifold Search (CSD-LMS). Inputs: a failure state (where the robot encountered an error not in its skill library), a latent skill encoder-decoder that maps between skill trajectories and low-dimensional codes, and a learned skill effect predictor. Output: a candidate new skill (latent code) that the robot can test. 

(B) **How it works**: 
```pseudocode
1. Detect failure: robot executes current skill s_current, gets failure signal f (e.g., object dropped).
2. Retrieve latent code z_current = encoder(trajectory of s_current).
3. Initialize candidate set Z_cand = {z_current}.
4. For each candidate z in Z_cand:
   - Compute predicted success probability p = Predictor(z, current_state, failure_context).
   - Compute gradient ∇_z p.
   - Generate new candidate z' = z + α * ∇_z p (with small random noise for exploration).
   - Add z' to Z_cand.
5. After K iterations, select top-M candidates with highest predicted p.
6. For each top candidate z*, decode to skill policy π_z* using decoder.
7. Execute π_z* in simulation (or real world) for N steps; record actual success.
8. If successful, add z* to skill library and update Predictor with (z*, state, outcome).
9. If not, discard and retry next candidate.
```
Hyperparameters: α=0.1, K=10, M=5, N=100. Predictor: neural network trained on (skill code, state) → success probability; trained from prior execution data and updated online with replay buffer.

(C) **Why this design**: We chose a gradient-based search over a latent manifold rather than genetic programming (ASPIRE) or LLM-based mutation (EvoEngineer) because gradient guidance from a learned outcome predictor allows efficient exploration of the skill space without requiring many costly executions; a domain expert would not describe this as a variant of ASPIRE or EvoEngineer because those methods rely on random mutation or LLM proposals without a learned predictor that directly guides the search. We chose a VAE-style encoder-decoder to embed skills into a continuous manifold because it enables smooth interpolation and gradient computation, accepting the cost of reconstruction loss that may lose fine-grained details. We chose to include random noise during gradient steps to maintain diversity and avoid getting stuck in poor local optima, accepting the cost of occasionally generating irrelevant candidates. We chose to evaluate top candidates in simulation before real-world testing because it filters out obviously bad skills, reducing wear and tear, but we accept the sim-to-real gap that may cause some failures. Finally, we chose to update the predictor online with new data using a replay buffer to improve its accuracy over time, at the cost of increased memory and computational overhead.

(D) **Why it measures what we claim**: The predicted success probability p = Predictor(z, state) operationalizes the concept of "counterfactual outcome" because it estimates the likelihood of success if skill z were executed from the given state, assuming the predictor approximates the true transition dynamics; this assumption fails when the predictor is inaccurate for unseen (z, state) pairs, in which case p reflects the predictor's inductive bias rather than true counterfactual. The gradient ∇_z p measures the direction in latent space that most increases predicted success, which operationalizes "guided search toward promising skills" because we assume the latent manifold is continuous and the predictor's gradient aligns with true improvement; this assumption fails near discontinuities in skill performance, where gradient may point to impractical codes. The selection of top-M candidates by p measures "hypothesis prioritization" because we assume p ranks skills by true success likelihood; this assumption fails when the predictor is poorly calibrated, leading to selection of overconfident but poor candidates. The online update of Predictor with actual outcomes measures "continual learning" because it incorporates new evidence to refine future predictions; this assumption fails if the predictor update overfits to recent data, degrading performance on past skills. By grounding each computational quantity in a motivation-level concept and naming the assumption that bridges them, we ensure the method is causally linked to the goal of discovering novel skill structures for failures not in the library.

## Contribution

(1) A novel algorithm, CSD-LMS, that searches over a learned latent behavioral manifold using counterfactual outcome predictions to hypothesize new skills for failures outside the existing library. (2) A design principle that gradient-based search guided by a learned predictor is more sample-efficient than evolutionary search for discovering novel skill structures, as demonstrated by the integration of outcome prediction with latent space exploration. (3) An online learning framework that updates the predictor with actual execution outcomes, enabling the system to improve its search efficiency over time.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|----------------------|
| Dataset | LIBERO-Pro manipulation tasks | Diverse, long-horizon, with perturbations. |
| Primary metric | Task success rate | Direct measure of skill effectiveness. |
| Baseline 1 | ASPIRE | Agentic skill discovery via LLM planning. |
| Baseline 2 | EvoEngineer | Evolutionary code mutation for skills. |
| Baseline 3 | Ablation: CSD-LMS w/o gradient (random search) | Isolates gradient guidance effect. |

### Why this setup validates the claim

This experimental design forms a falsifiable test of the claim that gradient-guided latent manifold search discovers better skills than alternative mutation-based methods. LIBERO-Pro provides a realistic, high-dimensional manipulation benchmark where failure states require novel skills, directly testing the method's core reactivity. Comparing against ASPIRE (which uses LLM planning) and EvoEngineer (which uses random mutation) isolates whether the learned predictor and gradient ascent provide a search advantage. The ablation (random search) measures the specific contribution of gradient guidance. Task success rate is the correct metric because it directly reflects whether a discovered skill overcomes the failure; a metric like execution time would conflate efficiency with effectiveness. If CSD-LMS outperforms both baselines and the ablation, it validates the claim that informed latent search is superior.

### Expected outcome and causal chain

**vs. ASPIRE** — On a case where a failure is due to a subtle physical nuance (e.g., object weight shift), ASPIRE's LLM proposes a skill based on language priors, which may be too coarse and fail because it lacks a learned effect predictor. Our method instead searches the latent space guided by a predictor trained on real executions, so it can find a nuanced skill (e.g., different grasp angle). We expect a noticeable gap on such subtle-failure subsets but parity on simple failures where language is sufficient.

**vs. EvoEngineer** — On a case requiring a combination of precise motions (e.g., inserting a peg), EvoEngineer's random mutations produce many irrelevant skills that rarely succeed within the trial limit because it has no gradient direction. Our method uses gradient ascent on predicted success to efficiently navigate toward promising latent codes, requiring fewer trials to find an effective skill. We expect CSD-LMS to achieve higher success rate in the same number of online evaluations (e.g., 5 attempts), with a wider gap on complex motions.

**vs. Ablation (random search)** — On any failure, random search samples uniformly in latent space, often generating skills that do not address the failure, while gradient search moves toward regions with higher predicted success. We expect gradient search to find a successful skill more often (e.g., 80% vs 30% success) and in fewer iterations on average.

### What would falsify this idea

If the success rate of CSD-LMS is not significantly higher than the random-search ablation, especially on failures requiring precise coordination, then the gradient guidance is not providing a benefit, falsifying the central claim. Alternatively, if ASPIRE or EvoEngineer match or exceed CSD-LMS on all failure types, then the learned predictor and latent manifold search do not offer advantages over existing methods.

## References

1. ASPIRE: Agentic /Skills Discovery for Robotics
2. EvoEngineer: Mastering Automated CUDA Kernel Code Evolution with Large Language Models
3. Eurekaverse: Environment Curriculum Generation via Large Language Models
4. Eureka: Human-Level Reward Design via Coding Large Language Models
5. RoboGen: Towards Unleashing Infinite Data for Automated Robot Learning via Generative Simulation
6. DeXtreme: Transfer of Agile In-hand Manipulation from Simulation to Reality
7. Code as Policies: Language Model Programs for Embodied Control
8. Evolution through Large Models
