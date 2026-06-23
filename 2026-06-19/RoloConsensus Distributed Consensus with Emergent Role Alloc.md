# RoloConsensus: Distributed Consensus with Emergent Role Allocation via Pairwise Embeddings

## Motivation

Existing multi-agent LLM consensus methods like AgentsNet assume homogeneous agents with identical capabilities, causing performance to degrade as network size scales because they cannot leverage heterogeneity for role specialization. This structural limitation prevents effective teaming in realistic heterogeneous settings. While Agora demonstrates self-organizing communication, it lacks principled role differentiation, leaving a gap for a method that jointly optimizes consensus and role allocation.

## Key Insight

The posterior over an agent's role can be expressed as a function of the consensus variable via pairwise compatibility scores, enabling joint optimization of consensus and role allocation through a single alternating message-passing loop.

## Method

### (A) What it is
RoloConsensus is a distributed protocol where each agent maintains a latent role embedding and exchanges pairwise compatibility scores to iteratively update a consensus variable and role assignments, outputting a final consensus decision and a role allocation.

### (B) How it works
```pseudocode
Input: Agents A = {1..N}, communication graph G, required roles R with prototype vectors μ_r, hyperparameters T=10 (rounds), α=0.5, β=0.1, γ=0.3, d=64
Initialize: For each agent i, random embedding e_i ∈ ℝ^d drawn from N(0, I/d), vote v_i (scalar, drawn from Uniform(0,1)), consensus variable c = 0

For t = 1 to T:
  // Phase 1: Exchange messages
  Each agent i sends (e_i, v_i) to neighbors N(i)
  For each received (e_j, v_j), compute compatibility score s_{ij} = exp(e_i^T e_j) / Σ_{k∈N(i)} exp(e_i^T e_k)
  
  // Phase 2: Update vote and embedding (local)
  v_i = (1 - α) * v_i + α * Σ_{j∈N(i)} s_{ij} * v_j   // weighted consensus
  e_i = e_i + β * Σ_{j∈N(i)} s_{ij} * (e_j - e_i)     // pull toward compatible neighbors
  
  // Phase 3: Update global consensus (distributed average)
  c = c + γ * (mean_i v_i - c)   // simulated central aggregation

// Oversmoothing verification: After each round, compute mean pairwise embedding distance among agents assigned to different roles (using role assignments from previous iteration). If distance < δ=0.1, abort and flag need for diversity term.

// Phase 4: Role assignment via stable matching
For each role r ∈ R:
  Compute compatibility of each agent i to role r: score_{i,r} = e_i^T μ_r
Execute Gale-Shapley with agents as proposers and roles as acceptors, each role having a quota of 1.

// Phase 5: Final consensus
Agents assigned to a 'leader' role (e.g., highest-score role) aggregate votes from all agents (weighted by role-compatibility) to produce final output.
```

### (C) Why this design
We chose pairwise compatibility scores over global attention (trade-off 1) because the sparse communication graph in distributed consensus makes full attention computationally prohibitive, accepting that local interactions may miss long-range dependencies; however, the iterative message passing propagates information across the network over rounds. For role assignment, we chose stable matching over soft probabilistic assignment (trade-off 2) because tasks require each role to be filled by exactly one agent, and stable matching guarantees a conflict-free allocation without a central coordinator, albeit losing flexibility in overlapping roles. The embedding update uses a gradient-like pull toward compatible neighbors (trade-off 3) rather than a learned update rule to avoid reliance on additional training data or offline computation, accepting slower convergence. These choices are consistent with the goal of a fully decentralized, low-computation protocol that can be deployed with off-the-shelf LLMs without fine-tuning.

### (D) Why it measures what we claim
The consensus variable c aggregates votes weighted by compatibility scores s_{ij}; this operationalizes *agreement* because agents with high compatibility are assumed to share similar task understanding, so their votes reflect collective alignment. This assumption fails when agents have complementary expertise (high compatibility but different roles), in which case c averages divergent views rather than indicating true consensus. The role compatibility score score_{i,r} = e_i^T μ_r measures *role fitness* because e_i is assumed to encode task-relevant capabilities and μ_r encodes capability requirements; this assumption fails if embeddings are not discriminative enough, leading to arbitrary role assignments. The stable matching ensures *role differentiation* by preventing duplicate assignments; it assumes roles are distinct and non-overlapping, which holds for the targeted consensus tasks but not for open-ended collaborative settings.

### (E) Compatibility score assumption and verification
The compatibility score s_{ij} is assumed to measure the likelihood of beneficial collaboration: agents with high dot product are expected to have complementary or aligned capabilities. We assume that embedding dot product captures task complementarity. A failure mode occurs when agents with high dot product have conflicting goals (e.g., both trying to lead), in which case high compatibility misleads consensus. To calibrate, we compute the correlation between compatibility scores and ground-truth collaboration success on a calibration set of 512 scenarios from the training distribution. If Pearson correlation < 0.3, we flag the assumption as violated and recommend adding a repulsive diversity term to embeddings.

## Contribution

(1) A distributed consensus protocol that jointly learns role embeddings and aggregates role-specific votes via alternating message passing, enabling heterogeneous agents to self-organize into roles. (2) The insight that role allocation can be cast as a stable matching problem on the learned embedding space, ensuring equitable and conflict-free assignment without central coordination. (3) An empirical demonstration on scaled AgentsNet tasks showing improved scalability over homogeneous baselines.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | Multi-Agent Spatial Inference (MASI) — 10,000 scenarios of 4 agents navigating a 2D 10x10 grid to reach distinct goals, with obstacles. Each agent has partial observability within a 3x3 patch. Task: all agents reach goals within 20 steps. Success if all goals occupied by correct agent. | Tests spatial coordination and consensus |
| Primary metric | Task success rate | Direct measure of collective goal achievement |
| Baseline 1 | Homogeneous agents with random messages (each agent sends a random embedding from N(0,I/d) and random vote from Uniform(0,1), unchanged across rounds) | Checks benefit of compatible communication |
| Baseline 2 | Fully connected network (each agent receives messages from all other agents) | Tests against full communication baseline |
| Baseline 3 | No role assignment (average voting: all agents vote equally, final consensus is mean of all votes) | Isolates effect of role differentiation |
| Ablation-of-ours | RoloConsensus without role assignment (skip Phase 4 and Phase 5, output mean vote as consensus) | Measures gain from stable matching |

### Why this setup validates the claim

The central claim is that RoloConsensus improves distributed spatial reasoning via iterative consensus and role assignment. The MASI dataset provides tasks requiring agents to coordinate spatial decisions (e.g., path planning, object manipulation), directly testing this claim. Baseline 1 (random messages) isolates the effect of compatibility-weighted communication; if our method outperforms it, the compatibility mechanism is validated. Baseline 2 (fully connected) tests whether localized exchanges suffice, given that full connectivity is often impractical but serves as an upper bound. Baseline 3 (no role assignment) isolates the role differentiation component, while the ablation (ours without roles) further pinpoints the contribution of the stable matching step. Task success rate is the primary metric because it directly captures whether the agents collectively achieved the spatial goal, reflecting both consensus quality and role appropriateness. The combination of baselines allows falsification: if our method fails to show a gap on the relevant subset where a specific mechanism is predicted to matter, the corresponding sub-claim is unsupported.

### Expected outcome and causal chain

**vs. Homogeneous agents with random messages** — On a task where two agents must swap positions in a narrow corridor, the baseline sends random embeddings and votes, leading to conflicting signals and deadlock because agents receive irrelevant or misleading information. Our method computes compatibility scores from learned embeddings, allowing agents to identify and follow compatible peers, thus coordinating a complementary swap. We expect a noticeable gap on tasks requiring tight coordination (e.g., success rate 0.75-0.85 vs. 0.25-0.35) but parity on independent tasks (e.g., both ~0.9).

**vs. Fully connected network** — On a large network (16 agents) performing spatial map assembly, the baseline forces each agent to process messages from all others, causing information overload and slow convergence because relevant signals are diluted among many. Our method uses local exchanges with iterative propagation, enabling efficient scaling and eventual global coordination. We expect our method to match or exceed on large graphs (e.g., success rate 0.7-0.8 vs. 0.6-0.7) but lag on small graphs (e.g., 0.85-0.95 vs. 0.95-0.98) due to reduced direct connectivity.

**vs. No role assignment (average voting)** — On a task where one agent must lead navigation while others follow, averaging votes yields a mediocre consensus that fails to exploit specialization, resulting in indecisive movement. Our method assigns roles via stable matching, giving the leader priority in influencing the final decision, so the group moves decisively. We expect a significant gap on role-structured tasks (e.g., success rate 0.8-0.9 vs. 0.4-0.6) but similar on symmetric tasks (e.g., both ~0.9).

### What would falsify this idea
If the full RoloConsensus does not significantly outperform its own ablation (without role assignment) on tasks requiring distinct roles, or if the performance gain is uniform across all task types (rather than concentrated on role-structured subsets), then the central claim that role assignment via stable matching is necessary for effective spatial reasoning would be falsified.

## References

1. AgentsNet: Coordination and Collaborative Reasoning in Multi-Agent LLMs
2. A Scalable Communication Protocol for Networks of Large Language Models
3. Problem-Solving in Language Model Networks
4. War and Peace (WarAgent): Large Language Model-based Multi-Agent Simulation of World Wars
5. S3: Social-network Simulation System with Large Language Model-Empowered Agents
6. Encouraging Divergent Thinking in Large Language Models through Multi-Agent Debate
7. PaLM: Scaling Language Modeling with Pathways
8. GLM-130B: An Open Bilingual Pre-trained Model
