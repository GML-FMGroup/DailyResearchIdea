# Causal Skill Abstraction for Teacher-Free Continual GUI Agent Learning

## Motivation

UI-MOPD's multi-teacher distillation assumes a pre-trained teacher for every new platform, which is unscalable as the number of platforms grows. The root cause is that the agent does not separate a skill's causal effect on system state from platform-specific appearances; without this abstraction, transferring knowledge requires platform-specific supervisory signals. A causal skill abstraction that captures invariant action effects could enable teacher-free adaptation via few self-supervised interactions.

## Key Insight

The causal effect of a primitive GUI action (e.g., click, type) on the system state is invariant across platforms when projected onto a low-dimensional state abstraction (e.g., UI element type, accessibility role), allowing skill transfer without platform-specific teachers.

## Method

We propose **CSAT (Causal Skill Abstraction for Teacher-free Adaptation)**, a framework that learns a library of causal effect functions for primitive skills during a multi-platform continual learning phase, then grounds these skills on a new platform via minimal self-supervised interaction (N=50 random actions).

### (A) What it is
CSAT maintains a set of skill prototypes, each defined by a causal effect model that maps a pre-action state vector to a post-action state change vector. On a new platform, it performs a short self-supervised exploration to estimate a projection matrix W (linear, size 64x64) that aligns the prototype effects to the platform's observed state dynamics, enabling zero-shot skill execution.

### (B) How it works
```python
# Training phase (on known platforms, e.g., P1, P2, P3)
for platform in {P1, P2, P3}:
    collect trajectories (state, action, next_state)  # ~5000 transitions each
    extract low-dimensional state features φ(s) = (element_type one-hot, position normalized) dim=64
    for each skill a in {click, type, scroll}:
        learn causal effect Δ_a(φ(s)) = φ(s') - φ(s) via ridge regression (α=0.1)
    aggregate effects across platforms: Δ_a* = average of platform-specific Δ_a
    # Verification: compute cross-platform variance σ²_a; if σ²_a > 0.1, discard Δ_a* (fallback to manual)

# Adaptation phase (on new platform)
# Step 1: Self-supervised exploration (N=50 random actions)
collect data D = {(φ(s_i), a_i, φ(s_i'))}  # 50 tuples
# Step 2: Learn platform-specific projection W (linear, no bias) that minimizes:
#   min_W Σ_{(φ(s),a,φ(s')) in D} || φ(s') - φ(s) - W(Δ_a*(φ(s))) ||²
# Step 3: For each task (specified as desired state change d in φ-space), select skill by:
#   a* = argmin_a || d - W(Δ_a*(current_state)) ||
```

### (C) Why this design
We chose **linear projection W** over a learned deep network for grounding because linearity preserves the interpretability of causal effects and reduces sample complexity (N=50 is feasible with random exploration); the cost is that W cannot capture nonlinear platform-specific distortions, which may require additional interactions if UI state features are poorly aligned. We chose **aggregated prototype effects Δ_a*** (averaging) rather than a generative model of causal effects because the invariance across platforms is approximate but consistent in mean; the trade-off is that high-variance platforms (e.g., non-standard UI) may degrade the prototype. We chose **random exploration** for grounding over curriculum-based exploration because it provides broad coverage of state-action pairs without requiring a task specification; the cost is that some state-action pairs may be unavailable (e.g., no editable text field to test 'type'), requiring fallback to manual grounding if coverage is insufficient. We chose **desired-effect comparison** for skill selection rather than value-based RL because it directly leverages the causal effect model without reward shaping; the cost is that tasks must be specified as state-change goals, which is natural for GUI tasks (e.g., 'open file' ~ 'dialog appears').

### (D) Why it measures what we claim
**Computational quantity Δ_a*(φ(s))** measures **the invariant causal effect of skill a** across platforms (assumption A: platform-specific effects are zero-mean and independent), because averaging cancels platform-specific noise; this assumption fails when a platform introduces a systematic bias (failure mode F: non-standard UI distorts the average), in which case Δ_a* reflects the average effect on the closest matching known element type instead. **Projection W** measures **platform-specific grounding of the invariant effect** because it minimizes the reconstruction error of actual state changes observed during random exploration under the assumption that the true mapping from abstract effect to observed change is linear; this assumption fails when the platform's state dynamics are nonlinear (e.g., animation delays), in which case W captures the best linear approximation but may mispredict effects. **Skill selection via argmin over desired effect** measures **task-appropriate skill choice** because it assumes the task goal can be expressed as a desired state change in φ-space; this assumption fails for goals not reducible to immediate state changes (e.g., long-horizon tasks), in which case argmin may choose a skill that does not lead to the overall goal.

## Contribution

(1) Introduces a causal skill abstraction framework that learns invariant causal effect models of primitive GUI actions across platforms, enabling teacher-free transfer. (2) Presents a grounding algorithm (linear projection from self-supervised exploration) that adapts abstract skill effects to a new platform's state space with minimal interaction. (3) Provides a skill selection mechanism based on desired state-change matching, eliminating the need for platform-specific teachers.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|-----------------------|
| Dataset | OSWorld & MobileWorld (multi-platform) | Covers desktop and mobile domains |
| Primary metric | Subset-weighted success rate on new platform | Measures adaptation without forgetting |
| Baseline 1 | EWC (standard continual learning) | Tests retention and transfer across platforms |
| Baseline 2 | Single-platform expert (no transfer) | Upper bound for zero-shot adaptation |
| Baseline 3 | Direct policy (no causal abstraction) | Ablates the causal effect structure |
| Ablation-of-ours | CSAT w/o projection (raw prototypes) | Isolates importance of grounding step |
| Ablation-of-ours | CSAT w/ MLP Projection (2-layer MLP, hidden=64, ReLU) | Tests impact of linearity assumption |

Hardware: single NVIDIA V100 (32GB), 8 CPU cores. Expected runtime: training on 3 platforms ~ 2 hours, adaptation on new platform ~ 10 minutes.

### Why this setup validates the claim

This design forms a falsifiable test of the central claim that causal effect abstraction plus minimal self-supervised grounding enables zero-shot skill execution on new platforms. The dataset spans desktop and mobile, testing cross-domain transfer. The primary metric directly measures adaptation success. EWC tests whether standard continual learning retains previous skills but fails on new platforms without causal abstraction. The single-platform expert provides an upper bound for adaptation performance. Direct policy (avoiding causal structure) tests the necessity of our causal effect modeling. The ablation (removing projection) isolates the grounding step's contribution; adding MLP projection tests the linearity assumption. If our method outperforms all baselines on common UI actions but not on novel ones, the claim is supported; otherwise, the underlying assumptions are invalid.

### Expected outcome and causal chain

**vs. EWC** — On a case where the new platform reverses scroll direction, EWC produces incorrect actions because it replays prior mappings, causing catastrophic interference. Our method instead learns a projection W from random exploration that aligns the invariant effect (scroll causes content shift) to new dynamics, so we expect a significant gap in success rate on new platform (e.g., 30% vs 80%) but similar retention on old platforms.

**vs. Single-platform expert** — On a case where the new platform shares basic UI primitives but with different spatial arrangements, the expert achieves high success (~90%). Our method, with only 50 random actions, approaches but does not exceed expert (~85%) because the causal prototype captures invariant effects, but linear projection may miss subtle nonlinearities. We expect our method to be within 10% of expert on linearizable dynamics, but gap widens on nonlinear platforms.

**vs. Direct policy (no causal abstraction)** — On a case with common UI actions (click, type, scroll), the direct policy overfits to prior platform-specific effect patterns and fails to generalize to new layouts. Our method uses invariant causal effects and adapts via W, outperforming by ~20% on these tasks. For rare actions with no invariant effect, both fail equally. Thus, we expect a clear advantage on common actions but parity on rare ones.

**vs. MLP Projection ablation** — On a case where the platform's state dynamics are nonlinear (e.g., due to animation delays), the linear projection may mispredict effects, while the MLP projection better captures the mapping. The MLP variant may achieve higher success (e.g., 5% improvement) on such platforms, confirming the linearity assumption is a limiting factor. On linear dynamics, both perform similarly.

### What would falsify this idea

If our method's success rate is not higher than baselines on tasks involving common UI actions (where causal invariance should hold), or if the improvement is uniform across all tasks (including those with novel effects), then the central claim is wrong. Specifically, if CSAT with and without projection perform identically, the grounding step is unnecessary. The added MLP ablation further tests the linearity assumption: if the MLP variant does not outperform the linear version on nonlinear platforms, then the choice of projection is not the bottleneck.

## References

1. UI-MOPD: Multi-Platform On-Policy Distillation for Continual GUI Agent Learning
2. UI-TARS-2 Technical Report: Advancing GUI Agent with Multi-Turn Reinforcement Learning
3. Mobile-Agent-v3: Fundamental Agents for GUI Automation
4. OS-Genesis: Automating GUI Agent Trajectory Construction via Reverse Task Synthesis
5. ShowUI: One Vision-Language-Action Model for GUI Visual Agent
6. OS-ATLAS: A Foundation Action Model for Generalist GUI Agents
7. Windows Agent Arena: Evaluating Multi-Modal OS Agents at Scale
8. Lemur: Harmonizing Natural Language and Code for Language Agents
