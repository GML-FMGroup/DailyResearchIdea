# LARA: Latent Action Rule Augmentation for Constraint-Aware Text-Based Game Agents

## Motivation

Existing LLM-based agents for text-based games either rely on separate verification modules (e.g., MC-DML uses tree search with an external verifier) or treat constraint checking as a post-hoc filter, failing to integrate feasibility into the generative process. This decoupling causes a structural mismatch: the generator produces actions without awareness of game rules, and the verifier only prunes after expensive generation. Prior work like OR-LLM-Agent shows that domain-specific verifiers can enforce constraints, but they require predefined rules. In text-based games, rules must be inferred from environment feedback, making a pre-specified verifier infeasible.

## Key Insight

By learning a shared latent space where action content and feasibility are jointly encoded via contrastive pairs of valid/invalid actions, the feasibility score computed during decoding is structurally tied to the same representation that generates the action, ensuring that constraint checking is an intrinsic property of the generative distribution rather than an exogenous filter.

## Method

### LARA: Latent Action Rule Augmentation

**(A) What it is:** LARA augments an LLM with a latent rule encoder that outputs a feasibility score alongside each generated action token. Input: current observation and action history; Output: next action sequence and per-token feasibility probability.

**(B) How it works:**

```pseudocode
# LARA Training and Inference

# Phase 1: Latent rule representation learning (pre-training)
for each game trial:
    collect (observation, action, valid_flag) triples from environment feedback
    for each valid action (state, act, valid=1):
        sample invalid action (state, act', valid=0) with 50% probability random (from action vocabulary)
        and 50% probability counterfactual (replace one token in act with random token)
    train contrastive loss with InfoNCE:
        h_valid = Encoder(state ⊕ act)   # concatenation of state and action tokens
        h_invalid = Encoder(state ⊕ act')
        L_contrast = -log( exp(sim(h_valid, p)/τ) / (exp(sim(h_valid, p)/τ) + exp(sim(h_invalid, p)/τ)) )
        # p ∈ ℝ^d is a learned prototype vector (d=512), τ=0.1
    update Encoder and LLM backbone via LoRA (rank=8, α=16) on frozen LLM

# Phase 2: Augmented decoding with feasibility scoring (calibration ablation)
Input: current observation O, history H
for each candidate action token a_i in beam search:
    h = LLM_encoder(O ⊕ H)   # hidden state at decoding step (d=512)
    f_i = σ(MLP(h))          # 2-layer MLP, hidden=256, GeLU, output sigmoid
    # optionally calibrate: f_cal = 1/(1+exp(-(a*f_i+b))) where a,b learned on 512 held-out (state,action) pairs
    generation_score = logP(a_i | O,H) + λ * log(f_i)   # λ=0.5
    select next token by combined score

# Phase 3: Tree search guidance (optional, for complex games)
if planning horizon > 1:
    use MCTS with LARA as rollout policy
    each node expansion uses LARA to compute Q-value = f_i * (R + γ * max_{a'} Q(s',a'))
    # feasibility score acts as prior for valid actions
```

Hyperparameters: τ=0.1, λ=0.5, MCTS iterations=50, negative sampling ratio=0.5 random/0.5 counterfactual, latent dimension d=512, LoRA rank=8, α=16.

**(C) Why this design:** We chose a contrastive objective over a direct binary classification loss because contrastive learning forces the encoder to discriminate fine-grained differences between valid and invalid actions in a shared embedding space, which generalizes better to unseen state-action pairs (trade-off: harder to train and more sensitive to negative sampling strategy). We selected a per-token feasibility score rather than a global action-level score because invalid actions may become valid at later tokens (trade-off: slightly more computation, but necessary for stepwise guidance). We used a learned MLP head rather than a rule-based threshold to allow the feasibility score to adapt to the game's implicit rules (trade-off: requires training data, but rule-based approaches would need manual specification). We added an optional MCTS integration to handle long-horizon planning, accepting cost of multiple LLM calls but leveraging the feasibility score to prune unpromising branches early. We assume that the contrastively trained embeddings cluster valid actions separately from invalid actions, enabling generalization to unseen states; this assumption is load-bearing. To mitigate failure, we add a calibration step (Platt scaling, trained on 512 held-out examples) and validate via nearest-neighbor analysis (see experiment).

**(D) Why it measures what we claim:** The contrastive loss operationalizes the concept of *action feasibility* by encoding the structural difference between valid and invalid actions in the latent space: the learned representation h must capture the game rule invariants that distinguish permissible from impermissible actions. The cosine similarity used in the contrastive objective measures *feasibility similarity* to a learned prototype p under the assumption that valid actions form a compact cluster in latent space; this assumption fails when game rules are non-stationary (e.g., inventory changes mid-game), in which case the prototype may shift and the score reflects an outdated cluster center. The sigmoid feasibility score f_i measures *per-token feasibility probability* because it is computed from the same hidden representation that conditions on the state, under the assumption that the neural head can approximate the true conditional probability of validity given the state and prefix; this assumption fails when the state representation is impoverished (e.g., missing agent location details), leading to a score that reflects spurious correlations from training data. The calibration step (Platt scaling) further adjusts the score to be a calibrated probability, under the assumption that the logit transformation is monotonic; this assumption fails if the miscalibration is non-monotonic.

## Contribution

(1) A novel contrastive learning framework (LARA) that jointly encodes action content and feasibility in a shared latent space, enabling per-token feasibility scores during LLM decoding without a separate verification module. (2) The design principle that constraint checking should be an intrinsic part of the generative distribution rather than an exogenous filter, demonstrated by integrating feasibility as a multiplicative factor in token selection. (3) A tree search augmentation that uses the learned feasibility scores as priors to guide MCTS, reducing exploration of invalid actions.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|---|---|---|
| Dataset | Jericho text-based games | Tests action feasibility in complex rule environments |
| Primary metric | Task success rate | Directly measures goal achievement |
| Baseline 1 | MCTS+LLM (standard) | Strong baseline without feasibility scoring |
| Baseline 2 | Tree-of-Thought (ToT) | Explores multiple reasoning paths without rule guidance |
| Baseline 3 | Zero-shot LLM | Naive baseline, no search or feasibility |
| Ablation-of-ours | LARA w/o contrastive pretraining | Isolates effect of latent rule encoding |
| Additional ablation | LARA w/ calibrated feasibility | Tests calibration effect (Platt scaling, 512 examples) |

### Why this setup validates the claim

This experimental design forms a falsifiable test of the central claim that LARA's latent rule encoder and feasibility scoring improve action validity and task completion. Jericho games require understanding implicit rules (e.g., inventory constraints, locked doors). Comparing against MCTS+LLM tests whether feasibility scoring adds value over pure likelihood-based search. ToT tests whether per-token feasibility is more efficient than breadth-first exploration. Zero-shot LLM establishes the lower bound. Ablation isolates the contrastive component. The additional ablation tests whether calibration further improves reliability. Success rate is the right metric because it directly reflects whether the agent avoids invalid actions and achieves goals. If LARA succeeds, it must show gains on games where rule knowledge is critical. We also measure valid action rate as a secondary metric to attribute success to feasibility. Additionally, we perform t-SNE visualization of the latent embeddings (h_valid, h_invalid) from the held-out calibration set to verify that valid actions form a compact cluster separate from invalid actions; we report the silhouette score.

### Expected outcome and causal chain

**vs. MCTS+LLM** — On a game where many actions are visually similar but one is invalid (e.g., "take key" vs "take rock"), MCTS+LLM may waste rollouts on invalid branches because its generation score ignores rule feasibility. Our method uses the contrastive encoder to assign low feasibility to invalid actions, pruning them early. We expect LARA to achieve ~20% higher success rate on games with high action ambiguity.

**vs. Tree-of-Thought** — On a multi-step puzzle where a single invalid action breaks the chain (e.g., in "Zork", crossing a chasm without a rope), ToT may explore many paths but often hits dead ends due to invalid intermediate steps. LARA's per-token feasibility prevents such dead ends, reducing required depth. We expect LARA to solve ~15% more levels on games with strict prerequisite chains.

**vs. Zero-shot LLM** — On any game, zero-shot LLM frequently generates actions like "eat the sword" which are invalid, leading to immediate failure. LARA's feasibility score blocks invalid tokens from being generated. We expect LARA to have <5% invalid action rate versus >50% for zero-shot.

**vs. Ablation (w/o contrastive)** — Without contrastive pretraining, the feasibility head gives near-uniform scores regardless of validity, effectively reducing to the baseline. We expect the ablation to match MCTS+LLM performance, confirming that contrastive learning is essential for discriminative feasibility.

**vs. LARA w/ calibrated feasibility** — On a held-out calibration set, Platt scaling improves the expected calibration error (ECE) from ~0.15 to <0.05, leading to better feasibility scores on unseen states. We expect LARA with calibration to achieve an additional ~5% success rate on games with distribution shift between training and evaluation (e.g., different room configurations).

### What would falsify this idea

If LARA's success rate improvement is uniform across all Jericho games (including simple ones where rule complexity is low) rather than concentrated on games with intricate rule sets (e.g., those requiring inventory or conditional actions), then the central claim that LARA learns game rule invariants is wrong—the gains would instead stem from a trivial bias. Additionally, if the t-SNE visualization shows no clear clustering of valid vs. invalid actions (silhouette score <0.1), then the contrastive loss fails to learn meaningful feasibility representations.

## References

1. Monte Carlo Planning with Large Language Model for Text-Based Game Agents
2. Plan-Seq-Learn: Language Model Guided RL for Solving Long Horizon Robotics Tasks
3. Everything of Thoughts: Defying the Law of Penrose Triangle for Thought Generation
4. Self-Consistency Improves Chain of Thought Reasoning in Language Models
5. Chain of Thought Prompting Elicits Reasoning in Large Language Models
6. Prompt a Robot to Walk with Large Language Models
7. SayTap: Language to Quadrupedal Locomotion
