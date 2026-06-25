# Self-Supervised Action Learning from UI Tree Diffs for Mobile GUI Agents

## Motivation

Existing mobile GUI agents, such as MemGUI-Agent, rely on expensive human-annotated action labels (e.g., ConAct) for supervised training, while synthetic trajectory generation methods like OS-Genesis still require VLM-based exploration and reward models, limiting scalability. The root cause is the assumption that action supervision must come from explicit demonstrations or powerful models, ignoring the structural property that each user action deterministically modifies a localized subtree of the UI hierarchy. This structural invariant makes it possible to derive weak supervision from the diff between consecutive UI trees alone.

## Key Insight

The deterministic and localized nature of GUI state transitions means that the diff between consecutive UI trees is a sufficient statistic for the action taken, enabling a self-supervised contrastive objective that requires no human annotations.

## Method

We propose **DIFF-Action**, a self-supervised method to train a mobile GUI agent using only unlabeled trajectories (screenshots + accessibility trees).

(A) **What it is**: DIFF-Action is a training objective that learns to predict the UI tree diff from the initial state and a candidate action, then uses contrastive learning to align the predicted diff with the observed diff for the true action. Input: initial UI tree \(T_t\), initial screenshot \(S_t\), and a set of candidate actions (e.g., clickable elements, text fields). Output: for each candidate action \(a\), a predicted diff embedding \(d_a\) and an action selection based on similarity to the observed diff embedding \(d_{\Delta}\).

(B) **How it works**:
```python
# For each unlabeled step (T_t, S_t, T_{t+1}, S_{t+1}):
Δ = ui_tree_diff(T_t, T_{t+1})  # e.g., set of changed node paths and attributes
# Extract observed diff embedding d_Δ using a frozen DiffEncoder (pre-trained on tree diffs)

# Candidate actions A = all clickable elements + text fields in T_t
for a in A:
    # Encode action: for click at (x,y), use positional embedding; for text field, use field ID embedding
    e_a = ActionEncoder(a)
    # Predict diff embedding: use a Transformer that attends to T_t, S_t patches, and e_a
    d_a = DiffPredictor(T_t, S_t, e_a)

# Contrastive loss: InfoNCE with temperature τ=0.1
loss = -log( exp(cosine_sim(d_Δ, d_a*)/τ) / sum_{a in A} exp(cosine_sim(d_Δ, d_a)/τ) )
# a* is the true action (unknown during training, but the contrastive loss implicitly selects the correct one)

# Additionally, train an auxiliary head to predict action type (click/type) from d_Δ
# to enable action type classification at inference.
```

Hyperparameters: temperature τ=0.1, embedding dimension 256, candidate action set size capped at 50 per state, Transformer with 4 layers and 8 attention heads.

(C) **Why this design**: We chose contrastive learning over generative state prediction because it avoids pixel-level reconstruction and scales efficiently with many candidate actions, focusing on discriminative action identification. We use UI tree diffs instead of screenshot diffs because tree diffs directly correspond to component-level changes, filtering out irrelevant visual variations (e.g., animations, color shifts). We opted for a learned diff predictor rather than a rule-based simulator because UI transitions can depend on app-specific logic (e.g., button toggles) that is not captured by simple tree operations. The trade-off is that the predictor may learn spurious correlations if trained on limited data; we mitigate this with data augmentation (e.g., shuffling action order) and by pre-training the DiffEncoder on synthetic tree diffs. We chose a Transformer architecture to handle variable-size UI trees and cross-attend to screenshots for visual grounding, accepting higher computational cost over a simpler MLP.

(D) **Why it measures what we claim**: The contrastive similarity between the observed diff embedding \(d_\Delta\) and the predicted diff embedding \(d_a\) for each candidate action operationalizes **action consistency**: the action that best predicts the observed state change is likely the true action. The assumption is that the UI transition is deterministic and that the diff is a complete representation of the action's effect. This assumption fails when two different actions produce identical diffs (e.g., two buttons opening the same dialog), in which case the metric cannot distinguish them; however, such collisions are rare because actions typically affect localized subtrees with distinct node paths. The embedding similarity further assumes that the DiffEncoder captures sufficient detail; if it is too coarse, it may merge distinct diffs, but negative examples in the contrastive loss enforce discriminative representations. Thus, the method provides a weak label that approximates the action without human annotation, with the caveat that indistinguishable actions are tied.

## Contribution

(1) A self-supervised framework (DIFF-Action) that uses UI tree diffs from unlabeled trajectories as weak supervision for training GUI agents. (2) A contrastive objective that aligns predicted action effects with observed state changes, eliminating the need for expensive action annotations. (3) A method to scalably acquire training data from unlabeled user interactions or automated exploration.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | AndroidWorld | Long-horizon mobile GUI tasks |
| Primary metric | Task success rate | Measures end-to-end completion |
| Baseline 1 | ReAct-style agent (AppAgent) | Tests need for learned action prediction |
| Baseline 2 | Supervised GUI agent (UI-TARS) | Upper bound with full supervision |
| Baseline 3 | Random action selection | Validates learning from unlabeled data |
| Ablation-of-ours | DIFF-Action without contrastive (MSE) | Tests importance of contrastive formulation |

### Why this setup validates the claim
This experimental design forms a falsifiable test of the central claim that self-supervised contrastive learning on UI tree diffs can produce an effective GUI agent without human labels. AndroidWorld provides diverse long-horizon tasks requiring complex action sequences. Task success rate directly measures the agent's ability to complete tasks, reflecting practical utility. The ReAct-style baseline (AppAgent) tests whether explicit learning from trajectories is necessary compared to pure prompting. The supervised baseline (UI-TARS) establishes an upper bound, quantifying how much performance is sacrificed by removing labels. The random baseline ensures the method actually learns. The ablation (MSE loss instead of contrastive) isolates the benefit of the contrastive objective. If DIFF-Action fails to outperform ReAct or random, the claim is invalid. Conversely, if it approaches the supervised baseline, it demonstrates the power of self-supervision. The ablation pinpoints the source of gain.

### Expected outcome and causal chain

**vs. ReAct-style agent (AppAgent)** — On a task like "turn on Wi-Fi" where multiple buttons have similar appearances but different effects, AppAgent relies on vague spatial descriptions and often clicks the wrong toggle because it lacks understanding of state changes. Our method predicts the exact tree diff for each candidate action; the contrastive loss selects the button whose predicted diff matches the observed diff (e.g., toggling a switch from off to on). Thus, we expect a noticeable gap on tasks with visually similar but semantically distinct actions, while parity on tasks with unique visible labels.

**vs. Supervised GUI agent (UI-TARS)** — On a task like "set alarm at 7 AM" where human annotations are scarce and expensive, UI-TARS overfits to limited training data and struggles on unseen app layouts. Our method leverages thousands of unlabeled trajectories, learning generalizable diff patterns. However, on very nuanced tasks requiring rare action sequences, supervised training may still win. We expect overall success rate within 85-90% of the supervised agent, with the gap concentrated on infrequent edge cases.

**vs. Random action selection** — Random baseline always fails (0% success) because it picks actions without any strategy. Our method should achieve clearly non-zero success (e.g., >30%) on all tasks, proving it learns from the unlabeled data. The minimal expectation is that success rate is statistically significantly above zero.

### What would falsify this idea
If DIFF-Action's task success rate is not significantly above random, or if its gains over the ReAct baseline are uniform across all task subsets rather than concentrated on tasks with ambiguous state transitions (e.g., visually similar buttons), then the central claim that diff contrastive learning captures action semantics is false.

## References

1. MemGUI-Agent: An End-to-End Long-Horizon Mobile GUI Agent with Proactive Context Management
2. UI-Venus Technical Report: Building High-performance UI Agents with RFT
3. Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory
4. Aguvis: Unified Pure Vision Agents for Autonomous GUI Interaction
5. OS-Genesis: Automating GUI Agent Trajectory Construction via Reverse Task Synthesis
6. Needle in the Haystack for Memory Based Large Language Models
7. You Only Look at Screens: Multimodal Chain-of-Action Agents
8. CogAgent: A Visual Language Model for GUI Agents
