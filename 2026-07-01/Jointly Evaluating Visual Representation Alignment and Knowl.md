# Jointly Evaluating Visual Representation Alignment and Knowledge Retention in Vision-Language-Action Models

## Motivation

Current VLA evaluation evaluates knowledge retention and visual representation alignment independently. The paper 'Does VLA Even Know the Basics?' measures commonsense knowledge retention via action-grounded QA, but ignores how visual representations align with the action space. Conversely, 'Don't Blind Your VLA' studies representation alignment but does not assess knowledge retention. This independent evaluation obscures a fundamental trade-off: improving visual alignment (e.g., by fine-tuning on pixel-level tasks) may degrade high-level knowledge retention as the model overfits to low-level visual features. We propose a joint evaluation framework to quantify this trade-off.

## Key Insight

The trade-off arises because knowledge retention relies on abstract semantic features while visual alignment relies on task-relevant local features; by measuring both on identical episodes and probing layerwise representations, we reveal that these two capacities are partially competing, especially in middle layers of the VLM backbone.

## Method

**(A) What it is:** JEDI (Joint Evaluation of Deployed Integrity) is an evaluation framework that takes a VLA model and a set of commonsense questions, and outputs two metrics: Knowledge Retention Score (KRS) and Visual Alignment Score (VAS). It also computes layerwise variants to localize where the trade-off occurs.

**(B) How it works:**
```python
# JEDI evaluation pseudocode
def compute_JEDI(model, questions, num_layers=3):  # reduced to 3 layers due to compute budget (1 A100 for 1 week)
    # Phase 1: Knowledge Retention Score
    KRS = 0
    episode_pool = []
    for q in questions:
        episode = construct_Act2Answer_episode(q)  # tabletop scene, single object placement
        success = model.execute(episode)            # action-grounded correctness
        KRS += success
        episode_pool.append(episode)
    KRS /= len(questions)

    # Phase 2: Visual Alignment Score (linear probing with hyperparameter tuning)
    # Train linear classifier with 5-fold cross-validation on C ∈ {0.01, 0.1, 1, 10, 100}
    features = []
    actions = []
    for episode in episode_pool:
        vis_feat = model.get_visual_features(episode.observation)  # last hidden state of vision encoder
        features.append(vis_feat)
        actions.append(episode.ground_truth_action)  # e.g., which object to place where
    X_train, X_test, y_train, y_test = train_test_split(features, actions, test_size=0.2, random_state=42)
    best_C = None
    best_val_acc = 0
    for C in [0.01, 0.1, 1, 10, 100]:
        clf = LogisticRegression(max_iter=1000, C=C, random_state=42)
        scores = cross_val_score(clf, X_train, y_train, cv=5)
        if scores.mean() > best_val_acc:
            best_val_acc = scores.mean()
            best_C = C
    clf = LogisticRegression(max_iter=1000, C=best_C, random_state=42).fit(X_train, y_train)
    VAS = accuracy_score(y_test, clf.predict(X_test))
    # Also compute nonlinear VAS (supplemental) using a 2-layer MLP (hidden=256, ReLU) trained with same CV
    # This serves as a check for the linear probing assumption.
    # ... (omitted for brevity but follows similar cross-validation on learning rate)

    # Phase 3: Layerwise probing for both metrics (only 3 layers: early, middle, late)
    layerwise_KRS = []
    layerwise_VAS = []
    for layer in [0, num_layers//2, num_layers-1]:  # 3 representative layers
        # For KRS: use intent probing from Act2Answer
        layer_KRS = probe_knowledge_at_layer(model, layer, episode_pool)
        layerwise_KRS.append(layer_KRS)
        # For VAS: train linear probe on features from layer (with CV as above)
        layer_feats = [model.get_layer_features(ep.observation, layer) for ep in episode_pool]
        # ... same linear probing as above
        layer_VAS = compute_VAS_from_features(layer_feats, actions)
        layerwise_VAS.append(layer_VAS)

    return KRS, VAS, layerwise_KRS, layerwise_VAS
```

**(C) Why this design:**
We chose to measure visual alignment via linear probing accuracy rather than representation similarity (e.g., CKA) because linear probing directly quantifies how much action-relevant information is linearly accessible in visual features, which matches our definition of alignment as 'ease of decoding the correct action'. This design decision accepts the cost that nonlinear alignments (e.g., relational reasoning) are underestimated, but ensures the metric is interpretable and comparable across models. Second, we use the same episodes for both KRS and VAS to control for task difficulty, coupling the two metrics; this means that a spurious correlation could arise if certain episodes are simply easier across the board. Third, we include layerwise probing to identify where in the network the trade-off manifests, at the cost of increased computation and the need for access to internal representations. Fourth, we choose logistic regression over a neural network probe to minimize additional free parameters and avoid overfitting, but this may miss more complex action-relevant patterns. To mitigate this, we supplement with a nonlinear probe (2-layer MLP) under cross-validation to verify that the trade-off persists under both linear and nonlinear decoding. Overall, the design prioritizes interpretability and direct measurement of the claimed concepts over raw performance or complexity.

**(D) Why it measures what we claim:**
KRS measures knowledge retention because it requires the model to recall high-level commonsense facts (e.g., 'a hammer is used for pounding') to select the correct action in a tabletop scenario; the underlying assumption is that action-grounded success reflects the model's internal access to that knowledge, which fails when motor execution errors (e.g., imprecise gripper control) cause failure despite correct knowledge, in which case KRS underestimates knowledge retention. VAS measures visual representation alignment because linear probing accuracy quantifies how well visual features linearly encode the correct action; **explicitly, VAS measures linearly decodable action information (not full alignment), assuming that action-relevant features are linearly separable—failure occurs if actions are encoded nonlinearly (e.g., via relational reasoning), causing VAS to underestimate alignment.** To verify this assumption, we additionally compute a nonlinear probe (2-layer MLP) which, if it yields higher accuracy, indicates nonlinear encoding and we note that VAS is a lower bound. Layerwise probing measures the degree to which knowledge or alignment is localized to specific layers; the assumption is that high activation at a layer indicates that the information is present there, which fails when information is distributed across layers, in which case probing may give noisy results. The joint interpretation of KRS and VAS (e.g., high KRS, low VAS) indicates that the model uses high-level reasoning not linearly captured in visual features, while low KRS, high VAS suggests overfitted visual features that do not support reasoning.

## Contribution

(1) JEDI, a novel evaluation framework that jointly measures commonsense knowledge retention and visual representation alignment in vision-language-action models on the same set of episodes, enabling direct analysis of their trade-off. (2) A systematic empirical finding that across several VLA models (e.g., RT-2, Octo, etc.), improvements in visual alignment (via linear probing) are often correlated with degraded knowledge retention on semantically complex categories (e.g., causality, function), with the trade-off most pronounced in middle layers of the VLM backbone. (3) A layerwise analysis methodology that extends Act2Answer's intent probing to also probe visual alignment, providing a finer-grained understanding of where the trade-off occurs.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Act2Answer benchmark tabletop scenes | Covers commonsense and action grounding; 100 episodes per model |
| Primary metric | Visual Alignment Score (VAS) | Quantifies action decodability from visual features |
| Baseline 1 | VQA-based commonsense evaluation | Ignores action grounding requirement |
| Baseline 2 | Task success rate (KRS only) | Misses visual alignment dimension |
| Baseline 3 | Representation similarity (CKA) on visual features vs. action labels | Measures representation overlap, not decodability |
| Baseline 4 | CKA between visual features and a learnable action embedding (linear probe) | Similar to VAS but using CKA; tests if linear probing offers distinct insights |
| Ablation-of-ours | Global metrics without layerwise probing | Tests importance of layer localization; we compare to layerwise version |
| Model variants | RT-2, Octo, OpenVLA | Test generalization of trade-off across architectures |

### Why this setup validates the claim

This experimental design provides a falsifiable test of JEDI’s central claim—that VLA models exhibit a trade-off between commonsense knowledge retention and visual representation alignment. The Act2Answer dataset requires both knowledge (e.g., object functions) and visual grounding (e.g., recognizing objects in scenes). By comparing against baselines that isolate single dimensions—VQA (knowledge without action), KRS-only (task success without alignment), and CKA (representation similarity without decodability)—we can attribute any observed dissociations to the JEDI framework. The primary metric VAS directly measures linear decodability, which operationalizes alignment as defined. The ablation (global metrics) tests whether layerwise localization is essential to detect the trade-off. To address the confound of episode difficulty, we stratify episodes by difficulty using a baseline VLA model (e.g., OpenVLA) and report KRS and VAS per difficulty quartile; if the trade-off persists within each quartile, it is not an artifact. If JEDI reveals patterns (e.g., high KRS, low VAS) that baselines miss, it validates the claim; if not, the idea is falsified.

### Expected outcome and causal chain

**vs. VQA-based evaluation** — On a case where a model memorizes commonsense facts but cannot ground them to actions (e.g., knows a hammer is for pounding but fails to select the correct hammer among distractors), VQA yields high accuracy because it only tests text. Our method requires action execution, so KRS drops due to failed motor grounding. VAS may also be low if visual features are poorly aligned. We expect a large gap: high VQA vs. lower KRS and VAS, indicating that VQA overestimates true grounded knowledge.

**vs. Task success rate (KRS only)** — On a case where a model relies on visual shortcuts (e.g., always picks the largest object) to succeed, KRS alone would appear high. However, VAS may also be high due to spurious visual correlations. Our method separately measures VAS, revealing that the model lacks true commonsense (low KRS) despite apparent task success. We expect KRS-only to show uniformly high scores, while JEDI reveals a dissociation: high KRS but low VAS for knowledge-poor models, or vice versa.

**vs. Representation similarity (CKA)** — On a case where visual representations align nonlinearly with action information (e.g., relational features), CKA may incorrectly indicate high alignment. Our linear probing VAS captures only linearly accessible information, so it would correctly show low alignment. We expect CKA to overestimate alignment compared to VAS, especially for models that encode actions nonlinearly, validating that VAS is a more action-relevant metric.

**vs. CKA with linear probe (Baseline 4)** — This baseline computes the CKA between visual features and the outputs of a linear probe trained to predict actions. If CKA aligns with VAS, the linear probing assumption is robust; if not, CKA may capture different information. We expect moderate correlation but CKA to be noisier due to sensitivity to feature dimensionalities.

### What would falsify this idea
If VAS and KRS are perfectly positively correlated across all models, tasks, and difficulty quartiles (no trade-off), then the central claim of a fundamental conflict between knowledge retention and visual alignment is falsified. Additionally, if layerwise probing shows uniform information distribution without any layer where one metric dominates, the claim that trade-offs are localized would be undermined.

## References

1. Does VLA Even Know the Basics? Measuring Commonsense and World Knowledge Retention in Vision-Language-Action Models
2. Cosmos-Reason1: From Physical Common Sense To Embodied Reasoning
3. The Llama 3 Herd of Models
4. HybridFlow: A Flexible and Efficient RLHF Framework
5. Expanding Performance Boundaries of Open-Source Multimodal Models with Model, Data, and Test-Time Scaling
6. Robotic Control via Embodied Chain-of-Thought Reasoning
7. NVLM: Open Frontier-Class Multimodal LLMs
8. BridgeData V2: A Dataset for Robot Learning at Scale
