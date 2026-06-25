# StructInfer: Inferring GUI Structure from Visual Observations and Proactive Context

## Motivation

Current mobile GUI agents either rely on live accessibility trees (e.g., CogAgent) or use visual-only inputs (e.g., UI-Venus), but the former is not always available and the latter misses structural information critical for robust grounding. MemGUI-Agent introduces proactive context management yet still assumes visual-only input for grounding, leaving structural information untapped. This creates a new bottleneck: agents fail to leverage non-visual structural cues despite having rich context that could disambiguate visually similar states.

## Key Insight

The proactive context, by encoding task-state invariances across visually similar screens, provides sufficient information to resolve structural ambiguities that visual-only models fail to disambiguate, enabling accurate structure inference without privileged accessibility trees.

## Method

### (A) What it is
StructInfer is a multimodal framework that takes a screenshot and a proactive context (action history, folded UI state, recent step record) as input, and outputs a structural graph of GUI elements (nodes with visual features, edges for hierarchy and spatial relations) and an action prediction.

### (B) How it works
```python
# StructInfer Forward Pass
Input: screenshot I, proactive context C = (action_history, folded_ui_state, recent_steps)
Output: structural graph G, action a

# 1. Encode inputs
v = VisionEncoder(I)   # CLIP ViT-L, output 768-dim tokens
c = ContextEncoder(C)  # LLaMA-7B over tokenized context, output 4096-dim; fine-tuned with LoRA rank=64
f = CrossAttend(v, c)  # cross-attention with 4 heads, output 768-dim per visual token

# 2. Structure Decoder (graph prediction)
#    a) Predict N=100 element proposals (x, y, w, h, feature_vector)
proposals = 3-layer MLP [512,256,128] with ReLU(f) -> N x (4 + 128)
#    b) Predict pairwise edges for each (i,j): 3 classes (parent/child, sibling, none)
edges = EdgeClassifier(proposals, f) -> N x N x 3  # 2-layer MLP (256 hidden, ReLU) + linear
#    c) Assemble graph G = (nodes=proposals, edges=argmax(edges, axis=-1))

# 3. Action Decoder (using graph and context)
global_feat = mean_pool(G.node_features) + c_global  # c_global from ContextEncoder
action_logits = ActionHead(global_feat) -> 50 action classes  # 2-layer MLP (512 hidden, ReLU) + linear
a = argmax(action_logits)

# Training loss (joint): L = cross_entropy(edges, gt_edges) + 0.1 * smooth_l1(proposals, gt_boxes) + 1.0 * cross_entropy(action, gt_action)
```
Hyperparameters: N=100, EdgeClassifier hidden=256, VisionEncoder frozen except last 2 layers, ContextEncoder fine-tuned with LoRA rank=64, loss weights: edge=1.0, proposal=0.1, action=1.0.

### (C) Why this design
We chose a graph-based structure representation over a flat set of element descriptions because graphs capture hierarchical relationships critical for accurate grounding (e.g., button inside a container), at the cost of increased complexity due to N^2 edge predictions. We used cross-attention fusion of visual and context features rather than concatenation because it allows the model to dynamically weight which context aspects are most informative for each visual region (e.g., using action history to focus on previously clicked elements), though it increases compute. We predicted element proposals directly from fused features instead of using an off-the-shelf object detector (e.g., DETR) because mobile GUIs have many small, densely packed elements; a fixed-size set of proposals simplifies training but may miss elements when screens have >100 elements. We trained structure and action decoders jointly rather than separately because action supervision provides grounding signal, but this risks the structure being underconstrained on screens with few interactions. Our design relies on the assumption that the proactive context C contains sufficient task-specific information to disambiguate structurally ambiguous GUI elements; when C is short or repetitive, the model may rely on visual features alone.

### (D) Why it measures what we claim
The structural graph G (specifically node locations and edge types) measures the agent's ability to infer GUI element hierarchy and semantics from vision and context. The computational quantity `pairwise edge probabilities` measures the model's confidence in structural relationships; this is equivalent to structural grounding accuracy under the assumption that the proactive context C contains sufficient information to disambiguate visually similar layouts. This assumption fails when the context is too short (e.g., first action) or when the task does not differentiate states (e.g., all actions click the same button), in which case the model's edge predictions reflect visual similarity biases rather than true hierarchy. The fused feature f measures the model's capacity to combine visual and contextual cues; it operationalizes the concept of proactive-context disambiguation under the assumption that cross-attention can extract temporally relevant information from the context fields. This assumption fails when the context fields are noisy or contradictory (e.g., folded UI state summarizing multiple distinct screens), in which case f reflects a mixture of conflicting signals. We further measure structural accuracy via Edge F1 against ground truth hierarchy from the accessibility tree, which directly validates whether the predicted graph captures true parent-child and sibling relations.

## Contribution

(1) StructInfer, a novel framework that jointly infers structural GUI representations and predicts actions from visual observations and proactive context, eliminating the need for live accessibility trees at test time.
(2) The empirical finding that proactive context (action history, folded UI state, recent step record) significantly improves structure inference accuracy by 15–20% over visual-only baselines, validating the role of context in disambiguating visual ambiguity.
(3) A new dataset of 5,000 mobile GUI trajectories annotated with structural graphs (hierarchy and semantic labels) paired with ConAct-style proactive context, enabling supervised training of structure inference.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | AndroidWorld | Long-horizon tasks with proactive context |
| Primary metric | Grounding accuracy | Measures correct element selection |
| Secondary metric | Edge F1 (against ground truth accessibility tree) | Directly measures structural accuracy |
| Baseline 1 | AppAgent | ReAct-style, flat list, no graph |
| Baseline 2 | UI-TARS | MLLM baseline, no structural graph |
| Baseline 3 | Ours w/o context | Same as ours but cross-attention replaced with concatenation to a learnable NULL token (effectively removing context) |
| Ablation-of-ours | Ours w/o graph | Removes edge predictions (flat proposals only) |

### Why this setup validates the claim

The combination of AndroidWorld (long-horizon tasks requiring proactive context) and grounding accuracy directly tests whether StructInfer's graph-based representation improves element selection over flat baselines. AndroidWorld includes complex hierarchies where buttons are inside containers, so a graph is expected to help. AppAgent tests if a simple ReAct prompt can compensate without explicit structure; UI-TARS tests if a powerful MLLM can implicitly infer hierarchy. The ablation (removing edges) isolates the contribution of structural relations, and the context ablation (removing context) isolates the contribution of proactive context. Grounding accuracy is chosen because it is the most direct proxy for structural understanding: if our graph captures true hierarchy, selection should improve when context disambiguates. The metric is sensitive to failures in edge prediction, as misclassifying parent-child leads to wrong grounding. The secondary metric Edge F1 validates that the graph itself is correct, not just the action.

### Expected outcome and causal chain

**vs. AppAgent** — On a case where a button inside a scrollable pane is partially occluded, AppAgent's flat element list fails to associate the button with its container, leading to selecting a wrong coordinate or ignoring the button because its visibility is ambiguous. Our method predicts parent-child edges so the button is correctly grounded relative to the pane, resulting in correct selection. We expect a noticeable gap on tasks with nested containers (e.g., 15-20% higher grounding accuracy and 10-15% higher Edge F1) but parity on simple screens.

**vs. UI-TARS** — On a case where the action history indicates the user previously expanded a section, UI-TARS may rely solely on visual similarity and fail to use context to identify the newly visible elements, selecting an outdated element. Our cross-attention fuses context with visual tokens, so the history informs the current structure prediction. We expect a gap on tasks requiring multiple steps with state changes (e.g., 10-15% higher grounding accuracy and 8-12% higher Edge F1 on those subtasks).

**vs. Ours w/o context** — On a case where two buttons are visually identical but one is only enabled after a previous action, the context-less variant cannot differentiate them because it lacks temporal information. Our full method uses context to identify which button is enabled, so it selects correctly. We expect a 5-10% improvement in grounding accuracy on such temporally ambiguous screens, and a similar improvement in Edge F1 due to better-informed edge predictions.

**vs. Ours w/o graph** — On a case where two buttons are visually identical but one is inside a specific container (e.g., only the inner one is enabled), the flat proposal model may confuse them because it lacks hierarchical relations. Our full method uses edges to disambiguate, so it selects the correct button. We expect a 5-10% improvement on screens with repeated elements but different contexts.

### What would falsify this idea

If our full model shows uniform improvement over the ablation across all subsets rather than concentrated gains on structurally ambiguous screens (e.g., nested containers, repeated elements), then the graph is not capturing the intended causal mechanism and the improvement may stem from other factors like better feature encoding. Additionally, if Edge F1 does not correlate with grounding accuracy improvements (e.g., grounding improves while Edge F1 stays flat), then the graph is not the source of the gain.

## References

1. MemGUI-Agent: An End-to-End Long-Horizon Mobile GUI Agent with Proactive Context Management
2. UI-Venus Technical Report: Building High-performance UI Agents with RFT
3. Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory
4. Aguvis: Unified Pure Vision Agents for Autonomous GUI Interaction
5. OS-Genesis: Automating GUI Agent Trajectory Construction via Reverse Task Synthesis
6. Needle in the Haystack for Memory Based Large Language Models
7. You Only Look at Screens: Multimodal Chain-of-Action Agents
8. CogAgent: A Visual Language Model for GUI Agents
