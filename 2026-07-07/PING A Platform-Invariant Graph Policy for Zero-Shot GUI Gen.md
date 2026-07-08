# PING: A Platform-Invariant Graph Policy for Zero-Shot GUI Generalization

## Motivation

Current GUI agents like UI-MOPD require platform-specific expert teachers, which are expensive to obtain for every target platform, limiting scalability to unseen platforms. The root cause is that these methods train on platform-dependent visual features and action distributions, failing to extract the shared functional structure of UI elements that persists across platforms.

## Key Insight

Accessibility trees encode UI functional structure (widget types, parent-child relations) that is invariant to visual style, so a graph neural network trained with a contrastive loss that matches elements with identical structural contexts across platforms learns representations that transfer zero-shot to new platforms.

## Method

(A) **What it is**: PING (Platform-Invariant Graph Policy) is a graph neural network policy that takes an accessibility tree as input and outputs action probabilities over UI elements and action types. It is trained on a multi-platform corpus using a self-supervised contrastive loss that aligns graph embeddings of UI elements with identical structural contexts, suppressing platform-specific style.

(B) **How it works**:
```pseudocode
# Training
for each batch of trees from multiple platforms:
    # 1. Compute structural context hash for each node
    for each node v in tree:
        context = [type(v), parent_type(v), sorted(children_types(v)), depth(v)]
        hash_v = hash(context)
    # 2. Encode all nodes with shared GNN (3-layer GIN, hidden=128)
    embeddings = GNN(tree_batch)  # node embeddings
    # 3. Contrastive loss (InfoNCE, tau=0.1)
    L_contrast = 0
    for anchor node in batch:
        positives = nodes with same hash from any platform (excluding self)
        negatives = random nodes with different hash
        L_contrast += -log(exp(sim(anchor, positive)/tau) / (sum(positives) + sum(negatives)))
    # 4. Policy loss (on subset with expert demos, lambda=0.1)
    L_policy = CE(action_type_pred, action_type_true) + CE(element_pred, element_true)
    total_loss = L_contrast + lambda * L_policy
    update GNN and policy head

# Zero-shot inference
1. Extract accessibility tree from novel platform (or estimate from vision)
2. Encode tree with frozen GNN
3. For each interaction step, compute action distribution via policy head (MLP on node embeddings)
4. Execute action (e.g., click on argmax element)
```

(C) **Why this design**: We chose structural context hashing over learned matching (e.g., a Siamese network) because hashing is deterministic and guarantees that nodes with identical functional roles are always paired, avoiding false negatives from a learned matcher. The cost is that it cannot capture semantic similarity when contexts differ slightly (e.g., a button in a form vs a button in a toolbar). We decoupled contrastive and supervised losses because the contrastive loss can leverage large amounts of unlabeled trees (accessible from many platforms via tooling), while the supervised loss requires expensive expert demonstrations but provides direct task-specific grounding. The trade-off is that if the supervised data is too scarce, the policy head may overfit to platform-specific patterns, but we rely on the contrastive loss to force the backbone to be invariant. We used a standard InfoNCE loss because it is computationally efficient and works well with large batch sizes; alternatives like Triplet loss with hard negative mining would be more sensitive to hyperparameters.

(D) **Why it measures what we claim**: The computational quantity `sim(embedding_anchor, embedding_positive)` measuring cosine similarity is used to enforce that nodes with same structural context hash are mapped to similar representations. This measures `invariance to platform-specific visual styles` under the assumption that the structural context hash (type, parent, children, depth) is a sufficient statistic for functional role across platforms; this assumption fails when two nodes have the same structural context but different functional roles (e.g., a "cancel" vs "delete" button with same parent type), in which case the metric instead measures `false invariance`, potentially confusing actions in safety-critical apps. The policy loss L_policy (cross-entropy on action type and element selection) measures `task-specific action correctness` under the assumption that the expert demonstrations are representative of optimal behavior on that platform; this assumption fails when the expert's demonstrations include platform-specific strategies (e.g., a different navigation pattern), in which case the metric measures `mimicry of a specific platform style` rather than universal task competence.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|------------------------|
| Dataset | OSWorld & MobileWorld | Standard multi-platform GUI benchmarks. |
| Primary metric | Average Success Rate | Directly measures task completion. |
| Baseline 1 | Single-platform agent | Shows no adaptation to new platforms. |
| Baseline 2 | EWC | Classic continual learning with weight regularization. |
| Baseline 3 | LwF | Knowledge distillation for old tasks. |
| Ablation-of-ours | PING w/o contrastive loss | Isolates effect of contrastive alignment. |

### Why this setup validates the claim
This setup tests PING's central claim of platform-invariant representations via contrastive learning on structural context hashes. The continual learning scenario (OSWorld followed by MobileWorld) directly assesses zero-shot transfer and retention. The single-platform baseline isolates the need for cross-platform generalization, while EWC and LwF represent standard continual learning approaches that rely on weight regularization or distillation. Our ablation removes the contrastive loss to attribute gains to the proposed alignment mechanism. The primary metric, Average Success Rate, captures both adaptation and forgetting in a single measure, making it a straightforward falsification tool: if PING's success on the new platform is not significantly higher than the single-platform agent, or if it forgets old tasks more than EWC, the invariance claim is undermined.

### Expected outcome and causal chain

**vs. Single-platform agent** — On a case where a new platform (e.g., mobile) has different UI layout and element styling, the single-platform agent fails because it memorized platform-specific visual patterns from its training platform (e.g., desktop). Our method uses structural context hashing to align nodes with identical functional roles across platforms, enabling zero-shot transfer. Thus, we expect a large gap on the new platform (e.g., >20% success rate difference) while parity on the original platform (within 5%).

**vs. EWC** — On a task requiring adapted action sequences on a new platform, EWC penalizes weight changes relative to old platform parameters, causing slow adaptation and potential failure on new task variants. Our method's contrastive loss trains a shared GNN backbone that is already invariant, so only the policy head needs adjustment. We expect PING to achieve >15% higher success on the new platform, while maintaining old-platform performance within 2% of EWC.

**vs. LwF** — On a scenario where the new platform introduces novel element types (e.g., gesture-based actions), LwF's distillation loss forces the model to mimic old platform output distributions, conflicting with new requirements. Our method's invariant representations are decoupled from platform-specific action biases, allowing the policy head to learn new actions without interference. We expect PING to outperform LwF by >10% on the new platform and >5% on the old platform due to reduced forgetting.

### What would falsify this idea
If PING's average success rate on the new platform is not significantly higher than the single-platform agent (within 5%), or if the ablation (without contrastive loss) achieves comparable overall performance, then the claimed invariance from structural context hashing and contrastive learning is not effective.

## References

1. UI-MOPD: Multi-Platform On-Policy Distillation for Continual GUI Agent Learning
2. UI-TARS-2 Technical Report: Advancing GUI Agent with Multi-Turn Reinforcement Learning
3. Mobile-Agent-v3: Fundamental Agents for GUI Automation
4. OS-Genesis: Automating GUI Agent Trajectory Construction via Reverse Task Synthesis
5. ShowUI: One Vision-Language-Action Model for GUI Visual Agent
6. OS-ATLAS: A Foundation Action Model for Generalist GUI Agents
7. Windows Agent Arena: Evaluating Multi-Modal OS Agents at Scale
8. Lemur: Harmonizing Natural Language and Code for Language Agents
