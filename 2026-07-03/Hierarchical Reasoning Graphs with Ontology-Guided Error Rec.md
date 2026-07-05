# Hierarchical Reasoning Graphs with Ontology-Guided Error Recovery for Medical VQA

## Motivation

Existing process-reward methods like MRPO detect and penalize invalid steps in linear chain-of-thought, but errors still propagate because the chain is fixed; backtracking is not possible. The root cause is the linear sequential assumption, which ignores the hierarchical nature of medical knowledge where multiple plausible alternatives coexist, making local error recovery more natural than global backtracking.

## Key Insight

Structuring reasoning as a directed acyclic graph and using ontology-grounded alternative generation enables error recovery as a local graph splicing operation rather than global backtracking, exploiting the closure property of medical ontologies.

## Method

## Hierarchical Reasoning Graph with Dynamic Error Recovery (HRG-DER)

We propose **Hierarchical Reasoning Graph with Dynamic Error Recovery (HRG-DER)**, a framework where reasoning steps form a directed acyclic graph (DAG) instead of a linear chain. Input: multimodal medical query (image + text). Output: answer with reasoning graph. The method jointly trains a policy network for expansion selection and a graph-convolutional value network for error detection and recovery.

### How it works

```pseudocode
Algorithm: HRG-DER Training (per episode)
Input: Query q, medical ontology O (e.g., UMLS)
Parameters: Policy network π (2-layer MLP, hidden=256, ReLU, parameters θ), 
            Value network V (2-layer GCN, hidden=128, ReLU, parameters φ)
Hyperparameters: K=4 (candidate expansions per node), K'=2 (selected expansions per node),
                 τ=0.3 (low-value threshold), γ=0.9 (discount factor), D_max=5 (max depth),
                 lr_θ=1e-4, lr_φ=1e-3, Adam optimizer

Initialize graph G with root node = q
for depth d = 1 to D_max:
    for each leaf node v in G:
        # Step 1: Policy proposes candidate expansions
        candidates = π(v) = [ (action a_i, step_text t_i) for i=1..K ]
        # Step 2: Value estimation for each candidate edge
        for each candidate c_i:
            value_i = V(v, a_i, t_i)  # GCN outputs scalar value for expanded subgraph
        # Step 3: Select top K' candidates (K' <= K) based on value_i
        selected = top_K'(candidates by value_i)
        Add selected edges to G (extending DAG)
        # Step 4: Error recovery for low-value existing nodes (after initial expansion)
    for each node v in G with degree > 0 and V(v) < τ:
        Retrieve alternatives from ontology O: alt = { (a_j, t_j) semantically related to v's original expansion }
        Replace v and its outgoing edges with subgraph generated from alt (splice)
    # Step 5: Compute step-wise process rewards via deterministic verifier
    for each step (node/edge) in G:
        # Reward: +1 if step's concepts (extracted via UMLS) are consistent with parent's concepts (subsumption/relation),
        # -1 otherwise. If no ontology mapping, perform 5 Monte Carlo rollouts with current policy and use average final reward
        r_step = ontology_consistency(step) OR Monte Carlo rollout value
    # Step 6: Graph-convolutional value learning using temporal-difference updates
    for each node v:
        TD_error = r_step + γ * V(v_next) - V(v)
        Update φ to minimize TD_error
    # Step 7: Policy gradient update using advantage A = TD_error
    Update θ to maximize log π(a|v) * A
until terminal (max depth or answer node reached)
```

### Why this design
We chose a DAG architecture over a linear chain because medical reasoning naturally branches (e.g., differential diagnosis), and a DAG allows alternative expansions to coexist without forcing a single path. This trades off increased graph complexity for the ability to recover from errors locally. Second, we use a graph-convolutional value network instead of a per-step binary reward because the value of a node depends on its entire subgraph context; a linear value head would ignore siblings and children. The cost is higher training compute due to graph convolutions. Third, error recovery through ontology splicing uses the ontology's hierarchical closure to generate plausible alternatives when a node is deemed low-value, rather than naive resampling from the policy, which could produce repeats. The trade-off is reliance on ontology coverage and structure, which may be incomplete in niche subdomains. Fourth, we employ policy gradient with process rewards (inspired by MRPO) to reward individual step correctness, but we augment it with value-based advantage to credit nodes that enable future correct steps. This addresses the forest-level meta-gap of linear credit assignment by distributing credit across graph branches.

### Why it measures what we claim
The graph-convolutional value V(v) measures the expected future correctness of a reasoning subgraph rooted at node v, because it aggregates local step rewards and discounted values of children via message passing; this relies on the assumption that correctness is compositional along the DAG structure. This assumption fails when a later error propagates to earlier nodes in a non-additive way; in that case, V(v) reflects a noisy proxy of local contribution. The policy network π(v) selects expansions by maximizing advantage A = TD_error; the advantage measures the causal necessity of a step relative to the total path reward, because TD_error isolates the contribution of that step given the current value estimate. This relies on the assumption that the value function is accurately learned; when it is not, advantage becomes a poor proxy for necessity. The threshold τ triggers ontology splicing for nodes with V(v) < τ, which measures the degree of local reasoning failure; τ is a hyperparameter that trades off sensitivity (more recovery) versus premature replacement. The ontology itself provides alternative expansions that are structurally plausible (via semantic relations), but its coverage limits the recovery scope; when ontology links are missing, splicing may reintroduce the same error. Together, these components operationalize the meta-gap of linear credit assignment by converting global backtracking into local graph splicing, but the method's effectiveness hinges on value accuracy and ontology completeness. **Load-bearing assumption:** V(v) measures expected future correctness only if step rewards are additive along the DAG and the value function is perfectly learned; failure mode is non-additive interactions or inaccurate V. To mitigate, we calibrate τ on a small set of 512 examples from Slake-VQA before main training.

## Contribution

(1) A novel DAG-based reasoning framework (HRG-DER) that dynamically constructs a reasoning graph and performs local error recovery via ontology-guided node splicing for medical VQA. (2) A joint reinforcement learning algorithm that trains a policy network for expansion selection and a graph-convolutional value network for error detection, integrating process rewards and advantage estimation. (3) A design principle demonstrating that exploiting the closure property of medical ontologies enables efficient local error recovery without global backtracking or separate controllers.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|-----------------------|
| Dataset | Slake-VQA, PathVQA, GSM8K (supplementary) | Med VQA + non-med reasoning generalization |
| Primary metric | Answer accuracy | Standard and directly measures performance |
| Baseline 1 | GRPO | State-of-the-art outcome-based RL for reasoning |
| Baseline 2 | MRPO | Step-aware RL with process rewards, strong baseline |
| Baseline 3 | Llava-Med + CoT | Popular non-RL medical MLLM baseline |
| Baseline 4 | HRG w/o recovery (same DAG but no ontology splicing) | Isolates benefit of error recovery |
| Baseline 5 | HRG w/ linear value (replace GCN with per-step MLP value) | Tests impact of graph-convolutional value |
| Ablation | HRG-DER w/ tree (no sibling branches) | Isolates benefit of DAG vs. tree |
| Ablation | HRG-DER w/ linear chain (no branching) | Removes graph structure, isolates benefit of DAG |

### Why this setup validates the claim
This setup tests our central claim that hierarchical DAG reasoning with dynamic error recovery improves over linear-step RL and non-RL methods. Comparing against GRPO isolates the benefit of step-aware process rewards versus outcome-only. Comparing against MRPO, which also uses step rewards but linearly, tests the advantage of graph-structured credit assignment and error recovery. Llava-Med + CoT serves as a supervised baseline to show that RL-driven search outperforms static CoT. Baseline 4 isolates the ontology splicing component; Baseline 5 tests the graph-convolutional value design. The tree ablation controls for branching structure, and the linear chain ablation directly measures the DAG contribution. Answer accuracy is the right metric because it captures the end goal and reflects cumulative reasoning quality; a method that improves reasoning steps should boost final answers, especially on multi-step, branching cases. Additionally, GSM8K tests whether the method generalizes beyond medical domains, addressing significance concerns.

### Expected outcome and causal chain

**vs. GRPO** — On a case requiring differential diagnosis (e.g., chest X-ray with multiple findings), GRPO may follow a single incorrect path because its outcome-based reward only evaluates the final answer, missing intermediate errors. Our method instead evaluates each step via graph-convolutional value; when a step is low-value, ontology splicing recovers. We expect HRG-DER to achieve higher accuracy on such complex cases (e.g., +10% absolute) while performing similarly on simple yes/no questions.

**vs. MRPO** — On a case with branching alternatives (e.g., two equally plausible diagnoses), MRPO's linear chain forces a single path; if early wrong step, no recovery. Our DAG allows both branches to coexist; the value network chooses the better one later. We expect HRG-DER to outperform MRPO on multi-branch reasoning (e.g., +5-7% absolute) but similar on single-path reasoning.

**vs. Llava-Med + CoT** — On a case requiring domain-specific knowledge (e.g., rare disease), CoT may hallucinate because it lacks iterative verification. Our method uses RL with process rewards to correct errors via exploration and ontology splicing. Expect HRG-DER to show larger gains on hard, knowledge-intensive questions (e.g., +15% absolute) than on simple ones.

**vs. HRG w/o recovery** — Without splicing, low-value nodes remain; thus, on questions where a single step is wrong, recovery is crucial. Expect HRG-DER to outperform by +3-5% absolute on such questions.

**vs. HRG w/ linear value** — Linear value head cannot aggregate subgraph context; thus, on branching questions, credit assignment is coarse. Expect HRG-DER to outperform by +2-4% absolute.

### What would falsify this idea
If HRG-DER does not outperform MRPO on questions with branching reasoning (e.g., differential diagnosis) by a noticeable margin, or if the ablation (linear chain) matches the full method, then the DAG structure and error recovery are not critical to performance gains. Additionally, if HRG w/o recovery matches HRG-DER, ontology splicing is unnecessary.

## References

1. Breaking Failure Cascades: Step-Aware Reinforcement Learning for Medical Multimodal Reasoning
2. Chiron-o1: Igniting Multimodal Large Language Models towards Generalizable Medical Reasoning via Mentor-Intern Collaborative Search
3. GRPO-λ: Credit Assignment improves LLM Reasoning
4. MedTVT-R1: A Multimodal LLM Empowering Medical Reasoning and Diagnosis
5. GLM-4.5V and GLM-4.1V-Thinking: Towards Versatile Multimodal Reasoning with Scalable Reinforcement Learning
6. R-4B: Incentivizing General-Purpose Auto-Thinking Capability in MLLMs via Bi-Mode Annealing and Reinforce Learning
7. MedCEG: Reinforcing Verifiable Medical Reasoning with Critical Evidence Graph
8. CAPO: Towards Enhancing LLM Reasoning through Generative Credit Assignment
