# Dynamic Action Synthesis Agent for Domain-Invariant Computer Use

## Motivation

Existing computer-use agents (GUI or CLI) rely on fixed action spaces—either predefined HTML elements or a static set of API functions—as shown in the GUI vs. CLI benchmark where CLI agents required manual skill augmentation to reach 69.3% pass rate. This static assumption forces agents to fail on novel interfaces or tools not seen during training. The structural root cause is that the action space is treated as a closed set, whereas real-world interactions require composing operations from context. We need a unified architecture that dynamically generates actions from environmental signals, enabling generalization across domains without manual specification.

## Key Insight

Actions in computer-use tasks are compositional—any interaction can be expressed as a (verb, target, value) triple—and the target can be inferred from the environment state via a generative model conditioned on the task, making the action space effectively open-vocabulary.

## Method

(A) **What it is**: DASA (Dynamic Action Synthesis Agent) is a unified agent that takes a task description and environment state (HTML DOM or API specification) and outputs an executable action (e.g., 'click element_id=42' or 'call weather_api(location=NY')). It uses a retrieval-augmented generator to produce candidate actions dynamically, without requiring a predefined action set.

(B) **How it works** (pseudocode):
```python
# Input: task T, environment state S (parsed DOM or API spec)
# Output: action A (verb, target, value)

# Step 1: Context encoding
context = encode(S)  # e.g., flatten DOM tree or serialize API endpoints into a structured list
# Step 2: Retrieve similar past successful actions from memory M
retrieved = retrieve(top_k=5, query=(T, S), memory=M)  # dense retriever (e.g., Contriever)
# Step 3: Generate candidate actions via LLM with context and retrieval
candidates = []
for _ in range(3):  # temperature sampling
    prompt = f"Task: {T}\nContext: {context}\nRetrieved examples: {retrieved}\nGenerate an action triple (verb, target, value):"
    response = LLM(prompt, max_tokens=50)
    candidates.append(parse_triple(response))
# Step 4: Score candidates with a learned classifier (small MLP on embedding of (T, S, candidate))
scores = classifier([candidate for candidate in candidates])
A = candidates[argmax(scores)]
# Step 5: Execute A and store outcome (success/fail) in M with embedding (T,S,A) for future retrieval
```
Hyperparameters: top_k=5 (set by validation on a held-out set of tasks), temperature=0.7 for diversity, classifier is a 2-layer MLP with 256 hidden units trained via binary cross-entropy on past successes.

(C) **Why this design**: We chose retrieval-augmented generation (RAG) over pure generation or pure retrieval because (1) generation alone lacks grounding in known effective patterns, leading to hallucinated actions; retrieval alone cannot produce novel combinations. The trade-off is higher latency due to retrieval and generation, but we accept this cost for generalization. (2) We score candidates with a learned classifier rather than log-probability because LLM log-probs are poorly calibrated for action correctness (see anti-pattern 3). The classifier is trained on a small set of success/failure outcomes from validation tasks, accepting the overhead of training data collection. (3) We sample three candidates instead of one to balance exploration vs. exploitation; more samples increase diversity but also computation. This design avoids the static action assumption by generating actions on-the-fly, and the memory M allows few-shot adaptation without manual skill addition.

(D) **Why it measures what we claim**: The action generator's output (verb, target, value) measures **compositional generalization** because it must infer novel target objects from the environment context; the assumption is that the context provides sufficient information to identify the target (e.g., element id or API parameter name). This assumption fails when the environment state is ambiguous or incomplete (e.g., multiple elements with similar text), in which case the generated action may be imprecise, but the scoring classifier learns to reject such candidates by preferring those with high embedding similarity to past successful actions. The retrieval step measures **task relevance** because the retrieved examples are those whose (T,S) embeddings are nearest to the current query; the assumption is that similar tasks in similar environments require similar actions. This assumption fails when the task is entirely novel (outside the training distribution), in which case retrieval may return irrelevant examples, but the generator can still produce plausible actions from context. The classifier score measures **action feasibility** under the assumption that the embedding space captures task-context-action compatibility; failure occurs when the classifier's training data does not cover the current action's distribution, leading to mis-scores. Together, these components ensure that DASA can dynamically synthesize actions without a fixed set.

## Contribution

(1) A novel dynamic action synthesis mechanism that generates (verb, target, value) triples from environment context using retrieval-augmented LLM, eliminating the need for predefined action spaces. (2) A unified agent architecture that jointly handles web HTML and API tool use by treating both as structured environment states, enabling domain-invariant operation. (3) A memory-augmented adaptation strategy that updates the retrieval database from online outcomes, supporting few-shot generalization to new interfaces.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | WebArena | Standard web task benchmark for computer-use agents |
| Primary metric | Task success rate | Primary measure of action correctness |
| Baseline: GUI-only agent | Screen-only GUI agent (from prior work) | No action synthesis flexibility |
| Baseline: CLI original-skill | CLI agent with original skills | Limited to predefined skill set |
| Baseline: CLI augmented-skill | CLI agent with augmented skills | Still relies on fixed skills |
| Ablation: DASA w/o retrieval | DASA without retrieval step | Tests value of retrieval component |

### Why this setup validates the claim
This experimental design forms a falsifiable test of DASA's central claim—dynamic action synthesis without a predefined set—by directly comparing against methods that rely on fixed action spaces or skill libraries. WebArena provides diverse web tasks requiring compositional generalization (e.g., combining multiple API calls, handling dynamic DOM elements). The baselines isolate key failure modes: GUI-only agents cannot generate novel actions beyond screen patterns; CLI agents with original skills lack coverage for unseen tasks; even augmented-skill CLI agents are bounded by their skill definitions. The ablation (DASA w/o retrieval) tests the necessity of retrieval-augmented generation. Task success rate is the appropriate metric because it directly measures whether the agent produces executable actions that complete the task, capturing both action correctness and feasibility. If DASA's advantage concentrates on tasks requiring novel actions or adaptation to dynamic environments, the claim is supported; if the gain is uniform, the mechanism is not operating as hypothesized.

### Expected outcome and causal chain

**vs. GUI-only agent** — On a case where a button appears only after a certain condition (e.g., a popup that triggers on mouseover), a GUI-only agent attempts to click a fixed element selector that does not exist in the DOM, leading to a failure. Our method instead retrieves past successful actions on similar dynamic elements, generates a new action targeting the popup's dynamic ID, and uses its classifier to confirm feasibility, so we expect a noticeable gap on tasks with dynamic or conditional elements but parity on static page interactions.

**vs. CLI original-skill agent** — On a case requiring a novel API sequence (e.g., combining a weather API and a calendar API to suggest outdoor events), CLI original-skill fails because no single skill covers the composition. Our method retrieves past examples of combining APIs, generates a new action triple (e.g., call weather_api(location=NY), then call calendar_api(search='outdoor')), and scores it high, so we expect a large gap on compositional tasks but little difference on single-step tasks.

**vs. CLI augmented-skill agent** — On a case where the task requires a completely bespoke action (e.g., a custom API call not in any skill set), CLI augmented-skill fails despite having many skills because the needed action is not in its library. Our method generates a novel action from context and retrieval of similar (but not identical) past actions, so we expect a moderate gap on tasks requiring out-of-skill actions, with smaller gains on tasks covered by existing skills.

### What would falsify this idea
If DASA's success rate improvement is uniform across all task subsets (e.g., static and dynamic; compositional and non-compositional) rather than concentrated on subsets where the baselines predictably fail (dynamic elements for GUI, novel compositions for CLI, etc.), then the central claim of dynamic action synthesis is not supported; instead, a flat gain would suggest the advantage comes from a generic improvement like better context encoding.

## References

1. GUI vs. CLI: Execution Bottlenecks in Screen-Only and Skill-Mediated Computer-Use Agents
2. API-Bank: A Comprehensive Benchmark for Tool-Augmented LLMs
3. Few-shot Learning with Retrieval Augmented Language Models
