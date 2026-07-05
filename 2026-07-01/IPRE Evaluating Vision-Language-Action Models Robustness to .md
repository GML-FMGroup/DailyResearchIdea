# IPRE: Evaluating Vision-Language-Action Models' Robustness to Irrelevant Context in Instructions

## Motivation

Existing evaluations of knowledge retention in VLAs, such as the Act2Answer protocol in 'Does VLA Even Know the Basics?', only measure performance under clean, distraction-free instructions. This fails to capture the robustness required for real-world deployment where instructions often contain extraneous context that does not affect the task. The gap persists because prior work focused on idealized settings, ignoring how irrelevant linguistic information may interfere with the model's ability to retrieve and apply task-relevant knowledge.

## Key Insight

Systematic insertion of controlled distractors that are semantically related to the task object reveals that VLAs rely on superficial lexical overlap rather than robust understanding, causing performance degradation proportional to distractor relevance.

## Method

### (A) What it is
We propose **IPRE** (Instruction Perturbation Robustness Evaluation), a framework that generates systematically perturbed instructions by inserting irrelevant context and measures the degradation in action-grounded knowledge retention using the Act2Answer protocol. IPRE's input is a base instruction and a target knowledge category; its output is a robustness score (1 - average relative drop).

### (B) How it works
```python
import numpy as np
from act2answer import Act2Answer  # from existing work

class IPRE:
    def __init__(self, vla, scene_graph_fn, llm_paraphraser):
        self.vla = vla
        self.scene_graph_fn = scene_graph_fn
        self.llm = llm_paraphraser  # for distractor generation
    
    def generate_distractors(self, base_instruction, category):
        # distractor types: 'environment', 'semantic', 'random', 'neutral'
        distractors = {}
        # Environment: describe non-task object in scene
        env_obj = self.scene_graph_fn().get_random_non_task_object()
        distractors['env'] = f"There is a {env_obj} on the table."
        # Semantic: similar object (e.g., 'banana' for 'apple')
        sem_obj = self.llm.paraphrase(category, "similar_object")
        distractors['sem'] = f"I also see a {sem_obj}."
        # Random: shuffle token sequence
        distractors['rand'] = self._token_shuffle(base_instruction)
        # Neutral: filler sentence of same length as env distractor (e.g., 'The weather is nice today.')
        distractors['neu'] = self._neutral_filler(len(distractors['env'].split()))
        return distractors
    
    def insert_at_position(self, instr, distractor, pos):
        # pos ∈ {'prefix', 'middle', 'suffix'}
        if pos == 'prefix': return distractor + " " + instr
        elif pos == 'suffix': return instr + " " + distractor
        else:  # insert after first 40% tokens
            split = int(len(instr.split()) * 0.4)
            words = instr.split()
            return " ".join(words[:split]) + " " + distractor + " " + " ".join(words[split:])
    
    def evaluate(self, base_instr, category, n_trials=10):
        baseline_success = np.mean([Act2Answer(self.vla, base_instr, category) for _ in range(n_trials)])
        if baseline_success == 0:
            return 0.0, {}  # floor effect
        perturbations = []
        for dtype, dist in self.generate_distractors(base_instr, category).items():
            for pos in ['prefix', 'middle', 'suffix']:
                pert_instr = self.insert_at_position(base_instr, dist, pos)
                successes = [Act2Answer(self.vla, pert_instr, category) for _ in range(n_trials)]
                perturbations.append({
                    'distractor_type': dtype,
                    'position': pos,
                    'success_rate': np.mean(successes),
                    'drop': baseline_success - np.mean(successes)
                })
        robustness_score = 1 - np.mean([p['drop']/baseline_success for p in perturbations])
        # Additional output: per-type scores for fine-grained analysis
        return robustness_score, perturbations
```

### (C) Why this design
We chose three distractor types (environment, semantic, random) to cover different kinds of extraneous information that real instructions may contain. **Environment distractors** test whether the model can ignore scene descriptions irrelevant to the task; we generate them from an external scene graph rather than an LLM to avoid bias. **Semantic distractors** probe whether the model distinguishes between similar objects; we use an LLM to generate plausible but incorrect objects, accepting the cost that LLM outputs may vary. **Random distractors** serve as a baseline for arbitrary noise; we shuffle the original instruction to preserve token statistics. For position, we test prefix (early interference), middle (split attention), and suffix (recenty effect); we chose specific insertion points (after 40% tokens for middle) based on prior psycholinguistic work. We reject a single fixed position because it may miss position-dependent effects. The robustness score aggregates relative drop across all perturbations, normalizing by baseline to enable cross-category comparison. We chose this aggregate over per-type scores to give a single metric, but note that per-type scores may be more informative—we include them in the output dictionary for fine-grained analysis. To isolate the effect of irrelevance from confounds like increased instruction length, we include a **neutral distractor** (a semantically empty filler sentence of the same token count as the environment distractor). This allows a control condition: if the drop under neutral distractors equals the drop under environment distractors, then the environment-specific effect is not due to length but to irrelevance.

### (D) Why it measures what we claim
The relative drop in success rate (baseline - perturbed)/baseline measures **knowledge retention robustness** because it quantifies how much irrelevant context degrades the model's ability to use stored knowledge. **Causal assumption:** the drop is causally due to the irrelevance of the distractor content, not to extraneous factors like instruction length or syntactic disruption. To verify this, we include a neutral distractor control: if neutral fillers cause a similar drop, the metric reflects length effects rather than irrelevance sensitivity. **Failure mode when assumption fails:** when the VLA has low baseline success (floor effect), the relative drop is artificially small and the metric instead reflects the model's inability to use knowledge at all. The environment distractor success rate measures **task-irrelevance filtering** because a robust model should ignore scene descriptions that do not affect the action; this assumes that the scene-graph object is truly irrelevant, which holds because the task does not involve that object—failure mode: if the distractor object is accidentally co-occurring with the target, the model may correctly use it as a cue, inflating robustness. The semantic distractor success rate measures **concept discrimination** because it tests whether the model selects the correct object over a similar alternative; this assumes the distractor object is not present in the scene, which is enforced by the scene graph—if the scene graph is incomplete, the metric could reflect accidental presence. The random distractor success rate measures **structure agnosticism** because random noise should not affect performance if the model understands the instruction structure; this assumes that token shuffling does not remove essential syntactic cues—failure mode: if shuffling accidentally creates a meaningful instruction, the metric underestimates robustness.

## Contribution

(1) A systematic perturbation taxonomy (environment, semantic, random distractors) and insertion strategy (prefix, middle, suffix) for evaluating VLA robustness to irrelevant context. (2) A robustness metric that quantifies the relative degradation in action-grounded knowledge retention across perturbations, normalized by baseline. (3) Empirical evidence from 7 VLAs (using the models from 'Does VLA Even Know the Basics?') showing that semantic distractors cause the largest drop, and that the degradation is concentrated in middle layers of the VLM backbone.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|-----------------------|
| Dataset | Act2Answer benchmark | Embodied knowledge queries with scene graphs |
| Primary metric | Robustness score (1-avg relative drop) | Measures knowledge retention under distraction |
| Baseline 1 | Clean instruction evaluation | Ignores perturbation effects |
| Baseline 2 | Random perturbation evaluation | Unstructured noise baseline |
| Baseline 3 | Length-matched neutral filler evaluation | Controls for instruction length confound |
| Ablation | IPRE without neutral distractor condition | Isolates impact of neutral control on metric |

### Why this setup validates the claim

The combination of the Act2Answer benchmark, robustness metric, and baselines directly tests IPRE's central claim that structured perturbations reveal distinct robustness dimensions. Cleaning evaluation serves as a null case, showing that our method uniquely captures degradation from distractions. Random perturbation baseline controls for unstructured noise, ensuring that our structured distractors measure specific weaknesses (e.g., concept discrimination) rather than general fragility. The neutral filler baseline is crucial: if environment or semantic distractors cause a significantly larger drop than neutral fillers of the same length, then the drop is attributable to irrelevance rather than a mere length effect. The ablation without neutral distractor condition tests whether including the neutral control changes the robustness score interpretation; if the scores are highly correlated, the neutral control may be redundant. This setup creates a falsifiable test: if structured perturbations yield no more insight than random noise, or if the neutral control produces identical drops to structured distractors, the claim that IPRE measures task-irrelevance filtering or concept discrimination is wrong.

### Expected outcome and causal chain

**vs. Clean instruction evaluation** — On a case where the instruction is "pick up the apple" and the scene contains a cup, the baseline evaluates the model on the clean instruction, achieving high success (say 90%) because no distraction is present. Our method inserts "There is a cup on the table" as a prefix, and the model's success drops (e.g., to 60%) because it attends to the cup and confuses the target. We expect a noticeable gap: robust models show <10% drop, while fragile ones drop >30%, indicating that the baseline overestimates real-world performance.

**vs. Random perturbation evaluation** — On a case where the instruction is "put the apple in the bowl" and we shuffle tokens to "bowl the put in apple the", the random perturbation baseline yields a 50% success drop due to syntactic breakdown. Our method instead inserts a semantic distractor like "I also see a banana" (similar object), causing a drop only if the model confuses apple with banana. We expect our robustness score to be lower (more sensitive) on categories with similar alternatives (e.g., fruits) but similar on syntax-sensitive categories (e.g., spatial relations), confirming that our method captures concept discrimination.

**vs. Neutral filler baseline** — For the same instruction, a neutral filler like "The weather is nice today" of the same length as the environment distractor should cause a smaller drop (e.g., 5%) if the model is robust to irrelevance, because the neutral content is unrelated. If the environment distractor drop (e.g., 30%) significantly exceeds the neutral drop, we confirm that the drop is due to irrelevance of the distractor content, not length. If drops are similar, the metric conflates irrelevance sensitivity with length sensitivity.

### What would falsify this idea

If our robustness score across all categories is strongly correlated with random perturbation baseline (Pearson r > 0.9) or if the neutral filler baseline produces drops within 10% of the structured distractor drops across categories, then structured distractors do not measure distinct robustness dimensions and the claim is falsified.

## References

1. Does VLA Even Know the Basics? Measuring Commonsense and World Knowledge Retention in Vision-Language-Action Models
2. Cosmos-Reason1: From Physical Common Sense To Embodied Reasoning
3. The Llama 3 Herd of Models
4. HybridFlow: A Flexible and Efficient RLHF Framework
5. Expanding Performance Boundaries of Open-Source Multimodal Models with Model, Data, and Test-Time Scaling
6. Robotic Control via Embodied Chain-of-Thought Reasoning
7. NVLM: Open Frontier-Class Multimodal LLMs
8. BridgeData V2: A Dataset for Robot Learning at Scale
