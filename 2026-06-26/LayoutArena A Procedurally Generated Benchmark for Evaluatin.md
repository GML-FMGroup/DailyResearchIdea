# LayoutArena: A Procedurally Generated Benchmark for Evaluating Web Agent Generalization Across Website Designs

## Motivation

Existing web agent benchmarks such as Running the Gauntlet evaluate performance on a small set of fixed website designs, failing to isolate how layout variation—common in real-world deployments—affects agent success. Because they test on either live websites with uncontrolled variability (WebVoyager) or static replicas (REAL), they cannot systematically measure generalization across layout structures. This limitation prevents the community from understanding why agents fail when encountering unfamiliar website designs.

## Key Insight

Task semantics and visual layout are structurally independent: by modeling each task as an abstract action sequence over a canonical element ontology, and procedurally generating distinct layouts that preserve solvability, we create a benchmark where layout becomes a controlled independent variable for causal analysis of agent robustness.

## Method

```python
# LayoutArena: Procedural benchmark generator
# Load-bearing assumption: Each required element type appears exactly once per variant, ensuring deterministic mapping works.

def generate_benchmark():
    # (A) What it is: LayoutArena is a procedural benchmark generator that takes a task ontology (list of tasks, each with steps defined via canonical element types) and produces multiple layout variants per task. Input: task ontology. Output: dataset of (task_id, layout_variant_id, HTML+CSS template, action_map) tuples.
    
    # (B) How it works:
    task_ontology = load_tasks()  # list of dicts: {id, steps: [{action, element_type, state_change}]}
    variants_per_task = 5
    seed = 42
    # Explicit grammar rules (pseudo-code):
    # layout_topology_grammar = {
    #   'page': ['header', 'main', 'footer'],
    #   'main': ['section', 'form', 'nav'],
    #   'form': ['input_group', 'submit_button'],
    #   'input_group': ['label', 'input']
    # }
    # CSS parameter distributions:
    # color_theme: uniform over ( 'light', 'dark', 'colorful' )
    # spacing: uniform from ['compact','normal','comfortable'] mapping to CSS padding/margin values
    # font: uniform over ['sans-serif', 'serif', 'monospace']
    # DOM complexity: num_elements ~ Uniform(50,200), max_depth ~ Uniform(3,8)
    for task in task_ontology:
        for variant in range(variants_per_task):
            # 1. Sample layout topology from probabilistic grammar (e.g., page tree with sections)
            layout_graph = sample_layout_topology(seed + task.id * 100 + variant)
            # 2. Sample CSS style (color theme, spacing, font) from controlled distributions
            css_params = sample_css(seed + variant + task.id)  # returns dict {color_theme, spacing, font}
            # 3. Sample DOM complexity (total elements, max depth) from uniform range [50, 200] depth [3, 8]
            dom_params = {'num_elements': random.randint(50,200), 'max_depth': random.randint(3,8)}
            # 4. Render template using Jinja2-like engine that fills layout graph with CSS and DOM structure
            page_html, element_mapping = render(layout_graph, css_params, dom_params)
            # 5. Ensure each required element type appears exactly once:
            for step in task['steps']:
                assert element_mapping[step['element_type']] has exactly 1 element, else reject and resample
            # 6. Map task steps to concrete elements: for each step's element_type, select the unique element from mapping
            action_map = {}
            for step_idx, step in enumerate(task['steps']):
                # unique element per type ensures deterministic mapping
                action_map[step_idx] = element_mapping[step['element_type']][0]
            # Validation script: check each required element is visible and clickable
            # pseudo-code:
            # for each step in task:
            #   el = action_map[step_idx]
            #   if not (el.is_visible and el.is_interactable):
            #       invalid = True; break
            # if invalid: skip and resample variant
            yield (task['id'], variant, page_html, action_map)

# Hyperparameters: variants_per_task=5, seed=42, DOM num_elements range [50,200], max_depth [3,8]
```

### (C) Why this design
We chose to separate task semantics (ontology) from layout generation via a grammar-based sampler rather than manually designing layouts or using an LLM to generate variations, because manual design is not scalable and LLM output may violate solvability constraints. Accepting the cost that grammar-defined layouts may look less organic than LLM-generated ones. We fixed the action mapping to be deterministic (based on unique element type) rather than learned, because learned mappings could introduce confounding errors; the trade-off is that the mapping may not always pick the most natural element (e.g., an old submit button vs. a prominent one), but this is a price for reproducibility. We chose to sample DOM complexity from a uniform range instead of fitting real websites, because controlled ranges allow precise manipulation, though it reduces ecological validity. These three decisions—grammar-based generation, deterministic mapping, uniform complexity sampling—ensure that layout variation is the sole independent variable across variants, at the cost of narrowing the universe of designs to those that fit our grammar.

### (D) Why it measures what we claim
The computational quantity *layout variant* (each distinct HTML/CSS produced from the same task ontology) measures *generalization to layout changes* because we hold the task semantics—the required action sequence and its mapping to canonical element types—constant across variants; this assumption fails if a layout variant makes an element functionally inaccessible (e.g., hidden by CSS or removed by the grammar), in which case a performance drop reflects task difficulty rather than layout generalization. To prevent this, we validate every variant with a script that checks that each required element type exists and is clickable/visible, and we reject any variant that fails. The deterministic mapping ensures that the action sequence stays the same across layouts, so differences in success rate can be attributed to layout differences. The sampling of CSS and DOM complexity introduces controlled perceptual and structural variation; these quantities measure the effect of perceptual load and structural complexity on agent robustness. The assumption that CSS changes do not change task semantics is valid as long as elements remain visible and interactable—verified by the validation script. Failure mode: if an agent relies on pixel-perfect layout cues (e.g., position of a button), the variant may drop performance even though the task is solvable, which is exactly what we want to measure: layout generalization.

## Contribution

(1) LayoutArena, a procedural benchmark generator that produces multiple website layout variants per task while preserving task solvability and semantics. (2) A new benchmark dataset of 100 tasks with 5 layout variants each (500 instances) spanning e-commerce, booking, and information retrieval domains. (3) The design principle that decoupling task semantics from visual layout through a canonical element ontology enables controlled study of agent robustness to layout changes.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|------------------------|
| Dataset | LayoutArena (generated from ontology) | Tests layout generalization directly. |
| Primary metric | Task success rate | Measures correct completion end-to-end. |
| Baseline 1 | WebVoyager on LayoutArena | State-of-the-art web agent baseline. |
| Baseline 2 | WebVoyager on fixed variant | Only one layout per task. |
| Baseline 3 | WebVoyager on random variants | No grammar control; random HTML/CSS. |
| Baseline 4 | WebVoyager on LLM-generated layouts | Compare to grammar vs. LLM generalization. |
| Ablation-of-ours | LayoutArena without validation | Allows invalid variants to slip in. |

### Why this setup validates the claim
This combination creates a falsifiable test of the central claim that LayoutArena measures layout generalization. Comparing WebVoyager on LayoutArena (with multiple variants) versus on fixed layouts isolates the effect of layout variation: if performance drops on LayoutArena relative to fixed, variation causes difficulty. The random-generation baseline tests whether our grammar-based control is necessary—if random variants cause similar or worse drops, then grammar adds value. The LLM-generated baseline tests whether grammar-based layouts provide better solvability guarantees and control, highlighting the novelty of our approach. The ablation (no validation) checks whether the validation step is critical to ensure tasks remain solvable; without it, variance may be inflated by broken tasks. The primary metric, task success rate, directly captures whether the agent completes the action sequence, which is the target outcome. This design ensures that any observed differences can be attributed to layout variation or its control, not to confounding factors like task difficulty or metric noise.

### Expected outcome and causal chain

**vs. WebVoyager on fixed variant** — On a task where the agent memorizes button coordinates from a single layout, fixed-layout baseline succeeds by exploiting position cues. Our LayoutArena presents 5 layout variants per task, each with different positions, colors, and element sizes. The agent relying on coordinates fails because buttons move; our method forces reliance on semantic understanding (e.g., find 'submit' by type). We expect a noticeable gap: WebVoyager on fixed layouts achieves ~80% success, while on LayoutArena it drops to ~60% on the same tasks, but the gap concentrates on tasks where visual layout varies most (e.g., form-filling vs. link-clicking).

**vs. WebVoyager on random variants** — On a task where a randomly generated layout places the submit button behind an opaque div, the random-baseline agent cannot interact because the element is invisible. Our LayoutArena's grammar ensures all required element types are visible and interactable (validated). The agent performs similarly on both benchmarks for simple tasks, but on complex tasks with many elements, the random baseline may have invalid hidden elements, causing spuriously low success. We expect LayoutArena to show higher average success than random layout variants (e.g., 60% vs. 45%) because validation removes broken instances, isolating layout variation as the independent variable.

**vs. WebVoyager on LLM-generated layouts** — On a task where an LLM-generated layout inadvertently creates a non-solvable variant (e.g., missing a required element), the LLM baseline may fail due to unsolvability, not layout variation. Our grammar guarantees solvability. We expect LayoutArena to achieve higher success than LLM-generated layouts (e.g., 60% vs. 50%) due to better task solvability, even though LLM layouts may look more organic.

**vs. LayoutArena without validation** — On a variant where a CSS rule disables clicks on a required button, the no-validation baseline includes it as valid, causing the agent to fail even though the task is logically the same. Our validation step rejects such variants, ensuring all variants are solvable. Thus, performance on the ablation is lower due to unsolvable tasks. We expect the no-validation ablation to have a lower success rate (e.g., 50% vs. 60%) and higher variance because a fraction of variants are broken, not because of layout difficulty. This confirms that our validation is necessary to measure layout generalization rather than task solvability.

### What would falsify this idea
If WebVoyager on LayoutArena performs identically to WebVoyager on fixed layouts (~80%), then layout variation does not actually challenge the agent, meaning our claim that variations measure generalization is false. Also, if the ablation without validation shows no drop in success (i.e., all variants are inherently solvable), then the validation step is unnecessary and not a key component.

## References

1. Running the Gauntlet: Re-evaluating the Capabilities of Agents Beyond Familiar Environments
2. WebVoyager: Building an End-to-End Web Agent with Large Multimodal Models
3. REAL: Benchmarking Autonomous Agents on Deterministic Simulations of Real Websites
4. GPT-4V in Wonderland: Large Multimodal Models for Zero-Shot Smartphone GUI Navigation
5. The Dawn of GUI Agent: A Preliminary Case Study with Claude 3.5 Computer Use
6. Is Your LLM Secretly a World Model of the Internet? Model-Based Planning for Web Agents
7. TheAgentCompany: Benchmarking LLM Agents on Consequential Real World Tasks
8. AgentOccam: A Simple Yet Strong Baseline for LLM-Based Web Agents
