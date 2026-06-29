# SpatialParseMLLM: Dynamic Instruction Parsing for Compositional Spatial Reasoning

## Motivation

Existing instruction-driven spatial reasoning models (e.g., ShutterMuse, InstructBLIP) rely on fixed templates or structured annotations, limiting their ability to handle arbitrary free-form spatial language. The root cause is treating spatial instructions as classification over predefined patterns rather than decomposing them into compositional relational constraints. This structural inflexibility prevents generalization to novel spatial expressions.

## Key Insight

Spatial language is inherently compositional: expressions like 'between X and Y' decompose into a conjunction of atomic spatial relations; by learning to identify these primitives and combine them into a constraint graph, we can handle arbitrarily complex instructions without predefined templates.

## Method

**What it is**: SpatialParseMLLM is a two-stage framework: a Parser Module (an LLM fine-tuned to output a spatial constraint graph) and a Generator Module (an MLLM conditioned on the constraint graph to produce or edit images). Input: free-form natural language instruction and optional visual context. Output: an image with objects placed according to the constraints.

**How it works**:
```python
# Training Parser
def train_parser():
    # Synthetic dataset: generate diverse spatial instructions paired with constraint graphs
    # Constraint graph: set of tuples (relation, obj1, obj2, ...)
    # e.g., left(apple, banana), between(apple, banana, orange)
    for instruction, constraints in synthetic_data:
        # Fine-tune a pretrained LLM (e.g., Llama-2) with instruction: "Parse spatial instruction: [instruction] -> [list of constraints]"
        # Use cross-entropy loss on constraint tokens
        loss = cross_entropy(llm_output, target_constraints)
        update(llm, loss)

# Inference
def inference(instruction, image_context=None):
    # Step 1: Parse
    parsed = llm_parse(instruction)  # outputs e.g., "left(apple,banana); on(apple,plate)"
    constraints = parse_to_list(parsed)  # handle formatting
    # Consistency check: detect conflicts (e.g., opposite relations) and warn
    if not consistent(constraints):
        # Fallback: request clarification or relax constraints
        pass
    # Step 2: Generate
    # Condition MLLM (e.g., LLaVA) on both image_context and constraints
    # Use a prompt template: "Place objects according to constraints: [constraints]. Generate an image."
    output_image = generator(image_context, constraints)
    return output_image
```

**Why this design**:
We chose to use a separate parser LLM (fine-tuned from a general LLM) rather than an end-to-end MLLM parsing because the parser can be trained on purely textual data, leveraging existing LLM capabilities, while the generator MLLM can focus on visual grounding. This separation (trade-off 1) reduces the need for large-scale paired image-text-constraint data, accepting the cost that the parser may miss visual context that could disambiguate spatial terms (e.g., 'the book to the left' depending on viewpoint). We chose a synthetic data generation approach (trade-off 2) to cover a wide range of compositional patterns, but this may introduce distribution shift from natural language; we mitigated by including paraphrasing and noise. We chose a constraint graph representation over a linearized form (trade-off 3) because a graph captures relations between objects explicitly, enabling consistency checks that improve generation robustness; the cost is increased complexity in decoding (graph-to-text for prompting the MLLM). We used a fixed set of atomic relations (left, right, above, below, between, on, inside) based on spatial language literature; this limits coverage of metaphorical or vague terms but ensures tractability.

**Why it measures what we claim**:
The computational quantity `parsed` (list of constraint tuples) measures `compositional understanding` of the instruction because we assume that any free-form spatial expression can be decomposed into a conjunction of atomic relations from our predefined set; this assumption fails when the instruction uses non-atomic spatial language (e.g., 'near' without specifying direction), in which case the parser may approximate with the most likely relation based on training data distribution. The `consistency check` measures `feasibility` of the instruction because we assume that a valid placement must satisfy all relations simultaneously (e.g., no contradictory pairs); this assumption fails when constraints conflict (e.g., left(A,B) and right(A,B) simultaneously), in which case the generation may produce an impossible arrangement, reflecting a need for conflict resolution. The `conditioned generation` in the MLLM measures `spatial grounding` because we assume that by prepending constraints in the prompt, the model will infer correct spatial positions; this assumption fails when the MLLM has weak compositional reasoning, in which case the generation may ignore or misinterpret constraints, and we would need to adjust the conditioning mechanism (e.g., using specialized attention masks).

## Contribution

(1) A dynamic instruction parser that converts free-form spatial language into a compositional constraint graph using a fine-tuned LLM. (2) A synthetic data generation pipeline for diverse spatial instructions covering a wide range of compositional patterns. (3) Integration with an MLLM that conditions object placement on parsed constraints, enabling handling of arbitrary spatial instructions without fixed templates.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|----------------------|
| Dataset | CLEVR-Ref+ (augmented with diverse spatial constraints) | Tests compositional spatial understanding. |
| Primary metric | Constraint satisfaction accuracy (fraction of images with all spatial relations correct) | Directly measures spatial grounding fidelity. |
| Baseline 1 | Direct MLLM (e.g., LLaVA + Stable Diffusion) | End-to-end generation without explicit parsing. |
| Baseline 2 | Layout baseline (LLM extracts boxes → GLIGEN renders) | Compares to layout-driven generation. |
| Ablation (Ours) | Ours with zero-shot parser (no fine-tuning) | Assesses parser fine-tuning necessity. |

### Why this setup validates the claim
This combination of dataset, baselines, and metric forms a falsifiable test of the central claim that separating parsing and generation with a fine-tuned parser improves compositional spatial reasoning. The dataset (CLEVR-Ref+) contains instructions with multiple atomic relations (e.g., left, between), enabling targeted evaluation of compositional understanding. The primary metric (constraint satisfaction accuracy) directly quantifies whether the generated image respects all spatial constraints, thus measuring the core capability. Baseline 1 (direct MLLM) tests whether an end-to-end model can implicitly learn spatial composition without explicit decomposition—if our method outperforms, it indicates that explicit parsing adds value. Baseline 2 (layout baseline) tests whether a conventional layout-based pipeline (that extracts bounding boxes) can achieve similar results; if our method wins, it suggests that constraint graphs are a more effective intermediate representation. The ablation (zero-shot parser) isolates the benefit of fine-tuning the parser on synthetic data—if fine-tuning is crucial, our full method should significantly outperform this ablation. Together, these comparisons allow us to attribute any gains to specific design choices: parser training, constraint graph representation, or consistency checking. The metric is sensitive enough to detect even small improvements in constraint satisfaction, ensuring we can observe predicted effects.

### Expected outcome and causal chain

**vs. Direct MLLM (LLaVA + SD)** — On an instruction like "place the apple to the left of the banana and above the plate", the baseline may generate an image where the apple is left of the banana but not above the plate, or misinterprets the dual constraint because its attention mechanism struggles to compose multiple spatial relations simultaneously. Our method first parses the instruction into explicit constraints: left(apple, banana), above(apple, plate). The generator then receives these constraints as a structured prompt, reducing ambiguity. We expect a noticeable gap (e.g., ~20% higher constraint satisfaction) on instructions with ≥2 constraints, but parity on single-constraint instructions.

**vs. Layout baseline (LLM→boxes→GLIGEN)** — On a case involving a vague term like "between" (e.g., "place the apple between the banana and the orange"), a typical LLM used for bounding box extraction may output boxes that approximate the relation but fail to enforce exact equidistance or alignment, causing visual inconsistency. Our constraint graph explicitly captures the ternary relation (between(apple, banana, orange)), which the generator can interpret via learned spatial priors. We expect our method to achieve higher accuracy on ternary relations, with an improvement of ~15% in satisfaction for such cases, while both methods perform similarly on binary relations like "left".

**vs. Ablation: zero-shot parser** — On instructions that involve compositional nesting (e.g., "the apple to the left of the banana that is on the plate"), the zero-shot parser (pretrained LLM without fine-tuning) may misinterpret dependencies, e.g., outputting left(apple, banana) and on(banana, plate) correctly but not on(apple, plate)? Actually it might omit the indirect relation. The fine-tuned parser from synthetic data learns to decompose such nested structures reliably. We expect the full method to surpass the ablation by ~10% overall, with larger gaps on instructions requiring hierarchical decomposition.

### What would falsify this idea
If the performance gain over the layout baseline is uniform across all constraint types (including single-relation instructions) rather than concentrated on multi-constraint or ternary cases, then the central claim of improved compositional reasoning is invalid—instead, the improvement may stem from better visual quality or prompt engineering rather than structured parsing.

## References

1. ShutterMuse: Capture-Time Photography Guidance with MLLMs
2. Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond
3. Image as a Foreign Language: BEiT Pretraining for All Vision and Vision-Language Tasks
4. OFASys: A Multi-Modal Multi-Task Learning System for Building Generalist Models
5. PaLI: A Jointly-Scaled Multilingual Language-Image Model
6. LAION-5B: An open large-scale dataset for training next generation image-text models
