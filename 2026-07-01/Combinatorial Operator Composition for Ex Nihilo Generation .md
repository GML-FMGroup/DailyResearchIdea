# Combinatorial Operator Composition for Ex Nihilo Generation of Diverse Mathematical Reasoning Strategies

## Motivation

Existing methods for generating diverse reasoning strategies, such as the human-calibrated LLM judge from 'Are We Measuring Strategy or Phrasing?', require either pre-existing solution sets or expensive human annotations, limiting scalability and cost-effectiveness. The root cause is a reliance on extracting diversity from a given set of alternatives rather than generating it from scratch, which our work overcomes.

## Key Insight

The space of reasoning strategies is spanned by a finite set of abstract operators; random composition of these operators covers distinct regions of that space without requiring any seed solutions.

## Method

### Combinatorial Operator Composition for Ex Nihilo Generation of Diverse Mathematical Reasoning Strategies

(A) **What it is**: FARA (Fully Abstract Reasoning Assembler) takes a math problem text as input and outputs a set of N=10 diverse solution strategies (each a sequence of operator applications) via random composition of a minimal set of five abstract reasoning operators, without any pre-existing solution set.

(B) **How it works**:
```pseudocode
Input: math problem P
Output: set S of N=10 strategies

Define operators O = {Algebraic_Manipulation, Logical_Deduction, Case_Analysis, Heuristic_Guess, Verification}

# Calibration step (performed once before generation):
# On 100 randomly selected MATH problems, generate one step per operator using GPT-4.
# Three human annotators classify each step into operator categories; compute pairwise Cohen's kappa (target ≥0.8).
# If confusion between any two operators >20%, merge or redefine operators.

For i = 1 to N:
    L = random integer uniformly from [3, 6]  # strategy length hyperparameter
    seq = []
    state = P
    for t in 1..L:
        op = random choice from O with uniform probability
        # Prompt template: "Given the problem: {P} and previous reasoning steps: {state}, apply the operator '{op}' to produce the next reasoning step. Output only the step."
        step = LLM_generate(prompt, model='gpt-4', temperature=0.7, max_tokens=200)
        seq.append((op, step))
        state = state + step  # append step to context
    add seq to S
Return S
```

(C) **Why this design**: We chose 5 operators to balance coverage and coherence: fewer risk insufficient diversity (e.g., two operators often produce overly similar strategies), while more increase computational cost and prompt redundancy. Random sequence length [3,6] was chosen because shorter sequences fail to solve complex problems, while longer ones amplify noise and cost. Allowing operator repetition (e.g., two Algebraic_Manipulations in a row) reflects that some problems benefit from repeated same-type steps, though this can reduce diversity; we accept this cost to maintain flexibility. We avoid any selection or refinement (like beam search) to keep the method lightweight and fully generative, at the expense of occasional incoherent strategies. The state is updated by appending steps verbatim, which is simple but may cause catastrophic forgetting; we trade off memory efficiency for consistency by keeping the full context. Prior to generation, we calibrate operator definitions via a human annotation study (see calibration details) to ensure minimal overlap.

(D) **Why it measures what we claim**: The operator sequence (e.g., [Case_Analysis, Algebraic_Manipulation]) measures strategic approach diversity because different operator orders imply distinct reasoning structures under the assumption that operators are disjoint in the reasoning space they span; this assumption fails when two operators produce similar reasoning patterns (e.g., Algebraic_Manipulation and Logical_Deduction might both reduce equations), in which case diversity measured by operator sequence overestimates true strategic variation. The actual text of each step measures surface-level content, not approach. To mitigate this, we design operators based on a cognitive taxonomy ensuring minimal overlap: Algebraic_Manipulation focuses on equation transformations, Logical_Deduction on rule-based inference, Case_Analysis on branching, Heuristic_Guess on approximate estimation, and Verification on checking consistency. This assumption is explicit: operators are chosen to be functionally orthogonal; under that, sequence variation directly corresponds to strategic diversity. The calibration study validates that distinct operator sequences indeed correspond to distinct reasoning strategies; without such validation, the metric would be unreliable.

## Contribution

(1) A novel framework (FARA) for generating diverse mathematical reasoning strategies from scratch via random composition of abstract reasoning operators, eliminating the need for pre-existing solution sets. (2) A demonstration that combinatorial variation of a small set of domain-agnostic operators yields a wide range of strategic approaches, supported by a design rationale for operator distinctiveness. (3) A set of prompt templates for five abstract operators that can be adapted to new domains.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|------------------------|
| Dataset | MATH dataset (500 problems sampled) | Diverse math problems requiring multi-step reasoning |
| Primary metric | LLM-judged approach diversity | Validated against human judgments in prior work |
| Baseline | Temperature sampling (T=1.0) | Standard strategy generation without diversity control |
| Baseline | Surface-level diversity filtering | Maximizes n-gram diversity, fails on approach level |
| Ablation | FARA without operator repetition | Tests if repetition boosts diversity or not |

**Compute budget**: Calibration study: ~$10 (500 GPT-4 API calls). Main experiment: ~$675 (22,500 GPT-4 calls on 500 MATH problems with N=10, average 4.5 steps per strategy).

### Why this setup validates the claim

This experimental design directly tests whether FARA produces superior approach-level diversity compared to natural baselines. The MATH dataset contains problems with multiple valid reasoning strategies, making it ideal for measuring diversity. The primary metric, an LLM judge assessing approach-level diversity, has been validated against human judgments in prior work, ensuring it captures strategic variation rather than surface phrasing. The temperature sampling baseline tests whether naive generation suffices; its expected failure would confirm the need for explicit diversity mechanisms. The surface-level filtering baseline tests whether maximizing n-gram diversity inadvertently captures approach diversity; its expected failure would highlight the gap between surface and approach levels. The ablation (no operator repetition) isolates the contribution of allowing repeated operator types, testing whether this flexibility is beneficial. Together, these components provide a falsifiable test: if FARA does not outperform both baselines on the primary metric, the claim that random operator composition yields high approach diversity is unsupported.

### Expected outcome and causal chain

**vs. Temperature sampling (T=1.0)** — On a case like "Prove that sqrt(2) is irrational", temperature sampling generates multiple proofs but often repeats the contradiction approach because the LLM follows the most common reasoning path. Our method instead forces different operator sequences (e.g., one uses Case_Analysis then Algebraic_Manipulation, another uses Heuristic_Guess then Verification), so we expect a clear gap: FARA will yield on average 3.5 distinct strategy categories per problem (out of a possible 5) versus 2.0 for temperature sampling, as judged by the LLM judge (which we validate per calibration).

**vs. Surface-level diversity filtering** — On a case where surface-level diverse solutions still share the same underlying approach (e.g., two different wordings of the same equation manipulation), the baseline selects strategies that look different but are strategically identical. Our method explicitly varies operator sequences, which correspond to different reasoning structures. Thus we expect that while surface-level filtering may achieve high n-gram diversity, FARA will have significantly higher approach diversity: e.g., FARA scores 4.0 vs. 2.5 (on a 1-5 Likert scale) on problems requiring multiple distinct reasoning steps.

**Ablation: FARA without operator repetition** — On a problem where repeating an operator (e.g., two Algebraic_Manipulations in a row) is necessary to solve in a certain strategy, the ablation may fail to generate that strategy because it cannot use the same operator twice. Our full method can, so we expect that full FARA covers a broader set of valid strategies. Specifically, full FARA generates on average 4.2 valid strategies out of 10 generated, compared to 3.1 for the no-repetition ablation. The approach diversity (measured by distinct operator sequences) for full FARA is also higher: 3.5 vs. 2.8.

### What would falsify this idea

If FARA does not significantly outperform temperature sampling on the primary metric (approach diversity) on the MATH dataset (e.g., less than 20% improvement in distinct strategies per problem), especially on problems where multiple distinct approaches exist, then the central claim that random operator composition yields diverse strategies is wrong. Also if ablation without repetition performs equally well (e.g., within 10% of full FARA on approach diversity), then repetition is unnecessary.

## References

1. Are We Measuring Strategy or Phrasing? The Gap Between Surface- and Approach-Level Diversity in LLM Math Reasoning
2. Artificial Hivemind: The Open-Ended Homogeneity of Language Models (and Beyond)
3. Reasoning Path Divergence: A New Metric and Curation Strategy to Unlock LLM Diverse Thinking
4. Post-training Large Language Models for Diverse High-Quality Responses
5. Diversity-Incentivized Exploration for Versatile Reasoning
6. GuidedSampling: Steering LLMs Towards Diverse Candidate Solutions at Inference-Time
7. Benchmarking Linguistic Diversity of Large Language Models
8. One fish, two fish, but not the whole sea: Alignment reduces language models' conceptual diversity
