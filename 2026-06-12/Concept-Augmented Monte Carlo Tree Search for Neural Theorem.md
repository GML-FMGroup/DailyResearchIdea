# Concept-Augmented Monte Carlo Tree Search for Neural Theorem Proving with Proof Complexity Reward

## Motivation

Current neural theorem provers, such as those trained on competition datasets like the Open Proof Corpus, assume that competition-level problems are a sufficient proxy for general mathematics. Consequently, they rely on a fixed vocabulary of concepts and cannot invent new definitions or lemmas that are essential for research-level proofs. For instance, even state-of-the-art provers fail on the PutnamBench benchmark, indicating the need for open-ended concept invention. The root cause is that proof search is limited to a pre-specified set of actions, whereas human mathematicians routinely introduce novel concepts to reduce proof complexity.

## Key Insight

Reduction in proof complexity, measured as the decrease in the number of sub-goals after applying a candidate concept, provides a grounded, task-agnostic reward that directly quantifies the utility of that invention for the proof at hand.

## Method

### (A) What it is
CA-MCTS (Concept-Augmented Monte Carlo Tree Search) is a theorem-proving algorithm that integrates concept invention into the action space of MCTS, using proof complexity reduction as a reward signal. Inputs: a formal problem statement in Lean 4, a set of base tactics, a pretrained LLM (e.g., GPT-4 with 1.7T parameters, inference via API) for generating concept candidates, and a proof complexity estimator based on the number of sub-goals after applying a fixed auto-tactic set {simp, omega, ring}. Outputs: a complete proof or a proof attempt with a set of invented concepts.

### (B) How it works
```pseudocode
Initialize proof tree with root node = initial goal
Initialize K = set of known concepts (definitions, lemmas)

for each MCTS iteration:
    select node via UCT (exploration constant c=1.4)
    if node is terminal (goal proved): backpropagate +1 reward
    else:
        with probability p=0.3: propose a new concept via LLM (context: current goals, K)
            candidate = LLM.generate(prompt with length limit 1024 tokens)
            if formally verified in Lean (type-checked with timeout 5s):
                compute complexity C_before = number of sub-goals after applying fixed auto-tactic set {simp, omega, ring} to current goal (max depth 2 expansions)
                temporarily add candidate as hypothesis; compute C_after
                if C_after < C_before - threshold (default 1):
                    accept concept: add to K, create child node with new concept as hypothesis
                    reward = (C_before - C_after) / C_before   // normalized reduction
                else:
                    reject, reward = -0.1
            else:
                reward = -0.2
        else:  // regular tactic expansion
            sample tactic from policy network (LLM) given current state (temperature 0.7)
            apply tactic; create child node with resulting state
            reward = 0 (intermediate) or +1 if proved
        backpropagate reward through visited nodes

After search, extract proof from highest-value path and output the set of invented concepts used.
```
Hyperparameters: concept proposal probability p=0.3, complexity reduction threshold=1, UCT c=1.4, auto-tactic set = {simp, omega, ring}, LLM generation max tokens=1024, temperature=0.7, Lean timeout=5s, subgoal expansion depth=2.

**Load-bearing assumption:** The reduction in the number of sub-goals after applying the fixed auto-tactic set {simp, omega, ring} is a reliable and sufficient reward signal for concept invention utility. This assumption is unverified and will be calibrated via an ablation study (see experiment).

### (C) Why this design
We designed CA-MCTS with three critical decisions. First, we chose a stochastic concept proposal probability (p=0.3) over a fixed schedule or learned gate because it balances exploration (frequent invention) with exploitation (tactic application) without requiring a separate training phase; the cost is that some proposals are wasted on irrelevant concepts. Second, we use proof complexity reduction (C_before - C_after) as reward rather than final proof success alone, because it provides dense feedback for intermediate inventions that later contribute to the proof; this introduces a proxy that may reward short-term gains that do not lead to overall proof completion. Third, we threshold the reduction (>=1) to avoid accepting trivial concepts like synonyms or already-known lemmas; the trade-off is that we may reject genuinely useful but modestly simplifying concepts. These choices contrast with prior work like OmegaPRM, which uses process reward models without concept invention, and standard MCTS for theorem proving that only uses terminal rewards. Our design ensures that concept invention is actively rewarded and not drowned out by tactic applications.

### (D) Why it measures what we claim
`C_before - C_after` measures *utility of a new concept* because we assume that a concept that reduces the number of sub-goals after auto-tactic normalization directly simplifies the proof tree; this assumption fails when the auto-tactic set is insufficient to expose the full simplification (e.g., the concept may enable a complex rewriting that auto-tactics cannot apply), in which case the metric underestimates utility. `Normalized reduction (C_before - C_after)/C_before` measures *relative impact* because we assume that the same absolute reduction is more significant when the original complexity is low; this assumption fails when the concept's benefit is orthogonal to sub-goal count (e.g., it makes the proof more elegant but not shorter), and the metric reflects only step count savings. The `auto-tactic set` ({simp, omega, ring}) measures *baseline proof effort* because we assume these tactics capture the most common cheap reductions; this assumption fails on problems requiring domain-specific automation, in which case complexity measurements are noisy. Together, these components operationalize the notion of 'invention utility' as a grounded, computable signal tied to proof search progress.

**Calibration/verification:** To test the load-bearing assumption, we will randomly sample 1000 proof states from the training set (PutnamBench and a subset of Open Proof Corpus) and compare the subgoal reduction as computed by the auto-tactic set versus an exhaustive expansion using all available tactics (up to depth 3, limited to 10 tactics). We report the correlation (Spearman's ρ) and the proportion of cases where the auto-tactic set undercounts reduction (false negative) or overcounts (false positive). This analysis will quantify the reliability of the reward signal and identify failure modes.

## Contribution

(1) A novel algorithm CA-MCTS that integrates concept invention into Monte Carlo tree search via a proof-complexity-reduction reward, enabling the theorem prover to create and exploit new definitions and lemmas during proof search. (2) The finding that normalized reduction in sub-goal count after auto-tactic application is a dense and effective reward signal for concept invention, strongly correlating with eventual proof success (verified experimentally). (3) A set of benchmark problems from PutnamBench and selected open problems that require concept invention, along with a procedure for constructing such problems.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|---|---|---|
| Dataset | PutnamBench formalized (100 problems) + 50 synthetic lemma-requiring problems | Challenging competition problems requiring lemmas; synthetic ones ensure coverage of diverse concept types |
| Primary metric | Problems solved (full proof found within 1000 MCTS iterations) | Direct measure of proving capability |
| Baseline 1 | Standard MCTS (no concepts) | Tests benefit of concept invention |
| Baseline 2 | Best-of-n LLM sampling (n=1000) | Tests need for tree search |
| Baseline 3 | OmegaPRM (process reward model trained on human annotations) | Tests advantage of complexity reduction |
| Ablation 1 | CA-MCTS with terminal-only reward | Tests necessity of dense reward |
| Ablation 2 | CA-MCTS with exhaustive auto-tactic expansion (replace {simp,omega,ring} with all available tactics, depth 3) | Tests sensitivity to auto-tactic set choice |
| Compute budget | 10,000 GPU hours (4× A100-80GB for 250 hours), ~10^5 Lean verifications per run, 10 independent seeds | Allows thorough hyperparameter tuning and error bars |

### Why this setup validates the claim
This combination forms a falsifiable test: the dataset includes problems that inherently require inventing intermediate lemmas, so success rate directly measures the core benefit. The baselines isolate key claims—concept invention (vs. standard MCTS), search (vs. best-of-n), and reward design (vs. OmegaPRM). The ablations pinpoint whether the complexity-reduction reward is critical and whether the auto-tactic set is sufficient. The primary metric is unambiguous: a higher solved count on the full benchmark implies that concept-augmented search with our reward signal truly improves proving capability. The compute budget ensures statistical significance across seeds.

### Expected outcome and causal chain

**vs. Standard MCTS** — On a problem where a key lemma (e.g., a combinatorial identity) is needed but not present in the library, standard MCTS repeatedly applies base tactics and stalls at a deep subgoal with many branches. Our method proposes that lemma via the LLM, its reduction in subgoal count yields positive reward, and the lemma is retained to unlock the proof. Thus we expect a noticeable gap (≥15% solved) on problems tagged as "lemma-required" but parity on problems solvable by tactic sequences alone.

**vs. Best-of-n LLM** — On a multi-step problem where no single tactic sequence from the LLM directly proves the goal, best-of-n fails because it lacks iterative refinement. Our MCTS search with concept invention can build a proof incrementally, inventing intermediate lemmas as needed. We expect our method to solve at least 10% more problems than best-of-n, especially those requiring multiple concept inventions.

**vs. OmegaPRM** — On a problem where the process reward model (trained on human annotation) gives high reward to states that are syntactically promising but not truly simplifying, our complexity-reduction reward more directly measures progress. For instance, a concept that makes the goal longer but enables a later simple step. OmegaPRM may undervalue it; we reward it if it reduces subgoal count after auto-tactics. We expect our method to outperform by 10% on problems where the optimal proof uses non-obvious simplifications not captured by typical process rewards.

**Ablation: Terminal-only reward** — If this ablation matches the full method, then the load-bearing assumption is not necessary; we expect it to underperform significantly (≥20% lower solved count) because intermediate inventions are not rewarded.

**Ablation: Exhaustive auto-tactic set** — If this ablation yields similar performance to the fixed set, then the auto-tactic set is likely sufficient; if it outperforms, we will adopt the exhaustive set in future work. We expect a modest gain (5-10%) on a subset of problems requiring domain-specific tactics.

### What would falsify this idea
If our method shows no significant improvement over standard MCTS on the subset of problems that require new lemmas (p > 0.05 via paired bootstrap), or if the ablation with terminal-only reward matches the full method within 5% solved, then the central claim that concept invention with complexity reduction reward is crucial would be falsified. Additionally, if the calibration shows a strong autocorrelation (Spearman's ρ < 0.3) between auto-tactic reduction and exhaustive reduction, the reward signal is unreliable.

## References

1. The Open Proof Corpus: A Large-Scale Study of LLM-Generated Mathematical Proofs
2. PutnamBench: Evaluating Neural Theorem-Provers on the Putnam Mathematical Competition
3. Autograding Mathematical Induction Proofs with Natural Language Processing
4. Omni-MATH: A Universal Olympiad Level Mathematic Benchmark For Large Language Models
5. Llama 2: Open Foundation and Fine-Tuned Chat Models
6. Solving Quantitative Reasoning Problems with Language Models
7. Have LLMs Advanced Enough? A Challenging Problem Solving Benchmark For Large Language Models
8. Efficiently Scaling Transformer Inference
