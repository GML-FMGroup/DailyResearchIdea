# DynaVerif: Decentralized Adversarial Proof Generation and Dynamic Axiom Discovery for Multi-Agent Formal Verification

## Motivation

Existing multi-agent discovery systems, such as the one proposed by 'Discovering mathematical concepts through a multi-agent system', rely on a fixed formal verifier (e.g., Lean) and static axiom sets. This structural dependence prevents generalization to mathematical domains where no pre-built formal environment exists, because the system cannot adapt its verification criteria or inference rules based on collective experience. The root cause is the centralized, static verification component that does not allow the formal environment to co-evolve with agent discoveries.

## Key Insight

By decoupling verification into an adversarial role rotation among agents, the system can treat the formal environment (axioms, inference rules) as a learnable artifact that is updated through collective proof failures, thereby enabling self-adaptation to new domains without a pre-specified verifier.

## Method

**DynaVerif** is a decentralized multi-agent framework where agents rotate as provers and verifiers, dynamically updating the formal environment (axioms and inference rules) through adversarial interactions. Input: initial axiom set S₀, a set of agents A, maximum rounds T. Output: final axiom set S* and a set of verified conjectures C.

### Load-bearing assumption
Verifiers can always produce a concrete counterexample for any invalid conjecture or proof step, or a proof of invalidity via a derivation that the conjecture does not follow from axioms. For undecided outcomes (e.g., when bounded search fails), no axiom update is triggered; the outcome is recorded as undecided and the proof attempt is discarded.

### (B) How it works

```python
# DynaVerif Pseudocode
Initialize axiom set S = S₀.
Initialize empty conjecture set C.
For round t = 1 to T:  # T = 100
    # Phase 1: Role assignment
    For each agent a in A:
        If hash(a || t) mod 2 == 0: role[a] = 'prover'
        Else: role[a] = 'verifier'
    # Phase 2: Proposer step (provers only)
    For each prover p:
        # Generate conjecture using a context-free grammar (max depth 5, nonterminals: Expr, Op, Var) conditioned on past successful axioms and rejected conjectures (history = last 10 rounds). Tautologies are filtered by attempting a proof with a lightweight SAT solver (3-second timeout); if the conjecture is a tautology, skip and resample.
        Conjecture c_p = generate_conjecture(S, history, grammar_depth=5)
        # Forward search proof attempt using iterative deepening depth-first search (IDDFS) up to 100 steps, each step applying an inference rule from S.
        Proof π_p = attempt_proof(c_p, S, max_steps=100, search='IDDFS')
        Submit (c_p, π_p) to community pool.
    # Phase 3: Verification step (verifiers only)
    For each verifier v:
        For each submitted (c, π) from pool:
            if random() < 0.8: # skip some to avoid overload
                # Verification using a symbolic verifier (propositional SAT solver for magma domain) that checks whether π is a valid proof under S. Returns 'accept' if valid; 'reject_with_counterexample' if invalid, providing a minimal counterexample (assignment of variables) that falsifies a step; 'undecided' if search bound exceeded.
                Outcome out = verify(c, π, S, verifier='MiniSat', timeout=10s)
                If out == 'reject_with_counterexample':
                    # Verifier provides a counterexample model (variable assignment) that refutes c or shows π is invalid.
                    Record (c, π, counterexample).
                If out == 'undecided':
                    # Do not trigger axiom update; record as undecided for later majority voting.
                    Record (c, π, undecided).
    # Phase 4: Axiom update
    For each rejected (c, π) with counterexample cx:
        # Use an LLM (e.g., GPT-3.5) with prompt: 'Given axiom set S, conjecture c, proof π, and counterexample cx, propose a new axiom that would block the invalid step. Output as a single logical formula.' Parse the LLM output to a formula and add to S with validity score 0.
        Candidate axiom a_new = propose_axiom(cx, S, LLM_prompt, model='gpt-3.5-turbo', temperature=0.2)
        Add a_new to S with a validity score = 0.
    # Phase 5: Score adjustment
    For each axiom s in S:
        For each successful verification (accepted proof) that used s: increase score by 1.
        For each failed verification (rejected with counterexample) that used s: decrease score by 1.
        # Score decay: each round, decrease score of all axioms by 0.1 to encourage recent utility.
        If score < threshold (threshold = -5): remove axiom (pruning).
    # Phase 6: Consolidate accepted conjectures
    For each accepted (c, π), add to C if not already present.
Return S (axioms with score > -5) and C.
```
Hyperparameters: T=100, threshold=-5, grammar depth up to 5, verifier = MiniSat (SAT solver) with 10-second timeout, LLM axiom generation using GPT-3.5-turbo.

### (C) Why this design
We chose adversarial role rotation (prover/verifier swap per round) over fixed roles because it prevents agents from specializing into pure exploiters: a prover who becomes lazy knows it will later be a verifier and must maintain verification competence. This incurs a computational cost: each agent must maintain both generative and verificative capabilities, doubling training complexity. We chose decentralized verification (each verifier independently checks a subset) over centralized verification to avoid a single point of failure and to enable scaling to larger agent populations; the trade-off is that inconsistent verification outcomes may arise, which we mitigate by majority voting across verifiers (implicit in the scoring phase). We chose dynamic axiom generation via counterexample-driven LLM proposals over fixed axioms because it directly addresses the motivation's core bottleneck—lack of adaptation; the cost is that unsound axioms may be introduced, requiring a robust scoring mechanism that slowly prunes low-utility axioms. This prevents the system from overfitting to initial conjectures.

### (D) Why it measures what we claim
The verification success rate (fraction of accepted conjectures per round) measures the adaptability of the formal environment because a rising success rate implies that axiom updates are aligning the formal system with the agents' inference capabilities; this assumption fails when agents simply learn to generate conjectures that are trivially verifiable under weak axioms (e.g., tautologies), in which case the metric reflects triviality rather than adaptation. To counter this, we exclude tautologies via a pre-filter (see Phase 2). The number of valid axioms retained after pruning measures the system's ability to discover non-trivial rules because only axioms that have been used in multiple successful proofs survive; this assumption fails when the score threshold is too low, allowing spurious axioms to persist, or when the proof search consistently avoids using certain axioms, making them appear useless. The frequency of counterexample generation measures the system's ability to detect and exploit gaps in the current formal environment; this assumes that verifiers can construct concrete counterexamples, which may be impossible in undecidable or infinite domains, in which case 'undecided' outcomes dominate and the metric collapses. We mitigate this by using a bounded model checker with increasing bounds and treating undecided as no axiom update. The novelty of discovered axioms (measured by Kolmogorov complexity relative to initial axioms) measures the system's genuine expansion of the formal environment; this assumes that the axiom generation process introduces rules that are not syntactically reducible to existing ones, which fails when the LLM simply paraphrases or recombines existing axioms without adding substantive new content.

## Contribution

(1) A decentralized adversarial verification protocol (DynaVerif) that replaces static formal verifiers with dynamic role rotation and axiom updating, enabling multi-agent systems to adapt their formal environment without human intervention. (2) A counterexample-driven axiom generation mechanism that uses failed proofs to propose new inference rules, grounded in a scoring and pruning process that retains only empirically useful axioms. (3) A demonstration that the system can self-adapt to a novel domain (e.g., a simple algebraic structure not covered by initial axioms) without requiring a pre-built formal environment, validated on synthetic domains with varying axiom completeness.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|----------------------|
| Dataset | Equational laws on a magma (ETP) | Formal domain with known implications |
| Primary metric | Verification success rate per round | Measures adaptability of formal environment |
| Baseline 1 | Static Axioms (no update) | Tests need for dynamic axiom generation |
| Baseline 2 | Random Conjectures (no learning) | Tests prover adaptation from feedback |
| Baseline 3 | Centralized Verification (single verifier) | Tests decentralization advantage |
| Ablation-of-ours | Fixed Roles (no rotation) | Tests benefit of adversarial rotation |
| Additional ablation | No tautology filter | Isolates effect of tautology removal on success rate |

### Why this setup validates the claim
This combination creates a controlled test of the central claim that adversarial role rotation and counterexample-driven axiom updates enable adaptive formal environments. The ETP dataset provides a ground-truth implication graph, making verification success rate a reliable proxy for alignment between the evolving axiom set and valid reasoning. Static Axioms isolates the effect of dynamic updates; Random Conjectures isolates the effect of learning from verification feedback; Centralized Verification isolates the advantage of decentralized redundancy; and Fixed Roles isolates the effect of role rotation. The additional ablation (no tautology filter) ensures that the success rate metric is not inflated by trivial conjectures. The chosen metric directly reflects how well the system discovers useful axioms—if success rate rises under our full method but not under baselines, the claim is supported.

### Expected outcome and causal chain

**vs. Static Axioms** — On a conjecture requiring a newly discovered axiom to prove, Static Axioms lacks that axiom and fails, because its fixed set cannot adapt. Our method generates the axiom via a counterexample, enabling proof, so we expect a large gap in success rate after a few rounds (e.g., ours >0.8 vs baseline <0.2 on nontrivial conjectures).

**vs. Random Conjectures** — On a hard conjecture type, Random Conjectures never learns to produce plausible conjectures, because it ignores verification feedback. Our provers condition on past rejections to avoid similar failures, so we expect our success rate to increase over rounds while Random remains flat near 0.

**vs. Centralized Verification** — On a corner case where the central verifier has a blind spot (e.g., a subtle logical error), a single verifier consistently accepts a wrong proof. Our decentralized verifiers with majority voting catch the error, leading to fewer false acceptances; we expect our false positive rate to be near zero while centralized exceeds 10%.

**vs. Fixed Roles** — Without rotation, provers may overfit to a single style and verifiers become complacent, leading to stagnation; we expect our success rate to plateau earlier (e.g., below 0.6) compared to ours with rotation (above 0.8 after 100 rounds).

### What would falsify this idea
If our method shows no performance gain over Static Axioms on conjecture types that genuinely require novel axioms, or if the gain is uniform across all subsets (including trivial ones) rather than concentrated where the predicted failure modes occur, then the central claim is invalid.

## References

1. Discovering mathematical concepts through a multi-agent system
2. Lean Copilot: Large Language Models as Copilots for Theorem Proving in Lean
3. The Equational Theories Project: Advancing Collaborative Mathematical Research at Scale
4. Machine learning and information theory concepts towards an AI Mathematician
5. Symbols and mental programs: a hypothesis about human singularity.
6. LeanDojo: Theorem Proving with Retrieval-Augmented Language Models
7. LLMSTEP: LLM proofstep suggestions in Lean
