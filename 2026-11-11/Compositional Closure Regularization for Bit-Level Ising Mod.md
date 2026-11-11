# Compositional Closure Regularization for Bit-Level Ising Model Edits

## Motivation

Tree-Ising and knowledge-neuron attribution both edit localized subnetworks without ever verifying that the edited behavior survives composition with other behavior-mediating directions; the closure of the edited subnetwork under multi-layer composition is never established. This structural gap recurs across branches—knowledge-neuron edits and refusal-direction edits—because no existing objective penalizes edits whose forward-pass effect is not preserved when other behavior directions are simultaneously activated. The root cause is that Tree-Ising's three-term QUBO objective is defined on single-direction forward statistics, so the optimizer has no signal that composition with refusal, loop, or conclude directions can undo the edit.

## Key Insight

Compositional closure is a property of the edited subnetwork's joint forward map over a basis of behavior directions, not of any single direction's statistics, so it can be enforced as a quadratic penalty on cross-direction interference within the same Ising/QUBO decomposition that Tree-Ising already solves exactly.

## Method

**What it is.** We introduce CLOIS (Compositional LOcal Ising Subnetwork closure), a bit-level edit procedure that augments Tree-Ising's per-row QUBO with a closure regularizer. Inputs: BF16 weights of the targeted layers, a set of behavior directions D = {loop, conclude, refusal, productive} identified by forward statistics, and a composition schedule of multi-layer prompts. Outputs: bit-level edits (sign flips, exponent shifts) whose forward-pass behavior on each direction is preserved when any pair of other directions is simultaneously activated.

**How it works.**

```pseudocode
# Phase 1: direction basis and composition probes
D = {loop, conclude, refusal, productive}          # from forward statistics
P_comp = { (d_i, d_j, prompt_k) }                  # pairs of directions co-activated on multi-layer prompts

# Phase 2: per-row QUBO with closure term
for each output row r in targeted layers:
    H_r = H_TreeIsing(r)                            # original 3-term objective
    for each pair (d_i, d_j) in D x D:
        # interference matrix between direction i and j on row r
        C_ij[r] = sum_k  ( <w_r, a_i^k> * <w_r, a_j^k> )^2
        H_r += lambda_closure * C_ij[r] * (1 - delta_ij)   # lambda_closure = 0.3
    solve H_r via simulated annealing on GPU (T=0.01, 10^4 sweeps)

# Phase 3: closure test (accept/reject)
for each edit e:
    for each (d_i, d_j, prompt_k) in P_comp:
        if | f_edited(prompt_k | d_i, d_j) - f_base(prompt_k | d_i) | > tau:
            reject e; return to Phase 2 with increased lambda_closure
# tau = 0.05 logit tolerance
```

The closure term C_ij[r] is the squared inner product between the row's response to direction i and its response to direction j, summed over composition prompts. It is quadratic in the bit-level edit variables (since each <w_r, a> is linear in the bits), so it folds into the existing QUBO without breaking exact per-row decomposition.

**Why this design.** We chose to fold the closure penalty into the per-row QUBO rather than run a separate post-hoc filter because Tree-Ising's exact decomposition is what makes bit-level optimization tractable, and a post-hoc filter would discard edits without guiding the optimizer away from non-closed regions; the cost is that the closure term couples direction pairs within a row, slightly increasing QUBO density. We chose a quadratic interference penalty over a hard constraint because hard constraints on composition would make the QUBO infeasible for many rows (no admissible bit edit satisfies all pairs), while a soft penalty with a reject-and-retry loop keeps the problem always solvable; the cost is that closure is enforced to tolerance tau rather than exactly. We chose simulated annealing over simulated bifurcation for the augmented QUBO because the closure term adds off-diagonal couplings that bifurcation handles less reliably at this density, accepting longer solve time per row. A domain expert would not describe CLOIS as 'Tree-Ising plus a filter' because the closure term changes the optimization landscape itself—it is a new objective term solved jointly with the original three, not a post-processing step, and it is the first objective in this line to make cross-direction composition an explicit training signal rather than an untested assumption.

**Why it measures what we claim.** The interference quantity C_ij[r] measures compositional non-closure because it equals the squared cross-term in the forward map's response to co-activated directions i and j under the assumption that the subnetwork's response is linear in the direction basis; this assumption fails when directions interact nonlinearly (e.g., refusal saturates before loop activates), in which case C_ij[r] reflects pairwise linear leakage rather than full nonlinear interference. The closure test tolerance tau measures behavioral invariance under composition because it compares the edited forward pass on co-activated prompts against the base forward pass on the single direction under the assumption that the base single-direction output is the correct reference for what the edit should preserve; this assumption fails when the base model's single-direction output is itself degenerate (e.g., base already loops), in which case tau measures deviation from a degenerate reference rather than from a desired behavior. The reject-and-retry loop measures whether the optimizer's closure penalty was sufficient because rejection triggers only when the realized forward pass violates tau under the assumption that the QUBO's linearization of the forward map is accurate at the edit magnitude; this assumption fails for large exponent shifts, in which case rejection reflects linearization error rather than true non-closure.

## Contribution

['A compositional closure regularizer for bit-level Ising/QUBO model editing that folds cross-direction interference into the per-row objective without breaking exact decomposition.', "A closure test protocol over paired behavior directions (loop, conclude, refusal, productive) that operationalizes 'edited subnetwork closure under multi-layer composition' as a measurable accept/reject criterion.", 'An empirical design principle: closure must be optimized jointly with the original edit objective, not filtered post-hoc, because post-hoc rejection discards edits the optimizer could have repaired.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | Compositional prompt suite (paired directions) | Directly probes cross-direction closure |
| Primary metric | Closure-violation rate at tau=0.05 | Measures the exact property claimed |
| Baseline | Astra agent edit (GPT-6) | Strong single-direction reference edit |
| Baseline | Tree-Ising (no closure term) | Isolates the closure contribution |
| Baseline | Base Qwen3.5-0.8B (unedited) | Lower bound on composition stability |
| Ablation-of-ours | CLOIS with lambda_closure=0 | Tests whether closure term drives the gain |

### Why this setup validates the claim

The central claim is that folding a closure penalty into the per-row QUBO produces bit edits whose single-direction behavior survives pairwise co-activation, where prior methods break. The paired-direction prompt suite is the right dataset because it instantiates exactly the composition the claim is about: each prompt activates two directions simultaneously, so closure violations become directly observable rather than hypothetical. Closure-violation rate at tau is the right primary metric because it is the quantity the reject-and-retry loop is designed to minimize; a method that genuinely reshapes the optimization landscape should lower it, while a method that only patches single directions should not. The two external baselines test different sub-claims: the Astra agent edit tests whether a strong single-direction edit already achieves closure (if it does, our premise collapses), and Tree-Ising without the closure term tests whether the closure term—not merely bit-level editing—is responsible. The lambda_closure=0 ablation is the sharpest test: it is CLOIS minus its distinguishing objective, so any gap between it and full CLOIS is attributable to the closure mechanism itself rather than to the SA solver, the direction basis, or the bit-level representation.

### Expected outcome and causal chain

**vs. Astra agent edit** — On a case where a loop-inducing prompt is co-activated with a conclude direction, the Astra edit produces a blended continuation that partially loops because its edit was optimized for single-direction behavior and never saw the co-activated context. Our method instead penalizes the row-level interference between loop and conclude responses during optimization, so the solver avoids bit configurations where those two direction responses share large inner product. We expect a noticeable gap on loop×conclude and refusal×productive pairs but rough parity on single-direction prompts, since both methods target single-direction fidelity.

**vs. Tree-Ising (no closure term)** — On a case where refusal and productive directions are co-activated, Tree-Ising produces an edit that satisfies each direction in isolation but leaks refusal signal into the productive response because its three-term objective has no cross-direction coupling and the SA solver is free to land on high-interference rows. Our method instead includes C_ij[r] in the same QUBO, so the solver is steered away from those rows before acceptance. We expect a measurable reduction in closure-violation rate concentrated on refusal×productive and loop×conclude pairs, with the largest gap on rows where the unedited interference C_ij was highest.

**vs. Base Qwen3.5-0.8B (unedited)** — On a case where the base model already loops, the unedited forward pass produces degenerate output that trivially fails any composition test, because there is no edit to stabilize behavior. Our method instead applies bit edits that reduce the degenerate direction's response magnitude, so the co-activated pass deviates less from the (now non-degenerate) single-direction reference. We expect CLOIS to beat the base by a wide margin on loop-containing pairs but to show only modest gains on pairs not involving the degenerate direction, since those were already stable.

### What would falsify this idea

If CLOIS's closure-violation reduction over Tree-Ising is uniform across all direction pairs rather than concentrated on the pairs with the highest unedited interference C_ij, the closure term is not doing the causal work claimed and the gain is attributable to solver or representation differences instead. Equivalently, if the lambda_closure=0 ablation matches full CLOIS, the central mechanism claim is wrong.

## References

1. Tree-Ising: Repairing Language Models with Bit-Level Ising Edits
