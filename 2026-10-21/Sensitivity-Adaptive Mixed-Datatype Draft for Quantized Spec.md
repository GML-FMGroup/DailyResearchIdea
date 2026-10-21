# Sensitivity-Adaptive Mixed-Datatype Draft for Quantized Speculative MoE Offloading

## Motivation

Specter's speculative offloading loop uses a single low-precision datatype for every expert in its draft model, implicitly assuming that expert-activation fidelity is uniform across experts; BitMoD demonstrates that expert sensitivity to quantization is sharply heterogeneous, so Specter's uniform draft systematically under-represents the activation patterns of high-sensitivity experts, causing mispredicted expert prefetches and wasted PCIe bandwidth. The root cause is structural: both lineages treat quantization as a static global or per-expert choice made before runtime, so neither can react to the per-expert activation-frequency and quantization-error signal that only becomes observable while the draft is actually running inside the offloading loop.

## Key Insight

Because the draft's only job is to predict expert activation, the fidelity that matters is per-expert activation-pattern agreement rather than per-tensor weight error, and this quantity is directly observable online from the draft's own routing decisions and quantization residuals, making per-expert bit-width a runtime control variable rather than a static design choice.

## Method

## SAD-MoD: Sensitivity-Adaptive Draft with Mixed-Datatype Experts

**(A) What it is.** SAD-MoD is a draft-model quantization controller that, for each MoE expert $e$ in the Specter draft, maintains a per-expert datatype $b_e \in \{2,3,4,8\}$ bits and updates it every $W=32$ draft steps from two online signals: the expert's routing frequency $f_e$ and its routing-disagreement residual $r_e$. Inputs: draft router logits, expert activations, and target-model verification outcomes; outputs: an updated per-expert bit-width table and a prefetch priority list for the offloading runtime.

**(B) How it works.**

```pseudocode
# State per expert e: b_e (init 4), f_e (EMA, alpha=0.1), r_e (EMA, alpha=0.1)
# Every draft step t:
for each token x_t in draft rollout:
    logits, acts = draft_forward(x_t)          # logits: top-2 gate logits; acts[e] = post-gate activation of expert e
    e* = argmax(logits)                         # routed expert
    f_{e*} <- (1-alpha) f_{e*} + alpha * 1
    # Routing-disagreement residual: margin shortfall and flip rate
    margin = logits[top1] - logits[top2]        # top-2 gate-logit margin
    flip = 1 if top1 != top1_full_precision else 0  # from verification outcome
    r_e <- (1-alpha) r_e + alpha * ( (1 - margin / margin_max) + flip ) / 2

# Every W=32 draft steps, run sensitivity estimator:
for each expert e with f_e > f_min (0.01):
    # sensitivity = routing-frequency-weighted residual, normalized by measured bit cost
    s_e = f_e * r_e / cost_e(b_e)               # cost_e(b) from precomputed residual-vs-bitwidth curve
    if s_e > tau_hi (0.15) and b_e < 8:  b_e <- next_higher(b_e)
    elif s_e < tau_lo (0.05) and b_e > 2: b_e <- next_lower(b_e)

# Feed updated b_e to offloading runtime:
prefetch_priority[e] = f_e * (1 + r_e)        # high-freq, high-residual experts prefetched first
```

**(C) Why this design.** We chose to drive bit-width from a routing-disagreement signal rather than from weight-quantization error (as BitMoD does) because the draft's downstream consumer is the router, and routing disagreement is what perturbs routing decisions, accepting the cost that we must instrument the draft's forward pass with a margin and flip-rate computation per expert. We chose a fixed-window ($W=32$) discrete controller over a learned or gradient-based bit allocation because the bit-width space is tiny (four levels) and a closed-form threshold rule is auditable and adds no training, accepting the cost that the controller cannot interpolate between levels. We chose to weight by routing frequency $f_e$ rather than uniform averaging because low-frequency experts contribute negligibly to prefetch errors, accepting the cost that a suddenly-hot expert has a lag of one window before its bit-width rises. We chose to couple the bit-width table to prefetch priority rather than treating them as separate knobs because the whole point is to spend PCIe bandwidth where the draft is least faithful, accepting the cost that a mis-set threshold degrades both signals together. A domain expert comparing this to Specter would not call it a variant: Specter fixes one datatype for the whole draft and never observes the draft's own residuals, whereas SAD-MoD's control variable is the per-expert routing-disagreement residual that Specter's design does not compute at all.

**(D) Why it measures what we claim.** The routing-disagreement residual $r_e$ measures per-expert routing fidelity loss because the assumption is that the normalized margin shortfall and flip rate are monotone in the probability that the draft's top-k routing differs from the full-precision routing; this assumption fails when two experts have near-tied gate logits, in which case $r_e$ reflects margin noise rather than actual routing disagreement. The routing frequency $f_e$ measures an expert's contribution to prefetch traffic because the assumption is that the offloading runtime's transfer volume per expert is proportional to how often the draft routes to it; this assumption fails under batched inference where a single expert serves many tokens per step, in which case $f_e$ undercounts its bandwidth share. The product $s_e = f_e \cdot r_e / \text{cost}_e(b_e)$ measures marginal fidelity gain per bit because the assumption is that residual reduction from adding one bit is approximately linear over the 2–8 bit range; this assumption fails at the 2-bit floor where residuals saturate, in which case $s_e$ overstates the benefit of the next increment and the controller holds at 2 bits longer than optimal. To calibrate $\text{cost}_e(b)$, we precompute a per-expert residual-vs-bitwidth curve from a calibration set of 512 examples, measuring the routing-disagreement residual at each bit-width $b \in \{2,3,4,8\}$ and fitting a monotone piecewise-linear function; this curve replaces the linear $\text{cost}(b)=b/4$ and is updated every 1000 draft steps.

## Contribution

['A per-expert mixed-datatype draft controller (SAD-MoD) that treats draft bit-width as an online control variable driven by activation-residual and routing-frequency signals observed inside the speculative offloading loop.', "A design finding that activation-residual-weighted sensitivity, not weight-quantization error, is the correct signal for allocating draft precision when the draft's purpose is expert-activation prediction rather than token likelihood.", "An integration of per-expert datatype tables into the offloading runtime's prefetch priority, yielding a single bandwidth-allocation surface rather than two independently tuned knobs."]

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | MoE workloads with skewed routing (e.g., Mixtral traces) | Skew exercises frequency-weighted controller |
| Primary metric | Time per output token (TPOT) | Directly reflects offloading + draft fidelity |
| Baseline | Specter (fixed-datatype draft) | Tests whether per-expert adaptivity beats uniform datatype |
| Baseline | SAD-MoD with frequency-only signal | Tests whether residual signal matters beyond frequency |
| Baseline | SAD-MoD with uniform prefetch priority | Tests whether coupling bit-table to prefetch helps |
| Baseline | Static per-expert BitMoD-style allocation (offline calibration) | Tests whether online controller beats static per-expert allocation |
| Ablation-of-ours | SAD-MoD with residual-only (no frequency weighting) | Isolates the frequency-weighting contribution |
| Ablation-of-ours | SAD-MoD with linear cost(b)=b/4 | Tests whether measured residual-vs-bitwidth curve matters |

### Why this setup validates the claim

The central claim is that driving per-expert draft bit-width from the routing-disagreement residual signal, weighted by routing frequency and coupled to prefetch priority, reduces TPOT on skewed MoE workloads relative to a fixed-datatype draft. Specter tests the core sub-claim: if per-expert adaptivity is what matters, SAD-MoD should beat a uniform-datatype draft specifically on the high-residual, high-frequency experts that Specter over-spends on or under-serves. The frequency-only and uniform-prefetch baselines are control variants that isolate two distinct mechanisms — the residual signal and the prefetch coupling — so a win over Specter that disappears against these controls would indicate the gain came from a different source than claimed. The static BitMoD-style baseline tests whether the online controller's runtime adaptivity provides any benefit over a well-tuned static per-expert allocation, which is the key novelty claim. The residual-only ablation tests whether frequency weighting is load-bearing or decorative. The linear-cost ablation tests whether the measured residual-vs-bitwidth curve is necessary. TPOT is the right metric because it aggregates both the draft's acceptance behavior and the offloading runtime's transfer cost, which are exactly the two channels our controller is designed to influence; a fidelity-only metric would miss the prefetch coupling, and a bandwidth-only metric would miss the draft-faithfulness channel.

### Expected outcome and causal chain

**vs. Specter (fixed-datatype draft)** — On a case where a few high-frequency experts have large routing-disagreement residuals while most experts are low-residual, Specter spends the same 4-bit budget on every expert, so the critical experts are under-quantized and their routing diverges from full precision, forcing extra verification rounds and raising TPOT. Our method instead raises those experts toward 8 bits within one 32-step window while dropping cold low-residual experts to 2–3 bits, so we expect a visible TPOT gap concentrated on skewed-routing subsets and near-parity on uniform-routing subsets where no expert dominates.

**vs. SAD-MoD with frequency-only signal** — On a case where two experts have similar routing frequency but very different routing-disagreement residuals (one smooth, one spiky), the frequency-only variant assigns them the same bit-width, so the spiky expert keeps perturbing router logits and its drafts get rejected at verification. Our method instead reads the routing-disagreement residual and lifts only the spiky expert's bit-width, so we expect the frequency-only variant to show higher rejection rates and TPOT specifically on the high-residual-frequency-matched subset, with the gap vanishing when residuals are homogeneous.

**vs. SAD-MoD with uniform prefetch priority** — On a case where PCIe bandwidth is saturated and the draft routes unevenly, uniform prefetch orders experts by index rather than by need, so high-frequency high-residual experts arrive late and stall the draft step. Our method instead ranks by $f_e(1+r_e)$, so the experts that actually gate the next step are resident first, and we expect a TPOT improvement that grows with bandwidth pressure and shrinks to parity when bandwidth is ample.

**vs. Static per-expert BitMoD-style allocation** — On a case where the routing distribution shifts mid-serving (e.g., a new domain with different expert frequencies), the static allocation, computed offline from calibration data, cannot adapt and will over- or under-provision experts that become hot or cold. Our method updates bit-widths every 32 steps based on online signals, so we expect it to maintain lower TPOT after the shift, with the gap growing with the magnitude of the distribution shift and vanishing when the serving distribution matches the calibration set.

### What would falsify this idea

If SAD-MoD's TPOT advantage over Specter is uniform across skewed and uniform-routing subsets rather than concentrated where high-frequency high-residual experts exist, then the residual-driven controller is not the causal source of the gain. Likewise, if the residual-only ablation matches the full method, the frequency weighting is decorative and the central mechanism claim is wrong. If the static BitMoD-style baseline matches or beats SAD-MoD even under distribution shift, the online adaptivity claim is falsified.

## References

1. Specter: Fast MoE Inference via Quantization-Enabled Multi-Step Speculative Offloading
