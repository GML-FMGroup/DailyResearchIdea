# Monotone Success-Conditioned Invariant Training for Speech-to-Tool-Use Models

## Motivation

Two-stage speech-LLM adaptation pipelines such as Apriel Voice 27B Preview train the speech projector for interface fidelity and then jointly adapt projector and backbone for reasoning, but neither stage optimizes the composite quantity that actually matters: speech-conditioned tool-call success. Because interface fidelity and reasoning are optimized on separate objectives, the pipeline provides no guarantee that improving speech representation fidelity cannot degrade downstream tool-use success — a speech embedding can become more faithful to the transcript while shifting the backbone's tool-selection distribution in a way that breaks execution. This is a meta-level structural failure of staged adaptation: task success is never a first-class training signal coupling the two stages, so the two objectives can move in opposite directions.

## Key Insight

If task success is made a monotone function of speech-interface fidelity via a counterfactual consistency loss over paired speech-text inputs, then any projector update that improves fidelity while degrading tool success incurs a directly penalizable violation, coupling the two stages through a single invariant rather than through shared gradients alone.

## Method

**What it is.** Monotone Success-Conditioned Invariant Training (MSCIT) is a training objective added to stage-two adaptation of a speech-LLM tool-use model. Inputs: counterfactual pairs `(speech s, transcript text t)` for the same dialogue turn, a differentiable task-success proxy `R` (verifier score from tool-call execution: 1.0 if the emitted tool call executes and the verifier accepts its arguments, 0.0 on execution/verification failure, plus a partial-credit term from argument-field exact match), and the current projector `P` and backbone `L`. Output: an updated projector and backbone trained under a loss that penalizes any fidelity improvement that does not weakly improve task success.

**Load-bearing assumption (stated explicitly).** The method assumes that the sign of the change in a behaviorally grounded fidelity measure between speech- and text-conditioned tool-call distributions is a valid causal proxy for the direction in which the speech path's tool-call success is moving, so that penalizing `-sign(delta_fidelity)*delta_success` couples fidelity improvements to success improvements rather than penalizing noise. This assumption fails when backbone updates from `L_task` systematically shift the text-path success `R_t` within the EMA window, in which case the violation term reflects backbone drift rather than a fidelity-success decoupling. To make the assumption testable, the violation is computed with the backbone frozen for the paired forward passes (see step 3), and a backbone-frozen control is reported in the experiment.

**How it works.**

```pseudocode
for each batch of turns (s, t, tool_spec) with gold tool call y*:
    # 0. Freeze backbone L for the paired forward passes used in the violation term.
    #    L_frozen = copy of L at the start of the current window (window = 50 steps).
    #    Gradients from L_inv flow to P only; L_task updates L normally.

    # 1. Fidelity score: behaviorally grounded, differentiable distributional agreement.
    h_s = L_frozen.project(P(encode(s)))    # speech path hidden at tool-call position
    h_t = L_frozen.project(encode_text(t))  # text path hidden at same position
    p_s = softmax(L_frozen.tool_logits(h_s, tool_spec) / T)  # T = 1.0
    p_t = softmax(L_frozen.tool_logits(h_t, tool_spec) / T)
    fidelity = -KL(p_t || p_s)              # scalar <= 0; higher is better

    # 2. Task-success proxy via verifier on executed tool calls.
    y_s = L_frozen.decode_tool_call(h_s, tool_spec)
    y_t = L_frozen.decode_tool_call(h_t, tool_spec)
    R_s = verifier(y_s, y*, tool_spec)      # in [0,1], execution + arg match
    R_t = verifier(y_t, y*, tool_spec)

    # 3. Monotone consistency violation: success must not drop when fidelity rises.
    #    delta_fidelity = fidelity - fidelity_prev (EMA of per-batch mean, decay 0.99)
    #    delta_success  = R_s - R_t  (speech vs text success gap, both under L_frozen)
    violation = relu( -sign(delta_fidelity) * delta_success + margin )  # margin = 0.05

    # 4. Success-conditioned invariant loss.
    L_inv = mean( violation )
    L_task = -log p(y* | h_s)               # standard CE on speech path, gradients to P and L
    L_fid  = -fidelity                      # interface fidelity (KL-based)
    L = L_task + lambda_fid * L_fid + lambda_inv * L_inv   # lambda_fid=0.1, lambda_inv=0.5

    # 5. Counterfactual hard-negative mining: every K=4 steps, select pairs where
    #    fidelity rose but R_s fell; upweight them 2x in L_inv.
    update(P, L) with grad(L)               # L_inv gradients only to P; L_task to P and L
    fidelity_prev = 0.99 * fidelity_prev + 0.01 * mean(fidelity)
    # Every 50 steps: refresh L_frozen <- L (new window).
```

**Why this design.** We chose a counterfactual speech-text pair loss over a single-path success loss because only the paired structure lets us define the directional violation `sign(delta_fidelity)*delta_success`, which is the quantity that operationalizes monotone consistency; the cost is that every step requires two forward passes through the backbone, roughly doubling stage-two compute. We chose a verifier-based execution reward over a learned reward model because tool-call execution is a deterministic, non-gameable signal for the tool-use domain, accepting that it is only available for tool calls whose schemas permit sandboxed execution. We chose an EMA-tracked `fidelity_prev` baseline over a fixed reference projector because fidelity naturally rises during training and a fixed baseline would saturate the violation term to zero after early steps, accepting that the EMA introduces a mild lag in detecting sudden fidelity jumps. We chose to upweight counterfactual hard negatives over uniform sampling because violations are rare once the invariant is partially satisfied, accepting a small variance increase in the gradient. We chose to freeze the backbone for the paired forward passes in the violation term because otherwise backbone updates from `L_task` contaminate `delta_success` with drift, accepting that the frozen copy lags the live backbone by up to 50 steps. Unlike VinePPO or process-reward-model step scoring, which assign credit per reasoning step, MSCIT never decomposes the trajectory: it acts on a paired-input invariant between two conditioning modalities, which is a structural property those methods do not produce.

**Why this measures what we claim.** The KL-based fidelity `fidelity = -KL(p_t || p_s)` measures *speech-interface fidelity* because it assumes the text-conditioned tool-call distribution is the correct target for the same turn; this assumption fails when the transcript itself is disfluent or the speech carries prosody that the text cannot encode, in which case fidelity reflects distributional alignment rather than semantic equivalence. The verifier score `R_s` measures *task success* because it assumes the sandboxed execution and verifier acceptance criteria match the deployment tool-use criteria; this assumption fails when the deployment tool has side effects not modeled by the sandbox, in which case `R_s` reflects sandbox pass rate rather than real success. The violation term `relu(-sign(delta_fidelity)*delta_success + margin)` measures *monotone consistency between fidelity and success* because it assumes that over a short EMA window the only systematic driver of `delta_success` is the change in speech conditioning (enforced by freezing the backbone for the paired forward passes); this assumption fails when the frozen backbone still drifts relative to the live backbone within the 50-step window, in which case the violation reflects residual backbone drift rather than a fidelity-success decoupling.

## Contribution

(1) A success-conditioned invariant loss, MSCIT, that couples the speech projector and language backbone through a differentiable execution-verifier reward and a directional monotone-consistency violation over counterfactual speech-text pairs. (2) A counterfactual hard-negative mining schedule that concentrates the invariant penalty on the rare turns where fidelity rises while task success falls, making the coupling signal trainable without uniform reweighting. (3) An analysis of when execution-verifier rewards are a valid proxy for tool-use success in speech-conditioned settings, including the conditions under which the monotone violation term reflects genuine fidelity-success decoupling versus backbone drift.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Speech-BFCL tool-use turns | Paired speech/text turns with executable tool schemas. |
| Primary metric | Verifier task-success rate | Deterministic execution signal, directly matches claim. |
| Baseline | Apriel Voice 27B Preview | Strong speech-LLM tool-use reference point. |
| Baseline | Single-path success loss (no pairing) | Tests whether counterfactual pairing is necessary. |
| Baseline | Fixed-reference fidelity baseline | Tests whether EMA baseline is necessary. |
| Baseline | Speech-text KL consistency without sign-based violation | Tests whether monotone-direction term is load-bearing. |
| Ablation-of-ours | MSCIT without invariant term | Isolates contribution of monotone consistency penalty. |
| Control | MSCIT with backbone frozen for violation (window=50) | Isolates fidelity-driven success from backbone drift. |

### Why this setup validates the claim

The central claim is that penalizing fidelity gains that do not weakly improve task success produces a speech-LLM whose tool-use success tracks its speech-interface fidelity. Speech-BFCL is the right dataset because it supplies paired speech/text turns whose tool calls are actually executable, so the verifier's success signal is deterministic and non-gameable rather than a learned proxy. Verifier task-success rate is the correct primary metric because the claim is about success, not about hidden-state similarity; fidelity is only an intermediate lever, so measuring success directly makes the test falsifiable. The baselines decompose the sub-claims: Apriel Voice 27B Preview establishes whether the method beats a strong unmodified speech-LLM; the single-path success loss tests whether the counterfactual pairing is load-bearing versus merely adding a success term; the fixed-reference fidelity baseline tests whether the EMA-tracked violation is load-bearing versus a static fidelity target; the speech-text KL consistency baseline (same pairing, same KL fidelity, but no sign-based violation term) tests whether the monotone-direction term is the novel load-bearing component rather than the pairing itself. The ablation without the invariant term isolates whether the monotone consistency penalty—not just fidelity or task CE—drives any gain. The backbone-frozen control (window=50 steps) tests the load-bearing assumption that backbone drift does not contaminate `delta_success`: if MSCIT's gain persists when the backbone is frozen for the violation computation, the mechanism is fidelity-driven; if the gain vanishes, the violation term was reacting to backbone drift. If the gain appears only for MSCIT and vanishes for the ablation, the paired-loss baseline, and the KL-consistency baseline, the mechanism is confirmed; if it appears for all, the pairing and invariant are inert.

### Expected outcome and causal chain

**vs. Apriel Voice 27B Preview** — On a turn where the speech is fluent but the transcript drops a disfluent filler that carried the argument slot, the baseline's speech path drifts toward a surface-aligned hidden state and emits a tool call with a mismatched argument, so the verifier rejects it. Our method instead sees fidelity rise while `R_s` falls, and the violation term pulls the speech path back toward a state whose success matches the text path, so we expect a noticeable gap on disfluent-transcript turns but parity on clean turns.

**vs. Single-path success loss (no pairing)** — On a turn where speech and text both yield a correct tool call but for different reasons, the single-path loss cannot see the speech-text gap, so it accepts a speech path that succeeds by accident while fidelity stays low. Our method computes `delta_success = R_s - R_t` against the paired text path, so it detects that success is not conditioned on fidelity and pushes the speech path toward the text-consistent representation, so we expect higher fidelity at matched success and fewer fragile successes on held-out turns.

**vs. Fixed-reference fidelity baseline** — On later training steps where fidelity has naturally risen, the fixed-reference violation term saturates to zero, so the baseline stops penalizing decoupling and success gains stall while fidelity keeps climbing. Our method's EMA baseline tracks the rising fidelity, so the violation stays informative and success continues to improve, so we expect the baseline to plateau in success mid-training while MSCIT keeps a positive success-fidelity slope.

**vs. Speech-text KL consistency without sign-based violation** — This baseline uses the same paired speech/text inputs and the same KL fidelity term but omits the `-sign(delta_fidelity)*delta_success` violation. On turns where fidelity rises while `R_s` falls, the baseline has no directional penalty, so it accepts the decoupling. Our method's violation term directly penalizes those turns, so we expect MSCIT to outperform the KL-consistency baseline specifically on the subset of turns where fidelity and success move in opposite directions, confirming that the monotone-direction term—not the pairing—is load-bearing.

**vs. MSCIT without invariant term** — On hard-negative turns mined where fidelity rose but `R_s` fell, the ablation has no term to react, so those turns stay wrong and the upweighting does nothing. Our method's `L_inv` directly penalizes them, so we expect the ablation to match MSCIT on easy turns but show a concentrated gap on the mined hard-negative subset.

**vs. MSCIT with backbone frozen for violation (window=50)** — This control freezes the backbone for the paired forward passes used in the violation term, so `delta_success` isolates the projector's effect from backbone drift. If MSCIT's gain over the no-invariant ablation persists under this control, the violation term is reacting to fidelity-success decoupling rather than backbone drift; if the gain vanishes, the original violation term was confounded. We expect the frozen-backbone control to retain most of MSCIT's gain, with a small drop attributable to residual drift within the 50-step window.

### What would falsify this idea

If MSCIT's gain over the single-path baseline is uniform across all subsets rather than concentrated on turns where fidelity rose while success fell, or if the no-invariant ablation or the speech-text KL consistency baseline matches MSCIT, or if the backbone-frozen control eliminates the gain, then the monotone-consistency mechanism is not what produces the effect and the central claim is wrong.

## References

1. Apriel Voice 27B Preview: An Open Model for Speech, Reasoning, and Tool Use
