# Recall-Gated Proactive Streaming: A Trigger-Conditioned Memory Sufficiency Contract for Model-Initiated Utterances and Tool Calls

## Motivation

Proactive streaming assistants such as AURA 2.0 (Beyond Answering: Learning to Speak First and Act in Streaming Video) decide when to initiate a conversation (MIC) or invoke an external tool (SATC) using a trigger head that is trained independently of the memory state the response will later be grounded in; the memory itself is inherited as a fixed hierarchical buffer with no recall guarantee at the arbitrary timestamp the model chooses to speak. Structurally, the trigger decision and the evidence sufficiency are two random variables that prior work treats as separable — the trigger is learned from video dynamics, the memory is a compression heuristic (StreamChat's training-free hierarchical buffer; InfiniPot-V's query-agnostic compression) — so a proactive utterance can fire at a moment when the retained evidence set does not actually contain the fact the utterance will assert. The residual gap is joint identifiability: no existing method ties the emission decision to a verifiable property of the retrieved evidence at the trigger timestamp.

## Key Insight

A proactive emission is safe iff the memory state at the trigger timestamp is a sufficient statistic for the utterance it will produce, and this sufficiency is identifiable from a paired contrast between the compressed evidence set and a held-out full-resolution evidence set for the same timestamp — a signal the trigger head cannot see unless it is trained against the memory contract rather than alongside it.

## Method

**What it is.** We introduce **TRACE** (Trigger-Recall Aligned Contract for Emissions), a streaming assistant that emits a Model-Initiated Conversation (MIC) or Stream-Aware Tool Call (SATC) only when a lightweight *sufficiency verifier* certifies that the current bounded memory state supports the candidate utterance. Inputs: the live frame stream, the inherited AURA 2.0 trigger head's candidate emission, and a dual-index memory. Outputs: a binary EMIT/SUPPRESS decision, and, if EMIT, the utterance/tool call conditioned on the retrieved evidence set.

**Load-bearing assumption (stated explicitly).** TRACE assumes that the paired-contrast recall estimate `rho` between the bounded retrieved set `E_ret` and a held-out full-resolution reference `E_full` is a valid, trainable proxy for whether the memory state is a sufficient statistic for the candidate utterance, such that penalizing low-`rho` emissions in `L_contract` teaches the trigger head to attend to memory-supported moments rather than merely suppressing emissions. This assumption is load-bearing and is tested directly by the frozen-verifier ablation and the scope-stratified analysis below.

**How it works.**

```pseudocode
# Dual-index memory M = (T, S), updated every frame f_t
#   T: temporal-redundancy-pruned token index (TaR metric, cap N_T = 4096)
#   S: value-norm-ranked semantic token index (VaN rank, cap N_S = 2048)
for each incoming frame f_t:
    append KV(f_t) to M
    if |M| > N_T + N_S:
        drop tokens with lowest TaR score from T   # temporal-axis redundancy
        drop tokens with lowest VaN rank from S       # semantic salience

# Candidate emission from inherited AURA 2.0 trigger head
tau_hat = trigger_head(M)                # candidate timestamp
c_tau   = candidate_utterance(M, tau_hat) # MIC topic or SATC tool call

# TRACE sufficiency contract
E_ret   = dual_index_retrieve(M, query=c_tau, tau=tau_hat)   # bounded evidence set, |E_ret| <= K=64
E_full  = full_resolution_window(video, tau_hat, W=8s)        # held-out reference evidence
s       = sufficiency_verifier(c_tau, E_ret)                  # scalar in [0,1]
rho     = recall_probe(E_ret, E_full, c_tau)                  # paired-contrast recall estimate
if s >= theta_s (0.7) and rho >= theta_rho (0.9):
    EMIT(c_tau grounded in E_ret)
else:
    SUPPRESS and log (tau_hat, E_ret, s, rho) for contract training

# Joint training of trigger head and verifier (one shared loss)
L = L_emit(trigger_head) + lambda * L_verifier(s, rho) + mu * L_contract(s, rho, EMIT)
# lambda = 1.0, mu = 0.5; L_contract penalizes EMIT when rho < theta_rho
```

**Concrete specification of `rho` (fact-level recall).** `rho` is defined as a set-recall over fact-level propositions extracted from `E_full` and `E_ret`. Extraction procedure: (i) run an off-the-shelf entailment model (DeBERTa-v3-large NLI, frozen) over each candidate utterance `c_tau` and each retrieved evidence span to produce a set of atomic propositions `P_full` from `E_full` and `P_ret` from `E_ret`; (ii) match propositions by bidirectional entailment (threshold 0.5); (iii) `rho = |P_full ∩ P_ret| / |P_full|`, clipped to [0,1]. Inter-annotator agreement on a 200-sample calibration set is measured by Cohen's kappa; we report it alongside `rho`'s Pearson correlation with human-judged utterance support on that same held-out sample.

**Concrete specification of the E_full-window sufficiency assumption and its failure mode.** The contract condition is: `E_full` (the 8s full-resolution window at `tau_hat`) contains all facts that `c_tau` can legitimately assert. This assumption fails when `c_tau`'s temporal scope exceeds `W=8s` or when `c_tau` draws on parametric knowledge rather than the video; in that case `rho` reflects only local-window recall and the verifier may certify an utterance whose evidence is globally incomplete. We therefore stratify unsafe-emission rate by utterance temporal scope (short-scope ≤8s vs. long-scope >8s) in the experiment.

**Why this design.** We chose a *paired-contrast recall probe* (comparing the compressed evidence set against a full-resolution window at the same timestamp) over a self-reported confidence score because LLM log-probs are uncalibrated for task correctness and would make the gate a learned independence head over operationally coupled variables — the exact anti-pattern prior work on proactive assistants falls into. We chose *joint* optimization of the trigger head and verifier over the two-stage AURA 2.0 pipeline because the trigger head's emission distribution must be shaped by the memory contract, not merely filtered post hoc; the cost is that the trigger head can no longer fire on visually salient but memory-unsupported events, which we accept because those are precisely the unsafe emissions. We chose a dual-index memory with TaR/VaN pruning (InfiniPot-V) over a single hierarchical buffer (StreamChat) because the two indices give the verifier two *independent* evidence channels whose agreement is diagnostic of sufficiency; the cost is a larger constant-factor memory footprint, which we bound by hard caps N_T, N_S. A domain expert would not describe TRACE as 'InfiniPot-V + AURA 2.0' because the load-bearing object is the *joint contract loss* L_contract that couples the emission decision to a recall estimate — neither InfiniPot-V nor AURA 2.0 trains any quantity against a recall target, and the paired-contrast probe is a new training signal, not a retrieval operation.

**Why this measures what we claim.** The quantity `rho` (paired-contrast recall between E_ret and E_full) measures *evidence sufficiency for the candidate utterance* because we assume the full-resolution window E_full contains all facts the utterance c_tau could legitimately assert (the window is chosen to cover the utterance's temporal scope); this assumption fails when the utterance refers to events outside W=8s, in which case rho reflects only local-window recall and the verifier may certify an utterance whose evidence is globally incomplete. The quantity `s` (verifier score) measures *grounding of the candidate utterance in the retrieved set* because we assume the verifier is trained on (utterance, evidence) pairs where the evidence is the sole support; this assumption fails when the utterance is supported by parametric knowledge rather than the video, in which case s reflects plausibility rather than grounding. The joint loss `L_contract` measures *alignment between emission policy and memory contract* because we assume that suppressing emissions with rho < theta_rho during training shifts the trigger head's distribution toward memory-supported moments; this assumption fails when the trigger head's features are dominated by visual salience that is uncorrelated with memory support, in which case L_contract penalizes emissions without teaching the trigger head what to attend to.

## Contribution

['TRACE, a trigger-conditioned memory sufficiency contract that gates MIC/SATC emissions on a paired-contrast recall estimate between a compressed dual-index evidence set and a full-resolution reference window at the trigger timestamp.', 'A joint training objective (L_contract) that couples the inherited AURA 2.0 trigger head to the memory contract, replacing the post-hoc filtering of proactive emissions with an in-the-loop sufficiency signal.', "A dual-index memory (TaR-pruned temporal tokens + VaN-ranked semantic tokens) with hard caps that makes the sufficiency verifier's two evidence channels independently diagnostic of recall."]

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | StreamBench (StreamChat) and OmniStar (LiveStar) with proactive-event annotations | Supplies ground-truth moments where emission is warranted |
| Primary metric | Unsafe-emission rate at matched emission recall | Directly tests the contract, not raw fluency |
| Secondary metric | Fact-level recall `rho` correlation with human-judged support (Pearson r on 200-sample calibration set) | Grounds the `rho` quantity empirically |
| Baseline | AURA 2.0 | Tests whether contract gating beats inherited trigger head |
| Baseline | StreamBridge | Tests whether joint contract beats post-hoc activation filtering |
| Baseline | LiveStar | Tests whether recall-probe gating beats timing-only decoding |
| Ablation-of-ours | TRACE without L_contract (frozen verifier gate) | Isolates the joint-loss contribution |
| Ablation-of-ours | TRACE with self-consistency probe (two independently pruned memory states at same timestamp) replacing paired-contrast E_full | Isolates the contrast structure as the novel object |
| Ablation-of-ours | TRACE with StreamBridge activation model as trigger head | Tests contract transfer across trigger heads |

### Why this setup validates the claim

The central claim is that coupling the emission decision to a paired-contrast recall estimate reduces unsafe Model-Initiated Conversation and Stream-Aware Tool Call emissions without sacrificing the emissions that are actually warranted. The design makes this falsifiable by measuring unsafe-emission rate at a matched emission-recall operating point: if TRACE merely emits less, recall drops and the comparison is void, so matching recall forces the comparison onto *which* moments are emitted. AURA 2.0 tests the sub-claim that the inherited trigger head alone over-fires on visually salient but memory-unsupported events. StreamBridge tests the sub-claim that a post-hoc activation filter cannot substitute for shaping the trigger distribution, since its lightweight activation model is not trained against any recall target. LiveStar tests the sub-claim that response-timing decoding, which optimizes *when* to speak, does not by itself certify *whether the evidence supports* the utterance. The frozen-verifier ablation is the load-bearing control: freezing the verifier and removing L_contract keeps the dual-index memory and the recall probe but removes the coupling, so any gap between it and full TRACE is attributable to the joint contract loss rather than to memory architecture or retrieval. The self-consistency-probe ablation isolates the paired-contrast structure: if replacing E_full with two independently pruned memory states yields the same unsafe-emission rate, the novelty is merely a second retrieval channel, not the contrast. The StreamBridge-trigger-head ablation tests whether the contract is a reusable criterion rather than a single-system patch. Unsafe-emission rate at matched recall is the right metric because the predicted effect is a redistribution of emissions away from unsupported moments, which a pure accuracy or fluency metric would average away. We additionally stratify unsafe-emission rate by utterance temporal scope (≤8s vs. >8s) to show where the E_full-window sufficiency assumption holds and where the gate degrades.

### Expected outcome and causal chain

**vs. AURA 2.0** — On a case where a visually salient event (a person entering frame) occurs but the supporting context needed to comment on it has already been pruned from bounded memory, AURA 2.0's trigger head fires on salience alone because nothing in its training ties emission to retrievable evidence. Our method instead retrieves E_ret for the candidate utterance, finds low paired-contrast recall against the full-resolution window, and suppresses because rho falls below theta_rho. We expect a concentrated gap on the salient-but-unsupported subset, with near-parity on the well-supported subset where AURA 2.0 already fires correctly.

**vs. StreamBridge** — On a multi-turn case where an early fact is compressed away by round-decayed buffer compression and a later utterance depends on it, StreamBridge's decoupled activation model emits because it filters on activation strength, not on whether the compressed memory still contains the needed fact. Our method's dual-index retrieval surfaces the semantic-index channel and the recall probe flags the mismatch, so the emission is suppressed or deferred. We expect the gain to appear mainly on long-horizon, fact-carrying turns rather than on short clips, where StreamBridge's buffer has not yet lost anything.

**vs. LiveStar** — On a case where the optimal response timing is correctly identified but the evidence for the specific claim is only locally complete, LiveStar's response-silence decoding fires at the right moment with an under-grounded utterance because its verification is a single forward pass over timing, not over evidence sufficiency. Our method accepts the timing but gates on rho and s, so it either suppresses or emits a narrower, evidence-bounded utterance. We expect fewer unsafe emissions at matched recall on claims spanning more than the local window, with parity on short-scope claims that fit inside W=8s.

**Ablation: self-consistency probe** — Replacing E_full with two independently pruned memory states at the same timestamp is expected to yield a higher unsafe-emission rate at matched recall than the paired-contrast probe, because self-consistency measures agreement between two compressed views rather than sufficiency against a full-resolution reference; the gap isolates the contrast structure as the novel object.

**Ablation: StreamBridge trigger head** — Applying the TRACE contract to StreamBridge's activation model is expected to reduce unsafe-emission rate at matched recall relative to StreamBridge alone, demonstrating that the contract is a reusable criterion rather than a single-system patch.

**Scope-stratified analysis** — Unsafe-emission rate is stratified by utterance temporal scope (≤8s vs. >8s). We expect the gain to concentrate on the ≤8s stratum where the E_full-window sufficiency assumption holds, and to degrade on the >8s stratum where rho reflects only local-window recall.

### What would falsify this idea

If TRACE's reduction in unsafe emissions is uniform across the salient-but-unsupported, long-horizon, and out-of-window subsets rather than concentrated where the recall probe is predicted to bite, the contract is not doing the causal work and the gain is a generic emission-rate effect. Equally fatal: if the frozen-verifier ablation matches full TRACE, then L_contract is inert and the central coupling claim is false; and if the self-consistency-probe ablation matches full TRACE, the paired-contrast structure is not the novel object.

## References

1. Beyond Answering: Learning to Speak First and Act in Streaming Video
