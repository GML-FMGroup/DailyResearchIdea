# Goal-Identifiability Harness: Belief-State Goal Representations with Distinguishability-Tested Action Distributions for Frontier-Model-as-Policy

## Motivation

Frontier-model-as-policy harnesses such as Embo-Harness achieve strong closed-loop success by feeding observations and instructions to a multimodal model and reading out actions, but they inherit the foundation-model-as-encoder assumption that the current observation (or a memory bank as in SAM2Act) is a sufficient statistic for the latent task goal. When the goal is only partially observable — occluded target objects, goals specified by earlier observations that have since decayed from memory — distinct latent goals can map to the same observation and therefore to the same action distribution, so the harness cannot distinguish them and executes the wrong task; Lift3D's positional mapping and RDT-1B's diffusion action space share this same sufficiency assumption, which is why the failure recurs across branches rather than being one tree's local bug.

## Key Insight

Identifiability of the latent goal can be enforced as a testable separation property on the policy's action distribution — requiring that distinct goal-belief states induce distinguishable action distributions — which turns an untested sufficiency assumption into an explicit, optimizable belief-state invariant that the harness must maintain under occlusion and memory decay.

## Method

**What it is.** GIH (Goal-Identifiability Harness) augments a frontier-model-as-policy loop with an explicit goal-belief state `b_t` over a discrete set of candidate latent goals, updated from observations and a bounded memory buffer, plus a distinguishability test that penalizes the policy whenever two distinct beliefs produce statistically indistinguishable action distributions. Inputs: observation `o_t`, instruction `l`, memory buffer `M_t`, candidate goal set `G`. Outputs: action `a_t`, updated belief `b_t`, and a distinguishability loss used at harness-refinement time.

**Load-bearing assumption (stated explicitly).** The method assumes that (i) the per-goal likelihood used in the Bayes update is calibrated enough that `b_t` tracks the true latent goal rather than the frontier model's prior, and (ii) the JS divergence between counterfactual action distributions conditioned on one-hot goal prompts measures latent-goal distinguishability rather than prompt-format sensitivity. Both are treated as testable and are checked empirically as described in Phase 2b and Phase 4b; the loss is gated on these checks.

**How it works.**

```pseudocode
# Phase 1: goal-belief initialization from instruction + first observation
def init_belief(o_0, l, G):
    # frontier model scores each candidate goal against instruction+observation
    s_g = frontier_score(l, o_0, g)  for g in G       # logits, temperature tau=1.0
    b_0 = softmax(s_g / tau)                            # belief over G
    return b_0

# Phase 2: belief update from observation + bounded memory (horizon H=8 steps)
def update_belief(b_{t-1}, o_t, M_t):
    # likelihood of o_t under each candidate goal, conditioned on memory
    lik_g = frontier_likelihood(o_t | g, M_t)  for g in G
    b_t   = normalize(b_{t-1} * lik_g)                 # Bayes update, no learned head
    M_t   = push(M_{t-1}, o_t); if len(M_t) > H: pop_oldest(M_t)
    return b_t, M_t

# Phase 2b: calibration gate for the likelihood (fit on held-out goal-labeled episodes)
#   Fit temperature T_goal by minimizing NLL of the true goal on a calibration set of
#   N_cal = 512 held-out episodes from RoboLab-120 with known goal labels.
#   Compute expected calibration error (ECE) with 10 equal-frequency bins on the same set.
#   If ECE > 0.10, replace frontier_likelihood with a small discriminative goal-classifier
#   head: 2-layer MLP (hidden=256, GeLU) trained on frozen frontier embeddings of (o_t, M_t)
#   to predict g in G, trained on the same N_cal episodes with cross-entropy loss.
#   The gate is applied per task family; the chosen likelihood is logged.

# Phase 3: belief-conditioned action generation
def act(o_t, b_t, M_t):
    # belief is injected as an explicit textual/numeric prefix to the prompt
    prompt = render(b_t, M_t, o_t)
    a_t    = frontier_policy(prompt)                   # action chunk, size K=8
    return a_t, prompt

# Phase 4: distinguishability test (harness refinement)
def distinguishability_loss(b_t, a_dist):
    # sample pairs of distinct candidate goals (g_i, g_j) with b_t[i],b_t[j] > eps=1e-3
    pairs = [(i,j) for i<j if b_t[i]>eps and b_t[j]>eps]
    L = 0
    for (i,j) in pairs:
        # counterfactual action distributions conditioned on each goal
        p_i = frontier_policy_dist(render(onehot(i), M_t, o_t))
        p_j = frontier_policy_dist(render(onehot(j), M_t, o_t))
        # separation: JS divergence must exceed margin delta=0.1
        L += max(0, delta - JS(p_i, p_j))
    return L / max(len(pairs),1)

# Phase 4b: interventional validation of the separation metric (run once per task family)
#   For each goal pair (i,j), measure JS(p_i, p_j) on a held-out set of N_val = 256
#   ambiguous-subset episodes where the true goal is known. Compute the point-biserial
#   correlation between JS separation and goal-conditional success (binary).
#   If correlation < 0.2, the pair is marked 'unvalidated' and its contribution to L is
#   set to 0 (loss gated off). Report the fraction of pairs that pass.
#   Also compute per-step (K=1) JS on the same pairs to check that the K=8 chunker is not
#   the source of separation; if per-step JS is near zero while chunk JS is high, flag
#   chunk-smoothing as the cause and report it.

# Training: harness refinement minimizes task loss + lambda * distinguishability_loss (lambda=0.5)
#   Refinement cost bound: sample at most one goal pair per step; cache counterfactual
#   distributions across steps (reuse if (o_t, M_t, g) hash matches). Report wall-clock
#   and API-token cost per refinement iteration.
```

**Why this design.** We chose a Bayes-updated belief over a discrete candidate goal set rather than a learned recurrent memory bank (as in SAM2Act) because a Bayes update gives an explicit, inspectable posterior whose identifiability can be tested directly, accepting the cost that the candidate set `G` must be enumerated per task family. We chose to inject the belief as an explicit prompt prefix rather than fine-tuning the frontier model's weights because the harness is model-agnostic and the belief must remain legible to the distinguishability test, accepting the cost that prompt length grows with `|G|`. We chose a counterfactual JS-divergence separation margin rather than a learned independence head because the separation is defined on the action distribution the policy actually emits, so it cannot be satisfied by a decoupled auxiliary head that the policy ignores, accepting the cost that we must sample action distributions for goal pairs at refinement time (compute overhead). A domain expert would not describe this as a variant of Embo-Harness's prompting recipe: Embo-Harness optimizes prompt/planning choices for average success, whereas GIH adds a testable separation constraint on the action distribution that Embo-Harness has no mechanism to express or enforce.

**Why it measures what we claim.** The posterior `b_t` measures the latent-goal belief because the Bayes update assumes the observation likelihood `frontier_likelihood(o_t | g, M_t)` is calibrated per candidate goal; this assumption fails when the frontier model's likelihood is systematically overconfident for visually similar goals, in which case `b_t` reflects model prior rather than evidence. We test this assumption via the calibration gate in Phase 2b (ECE on N_cal=512 held-out episodes; if ECE>0.10, a small discriminative head replaces the raw likelihood). The JS-divergence separation `JS(p_i, p_j)` measures goal distinguishability because it assumes the counterfactual action distributions `p_i`, `p_j` are computed from the same policy with only the goal conditioning changed; this assumption fails when the action chunker (K=8) smooths over the goal-relevant sub-step, in which case the metric reflects chunk-level averaging rather than goal separation. We test this via the per-step (K=1) JS check in Phase 4b. The memory horizon `H=8` measures memory decay because it assumes observations older than `H` steps carry no goal-relevant information; this assumption fails when the goal is specified by a landmark seen early and never revisited, in which case the belief collapses to the prior and the test reports low separation for a recoverable goal. We report the frequency of this failure mode on RoboLab-120.

## Contribution

(1) A goal-identifiability harness (GIH) that augments a frontier-model-as-policy loop with an explicit Bayes-updated goal-belief state and a counterfactual JS-divergence distinguishability loss over action distributions. (2) A design principle: identifiability of the latent goal under partial observation is enforceable as a separation margin on the policy's own action distribution, not as an auxiliary independence head. (3) A diagnostic protocol that reports belief-separation and goal-recovery rates under controlled occlusion and memory-horizon decay.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | RoboLab-120 plus real-world manipulation tasks | Matches frontier-as-policy benchmark for direct comparison |
| Primary metric | Goal-conditional success on ambiguous-goal subset | Directly detects belief separation, not average success |
| Baseline | Embo-Harness recipe | Strongest frontier-as-policy baseline; no separation mechanism |
| Baseline | SAM2Act+ memory policy | Tests whether learned memory substitutes for explicit belief |
| Baseline | RDT-1B diffusion policy | Tests whether action-scale pretraining absorbs goal ambiguity |
| Ablation-of-ours | GIH without distinguishability loss | Isolates the separation constraint from the belief state |
| Diagnostic (new) | JS separation measured on Embo-Harness and SAM2Act+ without modification | Tests whether the distinguishability test exposes failures in existing harnesses |
| Calibration check (new) | ECE of frontier_likelihood on N_cal=512 held-out goal-labeled episodes | Quantifies overconfidence failure mode; gates belief update |
| Sub-chunk check (new) | Per-step (K=1) JS separation on the same goal pairs | Verifies chunker is not the source of separation |

### Why this setup validates the claim

The central claim is that an explicit, testable goal-belief state improves frontier-as-policy behavior precisely on tasks where the latent goal is ambiguous from the instruction and early observations. RoboLab-120 is the right dataset because it is the benchmark on which Embo-Harness's prompting recipe is already tuned, so any gain cannot be attributed to better prompting; we add a real-world subset to check that the effect is not a simulator artifact. Embo-Harness tests the sub-claim that average-success prompt optimization is insufficient: if GIH beats it only where goal ambiguity is high, the separation constraint is doing work that prompting cannot. SAM2Act+ tests the sub-claim that a learned recurrent memory bank is not a substitute for an inspectable posterior, since it has memory but no explicit belief over enumerated goals. RDT-1B tests whether a large diffusion policy's action prior already resolves ambiguity, which would make the belief state redundant. The ablation is the load-bearing control: it keeps the Bayes belief but removes the JS-separation loss, so if the ablation matches full GIH, the gain comes from belief injection alone and the distinguishability test is decorative. Goal-conditional success on the ambiguous subset is the right primary metric because average success can rise from unrelated robustness improvements; conditioning on the subset where two candidate goals are both plausible forces the metric to reflect whether the policy actually commits to the correct goal. The calibration check (ECE on N_cal=512 episodes) directly tests the load-bearing assumption that frontier_likelihood is usable as evidence; if ECE>0.10, the discriminative head is used and we report how often this occurs. The sub-chunk check (per-step K=1 JS) tests the assumption that the K=8 chunker preserves goal-relevant separation; if per-step JS is near zero while chunk JS is high, we report chunk-smoothing as the cause. The diagnostic on Embo-Harness and SAM2Act+ tests whether the JS separation metric is a general diagnostic that exposes failures in existing harnesses, establishing community-wide relevance beyond our own method.

### Expected outcome and causal chain

**vs. Embo-Harness recipe** — On a case where the instruction names a target that is visually similar to a distractor and the first frame is ambiguous, Embo-Harness produces a plausible but wrong-target action chunk because its recipe optimizes prompt/planning choices for average success and has no representation of competing goal hypotheses to separate. GIH instead maintains a posterior over candidate goals, injects it as a prompt prefix, and its refinement loss penalizes action distributions that fail to separate the two candidates, so we expect a noticeable gap on the ambiguous-goal subset with rough parity on unambiguous tasks. The diagnostic JS measurement on Embo-Harness is expected to show low separation on the same ambiguous pairs, confirming the failure mode.

**vs. SAM2Act+ memory policy** — On a case where an early landmark disambiguates the goal and the robot must hold that commitment across several steps, SAM2Act+ produces drift back toward the prior because its memory bank stores observations but has no explicit posterior whose identifiability is checked, so a smoothed memory representation can blur the distinguishing cue. GIH instead keeps a Bayes posterior over enumerated goals with a bounded horizon, so the disambiguating likelihood persists in `b_t`, and we expect a gap concentrated on memory-dependent ambiguous tasks rather than uniform gains. The diagnostic JS measurement on SAM2Act+ is expected to show low separation on these memory-dependent pairs.

**vs. RDT-1B diffusion policy** — On a case where two goals share most of the action trajectory and diverge only at a late sub-step, RDT-1B produces a chunk that averages the two modes because its diffusion prior is trained for broad manipulation competence, not for separating task-specific goal hypotheses. GIH instead conditions the prompt on the posterior and its separation loss explicitly pushes the counterfactual action distributions apart, so we expect the gap to appear on late-divergence tasks and to shrink when the ablation without the separation loss is compared. The per-step (K=1) JS check is expected to show that separation is present at the sub-step level, ruling out chunk-smoothing as the source.

### What would falsify this idea

If GIH's gain over Embo-Harness is uniform across ambiguous and unambiguous subsets rather than concentrated where goal ambiguity is predicted, or if the no-separation ablation matches full GIH, then the explicit belief and distinguishability test are not doing the claimed work and the central claim is wrong. Additionally, if the calibration check shows ECE>0.10 and the discriminative head does not restore calibration, or if the interventional validation shows JS separation does not predict goal-conditional success (correlation < 0.2), then the distinguishability test is measuring prompt-format sensitivity rather than latent-goal separation and the method's load-bearing assumption fails.

## References

1. Embo-Harness: Towards Understanding Frontier-Model-as-Policy Harness
