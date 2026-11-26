# Set-Decomposition Consistency for Multi-ROI Grounding in Medical MLLMs

## Motivation

Region-level supervision in medical MLLMs is treated additively: MedTrinity-25M (E1) builds image-ROI-description triplets, MedRegA (E2) formulates region-centric tasks, and MIMO (E5) grounds terminology to pixels, yet none enforces that grounding a set of ROIs jointly decomposes into the same outputs as grounding each ROI independently. The root cause is structural: ROI annotations enter training as independent per-region targets, so the model has no gradient signal requiring the multi-ROI joint output to be the compositional closure of the singleton outputs, leaving joint grounding free to drift from the union of independently grounded regions.

## Key Insight

Compositional closure of multi-ROI grounding is an invariant that can be enforced on the output surface itself — the joint region-level prediction must equal the set-union of singleton-region predictions under a shared grounding head — so consistency becomes a measurable property of the model's own outputs rather than an assumption baked into independent per-ROI labels.

## Method

# SDC-Ground: Set-Decomposition Consistency regularizer for medical MLLM grounding
# Inputs: scan x, prompt p, ROI set R = {r_1,...,r_k} with k ~ U{1..K_max}, K_max=8
# Model: M_theta -> (text tokens, region mask logits m in [0,1]^{HxW})
# Load-bearing assumption (stated explicitly): The soft-OR union of stop-gradient singleton ROI masks is a valid, stable target for the joint multi-ROI mask, i.e., the joint grounding output can be driven to equal the union of independently grounded singleton outputs without the singleton branch and joint branch collapsing into a degenerate fixed point. This assumption is enforced by (i) supervising m_joint against ground-truth ROI masks when available, (ii) using an EMA-frozen teacher for singleton masks with a variance floor, and (iii) replacing soft-OR with an overlap-aware per-ROI assignment target.

for each training step (x, p, R, y_text, optional ground-truth masks {g_i}):
  # 1. Joint forward: ground the whole set at once
  y_joint, m_joint = M_theta(x, p, R)              # m_joint: soft mask over union
  # 2. Singleton forwards: ground each ROI independently using an EMA-frozen teacher
  #    Teacher parameters theta_ema updated as theta_ema = 0.999 * theta_ema + 0.001 * theta
  for i in 1..k:
    _, m_i = M_theta_ema(x, p, {r_i})   with stop_grad  # m_i: soft mask for r_i from EMA teacher
  # 3. Set-decomposition target: union of singleton masks (soft OR)
  m_union = 1 - prod_i (1 - m_i)                   # soft OR over singleton masks
  # 4. Compositional consistency loss (symmetric soft-Dice, tau=0.5 binarization for eval)
  L_comp = 1 - (2*sum(m_joint*m_union)+eps) / (sum(m_joint)+sum(m_union)+eps)
  # 5. Singleton re-decomposition loss: each m_i must be recoverable from m_joint
  #    via per-ROI prompt-conditioned mask heads (shared head, ROI embedding as query)
  for i in 1..k:
    m_i_hat = MaskHead(m_joint, e(r_i))             # e: ROI text embedding
    L_single += Dice(m_i_hat, m_i)
  # 6. Task losses: text CE + grounding BCE (with ground-truth when available)
  if ground-truth masks {g_i} available:
    L_task = CE(y_joint, y_text) + BCE(m_joint, g_union)  # g_union = union of g_i
  else:
    L_task = CE(y_joint, y_text) + BCE(m_joint, m_union.detach())
  # 7. Variance floor on per-ROI masks to prevent all-0/all-1 collapse
  #    For each ROI i, compute variance of m_i over spatial dimensions; penalize if below threshold
  L_var = sum_i max(0, 0.01 - Var(m_i))  # Var computed over HxW pixels
  # 8. Total
  L = L_task + lambda_comp * L_comp + lambda_single * L_single + lambda_var * L_var
  # lambda_comp=0.5, lambda_single=0.3, lambda_var=0.1, warmup 2 epochs on L_task only

**What it is.** SDC-Ground (Set-Decomposition Consistency Grounding) is a training-time regularizer plus a shared ROI-query mask head for medical MLLMs. Its inputs are a scan, a prompt, and a variable-size ROI set; its outputs are the joint region mask, per-ROI decomposed masks, and the text answer, with the loss enforcing that the joint mask equals the set-union of singleton masks and that each singleton mask is recoverable from the joint mask via an ROI query. The load-bearing assumption is stated explicitly above: the soft-OR union of stop-gradient singleton masks is a valid, stable target, enforced by ground-truth supervision when available, an EMA-frozen teacher, and a variance floor.

**Why this design.** We chose a *soft-OR set-union* target over an explicit enumeration of all 2^k subsets because the union is the minimal compositional closure that scales to K_max=8 without combinatorial blowup, accepting that it does not constrain order or overlap semantics beyond union. We chose *stop-gradient singleton forwards* over jointly optimizing singleton and joint branches in one pass because shared-gradient coupling would let the joint branch trivially satisfy consistency by degrading singleton masks, accepting a small bias from stale singleton targets. We chose a *shared ROI-query mask head* over per-ROI independent decoders because it forces the joint mask to be a sufficient statistic for each ROI, accepting that the head must learn to disambiguate overlapping ROIs from text embeddings alone. Unlike MedRegA (E2), which adds region-centric tasks as extra supervision, and MIMO (E5), which adds pixel grounding as an output modality, SDC-Ground adds no new task — it adds an *invariant on the output surface* that neither prior method enforces, so a domain expert would describe it as a consistency constraint, not a variant of region-task augmentation or pixel-grounding supervision.

**Why it measures what we claim.** `L_comp` (joint-vs-union Dice) measures *compositional closure* of multi-ROI grounding because it assumes the set-union of singleton soft masks is the correct decomposition target — an assumption that fails when ROIs overlap such that union loses per-region identity, in which case `L_comp` reflects union overlap rather than true compositional consistency. `L_single` (ROI-query mask recovery) measures *decomposability* of the joint output because it assumes the shared mask head conditioned on an ROI embedding can invert the joint mask — an assumption that fails when two ROIs share identical text embeddings, in which case `L_single` reflects embedding collision rather than decomposition fidelity. The stop-gradient on singleton forwards measures *independence of the reference decomposition* because it assumes frozen singleton masks are a stable target — an assumption that fails early in training when singleton masks are near-random, in which case consistency reflects noise rather than closure. The EMA teacher and variance floor are added to prevent the trivial all-0/all-1 fixed point, ensuring that `L_comp` and `L_single` reflect meaningful closure rather than degenerate solutions.

## Contribution

['A set-decomposition consistency regularizer (SDC-Ground) that enforces joint multi-ROI grounding outputs to equal the soft-union of independently grounded ROIs, turning region supervision from additive per-ROI targets into a compositional-closure constraint on the output surface.', 'A shared ROI-query mask head that makes the joint region mask a sufficient statistic for each singleton ROI, providing a concrete mechanism for measuring decomposability of multi-ROI grounding without adding new task labels.', 'An empirical design principle: compositional closure is enforceable as an output-surface invariant on top of existing region supervision (MedTrinity-25M, MedRegA, MIMO style) without modifying the underlying annotation pipeline.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | MedTrinity-25M ROI triplets | Multi-ROI sets with multigranular region annotations |
| Primary metric | Multi-ROI grounding Dice on held-out ROI sets | Directly measures joint-mask correctness under set inputs |
| Baseline | MedRegA | Region-aware MLLM without set-union consistency |
| Baseline | MIMO | Pixel-grounding MLLM without decomposition invariant |
| Baseline | LLaVA-Tri | Strong grounding MLLM without compositional regularizer |
| Ablation-of-ours | SDC-Ground w/o L_comp and L_single | Isolates contribution of consistency terms |
| Diagnostic | Joint-vs-union Dice gap of baselines | Establishes drift empirically before fix |
| Additional metric | MediConfusion below-random confusion rate | Links closure to failure modes |
| Compute analysis | Wall-clock overhead vs ablation | Bounds forward-pass cost |

### Why this setup validates the claim

The central claim is that adding a set-decomposition invariant on the output mask surface improves multi-ROI grounding beyond what region-task supervision (MedRegA) or pixel-grounding output supervision (MIMO) can achieve. The design makes this falsifiable by holding the dataset and task fixed and varying only the mechanism: MedRegA tests whether extra region-centric tasks alone close the gap, MIMO tests whether pixel-grounded output supervision alone closes it, and LLaVA-Tri tests whether strong single-region grounding transfers to sets. The ablation removes exactly the two consistency losses while keeping data, backbone, and task losses, so any residual gain of the full method over the ablation is attributable to the invariant rather than to capacity or data. Multi-ROI grounding Dice is the right primary metric because the claim is about the joint mask over a set, not about text or single-ROI accuracy; a metric computed on variable-size ROI sets directly exposes whether the joint output is compositionally closed and decomposable. If the invariant matters, the gap should concentrate on multi-ROI cases with overlapping or many regions, not on singletons. Additionally, we include a diagnostic experiment measuring the joint-vs-union Dice gap of MedRegA, MIMO, and LLaVA-Tri on held-out multi-ROI sets to empirically establish the claimed drift, and we evaluate on MediConfusion to show that SDC-Ground reduces the below-random confusion rate, linking compositional closure to real-world failure modes.

### Expected outcome and causal chain

**vs. MedRegA** — On a case where a prompt names three overlapping ROIs in one scan, MedRegA produces a joint mask that drifts toward the most visually salient region because its region-centric training never constrains the joint output to equal a union of singleton predictions. Our method instead supervises the joint mask against the soft-OR of frozen singleton masks and forces ROI-query recovery, so the joint output must remain a sufficient statistic for each region. We expect a noticeable Dice gap on multi-ROI overlapping subsets, with near-parity on single-ROI cases where the invariant is vacuous. The diagnostic experiment will first confirm that MedRegA exhibits a joint-vs-union Dice gap on held-out multi-ROI sets, establishing the drift empirically.

**vs. MIMO** — On a case where visual referring input must ground several terminologies at once, MIMO produces pixel-grounded outputs that are individually plausible but mutually inconsistent, because its supervision treats each grounding target as independent and never enforces set-union closure. Our method instead ties the joint mask to the union of singleton masks and re-decomposes it per ROI, so inconsistencies between regions are penalized directly. We expect the gain to concentrate on high-k and overlapping-ROI subsets, with little change on basic single-target grounding. The diagnostic experiment will quantify MIMO's joint-vs-union Dice gap to confirm the inconsistency.

**vs. LLaVA-Tri** — On a case where a strong single-region grounder is asked to ground a set of four ROIs, LLaVA-Tri produces a joint mask dominated by one region or smeared across the union without per-region identity, because its pretraining never exposed it to variable-size set inputs and no consistency term corrects this. Our method instead regularizes the joint output toward the singleton union and enforces recoverability through the shared ROI-query head, so the model must preserve per-region structure. We expect a clear gap that grows with k, largest at k≈6–8 and overlapping sets. The diagnostic experiment will measure LLaVA-Tri's joint-vs-union Dice gap to establish the baseline drift.

### What would falsify this idea

If the full method's gain over the ablation is uniform across singleton and multi-ROI subsets rather than concentrated on overlapping and high-k sets, the central claim that the consistency invariant drives the improvement is wrong. Likewise, if L_single recovery accuracy stays near chance while joint Dice still improves, the decomposability mechanism is not what produces the effect.

## References

1. Sage-Med: Scaling Medical Multimodal Intelligence
