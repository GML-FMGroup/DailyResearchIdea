# Profile-Anchor Consistency Evaluation for Role-Playing Agents

## Motivation

Existing role-playing evaluation benchmarks (e.g., RPEval) rely on static, predefined dimensions and fixed prompts, which fail to capture long-term character consistency across multi-turn interactions. The root structural limitation is that they treat consistency as a static property rather than a dynamic alignment with the character profile, making them unable to detect gradual drift or subtle inconsistencies that emerge over multiple turns. This is particularly problematic for evaluating LLM-based role-playing agents, which may exhibit high initial fidelity but degrade over time.

## Key Insight

Character profiles serve as an invariant anchor across turns, enabling consistency to be quantified as the information-theoretic divergence between the agent's response distribution and the distribution conditioned on the profile.

## Method

## Profile-Anchor Consistency Evaluation (PACE)

**(A) What it is:** PACE is a model-independent evaluation framework that measures long-term character consistency by computing the cross-entropy between the agent's responses and predictions from a profile-conditional language model (PCLM). Inputs: agent's dialogue history (sequence of turns) and a structured character profile. Output: per-turn consistency scores and an overall score (e.g., mean and variance).

**(B) How it works:** We train a small transformer (e.g., 300M parameters) as the PCLM on a diverse corpus of character-specific dialogues. The profile is encoded as a prefix with a special token; the model uses causal attention. For each turn *t* in the evaluation dialogue, we feed the profile and the dialogue history up to turn *t*-1 into the PCLM to compute the negative log-likelihood of the agent's response *r_t*:

```python
def pace_evaluate(agent_dialogue, profile, pclm, calib_model=None):
    scores = []
    attention_scores = []
    history = ""  # concatenated previous turns
    for turn in agent_dialogue:
        # Convert profile and history to input string
        input_text = f"[PROFILE] {profile} [HISTORY] {history}"
        # Tokenize and run PCLM
        outputs = pclm(input_text, labels=tokenizer(turn.response))
        loss = outputs.loss  # cross-entropy for the response
        scores.append(loss.item())
        # Compute mean attention over profile token positions from last layer
        attn = outputs.attentions[-1].mean(dim=1)[0]  # shape: (1, seq_len) after mean over heads
        profile_token_positions = torch.where(tokenizer(input_text).input_ids == tokenizer.profile_token_id)[0]
        attention_scores.append(attn[:, profile_token_positions].mean().item())
        # Append the agent's response to history
        history += turn.response
    # Calibration (if calib_model is provided)
    if calib_model:
        cal_scores = calib_model.predict(np.array(scores).reshape(-1, 1))
        scores = cal_scores.tolist()
    return {'per_turn_consistency': scores, 'overall_consistency': np.mean(scores),
            'profile_adherence': attention_scores}
```

Hyperparameters: PCLM is a 12-layer, 12-head transformer with 768 hidden dimensions (≈300M params). Trained with AdamW, learning rate 1e-4, batch size 64, for 10 epochs on a dataset of 100K character-dialogue pairs. A calibration model (if used) is a linear regression trained on a held-out set of 512 human-rated examples to map raw NLL to calibrated consistency scores in [0,1].

**(C) Why this design:** We chose a probabilistic generative model (PCLM) over a heuristic similarity metric (e.g., cosine similarity of embeddings) because it captures the character-specific language distribution rather than surface-level word overlap, accepting the cost of requiring a training corpus for each character type. We selected cross-entropy as the consistency measure instead of BLEU or ROUGE because it provides a principled, information-theoretic score that directly reflects how well the agent's response aligns with the profile-conditioned distribution; the trade-off is that cross-entropy can be sensitive to rare but valid completions. We included an attention-based profile adherence score to capture whether the model explicitly references the profile during generation, at the cost of additional computation and potential noise from attention distribution; this provides a complementary signal to the cross-entropy score. We opted for a small model (300M) rather than a large LLM to enable fast, reproducible, and API-independent evaluation, sacrificing some accuracy for practicality.

**(D) Why it measures what we claim:** The cross-entropy `-log p(response|profile,history)` measures character consistency because it quantifies the surprise of the agent's response under a model that has learned the conditional distribution of utterances given the character profile; this equivalence holds under the assumption that the PCLM is a good approximation of the true character-specific dialogue distribution. This assumption fails when the PCLM is undertrained or the character profile is ambiguous, in which case the score reflects the model's uncertainty rather than true consistency. To mitigate this, we optionally calibrate using a held-out set of human ratings, which empirically adjusts scores to better align with human judgment. The profile attention score measures explicit use of profile during response generation; this assumes that attention weights correlate with reliance on profile information, which is a common but imperfect assumption; failure occurs when the model attends to profile but does not use it meaningfully (e.g., due to position bias). Together, these two scores provide complementary views of consistency: cross-entropy captures overall alignment, while attention scores capture explicit reference use.

## Contribution

(1) A novel evaluation framework PACE that uses a profile-conditional language model to dynamically measure character consistency over multi-turn dialogues, overcoming the static limitations of existing benchmarks like RPEval. (2) An empirical finding that cross-entropy under a profile-conditioned model correlates well with human judgments of role-consistency, validated on multiple role-playing agents. (3) A lightweight (300M parameter) profile-conditional language model trained on a diverse corpus of character dialogues, along with an analysis of attention-based profile adherence as a complementary metric.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|----------|
| Dataset | CharacterDial, PersonaChat, and a custom set of 200 movie character dialogues with human ratings | CharacterDial provides long dialogues with profiles and human ratings; PersonaChat adds chit-chat; movie characters add variety. All have human consistency ratings (Likert 1-5) from 3 annotators. |
| Primary metric | Spearman correlation with human consistency | Measures alignment with perceived consistency. |
| Baseline1 | Unconditional LM perplexity | Ignores profile; fails on style shifts. Uses same PCLM without profile prefix. |
| Baseline2 | Embedding cosine similarity (SBERT sentence embeddings) | Captures topic not style. |
| Baseline3 | PACE with large PCLM (fine-tuned LLaMA-7B) | Assesses impact of PCLM quality; uses same cross-entropy calculation but with LLaMA. |
| Ablation-of-ours | PACE without attention score | Isolates contribution of profile attention. |

### Why this setup validates the claim
This setup tests whether PACE better predicts human judgments of character consistency than simpler baselines across multiple datasets. CharacterDial, PersonaChat, and movie character dialogues provide ground-truth ratings, enabling direct correlation comparison. The baselines represent common evaluation approaches: perplexity reflects fluency but not profile alignment, and cosine similarity captures semantic similarity but not stylistic distribution. The large PCLM baseline (LLaMA-7B) tests the sensitivity to PCLM quality. The ablation removes the attention-based profile adherence score to test its added value. If PACE achieves higher Spearman correlation than all baselines and the ablation across all datasets, it demonstrates that profile-conditioned cross-entropy and attention provide complementary signals that align with human perception, thereby validating the claim that probabilistic modeling of character dialogue distributions is a principled and effective consistency metric.

### Expected outcome and causal chain

**vs. Unconditional LM perplexity** — On a case where a knight character suddenly uses modern slang (e.g., "yo that dragon was lit"), the unconditional LM assigns low perplexity because the sentence is fluent common English, yet humans rate it highly inconsistent. Our PACE, conditioning on the knight profile, assigns high cross-entropy due to the improbable slang under the archaic speech distribution. We expect a large gap (ρ ≈ 0.7 for PACE vs. ≈ 0.3 for perplexity) on stylistic mismatches, but parity on generic utterances.

**vs. Embedding cosine similarity** — On a case where a wise wizard discusses ancient lore but uses a cheerful modern tone (e.g., "Haha, that spell is super cool!"), cosine similarity between response and profile embeddings may be high due to topic overlap, but humans detect tone inconsistency. PACE penalizes the tonal mismatch via high cross-entropy because the profile conditions on formal, serious language. We expect PACE to have higher correlation (ρ ≈ 0.6 vs. ≈ 0.4) on such cases, especially those with tone misalignment.

**vs. PACE with large PCLM (fine-tuned LLaMA-7B)** — On cases where the small PCLM is undertrained (e.g., rare characters), the large PCLM may assign more accurate NLL, leading to higher correlation with human ratings. We expect the small PCLM with calibration to perform comparably (ρ ≈ 0.65 vs. ≈ 0.70), demonstrating that calibration mitigates PCLM quality issues.

**Ablation (PACE without attention score)** — On a case where the agent explicitly references profile details (e.g., "As a dutiful knight, I swear loyalty"), the full PACE captures this via high attention to profile, boosting consistency score. The ablation (cross-entropy only) may give moderate score because the response is still somewhat likely under profile, but misses the explicit adherence signal. We expect full PACE to show higher correlation (ρ ≈ 0.7 vs. ≈ 0.6) on instances with overt profile references.

### What would falsify this idea
If PACE's correlation with human ratings is not significantly higher than all baselines (e.g., p > 0.05 for differences across datasets), or if the ablation performs equally to the full method across all subsets, then the proposed profile conditioning and attention mechanism do not capture true consistency, falsifying the central claim.

## References

1. Thinking in Character: Advancing Role-Playing Agents with Role-Aware Reasoning
2. Role-Playing Evaluation for Large Language Models
3. A Multi-Task Role-Playing Agent Capable of Imitating Character Linguistic Styles
4. Identity-Driven Hierarchical Role-Playing Agents
5. LLMs + Persona-Plug = Personalized LLMs
6. Knowledge-Augmented Large Language Models for Personalized Contextual Query Suggestion
7. RETA-LLM: A Retrieval-Augmented Large Language Model Toolkit
8. BlenderBot 3: a deployed conversational agent that continually learns to responsibly engage
