# Constitution: Prompt-Conditioned Audio Lane

## Identity
This is the practical audio-control lane. Its job is not to solve cognition. Its job is to make the generator respond to prompts without collapsing.

## Thesis
A hybrid conditioning path can perturb the chamber strongly enough to create prompt-sensitive audio while preserving the audio engine.

## What This Lane Exists to Prove
- prompt information can affect generated audio at all
- conditioning can be added without destroying the stable voice
- hybrid control is stronger than pure IFS nudging alone

## Core Allowed Modules
- hybrid FiLM + IFS conditioning
- diffusion output path
- paired local manifest prompts

## Central Evaluator
- Path-B scorecard
- `avg_coupling_score`
- `avg_state_effect`
- collapse rate

## Best Current Evidence
`prompt_conditioned_hybrid_smoke`
- `avg_coupling_score = 0.0003083`
- `avg_state_effect = 0.0021417`
- `collapse_rate_pct = 0.0`

## Success Signature
- prompt-conditioned run beats pure IFS prompt control on coupling/state effect
- no collapse

## Failure Signature
- conditioning optimizes itself away
- prompt A/B stays indistinguishable
- coupling collapses toward zero while audio remains fine

## What This Lane Is Not Allowed to Claim
- semantics are solved
- logic is solved
- CLAP-like vectors are the right long-term steering language

## Constitutional Verdict
Working as a weak but real control lane. It is a pragmatic audio lane, not yet a semantic one.
