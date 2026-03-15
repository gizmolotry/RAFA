# Conditioned Depth Failure Analysis (2026-03-12)

## Run
- Canonical run: `conditioned_depth_prompt_anchor20_ns6_a0_to120`
- Training summary: `step=120`, `epoch=001`, `train_loss=1.00037`, `val_loss=0.67784`, `sec=2795.9`
- Checkpoint: `D:\RAFA\checkpoints_diffusion_conditioned_depth_prompt_anchor20_ns6_a0\diff_ep1.pt`

## Core Comparison
Baseline probe summary:
- File: `D:\RAFA\research_track\infra\summaries\logic_diag_focus_prompt_anchor20_a0_ic0_ns6.json`
- `num_success = 2 / 12`
- `success_rate = 0.1667`
- `mean_final_rank = 4.8333`
- `mean_persistence = 0.1611`
- `mean_final_margin = -0.1820`
- `mean_energy_gain = 0.0163`

New trained checkpoint summary:
- File: `D:\RAFA\research_track\infra\summaries\conditioned_depth_prompt_anchor20_ns6_a0_to120.json`
- `num_success = 0 / 12`
- `success_rate = 0.0000`
- `mean_final_rank = 4.0833`
- `mean_persistence = 0.1222`
- `mean_final_margin = -0.2599`
- `mean_energy_gain = 0.0470`

Interpretation:
- Average rank improved slightly.
- Persistence got worse.
- Margin got worse.
- Energy gain increased by about `2.9x`.
- The run produced more motion, not more stable relational binding.

## Why Depth Alone Failed
Config for this run:
- `h0_anchor_enabled: false`
- `intermediate_consistency_enabled: false`
- `use_ifs_intermediate_consistency: false`
- File: `D:\RAFA\tmp\conditioned_depth_configs\conditioned_depth_prompt_anchor20_ns6_a0_to120.yaml`

Interpretation:
- This run changed recurrence depth to `6`, but added no anchor pressure and no intermediate consistency leash.
- It was still optimized by the diffusion/audio objective, not by a direct relational objective.

## Regression Case: 2x3 -> 6
Old checkpoint artifact:
- `D:\RAFA\research_track\logic_results\logic_logic_diag_focus_prompt_anchor20_a0_ic0_ns6_2x3_t6_s1337.json`
- `success = true`
- `final_rank = 1`
- `final_margin = 0.1322`
- `persistence_score = 0.6667`
- Early target energy at step `0`: `0.0087`
- Final target energy at step `24`: `0.0454`
- Final top distractor energy at step `24` was `10 -> 0.0401`

New checkpoint artifact:
- `D:\RAFA\research_track\logic_results\logic_conditioned_depth_prompt_anchor20_ns6_a0_to120_2x3_t6_s1337.json`
- `success = false`
- `final_rank = 5`
- `final_margin = -0.9011`
- `persistence_score = 0.0`
- Early target energy at step `0`: `0.0099`
- Final target energy at step `24`: `0.0109`
- Final dominant distractor energy at step `24` was `10 -> 0.1099`

Interpretation:
- The new checkpoint did not merely weaken the target.
- It redirected mass toward a distractor (`q=10`) and never recovered.
- This is a selectivity collapse.

## False-Positive Case: 4x6 -> 12
Old checkpoint artifact:
- `D:\RAFA\research_track\logic_results\logic_logic_diag_focus_prompt_anchor20_a0_ic0_ns6_4x6_t12_s1337.json`
- `success = true`
- `final_rank = 1`
- `final_margin = 0.0605`
- `persistence_score = 0.7333`

New checkpoint artifact:
- `D:\RAFA\research_track\logic_results\logic_conditioned_depth_prompt_anchor20_ns6_a0_to120_4x6_t12_s1337.json`
- `success = false`
- `final_rank = 1`
- `final_margin = 1.5466`
- `persistence_score = 0.2667`

Interpretation:
- The target wins strongly at the end.
- But it wins too late and for too little of the trajectory.
- This is a stability failure, not a search failure.

## Near-Miss Pattern
Tasks with `final_rank = 1` but `success = false`:
- `2x6 -> 6`: margin `0.2910`, persistence `0.5333`
- `3x4 -> 12`: margin `0.4135`, persistence `0.2667`
- `4x6 -> 12`: margin `1.5466`, persistence `0.2667`

Interpretation:
- The model can still settle onto the right target.
- It just does not do so early enough or consistently enough to count as a solve.
- The mechanism looks underconstrained rather than absent.

## Failure Signature
The trained `ns6/a0` continuation appears to have shifted the system into this regime:
- higher latent energy movement
- weaker target-vs-distractor separation on average
- less persistent occupancy of correct targets
- more late-stage target spikes without stable commitment

Short version:
- `depth added motion`
- `depth did not add binding`

## Practical Conclusion
The result does not support `depth alone` as the active ingredient.
The metrics support this narrower claim:
- deeper recurrence can produce stronger late target activation on some tasks
- without anchor or consistency pressure, that activation is not reliably stabilized
- the diffusion continuation objective can degrade prior relational wins while still training normally on audio loss
