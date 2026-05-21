# RAFA Causal Operator Routing Multihead Depth Push - 2026-05-15

## Scope

This cycle executed the next three causal-router steps:

1. Add a shared causal-selector module and multi-head selector contract.
2. Train/test causal labels on deeper grandchild substrates and an alternate parent ontology config.
3. Verify that learned causal residuals can prefer branch-authorized operators over harder decoys that move the parent more but worsen world-jump risk.

All changes remain assay-only. Circleworld production runtime and checkpoint promotion are unchanged.

## Code Changes

- `D:\RAFA\runtimes\circleworld_proto\causal_operator_selector.py`
  - Added shared `CausalOperatorSelector`.
  - Added `MultiHeadCausalOperatorSelector`.
  - Added shared pair feature contract: `causal_pair_feature_vector()` and `causal_pair_feature_dim()`.
  - Added `selector_route_scores()` so single-head and multi-head checkpoints can be consumed by the same assay.

- `D:\RAFA\runtimes\circleworld_proto\run_causal_operator_selector_probe.py`
  - Uses shared causal selector module instead of duplicated local class.
  - Added `--model-kind single|multihead`.
  - Multi-head training predicts:
    - route score
    - movement
    - jump safety
    - identity compatibility

- `D:\RAFA\runtimes\circleworld_proto\run_resonant_operator_causality_assay.py`
  - Causal selector checkpoint loader now supports single-head and multi-head checkpoints.
  - Reports multi-head movement, jump-safety, and identity predictions when available.

- `D:\RAFA\runtimes\circleworld_proto\test_resonant_attention_contract.py`
  - Added multi-head output contract test.

## Run 1: Depth-2 Multi-Head Causal Selector

Artifact:

- `D:\RAFA\outputs\circleworld_proto\causal_operator_selector_2026-05-15_naked_gpu_9100_9107_d2_multihead\CAUSAL_OPERATOR_SELECTOR_PROBE.md`
- `D:\RAFA\outputs\circleworld_proto\causal_operator_selector_2026-05-15_naked_gpu_9100_9107_d2_multihead\causal_operator_selector_probe.json`
- `D:\RAFA\outputs\circleworld_proto\causal_operator_selector_2026-05-15_naked_gpu_9100_9107_d2_multihead\causal_operator_selector_probe.pt`

Setup:

- Config: `parent_ontology_trained_phasor_authority_carrier_only_gate55_static_mode_replace`
- Grandchild depth: `2`
- Seeds: `9100-9107`
- Candidate rows: `880`
- Model kind: `multihead`

Metrics:

- Status: `pass_causal_operator_selector_probe`
- Final loss: `0.00017972060387754546`
- Final route loss: `0.000042825809422148655`
- Final movement loss: `0.00035067041920098873`
- Final jump-safety loss: `0.0003123854361807129`
- Final identity loss: `0.000021418127809218796`
- Mean holdout membrane target score: `0.756770656362291`
- Mean holdout target-top score: `0.7619132830734119`
- Mean holdout identity pass: `0.9090909090909091`
- Mean holdout same-local family: `0.5454545454545454`
- Mean holdout agreement with recurrence: `0.5113636363636364`

Interpretation:

Depth-2 generates a richer routing problem than depth-1: `11` law objects per case instead of `7`, and `110` candidate rows per held-out seed instead of `42`. The multi-head selector stays identity-compatible while learning separate movement, jump-safety, and identity heads.

## Run 2: Depth-2 Held-Out Residual Control

Artifact:

- `D:\RAFA\outputs\circleworld_proto\resonant_operator_causality_2026-05-15_naked_gpu_9108_9110_d2_multihead_residual\RESONANT_OPERATOR_CAUSALITY_ASSAY.md`
- `D:\RAFA\outputs\circleworld_proto\resonant_operator_causality_2026-05-15_naked_gpu_9108_9110_d2_multihead_residual\resonant_operator_causality_assay.json`

Setup:

- Train seeds: `9100-9107`
- Held-out seeds: `9108-9110`
- Grandchild depth: `2`
- Causal membrane preference: `off`
- Residual remains membrane-gated, but candidate ranking is not forcibly sorted by membrane first.

Metrics:

- Selected/causal parent divergence: `0.1816104125333835`
- Decoy parent divergence: `0.19017025714780425`
- Selected/causal world-jump proxy: `0.016863268703669793`
- Decoy world-jump proxy: `0.02171658817402315`
- Causal model score: `0.7804904580116272`
- Causal movement pred: `0.7475987474123637`
- Causal jump-safety pred: `0.5325419505437216`
- Causal identity pred: `0.999982476234436`
- Causal residual: `0.05`
- Causal same-local family: `1.0`
- Causal agreement with recurrence: `1.0`

Interpretation:

This is the harder decoy case we wanted. The wrong-family decoy moves the parent more, but it has worse world-jump proxy. The causal selector chooses the identity-compatible operator anyway. That means the learned route is now aligned with safe branch authority, not maximum parent motion.

## Run 3: Alternate Config Depth-2 Multi-Head Check

Artifact:

- `D:\RAFA\outputs\circleworld_proto\causal_operator_selector_2026-05-15_childphase_9100_9105_d2_multihead\CAUSAL_OPERATOR_SELECTOR_PROBE.md`
- `D:\RAFA\outputs\circleworld_proto\causal_operator_selector_2026-05-15_childphase_9100_9105_d2_multihead\causal_operator_selector_probe.json`
- `D:\RAFA\outputs\circleworld_proto\resonant_operator_causality_2026-05-15_childphase_9106_9107_d2_multihead_residual\RESONANT_OPERATOR_CAUSALITY_ASSAY.md`

Setup:

- Config: `parent_ontology_childphase_gate55_static_mode_replace`
- Train seeds: `9100-9105`
- Held-out seeds: `9106-9107`
- Grandchild depth: `2`
- Model kind: `multihead`

Training metrics:

- Candidate rows: `660`
- Final loss: `0.00022401182892887542`
- Final route loss: `0.00004040978334766502`
- Final movement loss: `0.00043349372087201726`
- Final jump-safety loss: `0.0004600728037379061`
- Final identity loss: `0.000024443694504346542`
- Mean holdout membrane target score: `0.7551642432142532`
- Mean holdout target-top score: `0.7619135470211008`
- Mean holdout identity pass: `0.9090909090909091`
- Mean holdout agreement with recurrence: `0.393939393939394`

Held-out residual metrics:

- Causal parent divergence: `0.1816117681765071`
- Decoy parent divergence: `0.19017172459560494`
- Causal world-jump proxy: `0.016863463350331842`
- Decoy world-jump proxy: `0.021716873558663474`
- Causal model score: `0.7796920239925385`
- Causal movement pred: `0.7452703714370728`
- Causal jump-safety pred: `0.5418965816497803`
- Causal identity pred: `0.9999830722808838`
- Causal same-local family: `1.0`
- Causal agreement with recurrence: `1.0`

Interpretation:

The same safe-authority pattern appears under an alternate parent ontology config. The wrong-family decoy moves more, but the causal selector chooses the lower-jump branch-authorized operator.

## Current Scientific Status

Supported now:

- Causal label generation scales to depth-2 grandchild substrates.
- Multi-head causal routing can separately learn route score, movement, jump safety, and identity.
- Hard decoys now exist: wrong-family decoys can move the parent more while worsening world-jump risk.
- Causal routing correctly prefers branch-authorized, lower-jump operators over those harder decoys.
- The pattern holds in both the trained phasor authority carrier-only config and the childphase config.

Still not supported:

- Runtime promotion.
- Learned branch-law replacement.
- Audio-facing continuity improvement.
- Nested sibling emergence.
- Multi-config generalization from one trained checkpoint to a different config. We trained/evaluated per config in this cycle.

## Interpretation

This is the cleanest operator-routing result so far.

Earlier retrieval embeddings answered: which law object is similar?

The multi-head causal selector now answers something closer to: which operator has authority to write back safely?

The key scientific shift is that maximum parent movement is no longer treated as the objective. The causal route is allowed to choose a smaller phase movement when that movement preserves branch identity and reduces world-jump risk.

That is much more faithful to the RAFA operator-attention constitution: value routing is not content retrieval; it is lawful operator permission.

## Next Step

The next cycle should test cross-config generalization directly:

1. Train one multi-head causal selector on one config and evaluate it on another config without retraining.
2. Add explicit hard-decoy summary metrics to every run: high-movement wrong-family count, high-jump decoy count, causal-over-decoy safety win rate.
3. Start a shadow learned branch-law table from causal-selector outputs: survival delta, collapse pressure, support delta, and writeback permission.
