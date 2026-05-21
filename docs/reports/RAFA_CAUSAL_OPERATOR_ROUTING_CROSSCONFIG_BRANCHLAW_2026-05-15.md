# RAFA Causal Operator Routing Cross-Config Branch-Law Pass - 2026-05-15

## Summary

This pass tightened causal operator routing around the exact failure mode we care about: wrong-family operators that move the parent more but carry worse world-jump risk.

The result is positive but bounded: the learned multi-head causal selector transfers across two nearby Circleworld ontology configs and keeps choosing the branch-authorized, lower-jump operator. This is not yet a production branch law, but it gives a concrete shadow branch-law target surface.

## Code Changes

- Added hard-decoy accounting to `run_causal_operator_selector_probe.py`.
- Added hard-decoy and causal safety-win aggregates to `run_resonant_operator_causality_assay.py`.
- Added `build_shadow_branch_law_table.py` to convert causality assay outputs into assay-only branch-law fields.
- Extended `test_resonant_attention_contract.py` with shadow branch-law and multi-head selector contract tests.

## Training Probe With Hard-Decoy Metrics

Artifact:

- `D:\RAFA\outputs\circleworld_proto\causal_operator_selector_2026-05-15_naked_gpu_9100_9107_d2_multihead_harddecoy\causal_operator_selector_probe.json`

Metrics:

- candidate rows: `880`
- feature dim: `116`
- model kind: `multihead`
- final loss: `0.00017972060387754546`
- final route loss: `0.000042825809422148655`
- final movement loss: `0.00035067041920098873`
- final jump-safety loss: `0.0003123854361807129`
- final identity loss: `0.000021418127809218796`
- holdout membrane target score: `0.756770656362291`
- holdout target-top score: `0.7619132830734119`
- holdout identity pass: `0.9090909090909091`
- hard-decoy fraction: `1.0`
- hard-decoy query count: `80`
- hard wrong-family candidate count: `168`
- mean best decoy movement gap: `0.028679464507539464`
- mean best decoy jump gap: `0.012153132748393514`
- membrane safety win fraction: `1.0`

Interpretation: the training substrate is not easy. Hard decoys are present everywhere: wrong-family candidates can move the parent more than the identity-compatible route, but they also jump harder. This is exactly the distinction a learned branch law must preserve.

## Cross-Config Transfer

### Phasor-authority selector to childphase substrate

Artifact:

- `D:\RAFA\outputs\circleworld_proto\resonant_operator_causality_2026-05-15_crossconfig_phasor_to_childphase_9108_9110_d2\resonant_operator_causality_assay.json`

Metrics:

- cases: `3`
- causal parent divergence: `0.1816104125333835`
- decoy parent divergence: `0.19017025714780425`
- causal world jump: `0.016863268703669793`
- decoy world jump: `0.02171658817402315`
- causal model score: `0.7804904580116272`
- causal identity prediction: `0.999982476234436`
- causal identity membrane pass: `1.0`
- hard decoy against causal: `1.0`
- causal safety win over decoy: `1.0`
- causal authority win over decoy: `1.0`

### Childphase selector to phasor-authority substrate

Artifact:

- `D:\RAFA\outputs\circleworld_proto\resonant_operator_causality_2026-05-15_crossconfig_childphase_to_phasor_9108_9110_d2\resonant_operator_causality_assay.json`

Metrics:

- cases: `3`
- causal parent divergence: `0.1816104125333835`
- decoy parent divergence: `0.19017025714780425`
- causal world jump: `0.016863268703669793`
- decoy world jump: `0.02171658817402315`
- causal model score: `0.7796913584073385`
- causal identity prediction: `0.9999830722808838`
- causal identity membrane pass: `1.0`
- hard decoy against causal: `1.0`
- causal safety win over decoy: `1.0`
- causal authority win over decoy: `1.0`

Interpretation: within these two nearby ontology configs, causal selector behavior transfers. The selector does not chase maximum parent movement; it preserves branch authority and lower world-jump risk.

Caution: the two substrates are still closely related and produce nearly identical operator outcome surfaces. This is useful generalization evidence, not broad distributional proof.

## Shadow Branch-Law Table

Artifact:

- `D:\RAFA\outputs\circleworld_proto\shadow_branch_law_table_2026-05-15_crossconfig_d2\shadow_branch_law_table.json`

Aggregate metrics over 4 source assay outputs and 11 cases:

- mean coexistence drive: `0.9008687791111402`
- mean merge drive: `0.9329408723695581`
- mean survival delta: `0.8714130184587133`
- mean collapse pressure: `0.12858698154128678`
- mean support delta: `-0.11779154159805992`
- mean coherence target: `0.8735253238827791`
- mean q-trace inheritance mix: `0.9432156252505849`
- mean writeback permission: `1.0`
- hard decoy against causal: `1.0`
- causal safety win over decoy: `1.0`

Interpretation: this table is now the first concrete bridge from causal operator routing toward a learned branch law. The fields are continuous and branch-law shaped: coexistence, merge, survival, collapse, support, coherence, q-trace inheritance, and writeback permission.

## What We Learned

1. The hard decoy is real.
Wrong-family candidates are not inert. They often move the parent more than the safe route. The reason not to pick them is not lack of action; it is worse world-jump behavior and failed identity authority.

2. The causal selector is learning the right kind of preference.
It predicts high identity and routes to the lower-jump branch-authorized operator, even when a decoy offers more movement.

3. Cross-config transfer is encouraging but narrow.
The selector transfers between the phasor-authority and childphase ontology configs, but those configs share much of the same substrate. Next we need broader configs, depths, and seed families.

4. Branch-law learning should now be trained on fields, not labels alone.
The shadow table gives the next target: train `RafaLearnedBranchLawV0` to emit survival/collapse/writeback/coherence fields directly, while reporting geometric spine components separately.

## Verification

- `python -m pytest -q D:\RAFA\runtimes\circleworld_proto\test_resonant_attention_contract.py` -> `13 passed`
- `python -m py_compile` over patched runtime files -> pass
- `git diff --check` over patched files -> pass

## Next Move

The next useful step is to train a tiny learned branch-law head directly against the shadow branch-law table, then evaluate it as a shadow law on wider configs and grandchild depths before allowing it anywhere near runtime branch updates.
