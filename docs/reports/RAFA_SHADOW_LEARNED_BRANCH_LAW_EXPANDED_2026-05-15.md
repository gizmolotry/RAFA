# RAFA Shadow Learned Branch Law Expanded Pass - 2026-05-15

## Summary

This pass moved the learned branch-law lane from a positive-only table into a mixed table with both safe writeback rows and denied writeback rows.

The important result is not that the tiny model fits the table. The important result is that the expanded table now contains a failure boundary: at grandchild depth 3 on wider ontology configs, the causal selector chooses a stronger-moving operator that also jumps harder, so shadow writeback permission drops to zero.

This is exactly the kind of pressure a learned branch law needs before runtime use.

## Code Changes

- Added `train_shadow_branch_law_from_table.py`.
- Reused the existing `RafaLearnedBranchLawV0` from `circleworld.py`.
- Added a contract test mapping shadow table rows to the learned branch-law output schema.
- No Circleworld runtime behavior was changed.

## Wider Depth-3 Probes

### Authority config, depth 3

Artifact:

- `D:\RAFA\outputs\circleworld_proto\resonant_operator_causality_2026-05-15_wide_authority_9111_9113_d3\resonant_operator_causality_assay.json`

Metrics:

- cases: `3`
- law objects per case: `15`
- causal parent divergence: `0.2369058623476279`
- decoy parent divergence: `0.1905976399472833`
- causal world jump: `0.030709791638601918`
- decoy world jump: `0.02278113050760137`
- hard decoy against causal: `0.0`
- causal safety win over decoy: `0.0`
- causal authority win over decoy: `0.0`

### No-delta authority config, depth 3

Artifact:

- `D:\RAFA\outputs\circleworld_proto\resonant_operator_causality_2026-05-15_wide_nodelta_9111_9113_d3\resonant_operator_causality_assay.json`

Metrics match the authority config on this seed/config slice:

- cases: `3`
- law objects per case: `15`
- causal parent divergence: `0.2369058623476279`
- decoy parent divergence: `0.1905976399472833`
- causal world jump: `0.030709791638601918`
- decoy world jump: `0.02278113050760137`
- hard decoy against causal: `0.0`
- causal safety win over decoy: `0.0`
- causal authority win over decoy: `0.0`

Interpretation: this is a useful negative edge. Depth 2 showed hard decoys where the safe route moved less and jumped less. Depth 3 flips the pressure: the causal route moves more, but now also jumps more. Shadow writeback should not be automatically permitted there.

## Expanded Shadow Table

Artifact:

- `D:\RAFA\outputs\circleworld_proto\shadow_branch_law_table_2026-05-15_expanded_d2_d3\shadow_branch_law_table.json`

Aggregate:

- sources: `6`
- rows: `17`
- mean coexistence drive: `0.9158034974716595`
- mean merge drive: `0.9326431071305543`
- mean survival delta: `0.7922386059614718`
- mean collapse pressure: `0.20846098178538122`
- mean support delta: `-0.13343306061099558`
- mean q-trace inheritance mix: `0.947388499627115`
- mean writeback permission: `0.6470588235294118`
- mean hard decoy against causal: `0.6470588235294118`
- mean causal safety win over decoy: `0.6470588235294118`

Source split:

- depth-2 sources: writeback permission `1.0`, survival about `0.87`, collapse about `0.13`
- depth-3 wide sources: writeback permission `0.0`, survival about `0.647`, collapse about `0.355`

## Shadow Learned Branch-Law Fit

Artifact:

- `D:\RAFA\outputs\circleworld_proto\shadow_learned_branch_law_2026-05-15_expanded_d2_d3\shadow_branch_law_training.json`
- `D:\RAFA\outputs\circleworld_proto\shadow_learned_branch_law_2026-05-15_expanded_d2_d3\shadow_rafa_learned_branch_law_v0.pt`

Metrics:

- rows: `17`
- sources: `6`
- folds: `6`
- final train loss: `5.164767458154529e-07`
- final mean MAE: `0.0003720878448802978`
- final max MAE: `0.0019611716270446777`
- mean holdout MAE: `0.0005046677769371399`
- max holdout MAE: `0.000824248360004276`

Top saliency:

- causal safety win over decoy: `3.4195172702311538e-06`
- hard decoy against causal: `2.4161299734259956e-06`
- decoy-causal phase divergence gap: `2.286690914843348e-06`
- recurrence compatibility: `2.2152851215651026e-06`
- decoy world jump proxy: `1.2977714050066425e-06`
- identity signal: `1.1883439583471045e-06`

Output fit:

- split drive MAE: `0.00008985575550468639`
- coexistence drive MAE: `0.00008428447472397238`
- merge drive MAE: `0.000016370240700780414`
- survival delta0 MAE: `0.001150040072388947`
- survival delta1 MAE: `0.0011480696266517043`
- collapse pressure MAE: `0.0005634742556139827`
- support delta0 MAE: `0.00003929348531528376`
- support delta1 MAE: `0.00003587717583286576`
- coherence target MAE: `0.0005828703287988901`
- qtrace inheritance mix MAE: `0.000010742860467871651`

## Interpretation

The learned branch-law head can now model a mixed branch-law surface with both allowed and denied writeback regimes. That is a real improvement over the previous positive-only table.

But the table remains engineered. The model is learning our shadow-law mapping, not discovering a branch law from raw dynamics. The next credibility step is to replace more of the engineered shadow fields with directly measured outcomes from wider branch candidates and to hold out entire config families that are not near-duplicates.

## Current Claim Status

Supported:

- A small learned module can fit assay-derived branch-law fields.
- The shadow fields can represent both safe writeback and denied writeback.
- Depth and child/grandchild volume change the safety boundary.

Not yet supported:

- The learned branch law improves Circleworld runtime evolution.
- The learned branch law generalizes across distant configs or audio anchors.
- The learned law has discovered ontology rather than compressed engineered targets.

## Verification

- `python -m pytest -q D:\RAFA\runtimes\circleworld_proto\test_learned_branch_law_child_ifs_contract.py D:\RAFA\runtimes\circleworld_proto\test_resonant_attention_contract.py` -> `18 passed`
- `python -m py_compile` for new trainer and table builder -> pass

## Next Move

Use the learned shadow checkpoint as a validator, not a runtime controller: score new candidate branch-law rows before table-building, compare predicted writeback/survival/collapse against measured causality, and only then consider a gated sandbox runtime branch-law path.
