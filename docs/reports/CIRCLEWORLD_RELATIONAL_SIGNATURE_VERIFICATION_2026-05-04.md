# Circleworld Relational Signature Verification (2026-05-04)

## Scope

This pass hardens the new `RafaRelationalSignatureV0` lane after the initial scaffold landed.

Goals:

- wire signature metrics into normal evaluation paths
- verify the new exports on more than a toy smoke
- compare active vs fallback Circleworld checkpoints on the new signature artifacts

Graduation was not touched.

## Code Additions In This Pass

### Runtime / lineage

- `D:\RAFA\lineages\04_positive_replacement\rafa_relational_signature.py`
- `D:\RAFA\lineages\04_positive_replacement\semantic_projector.py`
- `D:\RAFA\lineages\04_positive_replacement\circleworld.py`

### Runtime scripts

- `D:\RAFA\runtimes\circleworld_proto\build_law_token_library.py`
- `D:\RAFA\runtimes\circleworld_proto\build_relational_signature_library.py`
- `D:\RAFA\runtimes\circleworld_proto\evaluate_circleworld.py`
- `D:\RAFA\runtimes\circleworld_proto\train_circleworld.py`
- `D:\RAFA\runtimes\circleworld_proto\train_circleworld_real_anchor.py`
- `D:\RAFA\runtimes\circleworld_proto\README.md`

## What Was Verified

### 1. Compile sweep

`py_compile` passed for:

- `rafa_relational_signature.py`
- `semantic_projector.py`
- `circleworld.py`
- `build_law_token_library.py`
- `build_relational_signature_library.py`
- `evaluate_circleworld.py`
- `train_circleworld.py`
- `train_circleworld_real_anchor.py`

### 2. Held-out evaluator integration

The active checkpoint now reports signature metrics through the normal held-out evaluator.

Artifacts:

- `D:\RAFA\outputs\circleworld_proto\relational_eval_active_2026-05-04\heldout_summary.json`
- `D:\RAFA\outputs\circleworld_proto\relational_eval_active_2026-05-04\heldout_trajectory.csv`

New surfaced metrics include:

- `mean_num_relational_signatures`
- `mean_num_relational_signature_families`
- `mean_relational_signature_confidence`
- `mean_relational_signature_q_entropy`
- `mean_relational_branch_mass`
- `mean_dominant_relational_family_share`

Active checkpoint held-out read:

- `mean_num_relational_signatures = 28.0`
- `mean_num_relational_signature_families = 1.6667`
- `mean_relational_signature_confidence = 0.8592`
- `mean_relational_signature_q_entropy = 0.5660`
- `mean_relational_branch_mass = 0.4965`
- `mean_dominant_relational_family_share = 0.9383`

By source:

- `synthetic`
  - signatures: `36.0`
  - families: `2.0`
  - confidence: `0.9470`
  - branch mass: `0.5379`
- `naked_rafa`
  - signatures: `12.0`
  - families: `1.0`
  - confidence: `0.6836`
  - branch mass: `0.4136`

So the signature lane is now visible in the same real-vs-synthetic split as the rest of Circleworld.

### 3. Real-anchor trainer dataset path

The real-anchor trainer's `_evaluate_cfg_on_dataset(...)` path was exercised on a small subset after integration.

Quick verification output:

- `num_samples = 2`
- `mean_num_relational_signatures = 12.0`
- `aggregate_num_relational_signature_families = 2.0`
- `aggregate_relational_signature_confidence = 0.8820`
- `aggregate_relational_branch_mass = 0.4782`

That means the trainer-side aggregate signature library path is live, not just compiled.

### 4. Full anchor export: active checkpoint

Artifacts:

- `D:\RAFA\outputs\circleworld_proto\relational_signature_full_2026-05-04\law_token_library.json`
- `D:\RAFA\outputs\circleworld_proto\relational_signature_full_2026-05-04\relational_signature_library.json`
- `D:\RAFA\outputs\circleworld_proto\relational_signature_full_2026-05-04\relational_signatures.json`
- `D:\RAFA\outputs\circleworld_proto\relational_signature_full_2026-05-04\RELATIONAL_SIGNATURE_LIBRARY.md`

Headline numbers:

- cases: `12`
- `mean_num_relational_signatures = 12.0833`
- `mean_num_relational_signature_families = 1.4167`
- `mean_relational_signature_confidence = 0.6943`
- aggregate signature families: `8`

### 5. Full anchor export: conservative baseline

Artifacts:

- `D:\RAFA\outputs\circleworld_proto\relational_signature_full_parentmix_2026-05-04\law_token_library.json`
- `D:\RAFA\outputs\circleworld_proto\relational_signature_full_parentmix_2026-05-04\relational_signature_library.json`
- `D:\RAFA\outputs\circleworld_proto\relational_signature_full_parentmix_2026-05-04\relational_signatures.json`
- `D:\RAFA\outputs\circleworld_proto\relational_signature_full_parentmix_2026-05-04\RELATIONAL_SIGNATURE_LIBRARY.md`

Headline numbers:

- cases: `12`
- `mean_num_relational_signatures = 12.4167`
- `mean_num_relational_signature_families = 2.0833`
- `mean_relational_signature_confidence = 0.6777`
- aggregate signature families: `12`

### 6. Active vs baseline comparison

Comparison artifact:

- `D:\RAFA\outputs\circleworld_proto\relational_signature_compare_2026-05-04.json`

Key read:

- active checkpoint:
  - slightly higher signature confidence
  - fewer signature families
- parentmix baseline:
  - more signature families both per-case and aggregate
  - slightly lower confidence

Concrete numbers:

- active `mean_num_relational_signature_families = 1.4167`
- baseline `mean_num_relational_signature_families = 2.0833`
- active `aggregate_num_signature_families = 8`
- baseline `aggregate_num_signature_families = 12`
- active `mean_relational_signature_confidence = 0.6943`
- baseline `mean_relational_signature_confidence = 0.6777`

## Honest Read

This pass succeeded as infrastructure and evaluation hardening.

What is now clearly true:

- the signature layer is a first-class Circleworld output, not just a hidden sidecar
- evaluator summaries and trainer dataset summaries can now surface signature metrics
- full anchor exports run cleanly on both the active and baseline checkpoints
- the new signature lane is already producing a useful comparative signal

The most important new finding is this:

- the active checkpoint is **not** the most diverse signature producer
- the conservative `parentmix_220_100` baseline currently yields broader signature family spread on the 12-case anchor set

That is useful because it means the new token layer is not just re-reporting the old winner. It is revealing a different tradeoff surface.

## Recommendations

### Immediate

1. Treat signature diversity as a separate objective from branch survival.
2. Keep both reference checkpoints alive in this lane:
   - active branch-first: `agreement_scout_v1`
   - signature-diverse baseline: `parentmix_220_100`

### Next implementation step

3. Add signature-specific scoring terms to the real-anchor search only after we decide whether we want:
   - more signature families
   - higher signature confidence
   - higher branch mass
   - better synthetic/naked parity

### Next experiment

4. Run a metamer-style export pass on transformed anchor variants and test whether early signature structure is more stable than old law-packet family assignment.

That is the next real test of whether `RafaRelationalSignatureV0` is becoming more than a prettier packet summary.
