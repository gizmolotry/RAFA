# Circleworld Relational Signature V0 (2026-05-04)

## Scope

This pass adds a first dense RAFA relational-signature layer on top of the existing Circleworld law-packet and multimode runtime.

It does **not** replace the current packet / childworld / branch-law substrate.
It adds a new export and inspection layer that can later support:

- Matryoshka prefix training
- metamer consistency work
- branch usefulness supervision
- semantic projector conditioning

Graduation was not touched.

## What Was Added

### New lineage modules

- `D:\RAFA\lineages\04_positive_replacement\rafa_relational_signature.py`
- `D:\RAFA\lineages\04_positive_replacement\semantic_projector.py`

### Updated runtime files

- `D:\RAFA\lineages\04_positive_replacement\circleworld.py`
- `D:\RAFA\runtimes\circleworld_proto\build_law_token_library.py`
- `D:\RAFA\runtimes\circleworld_proto\README.md`

## Relational Signature V0

The new signature layer is:

- dense body: `h : [N, 768]`
- prefix schedule: `128 / 256 / 384 / 512 / 640 / 768`
- explicit observable heads:
  - `q_profile`
  - `arc_profile`
  - `temporal_profile`
  - `support_profile`
  - `branch_profile`
  - `operator_seed`
  - `confidence`

### Important implementation note

This is an **instrumented scaffold**, not a trained learned encoder yet.

The dense body is assembled from existing Circleworld observables with a soft prefix layout and residual mixing. That gives us a concrete token-like object and export path now, without pretending we already trained the full semantics.

## Runtime Integration

Every `recurse_circleworld(...)` run now emits:

- `relational_signatures`
- `relational_signature_library`

`summary_circleworld_run(...)` now surfaces:

- `num_relational_signatures`
- `num_relational_signature_families`
- `mean_relational_signature_confidence`
- `mean_relational_signature_q_entropy`
- `mean_relational_branch_mass`
- `dominant_relational_family_share`

## Semantic Projector Scaffold

`semantic_projector.py` adds the embedding-to-control interface only.

It supports:

- `SemanticProjectorConfig`
- `SemanticControlProfile`
- `project_embedding_to_controls(...)`
- `semantic_profile_from_state_dict(...)`

This is intentionally just the projector surface.
It does **not** claim prompt semantics are solved.
A frozen text embedding model plus trained projector weights still need to be supplied later.

## Smoke Run

Config:

- `D:\RAFA\checkpoints_circleworld_proto\training_run_2026-04-24_agreement_scout_v1\circleworld_real_anchor_config_cem_v1.json`

Cases:

- `airplane_takeoff`
- `airplane_landing`

Artifacts:

- `D:\RAFA\outputs\circleworld_proto\relational_signature_smoke_2026-05-04\law_token_library.json`
- `D:\RAFA\outputs\circleworld_proto\relational_signature_smoke_2026-05-04\relational_signature_library.json`
- `D:\RAFA\outputs\circleworld_proto\relational_signature_smoke_2026-05-04\relational_signatures.json`
- `D:\RAFA\outputs\circleworld_proto\relational_signature_smoke_2026-05-04\RELATIONAL_SIGNATURE_LIBRARY.md`

Headline numbers:

- cases: `2`
- mean law packets per case: `6.0`
- mean relational signatures per case: `6.0`
- aggregate law families: `1`
- aggregate relational-signature families: `1`
- mean relational-signature confidence: `0.6924`

## Honest Read

This pass is successful as infrastructure.

What is real now:

- Circleworld can export dense 768-d relational signatures per promoted law packet.
- Those signatures can be clustered into cross-case families.
- The current runtime can serialize both packet-law families and dense signature families from the same run.
- The semantic projector interface now has a proper place to live.

What is **not** true yet:

- this is not a trained semantic encoder
- this is not yet Matryoshka-trained
- this does not solve branching or nested sibling continuation
- the current smoke run is still low-diversity and low-q collapsed

## Best Next Step

The next serious step is not another ontology rewrite.
It is a training/evaluation pass on the new signature layer:

1. add signature-side losses:
   - prefix usefulness
   - metamer consistency
   - continuation usefulness
2. export signature banks on a broader anchor set
3. compare whether signature families diversify faster than the older law-token families
4. only then connect the semantic projector to a real embedding source
