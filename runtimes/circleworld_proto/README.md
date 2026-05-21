# Runtime: Circleworld Proto

This is the positive-replacement formalization prototype.

## Status

- active prototype
- ablation-focused
- not yet integrated into the main generation stack

## Canonical Entry Point

- `run_ablation.py`
- `train_circleworld.py`
- `evaluate_circleworld.py`

## Scope

This runtime is for:

- coherent-structure detection
- packet promotion
- recursive packet feedback

It is not yet a production generation runtime.

## Training Lane

`train_circleworld.py` runs an isolated parameter-training attempt for Circleworld itself.

It is intentionally separate from:
- Graduation restore
- diffusion parent prompt-conditioning
- semantic steering / tension lane

Artifacts from this lane live under:
- `outputs/circleworld_proto/`
- `checkpoints_circleworld_proto/`

## Evaluation

`evaluate_circleworld.py` runs a held-out config evaluation and exports:
- per-sample summary JSON
- per-time trajectory CSV

This is the main inspection path for trained Circleworld configs.

`benchmark_circleworld_real_anchor.py` runs the bit-stable real-audio benchmark used for
reference-vs-output preservation checks.

`benchmark_audio_continuity.py` runs the macro-time continuity / loop benchmark used for
repetition, re-entry, and time-world diagnostics.

`test_nested_commitment.py` runs the fork/resume assay used to test whether recursion
preserves a coarse world-law while later recursion only refines it.

`build_law_token_library.py` extracts promoted-law families from real anchor runs and
builds a reusable cross-case token library from the existing Circleworld packet stream.
Use this when you want to inspect relation-token candidates rather than only branch or
audio metrics.

`build_relational_signature_library.py` is the signature-first entry point for the same
artifact flow. Use this when you specifically want the dense 768-d signature bank and
signature-family exports rather than starting from the older law-token framing.

`compare_relational_signature_runs.py` compares two signature export / metamer / held-out
artifact bundles and writes one JSON + Markdown comparison verdict.

`run_relational_signature_experiment.py` runs a signature-focused scout with preset
profiles, then exports held-out, benchmark, continuity, signature library, and metamer
sidecars in one shot.

The same path now also exports a `RafaRelationalSignatureV0` sidecar:
- dense 768-d relational signatures on top of promoted law packets
- interpretable heads for q / arc / temporal / support / branch observables
- aggregate signature families for cross-case inspection

The signature lane is Circleworld-only scaffolding for now. It is intended to support
Matryoshka-style prefix training and future semantic projector work without changing the
current branch-law runtime.

`run_token_diversity_experiment.py` reproduces the Ramanujan token-diversity experiment:
real-anchor training, held-out eval, expanded benchmark, continuity sidecars, nested
commitment, and rebuilt law-token library in one run.

`run_branch_pressure_experiment.py` runs the Circleworld branch-pressure lane: it starts
from the token-diverse Ramanujan checkpoint, pushes the trainer toward defect-born local
multiplicity in the lattice, and then runs the same full sidecar stack so branch pressure
can be compared directly against token-diversity and prior multimode runs.

Current retrospective read:
- the best real-audio Circleworld runs are still preservation-heavy
- they do not yet create a new macro-time law
- the current fork/resume evidence points to over-rigid attractors rather than
  Matryoshka-like nested commitment
- the next frontier is reducing loop/re-entry without leaving the real-audio frontier

## Semantic Steering Scaffold

The lineage also now contains a semantic projector interface:
- `D:\RAFA\lineages\04_positive_replacement\semantic_projector.py`

This is only the embedding-to-control surface for now. It does not claim that text-to-
lattice semantics are solved; it exists so future training can map frozen prompt
embeddings into Circleworld boundary conditions without inventing a second ontology.
