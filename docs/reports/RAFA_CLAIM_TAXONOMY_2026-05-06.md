# RAFA Claim Taxonomy - 2026-05-06

This document separates RAFA into independently testable claim families.

The core discipline: do not treat RAFA, Ramanujan structure, phase-only operation, audio modeling, lattice dynamics, semantic projection, and dense signatures as one inseparable thesis. They are connected, but they must be allowed to succeed or fail independently.

## Why This Split Matters

If every RAFA idea is bundled into one package, we kneecap the research loop:

- A failed Ramanujan branch experiment could wrongly imply semi-tokenless modeling is impossible.
- A successful audio benchmark could wrongly imply phase-only reasoning worked.
- A good dense signature result could wrongly validate the lattice substrate.
- A child-world branch result could wrongly validate semantic grounding.
- A q-trace result could wrongly validate Ramanujan structure, even if the same result works with generic Fourier features.

The right stance is modular: each claim needs its own falsifiable test, ablation, and promotion rule.

## Claim Family A: RAFA As A Semi-Tokenless Modeling Paradigm

Claim:

RAFA can model structured continuation without relying primarily on discrete symbolic tokens.

Independent question:

Can a model operate over relational phase/law states instead of text-like token sequences and still produce stable, controllable continuation?

What this does not require:

- Ramanujan sums specifically.
- Audio specifically.
- Child worlds specifically.
- No learned dense signatures.
- No semantic projector.

Tests:

- Replace discrete event tokens with continuous relational signatures or phasor states.
- Compare continuation quality against tokenized baselines.
- Measure whether useful state can be carried through dynamics without SFT-style label imitation.

Failure mode:

The system needs hidden discrete labels or externally imposed symbolic scaffolding to remain coherent.

## Claim Family B: RAFA As An Audio Model

Claim:

RAFA-style phase/law dynamics can produce useful audio continuation.

Independent question:

Can a phase-relational model produce audio with acceptable continuity, structure, and diversity?

What this does not require:

- Fully phase-only operation.
- Ramanujan priors.
- Tokenless semantics.
- Nested branching.

Tests:

- Benchmark against audio continuation baselines.
- Run continuity and loop recurrence diagnostics.
- Compare phase-only, phase-plus-magnitude, and magnitude-assisted variants.

Failure mode:

Audio only works when magnitude/reconstruction shortcuts dominate, or output becomes looped/noisy despite good internal metrics.

## Claim Family C: No-SFT / Low-SFT Ambition

Claim:

RAFA should learn operational laws from self-supervised, metameric, continuity, and branch objectives rather than relying totally on SFT.

Independent question:

Can useful continuation laws emerge from intrinsic dynamical objectives?

What this does not require:

- Tokenlessness.
- Ramanujan priors.
- Human-readable semantics.
- Audio-only evaluation.

Tests:

- Train with no SFT, weak labels, synthetic laws, or self-supervised continuation.
- Compare to SFT-heavy baselines.
- Measure transfer to heldout seeds and transformed metamers.

Failure mode:

The model only becomes useful after supervised imitation supplies the actual structure.

## Claim Family D: Phase-Only / No-Magnitude Ambition

Claim:

RAFA can place meaningful law and branch information in phase relations without relying on magnitude as the primary semantic carrier.

Independent question:

Can phase-only state carry enough information for continuation, branching, and reentry?

What this does not require:

- Ramanujan q structure.
- Audio success.
- Dense relational signatures.
- Semantic projector success.

Tests:

- Enforce unit phasors.
- Apply global phase rotations and magnitude rescaling controls.
- Compare phase-only branch metrics with magnitude-assisted variants.
- Audit every runtime update for hidden magnitude writes.

Failure mode:

The model claims phase-only behavior but branch identity or audio quality disappears when magnitude cues are removed.

## Claim Family E: Lattice Substrate

Claim:

A local phasor lattice is a useful substrate for relational computation.

Independent question:

Does the lattice geometry itself provide useful locality, support, and evolution constraints?

What this does not require:

- Ramanujan sums.
- Hardy-Littlewood arcs.
- Audio.
- Tokenless semantics.

Tests:

- Compare lattice dynamics to non-lattice continuous state models.
- Ablate support locality and neighborhood transport.
- Test whether local laws compose over time better than flat vector-state updates.

Failure mode:

The lattice adds complexity but no measurable benefit over a simpler latent sequence model.

## Claim Family F: Ramanujan / Rational q Prior

Claim:

Ramanujan q structure is a useful inductive prior for periodic/rational relational law.

Independent question:

Does q-structured arithmetic help beyond generic spectral or q-trace features?

What this does not require:

- Lattice substrate success.
- Audio success.
- Tokenless modeling.
- Branching success.

Tests:

- Compare Ramanujan, qtrace-only, shuffled qset, uniform q weights, Fourier-only, and learned q basis.
- Measure q-family diversity, q persistence, continuation, branch identity, and metamer stability.

Failure mode:

Ramanujan features are decorative: qtrace-only or learned Fourier features perform as well or better.

## Claim Family G: Hardy-Littlewood / Major-Minor Arc Prior

Claim:

Major/minor arc structure helps identify coherent, promotable relational packets.

Independent question:

Does arc decomposition causally improve promotion, support, and law formation?

What this does not require:

- Ramanujan being the best q basis.
- Audio success.
- Dense signature success.

Tests:

- Flatten or randomize arc fields while preserving phase state.
- Disable promotability.
- Compare packet quality, child spawn/writeback, and continuation.

Failure mode:

Arc fields are good diagnostics but do not causally improve dynamics.

## Claim Family H: Native Branching / Multimode Dynamics

Claim:

RAFA can maintain multiple local candidate continuations inside the runtime, not only as perturbation analysis.

Independent question:

Can sibling continuations coexist, survive, diverge at meso scale, and avoid world jumps?

What this does not require:

- Ramanujan priors.
- Audio success.
- Semantic projector success.

Tests:

- Native multimode vs single-path.
- Child-world writeback on/off.
- Parent branch vs child readiness separated.
- Nested commitment assay with qualified identity carry.

Failure mode:

Branching exists only as decorative slot-2 activity, child registry persistence, or readout perturbation.

## Claim Family I: Dense RAFA Relational Signatures

Claim:

RAFA can define dense learned relational signatures that act token-like without being ordinary discrete tokens.

Independent question:

Can a learned dense body `h` carry world, q, arc, lifecycle, support, branch, and operator information in a useful way?

What this does not require:

- Fully tokenless runtime.
- Ramanujan priors.
- Audio-only success.

Tests:

- Train `h` with Matryoshka prefix losses.
- Decode q/arc/temporal/support/branch/operator heads.
- Run metamer consistency and continuation-usefulness tests.

Failure mode:

Signatures collapse to explicit metadata, law-token labels, or inert embeddings that do not improve future dynamics.

Current status after the 2026-05-07 dense-signature audit:

- Schema/interface: present. `D=768`, Matryoshka prefixes, explicit heads, branch profile, and operator seed are available.
- Runtime V0 dense body: not a learned token body. Current `h` is assembled from explicit factor-pack fields, so it is a deterministic baseline/interface.
- Broad dense-vs-metadata result: `dense_stability_only_not_token_claim`. Dense slightly improves metamer stability but loses separation and anti-collapse against explicit q/law metadata.
- Learned module surface: present as an off-runtime smoke. Shape/simplex/unit-norm/loss/gradient checks pass and a tiny synthetic overfit reduces loss. The learning contract now includes an explicit learned law-signature projection head in addition to the operator-seed head, but this is still not trained Circleworld behavior.
- Learned packet-factor scout: upgraded to a narrow heldout implementation signal, not a token proof. The corrected 52-row non-smoke compare is `heldout_candidate_found_needs_seed`: strict candidates `4/52`, two-axis `11/52`, stability+anti-collapse `5/52`, heldout candidates `1/35`, and heldout two-axis rows `4/35`. The v14 low-mid run-center contrastive + soft base-SVD family still passes `2/3` explicit seeds in-sample. The v24 h96/run-contrastive/base-SVD/six-transform profile gives the first heldout all-axis pass at seed `1729` with stability `+0.000425`, separation `+0.119588`, and effective-rank delta `+0.444968`, but seed replication fails at `2718` and `3141`, heldout rotations at offsets `3`, `6`, and `12` fail, and shuffled partitions `101` and `202` fail. The split-suite compare is `fragile_split_suite_candidate_only`: v24 and v28 each pass only `1/8` heldout splits, v30 train15/holdout3 passes `0/8`, v31/v32 law-head suites pass `0/8`, robust candidates are `0/5`, and partition candidate fraction remains `0/5`. Law-head v32 lowers mean stability loss to `-0.003846` and has low mean final law/operator losses (`0.032666` / `0.015981`), but still fails all candidate gates. The heldout case-failure atlas identifies `soundbible_steam_engine`, `footsteps_cement`, `bells_tibetan_large`, `audience_applause`, and `male_vocalized_a_z` as current worst stability blockers. This means the learned lane is trainable and no longer merely in-sample, but it is split-suite seed/partition-blocked.
- Operator-tail/future-law usefulness: mixed-to-negative, not proof. The old operator-tail predictive compare is `operator_tail_mixed_usefulness` with learned views winning `1/4` full law/operator rows. The stricter redacted future-law probe is `future_law_no_learned_usefulness` with learned wins `0/4`. The stricter redacted tail-increment probe is `tail_incremental_usefulness_mixed_control_confounded`: tail+metadata beats metadata on both targets, but prefix and shuffled controls also win and total negative-control wins are `6`.
- Promotion blocker: no RAFA-token claim until learned `h` beats explicit metadata on stability, separation, anti-collapse, seed/partition stability, and redacted continuation/operator usefulness without negative-control leakage.

## Claim Family J: Semantic Projector / Phonode Controls

Claim:

High-level prompts can map to physically meaningful control variables without becoming a normal text-token model.

Independent question:

Can prompts like `airplane`, `reed`, `bell`, and `impact` steer harmonic coupling, decay, noise, branch temperature, and support spread?

What this does not require:

- Full language understanding.
- SFT.
- Ramanujan priors.
- Nested branching success.

Tests:

- One-control-at-a-time sweeps.
- Frozen/random embedding negative controls.
- Prompt family transfer tests.

Failure mode:

Controls are non-identifiable, random projector performs as well, or prompt effects are only label leakage.

## Claim Family K: Anti-Loop / Reentry Law

Claim:

RAFA can learn reentry and continuation laws that avoid dumb loop attractors.

Independent question:

Can the system preserve identity while reducing macro-time recurrence?

What this does not require:

- Ramanujan priors.
- Branching.
- Phase-only operation.

Tests:

- Loop autocorrelation.
- First-chunk reentry.
- Nonlocal chunk repeat.
- Continuity under anti-fixation penalties.

Failure mode:

Loop metrics only improve by destroying audio/identity, or branch activity does not affect recurrence.

## Claim Family L: Resonant Attention / RAFA Post-Tokens

Claim:

RAFA can replace token-position attention with resonance-mediated law selection
over packet, child, grandchild, or relational-signature objects.

Independent question:

Can partial phase/q/arc/support queries retrieve and compose reusable law objects
whose values act as operators, not merely as content vectors?

What this does not require:

- audio benchmark improvement in the first assay
- CLAP or text prompting
- dense signatures being fully solved
- Ramanujan being the only useful q basis
- child ontology being promoted by existence alone

Tests:

- Resonant child retrieval: partial relational query vs compatible and decoy law
  objects.
- Cross-depth composition: parent query to child activation to grandchild
  refinement to parent return.
- Interference selectivity: compatible laws constructively activate while
  incompatible decoys suppress or cancel.
- Operator causality: selected values must change phase, support, q-trace,
  branch state, or parent writeback when enabled.
- Geometry/residual ablations: explicit q/arc/support geometry must matter
  separately from learned dense residuals.

Failure mode:

The system rediscovers dense dot-product routing, stored IDs, or scalar threshold
selection while claiming RAFA attention.

## Cross-Claim Dependency Map

Strong dependency:

- Phase-only audio depends on both audio modeling and phase-only law.
- Nested sibling depends on branching, identity carry, and continuation survival.
- Dense RAFA-token ambition depends on useful signatures, but not necessarily Ramanujan.
- RAFA post-token promotion depends on retrieval, composition, and operator
  causality, not merely dense signature training.

Weak dependency:

- Ramanujan may help audio, but audio success does not prove Ramanujan.
- Lattice locality may help branch survival, but branch survival does not prove lattice necessity.
- Semantic projector may help initialization, but semantic control does not prove RAFA reasoning.

Independent enough to test separately:

- Semi-tokenless modeling vs Ramanujan q prior.
- Audio continuation vs no-SFT training.
- Phase-only law vs lattice substrate.
- Dense signatures vs child-world branching.
- Semantic projector vs native branching.
- Anti-loop law vs q-family structure.
- Resonant attention vs audio quality.
- Resonant attention vs dense signature quality.

## Promotion Rule

No future report should say "RAFA worked" without naming which claim family worked.

Allowed statements:

- "The Ramanujan q prior improved q-family diversity."
- "The phase-only branch metric became nonzero on naked RAFA."
- "Dense signatures improved metamer stability."
- "Audio continuation improved but relied on magnitude."
- "Native child worlds carried raw identity but not qualified identity."
- "Relational queries retrieved child-law objects above decoys."

Disallowed statements:

- "RAFA works."
- "RAFA failed."
- "Ramanujan proves tokenlessness."
- "Audio benchmark improvement proves phase-only reasoning."
- "Child survival proves nested sibling."

## Immediate Next Experiment Set

Run separate claim tests:

1. Semi-tokenless baseline: continuous relational state vs discretized/tokenized proxy.
2. Audio-only baseline: best audio continuation regardless of Ramanujan/phase-only purity.
3. Phase-only audit: unit-phasor and gauge-invariance tests.
4. Ramanujan ablation: Ramanujan vs qtrace-only vs Fourier-only vs shuffled qset.
5. Lattice ablation: phasor lattice vs flat latent sequence dynamics.
6. Branch ontology ablation: parent branch, child branch, raw carry, qualified carry separated.
7. Dense signature ablation: dense body vs explicit heads only.
8. Resonant attention assay: query/key/value law objects vs dense dot-product and stored-ID controls.

The point is not to make the project smaller. The point is to stop one hypothesis from holding every other hypothesis hostage.
