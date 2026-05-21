# RAFA Retrospective And Recommendations

Date: 2026-04-13

## Purpose

This report consolidates what we have actually discovered across the two active RAFA lanes:

- Graduation RAFA: the sound-engine / certified runtime lane
- Circleworld RAFA: the recursive object-formation / ontology research lane

The goal is to separate what is now known from what is still aspirational, and to recommend a clean path forward for both.

## Executive Summary

The project has split into two real systems with different jobs.

Graduation RAFA is currently the engineering baseline:
- it has a recoverable runtime family
- it produces the benchmark local audio artifacts
- it is the right place to measure sound quality, local continuity, and baseline stability

Circleworld RAFA is currently a research prototype:
- it can intervene on coherent phase structure
- it can promote packets and recurse
- it can preserve real-audio anchors while making bounded local changes
- but it does not yet produce a genuinely new macro-time law
- and it does not yet show Matryoshka-like nested commitment under fork/resume

So the correct reading is:

- Graduation RAFA currently owns the sound-engine lane
- Circleworld RAFA currently owns the world-building hypothesis lane
- the bridge question is whether Circleworld can improve time development, repetition, and lawful continuation without destroying the real-audio frontier

## What We Restored And Learned About Graduation RAFA

The graduation-pack restoration work established several things clearly.

### What was false

The attempted restoration through:

- `sample_diffusion.py`
- `diffusion_models.py`
- `checkpoints_diffusion_rafa_full/diff_step1000.pt`

was the wrong path for the original graduation pack.

That path:
- produced 1-second outputs
- had dead prompt control
- did not match the original graduation artifact family

### What is now confirmed

The original graduation pack belongs to the stage-4 Blackwell family:

- runtime family: `NakedRAFA` + `NakedDenoiser`
- main export lane: `tools/export_audio.py`
- checkpoint family: `checkpoints_stage4/bound_weights_*.pt`

There is also a real checkpoint/runtime seam:

- pre-`ep50`: older 14-key family
- `ep50+`: newer 16-key family with `g1_w_ph` and `g1_w_mag`

That seam matters because the historical graduation artifacts appear to belong to the older family, while the modern runtime drifted toward the newer one.

### Best current restore status

The best current practical restore lane is:

- restore output: [D:\RAFA\outputs\graduation_pack_restore_candidate](D:\RAFA\outputs\graduation_pack_restore_candidate)
- canonical findings: [D:\RAFA\docs\reports\GRADPACK_RESTORE_FINDINGS_2026-04-05.md](D:\RAFA\docs\reports\GRADPACK_RESTORE_FINDINGS_2026-04-05.md)

Important read:
- we have a recoverable, usable, deterministic graduation-family baseline
- it is close enough to treat as the live engineering anchor
- it is not yet proven bit-exact to the original historical path

### Practical doctrine for Graduation RAFA

Graduation RAFA should be treated as:

- the sound engine
- the certified benchmark lane
- the continuity/stability baseline
- the place where restoration and production-style audio questions are answered

It should not be overloaded with proving the full metaphysics.

## What We Actually Have In Circleworld RAFA

Circleworld is no longer philosophy-only. It has real implementation.

### Implemented capabilities

Circleworld currently has:

- an arc-style rational concentration field
- promotability scoring
- packet promotion
- recursive packet feedback
- active vs passive packet ablations
- held-out phase evaluation
- real-anchor benchmark evaluation
- continuity / repetition benchmarking
- fork/resume nested-commitment testing

Core runtime and evaluation paths:

- [D:\RAFA\runtimes\circleworld_proto\README.md](D:\RAFA\runtimes\circleworld_proto\README.md)
- [D:\RAFA\runtimes\circleworld_proto\evaluate_circleworld.py](D:\RAFA\runtimes\circleworld_proto\evaluate_circleworld.py)
- [D:\RAFA\runtimes\circleworld_proto\benchmark_circleworld_real_anchor.py](D:\RAFA\runtimes\circleworld_proto\benchmark_circleworld_real_anchor.py)
- [D:\RAFA\runtimes\circleworld_proto\benchmark_audio_continuity.py](D:\RAFA\runtimes\circleworld_proto\benchmark_audio_continuity.py)
- [D:\RAFA\runtimes\circleworld_proto\test_nested_commitment.py](D:\RAFA\runtimes\circleworld_proto\test_nested_commitment.py)
- [D:\RAFA\lineages\04_positive_replacement\circleworld.py](D:\RAFA\lineages\04_positive_replacement\circleworld.py)

### Best current Circleworld listening baseline

The current best listening baseline is the constrained real-anchor lane:

- [D:\RAFA\outputs\circleworld_proto\inference_2026-04-09_realanchor_constrained_ab](D:\RAFA\outputs\circleworld_proto\inference_2026-04-09_realanchor_constrained_ab)

This is the current best compromise among:

- preserving real audio
- allowing some intervention
- avoiding the most pathological metric gaming

### What the retrospective grid says

The grid comparison established three clear regimes:

1. Preservation winners
- high real-audio similarity
- but structurally close to no-op

2. Structural winners
- strong held-out structural metrics
- but they damage or distort real audio

3. Balanced middle
- weaker than preservation on raw similarity
- weaker than structural winners on internal scores
- but the only regime that looks usable as a forward research baseline

Canonical grid report:

- [D:\RAFA\outputs\circleworld_proto\retrospective_grid_2026-04-10\RETROSPECTIVE_GRID.md](D:\RAFA\outputs\circleworld_proto\retrospective_grid_2026-04-10\RETROSPECTIVE_GRID.md)

### What the continuity benchmark says

The continuity benchmark asked the right question:

- is Circleworld actually changing the time-world?
- or is it mostly preserving the anchor's macro-time behavior?

Answer:

- the best current real-anchor Circleworld runs are almost identical to their anchor references on loop and reentry metrics
- they are not yet creating a new macro-time law

Canonical continuity report:

- [D:\RAFA\docs\reports\SHARED_BATTLEFIELD_CONTINUITY_2026-04-10.md](D:\RAFA\docs\reports\SHARED_BATTLEFIELD_CONTINUITY_2026-04-10.md)

This is a very important result because it means the current Circleworld wins are mostly:

- local intervention
- bounded preservation
- not long-horizon scene governance

### What the nested-commitment test says

The fork/resume assay asked the deeper Matryoshka question:

- does an early/coarse state define a world
- while later recursion only refines it?

We tested:

- preservation regime
- balanced regime
- structural regime

Result:

- none of them established nested commitment
- all three behaved as over-rigid attractors under fork/resume

That means:

- packet perturbations are effectively no-ops
- fine perturbations change texture slightly, but not meso development
- promotion perturbations do not generate lawful sibling branches

Canonical report:

- [D:\RAFA\docs\reports\NESTED_COMMITMENT_RETROSPECTIVE_2026-04-11.md](D:\RAFA\docs\reports\NESTED_COMMITMENT_RETROSPECTIVE_2026-04-11.md)

This is probably the most important Circleworld result so far.

It says the missing thing is not just "less looping."

The missing thing is:

- explicit hierarchical commitment
- coarse world-state
- meso development-state
- fine detail-state

without those being overwritten by later recursion

## What We Have Diverged Into

The project is no longer one experiment.

It is at least two real experiments plus a bridge.

### 1. Graduation RAFA

Question:
- can the system generate stable, coherent audio and serve as a trusted benchmark?

Primary role:
- sound engine
- continuity and stability anchor
- certified runtime family

### 2. Circleworld RAFA

Question:
- can recursive promotion and local-law machinery become object-like and eventually world-like?

Primary role:
- ontology research lane
- recursive formalization prototype
- candidate mechanism for future non-symbolic world-building

### 3. Shared battlefield

Question:
- does Circleworld help on the actual sound-engine problems that Graduation exposes?

That battlefield should remain small and concrete:

- collapse / stability
- repetition / reentry
- macro-time continuity
- structural evolution over time
- ablation sensitivity

This is where the two lanes can talk to each other without being confused for the same thing.

## What We Know, Cleanly

### Graduation RAFA

Known:

- the original graduation pack was not a diffusion-sampler artifact
- the stage-4 Blackwell family is the right restoration family
- a deterministic, usable restore candidate exists
- Graduation is the correct engineering anchor

Unknown:

- the exact historical pre-`g1_*` forward path
- whether the current restore candidate is literally the historical generating path

### Circleworld RAFA

Known:

- packetization and promotion exist
- active local intervention exists
- the model can preserve real-audio anchors fairly well in constrained regimes
- structural-only wins can be pathological
- continuity is currently mostly inherited from the anchor
- nested commitment is not yet established

Unknown:

- whether Circleworld can truly reduce looping without flattening itself into preservation
- whether packets can become lawful child objects rather than decorative summaries
- whether a true child-world / fractal-bin architecture will deliver meso-time development

## Recommendation For Graduation RAFA

### Keep doing

- treat Graduation as the engineering baseline
- preserve the stage-4 restore/runtime family carefully
- keep restoration, evaluation, and generation separate from Circleworld experiments
- use Graduation as the benchmark for:
  - local audio quality
  - baseline continuity
  - failure mode discovery

### Do next

1. Freeze the current restore path as the certified baseline
2. Add one stable benchmark harness for graduation audio outputs
3. Keep continuity and loop measurements attached to that lane
4. Recover the exact historical 14-key forward only if archival exactness becomes necessary

### Do not do

- do not use Circleworld metrics as proof that Graduation got better
- do not route Graduation restoration through diffusion-path confusion again
- do not mutate the graduation runtime in place while ontology work is happening

## Recommendation For Circleworld RAFA

### Keep doing

- keep Circleworld as a separate runtime and research lane
- keep evaluating on:
  - real-audio benchmark
  - continuity benchmark
  - nested-commitment benchmark

### Immediate recommendation

Do **not** jump straight into "optimize less looping" as the main next move.

The nested-commitment test showed why:

- if you optimize only for less looping now,
- you may simply make the system flatter, safer, and more frozen
- instead of making it more Matryoshka-like

### Actual next implementation move

The next implementation step should be architectural, not just search-based:

split the internal state into:

- coarse world-state
- meso development-state
- fine detail-state

and then require later recursion to refine rather than overwrite.

This is the real Matryoshka move.

### After that

Only after that split exists should Circleworld run a new continuity-aware training search.

That future search should explicitly reward:

- preserved coarse identity
- related but non-identical meso development
- variable fine detail
- reduced reentry / loop tendency

In other words:

- not "just less repetitive"
- but "lawful sibling continuation"

### Practical near-term Circleworld plan

1. Implement explicit hierarchical state bands
2. Re-run the nested-commitment assay
3. If nested commitment improves, then add continuity-aware training terms
4. Re-test against the shared battlefield with Graduation

## Recommendation For The Relationship Between The Two

The right relationship is:

- Graduation = sound engine
- Circleworld = world-building candidate
- shared battlefield = whether world-building improves sound-engine failure modes

That means:

- they should remain alive in parallel
- they should not be forced to answer the same question
- they should only be compared on explicit shared benchmarks

The bridge question from here on should be:

Can Circleworld produce a form of lawful continuation that helps Graduation-type generation avoid ghost loops and short attractor reentry?

That is the real bridge.

## Suggested Program From Here

### Graduation lane

1. Keep the restored stage-4 baseline stable
2. Maintain sound-engine benchmarks
3. Use it to surface concrete continuity failures

### Circleworld lane

1. Implement hierarchical state separation
2. Re-run fork/resume nested-commitment testing
3. Only then do continuity-aware search
4. Then test whether the resulting Circleworld behavior helps on the shared battlefield

### Shared evaluation lane

Keep a small permanent dashboard of:

- real-audio similarity
- continuity / reentry metrics
- nested-commitment status
- structural score

If Circleworld cannot improve the shared battlefield after the hierarchical-state change, then that will be a strong signal that the present packet formalism is too weak and needs a more explicit child-world architecture.

## Bottom Line

Graduation RAFA is currently real as an engineering system.

Circleworld RAFA is currently real as a research prototype.

But Circleworld has not yet crossed the line from:

- recursive packetization and bounded intervention

to:

- Matryoshka-like lawful world-building

That line is now identifiable.

And because we now have:

- a real-audio benchmark
- a continuity benchmark
- a nested-commitment benchmark

the next phase can be much less foggy than the last one.
