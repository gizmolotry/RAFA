# Circleworld Relation Token Build (2026-04-20)

## Scope

This report covers a Circleworld-only build pass for a new relation-token layer.

The user direction was:

- relation token should come through Ramanujan on the lattice
- analyze existing Circleworld machinery first
- avoid rebuilding structures that already exist

Graduation runtime was not modified.

## Redundancy Check

Before building anything new, I checked what Circleworld already had.

Already present in `D:\RAFA\lineages\04_positive_replacement\circleworld.py`:

- Hardy-Littlewood-like arc field
- promotability field
- promoted packets
- active packet feedback
- native multimode branch state
- relation-mediated branch law (`relational_qkv_v2`)
- q-trace inheritance
- masking sidecars and nested-commitment evaluation

That means Circleworld already had:

- local coherence detection
- packet selection
- branch-side relational coupling
- local rational history

What it did **not** have was:

- a reusable cross-case library of promoted law families
- a clean serialization of those families for downstream inspection
- a Circleworld-native place to say "this family of Ramanujan-shaped local laws recurs across cases"

So the new build did **not** add another packetizer or another branch sidecar.

It added a token layer **on top of** the existing promoted packets.

## What Was Built

Primary runtime file updated:

- `D:\RAFA\lineages\04_positive_replacement\circleworld.py`

New Circleworld concepts:

- `law_packets`
- `law_token_library`

New config surface:

- `law_packet_merge_threshold`
- `law_packet_min_score`
- `law_packet_topk_families`

New runtime script:

- `D:\RAFA\runtimes\circleworld_proto\build_law_token_library.py`

Supporting runtime updates:

- `D:\RAFA\runtimes\circleworld_proto\export_circleworld_audio.py`
- `D:\RAFA\runtimes\circleworld_proto\evaluate_circleworld.py`
- `D:\RAFA\runtimes\circleworld_proto\train_circleworld.py`
- `D:\RAFA\runtimes\circleworld_proto\README.md`

## Constitution Of The New Layer

Each promoted packet is now formalized into a richer law packet with:

- depth index
- major mass
- minor residue
- harmonic ratio
- persistence
- promotability
- concentration
- sharpness
- pair strength
- q-mass profile
- attack / decay / reentry terms
- normalized law signature

The token library then clusters those law packets by cosine similarity over the law signature.

So the relation token here is:

- not a symbolic label
- not a prompt token
- not a new branch slot

It is:

- a recurring family of promoted Ramanujan-shaped local laws extracted from the existing lattice dynamics

That is the closest non-redundant implementation of the user’s direction inside the current Circleworld code.

## Build Runs

Expanded anchor case set:

- `D:\RAFA\outputs\circleworld_proto\expanded_anchor_cases_2026-04-09.json`

Both runs used:

- clip length: `10 s`
- device: `cuda`
- mode: `native_multimode`

### 1. Ramanujan Relation-Token Library

Config:

- `D:\RAFA\outputs\circleworld_proto\branchlaw_ablation_series_2026-04-17\relational_qkv_v2_ramanujan\checkpoint\circleworld_real_anchor_config_cem_v1.json`

Artifacts:

- `D:\RAFA\outputs\circleworld_proto\law_token_library_2026-04-20_ramanujan\law_token_library.json`
- `D:\RAFA\outputs\circleworld_proto\law_token_library_2026-04-20_ramanujan\law_packets.json`
- `D:\RAFA\outputs\circleworld_proto\law_token_library_2026-04-20_ramanujan\LAW_TOKEN_LIBRARY.md`

Headline numbers:

- cases: `12`
- mean law packets per case: `48.0`
- mean law families per case: `2.167`
- aggregate law packets: `576`
- aggregate families: `6`

Top aggregate family:

- count: `426`
- top q: `2`
- mean score: `0.3465`
- mean major mass: `0.0660`
- mean minor residue: `0.0519`
- mean promotability: `0.2043`

### 2. Non-Ramanujan Comparison Library

Config:

- `D:\RAFA\outputs\circleworld_proto\branchlaw_ablation_series_2026-04-17\relational_qkv_v2_qtrace_only\checkpoint\circleworld_real_anchor_config_cem_v1.json`

Artifacts:

- `D:\RAFA\outputs\circleworld_proto\law_token_library_2026-04-20_qtrace_only\law_token_library.json`
- `D:\RAFA\outputs\circleworld_proto\law_token_library_2026-04-20_qtrace_only\law_packets.json`
- `D:\RAFA\outputs\circleworld_proto\law_token_library_2026-04-20_qtrace_only\LAW_TOKEN_LIBRARY.md`

Headline numbers:

- cases: `12`
- mean law packets per case: `34.25`
- mean law families per case: `2.167`
- aggregate law packets: `411`
- aggregate families: `6`

Top aggregate family:

- count: `273`
- top q: `2`
- mean score: `0.3736`
- mean major mass: `0.0753`
- mean minor residue: `0.0521`
- mean promotability: `0.2332`

## Interpretation

### 1. The token layer is real now

This is no longer just packet promotion.

Circleworld can now export:

- per-case law packets
- per-case law families
- a cross-case aggregate family library

So there is now a concrete runtime artifact that corresponds to a relation-token candidate.

### 2. Ramanujan increases token yield, not token diversity

Compared with `qtrace_only`, the Ramanujan run produced:

- more law packets per case: `48.0` vs `34.25`
- the same mean family count per case: `2.167`
- the same aggregate family count: `6`

So Ramanujan, in the current constitution, gives **more packetized law evidence**.

But it does **not** yet give richer family diversity.

### 3. The current law-token ontology is still low-q collapsed

In both libraries:

- every top family is dominated by `q = 2`
- one family massively dominates the token count

So the present token layer is still saying:

- "there is a recurring low-q lawful corridor"

more than it is saying:

- "there are many distinct relational object types"

That is the main limitation.

### 4. Voice-like cases still show the most family variation

In both runs, `voice_alt` was the strongest case for family diversity:

- `4` families

Most other cases still sit around:

- `2` families

That suggests the current token constitution is more sensitive to quasi-vocal / sustained spectral ambiguity than to engines, impacts, or drones.

### 5. This does not solve branching

The earlier branch-law ablation still stands:

- real live branching was not established
- nested commitment was not established

This new layer does not change that diagnosis.

It gives us a better way to inspect what local laws keep recurring while the system remains mostly single-path or over-rigid.

## What This Means

The clean read is:

- we now have a valid Circleworld relation-token build
- it is non-redundant with packets and branch law
- it is still structurally narrow

Right now the token library captures:

- repeated low-q lawful motifs

It does not yet capture:

- a broad society of distinct relational object types

So the build is successful as infrastructure, but only partial as ontology.

## Recommendations

### 1. Keep this token layer

This should stay.

It gives Circleworld a proper inspection surface for:

- repeated law families
- cross-case recurrence
- downstream token-conditioned experiments

### 2. Do not treat these families as semantic objects yet

The current evidence is too collapsed.

They are better understood as:

- recurrent low-q law families

not:

- clarinet token
- airplane token
- bell token

### 3. Next training should optimize token diversity explicitly

The current training objectives still allow one low-q family to dominate.

The next Circleworld-only objective should add pressure for:

- family diversity across cases
- reduced dominant-family share
- reduced dominant-q share inside the token library
- persistence of family identity across depth without full collapse to one prototype

### 4. Ramanujan should stay as a kernel, but not as an article of faith

Current data says:

- Ramanujan produces more law packets
- `qtrace_only` still wins the broader branch-law benchmark

So the right next stance is:

- keep Ramanujan in the constitution
- compare it empirically against other kernels
- do not force it as the only valid branch law until it wins on the right metrics

### 5. The next clean experiment

The most useful next experiment is:

- train Circleworld with an explicit token-diversity sidecar
- then rebuild the law-token library on the same 12 anchors
- compare:
  - aggregate family count
  - dominant family share
  - dominant q share
  - case-level family count
  - continuity / loop metrics

That would directly test whether the new ontology is becoming richer, rather than just more heavily packetized.

## Bottom Line

The relation token is now implemented in Circleworld in the least redundant way available in the current codebase:

- promoted packet -> law packet -> clustered law family

That is a valid build.

But the first result is still narrow:

- the ontology is real
- the ontology is mostly low-q
- branching is still weak
- semantic objecthood is not yet established
