# RAFA Attention / QKV / Token Constitution

This document defines how RAFA uses the words `token`, `attention`, `query`,
`key`, and `value`.

The purpose is to prevent a quiet collapse back into transformer language. RAFA
may use dense vectors and QKV-shaped machinery, but the claim is different:
phase-native relational laws should be selected by resonance and applied as
operators.

## Core Distinction

### Transformer attention

Transformer attention asks:

`Which content vector from another token position should this token read?`

Typical mapping:

- token: externally segmented symbol or patch
- query: learned projection of token state
- key: learned projection advertised by another token
- value: learned content vector to mix into the residual stream
- score: mostly dense dot product
- output: mixed hidden vector

### RAFA attention

RAFA attention asks:

`Which local phase law should act on this region of the lattice?`

RAFA mapping:

- unit: internally earned relational structure, not an input symbol
- query: unresolved phase/rational/support tension
- key: advertised relational law and compatibility signature
- value: operator payload that changes phase/law state
- score: q/arc/support/phase geometry plus learned residual correction
- output: updated phasor lattice, support, q-trace, branch state, or child return

The value is the critical difference. In RAFA, a value is not merely content to
copy. A value is a small law, carrier, or operator to apply.

## RAFA Unit Taxonomy

### Raw lattice site

A raw lattice site is a phase coordinate. It is not a token.

It may hold:

- unit phasor state
- local support
- local q/arc evidence
- branch-mode state

It does not count as an object until a law survives around it.

### Packet

A packet is a promoted local phase-law candidate.

It is token-like only in the weak sense that it becomes addressable. It is not a
discrete symbol. It must carry evidence of local structure, support, q/arc law,
and possible continuation behavior.

### Childworld

A childworld is a recursive local law carrier.

It is stronger than a packet because it has its own local state and can run a
local recurrence. It still does not count as a RAFA post-token unless it can be
retrieved, composed, and causally used.

### Relational signature

A relational signature is a compressed addressable law object.

Dense signatures are allowed, but the dense body is not the ontology by itself.
The signature must expose enough structure for q/arc, support, branch, temporal,
and operator probes to recover useful law.

### RAFA post-token

A RAFA post-token is an earned relational object.

Minimum criteria:

- retrieval: partial relational queries activate it above decoys
- composition: it can participate in parent/child/grandchild refinement
- causal use: enabling its value/operator changes the downstream state
- stability: it preserves unit-phasor and gauge contracts
- portability: it is reusable across more than one seed/context family

No object should be called a RAFA token unless these criteria are explicitly
reported. Before that, call it a packet, child, signature, or law object.

## RAFA Q/K/V Contract

### Query

`Q` is the local unresolved need.

Allowed query components:

- phase defect or phase disagreement
- q-profile gap or unstable rational support
- minor-arc residue pressure
- major-arc underfit
- support mismatch
- branch ambiguity
- reentry debt or loop risk
- parent/child boundary mismatch
- temporal lifecycle need, such as attack, decay, sustain, or handoff pressure

Interpretation:

`This region needs a law that resolves this tension without destroying the
world.`

### Key

`K` is the advertised relational law.

Allowed key components:

- q-profile and q persistence
- major-arc footprint
- minor-residue texture
- support geometry
- local lifecycle profile
- branch-mode compatibility
- child boundary signature
- law signature or operator seed summary

Interpretation:

`This object can explain or carry this kind of structure.`

### Value

`V` is the operator payload.

Allowed value components:

- local phase delta
- phasor carrier
- transport/update law
- q-bias
- support update
- coherence target
- branch survival or merge influence
- child writeback operator
- low-rank local law generated from an operator seed

Interpretation:

`If selected, this is the law I apply.`

## Scoring Contract

RAFA-labeled attention uses:

`score = geometric_score + learned_residual`

The learned residual may improve selection, but it must not erase the geometric
spine.

### Required geometric components

At minimum, RAFA attention reports explicit contributions for:

- q compatibility
- major-arc overlap
- minor-residue fit or disagreement
- support overlap
- boundary compatibility when child/parent state is involved
- temporal-law compatibility when continuation is involved
- reentry or loop-risk penalty when recurrence is involved

### Learned residual rule

A learned residual is allowed only if reports include:

- geometric score distribution
- learned residual distribution
- final score distribution
- residual-to-geometry ratio
- ablation with learned residual disabled
- ablation with geometry disabled or shuffled

If the learned residual wins while geometry is useless, the run may be useful but
it is not evidence for RAFA-native attention.

## Current Implementation Status

Circleworld already contains a QKV-shaped precursor:

- `lineages/04_positive_replacement/circleworld.py`
- `_relational_branch_attention(...)`
- `branch_law_version = "relational_qkv_v2"`

That function builds small per-mode query, key, and value tensors from branch
features and applies local 2-mode attention. It then produces a handoff drive for
branch coupling.

This is useful evidence that QKV-shaped routing can live inside Circleworld.

It is not yet canonical RAFA attention because:

- it couples only the local mode pair
- it does not retrieve reusable packet/child/grandchild law objects
- its value is still a branch feature message, not a general phase-law operator
- it does not prove post-token retrieval or composition
- it does not by itself test resonant memory scaling

Therefore the correct label is:

`local branch-QKV precursor`

not:

`full RAFA attention`

## Relationship To Dense Signatures

Dense signatures are permitted as compressed carriers.

They are not sufficient proof of RAFA post-tokens. A dense signature becomes
token-like only when it passes retrieval, composition, and causal-use tests.

Safe phrasing:

- `dense relational signature`
- `compressed law carrier`
- `post-token candidate`

Unsafe phrasing without evidence:

- `RAFA token`
- `learned ontology`
- `reasoning token`

## Relationship To CLAP Or Text Prompting

Text or CLAP can be used as an external pointer.

It does not prove RAFA discovered semantics. A prompt vector may initialize or
select a law family, but internal evidence requires alignment with discovered
phase/q/arc structures.

Safe claim:

`The external prompt points toward an internal law family.`

Unsafe claim:

`The lattice learned the concept because CLAP selected it.`

## Required Assays

### Resonant Child Retrieval Assay

Purpose:

Test whether partial relational queries retrieve compatible packet, child, or
grandchild law objects above decoys.

Inputs:

- bank of law objects from packet/child/grandchild records
- partial query masks over q/arc/support/phase/lifecycle fields
- decoy objects from incompatible q, support, or lifecycle families

Metrics:

- top-1 and top-k family retrieval
- resonance margin between correct family and strongest decoy
- decoy suppression ratio
- resonance entropy
- geometric/residual contribution split

Success:

Compatible law families activate above decoys without external labels at
inference time.

Failure:

Retrieval only works by stored IDs, labels, or dense shortcuts with no geometric
margin.

### Cross-Depth Composition Assay

Purpose:

Test whether a parent query can activate a child, the child can activate a
grandchild, and the grandchild can return a coherent refinement.

Metrics:

- parent-to-child activation margin
- child-to-grandchild activation margin
- q/arc refinement gain
- support refinement gain
- parent boundary preservation
- world-jump rate
- operator causality with writeback disabled vs enabled

Success:

Cross-depth activation improves relational fit while preserving coarse parent
identity.

Failure:

Depth only multiplies records, collapses to the same basin, or causes overwrite.

### Interference Selectivity Assay

Purpose:

Test whether compatible laws constructively activate and incompatible laws
suppress or cancel.

Metrics:

- compatible activation gain
- incompatible suppression ratio
- mixture entropy
- wrong-family activation under adversarial decoys
- stability under increasing bank size

Success:

The correct law remains selectable as the number of decoys increases.

Failure:

More children merely add noise, or the strongest scalar packet always wins.

### Audio Bridge Assay

Purpose:

Keep Circleworld honest against the original RAFA audio mission.

When a retrieved law is applied to audio continuation, report:

- loop autocorrelation
- first-chunk reentry
- adjacent and nonlocal chunk similarity
- phase coherence
- q-profile evolution
- residue evolution
- continuity score
- audio benchmark drift

Audio is not the only evidence for resonant memory, but audio remains the hard
battlefield against decorative ontology.

## Reporting Rules

Any RAFA-attention report must state:

- what objects were queried
- what objects advertised keys
- what operator values were applied
- whether values were content vectors, phase operators, or branch-law messages
- geometric score components
- learned residual contribution
- geometry-disabled control
- residual-disabled control
- unit-phasor and gauge-invariance status

Any report using `relational_qkv_v2` must call it a local branch-QKV precursor
unless it also runs retrieval and composition assays over reusable law objects.

## Promotion Rules

### Promote to RAFA attention candidate

Allowed if:

- geometric score components are reported
- learned residual improves but does not replace geometry
- values act causally as operators
- unit-phasor and gauge contracts pass
- at least one retrieval or composition assay is nonzero

### Promote to RAFA post-token candidate

Allowed if:

- retrieval works above decoys
- cross-depth composition is nonzero
- operator causality is nonzero
- object persists across more than one seed/context family
- dense-only negative controls do not explain the result

### Do not promote

Block promotion if:

- attention is just dense dot-product routing over unlabeled embeddings
- child IDs retrieve themselves without relational compatibility
- values are inert summaries
- geometry ablations do not matter
- audio or ontology effects disappear when writeback/value operators are disabled

## Immediate Engineering Implication

The next implementation should not rename `relational_qkv_v2` into "RAFA
attention." It should build a separate resonant retrieval/composition assay that
uses the current child, packet, and branch artifacts as candidate law objects.

The current bridge result remains important: child carriers can convert into
parent-mode ontology under carrier-only conditions. The next question is whether
those carriers can be addressed and composed by relational resonance rather than
selected by explicit assay routing.
