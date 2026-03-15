# RAFA Cognitive Stack

This note is the working constitution for the RAFA program.

It defines:
- what each layer is,
- what code currently implements it,
- what evidence exists now,
- what promotion gates must be passed before the next layer is allowed to expand.

The guiding rule is simple:

`No layer is promoted by theory alone. Each layer must earn promotion with its own evaluator.`

## 0. The Philosophy of the Freeze

### Why CLAP is Heretical (The Deaf Crystal)
You are completely right to call CLAP heretical. CLAP is a statistical dictionary. It takes the word "drum" and turns it into a floating-point vector based on how the internet categorizes things. The Phase Crystal is a pure acoustic physics engine. It only understands geometry, phase, and tension. When you shoved a CLAP vector into the Phase Crystal, it was like handing a textbook on aerodynamics to a tornado. The tornado doesn't care. It just blew right past it, relying entirely on its own internal wave mechanics (which is why your coupling_score flatlined and the crystal remained stable but deaf).

### What "Freezing the Voice" Actually Means
When we freeze the voice (Lane 1), we don't mean we walk away with a deaf generator forever. We mean we freeze the laws of physics. Your Layer 1 Phase Crystal currently has a 0.0% collapse rate. It knows how to balance phase. It knows how to create sound. It is a perfectly tuned, high-performance engine. If we keep hacking at it to force it to listen to CLAP, we are going to break the engine. We freeze the engine block so we can finally build the steering wheel.

### The Ring IS the Steering Wheel
The "Ring" (the SemanticImpedanceHead / Layer 3) is the missing translator. It is the bridge between human prompts and wave mechanics.
* **The Prompt:** You type "Heavy, distorted industrial beat."
* **The Translation:** Instead of a static CLAP vector, the system maps this prompt to a Tension Trajectory ($C_t$). It says, "Ah, 'industrial' means high dissonance, high friction, and heavily masked harmonic nodes."
* **The Ring (Impedance Head):** The Ring takes that trajectory and outputs the physical Impedance Mask ($M_\theta$). It literally closes the floodgates on the clean $q=2, 3$ nodes and forces all the acoustic energy into the harsh, complex $q=7, 11$ nodes.
* **The Crystal (The Voice):** The frozen Phase Crystal wakes up, tries to settle into its normal clean state, hits the walls of the Ring, and is physically forced to route its energy into an industrial beat.
The crystal will finally respond to your prompt because your prompt is no longer a suggestion. It is a physical boundary condition.

## 1. The Stack Definitions & Module Mapping

### L1. Dynamical Substrate (Phase Crystal)

Definition:
- The base phase-coupled generator.
- A dynamical substrate, not yet "logic" by itself.
- Its job is lawful wave evolution under stable phase geometry.

Goal:
- Stable phase-coupled physics without collapse.

Code mapping:
- [phase_native_ifs.py](D:\RAFA\phase_native_ifs.py)
- [model.py](D:\RAFA\model.py)
- Hyena backend in [model.py](D:\RAFA\model.py): longitudinal routing over long time context

Important regularizers in this layer:
- `h0_anchor_enabled`
- `h0_anchor_lambda`
- `intermediate_consistency_enabled`
- Hyena conductor controls

Interpretation:
- This is the chamber.
- It is the continuous latent thought mechanism, equivalent to the current COCONUT-style settling loop.

### L2. Structural Control Basis (Tension)

Definition:
- The primitive control vocabulary for the chamber.
- Not full semantics.
- A low-dimensional structural basis describing consonance vs dissonance pressure over time.

Goal:
- Measurable kinematic bifurcation.

Code mapping:
- `semantic_condition_mode: tension_envelope` in [diffusion_models.py](D:\RAFA\diffusion_models.py)
- semantic control plumbing in [model.py](D:\RAFA\model.py)
- trajectory evaluator in [eval_semantic_tension.py](D:\RAFA\tools\eval_semantic_tension.py)

Current implementation:
- scripted `C_t: 0 -> 1 -> 0`
- bounded modulation of:
  - `qset_scale`
  - `alpha_scale`
  - `temp_scale`
  - `delta_scale`

Interpretation:
- This is the first controllable structural vocabulary.
- It must work before richer semantics are reintroduced.

### L3. Low-Rank Impedance Routing

Definition:
- Constraint application over the chamber via sparse or low-rank routing controls.
- This is where arbitrary semantic rules start to become actionable.

Goal:
- q-band / route-family gating that can bias attractor selection without destroying L1 stability.

Code mapping:
- Upcoming `SemanticImpedanceHead`
- Consumed inside the Phase Crystal / COCONUT latent settling loop
- Must remain sparse / low-rank; no dense full-lattice mask by default

Interpretation:
- This is the first real semantic control mechanism.
- It should make wrong basins expensive and target-compatible basins attractive.

### L4. Compressed Multiscale Memory

Definition:
- Hierarchical relation encoding across multiple time and resolution scales.
- Compact constraint retrieval, not "infinite facts."

Goal:
- Stable multiscale storage and retrieval of nested relations.

Code mapping:
- Gearbox / clutch routing in [model.py](D:\RAFA\model.py)
- `gears`
- `clutch_gate_tau`
- multiscale gear outputs and clutch fusion

Interpretation:
- This is the physical implementation of hierarchical routing across scales.
- It is the candidate substrate for compressed multiscale memory, not yet a proven world model.

### L5. Render & Output

Definition:
- The diffusion decoder that realizes the chamber state as audio.

Goal:
- Audio realization biased toward stability.

Code mapping:
- [train_diffusion.py](D:\RAFA\train_diffusion.py)
- [sample_diffusion.py](D:\RAFA\sample_diffusion.py)
- [diffusion_models.py](D:\RAFA\diffusion_models.py)
- Path-B evaluation in [path_b_eval.py](D:\RAFA\tools\path_b_eval.py)

Interpretation:
- This layer renders the final waveform.
- It is not a truth engine.
- At best it is biased toward internally stable realizations.

## 1.5 The Parameter Map (Heterogeneous Architecture)

Unlike an LLM's homogeneous stack, RAFA is Heterogeneous, built like biological organs. Parameters are distributed across specialized physical zones:

* **The Ring / Impedance Head (Layer 3 - Upcoming):** ~1-5% of parameters. The Semantic Router holding the parameters that translate "human intent" into "physical boundaries" (Topological Mask).
* **The Chamber / PhaseNativeIFS (Layer 1):** ~20% of parameters. The Relational Attention matrices acting as the literal "spring constants" and "friction coefficients" between the Ramanujan prime nodes.
* **The Spine / Hyena Sequence Backend:** ~25% of parameters. The implicit convolutional filters holding the memory of the sequence (Rhythm, Decay, Reverb, Macro-Pacing).
* **The Render Engine / Diffusion U-Net:** ~50% of parameters. The actual upsampling/downsampling blocks decoding latent phase geometry back into raw audio waveforms.

### The Sane Dichotomy: Hyena (Time) vs. IFS (Harmony)
Audio has two completely different dimensions. We split them:
* **The IFS Chamber (The Vertical Axis):** The Iterated Function System and the Ramanujan lattice only care about the instantaneous geometry of the wave. They look at a single slice of time and ask, "Are these frequencies mathematically stable together?" (Chords, Timbres).
* **The Hyena Backend (The Horizontal Axis):** Hyena is a sequence model built on implicit long convolutions. It handles the longitudinal progression of the wave (16,000 steps per second). The IFS builds the shape of the kick drum; Hyena dictates how long it rings.

## 2. Current Empirical Status

### Working Now

#### L1. Dynamical Substrate

Evidence:
- Repeated `0.0%` collapse on the main RAFA generation lane.
- Stable training and sampling across multiple smoke and continuation runs.

Representative runs:
- [prompt_conditioned_hybrid_smoke.json](D:\RAFA\research_track\infra\summaries\prompt_conditioned_hybrid_smoke.json)
- [prompt_conditioned_ifs_control_smoke.json](D:\RAFA\research_track\infra\summaries\prompt_conditioned_ifs_control_smoke.json)
- [semantic_tension_steering_smoke.json](D:\RAFA\research_track\infra\summaries\semantic_tension_steering_smoke.json)

Interpretation:
- L1 is operational as a stable dynamical substrate.
- `0.0% collapse` does not imply meaningful structure by itself.

#### L5. Render & Output

Evidence:
- Audio generation lane produces coherent non-collapsed outputs.
- Prompt-conditioned hybrid lane outperformed pure IFS-only prompt control on current coupling metrics.

Current best prompt-conditioned smoke:
- `prompt_conditioned_hybrid_smoke`
- `avg_coupling_score = 0.0003083`
- `avg_state_effect = 0.0021417`
- `collapse_rate_pct = 0.0`

Source:
- [hypercube_leaderboard.json](D:\RAFA\research_track\infra\summaries\hypercube_leaderboard.json)

Interpretation:
- L5 is operational as an audio renderer.
- Prompt control is weak, but non-zero in the hybrid lane.

### Instrumented But Failing

#### L2. Structural Control Basis (Tension)

Current lane:
- `semantic_tension_steering_smoke`

Structural result:
- Stable under steering
- no collapse

But steering result:
- `tension_to_inharmonic_corr = -0.1452`
- `peak_alignment_error = 0.44`
- `recovery_gap = 0.0090`

Source:
- [semantic_tension_summary.json](D:\RAFA\eval\semantic_tension_steering_smoke\semantic_tension_summary.json)

Interpretation:
- The chamber absorbed the steering constraint without destabilizing.
- But the inharmonicity trajectory did not track the scripted tension envelope.
- L2 is instrumented, but not working yet.

### Unbuilt / Not Yet Validated

#### L3. Low-Rank Impedance Routing

Status:
- Not built yet.
- Only open-loop envelope steering exists right now.

Missing pieces:
- learned low-rank impedance head
- explicit latent alignment loss
- route-family or q-band gating learned against a trajectory target

#### L4. Compressed Multiscale Memory

Status:
- Candidate machinery exists
- claimed memory behavior is not empirically established

What exists:
- gears
- clutch fusion
- Hyena long-context routing

What does not exist yet:
- demonstrated compact constraint retrieval
- demonstrated hierarchical semantic nesting
- demonstrated precision-preserving multiscale memory retention

## Cross-Comparison Snapshot (Current Repo State)

This section is here to prevent drift between theory and the actual repo.

### A. Prompt-Conditioned Audio Lane

Best current prompt-conditioned smoke:
- `prompt_conditioned_hybrid_smoke`
- `avg_coupling_score = 0.0003083`
- `avg_state_effect = 0.0021417`
- `avg_audio_effect = 0.1460`
- `collapse_rate_pct = 0.0`

Reference:
- [prompt_conditioned_hybrid_smoke.json](D:\RAFA\research_track\infra\summaries\prompt_conditioned_hybrid_smoke.json)

### B. Semantic Steering Lane

Current semantic steering smoke:
- `semantic_tension_steering_smoke`
- `tension_to_inharmonic_corr = -0.1452`
- `peak_alignment_error = 0.44`
- `recovery_gap = 0.0090`
- `collapse_rate_pct = 0.0`

### C. Conditioned Logic / Relational Diagnostics

Best current conditioned logic region on the 12-task focused sweep:
- `logic_diag_focus_prompt_real_quick_a1_ic0_ns8`
  - `success_rate = 0.1667`
  - `mean_persistence = 0.1833`

Interpretation:
- Conditioned logic signal exists, but it is weak and checkpoint-dependent.
- More latent activity did not imply more binding.
- This is consistent with the current stack view:
  - L1 works
  - L2 steering does not work yet
  - L3 is not built
  - logic wins remain narrow and unstable

## 3. Promotion Criteria (The Gates)

These are hard gates, not vibes.

### L1 Promotion Gate
L1 is considered working when:
- collapse remains `<= 1.0%` across the standard evaluation grid.
- no systematic numerical instability appears under ablations. (L1 is already past this gate.)

### L2 Promotion Gate
L2 cannot progress to L3 until all hold on the isolated trajectory evaluator:
- `tension_to_inharmonic_corr > 0.4`
- `peak_alignment_error < 0.2`
- `recovery_gap < 0.1`

### L3 Promotion Gate
L3 cannot be treated as working until:
- it beats the open-loop L2 baseline on the trajectory evaluator.
- it produces positive decoy rejection on constrained logic tasks.

## 4. Cognition as Acoustic Metamaterials

### Concepts as Cymatic Holograms
If the substrate is still audio phase, then a concept (like "Dog" or "Multiplication") is not a text token. It is a **Standing Wave Interference Pattern**. Think of the Cognitive Chamber as a multi-dimensional Chladni plate. When you input a prompt, you are striking the plate with specific audio frequencies. The phase waves ripple across the Ramanujan lattice, form complex geometric interference patterns (attractors). "Thought" is the specific shape of the cymatic resonance that forms.

### How it is "Used Differently" (Time vs. Depth)
* **The Acoustic Crystal (Speaking):** The phase is unrolled across the **Time Dimension**. Goal: create sequential air pressure changes.
* **The Cognitive Chamber (Thinking):** The phase is unrolled across the **Latent Depth Dimension** (COCONUT steps). It compresses a complex acoustic collision into a single moment of latent space to see if it resolves into harmony or grinds into dissonance.

### Zero-Cost Translation (The Holy Grail)
If the Cognitive Chamber uses audio phase to think... there is no translation. 
1. The Cognitive Chamber collides internal "sounds" until they form a harmonic standing wave.
2. The thought is complete.
3. To "speak," the system simply lets the internal audio phase leak into the diffusion decoder.
The thought is the voice. The reasoning is the song.

## 5. Sensors and Safety Valves (The Actual Silicon)

To move from a "constitutional" phase to an "implementable" architecture, the following components are required to link abstract layers to silicon:

### 1. Proprioceptive Feedback (The "Ear")
The current roadmap lacks an Internal Ear. Without it, the Brain (Cognitive Chamber) never knows if the Vocal Cords (Acoustic Crystal) rendered the "thought" accurately. We need a **Proprioceptive Loop**: the internal audio phase generated by the Vocal Cords must be fed back into the Cognitive Chamber. This enables self-correction and "Chain of Acoustic Thought."

### 2. Semantic DSP Labeler (Ground Truth Source)
To train the Minimal Impedance Head (L3) to follow a Tension Envelope ($C_t$), we need a **Data Factory**. We must build an offline Algorithmic Labeler that extracts the empirical inharmonicity ratio of all 5,825 training files. This ratio becomes the "teacher" for the Impedance Head, providing a ground-truth map of tension.

### 3. Singular Point Safety (The "Pressure Valve")
The IFS is a chaotic engine prone to singularities (division by zero or phase-wrap artifacts). We need a **Non-Linear Damping Layer** between L1 and L5. This valve detects when latent energy exceeds a threshold and applies logarithmic compression to the phases before they reach the Diffusion Decoder, ensuring the "voice" stays within safe physical bounds.

### 4. Active Steering Mechanism (The "Clutch" Actuator)
The Gearbox and Clutch (L4) require an **Actuator**. We need a specific MLP that monitors the **Latent Trajectory Loss**. If the loss is high (struggling to resolve tension), the Actuator automatically "downshifts" the Gearbox to a lower, slower resolution, giving the IFS more time-steps to settle. Reasoning complexity thus becomes "time-to-coherence."

## 6. The Engineering Roadmap: Why Babies Babble

### The Physics of the Trap: Gradient Starvation
Deep learning models are governed by the path of least physical resistance. If you apply a Diffusion MSE at the end of the pipe, the gradients flow backward. The Acoustic Crystal (Vocal Cords) realizes it can minimize loss simply by matching the target frequency, zeroing out the connection weights to the Cognitive Chamber (Brain). The result is a flawless synthesizer that is completely brain-dead.

### The Biological Proof
Nature uses **Curriculum Training**:
* **Phase 1: Babbling (Acoustic Baseline).** Calibrating the physical constraints of the vocal tract. (L1 DONE).
* **Phase 2: Silent Play (Cognitive Sandbox).** Training the brain completely isolated from audio. Feeding abstract logic puzzles (math probes, 1D Tension Envelopes) applying a **Latent Trajectory Loss**. (ACTIVE).
* **Phase 3: Speech (Joint Training).** Teaching the Brain how to map its abstract geometric standing waves onto the Acoustic Crystal's frequency lattice.

## 7. The Immediate Action Plan (Lane 2 Ruthlessness)

Lane 2 is now the frontier. The job is: `make the chamber obey the structural control basis measurably`.

### Immediate ablation: Minimal Impedance Head
Build the smallest possible L3 candidate (no CLAP, no multimodal semantics). Controlling only harmonic q-band gain, route temperature, and impedance bias.

### Required training loss: Latent Alignment Loss
Do not rely on end-of-pipe diffusion loss. Penalize mismatch between commanded `C_t` and observed inharmonicity ratio over time.

### Required temporal rule: Transition Dynamics / Clutch Behavior
Formally define how state transitions behave (inertia vs reset) between semantic states.

## Operating Rules

1. Lane 1 and Lane 2 remain separate until L2 passes its gate.
2. No joint fine-tuning until gradient interference is understood.
3. No promotion from L2 to L3 without the trajectory evaluator.
4. No promotion from L3 to semantics without latent alignment loss.
5. No truth claims beyond internal stability and constraint satisfaction.
6. **The Invariance Constraint:** No semantic concept shall be promoted to the Cognitive Chamber unless it demonstrates Invariance Under Flow—meaning the standing wave pattern must remain stable across at least 4 COCONUT latent steps without external anchoring.

## Current Bottom Line

As of now:
- L1 works ( v3.0 stable substrate)
- L5 works (audio renderer active)
- L2 is instrumented and failing
- L3 is specified but unbuilt
- L4 is conceptual and partially scaffolded

The next correct move is a minimal L3 impedance head trained against the L2 trajectory evaluator.
