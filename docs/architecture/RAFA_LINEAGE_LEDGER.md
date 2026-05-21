# RAFA Lineage Ledger

This ledger formalizes the RAFA model evolution, categorizing every run into its proper philosophical and engineering branch.

## 1. Parent Artifact (The Anchor)
*The maximal proven hybrid. The reference artifact for all current logic.*

| Name | Family | Philosophical Purpose | Engineering Purpose | Question it tests | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Graduation Pack** | Parent | Maximal hybrid integration. | Establish strongest working baseline for audio synthesis. | Can a complex hybrid of Memory, Hyena, and Ramanujan gating produce stable, high-fidelity audio? | **Proven Working** |

### Checkpoint Analysis: Graduation Pack
- **Lineage Tag:** `rafa_full_v3`
- **Training Regime:** Path B (Diffusion on 5.8k audio corpus).
- **Evaluation Regime:** Path B Evaluation Grid (24 samples).
- **Key Metrics:** `CentVar` ~76k, `Collapse Rate` 0.0%.
- **Perceptual Verdict:** Stable, structured, clean synthesis.
- **Notes:** Reference files in `outputs/graduation_pack`. Anchor checkpoint: `bound_weights_ep49_step100.pt`. Refined restore baseline: `bound_weights_ep10_step1200.pt` with `qkv` slice index 4 (start=516) and `phase_mix=0.395`, yielding `mean_mse ~= 0.00958`. Found that the historical denoiser utilized a 3-channel guide, but current 2-channel path is more stable for reconstruction.

---

## 2. Local Ablations (The Knockouts)
*Studies within the Graduation Pack lineage removing one component at a time.*

| Name | Family | Parent | Engineering Purpose | Question it tests | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **no_ramanujan** | Local Ablation | Graduation Pack | Remove prime-lattice summary. | Is Ramanujan gating essential for frequency coordination? | **Inconclusive** |
| **no_memory** | Local Ablation | Graduation Pack | Remove Motif Memory. | Does long-term relational memory prevent mode collapse? | **Inconclusive** |
| **no_slow_clock** | Local Ablation | Graduation Pack | Remove multi-rate clocking. | Is temporal hierarchy necessary for structural stability? | **Inconclusive** |

### Checkpoint Analysis: Component Ablations
- **Training Regime:** 1000-step Path B continuation.
- **Evaluation Regime:** Path B Grid.
- **Key Metrics:** Minor increases in `CentVar` and `TopRatio` across all three.
- **Perceptual Verdict:** Models remain functional but show decreased structural coherence.
- **Notes:** Proves that while no single module is a "kill-switch," they all contribute to the stability attractor.

---

## 3. Subtractive Purity Fork (The Lobotomy)
*Attempting to purify the hybrid by aggressive stripping toward an austere phase-native form.*

| Name | Family | Parent | Engineering Purpose | Question it tests | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Mutilated RAFA** | Subtractive Fork | Graduation Pack | Strip hybrid toward minimal phase form. | Can we reach a pure phase-native state by removal? | **FAILED** |

### Checkpoint Analysis: Mutilated Graduation Pack
- **Lineage Tag:** `stage4_joint_final` (Lobotomised)
- **Training Regime:** 50-epoch "Graduation Marathon" with over-constrained Sm_120 registers.
- **Evaluation Regime:** Path B Grid.
- **Key Metrics:** `CentVar` 408,993 (**CRITICAL EXPLOSION**).
- **Perceptual Verdict:** "Washing-machine garbage" / High-frequency oscillatory noise.
- **Notes:** Over-optimization of the joint matching task created a high-gain feedback loop. The system collapsed into noise because it lacked the positive architectural structures to handle the pure phase pressure.

---

## 4. Positive Replacement Architecture (The Organism)
*Building the true alternative system directly, instead of carving pieces out of the hybrid.*

| Name | Family | Philosophical Purpose | Engineering Purpose | Question it tests | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Circleworld / Hardy-Littlewood** | Positive Replacement | Auto-formalize attractors via geometric laws. | Replace statistical snapshots with recursive structural mass laws. | Can a pure Hardy-Littlewood mass distribution replace the hybrid entirely? | **Untested** |
| **Resonant Attention / Post-Token Memory Lane** | Positive Replacement research lane | Test whether law objects can become addressable by resonance rather than token position. | Build retrieval/composition/operator assays over packets, childworlds, grandchildren, and dense signatures. | Can partial phase/q/arc/support queries select reusable law objects whose values causally update state? | **Scaffolded / Not Validated** |

### Checkpoint Analysis: Formalization Suite
- **Lineage Tag:** `h2_formal_v1`
- **Training Regime:** Target-based "Ontology Survival" (Attack/Decay laws).
- **Evaluation Regime:** "Hardy-Littlewood Sieve" (Structural Mass Measurement).
- **Key Metrics:** Major-Arc Mass vs. Minor-Arc Residue.
- **Perceptual Verdict:** TBD.
- **Notes:** Initial implementation in `tools/ablate_formalization.py`. Incorporates principles from `SemanticInferometer` (Circle-Method / Fractal IFS). Treats concepts (e.g., "Piano") as repeatable recursive geometries in phase space.

### Branch-Hierarchy Note: Resonant Attention / Post-Token Memory

- This is an active research lane under the Positive Replacement / Circleworld
  family, not a rename of the parent Graduation Pack and not a new proven
  lineage.
- Its current implementation evidence is limited to Circleworld's local
  branch-QKV mechanism: `branch_law_version = "relational_qkv_v2"` and
  `_relational_branch_attention(...)`.
- `relational_qkv_v2` remains only a local branch-QKV precursor. It must not be
  promoted to full RAFA attention unless retrieval and composition assays over
  reusable packet/child/grandchild/signature law objects pass.
- Dense relational signatures and childworld records are candidate law-object
  sources for this lane, but neither is sufficient by existence alone to claim a
  RAFA post-token.
