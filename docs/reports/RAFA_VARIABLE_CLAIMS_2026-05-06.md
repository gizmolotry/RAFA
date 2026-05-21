# RAFA Variable Claims - 2026-05-06

This file isolates RAFA components as falsifiable claims. The goal is to prevent one aggregate score from hiding whether a specific RAFA idea is actually working.

Higher-level paradigm claims are split in:

`D:\RAFA\docs\reports\RAFA_CLAIM_TAXONOMY_2026-05-06.md`

## Claim Matrix

| Claim | Variable | Falsifiable hypothesis | Intervention or ablation | Metric | Failure mode |
|---|---|---|---|---|---|
| Semi-tokenless audio model | `no_future_magnitude_continuation`, prefix state, flat continuation baseline, `copy_last_waveform_phase`, `audio_circle_delta_probe`, `audio_delta_mechanism_probe`, `audio_delta_mechanism_lockbox`, `audio_mechanism_family_sensitivity`, `audio_electric_motor_holdout`, `audio_steady_phenotype_manifest`, `audio_phase_phenotype_diagnostics`, `audio_delta_objective_scout`, exported audio sidecars | Circleworld variables can carry audio continuation structure without a token stream or future magnitude leakage. | Compare no-future-magnitude Circleworld continuation to flat, prefix, copy-last, copyphase seed, delta-off carriers, fixed prefix-only mechanism shaping, predeclared lockbox cases, sound-family slices, electric-motor/razor phenotype holdouts, disjoint steady phenotype manifests, baseline-correlation diagnostics, and fixed objective scout candidates on heldout audio probes. | Continuation corr, perceptual continuity, q/arc preservation, benchmark drift versus flat/prefix/copy-last/copyphase/delta-off, fixed mechanism delta, lockbox win/median robustness, family mean/median/win/outlier share, no-razor structural motor robustness, razor diagnostic outlier share, feature correlation with baseline quality versus prefix/phase variables, robust scout objective score. | Circleworld only matches carriers, loses to copy-last/copyphase on corr, improves only by MSE/correlation tradeoff or bad-baseline rescue, outlier/family/phenotype-only wins, requires future magnitude/phase leakage, or fails fixed mechanism/objective scouting. |
| Internal Circleworld phase law | `phase_law_precondition_gain`, `phase_law_velocity_mix`, `phase_law_stability_gain`, `phase_law_softclip`, `phase_law_low_rank`, `phase_law_consensus_mix`, `phase_law_consensus_damping`, `phase_law_median_guard`, `phase_law_local_velocity_mix`, `phase_law_local_coherence_damping`, `phase_law_curvature_guard`, `phase_law_reentry_mix`, `phase_law_reentry_accel_mix`, energy support masks, phase-derived support masks, `_precondition_phase_delta`, `run_internal_phase_law_objective_scout.py` | Internal phase-law coefficients, support geometry, and temporal reentry can causally change recurrence beyond export-time post-shaping. | Compare matched lockbox rows from default-off no-op config against internal-law variants, guarded consensus/median variants, local phase-carrier variants, v4 joint/nonbad variants, v5 low-energy target-row variants, v6 low-energy win variants, v7 low-energy gain-ladder variants, v8 energy support-mask variants, v9 phase-only mask variants, v10/v11 phase-router variants, and v12 temporal reentry variants; separately compare each variant to gain-0/copyphase carrier through a one-command locked-objective scout, same-row joint diagnostic, failure atlas, target sign-flip/family diagnostics, family/mask paired diagnostics, and phase-mask joint candidates. | Direct variant-vs-no-op corr/MSE delta, absolute lockbox corr/MSE delta, baseline-correlation bins, nonbad/good bin deltas, win fraction, median robustness, locked objective decision, joint-row candidate count, gain sign-flip category counts, support-mask win/median shifts by family, phase-mask same-row candidates, reentry-vs-router delta. | Variant improves relative to no-op but still loses to gain-0/copyphase carrier, gains occur only in bad-baseline bins, good/nonbad bins regress, support-mask wins are family-conditional without a learned routing rule, explicit target rows pass but global run selection still fails, or consensus/median/local/reentry pressure increases direct signal without creating robust joint rows. |
| Phase-only gauge invariance | `phase_only_real_branch_fraction`, `phase_only_excess_branch_fraction`, `mean_phase_only_branch_distinctness`, `decorative_slot2_low_phase_fraction` | True branch multiplicity can be produced from phase relations alone and is invariant to global phasor gauge shifts; amplitude claims require separate controls. | Run global phase rotation, normalized phase-preserving rescale/taper regression checks, and phase-only branch kernel; compare with Ramanujan kernel. | Naked and heldout phase-only branch above zero, stable under global gauge transform, low decorative slot-2 fraction. | Phase branch stays zero on `naked_rafa`, changes under global gauge transform, or only raises decorative slot-2 activity. |
| Lattice/substrate runtime contract | `phasor_normalize`, `phasor_apply_delta`, per-step state norms, q/arc/branch deltas under gauge transforms | Circleworld can alter law/phase while preserving unit-modulus phasors and avoiding hidden magnitude semantics. | Add assertions or audits after mode update, child spawn, child evolve, writeback, readout; rotate global phase and rescale/taper magnitudes. | Max phasor norm error near zero, q/arc/branch metric drift near numerical noise, no magnitude channel introduced into branch law. | Branch gains require magnitude writes, phasor norms drift, or q/arc/branch metrics move materially under gauge-preserving transforms. |
| Unit-phasor packet contract | promoted packet `seed_state`, `test_unit_phasor_contract.py`, checked runtime phasor tensors, raw multimode mixture norm | Promoted packet seeds must remain phase objects, not coherence-magnitude carriers, because downstream branch alignment assumes unit phasors. Weighted multimode readout must not rely on near-zero raw mixture cancellation before normalization. | Audit input/final/multimode/child/packet phasor tensors before and after robust packet seed fallback/normalization; audit heldout, seeded nested, export-like readout, direct WAV/STFT export, audio-continuation tensors, raw weighted mixture norms, and deliberate magnitude negative controls. | Pre-fix failure detected; post-fix all-surfaces max norm error `1.788e-7`, zero violations, finite fraction `1.0` across 3352 tensors, min raw mixture norm `0.997107` across 304 states. Supplemental 18-WAV direct export and audio-continuation checks pass 739 and 737 tensors with max error `1.192e-7`. Scale-0.83 negative control fails at `0.170000076` and is detected. | Packet seeds or child/harness states carry subunit magnitude, raw multimode mixture cancels near zero, or the audit fails to detect deliberate magnitude corruption. |
| Ramanujan/qtrace lane | `qset`, `q_weights`, `branch_kernel_version`, `q_trace`, `qtrace_momentum`, `child_survival_qtrace_weight`, `mean_law_top_q_entropy`, `num_law_top_q_unique` | Ramanujan q structure and qtrace survival must be separable; either can help or fail independently. | Compare `ramanujan`, `qtrace_only`, uniform q weights, shuffled qset, current-only qtrace, frozen/no-write qtrace, and child-qtrace-survival-off. | Benchmark corr, law families, top-q entropy, branch fraction by source, naked phase-only branch, child writeback decomposition. | Ramanujan performs no better than qtrace-only, or qtrace survival helps only by decorative branch pressure. |
| Hardy-Littlewood arc lane | `major_mass`, `minor_residue`, `harmonic_ratio`, `promotability_field` | Arc fields are causal for coherent packet promotion and child spawn/writeback, not retrospective diagnostics. | Flatten or randomize arc outputs while preserving phase state; disable promotability. | Major gain, residue drop, promotability gain, law packet count, child spawn/writeback, benchmark fidelity. | Metrics barely move under arc ablation or arc pressure improves packet scores without affecting branch/writeback. |
| Child support writeback | `child_support_writeback_mass`, `child_support_writeback_gain`, `child_support_writeback_floor_gain`, `child_writeback_mass`, `child_writeback_budget`, `child_parent_mix`, `child_parent_mix_early` | Child worlds become causal only when support-local writeback is nonzero and localized enough not to erase identity. | Sweep support writeback gain/floor, writeback gain, budget, parent mix; ablate support writeback to zero; force global writeback as negative control. | Support writeback mass, parent divergence, benchmark corr, nested sibling fraction, decorative slot-2 fraction. | Child count is nonzero with zero writeback, or writeback collapses to global overwrite. |
| Child operator phase writeback | `child_operator_seed_gain`, `child_operator_promotability_gain`, `child_writeback_operator_mix` | Operator-seeded phase return is necessary for parent-visible phase branching after child survival. | Disable operator delta; sweep operator seed and operator mix while holding support writeback fixed. | Naked parent branch, naked phase-only branch, phase-only distinctness, sibling divergence. | Writeback mass remains nonzero but parent/phase branch stays zero. |
| Child phase-floor return | `child_writeback_phase_floor_target`, `child_writeback_phase_floor_threshold_mult`, `child_writeback_phase_floor_gain`, `child_writeback_parent_mix_cap` | A limited phase-floor can push child identity across the parent-visible phase threshold without becoming an audio-destroying overwrite. | Sweep floor target/gain and parent-mix cap; compare floor off vs support-only. | Naked phase-only branch, mean major gain, benchmark corr, decorative slot-2 fraction, world-jump penalty. | Phase branch rises only with large audio regression or decorative slot-2 pressure. |
| Child identity carry | `same_child_carry_fraction`, `same_child_carry_steps`, `same_child_budget_retained` | Same-child persistence is necessary but insufficient for branch ontology. | Track exact child ID through continuation; replace-child events must not count. | Raw carry fraction, raw carry steps, retained budget. | Carry saturates at 1.0 while no active child law survives. |
| Qualified identity carry | `same_child_qualified_carry_fraction`, support/coherence/budget thresholds | Qualified carry distinguishes real branch identity from trivial child-record persistence. | Tighten support, coherence, and budget thresholds; compare old carry to qualified carry. | Gap between raw carry and qualified carry, nested sibling fraction under strict thresholds. | Raw carry succeeds but qualified carry collapses, proving previous success was record persistence only. |
| Readout sibling response | `mean_readout_sibling_response`, `coarse_env_corr`, `fine_q_profile_corr` | Readout difference is meaningful only if paired with branch survival and identity carry. | Compare readout-only, continuation-only, and combined child-led branches. | Readout response, coarse preservation, q preservation, qualified carry. | Readout response rises while child survival and qualified carry remain zero. |
| Branch survival | `mean_child_survival_signal`, `mean_live_child_fraction`, `mean_child_meso_response` | Branch ontology requires live child law during continuation, not just initial fork evidence. | Start nested probes from live-child states and measure active children after child-led continuation. | Child active fraction, survival signal, meso response, writeback count. | Live child exists at fork but disappears during the continuation interval. |
| Branch probe alias hygiene | `apply_passive_packets`, `apply_active_packets`, `perturb_circleworld_state`, `_child_only_pre_unroll`, `test_packet_alias_contract.py` | Branch probes must not mutate shared baseline tensors while constructing sibling perturbations. | Clone before packet slice writes; clone no-packet passthroughs; run alias audit for passive, active, empty packet, nested branch clone, child perturbation, and child pre-unroll paths. | Input max delta `0.0`, sibling max delta `0.0` where applicable, output delta nonzero for perturbing branches, no shared storage, output phasor norm preserved. | Packet perturbation, child perturbation, or pre-unroll changes the baseline branch or shares storage with input. |
| Parent branch truth | `parent_real_branch_fraction`, `child_real_branch_fraction`, `real_branch_fraction` | Headline branch fraction must not hide parent mode failure behind child readiness. | Report parent and child branch separately; ablate child registry while preserving parent multimode state. | Parent branch fraction, child branch fraction, phase-only branch fraction by source. | Overall branch fraction is high while parent branch stays near zero. |
| Dense relational signature | `RafaRelationalSignatureV0.h`, prefix schedule, signature families, confidence | Dense RAFA signatures expose a token-like objective not reducible to explicit metadata or law-packet labels. | Compare dense-body signature training to old explicit-field-only/law-token summaries. | Signature family count, confidence, prefix cosine, metamer family drift, dominant family share. | Signatures mirror law-token collapse or improve stability by destroying diversity. |
| Resonant Attention / Post-Token Memory | `relational_qkv_v2`, `_relational_branch_attention`, query/key/value law fields, geometric score components, learned residual, selected operator value, retrieval margin, composition margin | Phase-native law objects can become addressable by relational resonance and selected values can causally update state; `relational_qkv_v2` itself remains only a local branch-QKV precursor. | Build packet/child/grandchild/signature banks; run partial-query retrieval, cross-depth composition, interference selectivity, geometry-disabled/residual-disabled controls, stored-ID controls, dense-only controls, and inert-value ablations. | Top-k retrieval, resonance margin, decoy suppression, composition gain, operator-causality delta, unit-phasor/gauge pass, geometry/residual contribution split. | Dense dot-product routing, stored IDs, or scalar branch scores retrieve objects without geometric compatibility; selected values are inert; composition causes world jump; `relational_qkv_v2` is mislabeled as full RAFA attention. |
| Matryoshka signature prefixes | `h[:128]`, `h[:256]`, `h[:384]`, `h[:512]`, `h[:640]`, `h[:768]` | Prefixes encode increasing operational commitments: world, q/arc, lifecycle, support, branch, operator. | Train prefix prediction losses independently and ablate tail segments. | Prefix-specific prediction accuracy, metamer prefix stability, continuation improvement. | Only full signatures work, or early prefixes fail to preserve coarse world identity. |
| Semantic projector | `harmonic_coupling`, `decay_rate`, `noise_injection`, `branch_temperature`, `support_spread` | Five semantic controls steer boundary conditions without inventing a second ontology. | One-control-at-a-time sweeps; random projector negative control; fixed prompt families. | Monotonic response of harmonicity, decay, residue, branch fraction, support spread. | Controls are entangled, non-identifiable, or random projector performs as well. |
| Anti-loop/reentry | `mean_loop_autocorr_peak`, `mean_nonlocal_chunk_repeat`, `mean_adjacent_chunk_similarity`, `mean_first_chunk_reentry` | Branch-local law changes reduce macro-time reentry without sacrificing child identity. | Compare anti-fixation/reentry penalties, branch-local isolation, and no-branch baseline. | Lower loop autocorr/reentry with stable benchmark corr and stable qualified carry. | Loop improves only by degrading audio, or branch metrics improve while reentry is unchanged. |
| Runtime acceptance | Cross-metric contract | A runtime patch succeeds only if it moves the causal path, not only objective weights. | Re-run heldout, naked, nested, benchmark, continuity, and export sidecars on fixed seeds. | Naked parent or phase branch above zero, nonzero support writeback, nonzero qualified carry, no benchmark collapse. | Metrics improve only on synthetic rows, nested active response remains zero, or benchmark collapses. |

## Current-Cycle Status Split

Evidence source for the child-writeback split is the v4 seed-fallback harness at `D:\RAFA\runtimes\circleworld_proto\run_child_writeback_ablation.py`, which writes `child_writeback_ablation_summary.json` and `CHILD_WRITEBACK_ABLATION.md` under `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\child_writeback_ablation_v4_seed_fallback_cuda_2026_05_06\`. The unified paired comparison is `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\claim_isolation_cuda_mixed_2026_05_06\claim_isolation_suite_summary.json`. The unit-phasor audit is `D:\RAFA\runtimes\circleworld_proto\test_unit_phasor_contract.py`, with pre-fix failure under `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\unit_phasor_contract_prefix_fail_cuda_2026_05_06\unit_phasor_contract_report.json` and all-surfaces post-fix pass plus negative-control detection under `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\unit_phasor_contract_all_surfaces_cuda_2026_05_06\unit_phasor_contract_report.json`. Audio internal phase-law evidence is now rerunnable through `D:\RAFA\runtimes\circleworld_proto\run_internal_phase_law_objective_scout.py`; the full automated lockbox artifact is `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_objective_scout_auto_full_2026_05_07\TRACK_REPORT.md`, the focused coefficient search is `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_objective_scout_focused_v1_2026_05_07\TRACK_REPORT.md`, the wide coefficient search is `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_objective_scout_wide_v1_2026_05_07\TRACK_REPORT.md`, the v2 guarded consensus/median search is `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_objective_scout_v2_guard_v1_2026_05_07\TRACK_REPORT.md` with joint-row and atlas evidence under `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_joint_rows_v2_guard_v1_2026_05_07\` and `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_failure_atlas_v2_guard_v1_2026_05_07\`, the v3 local carrier search is `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_objective_scout_v3_local_v1_2026_05_07\TRACK_REPORT.md` with joint-row and atlas evidence under `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_joint_rows_v3_local_v1_2026_05_07\` and `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_failure_atlas_v3_local_v1_2026_05_07\`, and the completed v4 joint/nonbad scout root is `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_objective_scout_v4_joint_nonbad_v1_2026_05_07\` with objective score and joint-row diagnostics beneath that root plus failure-atlas evidence under `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_failure_atlas_v4_joint_nonbad_v1_2026_05_07\`. These specs are generated from `D:\RAFA\runtimes\circleworld_proto\build_internal_phase_law_variant_spec.py`. The report is `D:\RAFA\docs\reports\CIRCLEWORLD_INTERNAL_PHASE_LAW_SCOUT_2026-05-07.md`. Nested/readout evidence, five-anchor direct WAV/STFT export evidence, and five-anchor audio-continuation tensor evidence are now included in that all-surfaces audit, with supplemental 18-WAV direct/audio tensor-contract passes under `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\unit_phasor_broad_direct_export_cuda_2026_05_06\` and `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\unit_phasor_broad_audio_continuation_cuda_2026_05_06\`. Broad 18-WAV render benchmark quality evidence is under `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\benchmark_broad18_circleworld_4s_cuda_2026_05_06\benchmark_summary.json`. Broad no-future audio method comparison is under `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\audio_continuation_method_compare_broad18_cuda_2026_05_06\audio_continuation_method_compare.json`; copyphase benchmark evidence is under `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\audio_continuation_broad18_copyphase_prefix_hold_cuda_2026_05_06\audio_continuation_summary.json`, `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\audio_continuation_broad18_copyphase_flat_cuda_2026_05_06\audio_continuation_summary.json`, and `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\audio_continuation_method_compare_copyphase_broad18_cuda_2026_05_06\audio_continuation_method_compare.json`; broad phase-influence ablation is under `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\audio_phase_influence_broad18_cuda_2026_05_06\audio_phase_influence_ablation.json`; broad phase-seed ablation is under `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\audio_phase_seed_broad18_cuda_2026_05_06\audio_phase_seed_ablation.json`; weighted/fit phase-seed expansion is under `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\audio_phase_seed_weighted_fit_broad18_cuda_2026_05_06\audio_phase_seed_ablation.json`; delta-over-copyphase evidence is under `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\audio_circle_delta_probe_broad18_cuda_2026_05_06\audio_circle_delta_probe.json`; fixed mechanism delta evidence is under `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\audio_delta_mechanism_probe_broad18_cuda_2026_05_06\audio_delta_mechanism_probe.json`; predeclared fixed mechanism lockbox evidence is under `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\audio_delta_mechanism_probe_lockbox18_cuda_2026_05_06\audio_delta_mechanism_probe.json`; sound-family sensitivity evidence is under `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\audio_mechanism_family_sensitivity_broad_cuda_2026_05_06\audio_mechanism_family_sensitivity.json`; electric-motor/razor phenotype holdout evidence is under `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\audio_electric_motor_holdout_cuda_2026_05_06\audio_electric_motor_holdout.json`; disjoint steady phenotype evidence is under `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\audio_steady_phenotype_manifest_cuda_2026_05_06\audio_steady_phenotype_manifest.json`; phase phenotype diagnostics are under `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\audio_phase_phenotype_diagnostics_cuda_2026_05_06\audio_phase_phenotype_diagnostics.json`; objective scout evidence is under `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\audio_delta_objective_scout_broad18_cuda_2026_05_06\audio_delta_objective_scout.json` and `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\audio_delta_objective_scout_relsig_broad18_cuda_2026_05_06\audio_delta_objective_scout.json`. Dense-vs-explicit-metadata evidence is under `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\dense_signature_claim_broad18_cuda_2026_05_06\dense_signature_claim.json`; dense failure-mode evidence is under `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\dense_signature_failure_modes_2026_05_07\dense_signature_failure_modes.json`; relational-signature contract audit evidence is under `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\relational_signature_contract_audit_2026_05_07\relational_signature_contract_audit.json`; relational-signature learning contract evidence is under `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\relational_signature_learning_contract_2026_05_07\relational_signature_learning_contract.json`; learned-signature scout evidence is under `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\learned_signature_scout_cuda_2026_05_07\learned_signature_scout.json`; learned-signature scout comparison evidence is under `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\learned_signature_scout_compare_2026_05_07\learned_signature_scout_compare.json`; semantic-projector contract evidence is under `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\semantic_projector_contract_2026_05_06\semantic_projector_contract.json`. Branch probe alias hygiene is covered by `D:\RAFA\runtimes\circleworld_proto\test_packet_alias_contract.py` and `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\packet_alias_contract_v2_cuda_2026_05_06\packet_alias_contract_report.json`.

The completed v5 low-energy target-row scout root is `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_objective_scout_v5_low_energy_row_v1_2026_05_07\`, with objective score and joint-row diagnostics beneath that root plus failure-atlas evidence under `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_failure_atlas_v5_low_energy_row_v1_2026_05_07\`.

The completed v6 low-energy win scout root is `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_objective_scout_v6_low_energy_win_v1_2026_05_07\`, with objective score and joint-row diagnostics beneath that root plus failure-atlas evidence under `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_failure_atlas_v6_low_energy_win_v1_2026_05_07\`.

The completed v7 low-energy gain-ladder scout root is `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_objective_scout_v7_low_energy_gain_ladder_v1_2026_05_07\`, with objective score and joint-row diagnostics beneath that root, failure-atlas evidence under `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_failure_atlas_v7_low_energy_gain_ladder_v1_2026_05_07\`, and target sign-flip diagnostics under `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_target_flip_diagnostics_v7_low_energy_gain_ladder_v1_2026_05_07\`.

The completed v8 family/support-mask scout root is `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_objective_scout_v8_family_support_mask_v1_2026_05_07\`, with objective score and joint-row diagnostics beneath that root, failure-atlas evidence under `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_failure_atlas_v8_family_support_mask_v1_2026_05_07\`, and paired family/support-mask diagnostics under `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_family_mask_diagnostics_v8_family_support_mask_v1_2026_05_07\`.

The v8 support-router oracle is `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_support_router_oracle_v8_family_support_mask_v1_2026_05_07\internal_phase_law_support_router_oracle.json`.

The completed v9 phase-only mask scout root is `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_objective_scout_v9_phase_masks_v1_2026_05_07\`, with explicit phase-stable target reruns at `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_objective_score_v9_phase_stable_target_v1_2026_05_07\internal_phase_law_objective_score.json` and `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_joint_rows_v9_phase_stable_target_v1_2026_05_07\internal_phase_law_joint_row_diagnostics.json`, plus failure-atlas evidence under `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_failure_atlas_v9_phase_masks_v1_2026_05_07\`.

The completed v10 phase-router mask scout root is `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_objective_scout_v10_phase_router_masks_v1_2026_05_07\`, with explicit stable/coherent target reruns under `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_joint_rows_v10_phase_stable_target_v1_2026_05_07\` and `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_joint_rows_v10_phase_coherent_target_v1_2026_05_07\`. Candidate-class diagnostics are `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_candidate_classes_v9_phase_masks_v1_2026_05_07\internal_phase_law_candidate_classes.json` and `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_candidate_classes_v10_phase_router_v1_2026_05_07\internal_phase_law_candidate_classes.json`. The completed v11 phase-router direct scout root is `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_objective_scout_v11_phase_router_direct_v1_2026_05_07\`, with candidate-class diagnostics under `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_candidate_classes_v11_phase_router_direct_v1_2026_05_07\`. The completed v12 phase-reentry direct scout root is `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_objective_scout_v12_phase_reentry_direct_v1_2026_05_07\`, with candidate-class diagnostics under `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_candidate_classes_v12_phase_reentry_direct_v1_2026_05_07\`. The completed v13 causal phase-reentry direct scout root is `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_objective_scout_v13_causal_reentry_direct_v1_2026_05_07\`, with candidate-class diagnostics under `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_candidate_classes_v13_causal_reentry_direct_v1_2026_05_07\`. The reentry contract audit is `D:\RAFA\runtimes\circleworld_proto\test_internal_phase_law_reentry_contract.py`, with pass artifact at `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_reentry_contract_2026_05_07\internal_phase_law_reentry_contract.json`. The predeclared diagnostic class report is `D:\RAFA\runtimes\circleworld_proto\predeclare_internal_phase_law_diagnostic_class.py`, with artifact at `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_predeclared_diagnostic_class_v10_v13_2026_05_07\internal_phase_law_predeclared_diagnostic_class.json`.

| Independent hypothesis | Current status | Evidence split | Promotion bar |
|---|---|---|---|
| Semi-tokenless audio model | Testable, not proven. | No-future-magnitude continuation is now a valid broad 18-case test condition, and the new copyphase runs also report no future target phase reuse. Circleworld does not beat simple carriers: original velocity-seed prefix-hold Circleworld corr `-0.0110` / MAE `0.11218`, flat Circleworld corr `-0.00170` / MAE `0.09756`, while copy-last corr is `0.09023`. Original method comparison is `carrier_locked`. With the legal `copy_last_waveform_phase` seed, prefix-hold Circleworld corr rises to `0.088204`, but the copy-last waveform baseline remains higher at `0.090227`. Phase-seed ablation and weighted/fit expansion are `seed_policy_matters_circleworld_not_helpful`: Circleworld-minus-seed corr is only `0.001059`. Delta-over-copyphase is `delta_tradeoff_not_proof`: best corr delta is `+0.002819` and best MSE delta is `-0.000115`. Fixed prefix-only mechanism shaping improves the diagnostic signal but not enough for proof: `time_smooth_3` on `prefix_hold/all_bins` at gain `2.0` gives corr delta `+0.005640`, MSE delta `-0.000160`, and corr win fraction `0.666667`, so the rollup is `mechanism_tradeoff_not_proof` below the strict `+0.02` candidate bar. The predeclared lockbox repeats only that mechanism on 18 unused PCM cases and remains `mechanism_tradeoff_not_proof`: mean corr delta `+0.056605`, median corr delta `+0.000411`, corr win fraction `0.611111`, with the mean dominated by an `electric_razor` outlier. Sound-family sensitivity localizes the strongest effect to `electric_motor_hum`, but it is still `family_tradeoff_signal_only`: mean corr delta `+0.105432`, median `+0.000110`, min leave-one-out `+0.030514`, corr win fraction `0.500000`, and positive outlier share `0.720103`. The stricter provider-neutral electric-motor holdout narrows the signal rather than promoting it: no-razor structural motor is only tradeoff-level (`nonrazor_appliance_holdout` mean corr delta `+0.003838`, median `+0.000221`, outlier share `0.882072`), while the mixed razor diagnostic is dominated by one canonical `Electric Razor.wav` case (`+0.929525` case delta; mixed outlier share `0.998450`). The disjoint steady-phenotype manifest removes overlap: `60` selected cases are `60` unique paths/content hashes, but status is still `disjoint_phenotype_tradeoff_only`; steady-buzz mean corr delta is `+0.026831` with negative median and outlier share `0.670547`, and best no-razor motor mean corr delta is `+0.004305`. Post-hoc phenotype diagnostics show the strongest non-single-source predictor is bad gain-0 baseline alignment (`gain0_target_corr` Pearson `-0.686859`), not a clean phase law feature; prefix phase-velocity coherence is weak (`+0.265985`) and mean shaped-delta magnitude is negative (`-0.223002`). Fixed objective scouting does not rescue this: floor-high scout is `tradeoff_signal_only`, with best candidate `branch_aggressive_delta_high` corr delta `+0.002833`, corr win fraction `0.555556`; relsig scout is `no_candidate_found`, with best corr delta `+0.002037` and positive MSE delta. Broad 18-WAV 4s direct render shows preservation, not autonomy: mean corr `0.9630`, continuity delta band `near_identity`, recurrence delta `0.000406`. | Beat copy-last waveform phase seed, copy-last waveform baseline, delta-off copyphase carrier, and a baseline-correlation-stratified disjoint no-razor steady-motor/buzz manifest without future magnitude leakage, carrier lock, MSE-only tradeoff, low-yield phase gain, bad-baseline rescue, outlier/family/phenotype-only wins, simple config-knob/mechanism overfitting, or q/arc/branch collapse. |
| Internal Circleworld phase law | Causal recurrence and support-geometry variable, not audio promotion. | v7 gain-ladder targeting shows the target row `flat/low_energy_bins/raw/gain2` can keep positive direct mean and median, but direct win breadth stays pinned at `0.5081967213114754`. v8 shows support masks are a separable lever: for the objective-selected run, `all_bins` and `high_energy_bins` at gain `2.0` reach win fraction `0.6229508196721312` while `low_energy_bins` stays `0.5081967213114754`; the best direct row is `high_energy_bins` gain `1.75` with mean/median/wins `0.0008355768415039223` / `0.00008620824849370776` / `0.6557377049180327`. v9 phase-only support masks produce same-row joint candidates: explicit `flat/phase_stable_bins/raw/gain2` target rerun has `2/5` target candidates, best direct/absolute/nonbad/good deltas `0.00031847714441993473` / `0.002700202968841946` / `0.0008055606836901082` / `0.0`, and best target decision `candidate_joint_row`; the stricter candidate-class diagnostic still finds `0` strict audio and `0` phase-support candidates. v10 phase-router masks improve target-row quality: `phase_router_bins/gain2` has target joint candidates `4/5` with best direct/absolute/nonbad/good deltas `0.0008050548815987055` / `0.0034083202721412876` / `0.0011328893661807422` / `0.0`. The diagnostic class split finds `3` phase-support candidates and `0` strict audio candidates; the top class row is `v8_mask_local004_raw003_damp002_curv002_s100` / `phase_router_bins` / gain `2.0`. v11 narrows to the router substrate and yields `36/36` joint candidates, `12/12` target candidates, and `19` phase-support candidates, but still `0` strict audio candidates and `no_locked_candidate` / `hold_direct_too_small`. v12 adds phase-only temporal reentry and slightly raises target direct/nonbad deltas to `0.0008534610651987389` / `0.0014933570819019395`, but yields `35/36` joint candidates, `16` phase-support candidates, and still `0` strict audio candidates. v13 makes reentry causal and keeps the best target row slightly higher at direct/nonbad `0.00085477891809921` / `0.0014958846296200376`, but breadth drops to `24/30` joint candidates and `12` phase-support candidates. | Promotion requires row-level gains to survive global run selection and baseline-stratified carrier checks; v11-v13 support phase-derived support routing and causal temporal reentry as real variables, but falsify simple stronger-coefficient or simple reentry pressure as sufficient for audio continuation. |
| Phase-only/gauge | Global phase-rotation invariance supported; phase-only branch ontology not supported. | Global rotations move q/arc/branch metrics only around `1e-7` to `5e-7`; rescale/taper controls are normalized identity checks, not full amplitude-gauge proof. Corrected writeback v4 shows support-only has zero phase/operator/floor and zero phase branch despite support/logit/qtrace; raw phase-only gives synthetic branch but no naked branch. | Naked phase-only branch must survive without support/operator/floor help, remain gauge-stable, and avoid decorative slot-2 pressure. |
| Unit-phasor runtime contract | Supported for current childworld runtime after repair, including seeded nested, export-like readout, five-anchor all-surface direct/audio checks, and supplemental 18-WAV direct/audio checks. | The audit first failed on promoted packet `seed_state` with max norm error `0.180798`; after robust packet seed fallback/normalization, 3352 checked phasor tensors pass at max error `1.788e-7` with zero violations. Raw weighted mixture cancellation also passes with min norm `0.997107` across 304 states. Supplemental broad direct/audio checks pass 739 and 737 tensors with max error `1.192e-7`. Deliberate scale-0.83 corruption is detected. | Convert the broad WAV substrate into benchmark/continuity evidence before claiming audio-quality coverage. |
| Lattice/substrate | Weak regularizer, not primary branch-survival cause in the high-floor runs. | Synthetic-only and unified mixed-seed comparisons both show `pointwise_no_neighborhood` preserving child/phase branch. Shuffled/global smoothing slightly worsens loss and lowers writeback, so topology shapes quality but is not the branch-survival source here. | Require paired degradation on a stronger naked/branch-active suite before claiming lattice causality. |
| Ramanujan q prior / qtrace survival | Ramanujan demoted; child qtrace survival supported. | Post-repair q-basis ablation keeps `qtrace_only` numerically indistinguishable from Ramanujan. Uniform, shuffled, and compact controls increase law-family diversity but lose naked branch and major-arc gain. Qtrace inheritance ablation shows qtrace history and relation gain are inert here, while `child_survival_qtrace_weight` is causal. | Ramanujan must beat qtrace-only on heldout/naked branch and major gain; child qtrace survival must keep helping under non-Ramanujan learned/Fourier q controls. |
| Branch ontology | Rejected for promotion. | High-floor nested readout sibling response is nonzero, about `0.303`, but `nested_sibling` remains `0`. Qualified carry remains low/zero by branch even when same-child ID carry is mechanically `1.0`. Support+operator+floor-high yields nonzero naked phase branch but is still mostly decorative. | Nonzero `nested_sibling`, nonzero qualified carry, and non-decorative naked parent/phase branch must appear together. |
| Dense RAFA signature | Fragile heldout implementation signal; not a token claim. | Broad 18-case deterministic V0 still gives `dense_stability_only_not_token_claim`: stability improves (`+0.001848`) but separation and anti-collapse regress. The learned packet-factor scout compare covers `52` non-smoke profiles and returns `heldout_candidate_found_needs_seed`. Split-suite testing is stricter: v24/v28 are only `1/8`, v31/v32 law-head variants are `0/8`, robust candidates are `0/5`, and no partition passes. Redacted future-law utility is `0/4`; redacted tail-increment utility is mixed but control-confounded (`2/2` tail wins, `6` negative-control wins). | Dense signatures must pass held-out stability/separation/anti-collapse across seeds and rotated partitions, then prove redacted continuation/operator usefulness beyond explicit metadata and negative controls before claiming RAFA-token behavior. |
| Resonant Attention / Post-Token Memory | Scaffolded evaluator lane; not promoted. | `relational_qkv_v2` is a local Circleworld branch-QKV precursor only. Dense signatures and childworld records are candidate law-object sources, but no resonant retrieval, cross-depth composition, interference-selectivity, or operator-value assay has promoted them. | Promotion requires relational queries to retrieve compatible law objects above decoys, compose parent-child-grandchild refinements, and show selected values causally update state while geometry-disabled, residual-disabled, dense-only, stored-ID, and inert-value controls fail. |
| Hardy-Littlewood arc/promotability | Strongly supported as causal gateway across tested q families. | Flattening major/minor arc or promotability kills branch/writeback under Ramanujan, qtrace-only, uniform, shuffled, and compact q controls. Time-shuffling arc creates decorative activity without child/naked branch. | Test learned/Fourier q controls and see whether intact arcs can recover naked phase without Ramanujan/qtrace bookkeeping. |
| Semantic projector | Contract smoke only. | Exact five-control schema passes with prompt families `airplane`, `reed`, `bell`, `impact`; import works; deterministic mock path gives six diverse pairwise prompt outputs. This is not semantic proof because current projection requires explicit weights. | One-control sweeps must produce monotonic, identifiable changes and beat a random-projector control using a real learned/frozen embedding projector. |

## Internal Phase-Law v4 Joint/Nonbad Update

Run root:

`C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_objective_scout_v4_joint_nonbad_v1_2026_05_07`

Objective status is `no_locked_candidate`. Best run `v4_local005_damp020_curv025_p025_s150_lr4` fails as `hold_absolute_win_fraction`, with best score `-0.4853986778133745`, direct overall mean corr delta `+0.0014133001979746507`, direct best-row mean corr delta `+0.00410316475423101`, matched nonbad bin corr delta `-0.011955829830617834`, and matched good bin corr delta `-0.22175256182054173`.

Joint status is `no_joint_row_candidate`: candidate count `0`, row count `144`, and mean absolute nonbad-bin corr delta `-0.0040797997254572544`.

Best joint row:

- Run: `v4_local004_damp010_curv010_p000_s125_lr0`
- Row: `flat` / `low_energy_bins` / gain `2.0`
- Direct mean corr delta: `-6.83905271107356e-06`
- Absolute mean corr delta: `+0.0001417974014295602`
- Nonbad bin corr delta: `+0.00009552880734839431`
- Weak bin corr delta: `+0.00004287588986119856`
- Moderate bin corr delta: `+0.000806343193425537`
- Good bin corr delta: `0.0`
- Absolute corr win fraction: `0.5573770491803278`

Failure atlas status is `atlas_built`, with joint status `no_joint_row_candidate`, run count `12`, and family row count `108`. Baseline bins remain non-promotional:

| Bin | Mean corr delta | Corr win fraction |
|---|---:|---:|
| Bad | `+0.040754753013395` | `0.558974358974359` |
| Weak | `-0.00036189144602898705` | `0.4415807560137457` |
| Moderate | `-0.012559782386438872` | `0.4477317554240631` |
| Good | `-0.12593373735696445` | `0.5192307692307693` |

Claim impact: internal phase-law variables remain causal recurrence variables because direct deltas are positive in the best objective run. They remain non-promotional for audio because matched nonbad/good bins regress and the only row with small positive absolute/nonbad behavior does not also improve direct recurrence.

## Internal Phase-Law v5 Low-Energy Target Row Update

Run root:

`C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_objective_scout_v5_low_energy_row_v1_2026_05_07`

Objective status is `no_locked_candidate`. The best global run `v5_lowrow_local004_damp010_curv010_p000_s125_lr2` fails as `hold_absolute_win_fraction`.

The target row is `flat` / `low_energy_bins` / `raw` / gain `2.0`. Best target-row run:

- Run: `v5_lowrow_local004_damp005_curv005_p000_s125_lr0`
- Objective target decision: `hold_target_direct_win_fraction`
- Joint target decision: `hold_direct_median`
- Direct mean corr delta: `+0.0000010155743203960363`
- Direct win fraction: `0.47541`
- Absolute mean corr delta: `+0.00014965202846102982`
- Nonbad bin corr delta: `+0.00010164278219327388`
- Good bin corr delta: `0.0`
- Absolute corr win fraction: `0.5573770491803278`

Failure atlas status is `atlas_built`, with joint status `no_joint_row_candidate`, run count `12`, and family row count `108`. Baseline bins remain non-promotional:

| Bin | Mean corr delta | Corr win fraction |
|---|---:|---:|
| Bad | `+0.04362277679952343` | `0.5435897435897435` |
| Weak | `-0.0004220056700717941` | `0.43827650013217023` |
| Moderate | `-0.014665772940473651` | `0.4467455621301775` |
| Good | `-0.1378854093109648` | `0.5833333333333334` |

Claim impact: v5 shows the low-energy row is controllable enough to cross direct mean while preserving nonbad absolute structure. It does not prove robust continuation because direct median/win fails and global good-bin damage worsens.

## Internal Phase-Law v6 Low-Energy Win Update

Run root:

`C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_objective_scout_v6_low_energy_win_v1_2026_05_07`

Objective status is `no_locked_candidate`. The best global run `v6_lowwin_local004_raw003_damp003_curv003_p000_s125_lr0` fails as `hold_matched_bad_baseline_dominated`.

The target row is `flat` / `low_energy_bins` / `raw` / gain `2.0`. Best target-row run:

- Run: `v6_lowwin_local004_raw003_damp003_curv003_p000_s125_lr0`
- Objective target decision: `hold_target_direct_win_fraction`
- Joint target decision: `hold_direct_win_fraction`
- Direct mean corr delta: `+0.000008544895570530304`
- Direct median corr delta: `+0.000000196153106327146`
- Direct win fraction: `0.5081967213114754`
- Absolute mean corr delta: `+0.00015718134971116408`
- Absolute median corr delta: `+0.00001466761477929594`
- Absolute win fraction: `0.5901639344262295`
- Nonbad bin corr delta: `+0.0001230016167545306`
- Nonbad bin win fraction: `0.5862068965517241`
- Good bin corr delta: `0.0`

Joint diagnostic and atlas:

- Joint status: `no_joint_row_candidate`
- Candidate rows: `0`
- Joined rows: `168`
- Mean absolute nonbad-bin corr delta: `-0.0047463987725493975`
- Atlas bad / weak / moderate / good bin means: `+0.04729170141675616` / `-0.00043647997738324154` / `-0.015318471193859833` / `-0.1380008551313037`

Claim impact: v6 advances the internal phase-law claim from "mean-only ridge" to "positive mean plus positive median on the target row." It still does not prove audio continuation because the direct win fraction is only `31/61`, below the `>=0.55` target gate, and the global/baseline-bin read is still bad-baseline-rescue shaped.

## Internal Phase-Law v7 Low-Energy Gain-Ladder Update

Run root:

`C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_objective_scout_v7_low_energy_gain_ladder_v1_2026_05_07`

Objective status is `no_locked_candidate`. The best global run `v7_gain_local004_raw003_damp002_curv002_s100` fails as `hold_direct_too_small`; the direct variant read is only `variant_tradeoff_or_tiny_signal`.

Best target-row read:

- Run: `v7_gain_local0045_raw003_damp003_curv003_s125`
- Objective target decision: `hold_target_direct_win_fraction`
- Direct mean corr delta: `+0.000010019920859155829`
- Direct median corr delta: `+0.00000019353332522242112`
- Direct win fraction: `0.5081967213114754`
- Absolute mean corr delta: `+0.0001586563749997896`
- Absolute win fraction: `0.5901639344262295`
- Nonbad bin corr delta: `+0.00012720936557853514`
- Good bin corr delta: `0.0`

Target-flip diagnostic:

- Selected run: `v7_gain_local0035_raw003_damp003_curv003_s125`
- Target gain mean / median / wins: `+0.000008438469877048604` / `+0.0000002237293352498962` / `0.5081967213114754`
- Target gain cases: `31` positive, `28` negative, `2` zero
- Sign categories across gains: `29` positive all gains, `30` nonpositive all gains, `2` rescued by gain
- Leave-one-out excluding `steady_buzz_nonmotor_control` reaches win fraction `0.5490196078431373`

Claim impact: v7 rules out "turn the gain knob" as the missing proof. Gain can raise direct mean, but it barely changes which cases win. The next internal phase-law test should be family/case-conditional: preserve the saw/chain and combustion wins while removing the buzzy-synth and steady-buzz drag.

## Internal Phase-Law v8 Family/Support-Mask Update

Run root:

`C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_objective_scout_v8_family_support_mask_v1_2026_05_07`

Objective status is `no_locked_candidate`. The best global run `v8_mask_local004_raw003_damp002_curv002_s100` fails as `hold_matched_bad_baseline_dominated`, and the joint diagnostic has `0` candidates across `60` rows.

Best target-row read:

- Run: `v8_mask_local0045_raw003_damp003_curv003_s125`
- Objective target decision: `hold_target_direct_win_fraction`
- Direct mean corr delta: `+0.000010019920859155829`
- Direct median corr delta: `+0.00000019353332522242112`
- Direct win fraction: `0.5081967213114754`
- Absolute mean corr delta: `+0.0001586563749997896`
- Absolute win fraction: `0.5901639344262295`
- Nonbad bin corr delta: `+0.00012720936557853514`
- Good bin corr delta: `0.0`

Support-mask diagnostic for the objective-selected run at gain `2.0`:

| Mask | Mean corr d | Median corr d | Wins |
|---|---:|---:|---:|
| `all_bins` | `+0.001042980210303603` | `+0.0001348094873312427` | `0.6229508196721312` |
| `high_energy_bins` | `+0.0009958627140724337` | `+0.00009632557031789531` | `0.6229508196721312` |
| `low_energy_bins` | `+0.000010019920859155829` | `+0.00000019353332522242112` | `0.5081967213114754` |

Best direct row:

- Run: `v8_mask_local0045_raw003_damp003_curv003_s125`
- Mask/gain: `high_energy_bins` / `1.75`
- Mean / median / wins: `+0.0008355768415039223` / `+0.00008620824849370776` / `0.6557377049180327`

Claim impact: support geometry is now a separate RAFA variable. Global `all_bins` and `high_energy_bins` masks improve direct breadth, while the low-energy target row preserves the nonbad/good near-miss. The failure is that no single global mask satisfies the locked objective. The next test should learn or infer support routing from local phase/signature features, not hand-label families.

Support-router oracle:

| Router | Mean corr d | Median corr d | Wins |
|---|---:|---:|---:|
| `low_energy_fixed` | `+0.000010019920859155829` | `+0.00000019353332522242112` | `0.5081967213114754` |
| `global_fixed_best` (`all_bins`) | `+0.001042980210303603` | `+0.0001348094873312427` | `0.6229508196721312` |
| `family_oracle_router` | `+0.0010434926049990685` | `+0.00002535188285181407` | `0.7049180327868853` |
| `case_oracle_router` | `+0.0016661391383341617` | `+0.00014526865731809366` | `0.8032786885245902` |

Oracle impact: family labels are not an acceptable runtime dependency, but the ceiling justifies a v9 learned/local support-router experiment. The router should infer mask preference from local phase/signature/support observables.

## Internal Phase-Law v9 Phase-Only Mask Update

Run root:

`C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_objective_scout_v9_phase_masks_v1_2026_05_07`

The v9 assay adds default-off phase-derived support masks: `phase_stable_bins`, `phase_dynamic_bins`, and `phase_low_motion_bins`. These masks use prefix phase velocity rather than prefix energy.

Outcome:

- Objective status: `no_locked_candidate`
- Joint status: `candidate_joint_row_found`
- Candidate rows: `11` / `60`
- Best row: `v8_mask_local004_raw003_damp002_curv002_s100` / `phase_stable_bins` / gain `2.0`
- Direct mean / median / wins: `+0.00031847714441993473` / `+0.000046031686952028444` / `0.639344262295082`
- Absolute mean / median / wins: `+0.002700202968841946` / `+0.00007090066385779387` / `0.5737704918032787`
- Nonbad / weak / moderate / good corr d: `+0.0008055606836901082` / `+0.00048060491060858017` / `+0.005192463620290737` / `0.0`
- Explicit phase-stable target score candidates: `2` / `5`
- Explicit phase-stable target score best run / decision / score: `v8_mask_local004_raw003_damp002_curv002_s100` / `target_row_candidate` / `0.009769155093884003`
- Explicit phase-stable joint target candidates: `2` / `5`
- Explicit phase-stable joint target best run / decision / score: `v8_mask_local004_raw003_damp002_curv002_s100` / `candidate_joint_row` / `0.009428266037978807`

Claim impact: phase-only support is now a plausible RAFA variable, not just a philosophical constraint. The generalized locked target path confirms the explicit `phase_stable_bins/gain2` row clears row-level candidate criteria. This still does not promote audio continuation because the whole-run objective remains `no_locked_candidate` and global selection still fails `hold_absolute_win_fraction`; the next step is a learned/local phase-support router or phase-stable-focused coefficient scout that keeps the target-row win while improving global selection.

## Internal Phase-Law v10 Phase-Router Mask Update

Run root:

`C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_objective_scout_v10_phase_router_masks_v1_2026_05_07`

Outcome:

- Objective status: `no_locked_candidate`
- Objective best decision: `hold_direct_too_small`
- Joint status: `candidate_joint_row_found`
- Candidate rows: `29` / `80`
- Router target row: `flat` / `phase_router_bins` / `raw` / gain `2.0`
- Router target joint candidates: `4` / `5`
- Router target best run / decision / score: `v8_mask_local004_raw0035_damp003_curv003_s125` / `candidate_joint_row` / `0.012583116024110755`
- Router target direct / absolute / nonbad / good corr d: `+0.0008050548815987055` / `+0.0034083202721412876` / `+0.0011328893661807422` / `0.0`
- Stable target joint candidates / best score: `2` / `0.009428266037978807`
- Coherent-motion target joint candidates / best score: `2` / `0.00961552098200868`

Claim impact: phase-derived support routing is now stronger than the fixed stable-mask result. This is still not audio promotion because the global objective fails the current direct-size threshold, but the failure is much cleaner than earlier bad-baseline/nonbad regressions. The next isolation claim should test whether the `direct_best_mean_corr_delta >= 0.001` gate is genuinely necessary or whether a predeclared phase-router objective with smaller direct effect but positive absolute/nonbad/good behavior should be its own diagnostic acceptance class.

## Internal Phase-Law v11 Phase-Router Direct Update

- Run root: `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_objective_scout_v11_phase_router_direct_v1_2026_05_07\`
- Objective status / best decision: `no_locked_candidate` / `hold_direct_too_small`
- Objective target candidates: `12` / `12`
- Joint status / candidates: `candidate_joint_row_found` / `36` / `36`
- Joint best row: `v11_router_local004_raw0035_damp002_curv002_s100` / gain `2.25`
- Joint best direct / absolute / nonbad / good corr d: `+0.0007964143239529634` / `+0.0037224731069172433` / `+0.0013637684558537445` / `0.0`
- Candidate-class split: `0` strict audio candidates, `19` phase-support candidates

Claim impact: v11 strengthens the phase-derived support-routing claim while blocking the naive next step. The router substrate is now uniformly positive under same-row joint criteria, but increasing local/raw pressure does not push direct mean past `0.001`; it mostly raises absolute/nonbad support. This is evidence for a real support-geometry variable, not evidence for audio autonomy.

## Internal Phase-Law v12 Phase-Reentry Direct Update

- Run root: `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_objective_scout_v12_phase_reentry_direct_v1_2026_05_07\`
- Objective status / best decision: `no_locked_candidate` / `hold_direct_too_small`
- Objective target candidates: `11` / `12`
- Joint status / candidates: `candidate_joint_row_found` / `35` / `36`
- Joint best row: `v12_reentry_r006_a050_local004_raw003_damp002_curv002_s100` / gain `2.25`
- Joint best direct / absolute / nonbad / good corr d: `+0.000868096871660527` / `+0.003794155654624807` / `+0.0016471319853362948` / `0.0`
- Candidate-class split: `0` strict audio candidates, `16` phase-support candidates
- Top phase-support class row: `v12_reentry_r008_a075_local004_raw0035_damp002_curv002_s100` / gain `2.25`, direct `+0.000879655278537222`, absolute `+0.0038057140615015018`, nonbad `+0.001701117585137462`

Claim impact: v12 makes temporal reentry an explicit testable RAFA variable, and it slightly improves target direct/nonbad recurrence over v11. It does not solve audio autonomy: strict candidates remain zero, the best decision remains `hold_direct_too_small`, and phase-support breadth falls from `19` to `16`. Reentry should now be split into a causality-clean isolated claim rather than bundled into more coefficient pressure.

## Internal Phase-Law v13 Causal Phase-Reentry Direct Update

- Run root: `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_objective_scout_v13_causal_reentry_direct_v1_2026_05_07\`
- Objective status / best decision: `no_locked_candidate` / `hold_direct_too_small`
- Objective target candidates: `5` / `10`
- Joint status / candidates: `candidate_joint_row_found` / `24` / `30`
- Joint best row: `v13_causal_reentry_r006_a050_local004_raw003_damp002_curv002_s100` / gain `2.25`
- Joint best direct / absolute / nonbad / good corr d: `+0.0008692742414680994` / `+0.003795333024432379` / `+0.0016491977007333555` / `0.0`
- Candidate-class split: `0` strict audio candidates, `12` phase-support candidates
- Top phase-support class row: `v13_causal_reentry_r008_a075_local004_raw0035_damp002_curv002_s100` / gain `2.25`, direct `+0.000881642325190853`, absolute `+0.0038077011081551326`, nonbad `+0.0017039658929137432`

Claim impact: v13 shows the reentry signal is not just v12 boundary lookahead: the best target row survives under a zero-boundary causal carrier. The tradeoff is narrower breadth, so causal reentry should be treated as a real but weak internal variable. It still does not prove semi-tokenless audio viability.

## Internal Phase-Law Predeclared Diagnostic Class Update

- Artifact: `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_predeclared_diagnostic_class_v10_v13_2026_05_07\internal_phase_law_predeclared_diagnostic_class.json`
- Class: `small_direct_strong_absolute_nonbad_not_promotion`
- Status: `diagnostic_acceptance_found_not_promotion`
- Diagnostic / strict audio rows: `50` / `0`
- Total rows / tracks: `182` / `4`
- Best track by count: `v11_phase_router_direct`
- Track diagnostic counts: v10 `3`, v11 `19`, v12 `16`, v13 `12`
- Best row: `v13_causal_reentry_direct` / `v13_causal_reentry_r008_a075_local004_raw0035_damp002_curv002_s100` / gain `2.25`
- Best row direct / absolute / nonbad / good corr d: `+0.000881642325190853` / `+0.0038077011081551326` / `+0.0017039658929137432` / `0.0`

Claim impact: this formalizes the non-promotional landing zone. Internal phase-law variables have a reproducible diagnostic class, but strict audio candidates remain zero. This should stop the coefficient/reentry ladder unless we explicitly change the objective or move to learned dense signatures/operators.

## Dense Signature Failure-Mode Update

- Artifact: `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\dense_signature_failure_modes_2026_05_07\dense_signature_failure_modes.json`
- Status: `dense_stability_only_not_token_claim`
- Promotion effect: `none`
- Stability delta: `+0.001848042342397993`
- Separation delta: `-0.011797438375651836`
- Dominant cluster share delta: `+0.04166666666666663`
- Near-duplicate pair share delta: `+0.02486404776573181`
- Prefix status / monotonicity: `prefix_order_contract_weak` / `0.70`
- Operator-tail distinctness: `not_measured_in_current_artifact`
- Contract audit: `schema_ready_but_learned_body_absent`
- Learned body status: `not_learned_deterministic_factor_composite`
- Learning contract: `pass`
- Learning claim effect: `learned_dense_module_contract_ready_not_trained`
- Learning runtime effect: `none`
- Learning law head: explicit `law_signature` projection head added; contract checks law shape `[5, 18]` and unit-norm pass.
- Learning overfit smoke: `24` steps reduce total loss from `1.4686511754989624` to `0.12286952137947083`
- Learned scout: `learned_signature_scout_not_candidate`
- Learned scout split: `in_sample`
- Learned scout stability delta vs metadata: `-0.03790768318706084`
- Learned scout separation delta vs metadata: `+0.1711389534175396`
- Learned scout operator-tail effective clusters: `7.4648231571710655`
- Learned scout compare: `heldout_candidate_found_needs_seed`
- Learned scout candidates: strict `4` / `52`, two-axis `11` / `52`, stability+anti-collapse `5` / `52`, heldout `1` / `35`, heldout two-axis `4` / `35`
- Best heldout profile: `learned_signature_scout_v24_seed1729_train12_holdout6_h96_run125_tf6_cuda_2026_05_07`
- Best heldout deltas: stability `+0.000425`, separation `+0.119588`, dominant-cluster delta `-0.190476`, near-duplicate delta `-0.171893`, effective-rank delta `+0.444968`
- Heldout v24 seed replication: `1` / `3`; candidate seed `[1729]`, failed seeds `[2718, 3141]`
- Heldout v24 rotation check: offsets `3`, `6`, and `12` all fail; offset `3` keeps separation but loses stability, offset `6` loses stability and collapse, offset `12` collapses separation/rank.
- Learned split-suite compare: `fragile_split_suite_candidate_only`, robust candidates `0` / `5`, fragile candidates `2` / `5`.
- Best split-suite profile: `v28_tf6_meta025_stability_core`; heldout candidates `1` / `8`, seed candidate fraction `1` / `3`, partition candidate fraction `0` / `5`, mean stability delta `-0.007584`, mean separation delta `+0.112720`.
- Coverage split-suite profile: `v30_tf6_train15_holdout3_coverage_core`; heldout candidates `0` / `8`; broader training coverage improves rank but does not rescue seed or partition stability.
- Law-head split-suite profiles: v31 baseline is `0` / `8` with mean stability `-0.005775`; v32 meta-0.25 is `0` / `8` with mean stability `-0.003846`, mean separation `+0.032254`, mean final law loss `0.032666`, and mean final operator loss `0.015981`.
- Heldout case failure atlas: `case_failure_atlas_built`, `210` heldout case rows across `18` cases; worst mean-stability blockers are `soundbible_steam_engine`, `footsteps_cement`, `bells_tibetan_large`, `audience_applause`, and `male_vocalized_a_z`.
- Operator-tail predictive probe compare: `operator_tail_mixed_usefulness`, learned views win `1` / `4` full rows; learned operator tail wins `law_signature` only on offset `0`, while explicit non-law metadata wins `operator_seed` and both offset `3` targets.
- Redacted future-law probe: `future_law_no_learned_usefulness`, fixed ridge alpha, learned wins `0` / `4`, learned-vs-non-law wins `0` / `4`.
- Redacted tail incremental probe: `tail_incremental_usefulness_mixed_control_confounded`, fixed ridge alpha, tail wins `2` / `2`, tail-vs-prefix wins `0` / `2`, tail-vs-permuted/cross/random wins `1` / `2` each, and total negative-control wins `6`.
- Best in-sample all-axis family: `learned_signature_scout_v14*_runcontrast_lowmid_basesvd_soft_cuda_2026_05_07`; seeded v14 candidate fraction `2` / `3`

Claim impact: dense signatures are now isolated as their own RAFA-token claim, and the learned lane has advanced from interface/smoke to narrow heldout implementation plus stricter negative evidence. The deterministic V0 broad harness still shows only an invariance/stability signal and is not a learned token body. The packet-factor learned encoder finds an in-sample frontier with v14, plus one heldout all-axis pass with v24 h96/run-contrastive/base-SVD and six metamer transforms. The split suite downgrades this from "candidate" to "fragile signal": v24 and v28 each pass only one favorable seed split, v31/v32 law-head variants pass no splits despite low law/operator head losses, no rotated/shuffled partitions pass, and train15/holdout3 coverage does not rescue the objective. The redacted future-law probe is negative (`0/4` learned wins). The redacted tail-increment probe is mixed but control-confounded: tail+metadata beats metadata on both law/operator targets, but prefix/shuffled/cross/random controls also win (`6` total control wins). The lane is now seed/partition/usefulness-isolation blocked. The next signature work should make the objective seed-stable across rotated splits and force tail-local usefulness beyond explicit metadata and all negative controls.

## Resonant Attention / Post-Token Memory Lane Update

Current status: `scaffolded_evaluator_lane_not_promoted`.

This lane is downstream of the dense-signature and childworld artifact lanes, but
it is a separate claim. The current Circleworld QKV-shaped mechanism,
`relational_qkv_v2`, remains only a local branch-QKV precursor: it couples local
branch modes and emits a branch handoff message. It does not retrieve reusable
packet, child, grandchild, or signature law objects, and it does not by itself
prove RAFA attention or post-token memory.

Promotion requires a dedicated assay bundle:

- resonant child retrieval above decoys
- cross-depth parent-child-grandchild composition
- interference selectivity as law-object bank size grows
- selected value/operator causality
- unit-phasor and gauge-contract preservation
- stored-ID, dense-only, geometry-disabled, residual-disabled, and inert-value
  negative controls

Claim impact: the branch hierarchy is now explicit. Dense signatures can remain
a learned-carrier claim, Childworld can remain a branch/continuation claim, and
Resonant Attention / Post-Token Memory only promotes if relational Q/K/V law
objects are retrieved, composed, and causally applied by resonance.

## Predeclared Audio Lockbox v1 Update

New artifacts:

- Manifest: `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\audio_predeclared_lockbox_v1_2026_05_06\audio_predeclared_lockbox_manifest.json`
- Probe: `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\audio_predeclared_lockbox_probe_v1_cuda_2026_05_06\audio_delta_mechanism_probe.json`
- Compare: `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\audio_predeclared_lockbox_compare_v1_2026_05_06\audio_lockbox_result_compare.json`
- Compare without single-source probe: `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\audio_predeclared_lockbox_compare_v1_no_single_source_2026_05_06\audio_lockbox_result_compare.json`
- Phase-law scout: `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\audio_phase_law_scout_v1_cuda_2026_05_07\audio_delta_mechanism_probe.json`
- Phase-law scout no-single-source compare: `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\audio_phase_law_scout_compare_v1_no_single_source_2026_05_07\audio_lockbox_result_compare.json`

The lockbox builder freezes filename-derived case selection only, rejects non-RIFF `.wav` files, and enforces disjoint paths, content hashes, and provider-neutral stems. It does not select by target, future, baseline, correlation, or outcome metrics.

Outcome:

- Selected cases: `62`
- Frozen row: `prefix_hold` / `all_bins` / `time_smooth_3`, gain `2.0` against gain `0.0`
- Compare status: `tradeoff_signal_only`
- Mean corr delta: `+0.011134`
- Median corr delta: `-0.000039`
- Corr win fraction: `0.451613`
- Non-bad weighted mean corr delta: `-0.015601`
- Positive outlier share: `0.561650`
- Without single-source probe: mean corr delta `-0.004088`, median `-0.000074`, mean MSE delta `+0.001024`
- Phase-law scout best row without single-source probe: `stable_velocity_time_smooth_3`, gain `2.0`, status `bad_baseline_rescue_dominated`, mean corr delta `+0.004642`, median `-0.000162`, corr win fraction `0.393443`

Claim impact: this strengthens the negative read. The fixed mechanism still does not prove semi-tokenless audio viability; the apparent mean gain is bad-baseline rescue dominated.

## Child-Writeback Interpretation

Confirmed narrowly:

- Support/logit/q-trace writeback can be measured, but corrected support-only v4 is not phase causal.
- Operator/floor writeback can make naked phase branch nonzero only in the high-floor support+operator condition.
- Raw same-child ID carry can persist, but it is mechanical unless qualified carry also survives.

Rejected for promotion:

- `nested_sibling` is still not proven.
- Headline `real_branch_fraction` remains child-readiness dominated unless parent and phase branches are reported separately.
- Phase-floor strength can create visible branch metrics while remaining decorative or trading against audio/major-arc structure.
- Readout sibling response alone is insufficient evidence for branch ontology.

Next isolating experiment:

Run fixed-seed ablations on the same substrate and report parent branch, child branch, phase-only branch, decorative slot-2 fraction, readout sibling response, raw carry, qualified carry, benchmark drift, and continuity drift separately.

Fixed variants in the current v4 harness:

- `baseline_current`
- `support_only_phase_operator_floor_off`
- `raw_phase_only_support_operator_floor_off`
- `operator_floor_only_support_off`
- `support_operator_no_floor`
- `support_operator_floor_low`
- `support_operator_floor_fixed_2_10`
- `support_operator_floor_high`
