# RAFA-PC Evaluation Ledger
**Protocol (Path B):** All models are evaluated as diffusion denoisers on the 5.8k audio corpus. Success is measured by generative structural stability (avoiding mode collapse / beeps) across a 24-sample grid (6 seeds, 4 step counts: 8, 16, 32, 64).

### Scorecard Metrics Definition
*   **Collapse Rate:** % of samples with RMS < 0.001, TopRatio > 50, or CentVar < 100. Lower is better.
*   **Avg TopRatio:** Ratio of highest energy frequency bin to mean energy. Very high (>30) = single tone / beep. Lower (~5-10) = structured harmonic spread.
*   **Avg CentVar:** Variance of the spectral centroid over time. High = temporally evolving structure. Low (<1000) = static drone.
*   **Avg DPhiStd:** Standard deviation of phase velocity over time. Measures phase motion.

## The Grid

| Date | Git Commit | Run Name / Ablation | Checkpoint Steps | Collapse Rate | Avg TopRatio | Avg CentVar | Avg DPhiStd | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 2026-02-28 | `e2d36ec` | baseline_v3 | ep62 | 0.0% | 6.92 | 456101 | 2.6282 | Baseline model, SNR weighting, no RAFA. The Truth Anchor. |
| 2026-03-02 | `e2d36ec` | rafa_full | 1000 | 0.0% | 7.50 | 76045 | 1.1789 | Full RAFA-PC v3.0 (Memory + Hyena + Ramanujan + Slow Clock) |
| 2026-03-02 | `e2d36ec` | rafa_no_memory | 1000 | 0.0% | 8.02 | 85098 | 1.1983 | Ablation: Relational Motif Memory disabled |
| 2026-03-03 | `e2d36ec` | rafa_no_ramanujan | 1000 | 0.0% | 7.41 | 73101 | 1.1777 | Ablation: Ramanujan Gating disabled |
| 2026-03-03 | `e2d36ec` | rafa_no_slow_clock | 1000 | 0.0% | 7.47 | 75492 | 1.1860 | Ablation: Slow Clock (Multi-rate) disabled |
| 2026-03-04 | `e2d36ec` | rafa_full | 1000 | 0.0% | 7.56 | 76771 | 1.1839 | runner_validation |
| 2026-03-04 | `e2d36ec` | bridge_e2e_smoke | ep1 | 0.0% | 6.24 | 67766 | 0.5648 | bridge_smoke_e2e |
| 2026-03-04 | `e2d36ec` | bridge_lc_s30 | ep1 | 0.0% | 6.24 | 67766 | 0.5648 | bridge_learning_curve_step_30 |
| 2026-03-06 | `e2d36ec` | bridge_lc_vram_s300 | 300 | 0.0% | 13.01 | 144241 | 2.3937 | bridge_learning_curve_vram_step_300 |
| 2026-03-07 | `e2d36ec` | bridge_lc_vram_s3000 | 3000 | 0.0% | 13.20 | 146012 | 2.4028 | bridge_learning_curve_vram_step_3000 |
| 2026-03-09 | `e2d36ec` | bridge_lc_vram_s8000 | 8000 | 0.0% | 8.43 | 131625 | 1.1764 | bridge_learning_curve_vram_step_8000 |
| 2026-03-09 | `e2d36ec` | pathb_hypercube_mem1_ram1_slow1_cond1_meta1 | 1000 | 0.0% | 8.00 | 89083 | 1.1667 | hypercube memory=1 ramanujan=1 slow_clock=1 conductor=1 metamer=1 |
| 2026-03-10 | `e2d36ec` | pathb_hypercube_mem1_ram1_slow1_cond1_meta1 | 1000 | 0.0% | 7.91 | 94667 | 1.1695 | hypercube memory=1 ramanujan=1 slow_clock=1 conductor=1 metamer=1 |
| 2026-03-12 | `e2d36ec` | prompt_conditioned_ifs_control_smoke_to120 | ep1 | 0.0% | 3.39 | 57325 | 0.1643 | IFS-control smoke continuation to 120 steps after local split/val optimization. |
| 2026-03-12 | `e2d36ec` | prompt_conditioned_hybrid_smoke | ep1 | 0.0% | 2.11 | 4550 | 0.1614 | Hybrid FiLM + IFS smoke run on local multimodal manifest prompts. |
| 2026-03-13 | `e2d36ec` | semantic_tension_steering_smoke | ep1 | 0.0% | 3.07 | 16698 | 0.1633 | Semantic Hamiltonian smoke run: time-varying tension envelope steers harmonic vs inharmonic q-routing without text embeddings. |
