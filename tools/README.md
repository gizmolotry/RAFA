# Tools

`tools/` contains operational scripts.

## Main buckets
- Training stages: `train_stage1_voice.py`, `train_stage2_spine.py`, `train_stage3_brain.py`
- Evaluation: `path_b_eval.py`, `relational_probe.py`, `eval_semantic_tension.py`, `run_validation.py`
- Triton / runtime checks: `test_triton.py`, `test_triton_deq.py`, `verify.py`, `verify_physics.py`
- Audits and debugging: `audit_*`, `fault_localization.py`, `binding_stats.py`, `render_binding_comparison.py`
- Data prep / scraping: `dsp_labeler.py`, `extract_tension_targets.py`, `manifest_gen.py`, `scrape_*`

## Rule of thumb
If you want to do something to the repo rather than change the model itself, it is probably here.
