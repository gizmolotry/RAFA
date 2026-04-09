# Path B Hypercube Orchestration (Airflow + S3)

This folder adds a reproducible orchestration layer around existing Path B tools.

## What it does

1. Expands a hypercube of ablation settings into concrete jobs.
2. Runs each job with existing training/eval scripts:
   - `train_diffusion.py`
   - `tools/path_b_eval.py`
3. Optionally uploads checkpoints, eval wavs, and logs to S3.

Core training code is unchanged except `config.py` now supports `RAFA_CONFIG_PATH`, so each job can use an isolated temp config without mutating the main `config.yaml`.

## Files

- `hypercube_manifest.yaml`: dimensions and defaults.
- `generate_hypercube_jobs.py`: builds concrete job list JSON from the manifest.
- `run_hypercube_job.py`: executes one job (train/eval/upload).
- `../airflow/dags/rafa_pathb_hypercube.py`: Airflow DAG with dynamic task mapping.

## Local dry run

```powershell
python research_track/infra/generate_hypercube_jobs.py `
  --manifest research_track/infra/hypercube_manifest.yaml `
  --out research_track/infra/generated_jobs.json
```

Then run one job by writing a single-job payload:

```powershell
@'
{"job": {
  "name": "pathb_hypercube_mem1_ram1_slow1",
  "model_type": "rafa",
  "max_steps": 1000,
  "memory_enabled": true,
  "ablate_ramanujan": false,
  "ablate_slow_clock": false,
  "notes": "manual smoke",
  "use_existing_checkpoint": true,
  "s3": {"enabled": false, "bucket": "", "prefix": "rafa/path_b_hypercube"}
}}
'@ | Set-Content tmp\one_job.json

python research_track/infra/run_hypercube_job.py --job-json tmp\one_job.json
```

## S3

Set this per-job in the manifest (defaults or named run):

```yaml
s3:
  enabled: true
  bucket: "your-bucket-name"
  prefix: "rafa/path_b_hypercube"
```

Requires `boto3` and AWS credentials in environment or instance role.

## Airflow

DAG id: `rafa_pathb_hypercube`

Expected repo layout: DAG file stays in `research_track/airflow/dags` inside this repo so relative paths resolve correctly.

### Bounded DAG runs

The DAG now accepts `dag_run.conf` so you can execute only a slice of the hypercube:

- `start`: 0-based index into generated jobs
- `count`: number of jobs to run

Example trigger payload (first 4 jobs):

```json
{"start": 0, "count": 4}
```

If omitted, the DAG runs from `start=0` through all generated jobs.
