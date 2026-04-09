import json
import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CORE = ROOT / "core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from tools.export_audio_compat14 import render_graduation_suite_compat14


BEST_CONFIG = {
    "checkpoint": "D:\\RAFA\\checkpoints_stage4\\bound_weights_ep10_step1200.pt",
    "source": "qkv",
    "phase_mix": 0.35,
    "mag_mix": 0.05,
    "seed": 101,
}


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wf:
        return np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0


def compare_suite(original_dir: Path, candidate_dir: Path) -> dict:
    files = ["grad_engine.wav", "grad_voice.wav", "grad_impact.wav", "grad_drone.wav"]
    rows = {}
    mses, maes, corrs = [], [], []
    for name in files:
        a = read_wav(original_dir / name)
        b = read_wav(candidate_dir / name)
        n = min(len(a), len(b))
        a = a[:n]
        b = b[:n]
        mse = float(np.mean((a - b) ** 2))
        mae = float(np.mean(np.abs(a - b)))
        corr = float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else 0.0
        rows[name] = {"mse": mse, "mae": mae, "corr": corr, "n": int(n)}
        mses.append(mse)
        maes.append(mae)
        corrs.append(corr)
    return {
        "mean_mse": float(np.mean(mses)),
        "mean_mae": float(np.mean(maes)),
        "mean_corr": float(np.mean(corrs)),
        "per_file": rows,
    }


def main():
    root = ROOT
    out_dir = root / "outputs" / "graduation_pack_restore_candidate"
    original_dir = root / "outputs" / "graduation_pack"
    out_dir.mkdir(parents=True, exist_ok=True)

    render_graduation_suite_compat14(
        ckpt_path=BEST_CONFIG["checkpoint"],
        out_dir=str(out_dir),
        phase_mix=BEST_CONFIG["phase_mix"],
        mag_mix=BEST_CONFIG["mag_mix"],
        source=BEST_CONFIG["source"],
        seed=BEST_CONFIG["seed"],
    )
    summary = {
        **BEST_CONFIG,
        **compare_suite(original_dir, out_dir),
    }
    summary_path = out_dir / "comparison_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"WROTE {summary_path}")


if __name__ == "__main__":
    main()
