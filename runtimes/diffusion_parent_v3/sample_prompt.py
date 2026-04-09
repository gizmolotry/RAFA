from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    ap = argparse.ArgumentParser(description="Diffusion parent-v3 prompt sampler.")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default=str(ROOT / "outputs" / "sample_diffusion_runtime.wav"))
    ap.add_argument("--prompt", default="cinematic rhythmic texture")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--steps", type=int, default=None)
    args = ap.parse_args()

    cmd = [
        sys.executable,
        str(ROOT / "sample_diffusion.py"),
        "--ckpt", args.ckpt,
        "--out", args.out,
        "--prompt", args.prompt,
        "--seed", str(args.seed),
    ]
    if args.steps is not None:
        cmd.extend(["--steps", str(args.steps)])
    raise SystemExit(subprocess.call(cmd, cwd=str(ROOT)))


if __name__ == "__main__":
    main()
