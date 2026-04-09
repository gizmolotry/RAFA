from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "lineages" / "04_positive_replacement" / "ablate_formalization.py"


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Circleworld formalization ablation.")
    ap.add_argument("--mode", default="active_packets", choices=["no_promotion", "passive_packets", "active_packets"])
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cmd = [
        sys.executable,
        str(SCRIPT),
        "--mode", args.mode,
        "--depth", str(args.depth),
        "--device", args.device,
    ]
    if args.out:
        cmd.extend(["--out", args.out])
    raise SystemExit(subprocess.call(cmd, cwd=str(ROOT)))


if __name__ == "__main__":
    main()
