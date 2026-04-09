from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CORE = ROOT / "core"
for path in (ROOT, CORE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from tools.export_audio import render_graduation_suite


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage4 Blackwell 16 graduation export.")
    ap.add_argument("--ckpt", default=str(ROOT / "checkpoints_stage4" / "bound_weights_ep99_step200.pt"))
    ap.add_argument("--out-dir", default=str(ROOT / "outputs" / "graduation_pack_bw16"))
    args = ap.parse_args()
    render_graduation_suite(args.ckpt, args.out_dir)


if __name__ == "__main__":
    main()
