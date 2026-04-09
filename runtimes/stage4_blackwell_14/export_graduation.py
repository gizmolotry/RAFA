from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CORE = ROOT / "core"
for path in (ROOT, CORE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from tools.export_audio_compat14 import render_graduation_suite_compat14


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage4 Blackwell 14 graduation export.")
    ap.add_argument("--ckpt", default=str(ROOT / "checkpoints_stage4" / "bound_weights_ep10_step1200.pt"))
    ap.add_argument("--out-dir", default=str(ROOT / "outputs" / "graduation_pack_restore_candidate"))
    ap.add_argument("--source", default="qkv", choices=["spine", "qkv", "filt"])
    ap.add_argument("--phase-mix", type=float, default=0.35)
    ap.add_argument("--mag-mix", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=101)
    args = ap.parse_args()

    render_graduation_suite_compat14(
        ckpt_path=args.ckpt,
        out_dir=args.out_dir,
        source=args.source,
        phase_mix=args.phase_mix,
        mag_mix=args.mag_mix,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
