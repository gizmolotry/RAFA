from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_law_token_library import _default_cases_path, build_library


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a Circleworld relational-signature library from real anchor audio.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--cases-json", default=str(_default_cases_path()))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--clip-seconds", type=int, default=10)
    args = ap.parse_args()

    summary = build_library(
        config_path=Path(args.config),
        out_dir=Path(args.out_dir),
        cases_path=Path(args.cases_json),
        device_name=args.device,
        clip_seconds=int(args.clip_seconds),
    )
    print(
        json.dumps(
            {
                "saved": str(Path(args.out_dir) / "relational_signature_library.json"),
                "num_cases": summary["num_cases"],
                "mean_num_relational_signatures": summary["mean_num_relational_signatures"],
                "mean_num_relational_signature_families": summary["mean_num_relational_signature_families"],
                "aggregate_num_signature_families": summary["aggregate_relational_signature_library"]["num_families"],
                "aggregate_relational_effective_family_count": summary["aggregate_relational_effective_family_count"],
                "aggregate_relational_family_entropy": summary["aggregate_relational_family_entropy"],
                "aggregate_relational_dominant_family_pressure": summary["aggregate_relational_dominant_family_pressure"],
                "case_relational_signature_family_count_std": summary["case_relational_signature_family_count_std"],
                "case_relational_signature_confidence_std": summary["case_relational_signature_confidence_std"],
                "case_relational_branch_mass_std": summary["case_relational_branch_mass_std"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
