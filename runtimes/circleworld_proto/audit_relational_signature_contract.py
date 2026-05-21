from __future__ import annotations

import argparse
import dataclasses
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
LINEAGE = ROOT / "lineages" / "04_positive_replacement"
for path in (ROOT, LINEAGE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))



def _exit_if_help_requested_without_runtime() -> None:
    if any(arg in ("-h", "--help") for arg in sys.argv[1:]):
        print(f"usage: {Path(__file__).name} [runtime-coupled contract options]")
        print()
        print("This contract is coupled to the Circleworld runtime/signature lane.")
        print("Run it from a Circleworld integration worktree for full argument parsing and execution.")
        raise SystemExit(0)


_exit_if_help_requested_without_runtime()

from rafa_relational_signature import (  # noqa: E402
    RafaRelationalSignatureConfig,
    RafaRelationalSignatureV0,
    signature_prefix_contract,
)


SCHEMA = "rafa_relational_signature_contract_audit_v0"
TOKENBURST_ROOT = Path(r"C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst")
DEFAULT_SOURCE = LINEAGE / "rafa_relational_signature.py"
DEFAULT_OUT_DIR = TOKENBURST_ROOT / "relational_signature_contract_audit_2026_05_07"
EXPECTED_PREFIX_DIMS = [128, 256, 384, 512, 640, 768]
EXPECTED_FIELDS = {
    "h",
    "q_profile",
    "arc_profile",
    "temporal_profile",
    "support_profile",
    "branch_profile",
    "operator_seed",
    "confidence",
}


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _source_checks(source_text: str) -> dict[str, Any]:
    nn_module_defs = re.findall(r"class\s+\w+\s*\([^)]*nn\.Module[^)]*\)", source_text)
    torch_module_defs = re.findall(r"class\s+\w+\s*\([^)]*torch\.nn\.Module[^)]*\)", source_text)
    deterministic_assemblers = [
        name
        for name in (
            "_assemble_dense_body",
            "_assemble_dense_body_from_factor_pack",
            "construct_operator_seed",
            "_stretch_features",
        )
        if name in source_text
    ]
    learned_head_tokens = [
        token
        for token in (
            "nn.Linear",
            "torch.nn.Linear",
            "q_profile_head",
            "arc_profile_head",
            "temporal_head",
            "support_head",
            "branch_head",
            "operator_head",
        )
        if token in source_text
    ]
    return {
        "defines_nn_module": bool(nn_module_defs or torch_module_defs),
        "nn_module_defs": nn_module_defs + torch_module_defs,
        "deterministic_assemblers": deterministic_assemblers,
        "learned_head_tokens_present": learned_head_tokens,
        "has_dense_assembly_from_factor_pack": "_assemble_dense_body_from_factor_pack" in source_text,
        "has_extract_relational_signatures": "def extract_relational_signatures" in source_text,
    }


def audit_relational_signature_contract(source_path: Path, out_dir: Path) -> dict[str, Any]:
    source_text = source_path.read_text(encoding="utf-8")
    cfg = RafaRelationalSignatureConfig()
    prefix_contract = signature_prefix_contract(cfg)
    dataclass_fields = [field.name for field in dataclasses.fields(RafaRelationalSignatureV0)]
    missing_fields = sorted(EXPECTED_FIELDS.difference(dataclass_fields))
    source = _source_checks(source_text)
    prefix_dims = [int(dim) for dim in prefix_contract.get("prefix_dims", [])]

    schema_contract_passed = (
        int(prefix_contract.get("signature_dim", 0)) == 768
        and prefix_dims == EXPECTED_PREFIX_DIMS
        and not missing_fields
        and int(prefix_contract.get("operator_seed_dim", 0)) == 128
    )
    learned_body_present = bool(
        source["defines_nn_module"]
        and source["learned_head_tokens_present"]
        and not source["has_dense_assembly_from_factor_pack"]
    )
    if learned_body_present:
        learned_body_status = "learned_dense_body_present"
    elif source["has_dense_assembly_from_factor_pack"]:
        learned_body_status = "not_learned_deterministic_factor_composite"
    else:
        learned_body_status = "learned_body_not_established"

    if schema_contract_passed and learned_body_status == "not_learned_deterministic_factor_composite":
        claim_status = "schema_ready_but_learned_body_absent"
    elif schema_contract_passed and learned_body_present:
        claim_status = "learned_dense_contract_candidate"
    else:
        claim_status = "schema_contract_needs_review"

    summary = {
        "schema": SCHEMA,
        "created_utc": _utc_timestamp(),
        "source_path": str(source_path),
        "claim_status": claim_status,
        "schema_contract_passed": schema_contract_passed,
        "learned_body_status": learned_body_status,
        "prefix_contract": prefix_contract,
        "expected_prefix_dims": EXPECTED_PREFIX_DIMS,
        "dataclass_fields": dataclass_fields,
        "missing_expected_fields": missing_fields,
        "source_checks": source,
        "interpretation": (
            "The RAFA relational signature schema is present, but current v0 h is assembled from explicit factors. "
            "This supports interface readiness, not the stronger learned dense token-body claim."
            if claim_status == "schema_ready_but_learned_body_absent"
            else "Review the contract fields and learned-body status before using this as token evidence."
        ),
        "required_next_implementation": [
            "Introduce trainable projection heads or a learned encoder that produces h from phase/q/arc/support windows.",
            "Keep explicit q/arc/temporal/support/branch/operator fields as heads, diagnostics, or loss surfaces rather than the sole source of h.",
            "Train h with Matryoshka prefix, metamer consistency, predictive continuation, anti-collapse, and operator-tail losses.",
            "Retain the deterministic v0 factor pack as a baseline, not as proof of learned dense-body semantics.",
        ],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "relational_signature_contract_audit.json"
    md_path = out_dir / "RELATIONAL_SIGNATURE_CONTRACT_AUDIT.md"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_markdown(md_path, summary)
    return summary


def _write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Relational Signature Contract Audit",
        "",
        f"- schema: `{summary['schema']}`",
        f"- source: `{summary['source_path']}`",
        f"- claim status: `{summary['claim_status']}`",
        f"- schema contract passed: `{summary['schema_contract_passed']}`",
        f"- learned body status: `{summary['learned_body_status']}`",
        f"- signature dim: `{summary['prefix_contract'].get('signature_dim')}`",
        f"- prefix dims: `{summary['prefix_contract'].get('prefix_dims')}`",
        f"- missing expected fields: `{summary['missing_expected_fields']}`",
        "",
        "## Source Checks",
        "",
        f"- defines nn.Module: `{summary['source_checks'].get('defines_nn_module')}`",
        f"- learned head tokens present: `{summary['source_checks'].get('learned_head_tokens_present')}`",
        f"- deterministic assemblers: `{summary['source_checks'].get('deterministic_assemblers')}`",
        f"- has dense assembly from factor pack: `{summary['source_checks'].get('has_dense_assembly_from_factor_pack')}`",
        "",
        "## Interpretation",
        "",
        summary["interpretation"],
        "",
        "## Required Next Implementation",
        "",
    ]
    lines.extend([f"- {item}" for item in summary.get("required_next_implementation", [])])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Audit whether the RAFA relational signature implementation is only schema-ready or actually learned-token-body ready."
    )
    ap.add_argument("--source", default=str(DEFAULT_SOURCE))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = ap.parse_args()
    summary = audit_relational_signature_contract(
        source_path=Path(args.source),
        out_dir=Path(args.out_dir),
    )
    print(
        json.dumps(
            {
                "schema": summary["schema"],
                "saved": str(Path(args.out_dir) / "relational_signature_contract_audit.json"),
                "report": str(Path(args.out_dir) / "RELATIONAL_SIGNATURE_CONTRACT_AUDIT.md"),
                "claim_status": summary["claim_status"],
                "learned_body_status": summary["learned_body_status"],
                "schema_contract_passed": summary["schema_contract_passed"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
