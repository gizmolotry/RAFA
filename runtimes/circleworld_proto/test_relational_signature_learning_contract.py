from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
LINEAGE = ROOT / "lineages" / "04_positive_replacement"
for path in (ROOT, LINEAGE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from rafa_relational_signature import (  # noqa: E402
    RafaRelationalSignatureConfig,
    build_relational_factor_pack,
)
from rafa_relational_signature_learning import (  # noqa: E402
    RafaRelationalSignatureEncoderV0,
    matryoshka_signature_losses,
    summarize_learning_contract,
)


SCHEMA = "rafa_relational_signature_learning_contract_v0"
TOKENBURST_ROOT = Path(r"C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst")
DEFAULT_OUT_DIR = TOKENBURST_ROOT / "relational_signature_learning_contract_2026_05_07"


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _safe_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def _dummy_factor_pack(
    *,
    batch: int,
    q_dim: int,
    num_modes: int,
    branch_dim: int,
    law_dim: int,
    device: torch.device,
) -> Any:
    generator = torch.Generator(device=device)
    generator.manual_seed(731)
    q_profile = F.softmax(torch.randn(batch, q_dim, generator=generator, device=device), dim=-1)
    arc_profile = torch.sigmoid(torch.randn(batch, 5, generator=generator, device=device))
    temporal_profile = torch.sigmoid(torch.randn(batch, 8, generator=generator, device=device))
    support_profile = torch.sigmoid(torch.randn(batch, 6, generator=generator, device=device))
    branch_profile = torch.sigmoid(torch.randn(batch, num_modes, branch_dim, generator=generator, device=device))
    law_signature = F.normalize(torch.randn(batch, law_dim, generator=generator, device=device), dim=-1)
    confidence = torch.sigmoid(torch.randn(batch, 1, generator=generator, device=device))
    return build_relational_factor_pack(
        q_profile=q_profile,
        arc_profile=arc_profile,
        temporal_profile=temporal_profile,
        support_profile=support_profile,
        branch_profile=branch_profile,
        law_signature=law_signature,
        confidence=confidence,
        sig_cfg=RafaRelationalSignatureConfig(
            signature_dim=768,
            prefix_dims=(128, 256, 384, 512, 640, 768),
            operator_seed_dim=128,
            branch_profile_dim=branch_dim,
        ),
    )


def _grad_norm(model: torch.nn.Module) -> float:
    total = 0.0
    for param in model.parameters():
        if param.grad is None:
            continue
        total += float(param.grad.detach().float().norm().cpu().item())
    return total


def run_learning_contract(
    *,
    out_dir: Path,
    device_name: str,
    batch: int,
    hidden_dim: int,
    train_steps: int,
    learning_rate: float,
) -> dict[str, Any]:
    device = _safe_device(device_name)
    pack = _dummy_factor_pack(
        batch=int(batch),
        q_dim=7,
        num_modes=2,
        branch_dim=4,
        law_dim=18,
        device=device,
    )
    model = RafaRelationalSignatureEncoderV0.from_factor_pack(pack, hidden_dim=int(hidden_dim)).to(device)
    initial_outputs = model(pack)
    initial_losses = matryoshka_signature_losses(initial_outputs, pack)
    initial_total_loss = float(initial_losses["total_loss"].detach().cpu().item())
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate))
    for _step in range(max(0, int(train_steps))):
        optimizer.zero_grad(set_to_none=True)
        step_outputs = model(pack)
        step_losses = matryoshka_signature_losses(step_outputs, pack)
        step_losses["total_loss"].backward()
        optimizer.step()

    optimizer.zero_grad(set_to_none=True)
    outputs = model(pack)
    losses = matryoshka_signature_losses(outputs, pack)
    total_loss = losses["total_loss"]
    total_loss.backward()
    final_total_loss = float(total_loss.detach().cpu().item())
    contract = summarize_learning_contract(outputs, losses)
    grad_norm = _grad_norm(model)
    loss_reduction = float(initial_total_loss - final_total_loss)

    checks = {
        "h_shape_ok": contract["h_shape"] == [int(batch), 768],
        "q_shape_ok": contract["q_profile_shape"] == [int(batch), 7],
        "branch_shape_ok": contract["branch_profile_shape"] == [int(batch), 2, 4],
        "law_shape_ok": contract["law_signature_shape"] == [int(batch), 18],
        "operator_shape_ok": contract["operator_seed_shape"] == [int(batch), 128],
        "operator_tail_shape_ok": contract.get("operator_tail_shape") == [int(batch), 128],
        "h_unit_norm_ok": float(contract["max_h_unit_norm_error"]) <= 1.0e-5,
        "law_unit_norm_ok": float(contract["max_law_unit_norm_error"]) <= 1.0e-5,
        "operator_unit_norm_ok": float(contract["max_operator_unit_norm_error"]) <= 1.0e-5,
        "q_simplex_ok": float(contract["max_q_sum_error"]) <= 1.0e-5,
        "loss_finite": bool(torch.isfinite(total_loss).item()),
        "gradient_flow_ok": grad_norm > 0.0,
        "training_loss_reduces": int(train_steps) <= 0 or loss_reduction > 0.0,
    }
    status = "pass" if all(checks.values()) else "needs_review"
    summary = {
        "schema": SCHEMA,
        "created_utc": _utc_timestamp(),
        "status": status,
        "device": str(device),
        "requested_device": str(device_name),
        "batch": int(batch),
        "hidden_dim": int(hidden_dim),
        "train_steps": int(train_steps),
        "learning_rate": float(learning_rate),
        "initial_total_loss": initial_total_loss,
        "final_total_loss": final_total_loss,
        "loss_reduction": loss_reduction,
        "contract": contract,
        "gradient_norm": grad_norm,
        "checks": checks,
        "runtime_effect": "none",
        "claim_effect": (
            "learned_dense_module_contract_ready_not_trained"
            if status == "pass"
            else "learned_dense_module_contract_needs_review"
        ),
        "interpretation": (
            "A trainable dense-body encoder/head surface now exists and has finite losses/gradients. "
            "It is not wired into Circleworld runtime and does not prove learned RAFA-token behavior."
        ),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "relational_signature_learning_contract.json"
    md_path = out_dir / "RELATIONAL_SIGNATURE_LEARNING_CONTRACT.md"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_markdown(md_path, summary)
    return summary


def _write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Relational Signature Learning Contract",
        "",
        f"- schema: `{summary['schema']}`",
        f"- status: `{summary['status']}`",
        f"- device: `{summary['device']}`",
        f"- batch: `{summary['batch']}`",
        f"- hidden dim: `{summary['hidden_dim']}`",
        f"- train steps: `{summary['train_steps']}`",
        f"- learning rate: `{summary['learning_rate']}`",
        f"- runtime effect: `{summary['runtime_effect']}`",
        f"- claim effect: `{summary['claim_effect']}`",
        f"- initial total loss: `{summary['initial_total_loss']}`",
        f"- final total loss: `{summary['final_total_loss']}`",
        f"- loss reduction: `{summary['loss_reduction']}`",
        f"- gradient norm: `{summary['gradient_norm']}`",
        "",
        "## Checks",
        "",
    ]
    for name, value in summary.get("checks", {}).items():
        lines.append(f"- {name}: `{value}`")
    lines.extend(["", "## Contract", ""])
    contract = summary.get("contract", {})
    for name in (
        "h_shape",
        "q_profile_shape",
        "arc_profile_shape",
        "temporal_profile_shape",
        "support_profile_shape",
        "branch_profile_shape",
        "law_signature_shape",
        "operator_seed_shape",
        "operator_tail_shape",
        "law_operator_readout",
        "confidence_shape",
        "max_h_unit_norm_error",
        "max_law_unit_norm_error",
        "max_operator_unit_norm_error",
        "max_q_sum_error",
    ):
        lines.append(f"- {name}: `{contract.get(name)}`")
    lines.extend(["", "## Interpretation", "", summary["interpretation"]])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Smoke-test the learned RAFA relational signature encoder/head contract.")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch", type=int, default=5)
    ap.add_argument("--hidden-dim", type=int, default=96)
    ap.add_argument("--train-steps", type=int, default=24)
    ap.add_argument("--learning-rate", type=float, default=5.0e-3)
    args = ap.parse_args()
    summary = run_learning_contract(
        out_dir=Path(args.out_dir),
        device_name=str(args.device),
        batch=int(args.batch),
        hidden_dim=int(args.hidden_dim),
        train_steps=int(args.train_steps),
        learning_rate=float(args.learning_rate),
    )
    print(
        json.dumps(
            {
                "schema": summary["schema"],
                "saved": str(Path(args.out_dir) / "relational_signature_learning_contract.json"),
                "report": str(Path(args.out_dir) / "RELATIONAL_SIGNATURE_LEARNING_CONTRACT.md"),
                "status": summary["status"],
                "claim_effect": summary["claim_effect"],
                "checks": summary["checks"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
