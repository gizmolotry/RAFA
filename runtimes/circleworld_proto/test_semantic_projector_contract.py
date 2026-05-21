from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

SCHEMA = "rafa_semantic_projector_contract_v0"
EXPECTED_CONTROL_NAMES: tuple[str, ...] = (
    "harmonic_coupling",
    "decay_rate",
    "noise_injection",
    "branch_temperature",
    "support_spread",
)
PROMPT_FAMILIES: tuple[str, ...] = ("airplane", "reed", "bell", "impact")
LOCAL_PROMPT_TEXT: dict[str, str] = {
    "airplane": "broad engine surge, sustained thrust, noisy aerodynamic wash",
    "reed": "stable reed excitation, sustained tone, moderate noise, narrow support",
    "bell": "bright struck resonance, high harmonic coupling, long decay tail",
    "impact": "sharp transient hit, fast decay, wide initial support, low sustain",
}
CONTROL_BOUNDS: tuple[float, float] = (0.0, 1.0)
DIVERSITY_EPSILON = 1.0e-6
DEFAULT_EMBEDDING_DIM = 768


@dataclass(frozen=True)
class ImportResult:
    module: ModuleType | None
    path: Path
    ok: bool
    detail: str


@dataclass(frozen=True)
class ProjectionPath:
    name: str
    status: str
    detail: str
    project: Callable[[str, str], dict[str, float]]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_projector_path() -> Path:
    return _repo_root() / "lineages" / "04_positive_replacement" / "semantic_projector.py"


def _import_semantic_projector(path: Path) -> ImportResult:
    if not path.exists():
        return ImportResult(None, path, False, f"not found: {path}")
    try:
        spec = importlib.util.spec_from_file_location("rafa_semantic_projector_contract_target", path)
        if spec is None or spec.loader is None:
            return ImportResult(None, path, False, f"could not build import spec for {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return ImportResult(module, path, True, "imported")
    except Exception as exc:  # pragma: no cover - exercised by local environment state.
        return ImportResult(None, path, False, f"import failed: {type(exc).__name__}: {exc}")


def _module_contract(module: ModuleType | None) -> dict[str, Any]:
    if module is None:
        return {
            "control_dim": len(EXPECTED_CONTROL_NAMES),
            "control_names": list(EXPECTED_CONTROL_NAMES),
            "allowed_prompt_families": list(PROMPT_FAMILIES),
            "prompt_family_text": dict(LOCAL_PROMPT_TEXT),
            "source": "local_expected_contract",
        }

    if hasattr(module, "projector_contract"):
        try:
            contract = module.projector_contract()
            if isinstance(contract, dict):
                out = dict(contract)
                out["source"] = "semantic_projector.projector_contract"
                return out
        except Exception as exc:
            return {
                "control_dim": None,
                "control_names": [],
                "allowed_prompt_families": [],
                "prompt_family_text": {},
                "source": f"projector_contract_failed:{type(exc).__name__}:{exc}",
            }

    cfg_cls = getattr(module, "SemanticProjectorConfig", None)
    if cfg_cls is None:
        return {
            "control_dim": None,
            "control_names": [],
            "allowed_prompt_families": [],
            "prompt_family_text": {},
            "source": "semantic_projector_without_contract_or_config",
        }

    try:
        cfg = cfg_cls()
        return {
            "control_dim": int(getattr(cfg, "control_dim", -1)),
            "control_names": list(getattr(cfg, "control_names", ())),
            "allowed_prompt_families": list(getattr(module, "ALLOWED_PROMPT_FAMILIES", ())),
            "prompt_family_text": dict(getattr(module, "PROMPT_FAMILY_TEXT", {})),
            "source": "semantic_projector.SemanticProjectorConfig",
        }
    except Exception as exc:
        return {
            "control_dim": None,
            "control_names": [],
            "allowed_prompt_families": [],
            "prompt_family_text": {},
            "source": f"config_failed:{type(exc).__name__}:{exc}",
        }


def _require_exact_schema(contract: dict[str, Any]) -> dict[str, Any]:
    control_names = tuple(str(name) for name in contract.get("control_names", ()))
    control_dim = contract.get("control_dim")
    expected = list(EXPECTED_CONTROL_NAMES)
    checks = {
        "control_dim_is_5": control_dim == 5,
        "control_names_exact_order": control_names == EXPECTED_CONTROL_NAMES,
        "control_names_exact_set": set(control_names) == set(EXPECTED_CONTROL_NAMES),
    }
    if not all(checks.values()):
        raise AssertionError(
            "semantic projector control schema mismatch: "
            f"expected {expected}, got control_dim={control_dim!r}, control_names={list(control_names)!r}"
        )
    return checks


def _prompt_texts(contract: dict[str, Any]) -> dict[str, str]:
    raw_texts = contract.get("prompt_family_text", {})
    texts = dict(LOCAL_PROMPT_TEXT)
    if isinstance(raw_texts, dict):
        for family in PROMPT_FAMILIES:
            value = raw_texts.get(family)
            if isinstance(value, str) and value.strip():
                texts[family] = value.strip()
    return texts


def _stable_unit(seed: str) -> float:
    digest = hashlib.sha256(seed.encode("utf-8")).digest()
    integer = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return integer / float(2**64 - 1)


def _stable_signed(seed: str) -> float:
    return (_stable_unit(seed) * 2.0) - 1.0


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _mock_embedding(prompt_family: str, prompt_text: str, dim: int) -> list[float]:
    family_bias = {
        "airplane": (0.75, 0.40, 0.85, 0.60, 0.78),
        "reed": (0.48, 0.54, 0.28, 0.36, 0.26),
        "bell": (0.90, 0.82, 0.18, 0.46, 0.42),
        "impact": (0.34, 0.88, 0.42, 0.72, 0.68),
    }[prompt_family]
    base: list[float] = []
    for index in range(dim):
        feature = (family_bias[index % len(family_bias)] * 2.0) - 1.0
        jitter = 0.18 * _stable_signed(f"embedding:{prompt_family}:{prompt_text}:{index}")
        base.append(feature + jitter)
    norm = math.sqrt(sum(value * value for value in base)) or 1.0
    return [value / norm for value in base]


def _mock_weight(control_name: str, dim: int) -> list[float]:
    row = [_stable_signed(f"weight:{control_name}:{index}") for index in range(dim)]
    norm = math.sqrt(sum(value * value for value in row)) or 1.0
    return [value / norm for value in row]


def _deterministic_mock_project(prompt_family: str, prompt_text: str, dim: int = DEFAULT_EMBEDDING_DIM) -> dict[str, float]:
    embedding = _mock_embedding(prompt_family, prompt_text, dim)
    controls: dict[str, float] = {}
    for control_name in EXPECTED_CONTROL_NAMES:
        weight = _mock_weight(control_name, dim)
        dot = sum(left * right for left, right in zip(weight, embedding))
        calibration = _stable_signed(f"bias:{prompt_family}:{control_name}") * 0.25
        controls[control_name] = _sigmoid((2.35 * dot) + calibration)
    return controls


def _callable_requires_weight(callable_obj: Callable[..., Any]) -> bool:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return False
    parameter = signature.parameters.get("weight")
    if parameter is None:
        return False
    return parameter.default is inspect.Parameter.empty


def _extract_profile_dict(profile: Any) -> dict[str, float]:
    if hasattr(profile, "to_boundary_conditions"):
        profile = profile.to_boundary_conditions()
    elif hasattr(profile, "to_dict"):
        profile = profile.to_dict()
    if not isinstance(profile, dict):
        raise TypeError(f"projector returned unsupported profile type {type(profile).__name__}")
    return {name: float(profile[name]) for name in EXPECTED_CONTROL_NAMES}


def _select_projection_path(module: ModuleType | None) -> ProjectionPath:
    if module is None:
        return ProjectionPath(
            name="deterministic_local_mock",
            status="contract_smoke_only",
            detail="semantic_projector unavailable; using deterministic local mock vectors and projector weights",
            project=lambda family, text: _deterministic_mock_project(family, text),
        )

    prompt_projector = getattr(module, "project_prompt_family_to_controls", None)
    if callable(prompt_projector):
        def project_with_module(family: str, text: str) -> dict[str, float]:
            try:
                return _extract_profile_dict(prompt_projector(family))
            except TypeError:
                return _extract_profile_dict(prompt_projector(prompt_family=family, prompt_text=text))

        return ProjectionPath(
            name="semantic_projector.project_prompt_family_to_controls",
            status="semantic_proven",
            detail="semantic_projector exposes a prompt-family projection path without caller-provided weights",
            project=project_with_module,
        )

    candidate = getattr(module, "project_factors_to_controls", None)
    if callable(candidate) and _callable_requires_weight(candidate):
        return ProjectionPath(
            name="deterministic_local_mock",
            status="contract_smoke_only",
            detail="semantic_projector projection requires explicit weights; using deterministic local mock vectors and projector weights",
            project=lambda family, text: _deterministic_mock_project(family, text),
        )

    return ProjectionPath(
        name="deterministic_local_mock",
        status="contract_smoke_only",
        detail="no no-weight prompt projection path found; using deterministic local mock vectors and projector weights",
        project=lambda family, text: _deterministic_mock_project(family, text),
    )


def _validate_profile(profile: dict[str, float], prompt_family: str) -> dict[str, bool]:
    low, high = CONTROL_BOUNDS
    checks = {
        "has_exact_controls": tuple(profile.keys()) == EXPECTED_CONTROL_NAMES,
        "all_numeric": all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in profile.values()),
        "all_finite": all(math.isfinite(float(value)) for value in profile.values()),
        "all_bounded_0_1": all(low <= float(value) <= high for value in profile.values()),
    }
    if not all(checks.values()):
        raise AssertionError(f"invalid control profile for {prompt_family}: checks={checks}, profile={profile}")
    return checks


def _pairwise_diversity(outputs: dict[str, dict[str, float]]) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for left, right in combinations(PROMPT_FAMILIES, 2):
        distance = sum(abs(outputs[left][name] - outputs[right][name]) for name in EXPECTED_CONTROL_NAMES)
        diverse = distance > DIVERSITY_EPSILON
        if not diverse:
            raise AssertionError(f"prompt families {left!r} and {right!r} are not diverse: l1={distance}")
        pairs.append({"left": left, "right": right, "l1_distance": distance, "diverse": diverse})
    return pairs


def _json_ready(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 10)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    artifact_checks = payload.get("checks", {}).get("artifacts", {})
    lines = [
        "# RAFA Semantic Projector Contract",
        "",
        f"Schema: `{payload['schema']}`",
        f"Status: `{payload['status']}`",
        f"Projection path: `{payload['projection_path']['name']}`",
        "",
        "## Summary",
        "",
        f"- Imported projector: `{payload['semantic_projector']['import_ok']}`",
        f"- Projector detail: `{payload['semantic_projector']['detail']}`",
        f"- Contract source: `{payload['contract']['source']}`",
        f"- Control schema exact: `{payload['checks']['schema']['control_names_exact_order']}`",
        f"- Prompt families: `{', '.join(PROMPT_FAMILIES)}`",
        f"- Artifact readback: `{artifact_checks.get('json_readback_ok', 'pending')}`",
        "",
        "## Control Schema",
        "",
    ]
    lines.extend(f"- `{name}`" for name in EXPECTED_CONTROL_NAMES)
    lines.extend(["", "## Prompt Outputs", ""])
    header = "| prompt_family | " + " | ".join(EXPECTED_CONTROL_NAMES) + " |"
    separator = "|---" * (len(EXPECTED_CONTROL_NAMES) + 1) + "|"
    lines.extend([header, separator])
    for family in PROMPT_FAMILIES:
        profile = payload["outputs"][family]
        values = " | ".join(f"{profile[name]:.6f}" for name in EXPECTED_CONTROL_NAMES)
        lines.append(f"| {family} | {values} |")
    lines.extend(["", "## Pairwise Diversity", ""])
    lines.append("| left | right | l1_distance | diverse |")
    lines.append("|---|---|---:|---|")
    for pair in payload["checks"]["pairwise_diversity"]:
        lines.append(
            f"| {pair['left']} | {pair['right']} | {pair['l1_distance']:.10f} | {pair['diverse']} |"
        )
    lines.extend(["", "## Interpretation", ""])
    if payload["status"] == "semantic_proven":
        lines.append("The audit used a projector path that did not require caller-provided mock weights.")
    else:
        lines.append(
            "This is a contract smoke test only: the current projector path requires weights or is unavailable, "
            "so deterministic local mock vectors and projector weights were used."
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_audit(projector_path: Path, output_dir: Path) -> dict[str, Any]:
    import_result = _import_semantic_projector(projector_path)
    contract = _module_contract(import_result.module)
    schema_checks = _require_exact_schema(contract)
    projection_path = _select_projection_path(import_result.module)
    prompt_texts = _prompt_texts(contract)

    outputs: dict[str, dict[str, float]] = {}
    profile_checks: dict[str, dict[str, bool]] = {}
    for family in PROMPT_FAMILIES:
        profile = projection_path.project(family, prompt_texts[family])
        normalized = {name: float(profile[name]) for name in EXPECTED_CONTROL_NAMES}
        profile_checks[family] = _validate_profile(normalized, family)
        outputs[family] = normalized

    diversity = _pairwise_diversity(outputs)

    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "status": projection_path.status,
        "semantic_projector": {
            "path": str(projector_path),
            "import_ok": import_result.ok,
            "detail": import_result.detail,
        },
        "contract": {
            "source": contract.get("source"),
            "control_dim": contract.get("control_dim"),
            "control_names": list(contract.get("control_names", [])),
            "allowed_prompt_families": list(contract.get("allowed_prompt_families", [])),
        },
        "projection_path": {
            "name": projection_path.name,
            "detail": projection_path.detail,
        },
        "prompt_text": prompt_texts,
        "outputs": outputs,
        "checks": {
            "schema": schema_checks,
            "profiles": profile_checks,
            "pairwise_diversity": diversity,
            "artifacts": {},
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "semantic_projector_contract.json"
    markdown_path = output_dir / "SEMANTIC_PROJECTOR_CONTRACT.md"
    _write_json(json_path, payload)
    _write_markdown(markdown_path, payload)

    json_readback = json.loads(json_path.read_text(encoding="utf-8"))
    artifact_checks = {
        "json_path": str(json_path),
        "markdown_path": str(markdown_path),
        "json_exists": json_path.exists(),
        "markdown_exists": markdown_path.exists(),
        "json_readback_ok": json_readback.get("schema") == SCHEMA,
        "markdown_has_schema": SCHEMA in markdown_path.read_text(encoding="utf-8"),
    }
    if not all(value for key, value in artifact_checks.items() if key.endswith("exists") or key.endswith("ok") or key.endswith("schema")):
        raise AssertionError(f"artifact generation failed: {artifact_checks}")
    payload["checks"]["artifacts"] = artifact_checks
    _write_json(json_path, payload)
    _write_markdown(markdown_path, payload)
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit the RAFA semantic projector control-variable contract.")
    parser.add_argument(
        "--projector-path",
        type=Path,
        default=_default_projector_path(),
        help="Path to semantic_projector.py. Defaults to the RAFA lineage implementation.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory for semantic_projector_contract.json and SEMANTIC_PROJECTOR_CONTRACT.md.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = run_audit(args.projector_path.resolve(), args.output_dir.resolve())
    print(json.dumps({
        "schema": payload["schema"],
        "status": payload["status"],
        "json_path": payload["checks"]["artifacts"]["json_path"],
        "markdown_path": payload["checks"]["artifacts"]["markdown_path"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
