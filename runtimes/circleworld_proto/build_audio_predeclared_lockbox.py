from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import wave
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WAV_ROOT = ROOT / "wav_files"
OUTPUT_JSON = "audio_predeclared_lockbox_manifest.json"
OUTPUT_MD = "AUDIO_PREDECLARED_LOCKBOX_MANIFEST.md"
OUTPUT_CASES_JSON = "audio_predeclared_lockbox_cases.json"
DEFAULT_SEED = "circleworld-audio-predeclared-lockbox-v1"
DEFAULT_MAGNITUDE_MODE = "prefix_hold"
DEFAULT_MASK_MODE = "all_bins"
DEFAULT_MECHANISM = "time_smooth_3"
DEFAULT_TARGET_GAIN = 2.0
DEFAULT_BASELINE_GAIN = 0.0
DEFAULT_PREFIX_SECONDS = 1.0
DEFAULT_FUTURE_SECONDS = 1.0


EMBEDDED_KEYWORD_SPECS: "OrderedDict[str, dict[str, Any]]" = OrderedDict(
    [
        (
            "generic_electric_motor",
            {
                "role": "predeclared_motor_probe",
                "min_cases": 1,
                "max_cases": 8,
                "include": ("electric motor", "electrical motor", "servo motor"),
                "exclude": ("steam engine", "motorcycle", "motorbike", "muscle car"),
            },
        ),
        (
            "hvac_fan_airflow_motor",
            {
                "role": "predeclared_motor_probe",
                "min_cases": 2,
                "max_cases": 8,
                "include": ("air conditioner", "air-conditioner", "fan startup", "fan motor"),
                "exclude": ("applause", "audience", "ceiling fan blade only"),
            },
        ),
        (
            "household_appliance_motor",
            {
                "role": "predeclared_motor_probe",
                "min_cases": 3,
                "max_cases": 10,
                "include": (
                    "can opener electric",
                    "electric toothbrush",
                    "dishwasher",
                    "clothes dryer",
                    "hair dryer",
                    "vacuum",
                    "air zoom vacuum",
                ),
                "exclude": ("electric razor", "razor"),
            },
        ),
        (
            "rotary_tool_motor",
            {
                "role": "predeclared_motor_probe",
                "min_cases": 4,
                "max_cases": 10,
                "include": (
                    "air wrench",
                    "air wrench short",
                    "drill",
                    "drilling",
                    "drill press",
                    "cordless drill",
                    "dentist drill",
                    "electric sander",
                    "metal grinder",
                ),
                "exclude": ("dropping a wrench", "razor"),
            },
        ),
        (
            "saw_chain_tool_motor",
            {
                "role": "predeclared_motor_probe",
                "min_cases": 4,
                "max_cases": 10,
                "include": (
                    "chain saw",
                    "chainsaw",
                    "chainsaw quick",
                    "buzz saw",
                    "buzzsaw",
                    "electric saw",
                    "electric table saw",
                    "table saw buzz",
                ),
                "exclude": ("hacksaw", "wood saw", "bone saw", "small bow saw", "sawing wood", "fast sawing"),
            },
        ),
        (
            "steady_buzz_nonmotor_control",
            {
                "role": "predeclared_nonmotor_control",
                "min_cases": 4,
                "max_cases": 10,
                "include": (
                    "fluorescent light humming",
                    "old electrical buzz alarm",
                    "electrical sweep",
                    "electricity",
                    "buzzer",
                    "buzz fade",
                    "door buzzer",
                    "industrial alarm",
                ),
                "exclude": (
                    "buzzard",
                    "fly buzzing",
                    "mosquito buzzing",
                    "electric razor",
                    "razor",
                    "synth",
                    "buzzy synth",
                    "buzzy pad",
                    "buzz saw",
                    "buzzsaw",
                    "chainsaw",
                ),
            },
        ),
        (
            "typewriter_nonmotor_control",
            {
                "role": "predeclared_nonmotor_control",
                "min_cases": 2,
                "max_cases": 8,
                "include": ("electric typewriter", "typewriter", "typewriter and bell", "typing on old typewriter"),
                "exclude": (),
            },
        ),
        (
            "combustion_engine_control",
            {
                "role": "predeclared_nonmotor_control",
                "min_cases": 6,
                "max_cases": 12,
                "include": ("motorcycle", "motorbike", "fast bike or motorcycle", "engine", "tractor", "muscle car"),
                "exclude": ("electric motor", "servo motor", "steam train", "steam engine"),
            },
        ),
        (
            "buzzy_synth_control",
            {
                "role": "predeclared_nonmotor_control",
                "min_cases": 3,
                "max_cases": 8,
                "include": ("buzzy", "buzzy pad", "buzzy synth", "synth lead", "saw lead", "sawtooth", "saw c", "saw f"),
                "exclude": ("buzz saw", "buzzsaw", "table saw", "chain saw", "chainsaw"),
            },
        ),
        (
            "razor_single_source_probe",
            {
                "role": "predeclared_single_source_probe",
                "min_cases": 1,
                "max_cases": 4,
                "include": ("electric razor", "razor"),
                "exclude": (),
            },
        ),
    ]
)


FORBIDDEN_SPEC_KEY_PARTS = (
    "accuracy",
    "auc",
    "baseline",
    "corr",
    "correlation",
    "delta",
    "future",
    "improvement",
    "loss",
    "metric",
    "objective",
    "outcome",
    "score",
    "target",
)


def _die(message: str) -> None:
    raise SystemExit(f"ERROR: {message}")


def _jsonify(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception as exc:  # pragma: no cover - argparse-facing error detail
        _die(f"failed to read JSON from {path}: {exc}")


def _as_string_tuple(value: Any, field: str, group_id: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, Sequence):
        _die(f"group {group_id!r} field {field!r} must be a string or list of strings")
    out: list[str] = []
    for item in value:
        if not isinstance(item, str):
            _die(f"group {group_id!r} field {field!r} contains a non-string value")
        clean = item.strip()
        if clean:
            out.append(clean)
    return tuple(out)


def _as_int(value: Any, field: str, group_id: str, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        _die(f"group {group_id!r} field {field!r} must be an integer")
    try:
        out = int(value)
    except Exception:
        _die(f"group {group_id!r} field {field!r} must be an integer")
    if out < 0:
        _die(f"group {group_id!r} field {field!r} must be non-negative")
    return out


def _check_forbidden_spec_keys(raw: Any, path: tuple[str, ...] = ()) -> None:
    if isinstance(raw, dict):
        for key, value in raw.items():
            lower_key = str(key).lower()
            if any(part in lower_key for part in FORBIDDEN_SPEC_KEY_PARTS):
                where = ".".join((*path, str(key)))
                _die(f"forbidden outcome/metric-like selector field in spec: {where}")
            _check_forbidden_spec_keys(value, (*path, str(key)))
    elif isinstance(raw, list):
        for index, value in enumerate(raw):
            _check_forbidden_spec_keys(value, (*path, str(index)))


def _coerce_specs(raw: Any, source: str) -> "OrderedDict[str, dict[str, Any]]":
    if isinstance(raw, dict) and "groups" in raw:
        groups_raw = raw["groups"]
    else:
        groups_raw = raw

    items: list[tuple[str, dict[str, Any]]] = []
    if isinstance(groups_raw, dict):
        for group_id, spec in groups_raw.items():
            if not isinstance(spec, dict):
                _die(f"spec for group {group_id!r} must be a JSON object")
            _check_forbidden_spec_keys(spec, ("groups", str(group_id)))
            merged = dict(spec)
            merged.setdefault("group_id", str(group_id))
            items.append((str(group_id), merged))
    elif isinstance(groups_raw, list):
        for index, spec in enumerate(groups_raw):
            if not isinstance(spec, dict):
                _die(f"spec entry {index} must be a JSON object")
            _check_forbidden_spec_keys(spec, ("groups", str(index)))
            group_id = str(spec.get("group_id") or spec.get("name") or "").strip()
            if not group_id:
                _die(f"spec entry {index} is missing group_id")
            items.append((group_id, dict(spec)))
    else:
        _die(f"{source} must be a group object, a list of group objects, or an object with a groups field")

    out: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
    for group_id, spec in items:
        clean_group_id = _safe_id(group_id)
        if clean_group_id in out:
            _die(f"duplicate group_id after normalization: {group_id!r}")
        include = _as_string_tuple(spec.get("include") or spec.get("keywords"), "include", clean_group_id)
        if not include:
            _die(f"group {clean_group_id!r} must declare at least one include keyword")
        exclude = _as_string_tuple(spec.get("exclude"), "exclude", clean_group_id)
        min_cases = _as_int(spec.get("min_cases"), "min_cases", clean_group_id, 0)
        max_cases = _as_int(spec.get("max_cases"), "max_cases", clean_group_id, 12)
        if max_cases < min_cases:
            _die(f"group {clean_group_id!r} max_cases is smaller than min_cases")
        out[clean_group_id] = {
            "group_id": clean_group_id,
            "role": str(spec.get("role") or "predeclared_filename_group"),
            "min_cases": min_cases,
            "max_cases": max_cases,
            "include": include,
            "exclude": exclude,
            "source": source,
            "notes": str(spec.get("notes") or ""),
        }
    return out


def _load_specs(path: Path | None) -> "OrderedDict[str, dict[str, Any]]":
    if path is None:
        return _coerce_specs({"groups": EMBEDDED_KEYWORD_SPECS}, "embedded")
    return _coerce_specs(_read_json(path), str(path))


def _safe_id(value: str) -> str:
    clean = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    if not clean:
        _die(f"cannot derive a safe identifier from {value!r}")
    return clean


def _tokens(value: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[a-z0-9]+", value.lower()))


def _normalized_text(value: str) -> str:
    return " ".join(_tokens(value))


def _contains_phrase(text: str, phrase: str) -> bool:
    clean = _normalized_text(phrase)
    if not clean:
        return False
    return f" {clean} " in f" {text} "


def _is_fingerprint_token(token: str) -> bool:
    if len(token) >= 6 and re.fullmatch(r"[a-f0-9]+", token):
        return True
    if len(token) >= 4 and token.isdigit():
        return True
    return False


def _provider_neutral_stem_key(path: Path) -> str:
    tokens = list(_tokens(path.stem))
    if not tokens:
        return _safe_id(path.stem)

    removed_trailing_fingerprint = False
    while tokens and _is_fingerprint_token(tokens[-1]):
        tokens.pop()
        removed_trailing_fingerprint = True

    # Many collected files are named source_title_hash.wav.  When a trailing
    # fingerprint exists, drop one leading source token without naming providers.
    if removed_trailing_fingerprint and len(tokens) >= 3 and tokens[0].isalpha():
        tokens = tokens[1:]

    if not tokens:
        tokens = list(_tokens(path.stem))
    return "_".join(tokens)


def _content_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _wav_header(path: Path) -> dict[str, int]:
    with wave.open(str(path), "rb") as handle:
        return {
            "sample_rate": int(handle.getframerate()),
            "channels": int(handle.getnchannels()),
            "sample_width_bytes": int(handle.getsampwidth()),
            "frame_count": int(handle.getnframes()),
        }


def _rank(seed: str, namespace: str, *parts: object) -> str:
    joined = "\n".join((seed, namespace, *(str(part) for part in parts)))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def _iter_wavs(roots: Sequence[Path], recursive: bool) -> Iterable[tuple[int, Path, Path]]:
    for root_index, root in enumerate(roots):
        pattern = "**/*.wav" if recursive else "*.wav"
        for path in root.glob(pattern):
            if path.is_file():
                yield root_index, root, path


def _scan_wavs(roots: Sequence[Path], recursive: bool, seed: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    missing_roots = [str(root) for root in roots if not root.exists()]
    if missing_roots:
        _die("WAV root does not exist: " + ", ".join(missing_roots))

    seen_paths: set[str] = set()
    rows: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for root_index, root, path in _iter_wavs(roots, recursive):
        resolved = path.resolve()
        resolved_key = str(resolved).lower()
        if resolved_key in seen_paths:
            continue
        seen_paths.add(resolved_key)
        try:
            rel_path = str(resolved.relative_to(root.resolve()))
        except ValueError:
            rel_path = resolved.name
        try:
            size_bytes = resolved.stat().st_size
            sha256 = _content_sha256(resolved)
            wav_header = _wav_header(resolved)
        except Exception as exc:
            rejected.append(
                {
                    "reason": "unreadable_or_invalid_wav_file",
                    "path": str(resolved),
                    "detail": str(exc),
                }
            )
            continue
        provider_neutral_key = _provider_neutral_stem_key(resolved)
        row = {
            "path": str(resolved),
            "root": str(root.resolve()),
            "root_index": root_index,
            "relative_path": rel_path,
            "filename": resolved.name,
            "filename_text": _normalized_text(resolved.stem),
            "provider_neutral_stem_key": provider_neutral_key,
            "content_sha256": sha256,
            "size_bytes": size_bytes,
            "sample_rate": wav_header["sample_rate"],
            "channels": wav_header["channels"],
            "sample_width_bytes": wav_header["sample_width_bytes"],
            "frame_count": wav_header["frame_count"],
            "scan_rank": _rank(seed, "scan", provider_neutral_key, sha256, str(resolved).lower()),
        }
        rows.append(row)

    rows.sort(key=lambda row: (row["scan_rank"], row["path"].lower()))
    return rows, rejected


def _dedupe_rows(rows: Sequence[dict[str, Any]], seed: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rejected: list[dict[str, Any]] = []

    content_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        content_groups[str(row["content_sha256"])].append(dict(row))

    content_survivors: list[dict[str, Any]] = []
    for sha256, group in sorted(content_groups.items()):
        ranked = sorted(group, key=lambda row: (_rank(seed, "content_dedupe", sha256, row["path"]), row["path"].lower()))
        keeper = ranked[0]
        content_survivors.append(keeper)
        for duplicate in ranked[1:]:
            rejected.append(
                {
                    "reason": "duplicate_content_hash",
                    "path": duplicate["path"],
                    "duplicate_of": keeper["path"],
                    "content_sha256": sha256,
                    "provider_neutral_stem_key": duplicate["provider_neutral_stem_key"],
                }
            )

    stem_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in content_survivors:
        stem_groups[str(row["provider_neutral_stem_key"])].append(row)

    stem_survivors: list[dict[str, Any]] = []
    for stem_key, group in sorted(stem_groups.items()):
        ranked = sorted(
            group,
            key=lambda row: (_rank(seed, "provider_neutral_stem_dedupe", stem_key, row["content_sha256"], row["path"]), row["path"].lower()),
        )
        keeper = ranked[0]
        stem_survivors.append(keeper)
        for duplicate in ranked[1:]:
            rejected.append(
                {
                    "reason": "duplicate_provider_neutral_stem",
                    "path": duplicate["path"],
                    "duplicate_of": keeper["path"],
                    "content_sha256": duplicate["content_sha256"],
                    "provider_neutral_stem_key": stem_key,
                }
            )

    stem_survivors.sort(key=lambda row: (row["scan_rank"], row["path"].lower()))
    rejected.sort(key=lambda row: (str(row.get("reason", "")), str(row.get("provider_neutral_stem_key", "")), str(row.get("path", "")).lower()))
    return stem_survivors, rejected


def _matching_include(row: dict[str, Any], spec: dict[str, Any]) -> str | None:
    for keyword in spec["include"]:
        if _contains_phrase(str(row["filename_text"]), str(keyword)):
            return str(keyword)
    return None


def _matching_exclude(row: dict[str, Any], spec: dict[str, Any]) -> str | None:
    for keyword in spec["exclude"]:
        if _contains_phrase(str(row["filename_text"]), str(keyword)):
            return str(keyword)
    return None


def _select_groups(
    rows: Sequence[dict[str, Any]],
    specs: "OrderedDict[str, dict[str, Any]]",
    seed: str,
) -> tuple["OrderedDict[str, dict[str, Any]]", list[dict[str, Any]]]:
    selected_paths: set[str] = set()
    path_to_group: dict[str, str] = {}
    rejected: list[dict[str, Any]] = []
    groups: "OrderedDict[str, dict[str, Any]]" = OrderedDict()

    for group_id, spec in specs.items():
        candidates: list[dict[str, Any]] = []
        excluded_matches: list[dict[str, Any]] = []
        for row in rows:
            include_keyword = _matching_include(row, spec)
            if include_keyword is None:
                continue
            exclude_keyword = _matching_exclude(row, spec)
            if exclude_keyword is not None:
                excluded_matches.append(
                    {
                        "reason": "keyword_excluded",
                        "group_id": group_id,
                        "path": row["path"],
                        "matched_include": include_keyword,
                        "matched_exclude": exclude_keyword,
                        "provider_neutral_stem_key": row["provider_neutral_stem_key"],
                        "content_sha256": row["content_sha256"],
                    }
                )
                continue
            candidate = dict(row)
            candidate["matched_include"] = include_keyword
            candidate["selection_rank"] = _rank(
                seed,
                "group_selection",
                group_id,
                include_keyword,
                row["provider_neutral_stem_key"],
                row["content_sha256"],
                row["path"],
            )
            candidates.append(candidate)

        candidates.sort(key=lambda row: (row["selection_rank"], row["path"].lower()))
        selected: list[dict[str, Any]] = []
        for candidate in candidates:
            path = str(candidate["path"])
            if path in selected_paths:
                rejected.append(
                    {
                        "reason": "group_disjointness_conflict",
                        "group_id": group_id,
                        "path": path,
                        "already_selected_by": path_to_group[path],
                        "matched_include": candidate["matched_include"],
                        "provider_neutral_stem_key": candidate["provider_neutral_stem_key"],
                        "content_sha256": candidate["content_sha256"],
                    }
                )
                continue
            if len(selected) >= int(spec["max_cases"]):
                rejected.append(
                    {
                        "reason": "group_capacity_reached",
                        "group_id": group_id,
                        "path": path,
                        "matched_include": candidate["matched_include"],
                        "provider_neutral_stem_key": candidate["provider_neutral_stem_key"],
                        "content_sha256": candidate["content_sha256"],
                    }
                )
                continue
            selected_paths.add(path)
            path_to_group[path] = group_id
            selected.append(candidate)

        cases = []
        for index, row in enumerate(selected, start=1):
            case_slug = _safe_id(str(row["provider_neutral_stem_key"]))[:72]
            cases.append(
                {
                    "case_id": f"{group_id}__{index:03d}__{case_slug}",
                    "path": row["path"],
                    "root": row["root"],
                    "relative_path": row["relative_path"],
                    "filename": row["filename"],
                    "provider_neutral_stem_key": row["provider_neutral_stem_key"],
                    "content_sha256": row["content_sha256"],
                    "content_sha256_prefix": str(row["content_sha256"])[:16],
                    "size_bytes": row["size_bytes"],
                    "sample_rate": row["sample_rate"],
                    "channels": row["channels"],
                    "sample_width_bytes": row["sample_width_bytes"],
                    "frame_count": row["frame_count"],
                    "matched_include": row["matched_include"],
                    "selection_rank": row["selection_rank"],
                }
            )

        groups[group_id] = {
            "group_id": group_id,
            "role": spec["role"],
            "min_cases": spec["min_cases"],
            "max_cases": spec["max_cases"],
            "include": list(spec["include"]),
            "exclude": list(spec["exclude"]),
            "notes": spec["notes"],
            "candidate_count_before_disjoint_selection": len(candidates),
            "excluded_candidate_count": len(excluded_matches),
            "selected_count": len(cases),
            "met_min_cases": len(cases) >= int(spec["min_cases"]),
            "cases": cases,
        }
        rejected.extend(excluded_matches)

    rejected.sort(key=lambda row: (str(row.get("reason", "")), str(row.get("group_id", "")), str(row.get("path", "")).lower()))
    return groups, rejected


def _write_markdown(path: Path, manifest: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# Audio Predeclared Lockbox Manifest")
    lines.append("")
    lines.append("This manifest freezes case selection only. It does not run the audio mechanism and does not select on target, future, baseline, correlation, or outcome metrics.")
    lines.append("")
    lines.append("## Build Inputs")
    lines.append("")
    lines.append(f"- Schema: `{manifest['schema']}`")
    lines.append(f"- Seed: `{manifest['determinism']['seed']}`")
    lines.append(f"- Recursive scan: `{manifest['scan']['recursive']}`")
    lines.append(f"- WAV roots: `{len(manifest['scan']['wav_roots'])}`")
    lines.append(f"- Scanned WAV files: `{manifest['scan']['scanned_wav_count']}`")
    lines.append(f"- Usable after dedupe: `{manifest['dedupe']['usable_wav_count']}`")
    lines.append(f"- Duplicate/content/read rejections: `{manifest['dedupe']['rejected_count']}`")
    lines.append(f"- Selected cases: `{manifest['selected_case_count']}`")
    lines.append(f"- Probe cases JSON: `{manifest.get('cases_json_path')}`")
    lines.append("")
    lines.append("## Frozen Probe Row")
    lines.append("")
    row = manifest["predeclared_row"]
    lines.append(f"- Magnitude mode: `{row['magnitude_mode']}`")
    lines.append(f"- Mask mode: `{row['mask_mode']}`")
    lines.append(f"- Mechanism: `{row['mechanism']}`")
    lines.append(f"- Baseline gain: `{row['baseline_gain']}`")
    lines.append(f"- Target gain: `{row['gain']}`")
    lines.append(f"- Prefix seconds: `{manifest['timing']['prefix_seconds']}`")
    lines.append(f"- Future seconds: `{manifest['timing']['future_seconds']}`")
    lines.append("")
    lines.append("## Guardrails")
    lines.append("")
    for key, value in manifest["guardrails"].items():
        lines.append(f"- {key}: `{value}`")
    lines.append("")
    lines.append("## Groups")
    lines.append("")
    lines.append("| Group | Role | Selected | Min | Max | Met Min |")
    lines.append("| --- | --- | ---: | ---: | ---: | --- |")
    for group in manifest["groups"].values():
        lines.append(
            f"| `{group['group_id']}` | `{group['role']}` | {group['selected_count']} | {group['min_cases']} | {group['max_cases']} | `{group['met_min_cases']}` |"
        )
    lines.append("")
    lines.append("## Selected Cases")
    for group in manifest["groups"].values():
        lines.append("")
        lines.append(f"### {group['group_id']}")
        if not group["cases"]:
            lines.append("")
            lines.append("_No cases selected._")
            continue
        lines.append("")
        lines.append("| Case | Include | Stem Key | SHA256 Prefix | Path |")
        lines.append("| --- | --- | --- | --- | --- |")
        for case in group["cases"]:
            lines.append(
                f"| `{case['case_id']}` | `{case['matched_include']}` | `{case['provider_neutral_stem_key']}` | `{case['content_sha256_prefix']}` | `{case['path']}` |"
            )
    lines.append("")
    lines.append("## Rejections")
    lines.append("")
    lines.append(f"- Read/content/stem duplicate rejections: `{len(manifest['rejected_duplicates'])}`")
    lines.append(f"- Group candidate rejections: `{len(manifest['rejected_group_candidates'])}`")
    lines.append("")
    if manifest["unmet_min_case_groups"]:
        lines.append("## Unmet Minimums")
        lines.append("")
        for group_id in manifest["unmet_min_case_groups"]:
            group = manifest["groups"][group_id]
            lines.append(f"- `{group_id}` selected `{group['selected_count']}` of minimum `{group['min_cases']}`")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    specs = _load_specs(args.spec_json)
    roots = [Path(item).expanduser().resolve() for item in args.wav_root]
    scanned_rows, read_rejections = _scan_wavs(roots, recursive=not args.no_recursive, seed=args.seed)
    deduped_rows, duplicate_rejections = _dedupe_rows(scanned_rows, args.seed)
    groups, group_rejections = _select_groups(deduped_rows, specs, args.seed)
    unmet = [group_id for group_id, group in groups.items() if not group["met_min_cases"]]

    selected_hashes: set[str] = set()
    selected_stems: set[str] = set()
    selected_paths: set[str] = set()
    selected_cases: list[dict[str, Any]] = []
    for group in groups.values():
        for case in group["cases"]:
            if case["path"] in selected_paths:
                _die(f"internal disjointness failure for path {case['path']}")
            if case["content_sha256"] in selected_hashes:
                _die(f"internal content-hash uniqueness failure for {case['path']}")
            if case["provider_neutral_stem_key"] in selected_stems:
                _die(f"internal stem uniqueness failure for {case['path']}")
            selected_paths.add(case["path"])
            selected_hashes.add(case["content_sha256"])
            selected_stems.add(case["provider_neutral_stem_key"])
            selected_cases.append(
                {
                    **case,
                    "group": group["group_id"],
                    "role": group["role"],
                }
            )

    if unmet and args.strict_min_cases:
        _die("minimum case requirement not met for: " + ", ".join(unmet))

    rejected_duplicates = [*read_rejections, *duplicate_rejections]
    manifest = {
        "schema": "circleworld.audio_predeclared_lockbox_manifest.v1",
        "builder": {
            "script": str(Path(__file__).resolve()),
            "purpose": "Freeze filename-derived audio case selection before running any audio mechanism.",
            "timestamp_policy": "omitted_for_reproducibility",
        },
        "determinism": {
            "seed": args.seed,
            "rank_function": "sha256(seed, namespace, filename-derived keys, content hash, path)",
            "python_hash_randomization_used": False,
        },
        "guardrails": {
            "audio_mechanism_executed": False,
            "selection_uses_audio_content_features": False,
            "selection_uses_target_future_metrics": False,
            "selection_uses_baseline_metrics": False,
            "selection_uses_correlation_metrics": False,
            "selection_uses_filename_keywords": True,
            "selection_requires_riff_readable_wav": True,
            "selection_enforces_disjoint_groups": True,
            "selection_enforces_content_hash_uniqueness": True,
            "selection_enforces_provider_neutral_stem_dedupe": True,
            "probe_row_declared_before_execution": True,
        },
        "predeclared_row": {
            "magnitude_mode": args.magnitude_mode,
            "mask_mode": args.mask_mode,
            "mechanism": args.mechanism,
            "baseline_gain": float(args.baseline_gain),
            "gain": float(args.target_gain),
            "status": "predeclared_filename_lockbox_row",
        },
        "timing": {
            "prefix_seconds": float(args.prefix_seconds),
            "future_seconds": float(args.future_seconds),
        },
        "scan": {
            "wav_roots": [str(root) for root in roots],
            "recursive": not args.no_recursive,
            "scanned_wav_count": len(scanned_rows),
        },
        "dedupe": {
            "usable_wav_count": len(deduped_rows),
            "rejected_count": len(rejected_duplicates),
            "rejected_read_or_content_duplicate_count": len(read_rejections) + sum(1 for row in duplicate_rejections if row["reason"] == "duplicate_content_hash"),
            "rejected_provider_neutral_stem_duplicate_count": sum(
                1 for row in duplicate_rejections if row["reason"] == "duplicate_provider_neutral_stem"
            ),
        },
        "specs": {
            "source": str(args.spec_json) if args.spec_json else "embedded",
            "group_count": len(specs),
            "forbidden_spec_key_parts": list(FORBIDDEN_SPEC_KEY_PARTS),
        },
        "groups": _jsonify(groups),
        "cases": _jsonify(selected_cases),
        "selected_case_count": len(selected_cases),
        "selected_unique_paths": len(selected_paths),
        "selected_unique_content_hashes": len(selected_hashes),
        "selected_unique_provider_neutral_stems": len(selected_stems),
        "unmet_min_case_groups": unmet,
        "rejected_duplicates": _jsonify(rejected_duplicates),
        "rejected_group_candidates": _jsonify(group_rejections),
    }
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a predeclared Circleworld audio lockbox manifest from WAV filenames. "
            "The builder freezes case selection only and never runs the audio mechanism."
        )
    )
    parser.add_argument(
        "--wav-root",
        action="append",
        default=None,
        help="Root containing WAV files. May be repeated. Defaults to D:\\RAFA\\wav_files when run from this tree.",
    )
    parser.add_argument("--out-dir", required=True, type=Path, help="Directory for manifest JSON and Markdown summary.")
    parser.add_argument("--spec-json", type=Path, default=None, help="Optional keyword spec JSON. Replaces embedded specs when supplied.")
    parser.add_argument("--seed", default=DEFAULT_SEED, help="Deterministic selection seed.")
    parser.add_argument("--magnitude-mode", default=DEFAULT_MAGNITUDE_MODE, help="Frozen probe magnitude mode recorded in the manifest.")
    parser.add_argument("--mask-mode", default=DEFAULT_MASK_MODE, help="Frozen probe mask mode recorded in the manifest.")
    parser.add_argument("--mechanism", default=DEFAULT_MECHANISM, help="Frozen probe mechanism recorded in the manifest.")
    parser.add_argument("--target-gain", type=float, default=DEFAULT_TARGET_GAIN, help="Frozen nonbaseline gain recorded in the manifest.")
    parser.add_argument("--baseline-gain", type=float, default=DEFAULT_BASELINE_GAIN, help="Frozen baseline gain recorded in the manifest.")
    parser.add_argument("--prefix-seconds", type=float, default=DEFAULT_PREFIX_SECONDS, help="Prefix seconds recorded for the lockbox run.")
    parser.add_argument("--future-seconds", type=float, default=DEFAULT_FUTURE_SECONDS, help="Future seconds recorded for the lockbox run.")
    parser.add_argument("--no-recursive", action="store_true", help="Scan only direct WAV children of each --wav-root.")
    parser.add_argument("--strict-min-cases", action="store_true", help="Exit non-zero when any group selects fewer than min_cases.")
    args = parser.parse_args(argv)
    if args.wav_root is None:
        args.wav_root = [DEFAULT_WAV_ROOT]
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    manifest = build_manifest(args)
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / OUTPUT_JSON
    md_path = out_dir / OUTPUT_MD
    cases_path = out_dir / OUTPUT_CASES_JSON
    case_map = {str(case["case_id"]): str(case["path"]) for case in manifest["cases"]}
    manifest["cases_json_path"] = str(cases_path)
    manifest["cases_json_schema"] = "name_to_wav_path_for_run_audio_delta_mechanism_probe"
    cases_path.write_text(json.dumps(case_map, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    json_path.write_text(json.dumps(manifest, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    _write_markdown(md_path, manifest)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(f"Wrote {cases_path}")
    if manifest["unmet_min_case_groups"]:
        print("Unmet min-case groups: " + ", ".join(manifest["unmet_min_case_groups"]))


if __name__ == "__main__":
    main()
