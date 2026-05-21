from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from common_io import as_float as _as_float
from common_io import fmt as _fmt
from common_io import load_json as _load_json
from common_io import mean as _mean
from common_io import median as _median

OUTPUT_JSON = "internal_phase_law_variant_compare.json"
OUTPUT_MD = "INTERNAL_PHASE_LAW_VARIANT_COMPARE.md"


def _norm_path(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    try:
        return str(Path(raw).resolve()).lower()
    except (OSError, RuntimeError):
        return raw.replace("/", "\\").lower()


def _case_key(row: dict[str, Any]) -> str:
    path = _norm_path(row.get("source_wav") or row.get("path") or row.get("wav_path"))
    return path if path else str(row.get("name") or row.get("case_name") or "")


def _row_key(row: dict[str, Any]) -> tuple[str, str, str, float]:
    return (
        str(row.get("magnitude_mode", "")),
        str(row.get("mask_mode", "")),
        str(row.get("mechanism", "")),
        float(_as_float(row.get("gain"))),
    )


def _parse_labeled_path(raw: str) -> tuple[str, Path]:
    if "=" in raw:
        label, path = raw.split("=", 1)
        label = label.strip()
        return (label if label else Path(path).stem), Path(path)
    path = Path(raw)
    return path.stem, path


def _iter_result_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cases = payload.get("cases")
    if isinstance(cases, list):
        for case in cases:
            if not isinstance(case, dict):
                continue
            meta = {
                "name": case.get("name"),
                "source_wav": case.get("source_wav"),
                "sample_rate": case.get("sample_rate"),
                "prefix_samples": case.get("prefix_samples"),
                "future_samples": case.get("future_samples"),
            }
            for row in case.get("rows", []) or []:
                if isinstance(row, dict):
                    merged = dict(meta)
                    merged.update(row)
                    rows.append(merged)
    if rows:
        return rows
    flat = payload.get("rows")
    if isinstance(flat, list):
        return [dict(row) for row in flat if isinstance(row, dict)]
    return []


def _manifest_case_meta(manifest: dict[str, Any] | None) -> dict[str, dict[str, str]]:
    if not isinstance(manifest, dict):
        return {}
    out: dict[str, dict[str, str]] = {}

    def add(case: dict[str, Any], *, group: str | None = None, role: str | None = None) -> None:
        meta = {
            "group": str(case.get("group") or case.get("family") or group or ""),
            "role": str(case.get("role") or role or ""),
        }
        for value in (case.get("source_wav"), case.get("path"), case.get("wav_path"), case.get("name")):
            key = _norm_path(value) if value and Path(str(value)).suffix else str(value or "")
            if key:
                out[key] = {k: v for k, v in meta.items() if v}

    for case in manifest.get("cases", []) or []:
        if isinstance(case, dict):
            add(case)
    for group_key in ("groups", "families"):
        groups = manifest.get(group_key, []) or []
        group_iter = groups.values() if isinstance(groups, dict) else groups
        for group in group_iter:
            if not isinstance(group, dict):
                continue
            group_name = str(group.get("group") or group.get("group_id") or group.get("family") or group.get("name") or "")
            role = str(group.get("role") or "") or None
            for case in group.get("cases", []) or []:
                if isinstance(case, dict):
                    add(case, group=group_name, role=role)
                else:
                    add({"path": case, "name": Path(str(case)).stem}, group=group_name, role=role)
    return out


def _meta_for_case(case_meta: dict[str, dict[str, str]], row: dict[str, Any]) -> dict[str, str]:
    for key in (_case_key(row), str(row.get("name") or "")):
        if key in case_meta:
            return case_meta[key]
    return {}


def _status_for(rows: Sequence[dict[str, Any]]) -> str:
    mean_corr = _mean(_as_float(row.get("corr_delta_vs_baseline_config")) for row in rows)
    median_corr = _median(_as_float(row.get("corr_delta_vs_baseline_config")) for row in rows)
    win = _mean(1.0 if _as_float(row.get("corr_delta_vs_baseline_config")) > 0.0 else 0.0 for row in rows)
    mean_mse = _mean(_as_float(row.get("mse_delta_vs_baseline_config")) for row in rows)
    if mean_corr >= 0.001 and median_corr > 0.0 and win >= 0.55 and mean_mse <= 0.0:
        return "variant_improved"
    if mean_corr > 0.0 or mean_mse < 0.0:
        return "variant_tradeoff_or_tiny_signal"
    return "variant_no_improvement"


def _summarize(rows: Sequence[dict[str, Any]], group_keys: Sequence[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key) for key in group_keys)].append(row)
    summaries: list[dict[str, Any]] = []
    for key, local in sorted(grouped.items(), key=lambda item: item[0]):
        corr = [_as_float(row.get("corr_delta_vs_baseline_config")) for row in local]
        mse = [_as_float(row.get("mse_delta_vs_baseline_config")) for row in local]
        summary = {name: value for name, value in zip(group_keys, key)}
        summary.update(
            {
                "row_count": int(len(local)),
                "case_count": int(len({row.get("case_key") for row in local})),
                "status": _status_for(local),
                "mean_corr_delta_vs_baseline_config": _mean(corr),
                "median_corr_delta_vs_baseline_config": _median(corr),
                "corr_win_fraction_vs_baseline_config": _mean(1.0 if value > 0.0 else 0.0 for value in corr),
                "mean_mse_delta_vs_baseline_config": _mean(mse),
                "mse_win_fraction_vs_baseline_config": _mean(1.0 if value < 0.0 else 0.0 for value in mse),
            }
        )
        summaries.append(summary)
    return summaries


def compare_variants(
    *,
    baseline: tuple[str, Path],
    results: Sequence[tuple[str, Path]],
    out_dir: Path,
    manifest_path: Path | None = None,
    excluded_roles: Sequence[str] = (),
    include_gain0: bool = False,
) -> dict[str, Any]:
    baseline_label, baseline_path = baseline
    baseline_payload = _load_json(baseline_path)
    manifest = _load_json(manifest_path) if manifest_path else None
    case_meta = _manifest_case_meta(manifest)
    excluded = {role for role in excluded_roles if role}

    baseline_rows: dict[tuple[str, tuple[str, str, str, float]], dict[str, Any]] = {}
    skipped_baseline_gain0 = 0
    for row in _iter_result_rows(baseline_payload):
        if not include_gain0 and abs(_as_float(row.get("gain"))) < 1.0e-12:
            skipped_baseline_gain0 += 1
            continue
        meta = _meta_for_case(case_meta, row)
        if meta.get("role") in excluded:
            continue
        baseline_rows[(_case_key(row), _row_key(row))] = row

    deltas: list[dict[str, Any]] = []
    missing = 0
    skipped_gain0 = 0
    skipped_role = 0
    for label, result_path in results:
        payload = _load_json(result_path)
        for row in _iter_result_rows(payload):
            if not include_gain0 and abs(_as_float(row.get("gain"))) < 1.0e-12:
                skipped_gain0 += 1
                continue
            meta = _meta_for_case(case_meta, row)
            if meta.get("role") in excluded:
                skipped_role += 1
                continue
            case_key = _case_key(row)
            row_key = _row_key(row)
            base = baseline_rows.get((case_key, row_key))
            if base is None:
                missing += 1
                continue
            target_corr = _as_float(row.get("target_corr"))
            baseline_corr = _as_float(base.get("target_corr"))
            target_mse = _as_float(row.get("target_mse"))
            baseline_mse = _as_float(base.get("target_mse"))
            deltas.append(
                {
                    "run_label": label,
                    "result_path": str(result_path),
                    "baseline_label": baseline_label,
                    "baseline_path": str(baseline_path),
                    "case_key": case_key,
                    "case_name": row.get("name"),
                    "group": meta.get("group", ""),
                    "role": meta.get("role", ""),
                    "magnitude_mode": row_key[0],
                    "mask_mode": row_key[1],
                    "mechanism": row_key[2],
                    "gain": row_key[3],
                    "target_corr": target_corr,
                    "baseline_config_target_corr": baseline_corr,
                    "corr_delta_vs_baseline_config": target_corr - baseline_corr,
                    "target_mse": target_mse,
                    "baseline_config_target_mse": baseline_mse,
                    "mse_delta_vs_baseline_config": target_mse - baseline_mse,
                }
            )

    summaries = _summarize(deltas, ("run_label", "magnitude_mode", "mask_mode", "mechanism", "gain"))
    summaries.sort(
        key=lambda row: (
            _as_float(row.get("mean_corr_delta_vs_baseline_config")),
            _as_float(row.get("corr_win_fraction_vs_baseline_config")),
            -_as_float(row.get("mean_mse_delta_vs_baseline_config")),
        ),
        reverse=True,
    )
    overall = _summarize(deltas, ("run_label",))
    overall.sort(key=lambda row: _as_float(row.get("mean_corr_delta_vs_baseline_config")), reverse=True)
    status = "variant_improved" if any(row.get("status") == "variant_improved" for row in summaries) else (
        "variant_tradeoff_or_tiny_signal"
        if any(row.get("status") == "variant_tradeoff_or_tiny_signal" for row in summaries)
        else "variant_no_improvement"
    )
    summary = {
        "schema": "circleworld_internal_phase_law_variant_compare_v0",
        "status": status,
        "baseline": {"label": baseline_label, "path": str(baseline_path)},
        "manifest": str(manifest_path) if manifest_path else None,
        "excluded_roles": list(excluded_roles),
        "include_gain0": bool(include_gain0),
        "result_count": int(len(results)),
        "delta_row_count": int(len(deltas)),
        "case_count": int(len({row.get("case_key") for row in deltas})),
        "skipped_baseline_gain0": int(skipped_baseline_gain0),
        "skipped_gain0": int(skipped_gain0),
        "skipped_role": int(skipped_role),
        "missing_matches": int(missing),
        "overall_summaries": overall,
        "candidate_summaries": summaries,
        "case_deltas": deltas,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / OUTPUT_JSON).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / OUTPUT_MD).write_text(_markdown(summary), encoding="utf-8")
    return summary


def _markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Internal Phase Law Variant Compare",
        "",
        "This compares each internal phase-law config against the no-op control on matched lockbox rows.",
        "",
        f"- status: `{summary['status']}`",
        f"- baseline: `{summary['baseline']['label']}`",
        f"- results: `{summary['result_count']}`",
        f"- cases: `{summary['case_count']}`",
        f"- delta rows: `{summary['delta_row_count']}`",
        f"- excluded roles: `{', '.join(summary.get('excluded_roles') or []) or 'none'}`",
        "",
        "## Overall",
        "",
        "| run | rows | cases | status | mean corr d | median corr d | corr wins | mean MSE d | MSE wins |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary.get("overall_summaries", []):
        lines.append(
            "| {run} | {rows} | {cases} | {status} | {mc} | {med} | {cw} | {mm} | {mw} |".format(
                run=row.get("run_label"),
                rows=row.get("row_count"),
                cases=row.get("case_count"),
                status=row.get("status"),
                mc=_fmt(row.get("mean_corr_delta_vs_baseline_config")),
                med=_fmt(row.get("median_corr_delta_vs_baseline_config")),
                cw=_fmt(row.get("corr_win_fraction_vs_baseline_config")),
                mm=_fmt(row.get("mean_mse_delta_vs_baseline_config")),
                mw=_fmt(row.get("mse_win_fraction_vs_baseline_config")),
            )
        )
    lines.extend(
        [
            "",
            "## Candidate Rows",
            "",
            "| run | mode | mask | mechanism | gain | rows | cases | status | mean corr d | median corr d | corr wins | mean MSE d |",
            "|---|---|---|---|---:|---:|---:|---|---:|---:|---:|---:|",
        ]
    )
    for row in summary.get("candidate_summaries", [])[:80]:
        lines.append(
            "| {run} | {mode} | {mask} | {mechanism} | {gain} | {rows} | {cases} | {status} | {mc} | {med} | {cw} | {mm} |".format(
                run=row.get("run_label"),
                mode=row.get("magnitude_mode"),
                mask=row.get("mask_mode"),
                mechanism=row.get("mechanism"),
                gain=_fmt(row.get("gain")),
                rows=row.get("row_count"),
                cases=row.get("case_count"),
                status=row.get("status"),
                mc=_fmt(row.get("mean_corr_delta_vs_baseline_config")),
                med=_fmt(row.get("median_corr_delta_vs_baseline_config")),
                cw=_fmt(row.get("corr_win_fraction_vs_baseline_config")),
                mm=_fmt(row.get("mean_mse_delta_vs_baseline_config")),
            )
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare internal phase-law variants against a matched no-op lockbox run.")
    ap.add_argument("--baseline", required=True, help="Baseline result as label=path or path.")
    ap.add_argument("--result", action="append", required=True, help="Variant result as label=path or path. May be repeated.")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--exclude-role", action="append", default=[])
    ap.add_argument("--include-gain0", action="store_true")
    args = ap.parse_args()
    summary = compare_variants(
        baseline=_parse_labeled_path(args.baseline),
        results=[_parse_labeled_path(raw) for raw in args.result],
        out_dir=Path(args.out_dir),
        manifest_path=Path(args.manifest) if args.manifest else None,
        excluded_roles=args.exclude_role,
        include_gain0=bool(args.include_gain0),
    )
    print(json.dumps({"json": str(Path(args.out_dir) / OUTPUT_JSON), "markdown": str(Path(args.out_dir) / OUTPUT_MD), "status": summary["status"]}, indent=2))


if __name__ == "__main__":
    main()
