from __future__ import annotations

import argparse
import json
import wave
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from benchmark_audio_continuation import _chunk_vectors, _cosine, _loop_reentry_metrics  # noqa: E402


OUTPUT_JSON = "phase_native_audio_reentry_metric_audit.json"
OUTPUT_MD = "PHASE_NATIVE_AUDIO_REENTRY_METRIC_AUDIT.md"


def _json_load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(float(value) for value in values) / len(values))


def _median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def _read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sr = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
    if sample_width == 1:
        x = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sample_width == 2:
        x = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        x = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width {sample_width} in {path}")
    if channels > 1:
        x = x.reshape(-1, channels).mean(axis=1)
    return x.astype(np.float32), int(sr)


def _first_reentry_series(x: np.ndarray, sr: int) -> list[float]:
    duration = float(len(x) / float(sr)) if sr else 0.0
    chunk_seconds = min(1.0, max(0.1, duration / 4.0))
    chunks = _chunk_vectors(x, sr=sr, chunk_seconds=chunk_seconds)
    if not chunks:
        return []
    return [_cosine(chunks[0], chunks[i]) for i in range(1, len(chunks))]


def _chunk_profile_similarity(a: np.ndarray, b: np.ndarray, sr: int) -> float:
    duration = float(min(len(a), len(b)) / float(sr)) if sr else 0.0
    chunk_seconds = min(1.0, max(0.1, duration / 4.0))
    a_chunks = _chunk_vectors(a[: min(len(a), len(b))], sr=sr, chunk_seconds=chunk_seconds)
    b_chunks = _chunk_vectors(b[: min(len(a), len(b))], sr=sr, chunk_seconds=chunk_seconds)
    if not a_chunks or not b_chunks:
        return 0.0
    n = min(len(a_chunks), len(b_chunks))
    return _mean([_cosine(a_chunks[i], b_chunks[i]) for i in range(n)])


def _wave_metrics(path: Path) -> dict[str, float]:
    x, sr = _read_wav(path)
    import torch

    return _loop_reentry_metrics(torch.from_numpy(x), sr)


def _case_group(case_name: str) -> str:
    return str(case_name).split("__", 1)[0]


def _method_wav(row: dict[str, Any], method: str) -> Path | None:
    item = row.get("methods", {}).get(method)
    if not item:
        return None
    wav = item.get("wav_path")
    return Path(str(wav)) if wav else None


def _score_summary(summary: dict[str, Any], source_json: Path, *, target_margin: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in summary.get("rows", []):
        case_name = str(case.get("name"))
        target_path = Path(str(case.get("target_future_wav")))
        target_x, target_sr = _read_wav(target_path)
        target_loop = _wave_metrics(target_path)
        target_first_series = _first_reentry_series(target_x, target_sr)
        for method in sorted(case.get("methods", {})):
            wav_path = _method_wav(case, method)
            if wav_path is None or not wav_path.exists():
                continue
            pred_x, pred_sr = _read_wav(wav_path)
            if pred_sr != target_sr:
                raise ValueError(f"Sample-rate mismatch for {wav_path}: {pred_sr} vs {target_sr}")
            pred_loop = _wave_metrics(wav_path)
            pred_first_series = _first_reentry_series(pred_x, pred_sr)
            n = min(len(target_first_series), len(pred_first_series))
            first_profile_error = (
                _mean([abs(pred_first_series[i] - target_first_series[i]) for i in range(n)]) if n else 0.0
            )
            target_norm_reentry_delta = _as_float(pred_loop.get("first_chunk_reentry")) - _as_float(
                target_loop.get("first_chunk_reentry")
            )
            harmful_excess = max(0.0, target_norm_reentry_delta - float(target_margin))
            chunk_profile_sim = _chunk_profile_similarity(pred_x, target_x, pred_sr)
            rows.append(
                {
                    "source_json": str(source_json),
                    "case": case_name,
                    "group": _case_group(case_name),
                    "method": method,
                    "wav_path": str(wav_path),
                    "target_wav_path": str(target_path),
                    "target_first_chunk_reentry": _as_float(target_loop.get("first_chunk_reentry")),
                    "method_first_chunk_reentry": _as_float(pred_loop.get("first_chunk_reentry")),
                    "target_norm_reentry_delta": target_norm_reentry_delta,
                    "harmful_replay_excess": harmful_excess,
                    "first_reentry_profile_error": first_profile_error,
                    "target_loop_autocorr_peak": _as_float(target_loop.get("loop_autocorr_peak")),
                    "method_loop_autocorr_peak": _as_float(pred_loop.get("loop_autocorr_peak")),
                    "target_norm_loop_delta": _as_float(pred_loop.get("loop_autocorr_peak"))
                    - _as_float(target_loop.get("loop_autocorr_peak")),
                    "target_chunk_profile_similarity": chunk_profile_sim,
                    "metrics": case.get("methods", {}).get(method, {}).get("metrics", {}),
                }
            )
    return rows


def _aggregate(rows: Sequence[dict[str, Any]], group_key: str) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get(group_key, "")), str(row.get("method")))].append(row)
    out: list[dict[str, Any]] = []
    for (group_value, method), items in sorted(grouped.items()):
        harmful = [_as_float(row.get("harmful_replay_excess")) for row in items]
        target_delta = [_as_float(row.get("target_norm_reentry_delta")) for row in items]
        profile_error = [_as_float(row.get("first_reentry_profile_error")) for row in items]
        corr = [_as_float(row.get("metrics", {}).get("corr")) for row in items]
        mse = [_as_float(row.get("metrics", {}).get("mse")) for row in items]
        chunk_profile = [_as_float(row.get("target_chunk_profile_similarity")) for row in items]
        out.append(
            {
                group_key: group_value,
                "method": method,
                "case_count": len(items),
                "mean_harmful_replay_excess": _mean(harmful),
                "median_harmful_replay_excess": _median(harmful),
                "harmful_replay_case_fraction": _mean([1.0 if value > 0.0 else 0.0 for value in harmful]),
                "mean_target_norm_reentry_delta": _mean(target_delta),
                "median_target_norm_reentry_delta": _median(target_delta),
                "mean_first_reentry_profile_error": _mean(profile_error),
                "median_first_reentry_profile_error": _median(profile_error),
                "mean_target_chunk_profile_similarity": _mean(chunk_profile),
                "mean_corr": _mean(corr),
                "mean_mse": _mean(mse),
            }
        )
    return out


def _method_comparison(aggregate: Sequence[dict[str, Any]], baseline: str = "baseline_copy_last") -> list[dict[str, Any]]:
    by_scope: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in aggregate:
        scope = str(row.get("source_json", row.get("group", "all")))
        by_scope[scope][str(row["method"])] = row
    rows: list[dict[str, Any]] = []
    for scope, methods in sorted(by_scope.items()):
        base = methods.get(baseline)
        if not base:
            continue
        for method, item in sorted(methods.items()):
            if method == baseline:
                continue
            rows.append(
                {
                    "scope": scope,
                    "method": method,
                    "baseline": baseline,
                    "harmful_replay_excess_delta_vs_baseline": _as_float(item.get("mean_harmful_replay_excess"))
                    - _as_float(base.get("mean_harmful_replay_excess")),
                    "target_norm_reentry_delta_vs_baseline": _as_float(item.get("mean_target_norm_reentry_delta"))
                    - _as_float(base.get("mean_target_norm_reentry_delta")),
                    "first_reentry_profile_error_delta_vs_baseline": _as_float(
                        item.get("mean_first_reentry_profile_error")
                    )
                    - _as_float(base.get("mean_first_reentry_profile_error")),
                    "corr_delta_vs_baseline": _as_float(item.get("mean_corr")) - _as_float(base.get("mean_corr")),
                    "mse_delta_vs_baseline": _as_float(item.get("mean_mse")) - _as_float(base.get("mean_mse")),
                }
            )
    return rows


def _markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Phase-Native Audio Reentry Metric Audit",
        "",
        "This audit separates self-similarity from harmful replay by comparing each",
        "generated future's first-chunk reentry against the real target future's",
        "own first-chunk reentry profile.",
        "",
        "## Summary",
        "",
        f"- Status: `{summary['status']}`",
        f"- Target margin: `{summary['target_margin']}`",
        f"- Source summaries: `{len(summary['source_jsons'])}`",
        f"- Rows: `{summary['row_count']}`",
        "",
        "## Method Aggregate",
        "",
        "| method | cases | harmful excess | harmful fraction | target reentry delta | profile error | corr | MSE |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["method_aggregate"]:
        lines.append(
            "| `{method}` | {cases} | {harm:.9f} | {frac:.6f} | {tdelta:.9f} | {perr:.9f} | {corr:.9f} | {mse:.9f} |".format(
                method=row["method"],
                cases=row["case_count"],
                harm=_as_float(row["mean_harmful_replay_excess"]),
                frac=_as_float(row["harmful_replay_case_fraction"]),
                tdelta=_as_float(row["mean_target_norm_reentry_delta"]),
                perr=_as_float(row["mean_first_reentry_profile_error"]),
                corr=_as_float(row["mean_corr"]),
                mse=_as_float(row["mean_mse"]),
            )
        )
    lines.extend(["", "## Versus Copy-Last", ""])
    lines.extend(
        [
            "| method | harmful excess delta | target reentry delta | profile-error delta | corr delta | MSE delta |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary["method_vs_copy_last"]:
        lines.append(
            "| `{method}` | {harm:.9f} | {tdelta:.9f} | {perr:.9f} | {corr:.9f} | {mse:.9f} |".format(
                method=row["method"],
                harm=_as_float(row["harmful_replay_excess_delta_vs_baseline"]),
                tdelta=_as_float(row["target_norm_reentry_delta_vs_baseline"]),
                perr=_as_float(row["first_reentry_profile_error_delta_vs_baseline"]),
                corr=_as_float(row["corr_delta_vs_baseline"]),
                mse=_as_float(row["mse_delta_vs_baseline"]),
            )
        )
    lines.extend(["", "## Circleworld Group Breakdown", ""])
    lines.extend(
        [
            "| group | cases | harmful excess | harmful fraction | target reentry delta | profile error | corr | MSE |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary["group_method_aggregate"]:
        if row["method"] != "circleworld":
            continue
        lines.append(
            "| `{group}` | {cases} | {harm:.9f} | {frac:.6f} | {tdelta:.9f} | {perr:.9f} | {corr:.9f} | {mse:.9f} |".format(
                group=row["group"],
                cases=row["case_count"],
                harm=_as_float(row["mean_harmful_replay_excess"]),
                frac=_as_float(row["harmful_replay_case_fraction"]),
                tdelta=_as_float(row["mean_target_norm_reentry_delta"]),
                perr=_as_float(row["mean_first_reentry_profile_error"]),
                corr=_as_float(row["mean_corr"]),
                mse=_as_float(row["mean_mse"]),
            )
        )
    lines.extend(["", "## Interpretation", "", summary["interpretation"]])
    return "\n".join(lines) + "\n"


def run_audit(source_jsons: Sequence[Path], out_dir: Path, target_margin: float) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    payloads = []
    for source_json in source_jsons:
        payload = _json_load(source_json)
        payloads.append(payload)
        rows.extend(_score_summary(payload, source_json, target_margin=target_margin))
    method_aggregate = _aggregate(rows, "source_json")
    group_aggregate = _aggregate(rows, "group")
    # Fold all source summaries together by method.
    folded_rows = [dict(row, source_json="all") for row in rows]
    folded_aggregate = _aggregate(folded_rows, "source_json")
    method_vs_copy_last = _method_comparison(folded_aggregate)
    circle = next((row for row in folded_aggregate if row["method"] == "circleworld"), {})
    copy_last = next((row for row in folded_aggregate if row["method"] == "baseline_copy_last"), {})
    harmful_delta = _as_float(circle.get("mean_harmful_replay_excess")) - _as_float(
        copy_last.get("mean_harmful_replay_excess")
    )
    if harmful_delta <= 0.0:
        status = "target_normalized_reentry_pass"
        interpretation = (
            "Circleworld does not exceed copy-last harmful replay after normalizing to the target future's own "
            "first-chunk reentry. The older self-only guard may be too blunt for stationary audio."
        )
    else:
        status = "target_normalized_reentry_still_fails"
        interpretation = (
            "Circleworld still has more target-normalized harmful replay than copy-last. The reentry failure is not "
            "only an artifact of stationary targets."
        )
    summary = {
        "schema": "phase_native_audio_reentry_metric_audit_v1",
        "status": status,
        "source_jsons": [str(path) for path in source_jsons],
        "target_margin": float(target_margin),
        "row_count": len(rows),
        "method_aggregate": folded_aggregate,
        "source_method_aggregate": method_aggregate,
        "group_method_aggregate": group_aggregate,
        "method_vs_copy_last": method_vs_copy_last,
        "case_rows": rows,
        "interpretation": interpretation,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / OUTPUT_JSON).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / OUTPUT_MD).write_text(_markdown(summary), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit first-chunk reentry against target-normalized replay.")
    parser.add_argument("--source-json", required=True, action="append", type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--target-margin", type=float, default=0.01)
    args = parser.parse_args()
    summary = run_audit(args.source_json, args.out_dir, args.target_margin)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "json": str(Path(args.out_dir) / OUTPUT_JSON),
                "markdown": str(Path(args.out_dir) / OUTPUT_MD),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
