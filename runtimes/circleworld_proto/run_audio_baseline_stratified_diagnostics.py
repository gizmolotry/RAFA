from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Sequence


TOKENBURST_ROOT = Path(r"C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst")
DEFAULT_DIAGNOSTICS = (
    TOKENBURST_ROOT
    / "audio_phase_phenotype_diagnostics_cuda_2026_05_06"
    / "audio_phase_phenotype_diagnostics.json"
)
OUTPUT_JSON = "audio_baseline_stratified_diagnostics.json"
OUTPUT_MD = "AUDIO_BASELINE_STRATIFIED_DIAGNOSTICS.md"

FIXED_BINS = (
    ("bad_baseline", float("-inf"), -0.05),
    ("weak_baseline", -0.05, 0.05),
    ("moderate_baseline", 0.05, 0.20),
    ("good_baseline", 0.20, float("inf")),
)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _mean(values: Iterable[float]) -> float:
    vals = [float(value) for value in values]
    return float(sum(vals) / len(vals)) if vals else 0.0


def _median(values: Iterable[float]) -> float:
    vals = sorted(float(value) for value in values)
    if not vals:
        return 0.0
    mid = len(vals) // 2
    if len(vals) % 2:
        return vals[mid]
    return float(0.5 * (vals[mid - 1] + vals[mid]))


def _positive_outlier_share(rows: Sequence[dict[str, Any]]) -> float:
    positives = sorted((_as_float(row.get("corr_delta")) for row in rows if _as_float(row.get("corr_delta")) > 0.0), reverse=True)
    if not positives:
        return 0.0
    return float(positives[0] / max(1.0e-12, sum(positives)))


def _summarize_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    corr = [_as_float(row.get("corr_delta")) for row in rows]
    mse = [_as_float(row.get("mse_delta")) for row in rows]
    roles: dict[str, int] = {}
    groups: dict[str, int] = {}
    for row in rows:
        roles[str(row.get("role"))] = roles.get(str(row.get("role")), 0) + 1
        groups[str(row.get("group"))] = groups.get(str(row.get("group")), 0) + 1
    return {
        "case_count": len(rows),
        "mean_gain0_target_corr": _mean([_as_float(row.get("gain0_target_corr")) for row in rows]),
        "median_gain0_target_corr": _median([_as_float(row.get("gain0_target_corr")) for row in rows]),
        "mean_corr_delta": _mean(corr),
        "median_corr_delta": _median(corr),
        "corr_win_fraction": _mean([1.0 if value > 1.0e-6 else 0.0 for value in corr]),
        "mean_mse_delta": _mean(mse),
        "mse_win_fraction": _mean([1.0 if value < -1.0e-9 else 0.0 for value in mse]),
        "positive_outlier_share": _positive_outlier_share(rows),
        "role_counts": dict(sorted(roles.items())),
        "group_counts": dict(sorted(groups.items())),
        "top_positive_case": max(rows, key=lambda row: _as_float(row.get("corr_delta")), default={}),
        "top_negative_case": min(rows, key=lambda row: _as_float(row.get("corr_delta")), default={}),
    }


def _fixed_bin_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for name, lo, hi in FIXED_BINS:
        local = [
            row
            for row in rows
            if _as_float(row.get("gain0_target_corr")) >= lo and _as_float(row.get("gain0_target_corr")) < hi
        ]
        summary = _summarize_rows(local)
        summary.update({"bin": name, "lo": lo, "hi": hi})
        out.append(summary)
    return out


def _quantile_bin_rows(rows: Sequence[dict[str, Any]], bins: int) -> list[dict[str, Any]]:
    if not rows:
        return []
    ordered = sorted(rows, key=lambda row: _as_float(row.get("gain0_target_corr")))
    out: list[dict[str, Any]] = []
    bins = max(1, int(bins))
    for idx in range(bins):
        start = int(round(idx * len(ordered) / bins))
        end = int(round((idx + 1) * len(ordered) / bins))
        local = ordered[start:end]
        if not local:
            continue
        summary = _summarize_rows(local)
        summary.update(
            {
                "bin": f"q{idx + 1}_of_{bins}",
                "rank_start": start,
                "rank_end": end,
                "lo_gain0_target_corr": _as_float(local[0].get("gain0_target_corr")),
                "hi_gain0_target_corr": _as_float(local[-1].get("gain0_target_corr")),
            }
        )
        out.append(summary)
    return out


def _status(rows: Sequence[dict[str, Any]], fixed_bins: Sequence[dict[str, Any]]) -> str:
    if not rows:
        return "missing_cases"
    non_bad = [row for row in fixed_bins if row.get("bin") != "bad_baseline" and int(row.get("case_count") or 0) > 0]
    good = [row for row in fixed_bins if row.get("bin") == "good_baseline" and int(row.get("case_count") or 0) > 0]
    non_bad_positive = any(_as_float(row.get("median_corr_delta")) > 0.005 and _as_float(row.get("corr_win_fraction")) >= 0.67 for row in non_bad)
    good_positive = any(_as_float(row.get("median_corr_delta")) > 0.005 and _as_float(row.get("corr_win_fraction")) >= 0.67 for row in good)
    bad = next((row for row in fixed_bins if row.get("bin") == "bad_baseline"), {})
    bad_dominated = _as_float(bad.get("mean_corr_delta")) > 0.02 and _as_float(bad.get("positive_outlier_share")) > 0.5
    if good_positive:
        return "signal_survives_good_baseline_bin"
    if non_bad_positive:
        return "signal_survives_non_bad_baseline_bin"
    if bad_dominated:
        return "bad_baseline_rescue_dominated"
    return "no_stratified_signal"


def _write_markdown(summary: dict[str, Any], out_path: Path) -> None:
    def fmt(value: Any) -> str:
        if value is None:
            return "NA"
        if isinstance(value, (int, float)):
            if value == float("inf"):
                return "inf"
            if value == float("-inf"):
                return "-inf"
            return f"{float(value):.6g}"
        return str(value)

    lines = [
        "# Audio Baseline Stratified Diagnostics",
        "",
        "This bins the post-hoc phenotype diagnostic by gain-0 target correlation.",
        "It checks whether the frozen mechanism signal survives outside bad baseline cases.",
        "",
        f"- status: `{summary.get('status')}`",
        f"- diagnostics: `{summary.get('diagnostics_path')}`",
        f"- cases: `{fmt(summary.get('case_count'))}`",
        f"- excluded single-source probes: `{summary.get('exclude_single_source')}`",
        "",
        "## Fixed Bins",
        "",
        "| bin | range | cases | mean gain0 corr | mean corr d | median corr d | corr wins | mean MSE d | outlier share | top positive | top d |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for row in summary.get("fixed_bins", []) or []:
        top = row.get("top_positive_case") if isinstance(row.get("top_positive_case"), dict) else {}
        lines.append(
            "| {bin} | [{lo}, {hi}) | {cases} | {base} | {mean} | {median} | {wins} | {mse} | {share} | {top} | {topd} |".format(
                bin=row.get("bin"),
                lo=fmt(row.get("lo")),
                hi=fmt(row.get("hi")),
                cases=fmt(row.get("case_count")),
                base=fmt(row.get("mean_gain0_target_corr")),
                mean=fmt(row.get("mean_corr_delta")),
                median=fmt(row.get("median_corr_delta")),
                wins=fmt(row.get("corr_win_fraction")),
                mse=fmt(row.get("mean_mse_delta")),
                share=fmt(row.get("positive_outlier_share")),
                top=top.get("name"),
                topd=fmt(top.get("corr_delta")),
            )
        )
    lines.extend(
        [
            "",
            "## Quantile Bins",
            "",
            "| bin | gain0 corr range | cases | mean corr d | median corr d | corr wins | mean MSE d | outlier share |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary.get("quantile_bins", []) or []:
        lines.append(
            "| {bin} | [{lo}, {hi}] | {cases} | {mean} | {median} | {wins} | {mse} | {share} |".format(
                bin=row.get("bin"),
                lo=fmt(row.get("lo_gain0_target_corr")),
                hi=fmt(row.get("hi_gain0_target_corr")),
                cases=fmt(row.get("case_count")),
                mean=fmt(row.get("mean_corr_delta")),
                median=fmt(row.get("median_corr_delta")),
                wins=fmt(row.get("corr_win_fraction")),
                mse=fmt(row.get("mean_mse_delta")),
                share=fmt(row.get("positive_outlier_share")),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation Guard",
            "",
            "- If only bad-baseline bins improve, this is baseline-rescue evidence, not a robust audio-law signal.",
            "- This report is post-hoc for the current manifest and should define fixed bins for the next predeclared run.",
        ]
    )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_baseline_stratified_diagnostics(
    *,
    diagnostics_path: Path,
    out_dir: Path,
    exclude_single_source: bool,
    quantile_bins: int,
) -> dict[str, Any]:
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8-sig"))
    rows = [
        row
        for row in diagnostics.get("cases", []) or []
        if isinstance(row, dict) and not (exclude_single_source and row.get("role") == "single_source_probe")
    ]
    fixed = _fixed_bin_rows(rows)
    quantile = _quantile_bin_rows(rows, bins=quantile_bins)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "runtime": "circleworld_proto",
        "schema": "circleworld_audio_baseline_stratified_diagnostics_v0",
        "status": _status(rows, fixed),
        "diagnostics_path": str(diagnostics_path),
        "out_dir": str(out_dir),
        "exclude_single_source": bool(exclude_single_source),
        "case_count": len(rows),
        "source_case_count": diagnostics.get("case_count"),
        "fixed_bins": fixed,
        "quantile_bins": quantile,
    }
    (out_dir / OUTPUT_JSON).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_markdown(summary, out_dir / OUTPUT_MD)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Stratify frozen audio mechanism diagnostics by gain-0 baseline quality.")
    ap.add_argument("--diagnostics", default=str(DEFAULT_DIAGNOSTICS))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--include-single-source", action="store_true")
    ap.add_argument("--quantile-bins", type=int, default=4)
    args = ap.parse_args()
    summary = run_baseline_stratified_diagnostics(
        diagnostics_path=Path(args.diagnostics),
        out_dir=Path(args.out_dir),
        exclude_single_source=not bool(args.include_single_source),
        quantile_bins=int(args.quantile_bins),
    )
    print(
        json.dumps(
            {
                "json": str(Path(args.out_dir) / OUTPUT_JSON),
                "markdown": str(Path(args.out_dir) / OUTPUT_MD),
                "status": summary.get("status"),
                "case_count": summary.get("case_count"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
