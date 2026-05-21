from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

COSINE_KEYS: tuple[str, ...] = (
    "mean_pooled_full_cos",
    "mean_prefix_cos",
    "mean_mean_prefix_cos",
    "mean_q_profile_cos",
    "mean_arc_profile_cos",
    "mean_temporal_profile_cos",
    "mean_support_profile_cos",
    "mean_branch_profile_cos",
)

DRIFT_KEYS: tuple[str, ...] = (
    "mean_signature_count_delta",
    "mean_family_count_delta",
    "mean_confidence_delta",
    "mean_q_entropy_delta",
    "mean_branch_mass_delta",
    "mean_dominant_family_share_delta",
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _nested_get(payload: dict[str, Any], *keys: str, default: float = 0.0) -> float:
    cur: Any = payload
    for key in keys:
        if not isinstance(cur, dict):
            return float(default)
        cur = cur.get(key)
    if cur is None:
        return float(default)
    return float(cur)


def _first_number(payload: dict[str, Any], keys: list[str], default: float = 0.0) -> float:
    for key in keys:
        if key in payload and payload.get(key) is not None:
            return float(payload[key])
    return float(default)


def _export_metrics(summary: dict[str, Any]) -> dict[str, float]:
    agg = summary.get("aggregate_relational_signature_library", {}) if isinstance(summary.get("aggregate_relational_signature_library"), dict) else {}
    return {
        "mean_num_relational_signatures": float(summary.get("mean_num_relational_signatures", 0.0)),
        "mean_num_relational_signature_families": float(summary.get("mean_num_relational_signature_families", 0.0)),
        "mean_relational_signature_confidence": float(summary.get("mean_relational_signature_confidence", 0.0)),
        "aggregate_num_relational_signature_families": float(agg.get("num_families", 0.0)),
        "aggregate_mean_relational_family_size": float(agg.get("mean_family_size", 0.0)),
    }


def _heldout_metrics(summary: dict[str, Any]) -> dict[str, float]:
    return {
        "mean_real_branch_fraction": float(summary.get("mean_real_branch_fraction", 0.0)),
        "mean_relational_signature_confidence": float(summary.get("mean_relational_signature_confidence", 0.0)),
        "mean_relational_signature_q_entropy": float(summary.get("mean_relational_signature_q_entropy", 0.0)),
        "mean_relational_branch_mass": float(summary.get("mean_relational_branch_mass", 0.0)),
        "mean_num_relational_signature_families": float(summary.get("mean_num_relational_signature_families", 0.0)),
        "mean_dominant_relational_family_share": float(summary.get("mean_dominant_relational_family_share", 0.0)),
    }


def _metamer_metrics(summary: dict[str, Any]) -> dict[str, float]:
    return {
        "mean_pooled_full_cos": float(summary.get("mean_pooled_full_cos", 0.0)),
        "mean_prefix_cos": float(summary.get("mean_prefix_cos", 0.0)),
        "mean_q_profile_cos": float(summary.get("mean_q_profile_cos", 0.0)),
        "mean_arc_profile_cos": float(summary.get("mean_arc_profile_cos", 0.0)),
        "mean_temporal_profile_cos": float(summary.get("mean_temporal_profile_cos", 0.0)),
        "mean_support_profile_cos": float(summary.get("mean_support_profile_cos", 0.0)),
        "mean_branch_profile_cos": float(summary.get("mean_branch_profile_cos", 0.0)),
        "mean_signature_count_delta": float(summary.get("mean_signature_count_delta", 0.0)),
        "mean_family_count_delta": float(summary.get("mean_family_count_delta", 0.0)),
        "mean_confidence_delta": float(summary.get("mean_confidence_delta", 0.0)),
        "mean_q_entropy_delta": float(summary.get("mean_q_entropy_delta", 0.0)),
        "mean_branch_mass_delta": float(summary.get("mean_branch_mass_delta", 0.0)),
        "mean_dominant_family_share_delta": float(summary.get("mean_dominant_family_share_delta", 0.0)),
    }


def _benchmark_metrics(summary: dict[str, Any]) -> dict[str, float]:
    return {
        "mean_corr": float(summary.get("mean_corr", 0.0)),
        "mean_mae": float(summary.get("mean_mae", 0.0)),
        "mean_prefix_alignment": float(summary.get("mean_prefix_alignment", 0.0)),
        "mean_prefix_delta": float(summary.get("mean_prefix_delta", 0.0)),
    }


def _continuity_metrics(summary: dict[str, Any]) -> dict[str, float]:
    return {
        "mean_loop_autocorr_peak": float(summary.get("mean_loop_autocorr_peak", 0.0)),
        "mean_loop_period_seconds": float(summary.get("mean_loop_period_seconds", 0.0)),
        "mean_adjacent_chunk_similarity": float(summary.get("mean_adjacent_chunk_similarity", 0.0)),
        "mean_nonlocal_chunk_repeat": float(summary.get("mean_nonlocal_chunk_repeat", 0.0)),
        "mean_first_chunk_reentry": float(summary.get("mean_first_chunk_reentry", 0.0)),
    }


def _comparison_mode(
    key: str,
    *,
    higher_is_better: set[str],
    lower_is_better: set[str],
    closer_to_zero: set[str],
) -> str:
    if key in higher_is_better:
        return "higher_is_better"
    if key in closer_to_zero:
        return "closer_to_zero"
    if key in lower_is_better:
        return "lower_is_better"
    return "smaller_absolute_value"


def _metric_advantage(entry: dict[str, Any], side: str) -> float:
    mode = str(entry.get("comparison_mode"))
    a_val = float(entry.get("a", 0.0))
    b_val = float(entry.get("b", 0.0))
    if mode == "higher_is_better":
        return a_val - b_val if side == "a" else b_val - a_val
    if mode == "lower_is_better":
        return b_val - a_val if side == "a" else a_val - b_val
    return abs(b_val) - abs(a_val) if side == "a" else abs(a_val) - abs(b_val)


def _compare_block(
    a: dict[str, float],
    b: dict[str, float],
    *,
    higher_is_better: set[str],
    lower_is_better: set[str],
    closer_to_zero: set[str] | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    closer_to_zero = closer_to_zero or set()
    keys = sorted(set(a.keys()) | set(b.keys()))
    for key in keys:
        av = float(a.get(key, 0.0))
        bv = float(b.get(key, 0.0))
        delta = av - bv
        mode = _comparison_mode(key, higher_is_better=higher_is_better, lower_is_better=lower_is_better, closer_to_zero=closer_to_zero)
        if mode == "higher_is_better":
            winner = "a" if av > bv else "b" if bv > av else "tie"
        elif mode == "closer_to_zero":
            winner = "a" if abs(av) < abs(bv) else "b" if abs(bv) < abs(av) else "tie"
        elif mode == "lower_is_better":
            winner = "a" if av < bv else "b" if bv < av else "tie"
        else:
            winner = "a" if abs(av) < abs(bv) else "b" if abs(bv) < abs(av) else "tie"
        out[key] = {
            "a": av,
            "b": bv,
            "delta_a_minus_b": delta,
            "winner": winner,
            "comparison_mode": mode,
        }
        if mode in {"closer_to_zero", "smaller_absolute_value"}:
            out[key]["abs_a"] = abs(av)
            out[key]["abs_b"] = abs(bv)
            out[key]["abs_delta_a_minus_b"] = abs(av) - abs(bv)
    return out


def _transform_metric_row(row: dict[str, Any]) -> dict[str, float]:
    return {
        "mean_pooled_full_cos": _nested_get(row, "mean_pooled_full_cos"),
        "mean_prefix_cos": _first_number(row, ["mean_prefix_cos", "mean_mean_prefix_cos"]),
        "mean_mean_prefix_cos": _first_number(row, ["mean_mean_prefix_cos", "mean_prefix_cos"]),
        "mean_q_profile_cos": _nested_get(row, "mean_q_profile_cos"),
        "mean_arc_profile_cos": _nested_get(row, "mean_arc_profile_cos"),
        "mean_temporal_profile_cos": _nested_get(row, "mean_temporal_profile_cos"),
        "mean_support_profile_cos": _nested_get(row, "mean_support_profile_cos"),
        "mean_branch_profile_cos": _nested_get(row, "mean_branch_profile_cos"),
        "mean_signature_count_delta": _nested_get(row, "mean_signature_count_delta"),
        "mean_family_count_delta": _nested_get(row, "mean_family_count_delta"),
        "mean_confidence_delta": _nested_get(row, "mean_confidence_delta"),
        "mean_q_entropy_delta": _nested_get(row, "mean_q_entropy_delta"),
        "mean_branch_mass_delta": _nested_get(row, "mean_branch_mass_delta"),
        "mean_dominant_family_share_delta": _nested_get(row, "mean_dominant_family_share_delta"),
    }


def _headline_verdict(name: str, block: dict[str, Any], *, label_a: str, label_b: str) -> dict[str, Any]:
    a_wins = [key for key, entry in block.items() if entry.get("winner") == "a"]
    b_wins = [key for key, entry in block.items() if entry.get("winner") == "b"]
    ties = [key for key, entry in block.items() if entry.get("winner") == "tie"]
    a_cos = [key for key in a_wins if key in COSINE_KEYS]
    b_cos = [key for key in b_wins if key in COSINE_KEYS]
    a_drift = [key for key in a_wins if key in DRIFT_KEYS]
    b_drift = [key for key in b_wins if key in DRIFT_KEYS]
    if len(a_wins) > len(b_wins):
        winner = "a"
    elif len(b_wins) > len(a_wins):
        winner = "b"
    else:
        winner = "tie"
    split = (len(a_cos) > len(b_cos) and len(a_drift) < len(b_drift)) or (len(b_cos) > len(a_cos) and len(b_drift) < len(a_drift))
    verdict = "split" if split and winner != "tie" else "clear" if winner != "tie" else "tie"
    decisive_metric = None
    decisive_margin = None
    if winner in {"a", "b"}:
        winner_keys = a_wins if winner == "a" else b_wins
        if winner_keys:
            decisive_metric = max(winner_keys, key=lambda key: (_metric_advantage(block[key], winner), key))
            decisive_margin = float(_metric_advantage(block[decisive_metric], winner))
    winner_label = label_a if winner == "a" else label_b if winner == "b" else "tie"
    if winner == "tie":
        headline = f"{name} is a tie; cosine and drift lanes split without a decisive edge."
    else:
        lead_bits = []
        if decisive_metric is not None:
            lead_bits.append(f"decisive metric {decisive_metric}")
        if len(a_drift) != len(b_drift):
            drift_label = label_a if len(a_drift) > len(b_drift) else label_b
            lead_bits.append(f"{drift_label} wins the drift lane")
        if len(a_cos) != len(b_cos):
            cosine_label = label_a if len(a_cos) > len(b_cos) else label_b
            lead_bits.append(f"{cosine_label} wins the cosine lane")
        joined = "; ".join(lead_bits) if lead_bits else "no single lane dominates"
        headline = f"{winner_label} wins {name} ({len(a_wins)} to {len(b_wins)} metric wins); {joined}."
    return {
        "transform": name,
        "winner": winner,
        "winner_label": winner_label,
        "verdict": verdict,
        "metric_win_counts": {
            "a": len(a_wins),
            "b": len(b_wins),
            "tie": len(ties),
            "a_cosine": len(a_cos),
            "b_cosine": len(b_cos),
            "a_drift": len(a_drift),
            "b_drift": len(b_drift),
        },
        "decisive_metric": decisive_metric,
        "decisive_margin": decisive_margin,
        "headline": headline,
    }


def compare_runs(
    *,
    label_a: str,
    label_b: str,
    export_a_path: Path,
    export_b_path: Path,
    metamer_a_path: Path,
    metamer_b_path: Path,
    out_dir: Path,
    heldout_a_path: Path | None = None,
    heldout_b_path: Path | None = None,
    benchmark_a_path: Path | None = None,
    benchmark_b_path: Path | None = None,
    continuity_a_path: Path | None = None,
    continuity_b_path: Path | None = None,
) -> dict[str, Any]:
    export_a = _load_json(export_a_path)
    export_b = _load_json(export_b_path)
    metamer_a = _load_json(metamer_a_path)
    metamer_b = _load_json(metamer_b_path)
    heldout_a = _load_json(heldout_a_path) if heldout_a_path else None
    heldout_b = _load_json(heldout_b_path) if heldout_b_path else None
    benchmark_a = _load_json(benchmark_a_path) if benchmark_a_path else None
    benchmark_b = _load_json(benchmark_b_path) if benchmark_b_path else None
    continuity_a = _load_json(continuity_a_path) if continuity_a_path else None
    continuity_b = _load_json(continuity_b_path) if continuity_b_path else None

    export_cmp = _compare_block(
        _export_metrics(export_a),
        _export_metrics(export_b),
        higher_is_better={
            "mean_num_relational_signatures",
            "mean_num_relational_signature_families",
            "mean_relational_signature_confidence",
            "aggregate_num_relational_signature_families",
        },
        lower_is_better={"aggregate_mean_relational_family_size"},
    )
    metamer_cmp = _compare_block(
        _metamer_metrics(metamer_a),
        _metamer_metrics(metamer_b),
        higher_is_better=set(COSINE_KEYS),
        lower_is_better=set(),
        closer_to_zero=set(DRIFT_KEYS),
    )

    heldout_cmp = None
    if heldout_a is not None and heldout_b is not None:
        heldout_cmp = _compare_block(
            _heldout_metrics(heldout_a),
            _heldout_metrics(heldout_b),
            higher_is_better={
                "mean_real_branch_fraction",
                "mean_relational_signature_confidence",
                "mean_relational_signature_q_entropy",
                "mean_relational_branch_mass",
                "mean_num_relational_signature_families",
            },
            lower_is_better={"mean_dominant_relational_family_share"},
        )

    benchmark_cmp = None
    if benchmark_a is not None and benchmark_b is not None:
        benchmark_cmp = _compare_block(
            _benchmark_metrics(benchmark_a),
            _benchmark_metrics(benchmark_b),
            higher_is_better={"mean_corr", "mean_prefix_alignment"},
            lower_is_better={"mean_mae", "mean_prefix_delta"},
        )

    continuity_cmp = None
    if continuity_a is not None and continuity_b is not None:
        continuity_cmp = _compare_block(
            _continuity_metrics(continuity_a),
            _continuity_metrics(continuity_b),
            higher_is_better=set(),
            lower_is_better={
                "mean_loop_autocorr_peak",
                "mean_adjacent_chunk_similarity",
                "mean_nonlocal_chunk_repeat",
                "mean_first_chunk_reentry",
            },
        )

    by_transform: dict[str, Any] = {}
    transform_headlines: list[dict[str, Any]] = []
    transforms = sorted(
        set((metamer_a.get("by_transform") or {}).keys()) | set((metamer_b.get("by_transform") or {}).keys())
    )
    for name in transforms:
        a_row = (metamer_a.get("by_transform") or {}).get(name, {})
        b_row = (metamer_b.get("by_transform") or {}).get(name, {})
        block = _compare_block(
            _transform_metric_row(a_row),
            _transform_metric_row(b_row),
            higher_is_better=set(COSINE_KEYS),
            lower_is_better=set(),
            closer_to_zero=set(DRIFT_KEYS),
        )
        headline = _headline_verdict(name, block, label_a=label_a, label_b=label_b)
        block["headline_verdict"] = headline
        by_transform[name] = block
        transform_headlines.append(headline)

    transform_headlines.sort(
        key=lambda item: (
            0 if item["winner"] == "a" else 1 if item["winner"] == "b" else 2,
            -abs(float(item["decisive_margin"])) if item.get("decisive_margin") is not None else 0.0,
            str(item["transform"]),
        )
    )
    transform_winner_tally = {
        "a": sum(1 for item in transform_headlines if item["winner"] == "a"),
        "b": sum(1 for item in transform_headlines if item["winner"] == "b"),
        "tie": sum(1 for item in transform_headlines if item["winner"] == "tie"),
        "split": sum(1 for item in transform_headlines if item["verdict"] == "split"),
    }

    takeaways: list[str] = []
    if export_cmp["mean_num_relational_signature_families"]["winner"] == "a":
        takeaways.append(f"{label_a} is more signature-diverse on the anchor export.")
    elif export_cmp["mean_num_relational_signature_families"]["winner"] == "b":
        takeaways.append(f"{label_b} is more signature-diverse on the anchor export.")
    if export_cmp["mean_relational_signature_confidence"]["winner"] == "a":
        takeaways.append(f"{label_a} carries higher mean signature confidence.")
    elif export_cmp["mean_relational_signature_confidence"]["winner"] == "b":
        takeaways.append(f"{label_b} carries higher mean signature confidence.")
    if metamer_cmp["mean_family_count_delta"]["winner"] == "a":
        takeaways.append(f"{label_a} is more metamer-stable in family assignment under transforms.")
    elif metamer_cmp["mean_family_count_delta"]["winner"] == "b":
        takeaways.append(f"{label_b} is more metamer-stable in family assignment under transforms.")
    if metamer_cmp["mean_confidence_delta"]["winner"] == "a":
        takeaways.append(f"{label_a} keeps confidence drift closer to zero across metamers.")
    elif metamer_cmp["mean_confidence_delta"]["winner"] == "b":
        takeaways.append(f"{label_b} keeps confidence drift closer to zero across metamers.")
    if heldout_cmp is not None:
        if heldout_cmp["mean_real_branch_fraction"]["winner"] == "a":
            takeaways.append(f"{label_a} keeps stronger held-out real branching.")
        elif heldout_cmp["mean_real_branch_fraction"]["winner"] == "b":
            takeaways.append(f"{label_b} keeps stronger held-out real branching.")
    if transform_winner_tally["a"] > transform_winner_tally["b"]:
        takeaways.append(f"{label_a} wins more individual transform verdicts ({transform_winner_tally['a']} vs {transform_winner_tally['b']}).")
    elif transform_winner_tally["b"] > transform_winner_tally["a"]:
        takeaways.append(f"{label_b} wins more individual transform verdicts ({transform_winner_tally['b']} vs {transform_winner_tally['a']}).")

    summary = {
        "labels": {"a": label_a, "b": label_b},
        "paths": {
            "export_a": str(export_a_path),
            "export_b": str(export_b_path),
            "metamer_a": str(metamer_a_path),
            "metamer_b": str(metamer_b_path),
            "heldout_a": str(heldout_a_path) if heldout_a_path else None,
            "heldout_b": str(heldout_b_path) if heldout_b_path else None,
            "benchmark_a": str(benchmark_a_path) if benchmark_a_path else None,
            "benchmark_b": str(benchmark_b_path) if benchmark_b_path else None,
            "continuity_a": str(continuity_a_path) if continuity_a_path else None,
            "continuity_b": str(continuity_b_path) if continuity_b_path else None,
        },
        "export_compare": export_cmp,
        "heldout_compare": heldout_cmp,
        "benchmark_compare": benchmark_cmp,
        "continuity_compare": continuity_cmp,
        "metamer_compare": metamer_cmp,
        "by_transform_compare": by_transform,
        "transform_headlines": transform_headlines,
        "transform_winner_tally": transform_winner_tally,
        "takeaways": takeaways,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "relational_signature_compare.json"
    md_path = out_dir / "RELATIONAL_SIGNATURE_COMPARE.md"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md_lines = [
        "# Circleworld Relational Signature Comparison",
        "",
        f"- A: `{label_a}`",
        f"- B: `{label_b}`",
        "",
        "## Takeaways",
        "",
    ]
    md_lines.extend([f"- {line}" for line in takeaways] or ["- No strong directional takeaway found."])
    md_lines.extend(
        [
            "",
            "## Export Metrics",
            "",
            f"- mean signature families: `{label_a}={export_cmp['mean_num_relational_signature_families']['a']:.4f}` vs `{label_b}={export_cmp['mean_num_relational_signature_families']['b']:.4f}`",
            f"- mean signature confidence: `{label_a}={export_cmp['mean_relational_signature_confidence']['a']:.4f}` vs `{label_b}={export_cmp['mean_relational_signature_confidence']['b']:.4f}`",
            f"- aggregate signature families: `{label_a}={export_cmp['aggregate_num_relational_signature_families']['a']:.0f}` vs `{label_b}={export_cmp['aggregate_num_relational_signature_families']['b']:.0f}`",
            "",
            "## Metamer Stability",
            "",
            f"- pooled cosine: `{label_a}={metamer_cmp['mean_pooled_full_cos']['a']:.4f}` vs `{label_b}={metamer_cmp['mean_pooled_full_cos']['b']:.4f}`",
            f"- prefix cosine: `{label_a}={metamer_cmp['mean_prefix_cos']['a']:.4f}` vs `{label_b}={metamer_cmp['mean_prefix_cos']['b']:.4f}`",
            f"- family-count drift (closer to zero wins): `{label_a}={metamer_cmp['mean_family_count_delta']['a']:+.4f}` vs `{label_b}={metamer_cmp['mean_family_count_delta']['b']:+.4f}`",
            f"- confidence drift (closer to zero wins): `{label_a}={metamer_cmp['mean_confidence_delta']['a']:+.4f}` vs `{label_b}={metamer_cmp['mean_confidence_delta']['b']:+.4f}`",
            "",
        ]
    )
    if heldout_cmp is not None:
        md_lines.extend(
            [
                "## Held-out",
                "",
                f"- real branch fraction: `{label_a}={heldout_cmp['mean_real_branch_fraction']['a']:.4f}` vs `{label_b}={heldout_cmp['mean_real_branch_fraction']['b']:.4f}`",
                f"- held-out signature families: `{label_a}={heldout_cmp['mean_num_relational_signature_families']['a']:.4f}` vs `{label_b}={heldout_cmp['mean_num_relational_signature_families']['b']:.4f}`",
                f"- held-out signature confidence: `{label_a}={heldout_cmp['mean_relational_signature_confidence']['a']:.4f}` vs `{label_b}={heldout_cmp['mean_relational_signature_confidence']['b']:.4f}`",
                "",
            ]
        )
    if benchmark_cmp is not None:
        md_lines.extend(
            [
                "## Benchmark",
                "",
                f"- corr: `{label_a}={benchmark_cmp['mean_corr']['a']:.4f}` vs `{label_b}={benchmark_cmp['mean_corr']['b']:.4f}`",
                f"- mae: `{label_a}={benchmark_cmp['mean_mae']['a']:.4f}` vs `{label_b}={benchmark_cmp['mean_mae']['b']:.4f}`",
                "",
            ]
        )
    md_lines.extend(["## Transform Verdicts", ""])
    for item in transform_headlines:
        md_lines.append(f"- `{item['transform']}`: {item['headline']}")
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare two Circleworld relational-signature artifact sets.")
    ap.add_argument("--label-a", required=True)
    ap.add_argument("--label-b", required=True)
    ap.add_argument("--export-a", required=True)
    ap.add_argument("--export-b", required=True)
    ap.add_argument("--metamer-a", required=True)
    ap.add_argument("--metamer-b", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--heldout-a", default=None)
    ap.add_argument("--heldout-b", default=None)
    ap.add_argument("--benchmark-a", default=None)
    ap.add_argument("--benchmark-b", default=None)
    ap.add_argument("--continuity-a", default=None)
    ap.add_argument("--continuity-b", default=None)
    args = ap.parse_args()

    summary = compare_runs(
        label_a=args.label_a,
        label_b=args.label_b,
        export_a_path=Path(args.export_a),
        export_b_path=Path(args.export_b),
        metamer_a_path=Path(args.metamer_a),
        metamer_b_path=Path(args.metamer_b),
        out_dir=Path(args.out_dir),
        heldout_a_path=Path(args.heldout_a) if args.heldout_a else None,
        heldout_b_path=Path(args.heldout_b) if args.heldout_b else None,
        benchmark_a_path=Path(args.benchmark_a) if args.benchmark_a else None,
        benchmark_b_path=Path(args.benchmark_b) if args.benchmark_b else None,
        continuity_a_path=Path(args.continuity_a) if args.continuity_a else None,
        continuity_b_path=Path(args.continuity_b) if args.continuity_b else None,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
