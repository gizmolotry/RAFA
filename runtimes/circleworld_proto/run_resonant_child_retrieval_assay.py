from __future__ import annotations

import argparse
import json
from pathlib import Path

from resonant_law_objects import (
    composition_summary,
    load_law_objects_from_json,
    object_family_value,
    run_retrieval_assay,
    synthetic_law_objects,
    write_assay_report,
)


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _expand_inputs(paths: list[str]) -> list[Path]:
    out: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            out.extend(sorted(path.rglob("nested_commitment_summary.json")))
            out.extend(sorted(path.rglob("nested_commitment_report.json")))
            continue
        if path.exists():
            out.append(path)
    dedup: dict[str, Path] = {}
    for path in out:
        dedup[str(path.resolve()).lower()] = path
    return list(dedup.values())


def _case_retrieval(objects: list[dict], *, top_k: int, family_key_field: str = "family_key") -> dict:
    grouped: dict[str, list[dict]] = {}
    for obj in objects:
        case = str(obj.get("case", "unknown_case"))
        grouped.setdefault(case, []).append(obj)
    rows = []
    for case, case_objects in sorted(grouped.items()):
        retrieval = run_retrieval_assay(
            case_objects,
            top_k=top_k,
            family_key_field=family_key_field,
        )["summary"]
        composition = composition_summary(case_objects)
        rows.append(
            {
                "case": case,
                "object_count": len(case_objects),
                "family_count": retrieval.get("family_count", 0),
                "eligible_excluding_self_count": retrieval.get("eligible_excluding_self_count", 0),
                "status": retrieval.get("status", "unknown"),
                "top1_family_accuracy_excluding_self": retrieval.get(
                    "top1_family_accuracy_excluding_self", 0.0
                ),
                "topk_family_accuracy": retrieval.get("topk_family_accuracy", 0.0),
                "mean_resonance_margin": retrieval.get("mean_resonance_margin", 0.0),
                "mean_resonance_entropy": retrieval.get("mean_resonance_entropy", 0.0),
                "cross_depth_pair_count": composition.get("cross_depth_pair_count", 0),
                "mean_cross_depth_geometric_score": composition.get("mean_cross_depth_geometric_score", 0.0),
            }
        )
    return {
        "case_count": len(rows),
        "passing_case_count": sum(1 for row in rows if str(row["status"]).startswith("pass")),
        "mean_case_top1_family_accuracy_excluding_self": _mean(
            [float(row["top1_family_accuracy_excluding_self"]) for row in rows]
        ),
        "mean_case_resonance_margin": _mean([float(row["mean_resonance_margin"]) for row in rows]),
        "rows": rows,
    }


def _top_candidate_case_summary(objects: list[dict], retrieval: dict) -> dict:
    by_id = {str(obj.get("object_id", "")): obj for obj in objects if obj.get("object_id")}
    rows = []
    for row in retrieval.get("rows", []) or []:
        query = by_id.get(str(row.get("query_object_id", "")), {})
        top = by_id.get(str(row.get("top_object_id", "")), {})
        query_case = str(query.get("case", ""))
        top_case = str(top.get("case", ""))
        rows.append(
            {
                "query_case": query_case,
                "top_case": top_case,
                "same_case": 1.0 if query_case and query_case == top_case else 0.0,
                "cross_case": 1.0 if query_case and top_case and query_case != top_case else 0.0,
                "same_local_family": 1.0
                if object_family_value(query, "local_family_key") == object_family_value(top, "local_family_key")
                else 0.0,
                "same_structural_family": 1.0
                if object_family_value(query, "structural_family_key")
                == object_family_value(top, "structural_family_key")
                else 0.0,
            }
        )
    return {
        "row_count": len(rows),
        "top_same_case_fraction": _mean([float(row["same_case"]) for row in rows]),
        "top_cross_case_fraction": _mean([float(row["cross_case"]) for row in rows]),
        "top_same_local_family_fraction": _mean([float(row["same_local_family"]) for row in rows]),
        "top_same_structural_family_fraction": _mean([float(row["same_structural_family"]) for row in rows]),
    }


def _retrieval_scopes(
    objects: list[dict],
    *,
    top_k: int,
    local_retrieval: dict,
    local_case_retrieval: dict,
) -> dict:
    scopes = {
        "local": {
            "family_key_field": "local_family_key",
            "summary": local_retrieval["summary"],
            "case_retrieval": local_case_retrieval,
            "top_candidate_case_summary": _top_candidate_case_summary(objects, local_retrieval),
        }
    }
    for name, field in (("structural", "structural_family_key"),):
        retrieval = run_retrieval_assay(objects, top_k=top_k, family_key_field=field)
        scopes[name] = {
            "family_key_field": field,
            "summary": retrieval["summary"],
            "case_retrieval": _case_retrieval(objects, top_k=top_k, family_key_field=field),
            "top_candidate_case_summary": _top_candidate_case_summary(objects, retrieval),
        }
    return scopes


def run_assay(*, inputs: list[str], out_dir: Path, top_k: int, synthetic_smoke: bool = False) -> dict[str, str]:
    source_files: list[str] = []
    objects = []
    for path in _expand_inputs(inputs):
        loaded = load_law_objects_from_json(path)
        if loaded:
            source_files.append(str(path))
            objects.extend(loaded)
    if synthetic_smoke or not objects:
        source_files.append("synthetic_smoke")
        objects.extend(synthetic_law_objects())
    deduped = {}
    for obj in objects:
        object_id = str(obj.get("object_id", ""))
        if not object_id:
            continue
        deduped.setdefault(object_id, obj)
    objects = list(deduped.values())

    retrieval = run_retrieval_assay(objects, top_k=top_k)
    composition = composition_summary(objects)
    case_retrieval = _case_retrieval(objects, top_k=top_k)
    retrieval_scopes = _retrieval_scopes(
        objects,
        top_k=top_k,
        local_retrieval=retrieval,
        local_case_retrieval=case_retrieval,
    )
    payload = {
        "source_files": source_files,
        "top_k": int(top_k),
        "object_count": len(objects),
        "objects": objects,
        "retrieval": retrieval,
        "composition": composition,
        "case_retrieval": case_retrieval,
        "retrieval_scopes": retrieval_scopes,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "resonant_child_retrieval_assay.json"
    md_path = out_dir / "RESONANT_CHILD_RETRIEVAL_ASSAY.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_assay_report(md_path, payload)
    return {
        "json": str(json_path),
        "markdown": str(md_path),
        "status": str(retrieval["summary"]["status"]),
        "object_count": str(len(objects)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the Circleworld resonant child retrieval assay.")
    ap.add_argument("--input", nargs="*", default=[], help="Nested summary/report JSON files or directories.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--synthetic-smoke", action="store_true")
    args = ap.parse_args()
    print(
        json.dumps(
            run_assay(
                inputs=[str(item) for item in args.input],
                out_dir=Path(args.out_dir),
                top_k=int(args.top_k),
                synthetic_smoke=bool(args.synthetic_smoke),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
