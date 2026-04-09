from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable


def _hub_root() -> Path:
    if os.environ.get("HUGGINGFACE_HUB_CACHE"):
        return Path(os.environ["HUGGINGFACE_HUB_CACHE"])
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home) / "hub"
    user_profile = os.environ.get("USERPROFILE") or str(Path.home())
    return Path(user_profile) / ".cache" / "huggingface" / "hub"


def _repo_cache_dir(repo_id: str) -> Path:
    return _hub_root() / f"models--{repo_id.replace('/', '--')}"


def _snapshot_candidates(repo_id: str) -> list[Path]:
    repo_dir = _repo_cache_dir(repo_id)
    snapshots_dir = repo_dir / "snapshots"
    if not snapshots_dir.exists():
        return []

    out: list[Path] = []
    refs_main = repo_dir / "refs" / "main"
    if refs_main.exists():
        try:
            ref_name = refs_main.read_text(encoding="utf-8").strip()
            if ref_name:
                ref_path = snapshots_dir / ref_name
                if ref_path.exists():
                    out.append(ref_path)
        except OSError:
            pass

    others = sorted(
        [p for p in snapshots_dir.iterdir() if p.is_dir()],
        key=lambda p: p.stat().st_mtime_ns,
        reverse=True,
    )
    seen = {str(p).lower() for p in out}
    for p in others:
        key = str(p).lower()
        if key not in seen:
            out.append(p)
            seen.add(key)
    return out


def _has_any(path: Path, names: Iterable[str]) -> bool:
    return any((path / name).exists() for name in names)


def resolve_hf_pretrained_path(
    repo_id: str,
    *,
    need_model: bool = False,
    need_tokenizer: bool = False,
) -> str:
    direct = Path(repo_id)
    if direct.exists():
        return str(direct)

    for snap in _snapshot_candidates(repo_id):
        has_config = (snap / "config.json").exists()
        has_model = _has_any(snap, ("pytorch_model.bin", "model.safetensors"))
        has_tokenizer = _has_any(snap, ("tokenizer.json", "tokenizer_config.json", "vocab.json"))
        if need_model and not (has_config and has_model):
            continue
        if need_tokenizer and not has_tokenizer:
            continue
        if need_model and need_tokenizer and not (has_config and has_model and has_tokenizer):
            continue
        return str(snap)

    return repo_id
