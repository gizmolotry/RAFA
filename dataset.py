from pathlib import Path
from fnmatch import fnmatch
import csv
import wave

import numpy as np
import torch
import torch.nn.functional as F
# BLACKWELL FIX: No torchaudio allowed
# import torchaudio
from torch.utils.data import Dataset

from stft_utils import compute_stft

def native_resample(wav: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
    if orig_sr == target_sr:
        return wav
    # Linear interpolation for Blackwell compatibility
    # [1, T] -> [1, 1, T] -> [1, 1, T_new] -> [1, T_new]
    ratio = float(target_sr) / float(orig_sr)
    target_len = int(wav.size(1) * ratio)
    return F.interpolate(wav.unsqueeze(0), size=target_len, mode="linear", align_corners=False).squeeze(0)


class RafaGuerrillaDataset(Dataset):
    """GTZAN-backed dataset for RAFA training."""

    GENRES = [
        "blues",
        "classical",
        "country",
        "disco",
        "hiphop",
        "jazz",
        "metal",
        "pop",
        "reggae",
        "rock",
    ]

    def __init__(self, cfg, split: str | None = None):
        super().__init__()
        self.sr = int(cfg["data"]["sample_rate"])
        self.stft_cfg = cfg["data"]["stft"]
        self.clip_seconds = int(cfg["data"].get("clip_seconds", 5))
        self.local_min_rms = float(cfg["data"].get("local_min_rms", 0.0))
        self.local_target_rms = float(cfg["data"].get("local_target_rms", 0.0))
        self.exclude_globs = list(
            cfg["data"].get(
                "exclude_globs",
                ["outputs/**", "**/sample_*.wav", "**/reconstructed*.wav"],
            )
        )
        self.use_local_files = False
        self.local_files: list[Path] = []
        self.local_labels: list[str] = []
        self.local_prompts: list[str] = []

        gtzan_root = cfg["data"].get("gtzan_root", "datasets/GTZAN")
        gtzan_subset = split or cfg["data"].get("gtzan_subset", "training")
        gtzan_download = bool(cfg["data"].get("gtzan_download", False))
        local_wav_root = cfg["data"].get("local_wav_root", "wav_files")

        self.label_to_id = {name: i for i, name in enumerate(self.GENRES)}
        self.load_tension = bool(cfg["data"].get("load_tension", False))
        self.tension_root = Path("datasets/tension_tracks")
        force_local_wavs = bool(cfg["data"].get("force_local_wavs", False))
        local_root = Path(local_wav_root)
        manifest_path = self._resolve_manifest_path(cfg, split)

        if manifest_path is not None:
            valid, labels, prompts = self._collect_manifest_wavs(manifest_path, local_root)
            if not valid:
                raise RuntimeError(
                    f"Manifest configured but no decodable WAV files passed QC from {manifest_path}."
                )
            self.use_local_files = True
            self.local_files = valid
            self.local_labels = labels
            self.local_prompts = prompts
            self.ds = None
            return

        if force_local_wavs:
            valid = self._collect_local_wavs(local_root)
            if not valid:
                raise RuntimeError(
                    "force_local_wavs=true but no decodable local WAV files found under "
                    f"{local_root.resolve() if local_root.exists() else local_root}."
                )
            self.use_local_files = True
            self.local_files = valid
            self.local_labels = ["unknown"] * len(valid)
            self.local_prompts = ["unknown audio"] * len(valid)
            self.ds = None
            return

        try:
            # self.ds = GTZAN(root=gtzan_root, subset=gtzan_subset, download=gtzan_download)
            self.ds = None
            valid = self._collect_local_wavs(local_root)
        except Exception:
            valid = self._collect_local_wavs(local_root)
            if not valid:
                raise RuntimeError(
                    "GTZAN unavailable and no decodable local WAV files found under "
                    f"{local_root.resolve() if local_root.exists() else local_root}."
                )
            self.use_local_files = True
            self.local_files = valid
            self.local_labels = ["unknown"] * len(valid)
            self.local_prompts = ["unknown audio"] * len(valid)
            self.ds = None

    def _build_manifest_prompt(self, row: dict[str, str]) -> str:
        label = str(row.get("label", "")).strip().lower()
        style = str(row.get("style", "")).strip().lower()
        source = str(row.get("source", "")).strip().lower()
        if label and style and label != style:
            base = f"{style} {label} sound"
        elif style:
            base = f"{style} sound"
        elif label:
            base = f"{label} sound"
        else:
            base = "unknown audio"
        if source:
            return f"{base}, source {source}"
        return base

    def _resolve_manifest_path(self, cfg, split: str | None) -> Path | None:
        data_cfg = cfg.get("data", {})
        split_tag = str(split or data_cfg.get("gtzan_subset", "training")).strip().lower()
        manifest_train = data_cfg.get("manifest_train_csv")
        manifest_val = data_cfg.get("manifest_val_csv")
        manifest_all = data_cfg.get("manifest_csv")

        if split_tag in ("val", "validation", "dev") and manifest_val:
            p = Path(str(manifest_val))
            if p.exists():
                return p
        if split_tag in ("train", "training") and manifest_train:
            p = Path(str(manifest_train))
            if p.exists():
                return p
        if manifest_all:
            p = Path(str(manifest_all))
            if p.exists():
                return p
        return None

    def _collect_local_wavs(self, local_root: Path) -> list[Path]:
        candidates = sorted(local_root.rglob("*.wav")) if local_root.exists() else []
        valid: list[Path] = []
        for p in candidates:
            rel = p.relative_to(local_root).as_posix()
            if self._is_excluded(rel):
                continue
            # Skip expensive RMS check here, do it in __getitem__ if needed
            valid.append(p)
        return valid

    def _collect_manifest_wavs(self, manifest_path: Path, local_root: Path) -> tuple[list[Path], list[str], list[str]]:
        valid: list[Path] = []
        labels: list[str] = []
        prompts: list[str] = []
        with manifest_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None or "wav_path" not in reader.fieldnames:
                raise RuntimeError(f"Manifest {manifest_path} missing required column wav_path")
            for row in reader:
                wav_path = str(row.get("wav_path", "")).strip()
                if not wav_path:
                    continue
                p = Path(wav_path)
                if not p.is_absolute():
                    direct = p.resolve()
                    p = direct if direct.exists() else (local_root / p).resolve()
                
                # Fast check only
                if not p.exists() or p.suffix.lower() != ".wav":
                    continue

                rel = p.as_posix()
                if local_root.exists():
                    try:
                        rel = p.relative_to(local_root).as_posix()
                    except Exception:
                        rel = p.as_posix()
                if self._is_excluded(rel):
                    continue

                valid.append(p)
                label = str(row.get("style", "") or row.get("label", "")).strip().lower() or "unknown"
                labels.append(label)
                prompts.append(self._build_manifest_prompt(row))
        return valid, labels, prompts

    def _is_excluded(self, rel_path: str) -> bool:
        for pat in self.exclude_globs:
            pat = str(pat).replace("\\", "/")
            if fnmatch(rel_path, pat):
                return True
            if pat.startswith("**/") and fnmatch(rel_path, pat[3:]):
                return True
        return False

    def __len__(self):
        return len(self.local_files) if self.use_local_files else len(self.ds)

    def _load_local_pcm_wav(self, path: Path) -> tuple[torch.Tensor, int]:
        with wave.open(str(path), "rb") as wf:
            sr = int(wf.getframerate())
            ch = int(wf.getnchannels())
            sampwidth = int(wf.getsampwidth())
            nframes = int(wf.getnframes())
            raw = wf.readframes(nframes)

        if sampwidth == 1:
            arr = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
            arr = (arr - 128.0) / 128.0
        elif sampwidth == 2:
            arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        elif sampwidth == 3:
            b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
            v = (
                b[:, 0].astype(np.int32)
                | (b[:, 1].astype(np.int32) << 8)
                | (b[:, 2].astype(np.int32) << 16)
            )
            sign = 1 << 23
            v = (v ^ sign) - sign
            arr = v.astype(np.float32) / 8388608.0
        elif sampwidth == 4:
            arr = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            raise RuntimeError(f"Unsupported WAV sample width: {sampwidth} bytes")

        arr = arr.reshape(-1, ch).T
        wav = torch.from_numpy(arr)
        return wav, sr

    def __getitem__(self, idx: int):
        if self.use_local_files:
            wav, sr_orig = self._load_local_pcm_wav(self.local_files[idx])
            label = self.local_labels[idx] if idx < len(self.local_labels) else "unknown"
            prompt_text = self.local_prompts[idx] if idx < len(self.local_prompts) else str(label)
        else:
            wav, sr_orig, label = self.ds[idx]
            prompt_text = str(label)
        wav = wav.float()

        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        if int(sr_orig) != self.sr:
            wav = native_resample(wav, int(sr_orig), self.sr)

        target_len = self.sr * self.clip_seconds
        if wav.shape[1] < target_len:
            wav = torch.nn.functional.pad(wav, (0, target_len - wav.shape[1]))
        else:
            wav = wav[:, :target_len]

        if self.local_target_rms > 0.0:
            rms = torch.sqrt(torch.mean(wav**2) + 1e-12)
            wav = wav * (self.local_target_rms / rms.clamp_min(1e-6))
            wav = wav.clamp(-1.0, 1.0)

        mag, phase = compute_stft(wav, self.stft_cfg)

        label_id = self.label_to_id.get(str(label).lower(), -1)
        label_tensor = torch.tensor(label_id, dtype=torch.long)
        prompt_text = str(prompt_text).strip().lower() or "unknown"

        if self.load_tension and self.use_local_files:
            wav_p = self.local_files[idx]
            # Find the relative path to the wav_files root
            # local_wav_root is set in __init__
            root = Path(self.tension_root).parent / "wav_files" # tension_root is datasets/tension_tracks
            # Wait, better use the root we know:
            try:
                # Iterate up to find 'wav_files'
                curr = wav_p.parent
                rel_p = wav_p.name
                while curr.name and curr.name != "wav_files" and curr != curr.parent:
                    curr = curr.parent
                if curr.name == "wav_files":
                    rel_p = wav_p.relative_to(curr)
                
                track_p = self.tension_root / Path(rel_p).with_suffix(".npy")
                if track_p.exists():
                    tension = torch.from_numpy(np.load(track_p).astype(np.float32))
                    return mag.squeeze(0), phase.squeeze(0), label_tensor, prompt_text, tension
            except Exception as e:
                pass
        
        # Fallback if no tension or not requested
        return mag.squeeze(0), phase.squeeze(0), label_tensor, prompt_text, torch.zeros(mag.shape[-1])
