import asyncio
import csv
import hashlib
import json
import os
import random
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Iterable, Optional
from urllib.parse import parse_qs, urlparse
from urllib.robotparser import RobotFileParser

import httpx
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)


def sanitize_filename(text: str) -> str:
    text = re.sub(r"[\\/*?:\"<>|]+", "", text or "")
    text = re.sub(r"\s+", " ", text).strip()
    return text[:140] or "untitled"


def make_slug(text: str) -> str:
    text = sanitize_filename(text).lower()
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return text[:100] or "audio"


def _guess_ext(url: str, content_type: str, content_disposition: str = "") -> str:
    u = url.lower()
    ct = (content_type or "").lower()
    cd = (content_disposition or "").lower()
    parsed = urlparse(url)
    q = parse_qs(parsed.query)
    q_type = (q.get("type", [""])[0] or "").lower()
    if q_type in ("wav", "wave"):
        return ".wav"
    if q_type in ("mp3", "mpeg"):
        return ".mp3"
    if ".wav" in cd:
        return ".wav"
    if ".mp3" in cd:
        return ".mp3"
    if ".wav" in u or "wav" in ct:
        return ".wav"
    if ".mp3" in u or "mpeg" in ct or "mp3" in ct:
        return ".mp3"
    return ".bin"


def _looks_like_audio(raw: bytes, ext: str) -> bool:
    if len(raw) < 16:
        return False
    if ext == ".wav":
        return raw[:4] == b"RIFF" and raw[8:12] == b"WAVE"
    if ext == ".mp3":
        return raw[:3] == b"ID3" or (raw[0] == 0xFF and (raw[1] & 0xE0) == 0xE0)
    return raw[:4] == b"RIFF" or raw[:3] == b"ID3" or (raw[0] == 0xFF and (raw[1] & 0xE0) == 0xE0)


@dataclass
class AudioCandidate:
    source: str
    page_url: str
    audio_url: str
    title: str
    style: str = "unknown"
    tags: str = ""


@dataclass
class ScrapeConfig:
    out_dir: str = "wav_files"
    logs_dir: str = "logs"
    temp_dir: str = "tmp/scrape_audio"
    sample_rate: int = 16000
    min_seconds: float = 1.0
    max_seconds: float = 10.0
    min_rms: float = 0.005
    timeout_s: float = 40.0
    max_retries: int = 5
    concurrency: int = 16
    respect_robots: bool = True
    metadata_csv: str = "logs/scrape_metadata.csv"
    metadata_jsonl: str = "logs/scrape_metadata.jsonl"
    failures_log: str = "logs/scrape_failures.log"


class ScrapePipeline:
    def __init__(self, cfg: ScrapeConfig):
        self.cfg = cfg
        self.out_dir = Path(cfg.out_dir)
        self.logs_dir = Path(cfg.logs_dir)
        self.temp_dir = Path(cfg.temp_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.failures_log = Path(cfg.failures_log)
        self.metadata_csv_path = Path(cfg.metadata_csv)
        self.metadata_jsonl_path = Path(cfg.metadata_jsonl)
        self._robots: dict[str, RobotFileParser] = {}
        self._meta_lock = asyncio.Lock()
        self._seen_audio_urls: set[str] = set()

    async def _is_allowed_by_robots(self, client: httpx.AsyncClient, url: str) -> bool:
        if not self.cfg.respect_robots:
            return True
        parsed = urlparse(url)
        base = f"{parsed.scheme}://{parsed.netloc}"
        if base not in self._robots:
            rp = RobotFileParser()
            robots_url = f"{base}/robots.txt"
            try:
                resp = await client.get(robots_url, timeout=self.cfg.timeout_s)
                if resp.status_code == 200:
                    rp.parse(resp.text.splitlines())
                else:
                    rp = RobotFileParser()
                    rp.parse([])
            except Exception:
                rp = RobotFileParser()
                rp.parse([])
            self._robots[base] = rp
        return self._robots[base].can_fetch(USER_AGENT, url)

    async def _log_failure(self, source: str, url: str, reason: str) -> None:
        line = f"{source}\t{url}\t{reason}\n"
        async with self._meta_lock:
            with self.failures_log.open("a", encoding="utf-8") as f:
                f.write(line)

    async def _append_metadata(self, row: dict) -> None:
        async with self._meta_lock:
            write_header = not self.metadata_csv_path.exists()
            with self.metadata_csv_path.open("a", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "wav_path",
                        "source",
                        "style",
                        "tags",
                        "title",
                        "duration_sec",
                        "rms",
                        "audio_url",
                        "page_url",
                    ],
                )
                if write_header:
                    writer.writeheader()
                writer.writerow(row)
            with self.metadata_jsonl_path.open("a", encoding="utf-8") as jf:
                jf.write(json.dumps(row, ensure_ascii=True) + "\n")

    async def _fetch_bytes(self, client: httpx.AsyncClient, url: str) -> tuple[bytes, str, str]:
        delay = 0.8
        last_err = "unknown"
        for _ in range(self.cfg.max_retries):
            try:
                resp = await client.get(url, follow_redirects=True, timeout=self.cfg.timeout_s)
                if resp.status_code in (429, 500, 502, 503, 504):
                    last_err = f"http_{resp.status_code}"
                    await asyncio.sleep(delay + random.uniform(0.0, 0.5))
                    delay = min(delay * 2, 8.0)
                    continue
                resp.raise_for_status()
                return (
                    resp.content,
                    resp.headers.get("content-type", ""),
                    resp.headers.get("content-disposition", ""),
                )
            except Exception as exc:
                last_err = str(exc)
                await asyncio.sleep(delay + random.uniform(0.0, 0.5))
                delay = min(delay * 2, 8.0)
        raise RuntimeError(last_err)

    def _temp_path(self, source: str, audio_url: str, ext: str) -> Path:
        digest = hashlib.sha1(audio_url.encode("utf-8")).hexdigest()[:16]
        return self.temp_dir / f"{source}_{digest}{ext}"

    def _out_path(self, cand: AudioCandidate, audio_url: str) -> Path:
        slug = make_slug(cand.title)
        digest = hashlib.sha1(audio_url.encode("utf-8")).hexdigest()[:10]
        return self.out_dir / f"{cand.source}_{slug}_{digest}.wav"

    def _audio_qc(self, in_path: Path, out_path: Path) -> tuple[float, float]:
        wav, sr = sf.read(str(in_path), always_2d=True, dtype="float32")
        wav = np.asarray(wav, dtype=np.float32)
        if wav.shape[1] > 1:
            wav = wav.mean(axis=1, keepdims=True)
        if int(sr) != int(self.cfg.sample_rate):
            wav = resample_poly(wav, up=int(self.cfg.sample_rate), down=int(sr), axis=0).astype(np.float32)
        n = int(wav.shape[0])
        duration = float(n / float(self.cfg.sample_rate))
        if duration < self.cfg.min_seconds:
            raise RuntimeError(f"too_short:{duration:.3f}s")
        max_samples = int(self.cfg.max_seconds * self.cfg.sample_rate)
        if n > max_samples:
            wav = wav[:max_samples, :]
            n = int(wav.shape[0])
            duration = float(n / float(self.cfg.sample_rate))
        rms = float(np.sqrt(np.mean(np.square(wav), dtype=np.float64) + 1e-12))
        if rms < self.cfg.min_rms:
            raise RuntimeError(f"low_rms:{rms:.6f}")
        wav = np.clip(wav, -1.0, 1.0)
        sf.write(str(out_path), wav, self.cfg.sample_rate, subtype="PCM_16")
        return duration, rms

    async def _process_candidate(self, client: httpx.AsyncClient, cand: AudioCandidate) -> bool:
        if cand.audio_url in self._seen_audio_urls:
            return False
        self._seen_audio_urls.add(cand.audio_url)

        allowed = await self._is_allowed_by_robots(client, cand.audio_url)
        if not allowed:
            await self._log_failure(cand.source, cand.audio_url, "blocked_by_robots")
            return False

        try:
            raw, content_type, content_disposition = await self._fetch_bytes(client, cand.audio_url)
        except Exception as exc:
            await self._log_failure(cand.source, cand.audio_url, f"download_error:{exc}")
            return False

        ext = _guess_ext(cand.audio_url, content_type, content_disposition)
        if ext not in (".wav", ".mp3"):
            await self._log_failure(cand.source, cand.audio_url, f"bad_ext:{ext}")
            return False
        if not _looks_like_audio(raw, ext):
            await self._log_failure(cand.source, cand.audio_url, "bad_signature")
            return False

        temp_path = self._temp_path(cand.source, cand.audio_url, ext)
        out_path = self._out_path(cand, cand.audio_url)
        try:
            temp_path.write_bytes(raw)
            duration, rms = await asyncio.to_thread(self._audio_qc, temp_path, out_path)
            await self._append_metadata(
                {
                    "wav_path": str(out_path.as_posix()),
                    "source": cand.source,
                    "style": cand.style or "unknown",
                    "tags": cand.tags or "",
                    "title": cand.title,
                    "duration_sec": f"{duration:.4f}",
                    "rms": f"{rms:.6f}",
                    "audio_url": cand.audio_url,
                    "page_url": cand.page_url,
                }
            )
            return True
        except Exception as exc:
            await self._log_failure(cand.source, cand.audio_url, f"qc_error:{exc}")
            return False
        finally:
            try:
                temp_path.unlink(missing_ok=True)
            except Exception:
                pass

    async def run(
        self,
        candidates: Iterable[AudioCandidate],
        *,
        target_count: Optional[int] = None,
        progress_cb: Optional[Callable[[int, int], None]] = None,
    ) -> dict:
        headers = {"User-Agent": USER_AGENT, "Accept": "*/*"}
        limits = httpx.Limits(max_connections=max(32, self.cfg.concurrency * 2))
        sem = asyncio.Semaphore(self.cfg.concurrency)
        cand_list = list(candidates)

        saved = 0
        attempted = 0

        async with httpx.AsyncClient(headers=headers, limits=limits) as client:
            async def worker(cand: AudioCandidate):
                nonlocal saved, attempted
                async with sem:
                    attempted += 1
                    ok = await self._process_candidate(client, cand)
                    if ok:
                        saved += 1
                    if progress_cb:
                        progress_cb(saved, attempted)

            tasks = []
            for cand in cand_list:
                if target_count is not None and saved >= target_count:
                    break
                tasks.append(asyncio.create_task(worker(cand)))
            if tasks:
                await asyncio.gather(*tasks)

        return {
            "candidates": len(cand_list),
            "attempted": attempted,
            "saved": saved,
            "metadata_csv": str(self.metadata_csv_path),
            "metadata_jsonl": str(self.metadata_jsonl_path),
            "failures_log": str(self.failures_log),
        }


def print_progress(saved: int, attempted: int) -> None:
    if attempted % 25 == 0:
        print(f"[scrape] attempted={attempted} saved={saved}")
