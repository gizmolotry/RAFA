import argparse
import asyncio
from collections import deque
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from scrape_core import AudioCandidate, ScrapeConfig, ScrapePipeline, print_progress


BASE = "https://www.wavsource.com/"


def _get_soup(session: requests.Session, url: str) -> BeautifulSoup:
    r = session.get(url, timeout=30)
    r.raise_for_status()
    return BeautifulSoup(r.text, "html.parser")


def _same_host(url: str) -> bool:
    return urlparse(url).netloc.endswith("wavsource.com")


def build_candidates(max_pages: int = 3500) -> list[AudioCandidate]:
    out: list[AudioCandidate] = []
    seen_pages: set[str] = set()
    seen_audio: set[str] = set()
    q = deque([BASE])

    with requests.Session() as session:
        session.headers.update({"User-Agent": "Mozilla/5.0"})
        while q and len(seen_pages) < max_pages:
            page_url = q.popleft()
            if page_url in seen_pages:
                continue
            seen_pages.add(page_url)

            try:
                soup = _get_soup(session, page_url)
            except Exception:
                continue

            title_el = soup.select_one("title")
            style = title_el.get_text(" ", strip=True)[:80] if title_el else "unknown"

            for a in soup.select("a[href]"):
                href = (a.get("href") or "").strip()
                if not href:
                    continue
                full = urljoin(page_url, href)
                if not _same_host(full):
                    continue
                low = full.lower()
                if low.endswith((".htm", ".html")):
                    if full not in seen_pages:
                        q.append(full)
                    continue
                if ".wav" not in low and ".mp3" not in low:
                    continue
                if full in seen_audio:
                    continue
                seen_audio.add(full)
                title = a.get_text(" ", strip=True) or low.split("/")[-1]
                out.append(
                    AudioCandidate(
                        source="wavsource",
                        page_url=page_url,
                        audio_url=full,
                        title=title,
                        style=style,
                        tags="",
                    )
                )

            if len(seen_pages) % 100 == 0:
                print(f"[wavsource] crawled_pages={len(seen_pages)} candidates={len(out)}")

    print(f"[wavsource] final_pages={len(seen_pages)}")
    return out


async def main() -> None:
    ap = argparse.ArgumentParser(description="Scrape WavSource into normalized local wavs.")
    ap.add_argument("--max-pages", type=int, default=3500)
    ap.add_argument("--target-count", type=int, default=3000)
    ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--min-rms", type=float, default=0.005)
    args = ap.parse_args()

    candidates = build_candidates(max_pages=args.max_pages)
    print(f"[wavsource] candidates={len(candidates)}")

    cfg = ScrapeConfig(concurrency=args.concurrency, min_rms=args.min_rms)
    pipe = ScrapePipeline(cfg)
    result = await pipe.run(candidates, target_count=args.target_count, progress_cb=print_progress)
    print(f"[wavsource] done {result}")


if __name__ == "__main__":
    asyncio.run(main())

