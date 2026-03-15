import argparse
import asyncio
import re
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from scrape_core import AudioCandidate, ScrapeConfig, ScrapePipeline, print_progress


BASE = "https://freewavesamples.com/"


def _get_soup(session: requests.Session, url: str) -> BeautifulSoup:
    r = session.get(url, timeout=30)
    r.raise_for_status()
    return BeautifulSoup(r.text, "html.parser")


def _detect_pages(session: requests.Session, max_pages: int) -> int:
    soup = _get_soup(session, BASE)
    pages = [1]
    for a in soup.select("a.navarrow, a[href*='?page=']"):
        href = (a.get("href") or "").strip()
        m = re.search(r"[?&]page=(\d+)", href)
        if m:
            pages.append(int(m.group(1)))
    return min(max(pages), max_pages)


def build_candidates(max_pages: int = 400) -> list[AudioCandidate]:
    out: list[AudioCandidate] = []
    seen_audio = set()
    with requests.Session() as session:
        session.headers.update({"User-Agent": "Mozilla/5.0"})
        pages = _detect_pages(session, max_pages)
        print(f"[freewavesamples] pages={pages}")
        for page_num in range(1, pages + 1):
            page_url = f"{BASE}?page={page_num}"
            try:
                soup = _get_soup(session, page_url)
            except Exception as exc:
                print(f"[freewavesamples] skip page {page_num}: {exc}")
                continue

            sample_links = [urljoin(BASE, a.get("href", "")) for a in soup.select(".sample h2 a[href]")]
            for sample_url in sample_links:
                try:
                    ss = _get_soup(session, sample_url)
                except Exception:
                    continue

                title_tag = ss.select_one(".sample h2:not(.catname)")
                title = title_tag.get_text(" ", strip=True) if title_tag else "untitled"

                cat_link = ss.select_one(".sample h2.catname a:last-of-type")
                style = cat_link.get_text(" ", strip=True) if cat_link else "unknown"

                tags = []
                for tag_a in ss.select("a[href*='/sample-type/'], a[href*='/instrument/'], a[href*='/source/']"):
                    t = tag_a.get_text(" ", strip=True)
                    if t and t not in tags:
                        tags.append(t)
                tags_csv = ",".join(tags[:10])

                for a in ss.select("a[href]"):
                    href = (a.get("href") or "").strip()
                    if ".wav" not in href.lower():
                        continue
                    audio_url = urljoin(sample_url, href)
                    if audio_url in seen_audio:
                        continue
                    seen_audio.add(audio_url)
                    out.append(
                        AudioCandidate(
                            source="freewavesamples",
                            page_url=sample_url,
                            audio_url=audio_url,
                            title=title,
                            style=style,
                            tags=tags_csv,
                        )
                    )
    return out


async def main() -> None:
    ap = argparse.ArgumentParser(description="Scrape FreeWaveSamples into normalized local wavs.")
    ap.add_argument("--max-pages", type=int, default=400)
    ap.add_argument("--target-count", type=int, default=1500)
    ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--min-rms", type=float, default=0.005)
    args = ap.parse_args()

    candidates = build_candidates(max_pages=args.max_pages)
    print(f"[freewavesamples] candidates={len(candidates)}")

    cfg = ScrapeConfig(concurrency=args.concurrency, min_rms=args.min_rms)
    pipe = ScrapePipeline(cfg)
    result = await pipe.run(candidates, target_count=args.target_count, progress_cb=print_progress)
    print(f"[freewavesamples] done {result}")


if __name__ == "__main__":
    asyncio.run(main())

